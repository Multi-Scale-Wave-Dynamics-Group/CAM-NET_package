import os
import time
import datetime
import json
import numpy as np
from tqdm import tqdm
import pynvml
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import logging
import wandb
from makani.utils.dataloader import get_tracer_dataloader
from makani.utils.loss_tracer import TracerLossHandler
from makani.utils.metric import MetricsHandler
from makani.models import model_registry
from makani.utils import comm, visualize
from makani.utils.tracer_trainer import TracerTrainer

class TracerInferencer(TracerTrainer):
    def __init__(self, params, world_rank):
        self.params = params
        self.world_rank = world_rank
        self.device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger()
        pynvml.nvmlInit()
        self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(self.device.index)

        self.amp_enabled = hasattr(params, "amp_mode") and (params.amp_mode != "none")
        self.amp_dtype = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }.get(params.amp_mode, torch.float32)


        if getattr(params, "log_to_wandb", False):
            wandb.login()
            wandb.init(
                dir=params.experiment_dir,
                config=params,
                name=params.wandb_name,
                group=params.wandb_group,
                project=params.wandb_project,
                entity=params.wandb_entity,
                resume=params.resuming,
            )

        self.valid_dataloader, self.valid_dataset = get_tracer_dataloader(
            params, params.inf_data_path, params.tracer_inf_data_path, train=False
        )
        out_bias, out_scale, out_bias_tracer, out_scale_tracer = self.valid_dataloader.get_output_normalization()
        self.bias = out_bias_tracer[0, ...]
        self.scale = out_scale_tracer[0, ...]
        self.lat = self.valid_dataset.grid_converter.get_dst_coords()[0].cpu().numpy()
        self.lon = self.valid_dataset.grid_converter.get_dst_coords()[1].cpu().numpy() - np.pi
        params = self._update_parameters(params)
        self.params = params
        self.model = model_registry.get_model(params).to(self.device)
        self.preprocessor = self.model.preprocessor

        if self.world_rank == 0:
            print(self.model)
        
        self.restore_checkpoint(
                pretrained_checkpoint_path=self.params.pretrained_checkpoint_path,
                load_backbone_only=True,
                load_optimizer=False,
                load_scheduler=False,
                load_counters=False,
            )
        tracer_ckpt_path = self.params.tracer_checkpoint_path.format(mp_rank=comm.get_rank("model"))
        self.restore_tracer_checkpoint(tracer_ckpt_path)
        mult_cpu, clim, mult_cpu_tracer, clim_tracer = self._get_time_stats()
        self.metrics_tracer = MetricsHandler(self.params, mult_cpu_tracer, clim_tracer, self.device)
        self.metrics_tracer.initialize_buffers()
        self.loss_obj = TracerLossHandler(params).to(self.device)

    def _format_sample_time(self, year, local_idx):
        timestamp = datetime.datetime(year, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)
        timestamp += datetime.timedelta(hours=float(local_idx * self.params.dhours))
        return timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")

    def _raw_output_time_metadata(self, eval_step):
        """Reconstruct times for the first saved sample in a local inference batch."""
        source = getattr(self.valid_dataloader, "extsource", None)
        if source is None:
            return None

        batch_size = int(getattr(self.valid_dataloader, "batchsize", self.params.batch_size))
        idx_in_epoch = (eval_step - 1) * batch_size
        cycle_sample_idx = idx_in_epoch % source.num_samples_per_cycle_shard
        cycle_epoch_idx = idx_in_epoch // source.num_samples_per_cycle_shard

        if source.shuffle:
            rng = np.random.default_rng(seed=source.base_seed + cycle_epoch_idx)
            index_permutation = source.n_samples_offset + rng.permutation(source.n_samples_total)
        else:
            index_permutation = source.n_samples_offset + np.arange(source.n_samples_total)

        start = source.n_samples_shard * source.shard_id
        end = start + source.n_samples_shard
        index_permutation = index_permutation[start:end]
        sample_idx = int(index_permutation[cycle_sample_idx])

        year_idx = np.searchsorted(source.year_offsets, sample_idx, side="right") - 1
        local_idx = int(source.valid_indices_year[year_idx][sample_idx - source.year_offsets[year_idx]])

        if local_idx < source.dt * source.n_history:
            local_idx += source.dt * source.n_history
        if local_idx >= (source.n_samples_year[year_idx] - source.dt * (source.n_future + 1)):
            local_idx = source.n_samples_year[year_idx] - source.dt * (source.n_future + 1) - 1

        year = int(source.years[year_idx])
        input_start_idx = local_idx - source.dt * source.n_history
        target_indices = range(
            local_idx + source.dt,
            local_idx + source.dt * (source.n_future + 1) + 1,
            source.dt,
        )
        target_times = [self._format_sample_time(year, idx) for idx in target_indices]

        return {
            "sample_idx": sample_idx,
            "year": year,
            "local_idx": local_idx,
            "input_start_time": self._format_sample_time(year, input_start_idx),
            "initial_time": self._format_sample_time(year, local_idx),
            "lead0_time": target_times[0],
            "last_target_time": target_times[-1],
            "target_times": ",".join(target_times),
        }

    def _autoregressive_inference(self, data, compute_metrics=False, output_data=False):
        phys_in, _, tracer_input, tracer_tgt, zenith_in, zenith_tgt = map(
            lambda x: x.to(self.device, dtype=torch.float32), data
        )

        # === Preprocess ===
        physical, _ = self.preprocessor.cache_unpredicted_features(phys_in, _, xz=zenith_in, yz=zenith_tgt)
        physical = self.preprocessor.flatten_history(physical)
        tracer_input = self.preprocessor.flatten_history(tracer_input)
        tracer_tgt = self.preprocessor.flatten_history(tracer_tgt)
        c_out = self.params.N_tracer_out_channels
        tracer_tarlist = torch.split(tracer_tgt, c_out, dim=1)

        # === Setup storage ===
        pred_outputs = []
        targ_outputs = []

        base_model = self._get_base_model()

        # === Autoregressive rollout ===
        for idt, tracer_targ in enumerate(tracer_tarlist):

            # Append static/unpredicted features (no multi-scale extraction)
            inpa = self.preprocessor.append_unpredicted_features(physical)
            self.preprocessor.history_compute_stats(inpa)
            inpan = self.preprocessor.history_normalize(inpa, target=False)
            inpans = self.preprocessor.add_static_features(inpan)

            # Forward pass
            with amp.autocast(enabled=self.amp_enabled, dtype=self.amp_dtype):
                phy_features = base_model.physical_features(inpans)
                current_state = tracer_input[:, -self.params.N_tracer_out_channels:, ...]

                tracer_features = base_model.tracer_features(tracer_input)
                fused = torch.cat([phy_features, tracer_features], dim=1)

                pred_raw = base_model.predict_tracer(fused)

                if self.params.predict_delta:
                    tracer_pred = current_state + pred_raw   # convert delta → state
                    loss_target = tracer_targ - current_state
                    loss = self.loss_obj(pred_raw, loss_target, tracer_input)
                else:
                    tracer_pred = pred_raw
                    loss = self.loss_obj(tracer_pred, tracer_targ, tracer_input)
            # Predict physical outputs (no small/large split)
            phy_outputs = base_model.predict_physical(inpans)
            physical = self.preprocessor.append_history(physical, phy_outputs, idt)
            tracer_input = self.preprocessor.append_history(tracer_input, tracer_pred, idt)

            # === Metrics & optional output saving ===
            self.metrics_tracer.update(tracer_pred, tracer_targ, loss, idt)

            if output_data:
                pred_np = (self.scale * tracer_pred[0].detach().cpu().numpy()) + self.bias
                targ_np = (self.scale * tracer_targ[0].detach().cpu().numpy()) + self.bias
                pred_outputs.append(np.expand_dims(pred_np.astype(np.float32, copy=False), axis=0))
                targ_outputs.append(np.expand_dims(targ_np.astype(np.float32, copy=False), axis=0))

        # === Concatenate outputs ===
        if output_data:
            return np.concatenate(pred_outputs, axis=0), np.concatenate(targ_outputs, axis=0)
        return None, None

    def inference_single(self, single_time, compute_metrics=True, output_data=True):
        """
        Run tracer inference for one requested initial-condition time.

        This path expects the non-DALI tracer multifiles dataset, where
        single_time is the newest input-history timestamp.
        """
        world_size = comm.get_world_size() if dist.is_initialized() else 1
        if world_size > 1:
            raise RuntimeError(
                "Tracer single inference should be run with one process/GPU. "
                "Use one rank, or set --mode score for distributed inference."
            )

        if not hasattr(self.valid_dataset, "get_sample_times"):
            raise RuntimeError(
                "Tracer single inference requires the multifiles tracer dataset. "
                "Set params['multifiles'] = True before constructing TracerInferencer."
            )

        requested_time = np.datetime64(single_time, "h")
        sample_times = np.asarray(self.valid_dataset.get_sample_times(), dtype="datetime64[h]")
        n_history = int(self.params.n_history)
        dt_step = int(self.params.dt)
        n_future = int(getattr(self.valid_dataset, "n_future", self.params.valid_autoreg_steps))

        first_valid_initial = dt_step * n_history
        last_valid_initial_exclusive = len(sample_times) - dt_step * (n_future + 1)
        valid_initial_indices = np.arange(first_valid_initial, last_valid_initial_exclusive)
        if valid_initial_indices.size == 0:
            raise RuntimeError("No valid initial-condition times are available for tracer single inference.")

        valid_initial_times = sample_times[valid_initial_indices]
        time_diffs = np.abs(valid_initial_times - requested_time).astype("timedelta64[h]").astype(float)
        match_pos = int(np.argmin(time_diffs))
        initial_idx = int(valid_initial_indices[match_pos])
        matched_time = sample_times[initial_idx]
        dataset_idx = initial_idx - dt_step * n_history

        input_indices = list(range(dataset_idx, initial_idx + dt_step, dt_step))
        target_indices = list(range(initial_idx + dt_step, initial_idx + dt_step * (n_future + 1) + 1, dt_step))

        def format_time(value):
            return np.datetime_as_string(np.datetime64(value, "s"), unit="s") + "Z"

        input_times = [format_time(sample_times[idx]) for idx in input_indices]
        target_times = [format_time(sample_times[idx]) for idx in target_indices]

        print(f"Requested tracer initial time: {format_time(requested_time)}")
        print(f"Closest tracer initial time: {format_time(matched_time)} (raw time index {initial_idx})")
        print(f"Time difference: {time_diffs[match_pos]:.2f} hours")
        print(f"Input history times: {input_times}")
        print(f"Target rollout times: {target_times}")

        self._set_eval()
        torch.cuda.empty_cache()
        self.metrics_tracer.zero_buffers()

        data = self.valid_dataset[dataset_idx]
        if not isinstance(data, (list, tuple)):
            data = (data,)
        single_sample_data = tuple(tensor.unsqueeze(0) for tensor in data)

        with torch.inference_mode(), torch.no_grad():
            pred_batch, targ_batch = self._autoregressive_inference(
                single_sample_data,
                compute_metrics=compute_metrics,
                output_data=output_data,
            )

        logs = {}
        acc_curves = None
        rmse_curves = None
        if compute_metrics:
            logs, acc_curves, rmse_curves = self.metrics_tracer.finalize(final_inference=True)

        safe_time = format_time(matched_time).replace(":", "-").replace("Z", "")
        ic_dir = os.path.join(self.params.experiment_dir, "tracer_single_ic", f"ic_{safe_time}")
        os.makedirs(ic_dir, exist_ok=True)

        if output_data:
            np.save(os.path.join(ic_dir, "tracer_pred_outputs.npy"), pred_batch)
            np.save(os.path.join(ic_dir, "tracer_targ_outputs.npy"), targ_batch)

        if compute_metrics:
            np.save(os.path.join(ic_dir, "tracer_acc_curves.npy"), acc_curves.cpu().numpy())
            np.save(os.path.join(ic_dir, "tracer_rmse_curves.npy"), rmse_curves.cpu().numpy())

        metadata = {
            "requested_time_utc": format_time(requested_time),
            "matched_initial_time_utc": format_time(matched_time),
            "time_diff_hours": float(time_diffs[match_pos]),
            "dataset_index": int(dataset_idx),
            "initial_time_index": int(initial_idx),
            "input_history_times_utc": input_times,
            "target_times_utc": target_times,
            "lead_hours": float(dt_step * self.params.dhours),
            "prediction_shape": list(pred_batch.shape) if pred_batch is not None else None,
            "target_shape": list(targ_batch.shape) if targ_batch is not None else None,
        }
        with open(os.path.join(ic_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved tracer single inference outputs to {ic_dir}")
        return logs

    def inference_epoch(self):
        self._set_eval()
        torch.cuda.empty_cache()
        self.metrics_tracer.zero_buffers()
        output_data = bool(getattr(self.params, "save_raw_forecasts", False))
        raw_output_dir = os.path.join(self.params.experiment_dir, "tracer_raw_outputs")
        manifest_rows = []

        if output_data:
            os.makedirs(raw_output_dir, exist_ok=True)

        with torch.inference_mode(), torch.no_grad():
            eval_steps = 0
            for data in tqdm(self.valid_dataloader, desc="Scoring tracer", disable=not self.params.log_to_screen):
                eval_steps += 1
                pred_batch, targ_batch = self._autoregressive_inference(
                    data, compute_metrics=True, output_data=output_data
                )
                if output_data:
                    pred_name = f"tracer_pred_rank{self.world_rank:04d}_batch{eval_steps:05d}.npy"
                    targ_name = f"tracer_targ_rank{self.world_rank:04d}_batch{eval_steps:05d}.npy"
                    np.save(os.path.join(raw_output_dir, pred_name), pred_batch)
                    np.save(os.path.join(raw_output_dir, targ_name), targ_batch)
                    time_meta = self._raw_output_time_metadata(eval_steps)
                    if time_meta is None:
                        manifest_rows.append(
                            f"{eval_steps}\t{pred_name}\t{targ_name}\t{list(pred_batch.shape)}"
                            "\t\t\t\t\t\t\t\t\n"
                        )
                    else:
                        manifest_rows.append(
                            f"{eval_steps}\t{pred_name}\t{targ_name}\t{list(pred_batch.shape)}"
                            f"\t{time_meta['sample_idx']}\t{time_meta['year']}\t{time_meta['local_idx']}"
                            f"\t{time_meta['input_start_time']}\t{time_meta['initial_time']}"
                            f"\t{time_meta['lead0_time']}\t{time_meta['last_target_time']}"
                            f"\t{time_meta['target_times']}\n"
                        )
        logs, acc_curves, rmse_curves = self.metrics_tracer.finalize(final_inference=True)

        if output_data:
            manifest_name = f"manifest_rank{self.world_rank:04d}.tsv"
            with open(os.path.join(raw_output_dir, manifest_name), "w", encoding="utf-8") as f:
                f.write(
                    "batch\tprediction_file\ttarget_file\tshape\tsample_idx\tyear\tlocal_idx"
                    "\tinput_start_time_utc\tinitial_time_utc\tlead0_time_utc"
                    "\tlast_target_time_utc\ttarget_times_utc\n"
                )
                f.writelines(manifest_rows)

        if self.world_rank == 0:
            np.save(os.path.join(self.params.experiment_dir, "tracer_acc_curves.npy"), acc_curves.cpu().numpy())
            np.save(os.path.join(self.params.experiment_dir, "tracer_rmse_curves.npy"), rmse_curves.cpu().numpy())
            if output_data:
                print(f"Saved tracer prediction chunks to {raw_output_dir}")
            #visualize.plot_rollout_metrics(acc_curves, rmse_curves, self.params, epoch=0, model_name=self.params.nettype, comparison_channels=self.params.comparison_channels)

        if dist.is_initialized():
            dist.barrier(device_ids=[self.device.index])

        return logs

    def score_model(self):
        if self.params.log_to_screen:
            all_mem_gb = pynvml.nvmlDeviceGetMemoryInfo(self.nvml_handle).used / (1024.0 ** 3)
            max_mem_gb = torch.cuda.max_memory_allocated(device=self.device) / (1024.0 ** 3)
            self.logger.info(f"Memory usage: {all_mem_gb:.2f} GB (peak: {max_mem_gb:.2f} GB)")
            self.logger.info("Starting tracer model scoring...")

        if dist.is_initialized():
            dist.barrier(device_ids=[self.device.index])

        try:
            torch.cuda.reset_peak_memory_stats(self.device)
        except ValueError:
            pass

        start = time.time()
        scoring_logs = self.inference_epoch()
        end = time.time()

        self.log_score(scoring_logs, end - start)
        return

    def log_score(self, scoring_logs, scoring_time):
        separator = "-" * 50
        print_prefix = "    "

        def get_pad(nchar):
            return " " * nchar

        if self.params.log_to_screen:
            self.logger.info(separator)
            self.logger.info("Scoring summary:")
            self.logger.info("Total scoring time is {:.2f} sec".format(scoring_time))

            print_list = list(scoring_logs["metrics"].keys())
            max_len = max(len(x) for x in print_list)
            pad_len = [max_len - len(x) for x in print_list]

            self.logger.info("Metrics:")
            for idk, key in enumerate(print_list):
                value = scoring_logs["metrics"][key]
                self.logger.info(f"{print_prefix}{key}: {get_pad(pad_len[idk])}{value}")
            self.logger.info(separator)

    def _set_eval(self):
        base = self._get_base_model()
        base.physical_sfno.eval()
        base.tracer_encoder.eval()
        base.tracer_sfno.eval()
        self.loss_obj.eval()
        self.preprocessor.eval()
