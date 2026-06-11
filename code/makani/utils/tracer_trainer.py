# makani/utils/tracer_trainer.py
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import gc
import time
import subprocess

import torch
import torch.distributed as dist
import torch.cuda.amp as amp
import numpy as np
from tqdm import tqdm
import pynvml
import wandb

# Add project root (which contains `makani/`) to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from makani.utils.trainer import Trainer
from makani.utils import comm
from makani.utils.features import get_auxiliary_channels
from makani.models.tracer_stepper import TracerMultiStepWrapper


class TracerTrainer(Trainer):
    def __init__(self, params, world_rank, job_type="train"):
        import logging
        from makani.models import model_registry
        from makani.utils.losses import LossHandler
        from makani.utils.metric import MetricsHandler
        from makani.models.helpers import count_parameters
        from makani.utils.dataloader import get_tracer_dataloader
        import torch.cuda.amp as amp

        # === Setup parameters ===
        self.params = params
        self.world_rank = world_rank
        self.data_parallel_rank = comm.get_rank("data")
        tags = [
            f"ngpu{comm.get_world_size()}",
            f"mp{comm.get_size('model')}",
            f"sp{comm.get_size('spatial')}",
        ]
        self.device = torch.device(
            f"cuda:{torch.cuda.current_device()}"
            if torch.cuda.is_available() else "cpu"
        )

        if self.params.log_to_screen:
            self.logger = logging.getLogger()
            self.logger.info("Initializing TracerTrainer...")

        # === Data loading ===
        self.train_dataloader, self.train_dataset, self.train_sampler = get_tracer_dataloader(
            params,
            params.train_data_path,
            params.train_tracer_data_path,
            train=True,
        )

        self.valid_dataloader, self.valid_dataset = get_tracer_dataloader(
            params,
            params.valid_data_path,
            params.valid_tracer_data_path,
            train=False,
        )

        # === Custom parameter update ===
        self.params = self._update_parameters(self.params)

        # === Model + Preprocessor ===
        from makani.models.helpers import count_parameters
        self.model = model_registry.get_model(self.params).to(self.device)
        self.preprocessor = self.model.preprocessor

        base_model = self._get_base_model()
        # multistep tracer wrapper
        self.tracer_stepper = TracerMultiStepWrapper(
            self.params, base_model
        )

        # Only print model structure once (rank 0)
        if params.log_to_screen and dist.is_initialized() and dist.get_rank() == 0:
            self.logger.info(f"\n{self.model}")

        # === Sync model weights ===
        if dist.is_initialized():
            from makani.mpu.helpers import sync_params
            sync_params(self.model, mode="broadcast")

        # === Loss Handler ===
        if self.params.tracer_finetune:
            from makani.utils.loss_tracer import TracerLossHandler
            self.loss_obj = TracerLossHandler(self.params)
        else:
            self.loss_obj = LossHandler(self.params)
        self.loss_obj = self.loss_obj.to(self.device)

        # === Optimizer ===
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            betas=(self.params.optimizer_beta1, self.params.optimizer_beta2),
            lr=self.params.lr,
            weight_decay=self.params.weight_decay,
            foreach=True,
        )

        # === Scheduler ===
        self.scheduler = None

        if params.scheduler == "ReduceLROnPlateau":
            if not hasattr(params, "scheduler_mode"):
                params["scheduler_mode"] = "min"
            if params.skip_validation:
                raise ValueError(f"Error, you cannot skip validation when using ReduceLROnPlateau scheduler.")
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, factor=params.scheduler_factor, patience=params.scheduler_patience, mode=params.scheduler_mode
            )

        elif params.scheduler == "StepLR":
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=params.scheduler_step_size, gamma=params.scheduler_gamma)

        elif params.scheduler == "CosineAnnealingLR":
            if not hasattr(params, "scheduler_min_lr"):
                params["scheduler_min_lr"] = 0.0
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=params.scheduler_T_max, eta_min=params.scheduler_min_lr)

        elif params.scheduler == "OneCycleLR":
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=params.lr, total_steps=params.scheduler_T_max, steps_per_epoch=1)

        else:
            self.scheduler = None

        if params.lr_warmup_steps > 0:
            if params.scheduler == "ReduceLROnPlateau":
                raise NotImplementedError("Error, warmup scheduler not implemented for ReduceLROnPlateau scheduler")
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=params.lr_start, end_factor=1.0, total_iters=params.lr_warmup_steps)

            self.scheduler = torch.optim.lr_scheduler.SequentialLR(self.optimizer, [warmup_scheduler, self.scheduler], milestones=[params.lr_warmup_steps])

        # === NVML init ===
        if params.log_to_screen:
            pynvml.nvmlInit()
            self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(self.device.index)

        # === AMP Setup ===
        self.amp_enabled = (
            hasattr(self.params, "amp_mode")
            and self.params.amp_mode in ["fp16", "bf16"]
        )
        self.amp_dtype = (
            torch.float16 if self.params.amp_mode == "fp16" else
            torch.bfloat16 if self.params.amp_mode == "bf16" else
            torch.float32
        )
        self.gscaler = amp.GradScaler(enabled=(self.amp_dtype == torch.float16))

        from makani.mpu.mappings import init_gradient_reduction_hooks
        # === Graph/sample init ===
        self._set_train()
        iterator = iter(self.train_dataloader)
        data = next(iterator)
        gdata = map(lambda x: x.to(self.device, dtype=torch.float32), data)

        # Preprocess based on tracer mode
        if self.params.tracer_finetune:
            physical_input, physical_target, tracer_input, tracer_target, zenith_input, zenith_target = gdata
            inp, tar = self.preprocessor.cache_unpredicted_features(
                physical_input, physical_target,
                xz=zenith_input, yz=zenith_target
            )
        else:
            inp, tar = self.preprocessor.cache_unpredicted_features(*gdata)

        # Flatten input/target across history
        inp = self.preprocessor.flatten_history(inp)
        tar = self.preprocessor.flatten_history(tar)
        inp_shape = inp.shape
        tar_shape = tar.shape

        # === Compile or JIT the model ===
        self._compile_model(inp_shape)
        if not self.loss_obj.is_distributed():
            self.loss_obj = torch.jit.script(self.loss_obj)

        # === CUDA Graph capture (optional) ===
        self.graph = None
        if (
            self.params.cuda_graph_mode != "none"
            and self.device.type == "cuda"
        ):
            capture_stream = torch.cuda.Stream()
            self._capture_model(
                capture_stream, inp_shape, tar_shape,
                num_warmup_steps=20
            )

        # === Metrics Handler ===
        from makani.utils.metric import MetricsHandler
        mult_cpu, clim, multi_cpu_tracer, clim_tracer = self._get_time_stats()
        self.metrics = MetricsHandler(self.params, mult_cpu, clim, self.device)
        self.metrics.initialize_buffers()

        self.metrics_tracer = MetricsHandler(
            self.params, multi_cpu_tracer, clim_tracer, self.device
        )
        self.metrics_tracer.initialize_buffers()

        from makani.utils import visualize

        # === Visualization Setup ===
        if self.params.log_video:
            plot_list = [
                {"name": "O at lever of 20", "functor": "lambda x: x[0, ...]", "diverging": True},
                {"name": "O at level of 25", "functor": "lambda x: x[1, ...]", "diverging": True},
                {"name": "O at level of 30", "functor": "lambda x: x[3, ...]", "diverging": True},
            ]

            lat = self.valid_dataset.grid_converter.get_dst_coords()[0].cpu().numpy()
            lon = self.valid_dataset.grid_converter.get_dst_coords()[1].cpu().numpy() - np.pi

            out_bias, out_scale, out_bias_tracer, out_scale_tracer = (
                self.train_dataloader.get_output_normalization()
            )

            self.visualizer = visualize.VisualizationWrapper(
                log_to_wandb=self.params.log_to_wandb,
                path=None,
                prefix=None,
                plot_list=plot_list,
                lat=lat,
                lon=lon,
                scale=out_scale_tracer[0, ...],
                bias=out_bias_tracer[0, ...],
                num_workers=self.params.num_visualization_workers,
            )

            pin_memory = (self.device.type == "cuda")
            if self.device.type == "cuda":
                self.viz_stream = torch.cuda.Stream()
            else:
                self.viz_stream = None

            self.viz_prediction_cpu_tracer = torch.empty(
                (
                    (params.N_tracer_target_channels) // (params.n_future + 1),
                    params.img_shape_x,
                    params.img_shape_y,
                ),
                device="cpu",
                pin_memory=pin_memory,
            )

            self.viz_target_cpu_tracer = torch.empty_like(
                self.viz_prediction_cpu_tracer
            )

        # === Restore checkpoints (backbone or full) ===
        self.iters = 0
        self.startEpoch = 0

        if self.params.tracer_finetune:
            if self.params.pretrained_checkpoint_path is None:
                raise ValueError(
                    "tracer_finetune=True requires pretrained_checkpoint_path "
                    "so the physical SFNO backbone can be initialized."
                )

            self.restore_checkpoint(
                pretrained_checkpoint_path=self.params.pretrained_checkpoint_path,
                load_backbone_only=True,
                load_optimizer=False,
                load_scheduler=False,
                load_counters=False,
            )

            if self.params.resuming:
                if (
                    not hasattr(self.params, "tracer_checkpoint_path")
                    or self.params.tracer_checkpoint_path is None
                ):
                    raise ValueError(
                        "resuming=True requires tracer_checkpoint_path for tracer finetuning."
                    )

                ckpt_file = self.params.tracer_checkpoint_path.format(
                    mp_rank=comm.get_rank("model")
                )

                if not os.path.exists(ckpt_file):
                    raise FileNotFoundError(
                        f"Tracer checkpoint not found for resume: {ckpt_file}"
                    )

                if self.params.log_to_screen:
                    self.logger.info(f"Restoring tracer checkpoint from {ckpt_file}")
                self.restore_tracer_checkpoint(
                    ckpt_file,
                    load_counters=self.params.load_counters,
                    load_optimizer=self.params.load_optimizer,
                    load_scheduler=self.params.load_scheduler,
                )
            else:
                if self.params.log_to_screen:
                    self.logger.info(
                        "Fresh tracer finetune: loaded pretrained physical SFNO "
                        "and initialized tracer modules from scratch."
                    )

        elif self.params.resuming:
            raise NotImplementedError(
                "Full-model resume is not implemented in TracerTrainer. "
                "Use tracer_finetune=True to load the pretrained physical SFNO "
                "and resume tracer weights from tracer_checkpoint_path."
            )

        self.epoch = self.startEpoch

        if self.params.log_to_screen:
            pcount = count_parameters(self.model, self.device)
            self.logger.info(f"Number of trainable model parameters: {pcount}")

    # ----------------------------------------------------------------------
    #                         Parameter Update Function
    # ----------------------------------------------------------------------

    def _update_parameters(self, params):
        """
        Routine for updating parameters internally.
        This is the only place where parameters are updated.
        """
        params.N_in_channels = len(self.valid_dataset.in_channels)
        params.N_out_channels = len(self.valid_dataset.out_channels)
        params.N_tracer_in_channels = len(self.valid_dataset.tracer_in_channels)
        params.N_tracer_out_channels = len(self.valid_dataset.tracer_out_channels)

        params.img_shape_x = self.valid_dataset.img_shape_x
        params.img_shape_y = self.valid_dataset.img_shape_y

        params.img_crop_shape_x = self.valid_dataset.img_crop_shape_x
        params.img_crop_shape_y = self.valid_dataset.img_crop_shape_y
        params.img_crop_offset_x = self.valid_dataset.img_crop_offset_x
        params.img_crop_offset_y = self.valid_dataset.img_crop_offset_y

        params.img_local_shape_x = self.valid_dataset.img_local_shape_x
        params.img_local_shape_y = self.valid_dataset.img_local_shape_y
        params.img_local_offset_x = self.valid_dataset.img_local_offset_x
        params.img_local_offset_y = self.valid_dataset.img_local_offset_y

        # derived quantities
        params["N_in_predicted_channels"] = params.N_in_channels

        # --- Sanitization ---
        if not hasattr(params, "add_zenith"):
            params["add_zenith"] = False

        # Input channels
        if params.add_zenith:
            params.N_in_channels += 1

        if params.n_history >= 1:
            params.N_in_channels *= (params.n_history + 1)
            params.N_in_predicted_channels *= (params.n_history + 1)
            params.N_tracer_in_channels *= (params.n_history + 1)

        if params.add_grid:
            n_grid_chan = 2
            if (
                params.gridtype == "sinusoidal"
                and hasattr(params, "grid_num_frequencies")
            ):
                n_grid_chan *= params.grid_num_frequencies
            params.N_in_channels += n_grid_chan

        if params.add_orography:
            params.N_in_channels += 1

        if params.add_landmask:
            params.N_in_channels += 2

        # Auxiliary channel names
        params["aux_channel_names"] = get_auxiliary_channels(**params.to_dict())

        # Target channels
        params.N_target_channels = (
            (params.n_future + 1) * params.N_out_channels 
        )
        params.N_tracer_target_channels = (
            (params.n_future + 1) * params.N_tracer_out_channels
        )

        # Misc parameters
        defaults = {
            "history_normalization_mode": "none",
            "num_visualization_workers": 1,
            "log_video": 0,
            "log_weights_and_grads": 0,
            "skip_validation": False,
            "load_checkpoint": "legacy",
            "save_checkpoint": "legacy",
            "load_optimizer": True,
            "load_scheduler": True,
            "load_counters": True,
        }
        for k, v in defaults.items():
            if not hasattr(params, k):
                params[k] = v

        if not hasattr(self.params, "disable_ddp"):
            self.params.disable_ddp = False

        if not hasattr(self.params, "parameters_reduction_buffer_count"):
            self.params.parameters_reduction_buffer_count = 4

        if not hasattr(self.params, "enable_grad_anomaly_detection"):
            self.params.enable_grad_anomaly_detection = False

        if not hasattr(self.params, "checkpointing"):
            self.params.checkpointing = 0

        return params

    # ----------------------------------------------------------------------
    #                           Helper: unwrap model
    # ----------------------------------------------------------------------
    def _get_base_model(self):
        model = self.model
        if hasattr(model, "module"):  # DDP unwrap
            model = model.module
        if hasattr(model, "model"):  # unwrap SingleStepWrapper
            model = model.model
        return model

    def _sync_data_parallel_gradients(self):
        if (not dist.is_initialized()) or comm.get_size("data") == 1:
            return

        data_group = comm.get_group("data")
        data_size = float(comm.get_size("data"))
        for param in self.model.parameters():
            if param.grad is None:
                continue
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, group=data_group)
            param.grad.div_(data_size)

    # ----------------------------------------------------------------------
    #                           Train / Eval mode
    # ----------------------------------------------------------------------
    def _set_train(self):
        model = self._get_base_model()
        self.tracer_stepper.train()
        for p in model.physical_sfno.parameters():
            p.requires_grad = False
        model.physical_sfno.eval()
        model.tracer_encoder.train()
        model.tracer_sfno.train()
        self.loss_obj.train()
        self.preprocessor.train()

    def _set_eval(self):
        model = self._get_base_model()
        model.physical_sfno.eval()
        model.tracer_encoder.eval()
        model.tracer_sfno.eval()
        self.tracer_stepper.eval()
        self.loss_obj.eval()
        self.preprocessor.eval()

    # ----------------------------------------------------------------------
    #                           Helper functions
    # ----------------------------------------------------------------------
    def _tracer_c_out(self):
        return self.params.N_tracer_out_channels

    def _split_tracer_steps(self, x):
        """
        Split a flattened multistep tracer tensor into [B, C_out, H, W] chunks.
        Expects shape [B, (n_future+1)*C_out, H, W].
        """
        c = self._tracer_c_out()
        if x.shape[1] % c != 0:
            raise ValueError(
                f"Tracer channel dimension {x.shape[1]} is not divisible by C_out={c}"
            )
        return list(torch.split(x, c, dim=1))

    def _make_delta_targets(self, current_state, target_steps):
        """
        Convert future state targets into delta targets.
        current_state: [B, C_out, H, W]
        target_steps: list of future states [t+1, t+2, ...]
        """
        delta_targets = []
        prev = current_state
        for state in target_steps:
            delta_targets.append(state - prev)
            prev = state
        return delta_targets

    def _time_step_weight(self, step_idx):
        """
        Optional step weighting for multistep loss.
        """
        mode = getattr(self.params, "loss_time_step_mode", "linear")

        if mode == "linear":
            return float(step_idx + 1)
        if mode == "sqrt":
            return float(np.sqrt(step_idx + 1))
        if mode == "exp":
            gamma = float(getattr(self.params, "loss_time_step_gamma", 0.9))
            return float(gamma ** step_idx)

        return 1.0

    def _weighted_multistep_loss(self, pred_steps, target_steps, tracer_input):
        total_loss = 0.0
        total_weight = 0.0

        for t, (pred_t, targ_t) in enumerate(zip(pred_steps, target_steps)):
            w = self._time_step_weight(t)
            total_loss = total_loss + w * self.loss_obj(pred_t, targ_t, tracer_input)
            total_weight += w

        return total_loss / max(total_weight, 1e-8)

    # ----------------------------------------------------------------------
    #                           Training one epoch
    # ----------------------------------------------------------------------
    def train_one_epoch(self):
        self.epoch += 1
        total_data_bytes = 0
        self._set_train()

        train_steps = 0
        train_start = time.perf_counter_ns()

        base_model = self._get_base_model()
        c_out = self._tracer_c_out()

        for data in tqdm(
            self.train_dataloader,
            desc="Training tracer head",
            disable=not self.params.log_to_screen,
        ):
            train_steps += 1
            self.iters += 1

            # -------------------------------
            # 1) Move data to GPU
            # -------------------------------
            gdata = map(lambda x: x.to(self.device, dtype=torch.float32), data)
            (
                physical_input,
                physical_target,
                tracer_input,
                tracer_target,
                zenith_input,
                zenith_target,
            ) = gdata

            # -------------------------------
            # 2) Preprocess physical input
            # -------------------------------
            physical, _ = self.preprocessor.cache_unpredicted_features(
                physical_input,
                physical_target,
                xz=zenith_input,
                yz=zenith_target,
            )

            physical = self.preprocessor.flatten_history(physical)

            inpa = self.preprocessor.append_unpredicted_features(physical)
            self.preprocessor.history_compute_stats(inpa)
            inpan = self.preprocessor.history_normalize(inpa, target=False)
            inpans = self.preprocessor.add_static_features(inpan)

            with torch.no_grad():
                physical_features = base_model.physical_features(inpans)

            # -------------------------------
            # 3) Preprocess tracer input/target
            # -------------------------------
            tracer_input = self.preprocessor.flatten_history(tracer_input)
            tracer_target = self.preprocessor.flatten_history(tracer_target)

            total_data_bytes += (physical.numel() + tracer_target.numel()) * 4

            # Split into per-step chunks
            pred_out = self.tracer_stepper(
                tracer_input=tracer_input,
                physical_features=physical_features,
            )
            pred_steps = self._split_tracer_steps(pred_out)
            target_steps = self._split_tracer_steps(tracer_target)

            # If training in delta mode, convert future states to deltas
            if self.params.predict_delta:
                current = tracer_input[:, -c_out:, ...]
                target_steps = self._make_delta_targets(current, target_steps)

            # -------------------------------
            # 4) Loss
            # -------------------------------
            self.optimizer.zero_grad(set_to_none=True)

            with amp.autocast(enabled=self.amp_enabled, dtype=self.amp_dtype):
                loss = self._weighted_multistep_loss(
                    pred_steps, target_steps, tracer_input
                )

            self.gscaler.scale(loss).backward()
            self.gscaler.unscale_(self.optimizer)
            self._sync_data_parallel_gradients()

            if hasattr(self.params, "grad_clip"):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.params.grad_clip
                )

            self.gscaler.step(self.optimizer)
            self.gscaler.update()

            if (
                self.params.print_timings_frequency > 0
                and (self.iters % self.params.print_timings_frequency == 0)
                and self.params.log_to_screen
            ):
                running_train_time = time.perf_counter_ns() - train_start
                print(
                    f"Average step time after step {self.iters}: "
                    f"{running_train_time / float(train_steps) * 1e-6:.1f} ms"
                )
                print(
                    f"Average effective io rate after step {self.iters}: "
                    f"{total_data_bytes * float(comm.get_world_size()) / (float(running_train_time) * 1e-9 * 1024. * 1024. * 1024.):.2f} GB/s"
                )
                print(f"Current loss {loss.item()}")
                lr = self.optimizer.param_groups[0]["lr"]
                if self.data_parallel_rank == 0:
                    print(f"[Iter {self.iters}] LR = {lr:.6e}")

        logs = {
            "loss": float(loss.detach().item()),
            "train_steps": train_steps,
        }

        train_end = time.perf_counter_ns()
        train_time = (train_end - train_start) * 1e-9
        total_data_gb = (total_data_bytes / (1024 ** 3)) * float(comm.get_world_size())

        return train_time, total_data_gb, logs
    # ----------------------------------------------------------------------
    #                           Validation epoch
    # ----------------------------------------------------------------------
    def validate_one_epoch(self, epoch):
        self._set_eval()
        torch.cuda.empty_cache()
        self.metrics_tracer.zero_buffers()

        visualize = self.params.log_video and (epoch % self.params.log_video == 0)

        valid_start = time.time()
        base_model = self._get_base_model()
        c_out = self._tracer_c_out()

        viz_time = 0.0

        with torch.inference_mode(), torch.no_grad():
            eval_steps = 0
            for data in tqdm(
                self.valid_dataloader,
                desc="Validation progress",
                disable=not self.params.log_to_screen,
            ):
                eval_steps += 1

                # ---- map to gpu ----
                gdata = map(lambda x: x.to(self.device, dtype=torch.float32), data)
                (
                    physical_input,
                    physical_target,
                    tracer_input,
                    tracer_target,
                    zenith_input,
                    zenith_target,
                ) = gdata

                # ---- preprocess physical ----
                physical, _ = self.preprocessor.cache_unpredicted_features(
                    physical_input,
                    physical_target,
                    xz=zenith_input,
                    yz=zenith_target,
                )

                physical = self.preprocessor.flatten_history(physical)

                # ---- flatten tracer history once ----
                tracer_input = self.preprocessor.flatten_history(tracer_input)
                tracer_target = self.preprocessor.flatten_history(tracer_target)

                tracer_target_steps = self._split_tracer_steps(tracer_target)

                # Roll out one step at a time
                for idt, tracer_targ in enumerate(tracer_target_steps):
                    inpa = self.preprocessor.append_unpredicted_features(physical)
                    self.preprocessor.history_compute_stats(inpa)
                    inpan = self.preprocessor.history_normalize(inpa, target=False)
                    inpans = self.preprocessor.add_static_features(inpan)

                    physical_features = base_model.physical_features(inpans)

                    # current tracer state = latest frame in history
                    current_state = tracer_input[:, -c_out:, ...]

                    with amp.autocast(enabled=self.amp_enabled, dtype=self.amp_dtype):
                        pred_raw = self.tracer_stepper(
                            tracer_input=tracer_input,
                            physical_features=physical_features,
                        )

                    if self.params.predict_delta:
                        # pred_raw is delta, so convert to next state for rollout
                        tracer_next = current_state + pred_raw

                        # if your loss is trained on deltas, compare against delta target
                        loss_target = tracer_targ - current_state
                        loss = self.loss_obj(pred_raw, loss_target, tracer_input)
                    else:
                        tracer_next = pred_raw
                        loss = self.loss_obj(tracer_next, tracer_targ, tracer_input)

                    tracer_input = self.preprocessor.append_history(tracer_input, tracer_next, idt)
                    physical = self.preprocessor.append_history(physical, base_model.predict_physical(inpans), idt)

                    self.metrics_tracer.update(tracer_next, tracer_targ, loss, idt)

                    if visualize and eval_steps == 1:
                        self.plot_tracer_rollout(
                            [tracer_next],
                            [tracer_targ],
                            save_path="tracer_rollout.png",
                            channel_idx=0,
                            channel_name=None,
                        )
                        pred_cpu = tracer_next[0].detach().cpu().numpy()
                        targ_cpu = tracer_targ[0].detach().cpu().numpy()
                        tag = f"step{eval_steps}_time{str(idt).zfill(3)}"
                        if visualize and eval_steps == 1:
                            self.visualizer.add(tag, pred_cpu, targ_cpu)

            logs = self.metrics_tracer.finalize()

            if visualize:
                t0 = time.perf_counter_ns()
                self.visualizer.finalize()
                viz_time = (time.perf_counter_ns() - t0) * 1e-9

        if dist.is_initialized():
            dist.barrier(device_ids=[self.device.index])

        valid_time = time.time() - valid_start
        return valid_time, viz_time, logs
    # ----------------------------------------------------------------------
    #                           Epoch logging
    # ----------------------------------------------------------------------
    def log_epoch(self, train_logs, valid_logs, timing_logs):
        separator = "-" * 50
        print_prefix = "    "

        def pad(n):
            return " " * n

        if self.params.log_to_screen:
            self.logger.info(separator)
            self.logger.info(f"Epoch {self.epoch} summary:")
            self.logger.info("Performance Parameters:")
            self.logger.info(f"{print_prefix}training steps: {train_logs['train_steps']}")
            self.logger.info(f"{print_prefix}validation steps: {valid_logs['base']['validation steps']}")

            mem = pynvml.nvmlDeviceGetMemoryInfo(self.nvml_handle).used / (1024 ** 3)
            self.logger.info(f"{print_prefix}memory footprint [GB]: {mem:.2f}")

            for k in timing_logs:
                self.logger.info(f"{print_prefix}{k}: {timing_logs[k]:.2f}")

            # metric table formatting
            keys = ["training loss", "validation loss", "validation L1"] + list(valid_logs["metrics"].keys())
            max_len = max(len(k) for k in keys)
            pads = [max_len - len(k) for k in keys]

            # core metrics
            self.logger.info("Metrics:")
            self.logger.info(f"{print_prefix}training loss: {pad(pads[0])}{train_logs['loss']}")
            self.logger.info(f"{print_prefix}validation loss: {pad(pads[1])}{valid_logs['base']['validation loss']}")
            self.logger.info(f"{print_prefix}validation L1: {pad(pads[2])}{valid_logs['base']['validation L1']}")

            # extra metrics
            for i, key in enumerate(keys[3:], start=3):
                v = valid_logs["metrics"][key]
                if np.isscalar(v):
                    self.logger.info(f"{print_prefix}{key}: {pad(pads[i])}{v}")

            self.logger.info(separator)

        if self.params.log_to_wandb:
            wandb.log(train_logs, step=self.epoch)
            wandb.log(valid_logs["base"], step=self.epoch)
            wandb.log(valid_logs["metrics"], step=self.epoch)

        return

    # ----------------------------------------------------------------------
    #                           Restore checkpoint
    # ----------------------------------------------------------------------
    def restore_checkpoint(
        self,
        pretrained_checkpoint_path,
        checkpoint_mode="legacy",
        load_optimizer=False,
        load_scheduler=False,
        load_counters=False,
        load_backbone_only=False,
    ):
        base = self._get_base_model()

        # === Load only the physical SFNO ===
        if pretrained_checkpoint_path is not None and load_backbone_only:
            if self.params.log_to_screen:
                self.logger.info(f"Loading retrained physical SFNO from: {pretrained_checkpoint_path}")

            checkpoint_fname = pretrained_checkpoint_path.format(
                mp_rank=comm.get_rank("model")
            )
            if not os.path.exists(checkpoint_fname):
                raise FileNotFoundError(
                    f"Physical SFNO checkpoint not found: {checkpoint_fname}"
                )

            ckpt = torch.load(checkpoint_fname, map_location="cpu")
            state_dict = ckpt.get("model_state", ckpt.get("model_state_dict"))
            if state_dict is None:
                raise RuntimeError(
                    f"Checkpoint {checkpoint_fname} does not contain model_state."
                )

            # remove DDP / wrapper prefixes
            new_state = {}
            for k, v in state_dict.items():
                for prefix in ["module.model.", "module.", "model."]:
                    if k.startswith(prefix):
                        k = k[len(prefix):]
                new_state[k] = v

            missing, unexpected = base.physical_sfno.load_state_dict(
                new_state, strict=False
            )
            if self.params.log_to_screen:
                self.logger.info(f"Loaded physical SFNO from {checkpoint_fname}")
                self.logger.info(f"Missing physical SFNO keys: {missing}")
                self.logger.info(f"Unexpected physical SFNO keys: {unexpected}")
            return

    # ----------------------------------------------------------------------
    #                          Restore tracer checkpoint
    # ----------------------------------------------------------------------
    def restore_tracer_checkpoint(
        self, ckpt_path, load_optimizer=False, load_scheduler=False, load_counters=False
    ):
        print(f"Loading tracer checkpoint: {ckpt_path}")

        # ---- Load checkpoint file ----
        checkpoint = torch.load(ckpt_path, map_location="cpu")

        # Extract model weights
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict) and "model_state" in checkpoint:
            state = checkpoint["model_state"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state = checkpoint["state_dict"]
        else:
            state = checkpoint  # fallback

        if not isinstance(state, dict):
            raise RuntimeError(f"Checkpoint {ckpt_path} does not contain a state dict.")

        # ---- Remove wrapper/DDP prefixes ----
        cleaned = {}
        for k, v in state.items():
            changed = True
            while changed:
                changed = False
                for prefix in ("module.model.", "module.", "model.", "_orig_mod."):
                    if k.startswith(prefix):
                        k = k[len(prefix):]
                        changed = True
                        break
            cleaned[k] = v

        # ---- Keep only tracer-related keys ----
        tracer_keys = {
            k: v for k, v in cleaned.items()
            if k.startswith("tracer_encoder") or k.startswith("tracer_sfno")
        }
        if not tracer_keys:
            sample = list(cleaned.keys())[:10]
            raise RuntimeError(
                f"No tracer weights found in {ckpt_path}. "
                f"Sample checkpoint keys after prefix cleanup: {sample}"
            )

        encoder_state = {
            k[len("tracer_encoder."):]: v
            for k, v in tracer_keys.items()
            if k.startswith("tracer_encoder.")
        }
        sfno_state = {
            k[len("tracer_sfno."):]: v
            for k, v in tracer_keys.items()
            if k.startswith("tracer_sfno.")
        }
        if not encoder_state or not sfno_state:
            raise RuntimeError(
                f"Incomplete tracer checkpoint {ckpt_path}: "
                f"encoder tensors={len(encoder_state)}, sfno tensors={len(sfno_state)}"
            )

        # ---- Load into base model ----
        base = self._get_base_model()
        enc_missing, enc_unexpected = base.tracer_encoder.load_state_dict(encoder_state, strict=False)
        sfno_missing, sfno_unexpected = base.tracer_sfno.load_state_dict(sfno_state, strict=False)

        if enc_missing or enc_unexpected or sfno_missing or sfno_unexpected:
            raise RuntimeError(
                "Tracer checkpoint did not match the current tracer modules: "
                f"encoder missing={enc_missing}, encoder unexpected={enc_unexpected}, "
                f"sfno missing={sfno_missing}, sfno unexpected={sfno_unexpected}"
            )

        print(
            "   Tracer weights loaded "
            f"({len(encoder_state)} encoder tensors, {len(sfno_state)} sfno tensors)."
        )

        # ---- Restore optimizer ----
        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print("   ✓ Optimizer state restored.")

        # ---- Restore scheduler ----
        if load_scheduler and "scheduler_state_dict" in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            print("   ✓ Scheduler state restored.")

        # ---- Restore counters ----
        if load_counters:
            if "iters" in checkpoint:
                self.iters = checkpoint["iters"]
            if "epoch" in checkpoint:
                self.startEpoch = checkpoint["epoch"]
            print("   ✓ Counters restored.")


    # ----------------------------------------------------------------------
    #                          Save tracer checkpoint
    # ----------------------------------------------------------------------
    def save_tracer_checkpoint(self, checkpoint_path=None):
        checkpoint_path = checkpoint_path or self.params.tracer_checkpoint_path
        checkpoint_fname = checkpoint_path.format(
            mp_rank=comm.get_rank("model")
        )

        model = self._get_base_model()
        state = model.state_dict()

        tracer_state = {
            k: v for k, v in state.items()
            if "tracer_encoder" in k or "tracer_sfno" in k
        }

        checkpoint = {
            "iters": self.iters,
            "epoch": self.epoch,
            "model_state_dict": tracer_state,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "params": self.params,
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(checkpoint, checkpoint_fname)
        if self.params.log_to_screen:
            self.logger.info(f"Tracer checkpoint saved to {checkpoint_fname}")

    # ----------------------------------------------------------------------
    #                     Utility: print GPU memory usage
    # ----------------------------------------------------------------------
    def print_memory_usage(self, tag=""):
        pid = os.getpid()
        mem = torch.cuda.memory_allocated() / 1e9
        print(f"[{tag}] Rank {comm.get_world_rank()}, PID {pid} using {mem:.2f} GB")

    # ----------------------------------------------------------------------
    #                               Train loop
    # ----------------------------------------------------------------------
    def train(self):
        if self.params.log_to_screen:
            all_mem_gb = pynvml.nvmlDeviceGetMemoryInfo(
                self.nvml_handle
            ).used / (1024 ** 3)
            max_mem_gb = (
                torch.cuda.max_memory_allocated(device=self.device)
                / (1024 ** 3)
            )
            self.logger.info(
                f"Scaffolding memory high watermark: {all_mem_gb} GB "
                f"({max_mem_gb} GB for pytorch)"
            )
            self.logger.info("Starting Training Loop...")

        if dist.is_initialized():
            dist.barrier(device_ids=[self.device.index])

        try:
            torch.cuda.reset_peak_memory_stats(self.device)
        except ValueError:
            pass

        training_start = time.time()
        best_valid_loss = 1.0e6
        for epoch in range(self.startEpoch, self.params.max_epochs):
            if dist.is_initialized() and self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)

            epoch_start = time.time()
            train_time, train_data_gb, train_logs = self.train_one_epoch()
            if not self.params.skip_validation:
                valid_time, viz_time, valid_logs = self.validate_one_epoch(epoch)
            else:
                valid_time = 0
                viz_time = 0
                valid_logs = {"base": {}, "metrics": {}}
            # schedulers
            if self.params.scheduler == "ReduceLROnPlateau":
                self.scheduler.step(valid_logs["base"]["validation loss"])
            elif self.scheduler is not None:
                self.scheduler.step()
            # wandb lr logging
            if self.params.log_to_wandb:
                for pg in self.optimizer.param_groups:
                    lr = pg["lr"]
                wandb.log({"learning rate": lr}, step=self.epoch)
            print(f"Rank: {self.data_parallel_rank}, Save Checkpoint Mode: {self.params.save_checkpoint}")
            print(f"Checkpoint Path: {self.params.tracer_checkpoint_path}")
            # checkpoint saving
            if (self.data_parallel_rank == 0) and (self.params.save_checkpoint != "none"):
                self.save_tracer_checkpoint(self.params.tracer_checkpoint_path)
                best_path = self.params.best_checkpoint_path.format(
                    mp_rank=comm.get_rank("model")
                )
                best_exists = os.path.isfile(best_path)

                if (
                    not self.params.skip_validation
                    and ((not best_exists) or
                         (valid_logs["base"]["validation loss"] <= best_valid_loss))
                ):
                    self.save_tracer_checkpoint(self.params.best_checkpoint_path)
                    best_valid_loss = valid_logs["base"]["validation loss"]

            if dist.is_initialized():
                dist.barrier(device_ids=[self.device.index])
            epoch_end = time.time()

            timing_logs = {
                "epoch time [s]": epoch_end - epoch_start,
                "training time [s]": train_time,
                "validation time [s]": valid_time,
                "visualization time [s]": viz_time,
                "training step time [ms]": (train_time / train_logs["train_steps"]) * 1e3,
                "minimal IO rate [GB/s]": train_data_gb / train_time,
            }
            self.log_epoch(train_logs, valid_logs, timing_logs)
        total_time = time.time() - training_start
        if self.params.log_to_screen:
            self.logger.info(f"Total training time is {total_time:.2f} sec")
        return
    # ----------------------------------------------------------------------
    #                          Plot physical outputs
    # ----------------------------------------------------------------------
    def plot_physical_outputs(
        self, tensor, tar_tensor,
        save_path="phy_outputs.png",
        vmin=None, vmax=None
    ):
        """
        Compare model output tensor and target tensor by plotting the first 4 channels.

        Args:
            tensor (torch.Tensor): model output [B, C, H, W]
            tar_tensor (torch.Tensor): target tensor same shape
            save_path (str): output figure path
        """
        import matplotlib.pyplot as plt

        tensor = tensor.detach().cpu()
        tar_tensor = tar_tensor.detach().cpu()

        B, C, H, W = tensor.shape
        channels_to_plot = min(C, 4)

        fig, axes = plt.subplots(
            2, channels_to_plot,
            figsize=(channels_to_plot * 4, 8)
        )

        for i in range(channels_to_plot):
            ax_pred = axes[0, i]
            ax_true = axes[1, i]

            im = ax_pred.imshow(
                tensor[0, i],
                cmap='viridis',
                vmin=vmin, vmax=vmax
            )
            ax_pred.set_title(f"Predicted Channel {i}")
            ax_pred.axis('off')

            ax_true.imshow(
                tar_tensor[0, i],
                cmap='viridis',
                vmin=vmin, vmax=vmax
            )
            ax_true.set_title(f"Target Channel {i}")
            ax_true.axis('off')

            fig.colorbar(im, ax=[ax_pred, ax_true], shrink=0.6)

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Saved physical outputs to {save_path}")


    # Add these methods to your TracerTrainer class
    def plot_tracer_outputs(
        self, 
        pred_tensor, 
        targ_tensor,
        save_path="tracer_outputs.png",
        vmin=None, 
        vmax=None,
        channel_names=None
    ):
        """
        Compare tracer prediction and target by plotting channels.

        Args:
            pred_tensor (torch.Tensor): model prediction [B, C, H, W]
            targ_tensor (torch.Tensor): target tensor same shape
            save_path (str): output figure path
            vmin (float): minimum value for colorbar
            vmax (float): maximum value for colorbar
            channel_names (list): names of tracer channels (e.g., ['O3', 'CO', 'CH4'])
        """
        import matplotlib.pyplot as plt

        pred_tensor = pred_tensor.detach().cpu()
        targ_tensor = targ_tensor.detach().cpu()

        B, C, H, W = pred_tensor.shape
        channels_to_plot = min(C, 4)  # Plot up to 4 channels

        fig, axes = plt.subplots(
            2, channels_to_plot,
            figsize=(channels_to_plot * 4, 8)
        )
        
        # Handle single channel case
        if channels_to_plot == 1:
            axes = axes.reshape(2, 1)

        for i in range(channels_to_plot):
            ax_pred = axes[0, i]
            ax_true = axes[1, i]
            
            # Determine channel name
            if channel_names and i < len(channel_names):
                chan_name = channel_names[i]
            else:
                chan_name = f"Channel {i}"

            # Plot prediction
            im = ax_pred.imshow(
                pred_tensor[0, i],
                cmap='viridis',
                vmin=vmin, 
                vmax=vmax
            )
            ax_pred.set_title(f"Predicted {chan_name}")
            ax_pred.axis('off')

            # Plot target
            ax_true.imshow(
                targ_tensor[0, i],
                cmap='viridis',
                vmin=vmin, 
                vmax=vmax
            )
            ax_true.set_title(f"Target {chan_name}")
            ax_true.axis('off')

            # Add colorbar
            fig.colorbar(im, ax=[ax_pred, ax_true], shrink=0.6)

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Saved tracer outputs to {save_path}")


    def plot_tracer_difference(
        self,
        pred_tensor,
        targ_tensor,
        save_path="tracer_difference.png",
        channel_names=None
    ):
        """
        Plot the difference between prediction and target for tracers.
        
        Args:
            pred_tensor (torch.Tensor): model prediction [B, C, H, W]
            targ_tensor (torch.Tensor): target tensor same shape
            save_path (str): output figure path
            channel_names (list): names of tracer channels
        """
        import matplotlib.pyplot as plt
        import numpy as np

        pred_tensor = pred_tensor.detach().cpu()
        targ_tensor = targ_tensor.detach().cpu()
        
        # Calculate difference
        diff = pred_tensor - targ_tensor

        B, C, H, W = pred_tensor.shape
        channels_to_plot = min(C, 4)

        fig, axes = plt.subplots(
            1, channels_to_plot,
            figsize=(channels_to_plot * 4, 4)
        )
        
        if channels_to_plot == 1:
            axes = [axes]

        for i in range(channels_to_plot):
            # Determine channel name
            if channel_names and i < len(channel_names):
                chan_name = channel_names[i]
            else:
                chan_name = f"Channel {i}"
            
            # Plot difference with diverging colormap
            diff_max = max(abs(diff[0, i].min()), abs(diff[0, i].max()))
            im = axes[i].imshow(
                diff[0, i],
                cmap='RdBu_r',  # Red-Blue diverging
                vmin=-diff_max,
                vmax=diff_max
            )
            axes[i].set_title(f"Pred - Target\n{chan_name}")
            axes[i].axis('off')
            
            # Add colorbar
            plt.colorbar(im, ax=axes[i], shrink=0.8)

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Saved tracer difference to {save_path}")


    def plot_tracer_rollout(
        self,
        pred_list,
        targ_list,
        save_path="tracer_rollout.png",
        channel_idx=0,
        channel_name=None
    ):
        """
        Plot tracer predictions over multiple timesteps (rollout).
        
        Args:
            pred_list (list): List of prediction tensors, one per timestep
            targ_list (list): List of target tensors, one per timestep
            save_path (str): output figure path
            channel_idx (int): which tracer channel to plot
            channel_name (str): name of the tracer channel
        """
        import matplotlib.pyplot as plt

        n_steps = len(pred_list)
        
        fig, axes = plt.subplots(2, n_steps, figsize=(n_steps * 3, 6))
        
        if n_steps == 1:
            axes = axes.reshape(2, 1)
        
        vmin = min(targ_list[0][0, channel_idx].min().item() for _ in range(1))
        vmax = max(targ_list[0][0, channel_idx].max().item() for _ in range(1))

        for step in range(n_steps):
            pred = pred_list[step].detach().cpu()
            targ = targ_list[step].detach().cpu()
            
            # Plot prediction
            axes[0, step].imshow(
                pred[0, channel_idx],
                cmap='viridis',
                vmin=vmin,
                vmax=vmax
            )
            axes[0, step].set_title(f"Pred t={step}")
            axes[0, step].axis('off')
            
            # Plot target
            im = axes[1, step].imshow(
                targ[0, channel_idx],
                cmap='viridis',
                vmin=vmin,
                vmax=vmax
            )
            axes[1, step].set_title(f"Target t={step}")
            axes[1, step].axis('off')
        
        # Add colorbar
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
        
        if channel_name:
            fig.suptitle(f"Rollout: {channel_name}", fontsize=16)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Saved tracer rollout to {save_path}")
