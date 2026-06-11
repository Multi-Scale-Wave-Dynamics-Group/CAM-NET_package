# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time

import numpy as np
from tqdm import tqdm
import pynvml

import torch
import torch.cuda.amp as amp
import torch.distributed as dist

# from torch.nn.parallel import DistributedDataParallel

import logging
import wandb

from makani.utils.dataloader import get_dataloader
from makani.utils.trainer import Trainer
from makani.utils.losses import LossHandler
from makani.utils.metric import MetricsHandler

from makani.models import model_registry

# distributed computing stuff
from makani.utils import comm
from makani.utils import visualize


class Inferencer(Trainer):
    """
    Inferencer class holding all the necessary information to perform inference. Design is similar to Trainer, however only keeping the necessary information.
    """

    def __init__(self, params, world_rank):
        # init the trainer
        # super().__init__(params, world_rank, job_type="inference")

        self.params = None
        self.world_rank = world_rank

        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
        else:
            self.device = torch.device("cpu")

        # get logger
        if params.log_to_screen:
            self.logger = logging.getLogger()

        # nvml stuff
        if params.log_to_screen:
            pynvml.nvmlInit()
            self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(self.device.index)

        # set amp_parameters
        if hasattr(params, "amp_mode") and (params.amp_mode != "none"):
            self.amp_enabled = True
            if params.amp_mode == "fp16":
                self.amp_dtype = torch.float16
            elif params.amp_mode == "bf16":
                self.amp_dtype = torch.bfloat16
            else:
                raise ValueError(f"Unknown amp mode {params.amp_mode}")

            if params.log_to_screen:
                self.logger.info(f"Enabling automatic mixed precision in {params.amp_mode}.")
        else:
            self.amp_enabled = False
            self.amp_dtype = torch.float32

        # resuming needs is set to False so loading checkpoints does not attempt to set the optimizer state
        if hasattr(params, "log_to_wandb") and params.log_to_wandb:
            # login first:
            wandb.login()
            # init
            wandb.init(
                dir=params.experiment_dir,
                config=params,
                name=params.wandb_name,  # if not params.resuming else None,
                group=params.wandb_group,  # if not params.resuming else None,
                project=params.wandb_project,
                entity=params.wandb_entity,
                resume=params.resuming,
            )

        # data loader
        if params.log_to_screen:
            self.logger.info("initializing data loader")

        if not hasattr(params, "multifiles"):
            params["multifiles"] = True
        if not hasattr(params, "enable_synthetic_data"):
            params["enable_synthetic_data"] = False
        if not hasattr(params, "amp"):
            params["enable_synthetic_data"] = False

        # although it is called validation dataloader here, the file path is taken from inf_data_path to perform inference on the
        # out of sample dataset
        self.valid_dataloader, self.valid_dataset = get_dataloader(params, params.inf_data_path, train=False, final_eval=True, device=self.device)
        out_bias, out_scale = self.valid_dataloader.get_output_normalization()
        self.bias = out_bias[0,...]
        self.scale = out_scale[0,...]
        self.lat = self.valid_dataset.grid_converter.get_dst_coords()[0].cpu().numpy()
        self.lon=self.valid_dataset.grid_converter.get_dst_coords()[1].cpu().numpy() 
        self.time_stamps = self.valid_dataset.get_sample_times()
        print(f"✅ Loaded {len(self.time_stamps)} timestamps from dataset")
        print("Dataset first 5 times:", self.time_stamps[:5])
        print("Dataset last 5 times:", self.time_stamps[-5:])
        print("Dataset length:", len(self.time_stamps))

        if params.log_to_screen:
            self.logger.info("data loader initialized")

        # update params
        params = self._update_parameters(params)

        # save params
        self.params = params

        self.model = model_registry.get_model(params).to(self.device)
        self.preprocessor = self.model.preprocessor

        # print model
        if self.world_rank == 0:
            print(self.model)

        self.restore_checkpoint(params.pretrained_checkpoint_path, checkpoint_mode=params.load_checkpoint)

        # metrics handler
        mult_cpu, clim = self._get_time_stats()
        self.metrics = MetricsHandler(self.params, mult_cpu, clim, self.device)
        self.metrics.initialize_buffers()

        # loss handler
        self.loss_obj = LossHandler(self.params)
        self.loss_obj = self.loss_obj.to(self.device)

    def _autoregressive_inference(self, data, compute_metrics=False, output_data=False, output_channels=[0, 1]):
        # map to gpu
        gdata = map(lambda x: x.to(self.device, dtype=torch.float32), data)
        # preprocess
        inp, tar = self.preprocessor.cache_unpredicted_features(*gdata)
        inp = self.preprocessor.flatten_history(inp)
        self.pred_outputs = []
        self.targ_outputs = []
        # split list of targets
        tarlist = torch.split(tar, 1, dim=1)
        # do autoregression
        inpt = inp
        for idt, targ in enumerate(tarlist):
            targ = self.preprocessor.flatten_history(targ)

            # FW pass
            with amp.autocast(enabled=self.amp_enabled, dtype=self.amp_dtype):
                pred = self.model(inpt)
                loss = self.loss_obj(pred, targ, inpt)

            # put in the metrics handler
            if compute_metrics:
                self.metrics.update(pred, targ, loss, idt)
            if output_data:
                pred_denorm = self.scale * pred[0,:].cpu().numpy() + self.bias
                targ_denorm = self.scale * targ[0,:].cpu().numpy() + self.bias
                # Add an extra dimension to pred_denorm and targ_denorm before appending
                pred_denorm_expanded = np.expand_dims(pred_denorm, axis=0)  # Add a new axis at the end
                targ_denorm_expanded = np.expand_dims(targ_denorm, axis=0)  # Add a new axis at the end

                # Append the expanded arrays to the list
                self.pred_outputs.append(pred_denorm_expanded)
                self.targ_outputs.append(targ_denorm_expanded)
            # append history
            inpt = self.preprocessor.append_history(inpt, pred, idt)
        self.pred_output_concat = np.concatenate(self.pred_outputs, axis=0)
        self.targ_output_concat = np.concatenate(self.targ_outputs, axis=0)
        print(f"Shape of pred_output_concat: {self.pred_output_concat.shape}")
        return

    def inference_single(self, single_time, compute_metrics=False, output_data=False, output_channels=[0, 1]):
        """
        Runs the model in autoregressive inference mode on a single initial condition.
        For best results, run with --nproc_per_node=1 to avoid distributed synchronization.
        """
        # Check if we're in a multi-GPU setup (which we shouldn't be for single mode)
        world_size = comm.get_world_size() if dist.is_initialized() else 1
        if world_size > 1:
            print(f"⚠️  WARNING: Running single inference with {world_size} GPUs detected!")
            print(f"⚠️  This may cause NCCL timeout issues. Please run with --nproc_per_node=1")
            print(f"⚠️  Only rank 0 will perform inference, other ranks will exit.")
            
            # For ranks other than 0, exit immediately
            if self.world_rank != 0:
                print(f"⏸️  Rank {self.world_rank} exiting - only rank 0 performs single inference")
                return
        
       # 🔹 Always refresh timestamps from dataset (so each run uses fresh times)
        self.time_stamps = np.array(self.valid_dataset.get_sample_times(), dtype="datetime64[h]")

        # 🔹 Convert input to matching precision
        if isinstance(single_time, str):
            single_time = np.datetime64(single_time, "h")
        else:
            single_time = np.datetime64(single_time, "h")

        # 🔹 Compute time difference and find nearest index
        time_diffs = np.abs(self.time_stamps - single_time).astype("timedelta64[h]").astype(float)
        ic = int(np.argmin(time_diffs))
        matched_time = self.time_stamps[ic]
        time_diff_hours = time_diffs[ic]

        print(f"🕓 Requested time: {single_time}")
        print(f"📅 Closest dataset time: {matched_time} (index {ic})")
        print(f"⏱️  Time difference: {time_diff_hours:.2f} hours")
        # 🧭 (Optional debug)
        print(f"Dataset time range: {self.time_stamps[0]} → {self.time_stamps[-1]}")
        print(f"🕓 Requested time: {single_time}")
        print(f"📅 Closest dataset time: {matched_time} (index {ic})")
        print(f"⏱️  Time difference: {time_diff_hours:.2f} hours")
        # Warn if the match is not exact
        if time_diff_hours > 0.1:  # More than 6 minutes off
            print(f"⚠️  WARNING: No exact match found. Using closest available time.")
        
        self._set_eval()

        # clear cache
        torch.cuda.empty_cache()

        # initialize metrics buffers
        if compute_metrics:
            self.metrics.zero_buffers()
        if output_data:
            self.targ_outputs = []
            self.pred_outputs = []
        print(f"🔍 Loading data for time index {ic}...")
        with torch.inference_mode():
            with torch.no_grad():
                # Direct dataset access (no distributed logic for single inference)
                try:
                    # Verify the time coordinate matches the data
                    # Dataset __getitem__ at index i loads samples starting from i
                    # So time_stamps[ic] should be the time of the FIRST sample in history
                    print(f"📍 Time index {ic} corresponds to timestamp: {self.time_stamps[ic]}")
                    print(f"   This will be the FIRST timestep in the input history")
                    
                    # Show which timestamps will be loaded
                    n_history = self.params.n_history
                    dt = self.params.dt
                    history_indices = [ic + dt * offset for offset in range(n_history + 1)]
                    
                    print(f"   Input history will use indices: {history_indices}")
                    if all(idx < len(self.time_stamps) for idx in history_indices):
                        history_times = [self.time_stamps[idx] for idx in history_indices]
                        print(f"   Input history timestamps:")
                        for i, t in enumerate(history_times):
                            print(f"     t-{n_history-i}: {t}")
                    
                    data = self.valid_dataset[ic]
                    print(f"✅ Loaded time index {ic} via direct dataset access")
                    
                    # CHECK TIME SEQUENCE ORDER
                    print(f"\n🔍 VERIFYING TIME SEQUENCE ORDER:")
                    print(f"   Dataset loads samples in this order:")
                    print(f"   Index 0 (oldest):  time_stamps[{ic}] = {self.time_stamps[ic]}")
                    for offset in range(1, min(n_history + 1, 3)):  # Show first few
                        idx = ic + dt * offset
                        if idx < len(self.time_stamps):
                            print(f"   Index {offset}:           time_stamps[{idx}] = {self.time_stamps[idx]}")
                    if n_history > 2:
                        print(f"   ...")
                    idx_last = ic + dt * n_history
                    if idx_last < len(self.time_stamps):
                        print(f"   Index {n_history} (newest): time_stamps[{idx_last}] = {self.time_stamps[idx_last]}")
                    print(f"   → Data is loaded in CHRONOLOGICAL order (oldest to newest)")
                    print(f"   → For weather forecasting, newest time should be the initial condition\n")
                    
                    # Convert to list and add batch dimension
                    if not isinstance(data, (list, tuple)):
                        data = [data]
                    
                    single_sample_data = []
                    for i, tensor in enumerate(data):
                        # Dataset __getitem__ returns (time, channels, H, W)
                        # We need to add batch dimension since we're bypassing DataLoader
                        tensor = tensor.unsqueeze(0)  # (time, channels, H, W) → (1, time, channels, H, W)
                        tensor = tensor.to(self.device)
                        single_sample_data.append(tensor)
                        print(f"   Data tensor {i} shape after adding batch dim: {tensor.shape}")
                        print(f"   Time dimension has {tensor.shape[1]} timesteps")
                    
                    print(f"🚀 Running autoregressive inference for time: {matched_time}")
                    self._autoregressive_inference(single_sample_data, compute_metrics=compute_metrics, output_data=output_data, output_channels=output_channels)
                    
                    # Visualize and save results
                    visualize.plot_inference_comparison(
                        self.pred_output_concat, self.targ_output_concat, 
                        self.params, self.params.comparison_channels, 
                        lat=self.lat, lon=self.lon, cmap="twilight_shifted", 
                        projection="mollweide", diverging=True, figsize=(30, 20), 
                        vmax=None, title_str=f"Initial Condition: {matched_time}"
                    )
                    
                    # create final logs
                    if compute_metrics:
                        ogs, acc_curves, rmse_curves = self.metrics.finalize(final_inference=True)
                    
                    # Create output directory with timestamp
                    ic_dir = os.path.join(self.params.experiment_dir, f"ic_{ic}_{str(matched_time).replace(':', '-')}")
                    os.makedirs(ic_dir, exist_ok=True)
                    
                    # Save metadata about the initial condition
                    metadata = {
                        'requested_time': str(single_time),
                        'matched_time': str(matched_time),
                        'time_index': ic,
                        'time_diff_hours': float(time_diff_hours)
                    }
                    np.save(os.path.join(ic_dir, "metadata.npy"), metadata)
                    
                    if compute_metrics:
                        np.save(os.path.join(ic_dir, "acc_curves.npy"), acc_curves.cpu().numpy())
                        np.save(os.path.join(ic_dir, "rmse_curves.npy"), rmse_curves.cpu().numpy())
                    
                    np.save(os.path.join(ic_dir, "pred_outputs.npy"), self.pred_output_concat)
                    np.save(os.path.join(ic_dir, "targ_outputs.npy"), self.targ_output_concat)
                    print(f"💾 Saved predictions and targets to {ic_dir}")
                    
                    # visualize the result and log it to wandb
                    if compute_metrics:
                        visualize.plot_rollout_metrics(
                            acc_curves, rmse_curves, self.params, epoch=0, 
                            model_name=self.params.nettype, 
                            comparison_channels=self.params.comparison_channels
                        )
                
                except Exception as e:
                    print(f"❌ Error accessing dataset at index {ic}: {e}")
                    raise
        
        # Clean up: destroy process group on rank 0 as well
        if dist.is_initialized():
            try:
                dist.destroy_process_group()
                print(f"✅ Rank {self.world_rank} destroyed process group")
            except Exception as e:
                print(f"⚠️  Rank {self.world_rank} error destroying process group: {e}")

        return

    def inference_epoch(self):
        """
        Runs the model in autoregressive inference mode on the entire validation dataset. Computes metrics and scores the model.
        """

        # set to eval
        self._set_eval()

        # clear cache
        torch.cuda.empty_cache()

        # initialize metrics buffers
        self.metrics.zero_buffers()

        with torch.inference_mode():
            with torch.no_grad():
                eval_steps = 0
                for data in tqdm(self.valid_dataloader, desc="Scoring progress", disable=not self.params.log_to_screen):
                    eval_steps += 1
                    self._autoregressive_inference(data, compute_metrics=True, output_data=True, output_channels=False)
                    if eval_steps == 1:
                        visualize.plot_inference_comparison(self.pred_output_concat, self.targ_output_concat, self.params, self.params.comparison_channels, lat=self.lat, lon=self.lon, cmap="twilight_shifted", projection="mollweide", diverging=True, figsize=(30, 20), vmax=None, title_str=None)

        # create final logs
        logs, acc_curves, rmse_curves = self.metrics.finalize(final_inference=True)

        # save the acc curve
        if self.world_rank == 0:
            np.save(os.path.join(self.params.experiment_dir, "acc_curves.npy"), acc_curves.cpu().numpy())
            np.save(os.path.join(self.params.experiment_dir, "rmse_curves.npy"), rmse_curves.cpu().numpy())
            np.save(os.path.join(self.params.experiment_dir, "pred_outputs.npy"), self.pred_output_concat)
            np.save(os.path.join(self.params.experiment_dir, "targ_outputs.npy"), self.targ_output_concat)
            print(f"Saved predictions and targets to {self.params.experiment_dir}")
            # visualize the result and log it to wandb. The dummy epoch 0 is used for logging to wandb
            visualize.plot_rollout_metrics(acc_curves, rmse_curves, self.params, epoch=0, model_name=self.params.nettype, comparison_channels=self.params.comparison_channels)

        # global sync is in order
        if dist.is_initialized():
            dist.barrier()

        return logs

    def log_score(self, scoring_logs, scoring_time):
        # separator
        separator = "".join(["-" for _ in range(50)])
        print_prefix = "    "

        def get_pad(nchar):
            return "".join([" " for x in range(nchar)])

        if self.params.log_to_screen:
            # header:
            self.logger.info(separator)
            self.logger.info(f"Scoring summary:")
            self.logger.info("Total scoring time is {:.2f} sec".format(scoring_time))

            # compute padding:
            print_list = list(scoring_logs["metrics"].keys())
            max_len = max([len(x) for x in print_list])
            pad_len = [max_len - len(x) for x in print_list]
            # validation summary
            self.logger.info("Metrics:")
            for idk, key in enumerate(print_list):
                value = scoring_logs["metrics"][key]
                self.logger.info(f"{print_prefix}{key}: {get_pad(pad_len[idk])}{value}")
            self.logger.info(separator)

        return

    def score_model(self):
        # log parameters
        if self.params.log_to_screen:
            # log memory usage so far
            all_mem_gb = pynvml.nvmlDeviceGetMemoryInfo(self.nvml_handle).used / (1024.0 * 1024.0 * 1024.0)
            max_mem_gb = torch.cuda.max_memory_allocated(device=self.device) / (1024.0 * 1024.0 * 1024.0)
            self.logger.info(f"Scaffolding memory high watermark: {all_mem_gb} GB ({max_mem_gb} GB for pytorch)")
            # announce training start
            self.logger.info("Starting Scoring...")

        # perform a barrier here to make sure everybody is ready
        if dist.is_initialized():
            dist.barrier()

        try:
            torch.cuda.reset_peak_memory_stats(self.device)
        except ValueError:
            pass

        # start timer
        scoring_start = time.time()

        scoring_logs = self.inference_epoch()

        # end timer
        scoring_end = time.time()

        self.log_score(scoring_logs, scoring_end - scoring_start)

        return
    
