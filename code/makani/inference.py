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
import numpy as np
import argparse
import torch
import logging
# utilities
from makani.utils import logging_utils
from makani.utils.YParams import YParams
# distributed computing stuff
from makani.utils import comm
# import trainer
from makani.utils.parse_dataset_metada import parse_dataset_metadata
from makani.utils.parse_tracer_metada import parse_tracer_metadata
from makani import Inferencer
from makani.utils.tracer_inferencer import TracerInferencer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fin_parallel_size", default=1, type=int, help="Input feature paralellization")
    parser.add_argument("--fout_parallel_size", default=1, type=int, help="Output feature paralellization")
    parser.add_argument("--h_parallel_size", default=1, type=int, help="Spatial parallelism dimension in h")
    parser.add_argument("--w_parallel_size", default=1, type=int, help="Spatial parallelism dimension in w")
    parser.add_argument("--run_num", default="00", type=str)
    parser.add_argument("--yaml_config", default="./config/sfnonet.yaml", type=str)
    parser.add_argument("--config", default="base_73chq", type=str)
    parser.add_argument("--batch_size", default=-1, type=int, help="Switch for overriding batch size in the configuration file.")
    parser.add_argument("--tracer_checkpoint_path", default=None, type=str)
    parser.add_argument("--pretrained_checkpoint_path", default=None, type=str, help="Path to the pretrained checkpoint to use for inference.")
    parser.add_argument("--enable_synthetic_data", action="store_true")
    parser.add_argument("--amp_mode", default="none", type=str, choices=["none", "fp16", "bf16"], help="Specify the mixed precision mode which should be used.")
    parser.add_argument("--jit_mode", default="none", type=str, choices=["none", "script", "inductor"], help="Specify if and how to use torch jit.")
    parser.add_argument("--cuda_graph_mode", default="none", type=str, choices=["none", "fwdbwd", "step"], help="Specify which parts to capture under cuda graph")
    parser.add_argument("--enable_benchy", action="store_true")
    parser.add_argument("--disable_ddp", action="store_true")
    parser.add_argument("--enable_nhwc", action="store_true")
    parser.add_argument("--checkpointing_level", default=0, type=int, help="How aggressively checkpointing is used")
    parser.add_argument("--epsilon_factor", default=0, type=float)
    parser.add_argument("--split_data_channels", action="store_true")
    parser.add_argument("--mode", default="score", type=str, choices=["score", "ensemble","single"], help="Select inference mode")
    parser.add_argument("--enable_odirect", action="store_true")
    parser.add_argument("--single_time",type=str,default=None,help="Run inference for a single time stamp, e.g., '2014-07-10T12:00:00'.")
    parser.add_argument("--checkpoint_format", default="legacy", choices=["legacy", "flexible"], type=str, help="Format in which to load checkpoints.")

    # parse
    args = parser.parse_args()

    # parse parameters
    params = YParams(os.path.abspath(args.yaml_config), args.config)
    params["epsilon_factor"] = args.epsilon_factor

    # distributed
    params["fin_parallel_size"] = args.fin_parallel_size
    params["fout_parallel_size"] = args.fout_parallel_size
    params["h_parallel_size"] = args.h_parallel_size
    params["w_parallel_size"] = args.w_parallel_size

    params["model_parallel_sizes"] = [args.h_parallel_size, args.w_parallel_size, args.fin_parallel_size, args.fout_parallel_size]
    params["model_parallel_names"] = ["h", "w", "fin", "fout"]

    # checkpoint format
    params["load_checkpoint"] = params["save_checkpoint"] = args.checkpoint_format

    # make sure to reconfigure logger after the pytorch distributed init
    comm.init(model_parallel_sizes=params["model_parallel_sizes"],
              model_parallel_names=params["model_parallel_names"],
              verbose=False)
    world_rank = comm.get_world_rank()

    # update parameters
    params["world_size"] = comm.get_world_size()
    if args.batch_size > 0:
        params.batch_size = args.batch_size
    params["global_batch_size"] = params.batch_size
    assert params["global_batch_size"] % comm.get_size("data") == 0, f"Error, cannot evenly distribute {params['global_batch_size']} across {comm.get_size('data')} GPU."
    params["batch_size"] = int(params["global_batch_size"] // comm.get_size("data"))

    # set device
    torch.cuda.set_device(comm.get_local_rank())
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Set up directory
    expDir = os.path.join(params.exp_dir, args.config, str(args.run_num))
    if world_rank == 0:
        logging.info(f"writing output to {expDir}")
        if not os.path.isdir(expDir):
            os.makedirs(expDir, exist_ok=True)
            os.makedirs(os.path.join(expDir, "deterministic_scores"), exist_ok=True)
            os.makedirs(os.path.join(expDir, "deterministic_scores", "wandb"), exist_ok=True)

    params["experiment_dir"] = os.path.abspath(expDir)
    if args.tracer_checkpoint_path is None:
        params["tracer_checkpoint_path"] = os.path.join(expDir, "tracer_checkpoint_path/ckpt_mp{mp_rank}.tar")
        params["best_checkpoint_path"] = os.path.join(expDir, "training_checkpoints/best_ckpt_mp{mp_rank}.tar")
    else:
        params["tracer_checkpoint_path"] = os.path.join(args.tracer_checkpoint_path, "ckpt_mp{mp_rank}.tar")
        params["best_checkpoint_path"] = os.path.join(args.tracer_checkpoint_path, "best_ckpt_mp{mp_rank}.tar")
    
    if args.pretrained_checkpoint_path is None and params['tracer_finetune'] == False :
        params["pretrained_checkpoint_path"] = os.path.join(expDir, "training_checkpoints/ckpt_mp{mp_rank}.tar")
        params["best_pretrained_checkpoint_path"] = os.path.join(expDir, "training_checkpoints/best_ckpt_mp{mp_rank}.tar")
    else:
        params["best_pretrained_checkpoint_path"] = os.path.join(args.pretrained_checkpoint_path, "best_ckpt_mp{mp_rank}.tar")
        params["pretrained_checkpoint_path"] = os.path.join(args.pretrained_checkpoint_path, "ckpt_mp{mp_rank}.tar")
        # check if all files are there - do not comment out.  
    for mp_rank in range(comm.get_size("model")):
        checkpoint_fname = params.pretrained_checkpoint_path.format(mp_rank=mp_rank)
        if params["load_checkpoint"] == "legacy" or mp_rank < 1:
            assert os.path.isfile(checkpoint_fname)
    if params['tracer_finetune'] == True:
        for mp_rank in range(comm.get_size("model")):
            checkpoint_fname = params.tracer_checkpoint_path.format(mp_rank=mp_rank)
            if params["load_checkpoint"] == 'legacy' or mp_rank < 1:
                assert os.path.isfile(checkpoint_fname)
                
    params["resuming"] = True
    params["amp_mode"] = args.amp_mode
    params["jit_mode"] = args.jit_mode
    params["cuda_graph_mode"] = args.cuda_graph_mode
    params["enable_odirect"] = args.enable_odirect
    params["enable_benchy"] = args.enable_benchy
    params["disable_ddp"] = args.disable_ddp
    params["enable_nhwc"] = args.enable_nhwc
    params["checkpointing"] = args.checkpointing_level
    params["enable_synthetic_data"] = args.enable_synthetic_data
    params["split_data_channels"] = args.split_data_channels
    params["n_future"] = 0

    # wandb configuration
    if params["wandb_name"] is None:
        params["wandb_name"] = args.config + "_inference_" + str(args.run_num)
    if params["wandb_group"] is None:
        params["wandb_group"] = "makani" + args.config
    if not hasattr(params, "wandb_dir") or params["wandb_dir"] is None:
        params["wandb_dir"] = os.path.join(expDir, "deterministic_scores")

    if world_rank == 0:
        logging_utils.config_logger()
        logging_utils.log_to_file(logger_name=None, log_filename=os.path.join(expDir, "out.log"))
        logging_utils.log_versions()
        params.log(logging.getLogger())

    params["log_to_wandb"] = (world_rank == 0) and params["log_to_wandb"]
    params["log_to_screen"] = (world_rank == 0) and params["log_to_screen"]

    # Helper function to substitute variables in path strings
    def substitute_variables(path_str, params_dict):
        """Substitute ${variable} patterns in path strings with actual values from params."""
        if not isinstance(path_str, str):
            return path_str
        
        import re
        # Find all ${variable} patterns
        pattern = r'\$\{(\w+)\}'
        matches = re.findall(pattern, path_str)
        
        result = path_str
        for var_name in matches:
            if var_name in params_dict:
                result = result.replace(f'${{{var_name}}}', str(params_dict[var_name]))
        
        return result
    
    # Substitute variables in all path parameters
    path_keys = [
        'metadata_json_path', 'train_data_path', 'valid_data_path',
        'train_tracer_data_path', 'valid_tracer_data_path',
        'min_path', 'max_path', 'time_means_path', 'global_means_path',
        'global_stds_path', 'time_diff_means_path', 'time_diff_stds_path',
        'inf_data_path', 'tracer_inf_data_path', 'orography_path', 'landmask_path',
        'tracer_metadata_json_path'
    ]
    
    for key in path_keys:
        if key in params:
            params[key] = substitute_variables(params[key], params)

    # parse dataset metadata
    if "metadata_json_path" in params:
        params, _ = parse_dataset_metadata(params["metadata_json_path"], params=params)
    else:
        raise RuntimeError(f"Error, please specify a dataset descriptor file for physical variables in json format")
    
    if params["tracer_finetune"]:
        if "tracer_metadata_json_path" in params:
            params, _ = parse_tracer_metadata(params["tracer_metadata_json_path"], params=params)
        else:
            raise RuntimeError(f"Error, please specify a dataset descriptor file for tracer variables in json format")

    # instantiate trainer / inference / ensemble object
    if args.mode == "score":
        if not params["tracer_finetune"]:
            inferencer = Inferencer(params, world_rank)
            inferencer.score_model()
        if params["tracer_finetune"]:
            tracer_inferencer= TracerInferencer(params, world_rank)
            tracer_inferencer.score_model()
    elif args.mode == "single":
        if args.single_time is None:
            raise ValueError("Please specify --single_time YYYY-MM-DDThh:mm:ss")

        if params["tracer_finetune"]:
            params["multifiles"] = True
            tracer_inferencer = TracerInferencer(params, world_rank)
            tracer_inferencer.inference_single(single_time=args.single_time, compute_metrics=True, output_data=True)
        else:
            inferencer = Inferencer(params, world_rank)
            inferencer.inference_single(single_time=args.single_time, compute_metrics=True, output_data=True)
    else:
        raise ValueError(f"Unknown training mode {args.mode}")
