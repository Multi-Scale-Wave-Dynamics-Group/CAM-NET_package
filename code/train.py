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

import argparse
import logging
import os
import re

import torch
from torch.distributed import is_initialized as dist_is_initialized

from makani import Trainer
from makani.utils import comm, logging_utils
from makani.utils.YParams import YParams
from makani.utils.parse_dataset_metada import parse_dataset_metadata
from makani.utils.parse_tracer_metada import parse_tracer_metadata
from makani.utils.tracer_trainer import TracerTrainer


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description="Makani training launcher")

    parser.add_argument(
        "--training_mode",
        default="auto",
        choices=["auto", "physical", "tracer"],
        help="Select original physical backbone training or tracer finetuning.",
    )

    parser.add_argument("--fin_parallel_size", default=1, type=int, help="Input feature parallelization")
    parser.add_argument("--fout_parallel_size", default=1, type=int, help="Output feature parallelization")
    parser.add_argument("--h_parallel_size", default=1, type=int, help="Spatial parallelism dimension in h")
    parser.add_argument("--w_parallel_size", default=1, type=int, help="Spatial parallelism dimension in w")
    parser.add_argument(
        "--parameters_reduction_buffer_count",
        default=1,
        type=int,
        help="How many buffers will be used approximately for weight gradient reductions.",
    )
    parser.add_argument("--run_num", default="00", type=str)
    parser.add_argument("--yaml_config", default="./config/sfnonet.yaml", type=str)
    parser.add_argument("--config", default="base_73chq", type=str)
    parser.add_argument("--batch_size", default=-1, type=int, help="Override batch size in the configuration file.")
    parser.add_argument("--pretrained_checkpoint_path", default=None, type=str)
    parser.add_argument("--enable_synthetic_data", action="store_true")
    parser.add_argument(
        "--amp_mode",
        default="none",
        type=str,
        choices=["none", "fp16", "bf16"],
        help="Specify the mixed precision mode which should be used.",
    )
    parser.add_argument(
        "--jit_mode",
        default="none",
        type=str,
        choices=["none", "script", "inductor"],
        help="Specify if and how to use torch jit.",
    )
    parser.add_argument(
        "--cuda_graph_mode",
        default="none",
        type=str,
        choices=["none", "fwdbwd", "step"],
        help="Specify which parts to capture under cuda graph.",
    )
    parser.add_argument("--enable_odirect", action="store_true")
    parser.add_argument("--checkpointing_level", default=0, type=int, help="How aggressively checkpointing is used")
    parser.add_argument("--epsilon_factor", default=0, type=float)
    parser.add_argument("--split_data_channels", action="store_true")
    parser.add_argument("--print_timings_frequency", default=50, type=int)
    parser.add_argument("--num_data_workers", default=None, type=int)
    parser.add_argument("--num_visualization_workers", default=None, type=int)
    parser.add_argument("--log_video", default=None, type=int)
    parser.add_argument("--valid_autoreg_steps", default=None, type=int)
    parser.add_argument("--skip_validation", action="store_true")
    parser.add_argument("--mode", default="train", type=str, choices=["train", "test"])

    parser.add_argument("--save_checkpoint", default="legacy", choices=["none", "flexible", "legacy"], type=str)
    parser.add_argument("--load_checkpoint", default="legacy", choices=["flexible", "legacy"], type=str)
    parser.add_argument("--multistep_count", default=1, type=int)

    parser.add_argument("--enable_benchy", action="store_true")
    parser.add_argument("--disable_ddp", action="store_true")
    parser.add_argument("--enable_grad_anomaly_detection", action="store_true")

    args = parser.parse_args(argv)
    if args.batch_size > 0 and args.batch_size < 1:
        raise ValueError("Batch size must be a positive integer or -1 to use the config value.")
    return args


def normalize_checkpoint_template(path):
    """Accept a checkpoint file, a template, or a directory of rank checkpoints."""
    if path is None:
        return None
    path = os.path.abspath(os.path.expanduser(path))
    if path.endswith(".tar") or "{mp_rank}" in path:
        return path
    return os.path.join(path, "ckpt_mp{mp_rank}.tar")


def resolve_training_mode(args, params):
    if args.training_mode != "auto":
        return args.training_mode
    return "tracer" if params.get("tracer_finetune", False) else "physical"


def substitute_variables(params):
    params_dict = params.to_dict()
    substituted = {}
    for key, value in params_dict.items():
        if isinstance(value, str):
            substituted[key] = re.sub(
                r"\$\{(\w+)\}",
                lambda match: str(params_dict.get(match.group(1), match.group(0))),
                value,
            )
        else:
            substituted[key] = value

    for key, value in substituted.items():
        if isinstance(value, str) and value != params_dict.get(key):
            params[key] = value


def apply_runtime_overrides(params, args, training_mode):
    params["training_mode"] = training_mode
    params["tracer_finetune"] = training_mode == "tracer"
    params["epsilon_factor"] = args.epsilon_factor

    params["fin_parallel_size"] = args.fin_parallel_size
    params["fout_parallel_size"] = args.fout_parallel_size
    params["h_parallel_size"] = args.h_parallel_size
    params["w_parallel_size"] = args.w_parallel_size
    params["model_parallel_sizes"] = [
        args.h_parallel_size,
        args.w_parallel_size,
        args.fin_parallel_size,
        args.fout_parallel_size,
    ]
    params["model_parallel_names"] = ["h", "w", "fin", "fout"]
    params["parameters_reduction_buffer_count"] = args.parameters_reduction_buffer_count

    params["load_checkpoint"] = args.load_checkpoint
    params["save_checkpoint"] = args.save_checkpoint
    params["amp_mode"] = args.amp_mode
    params["jit_mode"] = args.jit_mode
    params["cuda_graph_mode"] = args.cuda_graph_mode
    params["skip_validation"] = args.skip_validation
    params["enable_odirect"] = args.enable_odirect
    params["checkpointing"] = args.checkpointing_level
    params["enable_synthetic_data"] = args.enable_synthetic_data
    params["split_data_channels"] = args.split_data_channels
    params["print_timings_frequency"] = args.print_timings_frequency
    if args.num_data_workers is not None:
        params["num_data_workers"] = args.num_data_workers
    if args.num_visualization_workers is not None:
        params["num_visualization_workers"] = args.num_visualization_workers
    if args.log_video is not None:
        params["log_video"] = args.log_video
    if args.valid_autoreg_steps is not None:
        params["valid_autoreg_steps"] = args.valid_autoreg_steps
    params["multistep_count"] = args.multistep_count
    params["n_future"] = args.multistep_count - 1

    params["enable_benchy"] = args.enable_benchy
    params["disable_ddp"] = args.disable_ddp
    params["enable_grad_anomaly_detection"] = args.enable_grad_anomaly_detection
    params["predict_delta"] = params.get("predict_delta", False)

    params["pretrained_checkpoint_path"] = args.pretrained_checkpoint_path or params.get(
        "pretrained_checkpoint_path", None
    )


def validate_distributed_settings(params):
    if params["global_batch_size"] % comm.get_size("data") != 0:
        raise ValueError(
            f"Cannot evenly distribute batch size {params['global_batch_size']} "
            f"across {comm.get_size('data')} data-parallel ranks."
        )
    params["batch_size"] = int(params["global_batch_size"] // comm.get_size("data"))

    if "optimizer_max_grad_norm" not in params:
        params["optimizer_max_grad_norm"] = 1.0


def setup_device():
    if torch.cuda.is_available():
        torch.cuda.set_device(comm.get_local_rank())
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def setup_experiment(params, args, training_mode, world_rank):
    exp_dir = os.path.join(params.exp_dir, args.config, str(args.run_num))
    if world_rank == 0:
        os.makedirs(exp_dir, exist_ok=True)
        os.makedirs(os.path.join(exp_dir, "training_checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(exp_dir, "wandb"), exist_ok=True)

    checkpoint_template = os.path.join(exp_dir, "training_checkpoints/ckpt_mp{mp_rank}.tar")
    params["experiment_dir"] = os.path.abspath(exp_dir)
    params["checkpoint_path"] = checkpoint_template
    params["best_checkpoint_path"] = os.path.join(exp_dir, "training_checkpoints/best_ckpt_mp{mp_rank}.tar")

    if training_mode == "tracer":
        params["tracer_checkpoint_path"] = checkpoint_template

    if not hasattr(params, "wandb_dir") or params["wandb_dir"] is None:
        params["wandb_dir"] = exp_dir

    return exp_dir


def checkpoint_files_exist(checkpoint_path, checkpoint_mode):
    resuming = True
    for mp_rank in range(comm.get_size("model")):
        checkpoint_fname = checkpoint_path.format(mp_rank=mp_rank)
        if checkpoint_mode == "legacy" or mp_rank < 1:
            resuming = resuming and os.path.isfile(checkpoint_fname)
    return resuming


def configure_resume(params, training_mode):
    if training_mode == "physical":
        params["resuming"] = checkpoint_files_exist(params.checkpoint_path, params.load_checkpoint)
    else:
        params["resuming"] = bool(params.get("resuming", False))


def configure_logging(params, exp_dir, world_rank):
    params["log_to_wandb"] = (world_rank == 0) and params["log_to_wandb"]
    params["log_to_screen"] = (world_rank == 0) and params["log_to_screen"]

    if world_rank == 0:
        logging_utils.config_logger()
        logging_utils.log_to_file(logger_name=None, log_filename=os.path.join(exp_dir, "out.log"))
        logging_utils.log_versions()
        logging.getLogger().info(f"Training mode: {params.training_mode}")
        logging.getLogger().info(f"Experiment directory: {exp_dir}")
        params.log(logging.getLogger())


def parse_metadata(params, training_mode):
    if "metadata_json_path" not in params:
        raise RuntimeError("Please specify metadata_json_path in the config.")

    params, _ = parse_dataset_metadata(params["metadata_json_path"], params=params)

    if training_mode == "tracer":
        if "tracer_metadata_json_path" not in params:
            raise RuntimeError("Tracer training requires tracer_metadata_json_path in the config.")
        params, _ = parse_tracer_metadata(params["tracer_metadata_json_path"], params=params)

    return params


def finalize_checkpoint_paths(params, training_mode):
    params["pretrained_checkpoint_path"] = normalize_checkpoint_template(
        params.get("pretrained_checkpoint_path", None)
    )

    if training_mode == "tracer" and params["pretrained_checkpoint_path"] is None:
        raise ValueError(
            "--training_mode=tracer requires --pretrained_checkpoint_path "
            "or pretrained_checkpoint_path in the config."
        )


def main(argv=None):
    try:
        args = parse_arguments(argv)
        params = YParams(os.path.abspath(args.yaml_config), args.config)
        training_mode = resolve_training_mode(args, params)
        apply_runtime_overrides(params, args, training_mode)

        comm.init(
            model_parallel_sizes=params["model_parallel_sizes"],
            model_parallel_names=params["model_parallel_names"],
            verbose=False,
        )
        world_rank = comm.get_world_rank()

        params["world_size"] = comm.get_world_size()
        if args.batch_size > 0:
            params.batch_size = args.batch_size
        params["global_batch_size"] = params.batch_size
        validate_distributed_settings(params)

        setup_device()

        if args.enable_grad_anomaly_detection:
            torch.autograd.set_detect_anomaly(True)

        exp_dir = setup_experiment(params, args, training_mode, world_rank)
        configure_resume(params, training_mode)
        configure_logging(params, exp_dir, world_rank)

        substitute_variables(params)
        finalize_checkpoint_paths(params, training_mode)
        params = parse_metadata(params, training_mode)

        if args.mode == "train":
            trainer_cls = TracerTrainer if training_mode == "tracer" else Trainer
            trainer = trainer_cls(params, world_rank)
            trainer.train()
        elif args.mode == "test":
            if training_mode != "physical":
                raise ValueError("--mode=test is only supported for --training_mode=physical")
            params["nettype"] = "DebugNet"
            trainer = Trainer(params, world_rank)
            trainer.test_autoregression_pipeline()
        else:
            raise ValueError(f"Unknown mode {args.mode}")
    except Exception as exc:
        if dist_is_initialized() and comm.get_world_rank() == 0:
            logging.getLogger().error(f"Training failed: {exc}", exc_info=True)
        elif not dist_is_initialized():
            logging.error(f"Training failed: {exc}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
