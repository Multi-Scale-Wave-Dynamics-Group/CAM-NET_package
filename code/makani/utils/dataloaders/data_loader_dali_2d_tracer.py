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

import logging
import glob
import torch
import random
import numpy as np
import torch
from torch import Tensor
import math

# import cv2

# DALI stuff
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as dali_types
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

# distributed stuff
import torch.distributed as dist
from makani.utils import comm

# es helper
import makani.utils.dataloaders.dali_es_helper_2d_tracer as esh
from makani.utils.grids import GridConverter


class ERA5DaliESDataloadertracer(object):
    def get_pipeline(self):
        pipeline = Pipeline(
            batch_size=self.batchsize, num_threads=2, device_id=self.device_index, py_num_workers=self.num_data_workers, py_start_method="spawn", seed=self.global_seed
        )

        img_shape_x = self.img_shape_x
        img_shape_y = self.img_shape_y
        in_channels = self.in_channels
        out_channels = self.out_channels

        with pipeline:
            # get input and target
            data = fn.external_source(
                source=self.extsource,
                num_outputs=6 if self.add_zenith else 4,
                layout=["FCHW", "FCHW", "FCHW", "FCHW", "FCHW", "FCHW"] if self.add_zenith else ["FCHW", "FCHW", "FCHW", "FCHW"],
                batch=False,
                no_copy=True,
                parallel=True,
                prefetch_queue_depth=self.num_data_workers,
            )

            if self.add_zenith:
                inp, tar, inp_tracer, tar_tracer, izen, tzen = data
            else:
                inp, tar, inp_tracer, tar_tracer = data            # upload to GPU
            inp = inp.gpu()
            tar = tar.gpu()
            inp_tracer = inp_tracer.gpu()
            tar_tracer = tar_tracer.gpu()
            if self.add_zenith:
                izen = izen.gpu()
                tzen = tzen.gpu()

            # roll if requested
            if self.train and self.roll:
                shift = fn.random.uniform(device="cpu", dtype=dali_types.INT32, range=[0, img_shape_y])
                inp = fn.cat(inp[:, :, :, shift:], inp[:, :, :, :shift], device="gpu", axis=3)
                tar = fn.cat(tar[:, :, :, shift:], tar[:, :, :, :shift], device="gpu", axis=3)
                inp_tracer = fn.cat(inp_tracer[:, :, :, shift:], inp_tracer[:, :, :, :shift], device="gpu", axis=3)
                tar_tracer = fn.cat(tar_tracer[:, :, :, shift:], tar_tracer[:, :, :, :shift], device="gpu", axis=3)
                if self.add_zenith:
                    izen = fn.cat(izen[:, :, :, shift:], izen[:, :, :, :shift], device="gpu", axis=3)
                    tzen = fn.cat(tzen[:, :, :, shift:], tzen[:, :, :, :shift], device="gpu", axis=3)

            # normalize if requested
            if self.normalize:
                inp = fn.normalize(inp, device="gpu", axis_names=self.norm_channels, batch=self.norm_batch, mean=self.in_bias, stddev=self.in_scale)
                tar = fn.normalize(tar, device="gpu", axis_names=self.norm_channels, batch=self.norm_batch, mean=self.out_bias, stddev=self.out_scale)
                inp_tracer = fn.normalize(inp_tracer, device="gpu", axis_names=self.norm_channels, batch=self.norm_batch, mean=self.in_bias_tracer, stddev=self.in_scale_tracer)
                tar_tracer = fn.normalize(tar_tracer, device="gpu", axis_names=self.norm_channels, batch=self.norm_batch, mean=self.out_bias_tracer, stddev=self.out_scale_tracer)
            # add noise if requested
            if self.add_noise:
                inp = fn.noise.gaussian(inp, device="gpu", stddev=self.noise_std, seed=self.local_seed)
                inp_tracer = fn.noise.gaussian(inp_tracer, device="gpu", stddev=self.noise_std, seed=self.local_seed)
            # add zenith angle if requested
            if self.add_zenith:
                pipeline.set_outputs(inp, tar, inp_tracer, tar_tracer, izen, tzen)
            else:
                pipeline.set_outputs(inp, tar, inp_tracer, tar_tracer)
        return pipeline

    def __init__(self, params, location_physical, location_tracer, train, seed=333, final_eval=False):
        self.num_data_workers = params.num_data_workers
        self.device_index = torch.cuda.current_device()
        self.device = torch.device(f"cuda:{self.device_index}")
        self.batchsize = int(params.batch_size)

        # set up seeds
        # this one is the same on all ranks
        self.global_seed = seed
        # this one is the same for all ranks of the same model
        model_id = comm.get_world_rank() // comm.get_size("model")
        self.model_seed = self.global_seed + model_id
        # this seed is supposed to be diffferent for every rank
        self.local_seed = self.global_seed + comm.get_world_rank()

        # we need to copy those
        self.location_physical = location_physical
        self.location_tracer = location_tracer
        self.train = train
        self.dt = params.dt
        self.dhours = params.dhours
        self.n_history = params.n_history
        self.n_future = params.n_future if train else params.valid_autoreg_steps
        self.in_channels = params.in_channels
        self.out_channels = params.out_channels
        self.tracer_in_channels = np.array(params.tracer_in_channels) if hasattr(params, "tracer_in_channels") else None
        self.tracer_out_channels = np.array(params.tracer_out_channels) if hasattr(params, "tracer_out_channels") else None
        self.n_tracer_in_channels = len(params.tracer_in_channels) if self.tracer_in_channels is not None else 0
        self.n_tracer_out_channels = len(params.tracer_out_channels) if self.tracer_out_channels is not None else 0
        self.add_noise = params.add_noise if train else False
        self.noise_std = params.noise_std
        self.add_zenith = params.add_zenith if hasattr(params, "add_zenith") else False
        if hasattr(params, "lat") and hasattr(params, "lon"):
            self.lat_lon = (params.lat, params.lon)
        else:
            self.lat_lon = None
        self.dataset_path = params.h5_path
        if train:
            self.n_samples = params.n_train_samples if hasattr(params, "n_train_samples") else None
            self.n_samples_per_epoch = params.n_train_samples_per_epoch if hasattr(params, "n_train_samples_per_epoch") else None
        else:
            self.n_samples = params.n_eval_samples if hasattr(params, "n_eval_samples") else None
            self.n_samples_per_epoch = params.n_eval_samples_per_epoch if hasattr(params, "n_eval_samples_per_epoch") else None

        if final_eval:
            self.n_samples = None
            self.n_samples_per_epoch = None

        # by default we normalize over space
        self.norm_channels = "FHW"
        self.norm_batch = False
        if hasattr(params, "normalization_mode"):
            split = params.data_normalization_mode.split("-")
            self.norm_mode = split[0]
            if len(split) > 1:
                self.norm_channels = split[1]
                if "B" in self.norm_channels:
                    self.norm_batch = True
                    self.norm_channels.replace("B", "")
        else:
            self.norm_mode = "offline"

        # set sharding
        self.num_shards = params.data_num_shards
        self.shard_id = params.data_shard_id

        # get cropping:
        crop_size = [params.crop_size_x if hasattr(params, "crop_size_x") else None, params.crop_size_y if hasattr(params, "crop_size_y") else None]
        crop_anchor = [params.crop_anchor_x if hasattr(params, "crop_anchor_x") else 0, params.crop_anchor_y if hasattr(params, "crop_anchor_y") else 0]

        # get the image sizes
        self.extsource = esh.GeneralEStracer(
            self.location_physical,
            self.location_tracer,
            max_samples=self.n_samples,
            samples_per_epoch=self.n_samples_per_epoch,
            train=self.train,
            batch_size=self.batchsize,
            dt=self.dt,
            dhours=self.dhours,
            n_history=self.n_history,
            n_future=self.n_future,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            tracer_in_channels=self.tracer_in_channels,
            tracer_out_channels=self.tracer_out_channels,
            crop_size=crop_size,
            crop_anchor=crop_anchor,
            num_shards=self.num_shards,
            shard_id=self.shard_id,
            io_grid=params.io_grid,
            io_rank=params.io_rank,
            device_id=self.device_index,
            truncate_old=True,
            zenith_angle=self.add_zenith,
            lat_lon=self.lat_lon,
            dataset_path=self.dataset_path,
            enable_odirect=params.enable_odirect,
            enable_logging=params.log_to_screen,
            seed=333,
            is_parallel=True,
        )

        # grid types
        self.grid_converter = GridConverter(
            params.data_grid_type,
            params.model_grid_type,
            torch.deg2rad(torch.tensor(self.extsource.lat_lon[0])).to(torch.float32).to(self.device),
            torch.deg2rad(torch.tensor(self.extsource.lat_lon[1])).to(torch.float32).to(self.device),
        )

        # some image properties
        self.img_shape_x = self.extsource.img_shape[0]
        self.img_shape_y = self.extsource.img_shape[1]
        self.img_crop_shape_x = self.extsource.crop_size[0]
        self.img_crop_shape_y = self.extsource.crop_size[1]
        self.img_crop_offset_x = self.extsource.crop_anchor[0]
        self.img_crop_offset_y = self.extsource.crop_anchor[1]
        self.img_local_shape_x = self.extsource.read_shape[0]
        self.img_local_shape_y = self.extsource.read_shape[1]
        self.img_local_offset_x = self.extsource.read_anchor[0]
        self.img_local_offset_y = self.extsource.read_anchor[1]

        # num steps
        self.num_steps_per_epoch = self.extsource.num_steps_per_epoch

        # load stats
        self.normalize = True
        self.roll = params.roll

        # in
        if self.norm_mode == "offline":
            if params.normalization == "minmax":
                mins = np.load(params.min_path)[:, self.in_channels]
                maxes = np.load(params.max_path)[:, self.in_channels]
                self.in_bias = mins
                self.in_scale = maxes - mins
            elif params.normalization == "zscore":
                means = np.load(params.global_means_path)[:, self.in_channels]
                stds = np.load(params.global_stds_path)[:, self.in_channels]
                self.in_bias = means
                self.in_scale = stds
                means_tracer = np.load(params.global_means_path_tracer)[:, self.tracer_in_channels]
                stds_tracer = np.load(params.global_stds_path_tracer)[:, self.tracer_in_channels]
                self.in_bias_tracer = means_tracer
                self.in_scale_tracer = stds_tracer
            elif params.normalization == "none":
                N_in_channels = len(self.in_channels)
                self.in_bias = np.zeros((1, N_in_channels, 1, 1))
                self.in_scale = np.ones((1, N_in_channels, 1, 1))

            # out
            if params.normalization == "minmax":
                mins = np.load(params.min_path)[:, self.out_channels]
                maxes = np.load(params.max_path)[:, self.out_channels]
                self.out_bias = mins
                self.out_scale = maxes - mins
            elif params.normalization == "zscore":
                means = np.load(params.global_means_path)[:, self.out_channels]
                stds = np.load(params.global_stds_path)[:, self.out_channels]
                self.out_bias = means
                self.out_scale = stds
                means_tracer = np.load(params.global_means_path_tracer)[:, self.tracer_out_channels]
                stds_tracer = np.load(params.global_stds_path_tracer)[:, self.tracer_out_channels]
                self.out_bias_tracer = means_tracer
                self.out_scale_tracer = stds_tracer
            elif params.normalization == "none":
                N_out_channels = len(self.out_channels)
                self.out_bias = np.zeros((1, N_out_channels, 1, 1))
                self.out_scale = np.ones((1, N_out_channels, 1, 1))

            # reformat the biases
            if self.norm_channels == "FHW":
                in_shape = (1, len(self.in_channels), 1, 1)
                out_shape = (1, len(self.out_channels), 1, 1)
                in_shape_tracer = (1, len(self.tracer_in_channels), 1, 1)
                out_shape_tracer = (1, len(self.tracer_out_channels), 1, 1)
            else:
                in_shape = (1, *self.in_bias.shape)
                out_shape = (1, *self.out_bias.shape)
                in_shape_tracer = (1, *self.in_bias_tracer.shape)
                out_shape_tracer = (1, *self.out_bias_tracer.shape)

            self.in_bias = np.reshape(self.in_bias, in_shape)
            self.in_scale = np.reshape(self.in_scale, in_shape)
            self.out_bias = np.reshape(self.out_bias, out_shape)
            self.out_scale = np.reshape(self.out_scale, out_shape)
            self.in_bias_tracer = np.reshape(self.in_bias_tracer, in_shape_tracer)
            self.in_scale_tracer = np.reshape(self.in_scale_tracer, in_shape_tracer)
            self.out_bias_tracer = np.reshape(self.out_bias_tracer, out_shape_tracer)
            self.out_scale_tracer = np.reshape(self.out_scale_tracer, out_shape_tracer)
        else:
            # in case of online normalization,
            # we do not need to set it here
            self.in_bias = None
            self.in_scale = None
            self.out_bias = None
            self.out_scale = None
            self.in_bias_tracer = None
            self.in_scale_tracer = None
            self.out_bias_tracer = None
            self.out_scale_tracer = None

        # create pipeline
        self.pipeline = self.get_pipeline()
        self.pipeline.start_py_workers()
        self.pipeline.build()

        # create iterator
        outnames = ["inp", "tar","inp_tracer", "tar_tracer"]
        if self.add_zenith:
            outnames += ["izen", "tzen"]
        self.iterator = DALIGenericIterator([self.pipeline], outnames, auto_reset=True, size=-1, last_batch_policy=LastBatchPolicy.DROP, prepare_first_batch=True)

    def get_input_normalization(self):
        if self.norm_mode == "offline":
            return self.in_bias, self.in_scale, self.in_bias_tracer, self.in_scale_tracer
        else:
            return 0.0, 1.0

    def get_output_normalization(self):
        if self.norm_mode == "offline":
            return self.out_bias, self.out_scale, self.out_bias_tracer, self.out_scale_tracer
        else:
            return 0.0, 1.0

    def reset_pipeline(self):
        self.pipeline.reset()
        self.iterator.reset()

    def __len__(self):
        return self.num_steps_per_epoch

    def __iter__(self):
        # self.iterator.reset()
        for token in self.iterator:
            inp = token[0]["inp"]
            tar = token[0]["tar"]
            inp_tracer = token[0]["inp_tracer"]
            tar_tracer = token[0]["tar_tracer"]

            if self.add_zenith:
                izen = token[0]["izen"]
                tzen = token[0]["tzen"]
                result = inp, tar, inp_tracer, tar_tracer, izen, tzen
            else:
                result = inp, tar, inp_tracer, tar_tracer

            # convert grid
            with torch.no_grad():
                result = map(lambda x: self.grid_converter(x), result)

            yield result
