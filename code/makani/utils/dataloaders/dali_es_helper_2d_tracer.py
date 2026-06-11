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

import time
import sys
import os
import glob
import numpy as np
import cupy as cp
import cupyx as cpx
import h5py
import zarr
import logging
from itertools import groupby, accumulate
import operator
from bisect import bisect_right

# for nvtx annotation
import torch

# we need this for the zenith angle feature
import datetime
import pytz

# import splitting logic
from modulus.distributed.utils import compute_split_shapes

class GeneralEStracer(object):
    def _get_slices(self, lst):
        for a, b in groupby(enumerate(lst), lambda pair: pair[1] - pair[0]):
            b = list(b)
            yield slice(b[0][1], b[-1][1] + 1)

    # very important: the seed has to be constant across the workers, or otherwise mayhem:
    def __init__(
        self,
        physical_location,
        tracer_location,
        max_samples,
        samples_per_epoch,
        train,
        batch_size,
        dt,
        dhours,
        n_history,
        n_future,
        in_channels,
        out_channels,
        tracer_in_channels,
        tracer_out_channels,
        crop_size,
        crop_anchor,
        num_shards,
        shard_id,
        io_grid,
        io_rank,
        device_id=0,
        truncate_old=True,
        enable_logging=True,
        zenith_angle=True,
        lat_lon=None,
        dataset_path="fields",
        enable_odirect=False,
        seed=333,
        is_parallel=True,
    ):
        self.batch_size = batch_size
        self.physical_location = physical_location
        self.tracer_location = tracer_location
        self.max_samples = max_samples
        self.n_samples_per_epoch = samples_per_epoch
        self.truncate_old = truncate_old
        self.train = train
        self.dt = dt
        self.dhours = dhours
        self.n_history = n_history
        self.n_future = n_future
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.tracer_in_channels = tracer_in_channels
        self.tracer_out_channels = tracer_out_channels
        self.n_in_channels = len(in_channels)
        self.n_out_channels = len(out_channels)
        self.n_tracer_in_channels = len(tracer_in_channels)
        self.n_tracer_out_channels = len(tracer_out_channels)
        self.crop_size = crop_size
        self.crop_anchor = crop_anchor
        self.base_seed = seed
        self.num_shards = num_shards
        self.device_id = device_id
        self.shard_id = shard_id
        self.is_parallel = is_parallel
        self.zenith_angle = zenith_angle
        self.dataset_path = dataset_path
        self.lat_lon = lat_lon

        # O_DIRECT specific stuff
        self.file_driver = "direct" if enable_odirect else None
        self.read_direct = False  # if enable_odirect else True
        self.num_retries = 5

        # set the read slices
        # we do not support channel parallelism yet
        assert io_grid[0] == 1
        self.io_grid = io_grid[1:]
        self.io_rank = io_rank[1:]

        # parse the files
        self._get_files_stats(enable_logging)
        self.shuffle = True if train else False

        # convert in_channels to list of slices:
        self.in_channels_slices = list(self._get_slices(self.in_channels))
        self.out_channels_slices = list(self._get_slices(self.out_channels))
        self.tracer_in_channels_slices = list(self._get_slices(self.tracer_in_channels))
        self.tracer_out_channels_slices = list(self._get_slices(self.tracer_out_channels))
        # we need some additional static fields in this case
        if self.lat_lon is None:
            resolution = 360.0 / float(self.img_shape[1])
            longitude = np.arange(0, 360, resolution)
            latitude = np.arange(-90, 90 + resolution, resolution)
            latitude = latitude[::-1]
            self.lat_lon = (latitude.tolist(), longitude.tolist())

        if self.zenith_angle:
            latitude = np.array(self.lat_lon[0])
            longitude = np.array(self.lat_lon[1])
            self.lon_grid, self.lat_grid = np.meshgrid(longitude, latitude)
            self.lat_grid_local = self.lat_grid[self.read_anchor[0] : self.read_anchor[0] + self.read_shape[0], self.read_anchor[1] : self.read_anchor[1] + self.read_shape[1]]
            self.lon_grid_local = self.lon_grid[self.read_anchor[0] : self.read_anchor[0] + self.read_shape[0], self.read_anchor[1] : self.read_anchor[1] + self.read_shape[1]]

    # HDF5 routines
    def _get_stats_h5(self, enable_logging):
        self.time_stamps_year_phys = []
        self.time_stamps_year_tracer = []

        for phys_path, tracer_path in zip(self.files_paths_phys, self.files_paths_tracer):
            with h5py.File(phys_path, "r") as f_phys, h5py.File(tracer_path, "r") as f_tracer:
                time_phys = f_phys["time"][:]
                time_tracer = f_tracer["time"][:]
                self.time_stamps_year_phys.append(time_phys)
                self.time_stamps_year_tracer.append(time_tracer)
                self.img_shape = f_phys[self.dataset_path].shape[2:4]
                self.total_channels_phys = f_phys[self.dataset_path].shape[1]
                self.total_channels_tracer = f_phys[self.dataset_path].shape[1]
        self.valid_indices_year = []
        for t_phys, t_tracer in zip(self.time_stamps_year_phys, self.time_stamps_year_tracer):
            set_phys = set(t_phys)
            set_tracer = set(t_tracer)
            matched_times = sorted(set_phys&set_tracer)
            indices = np.where(np.isin(t_phys, matched_times))[0]
            self.valid_indices_year.append(indices)
        self.time_to_idx_phys = [{t: i for i, t in enumerate(t_arr)} for t_arr in self.time_stamps_year_phys]
        self.time_to_idx_tracer = [{t: i for i, t in enumerate(t_arr)} for t_arr in self.time_stamps_year_tracer]
        self.n_samples_year = [len(v) for v in self.valid_indices_year]
        return

    def _get_year_h5(self, year_idx):
        # here we want to use the specific file driver
        self.files_tracer[year_idx] = h5py.File(self.files_paths_tracer[year_idx], "r", driver=self.file_driver)
        self.files_phys[year_idx] = h5py.File(self.files_paths_phys[year_idx], "r", driver=self.file_driver)
        self.dsets_tracer[year_idx] = self.files_tracer[year_idx][self.dataset_path]
        self.dsets_phys[year_idx] = self.files_phys[year_idx][self.dataset_path]
        return

    def _get_data_h5(
        self,
        inp,
        tar,
        inp_tracer,
        tar_tracer,
        dset_phys,
        dset_tracer,
        idx_phys,
        idx_tracer,
        start_x,
        end_x,
        start_y,
        end_y
    ):
    ### --- Physical IN --- ###
        off = 0
        for slice_in in self.in_channels_slices:
            start = off
            end = start + (slice_in.stop - slice_in.start)
            if self.read_direct:
                dset_phys.read_direct(
                    inp,
                    np.s_[(idx_phys - self.dt * self.n_history) : (idx_phys + 1) : self.dt, slice_in, start_x:end_x, start_y:end_y],
                    np.s_[:, start:end, ...]
                )
            else:
                inp[:, start:end, ...] = dset_phys[
                    (idx_phys - self.dt * self.n_history) : (idx_phys + 1) : self.dt,
                    slice_in,
                    start_x:end_x,
                    start_y:end_y
                ]

            off = end

        ### --- Physical OUT --- ###
        off = 0
        for slice_out in self.out_channels_slices:
            start = off
            end = start + (slice_out.stop - slice_out.start)
            if self.read_direct:
                dset_phys.read_direct(
                    tar,
                    np.s_[
                        (idx_phys + self.dt) : (idx_phys + self.dt * (self.n_future + 1) + 1) : self.dt,
                        slice_out,
                        start_x:end_x,
                        start_y:end_y
                    ],
                    np.s_[:, start:end, ...]
                )
            else:
                tar[:, start:end, ...] = dset_phys[
                    (idx_phys + self.dt) : (idx_phys + self.dt * (self.n_future + 1) + 1) : self.dt,
                    slice_out,
                    start_x:end_x,
                    start_y:end_y
                ]
            off = end

        ### --- Tracer IN --- ###
        off = 0
        for slice_in in self.tracer_in_channels_slices:
            start = off
            end = start + (slice_in.stop - slice_in.start)
            if self.read_direct:
                dset_tracer.read_direct(
                    inp_tracer,
                    np.s_[(idx_tracer - self.dt * self.n_history) : (idx_tracer + 1) : self.dt, slice_in, start_x:end_x, start_y:end_y],
                    np.s_[:, start:end, ...]
                )
            else:
                inp_tracer[:, start:end, ...] = dset_tracer[
                    (idx_tracer - self.dt * self.n_history) : (idx_tracer + 1) : self.dt,
                    slice_in,
                    start_x:end_x,
                    start_y:end_y
                ]
            off = end

        ### --- Tracer OUT --- ###
        off = 0
        for slice_out in self.tracer_out_channels_slices:
            start = off
            end = start + (slice_out.stop - slice_out.start)
            if self.read_direct:
                dset_tracer.read_direct(
                    tar_tracer,
                    np.s_[
                        (idx_tracer + self.dt) : (idx_tracer + self.dt * (self.n_future + 1) + 1) : self.dt,
                        slice_out,
                        start_x:end_x,
                        start_y:end_y
                    ],
                    np.s_[:, start:end, ...]
                )  
            else:
                tar_tracer[:, start:end, ...] = dset_tracer[
                    (idx_tracer + self.dt) : (idx_tracer + self.dt * (self.n_future + 1) + 1) : self.dt,
                    slice_out,
                    start_x:end_x,
                    start_y:end_y
                ]
            off = end
        return inp, tar, inp_tracer, tar_tracer

    def _get_files_stats(self, enable_logging):
        # --- PHYSICAL FILES ---
        self.files_paths_phys = []
        self.physical_location = [self.physical_location] if not isinstance(self.physical_location, list) else self.physical_location
        for location in self.physical_location:
            self.files_paths_phys += glob.glob(os.path.join(location, "????.h5"))
        self.files_paths_phys.sort()

        # --- TRACER FILES ---
        self.files_paths_tracer = []
        self.tracer_location = [self.tracer_location] if not isinstance(self.tracer_location, list) else self.tracer_location
        for location in self.tracer_location:
            self.files_paths_tracer += glob.glob(os.path.join(location, "????.h5"))
        self.files_paths_tracer.sort()
        assert len(self.files_paths_phys) == len(self.files_paths_tracer), "Mismatch between number of physical and tracer files!"

        # --- YEARS ---
        self.years = [int(os.path.splitext(os.path.basename(x))[0]) for x in self.files_paths_phys]
        self.n_years = len(self.files_paths_phys)

        # --- STATS ---
        self.file_format = "h5"
        self._get_stats_h5(enable_logging)  # still valid, we'll assume phys files define img_shape

        # sanitize the crops first
        if self.crop_size[0] is None:
            self.crop_size[0] = self.img_shape[0]
        if self.crop_size[1] is None:
            self.crop_size[1] = self.img_shape[1]
        assert self.crop_anchor[0] + self.crop_size[0] <= self.img_shape[0]
        assert self.crop_anchor[1] + self.crop_size[1] <= self.img_shape[1]
        # for x
        split_shapes_x = compute_split_shapes(self.crop_size[0], self.io_grid[0])
        read_shape_x = split_shapes_x[self.io_rank[0]]
        read_anchor_x = self.crop_anchor[0] + sum(split_shapes_x[:self.io_rank[0]]) #self.crop_anchor[0] + read_shape_x * self.io_rank[0]
        # for y
        split_shapes_y = compute_split_shapes(self.crop_size[1], self.io_grid[1])
        read_shape_y = split_shapes_y[self.io_rank[1]]
        read_anchor_y = self.crop_anchor[1] + sum(split_shapes_y[:self.io_rank[1]]) #self.crop_anchor[1] + read_shape_y * self.io_rank[1]
        self.read_anchor = [read_anchor_x, read_anchor_y]
        self.read_shape = [read_shape_x, read_shape_y]

        # do some sample indexing gymnastics
        self.year_offsets = list(accumulate(self.n_samples_year, operator.add))[:-1]
        self.year_offsets.insert(0, 0)
        self.n_samples_available = sum(self.n_samples_year)
        if self.max_samples is not None:
            self.n_samples_total = min(self.n_samples_available, self.max_samples)
        else:
            self.n_samples_total = self.n_samples_available

        # do the sharding
        self.n_samples_shard = self.n_samples_total // self.num_shards
        if self.truncate_old:
            self.n_samples_offset = self.n_samples_available - self.n_samples_total
        else:
            self.n_samples_offset = 0

        # number of steps per epoch
        self.num_steps_per_cycle = self.n_samples_shard // self.batch_size
        if self.n_samples_per_epoch is None:
            self.n_samples_per_epoch = self.n_samples_total
        self.num_steps_per_epoch = self.n_samples_per_epoch // (self.batch_size * self.num_shards)

        # we need those here
        self.num_samples_per_cycle_shard = self.num_steps_per_cycle * self.batch_size
        self.num_samples_per_epoch_shard = self.num_steps_per_epoch * self.batch_size
        # prepare  file lists
        # --- FILE HANDLES ---
        self.files_phys = [None for _ in range(self.n_years)]
        self.files_tracer = [None for _ in range(self.n_years)]
        self.dsets_phys = [None for _ in range(self.n_years)]
        self.dsets_tracer = [None for _ in range(self.n_years)]
        if enable_logging:
            logging.info("Average number of samples per year: {:.1f}".format(float(self.n_samples_total) / float(self.n_years)))
            logging.info(
                "Found data at path {}. Number of examples: {}. Full image Shape: {} x {} x {}. Read Shape: {} x {} x {}".format(
                    self.tracer_location, self.n_samples_available, self.img_shape[0], self.img_shape[1], self.total_channels_tracer, self.read_shape[0], self.read_shape[1], self.n_in_channels
                )
            )
            logging.info(
                "Using {} from the total number of available samples with {} samples per epoch (corresponds to {} steps for {} shards with local batch size {})".format(
                    self.n_samples_total, self.n_samples_per_epoch, self.num_steps_per_epoch, self.num_shards, self.batch_size
                )
            )
            logging.info("Delta t: {} hours".format(self.dhours * self.dt))
            logging.info("Including {} hours of past history in training at a frequency of {} hours".format(self.dhours * self.dt * (self.n_history + 1), self.dhours * self.dt))
            logging.info("Including {} hours of future targets in training at a frequency of {} hours".format(self.dhours * self.dt * (self.n_future + 1), self.dhours * self.dt))

        # some state variables
        self.last_cycle_epoch = None
        self.index_permutation = None

        # prepare buffers for double buffering
        if not self.is_parallel:
            self._init_buffers()
        # --- Rest of the code unchanged ---
        # (sharding, year offsets, read anchors, etc. stay same)

    def _init_buffers(self):
        # set device
        self.device = cp.cuda.Device(self.device_id)
        self.device.use()
        self.current_buffer = 0

        # --- Physical ---
        self.inp_buffs = [
            cpx.zeros_pinned((self.n_history + 1, self.n_in_channels, self.read_shape[0], self.read_shape[1]), dtype=np.float32),
            cpx.zeros_pinned((self.n_history + 1, self.n_in_channels, self.read_shape[0], self.read_shape[1]), dtype=np.float32),
        ]
        self.tar_buffs = [
            cpx.zeros_pinned((self.n_future + 1, self.n_out_channels, self.read_shape[0], self.read_shape[1]), dtype=np.float32),
            cpx.zeros_pinned((self.n_future + 1, self.n_out_channels, self.read_shape[0], self.read_shape[1]), dtype=np.float32),
        ]

        # --- Tracer ---
        self.inp_tracer_buffs = [
            cpx.zeros_pinned((self.n_history + 1, self.n_tracer_in_channels, self.read_shape[0], self.read_shape[1]), dtype=np.float32),
            cpx.zeros_pinned((self.n_history + 1, self.n_tracer_in_channels, self.read_shape[0], self.read_shape[1]), dtype=np.float32),
        ]
        self.tar_tracer_buffs = [
            cpx.zeros_pinned((self.n_future + 1, self.n_tracer_out_channels, self.read_shape[0], self.read_shape[1]), dtype=np.float32),
            cpx.zeros_pinned((self.n_future + 1, self.n_tracer_out_channels, self.read_shape[0], self.read_shape[1]), dtype=np.float32),
        ]

        if self.zenith_angle:
            self.zen_inp_buffs = [
                cpx.zeros_pinned((self.n_history + 1, 1, self.read_shape[0], self.read_shape[1]), dtype=np.float32),
                cpx.zeros_pinned((self.n_history + 1, 1, self.read_shape[0], self.read_shape[1]), dtype=np.float32),
            ]
            self.zen_tar_buffs = [
                cpx.zeros_pinned((self.n_future + 1, 1, self.read_shape[0], self.read_shape[1]), dtype=np.float32),
                cpx.zeros_pinned((self.n_future + 1, 1, self.read_shape[0], self.read_shape[1]), dtype=np.float32),
            ]
            
    def _compute_zenith_angle(self, zen_inp, zen_tar, local_idx, year_idx):
        # nvtx range
        torch.cuda.nvtx.range_push("GeneralES:_compute_zenith_angle")

        # import
        from makani.third_party.climt.zenith_angle import cos_zenith_angle

        # compute hours into the year
        year = self.years[year_idx]
        jan_01_epoch = datetime.datetime(year, 1, 1, 0, 0, 0, tzinfo=pytz.utc)

        # zenith angle for input
        inp_times = np.asarray([jan_01_epoch + datetime.timedelta(hours=idx * self.dhours) for idx in range(local_idx - self.dt * self.n_history, local_idx + 1, self.dt)])
        cos_zenith_inp = np.expand_dims(cos_zenith_angle(inp_times, self.lon_grid_local, self.lat_grid_local).astype(np.float32), axis=1)
        zen_inp[...] = cos_zenith_inp[...]

        # zenith angle for target:
        tar_times = np.asarray(
            [jan_01_epoch + datetime.timedelta(hours=idx * self.dhours) for idx in range(local_idx + self.dt, local_idx + self.dt * (self.n_future + 1) + 1, self.dt)]
        )
        cos_zenith_tar = np.expand_dims(cos_zenith_angle(tar_times, self.lon_grid_local, self.lat_grid_local).astype(np.float32), axis=1)
        zen_tar[...] = cos_zenith_tar[...]

        # nvtx range
        torch.cuda.nvtx.range_pop()

        return

    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)

        if self.file_format == "h5":
            self.get_year_handle = self._get_year_h5
            self.get_data_handle = self._get_data_h5
        else:
            self.get_year_handle = self._get_year_zarr
            self.get_data_handle = self._get_data_zarr

        if self.is_parallel:
            self._init_buffers()

    def __len__(self):
        return self.n_samples_shard

    def __del__(self):
        # close files
        for f in self.files_tracer:
            if f is not None:
                f.close()

    def __call__(self, sample_info):
        # compute global iteration index:
        global_sample_idx = sample_info.idx_in_epoch + sample_info.epoch_idx * self.num_samples_per_epoch_shard
        cycle_sample_idx = global_sample_idx % self.num_samples_per_cycle_shard
        cycle_epoch_idx = global_sample_idx // self.num_samples_per_cycle_shard

        if sample_info.iteration >= self.num_steps_per_epoch:
            raise StopIteration

        torch.cuda.nvtx.range_push("GeneralES:__call__")

        if cycle_epoch_idx != self.last_cycle_epoch:
            self.last_cycle_epoch = cycle_epoch_idx
            rng = np.random.default_rng(seed=self.base_seed + cycle_epoch_idx)
            if self.shuffle:
                self.index_permutation = self.n_samples_offset + rng.permutation(self.n_samples_total)
            else:
                self.index_permutation = self.n_samples_offset + np.arange(self.n_samples_total)
            start = self.n_samples_shard * self.shard_id
            end = start + self.n_samples_shard
            self.index_permutation = self.index_permutation[start:end]

        sample_idx = self.index_permutation[cycle_sample_idx]
        year_idx = bisect_right(self.year_offsets, sample_idx) - 1
        valid_indices = self.valid_indices_year[year_idx]
        local_idx = valid_indices[sample_idx - self.year_offsets[year_idx]]
        
        if local_idx < self.dt * self.n_history:
            local_idx += self.dt * self.n_history
        if local_idx >= (self.n_samples_year[year_idx] - self.dt * (self.n_future + 1)):
            local_idx = self.n_samples_year[year_idx] - self.dt * (self.n_future + 1) - 1
        if self.files_phys[year_idx] is None:
            self.get_year_handle(year_idx)  # assumes will load both physical and tracer datasets
        timestamp = self.time_stamps_year_phys[year_idx][local_idx]
        idx_phys = self.time_to_idx_phys[year_idx][timestamp]
        idx_tracer = self.time_to_idx_tracer[year_idx][timestamp]
        
        # Sanity check: ensure timestamps match
        t_phys = self.time_stamps_year_phys[year_idx][idx_phys]
        t_tracer = self.time_stamps_year_tracer[year_idx][idx_tracer]

        if t_phys != t_tracer:
            logging.warning(f"Timestamp mismatch! Year {self.years[year_idx]} | Phys: {t_phys}, Tracer: {t_tracer}")
        else:
            logging.debug(f"Physical timestamp: {t_phys}, Tracer timestamp: {t_tracer}")

        # --- Buffers ---
        inp = self.inp_buffs[self.current_buffer]
        tar = self.tar_buffs[self.current_buffer]
        inp_tracer = self.inp_tracer_buffs[self.current_buffer]
        tar_tracer = self.tar_tracer_buffs[self.current_buffer]
        if self.zenith_angle:
            zen_inp = self.zen_inp_buffs[self.current_buffer]
            zen_tar = self.zen_tar_buffs[self.current_buffer]

        self.current_buffer = (self.current_buffer + 1) % 2

        dset_phys = self.dsets_phys[year_idx]
        dset_tracer = self.dsets_tracer[year_idx]

        start_x = self.read_anchor[0]
        end_x = start_x + self.read_shape[0]
        start_y = self.read_anchor[1]
        end_y = start_y + self.read_shape[1]

        inp, tar, inp_tracer, tar_tracer = self.get_data_handle(
            inp, tar, inp_tracer, tar_tracer, dset_phys, dset_tracer, idx_phys, idx_tracer, start_x, end_x, start_y, end_y)
        logging.info("zenith angle: {}".format(self.zenith_angle))
        if self.zenith_angle:
            self._compute_zenith_angle(zen_inp, zen_tar, local_idx, year_idx)
            result = inp, tar, inp_tracer, tar_tracer, zen_inp, zen_tar
        else:
            result = inp, tar, inp_tracer, tar_tracer

        torch.cuda.nvtx.range_pop()
        return result
