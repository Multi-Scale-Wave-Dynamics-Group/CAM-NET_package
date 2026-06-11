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
import logging
import glob
from itertools import groupby, accumulate
import operator
from bisect import bisect_right

import torch
import numpy as np
from torch.utils.data import Dataset
import h5py

# for the zenith angle
import datetime
import pytz

# for grid conversion
from makani.utils.grids import GridConverter

# import splitting logic
from modulus.distributed.utils import compute_split_shapes

class MultifilesDataset(Dataset):
    def __init__(self, params, location, train, enable_logging=True):
        self.params = params
        self.location = location
        self.train = train
        self.dt = params.dt
        self.dhours = params.dhours
        self.n_history = params.n_history
        self.n_future = params.valid_autoreg_steps if not train else params.n_future
        self.in_channels = np.array(params.in_channels)
        self.out_channels = np.array(params.out_channels)
        self.n_in_channels = len(self.in_channels)
        self.n_out_channels = len(self.out_channels)
        self.add_zenith = params.add_zenith if hasattr(params, "add_zenith") else False
        self.dataset_path = params.h5_path
        if hasattr(params, "lat") and hasattr(params, "lon"):
            self.lat_lon = (params.lat, params.lon)
        else:
            self.lat_lon = None

        # multifiles dataloader doesn't support channel parallelism yet
        # set the read slices
        assert params.io_grid[0] == 1
        self.io_grid = params.io_grid[1:]
        self.io_rank = params.io_rank[1:]

        # get cropping:
        crop_size = [params.crop_size_x if hasattr(params, "crop_size_x") else None, params.crop_size_y if hasattr(params, "crop_size_y") else None]
        crop_anchor = [params.crop_anchor_x if hasattr(params, "crop_anchor_x") else 0, params.crop_anchor_y if hasattr(params, "crop_anchor_y") else 0]

        self.crop_size = crop_size
        self.crop_anchor = crop_anchor

        self._get_files_stats(enable_logging)

        self.add_f107 = params.add_f107 if hasattr(params, "add_f107") else False
        self.add_ap = params.add_ap if hasattr(params, "add_ap") else False
        self.add_kp = params.add_kp if hasattr(params, "add_kp") else False
        self.add_space_weather = self.add_f107 or self.add_ap or self.add_kp
        
        if self.add_space_weather:
            if train:
                self.sw_location = params.space_weather_train_path
            else:
                if hasattr(params, 'space_weather_inf_path'):
                    self.sw_location = params.space_weather_inf_path
                else:
                    self.sw_location = params.space_weather_valid_path
            
            self._get_space_weather_files_stats(enable_logging)
            self._load_space_weather_normalization(params)
        # for normalization load the statistics
        self.normalize = True
        if params.normalization == "minmax":
            self.in_bias = np.load(params.min_path)[:, self.in_channels]
            self.in_scale = np.load(params.max_path)[:, self.in_channels] - self.in_bias
            self.out_bias = np.load(params.min_path)[:, self.out_channels]
            self.out_scale = np.load(params.max_path)[:, self.out_channels] - self.out_bias
        elif params.normalization == "zscore":
            self.in_bias = np.load(params.global_means_path)[:, self.in_channels]
            self.in_scale = np.load(params.global_stds_path)[:, self.in_channels]
            self.out_bias = np.load(params.global_means_path)[:, self.out_channels]
            self.out_scale = np.load(params.global_stds_path)[:, self.out_channels]

        # we need some additional static fields in this case
        if self.lat_lon is None:
            resolution = 360.0 / float(self.img_shape[1])
            longitude = np.arange(0, 360, resolution)
            latitude = np.arange(-90, 90 + resolution, resolution)
            latitude = latitude[::-1]
            self.lat_lon = (latitude.tolist(), longitude.tolist())

        if self.add_zenith:
            latitude = np.array(self.lat_lon[0])
            longitude = np.array(self.lat_lon[1])
            self.lon_grid, self.lat_grid = np.meshgrid(longitude, latitude)
            self.lat_grid_local = self.lat_grid[self.read_anchor[0] : self.read_anchor[0] + self.read_shape[0], self.read_anchor[1] : self.read_anchor[1] + self.read_shape[1]]
            self.lon_grid_local = self.lon_grid[self.read_anchor[0] : self.read_anchor[0] + self.read_shape[0], self.read_anchor[1] : self.read_anchor[1] + self.read_shape[1]]

        # grid types
        self.grid_converter = GridConverter(
            params.data_grid_type,
            params.model_grid_type,
            torch.deg2rad(torch.tensor(self.lat_lon[0])).to(torch.float32),
            torch.deg2rad(torch.tensor(self.lat_lon[1])).to(torch.float32),
        )

    # HDF5 routines
    def _get_stats_h5(self, enable_logging):
        with h5py.File(self.files_paths[0], "r") as _f:
            if enable_logging:
                logging.info("Getting file stats from {}".format(self.files_paths[0]))
            # original image shape (before padding)
            self.img_shape = _f[self.dataset_path].shape[2:4]
            self.total_channels = _f[self.dataset_path].shape[1]

        # get all sample counts
        self.n_samples_year = []
        for filename in self.files_paths:
            with h5py.File(filename, "r") as _f:
                logging.info("=== DEBUG INFO ===")
                logging.info(f"Opening file: {filename}")
                logging.info(f"Dataset path: {self.dataset_path}")
                self.n_samples_year.append(_f[self.dataset_path].shape[0])
        return

    def _get_files_stats(self, enable_logging):
        # check for hdf5 files
        self.files_paths = []
        self.location = [self.location] if not isinstance(self.location, list) else self.location
        for location in self.location:
            self.files_paths = self.files_paths + glob.glob(os.path.join(location, "????.h5"))
        self.file_format = "h5"

        if not self.files_paths:
            raise IOError(f"Error, the specified file path {self.location} does not contain h5 files.")

        self.files_paths.sort()

        # extract the years from filenames
        self.years = [int(os.path.splitext(os.path.basename(x))[0]) for x in self.files_paths]
        self.files = [None for x in self.files_paths]

        # get stats
        self.n_years = len(self.files_paths)

        if self.file_format == "h5":
            self._get_stats_h5(enable_logging)

        # determine local read size:
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
        read_anchor_x = self.crop_anchor[0] + sum(split_shapes_x[:self.io_rank[0]])
        
        # for y
        split_shapes_y = compute_split_shapes(self.crop_size[1], self.io_grid[1])
        read_shape_y = split_shapes_y[self.io_rank[1]]
        read_anchor_y = self.crop_anchor[1] + sum(split_shapes_y[:self.io_rank[1]])

        # store the variables
        self.read_anchor = [read_anchor_x, read_anchor_y]
        self.read_shape = [read_shape_x, read_shape_y]

        # do some sample indexing gymnastics
        self.year_offsets = list(accumulate(self.n_samples_year, operator.add))[:-1]
        self.year_offsets.insert(0, 0)
        self.n_samples_available = sum(self.n_samples_year)
        self.n_samples_total = self.n_samples_available
        self.sample_times = []
        all_times = []

        for i, fpath in enumerate(self.files_paths):
            with h5py.File(fpath, "r") as f:
                if "time" in f:
                    times = f["time"][:]  # numeric values
                    attrs = f["time"].attrs

                    # --- Decode CF-style "units" attribute if available ---
                    if "units" in attrs:
                        units = attrs["units"]
                        if isinstance(units, bytes):
                            units = units.decode("utf-8")
                        if "since" in units:
                            base_str = units.split("since")[-1].strip()
                            ref_time = np.datetime64(base_str)
                            if "day" in units:
                                times_np = ref_time + (times * 24 * np.timedelta64(1, "h"))
                            elif "hour" in units:
                                times_np = ref_time + (times * np.timedelta64(1, "h"))
                            else:
                                # fallback if weird unit
                                times_np = ref_time + (times * np.timedelta64(1, "s"))
                        else:
                            # no recognizable "since" → assume days since 2000-01-01
                            ref_time = np.datetime64("2000-01-01T00")
                            times_np = ref_time + (times * np.timedelta64(1, "D"))
                    else:
                        # no attrs → fallback
                        ref_time = np.datetime64("2000-01-01T00")
                        times_np = ref_time + (times * np.timedelta64(1, "D"))

                    all_times.append(times_np)
                else:
                    # Fallback: synthetic time axis
                    start = np.datetime64(f"{self.years[i]}-01-01T00")
                    step = np.timedelta64(int(self.dhours * self.dt), "h")
                    n_samples = self.n_samples_year[i]
                    times_np = start + np.arange(n_samples) * step
                    all_times.append(times_np)
        # Merge across years
        self.sample_times = np.concatenate(all_times)


        if enable_logging:
            logging.info("Average number of samples per year: {:.1f}".format(float(self.n_samples_total) / float(self.n_years)))
            logging.info(
                "Found data at path {}. Number of examples: {}. Full image Shape: {} x {} x {}. Read Shape: {} x {} x {}".format(
                    self.location, self.n_samples_available, self.img_shape[0], self.img_shape[1], self.total_channels, self.read_shape[0], self.read_shape[1], self.n_in_channels
                )
            )
            logging.info("Delta t: {} hours".format(self.dhours * self.dt))
            logging.info("Including {} hours of past history in training at a frequency of {} hours".format(self.dhours * self.dt * (self.n_history + 1), self.dhours * self.dt))
            logging.info("Including {} hours of future targets in training at a frequency of {} hours".format(self.dhours * self.dt * (self.n_future + 1), self.dhours * self.dt))
            logging.info("Total number of samples: {}".format(self.n_samples_total))
            logging.info("Number of years: {}".format(self.n_years))
            logging.info("Number of samples per year: {}".format(self.n_samples_year))
            logging.info("Number of samples available: {}".format(self.n_samples_available))
            logging.info("Year offsets: {}".format(self.year_offsets))
        
        # set properties for compatibility
        self.img_shape_x = self.img_shape[0]
        self.img_shape_y = self.img_shape[1]

        self.img_crop_shape_x = self.crop_size[0]
        self.img_crop_shape_y = self.crop_size[1]
        self.img_crop_offset_x = self.crop_anchor[0]
        self.img_crop_offset_y = self.crop_anchor[1]

        self.img_local_shape_x = self.read_shape[0]
        self.img_local_shape_y = self.read_shape[1]
        self.img_local_offset_x = self.read_anchor[0]
        self.img_local_offset_y = self.read_anchor[1]

    def _compute_zenith_angle(self, local_idx, year_idx):
        # import
        from makani.third_party.climt.zenith_angle import cos_zenith_angle

        # compute hours into the year
        year = self.years[year_idx]
        jan_01_epoch = datetime.datetime(year, 1, 1, 0, 0, 0, tzinfo=pytz.utc)

        # zenith angle for input
        inp_times = np.asarray([jan_01_epoch + datetime.timedelta(hours=idx * self.dhours) for idx in range(local_idx - self.dt * self.n_history, local_idx + 1, self.dt)])
        cos_zenith_inp = np.expand_dims(cos_zenith_angle(inp_times, self.lon_grid_local, self.lat_grid_local).astype(np.float32), axis=1)

        # zenith angle for target:
        tar_times = np.asarray(
            [jan_01_epoch + datetime.timedelta(hours=idx * self.dhours) for idx in range(local_idx + self.dt, local_idx + self.dt * (self.n_future + 1) + 1, self.dt)]
        )
        cos_zenith_tar = np.expand_dims(cos_zenith_angle(tar_times, self.lon_grid_local, self.lat_grid_local).astype(np.float32), axis=1)

        return cos_zenith_inp, cos_zenith_tar

    def _open_file(self, year_idx):
        _file = h5py.File(self.files_paths[year_idx], "r")
        print(f"It's loading files from {self.files_paths}")
        self.files[year_idx] = _file[self.dataset_path]

    def __len__(self):
        return self.n_samples_total - self.dt * (self.n_history + self.n_future + 1)

    def __getitem__(self, global_idx):
        start_x = self.read_anchor[0]
        end_x = start_x + self.read_shape[0]
        start_y = self.read_anchor[1]
        end_y = start_y + self.read_shape[1]

        inp_list = []
        sw_inp_list = [] if self.add_space_weather else None
        
        # Load input history
        for offset_idx in range(self.n_history + 1):
            year_idx = bisect_right(self.year_offsets, global_idx + self.dt * offset_idx) - 1
            local_idx = global_idx + self.dt * offset_idx - self.year_offsets[year_idx]

            if self.files[year_idx] is None:
                self._open_file(year_idx)
            print(f"global index is{global_idx}, local_idx is {local_idx}, offset_idx is {offset_idx}")
            print(f"year index is {year_idx}, dt is {self.dt}, self.year_offsets[year_idx] is {self.year_offsets}")
            print(f"the channels are {self.in_channels}")
            inp = self.files[year_idx][local_idx : local_idx + 1, self.in_channels, start_x:end_x, start_y:end_y]
            if self.normalize:
                inp = (inp - self.in_bias) / self.in_scale
            inp_list.append(inp)
            
            # Load space weather
            if self.add_space_weather:
                if self.sw_files[year_idx] is None:
                    self._open_sw_file(year_idx)
                sw = self.sw_files[year_idx][local_idx : local_idx + 1, :, start_x:end_x, start_y:end_y]
                sw = (sw - self.sw_bias) / self.sw_scale
                sw_inp_list.append(sw)

        # Load targets
        tar_list = []
        sw_tar_list = [] if self.add_space_weather else None
        
        for offset_idx in range(self.n_history + 1, self.n_history + self.n_future + 2):
            year_idx = bisect_right(self.year_offsets, global_idx + self.dt * offset_idx) - 1
            local_idx = global_idx + self.dt * offset_idx - self.year_offsets[year_idx]

            if self.files[year_idx] is None:
                self._open_file(year_idx)

            tar = self.files[year_idx][local_idx : local_idx + 1, self.out_channels, start_x:end_x, start_y:end_y]
            if self.normalize:
                tar = (tar - self.out_bias) / self.out_scale
            tar_list.append(tar)
            
            # Load space weather targets
            if self.add_space_weather:
                if self.sw_files[year_idx] is None:
                    self._open_sw_file(year_idx)
                sw = self.sw_files[year_idx][local_idx : local_idx + 1, :, start_x:end_x, start_y:end_y]
                sw = (sw - self.sw_bias) / self.sw_scale
                sw_tar_list.append(sw)

        # Concatenate along time dimension
        # Shape: (n_history+1, n_in_channels, H, W) and (n_future+1, n_out_channels, H, W)
        # DO NOT add batch dimension here - DataLoader does it automatically
        inp = np.concatenate(inp_list, axis=0)
        tar = np.concatenate(tar_list, axis=0)
        
        if self.add_space_weather:
            sw_inp = np.concatenate(sw_inp_list, axis=0)
            sw_tar = np.concatenate(sw_tar_list, axis=0)

        # Handle zenith and space weather
        if self.add_zenith:
            year_idx = bisect_right(self.year_offsets, global_idx) - 1
            local_idx = global_idx - self.year_offsets[year_idx]
            zen_inp, zen_tar = self._compute_zenith_angle(local_idx, year_idx)
            
            if self.add_space_weather:
                result = inp, tar, zen_inp, zen_tar, sw_inp, sw_tar
            else:
                result = inp, tar, zen_inp, zen_tar
        else:
            if self.add_space_weather:
                result = inp, tar, sw_inp, sw_tar
            else:
                result = inp, tar

        result = tuple(torch.as_tensor(arr) for arr in result)
        result = tuple(map(lambda x: self.grid_converter(x), result))
        
        return result

    def get_sample_times(self):
        """
        Return the global list of timestamps for each dataset sample index.
        The order matches __getitem__ indexing.
        """
        return self.sample_times


    def get_output_normalization(self):
        return self.out_bias, self.out_scale

    def get_input_normalization(self):
        return self.in_bias, self.in_scale

    def _get_space_weather_files_stats(self, enable_logging):
        """Initialize space weather file handles"""
        import glob
        
        self.sw_files_paths = glob.glob(os.path.join(self.sw_location, "????.h5"))
        self.sw_files_paths.sort()
        
        if not self.sw_files_paths:
            raise IOError(f"No space weather files found at {self.sw_location}")
        
        self.sw_files = [None for _ in self.sw_files_paths]
        self.sw_dataset_path = "fields"
        
        if enable_logging:
            logging.info(f"Found {len(self.sw_files_paths)} space weather files at {self.sw_location}")

    def _load_space_weather_normalization(self, params):
        """Load normalization statistics for space weather"""
        if params.normalization == "zscore":
            sw_means = np.load(params.space_weather_means_path)
            sw_stds = np.load(params.space_weather_stds_path)
            
            indices_to_use = []
            if self.add_f107:
                indices_to_use.append(0)
            if self.add_ap:
                indices_to_use.append(1)
            if self.add_kp:
                indices_to_use.append(2)
            
            self.sw_bias = sw_means[:, indices_to_use, :, :]
            self.sw_scale = sw_stds[:, indices_to_use, :, :]
        else:
            self.sw_bias = np.zeros((1, 3, 1, 1))
            self.sw_scale = np.ones((1, 3, 1, 1))

    def _open_sw_file(self, year_idx):
        """Open space weather h5 file"""
        _file = h5py.File(self.sw_files_paths[year_idx], "r")
        self.sw_files[year_idx] = _file[self.sw_dataset_path]
