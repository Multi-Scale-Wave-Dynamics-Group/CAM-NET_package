# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import glob
import operator
from bisect import bisect_right
from itertools import accumulate
import logging
import numpy as np
import torch
from torch.utils.data import Dataset
import h5py
from makani.utils.grids import GridConverter
from modulus.distributed.utils import compute_split_shapes
# for the zenith angle
import datetime
import pytz
class TracerMultifilesDataset(Dataset):
    def __init__(self, params, physical_location, tracer_location, train, enable_logging=True):
        # get logger
        if params.log_to_screen:
            self.logger = logging.getLogger()
        self.params = params
        self.tracer_location = tracer_location
        self.physical_location = physical_location
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

        self.tracer_in_channels = np.array(params.tracer_in_channels) if hasattr(params, "tracer_in_channels") else None
        self.tracer_out_channels = np.array(params.tracer_out_channels) if hasattr(params, "tracer_out_channels") else None
        self.n_tracer_in_channels = len(self.tracer_in_channels) if self.tracer_in_channels is not None else 0
        self.n_tracer_out_channels = len(self.tracer_out_channels) if self.tracer_out_channels is not None else 0
        if hasattr(params, "lat") and hasattr(params, "lon"):
            self.lat_lon = (params.lat, params.lon)
        else:
            self.lat_lon = None
        # IO parallelism
        assert params.io_grid[0] == 1
        self.io_grid = params.io_grid[1:]
        self.io_rank = params.io_rank[1:]

                # get cropping:
        crop_size = [params.crop_size_x if hasattr(params, "crop_size_x") else None, params.crop_size_y if hasattr(params, "crop_size_y") else None]
        crop_anchor = [params.crop_anchor_x if hasattr(params, "crop_anchor_x") else 0, params.crop_anchor_y if hasattr(params, "crop_anchor_y") else 0]

        self.crop_size = crop_size
        self.crop_anchor = crop_anchor

        self._get_files_stats(enable_logging)

        # for normalization load the statistics

        self.normalize = True
        if params.normalization == "minmax":
            self.in_bias = np.load(params.min_path)[:, self.in_channels]
            self.in_scale = np.load(params.max_path)[:, self.in_channels] - self.in_bias
            self.out_bias = np.load(params.min_path)[:, self.out_channels]
            self.out_scale = np.load(params.max_path)[:, self.out_channels] - self.out_bias

            self.in_bias_tracer = np.load(params.min_path)[:, self.tracer_in_channels]
            self.in_scale_tracer = np.load(params.max_path)[:, self.tracer_in_channels] - self.in_bias_tracer
            self.out_bias_tracer = np.load(params.min_path)[:, self.tracer_out_channels]
            self.out_scale_tracer = np.load(params.max_path)[:, self.tracer_out_channels] - self.out_bias_tracer

        elif params.normalization == "zscore":
            self.in_bias = np.load(params.global_means_path)[:, self.in_channels]
            self.in_scale = np.load(params.global_stds_path)[:, self.in_channels]
            self.out_bias = np.load(params.global_means_path)[:, self.out_channels]
            self.out_scale = np.load(params.global_stds_path)[:, self.out_channels]
            self.in_bias_tracer = np.load(params.global_means_path_tracer)[:, self.tracer_in_channels]
            self.in_scale_tracer = np.load(params.global_stds_path_tracer)[:, self.tracer_in_channels]
            self.out_bias_tracer = np.load(params.global_means_path_tracer)[:, self.tracer_out_channels]
            self.out_scale_tracer = np.load(params.global_stds_path_tracer)[:, self.tracer_out_channels]

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

    def _get_stats_h5(self, enable_logging):
        with h5py.File(self.files_paths_physical[0], "r") as _f:
            if enable_logging:
                logging.info("Getting physical file stats from {}".format(self.files_paths_physical[0]))
                logging.info("Getting tracer file stats from {}".format(self.files_paths_tracer[0]))
            # original image shape (before padding)
            self.img_shape = _f[self.dataset_path].shape[2:4]
            self.total_channels = _f[self.dataset_path].shape[1]

        # get all sample counts
        self.n_samples_year = []
        sample_times = []
        for year, phys_filename, tracer_filename in zip(self.years, self.files_paths_physical, self.files_paths_tracer):
            with h5py.File(phys_filename, "r") as _f_phys, h5py.File(tracer_filename, "r") as _f:
                print("=== DEBUG INFO ===")
                print("Opening file:", tracer_filename)  # path to your .h5 file
                print("Dataset path:", self.dataset_path)  # path to the dataset within the .h5 file
                n_samples = _f[self.dataset_path].shape[0]
                self.n_samples_year.append(n_samples)
                sample_times.append(self._decode_sample_times(_f_phys, year, n_samples))
        self.sample_times = np.concatenate(sample_times).astype("datetime64[h]")
        return

    def _decode_sample_times(self, h5_file, year, n_samples):
        if "time" not in h5_file:
            start = np.datetime64(f"{year}-01-01T00", "h")
            step = np.timedelta64(int(self.dhours), "h")
            return start + np.arange(n_samples) * step

        time_dset = h5_file["time"]
        times = time_dset[:n_samples]
        attrs = time_dset.attrs

        if times.dtype.kind in ("S", "U", "O"):
            decoded = [t.decode("utf-8") if isinstance(t, bytes) else str(t) for t in times]
            return np.asarray(decoded, dtype="datetime64[h]")

        units = attrs.get("units")
        if isinstance(units, bytes):
            units = units.decode("utf-8")

        if isinstance(units, str) and "since" in units:
            ref_time = np.datetime64(units.split("since")[-1].strip(), "h")
            if "day" in units:
                return ref_time + (times * 24).astype("timedelta64[h]")
            if "hour" in units:
                return ref_time + times.astype("timedelta64[h]")
            if "second" in units:
                return (ref_time.astype("datetime64[s]") + times.astype("timedelta64[s]")).astype("datetime64[h]")

        start = np.datetime64(f"{year}-01-01T00", "h")
        step = np.timedelta64(int(self.dhours), "h")
        return start + np.arange(n_samples) * step
    
    def _get_files_stats(self, enable_logging):
        # check for hdf5 files
        self.physical_location = [self.physical_location] if not isinstance(self.physical_location, list) else self.physical_location
        self.tracer_location = [self.tracer_location] if not isinstance(self.tracer_location, list) else self.tracer_location

        self.files_paths_physical = []
        for phys_location in self.physical_location:
            self.files_paths_physical += glob.glob(os.path.join(phys_location, "????.h5"))
            self.logger.info(f"the phys location is {phys_location}")
        self.logger.info(f"the files paths are {self.files_paths_physical}")
        self.logger.info(f"the physical location is {self.physical_location}")
        self.files_paths_tracer = []
        for tra_location in self.tracer_location:
            self.files_paths_tracer += glob.glob(os.path.join(tra_location, "????.h5"))
            self.logger.info(f"the tra location is {tra_location}")
        self.logger.info(f"the files paths are {self.files_paths_tracer}")
        self.logger.info(f"the tracer location is {self.tracer_location}")
        self.file_format = "h5"

        if not self.files_paths_physical:
            raise IOError(f"Error, the specified file path {self.physical_location} does not contain physical h5 files.")
        if not self.files_paths_tracer:
            raise IOError(f"Error, the specified file path {self.tracer_location} does not contain tracer h5 files.")

        self.files_paths_physical.sort()
        self.files_paths_tracer.sort()

        # extract the years from filenames
        self.years = [int(os.path.splitext(os.path.basename(x))[0]) for x in self.files_paths_physical]
        self.files_physical = [None for x in self.files_paths_physical]
        self.files_tracer = [None for x in self.files_paths_tracer]

        # get stats
        self.n_years = len(self.files_paths_tracer)

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

        if enable_logging:
            logging.info("Average number of samples per year: {:.1f}".format(float(self.n_samples_total) / float(self.n_years)))
            logging.info(
                "Found data at path {}. Number of examples: {}. Full image Shape: {} x {} x {}. Read Shape: {} x {} x {}".format(
                    self.tracer_location, self.n_samples_available, self.img_shape[0], self.img_shape[1], self.total_channels, self.read_shape[0], self.read_shape[1], self.n_in_channels
                )
            )
            logging.info("Delta t: {} hours".format(self.dhours * self.dt))
            logging.info("Including {} hours of past history in training at a frequency of {} hours".format(self.dhours * self.dt * (self.n_history + 1), self.dhours * self.dt))
            logging.info("Including {} hours of future targets in training at a frequency of {} hours".format(self.dhours * self.dt * (self.n_future + 1), self.dhours * self.dt))

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


    def __len__(self):
        return self.n_samples_total - self.dt * (self.n_history + self.n_future + 1)

    def _open_physical_file(self, year_idx):
        _file = h5py.File(self.files_paths_physical[year_idx], "r")
        self.files_physical[year_idx] = _file[self.dataset_path]

    def _open_tracer_file(self, year_idx):
        _file = h5py.File(self.files_paths_tracer[year_idx], "r")
        self.files_tracer[year_idx] = _file[self.dataset_path]

    def __getitem__(self, global_idx):
        start_x, start_y = self.read_anchor
        end_x = start_x + self.read_shape[0]
        end_y = start_y + self.read_shape[1]

        # History/future slices for both physical and tracer
        phys_inp_list, phys_tar_list = [], []
        tracer_inp_list, tracer_tar_list = [], []

        for offset_idx in range(self.n_history + 1):
            year_idx = bisect_right(self.year_offsets, global_idx + self.dt * offset_idx) - 1
            local_idx = global_idx + self.dt * offset_idx - self.year_offsets[year_idx]
            self.logger.info(f"year index is {year_idx}, local index is {local_idx}")
            if self.files_physical[year_idx] is None:
                self._open_physical_file(year_idx)
            if self.files_tracer[year_idx] is None:
                self._open_tracer_file(year_idx)

            phys_inp = self.files_physical[year_idx][local_idx, self.in_channels, start_x:end_x, start_y:end_y]
            tracer_inp = self.files_tracer[year_idx][local_idx, self.tracer_in_channels, start_x:end_x, start_y:end_y]
            if self.normalize:
                phys_inp = (phys_inp - self.in_bias) / self.in_scale
                tracer_inp = (tracer_inp - self.in_bias_tracer) / self.in_scale_tracer

            phys_inp_list.append(phys_inp)
            tracer_inp_list.append(tracer_inp)

        for offset_idx in range(self.n_history + 1, self.n_history + self.n_future + 2):
            year_idx = bisect_right(self.year_offsets, global_idx + self.dt * offset_idx) - 1
            local_idx = global_idx + self.dt * offset_idx - self.year_offsets[year_idx]

            if self.files_physical[year_idx] is None:
                self._open_physical_file(year_idx)
            if self.files_tracer[year_idx] is None:
                self._open_tracer_file(year_idx)

            phys_tar = self.files_physical[year_idx][local_idx, self.out_channels, start_x:end_x, start_y:end_y]
            tracer_tar = self.files_tracer[year_idx][local_idx, self.tracer_out_channels, start_x:end_x, start_y:end_y]
            if self.normalize:
                phys_tar = (phys_tar - self.out_bias) / self.out_scale
                tracer_tar = (tracer_tar - self.out_bias_tracer) / self.out_scale_tracer
            phys_tar_list.append(phys_tar)
            tracer_tar_list.append(tracer_tar)

        # Concatenate history/future sequences
        phys_inp = np.concatenate(phys_inp_list, axis=0)
        phys_tar = np.concatenate(phys_tar_list, axis=0)
        tracer_inp = np.concatenate(tracer_inp_list, axis=0)
        tracer_tar = np.concatenate(tracer_tar_list, axis=0)

        if self.add_zenith:
            initial_idx = global_idx + self.dt * self.n_history
            year_idx = bisect_right(self.year_offsets, initial_idx) - 1
            local_idx = initial_idx - self.year_offsets[year_idx]
            zen_inp, zen_tar = self._compute_zenith_angle(local_idx, year_idx)

            result = phys_inp, phys_tar, tracer_inp, tracer_tar, zen_inp, zen_tar
        else:
            result = phys_inp, phys_tar, tracer_inp, tracer_tar, 

        result = tuple(torch.as_tensor(arr) for arr in result)
        # convert grid
        result = tuple(map(lambda x: self.grid_converter(x), result))

        return result

    def get_sample_times(self):
        """
        Return the global raw timestamp axis. The timestamp at index
        dataset_idx + dt * n_history is the initial condition for __getitem__.
        """
        return self.sample_times

    def get_output_normalization(self):
        return self.out_bias, self.out_scale, self.out_bias_tracer, self.out_scale_tracer

    def get_input_normalization(self):
        return self.in_bias, self.in_scale, self.in_bias_tracer, self.in_scale_tracer
    # HDF5 routines

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
