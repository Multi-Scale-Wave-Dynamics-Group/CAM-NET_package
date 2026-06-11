import os
import glob
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from bisect import bisect_right
import sys
# Add project root (which contains `makani/`) to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from makani.utils.dataloaders.data_loader_multifiles import MultifilesDataset

class TracerMultifilesDataset(MultifilesDataset):
    def __init__(self, params, location, tracer_location, train, enable_logging=True):
        super().__init__(params, location, train, enable_logging)
        self.tracer_location = tracer_location
        # Set tracer H5 path
        self.dataset_path = params.h5_path
        # Get physical and tracer file paths
        self.files_paths = sorted(glob.glob(os.path.join(location, "????.h5")))
        self.tracer_files_paths = sorted(glob.glob(os.path.join(tracer_location, "????.h5")))
        print(f"Tracer files paths: {self.tracer_files_paths}")
        print(f"Tracer files paths: {self.files_paths}")
        self.tracer_files = [None for _ in self.tracer_files_paths]
        self.tracer_in_channels = np.array(params.tracer_in_channels) if hasattr(params, "tracer_in_channels") else None
        self.tracer_out_channels = np.array(params.tracer_out_channels) if hasattr(params, "tracer_out_channels") else None
        self.n_tracer_in_channels = len(self.tracer_in_channels) if self.tracer_in_channels is not None else 0
        self.n_tracer_out_channels = len(self.tracer_out_channels) if self.tracer_out_channels is not None else 0
    def _open_tracer_file(self, year_idx):
        if self.tracer_files[year_idx] is None:
            self.tracer_files[year_idx] = h5py.File(self.tracer_files_paths[year_idx], 'r')[self.dataset_path]

    def __getitem__(self, global_idx):
        # Call parent MultifilesDataset __getitem__
        result = super().__getitem__(global_idx)
    
        # Always the first two
        physical_input = result[0]
        physical_target = result[1]

        # Handle zenith input/output if they exist
        zenith_input = None
        zenith_target = None
        if self.add_zenith and len(result) == 4:
            zenith_input = result[2]
            zenith_target = result[3]

        # Physical spatial slicing info
        start_x, start_y = self.read_anchor
        end_x = start_x + self.read_shape[0]
        end_y = start_y + self.read_shape[1]

        # --- Load tracer input history ---
        tracer_input_list = []
        for offset in range(self.n_history + 1):
            year_idx = bisect_right(self.year_offsets, global_idx + self.dt * offset) - 1
            local_idx = global_idx + self.dt * offset - self.year_offsets[year_idx]
            self._open_tracer_file(year_idx)
            tracer_input = self.tracer_files[year_idx][local_idx : local_idx + 1, self.tracer_out_channels, start_x:end_x, start_y:end_y]
            with open('/glade/campaign/univ/uerf0005/modulus-makani/bad_sample.log', "a") as f:
                f.write(f"Bad tracer input at idx {local_idx}, file {self.tracer_files_paths[year_idx]}, inp shape {tracer_input.shape}\n")
            tracer_input_list.append(tracer_input)
            
        # --- Load tracer future targets ---
        tracer_target_list = []
        for offset in range(self.n_history + 1, self.n_history + self.n_future + 2):
            year_idx = bisect_right(self.year_offsets, global_idx + self.dt * offset) - 1
            local_idx = global_idx + self.dt * offset - self.year_offsets[year_idx]
            self._open_tracer_file(year_idx)
            tracer_target = self.tracer_files[year_idx][local_idx : local_idx + 1, self.tracer_out_channels, start_x:end_x, start_y:end_y]
            with open('/glade/campaign/univ/uerf0005/modulus-makani/bad_sample.log', "a") as f:
                f.write(f"Bad tracer target at idx {local_idx}, file {self.tracer_files_paths[year_idx]}, inp shape {tracer_input.shape}\n")
            tracer_target_list.append(tracer_target)

        # --- Convert everything to tensors ---
        tracer_input = torch.tensor(np.concatenate(tracer_input_list, axis=0)).float()
        tracer_target = torch.tensor(np.concatenate(tracer_target_list, axis=0)).float()

        physical_input = torch.tensor(physical_input).float()
        physical_target = torch.tensor(physical_target).float()

        if self.add_zenith:
            zenith_input = torch.tensor(zenith_input).float()
            zenith_target = torch.tensor(zenith_target).float()
            return (
                physical_input,
                physical_target,
                tracer_input,
                tracer_target,
                zenith_input,
                zenith_target
            )
        else:
            return (
                physical_input,
                physical_target,
                tracer_input,
                tracer_target
            )
