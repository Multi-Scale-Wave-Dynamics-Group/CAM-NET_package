# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from makani.models import Preprocessor2D

class CAMPreprocessorWithTracer(Preprocessor2D):
    def __init__(self, params):
        """
        Enhanced preprocessor for physical+tracer variables with:
        - Separate normalization pipelines
        - Temporal history handling
        - Consistent interface with TracerTrainer
        
        Args:
            params: Configuration containing:
                - num_input_channels: Physical variable count
                - tracer_input_channels: Tracer variable count  
                - tracer_output_channels: Output tracer count
                - n_history: Temporal history length
                - grid_type: Mesh type ('equiangular' etc.)
                - normalization_mode: 'mean_std'/'min_max'/'none'
                - tracer_history_mode: 'exponential'/'mean'/'none'
        """
        super().__init__(params)
        
        # Tracer configuration
        self.tracer_in_channels = params.tracer_input_channels
        self.tracer_out_channels = getattr(params, 'tracer_output_channels', params.tracer_input_channels)
        self.tracer_history_mode = getattr(params, 'tracer_history_mode', 'none')
        self.tracer_decay = getattr(params, 'tracer_history_decay', 0.5)
        
        # Initialize normalization stats
        self.tracer_stats = {
            'input_mean': None,
            'input_std': None,
            'target_mean': None, 
            'target_std': None
        }
        
        # History weights
        self.register_buffer('tracer_weights', 
                           self._init_history_weights(self.tracer_history_mode, 
                                                    params.n_history+1,
                                                    self.tracer_decay))

    def _init_history_weights(self, mode, n_timesteps, decay):
        """Initialize temporal weighting"""
        if mode == "exponential":
            weights = torch.exp(-decay * torch.arange(n_timesteps, 0, -1, dtype=torch.float32))
            return weights / weights.sum()
        elif mode == "mean":
            return torch.ones(n_timesteps, dtype=torch.float32) / n_timesteps
        return torch.ones(n_timesteps, dtype=torch.float32)

    def _compute_stats(self, x, stats_key):
        """Compute and store normalization statistics"""
        self.tracer_stats[stats_key] = {
            'mean': x.mean(dim=(0,2,3), keepdim=True),
            'std': x.std(dim=(0,2,3), keepdim=True) + 1e-6
        }

    def normalize_tracer(self, x, is_target=False):
        """Normalize tracer variables with temporal weighting"""
        key = 'target' if is_target else 'input'
        
        if self.tracer_stats[f'{key}_mean'] is None:
            self._compute_stats(x, f'{key}_stats')
            
        stats = self.tracer_stats[f'{key}_stats']
        x_norm = (x - stats['mean']) / stats['std']
        
        if self.tracer_history_mode != 'none':
            x_norm = x_norm * self.tracer_weights.view(1, -1, 1, 1)
            
        return x_norm

    def cache_unpredicted_features(self, x_physical, x_tracer):
        """
        Preprocess and return features for training
        Returns:
            tuple: (physical_features, tracer_targets)
                   physical_features: [B, C_phys, H, W]
                   tracer_targets: [B, C_tracer, H, W]
        """
        # Normalize physical vars (with static features)
        x_phys = super().history_normalize(x_physical)
        x_phys = self.add_static_features(x_phys)
        
        # Normalize tracer targets
        x_tracer = self.normalize_tracer(x_tracer, is_target=True)
        
        return x_phys, x_tracer

    def flatten_history(self, x):
        """
        Handle both [B,T,C,H,W] and [B,C,H,W] inputs
        Returns:
            tensor: [B, C*T, H, W] if temporal, else [B,C,H,W]
        """
        if x.dim() == 5:
            return x.flatten(1, 2)
        return x

    def preprocess(self, x_physical, x_tracer):
        """
        Full preprocessing pipeline
        Returns:
            dict: {
                'physical': Processed physical vars [B,C,H,W],
                'tracer': Processed tracers [B,C,H,W]
            }
        """
        return {
            'physical': self.add_static_features(super().history_normalize(x_physical)),
            'tracer': self.normalize_tracer(x_tracer)
        }

    def denormalize(self, x, var_type='physical'):
        """
        Reverse normalization for outputs
        Args:
            x: Tensor to denormalize
            var_type: 'physical' or 'tracer'
        """
        if var_type == 'physical':
            return super().denormalize(x)
        
        if self.tracer_stats['target_mean'] is None:
            raise RuntimeError("Tracer stats not initialized")
            
        return x * self.tracer_stats['target_std'] + self.tracer_stats['target_mean']