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
import io
import numpy as np
import concurrent.futures as cf
from PIL import Image
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import wandb

import torch


def plot_comparison(
    pred,
    truth,
    lat=None,
    lon=None,
    pred_title="Prediction",
    truth_title="Ground truth",
    cmap="twilight_shifted",
    projection="mollweide",
    diverging=False,
    figsize=(6, 7),
    vmax=None,
    title_str=None
):
    """
    Visualization tool to plot a comparison between ground truth and prediction
    pred: 2d array
    truth: 2d array
    cmap: colormap
    projection: "mollweide", "hammer", "aitoff" or None
    """
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    assert len(pred.shape) == 2
    assert len(truth.shape) == 2
    assert pred.shape == truth.shape

    H, W = pred.shape
    if (lat is None) or (lon is None):
        #lon = np.linspace(-np.pi, np.pi, W)
        #lat = np.linspace(-np.pi / 2.0, np.pi / 2.0, H)
        lon = np.linspace(-180, 180, W)
        lat = np.linspace(-90, 90, H)
    Lon, Lat = np.meshgrid(np.degrees(lon), np.degrees(lat))

    # Define projection
    proj_dict = {
        "mollweide": ccrs.Mollweide(),
        "hammer": ccrs.Hammer(),
        "aitoff": ccrs.Aitoff(),
        None: ccrs.PlateCarree(),  # Default to lat/lon
    }
    projection = proj_dict.get(projection, ccrs.PlateCarree())
    print(f"Using projection: {projection}")


    # only normalize with the truth
    vmax = vmax or np.abs(truth).max()
    # vmax = vmax or max(np.abs(pred).max(), np.abs(truth).max())
    if diverging:
        vmin = -vmax
    else:
        vmin = 0.0

    fig, axes = plt.subplots(2,1,figsize=figsize,subplot_kw={"projection":projection})
    fig.suptitle(title_str, fontsize=14, fontweight="bold")
    ax = axes[0]
    pcm = ax.pcolormesh(Lon, Lat, pred, transform=ccrs.PlateCarree(), cmap=cmap, vmax=vmax, vmin=vmin)
    ax.set_title(pred_title)
    ax.add_feature(cfeature.COASTLINE, edgecolor="white")
    ax.add_feature(cfeature.BORDERS, linestyle=":", edgecolor="gray")
    gl = ax.gridlines(draw_labels=True, linestyle="--", color="gray", alpha=0.5)
    gl.right_labels = False
    gl.top_labels = False
    cbar = plt.colorbar(pcm, ax=ax, orientation="horizontal", pad=0.05)

    ax = axes[1]
    pcm = ax.pcolormesh(Lon,Lat, truth, transform=ccrs.PlateCarree(), cmap=cmap, vmax=vmax, vmin=vmin)
    ax.set_title(truth_title)
    ax.add_feature(cfeature.COASTLINE, edgecolor="white")
    ax.add_feature(cfeature.BORDERS, linestyle=":", edgecolor="gray")
    gl = ax.gridlines(draw_labels=True, linestyle="--", color="gray", alpha=0.5)
    gl.right_labels = False
    gl.top_labels = False
    cbar = plt.colorbar(pcm, ax=ax, orientation="horizontal", pad=0.05)

    plt.tight_layout()

    # save into memory buffer
    buf = io.BytesIO()
    plt.savefig(buf)
    plt.close(fig)
    buf.seek(0)

    # create image
    image = Image.open(buf)

    return image


import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os

def plot_inference_comparison_scale(predictions_small, predictions_large, targets_small, targets_large, params, comparison_channels, 
                              lat=None, lon=None, cmap="twilight_shifted", 
                              projection="mollweide", diverging=False, 
                              figsize=(20, 12), vmax=None, title_str=None, level_to_height=None):
    """
    Plots a 3xN grid comparing predictions, targets, and their differences over time.
    """
    assert predictions_small.shape == targets_small.shape, "Predictions and targets must have the same shape."
    time_steps = predictions_small.shape[0]
    channel_names = params.channel_names

    if lat is None or lon is None:
        lon = np.linspace(-180, 180, predictions_small.shape[3])
        lat = np.linspace(-90, 90, predictions_small.shape[2])
    Lon, Lat = np.meshgrid(np.degrees(lon), np.degrees(lat))

    proj_dict = {
        "mollweide": ccrs.Mollweide(),
        "hammer": ccrs.Hammer(),
        "aitoff": ccrs.Aitoff(),
        None: ccrs.PlateCarree(),
    }
    projection = proj_dict.get(projection, ccrs.PlateCarree())

    for comparison_var in comparison_channels:
        selected_pred_small = predictions_small[:, channel_names.index(comparison_var), :, :]
        selected_pred_large = predictions_large[:, channel_names.index(comparison_var), :, :]
        selected_targ_small = targets_small[:, channel_names.index(comparison_var), :, :]
        selected_targ_large = targets_large[:, channel_names.index(comparison_var), :, :]

        vmax = np.abs(selected_targ_large).max()
        vmin = -vmax if "T" not in comparison_var else 0  # Symmetric or non-symmetric scaling

        fig, axes = plt.subplots(7, 5, figsize=figsize, subplot_kw={"projection": projection})

        if title_str:
            fig.suptitle(title_str, fontsize=26, fontweight="bold", y=1.02)
            
        # Store PCMs for separate colorbars
        pcm_small_scale = None  # For first 2 rows
        pcm_large_scale = None  # For bottom 5 rows
        for t in range(5):
            prediction = selected_pred_small[t, ...] + selected_pred_large[t, ...]
            target = selected_targ_large[t, ...] + selected_targ_small[t, ...]
            data_pairs = [
                (selected_pred_small[t, ...], f"Small scale prediction at Time {t+1}"),
                (selected_targ_small[t, ...], f"Small scale target at Time {t+1}"),
                (selected_pred_large[t, ...], f"Large scale prediction at Time {t+1}"),
                (selected_targ_large[t, ...], f"Large scale target at Time {t+1}"),
                (prediction, f"Prediction at Time {t+1}"),
                (target, f"Target at Time {t+1}"),
                (target - prediction, f"Difference at Time {t+1}")  # Fixed subtraction
            ]
            for row, (data, title) in enumerate(data_pairs):
                ax = axes[row, t]
                # Scale for first and second row colorbars
                if row in [0, 1]:
                    pcm = ax.pcolormesh(Lon, Lat, data, transform=ccrs.PlateCarree(), cmap=cmap, vmax=0.1*vmax, vmin=0.1*vmin)
                    if pcm_small_scale is None:
                        pcm_small_scale = pcm
                else:
                    pcm = ax.pcolormesh(Lon, Lat, data, transform=ccrs.PlateCarree(), cmap=cmap, vmax=vmax, vmin=vmin)
                    if pcm_large_scale is None:
                        pcm_large_scale = pcm

                ax.set_title(title, fontsize=18)
                ax.add_feature(cfeature.COASTLINE, edgecolor="white")
                ax.add_feature(cfeature.BORDERS, linestyle=":", edgecolor="gray")

                gl = ax.gridlines(draw_labels=True, linestyle="--", color="gray", alpha=0.5)
                gl.right_labels = False
                gl.top_labels = False

                level_idx = channel_names.index(comparison_var)
                if level_idx in level_to_height:
                    height_label = level_to_height[level_idx]
                    ax.set_ylabel(f"Height: {height_label} km", fontsize=18, rotation=90, labelpad=10)
        if pcm_small_scale:
            cbar_ax1 = fig.add_axes([0.92, 0.7, 0.02, 0.25])
            cbar1 = fig.colorbar(pcm_small_scale, cax = cbar_ax1, orientation="vertical")
            cbar1.ax.tick_params(labelsize=18)
            cbar1.set_label("Small-scale data colorbar", fontsize = 18)
        if pcm_large_scale:
            cbar_ax2 = fig.add_axes([0.92, 0.1, 0.02, 0.55])
            cbar2 = fig.colorbar(pcm_large_scale, cax= cbar_ax2, orientation="vertical")
            cbar2.ax.tick_params(labelsize=18)
            cbar2.set_label("Large-scale data colorbar", fontsize = 18)


        # 🔹 Adjust subplot layout for better spacing
        fig.subplots_adjust(left=0.05, right=0.88, bottom=0.1, top=0.95, hspace=0.3, wspace=0.2)

        # 🔹 Save the figure
        fig.savefig(os.path.join(params.experiment_dir, f"comparison_plot_{comparison_var}.png"), dpi=300)

    return

def plot_inference_comparison(predictions, targets, params, comparison_channels, 
                              lat=None, lon=None, cmap="twilight_shifted", 
                              projection="mollweide", diverging=False, 
                              figsize=(20, 12), vmax=None, title_str=None):
    """
    Plots a 3xN grid comparing predictions, targets, and their differences over time.
    """
    assert predictions.shape == targets.shape, "Predictions and targets must have the same shape."
    time_steps = predictions.shape[0]
    channel_names = params.channel_names

    if lat is None or lon is None:
        lon = np.linspace(-180, 180, predictions.shape[3])
        lat = np.linspace(-90, 90, predictions.shape[2])
    Lon, Lat = np.meshgrid(np.degrees(lon), np.degrees(lat))

    proj_dict = {
        "mollweide": ccrs.Mollweide(),
        "hammer": ccrs.Hammer(),
        "aitoff": ccrs.Aitoff(),
        None: ccrs.PlateCarree(),
    }
    projection = proj_dict.get(projection, ccrs.PlateCarree())

    if vmax is None:
        vmax = max(np.abs(targets).max(), np.abs(predictions).max())
    vmin = -vmax if diverging else 0.0

    for comparison_var in comparison_channels:
        selected_pred = predictions[:, channel_names.index(comparison_var), :, :]
        selected_targ = targets[:, channel_names.index(comparison_var), :, :]

        fig, axes = plt.subplots(3, 5, figsize=figsize, subplot_kw={"projection": projection})

        if title_str:
            fig.suptitle(title_str, fontsize=24, fontweight="bold")

        for t in range(5):
            data_pairs = [
                (selected_pred[t, ...], f"Prediction Time {t+1}"),
                (selected_targ[t, ...], f"Target Time {t+1}"),
                (selected_targ[t, ...] - selected_pred[t, ...], f"Difference Time {t+1}")  # Fixed subtraction
            ]
            for row, (data, title) in enumerate(data_pairs):
                ax = axes[row, t]
                pcm = ax.pcolormesh(Lon, Lat, data, transform=ccrs.PlateCarree(), cmap=cmap, vmax=vmax, vmin=vmin)
                ax.set_title(title, fontsize=12)
                ax.add_feature(cfeature.COASTLINE, edgecolor="white")
                ax.add_feature(cfeature.BORDERS, linestyle=":", edgecolor="gray")

                gl = ax.gridlines(draw_labels=True, linestyle="--", color="gray", alpha=0.5)
                gl.right_labels = False
                gl.top_labels = False

                if row == 2:
                    cbar = plt.colorbar(pcm, ax=ax, orientation="horizontal", pad=0.05)
                    cbar.ax.tick_params(labelsize=10)

        # Reduce spacing between rows
        fig.subplots_adjust(hspace=0.2, wspace=0.1)  # Less vertical & horizontal gap
        fig.tight_layout()  # Auto-adjust layout

        fig.savefig(os.path.join(params.experiment_dir, f"comparison_plot_{comparison_var}.png"), dpi=300)
    
    return



def plot_rollout_metrics(acc_curves, rmse_curves, params, epoch, model_name, comparison_channels):
    "Plots rollout metrics such as RMSE and ACC and saves them to the experiment directory"

    channel_names = params.channel_names

    for metric in ["acc", "rmse"]:
        curves = acc_curves if metric == "acc" else rmse_curves

        for comparison_var in comparison_channels:
            model_metric = curves[channel_names.index(comparison_var), :].cpu().numpy()

            import matplotlib.pyplot as plt
            import matplotlib.ticker as ticker

            var_name = comparison_var

            fig, ax = plt.subplots()
            t = np.arange(1, len(model_metric) + 1, 1) * 6
            ax.plot(t, model_metric, ".-", label=model_name)
            xticks = np.arange(0, len(model_metric) + 1, 1) * 6
            x_locator = ticker.FixedLocator(xticks)
            ax.xaxis.set_major_locator(x_locator)
            y_locator = ticker.MaxNLocator(nbins=20)
            ax.yaxis.set_major_locator(y_locator)
            ax.grid(which="major", alpha=0.5)
            ax.legend()
            ax.set_xlabel("Time [h]")
            ax.set_ylabel(metric + " " + var_name)
            ax.set_title(params.wandb_name)
            plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment="right")
            fig.savefig(os.path.join(params.experiment_dir, metric + "_" + var_name + ".png"))
            # push to wandb
            if params.log_to_wandb:
                wandb.log({metric + "_" + var_name: wandb.Image(fig)}, step=epoch)


def visualize_field(tag, func_string, prediction, target, lat, lon, scale, bias, diverging, field_name):
    torch.cuda.nvtx.range_push("visualize_field")

    # get func handle:
    func_handle = eval(func_string)

    # unscale:
    pred = scale * prediction + bias
    targ = scale * target + bias

    # apply functor:
    pred = func_handle(pred)
    targ = func_handle(targ)

    # generate image
    image = plot_comparison(pred, targ, lat, lon, pred_title="Prediction", truth_title="Ground truth", projection="mollweide", diverging=diverging, title_str=field_name)

    torch.cuda.nvtx.range_pop()

    return tag, image


class VisualizationWrapper(object):
    "Handles visualization during training"

    def __init__(self, log_to_wandb, path, prefix, plot_list, lat=None, lon=None, scale=1.0, bias=0.0, num_workers=1):
        self.log_to_wandb = log_to_wandb
        self.generate_video = True
        self.path = path
        self.prefix = prefix
        self.plot_list = plot_list

        # grid
        self.lat = lat
        self.lon = lon

        # normalization
        self.scale = scale
        self.bias = bias

        # this is for parallel processing
        self.executor = cf.ProcessPoolExecutor(max_workers=num_workers)
        self.requests = []

    def reset(self):
        self.requests = []

    def add(self, tag, prediction, target):
        # go through the plot list
        for item in self.plot_list:
            field_name = item["name"]
            func_string = item["functor"]
            plot_diverge = item["diverging"]
            self.requests.append(
                self.executor.submit(visualize_field, (tag, field_name), func_string, np.copy(prediction), np.copy(target), self.lat, self.lon, self.scale, self.bias, plot_diverge, field_name)
            )

        return

    def finalize(self):
        torch.cuda.nvtx.range_push("VisualizationWrapper:finalize")

        results = {}
        for request in cf.as_completed(self.requests):
            token, image = request.result()
            tag, field_name = token
            prefix = field_name + "_" + tag
            results[prefix] = image

        if self.generate_video:
            if self.log_to_wandb:
                video = []

                # draw stuff that goes on every frame here
                for prefix, image in sorted(results.items()):
                    video.append(np.transpose(np.asarray(image), (2, 0, 1)))

                video = np.stack(video)
                results = [wandb.Video(video, fps=3, format="gif")]
            else:
                video = []

                # draw stuff that goes on every frame here
                for prefix, image in sorted(results.items()):
                    video.append(np.asarray(image))
                functor_name = prefix.split("_")[0]
                output_filename = f"video_output_{functor_name}.gif"
                video = ImageSequenceClip(video, fps=3)
                video.write_gif(output_filename)
                print(f"Saved GIF: {output_filename}")

        else:
            results = [wandb.Image(image, caption=prefix) for prefix, image in results.items()]

        if self.log_to_wandb and results:
            wandb.log({"Inference samples": results})

        # reset requests
        self.reset()

        torch.cuda.nvtx.range_pop()

        return
