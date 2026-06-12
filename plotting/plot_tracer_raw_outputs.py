#!/usr/bin/env python3
"""Plot chunked tracer inference outputs saved by TracerInferencer."""

import argparse
import datetime as dt
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

try:
    import cartopy.crs as ccrs
except ImportError:  # cartopy is nice for maps, but not required to inspect fields.
    ccrs = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot tracer_raw_outputs chunks in a prediction-vs-target map grid."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("outputs_first_stage/sfno_tracer_coupled/01/tracer_raw_outputs"),
        help="Directory containing tracer_pred_*.npy, tracer_targ_*.npy, and manifests.",
    )
    parser.add_argument("--rank", type=int, default=0, help="MPI rank shard to plot.")
    parser.add_argument("--batch", type=int, default=1, help="Batch/chunk number to plot.")
    parser.add_argument(
        "--channel",
        default=None,
        help="Single tracer channel name from the config, or a zero-based channel index.",
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        default=None,
        help="Tracer channel names or zero-based channel indices to plot as height groups.",
    )
    parser.add_argument(
        "--variable",
        default="EDens",
        help="Variable prefix used with --levels when --channels/--channel are omitted.",
    )
    parser.add_argument(
        "--levels",
        type=int,
        nargs="+",
        default=[30, 60, 90],
        help="Level indices used to build <variable>_level_idx_<level> channel names.",
    )
    parser.add_argument(
        "--height-labels",
        nargs="+",
        default=None,
        help='Height labels for each plotted channel, e.g. "350 km" "250 km" "150 km".',
    )
    parser.add_argument(
        "--leads",
        type=int,
        nargs="+",
        default=[3, 13, 18],
        help="Zero-based rollout lead indices to plot.",
    )
    parser.add_argument(
        "--pred-path",
        type=Path,
        default=None,
        help="Optional direct prediction .npy file. Overrides --raw-dir/--rank/--batch.",
    )
    parser.add_argument(
        "--targ-path",
        type=Path,
        default=None,
        help="Optional direct target .npy file. Overrides --raw-dir/--rank/--batch.",
    )
    parser.add_argument(
        "--column-labels",
        nargs="+",
        default=None,
        help='Column labels. Example: "2015-03-16 21 UT" "2015-03-18 03 UT".',
    )
    parser.add_argument(
        "--start-time",
        default=None,
        help="Optional timestamp for lead 0, formatted like 2015-03-16T21:00.",
    )
    parser.add_argument(
        "--lead-hours",
        type=float,
        default=None,
        help="Hours between rollout leads when computing labels from --start-time.",
    )
    parser.add_argument(
        "--date-format",
        default="%Y-%m-%d %H UT",
        help="strftime format for computed column labels.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/sfnonet.yaml"),
        help="YAML config containing tracer_channel_names.",
    )
    parser.add_argument(
        "--config-name",
        default="sfno_tracer_coupled",
        help="Config section containing tracer_channel_names.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Figure title. Defaults to a title inferred from selected channels.",
    )
    parser.add_argument(
        "--model-label",
        default="CAM-NET",
        help="Row label for predictions.",
    )
    parser.add_argument(
        "--target-label",
        default="WACCM-X",
        help="Row label for targets.",
    )
    parser.add_argument(
        "--units-label",
        default=None,
        help="Colorbar units label. Defaults to electron-density units for EDens.",
    )
    parser.add_argument(
        "--value-scale",
        type=float,
        default=None,
        help="Multiply plotted values by this factor. Defaults to 1e-6 for EDens, else 1.",
    )
    parser.add_argument("--cmap", default="RdBu_r", help="Matplotlib colormap name.")
    parser.add_argument("--vmin", type=float, default=None, help="Fixed colorbar minimum.")
    parser.add_argument("--vmax", type=float, default=None, help="Fixed colorbar maximum.")
    parser.add_argument(
        "--vmax-percentile",
        type=float,
        default=100.0,
        help="Percentile for auto vmax. Use 100 for the finite maximum.",
    )
    parser.add_argument(
        "--shared-color-scale",
        action="store_true",
        help="Use one color scale for all height groups instead of one per height.",
    )
    parser.add_argument(
        "--symmetric",
        action="store_true",
        help="Force symmetric color limits about zero.",
    )
    parser.add_argument(
        "--no-roll-longitude",
        action="store_true",
        help="Do not roll 0..360 longitude data to -180..180 before plotting.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG path. Defaults to a file in the raw output directory.",
    )
    parser.add_argument("--dpi", type=int, default=180, help="Output image DPI.")
    return parser.parse_args()


def load_config_section(config_path, config_name):
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config[config_name]


def resolve_channel(channel, channel_names):
    try:
        idx = int(channel)
    except ValueError:
        idx = channel_names.index(channel)
    if idx < 0 or idx >= len(channel_names):
        raise IndexError(f"channel index {idx} is outside [0, {len(channel_names) - 1}]")
    return idx, channel_names[idx]


def resolve_channels(args, channel_names):
    if args.channel is not None and args.channels is not None:
        raise ValueError("Use either --channel or --channels, not both.")

    requested = None
    if args.channel is not None:
        requested = [args.channel]
    elif args.channels is not None:
        requested = args.channels

    if requested is not None:
        return [resolve_channel(channel, channel_names) for channel in requested]

    resolved = []
    missing = []
    for level in args.levels:
        name = f"{args.variable}_level_idx_{level}"
        if name in channel_names:
            resolved.append((channel_names.index(name), name))
        else:
            missing.append(level)

    if not missing:
        return resolved

    # Fall back to the same level indices with whichever variable prefix exists
    # in this raw-output config. This lets the paper-style layout work for UI,
    # VI, WI, or EDens runs without editing the script.
    resolved = []
    pattern_by_level = {
        level: re.compile(rf"^(.+)_level_idx_{level}$") for level in args.levels
    }
    for level in args.levels:
        matches = [
            (idx, name)
            for idx, name in enumerate(channel_names)
            if pattern_by_level[level].match(name)
        ]
        if not matches:
            available = ", ".join(channel_names[:10])
            if len(channel_names) > 10:
                available += ", ..."
            raise ValueError(
                f"Could not find {args.variable}_level_idx_{level} or any channel "
                f"ending in _level_idx_{level}. Available channels: {available}"
            )
        resolved.append(matches[0])
    return resolved


def default_height_labels(channel_names):
    if len(channel_names) == 3:
        return ["350 km", "250 km", "150 km"]
    return channel_names


def default_panel_tags(n_panels):
    return [f"({chr(ord('a') + idx)})" for idx in range(n_panels)]


def infer_title(channel_names):
    prefixes = {name.split("_level_idx_")[0] for name in channel_names}
    if prefixes == {"EDens"}:
        return "Electron Density Prediction vs Target"
    if len(prefixes) == 1:
        return f"{next(iter(prefixes))} Prediction vs Target"
    return "Tracer Prediction vs Target"


def infer_units_label(channel_names):
    prefixes = {name.split("_level_idx_")[0] for name in channel_names}
    if prefixes == {"EDens"}:
        return r"$10^6$ cm$^{-3}$"
    return ""


def infer_value_scale(channel_names):
    prefixes = {name.split("_level_idx_")[0] for name in channel_names}
    if prefixes == {"EDens"}:
        return 1.0e-6
    return 1.0


def lead_hours_from_config(config_section):
    dhours = config_section.get("dhours")
    dt_step = config_section.get("dt", 1)
    if dhours is None:
        return None
    return float(dhours) * float(dt_step)


def parse_start_time(value):
    if value is None:
        return None
    cleaned = value.replace("Z", "+00:00")
    return dt.datetime.fromisoformat(cleaned)


def column_labels(args, config_section):
    if args.column_labels is not None:
        if len(args.column_labels) != len(args.leads):
            raise ValueError("--column-labels must have the same length as --leads.")
        return args.column_labels

    start_time = parse_start_time(args.start_time)
    if start_time is not None:
        lead_hours = args.lead_hours
        if lead_hours is None:
            lead_hours = lead_hours_from_config(config_section)
        if lead_hours is None:
            raise ValueError("--lead-hours is required when --start-time is used.")
        return [
            (start_time + dt.timedelta(hours=lead * lead_hours)).strftime(args.date_format)
            for lead in args.leads
        ]

    return [f"lead index {lead}" for lead in args.leads]


def roll_longitude(data, enabled):
    if not enabled:
        return data
    return np.roll(data, data.shape[-1] // 2, axis=-1)


def finite_values(*arrays):
    data = np.concatenate([np.ravel(np.asarray(a)) for a in arrays])
    return data[np.isfinite(data)]


def color_limits(*arrays, vmin=None, vmax=None, vmax_percentile=100.0, symmetric=False):
    data = finite_values(*arrays)
    if data.size == 0:
        raise ValueError("No finite values found for color limits.")

    auto_vmin = float(np.nanmin(data))
    if vmax_percentile >= 100.0:
        auto_vmax = float(np.nanmax(data))
    else:
        auto_vmax = float(np.nanpercentile(data, vmax_percentile))

    if symmetric or (vmin is None and vmax is None and auto_vmin < 0.0 < auto_vmax):
        bound = max(abs(auto_vmin), abs(auto_vmax))
        return -bound if vmin is None else vmin, bound if vmax is None else vmax

    lower = vmin if vmin is not None else (0.0 if auto_vmin >= 0.0 else auto_vmin)
    upper = vmax if vmax is not None else auto_vmax
    if lower == upper:
        upper = lower + 1.0
    return lower, upper


def imshow_map(ax, data, *, cmap, vmin, vmax):
    if ccrs is not None:
        kwargs = {
            "cmap": cmap,
            "origin": "lower",
            "extent": [-180, 180, -90, 90],
            "vmin": vmin,
            "vmax": vmax,
            "transform": ccrs.PlateCarree(),
        }
        im = ax.imshow(data, **kwargs)
        ax.set_global()
        ax.coastlines(linewidth=0.35)
        if "geo" in ax.spines:
            ax.spines["geo"].set_linewidth(0.45)
    else:
        nlat, nlon = data.shape
        lon_edges = np.linspace(-np.pi, np.pi, nlon + 1)
        lat_edges = np.linspace(-np.pi / 2.0, np.pi / 2.0, nlat + 1)
        im = ax.pcolormesh(
            lon_edges,
            lat_edges,
            data,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            shading="auto",
            rasterized=True,
        )
        ax.set_frame_on(True)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def add_map_axis(fig, gridspec_slot):
    if ccrs is not None:
        return fig.add_subplot(gridspec_slot, projection=ccrs.Mollweide())
    return fig.add_subplot(gridspec_slot, projection="mollweide")


def add_group_labels(fig, axes, height_labels, model_label, target_label):
    fig.canvas.draw()
    panel_tags = default_panel_tags(len(height_labels))
    for group_idx, height_label in enumerate(height_labels):
        pred_row = 2 * group_idx
        targ_row = pred_row + 1
        first_pred = axes[pred_row, 0]
        first_targ = axes[targ_row, 0]

        first_pred.text(
            -0.08,
            0.5,
            model_label,
            transform=first_pred.transAxes,
            rotation=90,
            va="center",
            ha="right",
            fontsize=9,
            family="serif",
        )
        first_targ.text(
            -0.08,
            0.5,
            target_label,
            transform=first_targ.transAxes,
            rotation=90,
            va="center",
            ha="right",
            fontsize=9,
            family="serif",
        )

        top = first_pred.get_position()
        bottom = first_targ.get_position()
        x = top.x0 - 0.07
        y = 0.5 * (top.y1 + bottom.y0)
        fig.text(
            x,
            y,
            f"{panel_tags[group_idx]} {height_label}",
            rotation=90,
            va="center",
            ha="center",
            fontsize=11,
            fontweight="bold",
            family="serif",
        )


def main():
    args = parse_args()
    config_section = load_config_section(args.config, args.config_name)
    channel_names = list(config_section["tracer_channel_names"])
    selected_channels = resolve_channels(args, channel_names)
    channel_indices = [idx for idx, _ in selected_channels]
    selected_channel_names = [name for _, name in selected_channels]

    if args.height_labels is None:
        height_labels = default_height_labels(selected_channel_names)
    else:
        if len(args.height_labels) != len(selected_channel_names):
            raise ValueError("--height-labels must have the same length as selected channels.")
        height_labels = args.height_labels

    title = args.title or infer_title(selected_channel_names)
    units_label = args.units_label if args.units_label is not None else infer_units_label(selected_channel_names)
    value_scale = args.value_scale if args.value_scale is not None else infer_value_scale(selected_channel_names)
    col_labels = column_labels(args, config_section)

    if (args.pred_path is None) != (args.targ_path is None):
        raise ValueError("--pred-path and --targ-path must be provided together.")

    if args.pred_path is not None:
        pred_path = args.pred_path
        targ_path = args.targ_path
    else:
        pred_path = args.raw_dir / f"tracer_pred_rank{args.rank:04d}_batch{args.batch:05d}.npy"
        targ_path = args.raw_dir / f"tracer_targ_rank{args.rank:04d}_batch{args.batch:05d}.npy"
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)
    if not targ_path.exists():
        raise FileNotFoundError(targ_path)

    pred = np.load(pred_path, mmap_mode="r")
    targ = np.load(targ_path, mmap_mode="r")
    if pred.shape != targ.shape:
        raise ValueError(f"prediction shape {pred.shape} != target shape {targ.shape}")
    if max(channel_indices) >= pred.shape[1]:
        raise IndexError(
            f"selected channel index {max(channel_indices)} is outside raw-output "
            f"channel range [0, {pred.shape[1] - 1}]"
        )

    n_leads = pred.shape[0]
    bad_leads = [lead for lead in args.leads if lead < 0 or lead >= n_leads]
    if bad_leads:
        raise IndexError(f"lead indices {bad_leads} are outside [0, {n_leads - 1}]")

    pred_maps = np.asarray(pred[np.ix_(args.leads, channel_indices)]) * value_scale
    targ_maps = np.asarray(targ[np.ix_(args.leads, channel_indices)]) * value_scale
    pred_maps = roll_longitude(pred_maps, not args.no_roll_longitude)
    targ_maps = roll_longitude(targ_maps, not args.no_roll_longitude)

    n_groups = len(channel_indices)
    n_cols = len(args.leads)
    n_rows = 2 * n_groups

    if args.shared_color_scale:
        shared_limits = color_limits(
            pred_maps,
            targ_maps,
            vmin=args.vmin,
            vmax=args.vmax,
            vmax_percentile=args.vmax_percentile,
            symmetric=args.symmetric,
        )
    else:
        shared_limits = None

    fig_width = 1.95 * n_cols + 1.05
    fig_height = 1.00 * n_rows + 1.05
    fig = plt.figure(figsize=(fig_width, fig_height), facecolor="white")
    grid = fig.add_gridspec(
        n_rows,
        n_cols + 1,
        width_ratios=[1.0] * n_cols + [0.045],
        hspace=0.06,
        wspace=0.0,
    )
    axes = np.empty((n_rows, n_cols), dtype=object)

    for row in range(n_rows):
        for col in range(n_cols):
            axes[row, col] = add_map_axis(fig, grid[row, col])

    for col, label in enumerate(col_labels):
        axes[0, col].set_title(label, fontsize=10, fontweight="bold", family="serif", pad=8)

    fig.subplots_adjust(left=0.12, right=0.94, top=0.88, bottom=0.04)

    for group_idx in range(n_groups):
        pred_row = 2 * group_idx
        targ_row = pred_row + 1
        if shared_limits is None:
            vmin, vmax = color_limits(
                pred_maps[:, group_idx],
                targ_maps[:, group_idx],
                vmin=args.vmin,
                vmax=args.vmax,
                vmax_percentile=args.vmax_percentile,
                symmetric=args.symmetric,
            )
        else:
            vmin, vmax = shared_limits

        group_im = None
        for col in range(n_cols):
            group_im = imshow_map(
                axes[pred_row, col],
                pred_maps[col, group_idx],
                cmap=args.cmap,
                vmin=vmin,
                vmax=vmax,
            )
            imshow_map(
                axes[targ_row, col],
                targ_maps[col, group_idx],
                cmap=args.cmap,
                vmin=vmin,
                vmax=vmax,
            )

        top_pos = axes[pred_row, -1].get_position()
        bottom_pos = axes[targ_row, -1].get_position()
        group_height = top_pos.y1 - bottom_pos.y0
        cax = fig.add_axes(
            [
                top_pos.x1 + 0.018,
                bottom_pos.y0 + 0.02 * group_height,
                0.012,
                0.96 * group_height,
            ]
        )
        cbar = fig.colorbar(group_im, cax=cax)
        if units_label:
            cbar.set_label(units_label, rotation=90, labelpad=8, fontsize=9)
        cbar.ax.tick_params(labelsize=8)

    add_group_labels(fig, axes, height_labels, args.model_label, args.target_label)
    fig.suptitle(title, fontsize=12, fontweight="bold", family="serif", y=0.965)

    out = args.out
    if out is None:
        safe_channels = "-".join(name.replace("/", "_") for name in selected_channel_names)
        out = args.raw_dir / f"plot_rank{args.rank:04d}_batch{args.batch:05d}_{safe_channels}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print("channels:", ", ".join(selected_channel_names))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
