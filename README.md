# JAMES Plotting Review Package

Created: 2026-06-11

This folder contains only the plotting script and the compact data needed to reproduce a tracer prediction-vs-target figure. It intentionally excludes training code, inference code, PBS jobs, model checkpoints, and the `makani/` source tree.

## Contents

- `plotting/plot_tracer_raw_outputs.py`: standalone Matplotlib script for tracer prediction-vs-target map grids.
- `config/tracer_plot.yaml`: minimal plotting config with tracer channel names and lead timing.
- `sample_data/tracer_raw_outputs/`: one raw prediction/target pair plus the rank manifest.
- `figures/`: generated example figure output.

## Directory Layout

```text
.
├── README.md
├── FILE_MANIFEST.tsv
├── SHA256SUMS
├── config/
│   └── tracer_plot.yaml
├── figures/
│   └── tracer_run01_example.png
├── plotting/
│   └── plot_tracer_raw_outputs.py
└── sample_data/
    └── tracer_raw_outputs/
        ├── manifest_rank0000.tsv
        ├── tracer_pred_rank0000_batch00001.npy
        └── tracer_targ_rank0000_batch00001.npy
```

## Requirements

The plotting example requires `numpy`, `pyyaml`, and `matplotlib`. `cartopy` is optional; if unavailable, the script falls back to Matplotlib's built-in Mollweide projection.

## Example

From this package root:

```bash
python plotting/plot_tracer_raw_outputs.py \
  --raw-dir sample_data/tracer_raw_outputs \
  --config config/tracer_plot.yaml \
  --config-name sfno_tracer_coupled \
  --rank 0 \
  --batch 1 \
  --channels UI_level_idx_30 UI_level_idx_60 UI_level_idx_90 \
  --leads 3 13 18 \
  --start-time 2015-03-16T09:00:00 \
  --out figures/tracer_run01_example.png
```

The expected inputs for this command are:

- `sample_data/tracer_raw_outputs/tracer_pred_rank0000_batch00001.npy`
- `sample_data/tracer_raw_outputs/tracer_targ_rank0000_batch00001.npy`
- `config/tracer_plot.yaml`
