# JAMES Reviewer Code Package

Created: 2026-06-11

This folder collects the runnable inference/training code, trained weights, job scripts, plotting code, and compact sample outputs needed for reviewer inspection.

## Contents

- `code/`: current Makani source tree, configuration, dataset helpers, tracer utilities, `train.py`, `pyproject.toml`, original `README.md`, and `LICENSE`.
- `jobs/`: PBS scripts used for physical training, tracer training, physical inference, and tracer inference.
- `weights/`: selected trained checkpoints.
  - `physical_run03/training_checkpoints/best_ckpt_mp0.tar`
  - `physical_run04/training_checkpoints/best_ckpt_mp0.tar`
  - `tracer_run01/training_checkpoints/best_ckpt_mp0.tar`
  - `tracer_run02/training_checkpoints/best_ckpt_mp0.tar`
- `plotting/plot_tracer_raw_outputs.py`: script for prediction-vs-target tracer maps.
- `sample_outputs/`: compact metadata, metric arrays, selected figure outputs, and one raw tracer prediction/target example.
- `SHA256SUMS`: checksums for files in this package.

## Notes

The configuration in `code/config/sfnonet.yaml` still contains the original absolute data paths on Derecho/GLADE. Reviewers running outside that environment should update the data/statistics paths in the YAML or provide equivalent files at those paths.

The full raw forecast archive and full processed training/inference datasets are not copied here because they are much larger than the code and trained weights. This package includes representative raw tracer output for plotting validation.

The plotting example requires `numpy`, `pyyaml`, and `matplotlib`; `cartopy` is optional and used only when available.

## Example: Physical Inference

From this folder:

```bash
cd code
export PYTHONPATH=$(pwd):${PYTHONPATH}

torchrun --nproc_per_node=1 \
  makani/inference.py \
  --yaml_config=config/sfnonet.yaml \
  --config=sfno_linear_73chq_sc3_layers8_edim384_asgl2 \
  --mode=single \
  --single_time=2015-03-18T00:00:00 \
  --pretrained_checkpoint_path=../weights/physical_run03/training_checkpoints \
  --run_num=03
```

`physical_run04` is also included because it is referenced by `jobs/inference.pbs`.

## Example: Tracer Inference

```bash
cd code
export PYTHONPATH=$(pwd):${PYTHONPATH}

torchrun --nproc_per_node=1 \
  makani/inference.py \
  --yaml_config=config/sfnonet.yaml \
  --config=sfno_tracer_coupled \
  --mode=single \
  --single_time=2015-03-16T09:00:00 \
  --batch_size=1 \
  --tracer_checkpoint_path=../weights/tracer_run01/training_checkpoints \
  --pretrained_checkpoint_path=../weights/physical_run03/training_checkpoints \
  --run_num=01
```

`tracer_run02` is also included because it is the checkpoint produced by `jobs/tracer_train.pbs`.

## Example: Tracer Plotting

From the package root:

```bash
python plotting/plot_tracer_raw_outputs.py \
  --raw-dir sample_outputs/tracer_run01/tracer_raw_outputs \
  --config code/config/sfnonet.yaml \
  --config-name sfno_tracer_coupled \
  --rank 0 \
  --batch 1 \
  --channels UI_level_idx_30 UI_level_idx_60 UI_level_idx_90 \
  --leads 3 13 18 \
  --lead-hours 1 \
  --start-time 2015-03-16T09:00:00 \
  --out figures/tracer_run01_example.png
```

The script uses Cartopy when available; otherwise it falls back to Matplotlib's built-in Mollweide projection.
