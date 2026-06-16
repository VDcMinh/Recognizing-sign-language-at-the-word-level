# ST-GCN++ Implementation Log

## Scope

This file records the work completed for integrating ST-GCN++ into the
`skeleton` branch training pipeline, along with the issues encountered during
implementation and validation.

Task goal:

- add an ST-GCN++-compatible model for skeleton graph tensors
- keep the existing training and evaluation pipeline unchanged
- support `selected_27` and `selected_31`
- keep `SimpleSTGCN` intact
- avoid rebuilding data or adding heavy dependencies

## What Was Implemented

### 1. New model: repo-local ST-GCN++

Added:

- `src/slr/branches/skeleton/models/stgcnpp.py`

This is a clean-room PyTorch implementation, not a direct copy from PYSKL,
MMAction2, or another upstream repository.

Design choices:

- pure PyTorch only
- accepts graph adjacency from `SkeletonGraph`
- works with input shape `(N, C, T, V, M)`
- supports variable `V`, so it works for both:
  - `selected_27`
  - `selected_31`
- includes:
  - spatial graph convolution
  - multi-branch temporal convolution
  - residual connections
  - batch normalization
  - dropout
  - global average pooling
  - linear classifier

Main classes:

- `SpatialGraphConv`
- `MultiScaleTemporalConv`
- `STGCNPPBlock`
- `STGCNPP`

### 2. Model factory integration

Updated:

- `src/slr/branches/skeleton/models/__init__.py`

`build_skeleton_model(cfg, graph)` now supports:

- `model.name: simple_stgcn`
- `model.name: stgcnpp`

Invalid names now raise a clear `ValueError`.

### 3. Training config defaults

Updated:

- `src/slr/branches/skeleton/train.py`

Added default model keys used by the new ST-GCN++ path:

- `base_channels`
- `stage_channels`
- `temporal_strides`

This keeps the existing resolved-config flow compatible with the new model.

### 4. New train configs

Added:

- `configs/train/skeleton_selected_27_stgcnpp.yaml`
- `configs/train/skeleton_selected_31_stgcnpp.yaml`

These configs:

- keep the existing dataset paths and graph setup
- switch the model to `stgcnpp`
- use SGD + cosine scheduler
- preserve CLI override compatibility

### 5. Documentation

Added:

- `docs/skeleton_stgcnpp_integration.md`

Updated:

- `docs/skeleton_training_baseline.md`
- `TRAINING.md`

The new doc covers:

- purpose of the integration
- input format
- new config files
- run commands
- Kaggle notes
- current limitations
- next steps

## Files Added

- `IMPLEMENTATION.md`
- `src/slr/branches/skeleton/models/stgcnpp.py`
- `configs/train/skeleton_selected_27_stgcnpp.yaml`
- `configs/train/skeleton_selected_31_stgcnpp.yaml`
- `docs/skeleton_stgcnpp_integration.md`

## Files Modified

- `src/slr/branches/skeleton/models/__init__.py`
- `src/slr/branches/skeleton/train.py`
- `docs/skeleton_training_baseline.md`
- `TRAINING.md`

## Validation Commands Run

All checks were run with:

```powershell
.\.venv-rtmw310\Scripts\python.exe
```

because the default system `python` in this workspace does not provide
`torch`.

### Import check

```powershell
.\.venv-rtmw310\Scripts\python.exe -c "from slr.branches.skeleton.models import build_skeleton_model; print('OK')"
```

Result:

- passed
- output: `OK`

### Compile check

```powershell
.\.venv-rtmw310\Scripts\python.exe -m compileall src\slr\branches\skeleton\models src\slr\branches\skeleton\train.py scripts\train_skeleton.py scripts\evaluate_skeleton.py
```

Result:

- passed

### Dry-run: selected_27

```powershell
.\.venv-rtmw310\Scripts\python.exe scripts\train_skeleton.py --config configs\train\skeleton_selected_27_stgcnpp.yaml --run-name dry-sel27-stgcnpp --dry-run --no-wandb
```

Result:

- passed
- batch shape: `(64, 3, 150, 27, 1)`
- logits shape: `(64, 100)`

### Dry-run: selected_31

```powershell
.\.venv-rtmw310\Scripts\python.exe scripts\train_skeleton.py --config configs\train\skeleton_selected_31_stgcnpp.yaml --run-name dry-sel31-stgcnpp --dry-run --no-wandb
```

Result:

- passed
- batch shape: `(64, 3, 150, 31, 1)`
- logits shape: `(64, 100)`

### Smoke train: selected_27

```powershell
.\.venv-rtmw310\Scripts\python.exe scripts\train_skeleton.py --config configs\train\skeleton_selected_27_stgcnpp.yaml --run-name smoke-sel27-stgcnpp --epochs 1 --limit-train 32 --limit-val 16 --limit-test 16 --batch-size 8 --num-workers 0 --no-wandb
```

Result:

- passed
- output dir: `outputs/skeleton/smoke-sel27-stgcnpp/`
- checkpoints created:
  - `best.pt`
  - `last.pt`
- metrics created:
  - `metrics.json`
  - `train_log.csv`
  - `summary.json`

Metrics:

- `best_val_top5 = 0.375`
- `test_loss = 4.499626874923706`
- `test_top1 = 0.0625`
- `test_top5 = 0.1875`

### Smoke train: selected_31

```powershell
.\.venv-rtmw310\Scripts\python.exe scripts\train_skeleton.py --config configs\train\skeleton_selected_31_stgcnpp.yaml --run-name smoke-sel31-stgcnpp --epochs 1 --limit-train 32 --limit-val 16 --limit-test 16 --batch-size 8 --num-workers 0 --no-wandb
```

Result:

- passed
- output dir: `outputs/skeleton/smoke-sel31-stgcnpp/`
- checkpoints created:
  - `best.pt`
  - `last.pt`
- metrics created:
  - `metrics.json`
  - `train_log.csv`
  - `summary.json`

Metrics:

- `best_val_top5 = 0.375`
- `test_loss = 4.49727988243103`
- `test_top1 = 0.0625`
- `test_top5 = 0.25`

### Evaluate checkpoint

```powershell
.\.venv-rtmw310\Scripts\python.exe scripts\evaluate_skeleton.py --config outputs\skeleton\smoke-sel31-stgcnpp\config_resolved.yaml --checkpoint outputs\skeleton\smoke-sel31-stgcnpp\checkpoints\best.pt --split test --batch-size 8
```

Result:

- passed
- output:
  - `loss = 4.49727988243103`
  - `top1 = 0.0625`
  - `top5 = 0.25`

## Checkpoint Metadata Verified

The saved checkpoint metadata was checked and contains the expected values:

- `model_name = stgcnpp`
- `keypoint_set = selected_31`
- `num_nodes = 31`

This confirms compatibility with the existing checkpoint save/load flow.

## Problems Encountered

### 1. `TRAINING.md` encoding issues in terminal output

Observed:

- `TRAINING.md` content appeared with mojibake in the current PowerShell
  terminal session.

Impact:

- reading and patching the file by exact line context became unreliable

Handling:

- I relied on code and structured docs for the source of truth
- I inspected the tail/head of the file before patching
- I appended the ST-GCN++ note carefully after confirming the visible content

Status:

- not a blocker
- file was updated, but terminal display still shows encoding artifacts in this
  shell environment

### 2. Default system Python was not usable for torch-based validation

Observed:

- the workspace notes and prior docs already suggested that the default
  `python` did not have `torch`

Impact:

- import checks and training validation could not safely assume `python`

Handling:

- all validation commands were run with:
  - `.\.venv-rtmw310\Scripts\python.exe`

Status:

- resolved for this task

### 3. Preexisting dataset warnings about non-contiguous class IDs

Observed during dry-run and smoke tests:

- warnings such as:
  - `class_id values are not contiguous. min=... max=... unique=...`

Cause:

- this comes from the dataset loader when using subset-limited or split-specific
  manifests
- for example, validation or test subsets may not contain all 100 classes

Impact:

- no crash
- no model issue
- no checkpoint issue

Handling:

- no code change was made here because the warning is legitimate and predates
  this integration

Status:

- expected behavior

### 4. Need to avoid heavy external dependencies

Constraint:

- the task explicitly required avoiding `mmcv` / `mmaction2` dependency for the
  training path

Handling:

- I implemented a clean-room ST-GCN++-compatible model in pure PyTorch instead
  of porting tightly coupled upstream code

Status:

- resolved

## Things Not Done

- no Language Label Smoothing
- no standard label smoothing integration into loss
- no CTR-GCN implementation
- no pose extraction changes
- no graph tensor rebuilding
- no W&B live smoke run was executed in this turn

## Current Status

The ST-GCN++ integration is complete enough to:

- build via config with `model.name: stgcnpp`
- run dry-run on `selected_27`
- run dry-run on `selected_31`
- train smoke runs on both configs
- save compatible checkpoints
- evaluate saved checkpoints using the existing evaluation script

The existing `SimpleSTGCN` baseline path remains intact.
