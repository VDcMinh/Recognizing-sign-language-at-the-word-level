# Project Structure Guide

## Purpose

This repository is organized around a staged backend pipeline for WLASL word-level recognition. The cleanup keeps active runtime code, groups configs and scripts by task, and separates model metadata from checkpoints.

## Top-level directories

- `src/slr/`: active Python package
- `configs/`: preprocessing, build-input, training, experiment, and archive configs
- `scripts/`: task-focused CLI wrappers
- `model_registry/`: metadata-only registry for future backend/UI model selection
- `docs/`: active docs plus archived history
- `reports/current/`: current reports grouped by topic
- `archive/`: scaffolds, deprecated code, and moved placeholders

## Source layout

- `src/slr/data/`: dataset indexing and video standardization
- `src/slr/pose/`: RTMW-l pose extraction and pose utilities
- `src/slr/branches/skeleton/`: selected-keypoint build, dataset, graph, and models
- `src/slr/branches/regions/`: crop build, dataset, and region models
- `src/slr/branches/fusion/`: paired dataset, model build, training, and packaging helpers
- `src/slr/registry/`: registry schema, loader, and validation
- `src/slr/training/`: shared training helpers such as checkpointing, metrics, and optimizers
- `src/slr/utils/`: IO, bbox, logging, image, video, and seed helpers

## Config layout

- `configs/preprocessing/index/`
- `configs/preprocessing/standardize/`
- `configs/preprocessing/pose/`
- `configs/preprocessing/regions/`
- `configs/build_inputs/skeleton/`
- `configs/train/skeleton/`
- `configs/train/regions/`
- `configs/train/fusion/`
- `configs/experiments/`
- `configs/archive/`

See `configs/README.md` for examples.

## Script layout

- `scripts/preprocess/`: index, standardize, pose, skeleton-input, and region-input entrypoints
- `scripts/train/`: skeleton, regions, and fusion training entrypoints
- `scripts/evaluate/`: evaluation entrypoints
- `scripts/verify/`: validation and packaging checks
- `scripts/package/`: bundle and packaging utilities
- `scripts/dev/`: local smoke tests and visualization helpers
- `scripts/common/`: shared helpers for grouped scripts

## Model registry

`model_registry/` stores metadata only.

- `registry.yaml`: summary index
- `models/*/model.yaml`: per-model metadata

Checkpoint files remain in their existing artifact locations and are referenced through `artifacts.*.local_path`.

## Current runtime models

- Skeleton: `slr.branches.skeleton.models.stgcnpp.STGCNPP`
- Regions: `slr.branches.regions.models.region_resnet18_gru.RegionResNet18GRU`
- Fusion: `slr.branches.fusion.models.gated_feature_fusion.GatedFeatureFusion`

## Notes

- `data/` and `UI/` are intentionally not reorganized here.
- Historical docs may still describe older paths; active docs in `docs/architecture/`, `docs/training/`, and `README.md` reflect the cleaned structure.
