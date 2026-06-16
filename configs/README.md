# Config Guide

## Structure

- `dataset/`: dataset-wide metadata
- `preprocessing/index/`: index build configs
- `preprocessing/standardize/`: video standardization configs
- `preprocessing/pose/`: RTMW-l pose extraction configs
- `preprocessing/regions/`: region-crop build configs
- `build_inputs/skeleton/`: selected-keypoint and graph-input configs
- `train/skeleton/`: skeleton training configs by subset and keypoint set
- `train/regions/`: regions training configs by subset and variant
- `train/fusion/`: fusion training configs
- `experiments/`: lightweight experiment configs
- `archive/`: legacy and deprecated configs kept for reference

## Common examples

- Pose extraction:
  `configs/preprocessing/pose/pose_rtmw_l.yaml`
- Region crop build:
  `configs/preprocessing/regions/region_crops_nslt1000.yaml`
- Skeleton input build:
  `configs/build_inputs/skeleton/nslt1000/selected_31.yaml`
- Skeleton training:
  `configs/train/skeleton/nslt1000/selected_31/stgcnpp_ce.yaml`
- Regions training:
  `configs/train/regions/nslt1000/full/region_resnet18_gru_ce.yaml`
- Fusion training:
  `configs/train/fusion/gated_feature/nslt1000/gated_feature_fusion_ce.yaml`

## Naming conventions

- Subset is encoded in the directory path, not only in the filename.
- Variants such as `standardls`, `aug`, `union`, `incremental`, or `debug` stay under the most specific branch/subset folder.
- Archived configs stay under `configs/archive/` and are not treated as active runtime configs.
