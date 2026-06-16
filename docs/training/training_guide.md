# Training Guide

## Active training entrypoints

- Skeleton: `scripts/train/train_skeleton.py`
- Regions: `scripts/train/train_regions.py`
- Fusion: `scripts/train/train_gated_fusion.py`

## Main config paths

- Skeleton NSLT1000 selected_31: `configs/train/skeleton/nslt1000/selected_31/stgcnpp_ce.yaml`
- Regions NSLT1000 face-hands: `configs/train/regions/nslt1000/full/region_resnet18_gru_ce.yaml`
- Fusion NSLT1000: `configs/train/fusion/gated_feature/nslt1000/gated_feature_fusion_ce.yaml`

## Typical flow

1. Build preprocessing outputs with `scripts/preprocess/`.
2. Train a branch model with one config under `configs/train/`.
3. Evaluate with the matching script under `scripts/evaluate/`.
4. Use `scripts/verify/` when preparing packaging or checking dataset integrity.

## Examples

```bash
python scripts/train/train_skeleton.py --config configs/train/skeleton/nslt1000/selected_31/stgcnpp_ce.yaml
python scripts/train/train_regions.py --config configs/train/regions/nslt1000/full/region_resnet18_gru_ce.yaml
python scripts/train/train_gated_fusion.py --config configs/train/fusion/gated_feature/nslt1000/gated_feature_fusion_ce.yaml
```

## Notes

- Use the project environment that already has PyTorch and the required training dependencies installed.
- Fusion training currently depends on the validated skeleton and regions backbone artifacts referenced in the fusion config.
- The model registry is metadata-only; training still reads concrete config and checkpoint paths from the training configs.
