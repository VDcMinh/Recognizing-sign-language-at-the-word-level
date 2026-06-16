# Skeleton Training Baseline

## Primary configs

- `configs/train/skeleton/nslt100/selected_31/stgcnpp_ce.yaml`
- `configs/train/skeleton/nslt300/selected_31/stgcnpp_ce.yaml`
- `configs/train/skeleton/nslt1000/selected_31/stgcnpp_ce.yaml`

## Variants

- Standard label smoothing variants live beside the baseline config under the same subset/keypoint-set directory.
- `selected_27` configs are available under the matching `selected_27/` directories.

## Entry script

```bash
python scripts/train/train_skeleton.py --config configs/train/skeleton/nslt1000/selected_31/stgcnpp_ce.yaml
```

## Expected inputs

- Skeleton branch root: `data/datasets/WLASL/branch_inputs/skeleton/rtmw_l`
- Manifests: `.../manifests/nslt1000_selected_31_{train,val,test}.csv`
- Tensor shape: `[3, 150, 31, 1]`

## Runtime model

- Class path: `slr.branches.skeleton.models.stgcnpp.STGCNPP`
- Registry entry: `skeleton_nslt1000_sel31_v1`
