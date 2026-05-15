# ST-GCN++ Integration for Skeleton Branch

## 1. Purpose

ST-GCN++ is the next main model for the skeleton branch.
`SimpleSTGCN` remains in the repo as a technical baseline used to verify the
training pipeline end to end.

This integration keeps the existing skeleton training stack intact:

- manifest-driven dataset loading
- `selected_27` and `selected_31` graph tensors
- CLI hyperparameter overrides
- optional W&B logging
- local checkpoint save/load
- local evaluation via `scripts/evaluate_skeleton.py`

## 2. Model Files

Added:

- `src/slr/branches/skeleton/models/stgcnpp.py`
- `configs/train/skeleton_selected_27_stgcnpp.yaml`
- `configs/train/skeleton_selected_31_stgcnpp.yaml`
- `docs/skeleton_stgcnpp_integration.md`

Updated:

- `src/slr/branches/skeleton/models/__init__.py`
- `src/slr/branches/skeleton/train.py`
- `docs/skeleton_training_baseline.md`
- `TRAINING.md`

## 3. Input Format

The model consumes the same precomputed graph tensors already used by the
current skeleton pipeline.

- `selected_27`: `(N, 3, 150, 27, 1)`
- `selected_31`: `(N, 3, 150, 31, 1)`

Where:

- `N`: batch size
- `C=3`: `x`, `y`, `confidence`
- `T=150`: fixed number of frames
- `V`: number of graph nodes
- `M=1`: number of persons

Adjacency comes from `SkeletonGraph` and has shape `(K, V, V)`, with `K=3`
under the current `spatial` strategy.

## 4. Configs

New train configs:

- `configs/train/skeleton_selected_27_stgcnpp.yaml`
- `configs/train/skeleton_selected_31_stgcnpp.yaml`

Both keep the same dataset and pipeline assumptions as the baseline configs,
but switch `model.name` to `stgcnpp`.

## 5. Run Commands

Dry-run `selected_27`:

```bash
python scripts/train_skeleton.py --config configs/train/skeleton_selected_27_stgcnpp.yaml --run-name dry-sel27-stgcnpp --dry-run --no-wandb
```

Dry-run `selected_31`:

```bash
python scripts/train_skeleton.py --config configs/train/skeleton_selected_31_stgcnpp.yaml --run-name dry-sel31-stgcnpp --dry-run --no-wandb
```

Smoke train `selected_27`:

```bash
python scripts/train_skeleton.py --config configs/train/skeleton_selected_27_stgcnpp.yaml --run-name smoke-sel27-stgcnpp --epochs 1 --limit-train 32 --limit-val 16 --limit-test 16 --batch-size 8 --num-workers 0 --no-wandb
```

Smoke train `selected_31`:

```bash
python scripts/train_skeleton.py --config configs/train/skeleton_selected_31_stgcnpp.yaml --run-name smoke-sel31-stgcnpp --epochs 1 --limit-train 32 --limit-val 16 --limit-test 16 --batch-size 8 --num-workers 0 --no-wandb
```

Full train `selected_27`:

```bash
python scripts/train_skeleton.py --config configs/train/skeleton_selected_27_stgcnpp.yaml --run-name sel27-stgcnpp-ce-001
```

Full train `selected_31`:

```bash
python scripts/train_skeleton.py --config configs/train/skeleton_selected_31_stgcnpp.yaml --run-name sel31-stgcnpp-ce-001
```

Evaluate:

```bash
python scripts/evaluate_skeleton.py --config outputs/skeleton/smoke-sel31-stgcnpp/config_resolved.yaml --checkpoint outputs/skeleton/smoke-sel31-stgcnpp/checkpoints/best.pt --split test --batch-size 8
```

## 6. Kaggle Notes

- Default config uses `batch_size=64`, but on Kaggle T4 you may need
  `--batch-size 32` or `--batch-size 16`.
- W&B entity can come from `WANDB_ENTITY` or `--wandb-entity`.
- The implementation is pure PyTorch and does not require `mmcv` or
  `mmaction2` for training on precomputed graph tensors.

## 7. Current Limitations

- This is a repo-local clean-room ST-GCN++-compatible implementation. It is
  not a direct code copy from PYSKL or MMAction2.
- The goal here is pipeline compatibility and a stronger skeleton backbone than
  `SimpleSTGCN`, not an exact upstream reproduction.
- Language Label Smoothing is intentionally out of scope for this task.
- CTR-GCN is intentionally out of scope for this task.

## 8. Next Steps

- Train `selected_27` with ST-GCN++ + cross entropy
- Train `selected_31` with ST-GCN++ + cross entropy
- Add standard label smoothing
- Add LanguageLS
- Add CTR-GCN
