# ST-GCN++ Integration

The active skeleton runtime model is `slr.branches.skeleton.models.stgcnpp.STGCNPP`.

## Relevant files

- Model: `src/slr/branches/skeleton/models/stgcnpp.py`
- Builder: `src/slr/branches/skeleton/models/__init__.py`
- Graph utilities: `src/slr/branches/skeleton/graph.py`
- Training script: `scripts/train/train_skeleton.py`

## Config families

- `configs/train/skeleton/nslt100/selected_31/`
- `configs/train/skeleton/nslt300/selected_31/`
- `configs/train/skeleton/nslt1000/selected_31/`

## Verified artifact reference

- Resolved config: `artifacts/fusion/nslt1000/configs/skeleton/config_resolved.yaml`
- Checkpoint: `artifacts/fusion/nslt1000/checkpoints/skeleton/best.pt`

These validated artifact paths are also referenced from `model_registry/models/skeleton_nslt1000_sel31/model.yaml`.
