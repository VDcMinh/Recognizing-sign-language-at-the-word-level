# NSLT100 Skeleton + Regions Late Fusion Workspace

This workspace is reserved for late-fusion evaluation artifacts only.

Please copy these files here manually before running the fusion evaluator:

- `artifacts/fusion/nslt100/checkpoints/skeleton/best.pt`
- `artifacts/fusion/nslt100/checkpoints/regions/best.pt`

If available, also copy the resolved configs used to produce those checkpoints:

- `artifacts/fusion/nslt100/configs/skeleton/config_resolved.yaml`
- `artifacts/fusion/nslt100/configs/regions/config_resolved.yaml`

Notes:

- Checkpoints and cached logits are ignored by git.
- Keep the `.gitkeep` files and this `README.md` committed.
- The fusion scripts will write logits into `artifacts/fusion/nslt100/logits/`.
- The fusion scripts will write reports into `artifacts/fusion/nslt100/reports/`.

Quick start:

```bash
python scripts/check_fusion_workspace.py --workspace artifacts/fusion/nslt100
python scripts/evaluate_skeleton_region_late_fusion.py --fusion-config configs/fusion/nslt100_skeleton_regions_late_fusion.yaml --dry-run
```
