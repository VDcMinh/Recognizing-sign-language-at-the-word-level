# Fusion Checkpoint Placement

The UI supports two ways to wire `Skeleton + Fusion`.

## Option A: Single fused checkpoint
- `UI/checkpoints/fusion/best.pt`
- `UI/configs/fusion/config_resolved.yaml`

## Option B: Late fusion with two branches
- `UI/checkpoints/fusion/skeleton_best.pt`
- `UI/checkpoints/fusion/regions_best.pt`
- `UI/configs/fusion/skeleton_config_resolved.yaml`
- `UI/configs/fusion/regions_config_resolved.yaml`
- `UI/configs/fusion/fusion_config.yaml`

Do not commit real checkpoint files to git.
