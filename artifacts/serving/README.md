# Serving Artifacts

This directory is the stable landing zone for future UI or backend inference artifacts.

Use one fixed slot per `branch/subset`:

- `skeleton/nslt100`
- `skeleton/nslt300`
- `skeleton/nslt1000`
- `regions/nslt100`
- `regions/nslt300`
- `regions/nslt1000`
- `fusion/nslt100`
- `fusion/nslt300`
- `fusion/nslt1000`

For each slot, later replace the placeholder files with real artifacts:

- `best.pt`
- `config_resolved.yaml`
- `metrics.json`
- `train_log.csv`

Do not place checkpoints inside `model_registry/`.  
The registry should only reference files stored here.

Current setup:

- serving artifacts come from `artifacts/serving/<branch>/<subset>/`
- class maps are referenced from `data/datasets/WLASL/index/subsets/<subset>/label_map.json`
