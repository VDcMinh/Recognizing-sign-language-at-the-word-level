# Fusion Serving Slots

Each subset directory here is the canonical UI-serving location for one gated fusion model.

Drop future files into:

- `nslt100/`
- `nslt300/`
- `nslt1000/`

Expected filenames:

- `best.pt`
- `config_resolved.yaml`
- `metrics.json`
- `train_log.csv`

Class map is currently read from:

- `data/datasets/WLASL/index/subsets/<subset>/label_map.json`

If a fusion training run writes `training_history.csv` or `training_summary.json`, copy them here as:

- `train_log.csv`
- `metrics.json`
