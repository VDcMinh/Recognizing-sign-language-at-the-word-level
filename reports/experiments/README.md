# Experiment Outputs

Store experiment artifacts in a branch-specific subdirectory under `experiments/`.

Recommended contents for each experiment run:

- resolved config snapshots
- training and evaluation logs
- metrics summaries
- confusion matrices or per-class reports
- prediction samples and qualitative notes

Avoid mixing preprocessing artifacts with experiment outputs. Preprocessing reports belong under `reports/preprocessing/` or the matching data layer report folders.
