# Skeleton Training Baseline

## Purpose

This baseline trains a simple skeleton graph model on precomputed WLASL graph tensors for:

- `selected_27`
- `selected_31`

The current baseline is intended to validate the end-to-end training stack before replacing the model with ST-GCN++ or CTR-GCN.

## Data Requirement

Train-ready skeleton data must already exist under:

```text
data/datasets/WLASL/branch_inputs/skeleton/rtmw_l/
```

Expected tensor shapes:

- `selected_27`: `(3, 150, 27, 1)`
- `selected_31`: `(3, 150, 31, 1)`

This training flow does not rebuild graph tensors, does not run pose extraction, and does not read raw videos or frames.

## Experiment Configs

Baseline configs:

- `configs/train/skeleton_selected_27_baseline.yaml`
- `configs/train/skeleton_selected_31_baseline.yaml`

Important fields:

- `experiment.name`: run name used for the output folder
- `train.learning_rate`: optimizer learning rate
- `dataloader.batch_size`: batch size
- `train.weight_decay`: weight decay
- `train.epochs`: number of epochs
- `logging.use_wandb`: enable or disable W&B logging

## Run Examples

Smoke run for `selected_27`:

```bash
python scripts/train_skeleton.py --config configs/train/skeleton_selected_27_baseline.yaml --run-name sel27-test1 --epochs 2
```

Smoke run for `selected_31`:

```bash
python scripts/train_skeleton.py --config configs/train/skeleton_selected_31_baseline.yaml --run-name sel31-test1 --epochs 2
```

Override hyperparameters from CLI:

```bash
python scripts/train_skeleton.py --config configs/train/skeleton_selected_31_baseline.yaml --run-name sel31-lr5e4-bs32 --lr 0.0005 --batch-size 32 --weight-decay 0.0001
```

Useful overrides:

- `--run-name`
- `--epochs`
- `--lr`
- `--batch-size`
- `--weight-decay`
- `--dropout`
- `--device`
- `--seed`
- `--num-workers`
- `--limit-train`
- `--limit-val`
- `--limit-test`
- `--dry-run`

## W&B Setup

Set the API key and entity through environment variables.

Linux or Kaggle:

```bash
export WANDB_API_KEY=...
export WANDB_ENTITY=...
```

Windows PowerShell:

```powershell
$env:WANDB_API_KEY="..."
$env:WANDB_ENTITY="..."
```

Notes:

- Do not hard-code secrets in config files.
- The default W&B project is `wlasl-skeleton`.
- If `wandb` is unavailable or auth vars are missing, the code reports a clear message and disables W&B logging.

## Output Directory

Each run writes to:

```text
outputs/skeleton/<run_name>/
```

Files:

- `checkpoints/best.pt`
- `checkpoints/last.pt`
- `config_resolved.yaml`
- `metrics.json`
- `train_log.csv`
- `summary.json`

## Download Model From Kaggle

From the Kaggle working directory:

```bash
cd /kaggle/working/Recognizing-sign-language-at-the-word-level
zip -r /kaggle/working/sel31-test1_outputs.zip outputs/skeleton/sel31-test1
```

Then download `sel31-test1_outputs.zip` from the Kaggle Output tab.

## Evaluate Locally

After downloading and unzipping the output folder into your local repo:

```bash
python scripts/evaluate_skeleton.py --config outputs/skeleton/sel31-test1/config_resolved.yaml --checkpoint outputs/skeleton/sel31-test1/checkpoints/best.pt --split test
```

Optional overrides:

- `--batch-size`
- `--device`

The evaluation command also writes:

```text
outputs/skeleton/<run_name>/eval_test_best.json
```

when `experiment.output_dir` exists in the resolved config.

## Next Steps

- Replace `SimpleSTGCN` with ST-GCN++ or CTR-GCN
- Add standard label smoothing
- Add LanguageLS
