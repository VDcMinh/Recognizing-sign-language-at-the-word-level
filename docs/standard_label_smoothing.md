# Standard Label Smoothing

## 1. Purpose

Standard Label Smoothing is added to the skeleton training pipeline to:

- regularize the classifier
- reduce overconfidence in the predicted class distribution
- provide a clean intermediate comparison point between plain cross entropy and LanguageLS

This task only adds standard label smoothing.
LanguageLS remains out of scope here.

## 2. Short Formula

For smoothing factor `epsilon`:

- correct class target becomes `1 - epsilon`
- the remaining `epsilon` mass is distributed uniformly across the wrong classes

In this repo, the implementation uses:

```python
torch.nn.CrossEntropyLoss(label_smoothing=epsilon)
```

The dataset still returns the same hard `class_id` labels as before.

## 3. Configs

Added configs:

- `configs/train/skeleton_selected_31_stgcnpp_standardls_eps01.yaml`
- `configs/train/skeleton_selected_31_stgcnpp_standardls_eps03.yaml`

These keep the same skeleton pipeline and ST-GCN++ model family, but switch:

- `train.loss: standard_label_smoothing`

and provide:

- `label_smoothing.epsilon: 0.1`
- `label_smoothing.epsilon: 0.3`

## 4. Run Examples

Smoke train with `epsilon=0.1`:

```bash
python scripts/train_skeleton.py --config configs/train/skeleton_selected_31_stgcnpp_standardls_eps01.yaml --run-name smoke-standardls-eps01 --epochs 1 --limit-train 32 --limit-val 16 --limit-test 16 --batch-size 8 --num-workers 0 --no-wandb
```

Smoke train with `epsilon=0.3`:

```bash
python scripts/train_skeleton.py --config configs/train/skeleton_selected_31_stgcnpp_standardls_eps03.yaml --run-name smoke-standardls-eps03 --epochs 1 --limit-train 32 --limit-val 16 --limit-test 16 --batch-size 8 --num-workers 0 --no-wandb
```

## 5. Recommended Full Runs

```bash
python scripts/train_skeleton.py --config configs/train/skeleton_selected_31_stgcnpp_standardls_eps01.yaml --run-name sel31-stgcnpp-standardls-eps01-lr005-bs16-wd001-ep180-v1
```

```bash
python scripts/train_skeleton.py --config configs/train/skeleton_selected_31_stgcnpp_standardls_eps03.yaml --run-name sel31-stgcnpp-standardls-eps03-lr005-bs16-wd001-ep180-v1
```

## 6. Comparison Plan

Recommended comparison sequence:

- CE baseline
- StandardLS `epsilon=0.1`
- StandardLS `epsilon=0.3`
- LanguageLS `epsilon=0.3` later
