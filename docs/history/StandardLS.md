# Standard Label Smoothing in Skeleton Training

## 1. Scope

Tai lieu nay mo ta chi tiet phan Standard Label Smoothing da duoc them vao
skeleton training pipeline trong repo nay.

Task nay chi cover:

- skeleton branch
- loss layer
- skeleton training / evaluation wiring
- config train moi
- logging va output metadata lien quan

Task nay khong cover:

- Language Label Smoothing
- pose extraction
- skeleton preprocessing / graph tensor rebuild
- data changes
- embedding similarity / language matrix

## 2. Muc tieu cua thay doi

Truoc thay doi, skeleton training chi support:

- `cross_entropy`

Sau thay doi, skeleton training support:

- `cross_entropy`
- `standard_label_smoothing`

Y tuong la giu nguyen toan bo dataset output va train loop, chi thay doi
factory build loss de cho phep dung:

```python
torch.nn.CrossEntropyLoss(label_smoothing=epsilon)
```

Nghia la:

- dataset van tra `class_id` hard label
- model output van la `logits`
- loss call van la `criterion(logits, labels)`
- metric top1/top5/top10 khong doi

## 3. Files da them / sua

### 3.1. Added

- `configs/train/skeleton_selected_31_stgcnpp_standardls_eps01.yaml`
- `configs/train/skeleton_selected_31_stgcnpp_standardls_eps03.yaml`
- `docs/standard_label_smoothing.md`
- `StandardLS.md`

### 3.2. Modified

- `src/slr/training/losses.py`
- `src/slr/branches/skeleton/train.py`

## 4. Loss logic da implement

### 4.1. File: `src/slr/training/losses.py`

Da bo sung cac helper sau:

- `get_loss_name(cfg)`
- `get_label_smoothing_epsilon(cfg)`
- `build_loss_from_config(cfg)`

Va van giu:

- `build_loss(...)`

de tranh pha backward compatibility.

### 4.2. `get_loss_name(cfg)`

Ham nay doc:

```python
cfg["train"]["loss"]
```

neu khong co thi fallback:

```python
"cross_entropy"
```

Loss name duoc normalize ve lowercase.

### 4.3. `get_label_smoothing_epsilon(cfg)`

Ham nay:

- neu loss la `cross_entropy` -> tra `0.0`
- neu loss la `standard_label_smoothing` -> doc
  `cfg["label_smoothing"]["epsilon"]`

Validation:

- phai convert duoc sang `float`
- phai thoa `0 <= epsilon < 1`

Neu khong hop le, code raise `ValueError` ro rang.

Vi du:

- thieu `label_smoothing.epsilon`
- `epsilon = "abc"`
- `epsilon = -0.1`
- `epsilon = 1.0`

deu se bi chan som.

### 4.4. `build_loss_from_config(cfg)`

Ham nay la entrypoint moi cho skeleton training.

Behavior:

- `train.loss == "cross_entropy"`
  -> `nn.CrossEntropyLoss()`

- `train.loss == "standard_label_smoothing"`
  -> `nn.CrossEntropyLoss(label_smoothing=epsilon)`

- loss name khac
  -> `ValueError("Unsupported loss type: ...")`

### 4.5. Vi sao van giu `build_loss(...)`

Repo da co code import `build_loss`.
De giam rui ro pha vo code cu, `build_loss(...)` van ton tai va duoc chuyen
huong noi bo sang `build_loss_from_config(...)`.

Nhu vay:

- code moi co API ro rang theo config
- code cu neu con goi `build_loss(...)` van khong bi vo

## 5. Train pipeline da duoc noi nhu the nao

### 5.1. File: `src/slr/branches/skeleton/train.py`

Truoc thay doi, criterion duoc build bang:

```python
build_loss(str(resolved_config["train"]["loss"]), ...)
```

Sau thay doi:

```python
criterion = build_loss_from_config(resolved_config)
```

Ca `run_training(...)` va `run_evaluation(...)` deu da dung chung path nay.

### 5.2. Khong doi train loop

Phan quan trong la train loop khong can sua logic tinh loss:

```python
loss = criterion(logits, labels)
```

Dieu nay hoat dong cho ca:

- CE
- StandardLS

vi ca hai deu la `CrossEntropyLoss`.

### 5.3. Khong doi dataset / metrics

Khong co thay doi nao o:

- dataset loader
- graph tensor
- labels
- accuracy top1/top5/top10
- checkpointing
- evaluate script

## 6. Metadata loss duoc gan vao runtime config

Trong `train.py`, da bo sung:

- `_attach_loss_metadata(config)`
- `_format_loss_log(config)`

### 6.1. `_attach_loss_metadata(config)`

Ham nay them vao `config["runtime"]`:

- `loss_type`
- `label_smoothing_epsilon`

Vi du:

CE:

```yaml
runtime:
  loss_type: cross_entropy
  label_smoothing_epsilon: 0.0
```

StandardLS eps 0.1:

```yaml
runtime:
  loss_type: standard_label_smoothing
  label_smoothing_epsilon: 0.1
```

### 6.2. `_format_loss_log(config)`

Ham nay phuc vu logging dau run.

Output:

- CE:
  `Loss: cross_entropy`

- StandardLS:
  `Loss: standard_label_smoothing epsilon=0.1`

hoac

`Loss: standard_label_smoothing epsilon=0.3`

## 7. Logging va output nao da thay doi

### 7.1. Terminal log

Dau moi run, logger in them:

- `Loss: cross_entropy`
hoac
- `Loss: standard_label_smoothing epsilon=...`

Moi epoch van giu nguyen log metrics:

- `train_loss`
- `train_top1`
- `train_top5`
- `train_top10`
- `val_loss`
- `val_top1`
- `val_top5`
- `val_top10`

### 7.2. `config_resolved.yaml`

Da chua:

- `train.loss`
- `label_smoothing.epsilon` neu dung StandardLS
- `runtime.loss_type`
- `runtime.label_smoothing_epsilon`

### 7.3. `summary.json`

Da bo sung:

- `loss_type`
- `label_smoothing_epsilon`

### 7.4. `metrics.json`

Da bo sung:

- `loss_type`
- `label_smoothing_epsilon`

Ngoai ra cac metric cu van duoc giu:

- `best_epoch`
- `best_val_top1`
- `best_val_top5`
- `best_val_top10`
- `test_loss`
- `test_top1`
- `test_top5`
- `test_top10`
- `final_train_loss`
- `final_val_loss`

### 7.5. W&B

Smoke run da dung `--no-wandb`, nhung ve mat code:

- `resolved_config` truyen vao `wandb.init(...)` da chua `runtime.loss_type`
  va `runtime.label_smoothing_epsilon`
- tag config moi da them:
  - `standard_label_smoothing`
  - `eps01` hoac `eps03`

Khong can doi epoch metric schema cho W&B vi loss type khong phat sinh metric moi
moi epoch.

## 8. Config moi da tao

### 8.1. `configs/train/skeleton_selected_31_stgcnpp_standardls_eps01.yaml`

Main setting:

- subset: `nslt100`
- keypoint_set: `selected_31`
- model: `stgcnpp`
- batch_size: `16`
- epochs: `180`
- optimizer: `sgd`
- learning_rate: `0.05`
- momentum: `0.9`
- weight_decay: `0.001`
- loss: `standard_label_smoothing`
- `label_smoothing.epsilon: 0.1`

Run name mac dinh:

- `sel31-stgcnpp-standardls-eps01-lr005-bs16-wd001-ep180-v1`

### 8.2. `configs/train/skeleton_selected_31_stgcnpp_standardls_eps03.yaml`

Main setting giong config tren, chi khac:

- `label_smoothing.epsilon: 0.3`

Run name mac dinh:

- `sel31-stgcnpp-standardls-eps03-lr005-bs16-wd001-ep180-v1`

## 9. Cach Standard Label Smoothing hoat dong

Cho bai toan phan loai `C` class:

- hard label binh thuong: target la one-hot
- label smoothing: target one-hot duoc lam mem

Neu `epsilon = 0.1`:

- class dung nhan xac suat `0.9`
- `0.1` con lai duoc phan deu cho cac class sai

Neu `epsilon = 0.3`:

- class dung nhan xac suat `0.7`
- `0.3` con lai duoc phan deu cho cac class sai

Tac dung:

- giam overconfidence
- tang regularization
- co the giup generalization tot hon

Tradeoff:

- `epsilon` qua cao co the lam model hoc cham hon
- `epsilon` qua cao co the lam giam do sac net cua xac suat class dung

Voi task nay, hai muc dang thu la:

- `0.1`
- `0.3`

## 10. Smoke tests da chay

### 10.1. Import check

Command:

```bash
python -c "from slr.training.losses import build_loss_from_config; print('loss import OK')"
```

Ket qua:

- pass

### 10.2. CE smoke sau khi them StandardLS

Command:

```bash
python scripts/train_skeleton.py --config configs/train/skeleton_selected_31_stgcnpp.yaml --run-name smoke-ce-after-standardls --epochs 1 --limit-train 32 --limit-val 16 --limit-test 16 --batch-size 8 --num-workers 0 --no-wandb
```

Ket qua:

- train chay xong
- `best.pt` / `last.pt` duoc tao
- top1/top5/top10 van hoat dong
- terminal log dau run in:
  `Loss: cross_entropy`

### 10.3. StandardLS eps=0.1 smoke

Command:

```bash
python scripts/train_skeleton.py --config configs/train/skeleton_selected_31_stgcnpp_standardls_eps01.yaml --run-name smoke-standardls-eps01 --epochs 1 --limit-train 32 --limit-val 16 --limit-test 16 --batch-size 8 --num-workers 0 --no-wandb
```

Ket qua:

- train chay xong
- `best.pt` / `last.pt` duoc tao
- metrics top1/top5/top10 van co
- terminal log dau run in:
  `Loss: standard_label_smoothing epsilon=0.1`
- `summary.json` co:
  - `loss_type = standard_label_smoothing`
  - `label_smoothing_epsilon = 0.1`

### 10.4. StandardLS eps=0.3 smoke

Command:

```bash
python scripts/train_skeleton.py --config configs/train/skeleton_selected_31_stgcnpp_standardls_eps03.yaml --run-name smoke-standardls-eps03 --epochs 1 --limit-train 32 --limit-val 16 --limit-test 16 --batch-size 8 --num-workers 0 --no-wandb
```

Ket qua:

- train chay xong
- `best.pt` / `last.pt` duoc tao
- metrics top1/top5/top10 van co
- terminal log dau run in:
  `Loss: standard_label_smoothing epsilon=0.3`

## 11. Output examples

### 11.1. `summary.json`

Run:

- `outputs/skeleton/smoke-standardls-eps01/summary.json`

Fields moi:

```json
{
  "loss_type": "standard_label_smoothing",
  "label_smoothing_epsilon": 0.1
}
```

### 11.2. `metrics.json`

Run:

- `outputs/skeleton/smoke-standardls-eps01/metrics.json`

Fields moi:

```json
{
  "loss_type": "standard_label_smoothing",
  "label_smoothing_epsilon": 0.1
}
```

### 11.3. `config_resolved.yaml`

Run:

- `outputs/skeleton/smoke-standardls-eps01/config_resolved.yaml`

Se thay:

- `train.loss: standard_label_smoothing`
- `label_smoothing.epsilon: 0.1`
- `runtime.loss_type: standard_label_smoothing`
- `runtime.label_smoothing_epsilon: 0.1`

## 12. Nhung gi co y khong lam trong task nay

Co y khong lam:

- khong implement LanguageLS
- khong tao similarity matrix
- khong doi dataset
- khong doi skeleton preprocessing
- khong doi graph tensor
- khong sua pose extraction
- khong them dependency moi
- khong sua config CE cu
- khong tao them selected_27 StandardLS config de giu thay doi toi thieu

## 13. Buoc tiep theo de xuat

Neu muon so sanh nghiem tuc, bo full run nen la:

1. CE baseline
2. StandardLS `epsilon=0.1`
3. StandardLS `epsilon=0.3`
4. LanguageLS sau

Command full run de xuat:

```bash
python scripts/train_skeleton.py --config configs/train/skeleton_selected_31_stgcnpp.yaml --run-name sel31-stgcnpp-ce-baseline-lr005-bs16-wd001-ep180-v1
```

```bash
python scripts/train_skeleton.py --config configs/train/skeleton_selected_31_stgcnpp_standardls_eps01.yaml --run-name sel31-stgcnpp-standardls-eps01-lr005-bs16-wd001-ep180-v1
```

```bash
python scripts/train_skeleton.py --config configs/train/skeleton_selected_31_stgcnpp_standardls_eps03.yaml --run-name sel31-stgcnpp-standardls-eps03-lr005-bs16-wd001-ep180-v1
```
