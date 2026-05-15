# Skeleton Training Baseline

## Mục đích của phần triển khai này

Tài liệu này mô tả chi tiết phần training baseline cho skeleton branch mà tôi vừa triển khai trong repo này.

Phạm vi của phần triển khai:

- train baseline cho `selected_27`
- train baseline cho `selected_31`
- dùng trực tiếp graph tensors đã precompute
- loss hiện tại là `CrossEntropyLoss`
- có CLI override hyperparameters
- có save checkpoint local
- có evaluate từ checkpoint local
- có tích hợp W&B ở mức optional

Phần này **không** làm các việc sau:

- không rebuild graph tensors
- không chạy pose extraction
- không đọc raw videos / standardized frames
- không implement Language Label Smoothing
- không implement standard label smoothing
- không thay ST-GCN++ / CTR-GCN thật

Mục tiêu chính là làm cho pipeline training skeleton chạy được end-to-end với một baseline model đơn giản nhưng hợp lý.

---

## Tổng quan luồng chạy

Luồng training hiện tại như sau:

```text
config YAML
-> apply CLI overrides
-> resolve run_name / output_dir
-> build SkeletonGraphDataset for train/val/test
-> build SkeletonGraph adjacency
-> build SimpleSTGCN model
-> build optimizer / optional scheduler / CE loss
-> train loop
-> validate mỗi epoch
-> save best.pt / last.pt
-> load best.pt
-> evaluate on test
-> save metrics.json / train_log.csv / summary.json
-> optional W&B logging
```

Input mà training dùng là:

- manifest CSV của skeleton branch
- graph tensor `.npz`

Training **không phụ thuộc MMPose** và không đụng tới các stage preprocessing cũ.

---

## Những file đã thêm / sửa

## File mới

### `src/slr/branches/skeleton/models/__init__.py`

Vai trò:

- export `SimpleSTGCN`
- cung cấp factory `build_skeleton_model(cfg, graph)`

Ý nghĩa:

- tách model construction khỏi training loop
- sau này thay `simple_stgcn` bằng `stgcnpp` hoặc `ctrgcn` sẽ dễ hơn

---

### `src/slr/branches/skeleton/models/simple_stgcn.py`

Đây là baseline model chính.

Model nhận:

- input shape `(N, C, T, V, M)`
- ví dụ:
  - `selected_27`: `(N, 3, 150, 27, 1)`
  - `selected_31`: `(N, 3, 150, 31, 1)`

Output:

- logits shape `(N, num_classes)`
- hiện tại `num_classes = 100`

Kiến trúc ở mức cao:

1. nhận adjacency matrix từ `SkeletonGraph`
2. merge hoặc average person dimension `M`
3. input batch norm
4. qua vài block graph-temporal
5. global average pool theo thời gian và node
6. linear classifier

Các class chính:

- `GraphConv2d`
  - project feature qua `1x1 conv`
  - aggregate theo adjacency bằng `einsum`
- `STGCNBlock`
  - graph conv
  - batch norm
  - temporal conv
  - residual
  - dropout / relu
- `SimpleSTGCN`
  - ghép nhiều block lại thành backbone
  - cuối cùng ra logits

Lý do chọn kiến trúc này:

- đủ nhẹ để smoke test nhanh
- vẫn bám đúng ý tưởng spatial-temporal graph baseline
- không bị “MLP trá hình”
- giữ API gần với các model graph mạnh hơn sau này

---

### `src/slr/branches/skeleton/train.py`

Đây là file quan trọng nhất của phần triển khai.

File này chứa:

- parser cho training CLI
- parser cho evaluate CLI
- logic load/normalize config
- apply CLI overrides
- build dataset / dataloader
- build graph / model
- training loop
- validation loop
- checkpointing
- test evaluation cuối run
- save outputs

Các function chính:

#### `build_parser()`

CLI cho train:

- `--config`
- `--run-name`
- `--epochs`
- `--batch-size`
- `--lr`
- `--weight-decay`
- `--dropout`
- `--device`
- `--seed`
- `--no-wandb`
- `--wandb-project`
- `--wandb-entity`
- `--output-root`
- `--num-workers`
- `--limit-train`
- `--limit-val`
- `--limit-test`
- `--dry-run`

#### `build_evaluate_parser()`

CLI cho evaluate:

- `--config`
- `--checkpoint`
- `--split`
- `--batch-size`
- `--device`

#### `_normalize_training_config(...)`

Vai trò:

- load YAML xong thì điền default cho các section còn thiếu
- tránh việc training loop phải check quá nhiều field nullable

Các section chính được chuẩn hóa:

- `experiment`
- `dataset`
- `dataloader`
- `graph`
- `model`
- `train`
- `scheduler`
- `logging`
- `runtime`

#### `apply_cli_overrides(...)`

Vai trò:

- lấy config đã normalize
- override bằng CLI values nếu user truyền vào

Ví dụ:

- `--run-name` override `experiment.name` và `logging.run_name`
- `--lr` override `train.learning_rate`
- `--batch-size` override `dataloader.batch_size`

#### `resolve_training_config(...)`

Vai trò:

- load YAML
- normalize
- apply CLI override
- validate consistency giữa:
  - `dataset.expected_shape`
  - `model.in_channels`
  - `model.num_nodes`
  - `dataset.num_classes`
  - `model.num_classes`

Nó giúp fail sớm nếu config bị lệch.

#### `build_skeleton_datasets(...)`

Vai trò:

- tạo `SkeletonGraphDataset` cho `train`, `val`, `test`
- truyền `limit_train`, `limit_val`, `limit_test` nếu có

#### `build_skeleton_dataloaders(...)`

Vai trò:

- tạo `DataLoader` cho từng split
- dùng `skeleton_collate_fn`
- `pin_memory` chỉ bật thực sự khi device là CUDA

#### `build_graph_and_model(...)`

Vai trò:

- build `SkeletonGraph`
- build model qua `build_skeleton_model(...)`

#### `run_one_epoch_with_shape(...)`

Đây là core của train/eval.

Nó:

- validate shape batch
- move tensor lên device
- forward
- compute loss
- backward nếu đang train
- optimizer step
- optional grad clip
- optional amp
- compute top1 / top5
- accumulate average loss / accuracy

Output trả về:

- `loss`
- `top1`
- `top5`

#### `run_training(...)`

Đây là full training entrypoint.

Nó làm lần lượt:

1. resolve config
2. set seed
3. build datasets
4. build dataloaders
5. build graph và model
6. dry-run nếu được yêu cầu
7. build criterion / optimizer / scheduler
8. init optional W&B
9. chạy train loop theo epoch
10. validate mỗi epoch
11. theo dõi best metric
12. save `best.pt` và `last.pt`
13. load best checkpoint
14. evaluate trên test split
15. save file outputs

#### `run_evaluation(...)`

Vai trò:

- load `config_resolved.yaml`
- build dataset/dataloader đúng split
- build graph/model
- load checkpoint
- evaluate
- save JSON kết quả nếu `experiment.output_dir` có trong config

---

### `src/slr/training/seed.py`

Vai trò:

- set seed cho Python / NumPy / PyTorch
- optional deterministic mode

Ý nghĩa:

- training reproducible hơn
- tách logic seed khỏi file train chính

---

### `src/slr/training/wandb_utils.py`

Vai trò:

- gom toàn bộ W&B logic vào một chỗ

Các function chính:

- `resolve_wandb_entity(...)`
- `init_wandb_run(...)`
- `log_wandb_metrics(...)`
- `log_wandb_model_artifact(...)`
- `finish_wandb_run(...)`

Luồng resolve entity:

1. CLI `--wandb-entity`
2. `logging.entity`
3. env var theo `logging.entity_env`

Luồng bật/tắt W&B:

- nếu `logging.use_wandb=false` thì bỏ qua
- nếu `wandb` chưa cài thì warning rồi disable
- nếu thiếu `WANDB_API_KEY` thì warning rồi disable
- nếu không resolve được entity thì warning rồi disable

Điểm này giúp training local hoặc CI không bị chết cứng chỉ vì thiếu W&B.

---

### `scripts/evaluate_skeleton.py`

Vai trò:

- CLI mỏng gọi vào `slr.branches.skeleton.train.evaluate_main`

Lệnh dùng:

```bash
python scripts/evaluate_skeleton.py --config outputs/skeleton/smoke-sel31/config_resolved.yaml --checkpoint outputs/skeleton/smoke-sel31/checkpoints/best.pt --split test
```

---

### `configs/train/skeleton_selected_27_baseline.yaml`

Config baseline cho `selected_27`.

Field quan trọng:

- `dataset.expected_shape: [3, 150, 27, 1]`
- `graph.layout: selected_27`
- `model.num_nodes: 27`
- manifest trỏ đúng vào `nslt100_selected_27_*.csv`

---

### `configs/train/skeleton_selected_31_baseline.yaml`

Config baseline cho `selected_31`.

Field quan trọng:

- `dataset.expected_shape: [3, 150, 31, 1]`
- `graph.layout: selected_31`
- `model.num_nodes: 31`
- manifest trỏ đúng vào `nslt100_selected_31_*.csv`

---

### `docs/skeleton_training_baseline.md`

Đây là tài liệu hướng dẫn ngắn gọn cho user cuối:

- cách chạy
- cách setup W&B
- output folder
- cách zip/tải model từ Kaggle
- cách evaluate local

File `TRAINING.md` này thì chi tiết hơn và thiên về “giải thích implementation”.

---

## File cũ được sửa

### `scripts/train_skeleton.py`

Trước đây file này chỉ gọi vào training scaffold chung.

Hiện tại:

- nó gọi vào training entrypoint thật của skeleton baseline

---

### `src/slr/branches/skeleton/__init__.py`

Đã export thêm:

- `build_skeleton_model`

Mục đích:

- dễ import từ branch package

---

### `src/slr/training/metrics.py`

Đã mở rộng thành:

- `AverageMeter`
- `accuracy_topk(...)`
- `top_k_accuracy(...)` backward-compatible

#### `accuracy_topk(...)`

Input:

- `logits`: `(N, num_classes)`
- `targets`: `(N,)`

Output:

- dict dạng:

```python
{
  "top1": 0.75,
  "top5": 0.93,
}
```

Giá trị là fraction `0..1`, nhất quán với training loop và W&B logging.

---

### `src/slr/training/checkpointing.py`

Trước đây chỉ có helper build path.

Hiện tại có thêm:

- `save_checkpoint(...)`
- `load_checkpoint(...)`

Format checkpoint:

```python
{
  "epoch": int,
  "model_state_dict": ...,
  "optimizer_state_dict": ...,
  "scheduler_state_dict": ...,
  "best_metric": float | None,
  "last_metrics": dict,
  "config": resolved_config,
  "keypoint_set": str,
  "num_classes": int,
  "num_nodes": int,
  "model_name": str,
  "class_id_to_gloss": dict,
}
```

Điểm quan trọng:

- chỉ lưu `state_dict`
- không pickle toàn bộ model object
- dễ load lại trên local hoặc Kaggle

---

### `src/slr/training/optim.py`

Đã thêm:

- `build_optimizer(...)`
- `build_scheduler(...)`

Optimizer hiện hỗ trợ:

- `adamw`
- `adam`
- `sgd`

Scheduler hiện hỗ trợ:

- `cosine`
- `step`

Mặc định config baseline đang để scheduler disabled.

---

### `src/slr/training/losses.py`

Đã thêm:

- `build_loss(...)`

Hiện tại hỗ trợ:

- `cross_entropy`

Thiết kế dạng factory giúp sau này thêm label smoothing hoặc LanguageLS mà không phải viết lại training loop.

---

### `requirements.txt` và `requirements-rtmw.txt`

Đã thêm:

- `wandb`

Lưu ý:

- môi trường local hiện tại mà tôi dùng để test chưa có `wandb` cài sẵn
- nên W&B smoke chưa được chạy thật trong workspace này

---

## Dataset / Graph đang được tận dụng như thế nào

## `src/slr/branches/skeleton/dataset.py`

File này đã có sẵn từ trước, và là nền tảng của phần training mới.

Training baseline tận dụng các khả năng sau của dataset loader:

- load manifest theo split
- filter `status == "ok"`
- resolve `graph_tensor_path`
- remap path giữa local / HF / Kaggle
- check tensor shape
- build label maps
- collate batch theo `skeleton_collate_fn`

Training loop không tự đọc `.npz` thô; tất cả đi qua `SkeletonGraphDataset`.

---

## `src/slr/branches/skeleton/graph.py`

Training baseline tận dụng:

- `SkeletonGraph(layout="selected_27" | "selected_31", strategy="spatial")`
- adjacency shape:
  - `selected_27`: `(3, 27, 27)`
  - `selected_31`: `(3, 31, 31)`

Model `SimpleSTGCN` nhận adjacency này như graph prior.

---

## Cấu trúc config baseline

Mỗi config có các section:

### `experiment`

Ví dụ:

```yaml
experiment:
  name: sel27-test1
  seed: 42
  output_root: outputs/skeleton
  monitor_metric: val/top1
  monitor_mode: max
  save_every_epoch: false
```

Ý nghĩa:

- `name`: tên run
- `output_root`: root output
- `monitor_metric`: metric để chọn best checkpoint
- `monitor_mode`: `max` hoặc `min`
- `save_every_epoch`: có lưu thêm `epoch_XXX.pt` hay không

---

### `dataset`

Ví dụ:

```yaml
dataset:
  keypoint_set: selected_27
  num_classes: 100
  expected_shape: [3, 150, 27, 1]
```

Ý nghĩa:

- xác định manifest nào được load
- xác định shape kỳ vọng để fail sớm nếu data sai

---

### `dataloader`

Ví dụ:

```yaml
dataloader:
  batch_size: 16
  num_workers: 2
  pin_memory: true
  shuffle_train: true
```

---

### `graph`

Ví dụ:

```yaml
graph:
  layout: selected_27
  strategy: spatial
  add_self_links: true
  normalize_adjacency: true
```

---

### `model`

Ví dụ:

```yaml
model:
  name: simple_stgcn
  in_channels: 3
  num_nodes: 27
  num_classes: 100
  hidden_channels: 64
  dropout: 0.5
```

---

### `train`

Ví dụ:

```yaml
train:
  epochs: 30
  device: auto
  optimizer: adamw
  learning_rate: 0.001
  weight_decay: 0.0005
  loss: cross_entropy
  grad_clip_norm: null
  amp: false
```

---

### `scheduler`

Ví dụ:

```yaml
scheduler:
  enabled: false
  name: cosine
  min_lr: 0.000001
```

---

### `logging`

Ví dụ:

```yaml
logging:
  use_wandb: true
  entity_env: WANDB_ENTITY
  project: wlasl-skeleton
  run_name: sel27-test1
  tags:
    - nslt100
    - selected_27
    - baseline
    - cross_entropy
  log_model: true
```

---

## CLI override hoạt động như thế nào

Các override hiện có:

- `--run-name`
- `--epochs`
- `--batch-size`
- `--lr`
- `--weight-decay`
- `--dropout`
- `--device`
- `--seed`
- `--no-wandb`
- `--wandb-project`
- `--wandb-entity`
- `--output-root`
- `--num-workers`
- `--limit-train`
- `--limit-val`
- `--limit-test`
- `--dry-run`

Ví dụ:

```bash
python scripts/train_skeleton.py \
  --config configs/train/skeleton_selected_31_baseline.yaml \
  --run-name sel31-lr5e4-bs32 \
  --epochs 5 \
  --lr 0.0005 \
  --batch-size 32 \
  --weight-decay 0.0001
```

Hiệu ứng:

- output sẽ vào `outputs/skeleton/sel31-lr5e4-bs32/`
- `config_resolved.yaml` sẽ phản ánh các override này

---

## Output của mỗi run

Mỗi run tạo:

```text
outputs/skeleton/<run_name>/
├── checkpoints/
│   ├── best.pt
│   └── last.pt
├── config_resolved.yaml
├── metrics.json
├── train_log.csv
└── summary.json
```

## `config_resolved.yaml`

Chứa:

- config cuối cùng sau khi merge default + YAML + CLI override

Mục đích:

- reproducibility
- local evaluate về sau

---

## `metrics.json`

Chứa summary metric cuối cùng:

- `best_epoch`
- `best_val_top1`
- `best_val_top5`
- `best_val_loss`
- `test_loss`
- `test_top1`
- `test_top5`
- `final_train_loss`
- `final_val_loss`

---

## `train_log.csv`

Một dòng mỗi epoch:

- `epoch`
- `lr`
- `train_loss`
- `train_top1`
- `train_top5`
- `val_loss`
- `val_top1`
- `val_top5`

---

## `summary.json`

Chứa metadata run:

- `run_name`
- `keypoint_set`
- `model_name`
- `output_dir`
- `best_checkpoint`
- `last_checkpoint`
- `config_path`
- dataset info
- optional `wandb_run_url`

---

## Checkpoint hoạt động như thế nào

Training loop hiện tại:

- luôn save `last.pt` mỗi epoch
- save `best.pt` khi metric monitor cải thiện
- optional save thêm `epoch_XXX.pt` nếu `save_every_epoch=true`

Default monitor:

- `val/top1`
- mode `max`

Cuối training:

- load lại `best.pt`
- evaluate trên test split

Điều này tránh việc report test metrics từ model ở epoch cuối nếu epoch tốt nhất là trước đó.

---

## W&B hoạt động như thế nào

Nếu `logging.use_wandb=true`, code sẽ thử:

1. import `wandb`
2. check `WANDB_API_KEY`
3. resolve entity
4. gọi `wandb.init(...)`
5. log metrics mỗi epoch
6. optional upload `best.pt` như artifact

Nếu một trong các điều kiện trên không đạt:

- code sẽ warning rõ ràng
- tự disable W&B
- training vẫn tiếp tục

Đây là quyết định có chủ đích để tránh làm run local chết vô ích.

---

## Evaluate local hoạt động như thế nào

Script:

- `scripts/evaluate_skeleton.py`

Nó nhận:

- `config_resolved.yaml`
- `best.pt` hoặc checkpoint bất kỳ
- split muốn evaluate

Ví dụ:

```bash
python scripts/evaluate_skeleton.py \
  --config outputs/skeleton/smoke-sel31/config_resolved.yaml \
  --checkpoint outputs/skeleton/smoke-sel31/checkpoints/best.pt \
  --split test
```

Output:

- in metrics ra stdout
- save `eval_<split>_<checkpoint_stem>.json`

Ví dụ thực tế:

- `eval_test_best.json`

---

## Những command tôi đã dùng để smoke test

Lưu ý quan trọng:

- `python` mặc định của máy hiện tại không có `torch`
- nên tôi dùng:

```powershell
.\.venv-rtmw310\Scripts\python.exe
```

Các lệnh đã chạy:

```powershell
.\.venv-rtmw310\Scripts\python.exe -m compileall src\slr\branches\skeleton src\slr\training scripts\train_skeleton.py scripts\evaluate_skeleton.py
.\.venv-rtmw310\Scripts\python.exe -c "from slr.branches.skeleton.models import build_skeleton_model; from slr.training.metrics import accuracy_topk; print('OK')"
.\.venv-rtmw310\Scripts\python.exe scripts\train_skeleton.py --config configs\train\skeleton_selected_27_baseline.yaml --run-name dry-sel27 --dry-run --no-wandb
.\.venv-rtmw310\Scripts\python.exe scripts\train_skeleton.py --config configs\train\skeleton_selected_31_baseline.yaml --run-name dry-sel31 --dry-run --no-wandb
.\.venv-rtmw310\Scripts\python.exe scripts\train_skeleton.py --config configs\train\skeleton_selected_27_baseline.yaml --run-name smoke-sel27 --epochs 1 --limit-train 32 --limit-val 16 --limit-test 16 --num-workers 0 --no-wandb
.\.venv-rtmw310\Scripts\python.exe scripts\train_skeleton.py --config configs\train\skeleton_selected_31_baseline.yaml --run-name smoke-sel31 --epochs 1 --limit-train 32 --limit-val 16 --limit-test 16 --num-workers 0 --no-wandb
.\.venv-rtmw310\Scripts\python.exe scripts\evaluate_skeleton.py --config outputs\skeleton\smoke-sel31\config_resolved.yaml --checkpoint outputs\skeleton\smoke-sel31\checkpoints\best.pt --split test --batch-size 16
```

---

## Kết quả smoke test đã có

## `smoke-sel27`

Output:

- `outputs/skeleton/smoke-sel27/`

Checkpoint:

- `best.pt`
- `last.pt`

Metric:

- train loss: `4.8618`
- train top1: `0.0000`
- train top5: `0.03125`
- val loss: `4.5984`
- val top1: `0.0000`
- val top5: `0.1875`
- test loss: `4.5581`
- test top1: `0.1250`
- test top5: `0.3125`

---

## `smoke-sel31`

Output:

- `outputs/skeleton/smoke-sel31/`

Checkpoint:

- `best.pt`
- `last.pt`

Metric:

- train loss: `4.9240`
- train top1: `0.0000`
- train top5: `0.0000`
- val loss: `4.6092`
- val top1: `0.0000`
- val top5: `0.2500`
- test loss: `4.5606`
- test top1: `0.0625`
- test top5: `0.3125`

Evaluate script:

- đã chạy thành công trên `smoke-sel31`

---

## Cảnh báo / giới hạn hiện tại

### 1. Baseline model còn đơn giản

`SimpleSTGCN` hiện chỉ là baseline functional.

Nó phù hợp để:

- verify training pipeline
- verify checkpoint / W&B / evaluate path
- smoke test selected_27 / selected_31

Nó chưa nhằm mục tiêu SOTA.

---

### 2. W&B chưa được verify live trong workspace này

Lý do:

- env test hiện tại chưa có `wandb` install sẵn
- tôi không cài package mới trong turn vừa rồi

Tôi đã:

- thêm `wandb` vào requirements
- thêm code integration

Nhưng chưa có run URL thật để ghi lại trong báo cáo.

---

### 3. Python mặc định trên máy hiện tại chưa có torch

`python` hiện tại của hệ thống không import được `torch`.

Vì vậy local run trên máy này nên dùng:

```powershell
.\.venv-rtmw310\Scripts\python.exe
```

hoặc cài `torch` vào env đang dùng.

---

### 4. Generic training scaffold cũ vẫn còn trong repo

Các file generic cũ trong `src/slr/training/train.py` và `evaluate.py` chưa được thay bằng pipeline thật.

Hiện tại đường chạy đúng là:

- `scripts/train_skeleton.py`
- `scripts/evaluate_skeleton.py`

---

## Cách chạy nhanh trong thực tế

## Dry run

```bash
python scripts/train_skeleton.py --config configs/train/skeleton_selected_27_baseline.yaml --run-name dry-sel27 --dry-run --no-wandb
```

Dry-run sẽ:

- load config
- build dataset
- build graph
- build model
- chạy thử một batch forward
- không train

---

## Smoke run

```bash
python scripts/train_skeleton.py --config configs/train/skeleton_selected_31_baseline.yaml --run-name smoke-sel31 --epochs 1 --limit-train 32 --limit-val 16 --limit-test 16 --no-wandb
```

---

## Full baseline run

```bash
python scripts/train_skeleton.py --config configs/train/skeleton_selected_31_baseline.yaml --run-name sel31-full-001 --epochs 30 --lr 0.001 --batch-size 16
```

---

## Evaluate best checkpoint

```bash
python scripts/evaluate_skeleton.py --config outputs/skeleton/sel31-full-001/config_resolved.yaml --checkpoint outputs/skeleton/sel31-full-001/checkpoints/best.pt --split test
```

---

## Cách tải model từ Kaggle về local

Trong Kaggle:

```bash
cd /kaggle/working/Recognizing-sign-language-at-the-word-level
zip -r /kaggle/working/sel31-full-001_outputs.zip outputs/skeleton/sel31-full-001
```

Sau đó:

- tải file zip từ Output tab
- giải nén vào local repo
- dùng `scripts/evaluate_skeleton.py` để evaluate lại local

---

## Gợi ý bước tiếp theo

Sau baseline này, roadmap hợp lý là:

1. thay `SimpleSTGCN` bằng ST-GCN++ wrapper
2. thêm CTR-GCN option
3. thêm standard label smoothing qua `build_loss(...)`
4. thêm LanguageLS ở loss layer
5. thêm confusion matrix / per-class metrics
6. thêm export prediction CSV cho analysis

---

## Kết luận ngắn

Phần training baseline skeleton hiện tại đã có:

- config-driven training
- CLI override
- selected_27 / selected_31 support
- CrossEntropyLoss
- local checkpointing
- local evaluation
- optional W&B integration
- tài liệu hướng dẫn chạy

Nói ngắn gọn: preprocessing skeleton đã có từ trước, và phần tôi vừa thêm vào là “nửa sau” của pipeline để bạn có thể train thử baseline end-to-end trên train-ready graph tensors.
---

## ST-GCN++ integration update

- Added a repo-local clean-room `stgcnpp` model for the skeleton branch.
- The current skeleton train/eval pipeline is unchanged:
  - `scripts/train_skeleton.py`
  - `scripts/evaluate_skeleton.py`
  - manifest-driven graph tensor loading
  - local checkpoints
  - optional W&B logging
- New configs:
  - `configs/train/skeleton_selected_27_stgcnpp.yaml`
  - `configs/train/skeleton_selected_31_stgcnpp.yaml`
- `SimpleSTGCN` remains the technical baseline used to verify the pipeline.
- LanguageLS and CTR-GCN are still deferred.

See also:

- `docs/skeleton_stgcnpp_integration.md`
