# LATEST INFO

## 1. Mục tiêu dự án

Repository này phục vụ bài toán **Word-Level Sign Language Recognition (WSLR)** trên bộ dữ liệu **WLASL**, với tư duy pipeline nhiều tầng:

```text
raw metadata + raw videos
-> index manifests
-> standardized frames/videos
-> shared RTMW-l pose
-> branch-specific inputs
-> training / evaluation
```

Trạng thái thực tế hiện nay:

- Nhánh **skeleton** là nhánh chính và đã chạy được end-to-end.
- Nhánh **regions** và **hand_poseflow** mới có scaffold CLI, dataset/model placeholder, chưa có implementation thật.
- Pipeline train/eval thực tế đang đi qua `src/slr/branches/skeleton/train.py`, không phải scaffold generic ở `src/slr/training/train.py`.

## 2. Ảnh chụp trạng thái workspace hiện tại

### 2.1 Những gì đã có thật trong workspace

- Dữ liệu gốc WLASL nằm dưới `data/datasets/WLASL/raw/`.
- Tầng `index/`, `standardized/`, `pose/rtmw_l/`, `branch_inputs/skeleton/rtmw_l/` đều đã tồn tại.
- Bộ dữ liệu skeleton train-ready cho `nslt100` đã tồn tại với:
  - `selected_27`: `1013` graph tensor `.npz`
  - `selected_31`: `1013` graph tensor `.npz`
- Manifest split hiện tại cho `nslt100`:
  - train: `748`
  - val: `165`
  - test: `100`
- Bundle phụ trợ đã có:
  - `hf_bundle/`
  - `kaggle_bundle/`
  - `kaggle_sub300_bundle/`
  - `kaggle_sub300_bundle.zip`

### 2.2 Các run huấn luyện hiện có trong `outputs/skeleton/`

Các run smoke test hiện thấy:

- `smoke-sel27`
- `smoke-sel31`
- `smoke-sel27-stgcnpp`
- `smoke-sel31-stgcnpp`
- `smoke-ce-after-standardls`
- `smoke-standardls-eps01`
- `smoke-standardls-eps03`
- `smoke-sel27-standardls-eps005`
- `smoke-sel27-standardls-eps01`
- `smoke-sel27-standardls-eps03`
- `smoke-sel31-standardls-eps005`

Một vài metric đã lưu:

| Run | test_top1 | test_top5 | test_loss | loss |
| --- | ---: | ---: | ---: | --- |
| `smoke-sel27` | 0.1250 | 0.3125 | 4.5581 | baseline CE |
| `smoke-sel31` | 0.0625 | 0.3125 | 4.5606 | baseline CE |
| `smoke-sel27-stgcnpp` | 0.0625 | 0.1875 | 4.4996 | ST-GCN++ |
| `smoke-sel31-stgcnpp` | 0.0625 | 0.2500 | 4.4973 | ST-GCN++ |
| `smoke-ce-after-standardls` | 0.0625 | 0.2500 | 4.4973 | CE |
| `smoke-standardls-eps01` | 0.0625 | 0.2500 | 4.5403 | StandardLS |
| `smoke-standardls-eps03` | 0.0625 | 0.2500 | 4.5686 | StandardLS |

## 3. Cách dự án hoạt động

### 3.1 Luồng preprocessing

1. `scripts/00_build_index.py`
   - Đọc `WLASL_v0.3.json`, `wlasl_class_list.txt`, `nslt_*.json`, video local.
   - Sinh manifest sạch cho từng subset và split.

2. `scripts/01_standardize_videos.py`
   - Đọc manifest ở `index/subsets_available/<subset>/`.
   - Crop theo bbox, resize + letterbox về `288x384`.
   - Ghi standardized frames/video và manifest.

3. `scripts/02_extract_pose_rtmw.py`
   - Đọc standardized frames.
   - Chạy RTMW-l qua MMPose.
   - Sinh pose `.npz` định dạng `(T, 133, 3)` và report chất lượng.

4. `scripts/03_build_skeleton_inputs.py`
   - Đọc pose 133 keypoint.
   - Rút gọn thành `selected_27` hoặc `selected_31`.
   - Chuẩn hóa tọa độ/confidence.
   - Cố định chiều dài chuỗi về `150`.
   - Đổi sang graph tensor `C x T x V x M`.

### 3.2 Luồng train/eval hiện dùng

1. `scripts/train_skeleton.py`
2. `src/slr/branches/skeleton/train.py`
3. `SkeletonGraphDataset` đọc manifest và `.npz`
4. `SkeletonGraph` tạo adjacency
5. `build_skeleton_model(...)` tạo `SimpleSTGCN` hoặc `STGCNPP`
6. train loop chạy theo config
7. lưu `best.pt`, `last.pt`, `metrics.json`, `summary.json`, `train_log.csv`
8. sau train, nạp lại `best.pt` rồi đánh giá trên test split

### 3.3 Những phần chưa hoàn thiện

- `regions`:
  - crop thật chưa implement
  - dataset/model chỉ placeholder
- `hand_poseflow`:
  - build hand sequences và poseflow mới tạo folder scaffold
  - dataset/model chỉ placeholder
- `src/slr/training/train.py`, `src/slr/training/evaluate.py`:
  - chỉ là unified scaffold, không phải đường chạy thật hiện tại
- `src/slr/inference/*`:
  - inference/visualization cho model chưa implement thật

## 4. Cấu trúc thư mục

### 4.1 Top-level

```text
.
├── configs/                      # YAML cho preprocessing, branch, train, debug
├── data/                         # toàn bộ dữ liệu WLASL nhiều tầng
├── docs/                         # tài liệu kỹ thuật ngắn
├── outputs/                      # output các run huấn luyện
├── reports/                      # báo cáo preprocessing/experiment
├── scripts/                      # CLI wrappers và utilities
├── src/slr/                      # code nguồn chính
├── checkpoints/                  # checkpoint RTMW-l và model
├── hf_bundle/                    # bundle skeleton để upload HF
├── kaggle_bundle/                # bundle Kaggle tiêu chuẩn
├── kaggle_sub300_bundle/         # bundle Kaggle cho nslt300
├── notebooks/                    # notebook phân tích
├── README.md
├── PROJECT_STRUCTURE_GUIDE.md
├── TRAINING.md
├── IMPLEMENTATION.md
├── StandardLS.md
└── LATEST_INFO.md
```

### 4.2 Cấu trúc dữ liệu thực tế

```text
data/datasets/WLASL/
├── raw/
├── index/
├── standardized/
├── pose/rtmw_l/
└── branch_inputs/skeleton/rtmw_l/
```

Ý nghĩa:

- `raw/`: dữ liệu gốc, không chỉnh sửa tại chỗ.
- `index/`: manifest và audit tầng metadata.
- `standardized/`: standardized frames/video.
- `pose/rtmw_l/`: shared wholebody pose.
- `branch_inputs/skeleton/rtmw_l/`: selected/normalized/graph tensors phục vụ train.

## 5. Phân tích chi tiết file theo nhóm

## 5.1 File gốc ở root

- `README.md`
  - Mô tả mục tiêu repo, dataset, pipeline chuẩn, schema các tầng dữ liệu.
  - Nêu rõ skeleton là nhánh ưu tiên.

- `PROJECT_STRUCTURE_GUIDE.md`
  - Tài liệu kiến trúc tổng quát của repo.
  - Có giá trị định hướng, nhưng một số chỗ mô tả repo như scaffold ban đầu; trạng thái code hiện tại đã tiến xa hơn, nhất là nhánh skeleton.

- `TRAINING.md`
  - Tài liệu rất chi tiết về training skeleton baseline và cập nhật ST-GCN++.
  - Gần với “implementation log” hơn là README cho end-user.

- `IMPLEMENTATION.md`
  - Nhật ký tích hợp ST-GCN++ repo-local.
  - Ghi rõ validation commands, smoke results, hạn chế hiện tại.

- `StandardLS.md`
  - Nhật ký chi tiết việc thêm Standard Label Smoothing vào training skeleton.

- `pyproject.toml`
  - Khai báo package `slr` theo `src` layout.
  - Dependencies để trống; repo dùng `requirements*.txt` riêng cho môi trường thực tế.

- `requirements.txt`
  - Bộ dependency nhẹ cho preprocessing/utilities: `numpy`, `pandas`, `opencv-python`, `pyyaml`, `tqdm`, `scikit-learn`, `matplotlib`, `wandb`.

- `requirements-rtmw.txt`
  - Môi trường nặng cho RTMW-l/MMPose: `torch`, `torchvision`, `mmengine`, `mmcv`, `mmdet`, `mmpose`, `xtcocotools`, `rich`, `wandb`.

- `requirements-kaggle-train.txt`
  - Gói tối thiểu cho train trên Kaggle bundle: `wandb`, `huggingface_hub`, `pyyaml`.

- `sitecustomize.py`
  - Tự thêm `src/` vào `sys.path` khi chạy ad-hoc từ root.

- `slr/__init__.py`
  - Import shim để `import slr` hoạt động mà không cần editable install.

- `src/slr/__init__.py`
  - Package gốc, chỉ khai báo version.

## 5.2 `configs/`

### 5.2.1 `configs/dataset/`

- `configs/dataset/wlasl.yaml`
  - Khai báo root dataset, `raw_metadata_path`, `raw_video_dir`, subset và split mặc định.

### 5.2.2 `configs/preprocessing/`

- `configs/preprocessing/index.yaml`
  - Config cho stage build index.
  - Chỉ ra metadata master, class list, mapping `nslt100/nslt300/nslt1000/nslt2000`, root output, rules như `video_id_width`.

- `configs/preprocessing/standardize.yaml`
  - Config chuẩn hóa video cho subset mặc định `nslt100`.
  - Resize về `288x384`, crop theo bbox, margin quanh bbox, save frames, không save video mặc định.

- `configs/preprocessing/standardize_nslt300.yaml`
  - Biến thể standardize cho subset `nslt300`.

- `configs/preprocessing/standardize_nslt1000.yaml`
  - Biến thể standardize cho subset `nslt1000`.

- `configs/preprocessing/pose_rtmw_l.yaml`
  - Config extract pose RTMW-l local.
  - Dùng standardized manifests, lưu wholebody_133, batch size 1, ưu tiên `cuda:0`, fallback CPU.

- `configs/preprocessing/pose_rtmw_l_kaggle.yaml`
  - Biến thể cho Kaggle. Nội dung hiện tại gần như cùng cấu trúc với local config, phục vụ bundle Kaggle.

- `configs/preprocessing/region_crops.yaml`
  - Config dành cho region branch; hiện branch này chưa có implementation thật.

- `configs/preprocessing/poseflow.yaml`
  - Config dành cho hand_poseflow branch; hiện mới có scaffold.

### 5.2.3 `configs/branches/`

- `configs/branches/skeleton/stgcnpp_27.yaml`
  - Config preprocessing/build inputs cho skeleton `selected_27`.
  - Chỉ ra đường dẫn pose manifest, output selected/normalized/graph tensor, config confidence normalization, target length `150`.

- `configs/branches/skeleton/stgcnpp_31.yaml`
  - Tương tự trên nhưng cho `selected_31`.
  - Có thêm mouth landmarks trong mapping thực tế.

- `configs/branches/skeleton/ctrgcn_31.yaml`
  - Ý định chạy CTR-GCN cho `selected_31`.
  - Repo hiện chưa có implementation CTR-GCN thật ở nhánh chạy.

- `configs/branches/skeleton/stgcnpp_31_languagels.yaml`
  - Cấu hình hướng tới Language Label Smoothing.
  - Code loss hiện chưa có implementation LanguageLS trong train loop hiện hành.

- `configs/branches/regions/face_hands_baseline.yaml`
  - Baseline config cho region branch, chủ yếu là scaffold.

- `configs/branches/hand_poseflow/hand_poseflow_baseline.yaml`
  - Baseline config cho hand poseflow branch, chủ yếu là scaffold.

### 5.2.4 `configs/train/`

Đây là nhóm config quan trọng nhất cho training hiện hành.

- `skeleton_selected_27.yaml`
  - Dataset/train config cho `selected_27`.

- `skeleton_selected_27_baseline.yaml`
  - Baseline `SimpleSTGCN` cho `selected_27`.

- `skeleton_selected_27_stgcnpp.yaml`
  - ST-GCN++ cross entropy cho `selected_27`.

- `skeleton_selected_27_stgcnpp_standardls_eps005.yaml`
  - ST-GCN++ + StandardLS epsilon `0.05`.

- `skeleton_selected_27_stgcnpp_standardls_eps01.yaml`
  - ST-GCN++ + StandardLS epsilon `0.1`.

- `skeleton_selected_27_stgcnpp_standardls_eps03.yaml`
  - ST-GCN++ + StandardLS epsilon `0.3`.

- `skeleton_selected_31.yaml`
  - Dataset/train config cho `selected_31`.

- `skeleton_selected_31_baseline.yaml`
  - Baseline `SimpleSTGCN` cho `selected_31`.

- `skeleton_selected_31_stgcnpp.yaml`
  - ST-GCN++ cross entropy cho `selected_31`.

- `skeleton_selected_31_stgcnpp_standardls_eps005.yaml`
  - ST-GCN++ + StandardLS epsilon `0.05`.

- `skeleton_selected_31_stgcnpp_standardls_eps01.yaml`
  - ST-GCN++ + StandardLS epsilon `0.1`.

- `skeleton_selected_31_stgcnpp_standardls_eps03.yaml`
  - ST-GCN++ + StandardLS epsilon `0.3`.

### 5.2.5 `configs/experiments/`

- `configs/experiments/skeleton_nslt100_debug.yaml`
  - File debug-level mô tả một experiment cụ thể, ghép branch config + dataset config.
  - Có ích cho tư duy experiment tracking, nhưng training production path hiện dùng trực tiếp các config trong `configs/train/`.

## 5.3 `src/slr/data/`

- `src/slr/data/manifests.py`
  - Khai báo schema cột chuẩn cho:
    - `master_instances`
    - `subset manifests`
    - `standardized manifests`
    - `pose manifests`
    - `skeleton input manifests`
  - Đây là file trung tâm giúp các stage ghi CSV nhất quán.

- `src/slr/data/validation.py`
  - Helper validate manifest:
    - kiểm tra cột bắt buộc
    - kiểm tra null ở key columns
    - ép thứ tự cột
    - kiểm tra split hợp lệ

- `src/slr/data/build_index.py`
  - File lớn nhất của tầng data.
  - Chức năng:
    - load config index
    - parse `WLASL_v0.3.json`
    - parse class list
    - parse `missing.txt`
    - parse tất cả `nslt_*.json`
    - scan video local
    - build master instances
    - build bảng class map
    - build `video_to_split`
    - build manifest cho từng subset
    - build reports coverage / invalid IDs
    - ghi output CSV/JSON/MD
  - Đây là stage đã implement thật, không còn là scaffold.

- `src/slr/data/standardize_videos.py`
  - Stage chuẩn hóa video thực tế.
  - Logic chính:
    - đọc manifest từ index layer
    - resolve frame range an toàn
    - resolve bbox và fallback full-frame
    - đọc frame từ video raw
    - crop + resize letterbox
    - ghi frames và/hoặc video
    - ghi manifest kết quả cùng report markdown
  - Có xử lý lỗi khá đầy đủ: missing video, invalid bbox, read/write error, empty video.

## 5.4 `src/slr/pose/`

- `src/slr/pose/pose_schema.py`
  - Định nghĩa layout RTMW-l wholebody 133.
  - Định nghĩa các keypoint sets:
    - `selected_27`
    - `selected_31`
    - `selected_49` chưa implement
  - Gồm mapping region indices, mouth landmarks, validation shape/index.

- `src/slr/pose/keypoint_selection.py`
  - Hàm slice pose từ 133 điểm xuống reduced keypoint set.
  - Tạo payload `.npz` ổn định cho selected keypoints.

- `src/slr/pose/pose_normalization.py`
  - Chuẩn hóa `x, y` về `[-1, 1]`.
  - Fit confidence scale theo percentile.
  - Normalize confidence và sanitize giá trị non-finite.

- `src/slr/pose/pose_quality.py`
  - Tính mean confidence chung/theo vùng.
  - Tính valid frame ratio.
  - Tóm tắt manifest pose cho report.

- `src/slr/pose/extract_rtmw.py`
  - Stage extract pose thực tế.
  - Chức năng:
    - load config
    - tìm config/checkpoint RTMW-l
    - khởi tạo `MMPoseInferencer`
    - đọc standardized manifest
    - duyệt từng frames dir
    - infer pose theo frame
    - chọn signer chính khi có nhiều người
    - ghi pose `.npz`
    - tính quality metrics
    - ghi pose manifests + report markdown
  - Có cơ chế fallback device và tái sử dụng pose cũ nếu `overwrite=false`.

## 5.5 `src/slr/branches/skeleton/`

- `src/slr/branches/skeleton/transforms.py`
  - Cố định độ dài chuỗi về `150`.
  - Chiến lược hiện hỗ trợ:
    - ngắn hơn: `repeat`
    - dài hơn: `head`
  - Chuyển `(T, V, 3)` thành `CTVM`.

- `src/slr/branches/skeleton/label_smoothing.py`
  - Cài đặt numpy-side cho:
    - standard label smoothing
    - language label smoothing
  - Lưu ý: train loop hiện dùng `torch.nn.CrossEntropyLoss(label_smoothing=...)` cho StandardLS; LanguageLS chưa được nối vào production path.

- `src/slr/branches/skeleton/graph.py`
  - Định nghĩa topology cho `selected_27` và `selected_31`.
  - Sinh adjacency tensor theo:
    - `uniform`
    - `spatial`
  - `SkeletonGraph` là object trung tâm để model lấy `A`.

- `src/slr/branches/skeleton/dataset.py`
  - Dataset loader production cho graph tensor `.npz`.
  - Điểm quan trọng:
    - lọc manifest `status == ok`
    - resolve path linh hoạt giữa local/HF/Kaggle
    - validate tensor shape
    - build label maps
    - trả tensor + metadata
    - `skeleton_collate_fn` trả về `data`, `labels`, `metadata`
  - Đây là file quan trọng nhất để repo mang tính “manifest-driven”.

- `src/slr/branches/skeleton/build_inputs.py`
  - Stage build input cho skeleton thực tế.
  - Chức năng:
    - load pose manifests
    - resolve pose file
    - fit confidence scale trên split train
    - select keypoints
    - normalize xy/confidence
    - sanitize non-finite values
    - save `selected`, `normalized`, `graph_tensor`
    - build per-split manifest + report
  - Đây là cầu nối từ shared pose sang train-ready tensors.

- `src/slr/branches/skeleton/models/__init__.py`
  - Factory `build_skeleton_model`.
  - Hỗ trợ:
    - `simple_stgcn`
    - `stgcnpp`

- `src/slr/branches/skeleton/models/simple_stgcn.py`
  - Baseline graph model gọn nhẹ bằng PyTorch.
  - Thành phần:
    - `GraphConv2d`
    - `STGCNBlock`
    - `SimpleSTGCN`
  - Dùng để xác thực pipeline end-to-end.

- `src/slr/branches/skeleton/models/stgcnpp.py`
  - Implementation ST-GCN++ repo-local, clean-room.
  - Thành phần:
    - `SpatialGraphConv`
    - `MultiScaleTemporalConv`
    - `STGCNPPBlock`
    - `STGCNPP`
  - Không phụ thuộc `mmaction2` hay `pyskl`.
  - Đây là backbone mạnh hơn baseline đang dùng được thật.

- `src/slr/branches/skeleton/train.py`
  - File production path quan trọng nhất hiện nay.
  - Chứa:
    - parser train/evaluate
    - normalize config
    - CLI override
    - build datasets/dataloaders
    - build graph/model
    - train loop
    - validate mỗi epoch
    - checkpointing
    - final test evaluation
    - output JSON/YAML/CSV
    - W&B integration
  - Hỗ trợ loss:
    - `cross_entropy`
    - `standard_label_smoothing`

## 5.6 `src/slr/training/`

- `src/slr/training/metrics.py`
  - `AverageMeter`, `accuracy_topk`, `top_k_accuracy`.
  - Hoạt động cả với numpy lẫn torch tensor.

- `src/slr/training/losses.py`
  - Factory loss hiện hành.
  - Hỗ trợ:
    - `cross_entropy`
    - `standard_label_smoothing`
  - Chưa nối LanguageLS vào train loop.

- `src/slr/training/optim.py`
  - Factory optimizer/scheduler.
  - Optimizer:
    - `adamw`
    - `adam`
    - `sgd`
  - Scheduler:
    - `cosine`
    - `step`

- `src/slr/training/checkpointing.py`
  - Ghi checkpoint bằng `state_dict` thuần.
  - Lưu cả metadata như `model_name`, `num_nodes`, `class_id_to_gloss`, `config`.

- `src/slr/training/seed.py`
  - Set seed cho Python, NumPy, PyTorch; hỗ trợ deterministic mode.

- `src/slr/training/wandb_utils.py`
  - Tích hợp W&B an toàn:
    - resolve entity từ CLI/config/env
    - disable mềm nếu thiếu `wandb` hoặc `WANDB_API_KEY`
    - log metrics và upload model artifact

- `src/slr/training/train.py`
  - Generic scaffold cho unified training CLI.
  - Hiện chỉ log plan, chưa có training thật.

- `src/slr/training/evaluate.py`
  - Generic scaffold cho unified evaluation CLI.
  - Hiện chỉ log plan, chưa có evaluation thật.

## 5.7 `src/slr/utils/`

- `src/slr/utils/io.py`
  - Helper đọc/ghi JSON, YAML, CSV, text; đảm bảo tạo thư mục cha.

- `src/slr/utils/logging.py`
  - Tạo logger stream/file nhất quán.

- `src/slr/utils/video.py`
  - `probe_video`, `read_frames`, `write_video_from_frames`.

- `src/slr/utils/image.py`
  - Đọc ảnh, ép kiểu `uint8`, resize letterbox, save image.

- `src/slr/utils/bbox.py`
  - Parse/expand/clip/serialize bbox.

- `src/slr/utils/seed.py`
  - Seed Python + NumPy mức đơn giản.

## 5.8 `src/slr/branches/regions/`

Trạng thái: **placeholder/scaffold**.

- `build_crops.py`
  - Tạo parser, log input/output, tạo folder rỗng, chưa crop thật.

- `dataset.py`
  - `RegionSequenceDataset` placeholder, `__len__=0`.

- `region_schema.py`
  - Khai báo `face`, `left_hand`, `right_hand`.

- `transforms.py`
  - Có duy nhất normalize ảnh `uint8 -> [0,1]`.

## 5.9 `src/slr/branches/hand_poseflow/`

Trạng thái: **placeholder/scaffold**.

- `build_hand_sequences.py`
  - Chỉ tạo folder scaffold cho hand sequences.

- `build_poseflow.py`
  - Chỉ tạo folder scaffold cho poseflow variants.

- `build_inputs.py`
  - Orchestrator gọi hai bước trên.

- `dataset.py`
  - `HandPoseFlowDataset` placeholder.

- `poseflow_schema.py`
  - Khai báo variants `selected_31`, `hands_only`.

## 5.10 `src/slr/models/`

Nhóm này chủ yếu là **API cấp cao/placeholder**, không phải implementation train path hiện nay.

- `src/slr/models/skeleton/stgcnpp.py`
  - Placeholder wrapper ST-GCN++.

- `src/slr/models/skeleton/ctrgcn.py`
  - Placeholder wrapper CTR-GCN.

- `src/slr/models/skeleton/heads.py`
  - Placeholder classification head.

- `src/slr/models/regions/cnn_lstm.py`
  - Placeholder model.

- `src/slr/models/regions/video_transformer.py`
  - Placeholder model.

- `src/slr/models/regions/heads.py`
  - Placeholder head.

- `src/slr/models/hand_poseflow/two_stream.py`
  - Placeholder model.

- `src/slr/models/hand_poseflow/heads.py`
  - Placeholder head.

## 5.11 `src/slr/inference/`

Trạng thái: **placeholder**.

- `predict_video.py`
  - Parser cho inference video đơn.
  - Hiện chỉ log config/video/checkpoint.

- `visualize_prediction.py`
  - Parser cho visualize prediction.
  - Hiện chỉ log file input/output.

## 5.12 `scripts/`

### 5.12.1 Wrappers mỏng

- `scripts/00_build_index.py`
  - Gọi `slr.data.build_index.main`.

- `scripts/01_standardize_videos.py`
  - Gọi `slr.data.standardize_videos.main`.

- `scripts/02_extract_pose_rtmw.py`
  - Gọi `slr.pose.extract_rtmw.main`.

- `scripts/03_build_skeleton_inputs.py`
  - Gọi `slr.branches.skeleton.build_inputs.main`.

- `scripts/04_build_region_inputs.py`
  - Wrapper cho region branch scaffold.

- `scripts/05_build_hand_poseflow_inputs.py`
  - Wrapper cho hand poseflow branch scaffold.

- `scripts/train_skeleton.py`
  - Gọi `slr.branches.skeleton.train.main`.

- `scripts/evaluate_skeleton.py`
  - Gọi `slr.branches.skeleton.train.evaluate_main`.

- `scripts/train_regions.py`
  - Wrapper tới generic training scaffold.

- `scripts/train_hand_poseflow.py`
  - Wrapper tới generic training scaffold.

- `scripts/evaluate.py`
  - Wrapper tới generic evaluation scaffold.

- `scripts/sitecustomize.py`
  - Thêm `src/` vào path khi chạy wrapper scripts.

### 5.12.2 Utility scripts

- `scripts/check_skeleton_dataset.py`
  - Sanity check cho dataset loader và graph topology.
  - In shape, min/max, batch shape, adjacency shape.

- `scripts/prepare_kaggle_bundle.py`
  - Script đóng gói standardized frames + repo subset + checkpoint RTMW-l cho Kaggle pose extraction.
  - Có validate chặt chẽ:
    - path nằm trong project root
    - không nhúng `.git`, `data` ngoài standardized phần được phép, `outputs`, `.venv`
    - verify zip không chứa path cấm
  - Hỗ trợ:
    - zip mode
    - copy tree mode `--no-zip`
  - Sinh `MANIFEST.json` và `README_KAGGLE_BUNDLE.md`.

- `scripts/prepare_hf_skeleton_bundle.py`
  - Đóng gói train-ready skeleton data để upload Hugging Face.
  - Validate:
    - đủ `1013` file cho mỗi keypoint set
    - metadata.json đúng shape và số class
    - manifests có `status=ok` toàn bộ, class_id `0..99`
  - Sinh nhiều zip theo nhóm: graph tensors, manifests, reports, logs.

### 5.12.3 Tool scripts

- `scripts/tools/test_rtmw_mmpose_video.py`
  - Smoke test RTMW-l/MMPose trên một video.
  - Tự tìm config/checkpoint trong `checkpoints/pose/rtmw_l/`.
  - Ghi summary JSON và markdown report dưới `_rtmw_mmpose_video_test_output/`.

- `scripts/tools/visualize_selected_27_samples.py`
  - Thực tế hỗ trợ cả `selected_27` và `selected_31`.
  - Đọc skeleton + pose manifests, chọn sample có hand confidence tốt, overlay keypoints lên standardized frames, sinh contact sheet + report.
  - Hữu ích để verify mapping reduced keypoints.

## 5.13 `docs/`

- `docs/skeleton_training_baseline.md`
  - Hướng dẫn ngắn cho baseline train/eval skeleton.

- `docs/skeleton_stgcnpp_integration.md`
  - Mô tả ST-GCN++ repo-local, input format, command chạy, limitation.

- `docs/standard_label_smoothing.md`
  - Mô tả Standard Label Smoothing, config tương ứng, command smoke/full runs.

## 5.14 `reports/`

- `reports/preprocessing/README.md`
  - Mô tả các layer preprocessing.

- `reports/preprocessing/nslt300_index_and_standardization_report.md`
  - Báo cáo chuẩn hóa/index cho `nslt300`.

- `reports/preprocessing/nslt1000_index_and_standardization_report.md`
  - Báo cáo chuẩn hóa/index cho `nslt1000`.

- `reports/preprocessing/kaggle_sub300_bundle_report.md`
  - Báo cáo đóng gói Kaggle bundle cho `nslt300`.
  - Xác nhận:
    - không chạy pose extraction
    - không train
    - chỉ copy standardized frames/manifests + repo subset + RTMW-l checkpoint

- `reports/experiments/README.md`
  - Quy ước lưu artifacts thí nghiệm.

## 5.15 `notebooks/`

- `notebooks/01_explore_wlasl.ipynb`
  - Khám phá metadata WLASL.

- `notebooks/02_check_pose_quality.ipynb`
  - Kiểm tra chất lượng pose extraction.

- `notebooks/03_visualize_keypoints.ipynb`
  - Trực quan hóa keypoints.

## 5.16 `data/`

### `data/datasets/WLASL/raw/`

- `metadata/WLASL_v0.3.json`
  - Master manifest gốc.

- `metadata/nslt_100.json`, `nslt_300.json`, `nslt_1000.json`, `nslt_2000.json`
  - Split manifests classification-oriented.

- `metadata/wlasl_class_list.txt`
  - Mapping class id -> gloss.

- `metadata/wlasl_class_list_corrected.txt`
  - Danh sách class sửa tay.

- `metadata/missing.txt`
  - Các video IDs thiếu local.

- `docs/README.md`
  - Báo cáo rất hữu ích về trạng thái raw data local: số video local, coverage, chênh lệch với master/NSLT.

- `docs/WLASL_raw_analysis_vi.md`
  - Tài liệu phân tích raw dataset bằng tiếng Việt.

- `videos/*.mp4`
  - Video gốc WLASL local snapshot.
  - Đây là artifact lớn; không phân tích từng file riêng lẻ.

### `data/datasets/WLASL/index/`

File sinh ra thực tế:

- `master_instances.csv`
- `available_instances.csv`
- `missing_instances.csv`
- `nslt_only_instances.csv`
- `class_id_to_gloss.csv`
- `video_to_split.csv`
- `video_to_split_all.csv`
- `reports/dataset_summary.md`
- `reports/coverage_by_split.json`
- `reports/coverage_by_class.json`
- `reports/invalid_ids.txt`
- `subsets/<subset>/{train,val,test}.csv`
- `subsets_available/<subset>/{train,val,test}.csv`

### `data/datasets/WLASL/standardized/`

Chứa:

- `frames/<subset>/<split>/<sample_id>/*.jpg`
- `videos/<subset>/<split>/<sample_id>.mp4` nếu bật save video
- `manifests/<subset>_<split>.csv`
- `reports/<subset>_standardization_report.md`
- `logs/standardize_<subset>.log`

### `data/datasets/WLASL/pose/rtmw_l/`

Chứa:

- `wholebody_133/<subset>/<split>/<sample_id>.npz`
- `manifests/<subset>_<split>.csv`
- `reports/<subset>_pose_quality_report.md`
- `logs/extract_pose_<subset>.log`

### `data/datasets/WLASL/branch_inputs/skeleton/rtmw_l/`

Chứa:

- `selected_27/<subset>/<split>/*.npz`
- `selected_31/<subset>/<split>/*.npz`
- `normalized/selected_27/<subset>/<split>/*.npz`
- `normalized/selected_31/<subset>/<split>/*.npz`
- `graph_tensors/selected_27/<subset>/<split>/*.npz`
- `graph_tensors/selected_31/<subset>/<split>/*.npz`
- `manifests/*.csv`
- `reports/*_skeleton_inputs_report.md`
- `logs/*.log`
- `README.md`
- `metadata.json`

## 5.17 `outputs/`

- `outputs/skeleton/<run_name>/`
  - `checkpoints/best.pt`
  - `checkpoints/last.pt`
  - `config_resolved.yaml`
  - `metrics.json`
  - `summary.json`
  - `train_log.csv`
  - `eval_test_best.json` ở một số run có evaluate sau train

## 5.18 `checkpoints/`

- `checkpoints/pose/rtmw_l/`
  - Chứa RTMW-l `.pth` và config `.py`.
  - Đây là đầu vào bắt buộc cho stage pose extraction.

## 5.19 `hf_bundle/`, `kaggle_bundle/`, `kaggle_sub300_bundle/`

- Đây là artifact đóng gói.
- Không phải code lõi, nhưng rất quan trọng cho workflow triển khai ra Kaggle/Hugging Face.

## 6. Điểm mạnh kiến trúc hiện tại

- Pipeline phân tầng khá rõ, ít chồng chéo giữa raw data và derived data.
- Skeleton branch đã có đường chạy manifest-driven tương đối sạch.
- Dataset loader có xử lý remap path giữa local/HF/Kaggle, hữu ích thực tế.
- Preprocessing tạo report/log ở từng tầng, dễ audit.
- Training loop hỗ trợ:
  - CLI overrides
  - checkpointing
  - dry-run
  - W&B optional
  - evaluate checkpoint local

## 7. Điểm yếu / hạn chế hiện tại

- `regions`, `hand_poseflow`, `inference`, generic `training/evaluate` vẫn là scaffold.
- Có hai “hệ model”:
  - `src/slr/branches/skeleton/models/*` là model thật
  - `src/slr/models/*` lại phần lớn là placeholder
  - Điều này dễ gây nhầm nếu mới vào repo.
- `selected_49` mới khai báo schema, chưa có mapping/graph thật.
- Language Label Smoothing mới có helper logic, chưa nối vào production train loop.
- Tài liệu cũ có chỗ mô tả repo như scaffold, trong khi code hiện đã có phần production path cho skeleton.

## 8. Cách chạy nhanh theo trạng thái hiện tại

### 8.1 Build index

```bash
python scripts/00_build_index.py --config configs/preprocessing/index.yaml
```

### 8.2 Standardize

```bash
python scripts/01_standardize_videos.py --config configs/preprocessing/standardize.yaml
```

### 8.3 Extract pose

```bash
python scripts/02_extract_pose_rtmw.py --config configs/preprocessing/pose_rtmw_l.yaml
```

### 8.4 Build skeleton inputs

```bash
python scripts/03_build_skeleton_inputs.py --config configs/branches/skeleton/stgcnpp_31.yaml
```

### 8.5 Dry-run train skeleton

```bash
python scripts/train_skeleton.py --config configs/train/skeleton_selected_31_stgcnpp.yaml --run-name dry-sel31-stgcnpp --dry-run --no-wandb
```

### 8.6 Smoke train

```bash
python scripts/train_skeleton.py --config configs/train/skeleton_selected_31_stgcnpp.yaml --run-name smoke-sel31-stgcnpp --epochs 1 --limit-train 32 --limit-val 16 --limit-test 16 --batch-size 8 --num-workers 0 --no-wandb
```

### 8.7 Evaluate

```bash
python scripts/evaluate_skeleton.py --config outputs/skeleton/smoke-sel31-stgcnpp/config_resolved.yaml --checkpoint outputs/skeleton/smoke-sel31-stgcnpp/checkpoints/best.pt --split test --batch-size 8
```

## 9. Kết luận ngắn

Đây không còn là repo scaffold thuần túy. Tính đến trạng thái hiện tại của folder:

- preprocessing `index -> standardize -> pose -> skeleton inputs` đã có implementation thật
- training/evaluation skeleton đã chạy được end-to-end
- ST-GCN++ repo-local đã tích hợp
- Standard Label Smoothing đã tích hợp
- dữ liệu train-ready `nslt100` cho `selected_27` và `selected_31` đã tồn tại
- các nhánh `regions`, `hand_poseflow`, `inference` vẫn chủ yếu ở mức khung

Nếu tiếp tục phát triển repo này, đường hợp lý nhất là:

1. tiếp tục tối ưu và benchmark nhánh `skeleton`
2. hoàn thiện `LanguageLS`
3. thêm `CTR-GCN`
4. chỉ sau đó mới mở rộng `regions` và `hand_poseflow`
