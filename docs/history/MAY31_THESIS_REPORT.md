# MAY31 Thesis Report

Ngày lập báo cáo: `2026-05-31`

## 1. Mục tiêu và phạm vi

File này tổng hợp việc đọc, đối chiếu và phân tích toàn bộ repo `Recognizing-sign-language-at-the-word-level` tại thời điểm `2026-05-31`, với mục tiêu:

1. Làm rõ cấu trúc thư mục và vai trò của từng nhóm file.
2. Mô tả luồng hoạt động thật sự của pipeline từ dữ liệu thô đến huấn luyện.
3. Chỉ ra file nào đang chạy production/research thực tế, file nào còn là scaffold hoặc placeholder.
4. Ghi nhận tình trạng hiện tại của các nhánh `skeleton`, `regions`, `hand_poseflow`, phần train/eval dùng chung, phần inference và phần đóng gói artifact.

Lưu ý về phạm vi:

- Tôi đã đọc các file mã nguồn Python, YAML, Markdown, script CLI và notebook trong repo.
- Với các thư mục artifact lớn như `data/`, `outputs/`, `hf_bundle/`, `hf_sub300_bundle/`, `hf_sub1000_bundle/`, `kaggle_bundle/`, `kaggle_sub300_bundle/`, `checkpoints/`, báo cáo phân tích theo cấu trúc, vai trò và cách chúng được tạo ra, thay vì liệt kê từng file nhị phân hoặc từng ảnh PNG riêng lẻ.
- Ba notebook trong `notebooks/` hiện chỉ có một cell markdown mở đầu, tức là mới đóng vai trò placeholder định hướng chứ chưa phải notebook phân tích hoàn chỉnh.

## 2. Kết luận ngắn gọn

Repo này không còn là scaffold thuần túy. Nó đã có một pipeline `skeleton` chạy được theo đúng chuỗi:

`raw metadata/video -> index manifests -> standardized frames/video -> RTMW-l wholebody_133 pose -> selected_27/selected_31 skeleton graph tensors -> train/evaluate`

Nhưng mức độ hoàn thiện giữa các phần rất khác nhau:

- Hoàn thiện và chạy thật: `src/slr/data`, `src/slr/pose`, `src/slr/branches/skeleton`, một phần `src/slr/training`, các script `00` tới `03`, `train_skeleton.py`, `evaluate_skeleton.py`, một số script đóng gói bundle.
- Mới ở mức scaffold/placeholder: `src/slr/branches/regions`, `src/slr/branches/hand_poseflow`, `src/slr/models/*` cấp generic, `src/slr/training/train.py`, `src/slr/training/evaluate.py`, `src/slr/inference/*`.
- Artifact và tài liệu đã khá dày: repo có dữ liệu đã chuẩn hóa, pose, graph tensor, output train smoke test, bundle cho Hugging Face/Kaggle, và nhiều báo cáo trung gian.

Điểm quan trọng nhất để đọc repo đúng:

- Đường chạy thật cho skeleton nằm ở `src/slr/branches/skeleton/train.py`, không nằm ở `src/slr/training/train.py`.
- Model skeleton chạy thật nằm ở `src/slr/branches/skeleton/models/`, không phải `src/slr/models/skeleton/`.
- Các thư mục `src/slr/models/*`, `src/slr/inference/*`, `src/slr/branches/regions/*`, `src/slr/branches/hand_poseflow/*` hiện chủ yếu là khung mở rộng cho tương lai.

## 3. Snapshot trạng thái hiện tại

| Thành phần | Trạng thái | Ghi chú |
| --- | --- | --- |
| Index layer | Implemented | Build manifest từ `WLASL_v0.3.json`, `nslt_*.json`, `missing.txt`, inventory video local |
| Standardized layer | Implemented | Crop theo bbox, resize + letterbox, sinh frame/video chuẩn hóa |
| RTMW pose extraction | Implemented | Tích hợp MMPose RTMW-l, fallback device, quality report |
| Skeleton input build | Implemented | Chọn keypoint, normalize, fit confidence scale, tạo graph tensor `C,T,V,M` |
| Skeleton dataset loader | Implemented | Load manifest, remap path local/HF/Kaggle, kiểm tra shape |
| Skeleton training/eval | Implemented | CE + standard label smoothing, checkpoint, scheduler, W&B optional |
| ST-GCN baseline | Implemented | `SimpleSTGCN` dùng để kiểm chứng pipeline |
| Repo-local ST-GCN++ | Implemented | Nằm ở `src/slr/branches/skeleton/models/stgcnpp.py` |
| CTR-GCN | Placeholder | Chỉ có placeholder ở `src/slr/models/skeleton/ctrgcn.py` |
| Region branch | Placeholder | Có CLI khung và config baseline nhưng chưa có crop/build/train thực |
| Hand poseflow branch | Placeholder | Có CLI khung và config baseline nhưng chưa có input/model thực |
| Generic training/eval | Placeholder | `src/slr/training/train.py`, `evaluate.py` chỉ log dry-run plan |
| Inference | Placeholder | Chỉ có parser và log |
| Hugging Face/Kaggle bundle | Partially implemented | Có script đóng gói và thư mục bundle đã sinh sẵn |

## 4. Luồng hoạt động end-to-end của hệ thống

### 4.1 Luồng dữ liệu thực tế

Luồng hiện tại của repo đi theo thứ tự:

1. `scripts/00_build_index.py`
2. `scripts/01_standardize_videos.py`
3. `scripts/02_extract_pose_rtmw.py`
4. `scripts/03_build_skeleton_inputs.py`
5. `scripts/train_skeleton.py`
6. `scripts/evaluate_skeleton.py`

### 4.2 Ý nghĩa của từng bước

#### Bước 1: build index

File lõi: `src/slr/data/build_index.py`

Đầu vào:

- `data/datasets/WLASL/raw/metadata/WLASL_v0.3.json`
- `data/datasets/WLASL/raw/metadata/nslt_100.json`
- `data/datasets/WLASL/raw/metadata/nslt_300.json`
- `data/datasets/WLASL/raw/metadata/nslt_1000.json`
- `data/datasets/WLASL/raw/metadata/nslt_2000.json`
- `data/datasets/WLASL/raw/metadata/wlasl_class_list.txt`
- `data/datasets/WLASL/raw/metadata/missing.txt`
- `data/datasets/WLASL/raw/videos/*.mp4`

Xử lý chính:

- Chuẩn hóa `video_id`.
- Đọc master manifest WLASL và flatten thành dataframe.
- Đọc class list.
- Đọc các manifest NSLT theo subset.
- Quét thư mục video local để biết sample nào có mặt thật trên máy.
- Build các bảng:
  - `master_instances`
  - `available_instances`
  - `missing_instances`
  - `nslt_only_instances`
  - `class_id_to_gloss`
  - `video_to_split`
  - `subset manifests`
- Ghi report coverage theo split và class.

Đầu ra:

- `data/datasets/WLASL/index/*.csv`
- `data/datasets/WLASL/index/subsets/<subset>/{train,val,test}.csv`
- `data/datasets/WLASL/index/subsets_available/<subset>/{train,val,test}.csv`
- `data/datasets/WLASL/index/reports/*`

#### Bước 2: standardize videos

File lõi: `src/slr/data/standardize_videos.py`

Đầu vào:

- Manifest khả dụng từ `data/datasets/WLASL/index/subsets_available/<subset>/<split>.csv`
- Video thô trong `raw/videos`

Xử lý chính:

- Đọc thông tin video bằng OpenCV.
- Giải quyết `start_frame`, `end_frame`.
- Dùng bbox từ metadata nếu hợp lệ, nếu không fallback sang full frame.
- Expand bbox theo margin trong config.
- Crop từng frame.
- Resize giữ tỉ lệ.
- Letterbox vào kích thước chuẩn `288x384`.
- Tùy config, ghi:
  - frame `.jpg`
  - hoặc video chuẩn hóa `.mp4`
- Sinh standardized manifest và report.

Đầu ra:

- `data/datasets/WLASL/standardized/frames/<subset>/<split>/<sample_id>/*.jpg`
- `data/datasets/WLASL/standardized/videos/<subset>/<split>/<sample_id>.mp4` nếu bật
- `data/datasets/WLASL/standardized/manifests/<subset>_<split>.csv`
- `data/datasets/WLASL/standardized/reports/*`

#### Bước 3: extract RTMW-l pose

File lõi: `src/slr/pose/extract_rtmw.py`

Đầu vào:

- Standardized manifests
- Standardized frame directories
- MMPose RTMW-l config/checkpoint trong `checkpoints/pose/rtmw_l/`

Xử lý chính:

- Resolve config/checkpoint RTMW-l.
- Import `torch` và `MMPoseInferencer`.
- Chọn device theo config, tự fallback CPU nếu CUDA không có.
- Chạy inferencer trên toàn bộ frame sequence của từng sample.
- Nếu có nhiều người trong một frame, chọn người chính bằng:
  - bbox score
  - bbox area
  - mean keypoint score
- Chuyển kết quả về tensor `(T, 133, 3)`.
- Với frame không detect được người, chèn frame rỗng `NaN xy + score 0`.
- Tính pose quality:
  - mean confidence
  - confidence theo body/face/left_hand/right_hand
  - valid frame ratio
- Ghi `.npz` pose và pose manifest.

Đầu ra:

- `data/datasets/WLASL/pose/rtmw_l/wholebody_133/<subset>/<split>/<sample_id>.npz`
- `data/datasets/WLASL/pose/rtmw_l/manifests/<subset>_<split>.csv`
- `data/datasets/WLASL/pose/rtmw_l/reports/<subset>_pose_quality_report.md`

#### Bước 4: build skeleton inputs

File lõi: `src/slr/branches/skeleton/build_inputs.py`

Đầu vào:

- Pose manifests
- Pose `.npz` wholebody 133
- Config branch `selected_27` hoặc `selected_31`

Xử lý chính:

- Load pose `(T, 133, 3)`.
- Chọn subset keypoints:
  - `selected_27 = 7 body + 10 left hand + 10 right hand`
  - `selected_31 = selected_27 + 4 mouth`
- Fit một `confidence_scale` từ split train theo percentile.
- Normalize `x,y` từ pixel sang `[-1, 1]`.
- Normalize confidence về `[0, 1]`.
- Sanitize non-finite values.
- Cố định độ dài sequence về `T = 150`:
  - sequence ngắn: repeat
  - sequence dài: lấy head
- Chuyển về graph tensor `(C, T, V, M)` với `M = 1`.
- Ghi selected pose, normalized pose, graph tensor, manifest, report.

Đầu ra:

- `data/datasets/WLASL/branch_inputs/skeleton/rtmw_l/selected_27/<subset>/<split>/<sample_id>.npz`
- `data/datasets/WLASL/branch_inputs/skeleton/rtmw_l/selected_31/<subset>/<split>/<sample_id>.npz`
- `data/datasets/WLASL/branch_inputs/skeleton/rtmw_l/normalized/<keypoint_set>/<subset>/<split>/<sample_id>.npz`
- `data/datasets/WLASL/branch_inputs/skeleton/rtmw_l/graph_tensors/<keypoint_set>/<subset>/<split>/<sample_id>.npz`
- `data/datasets/WLASL/branch_inputs/skeleton/rtmw_l/manifests/*.csv`
- `data/datasets/WLASL/branch_inputs/skeleton/rtmw_l/reports/*`

#### Bước 5: train skeleton model

File lõi: `src/slr/branches/skeleton/train.py`

Đầu vào:

- Config train trong `configs/train/*.yaml`
- Skeleton graph tensor manifests

Xử lý chính:

- Resolve config và CLI overrides.
- Validate sự nhất quán giữa:
  - `dataset.expected_shape`
  - `graph.layout`
  - `model.in_channels`
  - `model.num_nodes`
  - `model.num_classes`
- Build `SkeletonGraphDataset` cho `train/val/test`.
- Build `SkeletonGraph`.
- Build model qua `build_skeleton_model()`:
  - `simple_stgcn`
  - `stgcnpp`
- Build loss:
  - `cross_entropy`
  - `standard_label_smoothing`
- Build optimizer/scheduler.
- Optional AMP.
- Optional W&B.
- Train loop theo epoch, validate mỗi epoch.
- Save `best.pt`, `last.pt`, optional `epoch_XXX.pt`.
- Reload `best.pt`, evaluate trên `test`.
- Ghi `config_resolved.yaml`, `metrics.json`, `train_log.csv`, `summary.json`.

Đầu ra:

- `outputs/skeleton/<run_name>/...`

#### Bước 6: evaluate checkpoint

File lõi: vẫn là `src/slr/branches/skeleton/train.py` qua hàm `run_evaluation()`

Đầu vào:

- `config_resolved.yaml`
- checkpoint `.pt`
- split muốn đánh giá

Đầu ra:

- `eval_<split>_<checkpoint_stem>.json` trong `output_dir` nếu config có `experiment.output_dir`

## 5. Phân tích cấu trúc thư mục cấp cao

### 5.1 Root của repo

| File/thư mục | Vai trò | Đánh giá |
| --- | --- | --- |
| `README.md` | Giới thiệu ngắn gọn mục tiêu, dataset, pipeline, roadmap | Đúng định hướng nhưng ngắn hơn trạng thái code thật |
| `PROJECT_STRUCTURE_GUIDE.md` | Tài liệu giải thích kiến trúc thư mục và pipeline | Có giá trị onboarding, nhưng còn phản ánh giai đoạn scaffold nhiều hơn hiện trạng đầy đủ |
| `TRAINING.md` | Ghi log triển khai training baseline skeleton và ST-GCN++ | Là tài liệu kỹ thuật quan trọng để hiểu evolution của training |
| `IMPLEMENTATION.md` | Nhật ký tích hợp ST-GCN++ vào pipeline skeleton | Ghi rõ phần đã triển khai, smoke test, giới hạn |
| `LATEST_INFO.md`, `NEW_LATEST_INFO.md`, `bundle1k.md`, `StandardLS.md` | Các ghi chú trạng thái, experiment và bundle | Có tính snapshot, hữu ích nhưng không phải source of truth duy nhất |
| `MAY27_THESIS_REPORT.md` | Báo cáo phân tích repo trước đó | Là tài liệu nền để so sánh trạng thái cũ và mới |
| `pyproject.toml` | Khai báo package `slr` theo `src` layout | Rất gọn, chuẩn setuptools |
| `requirements.txt` | Dependency nền cho preprocessing và util | Gồm `numpy`, `pandas`, `opencv-python`, `pyyaml`, `tqdm`, `scikit-learn`, `matplotlib`, `wandb` |
| `requirements-rtmw.txt` | Dependency cho RTMW/MMPose | Gồm `torch`, `torchvision`, `mmengine`, `mmcv`, `mmdet`, `mmpose` |
| `requirements-kaggle-train.txt` | Dependency tối giản cho training/bundle trên Kaggle | Hiện rất gọn: `wandb`, `huggingface_hub`, `pyyaml` |
| `sitecustomize.py` | Auto thêm `src/` vào `sys.path` | Hữu ích để chạy script ad-hoc từ repo root |
| `slr/__init__.py` | Import shim để `import slr` map tới `src/slr` | Giúp không cần cài editable ngay |
| `.gitignore` | Quy tắc bỏ qua artifact | Hỗ trợ repo nghiên cứu có nhiều output |
| `.venv-rtmw310/` | Môi trường ảo cục bộ | Không phải phần logic dự án |
| `.git/` | Metadata git | Không phân tích ở mức code |

### 5.2 `configs/`

Thư mục này là lớp điều khiển toàn bộ hành vi pipeline. Config được chia thành các họ:

- `configs/dataset/`
- `configs/preprocessing/`
- `configs/branches/`
- `configs/train/`
- `configs/experiments/`

#### 5.2.1 `configs/dataset/`

| File | Vai trò |
| --- | --- |
| `configs/dataset/wlasl.yaml` | Mô tả dataset root, raw metadata/video/docs, danh sách subset (`nslt100`, `nslt300`, `nslt1000`, `nslt2000`) và split mặc định |

#### 5.2.2 `configs/preprocessing/`

| File | Vai trò |
| --- | --- |
| `configs/preprocessing/index.yaml` | Config cho index layer |
| `configs/preprocessing/standardize.yaml` | Config chuẩn hóa video cho `nslt100` mặc định |
| `configs/preprocessing/standardize_nslt300.yaml` | Variant chuẩn hóa cho `nslt300` |
| `configs/preprocessing/standardize_nslt1000.yaml` | Variant chuẩn hóa cho `nslt1000` |
| `configs/preprocessing/pose_rtmw_l.yaml` | Config trích xuất pose RTMW-l local |
| `configs/preprocessing/pose_rtmw_l_kaggle.yaml` | Variant pose config cho môi trường Kaggle |
| `configs/preprocessing/region_crops.yaml` | Placeholder config cho region branch |
| `configs/preprocessing/poseflow.yaml` | Placeholder config cho hand poseflow branch |
| `configs/preprocessing/index.yaml` | Source of truth cho bước tạo manifest index |

#### 5.2.3 `configs/branches/skeleton/`

Nhóm config này phục vụ bước build branch input, không phải train loop.

| File | Vai trò |
| --- | --- |
| `stgcnpp_27.yaml` | Build skeleton input `selected_27` cho `nslt100` |
| `stgcnpp_31.yaml` | Build skeleton input `selected_31` cho `nslt100` |
| `stgcnpp_27_nslt300.yaml` | Cùng logic nhưng cho `nslt300` |
| `stgcnpp_31_nslt300.yaml` | Cùng logic nhưng cho `nslt300` |
| `stgcnpp_27_nslt1000.yaml` | Cùng logic nhưng cho `nslt1000` |
| `stgcnpp_31_nslt1000.yaml` | Cùng logic nhưng cho `nslt1000` |
| `stgcnpp_31_languagels.yaml` | Config nhánh skeleton có ý định liên quan Language Label Smoothing |
| `ctrgcn_31.yaml` | Config dành cho hướng CTR-GCN, nhưng model CTR-GCN thật chưa hiện diện ở nhánh chạy production |

#### 5.2.4 `configs/train/`

Đây là nhóm config train thực sự đang dùng với `scripts/train_skeleton.py`.

Các mẫu tên chính:

- `skeleton_selected_27_baseline.yaml`
- `skeleton_selected_31_baseline.yaml`
- `skeleton_selected_27_stgcnpp.yaml`
- `skeleton_selected_31_stgcnpp.yaml`
- `skeleton_selected_27_stgcnpp_standardls_eps005.yaml`
- `skeleton_selected_27_stgcnpp_standardls_eps01.yaml`
- `skeleton_selected_27_stgcnpp_standardls_eps03.yaml`
- `skeleton_selected_31_stgcnpp_standardls_eps005.yaml`
- `skeleton_selected_31_stgcnpp_standardls_eps01.yaml`
- `skeleton_selected_31_stgcnpp_standardls_eps03.yaml`
- Các bản tương ứng cho `nslt300`
- Các bản tương ứng cho `nslt1000`

Quy luật đặt tên:

- `selected_27` hoặc `selected_31`: bộ keypoint.
- `baseline`: thường dùng `simple_stgcn`.
- `stgcnpp`: dùng model ST-GCN++ local.
- `standardls_epsXX`: bật standard label smoothing với `epsilon` tương ứng.
- `nslt300`, `nslt1000`: đổi subset và số lớp.

Điểm kỹ thuật quan trọng:

- Standard label smoothing đã thực sự được tích hợp trong loss factory bằng `nn.CrossEntropyLoss(label_smoothing=epsilon)`.
- File `src/slr/branches/skeleton/label_smoothing.py` chỉ là helper numpy riêng, không phải đường loss được train loop gọi trực tiếp.

#### 5.2.5 `configs/branches/regions/` và `configs/branches/hand_poseflow/`

| File | Vai trò | Trạng thái |
| --- | --- | --- |
| `configs/branches/regions/face_hands_baseline.yaml` | Khai báo branch regions, input dirs, model `cnn_lstm`, batch/lr | Placeholder |
| `configs/branches/hand_poseflow/hand_poseflow_baseline.yaml` | Khai báo branch hand_poseflow, hand dirs, poseflow dir, model `two_stream` | Placeholder |

#### 5.2.6 `configs/experiments/`

| File | Vai trò | Nhận xét |
| --- | --- | --- |
| `configs/experiments/skeleton_nslt100_debug.yaml` | Mô tả một experiment debug bằng cách trỏ tới `branch_config` và `dataset_config` | Mang hơi hướng kiến trúc cũ/generic; không phải đường config mà `scripts/train_skeleton.py` đang dùng trực tiếp |

### 5.3 `data/`

Thư mục dữ liệu hiện đã chứa cả raw lẫn các tầng dẫn xuất.

#### 5.3.1 `data/datasets/WLASL/raw/`

Vai trò:

- Giữ dataset gốc và metadata gốc.

Các file đáng chú ý:

| File | Vai trò |
| --- | --- |
| `raw/metadata/WLASL_v0.3.json` | Master manifest 2000 gloss |
| `raw/metadata/nslt_100.json` | Split manifest 100 lớp |
| `raw/metadata/nslt_300.json` | Split manifest 300 lớp |
| `raw/metadata/nslt_1000.json` | Split manifest 1000 lớp |
| `raw/metadata/nslt_2000.json` | Split manifest 2000 lớp |
| `raw/metadata/wlasl_class_list.txt` | Mapping class_id -> gloss |
| `raw/metadata/wlasl_class_list_corrected.txt` | Biến thể/class list đã chỉnh |
| `raw/metadata/missing.txt` | Danh sách video ID chưa có local |
| `raw/docs/README.md` | Tài liệu phân tích snapshot dữ liệu local |
| `raw/docs/WLASL_raw_analysis_vi.md` | Tài liệu phân tích dataset bằng tiếng Việt |

Theo `raw/docs/README.md`, snapshot local tại thời điểm tài liệu đó được tạo có:

- 2,000 gloss
- 21,083 master instances
- 11,980 MP4 local
- 9,103 video ID thiếu local

#### 5.3.2 `data/datasets/WLASL/index/`

Vai trò:

- Tầng manifest chuẩn cho toàn pipeline.

Nội dung điển hình:

- `master_instances.csv`
- `available_instances.csv`
- `missing_instances.csv`
- `nslt_only_instances.csv`
- `class_id_to_gloss.csv`
- `video_to_split.csv`
- `video_to_split_all.csv`
- `subsets/`
- `subsets_available/`
- `reports/`

#### 5.3.3 `data/datasets/WLASL/standardized/`

Vai trò:

- Tầng video/frame đã crop/resize/letterbox.

Nội dung:

- `frames/<subset>/<split>/<sample_id>/*.jpg`
- `videos/` nếu bật
- `manifests/<subset>_<split>.csv`
- `logs/`
- `reports/`

#### 5.3.4 `data/datasets/WLASL/pose/`

Vai trò:

- Tầng feature pose dùng chung.

Nội dung:

- `pose/rtmw_l/wholebody_133/<subset>/<split>/<sample_id>.npz`
- `pose/rtmw_l/manifests/*.csv`
- `pose/rtmw_l/logs/`
- `pose/rtmw_l/reports/`

#### 5.3.5 `data/datasets/WLASL/branch_inputs/skeleton/rtmw_l/`

Vai trò:

- Tầng input train-ready cho skeleton branch.

Nội dung quan sát được:

- `selected_27/`
- `selected_31/`
- `normalized/`
- `graph_tensors/`
- `manifests/`
- `reports/`
- `logs/`
- `metadata.json`
- `README.md`

Đây là thư mục quan trọng nhất cho training hiện tại.

#### 5.3.6 Các nhánh còn lại

Trong code, repo đã dành chỗ cho:

- `branch_inputs/regions/rtmw_l/...`
- `branch_inputs/hand_poseflow/rtmw_l/...`

Nhưng việc build nội dung cho hai nhánh này chưa được hiện thực end-to-end.

### 5.4 `checkpoints/`

Hiện repo dành riêng `checkpoints/pose/rtmw_l/` cho MMPose RTMW-l config/checkpoint.

Vai trò:

- Chứa model config `.py`
- Chứa weight `.pth`

Training checkpoint của skeleton branch lại được ghi vào `outputs/skeleton/<run_name>/checkpoints/`, tức là repo đang tách rất rõ:

- pose backbone checkpoint
- recognition model checkpoint

### 5.5 `outputs/`

Quan sát hiện có:

- `outputs/skeleton/`
- `outputs/smoke_metrics/`

Vai trò:

- Lưu output của các lần train/eval thật hoặc smoke test.

Theo logic code, mỗi run skeleton tạo:

- `config_resolved.yaml`
- `metrics.json`
- `summary.json`
- `train_log.csv`
- `checkpoints/best.pt`
- `checkpoints/last.pt`

### 5.6 `experiments/`

Hiện có khung:

- `experiments/skeleton/`
- `experiments/regions/`
- `experiments/hand_poseflow/`

Vai trò:

- Không phải output path đang được train skeleton hiện dùng mặc định.
- Là nơi dự kiến chứa artifact experiment theo cách tổ chức cấp cao hơn.

### 5.7 `reports/`

Hiện có:

- `reports/preprocessing/`
- `reports/training/`
- `reports/experiments/`

Vai trò:

- Chứa markdown report về preprocessing, training config, keypoint visualizations.

Các file đáng chú ý:

| File | Vai trò |
| --- | --- |
| `reports/preprocessing/README.md` | Mô tả các layer preprocessing |
| `reports/preprocessing/*_report.md` | Báo cáo riêng cho index/standardize/pose/skeleton/bundle |
| `reports/training/nslt300_training_config_report.md` | Ghi nhận config train subset 300 |
| `reports/training/nslt300_standardls_config_report.md` | Ghi nhận cấu hình standard label smoothing |
| `reports/training/nslt1000_training_config_report.md` | Ghi nhận cấu hình train subset 1000 |
| `reports/experiments/README.md` | Quy ước lưu artifact experiment |
| `reports/preprocessing/keypoint_visualizations/nstl100/*` | Ảnh minh họa overlay keypoint đã sinh |

### 5.8 `docs/`

| File | Vai trò |
| --- | --- |
| `docs/skeleton_training_baseline.md` | Hướng dẫn chạy training skeleton baseline |
| `docs/skeleton_stgcnpp_integration.md` | Tài liệu tích hợp ST-GCN++ |
| `docs/standard_label_smoothing.md` | Tài liệu nhánh experiment Standard LS |

### 5.9 `notebooks/`

Hiện cả 3 notebook chỉ có một markdown cell mở đầu:

| Notebook | Vai trò hiện tại |
| --- | --- |
| `01_explore_wlasl.ipynb` | Placeholder cho khám phá metadata WLASL |
| `02_check_pose_quality.ipynb` | Placeholder cho kiểm tra chất lượng pose |
| `03_visualize_keypoints.ipynb` | Placeholder cho trực quan hóa keypoint |

Nhận xét:

- Chưa có nội dung phân tích thực thi.
- Chỉ đóng vai trò định hướng chủ đề notebook sẽ làm.

### 5.10 Bundle directories

Repo hiện có cả bundle đã tạo sẵn:

| Thư mục | Vai trò |
| --- | --- |
| `hf_bundle/` | Bundle skeleton cho Hugging Face, subset nhỏ hơn |
| `hf_sub300_bundle/` | Bundle skeleton cho `nslt300` |
| `hf_sub1000_bundle/` | Bundle skeleton cho `nslt1000` |
| `kaggle_bundle/` | Bundle standardized dataset + repo + checkpoint cho Kaggle |
| `kaggle_sub300_bundle/` | Bundle Kaggle cho `nslt300` |
| `kaggle_sub1000_bundle/` | Thư mục có mặt, phục vụ đóng gói subset 1000 |

Nội dung quan sát được cho các bundle HF:

- `graph_tensors_selected_27*.zip`
- `graph_tensors_selected_31*.zip`
- `manifests.zip`
- `reports.zip`
- `logs.zip`
- `metadata.json`
- `README.md`

Nội dung quan sát được cho bundle Kaggle:

- `standardized_<subset>.zip`
- `repo.zip`
- `MANIFEST.json`
- `README_KAGGLE_BUNDLE.md`
- `checkpoints/`

## 6. Phân tích chi tiết package `src/slr`

## 6.1 `src/slr/utils`

Đây là tầng utility dùng lại ở nhiều nơi, và phần lớn đã hoàn thiện.

| File | Vai trò | Cách hoạt động |
| --- | --- | --- |
| `src/slr/utils/io.py` | I/O chung | `ensure_dir`, `read/write_json`, `read/write_yaml`, `read/write_csv`, `write_text` |
| `src/slr/utils/logging.py` | Logging chung | Có `get_logger()` cho logger nhẹ và `setup_logger()` cho logger console + file |
| `src/slr/utils/video.py` | Xử lý video bằng OpenCV | Probe metadata, đọc frame range, ghi video từ frames |
| `src/slr/utils/image.py` | Xử lý ảnh | Đọc ảnh, ép về `uint8`, resize giữ tỉ lệ với letterbox, lưu JPEG/PNG |
| `src/slr/utils/bbox.py` | Xử lý bbox | Parse bbox, validate, expand theo margin, clip, stringify |
| `src/slr/utils/seed.py` | Seed cơ bản | Seed Python và NumPy |

Nhận xét:

- `utils` gọn và thực dụng.
- Hầu như mọi stage preprocessing đều dựa vào `io.py`, `logging.py`, `video.py`, `image.py`, `bbox.py`.

## 6.2 `src/slr/data`

### 6.2.1 `src/slr/data/manifests.py`

Vai trò:

- Định nghĩa schema cột chuẩn cho mọi tầng manifest:
  - `MASTER_INSTANCE_COLUMNS`
  - `SUBSET_MANIFEST_COLUMNS`
  - `CLASS_MAP_COLUMNS`
  - `STANDARDIZED_COLUMNS`
  - `POSE_MANIFEST_COLUMNS`
  - `SKELETON_INPUT_MANIFEST_COLUMNS`

Ý nghĩa:

- Đây là xương sống schema của toàn pipeline.
- Các stage không ngầm hiểu cột; chúng validate schema rõ ràng.

### 6.2.2 `src/slr/data/validation.py`

Vai trò:

- Các validator chung:
  - thiếu cột
  - null ở khóa chính
  - validate schema order
  - validate split values

Ý nghĩa:

- Giúp fail sớm nếu manifest sai định dạng.

### 6.2.3 `src/slr/data/build_index.py`

Trạng thái: implemented, là một trong các file lõi nhất của repo.

Chức năng chính:

- `load_config()` normalize config index.
- `load_master_metadata()` đọc `WLASL_v0.3.json`.
- `load_class_list()` đọc class list.
- `load_nslt_metadata()` đọc tất cả `nslt_*.json`.
- `scan_local_videos()` quét video local thật trên đĩa.
- `build_master_instances()` dựng bảng master có thêm local path và notes.
- `build_available_and_missing()` tách sample có/không có video local.
- `build_nslt_only_instances()` xử lý ID chỉ có trong NSLT mà không có trong master.
- `build_class_map()` ghép class list với master gloss và coverage.
- `build_video_to_split()` tổng hợp split từ master và tất cả subset.
- `build_subset_manifests()` dựng manifest train/val/test cho từng subset.
- `build_reports()` tạo dataset summary và coverage json.
- `write_outputs()` ghi mọi CSV/JSON/MD.

Đánh giá:

- File này không chỉ là scaffold; nó xử lý nhiều edge case thật:
  - mismatch split
  - mismatch class id
  - ID có trong NSLT nhưng không có trong WLASL_v0.3
  - video không có local
  - duplicate video IDs

### 6.2.4 `src/slr/data/standardize_videos.py`

Trạng thái: implemented.

Chức năng chính:

- `load_config()` chuẩn hóa config standardization.
- `load_split_manifest()` đọc manifest index theo split.
- `_resolve_frame_range()` xử lý `start_frame/end_frame`.
- `_resolve_crop_box()` parse bbox, margin, fallback full frame.
- `standardize_one_sample()` chạy toàn bộ xử lý cho một sample.
- `standardize_split()` lặp toàn split, trả manifest và thống kê.
- `build_standardization_report()` viết report markdown.
- `run()` điều phối toàn bộ stage.

Điểm mạnh:

- Có `dry-run`.
- Có status/error rõ ràng cho từng sample.
- Có note logging cho bbox invalid, frame range invalid, fallback.

## 6.3 `src/slr/pose`

### 6.3.1 `src/slr/pose/pose_schema.py`

Vai trò:

- Định nghĩa schema `wholebody_133`.
- Định nghĩa các region indices:
  - body
  - foot
  - face
  - left_hand
  - right_hand
- Định nghĩa mapping:
  - `SELECTED_27`
  - `SELECTED_31`
- Ghi note giải thích mapping.
- Validate shape `(T, V, 3)`.

Điểm quan trọng:

- Đây là file quyết định ý nghĩa của `selected_27` và `selected_31`.
- `selected_49` được khai báo nhưng chưa implement.

### 6.3.2 `src/slr/pose/keypoint_selection.py`

Vai trò:

- Cắt pose 133 điểm xuống tập con theo indices hoặc theo tên keypoint set.
- Dựng payload `.npz` cho selected keypoints.

### 6.3.3 `src/slr/pose/pose_normalization.py`

Vai trò:

- Normalize tọa độ và confidence cho graph model.

Hàm chính:

- `normalize_xy_to_minus1_1()`
- `compute_confidence_scale()`
- `normalize_confidence()`
- `sanitize_non_finite_keypoints()`

Điểm quan trọng:

- Confidence scale được fit trên train split theo percentile, không hard-code.
- Khi scale không hợp lệ sẽ fallback `1.0`.

### 6.3.4 `src/slr/pose/pose_quality.py`

Vai trò:

- Tính mean confidence.
- Tính confidence theo vùng.
- Tính valid frame ratio.
- Tổng hợp pose manifest thành report-friendly summary.

### 6.3.5 `src/slr/pose/extract_rtmw.py`

Trạng thái: implemented.

Đây là file tích hợp thật với MMPose/RTMW-l.

Các khối logic quan trọng:

- `setup_pose_model()`:
  - resolve config/checkpoint file
  - import `torch`, `MMPoseInferencer`
  - fallback device
- `select_primary_person()`:
  - chọn signer chính khi một frame có nhiều người
- `extract_keypoints_from_result()`:
  - normalize output MMPose về `(133, 3)`
- `process_sample()`:
  - đọc standardized frames
  - chạy inferencer
  - tạo empty frame nếu không detect được người
  - tính quality
  - ghi pose `.npz`
- `process_split()`
- `build_pose_quality_report()`

Đánh giá:

- Đây là stage nặng nhất về dependency và runtime.
- Code được viết đủ robust để dùng trên local hoặc Kaggle.

## 6.4 `src/slr/branches/skeleton`

Đây là vùng quan trọng nhất của toàn repo.

### 6.4.1 `src/slr/branches/skeleton/build_inputs.py`

Trạng thái: implemented.

Vai trò:

- Chuyển `wholebody_133 pose` thành input training thực tế.

Các phần quan trọng:

- `load_config()`: resolve subset, keypoint set, output roots, expected tensor shape
- `compute_confidence_scale_from_train()`: fit thang confidence từ split train
- `process_sample()`:
  - load pose file
  - chọn keypoint
  - normalize xy/confidence
  - sanitize values
  - fix sequence length
  - convert sang `(C,T,V,M)`
  - save selected/normalized/graph outputs
- `process_split()`
- `build_report()`
- `run()`

Điểm mạnh:

- Có report rõ về số lượng output selected/normalized/graph tensor.
- Có theo dõi out-of-bounds và non-finite values.

### 6.4.2 `src/slr/branches/skeleton/transforms.py`

Vai trò:

- Xử lý sequence length.
- Chuyển `(T,V,3)` sang `(C,T,V,M)`.

Lưu ý:

- Hiện chỉ hỗ trợ `num_persons = 1`.
- Sequence ngắn được repeat, sequence dài lấy head.

### 6.4.3 `src/slr/branches/skeleton/graph.py`

Vai trò:

- Định nghĩa topology graph cho `selected_27` và `selected_31`.
- Build adjacency ST-GCN style:
  - `uniform`
  - `spatial`
- `SkeletonGraph` là container đóng gói layout, edges, node names, adjacency.

Điểm quan trọng:

- `selected_31` thêm 4 node miệng và 7 cạnh mở rộng.
- Adjacency output có thể là:
  - `(1, V, V)` cho `uniform`
  - `(3, V, V)` cho `spatial`

### 6.4.4 `src/slr/branches/skeleton/dataset.py`

Trạng thái: implemented.

Vai trò:

- Loader cho graph tensor đã precompute.

Các điểm kỹ thuật quan trọng:

- `load_skeleton_train_config()` đọc train config thành dataset config normalized.
- `resolve_graph_tensor_path()` remap path linh hoạt:
  - local project root
  - `data_root`
  - path từ HF/Kaggle/foreign absolute path
- `SkeletonGraphDataset`:
  - lọc `status == ok`
  - lọc theo split
  - validate `tensor_shape`
  - load `.npz` key `data` hoặc fallback key `tensor`
  - trả `torch.Tensor` + label + optional metadata
- `build_label_maps_from_manifest()` dựng `id_to_gloss`, `gloss_to_id`
- `skeleton_collate_fn()` stack batch và metadata

Đây là file rất quan trọng vì nó giải quyết vấn đề portability của manifest path giữa local, Hugging Face bundle và Kaggle bundle.

### 6.4.5 `src/slr/branches/skeleton/label_smoothing.py`

Vai trò:

- Cung cấp helper numpy cho:
  - `standard_label_smoothing`
  - `language_label_smoothing`

Trạng thái thực tế:

- Dùng như helper nghiên cứu.
- Train loop hiện tại không trực tiếp dùng distribution từ đây; standard label smoothing được map sang `CrossEntropyLoss(label_smoothing=epsilon)`.
- `language_label_smoothing()` hiện chưa được nối vào train loop thực tế.

### 6.4.6 `src/slr/branches/skeleton/models/__init__.py`

Vai trò:

- Model factory đang chạy thật.
- Hỗ trợ:
  - `simple_stgcn`
  - `stgcnpp`

### 6.4.7 `src/slr/branches/skeleton/models/simple_stgcn.py`

Trạng thái: implemented.

Vai trò:

- Baseline ST-GCN gọn để kiểm chứng pipeline.

Thành phần:

- `GraphConv2d`
- `STGCNBlock`
- `SimpleSTGCN`

Luồng forward:

1. Input `(N,C,T,V,M)`
2. Gộp person dimension
3. BatchNorm đầu vào
4. Qua chuỗi spatial-temporal graph blocks
5. Global average pooling
6. Linear classifier

### 6.4.8 `src/slr/branches/skeleton/models/stgcnpp.py`

Trạng thái: implemented.

Vai trò:

- ST-GCN++ local, thuần PyTorch, không phụ thuộc MMAction2/PYSKL.

Thành phần:

- `SpatialGraphConv`
- `MultiScaleTemporalConv`
- `STGCNPPBlock`
- `STGCNPP`

Đặc điểm:

- Có `edge_importance` learnable.
- Temporal conv nhiều nhánh.
- Residual blocks.
- Hỗ trợ cả `selected_27` và `selected_31`.

### 6.4.9 `src/slr/branches/skeleton/train.py`

Trạng thái: implemented, là training entrypoint thật của skeleton branch.

Các phần chính:

- Parser train và parser evaluate
- `_normalize_training_config()`
- `apply_cli_overrides()`
- `resolve_training_config()`
- `build_skeleton_datasets()`
- `build_skeleton_dataloaders()`
- `build_graph_and_model()`
- `run_one_epoch_with_shape()`
- `run_training()`
- `run_evaluation()`

Điểm đáng chú ý:

- Validate shape batch trước khi forward.
- Hỗ trợ top-1, top-5, top-10.
- Hỗ trợ optimizer `adamw`, `adam`, `sgd`.
- Hỗ trợ scheduler `cosine`, `step`.
- Hỗ trợ checkpoint đầy đủ.
- Hỗ trợ W&B tùy chọn.
- Dùng `best.pt` để report test metrics, không dùng epoch cuối một cách mù quáng.

## 6.5 `src/slr/training`

Đây là vùng dùng chung, nhưng chỉ một phần đang chạy thật.

| File | Trạng thái | Vai trò |
| --- | --- | --- |
| `src/slr/training/losses.py` | Implemented | Factory cho `cross_entropy` và `standard_label_smoothing` |
| `src/slr/training/metrics.py` | Implemented | `AverageMeter`, `accuracy_topk`, `top_k_accuracy` |
| `src/slr/training/optim.py` | Implemented | Build optimizer/scheduler |
| `src/slr/training/checkpointing.py` | Implemented | Save/load checkpoint bằng `state_dict` |
| `src/slr/training/seed.py` | Implemented | Seed Python/NumPy/PyTorch, optional deterministic |
| `src/slr/training/wandb_utils.py` | Implemented | W&B optional, có guard khi thiếu package/API key/entity |
| `src/slr/training/train.py` | Placeholder | Generic training scaffold, chỉ log config/device/dry-run |
| `src/slr/training/evaluate.py` | Placeholder | Generic eval scaffold, chỉ log config/checkpoint/split |

Nhận xét:

- Tầng helper của `training` dùng được thật cho skeleton branch.
- Nhưng generic entrypoint `train.py` và `evaluate.py` chưa được hiện thực ngoài mức scaffold.

## 6.6 `src/slr/models`

Đây là vùng generic model tree. Hiện phần lớn chưa chạy production.

### 6.6.1 `src/slr/models/skeleton`

| File | Trạng thái | Vai trò |
| --- | --- | --- |
| `stgcnpp.py` | Placeholder | Wrapper ST-GCN++ generic tương lai |
| `ctrgcn.py` | Placeholder | Wrapper CTR-GCN generic tương lai |
| `heads.py` | Placeholder | Classification head generic |

Điểm rất dễ nhầm:

- Các file này không phải model skeleton đang train thật.
- Model thật nằm ở `src/slr/branches/skeleton/models/`.

### 6.6.2 `src/slr/models/regions`

| File | Trạng thái |
| --- | --- |
| `cnn_lstm.py` | Placeholder |
| `video_transformer.py` | Placeholder |
| `heads.py` | Placeholder |

### 6.6.3 `src/slr/models/hand_poseflow`

| File | Trạng thái |
| --- | --- |
| `two_stream.py` | Placeholder |
| `heads.py` | Placeholder |

## 6.7 `src/slr/branches/regions`

Trạng thái chung: placeholder branch.

| File | Vai trò | Trạng thái |
| --- | --- | --- |
| `build_crops.py` | CLI khung để build face/hand crops | Chỉ tạo thư mục và log |
| `dataset.py` | Dataset interface cho region branch | `__len__ = 0`, `__getitem__` raise `NotImplementedError` |
| `transforms.py` | Normalize ảnh `uint8 -> [0,1]` | Helper rất nhỏ |
| `region_schema.py` | `REGION_NAMES = ("face", "left_hand", "right_hand")` | Implemented nhưng tối giản |

## 6.8 `src/slr/branches/hand_poseflow`

Trạng thái chung: placeholder branch.

| File | Vai trò | Trạng thái |
| --- | --- | --- |
| `build_inputs.py` | Orchestrator gọi build hand sequence rồi build poseflow | Có flow khung nhưng downstream vẫn placeholder |
| `build_hand_sequences.py` | Khung build chuỗi ảnh hai bàn tay | Chỉ tạo thư mục và log |
| `build_poseflow.py` | Khung build poseflow | Chỉ tạo thư mục và log |
| `dataset.py` | Dataset interface | Placeholder |
| `poseflow_schema.py` | Schema/constants cho poseflow | Có mặt để định hướng branch |

## 6.9 `src/slr/inference`

Trạng thái chung: placeholder.

| File | Vai trò |
| --- | --- |
| `predict_video.py` | Parser cho suy luận một video |
| `visualize_prediction.py` | Parser cho visualize prediction |

Nhưng hiện cả hai mới log input/output chứ chưa có infer loop thật.

## 7. Phân tích `scripts/`

Thư mục `scripts/` là lớp CLI mỏng đặt trên `src/slr/...`.

### 7.1 Wrapper pipeline chính

| Script | Gọi vào đâu | Trạng thái |
| --- | --- | --- |
| `scripts/00_build_index.py` | `slr.data.build_index.main` | Production |
| `scripts/01_standardize_videos.py` | `slr.data.standardize_videos.main` | Production |
| `scripts/02_extract_pose_rtmw.py` | `slr.pose.extract_rtmw.main` | Production |
| `scripts/03_build_skeleton_inputs.py` | `slr.branches.skeleton.build_inputs.main` | Production |
| `scripts/04_build_region_inputs.py` | `slr.branches.regions.build_crops.main` | Placeholder branch |
| `scripts/05_build_hand_poseflow_inputs.py` | `slr.branches.hand_poseflow.build_inputs.main` | Placeholder branch |
| `scripts/train_skeleton.py` | `slr.branches.skeleton.train.main` | Production |
| `scripts/evaluate_skeleton.py` | `slr.branches.skeleton.train.evaluate_main` | Production |
| `scripts/train_regions.py` | `slr.training.train.main` | Placeholder generic train |
| `scripts/train_hand_poseflow.py` | `slr.training.train.main` | Placeholder generic train |
| `scripts/evaluate.py` | `slr.training.evaluate.main` | Placeholder generic eval |

### 7.2 Script kiểm tra dataset

| Script | Vai trò |
| --- | --- |
| `scripts/check_skeleton_dataset.py` | Sanity-check loader, shape, class id range, graph topology cho manifest skeleton |

Đây là utility tốt để xác minh dữ liệu train-ready trước khi train.

### 7.3 Script merge/bundle

| Script | Vai trò | Đánh giá |
| --- | --- | --- |
| `scripts/merge_nslt300_pose_into_nslt1000.py` | Tái sử dụng pose đã có từ subset nhỏ hơn để dựng subset lớn hơn | Utility thực tế cho mở rộng dữ liệu |
| `scripts/prepare_hf_skeleton_bundle.py` | Đóng gói graph tensor/manifests/reports/logs để upload HF | Script thực dụng, phục vụ chia sẻ dataset train-ready |
| `scripts/prepare_kaggle_bundle.py` | Đóng gói standardized data + repo + checkpoints cho Kaggle pose extraction | Utility deployment |
| `scripts/prepare_kaggle_nslt1000_remaining_bundle.py` | Đóng gói phần còn lại cho bài toán xử lý `nslt1000` | Utility deployment nâng cao |

### 7.4 Script tools

| Script | Vai trò |
| --- | --- |
| `scripts/tools/visualize_pose_sets.py` | Sinh contact sheet so sánh `wholebody_133`, `selected_27`, `selected_31` |
| `scripts/tools/visualize_selected_27_samples.py` | Overlay keypoint reduced set lên frame standardized |
| `scripts/tools/test_rtmw_mmpose_video.py` | Smoke test RTMW-l/MMPose trên một video đơn |

Thư mục con:

- `scripts/tools/_rtmw_mmpose_video_test_output/`

đang chứa output của smoke test:

- `rtmw_test_report.md`
- `rtmw_test_summary.json`
- `predictions/*.json`
- `visualizations/*.jpg/png`

### 7.5 Script import shim

| File | Vai trò |
| --- | --- |
| `scripts/sitecustomize.py` | Thêm `src/` vào `sys.path` khi chạy script từ `scripts/` |
| `scripts/slr/__init__.py` | Shim package tương tự root `slr/__init__.py` |

## 8. Những file có vai trò “source of truth” quan trọng nhất

Nếu cần hiểu repo nhanh nhưng đúng bản chất, nên đọc theo thứ tự:

1. `README.md`
2. `configs/preprocessing/index.yaml`
3. `src/slr/data/build_index.py`
4. `src/slr/data/standardize_videos.py`
5. `src/slr/pose/pose_schema.py`
6. `src/slr/pose/extract_rtmw.py`
7. `configs/branches/skeleton/stgcnpp_27.yaml` hoặc `stgcnpp_31.yaml`
8. `src/slr/branches/skeleton/build_inputs.py`
9. `src/slr/branches/skeleton/dataset.py`
10. `src/slr/branches/skeleton/graph.py`
11. `src/slr/branches/skeleton/models/simple_stgcn.py`
12. `src/slr/branches/skeleton/models/stgcnpp.py`
13. `configs/train/skeleton_selected_27_stgcnpp.yaml` hoặc `skeleton_selected_31_stgcnpp.yaml`
14. `src/slr/branches/skeleton/train.py`
15. `src/slr/training/losses.py`, `optim.py`, `checkpointing.py`, `wandb_utils.py`

## 9. Những điểm dễ nhầm lẫn và cách đọc đúng

### 9.1 Có hai “vùng model skeleton”

- `src/slr/models/skeleton/*`: generic placeholder
- `src/slr/branches/skeleton/models/*`: model đang train thật

Kết luận:

- Khi đọc pipeline chạy thật, bỏ qua `src/slr/models/skeleton/*` trừ khi đang đánh giá hướng mở rộng generic.

### 9.2 Có hai “vùng training”

- `src/slr/training/train.py`, `evaluate.py`: scaffold chung
- `src/slr/branches/skeleton/train.py`: train/eval thật

Kết luận:

- `scripts/train_regions.py` và `scripts/train_hand_poseflow.py` hiện chưa train được branch tương ứng theo nghĩa production-ready.

### 9.3 Có hai tầng config khác nhau

- `configs/branches/skeleton/*.yaml`: build branch input
- `configs/train/*.yaml`: train model

Đây là tách biệt hợp lý:

- preprocessing branch config
- training run config

### 9.4 Standard label smoothing đã “có thật”, LanguageLS thì chưa

- `train.loss = standard_label_smoothing` đã được hỗ trợ trong `losses.py`.
- `language_label_smoothing()` chỉ tồn tại như helper numpy, chưa thấy được nối vào `run_training()`.

## 10. Đánh giá thiết kế tổng thể

### 10.1 Điểm mạnh

1. Repo tổ chức theo pipeline layer rất rõ: `raw -> index -> standardized -> pose -> branch_inputs -> training`.
2. Manifest schema được chuẩn hóa chặt.
3. Skeleton branch có đủ preprocessing, train, eval và bundle.
4. Dataset loader đã giải bài toán remap path giữa local, Kaggle và HF.
5. Training loop có mức engineering tốt cho repo nghiên cứu:
   - config-driven
   - checkpoint rõ
   - metrics top-k
   - optional W&B
   - test bằng best checkpoint

### 10.2 Điểm yếu / khoảng trống

1. `regions` và `hand_poseflow` mới dừng ở mức scaffold.
2. `src/slr/models/*` generic tree dễ gây hiểu nhầm vì cùng tên với branch-specific model tree nhưng chưa dùng.
3. `configs/experiments/skeleton_nslt100_debug.yaml` phản ánh một phong cách orchestration khác, hiện không phải đường chạy chính.
4. `selected_49` đã được khai báo trong schema nhưng chưa implement.
5. Inference sau huấn luyện chưa có pipeline hoàn chỉnh.
6. Notebook chưa được lấp đầy nội dung phân tích.

### 10.3 Mức trưởng thành của repo

Đánh giá thực tế:

- Ở mức “research pipeline đang hoạt động tốt cho một nhánh chính”.
- Chưa phải multi-branch framework hoàn chỉnh.
- Rất phù hợp để làm luận văn/thesis nếu skeleton branch là trục chính.
- Cần cẩn thận khi viết báo cáo học thuật: phải nói rõ `regions` và `hand_poseflow` hiện là kiến trúc dự kiến, không phải thực nghiệm hoàn thiện ngang skeleton branch.

## 11. Gợi ý cách mô tả repo trong luận văn

Cách mô tả chính xác nhất ở thời điểm `2026-05-31`:

> Hệ thống được tổ chức thành pipeline nhiều tầng trên WLASL, trong đó nhánh skeleton là nhánh triển khai hoàn chỉnh nhất. Dữ liệu video thô được lập chỉ mục, chuẩn hóa, trích xuất pose RTMW-l toàn thân, rút gọn thành các bộ keypoint `selected_27` hoặc `selected_31`, chuyển sang graph tensor để huấn luyện các mô hình graph-based như baseline ST-GCN và ST-GCN++ cài đặt thuần PyTorch. Các nhánh `regions` và `hand_poseflow` đã có scaffold cấu hình và mã khởi tạo nhưng chưa hoàn thiện end-to-end như nhánh skeleton.

## 12. Tóm tắt cuối cùng

Từ góc nhìn kỹ thuật, repo hiện có ba lớp rất rõ:

1. Lớp chạy thật cho skeleton branch:
   - `src/slr/data/*`
   - `src/slr/pose/*`
   - `src/slr/branches/skeleton/*`
   - một phần `src/slr/training/*`
   - `scripts/00..03`, `train_skeleton.py`, `evaluate_skeleton.py`

2. Lớp hạ tầng hỗ trợ nghiên cứu:
   - `utils`
   - `reports`
   - `docs`
   - `bundle scripts`
   - `visualization tools`

3. Lớp mở rộng cho tương lai:
   - `regions`
   - `hand_poseflow`
   - `generic models`
   - `generic training/eval`
   - `inference`

Nếu mục tiêu của bạn là hiểu “repo này hiện đang làm được gì”, câu trả lời ngắn nhất là:

- Repo đã có pipeline skeleton hoàn chỉnh từ WLASL raw đến train/eval graph-based SLR.
- Phần còn lại chủ yếu là bộ khung để mở rộng sau.

