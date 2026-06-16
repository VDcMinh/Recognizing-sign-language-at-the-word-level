# NEW LATEST INFO

## 1. Mục tiêu tài liệu

Tài liệu này là bản đọc hiểu toàn bộ workspace tại thời điểm hiện tại, nhằm trả lời 4 câu hỏi:

1. Dự án này thực chất dùng để làm gì.
2. Cấu trúc thư mục hiện tại đang được tổ chức ra sao.
3. File nào là mã chạy thật, file nào là scaffold, file nào là dữ liệu sinh ra.
4. Toàn bộ pipeline đang hoạt động end-to-end như thế nào.

Tài liệu này đọc toàn bộ folder dự án theo hai mức:

- Mức file nguồn/tài liệu/config: đọc trực tiếp từng file văn bản, mã nguồn, YAML, Markdown, notebook metadata.
- Mức artifact khối lượng lớn: đọc theo pattern thư mục và file đại diện, vì workspace chứa rất nhiều media/frame/npz/output đã sinh ra.

Lưu ý quan trọng:

- `.venv-rtmw310/` là môi trường cài package cục bộ, không phải logic của dự án.
- `data/` có khoảng `689120` file và `11045` thư mục, phần lớn là frame/video/npz sinh ra; không có giá trị khi liệt kê từng file media riêng lẻ, nên cần phân tích theo lớp dữ liệu.
- `kaggle_bundle/repo/` là bản sao đóng gói của repo cho Kaggle, không phải source of truth cần sửa trực tiếp.

## 2. Kết luận ngắn

Đây không còn là repo scaffold thuần túy.

Trạng thái hiện tại của dự án:

- Nhánh `skeleton` đã có pipeline chạy thật từ `index -> standardize -> pose -> skeleton inputs -> train/eval`.
- Nhánh `regions` và `hand_poseflow` chủ yếu vẫn là scaffold.
- Huấn luyện thực tế hiện đi qua `src/slr/branches/skeleton/train.py`, không đi qua scaffold chung `src/slr/training/train.py`.
- Bộ mô hình dùng thật hiện nằm trong `src/slr/branches/skeleton/models/`, trong khi `src/slr/models/` phần lớn vẫn là placeholder API.
- Repo đã có artifact train-ready cho `nslt100` và `nslt300`, cùng bundle dành cho Hugging Face và Kaggle.

Nói ngắn gọn: xương sống hiện tại của repo là pipeline skeleton dựa trên RTMW-l wholebody pose, reduced keypoints (`selected_27`, `selected_31`), graph tensors và mô hình `SimpleSTGCN` hoặc `STGCNPP`.

## 3. Ảnh chụp workspace hiện tại

### 3.1 Số lượng file/thư mục theo nhóm lớn

| Khu vực | Trạng thái |
| --- | --- |
| `configs/` | 32 file cấu hình |
| `scripts/` | 41 file wrapper/tool |
| `src/` | 167 file trong package nguồn |
| `docs/` | 3 file docs kỹ thuật |
| `reports/` | 7 file report mức repo |
| `outputs/` | 68 file artifact huấn luyện |
| `data/` | 689120 file artifact và dữ liệu |
| `hf_bundle/` | 7 file bundle nslt100 |
| `hf_sub300_bundle/` | 7 file bundle nslt300 |
| `kaggle_bundle/` | 95 file bundle Kaggle nslt100 |
| `kaggle_sub300_bundle/` | 6 file bundle Kaggle nslt300 |

### 3.2 Dữ liệu raw WLASL cục bộ

Theo `data/datasets/WLASL/raw/docs/README.md`:

- Vocabulary gốc: `2000` gloss.
- Master instances: `21083`.
- Video `.mp4` có sẵn local: `11980`.
- Video thiếu local: `9103`.
- Signer duy nhất: `119`.
- Nguồn video: `19`.

Độ phủ local theo split:

| Split | Tổng | Có video local | Thiếu | Độ phủ |
| --- | ---: | ---: | ---: | ---: |
| train | 14289 | 8313 | 5976 | 58.2% |
| val | 3916 | 2253 | 1663 | 57.5% |
| test | 2878 | 1414 | 1464 | 49.1% |

### 3.3 Các lớp dữ liệu đã sinh ra

Theo trạng thái thư mục hiện có:

- `index/`: đã có manifest và report cho `nslt100`, `nslt300`, `nslt1000`, `nslt2000`.
- `standardized/`: đã có manifest cho `nslt100`, `nslt300`, `nslt1000`.
- `pose/rtmw_l/`: hiện có manifest/report cho `nslt300`.
- `branch_inputs/skeleton/rtmw_l/`: hiện có manifest/report cho `nslt100` và `nslt300`.

Một số mốc cụ thể:

- `nslt100` skeleton train-ready: `1013` samples, `100` classes.
- `nslt300` standardized: `2660` samples, `155942` frames, `0` lỗi.
- `nslt300` pose: `2660` samples, `2660` status=`ok`, valid frame ratio trung bình `1.0`.
- `nslt300` skeleton `selected_31`: `2660` graph tensors, shape `(3, 150, 31, 1)`, `0` lỗi.

### 3.4 Các run huấn luyện hiện có

Hiện có các run trong `outputs/skeleton/`:

- `smoke-ce-after-standardls`
- `smoke-sel27`
- `smoke-sel27-standardls-eps005`
- `smoke-sel27-standardls-eps01`
- `smoke-sel27-standardls-eps03`
- `smoke-sel27-stgcnpp`
- `smoke-sel31`
- `smoke-sel31-standardls-eps005`
- `smoke-sel31-stgcnpp`
- `smoke-standardls-eps01`
- `smoke-standardls-eps03`

Snapshot metric:

| Run | test_top1 | test_top5 | test_loss |
| --- | ---: | ---: | ---: |
| `smoke-sel27` | 0.1250 | 0.3125 | 4.5581 |
| `smoke-sel31` | 0.0625 | 0.3125 | 4.5606 |
| `smoke-sel27-stgcnpp` | 0.0625 | 0.1875 | 4.4996 |
| `smoke-sel31-stgcnpp` | 0.0625 | 0.2500 | 4.4973 |
| `smoke-sel27-standardls-eps005` | 0.0625 | 0.2500 | 4.5273 |
| `smoke-sel27-standardls-eps01` | 0.0625 | 0.2500 | 4.5369 |
| `smoke-sel27-standardls-eps03` | 0.0625 | 0.2500 | 4.5665 |
| `smoke-sel31-standardls-eps005` | 0.0625 | 0.2500 | 4.5315 |

## 4. Bản đồ dự án

```text
Recognizing-sign-language-at-the-word-level/
├── checkpoints/                  # checkpoint pose và placeholder model dirs
├── configs/                      # toàn bộ YAML cho preprocessing, branch, train
├── data/                         # raw data + derived data nhiều tầng
├── docs/                         # docs kỹ thuật ngắn
├── experiments/                  # khung chứa output thí nghiệm, hiện mới có .gitkeep
├── hf_bundle/                    # bundle HF nslt100
├── hf_sub300_bundle/             # bundle HF nslt300
├── kaggle_bundle/                # bundle Kaggle nslt100
├── kaggle_sub300_bundle/         # bundle Kaggle nslt300
├── notebooks/                    # notebook khám phá, hiện chỉ có heading
├── outputs/                      # artifact huấn luyện skeleton
├── reports/                      # report preprocessing/training cấp repo
├── scripts/                      # wrapper CLI và tool scripts
├── slr/                          # import shim
├── src/slr/                      # package nguồn thực tế
├── README.md
├── PROJECT_STRUCTURE_GUIDE.md
├── TRAINING.md
├── IMPLEMENTATION.md
├── StandardLS.md
├── LATEST_INFO.md
└── NEW_LATEST_INFO.md
```

## 5. Cách hệ thống hoạt động end-to-end

Pipeline thực tế:

```text
WLASL raw metadata + raw videos
-> build index manifests
-> standardize frames/videos
-> extract RTMW-l wholebody_133 pose
-> build reduced skeleton inputs
-> train / evaluate skeleton graph model
```

Chi tiết từng tầng:

1. `index`
   - Đọc `WLASL_v0.3.json`, `wlasl_class_list.txt`, `missing.txt`, `nslt_*.json`, video local.
   - Sinh bảng master, available/missing, class map, video_to_split, subset manifests.

2. `standardized`
   - Đọc manifest subset có video thật.
   - Crop theo bbox nếu hợp lệ, resize về `288x384`, letterbox, xuất frame/video chuẩn hóa.

3. `pose/rtmw_l`
   - Đọc standardized frames.
   - Chạy RTMW-l qua MMPose.
   - Sinh `.npz` pose dạng `(T, 133, 3)` và pose quality report.

4. `branch_inputs/skeleton`
   - Chọn reduced keypoints (`selected_27` hoặc `selected_31`).
   - Chuẩn hóa `x, y` về `[-1, 1]`.
   - Chuẩn hóa confidence theo percentile trên split train.
   - Cố định độ dài sequence về `150`.
   - Đổi sang tensor `C x T x V x M`.

5. `training/evaluation`
   - Dataset đọc manifest và graph tensor `.npz`.
   - Graph builder tạo adjacency từ layout `selected_27` hoặc `selected_31`.
   - Model builder tạo `SimpleSTGCN` hoặc `STGCNPP`.
   - Train loop lưu `best.pt`, `last.pt`, `metrics.json`, `summary.json`, `train_log.csv`.
   - Sau train, nạp lại `best.pt` rồi evaluate trên split test.

## 6. Phân tích chi tiết file theo nhóm

## 6.1 File gốc ở root

| File | Vai trò | Nhận xét |
| --- | --- | --- |
| `README.md` | giới thiệu repo | Nêu mục tiêu, dataset, pipeline chuẩn, schema dữ liệu và roadmap. |
| `PROJECT_STRUCTURE_GUIDE.md` | tài liệu kiến trúc | Mô tả cấu trúc nhiều tầng của repo; một số đoạn phản ánh giai đoạn scaffold cũ hơn trạng thái code hiện tại. |
| `TRAINING.md` | training log tài liệu hóa | Ghi rất chi tiết về training skeleton baseline, smoke runs, output, checkpoint, W&B. |
| `IMPLEMENTATION.md` | implementation log | Nhật ký tích hợp `STGCNPP` repo-local, validation commands và giới hạn hiện tại. |
| `StandardLS.md` | implementation log | Nhật ký tích hợp Standard Label Smoothing vào pipeline skeleton. |
| `LATEST_INFO.md` | snapshot tài liệu trước | Đã tổng hợp khá tốt trạng thái hiện tại; `NEW_LATEST_INFO.md` mở rộng và cập nhật lại theo cùng tinh thần nhưng rõ hơn chuyện production path/scaffold/artifact. |
| `pyproject.toml` | package metadata | Dùng `src` layout cho package `slr`, dependencies để trống. |
| `requirements.txt` | dependency nhẹ | Dành cho preprocessing/utilities: `numpy`, `pandas`, `opencv-python`, `pyyaml`, `tqdm`, `scikit-learn`, `matplotlib`, `wandb`. |
| `requirements-rtmw.txt` | dependency nặng | Dành cho RTMW-l/MMPose: `torch`, `torchvision`, `mmengine`, `mmcv`, `mmdet`, `mmpose`, `xtcocotools`, `rich`, `wandb`. |
| `requirements-kaggle-train.txt` | dependency tối thiểu | Dành cho môi trường train/bundle phía Kaggle/HF. |
| `.gitignore` | chính sách theo dõi file | Bỏ qua video raw, tầng dữ liệu sinh ra, outputs, bundle zip, checkpoint tải về và cache local. |
| `sitecustomize.py` | import helper | Tự thêm `src/` vào `sys.path`. |
| `slr/__init__.py` | import shim | Giúp `import slr` chạy được dù chưa cài editable package. |

## 6.2 `configs/`

### 6.2.1 `configs/dataset/`

| File | Vai trò | Nhận xét |
| --- | --- | --- |
| `configs/dataset/wlasl.yaml` | dataset root config | Chỉ ra `raw_root`, `metadata_dir`, `videos_dir` và config dataset mức nền. |

### 6.2.2 `configs/preprocessing/`

| File | Vai trò | Nhận xét |
| --- | --- | --- |
| `configs/preprocessing/index.yaml` | config build index | Khai báo metadata master, class list, `nslt_*.json`, output root, zero-padding video_id. |
| `configs/preprocessing/standardize.yaml` | config chuẩn hóa nslt100 | Resize `288x384`, crop theo bbox, save frames, không save video mặc định. |
| `configs/preprocessing/standardize_nslt300.yaml` | config chuẩn hóa nslt300 | Biến thể theo subset `nslt300`. |
| `configs/preprocessing/standardize_nslt1000.yaml` | config chuẩn hóa nslt1000 | Biến thể theo subset `nslt1000`. |
| `configs/preprocessing/pose_rtmw_l.yaml` | config pose local | Dùng standardized manifests, RTMW-l, `wholebody_133`, `cuda:0` rồi fallback CPU. |
| `configs/preprocessing/pose_rtmw_l_kaggle.yaml` | config pose Kaggle | Cùng logic nhưng phục vụ bundle/notebook Kaggle. |
| `configs/preprocessing/region_crops.yaml` | config scaffold | Dành cho region branch, chưa có pipeline thực tế tương ứng. |
| `configs/preprocessing/poseflow.yaml` | config scaffold | Dành cho hand_poseflow branch, chưa có pipeline thực tế tương ứng. |

### 6.2.3 `configs/branches/skeleton/`

| File | Vai trò | Nhận xét |
| --- | --- | --- |
| `stgcnpp_27.yaml` | config build skeleton input | Chỉ ra pose manifest, output selected/normalized/graph tensor cho `selected_27`. |
| `stgcnpp_31.yaml` | config build skeleton input | Tương tự cho `selected_31`; đây là config quan trọng trong preprocessing skeleton. |
| `stgcnpp_27_nslt300.yaml` | build input nslt300 | Biến thể cho subset `nslt300`. |
| `stgcnpp_31_nslt300.yaml` | build input nslt300 | Biến thể cho subset `nslt300`. |
| `ctrgcn_31.yaml` | ý định CTR-GCN | Chỉ là config định hướng; code chạy thực chưa có CTR-GCN thật ở production path. |
| `stgcnpp_31_languagels.yaml` | ý định LanguageLS | Định hướng loss ngôn ngữ; production train loop chưa nối LanguageLS. |

### 6.2.4 `configs/branches/regions/` và `configs/branches/hand_poseflow/`

| File | Vai trò | Nhận xét |
| --- | --- | --- |
| `configs/branches/regions/face_hands_baseline.yaml` | baseline scaffold | Định nghĩa input/model/training cho regions branch nhưng branch chưa được hiện thực. |
| `configs/branches/hand_poseflow/hand_poseflow_baseline.yaml` | baseline scaffold | Tương tự cho hand poseflow. |

### 6.2.5 `configs/train/`

| File | Vai trò | Nhận xét |
| --- | --- | --- |
| `skeleton_selected_27.yaml` | config loader cơ bản | Chủ yếu chứa `dataset`, `dataloader`, `graph`; dùng cho sanity check và dữ liệu. |
| `skeleton_selected_31.yaml` | config loader cơ bản | Tương tự cho `selected_31`. |
| `skeleton_selected_27_baseline.yaml` | train baseline | Dùng `SimpleSTGCN` cho `nslt100`. |
| `skeleton_selected_31_baseline.yaml` | train baseline | Dùng `SimpleSTGCN` cho `nslt100`. |
| `skeleton_selected_27_stgcnpp.yaml` | train ST-GCN++ | Dùng `STGCNPP`, `SGD`, cosine scheduler, `nslt100`. |
| `skeleton_selected_31_stgcnpp.yaml` | train ST-GCN++ | Config production quan trọng cho `selected_31`. |
| `skeleton_selected_27_stgcnpp_standardls_eps005.yaml` | StandardLS | `epsilon=0.05`. |
| `skeleton_selected_27_stgcnpp_standardls_eps01.yaml` | StandardLS | `epsilon=0.1`. |
| `skeleton_selected_27_stgcnpp_standardls_eps03.yaml` | StandardLS | `epsilon=0.3`. |
| `skeleton_selected_31_stgcnpp_standardls_eps005.yaml` | StandardLS | `epsilon=0.05`. |
| `skeleton_selected_31_stgcnpp_standardls_eps01.yaml` | StandardLS | `epsilon=0.1`. |
| `skeleton_selected_31_stgcnpp_standardls_eps03.yaml` | StandardLS | `epsilon=0.3`. |
| `skeleton_selected_27_nslt300_stgcnpp.yaml` | train nslt300 | Train ST-GCN++ cho `nslt300`, `selected_27`, `300` classes. |
| `skeleton_selected_31_nslt300_stgcnpp.yaml` | train nslt300 | Train ST-GCN++ cho `nslt300`, `selected_31`, `300` classes. |

### 6.2.6 `configs/experiments/`

| File | Vai trò | Nhận xét |
| --- | --- | --- |
| `configs/experiments/skeleton_nslt100_debug.yaml` | experiment-level config | Hữu ích cho tư duy tổ chức thí nghiệm, nhưng production train hiện gọi trực tiếp config ở `configs/train/`. |

## 6.3 `src/slr/data/`

| File | Vai trò | Nhận xét |
| --- | --- | --- |
| `src/slr/data/manifests.py` | schema trung tâm | Khai báo chuẩn cột cho `master`, `subset`, `standardized`, `pose`, `skeleton input`. Đây là file nền để mọi stage ghi CSV nhất quán. |
| `src/slr/data/validation.py` | validation helper | Kiểm tra cột bắt buộc, null key fields, thứ tự schema và split values. |
| `src/slr/data/build_index.py` | stage build index thật | Đây là mã production của tầng index: parse metadata master, class list, missing IDs, NSLT splits, scan video local, build report và ghi nhiều CSV/JSON/MD. |
| `src/slr/data/standardize_videos.py` | stage standardize thật | Đọc split manifest, resolve frame range/bbox, đọc video, crop + resize + letterbox, ghi frames/video và standardized manifest/report. Có xử lý lỗi khá đầy đủ. |

## 6.4 `src/slr/pose/`

| File | Vai trò | Nhận xét |
| --- | --- | --- |
| `src/slr/pose/pose_schema.py` | schema keypoint | Định nghĩa `wholebody_133`, region indices, reduced sets `selected_27`, `selected_31`, note mapping và validation. `selected_49` chưa implement. |
| `src/slr/pose/keypoint_selection.py` | chọn reduced keypoints | Slice từ 133 keypoints xuống reduced set; tạo payload `.npz` ổn định kèm metadata. |
| `src/slr/pose/pose_normalization.py` | normalize tọa độ/confidence | Chuẩn hóa `x, y` về `[-1,1]`, fit confidence scale, normalize confidence, sanitize non-finite. |
| `src/slr/pose/pose_quality.py` | quality metrics | Tính mean confidence, valid frame ratio, confidence theo vùng body/face/hands và summarize manifest. |
| `src/slr/pose/extract_rtmw.py` | stage pose thật | Tải config/checkpoint RTMW-l, gọi `MMPoseInferencer`, chọn người ký chính, ghi `.npz` pose, manifest và quality report. Đây là production path của tầng pose. |

## 6.5 `src/slr/branches/skeleton/`

| File | Vai trò | Nhận xét |
| --- | --- | --- |
| `src/slr/branches/skeleton/build_inputs.py` | stage build skeleton input thật | Cầu nối từ shared pose sang train-ready tensors: load pose manifest, fit confidence scale trên train, chọn keypoints, normalize, fix sequence length, xuất selected/normalized/graph_tensor, ghi manifest/report. |
| `src/slr/branches/skeleton/dataset.py` | dataset loader production | File rất quan trọng. Đọc manifest, lọc `status=ok`, resolve `graph_tensor_path`, kiểm tra shape, build label maps, trả tensor + metadata. Hỗ trợ remap path giữa local/HF/Kaggle. |
| `src/slr/branches/skeleton/graph.py` | graph topology thật | Định nghĩa node names, edges, adjacency cho `selected_27` và `selected_31`, hỗ trợ `uniform` và `spatial`, normalize adjacency. |
| `src/slr/branches/skeleton/transforms.py` | transform sequence | Cố định sequence về `150`, hỗ trợ short=`repeat`, long=`head`, đổi `(T,V,3)` sang `CTVM`. |
| `src/slr/branches/skeleton/label_smoothing.py` | helper smoothing | Có cả standard LS và language-aware LS ở mức helper numpy. Dùng thật hiện tại mới có StandardLS ở loss torch. |
| `src/slr/branches/skeleton/models/__init__.py` | model factory | Build model theo `model.name`, hiện hỗ trợ `simple_stgcn` và `stgcnpp`. |
| `src/slr/branches/skeleton/models/simple_stgcn.py` | baseline model thật | Baseline PyTorch gọn nhẹ gồm `GraphConv2d`, `STGCNBlock`, `SimpleSTGCN`. Dùng để xác thực toàn bộ training path. |
| `src/slr/branches/skeleton/models/stgcnpp.py` | ST-GCN++ thật | Clean-room implementation repo-local, không phụ thuộc `mmaction2/pyskl`; gồm `SpatialGraphConv`, `MultiScaleTemporalConv`, `STGCNPPBlock`, `STGCNPP`. |
| `src/slr/branches/skeleton/train.py` | production train/eval path | File quan trọng nhất hiện tại. Chứa parser, config normalization, CLI override, dataset/dataloader build, graph/model build, train loop, validation, checkpointing, final test eval, output writing, W&B integration. |
| `src/slr/branches/skeleton/__init__.py` | package export | Export branch-level API cơ bản. |

## 6.6 `src/slr/training/`

| File | Vai trò | Nhận xét |
| --- | --- | --- |
| `src/slr/training/metrics.py` | metric utilities | Có `AverageMeter`, `accuracy_topk`, `top_k_accuracy`; dùng thật trong train loop. |
| `src/slr/training/losses.py` | loss factory | Hỗ trợ `cross_entropy` và `standard_label_smoothing` thông qua `torch.nn.CrossEntropyLoss(label_smoothing=...)`. |
| `src/slr/training/optim.py` | optimizer/scheduler factory | Hỗ trợ `adamw`, `adam`, `sgd`, và scheduler `cosine`, `step`. |
| `src/slr/training/checkpointing.py` | checkpoint utilities | Lưu và nạp `state_dict` cùng metadata như `model_name`, `num_nodes`, `class_id_to_gloss`, `config`. |
| `src/slr/training/seed.py` | seed cho training | Set seed cho Python, NumPy, PyTorch và deterministic mode. |
| `src/slr/training/wandb_utils.py` | W&B optional integration | Resolve entity từ CLI/config/env, disable mềm khi thiếu `wandb` hay `WANDB_API_KEY`, log metrics/artifact. |
| `src/slr/training/train.py` | generic scaffold | Chỉ log plan, chưa có training loop thật. Không phải đường chạy chính hiện tại. |
| `src/slr/training/evaluate.py` | generic scaffold | Chỉ log plan, chưa có evaluation loop thật. |

## 6.7 `src/slr/utils/`

| File | Vai trò | Nhận xét |
| --- | --- | --- |
| `src/slr/utils/io.py` | I/O helper | Đọc/ghi JSON, YAML, CSV, text; tạo thư mục cha trước khi ghi. |
| `src/slr/utils/logging.py` | logger helper | Tạo logger stream/file nhất quán cho script. |
| `src/slr/utils/video.py` | video helper | Probe metadata video, đọc frame, ghi video từ frames. |
| `src/slr/utils/image.py` | image helper | Đọc ảnh, ensure kiểu `uint8`, resize letterbox, save ảnh. |
| `src/slr/utils/bbox.py` | bbox helper | Parse, expand, clip, serialize bounding box. |
| `src/slr/utils/seed.py` | seed helper đơn giản | Bản nhẹ hơn so với `training/seed.py`. |

## 6.8 `src/slr/branches/regions/`

Trạng thái chung: scaffold.

| File | Vai trò | Nhận xét |
| --- | --- | --- |
| `build_crops.py` | CLI scaffold | Tạo parser, log input/output, tạo folder khung, chưa crop thật. |
| `dataset.py` | dataset placeholder | `RegionSequenceDataset` chưa có dữ liệu thật. |
| `region_schema.py` | constants | Khai báo `face`, `left_hand`, `right_hand`. |
| `transforms.py` | helper tối thiểu | Mới có normalize ảnh `uint8 -> [0,1]`. |

## 6.9 `src/slr/branches/hand_poseflow/`

Trạng thái chung: scaffold.

| File | Vai trò | Nhận xét |
| --- | --- | --- |
| `build_hand_sequences.py` | CLI scaffold | Mới tạo khung output cho hand sequences. |
| `build_poseflow.py` | CLI scaffold | Mới tạo khung output cho poseflow. |
| `build_inputs.py` | orchestrator scaffold | Gọi hai bước scaffold ở trên. |
| `dataset.py` | dataset placeholder | `HandPoseFlowDataset` chưa load dữ liệu thật. |
| `poseflow_schema.py` | constants | Khai báo variants như `selected_31`, `hands_only`. |

## 6.10 `src/slr/models/`

Nhóm này hiện chủ yếu là API mức cao hoặc placeholder, không phải nơi production path đang build model.

| File | Vai trò | Nhận xét |
| --- | --- | --- |
| `src/slr/models/skeleton/stgcnpp.py` | placeholder | Chỉ giữ args/kwargs, `forward()` ném `NotImplementedError`. |
| `src/slr/models/skeleton/ctrgcn.py` | placeholder | Tương tự, chưa có CTR-GCN thật. |
| `src/slr/models/skeleton/heads.py` | placeholder head | Khung phân loại mức API. |
| `src/slr/models/regions/cnn_lstm.py` | placeholder | `forward()` chưa triển khai. |
| `src/slr/models/regions/video_transformer.py` | placeholder | `forward()` chưa triển khai. |
| `src/slr/models/regions/heads.py` | placeholder | Head mức khung. |
| `src/slr/models/hand_poseflow/two_stream.py` | placeholder | Hai luồng chưa được hiện thực. |
| `src/slr/models/hand_poseflow/heads.py` | placeholder | Head mức khung. |

Nhận xét quan trọng:

- `src/slr/branches/skeleton/models/*` mới là model chạy thật.
- `src/slr/models/*` dễ gây nhầm vì tên “model” có vẻ canonical hơn nhưng thực tế chưa phải production path.

## 6.11 `src/slr/inference/`

| File | Vai trò | Nhận xét |
| --- | --- | --- |
| `predict_video.py` | inference CLI placeholder | Chỉ parse args rồi log config/video/checkpoint. |
| `visualize_prediction.py` | visualization CLI placeholder | Chỉ log input/output. |

## 6.12 `scripts/`

### 6.12.1 Wrapper scripts

| File | Vai trò | Nhận xét |
| --- | --- | --- |
| `scripts/00_build_index.py` | wrapper | Gọi `slr.data.build_index.main`. |
| `scripts/01_standardize_videos.py` | wrapper | Gọi `slr.data.standardize_videos.main`. |
| `scripts/02_extract_pose_rtmw.py` | wrapper | Gọi `slr.pose.extract_rtmw.main`. |
| `scripts/03_build_skeleton_inputs.py` | wrapper | Gọi `slr.branches.skeleton.build_inputs.main`. |
| `scripts/04_build_region_inputs.py` | wrapper | Gọi scaffold regions branch. |
| `scripts/05_build_hand_poseflow_inputs.py` | wrapper | Gọi scaffold hand_poseflow branch. |
| `scripts/train_skeleton.py` | wrapper production | Gọi `slr.branches.skeleton.train.main`. |
| `scripts/evaluate_skeleton.py` | wrapper production | Gọi `slr.branches.skeleton.train.evaluate_main`. |
| `scripts/train_regions.py` | wrapper scaffold | Trỏ về generic `slr.training.train.main`. |
| `scripts/train_hand_poseflow.py` | wrapper scaffold | Trỏ về generic `slr.training.train.main`. |
| `scripts/evaluate.py` | wrapper scaffold | Trỏ về generic `slr.training.evaluate.main`. |
| `scripts/sitecustomize.py` | import helper | Giống root `sitecustomize.py`, dành cho wrapper scripts. |
| `scripts/slr/__init__.py` | shim | Giúp import package từ thư mục script. |

### 6.12.2 Utility / tool scripts

| File | Vai trò | Nhận xét |
| --- | --- | --- |
| `scripts/check_skeleton_dataset.py` | sanity check | Kiểm tra dataset loader, sample shape, batch shape, class_id range, graph adjacency. Rất hữu ích trước khi train. |
| `scripts/prepare_hf_skeleton_bundle.py` | bundle HF | Validate chặt dữ liệu skeleton train-ready rồi đóng gói `graph_tensors`, `manifests`, `reports`, `logs` cho Hugging Face. |
| `scripts/prepare_kaggle_bundle.py` | bundle Kaggle | Đóng gói standardized frames + repo subset + RTMW-l checkpoint cho Kaggle pose extraction; có guard an toàn khá tốt về path và file cần loại trừ. |
| `scripts/tools/test_rtmw_mmpose_video.py` | smoke test MMPose | Chạy RTMW-l trên 1 video, ghi summary JSON và markdown report. |
| `scripts/tools/visualize_selected_27_samples.py` | visualization tool | Overlay reduced keypoints lên standardized frames, sinh contact sheet + report; thực tế hỗ trợ cả `selected_27` và `selected_31`. |

## 6.13 `docs/`

| File | Vai trò | Nhận xét |
| --- | --- | --- |
| `docs/skeleton_training_baseline.md` | quick-start doc | Hướng dẫn ngắn cho baseline training/eval skeleton. |
| `docs/skeleton_stgcnpp_integration.md` | ST-GCN++ doc | Mô tả integration ST-GCN++, input format, command, limitation. |
| `docs/standard_label_smoothing.md` | StandardLS doc | Mô tả configs, command smoke/full run và comparison plan. |

## 6.14 `reports/`

| File | Vai trò | Nhận xét |
| --- | --- | --- |
| `reports/preprocessing/README.md` | mô tả lớp preprocessing | Tóm tắt `raw/index/standardized/pose/branch_inputs`. |
| `reports/preprocessing/nslt300_index_and_standardization_report.md` | report subset | Mô tả kết quả index + standardize `nslt300`. |
| `reports/preprocessing/nslt1000_index_and_standardization_report.md` | report subset | Mô tả kết quả index + standardize `nslt1000`. |
| `reports/preprocessing/kaggle_sub300_bundle_report.md` | report bundle Kaggle | Giải thích nội dung gói Kaggle `nslt300`. |
| `reports/preprocessing/hf_sub300_bundle_report.md` | report bundle HF | Giải thích bundle skeleton `nslt300` cho HF. |
| `reports/training/nslt300_training_config_report.md` | report config train | Chứng minh `nslt300` config tương thích source code và bundle layout. |
| `reports/experiments/README.md` | guideline experiments | Quy ước artifact thí nghiệm nên chứa gì. |

## 6.15 `notebooks/`

| File | Vai trò | Nhận xét |
| --- | --- | --- |
| `notebooks/01_explore_wlasl.ipynb` | notebook placeholder | Hiện chỉ có heading `# Explore WLASL`, chưa có cell code thực. |
| `notebooks/02_check_pose_quality.ipynb` | notebook placeholder | Hiện chỉ có heading `# Check Pose Quality`. |
| `notebooks/03_visualize_keypoints.ipynb` | notebook placeholder | Hiện chỉ có heading `# Visualize Keypoints`. |

## 6.16 `data/`

Không nên hiểu `data/` như “một thư mục file”, mà phải hiểu như “các tầng dữ liệu”.

### `data/datasets/WLASL/raw/`

| Thành phần | Vai trò | Nhận xét |
| --- | --- | --- |
| `metadata/WLASL_v0.3.json` | master metadata | Source of truth giàu thông tin nhất: `gloss`, `split`, `bbox`, `signer_id`, `source`, `frame_start/end`. |
| `metadata/nslt_100.json` | split classification | Entry point nhanh cho bài toán 100 classes. |
| `metadata/nslt_300.json` | split classification | Entry point nhanh cho bài toán 300 classes. |
| `metadata/nslt_1000.json` | split classification | Entry point nhanh cho bài toán 1000 classes. |
| `metadata/nslt_2000.json` | split classification | Entry point nhanh cho full 2000 classes, nhưng có 12 ID ngoài master manifest. |
| `metadata/wlasl_class_list.txt` | class map | Ánh xạ `class_id -> gloss`. |
| `metadata/wlasl_class_list_corrected.txt` | corrected class list | Bản sửa tay để tham khảo. |
| `metadata/missing.txt` | missing IDs | Danh sách video ID có trong metadata nhưng thiếu file local. |
| `docs/README.md` | raw snapshot doc | Bản mô tả rất hữu ích về trạng thái local dataset. |
| `docs/WLASL_raw_analysis_vi.md` | raw analysis tiếng Việt | Giải thích kỹ bằng tiếng Việt về raw layer. |
| `videos/*.mp4` | raw video | Artifact lớn, không nên sửa tại chỗ; đây là data source thực để decode frame. |

### `data/datasets/WLASL/index/`

| Thành phần | Vai trò | Nhận xét |
| --- | --- | --- |
| `master_instances.csv` | bảng master phẳng | Flatten từ `WLASL_v0.3.json`. |
| `available_instances.csv` | các row có video local | Cầu nối từ metadata sang local training snapshot. |
| `missing_instances.csv` | các row thiếu video local | Hữu ích cho audit dữ liệu. |
| `nslt_only_instances.csv` | row chỉ thấy trong NSLT | Phơi ra 12 ID ngoài master manifest. |
| `class_id_to_gloss.csv` | class map audit | Dùng để kiểm tra nhất quán class list và gloss. |
| `video_to_split.csv`, `video_to_split_all.csv` | split map | Ánh xạ split giữa master và NSLT subsets. |
| `subsets/` | manifests đầy đủ | Chứa cả sample thiếu video local. |
| `subsets_available/` | manifests sạch | Chỉ giữ sample có video local, dùng thật cho downstream stages. |
| `reports/dataset_summary.md` | index summary | Ghi coverage cho `nslt100/nslt300/nslt1000/nslt2000`. |
| `logs/` | log stage index | Audit chạy thực tế. |

### `data/datasets/WLASL/standardized/`

| Thành phần | Vai trò | Nhận xét |
| --- | --- | --- |
| `frames/<subset>/<split>/<sample_id>/*.jpg` | standardized frames | Đầu vào thật cho pose extraction. |
| `videos/` | standardized video | Có thể lưu nếu bật config, hiện save frames là chính. |
| `manifests/nslt100_*.csv` | standardized manifest | Dùng downstream cho `nslt100`. |
| `manifests/nslt300_*.csv` | standardized manifest | Dùng downstream cho `nslt300`. |
| `manifests/nslt1000_*.csv` | standardized manifest | Dùng downstream cho `nslt1000`. |
| `reports/*.md` | standardization report | Cho biết số frames, số lỗi, output size, bbox behavior. |
| `logs/` | log stage | Audit chạy thực tế. |

### `data/datasets/WLASL/pose/rtmw_l/`

| Thành phần | Vai trò | Nhận xét |
| --- | --- | --- |
| `wholebody_133/<subset>/<split>/*.npz` | shared pose | Output thật của RTMW-l whole-body pose. |
| `manifests/nslt300_*.csv` | pose manifest | Hiện có cho `nslt300`. |
| `reports/nslt300_pose_quality_report.md` | pose report | Ghi quality, confidence theo vùng, device, model config/checkpoint. |
| `logs/` | log pose extraction | Audit stage pose. |

### `data/datasets/WLASL/branch_inputs/skeleton/rtmw_l/`

| Thành phần | Vai trò | Nhận xét |
| --- | --- | --- |
| `selected_27/` | reduced pose | `.npz` sau khi chọn keypoints. |
| `selected_31/` | reduced pose | `.npz` sau khi chọn keypoints. |
| `normalized/selected_27/` | normalized pose | Sau normalize `x, y, confidence`. |
| `normalized/selected_31/` | normalized pose | Sau normalize `x, y, confidence`. |
| `graph_tensors/selected_27/` | train-ready tensors | Đầu vào thật của skeleton models. |
| `graph_tensors/selected_31/` | train-ready tensors | Đầu vào thật của skeleton models. |
| `manifests/*.csv` | skeleton manifests | Hiện có cho `nslt100` và `nslt300`, cả `train/val/test/all`. |
| `reports/*_skeleton_inputs_report.md` | branch report | Ghi selected indices, confidence scale, sanitize stats, split summary. |
| `logs/` | log build inputs | Audit stage skeleton input. |
| `README.md` | dataset card local | Giải thích cấu trúc bundle skeleton. |
| `metadata.json` | metadata card | Ghi `subset`, shape, sample count, num_classes, manifests. |

## 6.17 `checkpoints/`

| Thành phần | Vai trò | Nhận xét |
| --- | --- | --- |
| `checkpoints/pose/rtmw_l/rtmw-l_8xb320-270e_cocktail14-384x288.py` | config MMPose | File cấu hình RTMW-l dùng khi extract pose. |
| `checkpoints/pose/rtmw_l/rtmw-dw-x-l_simcc-cocktail14_270e-384x288-20231122.pth` | checkpoint MMPose | Trọng số bắt buộc cho RTMW-l pose extraction. |
| `checkpoints/models/skeleton/.gitkeep` | placeholder dir | Chưa có checkpoint model nghiên cứu nào theo cấu trúc này. |
| `checkpoints/models/regions/.gitkeep` | placeholder dir | Placeholder. |
| `checkpoints/models/hand_poseflow/.gitkeep` | placeholder dir | Placeholder. |

## 6.18 `outputs/`

`outputs/skeleton/<run_name>/` là nơi train loop hiện tại ghi artifact thật:

- `checkpoints/best.pt`
- `checkpoints/last.pt`
- `config_resolved.yaml`
- `metrics.json`
- `summary.json`
- `train_log.csv`
- đôi khi có `eval_test_best.json`

Đây là nguồn đúng để xem kết quả thực tế của pipeline train/eval hiện hành.

## 6.19 `experiments/`

Thư mục này hiện chỉ mới là khung:

- `experiments/skeleton/.gitkeep`
- `experiments/regions/.gitkeep`
- `experiments/hand_poseflow/.gitkeep`

Nó phản ánh ý định tổ chức experiment outputs riêng, nhưng code hiện tại đang ghi thật vào `outputs/`.

## 6.20 `hf_bundle/` và `hf_sub300_bundle/`

Đây là artifact phát hành dữ liệu skeleton cho Hugging Face.

| Thư mục | Vai trò |
| --- | --- |
| `hf_bundle/` | bundle train-ready skeleton cho `nslt100` |
| `hf_sub300_bundle/` | bundle train-ready skeleton cho `nslt300` |

Mỗi bundle chứa:

- graph tensor zip cho `selected_27`
- graph tensor zip cho `selected_31`
- `manifests.zip`
- `reports.zip`
- `logs.zip`
- `metadata.json`
- `README.md`

## 6.21 `kaggle_bundle/` và `kaggle_sub300_bundle/`

Đây là artifact phát hành để chạy pose extraction trên Kaggle.

| Thư mục | Vai trò |
| --- | --- |
| `kaggle_bundle/` | bundle Kaggle cho `nslt100` |
| `kaggle_sub300_bundle/` | bundle Kaggle cho `nslt300` |

Điểm cần hiểu rõ:

- `repo.zip` hoặc `repo/` là bản sao một phần repo cần thiết cho Kaggle.
- `standardized_<subset>.zip` chứa standardized frames/manifests.
- `checkpoints/pose/rtmw_l/` chứa config + checkpoint RTMW-l.
- Đây không phải source of truth để sửa logic; chỉ là gói triển khai.

## 7. Điểm mạnh kiến trúc hiện tại

- Pipeline phân tầng rõ: raw, index, standardized, pose, branch inputs, training.
- Nhánh skeleton đã manifest-driven khá sạch.
- Dataset loader của skeleton xử lý tốt chuyện path remap local/HF/Kaggle.
- Mỗi stage preprocessing đều có logs và reports, thuận tiện audit.
- Production train loop có CLI override, dry-run, checkpointing, final test eval, optional W&B.
- `STGCNPP` đã được cài repo-local bằng PyTorch thuần, tránh phụ thuộc nặng vào upstream training stack.

## 8. Điểm yếu và rủi ro hiện tại

- Có hai “trung tâm mô hình”:
  - `src/slr/branches/skeleton/models/*` là model chạy thật.
  - `src/slr/models/*` lại là placeholder.
  Điều này dễ gây nhầm cho người mới.
- `regions`, `hand_poseflow`, `inference`, generic `training/evaluate` vẫn chủ yếu là scaffold.
- `selected_49` mới chỉ có chỗ giữ chỗ trong schema, chưa có mapping thật.
- Language Label Smoothing mới ở mức helper/config định hướng, chưa gắn vào production train loop.
- `experiments/` chưa phải nơi output thật, trong khi `outputs/` mới là thư mục đang được dùng.
- Notebooks hiện gần như trống, không phản ánh được các workflow phân tích đã có trong code.

## 9. Đường chạy thật và đường chạy giả

### 9.1 Đường chạy thật

- `scripts/00_build_index.py`
- `scripts/01_standardize_videos.py`
- `scripts/02_extract_pose_rtmw.py`
- `scripts/03_build_skeleton_inputs.py`
- `scripts/train_skeleton.py`
- `scripts/evaluate_skeleton.py`

Tương ứng với:

- `src/slr/data/build_index.py`
- `src/slr/data/standardize_videos.py`
- `src/slr/pose/extract_rtmw.py`
- `src/slr/branches/skeleton/build_inputs.py`
- `src/slr/branches/skeleton/train.py`

### 9.2 Đường scaffold / placeholder

- `scripts/train_regions.py`
- `scripts/train_hand_poseflow.py`
- `scripts/evaluate.py`
- `src/slr/training/train.py`
- `src/slr/training/evaluate.py`
- gần như toàn bộ `src/slr/models/*`
- toàn bộ `src/slr/inference/*`
- phần lớn `src/slr/branches/regions/*`
- phần lớn `src/slr/branches/hand_poseflow/*`

## 10. Cách chạy nhanh theo trạng thái hiện tại

### Build index

```bash
python scripts/00_build_index.py --config configs/preprocessing/index.yaml
```

### Standardize subset

```bash
python scripts/01_standardize_videos.py --config configs/preprocessing/standardize.yaml
```

### Extract pose

```bash
python scripts/02_extract_pose_rtmw.py --config configs/preprocessing/pose_rtmw_l.yaml
```

### Build skeleton inputs

```bash
python scripts/03_build_skeleton_inputs.py --config configs/branches/skeleton/stgcnpp_31.yaml
```

### Sanity check dataset

```bash
python scripts/check_skeleton_dataset.py --config configs/train/skeleton_selected_31_stgcnpp.yaml --split train --limit 4
```

### Dry-run train

```bash
python scripts/train_skeleton.py --config configs/train/skeleton_selected_31_stgcnpp.yaml --run-name dry-sel31-stgcnpp --dry-run --no-wandb
```

### Smoke train

```bash
python scripts/train_skeleton.py --config configs/train/skeleton_selected_31_stgcnpp.yaml --run-name smoke-sel31-stgcnpp --epochs 1 --limit-train 32 --limit-val 16 --limit-test 16 --batch-size 8 --num-workers 0 --no-wandb
```

### Evaluate checkpoint

```bash
python scripts/evaluate_skeleton.py --config outputs/skeleton/smoke-sel31-stgcnpp/config_resolved.yaml --checkpoint outputs/skeleton/smoke-sel31-stgcnpp/checkpoints/best.pt --split test --batch-size 8
```

## 11. Hướng phát triển hợp lý nhất

Nếu tiếp tục phát triển repo này, thứ tự hợp lý nhất là:

1. Tiếp tục benchmark và tối ưu nhánh `skeleton`.
2. Dọn rõ ranh giới giữa `src/slr/branches/skeleton/models` và `src/slr/models`.
3. Hoàn thiện Language Label Smoothing trong production train loop.
4. Thêm CTR-GCN thật vào nhánh skeleton production.
5. Chỉ sau đó mới đầu tư sâu vào `regions` và `hand_poseflow`.
6. Nếu cần inference thực tế, phải xây lại `src/slr/inference/*` thay vì dùng placeholder hiện tại.

## 12. Kết luận

Toàn bộ folder dự án hiện phản ánh một repo nghiên cứu đã có một trục production đủ rõ cho skeleton-based sign language recognition:

- raw WLASL local snapshot
- index manifests sạch
- standardized frames
- RTMW-l wholebody pose
- reduced skeleton graph tensors
- train/eval bằng `SimpleSTGCN` hoặc `STGCNPP`

Phần mạnh nhất của repo hiện nay không phải là số lượng branch, mà là việc một nhánh đã được làm đến nơi đến chốn.

Nhìn tổng thể:

- `skeleton` là nhánh chạy thật.
- `regions` và `hand_poseflow` là hướng mở rộng.
- `outputs/`, `reports/`, `hf_bundle/`, `kaggle_bundle/` cho thấy repo này đã vượt qua giai đoạn chỉ “thiết kế khung” và đang được dùng để chuẩn bị dữ liệu, đóng gói môi trường và chạy thí nghiệm thật.
