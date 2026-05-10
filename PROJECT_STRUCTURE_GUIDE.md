# Project Structure Guide

Tài liệu này giải thích chi tiết cấu trúc hiện tại của project, vai trò của từng nhóm file/thư mục, và luồng dữ liệu từ `raw/` đến các nhánh huấn luyện.

## 1. Mục tiêu của cấu trúc này

Project được tổ chức theo tư duy pipeline nghiên cứu:

```text
raw
-> index
-> standardized
-> shared pose
-> branch-specific inputs
-> training / evaluation
```

Điểm quan trọng:

- `raw/` là nguồn dữ liệu gốc, chỉ đọc, không chỉnh sửa.
- Mỗi tầng chỉ đọc từ tầng trước và ghi ra tầng của chính nó.
- Tầng pose là tầng dùng chung cho cả 3 nhánh.
- Skeleton branch là nhánh chính hiện tại.
- Region branch và Hand Pose Flow branch đã có scaffold để mở rộng dần.

Mục tiêu của cấu trúc này là tách rõ:

- dữ liệu gốc
- dữ liệu đã chuẩn hóa
- đặc trưng dùng chung
- đặc trưng riêng cho từng nhánh
- config thí nghiệm
- mã nguồn xử lý
- đầu ra huấn luyện và báo cáo

## 2. Toàn cảnh thư mục

```text
Recognizing-sign-language-at-the-word-level/
├── README.md
├── PROJECT_STRUCTURE_GUIDE.md
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── sitecustomize.py
├── slr/
├── configs/
├── data/
├── checkpoints/
├── src/
├── scripts/
├── experiments/
├── reports/
└── notebooks/
```

Ý nghĩa cấp cao:

- `README.md`: mô tả ngắn gọn mục tiêu project, dataset, pipeline và roadmap.
- `PROJECT_STRUCTURE_GUIDE.md`: tài liệu kiến trúc chi tiết này.
- `pyproject.toml`: khai báo package theo `src layout`.
- `requirements.txt`: dependency nền tảng cho preprocessing và utilities.
- `.gitignore`: quy ước file nào được theo dõi, file nào bỏ qua.
- `sitecustomize.py`: thêm `src/` vào import path khi chạy lệnh trực tiếp từ repo root.
- `slr/`: import shim để `import slr` hoạt động ngay cả khi chưa cài editable package.
- `configs/`: toàn bộ config dữ liệu, preprocessing, branch, experiment.
- `data/`: tất cả dữ liệu từ raw đến các tầng sinh ra sau này.
- `checkpoints/`: trọng số pose model và model huấn luyện.
- `src/`: mã nguồn chính của package `slr`.
- `scripts/`: wrapper CLI mỏng để chạy từng stage.
- `experiments/`: kết quả thí nghiệm theo từng branch.
- `reports/`: tài liệu, hình, báo cáo preprocessing và experiment.
- `notebooks/`: notebook phân tích, trực quan hóa, kiểm tra chất lượng.

## 3. Dữ liệu đi theo chiều nào

Luồng dữ liệu chuẩn của project như sau:

### Bước 1: Raw layer

Nguồn:

```text
data/datasets/WLASL/raw/
├── metadata/
├── videos/
└── docs/
```

Vai trò:

- `metadata/` chứa file JSON và class list gốc của WLASL.
- `videos/` chứa video gốc từng sample.
- `docs/` chứa ghi chú, phân tích, mô tả dataset.

Quy tắc:

- Không crop trực tiếp ở đây.
- Không đổi tên file gốc ở đây.
- Không tạo output suy diễn vào đây.

Đây là source of truth của toàn repo.

### Bước 2: Index layer

Đầu vào:

- `raw/metadata/`
- `raw/videos/`

Đầu ra:

```text
data/datasets/WLASL/index/
├── reports/
└── .gitkeep
```

Mục đích:

- đọc metadata gốc
- xác định subset như `nslt100`, `nslt300`, `nslt1000`, `nslt2000`
- xây manifest chuẩn
- gán `sample_id`, `class_id`, `split`
- kiểm tra sample nào có video, thiếu video, metadata lỗi

Kết quả kỳ vọng:

- file CSV hoặc Parquet index
- class map
- split manifests
- audit reports

Đây là tầng biến dữ liệu gốc thành bảng chỉ mục có thể dùng ổn định cho các stage sau.

### Bước 3: Standardized layer

Đầu vào:

- manifest từ `index/`
- video gốc từ `raw/videos/`

Đầu ra:

```text
data/datasets/WLASL/standardized/
├── videos/
├── frames/
├── manifests/
├── logs/
└── reports/
```

Mục đích:

- crop theo bbox nếu cần
- resize về kích thước chuẩn
- letterbox để giữ tỉ lệ hình
- chuẩn hóa quy ước frame count hoặc sampling
- tùy chọn xuất video chuẩn hóa và frame chuẩn hóa

Ý nghĩa:

- stage này giúp các downstream stage không cần xử lý sự khác biệt lung tung giữa video gốc.
- pose extractor, region crops và hand sequence sẽ làm việc trên đầu vào sạch, đồng nhất hơn.

### Bước 4: Shared pose layer

Đầu vào:

- `standardized/videos/` hoặc `standardized/frames/`

Đầu ra:

```text
data/datasets/WLASL/pose/rtmw_l/
├── wholebody_133/
├── manifests/
├── logs/
└── reports/
```

Mục đích:

- trích xuất RTMW-l whole-body 133 keypoints
- lưu pose theo từng sample
- sinh report chất lượng pose

Ý nghĩa:

- đây là tầng dùng chung cho tất cả nhánh downstream.
- thay vì mỗi branch tự chạy pose extractor riêng, repo chỉ chạy một lần rồi tái sử dụng.

### Bước 5: Branch-specific inputs

Đầu vào:

- pose dùng chung từ `pose/rtmw_l/wholebody_133/`
- standardized frames hoặc standardized videos nếu cần

Đầu ra:

```text
data/datasets/WLASL/branch_inputs/
├── skeleton/
├── regions/
└── hand_poseflow/
```

Mỗi branch sinh biểu diễn đầu vào riêng.

#### 5.1 Skeleton branch

```text
branch_inputs/skeleton/rtmw_l/
├── selected_27/
├── selected_31/
├── selected_49/
├── normalized/
├── graph_tensors/
├── manifests/
└── reports/
```

Xử lý:

- chọn subset keypoints từ 133 điểm
- chuẩn hóa tọa độ
- build tensor theo định dạng graph model
- chuẩn bị input cho ST-GCN++ hoặc CTR-GCN

Đây là nhánh ưu tiên vì phù hợp định hướng paper hiện tại.

#### 5.2 Region branch

```text
branch_inputs/regions/rtmw_l/
├── face/
├── left_hand/
├── right_hand/
├── combined/
├── manifests/
└── reports/
```

Xử lý:

- dùng pose hoặc bbox để cắt vùng mặt
- cắt vùng tay trái
- cắt vùng tay phải
- có thể ghép thành multi-region sequence cho model thị giác theo thời gian

Nhánh này giữ nhiều thông tin thị giác hơn skeleton, đặc biệt hữu ích khi shape tay và biểu cảm mặt quan trọng.

#### 5.3 Hand sequence + Pose Flow branch

```text
branch_inputs/hand_poseflow/rtmw_l/
├── hand_sequences/
│   ├── left_hand/
│   └── right_hand/
├── poseflow/
│   ├── selected_31/
│   └── hands_only/
├── combined/
├── manifests/
└── reports/
```

Xử lý:

- tạo chuỗi ảnh tay trái và tay phải
- sinh pose flow hoặc motion feature từ keypoints
- kết hợp hai nguồn thông tin thành input cho mô hình hai luồng

Nhánh này phù hợp khi muốn khai thác vừa tín hiệu thị giác của bàn tay, vừa động học chuyển động.

### Bước 6: Training / Evaluation

Đầu vào:

- branch-specific inputs
- branch config
- experiment config

Đầu ra:

- model checkpoints
- metrics
- logs
- confusion matrix
- kết quả dự đoán

Thư mục liên quan:

```text
checkpoints/
experiments/
reports/experiments/
```

## 4. Giải thích chi tiết từng nhóm thư mục

## 4.1 `configs/`

Đây là nơi mô tả toàn bộ “ý định chạy” của project, thay vì hard-code trong Python.

### `configs/dataset/wlasl.yaml`

Mô tả dataset:

- tên dataset
- root path
- vị trí raw metadata
- vị trí raw videos
- các subset khả dụng
- subset mặc định
- split mặc định

Vai trò:

- là nguồn cấu hình chung cho mọi stage cần biết dataset nằm ở đâu.

### `configs/preprocessing/index.yaml`

Mô tả stage build index:

- subset mặc định
- mapping tên file metadata
- nơi ghi output index
- nơi ghi report

### `configs/preprocessing/standardize.yaml`

Mô tả stage chuẩn hóa video:

- manifest đầu vào
- output root
- kích thước ảnh/video đầu ra
- sampling rule
- margin ratio
- bật/tắt save frames
- bật/tắt save video

### `configs/preprocessing/pose_rtmw_l.yaml`

Mô tả stage trích xuất pose:

- backend pose là `rtmw_l`
- số keypoints kỳ vọng là `133`
- input là video hoặc frame đã standardized
- nơi ghi output pose
- placeholder cho config/checkpoint của MMPose

### `configs/preprocessing/region_crops.yaml`

Mô tả cách tạo face/hand crop cho region branch:

- nguồn frames
- nguồn pose
- output directory
- crop size
- loại vùng cần cắt

### `configs/preprocessing/poseflow.yaml`

Mô tả cách sinh pose flow cho nhánh thứ ba:

- nguồn pose
- output directory
- variant feature
- temporal stride
- có chuẩn hóa motion hay không

### `configs/branches/skeleton/*.yaml`

Nhóm config quan trọng nhất hiện tại.

- `stgcnpp_31.yaml`: baseline ST-GCN++ với `selected_31`
- `ctrgcn_31.yaml`: baseline CTR-GCN với `selected_31`
- `stgcnpp_31_languagels.yaml`: ST-GCN++ với language label smoothing

Chúng quyết định:

- branch nào được huấn luyện
- bộ keypoint nào được dùng
- số frame vào model
- số channel đầu vào
- batch size
- learning rate
- optimizer
- có dùng label smoothing hay không

### `configs/branches/regions/*.yaml`

Config baseline cho region branch.

### `configs/branches/hand_poseflow/*.yaml`

Config baseline cho nhánh hand poseflow.

### `configs/experiments/*.yaml`

Đây là level config “thí nghiệm cụ thể”, nơi ghép:

- dataset config
- branch config
- subset
- output dir thí nghiệm

Khác biệt giữa branch config và experiment config:

- branch config mô tả cách model hoặc input branch hoạt động
- experiment config mô tả một lần chạy cụ thể

## 4.2 `data/`

Đây là trung tâm dữ liệu của repo.

### `data/datasets/WLASL/raw/`

Nguồn gốc, không sửa.

### `data/datasets/WLASL/index/`

Tầng metadata đã được chuẩn hóa để pipeline có thể vận hành nhất quán.

### `data/datasets/WLASL/standardized/`

Tầng dữ liệu video/frame đã được làm sạch và chuẩn hóa.

### `data/datasets/WLASL/pose/`

Tầng đặc trưng pose dùng chung.

### `data/datasets/WLASL/branch_inputs/`

Tầng đặc trưng riêng theo branch.

### `data/datasets/WLASL/audits/`

Nơi đặt kết quả kiểm tra chất lượng dữ liệu:

- `video_integrity/`: hỏng file, đọc không được, lệch metadata
- `pose_quality/`: pose thiếu tay, confidence thấp, outlier
- `branch_quality/`: lỗi crop, lỗi sequence, tensor không đúng shape

## 4.3 `src/slr/`

Đây là mã nguồn thật của package.

Tư duy của `src/slr/` là chia theo domain xử lý thay vì gom mọi thứ vào vài file lớn.

### `src/slr/data/`

Xử lý các tầng đầu pipeline.

- `build_index.py`: stage tạo index manifest từ raw metadata.
- `standardize_videos.py`: stage chuẩn hóa video và frame.
- `manifests.py`: khai báo schema cột chuẩn cho manifest.
- `validation.py`: helper kiểm tra thiếu cột hoặc dữ liệu lỗi.

### `src/slr/pose/`

Xử lý pose dùng chung.

- `extract_rtmw.py`: entrypoint cho RTMW-l extraction.
- `pose_schema.py`: định nghĩa constants về `wholebody_133` và các keypoint set.
- `keypoint_selection.py`: hàm chọn subset keypoint như `selected_31`.
- `pose_normalization.py`: helper chuẩn hóa tọa độ pose.
- `pose_quality.py`: helper audit chất lượng pose.

### `src/slr/branches/skeleton/`

Module quan trọng nhất hiện tại.

- `build_inputs.py`: build selected keypoints, normalized pose, graph tensors.
- `dataset.py`: dataset class cho skeleton branch.
- `graph.py`: nơi định nghĩa topology graph cho graph models.
- `transforms.py`: padding, trimming, sequence transform.
- `label_smoothing.py`: helper cho standard label smoothing và language label smoothing.

### `src/slr/branches/regions/`

Chuẩn bị input cho region branch.

- `build_crops.py`: build crop vùng mặt và hai tay.
- `dataset.py`: dataset class cho face/hand sequence.
- `transforms.py`: transform ảnh cho region branch.
- `region_schema.py`: constants tên vùng.

### `src/slr/branches/hand_poseflow/`

Chuẩn bị input cho nhánh thứ ba.

- `build_hand_sequences.py`: cắt chuỗi ảnh tay.
- `build_poseflow.py`: sinh feature pose flow.
- `build_inputs.py`: điều phối cả hai bước trên.
- `dataset.py`: dataset class cho two-stream input.
- `poseflow_schema.py`: constants cho biến thể poseflow.

### `src/slr/models/`

Nơi đặt lớp model theo từng branch.

#### `src/slr/models/skeleton/`

- `stgcnpp.py`: placeholder cho ST-GCN++.
- `ctrgcn.py`: placeholder cho CTR-GCN.
- `heads.py`: classification head.

#### `src/slr/models/regions/`

- `cnn_lstm.py`: baseline CNN-LSTM cho chuỗi crop.
- `video_transformer.py`: baseline transformer theo thời gian.
- `heads.py`: classification head.

#### `src/slr/models/hand_poseflow/`

- `two_stream.py`: placeholder cho model hai luồng.
- `heads.py`: classification head.

### `src/slr/training/`

Phần dùng chung cho train/evaluate.

- `train.py`: CLI train tổng quát.
- `evaluate.py`: CLI evaluate tổng quát.
- `metrics.py`: metric như top-k accuracy.
- `losses.py`: nơi chứa loss function hoặc factory.
- `optim.py`: nơi cấu hình optimizer.
- `checkpointing.py`: helper lưu tên checkpoint.

### `src/slr/inference/`

Phần phục vụ suy luận sau này.

- `predict_video.py`: chạy dự đoán trên video đơn.
- `visualize_prediction.py`: trực quan hóa kết quả.

### `src/slr/utils/`

Helper dùng lại ở nhiều nơi.

- `io.py`: đọc ghi JSON, YAML, CSV, tạo thư mục.
- `logging.py`: logger thống nhất.
- `video.py`: probe metadata video.
- `image.py`: helper đọc ảnh.
- `bbox.py`: xử lý bounding box.
- `seed.py`: cố định seed.

## 4.4 `scripts/`

Đây là lớp CLI mỏng nhất của repo.

Mục tiêu:

- cho phép chạy stage bằng lệnh đơn giản
- không nhồi logic vào script
- logic thật đặt trong `src/slr/...`

Ví dụ mapping:

- `scripts/00_build_index.py` -> `slr.data.build_index.main`
- `scripts/01_standardize_videos.py` -> `slr.data.standardize_videos.main`
- `scripts/02_extract_pose_rtmw.py` -> `slr.pose.extract_rtmw.main`
- `scripts/03_build_skeleton_inputs.py` -> `slr.branches.skeleton.build_inputs.main`
- `scripts/04_build_region_inputs.py` -> `slr.branches.regions.build_crops.main`
- `scripts/05_build_hand_poseflow_inputs.py` -> `slr.branches.hand_poseflow.build_inputs.main`
- `scripts/train_skeleton.py` -> `slr.training.train.main`
- `scripts/train_regions.py` -> `slr.training.train.main`
- `scripts/train_hand_poseflow.py` -> `slr.training.train.main`
- `scripts/evaluate.py` -> `slr.training.evaluate.main`

Ý nghĩa đánh số `00`, `01`, `02`, `03`, `04`, `05`:

- phản ánh đúng thứ tự preprocessing chuẩn của pipeline.

## 4.5 `checkpoints/`

Tách checkpoint theo bản chất:

- `checkpoints/pose/rtmw_l/`: trọng số hoặc file liên quan pose backend.
- `checkpoints/models/skeleton/`: model weight của skeleton branch.
- `checkpoints/models/regions/`: model weight của region branch.
- `checkpoints/models/hand_poseflow/`: model weight của nhánh thứ ba.

Điều này giúp không trộn:

- weight của backbone pose
- weight của model học nhận diện gloss

## 4.6 `experiments/`

Mỗi branch có vùng riêng:

- `experiments/skeleton/`
- `experiments/regions/`
- `experiments/hand_poseflow/`

Đây là nơi chứa output của từng lần chạy, ví dụ:

- config snapshot
- log train
- metric CSV
- confusion matrix
- prediction samples

Không nên dùng `experiments/` để chứa dữ liệu preprocessing.

## 4.7 `reports/`

Đây là vùng tài liệu và phân tích.

- `reports/preprocessing/`: mô tả pipeline preprocessing, tổng hợp lỗi và audit.
- `reports/experiments/`: mô tả cách ghi nhận kết quả thí nghiệm.
- `reports/figures/`: hình ảnh minh họa, biểu đồ, trực quan hóa.

## 4.8 `notebooks/`

Mục tiêu của notebook là khám phá và trực quan hóa, không phải là nơi chứa pipeline chính.

- `01_explore_wlasl.ipynb`: khám phá metadata và phân bố dữ liệu.
- `02_check_pose_quality.ipynb`: xem chất lượng pose extraction.
- `03_visualize_keypoints.ipynb`: trực quan selected keypoints và trajectory.

Pipeline chính vẫn nên đi qua `src/slr/` và `scripts/`.

## 5. Schema dữ liệu giữa các tầng

## 5.1 Index manifest

Các cột dự kiến:

- `sample_id`
- `video_id`
- `gloss`
- `class_id`
- `subset`
- `split`
- `raw_video_path`
- `has_video`
- `frame_start`
- `frame_end`
- `bbox_x1`
- `bbox_y1`
- `bbox_x2`
- `bbox_y2`
- `signer_id`
- `fps`
- `width`
- `height`
- `num_frames`

Vai trò:

- đây là bảng điều phối trung tâm của toàn pipeline.
- gần như mọi stage sau này sẽ join hoặc đọc theo `sample_id`.

## 5.2 Standardized manifest

Các cột dự kiến:

- `sample_id`
- `video_id`
- `gloss`
- `class_id`
- `split`
- `standardized_video_path`
- `frames_dir`
- `num_frames`
- `output_size`
- `crop_bbox`
- `status`
- `error_message`

Vai trò:

- là cầu nối giữa video gốc và tầng pose/region downstream.

## 5.3 Shared pose `.npz`

Trường dự kiến:

- `keypoints`: `(T, 133, 3)`
- `image_size`
- `sample_id`
- `video_id`
- `gloss`
- `class_id`
- `split`

Ý nghĩa:

- `3` thường là `x, y, score`.
- tầng này là chuẩn dùng chung trước khi chọn keypoint set.

## 5.4 Skeleton selected `.npz`

Trường dự kiến:

- `keypoints`: `(T, 31, 3)` nếu `selected_31`
- `keypoint_set`
- `sample_id`
- `video_id`
- `gloss`
- `class_id`
- `split`

Vai trò:

- giảm pose từ 133 điểm xuống tập điểm quan trọng cho nhánh skeleton.

## 5.5 Graph tensor

Định dạng dự kiến:

- `C x T x V x M`

Trong đó:

- `C`: số channels, thường là `2` hoặc `3`
- `T`: số frame, ví dụ `150`
- `V`: số node, ví dụ `31`
- `M`: số person, hiện thiết kế ưu tiên `1`

Đây là định dạng tiêu biểu cho ST-GCN++ hoặc CTR-GCN.

## 6. Dòng xử lý đi qua code như thế nào

Một workflow điển hình cho skeleton branch sẽ là:

1. Chạy `scripts/00_build_index.py`
2. Script gọi `slr.data.build_index.main()`
3. Module đọc `configs/preprocessing/index.yaml`
4. Module đọc dữ liệu từ `raw/metadata/` và kiểm tra `raw/videos/`
5. Ghi manifest vào `data/datasets/WLASL/index/`

Sau đó:

1. Chạy `scripts/01_standardize_videos.py`
2. Module đọc manifest từ `index/`
3. Đọc video gốc từ `raw/videos/`
4. Ghi video hoặc frame chuẩn hóa vào `standardized/`
5. Ghi standardized manifest

Sau đó:

1. Chạy `scripts/02_extract_pose_rtmw.py`
2. Module đọc standardized input
3. Gọi backend pose RTMW-l
4. Ghi `.npz` pose vào `pose/rtmw_l/wholebody_133/`
5. Ghi manifest và report chất lượng pose

Sau đó:

1. Chạy `scripts/03_build_skeleton_inputs.py`
2. Đọc pose `133`
3. Chọn `selected_31`
4. Chuẩn hóa pose
5. Build graph tensor
6. Ghi vào `branch_inputs/skeleton/rtmw_l/`

Cuối cùng:

1. Chạy `scripts/train_skeleton.py --config ...`
2. `slr.training.train.main()` đọc config branch hoặc experiment
3. Dataset branch load `graph_tensors`
4. Model skeleton được khởi tạo
5. Train loop chạy, checkpoint và metrics được ghi ra `experiments/` và `checkpoints/`

## 7. Tại sao cấu trúc này tốt hơn kiểu cũ

Repo này cố tình tránh quay về các kiểu thư mục như:

- `interim`
- `poses`
- `derived`
- `frames_256`
- `clips_T64`
- `derived_common.py`

Lý do:

- tên cũ thường mô tả file theo “kết quả kỹ thuật rời rạc” hơn là theo tầng pipeline.
- khó nhìn ra file nào là dữ liệu gốc, file nào là output tạm, file nào là input cho model nào.
- khó tái sử dụng pose chung cho nhiều branch.
- khó audit và tái chạy từng stage một cách minh bạch.

Cấu trúc mới tốt hơn vì:

- rõ dòng dữ liệu
- rõ shared layer và branch-specific layer
- dễ thêm branch mới
- dễ log, audit, benchmark
- dễ scale từ debug subset đến full subset

## 8. Trạng thái hiện tại của mã nguồn

Hiện tại project đang ở trạng thái scaffold có chủ đích:

- cấu trúc thư mục đã sẵn sàng
- config nền tảng đã có
- CLI entrypoint đã có
- schema và interface chính đã có
- nhiều module đang là placeholder sạch, có docstring rõ ràng

Điều này có nghĩa:

- repo đã sẵn sàng để implement từng stage mà không cần thiết kế lại cấu trúc
- nhưng chưa phải là pipeline đầy đủ end-to-end ngay lập tức

## 9. Thứ tự nên triển khai tiếp

Nếu bám theo ưu tiên hiện tại, thứ tự phù hợp nhất là:

1. hoàn thiện `src/slr/data/build_index.py`
2. hoàn thiện `src/slr/data/standardize_videos.py`
3. tích hợp MMPose trong `src/slr/pose/extract_rtmw.py`
4. chốt mapping keypoint trong `src/slr/pose/pose_schema.py`
5. hoàn thiện `src/slr/branches/skeleton/build_inputs.py`
6. hiện thực dataset, graph, model cho skeleton branch
7. hoàn thiện train/evaluate cho nhánh skeleton
8. sau đó mới mở rộng sang `regions` và `hand_poseflow`

## 10. Kết luận ngắn

Project này được thiết kế như một hệ pipeline nghiên cứu rõ tầng, trong đó:

- `raw/` là dữ liệu gốc bất biến
- `index/` là lớp chỉ mục chuẩn
- `standardized/` là lớp video/frame đã chuẩn hóa
- `pose/` là đặc trưng dùng chung
- `branch_inputs/` là đầu vào riêng cho từng nhánh
- `src/slr/` là nơi chứa logic thật
- `scripts/` là lớp chạy CLI mỏng
- `configs/` quyết định cách từng stage hoạt động
- `experiments/`, `reports/`, `checkpoints/` giữ đầu ra nghiên cứu có tổ chức

Nếu nhìn theo tư duy triển khai, skeleton branch chính là “xương sống” hiện tại của toàn repo, còn hai branch còn lại đã có khung chuẩn để gắn thêm logic sau.
