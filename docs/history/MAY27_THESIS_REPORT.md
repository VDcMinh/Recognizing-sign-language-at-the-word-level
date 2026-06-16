# MAY27 Thesis Report

Ngày lập báo cáo: `2026-05-27`

## 1. Mục tiêu và phạm vi

File này tổng hợp việc đọc, đối chiếu và phân tích toàn bộ cấu trúc repo `Recognizing-sign-language-at-the-word-level` để trả lời ba câu hỏi:

1. Repo này đang làm gì ở mức tổng thể.
2. Mỗi nhóm file có công dụng gì và đang hoạt động đến đâu.
3. Những điểm nào là pipeline chạy thật, những điểm nào mới là scaffold hoặc placeholder.

Lưu ý về phạm vi:

- Tôi đã đọc kỹ toàn bộ mã nguồn Python, file cấu hình, tài liệu Markdown và script chính trong repo.
- Với các thư mục artifact lớn như `data/`, `outputs/`, `checkpoints/`, `hf_bundle/`, `kaggle_bundle/` và các file sinh ra như video, `npz`, `json`, `png`, zip, báo cáo này phân tích theo vai trò, cấu trúc và cơ chế tạo ra chúng thay vì liệt kê từng file media riêng lẻ.
- Ba notebook trong `notebooks/` hiện chỉ là placeholder, gần như không chứa nội dung nghiên cứu thực thi.

## 2. Kết luận ngắn gọn

Đây không còn là một repo scaffold thuần túy. Repo đã có một pipeline skeleton branch chạy được end-to-end cho bài toán nhận diện ngôn ngữ ký hiệu ở mức word-level trên WLASL:

`index dữ liệu -> chuẩn hóa video -> trích xuất pose RTMW -> xây input skeleton -> train/evaluate ST-GCN/ST-GCN++`

Tuy nhiên, chỉ một nhánh là skeleton branch đang hoàn thiện tương đối. Các nhánh còn lại như `regions`, `hand_poseflow`, `generic models`, `generic training`, `inference` phần lớn mới là khung mã hoặc placeholder.

Nói cách khác:

- Phần production thật: `src/slr/data`, `src/slr/pose`, `src/slr/branches/skeleton`, một phần `src/slr/training`, các script `00` tới `03`, `train_skeleton.py`, `evaluate_skeleton.py`.
- Phần định hướng tương lai nhưng chưa hoàn chỉnh: `src/slr/branches/regions`, `src/slr/branches/hand_poseflow`, `src/slr/models`, `src/slr/inference`, `src/slr/training/train.py`, `src/slr/training/evaluate.py`.

## 3. Bức tranh kiến trúc tổng thể

### 3.1 Tổ chức repo

Repo được tổ chức theo hướng nghiên cứu có kỷ luật:

- `src/slr/`: mã nguồn chính.
- `scripts/`: entry points để chạy pipeline.
- `configs/`: YAML cấu hình cho dataset, preprocessing, branch, training.
- `data/`: dữ liệu gốc, metadata và dữ liệu đã chuẩn hóa.
- `outputs/`: kết quả sinh trong quá trình preprocessing.
- `checkpoints/`: trọng số train/eval.
- `reports/`: báo cáo trung gian về preprocessing/training.
- `docs/`: hướng dẫn kỹ thuật.
- `hf_bundle/`, `kaggle_bundle/`: gói artifact để mang đi môi trường khác.

### 3.2 Luồng dữ liệu thật của repo

Pipeline hiện tại hoạt động theo thứ tự:

1. `scripts/00_build_index.py`
2. `scripts/01_standardize_videos.py`
3. `scripts/02_extract_pose_rtmw.py`
4. `scripts/03_build_skeleton_inputs.py`
5. `scripts/train_skeleton.py`
6. `scripts/evaluate_skeleton.py`

Ý nghĩa từng bước:

1. Dò metadata WLASL và video local, tạo manifest chuẩn.
2. Crop/resize video thành chuẩn đầu vào thống nhất.
3. Dùng RTMW-l từ MMPose để trích xuất 133 whole-body keypoints.
4. Chọn subset keypoints, normalize, đổi sang graph tensor cho ST-GCN.
5. Train mô hình skeleton.
6. Đánh giá mô hình trên split test.

### 3.3 Điểm quan trọng dễ gây nhầm lẫn

- Có hai “cây model” trong repo:
  - `src/slr/branches/skeleton/models/`: model chạy thật.
  - `src/slr/models/`: khung generic, đa số chưa dùng.
- Có hai “cây training”:
  - `src/slr/branches/skeleton/train.py`: train/eval thật cho skeleton.
  - `src/slr/training/train.py`, `src/slr/training/evaluate.py`: scaffold chung, chưa phải entry production.

## 4. Phân tích các file ở root

### 4.1 File đóng vai trò định danh project

#### `README.md`

File giới thiệu chính của repo. Chức năng:

- Mô tả mục tiêu nghiên cứu word-level sign language recognition.
- Định vị repo theo WLASL.
- Cho cái nhìn tổng quan về cấu trúc và workflow.

Ý nghĩa:

- Đây là tài liệu mở đầu cho người mới.
- Nội dung mang tính định hướng, không phải nguồn sự thật chi tiết nhất về trạng thái code hiện tại.

#### `pyproject.toml`

Chức năng:

- Khai báo package Python tên `slr`.
- Dùng `setuptools`.
- Chỉ ra repo đang dùng `src` layout với package thật nằm trong `src/`.

Điều cần biết:

- Đây là nền tảng giúp `import slr` hoạt động chuẩn khi cài editable hoặc build package.

### 4.2 File hỗ trợ import local

#### `sitecustomize.py`

Chức năng:

- Tự động thêm thư mục `src/` vào `sys.path` khi chạy lệnh ad-hoc trong repo.

Giá trị thực tế:

- Giúp script chạy trực tiếp mà không cần luôn luôn `pip install -e .`.
- Rất hữu ích cho môi trường notebook, terminal, script thử nghiệm.

#### `slr/__init__.py`

Chức năng:

- Là import shim để `import slr` trỏ về `src/slr`.

Điều cần biết:

- Đây không phải package nguồn thật.
- Package thật nằm trong `src/slr`.

### 4.3 File yêu cầu môi trường

#### `requirements.txt`

Dependency tổng quát cho repo.

#### `requirements-rtmw.txt`

Dependency dành cho pose extraction bằng RTMW/MMPose.

#### `requirements-kaggle-train.txt`

Dependency tối ưu cho bối cảnh training trên Kaggle hoặc môi trường bundle hóa.

Ý nghĩa chung:

- Repo tách dependency theo use case thay vì một file duy nhất, phù hợp với môi trường nghiên cứu nhiều pipeline.

### 4.4 File tài liệu trạng thái và báo cáo

#### `IMPLEMENTATION.md`

Mô tả hiện trạng triển khai pipeline và các quyết định kỹ thuật.

#### `TRAINING.md`

Tập trung vào luồng huấn luyện, config, cách chạy và trạng thái training.

#### `LATEST_INFO.md`

Báo cáo/tóm tắt tình hình cập nhật gần nhất.

#### `NEW_LATEST_INFO.md`

Một trong những file phản ánh trạng thái thực tế tốt nhất của repo. Nội dung cho thấy:

- skeleton branch là nhánh hoàn thiện nhất.
- các nhánh còn lại chưa production-ready.
- nhiều artifact preprocessing/training đã tồn tại.

#### `StandardLS.md`

Tài liệu riêng về Standard Label Smoothing và các cấu hình thí nghiệm liên quan.

#### `PROJECT_STRUCTURE_GUIDE.md`

Giải thích cấu trúc thư mục và cách đọc repo.

Đánh giá chung:

- Bộ tài liệu root khá mạnh.
- Các file này bổ sung cho nhau: `README` cho onboarding, `IMPLEMENTATION` và `TRAINING` cho kỹ thuật, `LATEST_INFO`/`NEW_LATEST_INFO` cho snapshot trạng thái.

## 5. Phân tích mã nguồn trong `src/slr`

## 5.1 `src/slr/utils`

Đây là tầng tiện ích dùng lại giữa nhiều pipeline.

#### `src/slr/utils/io.py`

Chức năng:

- Tạo thư mục.
- Đọc/ghi JSON, YAML, CSV, text.

Ý nghĩa:

- Là lớp I/O chuẩn hóa cho toàn repo.
- Giúp các pipeline giảm lặp lại logic đọc ghi file.

#### `src/slr/utils/logging.py`

Chức năng:

- Tạo logger nhất quán.
- Thiết lập logging cho script/pipeline.

Ý nghĩa:

- Quan trọng cho pipeline dài như preprocessing và training.

#### `src/slr/utils/video.py`

Chức năng:

- Probe metadata video.
- Đọc frame từ video.
- Ghi video từ danh sách frame.

Ý nghĩa:

- Là lớp tiện ích nền cho chuẩn hóa video và visualization.

#### `src/slr/utils/bbox.py`

Chức năng:

- Biểu diễn bounding box.
- Parse, clip, expand, serialize bbox.

Ý nghĩa:

- Phục vụ trực tiếp cho standardization khi crop signer từ video.

#### `src/slr/utils/image.py`

Chức năng:

- Đọc/ghi ảnh.
- Resize theo kiểu letterbox.

Ý nghĩa:

- Đảm bảo đầu ra có kích thước chuẩn mà không làm méo khung hình.

#### `src/slr/utils/seed.py`

Chức năng:

- Seed Python và NumPy để tăng reproducibility.

### 5.2 `src/slr/data`

Đây là tầng xây manifest và chuẩn hóa dữ liệu video.

#### `src/slr/data/manifests.py`

Chức năng:

- Khai báo schema cột cho các manifest ở nhiều giai đoạn:
  - master
  - subset
  - standardized
  - pose
  - skeleton

Ý nghĩa:

- Đây là “hợp đồng dữ liệu” của repo.
- Nếu muốn mở rộng pipeline bền vững, đây là nơi cần giữ ổn định.

#### `src/slr/data/validation.py`

Chức năng:

- Kiểm tra cột bắt buộc.
- Kiểm tra null.
- Kiểm tra thứ tự schema.
- Kiểm tra split.

Ý nghĩa:

- Tạo kỷ luật dữ liệu trước khi bước sang preprocessing tiếp theo.

#### `src/slr/data/build_index.py`

Đây là một trong những file quan trọng nhất của repo.

Chức năng:

- Đọc metadata gốc WLASL:
  - `WLASL_v0.3.json`
  - `wlasl_class_list.txt`
  - `missing.txt`
  - `nslt_*.json`
- Quét thư mục video local.
- Kết hợp metadata gốc và file local.
- Tạo:
  - master manifest
  - manifest theo subset
  - map class
  - report phân tích số lượng available/missing

Ý nghĩa:

- Đây là bước hợp nhất “thế giới metadata” và “thế giới file thực tế”.
- Không có bước này thì pipeline sau sẽ không có manifest nhất quán để chạy.

Điểm mạnh:

- Tách hàm rõ ràng.
- Có khái niệm `available`, `missing`, `nslt_only`, `class_map`, `video_to_split`.

Điều cần biết:

- Pipeline phụ thuộc mạnh vào chất lượng metadata WLASL và tình trạng file local.
- Bộ WLASL thật sự có độ “bẩn” dữ liệu nhất định, nên file này mang vai trò làm sạch logic rất lớn.

#### `src/slr/data/standardize_videos.py`

Chức năng:

- Đọc subset manifest.
- Xử lý frame range.
- Áp bounding box.
- Crop signer.
- Resize về kích thước chuẩn bằng letterbox.
- Ghi video chuẩn hóa hoặc frame chuẩn hóa.
- Xuất standardized manifest và report.

Ý nghĩa:

- Đây là bước đưa dữ liệu video về một chuẩn hình học thống nhất.
- Là nền bắt buộc trước khi trích pose.

Điều quan trọng:

- Các config `standardize.yaml`, `standardize_nslt300.yaml`, `standardize_nslt1000.yaml` cho thấy repo hỗ trợ nhiều subset.

### 5.3 `src/slr/pose`

Đây là tầng xử lý pose estimation và chất lượng pose.

#### `src/slr/pose/pose_schema.py`

Chức năng:

- Khai báo schema keypoint cho RTMW-l wholebody 133.
- Định nghĩa các vùng cơ thể.
- Định nghĩa các tập keypoint rút gọn:
  - `selected_27`
  - `selected_31`
  - `selected_49` hiện chưa được triển khai thực chất
- Định nghĩa mouth indices.
- Có helper kiểm tra tính hợp lệ.

Ý nghĩa:

- Là file “từ điển giải phẫu” của pipeline pose.

Điểm rất quan trọng:

- `selected_49` đã được đặt tên nhưng chưa thật sự có nội dung hoàn chỉnh, nghĩa là đây là một hướng mở rộng dang dở.

#### `src/slr/pose/keypoint_selection.py`

Chức năng:

- Chọn tập keypoint con từ 133 keypoint gốc.
- Tạo payload `.npz` ổn định cho downstream.

Ý nghĩa:

- Cho phép mô hình skeleton không cần học trên toàn bộ 133 điểm, giúp giảm độ phức tạp và tập trung vào các vùng quan trọng.

#### `src/slr/pose/pose_normalization.py`

Chức năng:

- Chuẩn hóa tọa độ `xy` về khoảng `[-1, 1]`.
- Ước lượng thang confidence theo percentile.
- Chuẩn hóa confidence.
- Làm sạch giá trị không hữu hạn.

Ý nghĩa:

- Đây là bước rất quan trọng để biến pose thô thành input ổn định cho GCN.

#### `src/slr/pose/pose_quality.py`

Chức năng:

- Tính coverage theo confidence.
- Tính mean confidence tổng thể và theo từng vùng.
- Tính tỉ lệ frame hợp lệ.
- Tổng hợp chất lượng pose ở mức manifest.

Ý nghĩa:

- Là file phân tích chất lượng pose.
- Hữu ích cho báo cáo học thuật vì nó cho phép định lượng độ tin cậy của output từ pose extractor.

#### `src/slr/pose/extract_rtmw.py`

Một file cực kỳ quan trọng.

Chức năng:

- Gọi RTMW-l thông qua MMPose inferencer.
- Tự xử lý chọn device.
- Chọn người chính trong khung hình.
- Trích xuất keypoint theo frame.
- Ghi `.npz`.
- Tạo pose manifest.
- Tạo quality report.

Ý nghĩa:

- Đây là cầu nối giữa video chuẩn hóa và dữ liệu skeleton.
- Nếu bước này lỗi hoặc không ổn định, toàn bộ skeleton branch sẽ suy yếu.

Điều cần biết:

- File `requirements-rtmw.txt` và script test trong `scripts/tools/` tồn tại để phục vụ bước này.
- Repo đã có bằng chứng chạy thực tế cho RTMW qua file test/report trong `scripts/tools/_rtmw_mmpose_video_test_output/`.

### 5.4 `src/slr/branches/skeleton`

Đây là trái tim kỹ thuật của repo hiện tại.

#### `src/slr/branches/skeleton/graph.py`

Chức năng:

- Định nghĩa topology graph cho `selected_27` và `selected_31`.
- Tạo adjacency matrix.
- Chuẩn hóa adjacency.
- Đóng gói thành `SkeletonGraph`.

Ý nghĩa:

- Là phần quyết định “xương đồ thị” mà GCN sẽ học trên đó.

Điều cần biết:

- Thiết kế graph ảnh hưởng trực tiếp đến inductive bias của mô hình.

#### `src/slr/branches/skeleton/transforms.py`

Chức năng:

- Ép độ dài chuỗi về cố định, mặc định là `150` frame.
- Hỗ trợ chiến lược như `repeat`, `head`.
- Chuyển `(T, V, 3)` sang định dạng `(C, T, V, M)` cho GCN.

Ý nghĩa:

- Đây là bước chuẩn hóa tensor temporal trước khi nạp cho model.

#### `src/slr/branches/skeleton/dataset.py`

Chức năng:

- Nạp graph tensor `.npz`.
- Chỉ lấy mẫu `status=ok`.
- Remap path giữa local repo, Hugging Face bundle, Kaggle bundle.
- Validate shape.
- Tạo label map.
- Trả về tensor và metadata.
- Cung cấp `skeleton_collate_fn`.

Ý nghĩa:

- Đây là dataset production thực thụ cho training.

Điểm mạnh:

- Có ý thức rất rõ về tính di động môi trường.
- Không phụ thuộc cứng vào một layout path duy nhất.

Điều cần biết:

- Nếu mang bundle qua máy khác, logic remap path là phần rất đáng giá.

#### `src/slr/branches/skeleton/label_smoothing.py`

Chức năng:

- Chứa helper NumPy cho standard label smoothing và language-aware label smoothing.

Tình trạng thực tế:

- File này có giá trị nghiên cứu và thử nghiệm.
- Nhưng pipeline production hiện tại không đi trực tiếp qua helper này cho loss chính.

#### `src/slr/branches/skeleton/build_inputs.py`

Đây là file production rất quan trọng.

Chức năng:

- Đọc pose manifests.
- Ước lượng confidence scale từ train split.
- Chọn keypoint con.
- Normalize pose.
- Fix sequence length.
- Tạo:
  - `selected keypoints npz`
  - `normalized npz`
  - `graph tensor npz`
- Xuất skeleton manifests và reports.

Ý nghĩa:

- Đây là công đoạn chuyển từ “pose file” sang “tensor mà model dùng được”.

#### `src/slr/branches/skeleton/train.py`

Đây là entry production cho huấn luyện và đánh giá skeleton branch.

Chức năng:

- Parse CLI cho train/eval.
- Chuẩn hóa config và override.
- Chọn device.
- Build dataset, dataloader, graph, model.
- Chạy epoch loop.
- Kiểm tra shape đầu vào.
- Tính metric:
  - top-1
  - top-5
  - top-10
- Save checkpoint.
- Tích hợp tùy chọn W&B.
- Đánh giá test cuối cùng.

Ý nghĩa:

- Đây là file quan trọng nhất của giai đoạn learning.

Điểm mạnh:

- Có đủ các khối cần thiết cho một training pipeline nghiêm túc.
- Rõ ràng hơn nhiều so với các file generic ở `src/slr/training`.

#### `src/slr/branches/skeleton/models/__init__.py`

Factory chọn model skeleton thực tế.

#### `src/slr/branches/skeleton/models/simple_stgcn.py`

Chức năng:

- Cài đặt baseline ST-GCN gọn.

Ý nghĩa:

- Dùng làm mốc cơ sở.
- Phù hợp cho kiểm chứng pipeline và so sánh với model mạnh hơn.

#### `src/slr/branches/skeleton/models/stgcnpp.py`

Chức năng:

- Cài đặt ST-GCN++-style nội bộ của repo.
- Có các khối:
  - `SpatialGraphConv`
  - `MultiScaleTemporalConv`
  - `STGCNPPBlock`
  - `STGCNPP`

Ý nghĩa:

- Đây là model nâng cao và là hướng chính của repo hiện tại.

### 5.5 `src/slr/branches/regions`

Nhánh này mới ở mức scaffold.

#### `region_schema.py`

Khai báo tên các vùng như face/hands/body crop.

#### `transforms.py`

Có xử lý đơn giản như `normalize_uint8`.

#### `dataset.py`

`RegionSequenceDataset` mới là placeholder.

#### `build_crops.py`

Có khung tạo thư mục/log, nhưng logic crop region chưa hoàn chỉnh.

Kết luận:

- Nhánh regions thể hiện ý tưởng thiết kế tốt.
- Nhưng hiện chưa đủ để xem là pipeline production.

### 5.6 `src/slr/branches/hand_poseflow`

Nhánh này cũng mới ở mức scaffold.

#### `poseflow_schema.py`

Khai báo variant như `selected_31`, `hands_only`.

#### `dataset.py`

`HandPoseFlowDataset` là placeholder.

#### `build_hand_sequences.py`

Placeholder.

#### `build_poseflow.py`

Placeholder.

#### `build_inputs.py`

Mới đóng vai trò orchestrator gọi hai bước placeholder.

Kết luận:

- Đây là thiết kế cho một nhánh 2-stream hoặc hand-centric trong tương lai.
- Chưa phải thành phần có thể dùng để báo cáo kết quả chính thức.

### 5.7 `src/slr/training`

Đây là tầng hỗ trợ training dùng chung, nhưng chưa hoàn chỉnh ở cấp pipeline.

#### `metrics.py`

Chức năng:

- `AverageMeter`
- `accuracy_topk`
- `top_k_accuracy`

Đây là file hữu dụng và đang dùng được.

#### `losses.py`

Chức năng:

- Factory tạo loss.
- Hỗ trợ:
  - `cross_entropy`
  - `standard_label_smoothing`

Điểm rất quan trọng:

- Standard Label Smoothing ở đây dùng `nn.CrossEntropyLoss(label_smoothing=epsilon)`.
- Language Label Smoothing chưa được nối thành loss production ở cùng mức hoàn chỉnh.

#### `optim.py`

Chức năng:

- Factory optimizer: `adamw`, `adam`, `sgd`
- Factory scheduler: `cosine`, `step`

#### `checkpointing.py`

Chức năng:

- Save/load checkpoint kèm metadata.

#### `seed.py`

Chức năng:

- Seed Python/NumPy/PyTorch.

#### `wandb_utils.py`

Chức năng:

- Khởi tạo W&B.
- Log metric.
- Upload artifact.
- Kết thúc run.

#### `train.py`

Generic scaffold, chưa phải luồng training chính đang dùng.

#### `evaluate.py`

Generic scaffold, chưa phải luồng evaluate chính đang dùng.

### 5.8 `src/slr/models`

Đây là một khu vực dễ gây hiểu nhầm.

Nhận định:

- Tên thư mục khiến người đọc tưởng đây là model production.
- Nhưng trên thực tế phần lớn file ở đây mới là placeholder hoặc API-level skeleton.

#### `src/slr/models/skeleton/stgcnpp.py`

Placeholder, không phải implementation production chính.

#### `src/slr/models/skeleton/ctrgcn.py`

Placeholder cho hướng CTR-GCN.

#### `src/slr/models/skeleton/heads.py`

Placeholder.

#### `src/slr/models/regions/*.py`

Các file như `cnn_lstm.py`, `video_transformer.py`, `heads.py` chủ yếu là scaffold.

#### `src/slr/models/hand_poseflow/*.py`

Ví dụ `two_stream.py`, `heads.py`, cũng là scaffold.

Kết luận:

- Khi viết thesis hoặc trình bày kiến trúc, cần nói rất rõ rằng model dùng thật cho thí nghiệm skeleton nằm ở `src/slr/branches/skeleton/models/`, không phải `src/slr/models/`.

### 5.9 `src/slr/inference`

#### `predict_video.py`

Placeholder cho pipeline suy luận trên video đầu vào.

#### `visualize_prediction.py`

Placeholder cho trực quan hóa dự đoán.

Kết luận:

- Repo chưa có inference pipeline hoàn chỉnh ở mức deploy/demo.

## 6. Phân tích thư mục `scripts`

### 6.1 Script pipeline chính

#### `scripts/00_build_index.py`

Wrapper gọi `slr.data.build_index.main`.

#### `scripts/01_standardize_videos.py`

Wrapper gọi `slr.data.standardize_videos.main`.

#### `scripts/02_extract_pose_rtmw.py`

Wrapper gọi `slr.pose.extract_rtmw.main`.

#### `scripts/03_build_skeleton_inputs.py`

Wrapper gọi `slr.branches.skeleton.build_inputs.main`.

#### `scripts/04_build_region_inputs.py`

Trỏ tới branch regions, hiện chỉ scaffold.

#### `scripts/05_build_hand_poseflow_inputs.py`

Trỏ tới branch hand_poseflow, hiện chỉ scaffold.

### 6.2 Script training/eval

#### `scripts/train_skeleton.py`

Entry chạy training thật cho skeleton.

#### `scripts/evaluate_skeleton.py`

Entry chạy evaluate thật cho skeleton.

#### `scripts/train_regions.py`

Trỏ tới training generic hoặc scaffold.

#### `scripts/train_hand_poseflow.py`

Scaffold.

#### `scripts/evaluate.py`

Generic evaluate scaffold.

### 6.3 Script kiểm tra và đóng gói

#### `scripts/check_skeleton_dataset.py`

Chức năng:

- Sanity check dataset skeleton.
- Kiểm tra shape tensor.
- Kiểm tra graph topology.
- In thống kê class.

Ý nghĩa:

- Rất hữu ích trước khi train.

#### `scripts/prepare_hf_skeleton_bundle.py`

Chức năng:

- Đóng gói bundle cho Hugging Face hoặc môi trường chia sẻ.
- Validate manifest, count, shape.
- Tạo bundle theo subset.

Ý nghĩa:

- Hỗ trợ tái sử dụng dữ liệu tiền xử lý mà không phải build lại từ đầu.

#### `scripts/prepare_kaggle_bundle.py`

Chức năng:

- Tạo bundle cho Kaggle.
- Gom subset manifest, video chuẩn hóa và checkpoint RTMW cần thiết.
- Có kiểm soát an toàn path và kiểm tra hợp lệ.

Ý nghĩa:

- Cho thấy repo đã nghĩ tới portability và compute environment khác nhau.

### 6.4 Script trong `scripts/tools`

#### `scripts/tools/test_rtmw_mmpose_video.py`

Smoke test RTMW/MMPose trên một video.

#### `scripts/tools/visualize_selected_27_samples.py`

Tạo contact sheet vẽ keypoint đã rút gọn lên sample frames.

#### `scripts/tools/visualize_pose_sets.py`

So sánh trực quan wholebody 133 với các set rút gọn như `selected_27`, `selected_31`.

Ý nghĩa của nhóm tools:

- Đây là các công cụ chẩn đoán và minh họa rất tốt cho thesis.
- Chúng giúp giải thích rằng repo không chỉ train model mà còn có công cụ kiểm định chất lượng dữ liệu pose.

## 7. Phân tích thư mục `configs`

## 7.1 `configs/dataset`

#### `configs/dataset/wlasl.yaml`

File gốc mô tả:

- root dataset
- đường dẫn raw metadata/video/docs
- danh sách subset hỗ trợ
- default subset/split

Ý nghĩa:

- Đây là cấu hình nền cho toàn bộ pipeline WLASL.

## 7.2 `configs/preprocessing`

#### `index.yaml`

Cấu hình cho bước build index.

#### `standardize.yaml`

Chuẩn hóa video cho `nslt100`.

#### `standardize_nslt300.yaml`

Chuẩn hóa video cho `nslt300`.

#### `standardize_nslt1000.yaml`

Chuẩn hóa video cho `nslt1000`.

#### `pose_rtmw_l.yaml`

Cấu hình trích pose RTMW-l cho nhánh chuẩn.

#### `pose_rtmw_l_kaggle.yaml`

Biến thể cho bối cảnh Kaggle.

#### `region_crops.yaml`

Scaffold config cho branch regions.

#### `poseflow.yaml`

Scaffold config cho branch hand poseflow.

Nhận xét:

- Tầng preprocessing config là phần khá trưởng thành của repo.

## 7.3 `configs/branches/skeleton`

#### `stgcnpp_27.yaml`, `stgcnpp_31.yaml`

Cấu hình build input skeleton cho `selected_27` và `selected_31`.

#### `stgcnpp_27_nslt300.yaml`, `stgcnpp_31_nslt300.yaml`

Biến thể cho subset `nslt300`.

#### `ctrgcn_31.yaml`

Thể hiện ý định hỗ trợ CTR-GCN, nhưng code model production cho hướng này chưa hoàn chỉnh.

#### `stgcnpp_31_languagels.yaml`

Thể hiện định hướng language label smoothing, nhưng production path chưa đồng bộ đầy đủ.

Nhận xét:

- Cấu hình phong phú hơn mức triển khai thực tế.
- Một số config đi trước code.

## 7.4 `configs/train`

Đây là lớp cấu hình training quan trọng nhất.

### Nhóm dataset/loader

- `skeleton_selected_27.yaml`
- `skeleton_selected_31.yaml`

### Nhóm baseline CE

- `skeleton_selected_27_baseline.yaml`
- `skeleton_selected_31_baseline.yaml`
- `skeleton_selected_27_stgcnpp.yaml`
- `skeleton_selected_31_stgcnpp.yaml`

### Nhóm Standard Label Smoothing cho `nslt100`

- `skeleton_selected_27_stgcnpp_standardls_eps005.yaml`
- `skeleton_selected_27_stgcnpp_standardls_eps01.yaml`
- `skeleton_selected_27_stgcnpp_standardls_eps03.yaml`
- `skeleton_selected_31_stgcnpp_standardls_eps005.yaml`
- `skeleton_selected_31_stgcnpp_standardls_eps01.yaml`
- `skeleton_selected_31_stgcnpp_standardls_eps03.yaml`

### Nhóm `nslt300`

- `skeleton_selected_27_nslt300_stgcnpp.yaml`
- `skeleton_selected_31_nslt300_stgcnpp.yaml`
- `skeleton_selected_27_nslt300_stgcnpp_standardls_eps03.yaml`
- `skeleton_selected_31_nslt300_stgcnpp_standardls_eps01.yaml`
- `skeleton_selected_31_nslt300_stgcnpp_standardls_eps02.yaml`
- `skeleton_selected_31_nslt300_stgcnpp_standardls_eps03.yaml`

Ý nghĩa:

- Repo đã bước qua giai đoạn “chỉ có 1 config demo”.
- Đã có tư duy thí nghiệm bài bản: thay backbone, thay keypoint set, thay subset, thay epsilon smoothing.

## 7.5 `configs/branches/regions` và `configs/branches/hand_poseflow`

- `face_hands_baseline.yaml`
- `hand_poseflow_baseline.yaml`

Tình trạng:

- Chủ yếu mang vai trò định hướng tương lai.

## 8. Tài liệu trong `docs`, `reports`, `notebooks`

### 8.1 `docs`

#### `docs/skeleton_training_baseline.md`

Hướng dẫn baseline skeleton.

#### `docs/skeleton_stgcnpp_integration.md`

Mô tả cách tích hợp ST-GCN++.

#### `docs/standard_label_smoothing.md`

Tài liệu riêng về StandardLS.

Ý nghĩa:

- Bộ docs này hỗ trợ viết phần methodology trong thesis khá tốt.

### 8.2 `reports`

#### `reports/preprocessing/README.md`

Giải thích tầng preprocessing và artifact của nó.

#### `reports/experiments/README.md`

Giải thích cách lưu và tổ chức artifact thí nghiệm.

#### Các report cụ thể

- `reports/preprocessing/nslt300_index_and_standardization_report.md`
- `reports/preprocessing/nslt1000_index_and_standardization_report.md`
- `reports/preprocessing/hf_sub300_bundle_report.md`
- `reports/preprocessing/kaggle_sub300_bundle_report.md`
- `reports/training/nslt300_training_config_report.md`
- `reports/training/nslt300_standardls_config_report.md`

Ý nghĩa:

- Các file này là “dấu chân khoa học” của quá trình nghiên cứu.
- Chúng giúp thesis có bằng chứng quy trình chứ không chỉ có code.

### 8.3 `notebooks`

Ba notebook hiện tại:

- `notebooks/01_explore_wlasl.ipynb`
- `notebooks/02_check_pose_quality.ipynb`
- `notebooks/03_visualize_keypoints.ipynb`

Tình trạng thực tế:

- Mỗi notebook hiện chỉ có một markdown cell mô tả ý định sử dụng.
- Chưa chứa notebook analysis thực tế.

Kết luận:

- Không nên xem notebook là nguồn phân tích chính của repo ở thời điểm hiện tại.

## 9. Dữ liệu, artifact và bundle

### 9.1 `data/`

Đây là nơi chứa:

- raw WLASL metadata/video/docs
- dữ liệu sinh ra trong các bước chuẩn hóa theo cấu trúc dataset nội bộ

Điểm đáng chú ý từ tài liệu raw WLASL:

- Bộ dữ liệu có `2000` gloss.
- `21083` instances trong metadata gốc.
- Khoảng `11980` video local sẵn có.
- Khoảng `9103` instance đang thiếu local video.
- Có chênh lệch nhỏ giữa `nslt_2000.json` và master metadata.

Ý nghĩa:

- Repo đang xử lý một dataset có độ lệch giữa metadata và file thực, nên tầng manifest là cực kỳ quan trọng.

### 9.2 `outputs/`

Chứa kết quả sinh ra từ preprocessing:

- standardized videos/frames
- pose outputs
- skeleton tensors
- reports theo từng bước

Ý nghĩa:

- `outputs/` là vùng làm việc thật của pipeline.

### 9.3 `checkpoints/`

Chứa:

- checkpoint train của skeleton models
- có thể gồm cả checkpoint pose hoặc mô hình trung gian theo workflow

Ý nghĩa:

- Là sản phẩm phục vụ tái lập thí nghiệm.

### 9.4 `hf_bundle/`, `hf_sub300_bundle/`

Đây là các gói để chuyển dữ liệu skeleton preprocessing sang môi trường khác.

Giá trị:

- Hữu ích nếu muốn chia sẻ artifact qua Hugging Face hoặc máy khác mà không build lại.

### 9.5 `kaggle_bundle/`, `kaggle_sub300_bundle/`

Tương tự, nhưng tối ưu cho luồng Kaggle.

Điều cần biết:

- Đây là artifact triển khai/phân phối, không phải mã nguồn gốc.

### 9.6 `experiments/`

Nơi lưu kết quả chạy thí nghiệm, log, thống kê hoặc cấu hình snapshot.

Ý nghĩa:

- Quan trọng cho phần tái lập kết quả và so sánh mô hình.

## 10. Điểm mạnh kỹ thuật của repo

### 10.1 Pipeline dữ liệu được suy nghĩ khá kỹ

Từ raw metadata tới input graph tensor có các lớp trung gian rõ ràng:

- manifest
- standardized video
- pose file
- normalized pose
- graph tensor

Điều này rất tốt cho nghiên cứu vì:

- dễ debug
- dễ kiểm tra chất lượng từng tầng
- dễ đóng gói artifact

### 10.2 Skeleton branch có tính production rõ ràng

Biểu hiện:

- dataset thật
- graph topology thật
- transforms thật
- model thật
- train/eval thật
- configs đa dạng

### 10.3 Có tư duy portability

Các script bundle và logic remap path cho thấy repo không chỉ chạy cục bộ, mà đã tính tới:

- Kaggle
- Hugging Face style bundle
- môi trường máy khác

### 10.4 Có lớp báo cáo trung gian

Điều này rất giá trị cho thesis vì giúp minh chứng quy trình và kiểm soát chất lượng.

## 11. Hạn chế và các điểm cần cảnh báo

### 11.1 Repo chưa đồng đều về độ hoàn thiện

Chỉ skeleton branch là thật sự hoàn chỉnh tương đối.

Các phần chưa sẵn sàng:

- regions branch
- hand_poseflow branch
- inference pipeline
- generic models
- generic training/evaluate

### 11.2 Tên thư mục dễ gây nhầm

Đặc biệt là:

- `src/slr/models/` nhìn như model chính nhưng thực tế không phải.
- model dùng thật nằm ở `src/slr/branches/skeleton/models/`.

### 11.3 `selected_49` chưa được hiện thực đầy đủ

Đây là dấu hiệu cho một hướng nghiên cứu mở nhưng chưa hoàn tất.

### 11.4 Language Label Smoothing mới ở mức bán hoàn chỉnh

Repo có:

- helper
- tài liệu
- config định hướng

Nhưng loss production hoàn thiện nhất hiện tại là:

- cross entropy
- standard label smoothing

### 11.5 Notebook analysis còn rất mỏng

Nếu thesis cần phần exploratory analysis trực quan trong notebook, repo hiện chưa cung cấp mạnh ở mảng này.

## 12. Cách mô tả repo này trong thesis

Nếu cần mô tả học thuật ngắn gọn, có thể diễn đạt như sau:

“Đây là một codebase nghiên cứu có cấu trúc module hóa cho bài toán nhận diện ngôn ngữ ký hiệu ở mức từ trên bộ dữ liệu WLASL. Hệ thống hiện triển khai hoàn chỉnh một pipeline skeleton-based gồm xây dựng manifest dữ liệu, chuẩn hóa video, trích xuất pose whole-body bằng RTMW-l, chọn và chuẩn hóa keypoint, xây dựng graph tensor và huấn luyện các mô hình ST-GCN/ST-GCN++ để phân loại gloss. Các nhánh mở rộng như region-based modeling, hand poseflow và inference trực tiếp trên video mới đang ở giai đoạn scaffold.”

## 13. Những file quan trọng nhất cần nhớ

Nếu chỉ chọn ra các file “xương sống” của repo, đây là danh sách quan trọng nhất:

- `src/slr/data/build_index.py`
- `src/slr/data/standardize_videos.py`
- `src/slr/pose/extract_rtmw.py`
- `src/slr/pose/pose_schema.py`
- `src/slr/branches/skeleton/build_inputs.py`
- `src/slr/branches/skeleton/dataset.py`
- `src/slr/branches/skeleton/graph.py`
- `src/slr/branches/skeleton/train.py`
- `src/slr/branches/skeleton/models/simple_stgcn.py`
- `src/slr/branches/skeleton/models/stgcnpp.py`
- `src/slr/training/losses.py`
- `configs/train/*.yaml`
- `scripts/train_skeleton.py`
- `scripts/evaluate_skeleton.py`

## 14. Kết luận cuối cùng

Repo này đã vượt khỏi mức “đồ án khung” ở nhánh skeleton. Nó có một pipeline nghiên cứu khá nghiêm túc, chia tầng rõ, có manifest schema, có chuẩn hóa dữ liệu, có pose extraction, có graph construction, có training/evaluation và có cơ chế bundle artifact sang môi trường khác.

Tuy nhiên, repo chưa phải một hệ thống đồng nhất hoàn chỉnh ở mọi nhánh. Khi đánh giá hoặc trình bày trước hội đồng, cần nói rất rõ rằng đóng góp thực thi mạnh nhất hiện tại nằm ở skeleton-based recognition pipeline; các hướng regions, hand poseflow và inference end-user vẫn đang ở giai đoạn chuẩn bị hoặc mở rộng.

Nếu dùng repo này làm nền cho thesis, phần mạnh nhất để khai thác là:

- quy trình dữ liệu WLASL đến skeleton tensor
- lựa chọn tập keypoint `selected_27` và `selected_31`
- thiết kế graph skeleton
- so sánh baseline ST-GCN với ST-GCN++
- ảnh hưởng của Standard Label Smoothing trên các subset như `nslt100` và `nslt300`
