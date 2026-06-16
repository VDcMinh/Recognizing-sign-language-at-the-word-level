# Ghi Chú Triển Khai Bundle Kaggle Cho `nslt1000` Remaining

## 1. Mục tiêu

File này ghi lại chi tiết những gì đã được thực hiện để hỗ trợ đóng gói Kaggle bundle cho WLASL `nslt1000`, nhưng chỉ lấy phần dữ liệu còn thiếu so với pose output đã trích trước đó từ `nslt300`.

Mục tiêu thực tế là:

- không đóng gói full `nslt1000`
- không đổi tên subset sang `nslt1000_remaining`
- vẫn giữ layout bundle tương thích với notebook Kaggle hiện tại
- chỉ đưa vào bundle các sample thuộc `nslt1000` mà chưa có pose output hoàn chỉnh trong `nslt300`

## 2. Repo và các file đã được inspect

Các file/chỗ quan trọng đã được đọc để hiểu pipeline hiện tại:

- `scripts/prepare_kaggle_bundle.py`
- `configs/preprocessing/standardize_nslt300.yaml`
- `configs/preprocessing/standardize_nslt1000.yaml`
- `configs/preprocessing/pose/pose_rtmw_l.yaml`
- `configs/preprocessing/pose/pose_rtmw_l_kaggle.yaml`
- `scripts/preprocess/02_extract_pose_rtmw.py`
- `src/slr/pose/extract_rtmw.py`
- `src/slr/data/manifests.py`
- `src/slr/data/validation.py`
- `reports/preprocessing/kaggle_sub300_bundle_report.md`
- `README.md`

## 3. Kết luận dữ liệu trước khi code

Trước khi viết script, đã kiểm tra quan hệ dữ liệu giữa `nslt300` và `nslt1000`.

Kết quả:

- `nslt300` là subset thực sự của `nslt1000`
- ở mức metadata gốc, toàn bộ sample của `nslt300` đều nằm trong `nslt1000`
- ở mức manifest mà pipeline hiện tại dùng, `nslt300` cũng là subset của `nslt1000`

Với dữ liệu standardized hiện có trong repo:

- `nslt300` available rows: `2660`
- `nslt1000` available rows: `7232`
- phần remaining thực tế là: `7232 - 2660 = 4572`

Chia theo split:

- train: `3104`
- val: `844`
- test: `624`

Điểm này quan trọng vì:

- `1000 - 300 = 700` chỉ là chênh lệch số class
- số sample/video còn lại để chạy pose extraction là `4572`, không phải `700`

## 4. File mới đã tạo

Đã tạo file script mới:

- `scripts/prepare_kaggle_nslt1000_remaining_bundle.py`

Script này được viết độc lập, không sửa script bundle cũ.

## 5. Những gì script mới làm

### 5.1. Input CLI

Script hỗ trợ các tham số sau:

- `--project-root`
- `--standardized-root`
- `--pose-done-root`
- `--output-root`
- `--done-subset`
- `--target-subset`
- `--copy-repo`
- `--copy-checkpoints`
- `--make-zip`
- `--skip-missing-frames`
- `--overwrite`
- `--dry-run`
- `--verbose`

Mặc định phù hợp với repo hiện tại:

- `done-subset = nslt300`
- `target-subset = nslt1000`

### 5.2. Logic đọc manifest

Script đọc:

- standardized target manifest:
  - `data/datasets/WLASL/standardized/manifests/nslt1000_all.csv`
- pose manifest đã xong:
  - `data/datasets/WLASL/pose/rtmw_l/manifests/nslt300_all.csv`

Script validate:

- manifest phải tồn tại
- manifest phải có đủ cột cần thiết
- `split` phải thuộc `train`, `val`, `test`

### 5.3. Logic xác định sample đã xong

Script chỉ coi một dòng trong pose manifest `nslt300` là đã xong nếu:

- `status == "ok"` hoặc
- `status == "success"`

Script không dùng:

- `path`
- `class_id`

để xác định sample trùng nhau.

Thay vào đó, script match theo thứ tự ưu tiên:

1. `instance_uid`
2. `sample_id`
3. `video_id`

Ý nghĩa:

- nếu `instance_uid` có mặt thì dùng `instance_uid`
- nếu thiếu `instance_uid` thì fallback `sample_id`
- nếu thiếu tiếp thì fallback `video_id`

Đây là logic ổn định hơn vì path của `nslt300` và `nslt1000` khác nhau theo folder subset, còn `class_id` chỉ là nhãn lớp chứ không phải ID sample.

### 5.4. Logic tạo remaining

Remaining được tính theo dạng:

`nslt1000_remaining = nslt1000_all - nslt300_pose_done`

Tức là:

- đọc full standardized rows của `nslt1000`
- đọc pose rows đã xong của `nslt300`
- loại khỏi `nslt1000` những row đã được phủ bởi `nslt300` theo stable key

Kết quả trên dữ liệu thực:

- total target rows: `7232`
- covered by `nslt300` pose: `2660`
- remaining rows: `4572`

### 5.5. Logic copy frame folders

Script copy frame từ:

- `standardized-root/frames/nslt1000/{split}/{sample_folder}`

sang:

- `output-root/standardized_nslt1000/data/datasets/WLASL/standardized/frames/nslt1000/{split}/{sample_folder}`

Ưu tiên lấy `sample_folder` từ:

1. tên thư mục cuối cùng của `frames_dir`
2. nếu không resolve được thì fallback `sample_id`
3. nếu vẫn không được thì fallback `video_id`

Nếu dùng fallback, script log warning rõ ràng.

Nếu thiếu folder frames:

- mặc định script báo lỗi và dừng
- nếu bật `--skip-missing-frames` thì script bỏ qua row đó và ghi warning

### 5.6. Rewrite `frames_dir` trong manifest

Đây là điểm rất quan trọng.

Manifest standardized gốc hiện tại chứa `frames_dir` dạng absolute Windows path, ví dụ:

- `F:/DMV/Recognizing-sign-language-at-the-word-level/data/datasets/WLASL/standardized/frames/nslt1000/test/00625`

Trong Kaggle, extractor hiện tại đọc trực tiếp cột `frames_dir`, nên nếu giữ absolute path này thì notebook sẽ fail.

Vì vậy script đã rewrite `frames_dir` trong manifest bundle thành path tương thích Kaggle:

- `data/datasets/WLASL/standardized/frames/nslt1000/test/00625`

Đây là thay đổi bắt buộc để `scripts/preprocess/02_extract_pose_rtmw.py` và `src/slr/pose/extract_rtmw.py` có thể chạy đúng trong Kaggle working directory.

### 5.7. Tên folder và tên manifest

Script cố ý giữ nguyên:

- folder: `standardized_nslt1000`
- subset trong config: `nslt1000`
- manifest names:
  - `nslt1000_train.csv`
  - `nslt1000_val.csv`
  - `nslt1000_test.csv`
  - `nslt1000_all.csv`

Script không tạo:

- `standardized_nslt1000_remaining`
- `nslt1000_remaining_train.csv`
- `SUBSET = "nslt1000_remaining"`

Lý do:

- notebook Kaggle hiện tại vẫn nên chạy với `SUBSET = "nslt1000"`
- chỉ có nội dung manifest là remaining, còn naming phải giữ tương thích với pipeline đang có

### 5.8. Copy repo

Nếu bật `--copy-repo`, script copy các thành phần cần thiết vào:

- `output-root/repo/`

Bao gồm:

- `configs/`
- `scripts/`
- `slr/`
- `src/`
- `README.md`
- `pyproject.toml`
- `requirements.txt`
- `sitecustomize.py`

Script cố ý loại trừ artifact lớn và thứ không cần thiết, ví dụ:

- `data/`
- `checkpoints/`
- `experiments/`
- `reports/`
- `notebooks/`
- `outputs/`
- `hf_bundle/`
- `kaggle_bundle/`
- `kaggle_sub300_bundle/`
- `.git/`
- `.venv*`
- `__pycache__/`

### 5.9. Copy checkpoints

Nếu bật `--copy-checkpoints`, script copy checkpoint RTMW-l từ:

- `checkpoints/pose/rtmw_l/`

sang:

- `output-root/checkpoints/pose/rtmw_l/`

Script validate:

- folder checkpoint phải tồn tại
- phải có file model/config hợp lệ

### 5.10. Tạo config Kaggle cho `nslt1000`

Nếu trong repo copy chưa có file config phù hợp cho `nslt1000`, script tạo:

- `repo/configs/preprocessing/pose/pose_rtmw_l_kaggle_nslt1000.yaml`

Config này được dựng từ template hiện có:

- ưu tiên `pose/pose_rtmw_l_kaggle.yaml`
- fallback `pose/pose_rtmw_l.yaml`

Script chỉnh các trường sau:

- `input.subset = nslt1000`
- `input.splits = [train, val, test]`
- `input.standardized_manifests_root = data/datasets/WLASL/standardized/manifests`
- `input.manifest_filenames = nslt1000_train.csv/nslt1000_val.csv/nslt1000_test.csv`

Điểm quan trọng:

- config mới không hard-code `nslt300`
- output pose vẫn ghi vào:
  - `data/datasets/WLASL/pose/rtmw_l/`

Nghĩa là notebook có thể gọi:

```bash
python scripts/preprocess/02_extract_pose_rtmw.py --config configs/preprocessing/pose/pose_rtmw_l_kaggle_nslt1000.yaml
```

## 6. MANIFEST.json đã được thêm gì

Script tạo `MANIFEST.json` trong bundle với các trường chính:

- `target_subset`
- `done_subset`
- `key_used`
- `total_target_rows`
- `done_rows_from_nslt300_pose_manifest`
- `covered_target_rows`
- `remaining_rows`
- `remaining_rows_by_split`
- `covered_rows_by_key`
- `remaining_first_available_key_counts`
- `created_at`
- `source_project_root`
- `source_standardized_root`
- `source_pose_done_root`
- `output_root`
- `copy_repo`
- `copy_checkpoints`
- `make_zip`
- `skip_missing_frames`
- `notes`

`notes` nói rõ:

- đây chỉ là remaining subset của `nslt1000`
- không phải full standardized bundle
- dùng stable-key matching
- không chứa raw videos
- không chứa pose outputs
- không chứa graph tensors

## 7. README_KAGGLE_BUNDLE.md đã được thêm gì

Script tạo README trong bundle để giải thích cách dùng trên Kaggle.

README nhấn mạnh:

- mục đích bundle là chạy pose extraction cho phần còn thiếu của `nslt1000`
- dù tên Kaggle dataset có thể là `wlasl-nslt1000-remaining-standardized-rtmw`, notebook vẫn nên đặt:
  - `SUBSET = "nslt1000"`
  - `STANDARDIZED_DIRNAME = "standardized_nslt1000"`
- không đổi tên manifest thành `nslt1000_remaining`
- output pose sau khi chạy mới chỉ là remaining
- cần merge kết quả mới với pose `nslt300` cũ để có bộ `nslt1000` đầy đủ
- bundle không chứa:
  - raw videos
  - pose outputs
  - graph tensors

## 8. Validation cuối script

Script có validation cuối sau khi ghi bundle:

- kiểm tra các manifest output đã được tạo
- kiểm tra:
  - `all = train + val + test`
- kiểm tra mọi row trong manifest remaining đều có frame folder tương ứng trong bundle

Ngoài ra script còn chặn output path không an toàn, ví dụ:

- không cho `output-root` trùng project root
- không cho `output-root` nằm trong:
  - standardized root
  - pose done root
  - checkpoint root

## 9. Kết quả verify đã chạy

### 9.1. Verify cú pháp

Đã chạy:

```bash
python -m py_compile scripts/prepare_kaggle_nslt1000_remaining_bundle.py
```

Kết quả:

- pass

### 9.2. Dry-run trên dữ liệu thật

Đã chạy bằng interpreter trong venv:

```bash
.\.venv-rtmw310\Scripts\python.exe scripts/prepare_kaggle_nslt1000_remaining_bundle.py --project-root . --standardized-root data/datasets/WLASL/standardized --pose-done-root data/datasets/WLASL/pose/rtmw_l --output-root kaggle_nslt1000_remaining_bundle --done-subset nslt300 --target-subset nslt1000 --copy-repo --copy-checkpoints --make-zip --dry-run
```

Kết quả summary:

- total nslt1000 rows: `7232`
- rows already covered by nslt300 pose: `2660`
- done rows from nslt300 pose manifest: `2660`
- rows remaining: `4572`
- rows remaining per split:
  - train=`3104`
  - val=`844`
  - test=`624`

### 9.3. Smoke test end-to-end

Đã tạo một fixture nhỏ cục bộ:

- 3 rows target `nslt1000`
- 1 row nằm trong done pose `nslt300`
- 2 row là remaining

Sau đó đã chạy script thật để:

- tạo bundle output
- copy frame folders
- copy repo
- copy checkpoints
- tạo config `pose_rtmw_l_kaggle_nslt1000.yaml`
- tạo `MANIFEST.json`
- tạo `README_KAGGLE_BUNDLE.md`
- tạo zip

Kết quả:

- script chạy thành công
- summary:
  - total nslt1000 rows: `3`
  - covered: `1`
  - remaining: `2`

Đã kiểm tra thêm:

- bundle structure được tạo đúng
- config Kaggle mới được tạo đúng
- manifest output đã rewrite `frames_dir` sang path kiểu:
  - `data/datasets/WLASL/standardized/frames/nslt1000/...`

Sau khi test xong, fixture tạm đã được xóa.

## 10. File nào trong repo đã thay đổi

File mới đã thêm:

- `scripts/prepare_kaggle_nslt1000_remaining_bundle.py`
- `bundle1k.md`

Script cũ không bị sửa.

## 11. Cách chạy đề xuất

Nếu Python mặc định của máy bạn đã có dependency:

```bash
python scripts/prepare_kaggle_nslt1000_remaining_bundle.py \
  --project-root . \
  --standardized-root data/datasets/WLASL/standardized \
  --pose-done-root data/datasets/WLASL/pose/rtmw_l \
  --output-root kaggle_nslt1000_remaining_bundle \
  --done-subset nslt300 \
  --target-subset nslt1000 \
  --copy-repo \
  --copy-checkpoints \
  --make-zip
```

Nếu muốn dùng đúng venv đã dùng để verify:

```bash
.\.venv-rtmw310\Scripts\python.exe scripts/prepare_kaggle_nslt1000_remaining_bundle.py \
  --project-root . \
  --standardized-root data/datasets/WLASL/standardized \
  --pose-done-root data/datasets/WLASL/pose/rtmw_l \
  --output-root kaggle_nslt1000_remaining_bundle \
  --done-subset nslt300 \
  --target-subset nslt1000 \
  --copy-repo \
  --copy-checkpoints \
  --make-zip
```

## 12. Expected output

```text
kaggle_nslt1000_remaining_bundle.zip
kaggle_nslt1000_remaining_bundle/
|-- repo/
|-- checkpoints/
|-- standardized_nslt1000/
|-- MANIFEST.json
`-- README_KAGGLE_BUNDLE.md
```

Trong đó:

- `standardized_nslt1000/` chỉ chứa remaining samples
- nhưng naming vẫn giữ là `nslt1000`
- không có raw videos
- không có existing pose outputs
- không có graph tensors

## 13. Ý nghĩa vận hành

Sau khi upload bundle này lên Kaggle và chạy extractor:

- output mới sẽ chỉ là pose cho `4572` sample remaining
- output đó chưa phải full pose set của `nslt1000`

Muốn có bộ pose `nslt1000` hoàn chỉnh, bạn sẽ cần merge:

- pose `nslt300` cũ
- pose `nslt1000` remaining mới

thành một bộ pose unified cho `nslt1000`.
