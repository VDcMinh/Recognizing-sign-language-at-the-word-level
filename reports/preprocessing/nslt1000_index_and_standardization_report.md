# NSLT1000 Index and Standardization Report

## 1. Scope

Task nay chi cover hai tang:

- index verification cho `WLASL nslt1000`
- standardization cho `WLASL nslt1000`

Task nay khong dong vao:

- pose extraction
- skeleton inputs
- graph tensors
- training/evaluation
- region branch
- hand_poseflow branch
- placeholder models

Artifact `nslt100` va `nslt300` da duoc giu nguyen; khong co buoc xoa, clean, hoac overwrite artifact cua hai subset nay.

## 2. Repository files inspected

Da doc va kiem tra cac file/chay cac artifact lien quan sau:

- `configs/preprocessing/index.yaml`
- `configs/preprocessing/standardize.yaml`
- `configs/preprocessing/standardize_nslt300.yaml`
- `scripts/00_build_index.py`
- `scripts/01_standardize_videos.py`
- `src/slr/data/build_index.py`
- `src/slr/data/standardize_videos.py`
- `src/slr/data/manifests.py`
- `data/datasets/WLASL/index/subsets/nslt1000/train.csv`
- `data/datasets/WLASL/index/subsets/nslt1000/val.csv`
- `data/datasets/WLASL/index/subsets/nslt1000/test.csv`
- `data/datasets/WLASL/index/subsets/nslt1000/label_map.json`
- `data/datasets/WLASL/index/subsets_available/nslt1000/train.csv`
- `data/datasets/WLASL/index/subsets_available/nslt1000/val.csv`
- `data/datasets/WLASL/index/subsets_available/nslt1000/test.csv`
- `data/datasets/WLASL/index/subsets_available/nslt1000/label_map.json`
- `data/datasets/WLASL/index/master_instances.csv`
- `data/datasets/WLASL/index/available_instances.csv`
- `data/datasets/WLASL/index/missing_instances.csv`
- `data/datasets/WLASL/index/class_id_to_gloss.csv`
- `data/datasets/WLASL/index/video_to_split.csv`
- `data/datasets/WLASL/index/video_to_split_all.csv`
- `data/datasets/WLASL/standardized/`

## 3. Index verification

### 3.1. Files checked

Tat ca cac file sau deu ton tai va doc duoc:

- `data/datasets/WLASL/index/subsets/nslt1000/train.csv`
- `data/datasets/WLASL/index/subsets/nslt1000/val.csv`
- `data/datasets/WLASL/index/subsets/nslt1000/test.csv`
- `data/datasets/WLASL/index/subsets/nslt1000/label_map.json`
- `data/datasets/WLASL/index/subsets_available/nslt1000/train.csv`
- `data/datasets/WLASL/index/subsets_available/nslt1000/val.csv`
- `data/datasets/WLASL/index/subsets_available/nslt1000/test.csv`
- `data/datasets/WLASL/index/subsets_available/nslt1000/label_map.json`
- `data/datasets/WLASL/index/master_instances.csv`
- `data/datasets/WLASL/index/available_instances.csv`
- `data/datasets/WLASL/index/missing_instances.csv`
- `data/datasets/WLASL/index/class_id_to_gloss.csv`
- `data/datasets/WLASL/index/video_to_split.csv`
- `data/datasets/WLASL/index/video_to_split_all.csv`

### 3.2. Row counts

`subsets/nslt1000`:

- train: `8978`
- val: `2320`
- test: `1876`

`subsets_available/nslt1000`:

- train: `5001`
- val: `1290`
- test: `941`

### 3.3. Class coverage

- `label_map.json` trong `subsets/nslt1000`: `num_classes = 1000`
- `label_map.json` trong `subsets_available/nslt1000`: `num_classes = 1000`
- so `class_ids` trong ca hai file: `1000`
- range `class_id`: `0..999`
- so class thuc te co mat trong manifest:
  - `subsets/nslt1000`: `1000`
  - `subsets_available/nslt1000`: `1000`

Ket luan: class coverage day du cho `nslt1000`.

### 3.4. Integrity checks

Schema thuc te cua manifest subset la `video_path`, khong phai `raw_video_path`. Vi vay phan kiem tra path duoc doi chieu tren cot `video_path`.

Kiem tra tren `subsets/nslt1000`:

- `video_path` rong/null: `0`
- `video_path` khong ton tai tren may: `5942`
- split null: `0`
- split sai (`train/val/test`): `0`
- `class_id` null: `0`
- `class_id` ngoai range `0..999`: `0`
- duplicate `sample_id`: `0`
- duplicate `instance_uid`: `0`
- `class_id -> gloss` co nhieu gloss: `0`
- `gloss -> class_id` co nhieu class: `0`

Kiem tra tren `subsets_available/nslt1000`:

- `video_path` rong/null: `0`
- `video_path` khong ton tai tren may: `0`
- split null: `0`
- split sai (`train/val/test`): `0`
- `class_id` null: `0`
- `class_id` ngoai range `0..999`: `0`
- duplicate `sample_id`: `0`
- duplicate `instance_uid`: `0`
- `class_id -> gloss` co nhieu gloss: `0`
- `gloss -> class_id` co nhieu class: `0`
- class trong `subsets_available` khong co sample local: `0`

### 3.5. Gloss/class mapping check

Trong `data/datasets/WLASL/index/class_id_to_gloss.csv`:

- so dong co `nslt1000_total > 0`: `1000`
- so dong `nslt1000` co note `gloss_mismatch`: `1`

Mismatch duy nhat:

- `class_id = 163`
- `class_list_gloss = losear`
- `master_gloss = lose`

Day la canh bao metadata mapping lich su trong `class_id_to_gloss.csv`, khong tao duplicate, khong lam hong `subsets_available/nslt1000`, va khong chan stage standardization.

### 3.6. Index conclusion

Index `nslt1000` da san sang.

- Khong can chay lai `build_index`.
- Khong co loi blocker trong `subsets_available/nslt1000`.
- Viec `5942` path khong ton tai trong `subsets/nslt1000` la expected, vi day la manifest tong gom ca sample khong co raw video local.

## 4. Preflight check

### 4.1. Existing output state before running

Truoc khi chay standardization:

- `data/datasets/WLASL/standardized/frames/nslt1000/`: khong ton tai
- `data/datasets/WLASL/standardized/manifests/nslt1000_train.csv`: khong ton tai
- `data/datasets/WLASL/standardized/manifests/nslt1000_val.csv`: khong ton tai
- `data/datasets/WLASL/standardized/manifests/nslt1000_test.csv`: khong ton tai
- `data/datasets/WLASL/standardized/manifests/nslt1000_all.csv`: khong ton tai

Ket luan:

- khong co complete output
- khong co partial output
- can chay standardization moi cho `nslt1000`

### 4.2. Disk space

Preflight free space tren o `F:` truoc khi chay:

- total: `119.30 GB`
- free: `62.02 GB`

Muc free space nay du de tiep tuc task.

## 5. Config created

Da tao file config moi:

- `configs/preprocessing/standardize_nslt1000.yaml`

Khac biet so voi `configs/preprocessing/standardize.yaml` va `configs/preprocessing/standardize_nslt300.yaml`:

- chi doi `input.subset` thanh `nslt1000`

Moi tham so con lai duoc giu nguyen:

- output size `288x384`
- `crop_with_bbox: true`
- bbox margin giu nguyen
- `keep_aspect_ratio: true`
- `letterbox: true`
- `save_frames: true`
- `save_video: false`
- `overwrite: true`
- output root van la `data/datasets/WLASL/standardized`

Xac nhan:

- khong sua `configs/preprocessing/standardize.yaml`
- khong xoa artifact `nslt100`
- khong xoa artifact `nslt300`
- `overwrite: true` chi tac dong tren output subset `nslt1000`, vi pipeline resolve duong dan theo `frames/nslt1000/*`, `manifests/nslt1000_*.csv`, `reports/nslt1000_*`, `logs/standardize_nslt1000.log`

## 6. Standardization command

CLI hien tai cua `scripts/01_standardize_videos.py` da ho tro `--config`.

Command da chay:

```powershell
.\.venv-rtmw310\Scripts\python.exe scripts\01_standardize_videos.py --config configs\preprocessing\standardize_nslt1000.yaml
```

No code changes were required.

## 7. Standardized outputs

### 7.1. Output paths

Frames:

- `data/datasets/WLASL/standardized/frames/nslt1000/train/`
- `data/datasets/WLASL/standardized/frames/nslt1000/val/`
- `data/datasets/WLASL/standardized/frames/nslt1000/test/`

Manifests:

- `data/datasets/WLASL/standardized/manifests/nslt1000_train.csv`
- `data/datasets/WLASL/standardized/manifests/nslt1000_val.csv`
- `data/datasets/WLASL/standardized/manifests/nslt1000_test.csv`
- `data/datasets/WLASL/standardized/manifests/nslt1000_all.csv`

Report/log tu sinh:

- `data/datasets/WLASL/standardized/reports/nslt1000_standardization_report.md`
- `data/datasets/WLASL/standardized/logs/standardize_nslt1000.log`

### 7.2. Manifest counts

- train manifest rows: `5001`
- val manifest rows: `1290`
- test manifest rows: `941`
- all manifest rows: `7232`

### 7.3. Folder and JPG counts

- train sample folders: `5001`
- val sample folders: `1290`
- test sample folders: `941`
- total JPG frames: `434337`

Chi tiet theo split:

- train JPG frames: `302181`
- val JPG frames: `78252`
- test JPG frames: `53904`

### 7.4. Status summary

- success samples: `7232`
- failed samples: `0`
- skipped samples: `0`
- `status` unique values: `["ok"]`
- `error_message` non-empty count: `0`

## 8. Sanity checks

### 8.1. Aggregate checks

- manifest `all = train + val + test`: `Yes` (`7232 = 5001 + 1290 + 941`)
- folder count khop manifest count:
  - train: `Yes`
  - val: `Yes`
  - test: `Yes`
- folder rong (khong co JPG):
  - train: `0`
  - val: `0`
  - test: `0`
- `error_message` non-empty: `0`

### 8.2. Manifest schema and output size

Trong `nslt1000_all.csv`, cac cot quan trong deu ton tai:

- `instance_uid`
- `sample_id`
- `video_id`
- `gloss`
- `class_id`
- `split`
- `raw_video_path`
- `frames_dir`
- `num_frames`
- `fps`
- `output_width`
- `output_height`
- `status`
- `error_message`

Output size trong manifest la duy nhat:

- `288 x 384`

### 8.3. Manual sample checks

Train:

- `sample_id=414` -> folder ton tai, `54` JPG, file dau `000001.jpg`, size `288x384`
- `sample_id=35527` -> folder ton tai, `30` JPG, file dau `000001.jpg`, size `288x384`
- `sample_id=69536` -> folder ton tai, `74` JPG, file dau `000001.jpg`, size `288x384`

Val:

- `sample_id=421` -> folder ton tai, `29` JPG, file dau `000001.jpg`, size `288x384`
- `sample_id=37611` -> folder ton tai, `89` JPG, file dau `000001.jpg`, size `288x384`
- `sample_id=69546` -> folder ton tai, `62` JPG, file dau `000001.jpg`, size `288x384`

Test:

- `sample_id=625` -> folder ton tai, `33` JPG, file dau `000001.jpg`, size `288x384`
- `sample_id=35344` -> folder ton tai, `43` JPG, file dau `000001.jpg`, size `288x384`
- `sample_id=69547` -> folder ton tai, `58` JPG, file dau `000001.jpg`, size `288x384`

Tat ca sample check thu cong deu co folder va JPG that.

## 9. Issues or warnings

No blocking issues found.

Canh bao non-blocking:

- `class_id_to_gloss.csv` co `1` dong `gloss_mismatch` cho `class_id=163` (`losear` vs `lose`).
- Trong khi chay standardization, console co mot vai warning giai ma `h264/mp4` tu backend video. Tuy nhien manifest cuoi van cho `7232/7232` sample `status=ok`, `0` failed, `0` `error_message`, `0` folder rong.

## 10. Files created or modified

Created:

- `configs/preprocessing/standardize_nslt1000.yaml`
- `reports/preprocessing/nslt1000_index_and_standardization_report.md`

Generated artifacts:

- `data/datasets/WLASL/standardized/frames/nslt1000/`
- `data/datasets/WLASL/standardized/manifests/nslt1000_train.csv`
- `data/datasets/WLASL/standardized/manifests/nslt1000_val.csv`
- `data/datasets/WLASL/standardized/manifests/nslt1000_test.csv`
- `data/datasets/WLASL/standardized/manifests/nslt1000_all.csv`
- `data/datasets/WLASL/standardized/reports/nslt1000_standardization_report.md`
- `data/datasets/WLASL/standardized/logs/standardize_nslt1000.log`

Code modified:

- None

## 11. Next recommended step

Buoc tiep theo de xuat, chua thuc hien trong task nay:

`standardized/nslt1000 -> pose/rtmw_l/wholebody_133/nslt1000 -> selected_27/selected_31 -> graph_tensors`
