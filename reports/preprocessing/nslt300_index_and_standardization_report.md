# NSLT300 Index and Standardization Report

## 1. Scope

Task nay chi cover hai tang:

- index verification cho `WLASL nslt300`
- standardization cho `WLASL nslt300`

Task nay khong dong vao:

- pose extraction
- skeleton inputs
- graph tensors
- training/evaluation
- region branch
- hand_poseflow branch
- placeholder models

Artifact `nslt100` da duoc giu nguyen; khong co buoc xoa, clean, hoac overwrite artifact `nslt100`.

## 2. Repository files inspected

Da doc va kiem tra cac file/chay cac artifact lien quan sau:

- `configs/preprocessing/index.yaml`
- `configs/preprocessing/standardize.yaml`
- `scripts/00_build_index.py`
- `scripts/01_standardize_videos.py`
- `src/slr/data/build_index.py`
- `src/slr/data/standardize_videos.py`
- `src/slr/data/manifests.py`
- `data/datasets/WLASL/index/subsets/nslt300/train.csv`
- `data/datasets/WLASL/index/subsets/nslt300/val.csv`
- `data/datasets/WLASL/index/subsets/nslt300/test.csv`
- `data/datasets/WLASL/index/subsets/nslt300/label_map.json`
- `data/datasets/WLASL/index/subsets_available/nslt300/train.csv`
- `data/datasets/WLASL/index/subsets_available/nslt300/val.csv`
- `data/datasets/WLASL/index/subsets_available/nslt300/test.csv`
- `data/datasets/WLASL/index/subsets_available/nslt300/label_map.json`
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

- `data/datasets/WLASL/index/subsets/nslt300/train.csv`
- `data/datasets/WLASL/index/subsets/nslt300/val.csv`
- `data/datasets/WLASL/index/subsets/nslt300/test.csv`
- `data/datasets/WLASL/index/subsets/nslt300/label_map.json`
- `data/datasets/WLASL/index/subsets_available/nslt300/train.csv`
- `data/datasets/WLASL/index/subsets_available/nslt300/val.csv`
- `data/datasets/WLASL/index/subsets_available/nslt300/test.csv`
- `data/datasets/WLASL/index/subsets_available/nslt300/label_map.json`
- `data/datasets/WLASL/index/master_instances.csv`
- `data/datasets/WLASL/index/available_instances.csv`
- `data/datasets/WLASL/index/missing_instances.csv`
- `data/datasets/WLASL/index/class_id_to_gloss.csv`
- `data/datasets/WLASL/index/video_to_split.csv`
- `data/datasets/WLASL/index/video_to_split_all.csv`

### 3.2. Row counts

`subsets/nslt300`:

- train: `3549`
- val: `901`
- test: `668`

`subsets_available/nslt300`:

- train: `1897`
- val: `446`
- test: `317`

### 3.3. Class coverage

- `label_map.json` trong `subsets/nslt300`: `num_classes = 300`
- `label_map.json` trong `subsets_available/nslt300`: `num_classes = 300`
- so `class_ids` trong ca hai file: `300`
- range `class_id`: `0..299`
- so class thuc te co mat trong manifest:
  - `subsets/nslt300`: `300`
  - `subsets_available/nslt300`: `300`

Ket luan: class coverage day du cho `nslt300`.

### 3.4. Integrity checks

Schema thuc te cua manifest subset la `video_path`, khong phai `raw_video_path`. Vi vay phan kiem tra path duoc doi chieu tren cot `video_path`.

Kiem tra tren `subsets/nslt300`:

- `video_path` rong/null: `0`
- `video_path` khong ton tai tren may: `2458`
- split null: `0`
- split sai (`train/val/test`): `0`
- `class_id` null: `0`
- `class_id` ngoai range `0..299`: `0`
- duplicate `sample_id`: `0`
- duplicate `instance_uid`: `0`
- `class_id -> gloss` co nhieu gloss: `0`
- `gloss -> class_id` co nhieu class: `0`

Kiem tra tren `subsets_available/nslt300`:

- `video_path` rong/null: `0`
- `video_path` khong ton tai tren may: `0`
- `is_present_locally = false`: `0`
- split null: `0`
- split sai (`train/val/test`): `0`
- `class_id` null: `0`
- `class_id` ngoai range `0..299`: `0`
- duplicate `sample_id`: `0`
- duplicate `instance_uid`: `0`
- `class_id -> gloss` co nhieu gloss: `0`
- `gloss -> class_id` co nhieu class: `0`

### 3.5. Gloss/class mapping check

Trong `data/datasets/WLASL/index/class_id_to_gloss.csv`:

- tong so dong: `2000`
- so dong co `nslt300_total > 0`: `300`
- so dong `nslt300` co note `gloss_mismatch`: `1`

Mismatch duy nhat:

- `class_id = 163`
- `class_list_gloss = losear`
- `master_gloss = lose`

Day la canh bao metadata mapping lich su trong `class_id_to_gloss.csv`, khong lam hong `subsets_available/nslt300`, khong tao duplicate, va khong chan stage standardization.

### 3.6. Index conclusion

Index `nslt300` da san sang.

- Khong can chay lai `build_index`.
- Khong co loi blocker trong `subsets_available/nslt300`.
- Viec `2458` path khong ton tai trong `subsets/nslt300` la expected, vi day la manifest tong gom ca sample khong co raw video local.

## 4. Config created

Da tao file config moi:

- `configs/preprocessing/standardize_nslt300.yaml`

Khac biet so voi `configs/preprocessing/standardize.yaml`:

- chi doi `input.subset` tu `nslt100` sang `nslt300`

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

Khong can sua `configs/preprocessing/standardize.yaml` goc.

Xac nhan:

- khong xoa artifact `nslt100`
- khong ghi de manifest `nslt100`
- khong sua code pipeline standardization

## 5. Standardization command

CLI hien tai cua `scripts/01_standardize_videos.py` da ho tro `--config`, nen khong can sua code.

Command da chay:

```powershell
.\.venv-rtmw310\Scripts\python.exe scripts\01_standardize_videos.py --config configs\preprocessing\standardize_nslt300.yaml
```

Khong co thay doi nao trong `src/slr/data/standardize_videos.py`.

## 6. Standardized outputs

### 6.1. Output paths

Frames:

- `data/datasets/WLASL/standardized/frames/nslt300/train/`
- `data/datasets/WLASL/standardized/frames/nslt300/val/`
- `data/datasets/WLASL/standardized/frames/nslt300/test/`

Manifests:

- `data/datasets/WLASL/standardized/manifests/nslt300_train.csv`
- `data/datasets/WLASL/standardized/manifests/nslt300_val.csv`
- `data/datasets/WLASL/standardized/manifests/nslt300_test.csv`
- `data/datasets/WLASL/standardized/manifests/nslt300_all.csv`

Report/log tu sinh:

- `data/datasets/WLASL/standardized/reports/nslt300_standardization_report.md`
- `data/datasets/WLASL/standardized/logs/standardize_nslt300.log`

### 6.2. Manifest counts

- train manifest rows: `1897`
- val manifest rows: `446`
- test manifest rows: `317`
- all manifest rows: `2660`
- check `all = train + val + test`: `2660 = 1897 + 446 + 317`

### 6.3. Frame folder counts

- train sample folders: `1897`
- val sample folders: `446`
- test sample folders: `317`

Tat ca deu khop voi so dong manifest cua split tuong ung.

### 6.4. JPG counts

- train JPG frames: `110814`
- val JPG frames: `25889`
- test JPG frames: `19239`
- total JPG frames: `155942`

### 6.5. Status summary

- success samples: `2660`
- failed samples: `0`
- skipped samples: `0`
- `status != ok`: `0`
- `error_message` khong rong: `0`

## 7. Sanity checks

### 7.1. Manifest schema

Trong `nslt300_all.csv`, cac cot quan trong deu ton tai:

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

### 7.2. Aggregate checks

- manifest `all` bang tong `train + val + test`: `Yes`
- folder count khop manifest count:
  - train: `Yes`
  - val: `Yes`
  - test: `Yes`
- sample folder rong (khong co JPG):
  - train: `0`
  - val: `0`
  - test: `0`

### 7.3. Manual sample checks

Train:

- `sample_id=414` -> folder ton tai, `54` JPG, file dau `000001.jpg`
- `sample_id=32949` -> folder ton tai, `83` JPG, file dau `000001.jpg`
- `sample_id=69534` -> folder ton tai, `54` JPG, file dau `000001.jpg`

Val:

- `sample_id=421` -> folder ton tai, `29` JPG, file dau `000001.jpg`
- `sample_id=34586` -> folder ton tai, `67` JPG, file dau `000001.jpg`
- `sample_id=69546` -> folder ton tai, `62` JPG, file dau `000001.jpg`

Test:

- `sample_id=625` -> folder ton tai, `33` JPG, file dau `000001.jpg`
- `sample_id=32157` -> folder ton tai, `47` JPG, file dau `000001.jpg`
- `sample_id=69547` -> folder ton tai, `58` JPG, file dau `000001.jpg`

Tat ca sample check thu cong deu co folder va JPG that.

## 8. Issues or warnings

No blocking issues found.

Canh bao non-blocking:

- `class_id_to_gloss.csv` co `1` dong `gloss_mismatch` cho `class_id=163` (`losear` vs `lose`).
- Trong khi chay standardization, console co mot vai warning giai ma `h264/mp4` tu backend video. Tuy nhien manifest cuoi van cho `2660/2660` sample `status=ok`, `0` failed, `0` empty folder, va `0` `error_message`.

## 9. Next recommended step

Buoc tiep theo de xuat, chua thuc hien trong task nay:

`standardized/nslt300 -> pose/rtmw_l/wholebody_133/nslt300 -> selected_27/selected_31 -> graph_tensors`
