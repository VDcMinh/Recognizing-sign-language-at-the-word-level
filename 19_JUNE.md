# Phan tich repo ngay 19/06/2026

## 1. Pham vi va cach doc

Tai lieu nay duoc viet sau khi doc cac nhom file quan trong trong workspace hien tai:

- Toan bo `src/slr/` dang duoc su dung cho preprocessing, training, evaluation, fusion, registry.
- Toan bo `scripts/` de xac dinh wrapper nao chi la vo mong, script nao chua logic that.
- Toan bo `configs/`, `model_registry/`, `tests/`, `docs/` va cac README/report chinh.
- Cac thu muc runtime nhu `data/`, `outputs/`, `artifacts/`, `packaging_outputs/`, `reports/` duoc doc theo lop du lieu va naming pattern.

Gioi han can noi ro:

- Cac file sinh tu dong nhu `.npz`, `.jpg`, `.pt`, `.pth`, `.zip`, `.csv` hang loat trong `data/`, `outputs/`, `artifacts/`, `packaging_outputs/` khong co y nghia neu liet ke tung file mot. Tai lieu nay phan tich chung theo vai tro, duong vao, duong ra va quy uoc dat ten.
- Cac file `__pycache__/`, `.pyc`, `.gitkeep` khong chua logic nghiep vu; chi duoc nhac toi de phan loai.
- Workspace hien tai co mot so thay doi local va file chua track; tai lieu nay mo ta repo theo trang thai dang co trong may, khong chi theo git history.

## 2. Ket luan nhanh

Repo nay khong con la scaffold don gian. No da co 3 truc chay that:

1. `skeleton`: day du nhat, di duoc tu raw metadata/video den graph tensor, train, eval, registry.
2. `regions`: da co pipeline build crop/tensor, dataset, model, train/eval, packaging; khong con o muc placeholder nhu mot so tai lieu cu.
3. `fusion`: da co paired dataset, gated feature fusion, training, evaluation, late fusion score-level, packaging.

Hinh dung tong quat:

```text
WLASL raw metadata + raw videos
-> index manifests
-> standardized frames/videos
-> RTMW-l wholebody pose
-> branch inputs
   -> skeleton graph tensors
   -> region tensors
-> branch training/evaluation
-> fusion pairing/training/evaluation
-> packaging + registry
```

Neu can biet "source of truth" de sua code, uu tien theo thu tu nay:

1. `src/slr/...`
2. `scripts/...`
3. `configs/...`
4. `model_registry/...`
5. `tests/...`
6. `docs/architecture`, `docs/training`

Tai lieu trong `docs/history/` va `reports/current/*_implementation_report.md` co gia tri lich su, nhung khong nen xem la mo ta chinh xac nhat cua production path hien tai.

## 3. Ban do repo theo vai tro

| Khu vuc | Vai tro chinh | Tinh chat |
| --- | --- | --- |
| `src/slr/` | logic nghiep vu that | production/support |
| `scripts/` | CLI wrappers va tool van hanh | production/support |
| `configs/` | tham so pipeline/train/package | production/support/history |
| `model_registry/` | dang ky model/artifact | production |
| `tests/` | kiem tra registry va union dataset | production support |
| `docs/architecture` | mo ta kien truc hien tai | support |
| `docs/training` | huong dan train va integration | support |
| `docs/history` | snapshot lich su, tong ket tung moc | history |
| `data/` | raw data + derived layers | generated/source data |
| `outputs/` | output train/eval run | generated |
| `artifacts/` | workspace danh cho fusion/late-fusion/checks | generated |
| `packaging_outputs/` | bo dong goi trung gian/cuoi | generated |
| `reports/` | report ky thuat va ket qua theo giai doan | generated/support |
| `archive/`, `scripts/archive/`, `configs/archive/` | scaffold va legacy snapshot | history |
| `experiments/` | khung danh cho experiment artifacts | placeholder |
| `UI/` | hien tai trong workspace nay rong | placeholder |

## 4. Luong hoat dong end-to-end

### 4.1 Tu raw data den index

- Dau vao:
  - `data/datasets/WLASL/raw/metadata/WLASL_v0.3.json`
  - `data/datasets/WLASL/raw/metadata/wlasl_class_list.txt`
  - `data/datasets/WLASL/raw/metadata/missing.txt`
  - `data/datasets/WLASL/raw/metadata/nslt_100.json`
  - `data/datasets/WLASL/raw/metadata/nslt_300.json`
  - `data/datasets/WLASL/raw/metadata/nslt_1000.json`
  - `data/datasets/WLASL/raw/metadata/nslt_2000.json`
  - `data/datasets/WLASL/raw/videos/*.mp4`
- File xu ly:
  - `scripts/preprocess/00_build_index.py`
  - `src/slr/data/build_index.py`
- Dau ra:
  - `data/datasets/WLASL/index/master_instances.csv`
  - `available_instances.csv`, `missing_instances.csv`, `nslt_only_instances.csv`
  - `class_id_to_gloss.csv`, `video_to_split*.csv`
  - `subsets/` va `subsets_available/`
  - `reports/`, `logs/`

### 4.2 Tu index den standardized

- Dau vao:
  - `data/datasets/WLASL/index/subsets_available/...`
  - raw videos
- File xu ly:
  - `scripts/preprocess/01_standardize_videos.py`
  - `src/slr/data/standardize_videos.py`
- Dau ra:
  - `data/datasets/WLASL/standardized/frames/<subset>/<split>/<sample_id>/*.jpg`
  - `data/datasets/WLASL/standardized/videos/...` neu config bat
  - `data/datasets/WLASL/standardized/manifests/*.csv`
  - `reports/*_standardization_report.md`, `logs/`

### 4.3 Tu standardized den pose

- Dau vao:
  - standardized manifests
  - `checkpoints/pose/rtmw_l/*.py`
  - `checkpoints/pose/rtmw_l/*.pth`
- File xu ly:
  - `scripts/preprocess/02_extract_pose_rtmw.py`
  - `src/slr/pose/extract_rtmw.py`
- Dau ra:
  - `data/datasets/WLASL/pose/rtmw_l/wholebody_133/<subset>/<split>/*.npz`
  - `manifests/*.csv`
  - `reports/*_pose_quality_report.md`

### 4.4 Tu pose den skeleton inputs

- Dau vao:
  - pose manifests va pose `.npz`
- File xu ly:
  - `scripts/preprocess/03_build_skeleton_inputs.py`
  - `src/slr/branches/skeleton/build_inputs.py`
- Dau ra:
  - `branch_inputs/skeleton/rtmw_l/selected_27/...`
  - `branch_inputs/skeleton/rtmw_l/selected_31/...`
  - `normalized/...`
  - `graph_tensors/...`
  - skeleton manifests/report/logs

### 4.5 Tu pose + standardized den regions inputs

- Dau vao:
  - standardized manifests + frame folders
  - pose manifests + pose `.npz`
- File xu ly:
  - `scripts/preprocess/04_build_region_inputs.py`
  - `src/slr/branches/regions/build_crops.py`
- Dau ra:
  - `branch_inputs/regions/.../tensors/<subset>/<split>/*.npz`
  - preview images
  - region manifests/report/metadata/logs

### 4.6 Training branch

- Skeleton:
  - `scripts/train/train_skeleton.py`
  - `src/slr/branches/skeleton/train.py`
  - output thuong vao `outputs/skeleton/<run_name>/`
- Regions:
  - `scripts/train/train_regions.py`
  - `src/slr/branches/regions/train.py`
  - output thuong vao `outputs/regions/<run_name>/`
- Fusion:
  - `scripts/train/train_gated_fusion.py`
  - `src/slr/branches/fusion/train.py`
  - output thuong vao `outputs/fusion/<run_name>/`

### 4.7 Evaluation va packaging

- Skeleton eval:
  - `scripts/evaluate/evaluate_skeleton.py`
- Regions eval:
  - `scripts/evaluate/evaluate_regions.py`
- Fusion eval:
  - `scripts/evaluate/evaluate_gated_fusion.py`
- Score-level late fusion:
  - `scripts/evaluate/evaluate_skeleton_region_late_fusion.py`
  - output thuong vao `artifacts/fusion/<subset>/`
- Packaging/verification:
  - `scripts/package/*`
  - `scripts/verify/*`
  - `src/slr/branches/fusion/package_support.py`
  - `scripts/common/regions_nslt1000_incremental_common.py`

## 5. Phan tich file nguon chinh trong repo

## 5.1 Root files

| File | Tac dung | Ghi chu luong |
| --- | --- | --- |
| `README.md` | tong quan repo, 3 branch active, command chinh, dinh huong project | file mo dau de hieu repo |
| `pyproject.toml` | khai bao package `slr` theo `src` layout | phuc vu import/cai editable |
| `requirements.txt` | dependency co ban cho data/train/report | moi truong chinh |
| `requirements-rtmw.txt` | dependency nang cho MMPose/RTMW-l | can cho pose extraction |
| `requirements-kaggle-train.txt` | dependency toi thieu cho train/package tren Kaggle | dung cho bundle runtime |
| `.gitignore` | bo qua data/artifact/bundle/checkpoint/cache | ngan git bi phinh |
| `sitecustomize.py` | them `src/` vao `sys.path` | giup script chay tu root |

## 5.2 Import bootstrap va package roots

| File | Tac dung | Ghi chu |
| --- | --- | --- |
| `slr/__init__.py` | import shim de `import slr` hoat dong ngay ca khi chua install package | bridge root -> `src/slr` |
| `src/slr/__init__.py` | khai bao package version/co ban | diem vao package |
| `src/slr/branches/__init__.py` | danh dau namespace cho cac branch | to chuc module |
| `src/slr/data/__init__.py` | namespace data | khong chua nghiep vu |
| `src/slr/inference/__init__.py` | namespace inference | khong chua nghiep vu |
| `src/slr/pose/__init__.py` | namespace pose | khong chua nghiep vu |
| `src/slr/registry/__init__.py` | export loader/schema/validation registry | support model registry |
| `src/slr/training/__init__.py` | namespace training utils | support loop train |
| `src/slr/utils/__init__.py` | namespace utils | helper chung |

## 5.3 `src/slr/utils/`

| File | Tac dung | Luong su dung |
| --- | --- | --- |
| `src/slr/utils/bbox.py` | parse, clip, expand, serialize bounding box | duoc goi boi standardize va region crop |
| `src/slr/utils/image.py` | doc/ghi anh, resize letterbox, crop, tao black image | standardize, region crop, visualization |
| `src/slr/utils/io.py` | doc/ghi JSON, YAML, text, CSV; tao folder; remap path | duoc su dung khap repo |
| `src/slr/utils/logging.py` | tao logger file/console thong nhat | moi stage train/preprocess |
| `src/slr/utils/seed.py` | seed Python/NumPy muc don gian | helper phu |
| `src/slr/utils/video.py` | probe video, read frames, write video | standardization va smoke tools |

## 5.4 `src/slr/data/`

| File | Tac dung | Luong vao/ra |
| --- | --- | --- |
| `src/slr/data/manifests.py` | dinh nghia schema cot chuan cho master/index/standardized/pose/skeleton/regions manifests | file schema trung tam |
| `src/slr/data/validation.py` | check required columns, null, split values, column order | duoc goi boi cac stage doc/ghi CSV |
| `src/slr/data/build_index.py` | build data index tu raw metadata + local videos | viet vao `data/datasets/WLASL/index/` |
| `src/slr/data/standardize_videos.py` | crop/resize/letterbox video thanh standardized frames/videos + manifests | viet vao `standardized/` |

Chi tiet `build_index.py`:

- Parse config `configs/preprocessing/index/index.yaml`.
- Flatten `WLASL_v0.3.json` thanh `master_instances.csv`.
- Doi soat class list, split mapping, local video coverage.
- Xay subset manifests cho `nslt100`, `nslt300`, `nslt1000`, `nslt2000`.
- Tao report coverage theo split va class.

Chi tiet `standardize_videos.py`:

- Doc split manifest da co video.
- Resolve frame range hop le va bbox.
- Doc khung hinh tu video raw.
- Crop nguoi ky, resize ve kich thuoc chuan, letterbox neu can.
- Ghi frames, optional video, manifest, report, log.

## 5.5 `src/slr/pose/`

| File | Tac dung | Luong su dung |
| --- | --- | --- |
| `src/slr/pose/pose_schema.py` | dinh nghia wholebody_133, cac region indices, keypoint sets `selected_27`, `selected_31`, mo ta `selected_49` | nguon chuan cho skeleton va regions |
| `src/slr/pose/keypoint_selection.py` | rut gon 133 keypoint thanh mot keypoint set nho hon | duoc goi boi `build_inputs.py` |
| `src/slr/pose/pose_normalization.py` | center, normalize `x/y`, fit va normalize confidence, sanitize NaN/Inf | duoc goi boi skeleton preprocessing |
| `src/slr/pose/pose_quality.py` | tinh coverage va confidence theo body/face/hands | duoc goi boi pose report |
| `src/slr/pose/extract_rtmw.py` | pipeline chay MMPose RTMW-l tren standardized frames, chon primary signer, ghi pose `.npz`, manifest, report | viet vao `pose/rtmw_l/` |

`extract_rtmw.py` la mot production stage day du:

- Tu tim hoac nhan duong dan config/checkpoint RTMW-l.
- Khoi tao `MMPoseInferencer`.
- Chay tung frame, neu nhieu nguoi thi chon nguoi co box/chuan tin cay tot nhat.
- Ket hop frame-level ket qua thanh tensor `(T, 133, 3)`.
- Ghi metadata ve quality, valid frame ratio, region confidence.

## 5.6 `src/slr/branches/skeleton/`

| File | Tac dung | Luong su dung |
| --- | --- | --- |
| `src/slr/branches/skeleton/__init__.py` | namespace branch skeleton | to chuc module |
| `src/slr/branches/skeleton/transforms.py` | fix sequence length, pad/trim, doi `(T,V,C)` sang `C,T,V,M` | preprocessing skeleton |
| `src/slr/branches/skeleton/label_smoothing.py` | helper Standard Label Smoothing va Language Label Smoothing | logic ho tro loss |
| `src/slr/branches/skeleton/graph.py` | dinh nghia graph topology `selected_27`, `selected_31`, adjacency `uniform/spatial` | duoc model va train su dung |
| `src/slr/branches/skeleton/dataset.py` | dataset loader doc skeleton manifests, resolve tensor path, validate shape, build label maps | train/eval skeleton va fusion |
| `src/slr/branches/skeleton/build_inputs.py` | xay selected/normalized/graph tensors tu shared pose | viet vao `branch_inputs/skeleton/...` |
| `src/slr/branches/skeleton/train.py` | CLI train/eval, dataloader, graph+model build, epoch loop, checkpoint, final eval | output `outputs/skeleton/...` |

`dataset.py` la mot file cuc ky quan trong:

- Doc CSV theo split.
- Chi giu `status=ok`.
- Resolve `graph_tensor_path` linh hoat giua local workspace, package root, data root.
- Validate shape ky vong cho `selected_27` hoac `selected_31`.
- Tra ve tensor + `class_id` + metadata.

`build_inputs.py` la cau noi tu pose sang skeleton training:

- Doc pose manifests.
- Fit confidence scale tren split train.
- Chon keypoints giam chieu.
- Normalize `x/y/confidence`.
- Co dinh sequence ve do dai cau hinh.
- Ghi ba lop artifact:
  - selected keypoints
  - normalized keypoints
  - graph tensors train-ready

`train.py` la production training loop that:

- Resolve config tu YAML + CLI override.
- Xay `SkeletonGraphDataset`, `DataLoader`, `SkeletonGraph`.
- Build model qua `build_skeleton_model`.
- Dung loss tu `src/slr/training/losses.py`.
- Track metrics top-k, luu `best.pt` va `last.pt`.
- Cuoi run nap lai best checkpoint va danh gia split test.

### Models skeleton

| File | Tac dung | Ghi chu |
| --- | --- | --- |
| `src/slr/branches/skeleton/models/__init__.py` | factory chon model skeleton theo config | ho tro `simple_stgcn`, `stgcnpp` |
| `src/slr/branches/skeleton/models/simple_stgcn.py` | baseline ST-GCN nhe, tu cai dat | de xac minh pipeline va baseline |
| `src/slr/branches/skeleton/models/stgcnpp.py` | cai dat ST-GCN++ local, gom spatial graph conv + multi-scale temporal conv | backbone skeleton manh hon, dang duoc dung that |

## 5.7 `src/slr/branches/regions/`

Day la nhanh da co implementation that, khong con la scaffold.

| File | Tac dung | Luong su dung |
| --- | --- | --- |
| `src/slr/branches/regions/__init__.py` | namespace regions | to chuc module |
| `src/slr/branches/regions/region_schema.py` | dinh nghia ten regions, status/bbox source constants, shape metadata | schema cho region branch |
| `src/slr/branches/regions/crop_utils.py` | tinh bbox mat/tay tu wholebody133, crop va resize, fallback black crop | duoc `build_crops.py` goi |
| `src/slr/branches/regions/transforms.py` | normalize va augmentation cho region clip tensor | duoc dataset/train goi |
| `src/slr/branches/regions/dataset.py` | doc region manifests, resolve `.npz`, cat active regions, validate shape, tra `valid_mask` va metadata | train/eval regions va fusion |
| `src/slr/branches/regions/build_crops.py` | pipeline tao region tensor tu frames + pose, ke ca preview/report/quality metrics | viet vao `branch_inputs/regions/...` |
| `src/slr/branches/regions/train.py` | train/eval regions branch, support early stopping, augmentation, checkpoint, final eval | output `outputs/regions/...` |

### Models regions

| File | Tac dung | Ghi chu |
| --- | --- | --- |
| `src/slr/branches/regions/models/__init__.py` | factory chon model regions | build `region_cnn_gru` hoac `region_resnet18_gru` |
| `src/slr/branches/regions/models/region_cnn_gru.py` | model CNN encoder + temporal GRU thuan custom | baseline va phien ban nhe |
| `src/slr/branches/regions/models/region_resnet18_gru.py` | ResNet18 frame encoder + temporal GRU, co freeze/fine-tune backbone | model regions chinh hien tai |

`build_crops.py` lam nhieu viec:

- Ghep standardized manifest va pose manifest theo sample.
- Doc frames va pose.
- Tinh bbox cho `left_hand`, `right_hand`, `face`.
- Crop, resize, danh dau valid/fallback.
- Ghi tensor 5D va metadata ve quality.
- Tao preview image, low-quality report, split report.

`train.py` cho regions co nhieu diem giong skeleton, nhung bo sung:

- `active_regions`
- `valid_mask`
- augmentation tren clip images
- monitor metric co early stopping
- cau hinh optimizer/scheduler mang tinh CV hon

## 5.8 `src/slr/branches/fusion/`

Fusion branch la lop ket hop skeleton va regions, dong thoi la trung tam cho packaging/gated-fusion.

| File | Tac dung | Luong su dung |
| --- | --- | --- |
| `src/slr/branches/fusion/__init__.py` | namespace fusion | to chuc module |
| `src/slr/branches/fusion/dataset.py` | paired dataset ghep sample skeleton va regions theo `sample_id` da normalize | train/eval fusion |
| `src/slr/branches/fusion/build.py` | doc fusion config, load branch configs/checkpoints, build frozen backbones + fusion head | duoc train/eval fusion goi |
| `src/slr/branches/fusion/evaluate.py` | danh gia checkpoint fusion tren mot split | output JSON eval |
| `src/slr/branches/fusion/train.py` | training loop cho gated feature fusion, chi train head, ghi gate statistics | output `outputs/fusion/...` |
| `src/slr/branches/fusion/package_support.py` | helper lon cho audit manifests, resolve tensor sources, inspect checkpoints/configs, tao runtime package metadata | duoc script package/verify goi |

### Models fusion

| File | Tac dung | Ghi chu |
| --- | --- | --- |
| `src/slr/branches/fusion/models/__init__.py` | export model fusion | namespace |
| `src/slr/branches/fusion/models/gated_feature_fusion.py` | module hop nhat features skeleton + regions bang gate hoc duoc | model fusion chinh hien tai |

`train.py` cua fusion hoat dong nhu sau:

- Doc config fusion va branch tham chieu.
- Tai skeleton backbone va regions backbone tu checkpoint.
- Freeze backbone, chi toi uu fusion head.
- Kiem tra shape cap input ghép.
- Track gate statistics de biet branch nao dang duoc uu tien.
- Ghi checkpoint va summary.

`package_support.py` la file support rat quan trong:

- normalize/canonical sample id
- inspect checkpoint state_dict
- inspect config branch
- audit split manifests skeleton/regions
- xay alignment manifests cho package
- estimate disk usage
- materialize tensor files vao package
- tao runtime config/readme/metadata

## 5.9 `src/slr/training/`

| File | Tac dung | Luong su dung |
| --- | --- | --- |
| `src/slr/training/metrics.py` | `AverageMeter`, top-k accuracy cho numpy/torch | duoc moi train loop goi |
| `src/slr/training/losses.py` | factory loss tu config | skeleton, regions, fusion |
| `src/slr/training/optim.py` | factory optimizer/scheduler | skeleton, regions, fusion |
| `src/slr/training/checkpointing.py` | save/load checkpoint kieu `state_dict` thuần + metadata | skeleton, regions, fusion |
| `src/slr/training/seed.py` | set random seed Python/NumPy/PyTorch | train loops |
| `src/slr/training/wandb_utils.py` | init/log/finish W&B an toan, fallback neu chua setup | train loops |

Luu y:

- Trong repo hien tai, training loop that nam o `src/slr/branches/skeleton/train.py`, `src/slr/branches/regions/train.py`, `src/slr/branches/fusion/train.py`.
- Thu muc `src/slr/training/` dong vai tro shared utilities, khong phai noi gom mot mega-trainer chung.

## 5.10 `src/slr/registry/`

| File | Tac dung | Ghi chu |
| --- | --- | --- |
| `src/slr/registry/schema.py` | dataclass cho artifact ref, bundle, model identity, registry record | schema logic registry |
| `src/slr/registry/validation.py` | validate duong dan, class path, artifact refs, loaded registry | bao ve registry |
| `src/slr/registry/loader.py` | load `registry.yaml` + tung `model.yaml`, tao `LoadedRegistry` | duoc test va van hanh goi |

Registry layer dung de:

- dang ky model co san
- tro den config, checkpoint, manifests, package roots
- tao mot chi muc on dinh cho huan luyen, demo, packaging hoac external handoff

## 5.11 `src/slr/inference/`

| File | Tac dung | Ghi chu |
| --- | --- | --- |
| `src/slr/inference/visualize_prediction.py` | utility visualize keypoint/prediction hien co trong workspace | mang tinh debug/phan tich, khong phai infer end-user hoan chinh |

Nhan xet:

- Thu muc inference hien rat nho.
- No chua la mot serving/inference stack day du.
- Gia tri chinh hien nay la visualization va debug mau du lieu.

## 6. Phan tich `scripts/`

## 6.1 Scripts wrapper mong

Nhung file nay gan nhu khong chua nghiep vu, chi la diem vao CLI:

| File | Goi vao |
| --- | --- |
| `scripts/preprocess/00_build_index.py` | `slr.data.build_index.main` |
| `scripts/preprocess/01_standardize_videos.py` | `slr.data.standardize_videos.main` |
| `scripts/preprocess/02_extract_pose_rtmw.py` | `slr.pose.extract_rtmw.main` |
| `scripts/preprocess/03_build_skeleton_inputs.py` | `slr.branches.skeleton.build_inputs.main` |
| `scripts/preprocess/04_build_region_inputs.py` | `slr.branches.regions.build_crops.main` |
| `scripts/train/train_skeleton.py` | `slr.branches.skeleton.train.main` |
| `scripts/train/train_regions.py` | `slr.branches.regions.train.main` |
| `scripts/train/train_gated_fusion.py` | `slr.branches.fusion.train.main` |
| `scripts/evaluate/evaluate_skeleton.py` | `slr.branches.skeleton.train.evaluate_main` |
| `scripts/evaluate/evaluate_regions.py` | `slr.branches.regions.train.evaluate_main` |
| `scripts/evaluate/evaluate_gated_fusion.py` | `slr.branches.fusion.evaluate.main` |

## 6.2 Script evaluation dac biet

| File | Tac dung |
| --- | --- |
| `scripts/evaluate/evaluate_skeleton_region_late_fusion.py` | export logits skeleton/regions, align theo `sample_id`, sweep alpha, chon alpha tot nhat tren validation, danh gia test, ghi report late fusion |

Day la file khong chi wrapper. No chua logic score-level late fusion that.

## 6.3 Scripts verify/sanity-check

| File | Tac dung |
| --- | --- |
| `scripts/verify/check_skeleton_dataset.py` | kiem tra skeleton manifest, sample tensor, batch shape, graph topology |
| `scripts/verify/check_region_dataset.py` | kiem tra region manifest, tensor shape, quality columns, preview path, loader batch |
| `scripts/verify/check_fusion_workspace.py` | kiem tra workspace fusion co du file bat buoc |
| `scripts/verify/check_gated_fusion_setup.py` | smoke-check kha nang build/load fusion components |
| `scripts/verify/check_gated_fusion_nslt1000_packaging_requirements.py` | audit yeu cau package fusion nslt1000 |
| `scripts/verify/check_gated_fusion_nslt300_packaging_requirements.py` | audit yeu cau package fusion nslt300 va tao candidate config |
| `scripts/verify/check_regions_nslt1000_extraction.py` | kiem tra ket qua region extraction cho nslt1000 |
| `scripts/verify/check_regions_nslt1000_incremental_feasibility.py` | danh gia kha thi incremental pipeline cho regions nslt1000 |
| `scripts/verify/check_regions_nslt1000_incremental_pipeline_requirements.py` | preflight check truoc khi chay pipeline incremental |
| `scripts/verify/check_regions_nslt1000_incremental_progress.py` | tong hop tien do extraction incremental |
| `scripts/verify/check_regions_nslt1000_union_dataset.py` | validate union manifests/tensors/loader sau khi hop nhat |
| `scripts/verify/verify_gated_fusion_nslt1000_package.py` | verify package fusion nslt1000 da tao |
| `scripts/verify/verify_nslt100_branch_inputs_package.py` | verify package skeleton/regions nslt100 |
| `scripts/verify/verify_regions_nslt1000_incremental_package.py` | verify incremental regions package |

## 6.4 Scripts preprocess bo sung cho `regions`

| File | Tac dung |
| --- | --- |
| `scripts/common/regions_nslt1000_incremental_common.py` | helper dung chung cho feasibility check, missing manifests, union manifests, packaging va progress tracking cua regions nslt1000 incremental |
| `scripts/preprocess/prepare_regions_branch_inputs.py` | wrapper/orchestrator de chuan bi root, config va tham so cho regions branch inputs |
| `scripts/preprocess/build_regions_nslt1000_missing_manifests.py` | tao manifests cho cac sample nslt1000 con thieu tensor |
| `scripts/preprocess/build_regions_nslt1000_union_manifests.py` | hop manifest reusable + incremental thanh bo union |
| `scripts/preprocess/extract_regions_nslt1000_missing_only.py` | chay `build_crops` chi tren phan con thieu, co state/progress/disk checks |
| `scripts/preprocess/materialize_regions_nslt1000_kaggle_manifests.py` | bien logical manifests thanh manifest co duong dan tensor cu the cho package Kaggle |
| `scripts/preprocess/merge_nslt300_pose_into_nslt1000.py` | merge pose layer nslt300 vao nslt1000 khi dung overlap sample |

## 6.5 Scripts package

| File | Tac dung |
| --- | --- |
| `scripts/package/prepare_hf_skeleton_bundle.py` | dong goi skeleton branch inputs cho Hugging Face |
| `scripts/package/prepare_hf_regions_bundle.py` | dong goi regions branch inputs cho Hugging Face |
| `scripts/package/prepare_kaggle_bundle.py` | tao repo/data subset package cho Kaggle preprocessing |
| `scripts/package/prepare_kaggle_nslt1000_remaining_bundle.py` | dong goi phan con lai cua nslt1000 cho workflow Kaggle |
| `scripts/package/package_nslt100_branch_inputs.py` | dong goi skeleton + regions inputs cho nslt100 |
| `scripts/package/package_regions_nslt300_kaggle_dataset.py` | package regions nslt300 de train/eval tren Kaggle |
| `scripts/package/package_regions_nslt1000_incremental_kaggle_dataset.py` | package bo regions nslt1000 incremental |
| `scripts/package/create_regions_nslt1000_incremental_zip.py` | zip package incremental va verify hash/noi dung |
| `scripts/package/package_gated_fusion_nslt100_kaggle_dataset.py` | package fusion nslt100 |
| `scripts/package/package_gated_fusion_nslt300_kaggle_dataset.py` | package fusion nslt300 |
| `scripts/package/package_gated_fusion_nslt1000_kaggle_dataset.py` | package fusion nslt1000 |

## 6.6 Scripts dev

| File | Tac dung |
| --- | --- |
| `scripts/dev/test_rtmw_mmpose_video.py` | smoke-test RTMW-l tren mot video, sinh report |
| `scripts/dev/visualize_selected_27_samples.py` | visualize selected keypoints len standardized frames, tao contact sheet/report |
| `scripts/dev/cleanup_region_outputs.py` | don dep outputs regions theo root/subset, co safety checks |

## 6.7 Sitecustomize va shim trong `scripts/`

| File/family | Tac dung |
| --- | --- |
| `scripts/sitecustomize.py` | bo sung `src/` vao `sys.path` |
| `scripts/preprocess/sitecustomize.py` | tuong tu cho subfolder preprocess |
| `scripts/train/sitecustomize.py` | tuong tu cho subfolder train |
| `scripts/evaluate/sitecustomize.py` | tuong tu cho subfolder evaluate |
| `scripts/verify/sitecustomize.py` | tuong tu cho subfolder verify |
| `scripts/package/sitecustomize.py` | tuong tu cho subfolder package |
| `scripts/dev/sitecustomize.py` | tuong tu cho subfolder dev |
| `scripts/slr/__init__.py` | shim de import package tu trong `scripts/` |

## 7. Phan tich `configs/`

`configs/` chia dung theo tang, khong theo model don le. Day la mot diem manh cua repo.

## 7.1 Config data va preprocessing

| File/nhom | Tac dung |
| --- | --- |
| `configs/dataset/wlasl.yaml` | dataset root settings co ban |
| `configs/preprocessing/index/index.yaml` | config build index tu raw metadata |
| `configs/preprocessing/standardize/standardize_nslt100.yaml` | standardize subset nslt100 |
| `configs/preprocessing/standardize/standardize_nslt300.yaml` | standardize subset nslt300 |
| `configs/preprocessing/standardize/standardize_nslt1000.yaml` | standardize subset nslt1000 |
| `configs/preprocessing/pose/pose_rtmw_l.yaml` | extract pose RTMW-l tren local workspace |
| `configs/preprocessing/pose/pose_rtmw_l_kaggle.yaml` | extract pose RTMW-l trong layout Kaggle/package |
| `configs/preprocessing/regions/region_crops_nslt100.yaml` | xay region tensors cho nslt100 |
| `configs/preprocessing/regions/region_crops_nslt300.yaml` | xay region tensors cho nslt300 |
| `configs/preprocessing/regions/region_crops_nslt1000.yaml` | xay region tensors cho nslt1000 |

## 7.2 Config build skeleton inputs

| File/nhom | Tac dung |
| --- | --- |
| `configs/build_inputs/skeleton/nslt100/selected_27.yaml` | build graph tensors nslt100 voi `selected_27` |
| `configs/build_inputs/skeleton/nslt100/selected_31.yaml` | build graph tensors nslt100 voi `selected_31` |
| `configs/build_inputs/skeleton/nslt300/selected_27.yaml` | build graph tensors nslt300 voi `selected_27` |
| `configs/build_inputs/skeleton/nslt300/selected_31.yaml` | build graph tensors nslt300 voi `selected_31` |
| `configs/build_inputs/skeleton/nslt1000/selected_27.yaml` | build graph tensors nslt1000 voi `selected_27` |
| `configs/build_inputs/skeleton/nslt1000/selected_31.yaml` | build graph tensors nslt1000 voi `selected_31` |

## 7.3 Config train skeleton

| Nhom file | Y nghia |
| --- | --- |
| `configs/train/skeleton/nslt100/selected_27/simple_stgcn_ce.yaml` | baseline simple ST-GCN, CE |
| `configs/train/skeleton/nslt100/selected_27/stgcnpp_ce.yaml` | ST-GCN++ cho nslt100 selected_27 |
| `configs/train/skeleton/nslt100/selected_27/stgcnpp_standardls_eps005.yaml` | ST-GCN++ + StandardLS eps 0.05 |
| `configs/train/skeleton/nslt100/selected_27/stgcnpp_standardls_eps01.yaml` | ST-GCN++ + StandardLS eps 0.1 |
| `configs/train/skeleton/nslt100/selected_27/stgcnpp_standardls_eps03.yaml` | ST-GCN++ + StandardLS eps 0.3 |
| `configs/train/skeleton/nslt100/selected_31/simple_stgcn_ce.yaml` | baseline simple ST-GCN cho selected_31 |
| `configs/train/skeleton/nslt100/selected_31/stgcnpp_ce.yaml` | ST-GCN++ selected_31 |
| `configs/train/skeleton/nslt100/selected_31/stgcnpp_standardls_eps005.yaml` | ST-GCN++ + StandardLS eps 0.05 |
| `configs/train/skeleton/nslt100/selected_31/stgcnpp_standardls_eps01.yaml` | ST-GCN++ + StandardLS eps 0.1 |
| `configs/train/skeleton/nslt100/selected_31/stgcnpp_standardls_eps03.yaml` | ST-GCN++ + StandardLS eps 0.3 |
| `configs/train/skeleton/nslt300/selected_27/stgcnpp_ce.yaml` | ST-GCN++ nslt300 selected_27 |
| `configs/train/skeleton/nslt300/selected_27/stgcnpp_standardls_eps03.yaml` | bien the StandardLS nslt300 selected_27 |
| `configs/train/skeleton/nslt300/selected_31/stgcnpp_ce.yaml` | ST-GCN++ nslt300 selected_31 |
| `configs/train/skeleton/nslt300/selected_31/stgcnpp_standardls_eps01.yaml` | StandardLS eps 0.1 |
| `configs/train/skeleton/nslt300/selected_31/stgcnpp_standardls_eps02.yaml` | StandardLS eps 0.2 |
| `configs/train/skeleton/nslt300/selected_31/stgcnpp_standardls_eps03.yaml` | StandardLS eps 0.3 |
| `configs/train/skeleton/nslt1000/selected_27/stgcnpp_ce.yaml` | ST-GCN++ nslt1000 selected_27 |
| `configs/train/skeleton/nslt1000/selected_27/stgcnpp_standardls_eps01.yaml` | StandardLS eps 0.1 |
| `configs/train/skeleton/nslt1000/selected_27/stgcnpp_standardls_eps02.yaml` | StandardLS eps 0.2 |
| `configs/train/skeleton/nslt1000/selected_27/stgcnpp_standardls_eps03.yaml` | StandardLS eps 0.3 |
| `configs/train/skeleton/nslt1000/selected_31/stgcnpp_ce.yaml` | ST-GCN++ nslt1000 selected_31 |
| `configs/train/skeleton/nslt1000/selected_31/stgcnpp_standardls_eps01.yaml` | StandardLS eps 0.1 |
| `configs/train/skeleton/nslt1000/selected_31/stgcnpp_standardls_eps02.yaml` | StandardLS eps 0.2 |
| `configs/train/skeleton/nslt1000/selected_31/stgcnpp_standardls_eps03.yaml` | StandardLS eps 0.3 |

## 7.4 Config train regions

| Nhom file | Y nghia |
| --- | --- |
| `configs/train/regions/nslt100/face_hands/*.yaml` | train regions nslt100 voi ca face + hai hand |
| `configs/train/regions/nslt100/hands_only/*.yaml` | ablation dung chi hands |
| `configs/train/regions/nslt300/face_hands/*.yaml` | train regions nslt300 |
| `configs/train/regions/nslt1000/debug/region_resnet18_gru_smoke_ce.yaml` | smoke config cho nslt1000 |
| `configs/train/regions/nslt1000/full/*.yaml` | train full regions nslt1000 |
| `configs/train/regions/nslt1000/incremental/region_resnet18_gru_incremental_kaggle_ce.yaml` | train tren bo incremental/package layout |
| `configs/train/regions/nslt1000/incremental/*.template` | template tao config incremental runtime |
| `configs/train/regions/nslt1000/union/*.yaml` | train tren union dataset sau khi hop reusable + incremental |

## 7.5 Config train fusion

| File/nhom | Tac dung |
| --- | --- |
| `configs/train/fusion/gated_feature/nslt100/gated_feature_fusion_ce.yaml` | fusion feature-level cho nslt100 |
| `configs/train/fusion/gated_feature/nslt300/gated_feature_fusion_ce.yaml` | fusion feature-level cho nslt300 |
| `configs/train/fusion/gated_feature/nslt1000/gated_feature_fusion_ce.yaml` | fusion feature-level cho nslt1000 |
| `configs/train/fusion/gated_feature/nslt1000/gated_feature_fusion_kaggle_ce.yaml` | bien the cho package/Kaggle layout |
| `configs/train/fusion/late_fusion/nslt100/skeleton_regions_late_fusion.yaml` | config cho script score-level late fusion |

## 7.6 Config experiments va archive

| File/nhom | Tac dung |
| --- | --- |
| `configs/experiments/skeleton/nslt100_debug.yaml` | config debug experiment skeleton |
| `configs/archive/legacy/*.yaml` | cac moc config cu, luu de tham khao lich su |
| `configs/archive/deprecated/regions_cnn_gru/*.yaml` | cac bien the cu cua regions CNN-GRU, khong phai source of truth hien tai |
| `configs/archive/deprecated/regions_cnn_gru/README.md` | giai thich config cu cua nhanh regions |

## 8. `model_registry/`

| File | Tac dung |
| --- | --- |
| `model_registry/README.md` | mo ta schema registry va cach them model |
| `model_registry/registry.yaml` | index chinh tham chieu toi cac model record |
| `model_registry/models/skeleton_nslt1000_sel31/model.yaml` | record model skeleton nslt1000 selected_31 |
| `model_registry/models/regions_nslt1000_face_hands/model.yaml` | record model regions nslt1000 face_hands |
| `model_registry/models/gated_fusion_nslt1000/model.yaml` | record model fusion nslt1000 |

Moi `model.yaml` thong thuong chua:

- dinh danh model
- loai branch
- config file
- checkpoint
- artifact bundle
- mo ta performance/pham vi su dung

## 9. `tests/`

| File | Tac dung |
| --- | --- |
| `tests/test_registry_loader.py` | kiem tra `src/slr/registry/loader.py` va validation registry |
| `tests/test_regions_nslt1000_union_builder.py` | kiem tra quy trinh xay union manifests/normalize sample_id/shape cho pipeline regions nslt1000 |

Nhan xet:

- Test coverage hien tap trung vao registry va incremental regions pipeline.
- Chua co bo unit test day cho skeleton/regions/fusion train loops, nhung cac script `verify/` bo sung vai tro smoke/integration tests.

## 10. `docs/`

## 10.1 Docs nen uu tien tin

| File | Tac dung |
| --- | --- |
| `docs/architecture/project_structure_guide.md` | mo ta kien truc repo va vai tro tung khu vuc |
| `docs/architecture/implementation_log.md` | nhat ky trien khai, dac biet lien quan ST-GCN++/huong mo rong |
| `docs/training/training_guide.md` | huong dan cac lenh train/eval dang dung |
| `docs/training/skeleton_training_baseline.md` | guide cho baseline skeleton |
| `docs/training/skeleton_stgcnpp_integration.md` | note tich hop ST-GCN++ |
| `docs/training/standard_label_smoothing.md` | note ve Standard Label Smoothing |

## 10.2 Docs lich su

| File | Tac dung | Canh bao |
| --- | --- | --- |
| `docs/history/LATEST_INFO.md` | snapshot doc cu | mot so nhan dinh da loi thoi |
| `docs/history/NEW_LATEST_INFO.md` | phien ban tong hop cu hon | van co cho chua cap nhat ve regions/fusion |
| `docs/history/FINAL_SUMARY.md` | tong ket moc truoc | gia tri lich su |
| `docs/history/FINAL_SUMARY_V2.md` | tong ket moc truoc ban 2 | gia tri lich su |
| `docs/history/MAY27_THESIS_REPORT.md` | bao cao moc 27/05 | lich su nghien cuu |
| `docs/history/MAY31_THESIS_REPORT.md` | bao cao moc 31/05 | lich su nghien cuu |
| `docs/history/StandardLS.md` | note cu ve StandardLS | doc doi chieu voi docs/training |
| `docs/packaging/bundle1k.md` | ghi chu package nslt1000 | huong van hanh, co the co encoding issue khi mo bang terminal |

## 11. `reports/`, `outputs/`, `artifacts/`, `packaging_outputs/`

## 11.1 `reports/`

`reports/` la bo nho van hanh cua repo:

- `reports/current/preprocessing/`: report sau index/standardize/pose/build branch inputs.
- `reports/current/regions/`: report setup, extraction, incremental feasibility, progress, union checks.
- `reports/current/fusion/`: report training pipeline, late fusion, packaging readiness.
- `reports/current/packaging/`: report sau khi dong goi bundle/package.
- `reports/current/training/`: report cau hinh va training analyses.
- `reports/archive/`: report cu, vi du lien quan UI/chuyen doi.

No khong phai source code, nhung rat huu ich de tai tao ly do tai sao mot workflow duoc to chuc nhu vay.

## 11.2 `outputs/`

`outputs/` la noi train loops ghi artifact.

Mau chung mot run:

- `config_resolved.yaml`
- `checkpoints/best.pt`
- `checkpoints/last.pt`
- `metrics.json`
- `summary.json`
- `train.log` hoac `train_log.csv`
- `eval_*.json` trong mot so run

Trong workspace hien co:

- `outputs/skeleton/...`
- `outputs/regions/...`
- `outputs/fusion/...`

## 11.3 `artifacts/`

`artifacts/` duoc dung lam workspace cho:

- logits export
- late fusion sweeps
- dry-run/plan support
- mot so fusion package intermediate outputs

Thu muc nay mang tinh "lam viec tam thoi/co cau truc ky thuat", khac voi `outputs/` la output train run.

## 11.4 `packaging_outputs/`

No luu cac package da duoc materialize truoc khi zip/phat hanh:

- bo branch inputs
- bo fusion package
- bo incremental regions package

## 12. `data/` theo lop thay vi tung file

## 12.1 `data/datasets/WLASL/raw/`

Nguon su that cua du lieu:

- `metadata/WLASL_v0.3.json`: master metadata.
- `metadata/wlasl_class_list.txt`: class map chuan.
- `metadata/missing.txt`: local missing ids.
- `metadata/nslt_100.json`, `nslt_300.json`, `nslt_1000.json`, `nslt_2000.json`: subset declarations.
- `docs/*.md`: phan tich local snapshot cua raw data.
- `videos/*.mp4`: raw media.

## 12.2 `data/datasets/WLASL/index/`

Lop metadata da lam sach va doi soat:

- master/all rows
- rows co video local
- rows thieu video
- subset manifests
- split maps
- class maps
- reports/logs

## 12.3 `data/datasets/WLASL/standardized/`

Lop media chuan hoa:

- `frames/`
- optional `videos/`
- `manifests/`
- `reports/`
- `logs/`

Day la dau vao chung cho pose extraction va cac visualization.

## 12.4 `data/datasets/WLASL/pose/rtmw_l/`

Lop pose dung chung:

- `wholebody_133/<subset>/<split>/*.npz`
- `manifests/*.csv`
- `reports/*.md`
- `logs/`

## 12.5 `data/datasets/WLASL/branch_inputs/skeleton/rtmw_l/`

Lop train-ready cho skeleton:

- `selected_27/`, `selected_31/`
- `normalized/`
- `graph_tensors/`
- `manifests/`
- `reports/`
- `logs/`

## 12.6 `data/datasets/WLASL/branch_inputs/regions/`

Lop train-ready cho regions:

- tensors theo split
- manifests
- preview images
- metadata/reports/logs

## 13. `archive/`, `scripts/archive/`, `configs/archive/`

Nhung thu muc nay quan trong vi chung giai thich lich su phat trien, nhung khong nen sua truoc tien khi can thay doi pipeline hien tai.

| Khu vuc | Noi dung |
| --- | --- |
| `archive/scaffolds/src/slr/branches/hand_poseflow/*` | scaffold branch hand_poseflow cu |
| `archive/scaffolds/src/slr/models/*` | scaffold model wrappers cu cho skeleton/regions/hand_poseflow |
| `archive/scaffolds/src/slr/inference/predict_video.py` | scaffold inference cu |
| `archive/scaffolds/src/slr/training/*.py` | scaffold train/eval cu |
| `scripts/archive/scaffolds/*` | wrapper CLI cu cho scaffold |
| `configs/archive/legacy/*` | config lich su |
| `configs/archive/deprecated/regions_cnn_gru/*` | cac bien the cu cua regions branch |

Ket luan:

- Lich su duoc giu lai rat day du.
- Nhung source of truth hien tai da chuyen sang `src/slr/branches/*`, `scripts/*`, `configs/train/*`, `configs/preprocessing/*`.

## 14. `experiments/`, `UI/`, `notebooks/`

| Khu vuc | Trang thai hien tai |
| --- | --- |
| `experiments/` | chi co cac subfolder placeholder voi `.gitkeep`; la noi du kien luu experiment artifacts tach khoi preprocessing |
| `UI/` | workspace nay hien rong, chua tham gia luong production |
| `notebooks/` | khu vuc local kham pha/phan tich; khong phai mot phan bat buoc cua pipeline source code |

## 15. Cac file/nhom file quan trong nhat neu can onboard nhanh

Neu mot nguoi moi vao repo va can doc theo thu tu hieu qua nhat, nen doc:

1. `README.md`
2. `docs/architecture/project_structure_guide.md`
3. `docs/training/training_guide.md`
4. `src/slr/data/build_index.py`
5. `src/slr/data/standardize_videos.py`
6. `src/slr/pose/extract_rtmw.py`
7. `src/slr/branches/skeleton/build_inputs.py`
8. `src/slr/branches/skeleton/dataset.py`
9. `src/slr/branches/skeleton/train.py`
10. `src/slr/branches/regions/build_crops.py`
11. `src/slr/branches/regions/dataset.py`
12. `src/slr/branches/regions/train.py`
13. `src/slr/branches/fusion/dataset.py`
14. `src/slr/branches/fusion/build.py`
15. `src/slr/branches/fusion/train.py`
16. `src/slr/branches/fusion/package_support.py`
17. `scripts/evaluate/evaluate_skeleton_region_late_fusion.py`
18. `model_registry/registry.yaml`

## 16. Nhan xet kien truc

### 16.1 Diem manh

- Tach lop du lieu ro rang: raw -> index -> standardized -> pose -> branch inputs.
- Moi stage lon deu manifest-driven.
- `src/slr/utils/io.py` va `src/slr/data/manifests.py` tao duoc ngon ngu chung cho ca repo.
- Skeleton, regions va fusion deu da co train/eval code rieng, khong bi ep vao mot mega abstraction kho doc.
- Scripts verify/package cho thay repo da tien den giai doan van hanh artifact, khong chi nghien cuu demo.

### 16.2 Diem yeu/rui ro

- Nhieu doc lich su da loi thoi mot phan, dac biet lien quan muc do hoan thien cua regions/fusion.
- Rat nhieu script packaging/incremental lam tang do phuc tap van hanh.
- Test unit chua phu rong bang so luong logic trong repo.
- `data/`, `outputs/`, `reports/`, `artifacts/` rat lon, nguoi moi de nham generated artifact voi source code.

## 17. Tom tat mot cau cho tung khu vuc

- `src/slr/data/*`: xay va chuan hoa lop metadata/media.
- `src/slr/pose/*`: trich xuat va chuan hoa wholebody pose dung chung.
- `src/slr/branches/skeleton/*`: bien pose thanh graph tensor va train graph model.
- `src/slr/branches/regions/*`: cat face/hands tensor va train video-region model.
- `src/slr/branches/fusion/*`: ghep hai branch va hoc gate hop nhat.
- `src/slr/training/*`: utility chung cho loss/optim/checkpoint/metrics/W&B.
- `src/slr/registry/*`: dang ky model/artifact co cau truc.
- `scripts/*`: cach van hanh repo tu terminal.
- `configs/*`: bo tham so co cau truc cho tung tang.
- `model_registry/*`: danh muc model san xuat/ban giao.
- `tests/*`: kiem tra registry va pipeline union regions.
- `docs/*`: huong dan, kien truc, lich su.
- `data/*`, `outputs/*`, `artifacts/*`, `packaging_outputs/*`, `reports/*`: ket qua va dau ra cua pipeline.

## 18. Ket luan cuoi

Repo nay hien la mot workspace SLR theo kieu "data pipeline + multi-branch training + packaging":

- `skeleton` la nhanh on dinh va ro rang nhat.
- `regions` da o muc implementation that, co ca preprocessing, models, train/eval va package.
- `fusion` da vuot qua muc y tuong, co feature-level training, score-level late fusion, audit va package support.

Neu sau nay can sua code, khong nen bat dau tu `docs/history/` hay `archive/`.
Hay bat dau tu:

- `src/slr/branches/*`
- `src/slr/data/*`
- `src/slr/pose/*`
- `scripts/*`
- `configs/*`

Do day moi la duong chay that cua repo hien tai.
