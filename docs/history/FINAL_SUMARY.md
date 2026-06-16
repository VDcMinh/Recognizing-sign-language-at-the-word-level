# FINAL_SUMARY

## 1. Pham vi va cach doc repo

Tai lieu nay duoc viet theo snapshot workspace ngay `2026-06-16`.

Toi da doc va doi chieu cac nhom sau:

- Toan bo ma nguon trong `src/`
- Toan bo config trong `configs/`
- Toan bo wrapper/utility script trong `scripts/`
- Cac tai lieu goc o root va `docs/`
- Cau truc du lieu, output, artifact, packaging va reports

Luu y quan trong:

- Repo nay co rat nhieu file sinh ra tu dong: rieng `data/` co hon `757,848` file, `packaging_outputs/` co hon `19,072` file.
- Vi vay, phan code/config/script se duoc mo ta theo tung file.
- Cac thu muc du lieu/artefact sinh ra tu dong se duoc mo ta theo cau truc, pattern ten file va vai tro trong pipeline, thay vi liet ke tung `.npz`, `.jpg`, `.csv`.

## 2. Tom tat mot cau

Day la repo nghien cuu nhan dang ngon ngu ky hieu muc tu tren WLASL, duoc to chuc thanh pipeline:

`raw videos + metadata -> index -> standardized videos/frames -> RTMW-l pose -> branch inputs -> train/eval skeleton / regions / fusion`

Trong 4 branch hien co:

- `skeleton`: da hoan chinh va dung duoc
- `regions`: da hoan chinh va dung duoc
- `fusion`: da hoan chinh va dung duoc
- `hand_poseflow`: moi la scaffold, chua co training/inference that su

## 3. Tinh trang thuc te cua repo

| Thanh phan | Muc do hoan thien | Ghi chu |
| --- | --- | --- |
| Indexing du lieu WLASL | San sang | `src/slr/data/build_index.py` la pipeline that |
| Standardize video | San sang | `src/slr/data/standardize_videos.py` crop/letterbox/video manifest |
| RTMW-l pose extraction | San sang | `src/slr/pose/extract_rtmw.py` dung MMPose |
| Skeleton branch | San sang | build input, dataset, graph, model, train, eval |
| Regions branch | San sang | build crop tensors, dataset, model, train, eval |
| Gated feature fusion | San sang | dataset pair, build backbones, gated fusion, train, eval |
| Late fusion logits | San sang | script rieng de quet alpha |
| Hand poseflow branch | Scaffold | build placeholder, dataset placeholder, model placeholder |
| Generic `src/slr/models/*` | Chu yeu la placeholder | implementation that nam o `src/slr/branches/*/models/*` |
| Generic `src/slr/training/train.py` va `evaluate.py` | Placeholder | train/eval that dung branch-specific files |
| `UI/` | Rong | chua co code giao dien |
| `experiments/` | Gan nhu rong | chu yeu la noi giu cho |

## 4. Bo cuc tong the

```text
.
|-- src/slr/                  # package chinh
|   |-- data/                 # index, manifest, standardization
|   |-- pose/                 # RTMW-l, schema, normalization, quality
|   |-- branches/
|   |   |-- skeleton/         # skeleton pipeline that
|   |   |-- regions/          # face/hands region pipeline that
|   |   |-- fusion/           # gated feature fusion that
|   |   `-- hand_poseflow/    # scaffold
|   |-- training/             # helper train chung
|   |-- inference/            # visualization + placeholder predict
|   |-- utils/                # io, image, video, bbox, logging, seed
|   `-- models/               # model namespace generic, phan lon la placeholder
|-- configs/                  # dataset, preprocessing, branch, train, fusion configs
|-- scripts/                  # wrapper scripts, checks, packaging, evaluation
|-- data/datasets/WLASL/      # raw/index/standardized/pose/branch_inputs
|-- outputs/                  # output train/eval
|-- artifacts/                # artifact support, nhat la fusion
|-- packaging_outputs/        # bundle Kaggle/HF/package da duoc tao
|-- reports/                  # bao cao preprocessing/training/packaging/fusion
|-- docs/                     # ghi chu ky thuat ngan gon
|-- checkpoints/              # pose checkpoint va model checkpoint da chon
|-- UI/                       # rong
|-- experiments/              # placeholder
`-- tai lieu root/*.md        # README + implementation/training notes
```

## 5. Cac file o root va vai tro

| File/thu muc | Noi dung | Vai tro trong repo |
| --- | --- | --- |
| `README.md` | Gioi thieu ngan gon pipeline | Diem vao nhanh nhat de hieu du an |
| `PROJECT_STRUCTURE_GUIDE.md` | Huong dan cau truc repo chi tiet | Tai lieu kien truc tong quan |
| `TRAINING.md` | Ghi chu train cac branch | Huong dan thuc thi training |
| `IMPLEMENTATION.md` | Ghi chu tinh hinh implementation | Tong hop tien do code |
| `LATEST_INFO.md` | Nhat ky/cap nhat trang thai | Snapshot thong tin moi hon |
| `NEW_LATEST_INFO.md` | Ban cap nhat bo sung | Bo sung cho `LATEST_INFO.md` |
| `MAY27_THESIS_REPORT.md` | Bao cao theo moc thoi gian | Tai lieu hoc thuat/noi bo |
| `MAY31_THESIS_REPORT.md` | Bao cao theo moc thoi gian | Tai lieu hoc thuat/noi bo |
| `bundle1k.md` | Ghi chu bundle/package 1000 lop | Phu tro packaging |
| `StandardLS.md` | Ghi chu Standard Label Smoothing | Tai lieu thuat toan va config |
| `pyproject.toml` | Dinh nghia package `slr` theo `src` layout | Dong goi Python package |
| `requirements.txt` | Dependency co ban | Moi truong preprocess/train co ban |
| `requirements-rtmw.txt` | Dependency cho RTMW-l/MMPose | Moi truong pose extraction |
| `requirements-kaggle-train.txt` | Dependency bo sung cho train/package tren Kaggle | Moi truong packaging/train cloud |
| `sitecustomize.py` | Them `src/` vao `sys.path` | Cho phep chay lenh ad-hoc khong can editable install |
| `slr/__init__.py` | Import shim tro den `src/slr` | De `import slr` hoat dong ngay tai root |
| `.gitignore` | Rule bo qua file sinh ra | Tach code voi artifact/data/output |
| `.venv-rtmw310/` | Virtual environment local | Moi truong thuc thi, khong phai source |

## 6. Package `src/slr/`

### 6.1 File namespace chung

| File | Vai tro |
| --- | --- |
| `src/slr/__init__.py` | Danh dau package `slr` trong `src` layout |
| `src/slr/branches/__init__.py` | Namespace cho cac branch |
| `src/slr/data/__init__.py` | Namespace cho module du lieu |
| `src/slr/inference/__init__.py` | Namespace cho module inference |
| `src/slr/models/__init__.py` | Namespace model generic |
| `src/slr/pose/__init__.py` | Namespace cho pose modules |
| `src/slr/training/__init__.py` | Namespace helper train |
| `src/slr/utils/__init__.py` | Namespace helper chung |

### 6.2 `src/slr/data/`

| File | Noi dung chinh | Cac hoat dong |
| --- | --- | --- |
| `src/slr/data/manifests.py` | Dinh nghia schema cot cho cac manifest | La nguon su that cho ten cot CSV/NPZ trong toan repo |
| `src/slr/data/validation.py` | Ham kiem tra cot, split, schema | Dung de bao ve pipeline khoi manifest sai cau truc |
| `src/slr/data/build_index.py` | Pipeline lap chi muc WLASL | Doc raw metadata, scan video local, tao `master_instances.csv`, subset manifests, class maps, missing/invalid reports |
| `src/slr/data/standardize_videos.py` | Chuan hoa video | Doc bbox/frame range, crop, expand/clip bbox, fallback full frame, resize/letterbox, xuat frames/videos + standardized manifest |

### 6.3 `src/slr/pose/`

| File | Noi dung chinh | Cac hoat dong |
| --- | --- | --- |
| `src/slr/pose/pose_schema.py` | Schema RTMW-l wholebody_133 va cac tap keypoint chon loc | Dinh nghia body/face/hands, `selected_27`, `selected_31`, helper ten keypoint |
| `src/slr/pose/keypoint_selection.py` | Cat keypoint subset | Chuyen pose day du thanh skeleton subset can train |
| `src/slr/pose/pose_normalization.py` | Chuan hoa pose | Center/scale toa do, lam sach gia tri khong hop le, chuan hoa confidence |
| `src/slr/pose/pose_quality.py` | Do chat luong pose | Tinh mean confidence, ti le frame hop le, thong ke theo vung |
| `src/slr/pose/extract_rtmw.py` | RTMW-l extraction pipeline | Goi MMPose inferencer, chon nguoi chinh, ghi pose tensors/manifests/quality reports, ho tro CPU/GPU fallback |

### 6.4 `src/slr/utils/`

| File | Noi dung chinh | Cac hoat dong |
| --- | --- | --- |
| `src/slr/utils/io.py` | IO helpers | Doc/ghi JSON, YAML, CSV, text; remap path giua local/Kaggle/HF |
| `src/slr/utils/logging.py` | Logging helpers | Tao logger stream/file nhat quan |
| `src/slr/utils/video.py` | Video helpers | Probe/read/write video bang OpenCV |
| `src/slr/utils/image.py` | Image helpers | Doc/ghi/crop/letterbox anh |
| `src/slr/utils/bbox.py` | Bounding-box helpers | Parse, validate, expand, clip, square, stringify bbox |
| `src/slr/utils/seed.py` | Seed helper | Dat seed cho cac thanh phan co random |

### 6.5 `src/slr/training/`

| File | Trang thai | Vai tro |
| --- | --- | --- |
| `src/slr/training/checkpointing.py` | That | Save/load `state_dict`, metadata checkpoint |
| `src/slr/training/losses.py` | That | Factory tao loss; ho tro cross entropy va standard label smoothing |
| `src/slr/training/metrics.py` | That | `AverageMeter`, top-k accuracy |
| `src/slr/training/optim.py` | That | Factory tao optimizer/scheduler |
| `src/slr/training/seed.py` | That | Reproducibility cho training |
| `src/slr/training/wandb_utils.py` | That | Khoi tao/log W&B neu kha dung |
| `src/slr/training/train.py` | Placeholder | CLI train generic, chua phai entrypoint that duoc branch su dung |
| `src/slr/training/evaluate.py` | Placeholder | CLI eval generic, chua phai entrypoint that duoc branch su dung |

### 6.6 `src/slr/inference/`

| File | Trang thai | Vai tro |
| --- | --- | --- |
| `src/slr/inference/predict_video.py` | Placeholder | Dung parser va log input, chua co luong suy luan that |
| `src/slr/inference/visualize_prediction.py` | That | Ve/preview pose va region crop de kiem tra qualitative |

## 7. Cac branch chinh

### 7.1 Skeleton branch `src/slr/branches/skeleton/`

| File | Noi dung chinh | Cac hoat dong |
| --- | --- | --- |
| `src/slr/branches/skeleton/__init__.py` | Namespace branch skeleton | Export module branch |
| `src/slr/branches/skeleton/build_inputs.py` | Build branch input skeleton | Doc pose manifest, chon keypoint set, normalize, pad/trim den `T=150`, doi sang graph tensor va ghi tensor/manifests/reports |
| `src/slr/branches/skeleton/dataset.py` | Dataset loader skeleton | Doc graph tensor `.npz`, remap path giua moi truong, check shape nghiem ngat, tra ve tensor + label + metadata |
| `src/slr/branches/skeleton/graph.py` | Dinh nghia do thi skeleton | Khai bao edge cho `selected_27`/`selected_31`, tao adjacency `uniform`/`spatial` |
| `src/slr/branches/skeleton/label_smoothing.py` | Helper label smoothing | Ho tro tao target/loss phuc vu thi nghiem smoothing |
| `src/slr/branches/skeleton/transforms.py` | Tensor transforms | Pad/trim chuoi, fix sequence length, doi `(T,V,C)` sang `(C,T,V,M)` |
| `src/slr/branches/skeleton/train.py` | Training/eval that cua skeleton | Load config, tao dataset/dataloader/graph/model/loss/optimizer, train, save `best.pt`/`last.pt`, evaluate test |
| `src/slr/branches/skeleton/models/__init__.py` | Model factory | Chon `simple_stgcn` hoac `stgcnpp` |
| `src/slr/branches/skeleton/models/simple_stgcn.py` | Baseline model that | ST-GCN nhe de kiem tra end-to-end pipeline |
| `src/slr/branches/skeleton/models/stgcnpp.py` | Backbone that | Implementation ST-GCN++ compatible cho input graph tensor |

### 7.2 Regions branch `src/slr/branches/regions/`

| File | Noi dung chinh | Cac hoat dong |
| --- | --- | --- |
| `src/slr/branches/regions/__init__.py` | Namespace branch regions | Export module branch |
| `src/slr/branches/regions/region_schema.py` | Schema region tensors | Dinh nghia thu tu region, ma nguon, format tensor |
| `src/slr/branches/regions/crop_utils.py` | Helper crop | Rut bbox mat/tay tu wholebody_133, fallback khi bbox kem chat luong, crop + resize |
| `src/slr/branches/regions/transforms.py` | Region transforms | Normalize va augmentation cho clip tensor |
| `src/slr/branches/regions/build_crops.py` | Build region inputs | Doc frames standardized + pose, crop `left_hand`, `right_hand`, `face`, ghi tensor/manifests/previews/quality reports |
| `src/slr/branches/regions/dataset.py` | Dataset loader regions | Doc tensor `.npz`, chon active regions, normalize/augment, tra `valid_mask`, bbox metadata |
| `src/slr/branches/regions/train.py` | Training/eval that cua regions | Load config, tao dataloader/model/loss/optimizer, early stopping, save checkpoint va metrics |
| `src/slr/branches/regions/models/__init__.py` | Model factory | Chon `region_resnet18_gru` hoac `region_cnn_gru` |
| `src/slr/branches/regions/models/region_resnet18_gru.py` | Model that chinh | Ma hoa tung frame bang ResNet18, ma hoa thoi gian bang GRU cho moi region, fusion region feature, classifier |
| `src/slr/branches/regions/models/region_cnn_gru.py` | Model that thay the nhe hon | CNN tu xay + GRU; duoc giu cho baseline/deprecated configs |

### 7.3 Fusion branch `src/slr/branches/fusion/`

| File | Noi dung chinh | Cac hoat dong |
| --- | --- | --- |
| `src/slr/branches/fusion/__init__.py` | Namespace branch fusion | Export builder/dataset/model |
| `src/slr/branches/fusion/dataset.py` | Dataset pair skeleton-regions | Canh hang hai branch theo `sample_id`, check nhat quan `class_id`/`gloss`, tra ve mau pair |
| `src/slr/branches/fusion/build.py` | Builder fusion | Doc config branch, khoi tao skeleton model + region model + nap checkpoint, dong goi thanh fusion model |
| `src/slr/branches/fusion/train.py` | Training that cua gated fusion | Validate shape dau vao, dong bang backbone neu can, train fusion head, log gate stats, save checkpoint |
| `src/slr/branches/fusion/evaluate.py` | Evaluate fusion model | Chay eval tren split, ghi metrics |
| `src/slr/branches/fusion/package_support.py` | Utility packaging lon nhat repo | Audit manifest, resolve path tensor, check checkpoint/config, smoke test fusion, tao metadata/README/package verifiers cho NSLT1000 |
| `src/slr/branches/fusion/models/__init__.py` | Namespace model fusion | Export gated fusion model |
| `src/slr/branches/fusion/models/gated_feature_fusion.py` | Model fusion that | Rut feature skeleton + regions, project ve hidden dim chung, tinh sigmoid gate, tron feature va classifier |

### 7.4 Hand poseflow branch `src/slr/branches/hand_poseflow/`

| File | Trang thai | Vai tro |
| --- | --- | --- |
| `src/slr/branches/hand_poseflow/__init__.py` | Namespace | Export branch scaffold |
| `src/slr/branches/hand_poseflow/poseflow_schema.py` | Co that nhung nho | Khai bao `POSEFLOW_VARIANTS = ("selected_31", "hands_only")` |
| `src/slr/branches/hand_poseflow/build_inputs.py` | Scaffold | Dieu phoi hai buoc build hand sequence va poseflow |
| `src/slr/branches/hand_poseflow/build_hand_sequences.py` | Placeholder | Moi tao thu muc/log, chua build sequence that |
| `src/slr/branches/hand_poseflow/build_poseflow.py` | Placeholder | Moi tao thu muc/log, chua build poseflow that |
| `src/slr/branches/hand_poseflow/dataset.py` | Placeholder | `__len__`/`__getitem__` chua phuc vu training that |

## 8. Namespace `src/slr/models/` va ly do de nham lan

Repo co hai lop model:

- Lop that de train: `src/slr/branches/<branch>/models/*`
- Lop generic: `src/slr/models/*`

Lop generic hien tai chu yeu la placeholder:

| File | Trang thai | Ghi chu |
| --- | --- | --- |
| `src/slr/models/skeleton/stgcnpp.py` | Placeholder | Class `STGCNPP` nem `NotImplementedError` |
| `src/slr/models/skeleton/ctrgcn.py` | Placeholder | Class `CTRGCN` nem `NotImplementedError` |
| `src/slr/models/skeleton/heads.py` | Placeholder nhe | Chi giu `ClassificationHead` cuc ky don gian |
| `src/slr/models/regions/cnn_lstm.py` | Placeholder | `CNNLSTM` chua implement |
| `src/slr/models/regions/video_transformer.py` | Placeholder | `VideoTransformer` chua implement |
| `src/slr/models/regions/heads.py` | Placeholder nhe | Giu `RegionClassificationHead` |
| `src/slr/models/hand_poseflow/two_stream.py` | Placeholder | `TwoStreamHandPoseFlow` chua implement |
| `src/slr/models/hand_poseflow/heads.py` | Placeholder nhe | Giu `HandPoseFlowHead` |
| `src/slr/models/skeleton/__init__.py` | Namespace | Gom model skeleton generic |
| `src/slr/models/regions/__init__.py` | Namespace | Gom model regions generic |
| `src/slr/models/hand_poseflow/__init__.py` | Namespace | Gom model hand-poseflow generic |

Ket luan: khi chay that, can uu tien doc `src/slr/branches/*/models/*`, khong phai `src/slr/models/*`.

## 9. Config catalogue `configs/`

### 9.1 Dataset va preprocessing

| File | Vai tro |
| --- | --- |
| `configs/dataset/wlasl.yaml` | Config goc cho root du lieu WLASL, subset co san, split/mac dinh |
| `configs/preprocessing/index.yaml` | Config build index tu raw metadata/video |
| `configs/preprocessing/standardize.yaml` | Standardize mac dinh |
| `configs/preprocessing/standardize_nslt300.yaml` | Standardize cho subset NSLT300 |
| `configs/preprocessing/standardize_nslt1000.yaml` | Standardize cho subset NSLT1000 |
| `configs/preprocessing/pose_rtmw_l.yaml` | Pose extraction RTMW-l mac dinh |
| `configs/preprocessing/pose_rtmw_l_kaggle.yaml` | Pose extraction RTMW-l cho moi truong Kaggle |
| `configs/preprocessing/region_crops.yaml` | Build region crops mac dinh |
| `configs/preprocessing/region_crops_nslt100.yaml` | Build region crops NSLT100 |
| `configs/preprocessing/region_crops_nslt300.yaml` | Build region crops NSLT300 |
| `configs/preprocessing/region_crops_nslt1000.yaml` | Build region crops NSLT1000 |
| `configs/preprocessing/poseflow.yaml` | Scaffold preprocessing cho hand-poseflow |

### 9.2 Branch preprocessing configs

| File | Vai tro |
| --- | --- |
| `configs/branches/skeleton/stgcnpp_27.yaml` | Build input skeleton `selected_27` cho subset mac dinh |
| `configs/branches/skeleton/stgcnpp_27_nslt300.yaml` | Build input `selected_27` cho NSLT300 |
| `configs/branches/skeleton/stgcnpp_27_nslt1000.yaml` | Build input `selected_27` cho NSLT1000 |
| `configs/branches/skeleton/stgcnpp_31.yaml` | Build input skeleton `selected_31` cho subset mac dinh |
| `configs/branches/skeleton/stgcnpp_31_nslt300.yaml` | Build input `selected_31` cho NSLT300 |
| `configs/branches/skeleton/stgcnpp_31_nslt1000.yaml` | Build input `selected_31` cho NSLT1000 |
| `configs/branches/regions/face_hands_baseline.yaml` | Baseline branch config cho regions (face + two hands) |
| `configs/branches/hand_poseflow/hand_poseflow_baseline.yaml` | Baseline scaffold cho hand-poseflow |

### 9.3 Skeleton training configs

| File | Vai tro |
| --- | --- |
| `configs/train/skeleton_selected_27_baseline.yaml` | Train `SimpleSTGCN` tren `selected_27` |
| `configs/train/skeleton_selected_31_baseline.yaml` | Train `SimpleSTGCN` tren `selected_31` |
| `configs/train/skeleton_selected_27.yaml` | Config skeleton tong quat cho `selected_27` |
| `configs/train/skeleton_selected_31.yaml` | Config skeleton tong quat cho `selected_31` |
| `configs/train/skeleton_selected_27_stgcnpp.yaml` | ST-GCN++ cho `selected_27` |
| `configs/train/skeleton_selected_31_stgcnpp.yaml` | ST-GCN++ cho `selected_31` |
| `configs/train/skeleton_selected_27_stgcnpp_standardls_eps005.yaml` | ST-GCN++ `selected_27` + StandardLS epsilon 0.05 |
| `configs/train/skeleton_selected_27_stgcnpp_standardls_eps01.yaml` | ST-GCN++ `selected_27` + StandardLS epsilon 0.1 |
| `configs/train/skeleton_selected_27_stgcnpp_standardls_eps03.yaml` | ST-GCN++ `selected_27` + StandardLS epsilon 0.3 |
| `configs/train/skeleton_selected_31_stgcnpp_standardls_eps005.yaml` | ST-GCN++ `selected_31` + StandardLS epsilon 0.05 |
| `configs/train/skeleton_selected_31_stgcnpp_standardls_eps01.yaml` | ST-GCN++ `selected_31` + StandardLS epsilon 0.1 |
| `configs/train/skeleton_selected_31_stgcnpp_standardls_eps03.yaml` | ST-GCN++ `selected_31` + StandardLS epsilon 0.3 |
| `configs/train/skeleton_selected_27_nslt300_stgcnpp.yaml` | ST-GCN++ `selected_27` cho NSLT300 |
| `configs/train/skeleton_selected_27_nslt300_stgcnpp_standardls_eps03.yaml` | NSLT300 `selected_27` + StandardLS 0.3 |
| `configs/train/skeleton_selected_31_nslt300_stgcnpp.yaml` | ST-GCN++ `selected_31` cho NSLT300 |
| `configs/train/skeleton_selected_31_nslt300_stgcnpp_standardls_eps01.yaml` | NSLT300 `selected_31` + StandardLS 0.1 |
| `configs/train/skeleton_selected_31_nslt300_stgcnpp_standardls_eps02.yaml` | NSLT300 `selected_31` + StandardLS 0.2 |
| `configs/train/skeleton_selected_31_nslt300_stgcnpp_standardls_eps03.yaml` | NSLT300 `selected_31` + StandardLS 0.3 |
| `configs/train/skeleton_selected_27_nslt1000_stgcnpp.yaml` | ST-GCN++ `selected_27` cho NSLT1000 |
| `configs/train/skeleton_selected_27_nslt1000_stgcnpp_standardls_eps01.yaml` | NSLT1000 `selected_27` + StandardLS 0.1 |
| `configs/train/skeleton_selected_27_nslt1000_stgcnpp_standardls_eps02.yaml` | NSLT1000 `selected_27` + StandardLS 0.2 |
| `configs/train/skeleton_selected_27_nslt1000_stgcnpp_standardls_eps03.yaml` | NSLT1000 `selected_27` + StandardLS 0.3 |
| `configs/train/skeleton_selected_31_nslt1000_stgcnpp.yaml` | ST-GCN++ `selected_31` cho NSLT1000 |
| `configs/train/skeleton_selected_31_nslt1000_stgcnpp_standardls_eps01.yaml` | NSLT1000 `selected_31` + StandardLS 0.1 |
| `configs/train/skeleton_selected_31_nslt1000_stgcnpp_standardls_eps02.yaml` | NSLT1000 `selected_31` + StandardLS 0.2 |
| `configs/train/skeleton_selected_31_nslt1000_stgcnpp_standardls_eps03.yaml` | NSLT1000 `selected_31` + StandardLS 0.3 |

### 9.4 Regions training configs

| File | Vai tro |
| --- | --- |
| `configs/train/regions_resnet18_gru_nslt100.yaml` | Config goc regions cho NSLT100 |
| `configs/train/regions_resnet18_gru_nslt100_aug.yaml` | NSLT100 + augmentation |
| `configs/train/regions_resnet18_gru_nslt100_standardls_eps01.yaml` | NSLT100 + StandardLS 0.1 |
| `configs/train/regions_resnet18_gru_nslt100_hands_only.yaml` | Regions chi dung 2 tay, bo mat |
| `configs/train/regions_resnet18_gru_nslt100_hands_only_aug.yaml` | Hands-only + augmentation |
| `configs/train/regions_resnet18_gru_nslt100_hands_only_standardls_eps01.yaml` | Hands-only + StandardLS 0.1 |
| `configs/train/regions_resnet18_gru_nslt100_finetune.yaml` | Config fine-tune cho NSLT100 |
| `configs/train/regions_resnet18_gru_nslt300.yaml` | Regions cho NSLT300 |
| `configs/train/regions_resnet18_gru_nslt300_aug.yaml` | NSLT300 + augmentation |
| `configs/train/regions_resnet18_gru_nslt300_standardls_eps01.yaml` | NSLT300 + StandardLS 0.1 |
| `configs/train/regions_resnet18_gru_nslt1000.yaml` | Regions cho NSLT1000 |
| `configs/train/regions_resnet18_gru_nslt1000_aug.yaml` | NSLT1000 + augmentation |
| `configs/train/regions_resnet18_gru_nslt1000_standardls_eps01.yaml` | NSLT1000 + StandardLS 0.1 |
| `configs/train/regions_resnet18_gru_nslt1000_union.yaml` | Train tren union manifests cho NSLT1000 |
| `configs/train/regions_resnet18_gru_nslt1000_union_standardls_eps01.yaml` | Union NSLT1000 + StandardLS 0.1 |
| `configs/train/regions_resnet18_gru_nslt1000_incremental_kaggle.yaml` | Train package incremental regions tren Kaggle |
| `configs/train/regions_resnet18_gru_nslt1000_incremental_kaggle.yaml.template` | Template cho config incremental Kaggle |
| `configs/train/regions_resnet18_gru_nslt1000_test_run.yaml` | Config test/smoke run cho NSLT1000 |

### 9.5 Fusion training configs

| File | Vai tro |
| --- | --- |
| `configs/train/gated_feature_fusion_nslt100.yaml` | Train gated fusion cho NSLT100 |
| `configs/train/gated_feature_fusion_nslt300.yaml` | Train gated fusion cho NSLT300 |
| `configs/train/gated_feature_fusion_nslt1000.yaml` | Train gated fusion cho NSLT1000 local |
| `configs/train/gated_feature_fusion_nslt1000_kaggle.yaml` | Train gated fusion cho package Kaggle NSLT1000 |
| `configs/fusion/nslt100_skeleton_regions_late_fusion.yaml` | Config quet late-fusion alpha tren logits skeleton/regions |

### 9.6 Experiment va deprecated configs

| File | Vai tro |
| --- | --- |
| `configs/experiments/skeleton_nslt100_debug.yaml` | Config debug/thu nghiem skeleton |
| `configs/train/deprecated/regions_cnn_gru/README.md` | Giai thich cum config deprecated |
| `configs/train/deprecated/regions_cnn_gru/regions_cnn_gru_nslt100.yaml` | Baseline CNN-GRU cu cho NSLT100 |
| `configs/train/deprecated/regions_cnn_gru/regions_cnn_gru_nslt100_standardls_eps005.yaml` | CNN-GRU NSLT100 + StandardLS 0.05 |
| `configs/train/deprecated/regions_cnn_gru/regions_cnn_gru_nslt100_standardls_eps01.yaml` | CNN-GRU NSLT100 + StandardLS 0.1 |
| `configs/train/deprecated/regions_cnn_gru/regions_cnn_gru_nslt100_standardls_eps02.yaml` | CNN-GRU NSLT100 + StandardLS 0.2 |
| `configs/train/deprecated/regions_cnn_gru/regions_cnn_gru_nslt100_small_reg_ce.yaml` | CNN-GRU NSLT100 voi regularization nho, CE |
| `configs/train/deprecated/regions_cnn_gru/regions_cnn_gru_nslt100_small_reg_standardls_eps01.yaml` | Regularization nho + StandardLS 0.1 |
| `configs/train/deprecated/regions_cnn_gru/regions_cnn_gru_nslt100_small_reg_aug_ce.yaml` | Regularization nho + augmentation + CE |
| `configs/train/deprecated/regions_cnn_gru/regions_cnn_gru_nslt100_small_reg_aug_standardls_eps01.yaml` | Regularization nho + augmentation + StandardLS 0.1 |
| `configs/train/deprecated/regions_cnn_gru/regions_cnn_gru_nslt300.yaml` | Baseline CNN-GRU cu cho NSLT300 |
| `configs/train/deprecated/regions_cnn_gru/regions_cnn_gru_nslt300_standardls_eps005.yaml` | CNN-GRU NSLT300 + StandardLS 0.05 |
| `configs/train/deprecated/regions_cnn_gru/regions_cnn_gru_nslt300_standardls_eps01.yaml` | CNN-GRU NSLT300 + StandardLS 0.1 |
| `configs/train/deprecated/regions_cnn_gru/regions_cnn_gru_nslt300_standardls_eps02.yaml` | CNN-GRU NSLT300 + StandardLS 0.2 |
| `configs/train/deprecated/regions_cnn_gru/regions_cnn_gru_nslt300_small_reg_ce.yaml` | Regularization nho cho NSLT300, CE |
| `configs/train/deprecated/regions_cnn_gru/regions_cnn_gru_nslt300_small_reg_standardls_eps01.yaml` | Regularization nho + StandardLS 0.1 |
| `configs/train/deprecated/regions_cnn_gru/regions_cnn_gru_nslt300_small_reg_aug_ce.yaml` | Regularization nho + augmentation + CE |
| `configs/train/deprecated/regions_cnn_gru/regions_cnn_gru_nslt300_small_reg_aug_standardls_eps01.yaml` | Regularization nho + augmentation + StandardLS 0.1 |

## 10. Script catalogue `scripts/`

### 10.1 Wrapper preprocess/train/eval chinh

| File | Vai tro |
| --- | --- |
| `scripts/00_build_index.py` | Wrapper cho `slr.data.build_index.main` |
| `scripts/01_standardize_videos.py` | Wrapper cho `slr.data.standardize_videos.main` |
| `scripts/02_extract_pose_rtmw.py` | Wrapper cho `slr.pose.extract_rtmw.main` |
| `scripts/03_build_skeleton_inputs.py` | Wrapper build branch input skeleton |
| `scripts/04_build_region_inputs.py` | Wrapper build branch input regions |
| `scripts/05_build_hand_poseflow_inputs.py` | Wrapper scaffold cho hand-poseflow |
| `scripts/train_skeleton.py` | Entry train/eval skeleton that |
| `scripts/evaluate_skeleton.py` | Entry evaluate skeleton |
| `scripts/train_regions.py` | Entry train/eval regions that |
| `scripts/evaluate_regions.py` | Entry evaluate regions |
| `scripts/train_gated_fusion.py` | Entry train gated fusion |
| `scripts/evaluate_gated_fusion.py` | Entry evaluate gated fusion |
| `scripts/train_hand_poseflow.py` | Entry placeholder cho branch hand-poseflow |
| `scripts/evaluate.py` | Evaluate generic placeholder |
| `scripts/evaluate_skeleton_region_late_fusion.py` | Quet late-fusion tren logits skeleton va regions |

### 10.2 Script check, sanity va visualization

| File | Vai tro |
| --- | --- |
| `scripts/check_skeleton_dataset.py` | Kiem tra manifest/path/shape cua skeleton tensors |
| `scripts/check_region_dataset.py` | Kiem tra region dataset |
| `scripts/check_fusion_workspace.py` | Kiem tra artifact/config/checkpoint cho fusion workspace |
| `scripts/check_gated_fusion_setup.py` | Sanity check cau hinh gated fusion |
| `scripts/check_gated_fusion_nslt300_packaging_requirements.py` | Audit dieu kien packaging NSLT300 fusion |
| `scripts/check_gated_fusion_nslt1000_packaging_requirements.py` | Audit dieu kien packaging NSLT1000 fusion |
| `scripts/check_regions_nslt1000_extraction.py` | Kiem tra extraction regions NSLT1000 |
| `scripts/check_regions_nslt1000_incremental_feasibility.py` | Kiem tra tinh kha thi incremental strategy |
| `scripts/check_regions_nslt1000_incremental_pipeline_requirements.py` | Kiem tra requirement cua incremental pipeline |
| `scripts/check_regions_nslt1000_incremental_progress.py` | Theo doi tien do incremental extraction |
| `scripts/check_regions_nslt1000_union_dataset.py` | Kiem tra union dataset NSLT1000 |
| `scripts/tools/test_rtmw_mmpose_video.py` | Chay nhanh RTMW-l tren video de debug |
| `scripts/tools/visualize_pose_sets.py` | Ve keypoint set cho kiem tra truc quan |
| `scripts/tools/visualize_selected_27_samples.py` | Visualize mau `selected_27` |

### 10.3 Script packaging va bundling

| File | Vai tro |
| --- | --- |
| `scripts/prepare_hf_skeleton_bundle.py` | Tao bundle theo huong Hugging Face cho skeleton |
| `scripts/prepare_hf_regions_bundle.py` | Tao bundle theo huong Hugging Face cho regions |
| `scripts/prepare_kaggle_bundle.py` | Tao bundle Kaggle tong quat |
| `scripts/prepare_kaggle_nslt1000_remaining_bundle.py` | Tao bundle phan con lai cho NSLT1000 |
| `scripts/package_nslt100_branch_inputs.py` | Dong goi branch inputs NSLT100 |
| `scripts/package_regions_nslt300_kaggle_dataset.py` | Package regions NSLT300 cho Kaggle |
| `scripts/package_regions_nslt1000_incremental_kaggle_dataset.py` | Package incremental regions NSLT1000 |
| `scripts/package_gated_fusion_nslt100_kaggle_dataset.py` | Package fusion NSLT100 |
| `scripts/package_gated_fusion_nslt300_kaggle_dataset.py` | Package fusion NSLT300 |
| `scripts/package_gated_fusion_nslt1000_kaggle_dataset.py` | Package fusion NSLT1000 |
| `scripts/verify_nslt100_branch_inputs_package.py` | Verify package branch inputs NSLT100 |
| `scripts/verify_regions_nslt1000_incremental_package.py` | Verify package incremental regions NSLT1000 |
| `scripts/verify_gated_fusion_nslt1000_package.py` | Verify package fusion NSLT1000 |

### 10.4 Script ho tro NSLT1000 regions incremental/union

| File | Vai tro |
| --- | --- |
| `scripts/regions_nslt1000_incremental_common.py` | Helper dung chung cho nhieu script incremental |
| `scripts/build_regions_nslt1000_missing_manifests.py` | Tao manifest cho phan sample con thieu |
| `scripts/build_regions_nslt1000_union_manifests.py` | Tao union manifests cho NSLT1000 |
| `scripts/extract_regions_nslt1000_missing_only.py` | Chi extract phan tensor con thieu |
| `scripts/materialize_regions_nslt1000_kaggle_manifests.py` | Materialize manifest package cho Kaggle |
| `scripts/create_regions_nslt1000_incremental_zip.py` | Tao file zip incremental package |
| `scripts/cleanup_region_outputs.py` | Don dep output region phuc vu packaging |
| `scripts/merge_nslt300_pose_into_nslt1000.py` | Tron asset/manifests pose NSLT300 vao NSLT1000 |
| `scripts/prepare_regions_branch_inputs.py` | Dieu phoi buoc chuan bi region branch inputs |

### 10.5 Script shim import

| File | Vai tro |
| --- | --- |
| `scripts/sitecustomize.py` | Them `src/` vao `sys.path` khi chay trong `scripts/` |
| `scripts/slr/__init__.py` | Import shim de `scripts/*` resolve package `slr` |

## 11. `docs/`, `tests/`, `reports/`, `artifacts/`

### 11.1 `docs/`

| File | Noi dung |
| --- | --- |
| `docs/skeleton_training_baseline.md` | Giai thich baseline skeleton training pipeline va cach chay |
| `docs/skeleton_stgcnpp_integration.md` | Giai thich viec tich hop ST-GCN++ vao skeleton branch |
| `docs/standard_label_smoothing.md` | Giai thich Standard Label Smoothing va cac config lien quan |

### 11.2 `tests/`

| File | Noi dung |
| --- | --- |
| `tests/test_regions_nslt1000_union_builder.py` | Unit test cho logic union-builder cua regions NSLT1000, nhat la normalizing sample ID va alignment |

### 11.3 `reports/`

`reports/` hien co 74 file, chia thanh:

- `reports/preprocessing/`: bao cao index, standardization, region crops, bundle preprocessing, keypoint visualization
- `reports/training/`: bao cao implementation va config training cho skeleton/regions
- `reports/regions/`: bao cao setup, extraction, incremental feasibility, union pipeline
- `reports/fusion/`: bao cao late fusion, gated fusion training, packaging readiness
- `reports/packaging/`: bao cao dong goi Kaggle/HF/package verification
- `reports/experiments/`: ghi chu thuc nghiem
- Hai file `reports/ui_*`: bao cao lien quan UI/Streamlit, nhung ban than `UI/` hien khong co code thuc te

Ban chat cua `reports/` la tai lieu noi bo va artifact mo ta qua trinh lam, khong phai runtime core.

### 11.4 `artifacts/`

`artifacts/` co 22 file, trong do phan lon tap trung vao `artifacts/fusion/` cho cac subset `nslt100`, `nslt300`, `nslt1000`.

Vai tro:

- Luu config da resolve cho branch skeleton/regions dung trong fusion
- Giup tai tao fusion workspace hoac packaging
- La cau noi giua checkpoint da train va package cuoi

## 12. Cau truc du lieu va artifact sinh ra

### 12.1 `data/datasets/WLASL/`

| Thu muc | Vai tro |
| --- | --- |
| `raw/` | Metadata goc, json goc, video goc va thong tin raw |
| `index/` | Master manifest, subset manifests, class maps, split reports, missing/invalid bookkeeping |
| `standardized/` | Frames/videos da crop-resize-letterbox + standardized manifests |
| `pose/` | Pose tensors/manifests/quality report tu RTMW-l |
| `branch_inputs/` | Input da chuan hoa rieng cho skeleton, regions, hand-poseflow |

### 12.2 `branch_inputs/` theo branch

| Nhanh | Dang du lieu |
| --- | --- |
| Skeleton | Tensor graph `selected_27` hoac `selected_31`, shape mau la `(3,150,V,1)` |
| Regions | Tensor crop 3 vung, shape mau cho fusion la `(3,3,64,112,112)` |
| Hand poseflow | Scaffold directory, chua co artifact on dinh |

### 12.3 `outputs/`

| Thu muc | Noi dung |
| --- | --- |
| `outputs/skeleton/` | Moi run skeleton luu `config_resolved.yaml`, `metrics.json`, `summary.json`, `train_log.csv`, checkpoints |
| `outputs/regions/` | Tuong tu cho regions |
| `outputs/fusion/` | Tuong tu cho fusion |
| `outputs/smoke_metrics/` | Ket qua sanity/smoke metrics |

### 12.4 `checkpoints/`

| Thu muc | Noi dung |
| --- | --- |
| `checkpoints/pose/rtmw_l/` | Checkpoint va config cho RTMW-l pose model |
| `checkpoints/models/skeleton/` | Checkpoint skeleton branch da chon |
| `checkpoints/models/regions/` | Checkpoint regions branch da chon |

### 12.5 `packaging_outputs/`

Thu muc nay chua cac goi du lieu da duoc materialize. Hai cum de thay ro nhat la:

- `wlasl-nslt1000-regions-rtmw-l-incremental/`
- `wlasl-nslt1000-gated-fusion-ready/`

Ben trong thuong co:

- `README.md`
- `metadata.json`
- `verify/verify_package.py` va `verify_summary.json`
- `configs/`
- `branch_inputs/` hoac `regions/`
- `manifests/`
- Tensor `.npz` da duoc sap xep dung layout package

## 13. Luong hoat dong dau cuoi

### 13.1 Pipeline preprocessing tong quat

```text
WLASL raw metadata + raw videos
    |
    v
scripts/00_build_index.py
    -> src/slr/data/build_index.py
    -> tao master/subset manifests
    |
    v
scripts/01_standardize_videos.py
    -> src/slr/data/standardize_videos.py
    -> crop + resize + letterbox + standardized manifests
    |
    v
scripts/02_extract_pose_rtmw.py
    -> src/slr/pose/extract_rtmw.py
    -> RTMW-l pose tensors + pose manifests + quality reports
    |
    +--> scripts/03_build_skeleton_inputs.py
    |       -> selected_27/31 graph tensors
    |
    +--> scripts/04_build_region_inputs.py
    |       -> left_hand/right_hand/face region tensors
    |
    `--> scripts/05_build_hand_poseflow_inputs.py
            -> scaffold only
```

### 13.2 Skeleton model flow

```text
Pose wholebody_133
    -> keypoint_selection.py
    -> pose_normalization.py
    -> transforms.py (pad/trim/fixed T=150)
    -> graph tensor (C,T,V,M)
    -> dataset.py
    -> graph.py tao adjacency A
    -> model: SimpleSTGCN hoac STGCNPP
    -> classifier logits
```

Chi tiet:

- Dau vao training la graph tensor da tinh san, khong train truc tiep tren raw pose JSON.
- `selected_27` va `selected_31` la hai topology keypoint chinh.
- `graph.py` tao canh do thi theo layout va strategy.
- `train.py` tu chon model qua config, train, validate, save `best.pt`, roi test checkpoint tot nhat.

### 13.3 Regions model flow

```text
Standardized frames + RTMW-l pose
    -> crop_utils.py rut bbox face/left_hand/right_hand
    -> build_crops.py tao crop tensor theo clip
    -> dataset.py normalize/augment/chon active_regions
    -> model: RegionResNet18GRU hoac RegionCNNGRU
    -> fused region feature
    -> classifier logits
```

Chi tiet:

- Moi sample thuong co 3 region: `left_hand`, `right_hand`, `face`.
- `valid_mask` danh dau frame nao co crop hop le, giup model khong trung binh ca frame den/fallback vo nghia.
- `RegionResNet18GRU` la backbone chinh: encode frame bang ResNet18, sau do GRU tong hop theo thoi gian cho tung region.
- Fusion giua cac region trong model co the la `concat` hoac `average`.

### 13.4 Gated feature fusion model flow

```text
Skeleton tensor --------------------.
                                     -> skeleton backbone -> skeleton feature --.
Regions tensor ---------------------.                                      |    |
                                     -> region backbone   -> region feature ----+-> project
                                                                                 -> sigmoid gate
                                                                                 -> gate * skeleton + (1-gate) * region
                                                                                 -> classifier logits
```

Chi tiet:

- `dataset.py` pair skeleton va regions theo `sample_id` da normalize.
- `build.py` khoi tao lai 2 backbone tu config/checkpoint cua branch goc.
- `gated_feature_fusion.py` project ca hai feature ve cung `hidden_dim`, tinh gate theo sample, roi tron dac trung truoc khi classifier.
- `train.py` thuong chi train fusion head, giu backbone dong bang hoac fine-tune co kiem soat.

### 13.5 Late fusion flow

Late fusion khac gated fusion:

- Gated fusion tron feature truoc classifier trong mot model chung.
- Late fusion tron logits/xac suat sau khi skeleton va regions da duoc train doc lap.

Script lien quan:

- `scripts/evaluate_skeleton_region_late_fusion.py`
- `configs/fusion/nslt100_skeleton_regions_late_fusion.yaml`

### 13.6 Hand poseflow flow hien tai

Y tuong pipeline:

```text
Standardized hand crops + poseflow signal
    -> two-stream model
    -> logits
```

Nhung trong code hien tai:

- build step chi tao scaffold
- dataset chua doc tensor that
- model generic chi la placeholder
- chua co huong train/eval hoàn chỉnh

## 14. Thu tu file nen doc neu muon onboard nhanh

Neu muon hieu repo tu dau den cuoi, thu tu doc tot nhat la:

1. `README.md`
2. `configs/dataset/wlasl.yaml`
3. `src/slr/data/build_index.py`
4. `src/slr/data/standardize_videos.py`
5. `src/slr/pose/extract_rtmw.py`
6. `src/slr/branches/skeleton/build_inputs.py`
7. `src/slr/branches/regions/build_crops.py`
8. `src/slr/branches/skeleton/train.py`
9. `src/slr/branches/regions/train.py`
10. `src/slr/branches/fusion/build.py`
11. `src/slr/branches/fusion/models/gated_feature_fusion.py`
12. `src/slr/branches/fusion/train.py`

## 15. Ket luan kien truc

Repo nay khong con la scaffold nghien cuu don gian nua. Phan xuong song that cua no nam o ba nhanh:

- skeleton
- regions
- fusion

Ba nhanh nay da co day du:

- preprocessing dau vao
- dataset loader
- model implementation
- training/evaluation entrypoint
- packaging/verification tool phu tro

Phan chua hoan thien chu yeu gom:

- hand-poseflow
- generic `src/slr/models/*`
- generic `src/slr/training/train.py` va `evaluate.py`
- `UI/`

Neu coi repo nhu mot he thong van hanh, luong chinh dang su dung trong thuc te la:

`WLASL raw -> index -> standardize -> RTMW-l pose -> skeleton/regions branch inputs -> train branch -> fusion hoac late fusion -> package`
