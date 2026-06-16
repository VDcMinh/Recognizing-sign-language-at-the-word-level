# FINAL_SUMARY_V2

## 1. Muc dich cua ban V2

File nay la ban mo rong cua `FINAL_SUMARY.md`.

Muc tieu cua ban V2:

- trinh bay repo theo kieu "cay thu muc + chu thich"
- di sau hon vao `reports/`
- di sau hon vao `packaging_outputs/`
- phan biet ro:
  - source code runtime
  - config
  - scripts
  - reports/tai lieu noi bo
  - outputs/artifacts sinh ra
  - packages da dong goi de mang di train/verify

Luu y:

- Repo nay co so luong file sinh ra cuc lon.
- Rieng `data/` co hon `757,848` file.
- `packaging_outputs/` co hon `19,072` file.
- Vi vay, cay thu muc duoi day se di den muc "co y nghia kien truc", khong liet ke tung tensor `.npz`.

## 2. Cach nhin repo nay cho dung

Neu chia theo "vai tro", repo gom 6 lop:

1. `src/slr/`: logic runtime that su
2. `configs/`: tham so va layout pipeline
3. `scripts/`: wrapper, check, packaging, eval
4. `data/`: du lieu va intermediate artifacts
5. `outputs/`, `artifacts/`, `packaging_outputs/`: san pham da tao ra trong qua trinh train/package
6. `reports/`, `docs/`, cac file `.md` goc: tai lieu noi bo, bao cao, huong dan

Neu chia theo "dong chay nghiep vu", repo gom 3 khoi chinh dang hoat dong that:

- Skeleton
- Regions
- Fusion

Con `hand_poseflow` hien chua phai pipeline hoan chinh.

## 3. Cay thu muc tong the

```text
Recognizing-sign-language-at-the-word-level/
|-- src/
|   `-- slr/                          # package chinh cua du an
|       |-- data/                     # index + standardization
|       |-- pose/                     # RTMW-l pose extraction + schema
|       |-- branches/                 # skeleton / regions / fusion / hand_poseflow
|       |-- training/                 # loss, optim, metrics, checkpoint helpers
|       |-- inference/                # visualize + placeholder predict
|       |-- utils/                    # io, image, video, bbox, logging, seed
|       `-- models/                   # model namespace generic, phan lon la placeholder
|
|-- configs/                          # dataset + preprocessing + train configs
|-- scripts/                          # wrapper scripts, checks, packaging scripts
|
|-- data/
|   `-- datasets/WLASL/               # raw/index/standardized/pose/branch_inputs
|
|-- outputs/                          # output moi run train/eval
|-- artifacts/                        # artifacts tap trung nhieu o fusion
|-- packaging_outputs/                # package da dong goi de mang di noi khac
|-- checkpoints/                      # pose checkpoint va selected model checkpoints
|
|-- reports/                          # bao cao noi bo, guide, json/csv audit, hinh minh hoa
|-- docs/                             # docs ky thuat ngan gon
|-- experiments/                      # placeholder
|-- UI/                               # rong
|
|-- README.md
|-- PROJECT_STRUCTURE_GUIDE.md
|-- TRAINING.md
|-- IMPLEMENTATION.md
|-- FINAL_SUMARY.md
`-- FINAL_SUMARY_V2.md
```

## 4. Nhan dinh nhanh ve tung khoi

### 4.1 Khoi runtime chinh

```text
src/slr/
|-- data/             # that
|-- pose/             # that
|-- branches/
|   |-- skeleton/     # that
|   |-- regions/      # that
|   |-- fusion/       # that
|   `-- hand_poseflow/# scaffold
|-- training/         # helper that, nhung train/eval generic la placeholder
|-- inference/        # 1 file that, 1 file placeholder
|-- utils/            # that
`-- models/           # chu yeu placeholder
```

Y nghia:

- Neu muon hieu "repo dang chay that nhu the nao", tap trung vao:
  - `src/slr/data/`
  - `src/slr/pose/`
  - `src/slr/branches/skeleton/`
  - `src/slr/branches/regions/`
  - `src/slr/branches/fusion/`
- Neu mo `src/slr/models/` truoc, rat de bi nham vi o do nhieu class placeholder.

### 4.2 Khoi config

```text
configs/
|-- dataset/          # root du lieu WLASL va subset
|-- preprocessing/    # index, standardize, pose, region crop, poseflow
|-- branches/         # preprocessing config theo branch
|-- train/            # config train cho skeleton/regions/fusion
|-- fusion/           # config late fusion
`-- experiments/      # debug/thu nghiem
```

Y nghia:

- `configs/preprocessing/` dieu khien cac buoc sinh du lieu dau vao.
- `configs/branches/` dieu khien cach build branch inputs.
- `configs/train/` dieu khien train model.
- `configs/fusion/` khong phai train gated-fusion, ma la cho late-fusion logits.

### 4.3 Khoi script

```text
scripts/
|-- 00_build_index.py
|-- 01_standardize_videos.py
|-- 02_extract_pose_rtmw.py
|-- 03_build_skeleton_inputs.py
|-- 04_build_region_inputs.py
|-- 05_build_hand_poseflow_inputs.py
|-- train_*.py / evaluate_*.py
|-- check_*.py
|-- package_*.py / prepare_*.py / verify_*.py
|-- tools/
|-- sitecustomize.py
`-- slr/__init__.py
```

Y nghia:

- Day la lop "entrypoint thuc thi".
- Nhieu script packaging va check dac biet phuc vu NSLT1000 regions/fusion.

## 5. Cay du lieu va artifact

### 5.1 `data/datasets/WLASL/`

```text
data/datasets/WLASL/
|-- raw/               # metadata goc va video goc
|-- index/             # manifest tong hop, subset manifests, class maps
|-- standardized/      # video/frame da crop-resize-letterbox
|-- pose/              # RTMW-l output + pose manifests + quality
`-- branch_inputs/     # skeleton/regions/hand_poseflow inputs
```

Y nghia nghiep vu:

- `raw/`: nguon goc
- `index/`: lop "su that metadata" de toan pipeline dung chung
- `standardized/`: du lieu da dua ve 1 chuan khung hinh
- `pose/`: tang trung gian chung cho skeleton va regions
- `branch_inputs/`: tang dau vao rieng cho tung nhanh model

### 5.2 `outputs/`

```text
outputs/
|-- skeleton/         # moi run skeleton
|-- regions/          # moi run regions
|-- fusion/           # moi run fusion
`-- smoke_metrics/    # smoke/sanity outputs
```

Y nghia:

- Day la output "thuc nghiem/train local" cua repo.
- Moi run thuong co:
  - `config_resolved.yaml`
  - `metrics.json`
  - `summary.json`
  - `train_log.csv`
  - `checkpoints/best.pt`
  - `checkpoints/last.pt`

### 5.3 `artifacts/`

```text
artifacts/
`-- fusion/
    |-- nslt100/
    |-- nslt300/
    `-- nslt1000/
```

Y nghia:

- Day khong phai du lieu dau vao ban dau.
- Day la workspace artifact phuc vu build/package fusion.
- Thuong chua:
  - checkpoint branch skeleton/regions da chon
  - resolved config branch
  - logits hoac thong tin can cho fusion/packaging

## 6. `reports/` duoc to chuc nhu the nao

`reports/` la lop tai lieu noi bo lon nhat repo. No khong tham gia runtime truc tiep, nhung cho biet ro lich su phat trien, quyet dinh thiet ke va ket qua audit.

Thong ke tong:

- `experiments/`: 1 file
- `fusion/`: 8 file
- `packaging/`: 6 file
- `preprocessing/`: 23 file
- `regions/`: 25 file
- `training/`: 9 file
- 2 file root lien quan UI

### 6.1 Cay thu muc `reports/`

```text
reports/
|-- ui_runtime_integration_report.md
|-- ui_streamlit_conversion_report.md
|
|-- experiments/
|   `-- README.md
|
|-- fusion/
|   |-- gated_feature_fusion_step1_3_report.md
|   |-- gated_feature_fusion_training_pipeline_report.md
|   |-- gated_fusion_nslt1000_manual_packaging_guide.md
|   |-- gated_fusion_nslt1000_packaging_code_implementation_report.md
|   |-- gated_fusion_nslt1000_packaging_readiness_report.md
|   |-- gated_fusion_nslt1000_requirements_summary.json
|   |-- gated_fusion_nslt1000_requirements_summary_fullpaths.json
|   `-- nslt100_skeleton_regions_late_fusion_implementation_report.md
|
|-- packaging/
|   |-- gated_fusion_nslt100_kaggle_package_report.md
|   |-- gated_fusion_nslt300_kaggle_package_report.md
|   |-- gated_fusion_nslt300_requirement_check_report.md
|   |-- nslt100_branch_inputs_package_report.md
|   |-- nslt100_branch_inputs_sel31_package_report.md
|   `-- regions_nslt300_kaggle_package_report.md
|
|-- preprocessing/
|   |-- README.md
|   |-- hf_sub1000_bundle_report.md
|   |-- hf_sub300_bundle_report.md
|   |-- kaggle_sub300_bundle_report.md
|   |-- nslt1000_index_and_standardization_report.md
|   |-- nslt1000_pose_merge_and_skeleton_build_report.md
|   |-- nslt1000_pose_merge_script_report.md
|   |-- nslt100_regions_branch_implementation_summary.md
|   |-- nslt100_regions_crop_policy_update_report.md
|   |-- nslt100_regions_hf_bundle_report.md
|   |-- nslt300_index_and_standardization_report.md
|   |-- nslt300_regions_summary_and_run_guide.md
|   |
|   |-- face_keypoints/
|   |   `-- nslt100_face_keypoints_sample.png
|   |
|   `-- keypoint_visualizations/
|       `-- nslt100/
|           |-- README.md
|           |-- selected_27/
|           |   |-- test_62975_what.png
|           |   |-- train_62169_walk.png
|           |   `-- val_24641_give.png
|           |-- selected_31/
|           |   |-- test_62975_what.png
|           |   |-- train_62169_walk.png
|           |   `-- val_24641_give.png
|           `-- wholebody_133/
|               |-- test_62975_what.png
|               |-- train_62169_walk.png
|               `-- val_24641_give.png
|
|-- regions/
|   |-- regions_1000_local_train_code_check_report.md
|   |-- regions_nslt1000_extraction_setup_report.md
|   |-- regions_nslt1000_incremental_feasibility_report.md
|   |-- regions_nslt1000_incremental_manual_run_guide.md
|   |-- regions_nslt1000_incremental_to_package_implementation_report.md
|   |-- regions_nslt1000_incremental_zip_report.md
|   |-- regions_nslt1000_setup_report.md
|   |-- regions_nslt1000_training_ready_fix_report.md
|   |-- regions_nslt1000_union_builder_fix_report.md
|   `-- regions_nslt300_setup_report.md
|   |
|   |-- nslt1000_incremental_feasibility/
|   |   |-- class_id_mapping_check.csv
|   |   |-- loader_preview_check.csv
|   |   |-- reusable_tensor_check.csv
|   |   |-- sample_overlap.csv
|   |   |-- split_mapping_check.csv
|   |   `-- summary.json
|   |
|   `-- nslt1000_incremental_pipeline/
|       |-- missing_report.md
|       |-- missing_samples_all.csv
|       |-- missing_summary.json
|       |-- preflight_report.md
|       |-- preflight_summary.json
|       |-- progress_report.md
|       |-- progress_summary.json
|       |-- union_verify_report.md
|       `-- union_verify_summary.json
|
`-- training/
    |-- nslt1000_training_config_report.md
    |-- nslt300_standardls_config_report.md
    |-- nslt300_training_config_report.md
    |-- regions_cnn_gru_overfit_mitigation_config_report.md
    |-- regions_cnn_gru_standardls_config_report.md
    |-- regions_cnn_gru_training_implementation_report.md
    |-- regions_cnn_gru_weight_decay_check_report.md
    |-- regions_resnet18_gru_hands_only_ablation_report.md
    `-- regions_resnet18_gru_implementation_report.md
```

### 6.2 Giai thich tung cum trong `reports/`

#### A. Hai file UI o root `reports/`

| File | Y nghia |
| --- | --- |
| `reports/ui_runtime_integration_report.md` | Ghi chu ve viec gan UI vao runtime/pipeline |
| `reports/ui_streamlit_conversion_report.md` | Ghi chu ve huong Streamlit hoa giao dien |

Nhan xet:

- Co tai lieu UI, nhung `UI/` hien rong.
- Nghia la huong UI da duoc nghi/bao cao, nhung code giao dien chua thanh mot module on dinh trong repo.

#### B. `reports/experiments/`

| File | Y nghia |
| --- | --- |
| `reports/experiments/README.md` | Noi ghi chu cho cac thu nghiem, hien rat nhe |

Nhan xet:

- Thu muc nay hien chua phat trien thanh mot nhom artifact lon.

#### C. `reports/fusion/`

Cum nay gom tai lieu cho hai huong:

- gated feature fusion
- late fusion

Y nghia cua tung file:

| File | Y nghia |
| --- | --- |
| `gated_feature_fusion_step1_3_report.md` | Bao cao cac buoc dau cua pipeline fusion |
| `gated_feature_fusion_training_pipeline_report.md` | Tong ket train pipeline cua gated fusion |
| `gated_fusion_nslt1000_manual_packaging_guide.md` | Huong dan package NSLT1000 bang tay |
| `gated_fusion_nslt1000_packaging_code_implementation_report.md` | Giai thich code package fusion NSLT1000 |
| `gated_fusion_nslt1000_packaging_readiness_report.md` | Danh gia package da san sang chua |
| `gated_fusion_nslt1000_requirements_summary.json` | Tong hop machine-readable ve yeu cau package |
| `gated_fusion_nslt1000_requirements_summary_fullpaths.json` | Ban day du duong dan tuyet doi |
| `nslt100_skeleton_regions_late_fusion_implementation_report.md` | Bao cao implementation late fusion cho NSLT100 |

Nhan xet:

- Day la thu muc "thiet ke + readiness + packaging" cua fusion.
- Cac file JSON o day mang tinh audit may doc, rat hop de script/package tiep tuc su dung.

#### D. `reports/packaging/`

Cum nay la bao cao package theo dataset/subset:

| File | Y nghia |
| --- | --- |
| `gated_fusion_nslt100_kaggle_package_report.md` | Bao cao package fusion NSLT100 |
| `gated_fusion_nslt300_kaggle_package_report.md` | Bao cao package fusion NSLT300 |
| `gated_fusion_nslt300_requirement_check_report.md` | Kiem tra dieu kien package fusion NSLT300 |
| `nslt100_branch_inputs_package_report.md` | Package branch inputs NSLT100 |
| `nslt100_branch_inputs_sel31_package_report.md` | Package branch inputs NSLT100 cho layout `selected_31` |
| `regions_nslt300_kaggle_package_report.md` | Package regions NSLT300 |

Nhan xet:

- `reports/packaging/` la lop "story" va "outcome".
- `packaging_outputs/` moi la san pham package that.

#### E. `reports/preprocessing/`

Day la cum reports nhieu ngoc nhat truoc khi vao training.

Nhom file markdown:

| File | Y nghia |
| --- | --- |
| `README.md` | Mo ta chung cho cum preprocessing |
| `hf_sub1000_bundle_report.md` | Bao cao bundle cho subset 1000 theo huong HF |
| `hf_sub300_bundle_report.md` | Bao cao bundle cho subset 300 theo huong HF |
| `kaggle_sub300_bundle_report.md` | Bao cao bundle cho Kaggle subset 300 |
| `nslt1000_index_and_standardization_report.md` | Ket qua index + standardize cho NSLT1000 |
| `nslt1000_pose_merge_and_skeleton_build_report.md` | Merge pose va build skeleton dau vao cho NSLT1000 |
| `nslt1000_pose_merge_script_report.md` | Bao cao rieng cho script merge pose |
| `nslt100_regions_branch_implementation_summary.md` | Tong ket implementation branch regions cho NSLT100 |
| `nslt100_regions_crop_policy_update_report.md` | Giai thich thay doi chinh sach crop tay/mat |
| `nslt100_regions_hf_bundle_report.md` | Bao cao bundle regions cho HF |
| `nslt300_index_and_standardization_report.md` | Index + standardize cho NSLT300 |
| `nslt300_regions_summary_and_run_guide.md` | Tong ket regions NSLT300 va cach chay |

Nhom hinh minh hoa:

| File/thu muc | Y nghia |
| --- | --- |
| `face_keypoints/nslt100_face_keypoints_sample.png` | Anh minh hoa face keypoints |
| `keypoint_visualizations/nslt100/README.md` | Giai thich cum anh visualize keypoints |
| `keypoint_visualizations/nslt100/selected_27/*.png` | Vi du mau keypoint set `selected_27` |
| `keypoint_visualizations/nslt100/selected_31/*.png` | Vi du mau keypoint set `selected_31` |
| `keypoint_visualizations/nslt100/wholebody_133/*.png` | Vi du pose day du wholebody_133 |

Nhan xet:

- `reports/preprocessing/` la cau noi rat tot giua code preprocessing va ket qua qualitative.
- Neu can kiem tra "keypoint set da chon co hop ly khong", day la noi nen doc truoc.

#### F. `reports/regions/`

Day la thu muc co mat do van hanh va audit cao nhat trong `reports/`.

Nhom markdown chinh:

| File | Y nghia |
| --- | --- |
| `regions_1000_local_train_code_check_report.md` | Kiem tra code train local cua regions 1000 |
| `regions_nslt1000_extraction_setup_report.md` | Setup extraction cho NSLT1000 |
| `regions_nslt1000_incremental_feasibility_report.md` | Danh gia incremental strategy co kha thi khong |
| `regions_nslt1000_incremental_manual_run_guide.md` | Huong dan chay tay pipeline incremental |
| `regions_nslt1000_incremental_to_package_implementation_report.md` | Giai thich tu extraction incremental sang package |
| `regions_nslt1000_incremental_zip_report.md` | Bao cao dong goi zip incremental |
| `regions_nslt1000_setup_report.md` | Bao cao setup tong quat cho NSLT1000 regions |
| `regions_nslt1000_training_ready_fix_report.md` | Ghi lai cac sua de training san sang |
| `regions_nslt1000_union_builder_fix_report.md` | Ghi lai cac sua cho union manifest builder |
| `regions_nslt300_setup_report.md` | Bao cao setup cho NSLT300 regions |

Nhom `nslt1000_incremental_feasibility/`:

| File | Y nghia |
| --- | --- |
| `class_id_mapping_check.csv` | Kiem tra map nhan lop |
| `loader_preview_check.csv` | Preview dataset loader tren du lieu incremental |
| `reusable_tensor_check.csv` | Kiem tra tensor nao tai su dung duoc tu NSLT300 |
| `sample_overlap.csv` | Kiem tra overlap sample |
| `split_mapping_check.csv` | Kiem tra split mapping |
| `summary.json` | Tong hop feasibility theo JSON |

Nhom `nslt1000_incremental_pipeline/`:

| File | Y nghia |
| --- | --- |
| `missing_report.md` | Bao cao mau thieu |
| `missing_samples_all.csv` | Danh sach mau thieu day du |
| `missing_summary.json` | Tong hop machine-readable cho phan thieu |
| `preflight_report.md` | Bao cao preflight truoc khi chay pipeline |
| `preflight_summary.json` | Tong hop preflight |
| `progress_report.md` | Bao cao tien do pipeline |
| `progress_summary.json` | Tong hop tien do |
| `union_verify_report.md` | Bao cao verify union manifests/package |
| `union_verify_summary.json` | Tong hop verify theo JSON |

Nhan xet:

- Thu muc nay cho thay regions NSLT1000 la noi repo da dau tu rat nhieu cong suc de giai bai toan packaging tang dan.
- Rat nhieu file o day la dau vet cua quy trinh "incremental -> union -> package".

#### G. `reports/training/`

Cum nay tap trung vao model/config training:

| File | Y nghia |
| --- | --- |
| `nslt1000_training_config_report.md` | Tong ket config train cho NSLT1000 |
| `nslt300_standardls_config_report.md` | Tong ket config StandardLS cho NSLT300 |
| `nslt300_training_config_report.md` | Tong ket config train cho NSLT300 |
| `regions_cnn_gru_overfit_mitigation_config_report.md` | Bao cao giam overfit cho CNN-GRU cu |
| `regions_cnn_gru_standardls_config_report.md` | Bao cao StandardLS cho CNN-GRU cu |
| `regions_cnn_gru_training_implementation_report.md` | Bao cao implementation CNN-GRU cu |
| `regions_cnn_gru_weight_decay_check_report.md` | Bao cao kiem tra weight decay |
| `regions_resnet18_gru_hands_only_ablation_report.md` | Ablation chi dung 2 tay |
| `regions_resnet18_gru_implementation_report.md` | Bao cao implementation backbone regions hien dung |

Nhan xet:

- Day la noi cho thay su chuyen dich tu baseline `regions_cnn_gru` cu sang `region_resnet18_gru`.
- Cum nay cuc huu ich de hieu "vi sao config va backbone hien tai trong repo lai nhu vay".

### 6.3 Ket luan ve `reports/`

`reports/` trong repo nay khong phai thu muc trang tri. No la:

- nhat ky phat trien
- noi giai thich quyet dinh ky thuat
- audit trail cho preprocessing/packaging
- bo bang chung de doi chieu voi artifact da tao ra

Neu muon audit nhanh repo ma khong doc het code, 4 noi nen doc truoc la:

1. `reports/preprocessing/`
2. `reports/regions/`
3. `reports/fusion/`
4. `reports/training/`

## 7. `packaging_outputs/` duoc to chuc nhu the nao

`packaging_outputs/` la noi chua package da dong goi xong. Khac voi `reports/`, day la san pham co the mang sang moi truong khac de verify/train.

Top-level hien tai:

```text
packaging_outputs/
|-- gated_fusion_nslt1000_build_plan.json
|-- wlasl-nslt1000-gated-fusion-ready.sha256.txt
|-- wlasl-nslt1000-gated-fusion-ready.zip
|-- wlasl-nslt1000-gated-fusion-ready/
`-- wlasl-nslt1000-regions-rtmw-l-incremental/
```

Y nghia:

- `gated_fusion_nslt1000_build_plan.json`: mo ta ke hoach build package fusion
- `wlasl-nslt1000-gated-fusion-ready.sha256.txt`: checksum cho package fusion
- `wlasl-nslt1000-gated-fusion-ready.zip`: file package nen san pham cuoi
- Hai thu muc con la hai package da materialize

### 7.1 Package 1: `wlasl-nslt1000-gated-fusion-ready/`

Thong tin metadata noi bat:

- subset: `nslt1000`
- num_classes: `1000`
- skeleton model: `stgcnpp`
- regions model: `region_resnet18_gru`
- fusion hidden dim: `256`
- keypoint set skeleton: `selected_31`
- split counts:
  - train: `5001`
  - val: `1290`
  - test: `941`
  - total: `7232`
- tensor counts:
  - skeleton: `7232`
  - regions: `7232`
- link mode khi build local: `hardlink`

#### Cay package

```text
packaging_outputs/wlasl-nslt1000-gated-fusion-ready/
|-- README.md
|-- metadata.json
|-- build_state.json
|
|-- checkpoints/                         # 2 file
|   |-- skeleton/
|   |   `-- best.pt
|   `-- regions/
|       `-- best.pt
|
|-- configs/                             # 3 file
|   |-- gated_feature_fusion_nslt1000_kaggle.yaml
|   |-- skeleton_config_resolved.yaml
|   `-- regions_config_resolved.yaml
|
|-- branch_inputs/                       # 14,470 file
|   |-- skeleton/
|   |   `-- rtmw_l/
|   |       |-- manifests/
|   |       |   |-- nslt1000_selected_31_train.csv
|   |       |   |-- nslt1000_selected_31_val.csv
|   |       |   `-- nslt1000_selected_31_test.csv
|   |       `-- tensors/
|   |           `-- nslt1000/
|   |               |-- train/           # 5001 tensor
|   |               |-- val/             # 1290 tensor
|   |               `-- test/            # 941 tensor
|   |
|   `-- regions/
|       `-- rtmw_l/
|           |-- manifests/
|           |   |-- nslt1000_train.csv
|           |   |-- nslt1000_val.csv
|           |   `-- nslt1000_test.csv
|           `-- tensors/
|               `-- nslt1000/
|                   |-- train/           # 5001 tensor
|                   |-- val/             # 1290 tensor
|                   `-- test/            # 941 tensor
|
|-- verify/                              # 2 file
|   |-- verify_package.py
|   `-- verify_summary.json
|
`-- packaging_outputs/                   # verify artifact long ben trong package
    `-- wlasl-nslt1000-gated-fusion-ready/
        `-- verify/
            |-- verify_summary.json
            `-- verify_summary_full.json
```

#### Giai thich y nghia tung file/chum

| Muc | Y nghia |
| --- | --- |
| `README.md` | Huong dan package, split counts, expected shapes, sample ID policy, lenh verify/train tren Kaggle |
| `metadata.json` | Thong tin may doc ve model, shape, split, policy, build summary |
| `build_state.json` | Ban do cuc chi tiet sample nao duoc hardlink/copy tu dau den dau; file rat lon |
| `checkpoints/` | Hai backbone da train xong can cho fusion |
| `configs/` | Fusion config va hai resolved config cua branch skeleton/regions |
| `branch_inputs/skeleton/...` | Skeleton package input canonically materialized |
| `branch_inputs/regions/...` | Regions package input canonically materialized |
| `verify/verify_package.py` | Script verify package sau khi dong goi hoac sau khi mount len Kaggle |
| `verify/verify_summary.json` | Ket qua verify: alignment, shape, checkpoint, smoke test |
| `packaging_outputs/.../verify_summary_full.json` | Ban verify chi tiet hon, luu long-form audit trong package |

#### Package nay duoc thiet ke de lam gi

Muc dich:

- La package "all-in-one" de train gated fusion tren Kaggle ma khong can dung them package con nao khac.

The hien qua metadata/README:

- Chua du checkpoint cua skeleton va regions
- Chua du resolved configs
- Chua du branch inputs cho ca skeleton va regions
- Co verify script va verify summary di kem

#### Package nay "hoan chinh" den dau

Theo `verify/verify_summary.json`:

- status: `pass`
- matched samples: `7232/7232`
- khong co duplicate/collision sample IDs
- smoke forward pass tren CPU thanh cong
- checkpoint skeleton:
  - model `stgcnpp`
  - feature dim `256`
- checkpoint regions:
  - model `region_resnet18_gru`
  - feature dim `768`

Nghia la package nay khong chi la zip file, ma da duoc verify o muc:

- config
- checkpoint
- manifest alignment
- sample ID normalization
- mot forward pass thuc te

### 7.2 Package 2: `wlasl-nslt1000-regions-rtmw-l-incremental/`

Thong tin metadata noi bat:

- subset: `nslt1000`
- num_classes: `1000`
- purpose: `Missing-only Regions tensors for incremental NSLT1000 construction`
- format: `incremental_two_source`
- base subset can co san: `nslt300`
- base dataset required: `true`
- active regions:
  - `left_hand`
  - `right_hand`
  - `face`
- expected shape: `[3, 3, 64, 112, 112]`
- counts:
  - total NSLT1000: `7232`
  - reused NSLT300: `2660`
  - incremental new: `4572`
  - train new: `3104`
  - val new: `844`
  - test new: `624`

#### Cay package

```text
packaging_outputs/wlasl-nslt1000-regions-rtmw-l-incremental/
|-- README.md
|-- metadata.json
|
|-- configs/                               # 1 file
|   `-- regions_resnet18_gru_nslt1000_incremental_kaggle.yaml.template
|
|-- manifests/                             # 6 file
|   |-- logical/
|   |   |-- nslt1000_train.csv
|   |   |-- nslt1000_val.csv
|   |   `-- nslt1000_test.csv
|   `-- missing/
|       |-- nslt1000_missing_train.csv
|       |-- nslt1000_missing_val.csv
|       `-- nslt1000_missing_test.csv
|
|-- regions/                               # 4572 file
|   `-- rtmw_l_incremental/
|       `-- tensors/
|           `-- nslt1000/
|               |-- train/                 # 3104 tensor moi
|               |-- val/                   # 844 tensor moi
|               `-- test/                  # 624 tensor moi
|
|-- reports/                               # 3 file
|   |-- extraction_summary.json
|   |-- package_report.md
|   `-- union_verify_summary.json
|
|-- scripts/                               # 2 file
|   |-- materialize_regions_nslt1000_kaggle_manifests.py
|   `-- verify_incremental_package.py
|
`-- verify/                                # 1 file
    `-- verify_summary.json
```

#### Giai thich y nghia tung file/chum

| Muc | Y nghia |
| --- | --- |
| `README.md` | Noi ro day la incremental package; khong duplicate NSLT300 tensors; can attach package base NSLT300 khi train tren Kaggle |
| `metadata.json` | Mo ta format incremental-two-source va thong ke reused/new tensors |
| `configs/*.template` | Template config de train trong moi truong attach package |
| `manifests/logical/*.csv` | Full logical manifest cho NSLT1000 |
| `manifests/missing/*.csv` | Chi danh sach mau can them vao so voi base NSLT300 |
| `regions/rtmw_l_incremental/tensors/...` | Chi chua 4572 tensor con thieu, khong chua 2660 tensor da co trong NSLT300 |
| `reports/` | Audit package va extraction summary |
| `scripts/materialize_regions_nslt1000_kaggle_manifests.py` | Script tao runtime manifests khi mount package tren Kaggle |
| `scripts/verify_incremental_package.py` | Script verify package incremental |
| `verify/verify_summary.json` | Ket qua verify package |

#### Package nay duoc thiet ke de lam gi

Muc dich:

- Khong dong goi lai toan bo 7232 tensor regions cua NSLT1000.
- Chi dong goi 4572 tensor "phan con thieu".
- Khi dua len Kaggle, can mount:
  - package base NSLT300
  - package incremental nay
- Sau do materialize logical manifests de co du full NSLT1000 runtime view.

Nghia la day la package "tiet kiem dung luong", khong phai package standalone.

#### Package nay "hoan chinh" den dau

Theo `verify/verify_summary.json`:

- status: `pass`
- khong thieu file bat buoc
- logical counts:
  - train: `5001`
  - val: `1290`
  - test: `941`

Nghia la ve mat package logic, no hop le theo mo hinh incremental hai nguon.

### 7.3 Moi quan he giua hai package lon

Quan he rat quan trong:

| Package | Kieu | Doc lap hay phu thuoc | Muc dich |
| --- | --- | --- | --- |
| `wlasl-nslt1000-gated-fusion-ready` | Full package | Gan nhu doc lap | Train gated fusion ngay tren Kaggle |
| `wlasl-nslt1000-regions-rtmw-l-incremental` | Incremental package | Phu thuoc base NSLT300 | Hoan tat regions NSLT1000 ma khong duplicate du lieu cu |

Noi cach khac:

- Package fusion la "san pham cuoi"
- Package regions incremental la "mau ghep trung gian/chien luoc tiet kiem kich thuoc"

### 7.4 Y nghia cua cac file top-level trong `packaging_outputs/`

| File | Y nghia |
| --- | --- |
| `gated_fusion_nslt1000_build_plan.json` | Ke hoach/blueprint build package fusion NSLT1000 |
| `wlasl-nslt1000-gated-fusion-ready.sha256.txt` | Checksum de kiem tra tinh toan ven cua zip/package |
| `wlasl-nslt1000-gated-fusion-ready.zip` | Ban dong goi nem duoc/mang di ngay |

## 8. So sanh `reports/`, `outputs/`, `artifacts/`, `packaging_outputs/`

Day la diem de nham lan nhat trong repo.

| Thu muc | Ban chat | Dung luc nao |
| --- | --- | --- |
| `reports/` | Tai lieu va audit trail | Khi can hieu quyet dinh, setup, tien do, feasibility |
| `outputs/` | Ket qua train/eval cua tung run | Khi can checkpoint/metrics cua mot lan chay cu the |
| `artifacts/` | Workspace artifact trung gian, nhat la cho fusion | Khi can tap hop checkpoint/config da chon cho packaging |
| `packaging_outputs/` | San pham package da dong goi | Khi can chuyen du lieu/model sang Kaggle hay moi truong khac |

Co the hieu nhu sau:

- `reports/` = "vi sao va da lam gi"
- `outputs/` = "lan train/eval nay cho ra gi"
- `artifacts/` = "nhung thanh phan da duoc chon de dung tiep"
- `packaging_outputs/` = "goi da ship duoc"

## 9. Thu tu doc neu muon audit phan package va report

Neu muon hieu nhanh nhat phan report + package, toi khuyen doc theo thu tu nay:

1. `FINAL_SUMARY.md`
2. `reports/preprocessing/README.md`
3. `reports/regions/regions_nslt1000_incremental_feasibility_report.md`
4. `reports/regions/regions_nslt1000_incremental_manual_run_guide.md`
5. `reports/fusion/gated_fusion_nslt1000_packaging_readiness_report.md`
6. `packaging_outputs/wlasl-nslt1000-regions-rtmw-l-incremental/README.md`
7. `packaging_outputs/wlasl-nslt1000-regions-rtmw-l-incremental/metadata.json`
8. `packaging_outputs/wlasl-nslt1000-gated-fusion-ready/README.md`
9. `packaging_outputs/wlasl-nslt1000-gated-fusion-ready/metadata.json`
10. `packaging_outputs/wlasl-nslt1000-gated-fusion-ready/verify/verify_summary.json`

## 10. Ket luan cua ban V2

Neu `FINAL_SUMARY.md` tra loi cau hoi "repo nay hoat dong nhu the nao", thi ban V2 tra loi them hai cau hoi:

1. `reports/` dung de ghi lai va audit nhung gi?
2. `packaging_outputs/` duoc to chuc nhu the nao de co the ship sang Kaggle/production-like environment?

Ket luan sau khi doc toan bo cau truc lien quan:

- `reports/` la bo nho to chuc cua du an
- `outputs/` la ket qua cua tung lan chay
- `artifacts/` la workspace chon loc cho package/fusion
- `packaging_outputs/` la san pham giao nhan that su

Va o thoi diem hien tai, phan package truong thanh nhat cua repo la:

- NSLT1000 regions incremental packaging
- NSLT1000 gated-fusion ready packaging

Hai package nay cho thay repo da di xa hon muc "code nghien cuu", va da co tu duy:

- tai su dung artifact
- kiem toan sample ID
- verify package
- layout cho Kaggle
- tach full package va incremental package theo chi phi luu tru
