# Recognizing Sign Language at the Word Level

Word-level sign language recognition workspace for WLASL with three active backend tracks:

- `skeleton`: RTMW-l pose -> selected keypoints -> graph tensors -> ST-GCN++
- `regions`: RTMW-l guided face and hand crops -> region tensors -> ResNet18-GRU
- `fusion`: gated feature fusion over validated skeleton and regions backbones

## Current layout

- Source code: `src/slr/`
- Active configs: `configs/`
- CLI entrypoints: `scripts/preprocess/`, `scripts/train/`, `scripts/evaluate/`, `scripts/verify/`, `scripts/package/`
- Model metadata: `model_registry/`
- UI-serving artifact slots: `artifacts/serving/`
- React demo UI: `apps/react-ui/`
- React demo backend: `src/slr/demo_ui/` and `scripts/ui/run_react_demo_api.py`
- Historical notes: `docs/history/`
- Current reports: `reports/current/`

## Important constraints

- `data/` is treated as immutable project data and was not reorganized here.
- `UI/` is intentionally untouched. This cleanup only prepares backend, config, and registry support for future UI integration.
- Checkpoints are referenced from metadata; they are not stored inside `model_registry/`.
- For future UI or backend inference integration, prefer `artifacts/serving/` plus `model_registry/registry_serving.yaml` as the stable deployment-facing layout.

## Active config examples

- Skeleton train: `configs/train/skeleton/nslt1000/selected_31/stgcnpp_ce.yaml`
- Regions train: `configs/train/regions/nslt1000/full/region_resnet18_gru_ce.yaml`
- Fusion train: `configs/train/fusion/gated_feature/nslt1000/gated_feature_fusion_ce.yaml`
- Pose extraction: `configs/preprocessing/pose/pose_rtmw_l.yaml`
- Region crop build: `configs/preprocessing/regions/region_crops_nslt1000.yaml`
- Skeleton input build: `configs/build_inputs/skeleton/nslt1000/selected_31.yaml`

## Common commands

```bash
python scripts/preprocess/00_build_index.py
python scripts/preprocess/01_standardize_videos.py --config configs/preprocessing/standardize/standardize_nslt1000.yaml
python scripts/preprocess/02_extract_pose_rtmw.py --config configs/preprocessing/pose/pose_rtmw_l.yaml --subset nslt1000
python scripts/preprocess/03_build_skeleton_inputs.py --config configs/build_inputs/skeleton/nslt1000/selected_31.yaml
python scripts/train/train_skeleton.py --config configs/train/skeleton/nslt1000/selected_31/stgcnpp_ce.yaml
python scripts/train/train_regions.py --config configs/train/regions/nslt1000/full/region_resnet18_gru_ce.yaml
python scripts/train/train_gated_fusion.py --config configs/train/fusion/gated_feature/nslt1000/gated_feature_fusion_ce.yaml
```

## Model Registry

`model_registry/registry.yaml` is the discovery index for future backend/UI selection. The current registry contains:

- `skeleton_nslt1000_sel31_v1`: `ready`
- `regions_nslt1000_face_hands_v1`: `ready`
- `gated_fusion_nslt1000_v1`: `incomplete`

The fusion entry is marked incomplete because this repo snapshot contains validated backbone checkpoints and configs, but no verified local fusion checkpoint.

For the local React demo UI, use `model_registry/registry_serving.yaml` together with `artifacts/serving/`.

## React Demo UI

The repo now includes one local React-based demo that can:

- upload one video
- run prediction through `skeleton`, `regions`, or `fusion`
- read the active subset from backend settings
- resolve the matching `best.pt` from `artifacts/serving/<branch>/<subset>/`

Manual subset switch:

- edit `ACTIVE_SUBSET` in `src/slr/demo_ui/settings.py`
- allowed values: `nslt100`, `nslt300`, `nslt1000`

Run backend:

```bash
python scripts/ui/run_react_demo_api.py
```

Run frontend:

```bash
cd apps/react-ui
npm install
npm run dev
```

Default local addresses:

- backend: `http://127.0.0.1:8008`
- frontend: `http://127.0.0.1:5173`

Important runtime note:

- real prediction depends on your local environment already having the required PyTorch, torchvision, and RTMW/MMPose dependencies available
- the backend uses the existing project pipeline: standardize video -> extract pose -> build skeleton/regions tensors -> load serving checkpoint -> predict

## Documentation

- Structure guide: `docs/architecture/project_structure_guide.md`
- Training guide: `docs/training/training_guide.md`
- Packaging notes: `docs/packaging/`
- Cleanup history and archived reports: `docs/history/`
