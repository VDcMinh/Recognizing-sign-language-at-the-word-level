# UI Implementation Report

## 1. Goal
Build a dedicated `UI/` demo package for a word-level sign language recognition app using Streamlit, while keeping the training code untouched unless needed.

## 2. Design Requirements
- Beige outer background
- White rounded dashboard shell
- Plum and pink accent palette
- No green or blue in the main UI theme
- Desktop two-column layout with mobile-friendly stacking
- Upload, recent videos, result card, and prediction details

## 3. Folder Structure
The requested `UI/` folder structure has been created, including `checkpoints/`, `configs/`, `assets/`, `outputs/`, and `ui_core/`.

## 4. Files Added
- `UI/app.py`
- `UI/requirements.txt`
- `UI/README_UI.md`
- `UI/UI_IMPLEMENTATION_REPORT.md`
- `UI/ui_core/__init__.py`
- `UI/ui_core/inference.py`
- `UI/ui_core/model_registry.py`
- `UI/ui_core/video_utils.py`
- `UI/ui_core/pipeline.py`
- `UI/ui_core/styles.py`
- `UI/ui_core/state.py`
- Checkpoint README placeholders and `.gitkeep` files

## 5. UI Layout
The app renders a premium-looking shell with:

- Top action row for `Predict` and `Models`
- Left column for upload, preview, and recent videos
- Right column for the predicted word and detailed probabilities

## 6. Color Palette
- Background: `#FAF7F5`
- Card white: `#FFFFFF`
- Dark panel: `#241D2B`
- Accent plum: `#9B3F68`
- Accent light pink: `#D98CA6`
- Accent peach pink: `#F2B8C6`
- Main text: `#1F1D2B`
- Secondary text: `#77717C`
- Border light: `#E8DEE3`

## 7. Model Selection
The UI exposes exactly two choices:

- `Skeleton`
- `Skeleton + Fusion`

The internal registry supports:

- Skeleton single checkpoint
- Fusion single checkpoint
- Fusion late-fusion layout with separate skeleton and regions checkpoints

## 8. Checkpoint Locations
Skeleton:

- `UI/checkpoints/skeleton/best.pt`
- `UI/configs/skeleton/config_resolved.yaml`

Skeleton + Fusion single-checkpoint mode:

- `UI/checkpoints/fusion/best.pt`
- `UI/configs/fusion/config_resolved.yaml`

Skeleton + Fusion late-fusion mode:

- `UI/checkpoints/fusion/skeleton_best.pt`
- `UI/checkpoints/fusion/regions_best.pt`
- `UI/configs/fusion/skeleton_config_resolved.yaml`
- `UI/configs/fusion/regions_config_resolved.yaml`
- `UI/configs/fusion/fusion_config.yaml`

## 9. Upload Flow
Upload flow implemented in the UI:

1. Validate extension and size
2. Save the uploaded file under `UI/outputs/uploads/`
3. Extract metadata with OpenCV
4. Add the saved clip to `st.session_state["recent_videos"]`
5. Auto-select the latest uploaded clip for preview and prediction

## 10. Prediction Pipeline
The orchestration lives in `UI/ui_core/pipeline.py` and explicitly performs:

1. Validate video
2. Save uploaded video
3. Select model mode
4. Preprocess video
5. Build model input
6. Load checkpoint and config
7. Run inference
8. Convert logits to probabilities
9. Get top-k predictions
10. Render result in UI
11. Save prediction JSON and log

## 11. Mock Mode / Real Inference Status
Real raw-video inference is partially connected.

What the repo already has:

- pose extraction modules under `src/slr/pose/`
- skeleton and regions training/evaluation code
- a late-fusion evaluation script for branch logits

Current status:

- default UI mode: mock
- real `Skeleton` inference: connected
- real `Skeleton + Fusion` inference: not connected yet
- signal in UI: warning card explains mock mode when enabled
- signal in artifacts: prediction JSON stores `demo_mode: true` or `false`

Implemented real Skeleton flow:

- uploaded video
- frame extraction and resize to RTMW-l pose input resolution
- RTMW-l whole-body pose extraction
- selected keypoint subset build
- xy normalization and confidence normalization
- fixed-length graph tensor build
- ST-GCN++ checkpoint forward pass
- logits to probabilities in the shared UI pipeline

Still missing:

- raw-video regions preprocessing for fusion
- real fusion logits execution and combination for uploaded videos

Files to extend later:

- `UI/ui_core/inference.py`

Functions still needing future work:

- `build_fusion_inputs_from_video()`
- fusion branch inside `_run_real_inference()`

## 12. Error Handling
Implemented cases:

- No video uploaded
- Unsupported file format
- File too large
- Missing checkpoint with mock mode enabled
- Missing checkpoint with mock mode disabled
- Real inference requested before adapter implementation

## 13. How to Run
```bash
pip install -r UI/requirements.txt
streamlit run UI/app.py
```

To support 500MB uploads through Streamlit:

```bash
streamlit run UI/app.py --server.maxUploadSize 500
```

## 14. Smoke Test Results
Executed smoke tests:

- `python -m py_compile UI/app.py`
- `python -m py_compile UI/ui_core/inference.py`
- `python -m py_compile UI/ui_core/model_registry.py`
- `python -m py_compile UI/ui_core/video_utils.py`
- `python -m py_compile UI/ui_core/pipeline.py`
- `python -m py_compile UI/ui_core/styles.py`
- `python -m py_compile UI/ui_core/state.py`

Result:

- `py_compile`: pass
- Verified command:
  `python -m py_compile UI/app.py UI/ui_core/inference.py UI/ui_core/model_registry.py UI/ui_core/video_utils.py UI/ui_core/pipeline.py UI/ui_core/styles.py UI/ui_core/state.py`

Streamlit launch smoke test:

- pass in project virtual environment
- Verified command:
  `.venv-rtmw310\\Scripts\\python.exe -m streamlit run UI/app.py --server.headless true --server.port 8510 --server.maxUploadSize 500`
- Observed startup signal:
  Streamlit started and exposed `http://localhost:8510`, then the process was stopped after the smoke check

Real Skeleton inference smoke test:

- pass
- Demo mode disabled with `UI_DEMO_MOCK=0`
- Verified via `run_prediction_pipeline(..., 'Skeleton')` on a sample uploaded video
- Produced a real prediction artifact under `UI/outputs/predictions/`

## 15. Known Limitations
- Real single-video inference from raw upload is currently connected for `Skeleton` only
- `Skeleton + Fusion` real uploaded-video inference is still not implemented
- Mock mode is enabled by default for a smooth demo experience
- Streamlit server upload size may need `--server.maxUploadSize 500`
- Recent videos are session-based only and are not persisted in a database

## 16. Next Steps
1. Implement real uploaded-video preprocessing for the regions branch
2. Add real `Skeleton + Fusion` execution for raw videos
3. Expose device and inference diagnostics in the UI if needed
4. Keep mock mode as a fallback while broadening real model coverage
