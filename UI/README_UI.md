# Word-level Sign Language Recognition UI Demo

## 1. Overview
This folder contains a self-contained Streamlit demo UI for word-level sign language recognition. The interface is designed to run even when real checkpoints are not available by using a clearly labeled mock mode.

## 2. UI Design
The UI uses a soft dashboard layout with a beige background, white rounded shell, plum and pink accents, and dark upload/result panels. The main desktop layout uses two columns:

- Left: upload area, video preview, recent videos
- Right: result card and prediction details

On smaller screens, Streamlit naturally stacks the layout into one column.

## 3. Folder Structure
```text
UI/
├── app.py
├── requirements.txt
├── README_UI.md
├── UI_IMPLEMENTATION_REPORT.md
├── checkpoints/
│   ├── skeleton/
│   │   ├── .gitkeep
│   │   └── README.md
│   └── fusion/
│       ├── .gitkeep
│       └── README.md
├── configs/
│   ├── skeleton/
│   │   └── .gitkeep
│   └── fusion/
│       └── .gitkeep
├── assets/
│   └── .gitkeep
├── outputs/
│   ├── uploads/
│   │   └── .gitkeep
│   ├── predictions/
│   │   └── .gitkeep
│   └── logs/
│       └── .gitkeep
└── ui_core/
    ├── __init__.py
    ├── inference.py
    ├── model_registry.py
    ├── video_utils.py
    ├── pipeline.py
    ├── styles.py
    └── state.py
```

## 4. Model Modes
The UI exposes two user-facing model choices only:

- `Skeleton`
- `Skeleton + Fusion`

The fusion option is flexible enough to support either:

- Single fused checkpoint
- Late fusion between a skeleton branch and a regions branch

## 5. Checkpoint Placement
Skeleton:

- `UI/checkpoints/skeleton/best.pt`

Skeleton config:

- `UI/configs/skeleton/config_resolved.yaml`

Skeleton + Fusion, option A, single checkpoint:

- `UI/checkpoints/fusion/best.pt`
- `UI/configs/fusion/config_resolved.yaml`

Skeleton + Fusion, option B, late fusion:

- `UI/checkpoints/fusion/skeleton_best.pt`
- `UI/checkpoints/fusion/regions_best.pt`
- `UI/configs/fusion/skeleton_config_resolved.yaml`
- `UI/configs/fusion/regions_config_resolved.yaml`
- `UI/configs/fusion/fusion_config.yaml`

## 6. Config Placement
Place resolved YAML configs under:

- `UI/configs/skeleton/`
- `UI/configs/fusion/`

The UI falls back to reading repo training or fusion configs when the UI-resolved config files are not present, but real inference is still not connected yet.

## 7. How to Run
Install dependencies:

```bash
pip install -r UI/requirements.txt
```

Run the app:

```bash
streamlit run UI/app.py
```

If you want to allow uploads up to 500MB through Streamlit itself, launch with:

```bash
streamlit run UI/app.py --server.maxUploadSize 500
```

## 8. Prediction Flow
The UI pipeline is intentionally explicit:

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

The entry point is:

```python
run_prediction_pipeline(video_path: str, model_name: str) -> dict
```

## 9. Mock Mode vs Real Inference
The demo uses mock mode by default:

- `UI_DEMO_MOCK=1` or unset: mock mode enabled
- `UI_DEMO_MOCK=0`: mock mode disabled

When mock mode is enabled, the UI writes `demo_mode: true` into prediction artifacts and shows warnings in the UI if checkpoints are missing.

When mock mode is disabled:

- `Skeleton` runs real inference if the skeleton checkpoint, config, and RTMW-l pose assets are available
- `Skeleton + Fusion` still returns a clear not-implemented message for real raw-video inference

## 10. How to Connect Real Inference
The repo already contains useful building blocks:

- `src/slr/pose/extract_rtmw.py`
- `src/slr/branches/skeleton/`
- `src/slr/branches/regions/`
- `scripts/evaluate_skeleton.py`
- `scripts/evaluate_skeleton_region_late_fusion.py`

Current real-inference status:

1. `Skeleton`
   already connected for `uploaded video -> RTMW-l pose -> selected keypoints -> normalized graph tensor -> ST-GCN++ checkpoint -> logits`
2. `Skeleton + Fusion`
   still needs raw-video regions preprocessing and fusion execution

To complete real fusion inference later:

1. Implement `build_fusion_inputs_from_video()` in `UI/ui_core/inference.py`
2. Extend `_run_real_inference()` in `UI/ui_core/inference.py` for fusion execution
3. Keep `UI/ui_core/pipeline.py` as the orchestration layer

## 11. Troubleshooting
- `Unsupported file format`: upload MP4, MOV, AVI, or MKV only
- `File is too large`: keep the upload below 500MB
- `Checkpoint not found for selected model`: place the expected checkpoint files in `UI/checkpoints/...`
- `Real fusion inference is not implemented yet`: only the `Skeleton` model is currently wired for real uploaded-video inference
