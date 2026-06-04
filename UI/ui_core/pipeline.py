from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Callable

from ui_core.inference import (
    RealInferenceNotImplementedError,
    convert_logits_to_probabilities,
    get_mock_or_checkpoint_warning,
    get_required_asset_error,
    get_topk_predictions,
    is_mock_mode_enabled,
    load_runtime_config,
    prepare_model_input,
    run_inference,
)
from ui_core.model_registry import describe_model_assets
from ui_core.video_utils import get_video_metadata, validate_saved_video_path


UI_ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_DIR = UI_ROOT / "outputs" / "predictions"
LOGS_DIR = UI_ROOT / "outputs" / "logs"


def _emit(message: str, callback: Callable[[str], None] | None) -> None:
    if callback is not None:
        callback(message)


def _save_prediction_artifacts(payload: dict) -> tuple[str, str]:
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    predicted_word = str(payload.get("predicted_word", "unknown")).replace(" ", "_")
    prediction_path = PREDICTIONS_DIR / f"{timestamp}_{predicted_word}.json"
    log_path = LOGS_DIR / f"{timestamp}_{predicted_word}.log"
    payload_with_paths = dict(payload)
    payload_with_paths["prediction_path"] = prediction_path.as_posix()
    payload_with_paths["log_path"] = log_path.as_posix()

    prediction_path.write_text(
        json.dumps(payload_with_paths, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log_lines = [
        f"timestamp: {timestamp}",
        f"model_name: {payload.get('model_name')}",
        f"video_path: {payload.get('video_path')}",
        f"predicted_word: {payload.get('predicted_word')}",
        f"confidence: {payload.get('confidence')}",
        f"demo_mode: {payload.get('demo_mode')}",
        f"runtime_mode: {payload.get('runtime_mode')}",
    ]
    for warning in payload.get("warnings", []):
        log_lines.append(f"warning: {warning}")
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    return prediction_path.as_posix(), log_path.as_posix()


def run_prediction_pipeline(
    video_path: str,
    model_name: str,
    status_callback: Callable[[str], None] | None = None,
) -> dict:
    try:
        # 1. Validate video
        _emit("Validating video...", status_callback)
        is_valid, validation_message = validate_saved_video_path(video_path)
        if not is_valid:
            return {"status": "error", "message": validation_message}

        saved_video_path = Path(video_path)
        video_metadata = get_video_metadata(saved_video_path)

        # 2. Save uploaded video
        _emit("Using saved uploaded video...", status_callback)

        # 3. Select model mode
        _emit("Selecting model mode...", status_callback)
        asset_report = describe_model_assets(model_name)
        warnings: list[str] = []
        if is_mock_mode_enabled():
            warning_message = get_mock_or_checkpoint_warning(model_name)
            if warning_message:
                warnings.append(warning_message)
        else:
            asset_error = get_required_asset_error(model_name)
            if asset_error:
                return {"status": "error", "message": asset_error}

        # 4. Preprocess video
        _emit("Preparing model input...", status_callback)
        runtime_config = load_runtime_config(model_name)

        # 5. Build model input
        model_input = prepare_model_input(saved_video_path.as_posix(), model_name, runtime_config)

        # 6. Load checkpoint/config
        _emit("Loading checkpoint and config...", status_callback)
        _ = asset_report

        # 7. Run inference
        _emit("Running inference...", status_callback)
        inference_output = run_inference(saved_video_path.as_posix(), model_name, model_input)

        # 8. Convert logits to probabilities
        _emit("Converting logits to probabilities...", status_callback)
        probabilities = convert_logits_to_probabilities(inference_output["logits"])

        # 9. Get top-k predictions
        _emit("Generating top-k predictions...", status_callback)
        topk = get_topk_predictions(probabilities, inference_output["labels"], k=5)

        # 10. Render result in UI
        _emit("Preparing result for the UI...", status_callback)
        predicted_word = str(topk[0]["word"])
        confidence = float(topk[0]["prob"])

        response = {
            "status": "success",
            "model_name": model_name,
            "predicted_word": predicted_word,
            "confidence": confidence,
            "topk": topk,
            "video_path": saved_video_path.as_posix(),
            "video_metadata": video_metadata,
            "demo_mode": bool(is_mock_mode_enabled()),
            "runtime_mode": asset_report["runtime_mode"],
            "warnings": warnings,
            "backend": inference_output.get("backend", "unknown"),
        }

        # 11. Save prediction JSON/log
        _emit("Saving prediction log...", status_callback)
        prediction_path, log_path = _save_prediction_artifacts(response)
        response["prediction_path"] = prediction_path
        response["log_path"] = log_path
        return response

    except RealInferenceNotImplementedError as exc:
        return {"status": "error", "message": str(exc)}
    except Exception as exc:  # pragma: no cover - final safety net for demo app
        return {"status": "error", "message": str(exc)}
