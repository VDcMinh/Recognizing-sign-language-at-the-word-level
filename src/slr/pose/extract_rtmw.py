"""Extract shared RTMW-l whole-body pose from standardized WLASL inputs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from slr.data.manifests import POSE_MANIFEST_COLUMNS, STANDARDIZED_COLUMNS
from slr.data.validation import require_columns, validate_manifest_schema
from slr.pose.pose_quality import (
    compute_mean_confidence,
    compute_region_mean_confidence,
    compute_valid_frames,
    summarize_pose_manifest,
)
from slr.pose.pose_schema import (
    RTMW_L_BACKEND,
    WHOLEBODY_133_LAYOUT,
    WHOLEBODY_133_NUM_KEYPOINTS,
    validate_keypoints_shape,
)
from slr.utils.io import ensure_dir, read_csv, read_yaml, write_dataframe_csv, write_text
from slr.utils.logging import setup_logger


DEFAULT_CONFIG_PATH = Path("configs/preprocessing/pose_rtmw_l.yaml")
ALLOWED_SPLITS = ("train", "val", "test")
LOGGER = setup_logger(__name__)


def parse_args() -> argparse.ArgumentParser:
    """Create the CLI parser for RTMW-l extraction."""

    parser = argparse.ArgumentParser(
        description="Extract RTMW-l wholebody-133 pose from standardized WLASL frames."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the RTMW-l extraction configuration.",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        help="Optional subset override. Defaults to the value from config.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional device override, e.g. cuda:0 or cpu.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional per-split sample limit for debugging.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and manifests without writing pose outputs.",
    )
    return parser


def _resolve_path(base_dir: Path, value: str | Path | None) -> Path | None:
    """Resolve an absolute or repo-relative path."""

    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else (base_dir / path)


def _stringify_path(path: Path | None) -> str:
    """Return a stable POSIX-like string for a path."""

    if path is None:
        return ""
    return path.as_posix()


def safe_str(value: Any, default: str = "") -> str:
    """Convert nullable pandas/scalar values to a safe string."""

    if value is None:
        return default
    try:
        is_na = pd.isna(value)
    except TypeError:
        is_na = False
    if isinstance(is_na, (bool, np.bool_)) and is_na:
        return default
    return str(value)


def _split_notes(value: Any) -> list[str]:
    """Split a semicolon-delimited note string into a list."""

    if value is None or pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    return [item for item in text.split(";") if item]


def _add_note(notes: list[str], note: str | None) -> None:
    """Append a note once while preserving insertion order."""

    if note and note not in notes:
        notes.append(note)


def _join_notes(notes: list[str]) -> str:
    """Join note tokens into a stable string."""

    merged: list[str] = []
    for note in notes:
        _add_note(merged, note)
    return ";".join(merged)


def _parse_optional_int(value: Any) -> int | None:
    """Convert nullable numeric-like values to ``int``."""

    if value is None or pd.isna(value):
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        value = text
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def load_config(
    config_path: Path,
    subset_override: str | None = None,
    device_override: str | None = None,
) -> dict[str, Any]:
    """Load, normalize, and validate the pose extraction config."""

    base_dir = Path.cwd()
    config = read_yaml(config_path)

    dataset_cfg = config.get("dataset", {})
    input_cfg = config.get("input", {})
    output_cfg = config.get("output", {})
    pose_cfg = config.get("pose", {})
    mmpose_cfg = config.get("mmpose", {})
    quality_cfg = config.get("quality", {})

    subset = subset_override or input_cfg.get("subset", "nslt100")
    splits = list(input_cfg.get("splits", list(ALLOWED_SPLITS)))
    invalid_splits = [split for split in splits if split not in ALLOWED_SPLITS]
    if invalid_splits:
        raise ValueError(f"Unsupported splits in config: {invalid_splits}")

    dataset_root = _resolve_path(
        base_dir, dataset_cfg.get("root", "data/datasets/WLASL")
    )
    output_root = _resolve_path(base_dir, output_cfg.get("root", dataset_root / "pose" / "rtmw_l"))
    keypoints_root = _resolve_path(
        base_dir,
        output_cfg.get("keypoints_root", output_root / "wholebody_133"),
    )

    resolved = {
        "config_path": config_path,
        "dataset": {
            "name": dataset_cfg.get("name", "WLASL"),
            "root": dataset_root,
        },
        "input": {
            "subset": subset,
            "splits": splits,
            "standardized_manifests_root": _resolve_path(
                base_dir,
                input_cfg.get(
                    "standardized_manifests_root",
                    dataset_root / "standardized" / "manifests",
                ),
            ),
            "manifest_filenames": {
                split: input_cfg.get("manifest_filenames", {}).get(
                    split, f"{subset}_{split}.csv"
                )
                for split in splits
            },
            "input_source": str(input_cfg.get("input_source", "frames")),
            "require_standardized_status_ok": bool(
                input_cfg.get("require_standardized_status_ok", True)
            ),
        },
        "output": {
            "root": output_root,
            "keypoints_root": keypoints_root,
            "manifests_root": _resolve_path(
                base_dir, output_cfg.get("manifests_root", output_root / "manifests")
            ),
            "reports_root": _resolve_path(
                base_dir, output_cfg.get("reports_root", output_root / "reports")
            ),
            "logs_root": _resolve_path(
                base_dir, output_cfg.get("logs_root", output_root / "logs")
            ),
        },
        "pose": {
            "backend": str(pose_cfg.get("backend", RTMW_L_BACKEND)),
            "keypoint_layout": str(
                pose_cfg.get("keypoint_layout", WHOLEBODY_133_LAYOUT)
            ),
            "num_keypoints": int(
                pose_cfg.get("num_keypoints", WHOLEBODY_133_NUM_KEYPOINTS)
            ),
            "input_width": int(pose_cfg.get("input_width", 288)),
            "input_height": int(pose_cfg.get("input_height", 384)),
            "device": device_override or str(pose_cfg.get("device", "cuda:0")),
            "fallback_device": str(pose_cfg.get("fallback_device", "cpu")),
            "batch_size": int(pose_cfg.get("batch_size", 1)),
            "save_visualizations": bool(pose_cfg.get("save_visualizations", False)),
            "overwrite": bool(pose_cfg.get("overwrite", True)),
        },
        "mmpose": {
            "config_file": _resolve_path(base_dir, mmpose_cfg.get("config_file")),
            "checkpoint_file": _resolve_path(
                base_dir, mmpose_cfg.get("checkpoint_file")
            ),
            "detector_config_file": _resolve_path(
                base_dir, mmpose_cfg.get("detector_config_file")
            ),
            "detector_checkpoint_file": _resolve_path(
                base_dir, mmpose_cfg.get("detector_checkpoint_file")
            ),
            "use_detector": bool(mmpose_cfg.get("use_detector", False)),
        },
        "quality": {
            "min_mean_confidence": float(
                quality_cfg.get("min_mean_confidence", 0.10)
            ),
            "missing_keypoint_confidence_threshold": float(
                quality_cfg.get("missing_keypoint_confidence_threshold", 0.01)
            ),
            "min_valid_frames_ratio": float(
                quality_cfg.get("min_valid_frames_ratio", 0.50)
            ),
            "important_regions": list(
                quality_cfg.get(
                    "important_regions",
                    ["body", "face", "left_hand", "right_hand"],
                )
            ),
        },
    }
    return resolved


def resolve_paths(config: dict[str, Any]) -> dict[str, Path]:
    """Resolve output and manifest paths for the requested subset."""

    subset = config["input"]["subset"]
    output_cfg = config["output"]
    return {
        "keypoints_subset_root": Path(output_cfg["keypoints_root"]) / subset,
        "manifests_root": Path(output_cfg["manifests_root"]),
        "reports_root": Path(output_cfg["reports_root"]),
        "logs_root": Path(output_cfg["logs_root"]),
        "standardized_manifests_root": Path(config["input"]["standardized_manifests_root"]),
    }


def _resolve_model_file(path: Path | None, pattern: str) -> Path:
    """Resolve a configured model file or discover one under its parent directory."""

    if path is not None and path.exists():
        return path

    search_dir = path.parent if path is not None else Path("checkpoints/pose/rtmw_l")
    candidates = sorted(search_dir.glob(pattern)) if search_dir.exists() else []
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        return candidates[0]

    raise FileNotFoundError(
        "RTMW-l config/checkpoint not found. Please place files under "
        "`checkpoints/pose/rtmw_l/` or update `configs/preprocessing/pose_rtmw_l.yaml`."
    )


def setup_pose_model(config: dict[str, Any]) -> tuple[Any | None, dict[str, Any]]:
    """Initialize MMPose RTMW-l inferencer with device fallback."""

    pose_cfg = config["pose"]
    mmpose_cfg = config["mmpose"]

    try:
        config_file = _resolve_model_file(mmpose_cfg["config_file"], "*rtmw*.py")
        checkpoint_file = _resolve_model_file(mmpose_cfg["checkpoint_file"], "*.pth")
    except Exception as exc:
        return None, {
            "device": None,
            "config_file": mmpose_cfg["config_file"],
            "checkpoint_file": mmpose_cfg["checkpoint_file"],
            "error": str(exc),
        }

    try:
        import torch
        from mmpose.apis import MMPoseInferencer
    except Exception as exc:
        return None, {
            "device": None,
            "config_file": config_file,
            "checkpoint_file": checkpoint_file,
            "error": f"Could not import MMPose/torch: {exc}",
        }

    requested_device = pose_cfg["device"]
    fallback_device = pose_cfg["fallback_device"]
    devices_to_try: list[str] = []
    for device in (requested_device, fallback_device):
        if device and device not in devices_to_try:
            devices_to_try.append(device)

    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        devices_to_try = [fallback_device] if fallback_device else ["cpu"]

    init_errors: list[str] = []
    for device in devices_to_try:
        attempts: list[dict[str, Any]] = []
        base_kwargs = {
            "pose2d": str(config_file),
            "pose2d_weights": str(checkpoint_file),
            "device": device,
        }
        if not mmpose_cfg["use_detector"]:
            attempts.append({**base_kwargs, "det_model": "whole_image"})
            attempts.append(base_kwargs)
        else:
            det_model = (
                str(mmpose_cfg["detector_config_file"])
                if mmpose_cfg["detector_config_file"] is not None
                else None
            )
            det_weights = (
                str(mmpose_cfg["detector_checkpoint_file"])
                if mmpose_cfg["detector_checkpoint_file"] is not None
                else None
            )
            attempts.append(
                {
                    **base_kwargs,
                    "det_model": det_model,
                    "det_weights": det_weights,
                }
            )

        for kwargs in attempts:
            try:
                inferencer = MMPoseInferencer(**kwargs)
                return inferencer, {
                    "device": device,
                    "config_file": config_file,
                    "checkpoint_file": checkpoint_file,
                    "error": "",
                }
            except Exception as exc:  # pragma: no cover - runtime env dependent
                init_errors.append(f"{device}: {exc}")

    return None, {
        "device": None,
        "config_file": config_file,
        "checkpoint_file": checkpoint_file,
        "error": " ; ".join(init_errors) if init_errors else "Unknown model init error.",
    }


def load_standardized_manifest(manifest_path: Path, split: str) -> pd.DataFrame:
    """Load and validate a standardized manifest for one split."""

    dtype_map = {
        "instance_uid": "string",
        "sample_id": "string",
        "video_id": "string",
        "gloss": "string",
        "split": "string",
        "raw_video_path": "string",
        "standardized_video_path": "string",
        "frames_dir": "string",
        "original_bbox": "string",
        "used_bbox": "string",
        "status": "string",
        "error_message": "string",
        "notes": "string",
    }
    frame = read_csv(manifest_path, dtype=dtype_map)
    require_columns(frame, STANDARDIZED_COLUMNS, name=f"standardized:{split}")
    frame = frame.copy()
    frame["split"] = frame["split"].fillna("").astype(str).str.strip().str.lower()
    frame = frame[frame["split"] == split].reset_index(drop=True)
    return frame


def collect_frame_paths(frames_dir: Path) -> list[Path]:
    """Collect standardized frame paths in stable lexicographic order."""

    paths: list[Path] = []
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        paths.extend(frames_dir.glob(pattern))
    unique_paths = sorted({path.resolve() for path in paths}, key=lambda item: item.name)
    return [Path(path) for path in unique_paths]


def run_pose_on_frame(inferencer: Any, frame_path: Path) -> list[dict[str, Any]]:
    """Run RTMW-l on a single frame and return predicted instances."""

    result = next(
        inferencer(
            str(frame_path),
            batch_size=1,
            show=False,
            return_vis=False,
        )
    )
    predictions = result.get("predictions", [])
    return predictions[0] if predictions else []


def _empty_keypoint_frame() -> np.ndarray:
    """Return an empty keypoint frame with NaN coordinates and zero confidence."""

    frame = np.zeros((WHOLEBODY_133_NUM_KEYPOINTS, 3), dtype=np.float32)
    frame[:, :2] = np.nan
    return frame


def _coerce_score(value: Any) -> float:
    """Convert detector/person scores to a comparable float."""

    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == 0:
            return 0.0
        value = value[0]
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _bbox_area(instance: dict[str, Any]) -> float:
    """Estimate instance bbox area."""

    bbox = instance.get("bbox")
    if isinstance(bbox, list) and bbox and isinstance(bbox[0], list):
        coords = bbox[0]
    elif isinstance(bbox, (list, tuple)):
        coords = bbox
    else:
        return 0.0
    if len(coords) < 4:
        return 0.0
    x1, y1, x2, y2 = (float(coords[i]) for i in range(4))
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def select_primary_person(instances: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, list[str]]:
    """Choose the primary signer by bbox score, then bbox area, then mean keypoint score."""

    if not instances:
        return None, []

    def rank(instance: dict[str, Any]) -> tuple[float, float, float]:
        bbox_score = _coerce_score(instance.get("bbox_score", 0.0))
        bbox_area = _bbox_area(instance)
        keypoint_scores = np.asarray(instance.get("keypoint_scores", []), dtype=np.float32)
        mean_score = float(keypoint_scores.mean()) if keypoint_scores.size else 0.0
        return bbox_score, bbox_area, mean_score

    if len(instances) == 1:
        return instances[0], []
    return max(instances, key=rank), ["multiple_people_selected_highest_score"]


def extract_keypoints_from_result(
    instance: dict[str, Any],
    expected_num_keypoints: int = WHOLEBODY_133_NUM_KEYPOINTS,
) -> tuple[np.ndarray, list[str]]:
    """Convert one MMPose instance dict into a ``(133, 3)`` float32 array."""

    notes: list[str] = []
    keypoints = np.asarray(instance.get("keypoints", []), dtype=np.float32)
    if keypoints.ndim != 2 or keypoints.shape[0] != expected_num_keypoints:
        raise ValueError(
            f"Expected keypoints with shape ({expected_num_keypoints}, 2/3), got {keypoints.shape}."
        )

    if keypoints.shape[1] >= 3:
        xy = keypoints[:, :2]
        scores = keypoints[:, 2]
    elif keypoints.shape[1] == 2:
        xy = keypoints
        scores = np.asarray(instance.get("keypoint_scores", []), dtype=np.float32)
        if scores.shape[0] != expected_num_keypoints:
            scores = np.ones(expected_num_keypoints, dtype=np.float32)
            notes.append("missing_keypoint_scores_filled_ones")
    else:
        raise ValueError(f"Invalid keypoint channel dimension: {keypoints.shape}.")

    if "keypoint_scores" in instance:
        candidate_scores = np.asarray(instance["keypoint_scores"], dtype=np.float32)
        if candidate_scores.shape[0] == expected_num_keypoints:
            scores = candidate_scores

    pose_frame = np.concatenate([xy, scores[:, None]], axis=1).astype(np.float32)
    if pose_frame.shape != (expected_num_keypoints, 3):
        raise ValueError(f"Invalid final pose shape: {pose_frame.shape}.")
    return pose_frame, notes


def write_pose_npz(
    pose_path: Path,
    keypoints: np.ndarray,
    row: pd.Series,
    image_height: int,
    image_width: int,
    keypoint_layout: str,
    pose_backend: str,
) -> None:
    """Write a sample pose tensor and metadata to a compressed ``.npz`` file."""

    ensure_dir(pose_path.parent)
    np.savez_compressed(
        pose_path,
        keypoints=keypoints.astype(np.float32),
        image_size=np.asarray([image_height, image_width], dtype=np.int32),
        sample_id=np.asarray(safe_str(row.get("sample_id"))),
        video_id=np.asarray(safe_str(row.get("video_id"))),
        gloss=np.asarray(safe_str(row.get("gloss"))),
        class_id=np.asarray(_parse_optional_int(row.get("class_id")) or -1, dtype=np.int32),
        split=np.asarray(safe_str(row.get("split"))),
        num_frames=np.asarray(int(keypoints.shape[0]), dtype=np.int32),
        keypoint_layout=np.asarray(keypoint_layout),
        pose_backend=np.asarray(pose_backend),
    )


def compute_pose_quality(keypoints: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    """Compute per-sample quality statistics from a pose tensor."""

    quality_cfg = config["quality"]
    valid_frames, valid_frames_ratio, missing_frames = compute_valid_frames(
        keypoints,
        confidence_threshold=quality_cfg["missing_keypoint_confidence_threshold"],
    )
    return {
        "mean_confidence": compute_mean_confidence(keypoints),
        "body_mean_confidence": compute_region_mean_confidence(keypoints, "body"),
        "face_mean_confidence": compute_region_mean_confidence(keypoints, "face"),
        "left_hand_mean_confidence": compute_region_mean_confidence(keypoints, "left_hand"),
        "right_hand_mean_confidence": compute_region_mean_confidence(keypoints, "right_hand"),
        "valid_frames": valid_frames,
        "valid_frames_ratio": valid_frames_ratio,
        "missing_frames": missing_frames,
    }


def _read_existing_pose(
    pose_path: Path,
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load an existing pose file and recompute its quality stats."""

    with np.load(pose_path, allow_pickle=False) as payload:
        keypoints = payload["keypoints"].astype(np.float32)
    validate_keypoints_shape(keypoints, expected_v=config["pose"]["num_keypoints"])
    return keypoints, compute_pose_quality(keypoints, config)


def process_sample(
    row: pd.Series,
    inferencer: Any | None,
    model_state: dict[str, Any],
    config: dict[str, Any],
    split_pose_root: Path,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Process one standardized sample into a shared pose ``.npz``."""

    pose_cfg = config["pose"]
    input_cfg = config["input"]
    sample_id = safe_str(row.get("sample_id")).strip()
    split = safe_str(row.get("split")).strip().lower()
    frames_dir = Path(safe_str(row.get("frames_dir")).strip())
    pose_path = split_pose_root / f"{sample_id}.npz"
    notes = _split_notes(safe_str(row.get("notes")))
    image_height = _parse_optional_int(row.get("output_height")) or pose_cfg["input_height"]
    image_width = _parse_optional_int(row.get("output_width")) or pose_cfg["input_width"]

    result = {
        "instance_uid": safe_str(row.get("instance_uid")),
        "sample_id": sample_id,
        "video_id": safe_str(row.get("video_id")),
        "gloss": safe_str(row.get("gloss")),
        "class_id": row.get("class_id"),
        "split": split,
        "frames_dir": _stringify_path(frames_dir),
        "pose_path": "",
        "keypoint_layout": pose_cfg["keypoint_layout"],
        "pose_backend": pose_cfg["backend"],
        "num_frames_input": _parse_optional_int(row.get("num_frames")) or 0,
        "num_frames_pose": 0,
        "image_height": image_height,
        "image_width": image_width,
        "mean_confidence": np.nan,
        "body_mean_confidence": np.nan,
        "face_mean_confidence": np.nan,
        "left_hand_mean_confidence": np.nan,
        "right_hand_mean_confidence": np.nan,
        "valid_frames": 0,
        "valid_frames_ratio": 0.0,
        "missing_frames": 0,
        "status": "failed",
        "error_message": "",
        "notes": "",
    }

    try:
        standardized_status = safe_str(row.get("status")).strip().lower()
        standardized_error = safe_str(row.get("error_message")).strip()
        if input_cfg["require_standardized_status_ok"] and standardized_status != "ok":
            result["status"] = "skipped_standardization_not_ok"
            result["error_message"] = standardized_error
            return result

        if inferencer is None:
            result["status"] = "model_load_error"
            result["error_message"] = model_state.get("error", "Pose model is unavailable.")
            return result

        if not frames_dir.exists():
            result["status"] = "missing_frames_dir"
            result["error_message"] = "Standardized frames directory does not exist."
            return result

        frame_paths = collect_frame_paths(frames_dir)
        if not frame_paths:
            result["status"] = "empty_frames_dir"
            result["error_message"] = "No standardized frames were found."
            return result

        result["num_frames_input"] = len(frame_paths)
        if not pose_cfg["overwrite"] and pose_path.exists():
            keypoints, quality = _read_existing_pose(pose_path, config)
            result["pose_path"] = _stringify_path(pose_path)
            result["num_frames_pose"] = int(keypoints.shape[0])
            result.update(quality)
            result["status"] = "ok"
            _add_note(notes, "reused_existing_pose")
            return result

        if dry_run:
            result["status"] = "ok"
            result["pose_path"] = _stringify_path(pose_path)
            result["num_frames_pose"] = len(frame_paths)
            return result

        if pose_path.exists():
            pose_path.unlink()

        keypoint_frames: list[np.ndarray] = []
        try:
            frame_inputs = [str(path) for path in frame_paths]
            inference_stream = inferencer(
                frame_inputs,
                batch_size=pose_cfg["batch_size"],
                show=False,
                return_vis=False,
            )
            for batch_result in inference_stream:
                batch_predictions = batch_result.get("predictions", [])
                for instance_predictions in batch_predictions:
                    primary_person, selection_notes = select_primary_person(instance_predictions)
                    for note in selection_notes:
                        _add_note(notes, note)
                    if primary_person is None:
                        _add_note(notes, "missing_person_detection")
                        keypoint_frames.append(_empty_keypoint_frame())
                        continue
                    pose_frame, extract_notes = extract_keypoints_from_result(primary_person)
                    for note in extract_notes:
                        _add_note(notes, note)
                    keypoint_frames.append(pose_frame)
        except ValueError as exc:
            if "Expected keypoints" in str(exc) or "Invalid keypoint" in str(exc):
                result["status"] = "invalid_keypoint_shape"
            else:
                result["status"] = "inference_error"
            result["error_message"] = str(exc)
            return result
        except Exception as exc:  # pragma: no cover - runtime env dependent
            result["status"] = "inference_error"
            result["error_message"] = str(exc)
            return result

        if len(keypoint_frames) != len(frame_paths):
            result["status"] = "inference_error"
            result["error_message"] = (
                f"Inference returned {len(keypoint_frames)} frames for {len(frame_paths)} inputs."
            )
            return result

        keypoints = np.stack(keypoint_frames, axis=0).astype(np.float32)
        validate_keypoints_shape(keypoints, expected_v=pose_cfg["num_keypoints"])
        quality = compute_pose_quality(keypoints, config)

        try:
            write_pose_npz(
                pose_path=pose_path,
                keypoints=keypoints,
                row=row,
                image_height=image_height,
                image_width=image_width,
                keypoint_layout=pose_cfg["keypoint_layout"],
                pose_backend=pose_cfg["backend"],
            )
        except Exception as exc:  # pragma: no cover - file-system failures
            result["status"] = "write_error"
            result["error_message"] = str(exc)
            return result

        result["pose_path"] = _stringify_path(pose_path)
        result["num_frames_pose"] = int(keypoints.shape[0])
        result.update(quality)
        result["status"] = "ok"
        return result
    finally:
        result["notes"] = _join_notes(notes)


def process_split(
    split: str,
    manifest: pd.DataFrame,
    inferencer: Any | None,
    model_state: dict[str, Any],
    config: dict[str, Any],
    paths: dict[str, Path],
    limit: int | None = None,
    dry_run: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Process all samples for one split."""

    split_pose_root = ensure_dir(paths["keypoints_subset_root"] / split)
    working_manifest = manifest.copy()
    working_manifest = working_manifest.sort_values(by=["sample_id", "video_id"]).reset_index(drop=True)
    if limit is not None:
        working_manifest = working_manifest.head(limit).reset_index(drop=True)

    LOGGER.info("Processing split=%s subset=%s samples=%s", split, config["input"]["subset"], len(working_manifest))

    rows: list[dict[str, Any]] = []
    for _, row in working_manifest.iterrows():
        rows.append(
            process_sample(
                row=row,
                inferencer=inferencer,
                model_state=model_state,
                config=config,
                split_pose_root=split_pose_root,
                dry_run=dry_run,
            )
        )

    output = pd.DataFrame(rows)
    if output.empty:
        output = pd.DataFrame(columns=POSE_MANIFEST_COLUMNS)
    output = validate_manifest_schema(output, POSE_MANIFEST_COLUMNS, name=f"pose:{split}")
    stats = {
        "split": split,
        "input_samples": int(len(working_manifest)),
        "ok_samples": int((output["status"] == "ok").sum()),
        "error_samples": int((output["status"] != "ok").sum()),
        "status_counts": output["status"].value_counts().to_dict(),
        "frames_pose": int(output["num_frames_pose"].fillna(0).astype(int).sum()),
    }
    return output, stats


def build_pose_quality_report(
    combined_manifest: pd.DataFrame,
    split_stats: list[dict[str, Any]],
    config: dict[str, Any],
    output_manifest_paths: list[Path],
    model_state: dict[str, Any],
) -> str:
    """Build the shared pose markdown report."""

    subset = config["input"]["subset"]
    pose_cfg = config["pose"]
    quality_cfg = config["quality"]
    summary = summarize_pose_manifest(
        combined_manifest,
        min_mean_confidence=quality_cfg["min_mean_confidence"],
        min_valid_frames_ratio=quality_cfg["min_valid_frames_ratio"],
    )

    lines = [
        f"# WLASL Pose Quality Report: {subset}",
        "",
        f"- Subset: `{subset}`",
        f"- Splits processed: `{', '.join(config['input']['splits'])}`",
        f"- Backend: `{pose_cfg['backend']}`",
        f"- Keypoint layout: `{pose_cfg['keypoint_layout']}`",
        f"- Model config: `{_stringify_path(model_state.get('config_file'))}`",
        f"- Model checkpoint: `{_stringify_path(model_state.get('checkpoint_file'))}`",
        f"- Device used: `{model_state.get('device')}`",
        f"- Total samples: `{summary['num_samples']}`",
        f"- Total status=ok: `{summary['num_ok']}`",
        f"- Total errors: `{summary['num_errors']}`",
        f"- Total frames with pose: `{summary['total_frames_pose']}`",
        f"- Mean confidence avg: `{summary['mean_confidence_avg']:.6f}`",
        f"- Valid frames ratio avg: `{summary['valid_frames_ratio_avg']:.6f}`",
        "",
        "## Region Confidence",
        "",
        f"- body: `{summary['body_mean_confidence_avg']:.6f}`",
        f"- face: `{summary['face_mean_confidence_avg']:.6f}`",
        f"- left_hand: `{summary['left_hand_mean_confidence_avg']:.6f}`",
        f"- right_hand: `{summary['right_hand_mean_confidence_avg']:.6f}`",
        "",
        "## Threshold Counts",
        "",
        f"- Samples below mean confidence threshold `{quality_cfg['min_mean_confidence']}`: `{summary['low_mean_confidence_samples']}`",
        f"- Samples below valid frames ratio threshold `{quality_cfg['min_valid_frames_ratio']}`: `{summary['low_valid_frames_ratio_samples']}`",
        "",
        "## Split Summary",
        "",
    ]

    for stats in split_stats:
        lines.extend(
            [
                f"### {stats['split']}",
                "",
                f"- Input samples: `{stats['input_samples']}`",
                f"- Success samples: `{stats['ok_samples']}`",
                f"- Error samples: `{stats['error_samples']}`",
                f"- Frames with pose: `{stats['frames_pose']}`",
                "",
            ]
        )

    lines.extend(["## Common Errors", ""])
    for status, count in sorted(summary["status_counts"].items()):
        lines.append(f"- {status}: `{count}`")

    lines.extend(
        [
            "",
            "## Output Manifests",
            "",
        ]
    )
    for manifest_path in output_manifest_paths:
        lines.append(f"- `{_stringify_path(manifest_path)}`")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `selected_31`, normalization, and graph tensor export are intentionally deferred to the skeleton branch.",
            "",
        ]
    )
    return "\n".join(lines)


def run(
    config_path: Path,
    subset: str | None = None,
    device: str | None = None,
    limit: int | None = None,
    dry_run: bool = False,
) -> int:
    """Run shared RTMW-l pose extraction for one subset."""

    config = load_config(
        config_path=config_path,
        subset_override=subset,
        device_override=device,
    )
    paths = resolve_paths(config)
    subset_name = config["input"]["subset"]

    ensure_dir(paths["keypoints_subset_root"])
    ensure_dir(paths["manifests_root"])
    ensure_dir(paths["reports_root"])
    ensure_dir(paths["logs_root"])

    global LOGGER
    LOGGER = setup_logger(
        __name__,
        paths["logs_root"] / f"extract_pose_{subset_name}.log",
    )

    LOGGER.info("Starting RTMW-l pose extraction.")
    LOGGER.info("Config path: %s", _stringify_path(config_path))
    LOGGER.info("Subset: %s", subset_name)
    LOGGER.info("Splits: %s", ", ".join(config["input"]["splits"]))
    LOGGER.info("Limit per split: %s", limit)
    LOGGER.info("Dry run: %s", dry_run)

    inferencer = None
    model_state: dict[str, Any] = {
        "device": None,
        "config_file": config["mmpose"]["config_file"],
        "checkpoint_file": config["mmpose"]["checkpoint_file"],
        "error": "",
    }
    if not dry_run:
        inferencer, model_state = setup_pose_model(config)
        LOGGER.info("Model config: %s", _stringify_path(model_state.get("config_file")))
        LOGGER.info("Model checkpoint: %s", _stringify_path(model_state.get("checkpoint_file")))
        LOGGER.info("Device used: %s", model_state.get("device"))
        if inferencer is None:
            LOGGER.error("Model setup failed: %s", model_state.get("error"))

    split_outputs: list[pd.DataFrame] = []
    split_stats: list[dict[str, Any]] = []
    output_manifest_paths: list[Path] = []

    for split in config["input"]["splits"]:
        manifest_path = paths["standardized_manifests_root"] / config["input"]["manifest_filenames"][split]
        LOGGER.info("Loading standardized manifest for split=%s: %s", split, _stringify_path(manifest_path))
        manifest = load_standardized_manifest(manifest_path, split)
        split_output, stats = process_split(
            split=split,
            manifest=manifest,
            inferencer=inferencer,
            model_state=model_state,
            config=config,
            paths=paths,
            limit=limit,
            dry_run=dry_run,
        )
        split_outputs.append(split_output)
        split_stats.append(stats)
        split_manifest_path = paths["manifests_root"] / f"{subset_name}_{split}.csv"
        output_manifest_paths.append(split_manifest_path)
        if not dry_run:
            write_dataframe_csv(split_output, split_manifest_path)
        LOGGER.info(
            "Finished split=%s ok=%s errors=%s frames_pose=%s",
            split,
            stats["ok_samples"],
            stats["error_samples"],
            stats["frames_pose"],
        )

    combined_manifest = (
        pd.concat(split_outputs, ignore_index=True)
        if split_outputs
        else pd.DataFrame(columns=POSE_MANIFEST_COLUMNS)
    )
    if not combined_manifest.empty:
        combined_manifest = combined_manifest.sort_values(
            by=["split", "sample_id", "video_id"]
        ).reset_index(drop=True)
    combined_manifest = validate_manifest_schema(
        combined_manifest,
        POSE_MANIFEST_COLUMNS,
        name="pose_all",
    )

    all_manifest_path = paths["manifests_root"] / f"{subset_name}_all.csv"
    report_path = paths["reports_root"] / f"{subset_name}_pose_quality_report.md"
    output_manifest_paths.append(all_manifest_path)
    report_text = build_pose_quality_report(
        combined_manifest=combined_manifest,
        split_stats=split_stats,
        config=config,
        output_manifest_paths=output_manifest_paths,
        model_state=model_state,
    )

    if not dry_run:
        write_dataframe_csv(combined_manifest, all_manifest_path)
        write_text(report_text, report_path)

    LOGGER.info("Combined manifest: %s", _stringify_path(all_manifest_path))
    LOGGER.info("Report path: %s", _stringify_path(report_path))
    LOGGER.info(
        "Finished RTMW-l pose extraction. total=%s ok=%s errors=%s",
        len(combined_manifest),
        int((combined_manifest["status"] == "ok").sum()),
        int((combined_manifest["status"] != "ok").sum()),
    )
    return 0


def main() -> int:
    """CLI entrypoint."""

    parser = parse_args()
    args = parser.parse_args()
    return run(
        config_path=args.config,
        subset=args.subset,
        device=args.device,
        limit=args.limit,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
