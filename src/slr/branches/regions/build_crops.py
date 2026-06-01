"""Build local-image region inputs from standardized frames and RTMW-l pose."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from slr.branches.regions.crop_utils import (
    RegionBBoxResult,
    black_fallback_crop,
    crop_and_resize,
    face_bbox_from_wholebody133,
    hand_bbox_from_wholebody133,
)
from slr.branches.regions.region_schema import (
    BBOX_SOURCE_BLACK_CROP_FAILED,
    BBOX_SOURCE_CURRENT_KEYPOINTS,
    BBOX_SOURCE_NAMES,
    BBOX_SOURCE_PREVIOUS_BBOX_FALLBACK,
    DEFAULT_CLIP_LEN,
    DEFAULT_CROP_SIZE,
    DEFAULT_IMAGE_DTYPE,
    NUM_CHANNELS,
    NUM_REGIONS,
    REGION_NAMES,
    REGION_TO_INDEX,
    TENSOR_FORMAT,
)
from slr.data.manifests import (
    POSE_MANIFEST_COLUMNS,
    REGION_INPUT_MANIFEST_COLUMNS,
    STANDARDIZED_COLUMNS,
)
from slr.data.validation import require_columns, validate_manifest_schema
from slr.pose.pose_schema import RTMW_L_BACKEND, WHOLEBODY_133_LAYOUT, validate_keypoints_shape
from slr.utils.bbox import BoundingBox
from slr.utils.image import read_image, save_image
from slr.utils.io import (
    ensure_dir,
    read_csv,
    read_yaml,
    remap_wlasl_path,
    stringify_path,
    write_dataframe_csv,
    write_json,
    write_text,
)
from slr.utils.logging import setup_logger


DEFAULT_CONFIG_PATH = Path("configs/preprocessing/region_crops.yaml")
ALLOWED_SPLITS = ("train", "val", "test")
DEFAULT_PREVIEW_FRAME_INDICES = (0, 15, 31, 47, 63)
DEFAULT_BLACK_CROP_RATIO_THRESHOLD = 0.3
DEFAULT_PREVIOUS_FALLBACK_RATIO_THRESHOLD = 0.4
LOGGER = setup_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for region input generation."""

    parser = argparse.ArgumentParser(
        description="Build local-image face/hand region tensors from standardized frames and RTMW-l pose."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the region preprocessing config.",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        help="Optional subset override. Defaults to the value from config.",
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
        help="Validate inputs without writing crops, tensors, manifests, or previews.",
    )
    return parser


def _resolve_path(base_dir: Path, value: str | Path) -> Path:
    """Resolve an absolute or repo-relative path."""

    path = Path(value)
    return path if path.is_absolute() else (base_dir / path)


def _safe_str(value: Any, default: str = "") -> str:
    """Convert nullable values to stable strings."""

    if value is None:
        return default
    try:
        is_na = pd.isna(value)
    except TypeError:
        is_na = False
    if isinstance(is_na, (bool, np.bool_)) and is_na:
        return default
    return str(value)


def _parse_optional_int(value: Any) -> int | None:
    """Convert one nullable numeric-like value to int."""

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


def _parse_optional_float(value: Any) -> float | None:
    """Convert one nullable numeric-like value to float."""

    if value is None or pd.isna(value):
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        value = text
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _json_text(values: list[Any] | tuple[Any, ...] | np.ndarray) -> str:
    """Serialize one list-like object as stable JSON text."""

    if isinstance(values, np.ndarray):
        values = values.tolist()
    return json.dumps(list(values), ensure_ascii=False)


def _split_notes(value: Any) -> list[str]:
    """Split a semicolon-delimited note string into a list."""

    text = _safe_str(value).strip()
    if not text:
        return []
    return [item for item in text.split(";") if item]


def _add_note(notes: list[str], note: str | None) -> None:
    """Append one note once while preserving insertion order."""

    if note and note not in notes:
        notes.append(note)


def _join_notes(notes: list[str]) -> str:
    """Join note tokens into a stable string."""

    merged: list[str] = []
    for note in notes:
        _add_note(merged, note)
    return ";".join(merged)


def _id_variants(*values: Any) -> tuple[str, ...]:
    """Generate stable ID variants for zero-padded and raw manifest values."""

    variants: list[str] = []
    for value in values:
        text = _safe_str(value).strip()
        if not text:
            continue
        candidates = [text]
        if ":" in text:
            candidates.append(text.split(":")[-1].strip())
        for candidate in list(candidates):
            if candidate.isdigit():
                candidates.append(str(int(candidate)))
        for candidate in candidates:
            normalized = candidate.strip()
            if normalized and normalized not in variants:
                variants.append(normalized)
    return tuple(variants)


def _manifest_filename_map(subset: str, manifest_cfg: dict[str, Any]) -> dict[str, str]:
    """Resolve per-split manifest filenames."""

    return {
        split: str(manifest_cfg.get(split, f"{subset}_{split}.csv"))
        for split in ALLOWED_SPLITS
    }


def _resolve_quality_threshold_map(
    value: Any,
    *,
    defaults: dict[str, float],
) -> dict[str, float]:
    """Resolve per-region quality thresholds from one scalar or mapping."""

    if isinstance(value, dict):
        return {
            region: float(value.get(region, defaults[region]))
            for region in REGION_NAMES
        }
    if value is None:
        return {region: float(defaults[region]) for region in REGION_NAMES}
    scalar = float(value)
    return {region: scalar for region in REGION_NAMES}


def load_config(config_path: Path, subset_override: str | None = None) -> dict[str, Any]:
    """Load and normalize the region preprocessing config."""

    base_dir = Path.cwd()
    config = read_yaml(config_path)

    dataset_cfg = config.get("dataset", {})
    input_cfg = config.get("input", {})
    output_cfg = config.get("output", {})
    outputs_cfg = config.get("outputs", {})
    crop_cfg = config.get("crop", {})
    preview_cfg = config.get("preview", {})
    options_cfg = config.get("options", {})
    quality_cfg = config.get("quality", {})
    hand_cfg = config.get("hand", {})
    face_cfg = config.get("face", {})
    regions_cfg = config.get("regions", {})

    subset = subset_override or dataset_cfg.get("subset") or "nslt100"
    dataset_root = _resolve_path(base_dir, dataset_cfg.get("root", "data/datasets/WLASL"))
    standardized_frames_root = _resolve_path(
        base_dir,
        input_cfg.get("standardized_frames_root", dataset_root / "standardized" / "frames"),
    )
    standardized_manifests_root = _resolve_path(
        base_dir,
        input_cfg.get("standardized_manifests_root", dataset_root / "standardized" / "manifests"),
    )
    pose_backend = str(input_cfg.get("pose_backend", RTMW_L_BACKEND))
    pose_layout = str(input_cfg.get("pose_layout", WHOLEBODY_133_LAYOUT))
    pose_root = _resolve_path(
        base_dir,
        input_cfg.get("pose_root", dataset_root / "pose" / pose_backend / subset / pose_layout),
    )
    pose_backend_root = _resolve_path(
        base_dir,
        input_cfg.get("pose_backend_root", dataset_root / "pose" / pose_backend),
    )
    pose_manifest_root = _resolve_path(
        base_dir,
        input_cfg.get("pose_manifest_root", dataset_root / "pose" / pose_backend / subset / "manifests"),
    )
    output_root = _resolve_path(
        base_dir,
        output_cfg.get("root", dataset_root / "branch_inputs" / "regions" / pose_backend),
    )

    splits = list(input_cfg.get("splits", list(ALLOWED_SPLITS)))
    invalid_splits = [split for split in splits if split not in ALLOWED_SPLITS]
    if invalid_splits:
        raise ValueError(f"Unsupported splits in config: {invalid_splits}")

    regions = tuple(
        regions_cfg.get(
            "order",
            output_cfg.get("regions", crop_cfg.get("regions", REGION_NAMES)),
        )
    )
    if tuple(regions) != REGION_NAMES:
        raise ValueError(f"Region order must be {REGION_NAMES}, got {regions}.")

    clip_length = int(
        config.get("clip_len", crop_cfg.get("clip_length", crop_cfg.get("clip_len", DEFAULT_CLIP_LEN)))
    )
    crop_size = int(config.get("crop_size", crop_cfg.get("crop_size", DEFAULT_CROP_SIZE)))
    preview_frame_indices = tuple(int(index) for index in preview_cfg.get("frame_indices", DEFAULT_PREVIEW_FRAME_INDICES))
    low_valid_ratio_thresholds = _resolve_quality_threshold_map(
        quality_cfg.get("low_valid_ratio_thresholds", quality_cfg.get("low_valid_ratio_threshold")),
        defaults={"left_hand": 0.5, "right_hand": 0.5, "face": 0.7},
    )
    high_black_crop_ratio_thresholds = _resolve_quality_threshold_map(
        quality_cfg.get("high_black_crop_ratio_thresholds", quality_cfg.get("high_black_crop_ratio_threshold")),
        defaults={"left_hand": 0.3, "right_hand": 0.3, "face": 0.2},
    )
    high_previous_fallback_ratio_thresholds = _resolve_quality_threshold_map(
        quality_cfg.get(
            "high_previous_fallback_ratio_thresholds",
            quality_cfg.get("high_previous_fallback_ratio_threshold"),
        ),
        defaults={
            "left_hand": DEFAULT_PREVIOUS_FALLBACK_RATIO_THRESHOLD,
            "right_hand": DEFAULT_PREVIOUS_FALLBACK_RATIO_THRESHOLD,
            "face": DEFAULT_PREVIOUS_FALLBACK_RATIO_THRESHOLD,
        },
    )

    resolved = {
        "config_path": config_path,
        "dataset": {
            "name": str(dataset_cfg.get("name", "WLASL")),
            "root": dataset_root,
            "subset": subset,
        },
        "input": {
            "splits": splits,
            "pose_backend": pose_backend,
            "pose_layout": pose_layout,
            "standardized_frames_root": standardized_frames_root,
            "standardized_manifests_root": standardized_manifests_root,
            "standardized_manifest_filenames": _manifest_filename_map(
                subset,
                input_cfg.get("standardized_manifest_filenames", {}),
            ),
            "pose_root": pose_root,
            "pose_backend_root": pose_backend_root,
            "pose_manifest_root": pose_manifest_root,
            "pose_manifest_filenames": _manifest_filename_map(
                subset,
                input_cfg.get("pose_manifest_filenames", {}),
            ),
            "require_standardized_status_ok": bool(input_cfg.get("require_standardized_status_ok", True)),
            "require_pose_status_ok": bool(input_cfg.get("require_pose_status_ok", True)),
        },
        "output": {
            "root": output_root,
            "regions": regions,
            "crops_root": _resolve_path(base_dir, output_cfg.get("crops_root", output_root / "crops")),
            "tensors_root": _resolve_path(base_dir, output_cfg.get("tensors_root", output_root / "tensors")),
            "manifests_root": _resolve_path(base_dir, output_cfg.get("manifests_root", output_root / "manifests")),
            "previews_root": _resolve_path(base_dir, output_cfg.get("previews_root", output_root / "previews")),
            "reports_root": _resolve_path(base_dir, output_cfg.get("reports_root", output_root / "reports")),
            "logs_root": _resolve_path(base_dir, output_cfg.get("logs_root", output_root / "logs")),
            "metadata_path": _resolve_path(base_dir, output_cfg.get("metadata_path", output_root / "metadata.json")),
        },
        "crop": {
            "clip_length": clip_length,
            "crop_size": crop_size,
            "hand": {
                "conf_threshold": float(hand_cfg.get("conf_threshold", crop_cfg.get("confidence_threshold", 0.2))),
                "min_points": int(hand_cfg.get("min_points", crop_cfg.get("min_hand_points", 5))),
                "margin": float(hand_cfg.get("margin", crop_cfg.get("hand_margin", 1.7))),
                "max_bbox_size_ratio": float(hand_cfg.get("max_bbox_size_ratio", 0.40)),
                "min_bbox_size_px": float(hand_cfg.get("min_bbox_size_px", 12)),
                "max_consecutive_fallback": int(hand_cfg.get("max_consecutive_fallback", 3)),
            },
            "face": {
                "conf_threshold": float(face_cfg.get("conf_threshold", crop_cfg.get("confidence_threshold", 0.2))),
                "min_face_points": int(face_cfg.get("min_face_points", crop_cfg.get("min_face_points", 8))),
                "min_anchor_points": int(face_cfg.get("min_anchor_points", crop_cfg.get("min_face_anchor_points", 2))),
                "margin": float(face_cfg.get("margin", crop_cfg.get("face_margin", 1.45))),
                "fallback_anchor_margin": float(
                    face_cfg.get("fallback_anchor_margin", crop_cfg.get("face_fallback_margin", 2.2))
                ),
                "max_bbox_size_ratio": float(face_cfg.get("max_bbox_size_ratio", 0.70)),
                "min_bbox_size_px": float(face_cfg.get("min_bbox_size_px", 24)),
                "max_consecutive_fallback": int(face_cfg.get("max_consecutive_fallback", 5)),
            },
        },
        "preview": {
            "frame_indices": preview_frame_indices,
        },
        "options": {
            "overwrite": bool(options_cfg.get("overwrite", True)),
            "save_crops": bool(outputs_cfg.get("save_crops", options_cfg.get("save_crops", True))),
            "save_tensors": bool(outputs_cfg.get("save_tensors", options_cfg.get("save_tensors", True))),
            "save_previews": bool(outputs_cfg.get("save_previews", options_cfg.get("save_previews", True))),
        },
        "quality": {
            "low_valid_ratio_thresholds": low_valid_ratio_thresholds,
            "high_black_crop_ratio_thresholds": high_black_crop_ratio_thresholds,
            "high_previous_fallback_ratio_thresholds": high_previous_fallback_ratio_thresholds,
        },
    }

    if resolved["input"]["pose_backend"] != RTMW_L_BACKEND:
        raise ValueError("This preprocessing step currently supports pose_backend=rtmw_l only.")
    if resolved["input"]["pose_layout"] != WHOLEBODY_133_LAYOUT:
        raise ValueError("This preprocessing step currently supports pose_layout=wholebody_133 only.")
    if clip_length <= 0 or crop_size <= 0:
        raise ValueError("clip_length and crop_size must be positive.")
    return resolved


def resolve_paths(config: dict[str, Any]) -> dict[str, Path]:
    """Resolve output and manifest paths."""

    output_cfg = config["output"]
    subset = config["dataset"]["subset"]
    return {
        "crops_subset_root": Path(output_cfg["crops_root"]) / subset,
        "tensors_subset_root": Path(output_cfg["tensors_root"]) / subset,
        "previews_subset_root": Path(output_cfg["previews_root"]) / subset,
        "manifests_root": Path(output_cfg["manifests_root"]),
        "reports_root": Path(output_cfg["reports_root"]),
        "logs_root": Path(output_cfg["logs_root"]),
        "metadata_path": Path(output_cfg["metadata_path"]),
        "low_quality_csv_path": Path(output_cfg["reports_root"]) / f"{subset}_region_low_quality_samples.csv",
    }


def load_standardized_manifest(manifest_path: Path, split: str) -> pd.DataFrame:
    """Load and validate one standardized manifest."""

    dtype_map = {
        "instance_uid": "string",
        "sample_id": "string",
        "video_id": "string",
        "gloss": "string",
        "split": "string",
        "raw_video_path": "string",
        "standardized_video_path": "string",
        "frames_dir": "string",
        "status": "string",
        "error_message": "string",
        "notes": "string",
    }
    frame = read_csv(manifest_path, dtype=dtype_map)
    require_columns(frame, STANDARDIZED_COLUMNS, name=f"standardized_manifest:{split}")
    working = frame.copy()
    working["split"] = working["split"].fillna("").astype(str).str.strip().str.lower()
    return working[working["split"] == split].reset_index(drop=True)


def load_pose_manifest(manifest_path: Path, split: str) -> pd.DataFrame:
    """Load and validate one pose manifest."""

    dtype_map = {
        "instance_uid": "string",
        "sample_id": "string",
        "video_id": "string",
        "gloss": "string",
        "split": "string",
        "frames_dir": "string",
        "pose_path": "string",
        "keypoint_layout": "string",
        "pose_backend": "string",
        "status": "string",
        "error_message": "string",
        "notes": "string",
    }
    frame = read_csv(manifest_path, dtype=dtype_map)
    require_columns(frame, POSE_MANIFEST_COLUMNS, name=f"pose_manifest:{split}")
    working = frame.copy()
    working["split"] = working["split"].fillna("").astype(str).str.strip().str.lower()
    return working[working["split"] == split].reset_index(drop=True)


def _working_manifest(manifest: pd.DataFrame, limit: int | None = None) -> pd.DataFrame:
    """Sort one manifest deterministically and apply an optional limit."""

    working = manifest.copy()
    working["sample_id"] = working["sample_id"].fillna("").astype(str)
    working["video_id"] = working["video_id"].fillna("").astype(str)
    working = working.sort_values(by=["sample_id", "video_id"]).reset_index(drop=True)
    if limit is not None:
        working = working.head(int(limit)).reset_index(drop=True)
    return working


def _resolve_manifest_path(candidates: list[Path]) -> Path:
    """Return the first existing manifest path or the first candidate."""

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def resolve_standardized_manifest_path(config: dict[str, Any], split: str) -> Path:
    """Resolve one standardized manifest path."""

    return (
        Path(config["input"]["standardized_manifests_root"])
        / config["input"]["standardized_manifest_filenames"][split]
    )


def resolve_pose_manifest_path(config: dict[str, Any], split: str) -> Path:
    """Resolve one pose manifest path across subset-specific and shared layouts."""

    filename = config["input"]["pose_manifest_filenames"][split]
    subset = config["dataset"]["subset"]
    backend_root = Path(config["input"]["pose_backend_root"])
    return _resolve_manifest_path(
        [
            Path(config["input"]["pose_manifest_root"]) / filename,
            backend_root / "manifests" / filename,
            backend_root / "manifests" / subset / filename,
        ]
    )


def resolve_frames_dir(row: pd.Series, config: dict[str, Any]) -> Path:
    """Resolve the standardized frame directory for one sample."""

    subset = config["dataset"]["subset"]
    split = _safe_str(row.get("split")).strip().lower()
    sample_id = _safe_str(row.get("sample_id")).strip()
    preferred = Path(config["input"]["standardized_frames_root"]) / subset / split / sample_id
    if preferred.exists():
        return preferred.resolve()
    manifest_path = remap_wlasl_path(
        _safe_str(row.get("frames_dir")).strip(),
        project_root=Path.cwd(),
        dataset_root=config["dataset"]["root"],
    )
    if manifest_path.exists():
        return manifest_path.resolve()
    return preferred.resolve(strict=False)


def resolve_pose_path(
    sample_id: str,
    split: str,
    config: dict[str, Any],
    pose_row: pd.Series | None = None,
) -> Path:
    """Resolve one pose file path across local and stored-manifest layouts."""

    subset = config["dataset"]["subset"]
    pose_root = Path(config["input"]["pose_root"])
    backend_root = Path(config["input"]["pose_backend_root"])

    id_variants = _id_variants(sample_id)
    candidates: list[Path] = []
    for sample_variant in id_variants:
        candidates.extend(
            [
                pose_root / subset / split / f"{sample_variant}.npz",
                pose_root / split / f"{sample_variant}.npz",
                backend_root / subset / WHOLEBODY_133_LAYOUT / subset / split / f"{sample_variant}.npz",
                backend_root / WHOLEBODY_133_LAYOUT / subset / split / f"{sample_variant}.npz",
            ]
        )
    if pose_row is not None:
        stored_text = _safe_str(pose_row.get("pose_path")).strip()
        if stored_text:
            candidates.insert(
                0,
                remap_wlasl_path(
                    stored_text,
                    project_root=Path.cwd(),
                    dataset_root=config["dataset"]["root"],
                ),
            )

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve(strict=False)


def collect_frame_paths(frames_dir: Path) -> list[Path]:
    """Collect frame files in stable lexicographic order."""

    return sorted(path for path in frames_dir.glob("*.jpg") if path.is_file())


def load_pose_keypoints(pose_path: Path) -> np.ndarray:
    """Load and validate one wholebody_133 pose tensor."""

    with np.load(pose_path, allow_pickle=False) as payload:
        keypoints = payload["keypoints"].astype(np.float32)
    validate_keypoints_shape(keypoints)
    return keypoints


def _region_metric_defaults() -> dict[str, float]:
    """Return default per-region numeric metrics."""

    return {region: 0.0 for region in REGION_NAMES}


def _region_int_defaults() -> dict[str, int]:
    """Return default per-region integer counters."""

    return {region: 0 for region in REGION_NAMES}


def _directory_size_bytes(path: Path) -> int:
    """Return the recursive size for one file or directory."""

    if not path.exists():
        return 0
    if path.is_file():
        return int(path.stat().st_size)
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += int(child.stat().st_size)
    return total


def _format_size(num_bytes: int) -> str:
    """Format a byte count into a compact human-readable string."""

    value = float(max(0, int(num_bytes)))
    units = ("B", "KB", "MB", "GB", "TB")
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def _manifest_ratio_columns_for_region(region_name: str) -> tuple[str, str, str]:
    """Return manifest column names for one region's bbox-source ratios."""

    return (
        f"{region_name}_current_bbox_ratio",
        f"{region_name}_previous_fallback_ratio",
        f"{region_name}_black_crop_ratio",
    )


def _box_to_array(box: BoundingBox | None) -> np.ndarray:
    """Serialize one bbox to a fixed float32 array."""

    if box is None:
        return np.asarray([-1.0, -1.0, -1.0, -1.0], dtype=np.float32)
    return np.asarray([box.x1, box.y1, box.x2, box.y2], dtype=np.float32)


def _write_npz(payload: dict[str, Any], path: Path, overwrite: bool) -> None:
    """Write a compressed npz file with overwrite handling."""

    ensure_dir(path.parent)
    if path.exists():
        if overwrite:
            path.unlink()
        else:
            raise FileExistsError(f"Output file already exists: {path}")
    np.savez_compressed(path, **payload)


def _render_preview_tile(
    image: np.ndarray,
    text: str,
    *,
    footer_height: int = 22,
    text_color: tuple[int, int, int] = (240, 240, 240),
    bg_color: tuple[int, int, int] = (24, 24, 24),
) -> np.ndarray:
    """Add one compact footer caption to a crop image."""

    tile = np.full((image.shape[0] + footer_height, image.shape[1], 3), bg_color, dtype=np.uint8)
    tile[: image.shape[0], :, :] = image
    cv2.putText(
        tile,
        text,
        (6, image.shape[0] + 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        text_color,
        1,
        cv2.LINE_AA,
    )
    return tile


def build_preview_image(
    region_data: np.ndarray,
    frame_indices: np.ndarray,
    preview_frame_indices: tuple[int, ...],
    sample_id: str,
    split: str,
) -> np.ndarray:
    """Build a 3-row contact sheet from the sampled region crops."""

    tiles_by_row: list[np.ndarray] = []
    safe_indices = [min(max(0, index), region_data.shape[1] - 1) for index in preview_frame_indices]

    for region_name in REGION_NAMES:
        row_tiles: list[np.ndarray] = []
        region_index = REGION_TO_INDEX[region_name]
        for clip_index in safe_indices:
            image = region_data[region_index, clip_index]
            source_index = int(frame_indices[clip_index])
            row_tiles.append(_render_preview_tile(image, text=f"clip={clip_index} src={source_index}"))
        row_strip = cv2.hconcat(row_tiles)
        label_width = 118
        label_canvas = np.full((row_strip.shape[0], label_width, 3), fill_value=18, dtype=np.uint8)
        cv2.putText(
            label_canvas,
            region_name,
            (10, row_strip.shape[0] // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
        tiles_by_row.append(cv2.hconcat([label_canvas, row_strip]))

    grid = cv2.vconcat(tiles_by_row)
    header = np.full((44, grid.shape[1], 3), fill_value=10, dtype=np.uint8)
    cv2.putText(
        header,
        f"sample={sample_id} split={split} regions={','.join(REGION_NAMES)}",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (250, 250, 250),
        1,
        cv2.LINE_AA,
    )
    return cv2.vconcat([header, grid])


def _build_region_result(
    region_name: str,
    frame_keypoints: np.ndarray,
    image_w: int,
    image_h: int,
    config: dict[str, Any],
) -> RegionBBoxResult:
    """Compute the current-frame bbox candidate for one region."""

    if region_name == "face":
        face_cfg = config["crop"]["face"]
        return face_bbox_from_wholebody133(
            frame_keypoints,
            image_w=image_w,
            image_h=image_h,
            conf_thr=face_cfg["conf_threshold"],
            primary_margin=face_cfg["margin"],
            fallback_margin=face_cfg["fallback_anchor_margin"],
            min_face_points=face_cfg["min_face_points"],
            min_anchor_points=face_cfg["min_anchor_points"],
            min_bbox_size_px=face_cfg["min_bbox_size_px"],
            max_bbox_size_ratio=face_cfg["max_bbox_size_ratio"],
        )
    hand_cfg = config["crop"]["hand"]
    return hand_bbox_from_wholebody133(
        frame_keypoints,
        region_name=region_name,
        image_w=image_w,
        image_h=image_h,
        conf_thr=hand_cfg["conf_threshold"],
        margin=hand_cfg["margin"],
        min_points=hand_cfg["min_points"],
        min_bbox_size_px=hand_cfg["min_bbox_size_px"],
        max_bbox_size_ratio=hand_cfg["max_bbox_size_ratio"],
    )


def process_sample(
    standardized_row: pd.Series,
    pose_row: pd.Series | None,
    config: dict[str, Any],
    paths: dict[str, Path],
    *,
    dry_run: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build region crops/tensor outputs for one sample."""

    subset = config["dataset"]["subset"]
    clip_length = int(config["crop"]["clip_length"])
    crop_size = int(config["crop"]["crop_size"])
    overwrite = bool(config["options"]["overwrite"])
    sample_id = _safe_str(standardized_row.get("sample_id")).strip()
    video_id = _safe_str(standardized_row.get("video_id")).strip()
    gloss = _safe_str(standardized_row.get("gloss")).strip()
    split = _safe_str(standardized_row.get("split")).strip().lower()
    class_id = _parse_optional_int(standardized_row.get("class_id"))
    instance_uid = _safe_str(standardized_row.get("instance_uid")).strip()

    sample_crop_root = paths["crops_subset_root"] / split / sample_id
    tensor_path = paths["tensors_subset_root"] / split / f"{sample_id}.npz"
    preview_path = paths["previews_subset_root"] / split / f"{sample_id}_preview.jpg"
    tensor_shape = (NUM_REGIONS, NUM_CHANNELS, clip_length, crop_size, crop_size)

    notes = _split_notes(standardized_row.get("notes"))
    if pose_row is not None:
        for note in _split_notes(pose_row.get("notes")):
            _add_note(notes, note)

    result = {
        "instance_uid": instance_uid,
        "sample_id": sample_id,
        "video_id": video_id,
        "gloss": gloss,
        "class_id": class_id if class_id is not None else -1,
        "split": split,
        "tensor_path": "",
        "crop_root": "",
        "preview_path": "",
        "num_frames_original": 0,
        "num_frames_used": clip_length,
        "tensor_shape": _json_text(tensor_shape),
        "left_hand_valid_ratio": 0.0,
        "right_hand_valid_ratio": 0.0,
        "face_valid_ratio": 0.0,
        "left_hand_current_bbox_ratio": 0.0,
        "left_hand_previous_fallback_ratio": 0.0,
        "left_hand_black_crop_ratio": 0.0,
        "right_hand_current_bbox_ratio": 0.0,
        "right_hand_previous_fallback_ratio": 0.0,
        "right_hand_black_crop_ratio": 0.0,
        "face_current_bbox_ratio": 0.0,
        "face_previous_fallback_ratio": 0.0,
        "face_black_crop_ratio": 0.0,
        "mean_left_hand_conf": 0.0,
        "mean_right_hand_conf": 0.0,
        "mean_face_conf": 0.0,
        "status": "error",
        "error_message": "",
        "notes": "",
    }
    stats = {
        "split": split,
        "status": "error",
        "region_valid_ratio": _region_metric_defaults(),
        "region_mean_conf": _region_metric_defaults(),
    }

    try:
        standardized_status = _safe_str(standardized_row.get("status")).strip().lower()
        if config["input"]["require_standardized_status_ok"] and standardized_status != "ok":
            result["error_message"] = (
                _safe_str(standardized_row.get("error_message")).strip()
                or "Standardized manifest status is not ok."
            )
            return result, stats

        if pose_row is not None:
            pose_status = _safe_str(pose_row.get("status")).strip().lower()
            if config["input"]["require_pose_status_ok"] and pose_status != "ok":
                result["error_message"] = (
                    _safe_str(pose_row.get("error_message")).strip()
                    or "Pose manifest status is not ok."
                )
                return result, stats
        else:
            _add_note(notes, "pose_manifest_row_missing")

        frames_dir = resolve_frames_dir(standardized_row, config)
        pose_path = resolve_pose_path(sample_id, split, config, pose_row)
        if not frames_dir.exists():
            raise FileNotFoundError(f"Standardized frames directory does not exist: {frames_dir}")
        if not pose_path.exists():
            raise FileNotFoundError(f"Pose file does not exist: {pose_path}")

        frame_paths = collect_frame_paths(frames_dir)
        if not frame_paths:
            raise FileNotFoundError(f"No standardized frames were found in {frames_dir}")
        keypoints = load_pose_keypoints(pose_path)

        usable_frames = min(len(frame_paths), int(keypoints.shape[0]))
        result["num_frames_original"] = usable_frames
        if usable_frames <= 0:
            raise ValueError("No aligned frame/pose pairs are available for this sample.")
        if len(frame_paths) != int(keypoints.shape[0]):
            _add_note(notes, f"frame_pose_length_mismatch={len(frame_paths)}vs{int(keypoints.shape[0])}")
        if usable_frames < clip_length:
            _add_note(notes, "sampled_with_repeated_indices")

        sampled_indices = np.linspace(0, usable_frames - 1, clip_length).round().astype(np.int32)
        region_data = np.zeros((NUM_REGIONS, clip_length, crop_size, crop_size, 3), dtype=np.uint8)
        valid_mask = np.zeros((NUM_REGIONS, clip_length), dtype=np.uint8)
        bbox_source = np.zeros((NUM_REGIONS, clip_length), dtype=np.uint8)
        bboxes = np.full((NUM_REGIONS, clip_length, 4), fill_value=-1.0, dtype=np.float32)
        confidence_sums = _region_metric_defaults()
        confidence_counts = {region: 0 for region in REGION_NAMES}
        previous_boxes: dict[str, BoundingBox | None] = {region: None for region in REGION_NAMES}
        consecutive_fallback_counts = _region_int_defaults()
        prev_fallback_counts = _region_int_defaults()
        black_counts = _region_int_defaults()
        frame_read_errors = 0
        frame_region_errors = _region_int_defaults()

        if not dry_run and config["options"]["save_crops"]:
            ensure_dir(sample_crop_root)

        for clip_index, source_index in enumerate(sampled_indices):
            frame_path = frame_paths[int(source_index)]
            try:
                image = read_image(frame_path)
            except Exception as exc:  # pragma: no cover - IO failures are runtime-dependent
                frame_read_errors += 1
                _add_note(notes, f"frame_read_error={type(exc).__name__}")
                image = black_fallback_crop(crop_size)
                image = cv2.resize(image, (crop_size, crop_size), interpolation=cv2.INTER_NEAREST)
                for region_name in REGION_NAMES:
                    region_index = REGION_TO_INDEX[region_name]
                    region_data[region_index, clip_index] = black_fallback_crop(crop_size)
                    bbox_source[region_index, clip_index] = BBOX_SOURCE_BLACK_CROP_FAILED
                    consecutive_fallback_counts[region_name] = 0
                    black_counts[region_name] += 1
                continue

            frame_keypoints = keypoints[int(source_index)]
            image_h, image_w = image.shape[:2]
            for region_name in REGION_NAMES:
                region_index = REGION_TO_INDEX[region_name]
                try:
                    max_consecutive_fallback = (
                        int(config["crop"]["face"]["max_consecutive_fallback"])
                        if region_name == "face"
                        else int(config["crop"]["hand"]["max_consecutive_fallback"])
                    )
                    bbox_result = _build_region_result(
                        region_name,
                        frame_keypoints=frame_keypoints,
                        image_w=image_w,
                        image_h=image_h,
                        config=config,
                    )
                    box = bbox_result.box
                    if (
                        box is None
                        and previous_boxes[region_name] is not None
                        and consecutive_fallback_counts[region_name] < max_consecutive_fallback
                    ):
                        box = previous_boxes[region_name]
                        prev_fallback_counts[region_name] += 1
                        consecutive_fallback_counts[region_name] += 1
                        bbox_source[region_index, clip_index] = BBOX_SOURCE_PREVIOUS_BBOX_FALLBACK
                    crop = crop_and_resize(image, box, crop_size)
                    if bbox_result.box is not None:
                        previous_boxes[region_name] = bbox_result.box
                        consecutive_fallback_counts[region_name] = 0
                        valid_mask[region_index, clip_index] = 1
                        bbox_source[region_index, clip_index] = BBOX_SOURCE_CURRENT_KEYPOINTS
                        confidence_sums[region_name] += float(bbox_result.mean_confidence)
                        confidence_counts[region_name] += 1
                    elif box is not None:
                        valid_mask[region_index, clip_index] = 1
                    else:
                        if (
                            previous_boxes[region_name] is not None
                            and consecutive_fallback_counts[region_name] >= max_consecutive_fallback
                        ):
                            _add_note(
                                notes,
                                f"{region_name}_fallback_limit_reached={max_consecutive_fallback}",
                            )
                        consecutive_fallback_counts[region_name] = 0
                        bbox_source[region_index, clip_index] = BBOX_SOURCE_BLACK_CROP_FAILED
                        black_counts[region_name] += 1
                    bboxes[region_index, clip_index] = _box_to_array(box)
                    region_data[region_index, clip_index] = crop
                except Exception as exc:  # pragma: no cover - runtime data issues are sample-specific
                    frame_region_errors[region_name] += 1
                    _add_note(notes, f"{region_name}_crop_error={type(exc).__name__}")
                    region_data[region_index, clip_index] = black_fallback_crop(crop_size)
                    bbox_source[region_index, clip_index] = BBOX_SOURCE_BLACK_CROP_FAILED
                    black_counts[region_name] += 1

                if not dry_run and config["options"]["save_crops"]:
                    crop_path = sample_crop_root / region_name / f"{clip_index:06d}.jpg"
                    save_image(crop_path, region_data[region_index, clip_index])

        tensor = np.transpose(region_data, (0, 4, 1, 2, 3)).astype(np.uint8, copy=False)
        preview_image = build_preview_image(
            region_data=region_data,
            frame_indices=sampled_indices,
            preview_frame_indices=config["preview"]["frame_indices"],
            sample_id=sample_id,
            split=split,
        )

        if not dry_run and config["options"]["save_tensors"]:
            tensor_payload = {
                "data": tensor,
                "valid_mask": valid_mask.astype(np.uint8),
                "bbox_source": bbox_source.astype(np.uint8),
                "bboxes": bboxes.astype(np.float32),
                "frame_indices": sampled_indices.astype(np.int32),
                "region_names": np.asarray(REGION_NAMES),
                "label": np.asarray(class_id if class_id is not None else -1, dtype=np.int32),
                "sample_id": np.asarray(sample_id),
                "video_id": np.asarray(video_id),
                "gloss": np.asarray(gloss),
            }
            _write_npz(tensor_payload, tensor_path, overwrite=overwrite)
            result["tensor_path"] = stringify_path(tensor_path)
        else:
            result["tensor_path"] = stringify_path(tensor_path)

        if not dry_run and config["options"]["save_previews"]:
            ensure_dir(preview_path.parent)
            save_image(preview_path, preview_image)
            result["preview_path"] = stringify_path(preview_path)
        elif config["options"]["save_previews"]:
            result["preview_path"] = stringify_path(preview_path)

        if config["options"]["save_crops"]:
            result["crop_root"] = stringify_path(sample_crop_root)

        for region_name in REGION_NAMES:
            region_index = REGION_TO_INDEX[region_name]
            valid_ratio = float(valid_mask[region_index].mean())
            current_ratio = float((bbox_source[region_index] == BBOX_SOURCE_CURRENT_KEYPOINTS).mean())
            previous_ratio = float((bbox_source[region_index] == BBOX_SOURCE_PREVIOUS_BBOX_FALLBACK).mean())
            black_ratio = float((bbox_source[region_index] == BBOX_SOURCE_BLACK_CROP_FAILED).mean())
            mean_conf = (
                float(confidence_sums[region_name] / confidence_counts[region_name])
                if confidence_counts[region_name] > 0
                else 0.0
            )
            result[f"{region_name}_valid_ratio"] = valid_ratio
            result[f"{region_name}_current_bbox_ratio"] = current_ratio
            result[f"{region_name}_previous_fallback_ratio"] = previous_ratio
            result[f"{region_name}_black_crop_ratio"] = black_ratio
            result[f"mean_{region_name}_conf"] = mean_conf
            stats["region_valid_ratio"][region_name] = valid_ratio
            stats["region_mean_conf"][region_name] = mean_conf

        if frame_read_errors:
            _add_note(notes, f"frame_read_errors={frame_read_errors}")
        for region_name in REGION_NAMES:
            if prev_fallback_counts[region_name]:
                _add_note(notes, f"{region_name}_prev_bbox_fallback_frames={prev_fallback_counts[region_name]}")
            if black_counts[region_name]:
                _add_note(notes, f"{region_name}_black_frames={black_counts[region_name]}")
            if frame_region_errors[region_name]:
                _add_note(notes, f"{region_name}_runtime_errors={frame_region_errors[region_name]}")

        result["status"] = "ok"
        stats["status"] = "ok"
        return result, stats
    except Exception as exc:  # pragma: no cover - runtime failures are sample-specific
        result["status"] = "error"
        result["error_message"] = str(exc)
        return result, stats
    finally:
        result["notes"] = _join_notes(notes)


def _build_pose_lookup(pose_manifest: pd.DataFrame) -> dict[str, pd.Series]:
    """Build a sample_id -> pose row lookup."""

    lookup: dict[str, pd.Series] = {}
    for _, row in _working_manifest(pose_manifest).iterrows():
        for variant in _id_variants(row.get("sample_id"), row.get("video_id"), row.get("instance_uid")):
            if variant not in lookup:
                lookup[variant] = row
    return lookup


def _resolve_pose_row(
    standardized_row: pd.Series,
    pose_lookup: dict[str, pd.Series],
) -> pd.Series | None:
    """Find the best pose-manifest row for one standardized sample."""

    for variant in _id_variants(
        standardized_row.get("sample_id"),
        standardized_row.get("video_id"),
        standardized_row.get("instance_uid"),
    ):
        if variant in pose_lookup:
            return pose_lookup[variant]
    return None


def process_split(
    split: str,
    standardized_manifest: pd.DataFrame,
    pose_manifest: pd.DataFrame,
    config: dict[str, Any],
    paths: dict[str, Path],
    *,
    limit: int | None = None,
    dry_run: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Process one split into region inputs."""

    working = _working_manifest(standardized_manifest, limit=limit)
    pose_lookup = _build_pose_lookup(pose_manifest)
    LOGGER.info(
        "Processing split=%s subset=%s samples=%s",
        split,
        config["dataset"]["subset"],
        len(working),
    )

    rows: list[dict[str, Any]] = []
    for _, row in working.iterrows():
        pose_row = _resolve_pose_row(row, pose_lookup)
        result_row, _ = process_sample(
            standardized_row=row,
            pose_row=pose_row,
            config=config,
            paths=paths,
            dry_run=dry_run,
        )
        rows.append(result_row)

    output = pd.DataFrame(rows)
    if output.empty:
        output = pd.DataFrame(columns=REGION_INPUT_MANIFEST_COLUMNS)
    output = validate_manifest_schema(output, REGION_INPUT_MANIFEST_COLUMNS, name=f"regions:{split}")

    ok_mask = output["status"] == "ok"
    stats = {
        "split": split,
        "input_samples": int(len(working)),
        "ok_samples": int(ok_mask.sum()),
        "error_samples": int((~ok_mask).sum()),
        "status_counts": output["status"].value_counts().to_dict(),
    }
    for region_name in REGION_NAMES:
        valid_values = pd.to_numeric(output[f"{region_name}_valid_ratio"], errors="coerce")
        current_values = pd.to_numeric(output[f"{region_name}_current_bbox_ratio"], errors="coerce")
        previous_values = pd.to_numeric(output[f"{region_name}_previous_fallback_ratio"], errors="coerce")
        black_values = pd.to_numeric(output[f"{region_name}_black_crop_ratio"], errors="coerce")
        conf_values = pd.to_numeric(output[f"mean_{region_name}_conf"], errors="coerce")
        stats[f"{region_name}_valid_ratio_avg"] = float(valid_values[ok_mask].mean()) if ok_mask.any() else 0.0
        stats[f"{region_name}_current_bbox_ratio_avg"] = (
            float(current_values[ok_mask].mean()) if ok_mask.any() else 0.0
        )
        stats[f"{region_name}_previous_fallback_ratio_avg"] = (
            float(previous_values[ok_mask].mean()) if ok_mask.any() else 0.0
        )
        stats[f"{region_name}_black_crop_ratio_avg"] = (
            float(black_values[ok_mask].mean()) if ok_mask.any() else 0.0
        )
        stats[f"{region_name}_mean_conf_avg"] = float(conf_values[ok_mask].mean()) if ok_mask.any() else 0.0
    LOGGER.info(
        "Finished split=%s ok=%s errors=%s",
        split,
        stats["ok_samples"],
        stats["error_samples"],
    )
    return output, stats


def build_metadata(config: dict[str, Any]) -> dict[str, Any]:
    """Build the metadata payload for the region branch output root."""

    crop_cfg = config["crop"]
    return {
        "dataset": config["dataset"]["name"],
        "subset": config["dataset"]["subset"],
        "pose_backend": config["input"]["pose_backend"],
        "pose_layout": config["input"]["pose_layout"],
        "regions": list(REGION_NAMES),
        "tensor_format": TENSOR_FORMAT,
        "num_regions": NUM_REGIONS,
        "num_channels": NUM_CHANNELS,
        "clip_len": int(crop_cfg["clip_length"]),
        "crop_size": int(crop_cfg["crop_size"]),
        "image_dtype": DEFAULT_IMAGE_DTYPE,
        "coordinate_source": "standardized_frames",
        "crop_source": "wholebody_133_keypoints",
        "bbox_source_codes": {str(key): value for key, value in BBOX_SOURCE_NAMES.items()},
        "overwrite": bool(config["options"]["overwrite"]),
        "hand_policy": dict(crop_cfg["hand"]),
        "face_policy": dict(crop_cfg["face"]),
    }


def build_low_quality_frame(
    combined_manifest: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Build the suspicious-sample dataframe for manual inspection."""

    conditions: list[pd.Series] = [combined_manifest["status"].fillna("").astype(str).str.lower() == "error"]
    for region_name in REGION_NAMES:
        low_valid_thr = float(config["quality"]["low_valid_ratio_thresholds"][region_name])
        high_black_thr = float(config["quality"]["high_black_crop_ratio_thresholds"][region_name])
        high_previous_thr = float(config["quality"]["high_previous_fallback_ratio_thresholds"][region_name])
        valid_values = pd.to_numeric(combined_manifest[f"{region_name}_valid_ratio"], errors="coerce").fillna(0.0)
        black_values = pd.to_numeric(combined_manifest[f"{region_name}_black_crop_ratio"], errors="coerce").fillna(0.0)
        previous_values = pd.to_numeric(
            combined_manifest[f"{region_name}_previous_fallback_ratio"],
            errors="coerce",
        ).fillna(0.0)
        conditions.append(valid_values < low_valid_thr)
        conditions.append(black_values > high_black_thr)
        conditions.append(previous_values > high_previous_thr)

    mask = conditions[0].copy()
    for condition in conditions[1:]:
        mask = mask | condition

    selected_columns = [
        "sample_id",
        "video_id",
        "split",
        "class_id",
        "gloss",
        "preview_path",
        "left_hand_valid_ratio",
        "right_hand_valid_ratio",
        "face_valid_ratio",
        "left_hand_black_crop_ratio",
        "right_hand_black_crop_ratio",
        "face_black_crop_ratio",
        "left_hand_previous_fallback_ratio",
        "right_hand_previous_fallback_ratio",
        "face_previous_fallback_ratio",
        "status",
        "error_message",
        "notes",
    ]
    output = combined_manifest.loc[mask, selected_columns].copy()
    if output.empty:
        return output

    output["_max_black_crop_ratio"] = output[
        ["left_hand_black_crop_ratio", "right_hand_black_crop_ratio", "face_black_crop_ratio"]
    ].apply(pd.to_numeric, errors="coerce").max(axis=1)
    output["_min_valid_ratio"] = output[
        ["left_hand_valid_ratio", "right_hand_valid_ratio", "face_valid_ratio"]
    ].apply(pd.to_numeric, errors="coerce").min(axis=1)
    output = output.sort_values(
        by=["status", "_max_black_crop_ratio", "_min_valid_ratio", "split", "sample_id"],
        ascending=[False, False, True, True, True],
        kind="stable",
    ).reset_index(drop=True)
    return output.drop(columns=["_max_black_crop_ratio", "_min_valid_ratio"])


def _aggregate_region_metrics(frame: pd.DataFrame, region_name: str) -> dict[str, float]:
    """Aggregate one region's quality metrics over a manifest frame."""

    if frame.empty:
        return {
            "valid_ratio_avg": 0.0,
            "current_bbox_ratio_avg": 0.0,
            "previous_fallback_ratio_avg": 0.0,
            "black_crop_ratio_avg": 0.0,
            "mean_conf_avg": 0.0,
        }
    return {
        "valid_ratio_avg": float(pd.to_numeric(frame[f"{region_name}_valid_ratio"], errors="coerce").mean()),
        "current_bbox_ratio_avg": float(
            pd.to_numeric(frame[f"{region_name}_current_bbox_ratio"], errors="coerce").mean()
        ),
        "previous_fallback_ratio_avg": float(
            pd.to_numeric(frame[f"{region_name}_previous_fallback_ratio"], errors="coerce").mean()
        ),
        "black_crop_ratio_avg": float(
            pd.to_numeric(frame[f"{region_name}_black_crop_ratio"], errors="coerce").mean()
        ),
        "mean_conf_avg": float(pd.to_numeric(frame[f"mean_{region_name}_conf"], errors="coerce").mean()),
    }


def build_report(
    combined_manifest: pd.DataFrame,
    split_stats: list[dict[str, Any]],
    config: dict[str, Any],
    manifest_paths: list[Path],
    metadata: dict[str, Any],
    paths: dict[str, Path],
    low_quality_frame: pd.DataFrame,
    commands_run: list[str],
) -> str:
    """Build the region crop quality report."""

    subset = config["dataset"]["subset"]
    low_valid_ratio_thresholds = config["quality"]["low_valid_ratio_thresholds"]
    high_black_ratio_thresholds = config["quality"]["high_black_crop_ratio_thresholds"]
    high_previous_fallback_ratio_thresholds = config["quality"]["high_previous_fallback_ratio_thresholds"]
    ok_frame = combined_manifest[combined_manifest["status"] == "ok"].copy()
    tensor_shape = (
        metadata["num_regions"],
        metadata["num_channels"],
        metadata["clip_len"],
        metadata["crop_size"],
        metadata["crop_size"],
    )
    split_frames = {
        split: combined_manifest[combined_manifest["split"] == split].copy()
        for split in config["input"]["splits"]
    }
    ok_split_frames = {
        split: frame[frame["status"] == "ok"].copy()
        for split, frame in split_frames.items()
    }
    all_metrics = {region: _aggregate_region_metrics(ok_frame, region) for region in REGION_NAMES}
    size_crops = _directory_size_bytes(paths["crops_subset_root"])
    size_tensors = _directory_size_bytes(paths["tensors_subset_root"])
    size_previews = _directory_size_bytes(paths["previews_subset_root"])
    size_manifests = _directory_size_bytes(paths["manifests_root"])
    size_reports = _directory_size_bytes(paths["reports_root"])
    total_output_size = size_crops + size_tensors + size_previews + size_manifests + size_reports

    lines = [
        f"# WLASL Region Crop Quality Report: {subset}",
        "",
        "## Overview",
        "",
        f"- subset: `{subset}`",
        f"- region order: `{_json_text(REGION_NAMES)}`",
        f"- tensor shape: `{tensor_shape}`",
        f"- clip length: `{metadata['clip_len']}`",
        f"- crop size: `{metadata['crop_size']}`",
        f"- output root: `{stringify_path(config['output']['root'])}`",
        f"- overwrite enabled: `{metadata['overwrite']}`",
        f"- total samples: `{len(combined_manifest)}`",
        f"- total ok samples: `{int((combined_manifest['status'] == 'ok').sum())}`",
        f"- total error samples: `{int((combined_manifest['status'] != 'ok').sum())}`",
        "",
        "## Split Summary",
        "",
    ]

    for stats in split_stats:
        lines.extend(
            [
                f"### {stats['split']}",
                "",
                f"- total samples: `{stats['input_samples']}`",
                f"- ok samples: `{stats['ok_samples']}`",
                f"- error samples: `{stats['error_samples']}`",
                "",
            ]
        )

    lines.extend(["## Valid Ratio", ""])
    for split, frame in ok_split_frames.items():
        lines.append(f"### {split}")
        lines.append("")
        metrics = {region: _aggregate_region_metrics(frame, region) for region in REGION_NAMES}
        for region_name in REGION_NAMES:
            lines.append(
                f"- {region_name} valid ratio avg: `{metrics[region_name]['valid_ratio_avg']:.6f}`"
            )
        lines.append("")

    lines.append("### all")
    lines.append("")
    for region_name in REGION_NAMES:
        lines.append(f"- {region_name} valid ratio avg: `{all_metrics[region_name]['valid_ratio_avg']:.6f}`")

    lines.extend(["", "## Bbox Source Ratio", ""])
    for split, frame in ok_split_frames.items():
        lines.append(f"### {split}")
        lines.append("")
        metrics = {region: _aggregate_region_metrics(frame, region) for region in REGION_NAMES}
        for region_name in REGION_NAMES:
            lines.extend(
                [
                    f"- {region_name} current keypoint ratio: `{metrics[region_name]['current_bbox_ratio_avg']:.6f}`",
                    f"- {region_name} previous bbox fallback ratio: `{metrics[region_name]['previous_fallback_ratio_avg']:.6f}`",
                    f"- {region_name} black crop ratio: `{metrics[region_name]['black_crop_ratio_avg']:.6f}`",
                ]
            )
        lines.append("")

    lines.append("### all")
    lines.append("")
    for region_name in REGION_NAMES:
        lines.extend(
            [
                f"- {region_name} current keypoint ratio: `{all_metrics[region_name]['current_bbox_ratio_avg']:.6f}`",
                f"- {region_name} previous bbox fallback ratio: `{all_metrics[region_name]['previous_fallback_ratio_avg']:.6f}`",
                f"- {region_name} black crop ratio: `{all_metrics[region_name]['black_crop_ratio_avg']:.6f}`",
            ]
        )

    lines.extend(["", "## Confidence", ""])
    for split, frame in ok_split_frames.items():
        lines.append(f"### {split}")
        lines.append("")
        metrics = {region: _aggregate_region_metrics(frame, region) for region in REGION_NAMES}
        for region_name in REGION_NAMES:
            lines.append(
                f"- {region_name} mean confidence: `{metrics[region_name]['mean_conf_avg']:.6f}`"
            )
        lines.append("")

    lines.append("### all")
    lines.append("")
    for region_name in REGION_NAMES:
        lines.append(f"- {region_name} mean confidence: `{all_metrics[region_name]['mean_conf_avg']:.6f}`")

    lines.extend(
        [
            "",
            "## Low-Quality Summary",
            "",
            f"- low valid ratio thresholds: `{json.dumps(low_valid_ratio_thresholds, ensure_ascii=False)}`",
            f"- high black crop ratio thresholds: `{json.dumps(high_black_ratio_thresholds, ensure_ascii=False)}`",
            f"- high previous fallback ratio thresholds: `{json.dumps(high_previous_fallback_ratio_thresholds, ensure_ascii=False)}`",
            f"- low-quality sample count: `{len(low_quality_frame)}`",
            f"- low-quality CSV: `{stringify_path(paths['low_quality_csv_path'])}`",
            "",
            "### Top 20 Suspicious Samples",
            "",
        ]
    )
    if low_quality_frame.empty:
        lines.append("- none")
    else:
        preview_columns = [
            "sample_id",
            "split",
            "left_hand_black_crop_ratio",
            "right_hand_black_crop_ratio",
            "face_black_crop_ratio",
            "left_hand_valid_ratio",
            "right_hand_valid_ratio",
            "face_valid_ratio",
            "status",
            "preview_path",
        ]
        ranked = low_quality_frame.copy()
        ranked["_max_black_crop_ratio"] = ranked[
            ["left_hand_black_crop_ratio", "right_hand_black_crop_ratio", "face_black_crop_ratio"]
        ].apply(pd.to_numeric, errors="coerce").max(axis=1)
        ranked["_min_valid_ratio"] = ranked[
            ["left_hand_valid_ratio", "right_hand_valid_ratio", "face_valid_ratio"]
        ].apply(pd.to_numeric, errors="coerce").min(axis=1)
        ranked = ranked.sort_values(
            by=["_max_black_crop_ratio", "_min_valid_ratio", "split", "sample_id"],
            ascending=[False, True, True, True],
            kind="stable",
        ).head(20)
        for _, row in ranked[preview_columns].iterrows():
            lines.append(
                "- "
                f"{row['sample_id']} ({row['split']}) "
                f"black=[{float(row['left_hand_black_crop_ratio']):.3f}, "
                f"{float(row['right_hand_black_crop_ratio']):.3f}, "
                f"{float(row['face_black_crop_ratio']):.3f}] "
                f"valid=[{float(row['left_hand_valid_ratio']):.3f}, "
                f"{float(row['right_hand_valid_ratio']):.3f}, "
                f"{float(row['face_valid_ratio']):.3f}] "
                f"status=`{row['status']}` preview=`{row['preview_path']}`"
            )

    lines.extend(
        [
            "",
            "## Output Size",
            "",
            f"- crops/nslt100: `{_format_size(size_crops)}`",
            f"- tensors/nslt100: `{_format_size(size_tensors)}`",
            f"- previews/nslt100: `{_format_size(size_previews)}`",
            f"- manifests: `{_format_size(size_manifests)}`",
            f"- reports: `{_format_size(size_reports)}`",
            f"- total regions output size for nslt100: `{_format_size(total_output_size)}`",
            "",
            "## Output Paths",
            "",
            f"- crops root: `{stringify_path(paths['crops_subset_root'])}`",
            f"- tensors root: `{stringify_path(paths['tensors_subset_root'])}`",
            f"- previews root: `{stringify_path(paths['previews_subset_root'])}`",
            f"- manifests root: `{stringify_path(paths['manifests_root'])}`",
            f"- metadata path: `{stringify_path(paths['metadata_path'])}`",
        ]
    )
    for manifest_path in manifest_paths:
        lines.append(f"- manifest: `{stringify_path(manifest_path)}`")

    lines.extend(["", "## Commands Run", ""])
    for command in commands_run:
        lines.append(f"- `{command}`")

    lines.extend(["", "## Status Counts", ""])
    status_counts = combined_manifest["status"].value_counts().to_dict()
    for status, count in sorted(status_counts.items()):
        lines.append(f"- {status}: `{count}`")

    lines.extend(["", "## Conclusion", ""])
    lines.append(
        f"- full build success: `{int((combined_manifest['status'] != 'ok').sum()) == 0}`"
    )
    lines.append(f"- total error samples: `{int((combined_manifest['status'] != 'ok').sum())}`")
    lines.append(
        "- crop readiness for training: "
        f"`{'needs_manual_review' if len(low_quality_frame) > 0 or int((combined_manifest['status'] != 'ok').sum()) > 0 else 'looks_ready'}`"
    )
    if low_quality_frame.empty and int((combined_manifest["status"] != "ok").sum()) == 0:
        lines.append("- next action: proceed to manual preview inspection, then move to training when satisfied.")
    else:
        lines.append("- next action: inspect low-quality previews before starting any training run.")

    lines.append("")
    return "\n".join(lines)


def run(
    config_path: Path,
    subset: str | None = None,
    limit: int | None = None,
    dry_run: bool = False,
) -> int:
    """Run region input generation for one subset."""

    config = load_config(config_path, subset_override=subset)
    paths = resolve_paths(config)
    subset_name = config["dataset"]["subset"]

    ensure_dir(paths["logs_root"])
    global LOGGER
    LOGGER = setup_logger(
        __name__,
        paths["logs_root"] / f"{subset_name}_build_regions.log",
    )

    LOGGER.info("Starting region input build.")
    LOGGER.info("Config path: %s", stringify_path(config["config_path"]))
    LOGGER.info("Subset: %s", subset_name)
    LOGGER.info("Splits: %s", ", ".join(config["input"]["splits"]))
    LOGGER.info("Limit per split: %s", limit)
    LOGGER.info("Dry run: %s", dry_run)
    LOGGER.info("Region order: %s", _json_text(REGION_NAMES))
    LOGGER.info(
        "Expected tensor shape: %s",
        (
            NUM_REGIONS,
            NUM_CHANNELS,
            config["crop"]["clip_length"],
            config["crop"]["crop_size"],
            config["crop"]["crop_size"],
        ),
    )
    commands_run = [" ".join([sys.executable, *sys.argv])]

    if not dry_run:
        ensure_dir(paths["crops_subset_root"])
        ensure_dir(paths["tensors_subset_root"])
        ensure_dir(paths["previews_subset_root"])
        ensure_dir(paths["manifests_root"])
        ensure_dir(paths["reports_root"])

    standardized_manifests: dict[str, pd.DataFrame] = {}
    pose_manifests: dict[str, pd.DataFrame] = {}
    for split in config["input"]["splits"]:
        standardized_manifest_path = resolve_standardized_manifest_path(config, split)
        pose_manifest_path = resolve_pose_manifest_path(config, split)
        LOGGER.info("Loading standardized manifest for split=%s: %s", split, stringify_path(standardized_manifest_path))
        LOGGER.info("Loading pose manifest for split=%s: %s", split, stringify_path(pose_manifest_path))
        standardized_manifests[split] = load_standardized_manifest(standardized_manifest_path, split)
        pose_manifests[split] = load_pose_manifest(pose_manifest_path, split)

    split_outputs: list[pd.DataFrame] = []
    split_stats: list[dict[str, Any]] = []
    manifest_paths: list[Path] = []

    for split in config["input"]["splits"]:
        split_output, stats = process_split(
            split=split,
            standardized_manifest=standardized_manifests[split],
            pose_manifest=pose_manifests[split],
            config=config,
            paths=paths,
            limit=limit,
            dry_run=dry_run,
        )
        split_outputs.append(split_output)
        split_stats.append(stats)

        split_manifest_path = paths["manifests_root"] / f"{subset_name}_{split}.csv"
        manifest_paths.append(split_manifest_path)
        if not dry_run:
            write_dataframe_csv(split_output, split_manifest_path)

    combined_manifest = (
        pd.concat(split_outputs, ignore_index=True)
        if split_outputs
        else pd.DataFrame(columns=REGION_INPUT_MANIFEST_COLUMNS)
    )
    if not combined_manifest.empty:
        combined_manifest = combined_manifest.sort_values(by=["split", "sample_id", "video_id"]).reset_index(drop=True)
    combined_manifest = validate_manifest_schema(
        combined_manifest,
        REGION_INPUT_MANIFEST_COLUMNS,
        name="regions_all",
    )

    metadata = build_metadata(config)
    low_quality_frame = build_low_quality_frame(combined_manifest, config)
    report_path = paths["reports_root"] / f"{subset_name}_region_crop_quality_report.md"
    report_text = build_report(
        combined_manifest=combined_manifest,
        split_stats=split_stats,
        config=config,
        manifest_paths=manifest_paths,
        metadata=metadata,
        paths=paths,
        low_quality_frame=low_quality_frame,
        commands_run=commands_run,
    )

    if not dry_run:
        write_json(metadata, paths["metadata_path"])
        write_dataframe_csv(low_quality_frame, paths["low_quality_csv_path"])
        write_text(report_text, report_path)

    LOGGER.info("Report path: %s", stringify_path(report_path))
    LOGGER.info("Low-quality CSV path: %s", stringify_path(paths["low_quality_csv_path"]))
    LOGGER.info("Metadata path: %s", stringify_path(paths["metadata_path"]))
    LOGGER.info(
        "Finished region input build. total=%s ok=%s errors=%s",
        len(combined_manifest),
        int((combined_manifest["status"] == "ok").sum()),
        int((combined_manifest["status"] != "ok").sum()),
    )
    return 0


def main() -> int:
    """CLI entrypoint."""

    parser = build_parser()
    args = parser.parse_args()
    return run(
        config_path=args.config,
        subset=args.subset,
        limit=args.limit,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
