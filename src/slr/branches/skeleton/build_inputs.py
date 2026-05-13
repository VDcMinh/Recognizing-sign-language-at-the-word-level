"""Build skeleton branch inputs from shared RTMW-l pose outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from slr.branches.skeleton.transforms import fix_sequence_length, to_graph_tensor_ctvm
from slr.data.manifests import POSE_MANIFEST_COLUMNS, SKELETON_INPUT_MANIFEST_COLUMNS
from slr.data.validation import require_columns, validate_manifest_schema
from slr.pose.keypoint_selection import build_selected_keypoints_npz_payload, select_keypoints
from slr.pose.pose_normalization import (
    compute_confidence_scale,
    normalize_confidence,
    normalize_xy_to_minus1_1,
    sanitize_non_finite_keypoints,
)
from slr.pose.pose_schema import (
    WHOLEBODY_133_LAYOUT,
    get_keypoint_component_indices,
    get_keypoint_indices,
    get_keypoint_names,
    get_keypoint_set_names,
    get_keypoint_set_note,
    validate_keypoints_shape,
)
from slr.utils.io import ensure_dir, read_csv, read_yaml, write_dataframe_csv, write_text
from slr.utils.logging import setup_logger


DEFAULT_CONFIG_PATH = Path("configs/branches/skeleton/stgcnpp_27.yaml")
ALLOWED_SPLITS = ("train", "val", "test")
LOGGER = setup_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for skeleton branch input building."""

    parser = argparse.ArgumentParser(
        description="Build skeleton branch inputs from shared RTMW-l pose files."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the skeleton branch preprocessing config.",
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
        help="Validate config and pose manifests without writing outputs.",
    )
    return parser


def _resolve_path(base_dir: Path, value: str | Path) -> Path:
    """Resolve an absolute or repo-relative path."""

    path = Path(value)
    return path if path.is_absolute() else (base_dir / path)


def _stringify_path(path: Path | None) -> str:
    """Return a stable POSIX-like string for a path."""

    if path is None:
        return ""
    return path.as_posix()


def _safe_str(value: Any, default: str = "") -> str:
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


def _json_array_text(values: list[int] | tuple[int, ...] | list[str] | tuple[str, ...]) -> str:
    """Serialize a list-like value into stable JSON text."""

    return json.dumps(list(values), ensure_ascii=False)


def load_config(config_path: Path, subset_override: str | None = None) -> dict[str, Any]:
    """Load, normalize, and validate the skeleton preprocessing config."""

    base_dir = Path.cwd()
    config = read_yaml(config_path)

    dataset_cfg = config.get("dataset", {})
    input_cfg = config.get("input", {})
    output_cfg = config.get("output", {})
    keypoints_cfg = config.get("keypoints", {})
    normalization_cfg = config.get("normalization", {})
    sequence_cfg = config.get("sequence", {})
    graph_tensor_cfg = config.get("graph_tensor", {})
    options_cfg = config.get("options", {})

    subset = subset_override or dataset_cfg.get("subset") or input_cfg.get("subset") or "nslt100"
    splits = list(input_cfg.get("splits", list(ALLOWED_SPLITS)))
    invalid_splits = [split for split in splits if split not in ALLOWED_SPLITS]
    if invalid_splits:
        raise ValueError(f"Unsupported splits in config: {invalid_splits}")

    dataset_root = _resolve_path(base_dir, dataset_cfg.get("root", "data/datasets/WLASL"))
    pose_root = _resolve_path(base_dir, input_cfg.get("pose_root", dataset_root / "pose" / "rtmw_l"))
    pose_manifest_root = _resolve_path(
        base_dir,
        input_cfg.get("pose_manifest_root", pose_root / "manifests"),
    )

    keypoint_set = str(keypoints_cfg.get("keypoint_set", "selected_27"))
    selected_indices = get_keypoint_indices(keypoint_set)
    selected_names = get_keypoint_set_names(keypoint_set)
    mouth_indices = get_keypoint_component_indices(keypoint_set, "mouth")
    mouth_names = get_keypoint_names(mouth_indices) if mouth_indices else ()
    num_selected_keypoints = int(keypoints_cfg.get("num_selected_keypoints", len(selected_indices)))
    if num_selected_keypoints != len(selected_indices):
        raise ValueError(
            f"Config num_selected_keypoints={num_selected_keypoints} does not match {len(selected_indices)} indices."
        )

    output_root = _resolve_path(base_dir, output_cfg.get("root", dataset_root / "branch_inputs" / "skeleton" / "rtmw_l"))
    selected_root = _resolve_path(base_dir, output_cfg.get("selected_root", output_root / keypoint_set))
    normalized_root = _resolve_path(
        base_dir,
        output_cfg.get("normalized_root", output_root / "normalized" / keypoint_set),
    )
    graph_tensor_root = _resolve_path(
        base_dir,
        output_cfg.get("graph_tensor_root", output_root / "graph_tensors" / keypoint_set),
    )

    channels = list(graph_tensor_cfg.get("channels", ["x", "y", "confidence"]))
    num_channels = int(graph_tensor_cfg.get("num_channels", len(channels)))
    if num_channels != len(channels):
        raise ValueError("graph_tensor.num_channels must match the number of configured channels.")
    expected_shape = list(graph_tensor_cfg.get("expected_shape", [num_channels, 150, len(selected_indices), 1]))
    if len(expected_shape) != 4:
        raise ValueError("graph_tensor.expected_shape must have 4 values: [C, T, V, M].")

    resolved = {
        "config_path": config_path,
        "dataset": {
            "name": dataset_cfg.get("name", "WLASL"),
            "root": dataset_root,
            "subset": subset,
        },
        "input": {
            "pose_root": pose_root,
            "pose_manifest_root": pose_manifest_root,
            "splits": splits,
            "manifest_filenames": {
                split: input_cfg.get("manifest_filenames", {}).get(split, f"{subset}_{split}.csv")
                for split in splits
            },
            "require_pose_status_ok": bool(input_cfg.get("require_pose_status_ok", True)),
            "source_pose_layout_root": pose_root / keypoints_cfg.get("source_layout", WHOLEBODY_133_LAYOUT),
        },
        "output": {
            "root": output_root,
            "selected_root": selected_root,
            "normalized_root": normalized_root,
            "graph_tensor_root": graph_tensor_root,
            "manifests_root": _resolve_path(base_dir, output_cfg.get("manifests_root", output_root / "manifests")),
            "reports_root": _resolve_path(base_dir, output_cfg.get("reports_root", output_root / "reports")),
            "logs_root": _resolve_path(base_dir, output_cfg.get("logs_root", output_root / "logs")),
        },
        "keypoints": {
            "source_layout": str(keypoints_cfg.get("source_layout", WHOLEBODY_133_LAYOUT)),
            "keypoint_set": keypoint_set,
            "num_source_keypoints": int(keypoints_cfg.get("num_source_keypoints", 133)),
            "num_selected_keypoints": num_selected_keypoints,
            "selected_indices": selected_indices,
            "selected_names": selected_names,
            "mouth_indices": mouth_indices,
            "mouth_names": mouth_names,
            "mapping_note": get_keypoint_set_note(keypoint_set),
        },
        "normalization": {
            "image_width": int(normalization_cfg.get("image_width", 288)),
            "image_height": int(normalization_cfg.get("image_height", 384)),
            "clip_xy_to_image": bool(normalization_cfg.get("clip_xy_to_image", True)),
            "xy_range": list(normalization_cfg.get("xy_range", [-1, 1])),
            "confidence": {
                "enabled": bool(normalization_cfg.get("confidence", {}).get("enabled", True)),
                "method": str(normalization_cfg.get("confidence", {}).get("method", "percentile")),
                "percentile": float(normalization_cfg.get("confidence", {}).get("percentile", 95)),
                "fit_on_split": str(normalization_cfg.get("confidence", {}).get("fit_on_split", "train")),
                "clip_range": list(normalization_cfg.get("confidence", {}).get("clip_range", [0, 1])),
                "save_scale_to_report": bool(
                    normalization_cfg.get("confidence", {}).get("save_scale_to_report", True)
                ),
            },
        },
        "sequence": {
            "target_num_frames": int(sequence_cfg.get("target_num_frames", 150)),
            "short_sequence_strategy": str(sequence_cfg.get("short_sequence_strategy", "repeat")),
            "long_sequence_strategy": str(sequence_cfg.get("long_sequence_strategy", "head")),
            "preserve_original_num_frames": bool(sequence_cfg.get("preserve_original_num_frames", True)),
        },
        "graph_tensor": {
            "channels": channels,
            "num_channels": num_channels,
            "num_persons": int(graph_tensor_cfg.get("num_persons", 1)),
            "layout": str(graph_tensor_cfg.get("layout", "CTVM")),
            "expected_shape": expected_shape,
        },
        "options": {
            "overwrite": bool(options_cfg.get("overwrite", True)),
            "save_selected": bool(options_cfg.get("save_selected", True)),
            "save_normalized": bool(options_cfg.get("save_normalized", True)),
            "save_graph_tensor": bool(options_cfg.get("save_graph_tensor", True)),
            "fail_on_missing_pose_file": bool(options_cfg.get("fail_on_missing_pose_file", False)),
        },
    }

    if resolved["keypoints"]["source_layout"] != WHOLEBODY_133_LAYOUT:
        raise ValueError("This preprocessing step currently supports source_layout=wholebody_133 only.")
    if resolved["keypoints"]["keypoint_set"] not in {"selected_27", "selected_31"}:
        raise ValueError("This preprocessing step currently supports keypoint_set=selected_27 or selected_31 only.")
    if resolved["graph_tensor"]["layout"] != "CTVM":
        raise ValueError("This preprocessing step requires graph_tensor.layout=CTVM.")
    expected_shape_tuple = tuple(int(value) for value in resolved["graph_tensor"]["expected_shape"])
    target_shape = (
        resolved["graph_tensor"]["num_channels"],
        resolved["sequence"]["target_num_frames"],
        resolved["keypoints"]["num_selected_keypoints"],
        resolved["graph_tensor"]["num_persons"],
    )
    if expected_shape_tuple != target_shape:
        raise ValueError(
            f"graph_tensor.expected_shape={expected_shape_tuple} does not match the resolved target shape {target_shape}."
        )
    return resolved


def resolve_paths(config: dict[str, Any]) -> dict[str, Path]:
    """Resolve subset-specific output and manifest paths."""

    subset = config["dataset"]["subset"]
    output_cfg = config["output"]
    return {
        "selected_subset_root": Path(output_cfg["selected_root"]) / subset,
        "normalized_subset_root": Path(output_cfg["normalized_root"]) / subset,
        "graph_tensor_subset_root": Path(output_cfg["graph_tensor_root"]) / subset,
        "manifests_root": Path(output_cfg["manifests_root"]),
        "reports_root": Path(output_cfg["reports_root"]),
        "logs_root": Path(output_cfg["logs_root"]),
        "pose_manifest_root": Path(config["input"]["pose_manifest_root"]),
    }


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
    frame = frame.copy()
    frame["split"] = frame["split"].fillna("").astype(str).str.strip().str.lower()
    frame = frame[frame["split"] == split].reset_index(drop=True)
    return frame


def _working_manifest(manifest: pd.DataFrame, limit: int | None = None) -> pd.DataFrame:
    """Sort one manifest deterministically and apply an optional limit."""

    working = manifest.copy()
    working = working.sort_values(by=["sample_id", "video_id"]).reset_index(drop=True)
    if limit is not None:
        working = working.head(limit).reset_index(drop=True)
    return working


def resolve_pose_file_path(row: pd.Series, config: dict[str, Any]) -> Path:
    """Resolve the local pose file path for one manifest row."""

    subset = config["dataset"]["subset"]
    split = _safe_str(row.get("split")).strip().lower()
    sample_id = _safe_str(row.get("sample_id")).strip()
    derived = (
        Path(config["input"]["source_pose_layout_root"])
        / subset
        / split
        / f"{sample_id}.npz"
    )
    if derived.exists():
        return derived

    pose_path_text = _safe_str(row.get("pose_path")).strip()
    if pose_path_text:
        manifest_pose_path = Path(pose_path_text)
        if manifest_pose_path.exists():
            return manifest_pose_path

        normalized_text = pose_path_text.replace("\\", "/")
        marker = "/data/datasets/WLASL/pose/rtmw_l/"
        if marker in normalized_text:
            suffix = normalized_text.split(marker, 1)[1]
            remapped = Path.cwd() / "data" / "datasets" / "WLASL" / "pose" / "rtmw_l" / Path(suffix)
            if remapped.exists():
                return remapped

    return derived if pose_path_text else derived


def load_pose_keypoints(pose_path: Path, expected_num_keypoints: int) -> np.ndarray:
    """Load one pose ``.npz`` file and validate its keypoint tensor shape."""

    with np.load(pose_path, allow_pickle=False) as payload:
        keypoints = payload["keypoints"].astype(np.float32)
    validate_keypoints_shape(keypoints, expected_v=expected_num_keypoints)
    return keypoints


def _build_normalization_metadata(config: dict[str, Any]) -> dict[str, Any]:
    """Build a compact normalization metadata payload."""

    return {
        "xy": {
            "clip_to_image": bool(config["normalization"]["clip_xy_to_image"]),
            "image_width": int(config["normalization"]["image_width"]),
            "image_height": int(config["normalization"]["image_height"]),
            "range": list(config["normalization"]["xy_range"]),
        },
        "confidence": {
            "enabled": bool(config["normalization"]["confidence"]["enabled"]),
            "method": str(config["normalization"]["confidence"]["method"]),
            "percentile": float(config["normalization"]["confidence"]["percentile"]),
            "clip_range": list(config["normalization"]["confidence"]["clip_range"]),
        },
    }


def _write_npz(payload: dict[str, Any], path: Path, overwrite: bool) -> None:
    """Write a compressed ``.npz`` file with overwrite handling."""

    ensure_dir(path.parent)
    if path.exists():
        if overwrite:
            path.unlink()
        else:
            raise FileExistsError(f"Output file already exists: {path}")
    np.savez_compressed(path, **payload)


def compute_confidence_scale_from_train(
    train_manifest: pd.DataFrame,
    config: dict[str, Any],
    limit: int | None = None,
) -> dict[str, Any]:
    """Fit one confidence scale from selected training-pose confidence values."""

    LOGGER.info("Starting confidence scale computation from train split.")
    working = _working_manifest(train_manifest, limit=limit)
    indices = config["keypoints"]["selected_indices"]
    require_ok = config["input"]["require_pose_status_ok"]
    confidence_arrays: list[np.ndarray] = []
    skipped_status = 0
    missing_pose_files = 0
    invalid_pose_shapes = 0

    for _, row in working.iterrows():
        pose_status = _safe_str(row.get("status")).strip().lower()
        if require_ok and pose_status != "ok":
            skipped_status += 1
            continue

        pose_path = resolve_pose_file_path(row, config)
        if not pose_path.exists():
            missing_pose_files += 1
            continue

        try:
            keypoints = load_pose_keypoints(
                pose_path,
                expected_num_keypoints=config["keypoints"]["num_source_keypoints"],
            )
        except ValueError:
            invalid_pose_shapes += 1
            continue

        if keypoints.shape[0] == 0:
            continue
        selected = select_keypoints(keypoints, indices)
        confidence_arrays.append(selected[..., 2])

    scale_info = compute_confidence_scale(
        confidence_arrays,
        method=config["normalization"]["confidence"]["method"],
        percentile=config["normalization"]["confidence"]["percentile"],
    )
    scale_info.update(
        {
            "fit_split": config["normalization"]["confidence"]["fit_on_split"],
            "rows_considered": int(len(working)),
            "rows_skipped_pose_status": int(skipped_status),
            "rows_missing_pose_file": int(missing_pose_files),
            "rows_invalid_pose_shape": int(invalid_pose_shapes),
        }
    )
    LOGGER.info(
        "Confidence scale computation finished. scale=%s arrays=%s values=%s fallback=%s",
        scale_info["scale"],
        scale_info["num_arrays"],
        scale_info["num_values"],
        scale_info["fallback_used"],
    )
    if scale_info.get("warning"):
        LOGGER.warning("%s", scale_info["warning"])
    return scale_info


def process_sample(
    row: pd.Series,
    config: dict[str, Any],
    confidence_scale: float,
    dry_run: bool = False,
) -> tuple[dict[str, Any], dict[str, int]]:
    """Build selected, normalized, and graph-tensor outputs for one sample."""

    subset = config["dataset"]["subset"]
    keypoint_cfg = config["keypoints"]
    normalization_cfg = config["normalization"]
    confidence_cfg = normalization_cfg["confidence"]
    sequence_cfg = config["sequence"]
    graph_cfg = config["graph_tensor"]
    options_cfg = config["options"]

    sample_id = _safe_str(row.get("sample_id")).strip()
    video_id = _safe_str(row.get("video_id")).strip()
    gloss = _safe_str(row.get("gloss")).strip()
    split = _safe_str(row.get("split")).strip().lower()
    class_id = _parse_optional_int(row.get("class_id"))
    pose_path = resolve_pose_file_path(row, config)

    selected_path = (
        Path(config["output"]["selected_root"]) / subset / split / f"{sample_id}.npz"
        if options_cfg["save_selected"]
        else None
    )
    normalized_path = (
        Path(config["output"]["normalized_root"]) / subset / split / f"{sample_id}.npz"
        if options_cfg["save_normalized"]
        else None
    )
    graph_tensor_path = (
        Path(config["output"]["graph_tensor_root"]) / subset / split / f"{sample_id}.npz"
        if options_cfg["save_graph_tensor"]
        else None
    )

    notes = _split_notes(row.get("notes"))
    tensor_shape_text = _json_array_text(graph_cfg["expected_shape"])
    result = {
        "instance_uid": _safe_str(row.get("instance_uid")),
        "sample_id": sample_id,
        "video_id": video_id,
        "gloss": gloss,
        "class_id": class_id,
        "split": split,
        "pose_path": _stringify_path(pose_path),
        "selected_path": _stringify_path(selected_path),
        "normalized_path": _stringify_path(normalized_path),
        "graph_tensor_path": _stringify_path(graph_tensor_path),
        "source_layout": keypoint_cfg["source_layout"],
        "keypoint_set": keypoint_cfg["keypoint_set"],
        "selected_indices": _json_array_text(keypoint_cfg["selected_indices"]),
        "num_frames_original": 0,
        "num_frames_output": 0,
        "num_source_keypoints": keypoint_cfg["num_source_keypoints"],
        "num_selected_keypoints": keypoint_cfg["num_selected_keypoints"],
        "num_channels": graph_cfg["num_channels"],
        "num_persons": graph_cfg["num_persons"],
        "tensor_shape": tensor_shape_text,
        "confidence_scale": float(confidence_scale),
        "xy_normalization": f"clip_to_[0,{normalization_cfg['image_width']}]x[0,{normalization_cfg['image_height']}]_then_scale_to_[-1,1]",
        "status": "failed",
        "error_message": "",
        "notes": "",
    }
    stats = {
        "selected_outputs": 0,
        "normalized_outputs": 0,
        "graph_outputs": 0,
        "x_out_of_bounds": 0,
        "y_out_of_bounds": 0,
        "xy_out_of_bounds": 0,
        "non_finite_keypoints": 0,
    }

    try:
        pose_status = _safe_str(row.get("status")).strip().lower()
        pose_error = _safe_str(row.get("error_message")).strip()
        if config["input"]["require_pose_status_ok"] and pose_status != "ok":
            result["status"] = "skipped_pose_not_ok"
            result["error_message"] = pose_error or "Pose manifest status is not ok."
            return result, stats

        if not pose_path.exists():
            if options_cfg["fail_on_missing_pose_file"]:
                raise FileNotFoundError(f"Pose file does not exist: {pose_path}")
            result["status"] = "missing_pose_file"
            result["error_message"] = "Pose file does not exist."
            return result, stats

        try:
            keypoints = load_pose_keypoints(
                pose_path,
                expected_num_keypoints=keypoint_cfg["num_source_keypoints"],
            )
        except ValueError as exc:
            result["status"] = "invalid_pose_shape"
            result["error_message"] = str(exc)
            return result, stats

        num_frames_original = int(keypoints.shape[0])
        result["num_frames_original"] = num_frames_original
        if num_frames_original == 0:
            result["status"] = "empty_pose_sequence"
            result["error_message"] = "Pose sequence contains zero frames."
            return result, stats

        selected_keypoints = select_keypoints(keypoints, keypoint_cfg["selected_indices"])
        if options_cfg["save_selected"] and not dry_run and selected_path is not None:
            selected_payload = build_selected_keypoints_npz_payload(
                keypoints=selected_keypoints,
                keypoint_set=keypoint_cfg["keypoint_set"],
                sample_id=sample_id,
                video_id=video_id,
                gloss=gloss,
                class_id=class_id if class_id is not None else -1,
                split=split,
                source_layout=keypoint_cfg["source_layout"],
            )
            _write_npz(selected_payload, selected_path, overwrite=options_cfg["overwrite"])
        if options_cfg["save_selected"]:
            stats["selected_outputs"] = 1

        normalized_keypoints, xy_stats = normalize_xy_to_minus1_1(
            selected_keypoints,
            image_width=normalization_cfg["image_width"],
            image_height=normalization_cfg["image_height"],
            clip=normalization_cfg["clip_xy_to_image"],
        )
        for key, value in xy_stats.items():
            stats[key] = int(value)

        if confidence_cfg["enabled"]:
            normalized_keypoints = normalize_confidence(
                normalized_keypoints,
                confidence_scale=confidence_scale,
                clip_min=float(confidence_cfg["clip_range"][0]),
                clip_max=float(confidence_cfg["clip_range"][1]),
            )
        normalized_keypoints, non_finite_count = sanitize_non_finite_keypoints(normalized_keypoints)
        stats["non_finite_keypoints"] = int(non_finite_count)

        if options_cfg["save_normalized"] and not dry_run and normalized_path is not None:
            normalized_payload = {
                "keypoints": np.asarray(normalized_keypoints, dtype=np.float32),
                "selected_indices": np.asarray(keypoint_cfg["selected_indices"], dtype=np.int32),
                "selected_names": np.asarray(keypoint_cfg["selected_names"]),
                "keypoint_set": np.asarray(keypoint_cfg["keypoint_set"]),
                "normalization": np.asarray(
                    json.dumps(_build_normalization_metadata(config), ensure_ascii=False)
                ),
                "confidence_scale": np.asarray(float(confidence_scale), dtype=np.float32),
                "sample_id": np.asarray(sample_id),
                "video_id": np.asarray(video_id),
                "gloss": np.asarray(gloss),
                "class_id": np.asarray(class_id if class_id is not None else -1, dtype=np.int32),
                "split": np.asarray(split),
                "num_frames_original": np.asarray(num_frames_original, dtype=np.int32),
            }
            _write_npz(normalized_payload, normalized_path, overwrite=options_cfg["overwrite"])
        if options_cfg["save_normalized"]:
            stats["normalized_outputs"] = 1

        try:
            fixed_keypoints = fix_sequence_length(
                normalized_keypoints,
                target_num_frames=sequence_cfg["target_num_frames"],
                short_strategy=sequence_cfg["short_sequence_strategy"],
                long_strategy=sequence_cfg["long_sequence_strategy"],
            )
        except ValueError as exc:
            if str(exc) == "empty_pose_sequence":
                result["status"] = "empty_pose_sequence"
            else:
                result["status"] = "failed"
            result["error_message"] = str(exc)
            return result, stats

        graph_tensor = to_graph_tensor_ctvm(
            fixed_keypoints,
            num_persons=graph_cfg["num_persons"],
        )
        expected_shape = tuple(int(value) for value in graph_cfg["expected_shape"])
        if tuple(graph_tensor.shape) != expected_shape:
            result["status"] = "invalid_graph_tensor_shape"
            result["error_message"] = (
                f"Expected graph tensor shape {expected_shape}, got {tuple(graph_tensor.shape)}."
            )
            return result, stats

        result["num_frames_output"] = int(graph_tensor.shape[1])
        if options_cfg["save_graph_tensor"] and not dry_run and graph_tensor_path is not None:
            graph_payload = {
                "data": np.asarray(graph_tensor, dtype=np.float32),
                "keypoint_set": np.asarray(keypoint_cfg["keypoint_set"]),
                "layout": np.asarray(graph_cfg["layout"]),
                "selected_indices": np.asarray(keypoint_cfg["selected_indices"], dtype=np.int32),
                "selected_names": np.asarray(keypoint_cfg["selected_names"]),
                "sample_id": np.asarray(sample_id),
                "video_id": np.asarray(video_id),
                "gloss": np.asarray(gloss),
                "class_id": np.asarray(class_id if class_id is not None else -1, dtype=np.int32),
                "split": np.asarray(split),
                "num_frames_original": np.asarray(num_frames_original, dtype=np.int32),
                "num_frames_output": np.asarray(int(graph_tensor.shape[1]), dtype=np.int32),
                "tensor_shape": np.asarray(graph_tensor.shape, dtype=np.int32),
            }
            _write_npz(graph_payload, graph_tensor_path, overwrite=options_cfg["overwrite"])
        if options_cfg["save_graph_tensor"]:
            stats["graph_outputs"] = 1

        result["status"] = "ok"
        return result, stats
    except (FileExistsError, OSError) as exc:
        result["status"] = "write_error"
        result["error_message"] = str(exc)
        return result, stats
    except FileNotFoundError as exc:
        result["status"] = "missing_pose_file"
        result["error_message"] = str(exc)
        return result, stats
    except Exception as exc:  # pragma: no cover - runtime failures
        result["status"] = "failed"
        result["error_message"] = str(exc)
        return result, stats
    finally:
        result["notes"] = _join_notes(notes)


def process_split(
    split: str,
    manifest: pd.DataFrame,
    config: dict[str, Any],
    confidence_scale: float,
    limit: int | None = None,
    dry_run: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Process one split of the pose manifest into skeleton branch inputs."""

    working = _working_manifest(manifest, limit=limit)
    LOGGER.info(
        "Processing split=%s subset=%s samples=%s",
        split,
        config["dataset"]["subset"],
        len(working),
    )

    rows: list[dict[str, Any]] = []
    totals = {
        "selected_outputs": 0,
        "normalized_outputs": 0,
        "graph_outputs": 0,
        "x_out_of_bounds": 0,
        "y_out_of_bounds": 0,
        "xy_out_of_bounds": 0,
        "non_finite_keypoints": 0,
    }

    for _, row in working.iterrows():
        result_row, result_stats = process_sample(
            row=row,
            config=config,
            confidence_scale=confidence_scale,
            dry_run=dry_run,
        )
        rows.append(result_row)
        for key, value in result_stats.items():
            totals[key] += int(value)

    output = pd.DataFrame(rows)
    if output.empty:
        output = pd.DataFrame(columns=SKELETON_INPUT_MANIFEST_COLUMNS)
    output = validate_manifest_schema(output, SKELETON_INPUT_MANIFEST_COLUMNS, name=f"skeleton:{split}")
    stats = {
        "split": split,
        "input_samples": int(len(working)),
        "ok_samples": int((output["status"] == "ok").sum()),
        "error_samples": int((output["status"] != "ok").sum()),
        "status_counts": output["status"].value_counts().to_dict(),
        **totals,
    }
    LOGGER.info(
        "Finished split=%s ok=%s errors=%s graph_outputs=%s",
        split,
        stats["ok_samples"],
        stats["error_samples"],
        stats["graph_outputs"],
    )
    return output, stats


def build_report(
    combined_manifest: pd.DataFrame,
    split_stats: list[dict[str, Any]],
    config: dict[str, Any],
    manifest_paths: list[Path],
    confidence_scale_info: dict[str, Any],
) -> str:
    """Build the skeleton-input preprocessing report."""

    subset = config["dataset"]["subset"]
    keypoint_cfg = config["keypoints"]
    normalization_cfg = config["normalization"]
    sequence_cfg = config["sequence"]
    graph_cfg = config["graph_tensor"]
    mouth_indices = keypoint_cfg.get("mouth_indices", ())
    mouth_names = keypoint_cfg.get("mouth_names", ())

    status_counts = combined_manifest["status"].value_counts().to_dict()
    total_selected = int(sum(stats["selected_outputs"] for stats in split_stats))
    total_normalized = int(sum(stats["normalized_outputs"] for stats in split_stats))
    total_graph = int(sum(stats["graph_outputs"] for stats in split_stats))
    total_xy_oob = int(sum(stats["xy_out_of_bounds"] for stats in split_stats))
    total_x_oob = int(sum(stats["x_out_of_bounds"] for stats in split_stats))
    total_y_oob = int(sum(stats["y_out_of_bounds"] for stats in split_stats))
    total_non_finite = int(sum(stats["non_finite_keypoints"] for stats in split_stats))

    lines = [
        f"# WLASL Skeleton Inputs Report: {subset} {keypoint_cfg['keypoint_set']}",
        "",
        f"- subset: `{subset}`",
        f"- keypoint_set: `{keypoint_cfg['keypoint_set']}`",
        f"- source_layout: `{keypoint_cfg['source_layout']}`",
        f"- selected indices: `{_json_array_text(keypoint_cfg['selected_indices'])}`",
        f"- selected names: `{_json_array_text(keypoint_cfg['selected_names'])}`",
        f"- mouth indices: `{_json_array_text(mouth_indices)}`",
        f"- mouth names: `{_json_array_text(mouth_names)}`",
        f"- selected mapping note: {keypoint_cfg['mapping_note']}",
        f"- total samples: `{len(combined_manifest)}`",
        f"- status=ok count: `{int((combined_manifest['status'] == 'ok').sum())}`",
        f"- error count: `{int((combined_manifest['status'] != 'ok').sum())}`",
        f"- input pose count: `{len(combined_manifest)}`",
        f"- selected output count: `{total_selected}`",
        f"- normalized output count: `{total_normalized}`",
        f"- graph tensor output count: `{total_graph}`",
        f"- target_num_frames: `{sequence_cfg['target_num_frames']}`",
        f"- graph tensor shape: `{tuple(graph_cfg['expected_shape'])}`",
        f"- image_width/image_height: `{normalization_cfg['image_width']}x{normalization_cfg['image_height']}`",
        "",
        "## Confidence Normalization",
        "",
        f"- method: `{confidence_scale_info['method']}`",
        f"- percentile: `{confidence_scale_info['percentile']}`",
        f"- fit split: `{confidence_scale_info['fit_split']}`",
        f"- confidence_scale: `{confidence_scale_info['scale']}`",
        f"- train arrays used: `{confidence_scale_info['num_arrays']}`",
        f"- train values used: `{confidence_scale_info['num_values']}`",
        f"- fallback used: `{confidence_scale_info['fallback_used']}`",
        "",
        "## Split Summary",
        "",
    ]

    for stats in split_stats:
        lines.extend(
            [
                f"### {stats['split']}",
                "",
                f"- input: `{stats['input_samples']}`",
                f"- ok: `{stats['ok_samples']}`",
                f"- errors: `{stats['error_samples']}`",
                f"- selected outputs: `{stats['selected_outputs']}`",
                f"- normalized outputs: `{stats['normalized_outputs']}`",
                f"- graph outputs: `{stats['graph_outputs']}`",
                "",
            ]
        )

    lines.extend(
        [
            "## Coordinate Sanitization",
            "",
            f"- x out-of-bounds before clipping: `{total_x_oob}`",
            f"- y out-of-bounds before clipping: `{total_y_oob}`",
            f"- total x/y out-of-bounds before clipping: `{total_xy_oob}`",
            f"- non-finite keypoint values replaced: `{total_non_finite}`",
            "",
            "## Common Errors",
            "",
        ]
    )
    if not status_counts:
        lines.append("- none")
    else:
        for status, count in sorted(status_counts.items()):
            lines.append(f"- {status}: `{count}`")

    lines.extend(["", "## Output Manifests", ""])
    for manifest_path in manifest_paths:
        lines.append(f"- `{_stringify_path(manifest_path)}`")

    if confidence_scale_info.get("warning"):
        lines.extend(["", "## Warnings", "", f"- {confidence_scale_info['warning']}"])

    lines.append("")
    return "\n".join(lines)


def run(
    config_path: Path,
    subset: str | None = None,
    limit: int | None = None,
    dry_run: bool = False,
) -> int:
    """Run skeleton input generation for one subset and reduced keypoint set."""

    config = load_config(config_path, subset_override=subset)
    paths = resolve_paths(config)
    subset_name = config["dataset"]["subset"]
    keypoint_set = config["keypoints"]["keypoint_set"]

    ensure_dir(paths["logs_root"])
    global LOGGER
    LOGGER = setup_logger(
        __name__,
        paths["logs_root"] / f"build_skeleton_{keypoint_set}_{subset_name}.log",
    )

    LOGGER.info("Starting skeleton input build.")
    LOGGER.info("Config path: %s", _stringify_path(config["config_path"]))
    LOGGER.info("Subset: %s", subset_name)
    LOGGER.info("Keypoint set: %s", keypoint_set)
    LOGGER.info("Splits: %s", ", ".join(config["input"]["splits"]))
    LOGGER.info("Limit per split: %s", limit)
    LOGGER.info("Dry run: %s", dry_run)
    LOGGER.info("Selected indices: %s", _json_array_text(config["keypoints"]["selected_indices"]))
    if config["keypoints"]["mouth_indices"]:
        LOGGER.info("Mouth indices: %s", _json_array_text(config["keypoints"]["mouth_indices"]))
        LOGGER.info("Mouth names: %s", _json_array_text(config["keypoints"]["mouth_names"]))
    LOGGER.info("Expected graph tensor shape: %s", tuple(config["graph_tensor"]["expected_shape"]))

    if not dry_run:
        ensure_dir(paths["selected_subset_root"])
        ensure_dir(paths["normalized_subset_root"])
        ensure_dir(paths["graph_tensor_subset_root"])
        ensure_dir(paths["manifests_root"])
        ensure_dir(paths["reports_root"])

    manifests_by_split: dict[str, pd.DataFrame] = {}
    for split in config["input"]["splits"]:
        manifest_path = paths["pose_manifest_root"] / config["input"]["manifest_filenames"][split]
        LOGGER.info("Loading pose manifest for split=%s: %s", split, _stringify_path(manifest_path))
        manifests_by_split[split] = load_pose_manifest(manifest_path, split)

    LOGGER.info(
        "Computing confidence scale from split=%s.",
        config["normalization"]["confidence"]["fit_on_split"],
    )
    confidence_scale_info = compute_confidence_scale_from_train(
        manifests_by_split[config["normalization"]["confidence"]["fit_on_split"]],
        config=config,
        limit=limit,
    )
    confidence_scale = float(confidence_scale_info["scale"])
    LOGGER.info("Confidence scale value: %s", confidence_scale)

    split_outputs: list[pd.DataFrame] = []
    split_stats: list[dict[str, Any]] = []
    manifest_paths: list[Path] = []

    for split in config["input"]["splits"]:
        split_output, stats = process_split(
            split=split,
            manifest=manifests_by_split[split],
            config=config,
            confidence_scale=confidence_scale,
            limit=limit,
            dry_run=dry_run,
        )
        split_outputs.append(split_output)
        split_stats.append(stats)

        split_manifest_path = paths["manifests_root"] / f"{subset_name}_{keypoint_set}_{split}.csv"
        manifest_paths.append(split_manifest_path)
        if not dry_run:
            write_dataframe_csv(split_output, split_manifest_path)

    combined_manifest = (
        pd.concat(split_outputs, ignore_index=True)
        if split_outputs
        else pd.DataFrame(columns=SKELETON_INPUT_MANIFEST_COLUMNS)
    )
    if not combined_manifest.empty:
        combined_manifest = combined_manifest.sort_values(
            by=["split", "sample_id", "video_id"]
        ).reset_index(drop=True)
    combined_manifest = validate_manifest_schema(
        combined_manifest,
        SKELETON_INPUT_MANIFEST_COLUMNS,
        name="skeleton_all",
    )

    all_manifest_path = paths["manifests_root"] / f"{subset_name}_{keypoint_set}_all.csv"
    report_path = paths["reports_root"] / f"{subset_name}_{keypoint_set}_skeleton_inputs_report.md"
    manifest_paths.append(all_manifest_path)
    report_text = build_report(
        combined_manifest=combined_manifest,
        split_stats=split_stats,
        config=config,
        manifest_paths=manifest_paths,
        confidence_scale_info=confidence_scale_info,
    )

    if not dry_run:
        write_dataframe_csv(combined_manifest, all_manifest_path)
        write_text(report_text, report_path)

    LOGGER.info("Output manifest paths: %s", ", ".join(_stringify_path(path) for path in manifest_paths))
    LOGGER.info("Report path: %s", _stringify_path(report_path))
    LOGGER.info(
        "Finished skeleton input build. total=%s ok=%s errors=%s",
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
