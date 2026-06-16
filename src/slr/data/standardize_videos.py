"""Standardize WLASL videos into a shared normalized layer."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from slr.data.manifests import STANDARDIZED_COLUMNS, SUBSET_MANIFEST_COLUMNS
from slr.data.validation import require_columns, validate_manifest_schema
from slr.utils.bbox import (
    BoundingBox,
    bbox_to_int,
    bbox_to_string,
    clip_bbox,
    expand_bbox,
    is_valid_bbox,
    parse_bbox,
)
from slr.utils.image import resize_letterbox, save_image
from slr.utils.io import ensure_dir, read_csv, read_yaml, write_dataframe_csv, write_text
from slr.utils.logging import setup_logger
from slr.utils.video import probe_video_basic, read_frames, write_video_from_frames


DEFAULT_CONFIG_PATH = Path("configs/preprocessing/standardize/standardize_nslt100.yaml")
ALLOWED_SPLITS = ("train", "val", "test")
LOGGER = setup_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for standardized layer generation."""

    parser = argparse.ArgumentParser(
        description="Standardize WLASL videos by crop, resize, and letterbox rules."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the standardization configuration.",
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
        help="Validate config and manifests without writing outputs.",
    )
    return parser.parse_args()


def _resolve_path(base_dir: Path, value: str | Path) -> Path:
    """Resolve an absolute or repo-relative path."""

    path = Path(value)
    return path if path.is_absolute() else (base_dir / path)


def _stringify_path(path: Path | None) -> str:
    """Return a stable POSIX-like string for a path."""

    if path is None:
        return ""
    return path.as_posix()


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


def _normalize_bool(value: Any) -> bool:
    """Interpret manifest boolean-like values robustly."""

    if isinstance(value, bool):
        return value
    if value is None or pd.isna(value):
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


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


def load_config(config_path: Path, subset_override: str | None = None) -> dict[str, Any]:
    """Load, normalize, and validate the standardization config."""

    base_dir = Path.cwd()
    config = read_yaml(config_path)

    dataset_cfg = config.get("dataset", {})
    input_cfg = config.get("input", {})
    output_cfg = config.get("output", {})
    standardization_cfg = config.get("standardization", {})

    subset = subset_override or input_cfg.get("subset", "nslt100")
    splits = list(input_cfg.get("splits", list(ALLOWED_SPLITS)))
    invalid_splits = [split for split in splits if split not in ALLOWED_SPLITS]
    if invalid_splits:
        raise ValueError(f"Unsupported splits in config: {invalid_splits}")

    dataset_root = _resolve_path(
        base_dir, dataset_cfg.get("root", "data/datasets/WLASL")
    )
    output_root = _resolve_path(
        base_dir, output_cfg.get("root", dataset_root / "standardized")
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
            "manifest_root": _resolve_path(
                base_dir,
                input_cfg.get(
                    "manifest_root",
                    dataset_root / "index" / "subsets_available",
                ),
            ),
            "manifest_filenames": {
                split: input_cfg.get("manifest_filenames", {}).get(
                    split, f"{split}.csv"
                )
                for split in splits
            },
        },
        "output": {
            "root": output_root,
            "frames_root": _resolve_path(base_dir, output_cfg.get("frames_root", output_root / "frames")),
            "videos_root": _resolve_path(base_dir, output_cfg.get("videos_root", output_root / "videos")),
            "manifests_root": _resolve_path(base_dir, output_cfg.get("manifests_root", output_root / "manifests")),
            "reports_root": _resolve_path(base_dir, output_cfg.get("reports_root", output_root / "reports")),
            "logs_root": _resolve_path(base_dir, output_cfg.get("logs_root", output_root / "logs")),
        },
        "standardization": {
            "output_size": {
                "width": int(
                    standardization_cfg.get("output_size", {}).get("width", 288)
                ),
                "height": int(
                    standardization_cfg.get("output_size", {}).get("height", 384)
                ),
            },
            "keep_aspect_ratio": bool(
                standardization_cfg.get("keep_aspect_ratio", True)
            ),
            "letterbox": bool(standardization_cfg.get("letterbox", True)),
            "letterbox_value": int(standardization_cfg.get("letterbox_value", 0)),
            "crop_with_bbox": bool(standardization_cfg.get("crop_with_bbox", True)),
            "bbox_margin": {
                "left": float(
                    standardization_cfg.get("bbox_margin", {}).get("left", 0.15)
                ),
                "right": float(
                    standardization_cfg.get("bbox_margin", {}).get("right", 0.15)
                ),
                "top": float(
                    standardization_cfg.get("bbox_margin", {}).get("top", 0.10)
                ),
                "bottom": float(
                    standardization_cfg.get("bbox_margin", {}).get("bottom", 0.15)
                ),
            },
            "fallback_to_full_frame_if_bbox_invalid": bool(
                standardization_cfg.get(
                    "fallback_to_full_frame_if_bbox_invalid",
                    True,
                )
            ),
            "frame_index_base": int(standardization_cfg.get("frame_index_base", 0)),
            "save_frames": bool(standardization_cfg.get("save_frames", True)),
            "save_video": bool(standardization_cfg.get("save_video", False)),
            "frame_format": str(standardization_cfg.get("frame_format", "jpg")).lower(),
            "jpg_quality": int(standardization_cfg.get("jpg_quality", 95)),
            "video_codec": str(standardization_cfg.get("video_codec", "mp4v")),
            "overwrite": bool(standardization_cfg.get("overwrite", True)),
        },
    }

    if not resolved["standardization"]["letterbox"]:
        raise ValueError("This standardized layer requires letterbox=True.")
    if not resolved["standardization"]["keep_aspect_ratio"]:
        raise ValueError("This standardized layer requires keep_aspect_ratio=True.")
    return resolved


def resolve_paths(config: dict[str, Any]) -> dict[str, Path]:
    """Resolve output and manifest paths for the requested subset."""

    subset = config["input"]["subset"]
    manifest_root = Path(config["input"]["manifest_root"]) / subset
    output_cfg = config["output"]
    return {
        "manifest_root": manifest_root,
        "frames_subset_root": Path(output_cfg["frames_root"]) / subset,
        "videos_subset_root": Path(output_cfg["videos_root"]) / subset,
        "manifests_root": Path(output_cfg["manifests_root"]),
        "reports_root": Path(output_cfg["reports_root"]),
        "logs_root": Path(output_cfg["logs_root"]),
    }


def load_split_manifest(manifest_path: Path, split: str) -> pd.DataFrame:
    """Load and validate a single split manifest from the index layer."""

    dtype_map = {
        "instance_uid": "string",
        "sample_id": "string",
        "video_id": "string",
        "gloss": "string",
        "split": "string",
        "video_path": "string",
        "bbox": "string",
        "source": "string",
        "url": "string",
        "notes": "string",
    }
    frame = read_csv(manifest_path, dtype=dtype_map)
    require_columns(frame, SUBSET_MANIFEST_COLUMNS, name=f"manifest:{split}")
    frame = frame.copy()
    frame["split"] = frame["split"].fillna("").astype(str).str.strip().str.lower()
    frame = frame[frame["split"] == split].reset_index(drop=True)
    return frame


def _resolve_frame_range(
    row: pd.Series,
    total_frames: int,
    frame_index_base: int,
    notes: list[str],
) -> tuple[int, int, int | None, int | None]:
    """Resolve a safe frame range for one sample."""

    original_start = _parse_optional_int(row.get("start_frame"))
    original_end = _parse_optional_int(row.get("end_frame"))
    if total_frames <= 0:
        return 0, -1, original_start, original_end

    if original_start is None or original_end is None or original_end < 0:
        _add_note(notes, "frame_range_full_video_missing_or_invalid_bounds")
        return 0, total_frames - 1, original_start, original_end

    used_start = original_start - frame_index_base
    used_end = original_end - frame_index_base

    if used_start < 0:
        _add_note(notes, "start_frame_clipped_to_zero")
        used_start = 0
    if used_end >= total_frames:
        _add_note(notes, "end_frame_clipped_to_video_end")
        used_end = total_frames - 1
    if used_end < 0 or used_start > used_end:
        _add_note(notes, "invalid_frame_range_fallback_full_video")
        return 0, total_frames - 1, original_start, original_end
    return used_start, used_end, original_start, original_end


def _resolve_crop_box(
    row: pd.Series,
    width: int,
    height: int,
    config: dict[str, Any],
    notes: list[str],
) -> tuple[BoundingBox, bool, bool]:
    """Resolve the bbox to crop with, including fallback behavior."""

    standardization_cfg = config["standardization"]
    full_frame_box = BoundingBox(0.0, 0.0, float(width), float(height))
    if not standardization_cfg["crop_with_bbox"]:
        return full_frame_box, False, False

    parsed = parse_bbox(row.get("bbox"))
    if not is_valid_bbox(parsed):
        if standardization_cfg["fallback_to_full_frame_if_bbox_invalid"]:
            _add_note(notes, "invalid_bbox_fallback_full_frame")
            return full_frame_box, False, True
        raise ValueError("invalid_bbox")

    margins = standardization_cfg["bbox_margin"]
    expanded = expand_bbox(
        parsed,
        left=margins["left"],
        right=margins["right"],
        top=margins["top"],
        bottom=margins["bottom"],
    )
    clipped = clip_bbox(expanded, width=width, height=height)
    if not is_valid_bbox(clipped):
        if standardization_cfg["fallback_to_full_frame_if_bbox_invalid"]:
            _add_note(notes, "invalid_bbox_fallback_full_frame")
            return full_frame_box, False, True
        raise ValueError("invalid_bbox")
    return clipped, True, False


def read_video_frame_range(
    video_path: Path,
    start_frame: int,
    end_frame: int,
) -> list[Any]:
    """Read a contiguous frame range from the raw video."""

    return read_frames(video_path, start_frame=start_frame, end_frame=end_frame)


def write_standardized_frames(
    frames: list[Any],
    output_dir: Path,
    frame_format: str,
    jpg_quality: int,
) -> int:
    """Write standardized frames with stable filenames."""

    ensure_dir(output_dir)
    written = 0
    suffix = frame_format.lstrip(".")
    for index, frame in enumerate(frames, start=1):
        frame_path = output_dir / f"{index:06d}.{suffix}"
        save_image(frame_path, frame, jpg_quality=jpg_quality)
        written += 1
    return written


def write_standardized_video(
    frames: list[Any],
    output_path: Path,
    fps: float,
    codec: str,
) -> None:
    """Write a standardized video file from in-memory frames."""

    write_video_from_frames(frames, output_path, fps=fps, codec=codec)


def standardize_one_sample(
    row: pd.Series,
    subset: str,
    config: dict[str, Any],
    split_frames_root: Path,
    split_videos_root: Path,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Standardize one sample and return a manifest row."""

    standardization_cfg = config["standardization"]
    output_size = standardization_cfg["output_size"]
    sample_id = str(row.get("sample_id", "")).strip()
    split = str(row.get("split", "")).strip().lower()
    raw_video_path = Path(str(row.get("video_path", "")).strip())

    frames_dir = split_frames_root / sample_id
    standardized_video_path = split_videos_root / f"{sample_id}.mp4"
    notes = _split_notes(row.get("notes"))

    result = {
        "instance_uid": str(row.get("instance_uid", "") or ""),
        "sample_id": sample_id,
        "video_id": str(row.get("video_id", "") or ""),
        "gloss": str(row.get("gloss", "") or ""),
        "class_id": row.get("class_id"),
        "split": split,
        "raw_video_path": _stringify_path(raw_video_path),
        "standardized_video_path": (
            _stringify_path(standardized_video_path)
            if standardization_cfg["save_video"]
            else ""
        ),
        "frames_dir": (
            _stringify_path(frames_dir) if standardization_cfg["save_frames"] else ""
        ),
        "num_frames": 0,
        "fps": row.get("fps"),
        "original_width": None,
        "original_height": None,
        "output_width": output_size["width"],
        "output_height": output_size["height"],
        "original_start_frame": _parse_optional_int(row.get("start_frame")),
        "original_end_frame": _parse_optional_int(row.get("end_frame")),
        "used_start_frame": None,
        "used_end_frame": None,
        "original_bbox": str(row.get("bbox", "") or ""),
        "used_bbox": "",
        "crop_applied": False,
        "bbox_fallback_used": False,
        "save_frames": standardization_cfg["save_frames"],
        "save_video": standardization_cfg["save_video"],
        "status": "failed",
        "error_message": "",
        "notes": "",
    }

    try:
        if not _normalize_bool(row.get("is_present_locally", True)):
            result["status"] = "missing_video"
            result["error_message"] = "Manifest marks sample as unavailable locally."
            return result
        if not raw_video_path.exists():
            result["status"] = "missing_video"
            result["error_message"] = "Raw video file does not exist."
            return result

        video_info = probe_video_basic(raw_video_path)
        result["fps"] = (
            float(video_info["fps"])
            if float(video_info["fps"]) > 0
            else float(row.get("fps") or 25.0)
        )
        result["original_width"] = int(video_info["width"])
        result["original_height"] = int(video_info["height"])
        total_frames = int(video_info["num_frames"])
        if total_frames <= 0:
            result["status"] = "empty_video"
            result["error_message"] = "Video contains zero readable frames."
            return result

        used_start, used_end, original_start, original_end = _resolve_frame_range(
            row=row,
            total_frames=total_frames,
            frame_index_base=standardization_cfg["frame_index_base"],
            notes=notes,
        )
        result["original_start_frame"] = original_start
        result["original_end_frame"] = original_end
        result["used_start_frame"] = used_start
        result["used_end_frame"] = used_end

        crop_box, crop_applied, bbox_fallback_used = _resolve_crop_box(
            row=row,
            width=result["original_width"],
            height=result["original_height"],
            config=config,
            notes=notes,
        )
        result["used_bbox"] = bbox_to_string(crop_box)
        result["crop_applied"] = crop_applied
        result["bbox_fallback_used"] = bbox_fallback_used

        if dry_run:
            result["status"] = "ok"
            result["notes"] = _join_notes(notes)
            return result

        overwrite = standardization_cfg["overwrite"]
        if standardization_cfg["save_frames"] and frames_dir.exists():
            if overwrite:
                shutil.rmtree(frames_dir)
            else:
                raise FileExistsError(f"Frames dir already exists: {frames_dir}")
        if standardization_cfg["save_video"] and standardized_video_path.exists():
            if overwrite:
                standardized_video_path.unlink()
            else:
                raise FileExistsError(
                    f"Standardized video already exists: {standardized_video_path}"
                )

        raw_frames = read_video_frame_range(raw_video_path, used_start, used_end)
        if not raw_frames:
            result["status"] = "read_error"
            result["error_message"] = "No frames could be read for the resolved range."
            return result

        x1, y1, x2, y2 = bbox_to_int(crop_box)
        standardized_frames = []
        for raw_frame in raw_frames:
            cropped = raw_frame[y1:y2, x1:x2]
            if cropped.size == 0:
                result["status"] = "invalid_bbox"
                result["error_message"] = "Crop box produced an empty frame."
                return result
            standardized_frames.append(
                resize_letterbox(
                    cropped,
                    target_width=output_size["width"],
                    target_height=output_size["height"],
                    letterbox_value=standardization_cfg["letterbox_value"],
                )
            )

        if standardization_cfg["save_frames"]:
            try:
                written = write_standardized_frames(
                    standardized_frames,
                    frames_dir,
                    frame_format=standardization_cfg["frame_format"],
                    jpg_quality=standardization_cfg["jpg_quality"],
                )
                result["num_frames"] = written
            except Exception as exc:  # pragma: no cover - exercised via runtime failures
                result["status"] = "frame_write_error"
                result["error_message"] = str(exc)
                return result
        else:
            result["num_frames"] = len(standardized_frames)

        if standardization_cfg["save_video"]:
            try:
                write_standardized_video(
                    standardized_frames,
                    standardized_video_path,
                    fps=float(result["fps"] or 25.0),
                    codec=standardization_cfg["video_codec"],
                )
            except Exception as exc:  # pragma: no cover - exercised via runtime failures
                result["status"] = "video_write_error"
                result["error_message"] = str(exc)
                return result

        result["status"] = "ok"
        result["notes"] = _join_notes(notes)
        return result
    except FileNotFoundError as exc:
        result["status"] = "missing_video"
        result["error_message"] = str(exc)
        return result
    except ValueError as exc:
        if str(exc) == "invalid_bbox":
            result["status"] = "invalid_bbox"
        else:
            result["status"] = "invalid_frame_range"
        result["error_message"] = str(exc)
        return result
    except Exception as exc:  # pragma: no cover - exercised via runtime failures
        result["status"] = "failed"
        result["error_message"] = str(exc)
        return result
    finally:
        result["notes"] = _join_notes(notes if "notes" in locals() else [])


def standardize_split(
    split: str,
    manifest: pd.DataFrame,
    subset: str,
    config: dict[str, Any],
    paths: dict[str, Path],
    limit: int | None = None,
    dry_run: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Standardize all samples for one split."""

    frame_split_root = ensure_dir(paths["frames_subset_root"] / split)
    video_split_root = ensure_dir(paths["videos_subset_root"] / split)

    working_frame = manifest.copy()
    working_frame = working_frame.sort_values(by=["sample_id", "video_id"]).reset_index(drop=True)
    if limit is not None:
        working_frame = working_frame.head(limit).reset_index(drop=True)

    LOGGER.info("Processing split=%s subset=%s samples=%s", split, subset, len(working_frame))
    rows: list[dict[str, Any]] = []

    for _, row in working_frame.iterrows():
        rows.append(
            standardize_one_sample(
                row=row,
                subset=subset,
                config=config,
                split_frames_root=frame_split_root,
                split_videos_root=video_split_root,
                dry_run=dry_run,
            )
        )

    output = pd.DataFrame(rows)
    if output.empty:
        output = pd.DataFrame(columns=STANDARDIZED_COLUMNS)
    output = validate_manifest_schema(output, STANDARDIZED_COLUMNS, name=f"standardized:{split}")
    stats = {
        "split": split,
        "input_samples": int(len(working_frame)),
        "ok_samples": int((output["status"] == "ok").sum()),
        "error_samples": int((output["status"] != "ok").sum()),
        "status_counts": output["status"].value_counts().to_dict(),
        "frames_written": int(output["num_frames"].fillna(0).astype(int).sum()),
    }
    return output, stats


def build_standardization_report(
    combined_manifest: pd.DataFrame,
    split_stats: list[dict[str, Any]],
    config: dict[str, Any],
    output_manifest_paths: list[Path],
) -> str:
    """Build the standardized layer markdown report."""

    subset = config["input"]["subset"]
    standardization_cfg = config["standardization"]
    output_size = standardization_cfg["output_size"]

    lines = [
        f"# WLASL Standardization Report: {subset}",
        "",
        f"- Subset: `{subset}`",
        f"- Splits processed: `{', '.join(config['input']['splits'])}`",
        f"- Total samples: `{len(combined_manifest)}`",
        f"- Total status=ok: `{int((combined_manifest['status'] == 'ok').sum())}`",
        f"- Total errors: `{int((combined_manifest['status'] != 'ok').sum())}`",
        f"- Total frames written: `{int(combined_manifest['num_frames'].fillna(0).astype(int).sum())}`",
        f"- Output size: `{output_size['width']}x{output_size['height']}`",
        f"- crop_with_bbox: `{standardization_cfg['crop_with_bbox']}`",
        (
            "- bbox_margin: "
            f"`left={standardization_cfg['bbox_margin']['left']}, "
            f"right={standardization_cfg['bbox_margin']['right']}, "
            f"top={standardization_cfg['bbox_margin']['top']}, "
            f"bottom={standardization_cfg['bbox_margin']['bottom']}`"
        ),
        f"- save_frames: `{standardization_cfg['save_frames']}`",
        f"- save_video: `{standardization_cfg['save_video']}`",
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
                f"- Frames written: `{stats['frames_written']}`",
                "",
            ]
        )

    status_counts = combined_manifest["status"].value_counts().to_dict()
    note_counts: dict[str, int] = {}
    for note_string in combined_manifest["notes"].fillna("").astype(str):
        for note in _split_notes(note_string):
            note_counts[note] = note_counts.get(note, 0) + 1

    lines.extend(["## Common Errors", ""])
    if not status_counts and not note_counts:
        lines.append("- No samples were processed.")
    else:
        for status, count in sorted(status_counts.items()):
            lines.append(f"- {status}: `{count}`")
        for note, count in sorted(note_counts.items()):
            if "invalid_bbox_fallback_full_frame" in note or "frame_range" in note:
                lines.append(f"- {note}: `{count}`")

    lines.extend(["", "## Output Manifests", ""])
    for manifest_path in output_manifest_paths:
        lines.append(f"- `{_stringify_path(manifest_path)}`")

    lines.append("")
    return "\n".join(lines)


def run(
    config_path: Path,
    subset: str | None = None,
    limit: int | None = None,
    dry_run: bool = False,
) -> int:
    """Run standardized layer generation for one subset."""

    config = load_config(config_path, subset_override=subset)
    paths = resolve_paths(config)
    subset_name = config["input"]["subset"]

    ensure_dir(paths["manifest_root"])
    ensure_dir(paths["frames_subset_root"])
    ensure_dir(paths["videos_subset_root"])
    ensure_dir(paths["manifests_root"])
    ensure_dir(paths["reports_root"])
    ensure_dir(paths["logs_root"])

    global LOGGER
    LOGGER = setup_logger(
        __name__,
        paths["logs_root"] / f"standardize_{subset_name}.log",
    )

    LOGGER.info("Starting WLASL standardization.")
    LOGGER.info("Config path: %s", _stringify_path(config_path))
    LOGGER.info("Subset: %s", subset_name)
    LOGGER.info("Splits: %s", ", ".join(config["input"]["splits"]))
    LOGGER.info("Limit per split: %s", limit)
    LOGGER.info("Dry run: %s", dry_run)
    LOGGER.info(
        "Output size: width=%s height=%s",
        config["standardization"]["output_size"]["width"],
        config["standardization"]["output_size"]["height"],
    )
    LOGGER.info(
        "Crop with bbox: %s | Letterbox: %s | Save frames: %s | Save video: %s",
        config["standardization"]["crop_with_bbox"],
        config["standardization"]["letterbox"],
        config["standardization"]["save_frames"],
        config["standardization"]["save_video"],
    )

    split_outputs: list[pd.DataFrame] = []
    split_stats: list[dict[str, Any]] = []
    output_manifest_paths: list[Path] = []

    for split in config["input"]["splits"]:
        manifest_path = paths["manifest_root"] / config["input"]["manifest_filenames"][split]
        LOGGER.info("Loading manifest for split=%s: %s", split, _stringify_path(manifest_path))
        manifest = load_split_manifest(manifest_path, split=split)
        split_output, stats = standardize_split(
            split=split,
            manifest=manifest,
            subset=subset_name,
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
            "Finished split=%s ok=%s errors=%s frames=%s",
            split,
            stats["ok_samples"],
            stats["error_samples"],
            stats["frames_written"],
        )

    combined_manifest = (
        pd.concat(split_outputs, ignore_index=True)
        if split_outputs
        else pd.DataFrame(columns=STANDARDIZED_COLUMNS)
    )
    if not combined_manifest.empty:
        combined_manifest = combined_manifest.sort_values(
            by=["split", "sample_id", "video_id"]
        ).reset_index(drop=True)
    combined_manifest = validate_manifest_schema(
        combined_manifest,
        STANDARDIZED_COLUMNS,
        name="standardized_all",
    )

    all_manifest_path = paths["manifests_root"] / f"{subset_name}_all.csv"
    report_path = paths["reports_root"] / f"{subset_name}_standardization_report.md"
    output_manifest_paths.append(all_manifest_path)
    report_text = build_standardization_report(
        combined_manifest=combined_manifest,
        split_stats=split_stats,
        config=config,
        output_manifest_paths=output_manifest_paths,
    )

    if not dry_run:
        write_dataframe_csv(combined_manifest, all_manifest_path)
        write_text(report_text, report_path)

    LOGGER.info("Combined manifest: %s", _stringify_path(all_manifest_path))
    LOGGER.info("Report path: %s", _stringify_path(report_path))
    LOGGER.info(
        "Finished WLASL standardization. total=%s ok=%s errors=%s",
        len(combined_manifest),
        int((combined_manifest["status"] == "ok").sum()),
        int((combined_manifest["status"] != "ok").sum()),
    )
    return 0


def main() -> int:
    """CLI entrypoint."""

    args = parse_args()
    return run(
        config_path=args.config,
        subset=args.subset,
        limit=args.limit,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
