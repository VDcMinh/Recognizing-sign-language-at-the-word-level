"""Shared helpers for the NSLT1000 incremental regions pipeline."""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from slr.branches.regions.dataset import RegionClipDataset, region_collate_fn, resolve_region_tensor_path
from slr.branches.regions.region_schema import REGION_NAMES
from slr.data.manifests import POSE_MANIFEST_COLUMNS, REGION_INPUT_MANIFEST_COLUMNS, STANDARDIZED_COLUMNS
from slr.data.validation import require_columns
from slr.utils.io import ensure_dir, read_csv, remap_wlasl_path, stringify_path, write_dataframe_csv, write_json, write_text


ALLOWED_SPLITS = ("train", "val", "test")
DEFAULT_BASE_SUBSET = "nslt300"
DEFAULT_TARGET_SUBSET = "nslt1000"
DEFAULT_ACTIVE_REGIONS = tuple(REGION_NAMES)
DEFAULT_EXPECTED_SHAPE = (3, 3, 64, 112, 112)
DEFAULT_REPORT_ROOT = Path("reports/current/regions/nslt1000_incremental_pipeline")
DEFAULT_BASE_ROOT = Path("data/datasets/WLASL/branch_inputs/regions/rtmw_l")
DEFAULT_TARGET_SOURCE_ROOT = Path("data/datasets/WLASL/standardized")
DEFAULT_INCREMENTAL_ROOT = Path("data/datasets/WLASL/branch_inputs/regions/rtmw_l_incremental")
DEFAULT_UNION_ROOT = Path("data/datasets/WLASL/branch_inputs/regions/rtmw_l_union")
DEFAULT_PACKAGE_ROOT = Path("packaging_outputs")
DEFAULT_PACKAGE_NAME = "wlasl-nslt1000-regions-rtmw-l-incremental"
DEFAULT_PREPROCESS_CONFIG = Path("configs/preprocessing/regions/region_crops_nslt1000.yaml")
FEASIBILITY_REPORT_PATH = Path("reports/current/regions/regions_nslt1000_incremental_feasibility_report.md")
FEASIBILITY_SCRIPT_PATH = Path("scripts/verify/check_regions_nslt1000_incremental_feasibility.py")


REQUIRED_MISSING_COLUMNS = (
    "sample_id",
    "video_id",
    "class_id",
    "gloss",
    "split",
    "source_frames_path",
    "pose_path",
    "expected_tensor_path",
    "status",
    "needs_extraction",
)

UNION_MANIFEST_COLUMNS = (
    "sample_id",
    "video_id",
    "class_id",
    "gloss",
    "split",
    "tensor_path",
    "tensor_shape",
    "status",
    "region_order",
    "reuse_source",
    "source_subset",
    "needs_extraction",
)

LOGICAL_MANIFEST_COLUMNS = (
    "sample_id",
    "video_id",
    "class_id",
    "gloss",
    "split",
    "tensor_source",
    "tensor_relpath",
    "tensor_shape",
    "region_order",
    "status",
)


@dataclass(frozen=True)
class TensorCheckResult:
    """One tensor validation result."""

    exists: bool
    valid: bool
    resolved_path: str
    shape: list[int]
    region_order: list[str]
    size_bytes: int
    error: str


def parse_csv_list(value: str | None) -> list[str]:
    """Parse a comma-separated string."""

    if value is None:
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def parse_shape(value: str | list[int] | tuple[int, ...] | None) -> tuple[int, ...]:
    """Parse one shape-like value."""

    if value is None:
        return DEFAULT_EXPECTED_SHAPE
    if isinstance(value, (list, tuple)):
        shape = tuple(int(item) for item in value)
    else:
        shape = tuple(int(part.strip()) for part in str(value).split(",") if part.strip())
    if len(shape) != 5:
        raise ValueError(f"Expected a 5D shape, got {shape}.")
    return shape


def format_size(num_bytes: float | int) -> str:
    """Format a byte count for reports."""

    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def safe_str(value: Any, default: str = "") -> str:
    """Convert nullable values to stable strings."""

    if value is None:
        return default
    try:
        is_na = pd.isna(value)
    except TypeError:
        is_na = False
    if isinstance(is_na, (bool, np.bool_)) and is_na:
        return default
    return str(value).strip()


def normalize_split(value: Any) -> str:
    """Normalize split values."""

    return safe_str(value).lower()


def normalize_sample_id(value: object, width: int = 5) -> str:
    """Normalize one numeric sample ID while preserving leading-zero semantics."""

    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    if not text:
        raise ValueError("sample_id must not be empty")
    if not text.isdigit():
        raise ValueError(f"Invalid sample_id: {value!r}")
    return text.zfill(width)


def normalize_sample_id_column(
    frame: pd.DataFrame,
    *,
    frame_name: str,
    column: str = "sample_id",
    width: int = 5,
) -> pd.DataFrame:
    """Normalize one manifest sample-id column and reject duplicates."""

    if column not in frame.columns:
        raise ValueError(f"{frame_name} is missing required column {column!r}.")

    working = frame.copy()
    try:
        working[column] = working[column].map(lambda value: normalize_sample_id(value, width=width))
    except ValueError as exc:
        raise ValueError(f"Invalid sample_id in {frame_name}: {exc}") from exc

    duplicates = working[column].duplicated(keep=False)
    if duplicates.any():
        duplicate_values = working.loc[duplicates, column].drop_duplicates().tolist()
        raise ValueError(
            f"Duplicate sample_id values after normalization in {frame_name}: {duplicate_values[:20]}"
        )
    return working


def normalize_manifest(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize common manifest columns."""

    working = frame.copy()
    for column in (
        "instance_uid",
        "sample_id",
        "video_id",
        "gloss",
        "split",
        "status",
        "tensor_path",
        "frames_dir",
        "pose_path",
        "preview_path",
        "crop_root",
        "error_message",
        "notes",
        "tensor_shape",
    ):
        if column in working.columns:
            working[column] = working[column].fillna("").astype(str).str.strip()
    if "split" in working.columns:
        working["split"] = working["split"].str.lower()
    if "status" in working.columns:
        working["status"] = working["status"].str.lower()
    if "class_id" in working.columns:
        working["class_id"] = pd.to_numeric(working["class_id"], errors="coerce").astype("Int64")
    return working


def load_standardized_manifest(source_root: Path, subset: str, split: str) -> pd.DataFrame:
    """Load one standardized split manifest."""

    path = source_root / "manifests" / f"{subset}_{split}.csv"
    frame = read_csv(path, dtype={"sample_id": "string", "video_id": "string", "gloss": "string", "split": "string"})
    require_columns(frame, STANDARDIZED_COLUMNS, name=f"standardized_manifest:{path.name}")
    return normalize_sample_id_column(
        normalize_manifest(frame),
        frame_name=f"standardized_manifest:{path.name}",
    )


def load_pose_manifest(source_root: Path, subset: str, split: str) -> pd.DataFrame:
    """Load one pose split manifest."""

    dataset_root = source_root.parent if source_root.name.lower() == "standardized" else source_root
    root = dataset_root / "pose" / "rtmw_l"
    candidate_paths = (
        root / "manifests" / f"{subset}_{split}.csv",
        root / subset / "manifests" / f"{subset}_{split}.csv",
    )
    for path in candidate_paths:
        if path.exists():
            frame = read_csv(
                path,
                dtype={"sample_id": "string", "video_id": "string", "gloss": "string", "split": "string"},
            )
            require_columns(frame, POSE_MANIFEST_COLUMNS, name=f"pose_manifest:{path.name}")
            return normalize_sample_id_column(
                normalize_manifest(frame),
                frame_name=f"pose_manifest:{path.name}",
            )
    raise FileNotFoundError(
        f"Could not find a pose manifest for subset={subset} split={split}. Tried: "
        + ", ".join(stringify_path(path) for path in candidate_paths)
    )


def load_regions_manifest(base_root: Path, subset: str, split: str) -> pd.DataFrame:
    """Load one regions manifest."""

    path = base_root / "manifests" / f"{subset}_{split}.csv"
    frame = read_csv(path, dtype={"sample_id": "string", "video_id": "string", "gloss": "string", "split": "string"})
    require_columns(frame, REGION_INPUT_MANIFEST_COLUMNS, name=f"regions_manifest:{path.name}")
    return normalize_sample_id_column(
        normalize_manifest(frame),
        frame_name=f"regions_manifest:{path.name}",
    )


def load_manifest_set(loader, root: Path, subset: str) -> dict[str, pd.DataFrame]:
    """Load a split -> dataframe mapping via one loader."""

    return {split: loader(root, subset, split) for split in ALLOWED_SPLITS}


def flatten_manifest_map(manifests: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Concatenate per-split manifests."""

    frames = [manifests[split].copy() for split in ALLOWED_SPLITS]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def repo_relative(path: str | Path, project_root: Path | None = None) -> str:
    """Return a stable project-relative path when possible."""

    candidate = Path(path)
    project_path = Path(project_root or Path.cwd()).resolve()
    try:
        return stringify_path(candidate.resolve().relative_to(project_path))
    except Exception:
        return stringify_path(candidate)


def resolve_standardized_frames_path(
    row: pd.Series,
    *,
    target_source_root: Path,
    subset: str,
) -> Path:
    """Resolve one standardized frames directory."""

    fallback = target_source_root / "frames" / subset / normalize_split(row.get("split")) / safe_str(row.get("sample_id"))
    if fallback.exists():
        return fallback.resolve()
    frames_dir = safe_str(row.get("frames_dir"))
    if frames_dir:
        resolved = remap_wlasl_path(
            frames_dir,
            project_root=Path.cwd(),
            dataset_root=target_source_root.parent,
        )
        if resolved.exists():
            return resolved.resolve()
    return fallback.resolve(strict=False)


def resolve_pose_tensor_path(
    row: pd.Series,
    *,
    target_source_root: Path,
    subset: str,
) -> Path:
    """Resolve one pose tensor file path."""

    pose_root = target_source_root.parent / "pose" / "rtmw_l" / "wholebody_133" / subset
    fallback = pose_root / normalize_split(row.get("split")) / f"{safe_str(row.get('sample_id'))}.npz"
    pose_path = safe_str(row.get("pose_path"))
    if pose_path:
        resolved = remap_wlasl_path(
            pose_path,
            project_root=Path.cwd(),
            dataset_root=target_source_root.parent,
        )
        if resolved.exists():
            return resolved.resolve()
    if fallback.exists():
        return fallback.resolve()
    return fallback.resolve(strict=False)


def id_variants(*values: Any) -> tuple[str, ...]:
    """Build stable key variants for cross-manifest matching."""

    variants: list[str] = []
    for value in values:
        text = safe_str(value)
        if not text:
            continue
        candidates = [text]
        if ":" in text:
            candidates.append(text.split(":")[-1].strip())
        for candidate in list(candidates):
            if candidate.isdigit():
                candidates.append(str(int(candidate)))
        for candidate in candidates:
            candidate = candidate.strip()
            if candidate and candidate not in variants:
                variants.append(candidate)
    return tuple(variants)


def build_lookup_by_sample(frame: pd.DataFrame) -> dict[str, pd.Series]:
    """Build a sample-id style lookup from one manifest."""

    lookup: dict[str, pd.Series] = {}
    for _, row in frame.iterrows():
        for key in id_variants(row.get("sample_id"), row.get("video_id"), row.get("instance_uid")):
            if key not in lookup:
                lookup[key] = row
    return lookup


def lookup_row(row: pd.Series, lookup: dict[str, pd.Series]) -> pd.Series | None:
    """Resolve one matching row from a lookup."""

    for key in id_variants(row.get("sample_id"), row.get("video_id"), row.get("instance_uid")):
        found = lookup.get(key)
        if found is not None:
            return found
    return None


def tensor_check(
    path_text: str | Path,
    *,
    expected_shape: tuple[int, ...],
    active_regions: list[str],
    project_root: Path | None = None,
    data_root: Path | None = None,
) -> TensorCheckResult:
    """Validate one tensor path and payload."""

    try:
        resolved = resolve_region_tensor_path(
            path_text,
            project_root=project_root or Path.cwd(),
            data_root=data_root,
        )
    except Exception as exc:
        return TensorCheckResult(
            exists=False,
            valid=False,
            resolved_path="",
            shape=[],
            region_order=[],
            size_bytes=0,
            error=str(exc),
        )

    if not resolved.exists():
        return TensorCheckResult(
            exists=False,
            valid=False,
            resolved_path=stringify_path(resolved),
            shape=[],
            region_order=[],
            size_bytes=0,
            error="tensor_path_not_found",
        )

    try:
        with np.load(resolved, allow_pickle=False) as payload:
            if "data" not in payload:
                raise KeyError("missing_data_key")
            data = np.asarray(payload["data"])
            region_order = payload["region_names"].tolist() if "region_names" in payload else []
    except Exception as exc:
        return TensorCheckResult(
            exists=True,
            valid=False,
            resolved_path=stringify_path(resolved),
            shape=[],
            region_order=[],
            size_bytes=int(resolved.stat().st_size),
            error=str(exc),
        )

    shape = [int(value) for value in data.shape]
    region_names = [safe_str(value) for value in region_order]
    valid = tuple(shape) == tuple(expected_shape) and region_names == list(active_regions)
    error = "" if valid else "shape_or_region_order_mismatch"
    return TensorCheckResult(
        exists=True,
        valid=valid,
        resolved_path=stringify_path(resolved),
        shape=shape,
        region_order=region_names,
        size_bytes=int(resolved.stat().st_size),
        error=error,
    )


def collect_base_rows(
    *,
    base_root: Path,
    base_subset: str,
    expected_shape: tuple[int, ...],
    active_regions: list[str],
    verify_payload: bool = False,
) -> pd.DataFrame:
    """Collect existing reusable base rows across all splits."""

    output_columns = [
        "instance_uid",
        "sample_id",
        "video_id",
        "class_id",
        "gloss",
        "split",
        "tensor_path",
        "resolved_tensor_path",
        "tensor_shape",
        "status",
        "base_row_valid",
        "tensor_size_bytes",
        "validation_error",
    ]
    rows: list[dict[str, Any]] = []
    for split in ALLOWED_SPLITS:
        frame = load_regions_manifest(base_root, base_subset, split)
        frame = frame[(frame["status"] == "ok") & (frame["tensor_path"] != "")].copy()
        for _, row in frame.iterrows():
            if verify_payload:
                check = tensor_check(
                    row["tensor_path"],
                    expected_shape=expected_shape,
                    active_regions=active_regions,
                    project_root=Path.cwd(),
                    data_root=base_root,
                )
                valid = check.valid
                resolved_path = check.resolved_path
                size_bytes = check.size_bytes
                error = check.error
                shape = check.shape or json.loads(row.get("tensor_shape", "[]") or "[]")
            else:
                resolved_path = ""
                try:
                    resolved = resolve_region_tensor_path(row["tensor_path"], project_root=Path.cwd(), data_root=base_root)
                    resolved_path = stringify_path(resolved)
                    valid = resolved.exists()
                    size_bytes = int(resolved.stat().st_size) if resolved.exists() else 0
                except Exception as exc:
                    valid = False
                    size_bytes = 0
                    error = str(exc)
                else:
                    error = ""
                try:
                    shape = json.loads(row.get("tensor_shape", "[]") or "[]")
                except json.JSONDecodeError:
                    shape = []
            rows.append(
                {
                    "instance_uid": safe_str(row.get("instance_uid")),
                    "sample_id": safe_str(row.get("sample_id")),
                    "video_id": safe_str(row.get("video_id")),
                    "class_id": int(row.get("class_id")),
                    "gloss": safe_str(row.get("gloss")),
                    "split": safe_str(row.get("split")),
                    "tensor_path": safe_str(row.get("tensor_path")),
                    "resolved_tensor_path": resolved_path,
                    "tensor_shape": json.dumps(shape),
                    "status": safe_str(row.get("status")),
                    "base_row_valid": bool(valid),
                    "tensor_size_bytes": int(size_bytes),
                    "validation_error": error,
                }
    )
    output = pd.DataFrame(rows, columns=output_columns)
    if output.empty:
        return output
    output = normalize_manifest(output)
    output = normalize_sample_id_column(output, frame_name=f"regions_subset:{base_subset}")
    output["base_row_valid"] = output["base_row_valid"].astype(bool)
    return output


def build_overlap_frames(
    *,
    base_root: Path,
    target_source_root: Path,
    base_subset: str,
    target_subset: str,
    expected_shape: tuple[int, ...],
    active_regions: list[str],
    verify_base_payload: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return target, base, and joined overlap frames."""

    target_map = load_manifest_set(load_standardized_manifest, target_source_root, target_subset)
    base_frame = collect_base_rows(
        base_root=base_root,
        base_subset=base_subset,
        expected_shape=expected_shape,
        active_regions=active_regions,
        verify_payload=verify_base_payload,
    )
    target_frame = flatten_manifest_map(target_map)
    target_frame = target_frame[target_frame["status"].isin(["", "ok"])].copy()
    target_frame = normalize_sample_id_column(
        target_frame,
        frame_name=f"standardized_subset:{target_subset}",
    )
    if not base_frame.empty:
        base_frame = normalize_sample_id_column(
            base_frame,
            frame_name=f"regions_subset:{base_subset}",
        )

    compare = target_frame[
        ["instance_uid", "sample_id", "video_id", "class_id", "gloss", "split", "status"]
    ].merge(
        base_frame[
            [
                "sample_id",
                "video_id",
                "class_id",
                "gloss",
                "split",
                "tensor_path",
                "resolved_tensor_path",
                "tensor_shape",
                "base_row_valid",
            ]
        ],
        on="sample_id",
        how="left",
        suffixes=("_target", "_base"),
    )
    compare["is_overlap"] = compare["tensor_path"].fillna("").astype(str) != ""
    compare["base_row_valid"] = compare["base_row_valid"].astype("boolean").fillna(False)
    compare["reusable_from_base"] = compare["is_overlap"] & compare["base_row_valid"].astype(bool)
    compare["same_split"] = compare["split_target"].fillna("") == compare["split_base"].fillna("")
    compare["same_class_id"] = compare["class_id_target"].astype("Int64") == compare["class_id_base"].astype("Int64")
    compare["same_gloss"] = compare["gloss_target"].fillna("") == compare["gloss_base"].fillna("")
    return target_frame, base_frame, compare


def summarize_counts(compare: pd.DataFrame) -> dict[str, Any]:
    """Summarize overlap and missing counts."""

    summary: dict[str, Any] = {
        "total_target_rows": int(len(compare)),
        "reused_base_rows": int(compare["reusable_from_base"].sum()),
        "missing_rows": int((~compare["reusable_from_base"]).sum()),
        "split_mismatch_count": int((compare["is_overlap"] & ~compare["same_split"]).sum()),
        "class_id_mismatch_count": int((compare["is_overlap"] & ~compare["same_class_id"]).sum()),
        "gloss_mismatch_count": int((compare["is_overlap"] & ~compare["same_gloss"]).sum()),
        "per_split": {},
    }
    for split in ALLOWED_SPLITS:
        split_frame = compare[compare["split_target"] == split]
        summary["per_split"][split] = {
            "target_rows": int(len(split_frame)),
            "reused_base_rows": int(split_frame["reusable_from_base"].sum()),
            "missing_rows": int((~split_frame["reusable_from_base"]).sum()),
        }
    return summary


def estimate_size_from_base(base_frame: pd.DataFrame, missing_rows: int) -> dict[str, Any]:
    """Estimate missing-only disk usage from current base tensors."""

    valid_sizes = pd.to_numeric(base_frame.loc[base_frame["base_row_valid"], "tensor_size_bytes"], errors="coerce").fillna(0.0)
    mean_size = float(valid_sizes.mean()) if not valid_sizes.empty else 0.0
    reused_rows = int(base_frame["base_row_valid"].sum())
    return {
        "mean_tensor_size_bytes": mean_size,
        "mean_tensor_size_human": format_size(mean_size),
        "estimated_missing_only_bytes": float(mean_size * max(missing_rows, 0)),
        "estimated_missing_only_human": format_size(mean_size * max(missing_rows, 0)),
        "estimated_reused_bytes": float(mean_size * max(reused_rows, 0)),
        "estimated_reused_human": format_size(mean_size * max(reused_rows, 0)),
    }


def get_free_disk_bytes(path: Path) -> int:
    """Return free disk bytes for one path."""

    usage = shutil.disk_usage(path.resolve())
    return int(usage.free)


def render_markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    """Render a simple markdown table."""

    output = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        output.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(output)


def ensure_incremental_layout(root: Path) -> dict[str, Path]:
    """Create the incremental folder layout when needed."""

    return {
        "root": ensure_dir(root),
        "manifests": ensure_dir(root / "manifests"),
        "tensors": ensure_dir(root / "tensors" / DEFAULT_TARGET_SUBSET),
        "reports": ensure_dir(root / "reports"),
        "state": ensure_dir(root / "state"),
    }


def state_file_path(incremental_root: Path, split: str) -> Path:
    """Return the per-split state JSON path."""

    return incremental_root / "state" / f"{DEFAULT_TARGET_SUBSET}_{split}_state.json"


def load_state(path: Path) -> dict[str, Any]:
    """Load one state file if it exists."""

    if not path.exists():
        return {"split": path.stem, "samples": {}, "updated_at": ""}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"State payload must be a JSON object: {path}")
    payload.setdefault("samples", {})
    return payload


def save_state(path: Path, payload: dict[str, Any]) -> None:
    """Persist one state JSON file."""

    write_json(payload, path, indent=2)


def save_manifest(frame: pd.DataFrame, path: Path, required_columns: tuple[str, ...] = REQUIRED_MISSING_COLUMNS) -> None:
    """Persist one manifest with a stable column order."""

    ordered = frame.copy()
    for column in required_columns:
        if column not in ordered.columns:
            ordered[column] = ""
    write_dataframe_csv(ordered.loc[:, list(required_columns) + [column for column in ordered.columns if column not in required_columns]], path)


def save_report_pair(summary: dict[str, Any], report_text: str, *, summary_path: Path, report_path: Path) -> None:
    """Write one JSON/Markdown report pair."""

    write_json(summary, summary_path, indent=2)
    write_text(report_text, report_path)


def loader_preview_pass(
    manifest_path: Path,
    *,
    expected_shape: tuple[int, ...],
    active_regions: list[str],
    num_classes: int,
    limit: int = 2,
) -> dict[str, Any]:
    """Run a tiny loader sanity check on one manifest."""

    dataset = RegionClipDataset(
        manifest_path=manifest_path,
        project_root=Path.cwd(),
        split=None,
        expected_shape=expected_shape,
        num_classes=num_classes,
        region_order=DEFAULT_ACTIVE_REGIONS,
        active_regions=active_regions,
        return_metadata=True,
        strict_shape_check=True,
        limit=limit,
    )
    batch = region_collate_fn([dataset[index] for index in range(min(len(dataset), max(1, limit)))])
    return {
        "ok": True,
        "rows_loaded": int(len(dataset)),
        "batch_shape": [int(value) for value in batch["data"].shape],
    }


def incomplete_sample_ids(frame: pd.DataFrame) -> list[str]:
    """Return sample IDs that are not yet marked ok."""

    statuses = frame["status"].fillna("").astype(str).str.lower()
    return frame.loc[statuses != "ok", "sample_id"].astype(str).tolist()


def determine_chunk(
    frame: pd.DataFrame,
    *,
    chunk_size: int | None,
    chunk_index: int | None,
    start_index: int,
    max_samples: int | None,
    next_chunk: bool,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Select the active chunk from one manifest frame."""

    working = frame.reset_index(drop=True).copy()
    total_rows = len(working)
    if start_index < 0:
        raise ValueError("start-index must be non-negative.")
    if start_index >= total_rows and total_rows > 0:
        raise ValueError(f"start-index {start_index} is outside the manifest row range 0..{total_rows - 1}.")

    selected_chunk_index = chunk_index if chunk_index is not None else 0
    if chunk_size is not None and chunk_size <= 0:
        raise ValueError("chunk-size must be positive.")

    if next_chunk and chunk_size is not None:
        statuses = working["status"].fillna("").astype(str).str.lower()
        pending_positions = working.index[statuses != "ok"].tolist()
        if pending_positions:
            selected_chunk_index = pending_positions[0] // chunk_size
        else:
            selected_chunk_index = 0

    chunk_start = start_index
    chunk_end = total_rows
    if chunk_size is not None:
        chunk_start = selected_chunk_index * chunk_size
        chunk_end = min(chunk_start + chunk_size, total_rows)
    selected = working.iloc[chunk_start:chunk_end].copy()
    if max_samples is not None:
        selected = selected.head(int(max_samples)).copy()
    meta = {
        "total_rows": int(total_rows),
        "chunk_index": int(selected_chunk_index),
        "chunk_start": int(chunk_start),
        "chunk_end": int(chunk_start + len(selected)),
    }
    return selected.reset_index(drop=True), meta
