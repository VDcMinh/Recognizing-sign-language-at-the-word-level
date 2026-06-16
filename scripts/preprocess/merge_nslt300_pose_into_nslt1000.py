from __future__ import annotations

import argparse
import shutil
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from slr.data.manifests import POSE_MANIFEST_COLUMNS, STANDARDIZED_COLUMNS
from slr.data.validation import validate_manifest_schema
from slr.utils.io import ensure_dir, write_dataframe_csv


KEY_PRIORITY = ("instance_uid", "sample_id", "video_id")
ALLOWED_SPLITS = ("train", "val", "test")
DONE_STATUSES = {"ok", "success"}
DEFAULT_STANDARDIZED_ROOT = Path("data/datasets/WLASL/standardized")
DEFAULT_POSE_ROOT = Path("data/datasets/WLASL/pose/rtmw_l")
DEFAULT_DONE_SUBSET = "nslt300"
DEFAULT_TARGET_SUBSET = "nslt1000"
DEFAULT_LAYOUT = "wholebody_133"
DEFAULT_BACKEND = "rtmw_l"

POSE_METRIC_COLUMNS = [
    "num_frames_input",
    "num_frames_pose",
    "image_height",
    "image_width",
    "mean_confidence",
    "body_mean_confidence",
    "face_mean_confidence",
    "left_hand_mean_confidence",
    "right_hand_mean_confidence",
    "valid_frames",
    "valid_frames_ratio",
    "missing_frames",
]


class MergeError(RuntimeError):
    """Raised when the merge inputs or outputs are invalid."""


@dataclass(frozen=True)
class PreparedPoseRow:
    """Resolved pose metadata for one manifest row."""

    source_name: str
    subset: str
    split: str
    row: dict[str, Any]
    local_pose_path: Path
    matched_key_name: str | None = None
    matched_key_value: str | None = None


@dataclass(frozen=True)
class CopyPlan:
    """One copy action from the done subset into the target subset."""

    source_name: str
    matched_key_name: str
    matched_key_value: str
    source_path: Path
    destination_path: Path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Merge pose extracted for an already-done subset into a target subset "
            "and rebuild the target pose manifests from the full standardized manifest."
        )
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument(
        "--standardized-root",
        type=Path,
        default=DEFAULT_STANDARDIZED_ROOT,
    )
    parser.add_argument(
        "--pose-root",
        type=Path,
        default=DEFAULT_POSE_ROOT,
    )
    parser.add_argument("--done-subset", type=str, default=DEFAULT_DONE_SUBSET)
    parser.add_argument("--target-subset", type=str, default=DEFAULT_TARGET_SUBSET)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _safe_text(value: Any) -> str:
    """Convert nullable values to normalized text."""

    if value is None:
        return ""
    try:
        is_na = pd.isna(value)
    except TypeError:
        is_na = False
    try:
        is_missing = bool(is_na)
    except (TypeError, ValueError):
        is_missing = False
    if is_missing:
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def _normalize_status(value: Any) -> str:
    """Normalize a manifest status string."""

    return _safe_text(value).lower()


def _path_text(path: Path | None) -> str:
    """Render a path with stable separators."""

    if path is None:
        return ""
    return path.as_posix()


def _resolve_path(project_root: Path, value: Path) -> Path:
    """Resolve a path relative to the project root when needed."""

    return value.resolve() if value.is_absolute() else (project_root / value).resolve()


def _require_exists(path: Path, label: str) -> None:
    """Raise when a required path is missing."""

    if not path.exists():
        raise MergeError(f"Missing {label}: {_path_text(path)}")


def _read_csv(path: Path, required_columns: list[str], label: str) -> pd.DataFrame:
    """Read and validate a CSV file."""

    _require_exists(path, label)
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        raise MergeError(f"Failed to read {label}: {_path_text(path)} ({exc})") from exc
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise MergeError(f"{label} is missing required columns: {missing}")
    return frame.copy()


def _validate_split_values(frame: pd.DataFrame, label: str) -> None:
    """Ensure every split value is supported."""

    invalid = sorted(
        {
            _safe_text(value).lower()
            for value in frame["split"].tolist()
            if _safe_text(value) and _safe_text(value).lower() not in ALLOWED_SPLITS
        }
    )
    if invalid:
        raise MergeError(f"{label} has invalid split values: {invalid}")


def _manifest_candidates(pose_root: Path, subset: str, split_name: str) -> list[Path]:
    """Return supported manifest path candidates for one subset/split."""

    filename = f"{subset}_{split_name}.csv"
    return [
        pose_root / "manifests" / subset / filename,
        pose_root / "manifests" / filename,
        pose_root / subset / "manifests" / filename,
    ]


def _resolve_existing_pose_manifest_path(pose_root: Path, subset: str, split_name: str) -> Path:
    """Find the first existing pose manifest path for one subset/split."""

    for candidate in _manifest_candidates(pose_root, subset, split_name):
        if candidate.exists():
            return candidate
    attempted = ", ".join(_path_text(path) for path in _manifest_candidates(pose_root, subset, split_name))
    raise MergeError(
        f"Could not find pose manifest for subset={subset!r}, split={split_name!r}. "
        f"Checked: {attempted}"
    )


def _output_manifest_paths(
    pose_root: Path,
    subset: str,
    split_name: str,
    existing_manifest_path: Path | None = None,
) -> list[Path]:
    """Return every manifest path that should stay in sync after writing."""

    filename = f"{subset}_{split_name}.csv"
    candidates = [
        pose_root / "manifests" / filename,
        pose_root / "manifests" / subset / filename,
    ]
    if existing_manifest_path is not None:
        candidates.append(existing_manifest_path)

    resolved: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        normalized = candidate.resolve()
        if normalized in seen:
            continue
        seen.add(normalized)
        resolved.append(normalized)
    return resolved


def _backup_path(path: Path) -> Path:
    """Return a timestamped backup path for an existing manifest."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return path.with_name(f"{path.stem}.before_merge_remaining_only_{timestamp}{path.suffix}")


def _backup_existing_manifests(paths: list[Path], dry_run: bool) -> list[tuple[Path, Path]]:
    """Backup manifests before overwrite."""

    backups: list[tuple[Path, Path]] = []
    for path in paths:
        if not path.exists():
            continue
        backup = _backup_path(path)
        backups.append((path, backup))
        if dry_run:
            continue
        ensure_dir(backup.parent)
        shutil.copy2(path, backup)
    return backups


def _normalize_key_series(frame: pd.DataFrame, column: str) -> pd.Series:
    """Return a normalized string series for one key column."""

    return frame[column].map(_safe_text)


def _validate_unique_keys(frame: pd.DataFrame, label: str) -> None:
    """Ensure stable-key columns are not duplicated within a manifest."""

    for key_name in KEY_PRIORITY:
        if key_name not in frame.columns:
            continue
        values = _normalize_key_series(frame, key_name)
        non_empty = values[values != ""]
        duplicates = non_empty[non_empty.duplicated(keep=False)]
        if duplicates.empty:
            continue
        preview = sorted(set(duplicates.tolist()))[:10]
        raise MergeError(f"{label} has duplicate values in {key_name}: {preview}")


def _pose_filename_candidates(row: dict[str, Any]) -> list[str]:
    """Return candidate pose filenames in priority order."""

    names: list[str] = []
    pose_path_text = _safe_text(row.get("pose_path"))
    if pose_path_text:
        basename = Path(pose_path_text).name.strip()
        if basename:
            names.append(basename)

    for key_name in ("sample_id", "video_id"):
        key_value = _safe_text(row.get(key_name))
        if key_value:
            names.append(f"{key_value}.npz")

    deduped: list[str] = []
    seen: set[str] = set()
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        deduped.append(name)
    return deduped


def _resolve_local_pose_path(pose_root: Path, subset: str, row: dict[str, Any]) -> Path:
    """Resolve the local pose path for one manifest row."""

    split = _safe_text(row.get("split")).lower()
    if split not in ALLOWED_SPLITS:
        raise MergeError(f"Unsupported split {split!r} in subset {subset!r}.")

    direct_path = Path(_safe_text(row.get("pose_path")))
    if direct_path.is_absolute() and direct_path.exists():
        return direct_path.resolve()

    split_root = pose_root / DEFAULT_LAYOUT / subset / split
    candidates = _pose_filename_candidates(row)
    if not candidates:
        raise MergeError(f"Could not infer pose filename for subset {subset!r}: {row}")

    for filename in candidates:
        candidate = split_root / filename
        if candidate.exists():
            return candidate.resolve()
    return (split_root / candidates[0]).resolve()


def _build_source_lookup(
    frame: pd.DataFrame,
    pose_root: Path,
    subset: str,
    source_name: str,
) -> tuple[dict[str, dict[str, PreparedPoseRow]], pd.DataFrame]:
    """Build per-key lookups from rows whose source manifest reports usable pose."""

    eligible = frame[frame["status"].map(_normalize_status).isin(DONE_STATUSES)].copy()
    _validate_unique_keys(eligible, f"{source_name} pose manifest")

    lookup: dict[str, dict[str, PreparedPoseRow]] = {key_name: {} for key_name in KEY_PRIORITY}
    prepared_rows: list[PreparedPoseRow] = []

    for row_dict in eligible.to_dict(orient="records"):
        split = _safe_text(row_dict.get("split")).lower()
        if split not in ALLOWED_SPLITS:
            raise MergeError(f"{source_name} pose manifest has invalid split: {split!r}")
        prepared = PreparedPoseRow(
            source_name=source_name,
            subset=subset,
            split=split,
            row=row_dict,
            local_pose_path=_resolve_local_pose_path(pose_root, subset, row_dict),
        )
        prepared_rows.append(prepared)
        for key_name in KEY_PRIORITY:
            key_value = _safe_text(row_dict.get(key_name))
            if not key_value:
                continue
            lookup[key_name][key_value] = prepared

    prepared_frame = pd.DataFrame([item.row for item in prepared_rows])
    return lookup, prepared_frame


def _expected_target_pose_path(
    pose_root: Path,
    target_subset: str,
    split: str,
    target_row: pd.Series,
    matched_row: PreparedPoseRow | None,
) -> Path:
    """Resolve the destination pose path inside the target subset."""

    filename_stem = _safe_text(target_row.get("sample_id")) or _safe_text(target_row.get("video_id"))
    if not filename_stem and matched_row is not None:
        filename_stem = matched_row.local_pose_path.stem
    if not filename_stem:
        raise MergeError(f"Could not infer target pose filename for row: {target_row.to_dict()}")
    return (pose_root / DEFAULT_LAYOUT / target_subset / split / f"{filename_stem}.npz").resolve()


def _match_pose_row(
    target_row: pd.Series,
    remaining_lookup: dict[str, dict[str, PreparedPoseRow]],
    done_lookup: dict[str, dict[str, PreparedPoseRow]],
) -> PreparedPoseRow | None:
    """Match one target standardized row against remaining then done pose sources."""

    for key_name in KEY_PRIORITY:
        key_value = _safe_text(target_row.get(key_name))
        if not key_value:
            continue
        remaining = remaining_lookup[key_name].get(key_value)
        if remaining is not None:
            return PreparedPoseRow(
                source_name=remaining.source_name,
                subset=remaining.subset,
                split=remaining.split,
                row=remaining.row,
                local_pose_path=remaining.local_pose_path,
                matched_key_name=key_name,
                matched_key_value=key_value,
            )
        done = done_lookup[key_name].get(key_value)
        if done is not None:
            return PreparedPoseRow(
                source_name=done.source_name,
                subset=done.subset,
                split=done.split,
                row=done.row,
                local_pose_path=done.local_pose_path,
                matched_key_name=key_name,
                matched_key_value=key_value,
            )
    return None


def _join_notes(*values: Any) -> str:
    """Join semicolon-delimited note fragments without duplicates."""

    parts: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _safe_text(value)
        if not text:
            continue
        for item in text.split(";"):
            note = item.strip()
            if not note or note in seen:
                continue
            seen.add(note)
            parts.append(note)
    return ";".join(parts)


def _source_missing_reason(
    matched_row: PreparedPoseRow | None,
    destination_path: Path,
) -> str:
    """Describe why a pose row is still missing."""

    if matched_row is None:
        return "No pose row matched in target remaining or done subset manifests."
    if matched_row.source_name == "target_remaining_pose":
        return (
            "Matched target remaining pose manifest row, but target pose file is missing: "
            f"{_path_text(destination_path)}"
        )
    return (
        f"Matched {matched_row.source_name} row, but source pose file is missing: "
        f"{_path_text(matched_row.local_pose_path)}"
    )


def _build_final_manifest(
    target_standardized: pd.DataFrame,
    pose_root: Path,
    target_subset: str,
    remaining_lookup: dict[str, dict[str, PreparedPoseRow]],
    done_lookup: dict[str, dict[str, PreparedPoseRow]],
    overwrite: bool,
    dry_run: bool,
) -> tuple[pd.DataFrame, list[CopyPlan], dict[str, Any]]:
    """Build the final target pose manifest and the copy plan."""

    output_rows: list[dict[str, Any]] = []
    copy_plans: list[CopyPlan] = []

    matched_source_counts: Counter[str] = Counter()
    matched_key_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    copied_candidates = 0
    missing_source_files = 0
    skipped_existing_files = 0

    for row in target_standardized.to_dict(orient="records"):
        target_row = pd.Series(row)
        split = _safe_text(target_row.get("split")).lower()
        if split not in ALLOWED_SPLITS:
            raise MergeError(f"Target standardized manifest has invalid split: {split!r}")

        matched = _match_pose_row(target_row, remaining_lookup, done_lookup)
        destination_path = _expected_target_pose_path(
            pose_root=pose_root,
            target_subset=target_subset,
            split=split,
            target_row=target_row,
            matched_row=matched,
        )

        copied_from_done_subset = False
        planned_ok = destination_path.exists()
        actual_pose_path = destination_path

        if matched is not None:
            matched_source_counts[matched.source_name] += 1
            if matched.matched_key_name:
                matched_key_counts[matched.matched_key_name] += 1

            if matched.source_name == "target_remaining_pose":
                actual_pose_path = matched.local_pose_path
                planned_ok = matched.local_pose_path.exists()
            else:
                if matched.local_pose_path.exists():
                    copied_candidates += 1
                    copied_from_done_subset = True
                    if destination_path.exists() and not overwrite:
                        skipped_existing_files += 1
                    else:
                        copy_plans.append(
                            CopyPlan(
                                source_name=matched.source_name,
                                matched_key_name=matched.matched_key_name or "",
                                matched_key_value=matched.matched_key_value or "",
                                source_path=matched.local_pose_path,
                                destination_path=destination_path,
                            )
                        )
                    planned_ok = True
                else:
                    if not destination_path.exists():
                        missing_source_files += 1

        status = "ok" if planned_ok else "missing_pose"
        if dry_run and copied_from_done_subset:
            actual_pose_path = destination_path
        elif matched is not None and matched.source_name != "target_remaining_pose":
            actual_pose_path = destination_path

        notes = _join_notes(
            matched.row.get("notes") if matched is not None else "",
            f"merged_from_{matched.source_name}" if matched is not None else "",
            "copied_from_done_subset" if copied_from_done_subset else "",
        )

        output_row = {
            "instance_uid": target_row.get("instance_uid"),
            "sample_id": target_row.get("sample_id"),
            "video_id": target_row.get("video_id"),
            "gloss": target_row.get("gloss"),
            "class_id": target_row.get("class_id"),
            "split": target_row.get("split"),
            "frames_dir": target_row.get("frames_dir"),
            "pose_path": _path_text(actual_pose_path) if status == "ok" else "",
            "keypoint_layout": matched.row.get("keypoint_layout") if matched is not None else DEFAULT_LAYOUT,
            "pose_backend": matched.row.get("pose_backend") if matched is not None else DEFAULT_BACKEND,
            "status": status,
            "error_message": "" if status == "ok" else _source_missing_reason(matched, destination_path),
            "notes": notes,
        }

        for metric_name in POSE_METRIC_COLUMNS:
            output_row[metric_name] = matched.row.get(metric_name) if matched is not None else None

        output_rows.append(output_row)
        split_counts[split] += 1

    final_frame = pd.DataFrame(output_rows)
    final_frame = validate_manifest_schema(final_frame, POSE_MANIFEST_COLUMNS, name="final target pose manifest")
    summary = {
        "total_target_rows": int(len(target_standardized)),
        "rows_from_target_remaining_pose": int(matched_source_counts["target_remaining_pose"]),
        "rows_from_done_subset_pose": int(matched_source_counts["done_subset_pose"]),
        "rows_merged_total": int(
            matched_source_counts["target_remaining_pose"] + matched_source_counts["done_subset_pose"]
        ),
        "missing_pose_rows": int((final_frame["status"] != "ok").sum()),
        "copy_candidates_from_done_subset": int(copied_candidates),
        "copied_files_planned": int(len(copy_plans)),
        "skipped_existing_files": int(skipped_existing_files),
        "missing_source_files": int(missing_source_files),
        "matched_rows_by_key": dict(matched_key_counts),
        "final_split_counts": dict(split_counts),
    }
    return final_frame, copy_plans, summary


def _execute_copy_plan(copy_plans: list[CopyPlan], overwrite: bool, dry_run: bool) -> dict[str, int]:
    """Copy pose files from the done subset into the target subset."""

    copied_files = 0
    skipped_existing_files = 0
    missing_source_files = 0

    for plan in copy_plans:
        if not plan.source_path.exists():
            missing_source_files += 1
            continue
        if plan.destination_path.exists() and not overwrite:
            skipped_existing_files += 1
            continue
        if dry_run:
            copied_files += 1
            continue
        ensure_dir(plan.destination_path.parent)
        shutil.copy2(plan.source_path, plan.destination_path)
        copied_files += 1

    return {
        "copied_files": copied_files,
        "skipped_existing_files": skipped_existing_files,
        "missing_source_files": missing_source_files,
    }


def _validate_final_manifest(
    final_frame: pd.DataFrame,
    target_standardized: pd.DataFrame,
    copy_plans: list[CopyPlan],
    dry_run: bool,
) -> dict[str, Any]:
    """Validate the merged target manifest."""

    if len(final_frame) != len(target_standardized):
        raise MergeError(
            "Final nslt1000_all.csv row count does not match the full standardized manifest: "
            f"{len(final_frame)} != {len(target_standardized)}"
        )

    _validate_unique_keys(final_frame, "final target pose manifest")

    standardized_split_counts = {
        split: int((_normalize_key_series(target_standardized, "split").str.lower() == split).sum())
        for split in ALLOWED_SPLITS
    }
    final_split_counts = {
        split: int((_normalize_key_series(final_frame, "split").str.lower() == split).sum())
        for split in ALLOWED_SPLITS
    }
    if standardized_split_counts != final_split_counts:
        raise MergeError(
            "Final split counts do not match the full standardized manifest. "
            f"expected={standardized_split_counts}, actual={final_split_counts}"
        )

    missing_rows = final_frame[final_frame["status"] != "ok"].copy()
    coverage_ok = missing_rows.empty

    missing_ok_paths: list[str] = []
    if not dry_run:
        ok_rows = final_frame[final_frame["status"] == "ok"].copy()
        for row in ok_rows.to_dict(orient="records"):
            pose_path_text = _safe_text(row.get("pose_path"))
            if not pose_path_text:
                missing_ok_paths.append("<empty pose_path>")
                continue
            if not Path(pose_path_text).exists():
                missing_ok_paths.append(pose_path_text)
        if missing_ok_paths:
            preview = missing_ok_paths[:10]
            raise MergeError(
                "Some status=ok rows point to missing pose files after merge: "
                f"{preview}"
            )

    return {
        "standardized_split_counts": standardized_split_counts,
        "final_split_counts": final_split_counts,
        "missing_pose_rows": int(len(missing_rows)),
        "missing_pose_preview": missing_rows.loc[:, ["instance_uid", "sample_id", "video_id", "split", "status"]]
        .head(10)
        .to_dict(orient="records"),
        "coverage_ok": coverage_ok,
        "planned_copy_count": int(len(copy_plans)),
        "dry_run": dry_run,
    }


def _write_manifests(
    final_frame: pd.DataFrame,
    pose_root: Path,
    target_subset: str,
    existing_manifest_paths: dict[str, Path],
    dry_run: bool,
) -> list[Path]:
    """Write all split manifests to every supported target path."""

    written_paths: list[Path] = []
    frames_by_name = {
        "all": final_frame,
        "train": final_frame[_normalize_key_series(final_frame, "split").str.lower() == "train"].copy(),
        "val": final_frame[_normalize_key_series(final_frame, "split").str.lower() == "val"].copy(),
        "test": final_frame[_normalize_key_series(final_frame, "split").str.lower() == "test"].copy(),
    }

    for split_name, frame in frames_by_name.items():
        frame = validate_manifest_schema(frame, POSE_MANIFEST_COLUMNS, name=f"target pose manifest {split_name}")
        output_paths = _output_manifest_paths(
            pose_root=pose_root,
            subset=target_subset,
            split_name=split_name,
            existing_manifest_path=existing_manifest_paths.get(split_name),
        )
        _backup_existing_manifests(output_paths, dry_run=dry_run)
        for output_path in output_paths:
            written_paths.append(output_path)
            if dry_run:
                continue
            ensure_dir(output_path.parent)
            write_dataframe_csv(frame, output_path)

    return written_paths


def _print_summary(
    standardized_manifest_path: Path,
    done_manifest_path: Path,
    target_remaining_manifest_path: Path,
    copy_results: dict[str, int],
    build_summary: dict[str, Any],
    validation_summary: dict[str, Any],
    written_paths: list[Path],
    dry_run: bool,
) -> None:
    """Print the merge summary."""

    print("Merge pose summary")
    print(f"- dry_run: {dry_run}")
    print(f"- target standardized manifest: {_path_text(standardized_manifest_path)}")
    print(f"- done subset pose manifest: {_path_text(done_manifest_path)}")
    print(f"- target remaining pose manifest: {_path_text(target_remaining_manifest_path)}")
    print(f"- total target rows: {build_summary['total_target_rows']}")
    print(f"- rows from nslt1000 remaining pose: {build_summary['rows_from_target_remaining_pose']}")
    print(f"- rows from nslt300 pose: {build_summary['rows_from_done_subset_pose']}")
    print(f"- rows merged total: {build_summary['rows_merged_total']}")
    print(f"- missing pose rows: {build_summary['missing_pose_rows']}")
    print(f"- copied files: {copy_results['copied_files']}")
    print(f"- skipped existing files: {build_summary['skipped_existing_files']}")
    print(f"- missing source files: {build_summary['missing_source_files']}")
    print(f"- matched rows by key: {build_summary['matched_rows_by_key']}")
    print(f"- standardized split counts: {validation_summary['standardized_split_counts']}")
    print(f"- final split counts: {validation_summary['final_split_counts']}")
    print(f"- coverage ok: {validation_summary['coverage_ok']}")
    if validation_summary["missing_pose_preview"]:
        print(f"- missing pose preview: {validation_summary['missing_pose_preview']}")
    print("- output manifests:")
    for path in written_paths:
        print(f"  {_path_text(path)}")


def main() -> None:
    """Run the merge workflow."""

    args = parse_args()

    project_root = _resolve_path(Path.cwd(), args.project_root)
    standardized_root = _resolve_path(project_root, args.standardized_root)
    pose_root = _resolve_path(project_root, args.pose_root)

    _require_exists(project_root, "project root")
    _require_exists(standardized_root, "standardized root")
    _require_exists(pose_root, "pose root")

    standardized_manifest_path = standardized_root / "manifests" / f"{args.target_subset}_all.csv"
    done_manifest_path = _resolve_existing_pose_manifest_path(pose_root, args.done_subset, "all")
    target_remaining_manifest_path = _resolve_existing_pose_manifest_path(pose_root, args.target_subset, "all")

    target_standardized = _read_csv(
        standardized_manifest_path,
        required_columns=STANDARDIZED_COLUMNS,
        label=f"standardized manifest for {args.target_subset}",
    )
    done_pose = _read_csv(
        done_manifest_path,
        required_columns=POSE_MANIFEST_COLUMNS,
        label=f"pose manifest for {args.done_subset}",
    )
    target_remaining_pose = _read_csv(
        target_remaining_manifest_path,
        required_columns=POSE_MANIFEST_COLUMNS,
        label=f"pose manifest for {args.target_subset}",
    )

    _validate_split_values(target_standardized, f"{args.target_subset} standardized manifest")
    _validate_split_values(done_pose, f"{args.done_subset} pose manifest")
    _validate_split_values(target_remaining_pose, f"{args.target_subset} pose manifest")

    _validate_unique_keys(target_standardized, f"{args.target_subset} standardized manifest")

    remaining_lookup, _ = _build_source_lookup(
        frame=target_remaining_pose,
        pose_root=pose_root,
        subset=args.target_subset,
        source_name="target_remaining_pose",
    )
    done_lookup, _ = _build_source_lookup(
        frame=done_pose,
        pose_root=pose_root,
        subset=args.done_subset,
        source_name="done_subset_pose",
    )

    final_frame, copy_plans, build_summary = _build_final_manifest(
        target_standardized=target_standardized,
        pose_root=pose_root,
        target_subset=args.target_subset,
        remaining_lookup=remaining_lookup,
        done_lookup=done_lookup,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )

    copy_results = _execute_copy_plan(
        copy_plans=copy_plans,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    build_summary["copied_files_actual"] = copy_results["copied_files"]
    build_summary["skipped_existing_files_actual"] = copy_results["skipped_existing_files"]
    build_summary["missing_source_files_actual"] = copy_results["missing_source_files"]

    validation_summary = _validate_final_manifest(
        final_frame=final_frame,
        target_standardized=target_standardized,
        copy_plans=copy_plans,
        dry_run=args.dry_run,
    )

    existing_manifest_paths = {
        split_name: _resolve_existing_pose_manifest_path(pose_root, args.target_subset, split_name)
        for split_name in ("train", "val", "test", "all")
    }
    written_paths = _write_manifests(
        final_frame=final_frame,
        pose_root=pose_root,
        target_subset=args.target_subset,
        existing_manifest_paths=existing_manifest_paths,
        dry_run=args.dry_run,
    )

    _print_summary(
        standardized_manifest_path=standardized_manifest_path,
        done_manifest_path=done_manifest_path,
        target_remaining_manifest_path=target_remaining_manifest_path,
        copy_results=copy_results,
        build_summary=build_summary,
        validation_summary=validation_summary,
        written_paths=written_paths,
        dry_run=args.dry_run,
    )

    if not validation_summary["coverage_ok"]:
        raise MergeError(
            "Merged manifests were written with missing_pose rows. "
            "Inspect the summary and fix missing source pose files before downstream steps."
        )


if __name__ == "__main__":
    main()
