from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


KEY_PRIORITY = ("instance_uid", "sample_id", "video_id")
DONE_STATUSES = {"ok", "success"}
ALLOWED_SPLITS = ("train", "val", "test")

REPO_DIR_INCLUDES = ("configs", "scripts", "slr", "src")
REPO_FILE_INCLUDES = ("README.md", "pyproject.toml", "requirements.txt", "sitecustomize.py")
CHECKPOINT_SUFFIXES = {".pth", ".pt", ".py", ".yaml", ".yml"}
REQUIRED_BUNDLE_SOURCE_FILES = (
    Path("repo") / "src" / "slr" / "data" / "manifests.py",
    Path("repo") / "src" / "slr" / "data" / "validation.py",
    Path("repo") / "src" / "slr" / "pose" / "extract_rtmw.py",
    Path("repo") / "scripts" / "preprocess" / "02_extract_pose_rtmw.py",
)

STANDARDIZED_COLUMNS = [
    "instance_uid",
    "sample_id",
    "video_id",
    "gloss",
    "class_id",
    "split",
    "raw_video_path",
    "standardized_video_path",
    "frames_dir",
    "num_frames",
    "fps",
    "original_width",
    "original_height",
    "output_width",
    "output_height",
    "original_start_frame",
    "original_end_frame",
    "used_start_frame",
    "used_end_frame",
    "original_bbox",
    "used_bbox",
    "crop_applied",
    "bbox_fallback_used",
    "save_frames",
    "save_video",
    "status",
    "error_message",
    "notes",
]

POSE_MANIFEST_COLUMNS = [
    "instance_uid",
    "sample_id",
    "video_id",
    "gloss",
    "class_id",
    "split",
    "frames_dir",
    "pose_path",
    "keypoint_layout",
    "pose_backend",
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
    "status",
    "error_message",
    "notes",
]


class BundleError(RuntimeError):
    """Raised when bundle preparation fails."""


@dataclass(frozen=True)
class FrameCopyPlan:
    """Resolved source and destination for one remaining sample."""

    row_index: int
    split: str
    sample_folder: str
    source_dir: Path
    bundle_manifest_frames_dir: str
    bundle_output_dir: Path
    key_name: str
    key_value: str
    fallback_used: bool


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Prepare a Kaggle bundle that contains only the remaining "
            "nslt1000 standardized samples not already covered by nslt300 pose outputs."
        )
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument(
        "--standardized-root",
        type=Path,
        default=Path("data/datasets/WLASL/standardized"),
    )
    parser.add_argument(
        "--pose-done-root",
        type=Path,
        default=Path("data/datasets/WLASL/pose/rtmw_l"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("kaggle_nslt1000_remaining_bundle"),
    )
    parser.add_argument("--done-subset", type=str, default="nslt300")
    parser.add_argument("--target-subset", type=str, default="nslt1000")
    parser.add_argument("--copy-repo", action="store_true")
    parser.add_argument("--copy-checkpoints", action="store_true")
    parser.add_argument("--make-zip", action="store_true")
    parser.add_argument("--skip-missing-frames", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _path_text(path: Path) -> str:
    """Render a path with stable separators."""

    return path.as_posix()


def _safe_text(value: Any) -> str:
    """Convert manifest values to normalized text."""

    if value is None:
        return ""
    if pd.isna(value):
        return ""
    return str(value).strip()


def _normalize_status(value: Any) -> str:
    """Normalize manifest statuses."""

    return _safe_text(value).lower()


def _resolve_under(base: Path, value: Path) -> Path:
    """Resolve a path relative to base when needed."""

    return value.resolve() if value.is_absolute() else (base / value).resolve()


def _is_within(path: Path, root: Path) -> bool:
    """Return True when path is equal to or nested under root."""

    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _validate_output_root(
    output_root: Path,
    project_root: Path,
    standardized_root: Path,
    pose_done_root: Path,
) -> None:
    """Guard against unsafe output locations."""

    checkpoint_root = project_root / "checkpoints"
    if output_root == project_root:
        raise BundleError("Output root must not be the project root.")
    for label, blocked_root in (
        ("standardized root", standardized_root),
        ("pose done root", pose_done_root),
        ("checkpoint root", checkpoint_root),
    ):
        if _is_within(output_root, blocked_root):
            raise BundleError(
                f"Output root must not be inside the {label}: {_path_text(output_root)}"
            )


def _ensure_empty_output_dir(output_root: Path, overwrite: bool) -> None:
    """Create an empty output directory, or fail if it already has content."""

    if output_root.exists() and any(output_root.iterdir()):
        if not overwrite:
            raise BundleError(
                "Output directory already exists and is not empty. "
                f"Use --overwrite or choose a different path: {_path_text(output_root)}"
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)


def _require_exists(path: Path, label: str) -> None:
    """Raise if path is missing."""

    if not path.exists():
        raise BundleError(f"Required {label} does not exist: {_path_text(path)}")


def _require_columns(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    """Validate required columns exist."""

    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise BundleError(f"{name} is missing required columns: {missing}")


def _validate_split_values(frame: pd.DataFrame, column: str, name: str) -> None:
    """Validate split values are supported."""

    invalid = sorted(
        {
            _safe_text(value).lower()
            for value in frame[column].tolist()
            if _safe_text(value) and _safe_text(value).lower() not in ALLOWED_SPLITS
        }
    )
    if invalid:
        raise BundleError(f"{name} has invalid split values: {invalid}")


def _read_csv(path: Path, required_columns: list[str], name: str) -> pd.DataFrame:
    """Read and validate a CSV manifest."""

    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        raise BundleError(f"Failed to read {name}: {_path_text(path)} ({exc})") from exc
    _require_columns(frame, required_columns, name)
    return frame.copy()


def _load_target_manifest(standardized_root: Path, target_subset: str) -> pd.DataFrame:
    """Load the full standardized manifest for the target subset."""

    manifest_path = standardized_root / "manifests" / f"{target_subset}_all.csv"
    frame = _read_csv(
        manifest_path,
        STANDARDIZED_COLUMNS,
        name=f"standardized manifest for {target_subset}",
    )
    _validate_split_values(frame, "split", f"{target_subset}_all.csv")
    return frame


def _load_done_pose_manifest(pose_done_root: Path, done_subset: str) -> pd.DataFrame:
    """Load the pose manifest that marks already extracted samples."""

    manifest_path = pose_done_root / "manifests" / f"{done_subset}_all.csv"
    frame = _read_csv(
        manifest_path,
        POSE_MANIFEST_COLUMNS,
        name=f"pose manifest for {done_subset}",
    )
    _validate_split_values(frame, "split", f"{done_subset}_all.csv")
    return frame


def _build_done_lookup(done_pose_frame: pd.DataFrame) -> tuple[dict[str, set[str]], pd.DataFrame]:
    """Build done-key lookup sets from pose rows with accepted statuses."""

    filtered = done_pose_frame[
        done_pose_frame["status"].map(_normalize_status).isin(DONE_STATUSES)
    ].copy()

    lookup: dict[str, set[str]] = {}
    for field in KEY_PRIORITY:
        if field in filtered.columns:
            lookup[field] = {
                value
                for value in filtered[field].map(_safe_text).tolist()
                if value
            }
        else:
            lookup[field] = set()
    return lookup, filtered


def _compute_remaining_rows(
    target_frame: pd.DataFrame,
    done_lookup: dict[str, set[str]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Subtract already-done pose rows from the target standardized manifest."""

    covered_indices: list[int] = []
    covered_key_counts: Counter[str] = Counter()
    remaining_first_key_counts: Counter[str] = Counter()
    available_first_key_counts: Counter[str] = Counter()

    for row_index, row in target_frame.iterrows():
        first_available_key_name = ""
        is_covered = False

        for key_name in KEY_PRIORITY:
            key_value = _safe_text(row.get(key_name))
            if not key_value:
                continue
            if not first_available_key_name:
                first_available_key_name = key_name
            if key_value in done_lookup[key_name]:
                is_covered = True
                covered_key_counts[key_name] += 1
                break

        if not first_available_key_name:
            raise BundleError(
                "Target standardized manifest contains a row without any stable key "
                f"({', '.join(KEY_PRIORITY)}). Row index: {row_index}"
            )

        available_first_key_counts[first_available_key_name] += 1
        if is_covered:
            covered_indices.append(row_index)
        else:
            remaining_first_key_counts[first_available_key_name] += 1

    covered_mask = target_frame.index.isin(covered_indices)
    remaining_frame = target_frame.loc[~covered_mask].copy().reset_index(drop=True)
    stats = {
        "key_used": "instance_uid > sample_id > video_id",
        "covered_rows": int(covered_mask.sum()),
        "covered_rows_by_key": dict(covered_key_counts),
        "target_first_available_key_counts": dict(available_first_key_counts),
        "remaining_first_available_key_counts": dict(remaining_first_key_counts),
    }
    return remaining_frame, stats


def _candidate_sample_folders(row: pd.Series) -> list[tuple[str, str]]:
    """Return candidate sample folder names in priority order."""

    candidates: list[tuple[str, str]] = []
    frames_dir_text = _safe_text(row.get("frames_dir"))
    if frames_dir_text:
        folder_name = Path(frames_dir_text).name.strip()
        if folder_name:
            candidates.append((folder_name, "frames_dir"))

    for field in ("sample_id", "video_id"):
        value = _safe_text(row.get(field))
        if value:
            candidates.append((value, field))

    deduped: list[tuple[str, str]] = []
    seen: set[str] = set()
    for folder_name, source_name in candidates:
        if folder_name in seen:
            continue
        seen.add(folder_name)
        deduped.append((folder_name, source_name))
    return deduped


def _resolve_frame_copy_plan(
    row: pd.Series,
    row_index: int,
    standardized_root: Path,
    target_subset: str,
    standardized_bundle_root: Path,
) -> FrameCopyPlan:
    """Resolve source and destination frame directories for one sample."""

    split = _safe_text(row.get("split")).lower()
    if split not in ALLOWED_SPLITS:
        raise BundleError(f"Unsupported split for row {row_index}: {split!r}")

    split_root = standardized_root / "frames" / target_subset / split
    if not split_root.exists():
        raise BundleError(
            f"Expected standardized split directory does not exist: {_path_text(split_root)}"
        )

    candidates = _candidate_sample_folders(row)
    if not candidates:
        raise BundleError(f"Could not determine sample folder for row {row_index}.")

    selected_source: Path | None = None
    selected_folder = ""
    selected_key = ""
    fallback_used = False

    for folder_name, source_name in candidates:
        candidate_dir = split_root / folder_name
        if candidate_dir.exists() and candidate_dir.is_dir():
            selected_source = candidate_dir
            selected_folder = folder_name
            selected_key = source_name
            fallback_used = source_name != "frames_dir"
            break

    if selected_source is None:
        tried = ", ".join(folder_name for folder_name, _ in candidates)
        raise FileNotFoundError(
            f"Missing standardized frames directory for row {row_index} "
            f"(split={split}, tried={tried})."
        )

    bundle_frames_dir = (
        Path("data")
        / "datasets"
        / "WLASL"
        / "standardized"
        / "frames"
        / target_subset
        / split
        / selected_folder
    )
    bundle_output_dir = standardized_bundle_root / bundle_frames_dir
    key_name = selected_key if selected_key in {"sample_id", "video_id"} else "frames_dir"
    key_value = selected_folder

    return FrameCopyPlan(
        row_index=row_index,
        split=split,
        sample_folder=selected_folder,
        source_dir=selected_source,
        bundle_manifest_frames_dir=bundle_frames_dir.as_posix(),
        bundle_output_dir=bundle_output_dir,
        key_name=key_name,
        key_value=key_value,
        fallback_used=fallback_used,
    )


def _prepare_remaining_manifests_and_copy_plan(
    remaining_frame: pd.DataFrame,
    standardized_root: Path,
    target_subset: str,
    standardized_bundle_root: Path,
    skip_missing_frames: bool,
    verbose: bool,
) -> tuple[pd.DataFrame, list[FrameCopyPlan], list[str]]:
    """Rewrite remaining manifests for Kaggle and resolve all frame copy actions."""

    rewritten_rows: list[dict[str, Any]] = []
    copy_plans: list[FrameCopyPlan] = []
    warnings: list[str] = []

    for row_index, row in remaining_frame.iterrows():
        try:
            plan = _resolve_frame_copy_plan(
                row=row,
                row_index=row_index,
                standardized_root=standardized_root,
                target_subset=target_subset,
                standardized_bundle_root=standardized_bundle_root,
            )
        except FileNotFoundError as exc:
            if not skip_missing_frames:
                raise BundleError(str(exc)) from exc
            warnings.append(f"WARNING: {exc}")
            continue

        rewritten = row.to_dict()
        rewritten["frames_dir"] = plan.bundle_manifest_frames_dir
        notes = _safe_text(rewritten.get("notes"))
        note_suffix = "kaggle_bundle_frames_dir_rewritten"
        rewritten["notes"] = f"{notes};{note_suffix}" if notes else note_suffix
        rewritten_rows.append(rewritten)
        copy_plans.append(plan)

        if plan.fallback_used:
            warnings.append(
                "WARNING: "
                f"row {row_index} used fallback sample folder resolution via {plan.key_name}="
                f"{plan.key_value!r}."
            )
        elif verbose:
            print(
                "Resolved sample folder:",
                f"split={plan.split}",
                f"sample_folder={plan.sample_folder}",
                f"source={_path_text(plan.source_dir)}",
            )

    if not rewritten_rows:
        raise BundleError(
            "No remaining rows were left after frame resolution. "
            "Nothing would be bundled."
        )

    rewritten_frame = pd.DataFrame(rewritten_rows)
    rewritten_frame = rewritten_frame.loc[:, list(remaining_frame.columns)]
    return rewritten_frame.reset_index(drop=True), copy_plans, warnings


def _write_manifest_csv(frame: pd.DataFrame, path: Path) -> None:
    """Write a UTF-8 CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    """Write a UTF-8 text file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a UTF-8 JSON file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _copy_tree_contents(source_dir: Path, destination_dir: Path) -> None:
    """Copy a directory tree."""

    if destination_dir.exists():
        shutil.rmtree(destination_dir)
    shutil.copytree(source_dir, destination_dir)


def _copy_remaining_frames(copy_plans: list[FrameCopyPlan], dry_run: bool) -> None:
    """Copy standardized frames for remaining samples."""

    for plan in copy_plans:
        if dry_run:
            continue
        plan.bundle_output_dir.parent.mkdir(parents=True, exist_ok=True)
        _copy_tree_contents(plan.source_dir, plan.bundle_output_dir)


def _is_generated_repo_dir(name: str) -> bool:
    """Return True for generated helper directories."""

    return name.startswith("_") and name.endswith("_output")


def _should_skip_repo_path(relative_path: Path, blocked_top_level_parts: set[str]) -> bool:
    """Return True when a repo path should be excluded from the copied repo.

    Important:
    - artifact exclusions such as ``data/`` apply only at the repository root
    - nested source paths like ``src/slr/data/`` must stay copyable
    """

    if relative_path.parts:
        first_part = relative_path.parts[0]
        if first_part in blocked_top_level_parts:
            return True

    for part in relative_path.parts:
        if part == "__pycache__":
            return True
        if part.startswith(".venv"):
            return True
        if _is_generated_repo_dir(part):
            return True
    return False


def _copy_repo_subset(
    project_root: Path,
    repo_output_root: Path,
    output_root_name: str,
    dry_run: bool,
) -> None:
    """Copy the minimal repo subset needed by the Kaggle notebook."""

    blocked_top_level_parts = {
        ".git",
        "data",
        "checkpoints",
        "experiments",
        "reports",
        "notebooks",
        "outputs",
        "hf_bundle",
        "hf_sub300_bundle",
        "kaggle_bundle",
        "kaggle_sub300_bundle",
        output_root_name,
    }

    for file_name in REPO_FILE_INCLUDES:
        source_path = project_root / file_name
        if not source_path.exists():
            continue
        destination_path = repo_output_root / file_name
        if dry_run:
            continue
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)

    for dir_name in REPO_DIR_INCLUDES:
        source_root = project_root / dir_name
        if not source_root.exists():
            continue
        for source_path in source_root.rglob("*"):
            if not source_path.is_file():
                continue
            relative_path = source_path.relative_to(project_root)
            if _should_skip_repo_path(relative_path, blocked_top_level_parts):
                continue
            if source_path.suffix.lower() in {".pyc", ".zip"}:
                continue
            destination_path = repo_output_root / relative_path
            if dry_run:
                continue
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination_path)


def _validate_required_bundle_source_files(output_root: Path) -> None:
    """Ensure the copied repo still contains the source files required by Kaggle."""

    for relative_path in REQUIRED_BUNDLE_SOURCE_FILES:
        bundle_path = output_root / relative_path
        if not bundle_path.exists() or not bundle_path.is_file():
            raise BundleError(
                f"Missing required source file in bundle: {relative_path.as_posix()}"
            )


def _copy_checkpoints(project_root: Path, output_root: Path, dry_run: bool) -> list[Path]:
    """Copy RTMW-l checkpoint/config files into the bundle."""

    source_dir = project_root / "checkpoints" / "pose" / "rtmw_l"
    _require_exists(source_dir, "checkpoint directory")

    files = sorted(
        path
        for path in source_dir.iterdir()
        if path.is_file() and path.suffix.lower() in CHECKPOINT_SUFFIXES
    )
    if not files:
        raise BundleError(f"No checkpoint/config files found in {_path_text(source_dir)}")

    destination_dir = output_root / "checkpoints" / "pose" / "rtmw_l"
    copied_paths: list[Path] = []
    for source_path in files:
        destination_path = destination_dir / source_path.name
        copied_paths.append(destination_path)
        if dry_run:
            continue
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)
    return copied_paths


def _create_bundle_pose_config(repo_output_root: Path, target_subset: str, dry_run: bool) -> Path:
    """Create a Kaggle pose config for the target subset in the copied repo."""

    config_dir = repo_output_root / "configs" / "preprocessing" / "pose"
    candidate_paths = [
        config_dir / f"pose_rtmw_l_kaggle_{target_subset}.yaml",
        config_dir / "pose_rtmw_l_kaggle.yaml",
        config_dir / "pose_rtmw_l.yaml",
    ]

    template_path = next((path for path in candidate_paths[1:] if path.exists()), None)
    if template_path is None:
        raise BundleError(
            "Could not find a template pose config in the copied repo under "
            f"{_path_text(config_dir)}"
        )

    with template_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise BundleError(f"Pose config template is not a YAML mapping: {_path_text(template_path)}")

    payload.setdefault("dataset", {})
    payload.setdefault("input", {})
    payload.setdefault("output", {})
    payload.setdefault("pose", {})
    payload.setdefault("mmpose", {})
    payload.setdefault("quality", {})

    payload["input"]["subset"] = target_subset
    payload["input"]["splits"] = list(ALLOWED_SPLITS)
    payload["input"]["standardized_manifests_root"] = "data/datasets/WLASL/standardized/manifests"
    payload["input"]["manifest_filenames"] = {
        split: f"{target_subset}_{split}.csv" for split in ALLOWED_SPLITS
    }

    output_path = config_dir / f"pose_rtmw_l_kaggle_{target_subset}.yaml"
    if not dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)
    return output_path


def _build_bundle_manifest(
    args: argparse.Namespace,
    target_frame: pd.DataFrame,
    done_pose_ok_frame: pd.DataFrame,
    remaining_frame: pd.DataFrame,
    remaining_stats: dict[str, Any],
    warnings: list[str],
    project_root: Path,
    standardized_root: Path,
    pose_done_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Build MANIFEST.json content."""

    remaining_rows_by_split = {
        split: int((remaining_frame["split"].map(_safe_text).str.lower() == split).sum())
        for split in ALLOWED_SPLITS
    }

    return {
        "target_subset": args.target_subset,
        "done_subset": args.done_subset,
        "key_used": remaining_stats["key_used"],
        "total_target_rows": int(len(target_frame)),
        "done_rows_from_nslt300_pose_manifest": int(len(done_pose_ok_frame)),
        "covered_target_rows": int(remaining_stats["covered_rows"]),
        "remaining_rows": int(len(remaining_frame)),
        "remaining_rows_by_split": remaining_rows_by_split,
        "covered_rows_by_key": remaining_stats["covered_rows_by_key"],
        "remaining_first_available_key_counts": remaining_stats["remaining_first_available_key_counts"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_project_root": _path_text(project_root),
        "source_standardized_root": _path_text(standardized_root),
        "source_pose_done_root": _path_text(pose_done_root),
        "output_root": _path_text(output_root),
        "copy_repo": bool(args.copy_repo),
        "copy_checkpoints": bool(args.copy_checkpoints),
        "make_zip": bool(args.make_zip),
        "skip_missing_frames": bool(args.skip_missing_frames),
        "notes": [
            "This bundle contains only the remaining standardized rows for the target subset.",
            f"It is not a full {args.target_subset} standardized bundle.",
            "Already-done rows are inferred from the pose manifest of the done subset.",
            "Stable-key matching uses priority: instance_uid, then sample_id, then video_id.",
            "No raw videos, pose outputs, or graph tensors are included.",
        ]
        + warnings,
    }


def _build_readme_text(
    args: argparse.Namespace,
    remaining_frame: pd.DataFrame,
    repo_config_path: Path | None,
) -> str:
    """Create README_KAGGLE_BUNDLE.md text."""

    remaining_rows_by_split = {
        split: int((remaining_frame["split"].map(_safe_text).str.lower() == split).sum())
        for split in ALLOWED_SPLITS
    }
    config_text = (
        f"configs/preprocessing/{repo_config_path.name}"
        if repo_config_path is not None
        else "configs/preprocessing/pose/pose_rtmw_l_kaggle_nslt1000.yaml"
    )
    repo_step = (
        "2. Copy `repo/` into `/kaggle/working/Recognizing-sign-language-at-the-word-level`."
        if args.copy_repo
        else "2. Use your existing repo checkout at `/kaggle/working/Recognizing-sign-language-at-the-word-level`."
    )
    checkpoint_step = (
        "4. Copy `checkpoints/` into that project root."
        if args.copy_checkpoints
        else "4. Ensure the RTMW-l checkpoints already exist under that project root."
    )

    lines = [
        f"# Kaggle Bundle for Remaining {args.target_subset} RTMW-l Pose Extraction",
        "",
        "## Purpose",
        "",
        (
            f"This bundle is meant for extracting RTMW-l pose only for the remaining "
            f"rows of `{args.target_subset}` that are not already covered by "
            f"`{args.done_subset}` pose outputs."
        ),
        "",
        "This is not a full target-subset standardized bundle.",
        "Do not rename the manifests to `nslt1000_remaining`.",
        "Keep the subset name as `nslt1000` in the Kaggle notebook.",
        "",
        "## Notebook Settings",
        "",
        "```python",
        "from pathlib import Path",
        "",
        'INPUT_ROOT = Path("/kaggle/input/wlasl-nslt1000-remaining-standardized-rtmw")',
        'SUBSET = "nslt1000"',
        'STANDARDIZED_DIRNAME = "standardized_nslt1000"',
        "```",
        "",
        "## Important Notes",
        "",
        "- The standardized folder name stays `standardized_nslt1000`.",
        "- The manifests stay named `nslt1000_train.csv`, `nslt1000_val.csv`, `nslt1000_test.csv`, and `nslt1000_all.csv`.",
        "- The manifests inside this bundle contain only the remaining rows.",
        "- The extraction notebook should still run with `SUBSET = \"nslt1000\"`.",
        "- The generated pose outputs after Kaggle will still only cover the remaining rows.",
        f"- You must merge the new `{args.target_subset}` pose outputs with the existing `{args.done_subset}` pose outputs to obtain a complete `{args.target_subset}` pose set.",
        "- This bundle does not contain raw videos.",
        "- This bundle does not contain existing pose outputs.",
        "- This bundle does not contain graph tensors.",
        "",
        "## Counts",
        "",
        f"- Remaining train rows: `{remaining_rows_by_split['train']}`",
        f"- Remaining val rows: `{remaining_rows_by_split['val']}`",
        f"- Remaining test rows: `{remaining_rows_by_split['test']}`",
        f"- Remaining total rows: `{len(remaining_frame)}`",
        "",
        "## Kaggle Usage",
        "",
        "1. Upload the bundle contents to a private Kaggle Dataset.",
        repo_step,
        "3. Copy `standardized_nslt1000/` into that project root.",
        checkpoint_step,
        "5. Run the pose extractor with:",
        "",
        "```bash",
        f"python scripts/preprocess/02_extract_pose_rtmw.py --config {config_text}",
        "```",
        "",
    ]
    return "\n".join(lines)


def _write_bundle_manifests(
    remaining_frame: pd.DataFrame,
    standardized_bundle_root: Path,
    target_subset: str,
    dry_run: bool,
) -> dict[str, Path]:
    """Write all remaining manifests under the standardized bundle tree."""

    manifests_root = standardized_bundle_root / "data" / "datasets" / "WLASL" / "standardized" / "manifests"
    manifest_paths = {
        "all": manifests_root / f"{target_subset}_all.csv",
        "train": manifests_root / f"{target_subset}_train.csv",
        "val": manifests_root / f"{target_subset}_val.csv",
        "test": manifests_root / f"{target_subset}_test.csv",
    }

    if dry_run:
        return manifest_paths

    _write_manifest_csv(remaining_frame, manifest_paths["all"])
    for split in ALLOWED_SPLITS:
        split_frame = remaining_frame[
            remaining_frame["split"].map(_safe_text).str.lower() == split
        ].copy()
        _write_manifest_csv(split_frame, manifest_paths[split])
    return manifest_paths


def _validate_bundle_outputs(
    standardized_bundle_root: Path,
    manifest_paths: dict[str, Path],
    remaining_frame: pd.DataFrame,
) -> None:
    """Validate manifest counts and frame-directory coverage."""

    for label, manifest_path in manifest_paths.items():
        _require_exists(manifest_path, f"bundle manifest {label}")

    all_frame = pd.read_csv(manifest_paths["all"])
    train_frame = pd.read_csv(manifest_paths["train"])
    val_frame = pd.read_csv(manifest_paths["val"])
    test_frame = pd.read_csv(manifest_paths["test"])

    if len(all_frame) != len(train_frame) + len(val_frame) + len(test_frame):
        raise BundleError(
            "Bundle manifest counts are inconsistent: "
            f"all={len(all_frame)} train={len(train_frame)} val={len(val_frame)} test={len(test_frame)}"
        )
    if len(all_frame) != len(remaining_frame):
        raise BundleError(
            f"Bundle manifest all-row count does not match remaining rows: {len(all_frame)} != {len(remaining_frame)}"
        )

    missing_dirs: list[str] = []
    for row in all_frame.to_dict(orient="records"):
        frames_dir_text = _safe_text(row.get("frames_dir"))
        if not frames_dir_text:
            missing_dirs.append("empty frames_dir")
            continue
        bundle_frames_dir = standardized_bundle_root / Path(frames_dir_text)
        if not bundle_frames_dir.exists() or not bundle_frames_dir.is_dir():
            missing_dirs.append(_path_text(bundle_frames_dir))

    if missing_dirs:
        raise BundleError(
            "Some remaining manifest rows do not have copied frame folders in the bundle. "
            f"Examples: {missing_dirs[:5]}"
        )


def _create_zip(output_root: Path, dry_run: bool) -> Path:
    """Create a flat zip archive from the output root contents."""

    zip_path = output_root.with_suffix(".zip")
    if dry_run:
        return zip_path
    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output_root.rglob("*")):
            if path.is_dir():
                continue
            archive.write(path, path.relative_to(output_root).as_posix())
    return zip_path


def _print_summary(
    output_root: Path,
    target_frame: pd.DataFrame,
    done_pose_ok_frame: pd.DataFrame,
    remaining_stats: dict[str, Any],
    remaining_frame: pd.DataFrame,
    zip_path: Path | None,
) -> None:
    """Print a concise summary at the end."""

    remaining_rows_by_split = {
        split: int((remaining_frame["split"].map(_safe_text).str.lower() == split).sum())
        for split in ALLOWED_SPLITS
    }
    print("")
    print("Summary:")
    print(f"- total nslt1000 rows: {len(target_frame)}")
    print(f"- rows already covered by nslt300 pose: {remaining_stats['covered_rows']}")
    print(f"- done rows from nslt300 pose manifest: {len(done_pose_ok_frame)}")
    print(f"- rows remaining: {len(remaining_frame)}")
    print(
        "- rows remaining per split: "
        f"train={remaining_rows_by_split['train']} "
        f"val={remaining_rows_by_split['val']} "
        f"test={remaining_rows_by_split['test']}"
    )
    print(f"- output bundle path: {_path_text(output_root)}")
    if zip_path is not None:
        print(f"- zip path: {_path_text(zip_path)}")


def main() -> int:
    """CLI entrypoint."""

    args = parse_args()
    project_root = _resolve_under(Path.cwd(), args.project_root)
    standardized_root = _resolve_under(project_root, args.standardized_root)
    pose_done_root = _resolve_under(project_root, args.pose_done_root)
    output_root = _resolve_under(project_root, args.output_root)

    standardized_bundle_root = output_root / f"standardized_{args.target_subset}"
    repo_output_root = output_root / "repo"

    try:
        _require_exists(project_root, "project root")
        _require_exists(standardized_root, "standardized root")
        _require_exists(pose_done_root, "pose done root")
        _validate_output_root(output_root, project_root, standardized_root, pose_done_root)

        target_frame = _load_target_manifest(standardized_root, args.target_subset)
        done_pose_frame = _load_done_pose_manifest(pose_done_root, args.done_subset)
        done_lookup, done_pose_ok_frame = _build_done_lookup(done_pose_frame)
        remaining_frame, remaining_stats = _compute_remaining_rows(target_frame, done_lookup)

        remaining_frame, copy_plans, warnings = _prepare_remaining_manifests_and_copy_plan(
            remaining_frame=remaining_frame,
            standardized_root=standardized_root,
            target_subset=args.target_subset,
            standardized_bundle_root=standardized_bundle_root,
            skip_missing_frames=args.skip_missing_frames,
            verbose=args.verbose,
        )

        if args.dry_run:
            for warning in warnings:
                print(warning)
            _print_summary(
                output_root=output_root,
                target_frame=target_frame,
                done_pose_ok_frame=done_pose_ok_frame,
                remaining_stats=remaining_stats,
                remaining_frame=remaining_frame,
                zip_path=output_root.with_suffix(".zip") if args.make_zip else None,
            )
            return 0

        _ensure_empty_output_dir(output_root, overwrite=args.overwrite)

        manifest_paths = _write_bundle_manifests(
            remaining_frame=remaining_frame,
            standardized_bundle_root=standardized_bundle_root,
            target_subset=args.target_subset,
            dry_run=False,
        )
        _copy_remaining_frames(copy_plans, dry_run=False)

        repo_config_path: Path | None = None
        if args.copy_repo:
            _copy_repo_subset(
                project_root=project_root,
                repo_output_root=repo_output_root,
                output_root_name=output_root.name,
                dry_run=False,
            )
            _validate_required_bundle_source_files(output_root)
            repo_config_path = _create_bundle_pose_config(
                repo_output_root=repo_output_root,
                target_subset=args.target_subset,
                dry_run=False,
            )

        if args.copy_checkpoints:
            _copy_checkpoints(project_root=project_root, output_root=output_root, dry_run=False)

        manifest_payload = _build_bundle_manifest(
            args=args,
            target_frame=target_frame,
            done_pose_ok_frame=done_pose_ok_frame,
            remaining_frame=remaining_frame,
            remaining_stats=remaining_stats,
            warnings=warnings,
            project_root=project_root,
            standardized_root=standardized_root,
            pose_done_root=pose_done_root,
            output_root=output_root,
        )
        _write_json(output_root / "MANIFEST.json", manifest_payload)
        _write_text(
            output_root / "README_KAGGLE_BUNDLE.md",
            _build_readme_text(
                args=args,
                remaining_frame=remaining_frame,
                repo_config_path=repo_config_path,
            ),
        )

        _validate_bundle_outputs(
            standardized_bundle_root=standardized_bundle_root,
            manifest_paths=manifest_paths,
            remaining_frame=remaining_frame,
        )

        zip_path = _create_zip(output_root, dry_run=False) if args.make_zip else None

        for warning in warnings:
            print(warning)
        _print_summary(
            output_root=output_root,
            target_frame=target_frame,
            done_pose_ok_frame=done_pose_ok_frame,
            remaining_stats=remaining_stats,
            remaining_frame=remaining_frame,
            zip_path=zip_path,
        )
        return 0
    except BundleError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
