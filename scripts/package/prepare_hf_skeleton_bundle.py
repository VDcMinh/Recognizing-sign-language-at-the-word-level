from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_SUBSET = "nslt100"
DEFAULT_PREFIX = "data/datasets/WLASL/branch_inputs/skeleton/rtmw_l/"
KEYPOINT_SETS = ("selected_27", "selected_31")
GRAPH_SHAPES = {
    "selected_27": [3, 150, 27, 1],
    "selected_31": [3, 150, 31, 1],
}
REQUIRED_MANIFEST_SPLITS = ("train", "val", "test", "all")


class BundleError(RuntimeError):
    """Raised when the HF skeleton bundle cannot be prepared safely."""


@dataclass(frozen=True)
class ManifestSummary:
    """Validation summary for one keypoint-set all-manifest file."""

    keypoint_set: str
    path: Path
    rows: int
    ok_count: int
    class_id_min: int
    class_id_max: int
    class_id_nunique: int
    split_counts: dict[str, int]


@dataclass(frozen=True)
class ValidationResult:
    """Resolved and validated bundle inputs."""

    project_root: Path
    data_root: Path
    output_dir: Path
    subset: str
    pose_backend: str
    source_layout: str
    split_counts: dict[str, int]
    total_samples: int
    num_classes: int
    selected_27_root: Path
    selected_31_root: Path
    manifests_root: Path
    reports_root: Path
    logs_root: Path | None
    selected_27_files: tuple[Path, ...]
    selected_31_files: tuple[Path, ...]
    manifest_files: tuple[Path, ...]
    report_files: tuple[Path, ...]
    log_files: tuple[Path, ...]
    metadata: dict[str, Any]
    readme_text: str
    manifest_summaries: tuple[ManifestSummary, ...]


def parse_args() -> argparse.Namespace:
    """Parse CLI options for Hugging Face skeleton bundle creation."""

    parser = argparse.ArgumentParser(
        description="Prepare a Hugging Face upload bundle for train-ready WLASL skeleton data."
    )
    parser.add_argument(
        "--subset",
        type=str,
        default=DEFAULT_SUBSET,
        help="Dataset subset to package, e.g. nslt100 or nslt300.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/datasets/WLASL/branch_inputs/skeleton/rtmw_l"),
        help="Root directory containing train-ready skeleton data.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("hf_bundle"),
        help="Output directory for the Hugging Face bundle.",
    )
    parser.add_argument(
        "--pose-backend",
        type=str,
        default="rtmw_l",
        help="Pose backend name recorded in metadata and README.",
    )
    parser.add_argument(
        "--source-layout",
        type=str,
        default="wholebody_133",
        help="Source pose layout recorded in metadata and README.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace any existing output directory contents.",
    )
    parser.add_argument(
        "--no-logs",
        action="store_true",
        help="Skip creating logs.zip even if logs/ exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print the plan without writing bundle files.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print additional validation and file-level details.",
    )
    return parser.parse_args()


def _format_bytes(size: int) -> str:
    """Pretty-print a byte count."""

    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{size} B"


def _resolve_under(base: Path, value: Path) -> Path:
    """Resolve an absolute or project-relative path."""

    return value.resolve() if value.is_absolute() else (base / value).resolve()


def _ensure_exists(path: Path, label: str) -> None:
    """Raise when one required path is missing."""

    if not path.exists():
        raise BundleError(f"Missing required {label}: {path}")


def _iter_files(root: Path, suffix: str | None = None) -> tuple[Path, ...]:
    """Collect files recursively under one directory in stable order."""

    if not root.exists():
        return ()
    return tuple(
        sorted(
            path
            for path in root.rglob("*")
            if path.is_file() and (suffix is None or path.suffix.lower() == suffix.lower())
        )
    )


def _read_manifest(path: Path) -> pd.DataFrame:
    """Read one manifest and require the core columns used for packaging."""

    frame = pd.read_csv(path)
    required_columns = {"status", "class_id", "graph_tensor_path", "split"}
    missing = sorted(required_columns - set(frame.columns))
    if missing:
        raise BundleError(f"Manifest {path.name} is missing required columns: {missing}")
    return frame


def _validate_class_ids(frame: pd.DataFrame, manifest_path: Path) -> tuple[int, int, int]:
    """Validate numeric class ids and return min/max/unique stats."""

    class_ids = pd.to_numeric(frame["class_id"], errors="coerce")
    if class_ids.isna().any():
        raise BundleError(f"Manifest {manifest_path.name} contains non-numeric class_id values.")
    return int(class_ids.min()), int(class_ids.max()), int(class_ids.nunique())


def _validate_manifest_set(
    manifests_root: Path,
    subset: str,
    keypoint_set: str,
) -> tuple[tuple[Path, ...], ManifestSummary]:
    """Validate all manifests for one keypoint set."""

    manifest_paths = tuple(
        manifests_root / f"{subset}_{keypoint_set}_{split}.csv"
        for split in REQUIRED_MANIFEST_SPLITS
    )
    for path in manifest_paths:
        _ensure_exists(path, f"manifest file {path.name}")

    split_counts: dict[str, int] = {}
    total_from_splits = 0
    for split in ("train", "val", "test"):
        frame = _read_manifest(manifest_paths[REQUIRED_MANIFEST_SPLITS.index(split)])
        status_ok = frame["status"].fillna("").astype(str).str.strip().str.lower().eq("ok")
        if not bool(status_ok.all()):
            raise BundleError(f"Manifest {manifest_paths[REQUIRED_MANIFEST_SPLITS.index(split)].name} contains non-ok rows.")
        graph_paths = frame["graph_tensor_path"].fillna("").astype(str).str.strip()
        if bool((graph_paths == "").any()):
            raise BundleError(
                f"Manifest {manifest_paths[REQUIRED_MANIFEST_SPLITS.index(split)].name} contains empty graph_tensor_path values."
            )
        split_count = int(len(frame))
        split_counts[split] = split_count
        total_from_splits += split_count

    all_path = manifest_paths[REQUIRED_MANIFEST_SPLITS.index("all")]
    all_frame = _read_manifest(all_path)
    all_rows = int(len(all_frame))
    all_ok = int(all_frame["status"].fillna("").astype(str).str.strip().str.lower().eq("ok").sum())
    if all_rows != total_from_splits:
        raise BundleError(
            f"Manifest {all_path.name} has {all_rows} rows but train+val+test sum to {total_from_splits}."
        )
    if all_ok != all_rows:
        raise BundleError(f"Manifest {all_path.name} has {all_ok} rows with status=ok, expected {all_rows}.")
    class_id_min, class_id_max, class_id_nunique = _validate_class_ids(all_frame, all_path)
    graph_paths = all_frame["graph_tensor_path"].fillna("").astype(str).str.strip()
    if bool((graph_paths == "").any()):
        raise BundleError(f"Manifest {all_path.name} contains empty graph_tensor_path values.")

    summary = ManifestSummary(
        keypoint_set=keypoint_set,
        path=all_path,
        rows=all_rows,
        ok_count=all_ok,
        class_id_min=class_id_min,
        class_id_max=class_id_max,
        class_id_nunique=class_id_nunique,
        split_counts=split_counts,
    )
    return manifest_paths, summary


def _load_graph_shape(files: tuple[Path, ...], keypoint_set: str) -> list[int]:
    """Load one graph tensor sample and validate its shape."""

    if not files:
        raise BundleError(f"No graph tensor files found for {keypoint_set}.")
    with np.load(files[0], allow_pickle=False) as payload:
        data = payload["data"]
    shape = list(data.shape)
    expected_shape = GRAPH_SHAPES[keypoint_set]
    if shape != expected_shape:
        raise BundleError(
            f"{keypoint_set} sample tensor shape mismatch: expected {expected_shape}, got {shape}."
        )
    return shape


def _graph_zip_name(subset: str, keypoint_set: str) -> str:
    """Return the graph zip filename while preserving the legacy nslt100 names."""

    if subset == DEFAULT_SUBSET:
        return f"graph_tensors_{keypoint_set}.zip"
    return f"graph_tensors_{keypoint_set}_{subset}.zip"


def _build_metadata(
    subset: str,
    pose_backend: str,
    source_layout: str,
    split_counts: dict[str, int],
    num_classes: int,
) -> dict[str, Any]:
    """Build the bundle metadata payload."""

    total_samples = int(sum(split_counts[split] for split in ("train", "val", "test")))
    bundle_files = [
        _graph_zip_name(subset, "selected_27"),
        _graph_zip_name(subset, "selected_31"),
        "manifests.zip",
        "reports.zip",
        "logs.zip",
    ]
    return {
        "dataset": "WLASL",
        "subset": subset,
        "num_classes": num_classes,
        "total_samples": total_samples,
        "splits": {
            "train": int(split_counts["train"]),
            "val": int(split_counts["val"]),
            "test": int(split_counts["test"]),
        },
        "pose_estimator": pose_backend,
        "source_pose_layout": source_layout,
        "target_num_frames": 150,
        "graph_tensor_layout": "CTVM",
        "keypoint_sets": {
            "selected_27": {
                "graph_tensor_shape": GRAPH_SHAPES["selected_27"],
                "total_files": total_samples,
            },
            "selected_31": {
                "graph_tensor_shape": GRAPH_SHAPES["selected_31"],
                "total_files": total_samples,
            },
        },
        "expected_training_use": [
            "skeleton branch",
            "graph tensors train-ready",
        ],
        "bundle_files": bundle_files,
        "created_at": date.today().isoformat(),
    }


def _build_readme(metadata: dict[str, Any]) -> str:
    """Build the bundle README text."""

    subset = metadata["subset"]
    shape_27 = tuple(metadata["keypoint_sets"]["selected_27"]["graph_tensor_shape"])
    shape_31 = tuple(metadata["keypoint_sets"]["selected_31"]["graph_tensor_shape"])
    return "\n".join(
        [
            f"# WLASL {subset} Skeleton Graph Tensors",
            "",
            f"This bundle is intended for training skeleton-based SLR models on the WLASL `{subset}` subset.",
            "",
            "## What this bundle contains",
            "",
            f"- Train-ready graph tensors for `selected_27` and `selected_31` built from `{metadata['pose_estimator']}` pose outputs.",
            "- Skeleton manifests for both keypoint sets.",
            "- Build reports and logs for preprocessing traceability.",
            "",
            "## What this bundle does not contain",
            "",
            "- Raw videos.",
            "- Standardized frames.",
            f"- Full source pose files in `{metadata['source_pose_layout']}` layout.",
            "",
            "## Tensor shapes",
            "",
            f"- `selected_27`: `{shape_27}`",
            f"- `selected_31`: `{shape_31}`",
            "",
            "Channel order is `(x, y, confidence)` and labels are read from the manifest column `class_id`.",
            "",
            "## Upload to Hugging Face",
            "",
            "1. Create or open a Hugging Face Dataset repository.",
            "2. Upload the full contents of this bundle directory.",
            "3. Keep the zip files, `metadata.json`, and `README.md` together at the dataset root.",
            "",
            "## Training-side download and unzip",
            "",
            "1. Download the dataset bundle in the training notebook or mount it as a dataset.",
            "2. Unzip the graph tensor archives into the project root so the internal paths resolve under:",
            "   `data/datasets/WLASL/branch_inputs/skeleton/rtmw_l/`",
            "3. Unzip `manifests.zip`, `reports.zip`, and `logs.zip` into the same project root if you need metadata or provenance files.",
            "4. Run the skeleton training notebook or manifest-driven training loader against the unpacked graph tensors.",
            "",
            "## Notes",
            "",
            "- This bundle is purpose-built for the skeleton branch.",
            "- Graph tensors are precomputed and should not be regenerated during training.",
            "- The graph tensor zips contain train/val/test splits only for the packaged subset.",
        ]
    )


def validate_inputs(
    project_root: Path,
    data_root: Path,
    output_dir: Path,
    subset: str,
    pose_backend: str,
    source_layout: str,
    include_logs: bool,
) -> ValidationResult:
    """Validate the bundle inputs and collect all files to archive."""

    _ensure_exists(data_root, "data_root")

    selected_27_root = data_root / "graph_tensors" / "selected_27" / subset
    selected_31_root = data_root / "graph_tensors" / "selected_31" / subset
    manifests_root = data_root / "manifests"
    reports_root = data_root / "reports"
    logs_root = data_root / "logs"

    _ensure_exists(selected_27_root, "selected_27 graph tensor root")
    _ensure_exists(selected_31_root, "selected_31 graph tensor root")
    _ensure_exists(manifests_root, "manifests root")
    _ensure_exists(reports_root, "reports root")

    selected_27_files = _iter_files(selected_27_root, suffix=".npz")
    selected_31_files = _iter_files(selected_31_root, suffix=".npz")
    if not selected_27_files:
        raise BundleError(f"selected_27 contains no .npz files: {selected_27_root}")
    if not selected_31_files:
        raise BundleError(f"selected_31 contains no .npz files: {selected_31_root}")

    manifest_files_set_27, summary_27 = _validate_manifest_set(manifests_root, subset, "selected_27")
    manifest_files_set_31, summary_31 = _validate_manifest_set(manifests_root, subset, "selected_31")

    if summary_27.rows != len(selected_27_files):
        raise BundleError(
            f"selected_27 graph tensors contain {len(selected_27_files)} files but manifest has {summary_27.rows} rows."
        )
    if summary_31.rows != len(selected_31_files):
        raise BundleError(
            f"selected_31 graph tensors contain {len(selected_31_files)} files but manifest has {summary_31.rows} rows."
        )
    if summary_27.split_counts != summary_31.split_counts:
        raise BundleError("selected_27 and selected_31 split counts do not match.")
    if summary_27.class_id_nunique != summary_31.class_id_nunique:
        raise BundleError("selected_27 and selected_31 class counts do not match.")
    if summary_27.class_id_min != summary_31.class_id_min or summary_27.class_id_max != summary_31.class_id_max:
        raise BundleError("selected_27 and selected_31 class_id ranges do not match.")

    _load_graph_shape(selected_27_files, "selected_27")
    _load_graph_shape(selected_31_files, "selected_31")

    report_files = tuple(sorted(reports_root.glob(f"{subset}_*_skeleton_inputs_report.md")))
    if not report_files:
        raise BundleError(f"No subset-specific report files found for {subset} under {reports_root}.")

    log_files: tuple[Path, ...] = ()
    resolved_logs_root: Path | None = None
    if include_logs:
        _ensure_exists(logs_root, "logs root")
        log_files = tuple(sorted(logs_root.glob(f"build_skeleton_*_{subset}.log")))
        if not log_files:
            raise BundleError(f"No subset-specific log files found for {subset} under {logs_root}.")
        resolved_logs_root = logs_root

    manifest_files = manifest_files_set_27 + manifest_files_set_31
    split_counts = dict(summary_27.split_counts)
    total_samples = int(summary_27.rows)
    num_classes = int(summary_27.class_id_nunique)
    metadata = _build_metadata(
        subset=subset,
        pose_backend=pose_backend,
        source_layout=source_layout,
        split_counts=split_counts,
        num_classes=num_classes,
    )
    readme_text = _build_readme(metadata)

    return ValidationResult(
        project_root=project_root,
        data_root=data_root,
        output_dir=output_dir,
        subset=subset,
        pose_backend=pose_backend,
        source_layout=source_layout,
        split_counts=split_counts,
        total_samples=total_samples,
        num_classes=num_classes,
        selected_27_root=selected_27_root,
        selected_31_root=selected_31_root,
        manifests_root=manifests_root,
        reports_root=reports_root,
        logs_root=resolved_logs_root,
        selected_27_files=selected_27_files,
        selected_31_files=selected_31_files,
        manifest_files=manifest_files,
        report_files=report_files,
        log_files=log_files,
        metadata=metadata,
        readme_text=readme_text,
        manifest_summaries=(summary_27, summary_31),
    )


def validate_output_dir(output_dir: Path, overwrite: bool) -> None:
    """Validate or clear the output directory."""

    if output_dir.exists():
        if not overwrite and any(output_dir.iterdir()):
            raise BundleError(
                f"Output directory already exists and is not empty: {output_dir}. Use --overwrite."
            )
        if overwrite:
            shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _write_zip(zip_path: Path, project_root: Path, files: tuple[Path, ...]) -> int:
    """Write one zip archive using paths relative to project root."""

    if not files:
        raise BundleError(f"Cannot create empty zip archive: {zip_path.name}")
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for source_path in files:
            archive.write(source_path, source_path.relative_to(project_root))
    return int(zip_path.stat().st_size)


def _write_bundle_text_files(validation: ValidationResult, dry_run: bool) -> tuple[Path, Path]:
    """Write README.md and metadata.json into the bundle root."""

    output_readme = validation.output_dir / "README.md"
    output_metadata = validation.output_dir / "metadata.json"
    if not dry_run:
        output_readme.write_text(validation.readme_text, encoding="utf-8")
        output_metadata.write_text(json.dumps(validation.metadata, indent=2), encoding="utf-8")
    return output_readme, output_metadata


def _verify_zip(zip_path: Path, expected_contains: str | None = None) -> list[str]:
    """Verify zip internal paths and return the first five archive names."""

    with zipfile.ZipFile(zip_path, mode="r") as archive:
        names = archive.namelist()
    if not names:
        raise BundleError(f"Zip archive is empty: {zip_path}")
    for name in names:
        if not name.startswith(DEFAULT_PREFIX):
            raise BundleError(
                f"Unexpected internal path in {zip_path.name}: {name}. Expected prefix {DEFAULT_PREFIX}."
            )
    if expected_contains is not None and not any(expected_contains in name for name in names):
        raise BundleError(f"{zip_path.name} is missing expected path fragment: {expected_contains}")
    return names[:5]


def _print_validation_summary(validation: ValidationResult, include_logs: bool) -> None:
    """Print the input validation summary."""

    print(f"subset: {validation.subset}")
    print(f"data_root: {validation.data_root}")
    print(f"output_dir: {validation.output_dir}")
    print(f"selected_27 npz count: {len(validation.selected_27_files)}")
    print(f"selected_31 npz count: {len(validation.selected_31_files)}")
    print(
        "split counts: "
        f"train={validation.split_counts['train']}, "
        f"val={validation.split_counts['val']}, "
        f"test={validation.split_counts['test']}"
    )
    print(f"total samples: {validation.total_samples}")
    print(f"num classes: {validation.num_classes}")
    print(f"manifest files count: {len(validation.manifest_files)}")
    print(f"report files count: {len(validation.report_files)}")
    if include_logs:
        print(f"logs files count: {len(validation.log_files)}")
    else:
        print("logs packaging: skipped by --no-logs")
    for summary in validation.manifest_summaries:
        print(
            f"manifest validation [{summary.path.name}]: "
            f"rows={summary.rows}, ok={summary.ok_count}, "
            f"class_id_range={summary.class_id_min}..{summary.class_id_max}, "
            f"class_id_nunique={summary.class_id_nunique}"
        )


def main() -> int:
    """Prepare the Hugging Face upload bundle for skeleton train-ready data."""

    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    data_root = _resolve_under(project_root, args.data_root)
    output_dir = _resolve_under(project_root, args.output_dir)
    include_logs = not args.no_logs

    validation = validate_inputs(
        project_root=project_root,
        data_root=data_root,
        output_dir=output_dir,
        subset=args.subset,
        pose_backend=args.pose_backend,
        source_layout=args.source_layout,
        include_logs=include_logs,
    )
    _print_validation_summary(validation, include_logs=include_logs)

    zip_plan: list[tuple[str, tuple[Path, ...], str | None]] = [
        (
            _graph_zip_name(validation.subset, "selected_27"),
            validation.selected_27_files,
            f"graph_tensors/selected_27/{validation.subset}/train/",
        ),
        (
            _graph_zip_name(validation.subset, "selected_31"),
            validation.selected_31_files,
            f"graph_tensors/selected_31/{validation.subset}/train/",
        ),
        (
            "manifests.zip",
            validation.manifest_files,
            f"manifests/{validation.subset}_selected_27_train.csv",
        ),
        (
            "reports.zip",
            validation.report_files,
            f"reports/{validation.subset}_selected_27_skeleton_inputs_report.md",
        ),
    ]
    if include_logs:
        zip_plan.append(
            (
                "logs.zip",
                validation.log_files,
                f"logs/build_skeleton_selected_27_{validation.subset}.log",
            )
        )

    if args.verbose:
        print("planned zip outputs:")
        for zip_name, files, _ in zip_plan:
            print(f"- {zip_name}: {len(files)} files")

    if args.dry_run:
        print("dry-run: validation succeeded, no files were written.")
        return 0

    validate_output_dir(output_dir, overwrite=args.overwrite)
    output_readme, output_metadata = _write_bundle_text_files(validation, dry_run=False)

    created_archives: list[tuple[Path, int, list[str]]] = []
    for zip_name, files, expected_fragment in zip_plan:
        zip_path = output_dir / zip_name
        size_bytes = _write_zip(zip_path, validation.project_root, files)
        sample_paths = _verify_zip(zip_path, expected_contains=expected_fragment)
        created_archives.append((zip_path, size_bytes, sample_paths))

    if not output_readme.exists():
        raise BundleError(f"README creation failed: {output_readme}")
    if not output_metadata.exists():
        raise BundleError(f"metadata creation failed: {output_metadata}")

    total_size_bytes = sum(size for _, size, _ in created_archives)
    total_size_bytes += output_readme.stat().st_size + output_metadata.stat().st_size

    print()
    print("zip files created:")
    for zip_path, size_bytes, sample_paths in created_archives:
        print(f"- {zip_path.name}: {_format_bytes(size_bytes)}")
        print("  sample internal paths:")
        for name in sample_paths:
            print(f"  - {name}")

    print()
    print(f"bundle README: {output_readme}")
    print(f"bundle metadata: {output_metadata}")
    print(f"total bundle size: {_format_bytes(total_size_bytes)}")
    print()
    print("next steps:")
    print(f"1. Upload {validation.output_dir.name}/ to a Hugging Face Dataset repository.")
    print("2. On Kaggle or another training environment, download the dataset bundle.")
    print("3. Unzip the zip files into the project root so the archived internal paths land in place.")
    print("4. Run the skeleton sanity checks before training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
