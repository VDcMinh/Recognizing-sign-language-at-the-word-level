from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


EXPECTED_SUBSET = "nslt100"
EXPECTED_NUM_SAMPLES = 1013
EXPECTED_NUM_CLASSES = 100
EXPECTED_LABEL_MIN = 0
EXPECTED_LABEL_MAX = 99
EXPECTED_PREFIX = "data/datasets/WLASL/branch_inputs/skeleton/rtmw_l/"
EXPECTED_SELECTED_27_SHAPE = [3, 150, 27, 1]
EXPECTED_SELECTED_31_SHAPE = [3, 150, 31, 1]
REQUIRED_MANIFEST_FILES = (
    "nslt100_selected_27_train.csv",
    "nslt100_selected_27_val.csv",
    "nslt100_selected_27_test.csv",
    "nslt100_selected_27_all.csv",
    "nslt100_selected_31_train.csv",
    "nslt100_selected_31_val.csv",
    "nslt100_selected_31_test.csv",
    "nslt100_selected_31_all.csv",
)


class BundleError(RuntimeError):
    """Raised when the HF skeleton bundle cannot be prepared safely."""


@dataclass(frozen=True)
class ManifestSummary:
    """Validation summary for one all-manifest file."""

    path: Path
    rows: int
    ok_count: int
    class_id_min: int
    class_id_max: int
    class_id_nunique: int


@dataclass(frozen=True)
class ValidationResult:
    """Resolved and validated bundle inputs."""

    project_root: Path
    data_root: Path
    output_dir: Path
    readme_path: Path
    metadata_path: Path
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
    manifest_summaries: tuple[ManifestSummary, ...]


def parse_args() -> argparse.Namespace:
    """Parse CLI options for Hugging Face skeleton bundle creation."""

    parser = argparse.ArgumentParser(
        description="Prepare a Hugging Face upload bundle for train-ready WLASL skeleton data."
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
    files = tuple(
        sorted(
            path
            for path in root.rglob("*")
            if path.is_file() and (suffix is None or path.suffix.lower() == suffix.lower())
        )
    )
    return files


def _validate_metadata(metadata_path: Path) -> dict[str, Any]:
    """Load metadata.json and validate a few required fields."""

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BundleError(f"metadata.json is not valid JSON: {metadata_path}") from exc

    checks = {
        "num_samples": EXPECTED_NUM_SAMPLES,
        "num_classes": EXPECTED_NUM_CLASSES,
        "selected_27_shape": EXPECTED_SELECTED_27_SHAPE,
        "selected_31_shape": EXPECTED_SELECTED_31_SHAPE,
    }
    for key, expected in checks.items():
        actual = metadata.get(key)
        if actual != expected:
            raise BundleError(
                f"metadata.json field {key!r} mismatch: expected {expected!r}, got {actual!r}."
            )
    return metadata


def _validate_manifest(manifest_path: Path) -> ManifestSummary:
    """Validate one all-manifest file against expected train-ready assumptions."""

    frame = pd.read_csv(manifest_path)
    required_columns = {"status", "class_id", "graph_tensor_path"}
    missing = sorted(required_columns - set(frame.columns))
    if missing:
        raise BundleError(f"Manifest {manifest_path.name} is missing required columns: {missing}")

    rows = int(len(frame))
    ok_count = int((frame["status"].fillna("").astype(str).str.strip().str.lower() == "ok").sum())
    if rows != EXPECTED_NUM_SAMPLES:
        raise BundleError(
            f"Manifest {manifest_path.name} has {rows} rows, expected {EXPECTED_NUM_SAMPLES}."
        )
    if ok_count != EXPECTED_NUM_SAMPLES:
        raise BundleError(
            f"Manifest {manifest_path.name} has {ok_count} rows with status=ok, expected {EXPECTED_NUM_SAMPLES}."
        )

    class_ids = pd.to_numeric(frame["class_id"], errors="coerce")
    if class_ids.isna().any():
        raise BundleError(f"Manifest {manifest_path.name} contains non-numeric class_id values.")
    class_id_min = int(class_ids.min())
    class_id_max = int(class_ids.max())
    class_id_nunique = int(class_ids.nunique())

    if class_id_min != EXPECTED_LABEL_MIN or class_id_max != EXPECTED_LABEL_MAX:
        raise BundleError(
            f"Manifest {manifest_path.name} class_id range is {class_id_min}..{class_id_max}, "
            f"expected {EXPECTED_LABEL_MIN}..{EXPECTED_LABEL_MAX}."
        )
    if class_id_nunique != EXPECTED_NUM_CLASSES:
        raise BundleError(
            f"Manifest {manifest_path.name} has {class_id_nunique} unique classes, "
            f"expected {EXPECTED_NUM_CLASSES}."
        )

    graph_paths = frame["graph_tensor_path"].fillna("").astype(str).str.strip()
    if (graph_paths == "").any():
        raise BundleError(f"Manifest {manifest_path.name} contains empty graph_tensor_path values.")

    return ManifestSummary(
        path=manifest_path,
        rows=rows,
        ok_count=ok_count,
        class_id_min=class_id_min,
        class_id_max=class_id_max,
        class_id_nunique=class_id_nunique,
    )


def validate_inputs(
    project_root: Path,
    data_root: Path,
    output_dir: Path,
    include_logs: bool,
) -> ValidationResult:
    """Validate the bundle inputs and collect all files to archive."""

    _ensure_exists(data_root, "data_root")

    selected_27_root = data_root / "graph_tensors" / "selected_27" / EXPECTED_SUBSET
    selected_31_root = data_root / "graph_tensors" / "selected_31" / EXPECTED_SUBSET
    manifests_root = data_root / "manifests"
    reports_root = data_root / "reports"
    logs_root = data_root / "logs"
    readme_path = data_root / "README.md"
    metadata_path = data_root / "metadata.json"

    _ensure_exists(selected_27_root, "selected_27 graph tensor root")
    _ensure_exists(selected_31_root, "selected_31 graph tensor root")
    _ensure_exists(manifests_root, "manifests root")
    _ensure_exists(reports_root, "reports root")
    _ensure_exists(readme_path, "README.md")
    _ensure_exists(metadata_path, "metadata.json")

    selected_27_files = _iter_files(selected_27_root, suffix=".npz")
    selected_31_files = _iter_files(selected_31_root, suffix=".npz")
    if len(selected_27_files) != EXPECTED_NUM_SAMPLES:
        raise BundleError(
            f"selected_27 contains {len(selected_27_files)} .npz files, expected {EXPECTED_NUM_SAMPLES}."
        )
    if len(selected_31_files) != EXPECTED_NUM_SAMPLES:
        raise BundleError(
            f"selected_31 contains {len(selected_31_files)} .npz files, expected {EXPECTED_NUM_SAMPLES}."
        )

    manifest_files = tuple(manifests_root / name for name in REQUIRED_MANIFEST_FILES)
    for path in manifest_files:
        _ensure_exists(path, f"manifest file {path.name}")

    report_files = _iter_files(reports_root)
    if not report_files:
        raise BundleError(f"reports/ contains no files: {reports_root}")

    log_files: tuple[Path, ...] = ()
    resolved_logs_root: Path | None = None
    if include_logs:
        _ensure_exists(logs_root, "logs root")
        log_files = _iter_files(logs_root)
        if not log_files:
            raise BundleError(f"logs/ contains no files: {logs_root}")
        resolved_logs_root = logs_root

    metadata = _validate_metadata(metadata_path)
    manifest_summaries = (
        _validate_manifest(manifests_root / "nslt100_selected_27_all.csv"),
        _validate_manifest(manifests_root / "nslt100_selected_31_all.csv"),
    )

    return ValidationResult(
        project_root=project_root,
        data_root=data_root,
        output_dir=output_dir,
        readme_path=readme_path,
        metadata_path=metadata_path,
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
        manifest_summaries=manifest_summaries,
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


def _copy_metadata_files(validation: ValidationResult, dry_run: bool) -> tuple[Path, Path]:
    """Copy README.md and metadata.json into the bundle root."""

    output_readme = validation.output_dir / "README.md"
    output_metadata = validation.output_dir / "metadata.json"
    if not dry_run:
        shutil.copy2(validation.readme_path, output_readme)
        shutil.copy2(validation.metadata_path, output_metadata)
    return output_readme, output_metadata


def _verify_zip(zip_path: Path, expected_contains: str | None = None) -> list[str]:
    """Verify zip internal paths and return the first five archive names."""

    with zipfile.ZipFile(zip_path, mode="r") as archive:
        names = archive.namelist()
    if not names:
        raise BundleError(f"Zip archive is empty: {zip_path}")
    for name in names:
        if not name.startswith(EXPECTED_PREFIX):
            raise BundleError(
                f"Unexpected internal path in {zip_path.name}: {name}. "
                f"Expected prefix {EXPECTED_PREFIX}."
            )
    if expected_contains is not None and not any(expected_contains in name for name in names):
        raise BundleError(
            f"{zip_path.name} is missing expected path fragment: {expected_contains}"
        )
    return names[:5]


def _print_validation_summary(validation: ValidationResult, include_logs: bool) -> None:
    """Print the input validation summary."""

    print(f"data_root: {validation.data_root}")
    print(f"output_dir: {validation.output_dir}")
    print(f"selected_27 npz count: {len(validation.selected_27_files)}")
    print(f"selected_31 npz count: {len(validation.selected_31_files)}")
    print(f"manifest files count: {len(validation.manifest_files)}")
    print(f"README path: {validation.readme_path}")
    print(f"metadata path: {validation.metadata_path}")
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
        include_logs=include_logs,
    )
    _print_validation_summary(validation, include_logs=include_logs)

    zip_plan: list[tuple[str, tuple[Path, ...], str | None]] = [
        ("graph_tensors_selected_27.zip", validation.selected_27_files, "graph_tensors/selected_27/nslt100/train/"),
        ("graph_tensors_selected_31.zip", validation.selected_31_files, "graph_tensors/selected_31/nslt100/train/"),
        ("manifests.zip", validation.manifest_files, "manifests/nslt100_selected_27_train.csv"),
        ("reports.zip", validation.report_files, "reports/"),
    ]
    if include_logs:
        zip_plan.append(("logs.zip", validation.log_files, "logs/"))

    if args.verbose:
        print("planned zip outputs:")
        for zip_name, files, _ in zip_plan:
            print(f"- {zip_name}: {len(files)} files")

    if args.dry_run:
        print("dry-run: validation succeeded, no files were written.")
        return 0

    validate_output_dir(output_dir, overwrite=args.overwrite)
    output_readme, output_metadata = _copy_metadata_files(validation, dry_run=False)

    created_archives: list[tuple[Path, int, list[str]]] = []
    for zip_name, files, expected_fragment in zip_plan:
        zip_path = output_dir / zip_name
        size_bytes = _write_zip(zip_path, validation.project_root, files)
        sample_paths = _verify_zip(zip_path, expected_contains=expected_fragment)
        created_archives.append((zip_path, size_bytes, sample_paths))

    if not output_readme.exists():
        raise BundleError(f"README copy failed: {output_readme}")
    if not output_metadata.exists():
        raise BundleError(f"metadata copy failed: {output_metadata}")

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
    print("1. Upload hf_bundle/ to a Hugging Face Dataset repository.")
    print("2. On Kaggle, download the dataset with snapshot_download or as a dataset mount.")
    print("3. Unzip the zip files into the project root.")
    print("4. Run the sanity checks for selected_27 and selected_31 before training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
