"""Package NSLT300 regions branch inputs into one Kaggle-ready dataset."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np


DEFAULT_SOURCE_ROOT = Path("data/datasets/WLASL/branch_inputs/regions/rtmw_l")
DEFAULT_OUTPUT_ROOT = Path("packaging_outputs")
DEFAULT_PACKAGE_NAME = "wlasl-nslt300-regions-rtmw-l-allregions"
EXPECTED_SHAPE = (3, 3, 64, 112, 112)
ACTIVE_REGIONS = ("left_hand", "right_hand", "face")
SPLITS = ("train", "val", "test")
REQUIRED_MANIFEST_COLUMNS = ("sample_id", "class_id", "gloss", "split", "tensor_path", "status")
REPORT_CANDIDATES = (
    "nslt300_region_crop_quality_report.md",
    "nslt300_region_low_quality_samples.csv",
)


class PackagingError(RuntimeError):
    """Raised when the package cannot be built safely."""


@dataclass(frozen=True)
class SplitStats:
    split: str
    manifest_path: Path
    tensor_dir: Path
    manifest_rows: int
    tensor_files: int
    class_id_min: int
    class_id_max: int
    class_id_unique: int
    sample_ids_checked: tuple[str, ...]
    tensor_basenames: tuple[str, ...]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Package Kaggle-ready NSLT300 regions branch inputs."
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--package-name", type=str, default=DEFAULT_PACKAGE_NAME)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--clean", action="store_true", help="Remove the target package folder/zip before rebuilding.")
    parser.add_argument("--zip", dest="create_zip", action="store_true", help="Create a combined zip archive.")
    parser.set_defaults(create_zip=False)
    return parser


def _resolve_under(project_root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _ensure_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise PackagingError(f"Missing required {label}: {path}")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _read_manifest(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise PackagingError(f"Manifest has no header: {path}")
        rows = [{key: value or "" for key, value in row.items()} for row in reader]
        missing = [column for column in REQUIRED_MANIFEST_COLUMNS if column not in reader.fieldnames]
        if missing:
            raise PackagingError(f"Manifest {path.name} is missing required columns: {missing}")
        return list(reader.fieldnames), rows


def _write_manifest(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    _ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _count_files(root: Path) -> int:
    return sum(1 for path in root.rglob("*") if path.is_file())


def _sum_size(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _format_size(size_bytes: int) -> str:
    value = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{size_bytes} B"


def _class_id_from_row(row: dict[str, str], *, manifest_name: str) -> int:
    raw = str(row.get("class_id", "")).strip()
    try:
        value = int(float(raw))
    except (TypeError, ValueError) as exc:
        raise PackagingError(f"Manifest {manifest_name} has non-integer class_id: {raw!r}") from exc
    if not 0 <= value <= 299:
        raise PackagingError(f"Manifest {manifest_name} has class_id outside 0..299: {value}")
    return value


def _verify_tensor_shape(tensor_path: Path) -> tuple[int, ...]:
    with np.load(tensor_path, allow_pickle=False) as payload:
        if "data" not in payload:
            raise PackagingError(f"Tensor file is missing 'data' key: {tensor_path}")
        shape = tuple(int(value) for value in payload["data"].shape)
        region_names = tuple(str(item) for item in payload["region_names"].tolist()) if "region_names" in payload else ()
    if shape != EXPECTED_SHAPE:
        raise PackagingError(
            f"Tensor shape mismatch for {tensor_path.name}: expected {EXPECTED_SHAPE}, got {shape}"
        )
    if region_names and region_names != ACTIVE_REGIONS:
        raise PackagingError(
            f"Region names mismatch for {tensor_path.name}: expected {ACTIVE_REGIONS}, got {region_names}"
        )
    return shape


def _validate_split(source_root: Path, split: str) -> SplitStats:
    manifest_path = source_root / "manifests" / f"nslt300_{split}.csv"
    tensor_dir = source_root / "tensors" / "nslt300" / split
    _ensure_exists(manifest_path, f"{split} manifest")
    _ensure_exists(tensor_dir, f"{split} tensor directory")

    fieldnames, rows = _read_manifest(manifest_path)
    if "nslt100" in manifest_path.name.lower():
        raise PackagingError(f"Manifest path incorrectly points to nslt100: {manifest_path}")
    if not rows:
        raise PackagingError(f"Manifest is empty: {manifest_path}")

    class_ids: list[int] = []
    checked_ids: list[str] = []
    tensor_basenames: list[str] = []
    for index, row in enumerate(rows):
        split_value = str(row.get("split", "")).strip().lower()
        if split_value != split:
            raise PackagingError(
                f"Manifest {manifest_path.name} has row with split={split_value!r}, expected {split!r}"
            )
        sample_id = str(row.get("sample_id", "")).strip()
        if not sample_id:
            raise PackagingError(f"Manifest {manifest_path.name} has empty sample_id at row {index + 2}")
        if not str(row.get("gloss", "")).strip():
            raise PackagingError(f"Manifest {manifest_path.name} has empty gloss for sample_id={sample_id}")
        status = str(row.get("status", "")).strip().lower()
        if status != "ok":
            raise PackagingError(
                f"Manifest {manifest_path.name} contains non-ok row for sample_id={sample_id}: status={status!r}"
            )
        class_ids.append(_class_id_from_row(row, manifest_name=manifest_path.name))
        tensor_raw = str(row.get("tensor_path", "")).strip()
        if not tensor_raw:
            raise PackagingError(f"Manifest {manifest_path.name} has empty tensor_path for sample_id={sample_id}")
        basename = Path(tensor_raw).name
        if not basename:
            raise PackagingError(f"Manifest {manifest_path.name} has invalid tensor_path for sample_id={sample_id}")
        tensor_path = tensor_dir / basename
        if not tensor_path.exists():
            raise PackagingError(
                f"Manifest {manifest_path.name} references missing tensor for sample_id={sample_id}: {tensor_path}"
            )
        tensor_basenames.append(basename)
        if index < 3:
            _verify_tensor_shape(tensor_path)
            checked_ids.append(sample_id)

    tensor_files = sorted(tensor_dir.glob("*.npz"))
    if not tensor_files:
        raise PackagingError(f"No tensor files found in {tensor_dir}")
    if len(tensor_files) < len(rows):
        raise PackagingError(
            f"Tensor count is smaller than manifest rows for split={split}: rows={len(rows)} tensor_files={len(tensor_files)}"
        )
    if len(tensor_files) != len(rows):
        raise PackagingError(
            f"Tensor count mismatch for split={split}: rows={len(rows)} tensor_files={len(tensor_files)}"
        )

    return SplitStats(
        split=split,
        manifest_path=manifest_path,
        tensor_dir=tensor_dir,
        manifest_rows=len(rows),
        tensor_files=len(tensor_files),
        class_id_min=min(class_ids),
        class_id_max=max(class_ids),
        class_id_unique=len(set(class_ids)),
        sample_ids_checked=tuple(checked_ids),
        tensor_basenames=tuple(tensor_basenames),
    )


def _collect_reports(source_root: Path) -> tuple[Path, ...]:
    reports_root = source_root / "reports"
    if not reports_root.exists():
        return ()
    found: list[Path] = []
    for filename in REPORT_CANDIDATES:
        path = reports_root / filename
        if path.exists() and path.is_file():
            found.append(path)
    return tuple(found)


def _sanitize_rows(fieldnames: list[str], rows: list[dict[str, str]], split: str) -> list[dict[str, str]]:
    sanitized: list[dict[str, str]] = []
    for row in rows:
        updated = {key: value for key, value in row.items() if key in fieldnames}
        basename = Path(str(updated.get("tensor_path", "")).strip()).name
        updated["tensor_path"] = f"tensors/nslt300/{split}/{basename}"
        if "crop_root" in updated:
            updated["crop_root"] = ""
        if "preview_path" in updated:
            updated["preview_path"] = ""
        sanitized.append(updated)
    return sanitized


def _link_or_copy_file(source_path: Path, destination_path: Path) -> None:
    _ensure_dir(destination_path.parent)
    try:
        if destination_path.exists():
            destination_path.unlink()
        os.link(source_path, destination_path)
    except OSError:
        shutil.copy2(source_path, destination_path)


def _materialize_tensor_split(source_dir: Path, destination_dir: Path) -> int:
    _ensure_dir(destination_dir)
    count = 0
    for source_path in sorted(source_dir.glob("*.npz")):
        _link_or_copy_file(source_path, destination_dir / source_path.name)
        count += 1
    return count


def _build_verify_script_text() -> str:
    return """from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

EXPECTED_SHAPE = (3, 3, 64, 112, 112)
ACTIVE_REGIONS = ("left_hand", "right_hand", "face")
SPLITS = ("train", "val", "test")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify a Kaggle-ready NSLT300 regions package."
    )
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--sample-checks", type=int, default=3)
    parser.add_argument("--summary-path", type=Path, default=None)
    return parser


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [{key: value or "" for key, value in row.items()} for row in reader]


def resolve_path(base: Path, value: str) -> Path:
    raw = Path(value)
    return raw if raw.is_absolute() else (base / raw)


def write_summary(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    package_root = args.package_root.resolve()
    metadata_path = package_root / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata.json: {metadata_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    branch_root = package_root / "regions" / "rtmw_l"
    manifest_root = branch_root / "manifests"
    tensors_root = branch_root / "tensors" / "nslt300"
    verify_root = package_root / "verify"
    summary_path = args.summary_path.resolve() if args.summary_path is not None else (verify_root / "verify_summary.json")

    results = {
        "status": "pass",
        "package_name": metadata.get("package_name", package_root.name),
        "package_root": package_root.as_posix(),
        "subset": metadata.get("subset", "nslt300"),
        "active_regions": list(metadata.get("active_regions", list(ACTIVE_REGIONS))),
        "expected_shape": list(metadata.get("expected_shape", list(EXPECTED_SHAPE))),
        "splits": {},
    }

    for split in SPLITS:
        manifest_path = manifest_root / f"nslt300_{split}.csv"
        tensor_dir = tensors_root / split
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest: {manifest_path}")
        if not tensor_dir.exists():
            raise FileNotFoundError(f"Missing tensor directory: {tensor_dir}")

        rows = read_manifest(manifest_path)
        tensor_files = sorted(tensor_dir.glob("*.npz"))
        if len(rows) != len(tensor_files):
            raise ValueError(
                f"Split {split} mismatch: manifest_rows={len(rows)} tensor_files={len(tensor_files)}"
            )

        checked_sample_ids: list[str] = []
        for row in rows[: max(1, int(args.sample_checks))]:
            class_id = int(float(str(row.get("class_id", "")).strip()))
            if not 0 <= class_id <= 299:
                raise ValueError(f"Split {split} has class_id outside 0..299: {class_id}")
            tensor_path = resolve_path(branch_root, str(row.get("tensor_path", "")).strip())
            if not tensor_path.exists():
                raise FileNotFoundError(f"Missing tensor referenced by manifest: {tensor_path}")
            with np.load(tensor_path, allow_pickle=False) as payload:
                if "data" not in payload:
                    raise KeyError(f"Tensor file missing 'data' key: {tensor_path}")
                shape = tuple(int(value) for value in payload["data"].shape)
                if shape != EXPECTED_SHAPE:
                    raise ValueError(
                        f"Tensor shape mismatch for {tensor_path.name}: expected {EXPECTED_SHAPE}, got {shape}"
                    )
                if "region_names" in payload:
                    region_names = tuple(str(item) for item in payload["region_names"].tolist())
                    if region_names != ACTIVE_REGIONS:
                        raise ValueError(
                            f"Region names mismatch for {tensor_path.name}: expected {ACTIVE_REGIONS}, got {region_names}"
                        )
            checked_sample_ids.append(str(row.get("sample_id", "")))

        class_ids = [int(float(str(row.get("class_id", "")).strip())) for row in rows]
        results["splits"][split] = {
            "manifest": str(Path("regions/rtmw_l/manifests") / f"nslt300_{split}.csv").replace("\\\\", "/"),
            "tensor_dir": str(Path("regions/rtmw_l/tensors/nslt300") / split).replace("\\\\", "/"),
            "manifest_rows": len(rows),
            "tensor_files": len(tensor_files),
            "class_id_min": min(class_ids),
            "class_id_max": max(class_ids),
            "class_id_unique": len(set(class_ids)),
            "checked_sample_ids": checked_sample_ids,
        }

    write_summary(summary_path, results)
    print("VERIFY PASS")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""


def _write_verify_script(package_root: Path) -> Path:
    verify_root = package_root / "verify"
    _ensure_dir(verify_root)
    script_path = verify_root / "verify_package.py"
    script_path.write_text(_build_verify_script_text(), encoding="utf-8")
    return script_path


def _write_metadata(package_root: Path, split_stats: dict[str, SplitStats]) -> Path:
    metadata = {
        "package_name": package_root.name,
        "purpose": "Kaggle-ready Regions branch inputs for training RegionResNet18GRU on NSLT300",
        "subset": "nslt300",
        "branch": "regions",
        "pose_backend": "rtmw_l",
        "active_regions": list(ACTIVE_REGIONS),
        "expected_shape": list(EXPECTED_SHAPE),
        "num_classes": 300,
        "splits": {
            split: {
                "manifest": f"regions/rtmw_l/manifests/nslt300_{split}.csv",
                "tensor_dir": f"regions/rtmw_l/tensors/nslt300/{split}",
                "manifest_rows": split_stats[split].manifest_rows,
                "tensor_files": split_stats[split].tensor_files,
            }
            for split in SPLITS
        },
        "notes": [
            "This package does not include raw videos.",
            "This package does not include model checkpoints.",
            "This package is intended for training the Regions branch for NSLT300 on Kaggle.",
        ],
    }
    path = package_root / "metadata.json"
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _write_readme(package_root: Path) -> Path:
    package_name = package_root.name
    kaggle_root = (
        f"/kaggle/input/{package_name}/{package_name}/regions/rtmw_l"
    )
    text = "\n".join(
        [
            "# WLASL NSLT300 Regions RTMW-l All-Regions Dataset",
            "",
            "## 1. Purpose",
            "This package provides Kaggle-ready Regions branch inputs for training `RegionResNet18GRU` on WLASL `nslt300`.",
            "",
            "## 2. What is included",
            "- NSLT300 manifests for train/val/test.",
            "- Region tensors for train/val/test.",
            "- Lightweight subset-specific region reports when available.",
            "- Package metadata and a self-contained verify script.",
            "",
            "## 3. Folder structure",
            "```text",
            f"{package_name}/",
            "|-- README.md",
            "|-- metadata.json",
            "|-- regions/",
            "|   `-- rtmw_l/",
            "|       |-- manifests/",
            "|       |   |-- nslt300_train.csv",
            "|       |   |-- nslt300_val.csv",
            "|       |   `-- nslt300_test.csv",
            "|       |-- tensors/",
            "|       |   `-- nslt300/",
            "|       |       |-- train/",
            "|       |       |-- val/",
            "|       |       `-- test/",
            "|       `-- reports/",
            "`-- verify/",
            "    |-- verify_package.py",
            "    `-- verify_summary.json",
            "```",
            "",
            "## 4. Dataset details",
            "- Subset: `nslt300`",
            "- Num classes: `300`",
            "- Pose backend: `rtmw_l`",
            "- Branch: `regions`",
            "",
            "## 5. Active regions",
            "Active regions: `left_hand + right_hand + face`",
            "",
            "## 6. Expected tensor shape",
            "Expected shape: `(3, 3, 64, 112, 112)`",
            "",
            "## 7. How to upload to Kaggle Dataset",
            "Create a private Kaggle Dataset and upload either the package folder contents or the generated zip file.",
            "",
            "## 8. How to add to Kaggle Notebook",
            "Attach the Kaggle Dataset to the notebook. The mounted input path will typically start with `/kaggle/input/<dataset-slug>/`.",
            "",
            "## 9. Suggested data_root",
            kaggle_root,
            "",
            "## 10. How to verify",
            "Local package root:",
            "```bash",
            "python verify/verify_package.py --package-root .",
            "```",
            "From the repo:",
            "```bash",
            f"python packaging_outputs/{package_name}/verify/verify_package.py --package-root packaging_outputs/{package_name}",
            "```",
            "Note: `/kaggle/input` is read-only. If you want to write a fresh summary on Kaggle, copy the package to `/kaggle/working` or pass `--summary-path /kaggle/working/verify_summary.json`.",
            "",
            "## 11. How to use with regions training config",
            "Suggested config values:",
            "```yaml",
            "dataset:",
            "  subset: nslt300",
            "  num_classes: 300",
            f"  data_root: {kaggle_root}",
            "  active_regions: [left_hand, right_hand, face]",
            "  expected_shape: [3, 3, 64, 112, 112]",
            "  manifests:",
            f"    train: {kaggle_root}/manifests/nslt300_train.csv",
            f"    val: {kaggle_root}/manifests/nslt300_val.csv",
            f"    test: {kaggle_root}/manifests/nslt300_test.csv",
            "```",
            "",
            "## 12. Notes and limitations",
            "- This package does not include raw videos.",
            "- This package does not include model checkpoints.",
            "- This package does not include W&B logs or old training outputs.",
            "- This package keeps the all-regions setting and does not support hands-only packaging.",
        ]
    )
    path = package_root / "README.md"
    path.write_text(text, encoding="utf-8")
    return path


def _zip_directory(source_dir: Path, zip_path: Path, *, prefix: str) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with ZipFile(zip_path, mode="w", compression=ZIP_DEFLATED) as archive:
        for path in sorted(source_dir.rglob("*")):
            if path.is_dir():
                continue
            archive.write(path, arcname=f"{prefix}/{path.relative_to(source_dir).as_posix()}")


def _run_verify(script_path: Path, package_root: Path) -> dict[str, Any]:
    command = [sys.executable, str(script_path), "--package-root", str(package_root)]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    summary_path = package_root / "verify" / "verify_summary.json"
    if not summary_path.exists():
        raise PackagingError(f"Verify script did not write summary file: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if str(summary.get("status", "")).lower() != "pass":
        raise PackagingError(f"Verify script did not report pass: {summary}")
    return {
        "stdout": completed.stdout,
        "summary": summary,
    }


def _remove_targets(package_root: Path, zip_path: Path) -> None:
    if package_root.exists():
        shutil.rmtree(package_root)
    if zip_path.exists():
        zip_path.unlink()


def main() -> int:
    args = build_parser().parse_args()
    project_root = Path(__file__).resolve().parents[1]
    source_root = _resolve_under(project_root, args.source_root)
    output_root = _resolve_under(project_root, args.output_root)
    package_name = str(args.package_name).strip()
    if not package_name:
        raise PackagingError("package-name must not be empty.")

    package_root = output_root / package_name
    zip_path = output_root / f"{package_name}.zip"

    _ensure_exists(source_root, "source root")
    split_stats = {split: _validate_split(source_root, split) for split in SPLITS}
    reports = _collect_reports(source_root)

    if args.clean:
        _remove_targets(package_root, zip_path)
    elif package_root.exists():
        raise PackagingError(
            f"Package folder already exists: {package_root}. Re-run with --clean to rebuild it."
        )
    elif args.create_zip and zip_path.exists():
        raise PackagingError(
            f"Zip file already exists: {zip_path}. Re-run with --clean to rebuild it."
        )

    _ensure_dir(output_root)
    branch_root = package_root / "regions" / "rtmw_l"
    manifests_root = branch_root / "manifests"
    tensors_root = branch_root / "tensors" / "nslt300"
    reports_root = branch_root / "reports"

    _ensure_dir(manifests_root)
    _ensure_dir(tensors_root)
    _ensure_dir(reports_root)

    for split in SPLITS:
        source_manifest = split_stats[split].manifest_path
        fieldnames, rows = _read_manifest(source_manifest)
        sanitized_rows = _sanitize_rows(fieldnames, rows, split)
        _write_manifest(manifests_root / source_manifest.name, fieldnames, sanitized_rows)
        copied_count = _materialize_tensor_split(split_stats[split].tensor_dir, tensors_root / split)
        if copied_count != split_stats[split].tensor_files:
            raise PackagingError(
                f"Copied tensor count mismatch for split={split}: expected {split_stats[split].tensor_files}, got {copied_count}"
            )

    copied_reports: list[str] = []
    for path in reports:
        destination = reports_root / path.name
        shutil.copy2(path, destination)
        copied_reports.append(path.name)

    readme_path = _write_readme(package_root)
    metadata_path = _write_metadata(package_root, split_stats)
    verify_script_path = _write_verify_script(package_root)
    verify_result = _run_verify(verify_script_path, package_root)

    if args.create_zip:
        _zip_directory(package_root, zip_path, prefix=package_name)

    package_file_count = _count_files(package_root)
    zip_size = zip_path.stat().st_size if zip_path.exists() else 0
    result = {
        "package_path": str(package_root),
        "zip_path": str(zip_path) if zip_path.exists() else None,
        "zip_size_bytes": zip_size if zip_path.exists() else None,
        "zip_size_human": _format_size(zip_size) if zip_path.exists() else None,
        "file_count": package_file_count,
        "manifest_counts": {split: split_stats[split].manifest_rows for split in SPLITS},
        "tensor_counts": {split: split_stats[split].tensor_files for split in SPLITS},
        "expected_shape": list(EXPECTED_SHAPE),
        "active_regions": list(ACTIVE_REGIONS),
        "reports_copied": copied_reports,
        "verify_result": verify_result["summary"].get("status"),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
