"""Package incremental NSLT1000 region tensors for Kaggle without duplicating NSLT300."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd

from scripts.common.regions_nslt1000_incremental_common import (
    ALLOWED_SPLITS,
    DEFAULT_INCREMENTAL_ROOT,
    DEFAULT_PACKAGE_NAME,
    DEFAULT_PACKAGE_ROOT,
    DEFAULT_UNION_ROOT,
    LOGICAL_MANIFEST_COLUMNS,
    format_size,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Package incremental NSLT1000 region tensors for Kaggle."
    )
    parser.add_argument("--incremental-root", type=Path, default=DEFAULT_INCREMENTAL_ROOT)
    parser.add_argument("--union-root", type=Path, default=DEFAULT_UNION_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_PACKAGE_ROOT)
    parser.add_argument("--package-name", type=str, default=DEFAULT_PACKAGE_NAME)
    parser.add_argument("--link-mode", type=str, choices=("hardlink", "copy"), default="hardlink")
    parser.add_argument("--zip", action="store_true")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _load_union_verify_status() -> dict:
    summary_path = Path("reports/current/regions/nslt1000_incremental_pipeline/union_verify_summary.json")
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing union verify summary: {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _ensure_ready_for_packaging() -> None:
    summary = _load_union_verify_status()
    if str(summary.get("status", "")).strip().upper() != "VERIFY PASS":
        raise RuntimeError(
            "Union verify has not passed, so packaging is blocked. "
            f"Current status: {summary.get('status')!r}"
        )


def _remove_target(path: Path) -> None:
    if path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def _link_or_copy(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    if mode == "hardlink":
        try:
            os.link(source, destination)
            return
        except OSError:
            shutil.copy2(source, destination)
            return
    shutil.copy2(source, destination)


def _build_logical_manifest(union_path: Path, destination: Path) -> int:
    frame = pd.read_csv(union_path)
    rows = []
    for _, row in frame.iterrows():
        tensor_path = Path(str(row["tensor_path"]).replace("\\", "/"))
        split = str(row["split"]).strip()
        if str(row["reuse_source"]).strip() == "nslt300":
            tensor_source = "nslt300_base"
            tensor_relpath = Path("tensors") / "nslt300" / split / tensor_path.name
        else:
            tensor_source = "nslt1000_incremental"
            tensor_relpath = Path("regions") / "rtmw_l_incremental" / "tensors" / "nslt1000" / split / tensor_path.name
        rows.append(
            {
                "sample_id": row["sample_id"],
                "video_id": row["video_id"],
                "class_id": row["class_id"],
                "gloss": row["gloss"],
                "split": row["split"],
                "tensor_source": tensor_source,
                "tensor_relpath": str(tensor_relpath).replace("\\", "/"),
                "tensor_shape": row["tensor_shape"],
                "region_order": row["region_order"],
                "status": row["status"],
            }
        )
    logical = pd.DataFrame(rows)
    destination.parent.mkdir(parents=True, exist_ok=True)
    logical.loc[:, list(LOGICAL_MANIFEST_COLUMNS)].to_csv(destination, index=False, encoding="utf-8")
    return int(len(logical))


def _copy_verify_scripts(package_root: Path) -> None:
    scripts_root = package_root / "scripts"
    scripts_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        "scripts/preprocess/materialize_regions_nslt1000_kaggle_manifests.py",
        scripts_root / "materialize_regions_nslt1000_kaggle_manifests.py",
    )
    shutil.copy2(
        "scripts/verify/verify_regions_nslt1000_incremental_package.py",
        scripts_root / "verify_incremental_package.py",
    )


def _write_readme(package_root: Path, package_name: str) -> None:
    readme = "\n".join(
        [
            f"# {package_name}",
            "",
            "Incremental NSLT1000 regions package.",
            "",
            "This package does not duplicate NSLT300 tensors.",
            "Attach both the NSLT300 base dataset and this incremental dataset on Kaggle.",
            "",
            "Important:",
            "- The `.npz` tensors are already compressed, so `--zip` may not reduce size much.",
            "- Materialize runtime manifests under `/kaggle/working` before training.",
            "",
        ]
    )
    (package_root / "README.md").write_text(readme, encoding="utf-8")


def _write_metadata(package_root: Path, counts: dict[str, int]) -> None:
    metadata = {
        "package_name": package_root.name,
        "subset": "nslt1000",
        "num_classes": 1000,
        "purpose": "Missing-only Regions tensors for incremental NSLT1000 construction",
        "format": "incremental_two_source",
        "base_subset": "nslt300",
        "base_dataset_required": True,
        "active_regions": ["left_hand", "right_hand", "face"],
        "expected_shape": [3, 3, 64, 112, 112],
        "counts": counts,
    }
    (package_root / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_package_report(package_root: Path, counts: dict[str, int]) -> None:
    text = "\n".join(
        [
            "# Incremental Package Report",
            "",
            "Package folder was built successfully.",
            "",
            f"- total_nslt1000: `{counts['total_nslt1000']}`",
            f"- reused_nslt300: `{counts['reused_nslt300']}`",
            f"- incremental_new: `{counts['incremental_new']}`",
            "",
        ]
    )
    reports_root = package_root / "reports"
    reports_root.mkdir(parents=True, exist_ok=True)
    (reports_root / "package_report.md").write_text(text, encoding="utf-8")


def _zip_directory(source_dir: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with ZipFile(zip_path, mode="w", compression=ZIP_DEFLATED) as archive:
        for path in sorted(source_dir.rglob("*")):
            if path.is_dir():
                continue
            archive.write(path, arcname=f"{source_dir.name}/{path.relative_to(source_dir).as_posix()}")


def main() -> int:
    args = build_parser().parse_args()
    _ensure_ready_for_packaging()

    package_root = args.output_root / args.package_name
    zip_path = args.output_root / f"{args.package_name}.zip"
    if args.verify_only:
        summary = _load_union_verify_status()
        print(json.dumps({"status": "ready_for_packaging", "union_verify": summary.get("status")}, indent=2))
        return 0

    if args.clean:
        _remove_target(package_root)
        _remove_target(zip_path)
    elif package_root.exists():
        raise FileExistsError(f"Package output already exists: {package_root}. Re-run with --clean.")

    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "dry_run",
                    "package_root": str(package_root).replace("\\", "/"),
                    "zip_path": str(zip_path).replace("\\", "/") if args.zip else None,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    package_root.mkdir(parents=True, exist_ok=True)
    manifests_root = package_root / "manifests"
    logical_root = manifests_root / "logical"
    missing_root = manifests_root / "missing"
    incremental_tensor_root = package_root / "regions" / "rtmw_l_incremental" / "tensors" / "nslt1000"
    config_root = package_root / "configs"
    verify_root = package_root / "verify"
    reports_root = package_root / "reports"
    for path in (logical_root, missing_root, incremental_tensor_root, config_root, verify_root, reports_root):
        path.mkdir(parents=True, exist_ok=True)

    total_union_rows = 0
    incremental_new = 0
    train_new = 0
    val_new = 0
    test_new = 0
    for split in ALLOWED_SPLITS:
        union_path = args.union_root / "manifests" / f"nslt1000_{split}.csv"
        missing_path = args.incremental_root / "manifests" / f"nslt1000_missing_{split}.csv"
        total_union_rows += _build_logical_manifest(union_path, logical_root / f"nslt1000_{split}.csv")
        shutil.copy2(missing_path, missing_root / missing_path.name)
        missing_frame = pd.read_csv(missing_path)
        split_new_count = int((missing_frame.get("status", pd.Series("", index=missing_frame.index)).astype(str).str.lower() == "ok").sum())
        if split == "train":
            train_new = split_new_count
        elif split == "val":
            val_new = split_new_count
        else:
            test_new = split_new_count
        incremental_new += split_new_count
        source_dir = args.incremental_root / "tensors" / "nslt1000" / split
        dest_dir = incremental_tensor_root / split
        dest_dir.mkdir(parents=True, exist_ok=True)
        for tensor_path in sorted(source_dir.glob("*.npz")):
            _link_or_copy(tensor_path, dest_dir / tensor_path.name, args.link_mode)

    counts = {
        "total_nslt1000": total_union_rows,
        "reused_nslt300": total_union_rows - incremental_new,
        "incremental_new": incremental_new,
        "train_new": train_new,
        "val_new": val_new,
        "test_new": test_new,
    }
    _write_readme(package_root, args.package_name)
    _write_metadata(package_root, counts)
    _copy_verify_scripts(package_root)
    shutil.copy2(
        "configs/train/regions/nslt1000/incremental/region_resnet18_gru_incremental_kaggle_ce.yaml.template",
        config_root / "region_resnet18_gru_incremental_kaggle_ce.yaml.template",
    )
    extraction_summary = Path("reports/current/regions/nslt1000_incremental_pipeline/progress_summary.json")
    union_verify_summary = Path("reports/current/regions/nslt1000_incremental_pipeline/union_verify_summary.json")
    if extraction_summary.exists():
        shutil.copy2(extraction_summary, reports_root / "extraction_summary.json")
    if union_verify_summary.exists():
        shutil.copy2(union_verify_summary, reports_root / "union_verify_summary.json")
    _write_package_report(package_root, counts)

    verify_script = package_root / "scripts" / "verify_incremental_package.py"
    subprocess.run([sys.executable, str(verify_script), "--package-root", str(package_root)], check=True)
    if args.zip:
        _zip_directory(package_root, zip_path)

    print(
        json.dumps(
            {
                "status": "ok",
                "package_root": str(package_root).replace("\\", "/"),
                "zip_path": str(zip_path).replace("\\", "/") if args.zip else None,
                "counts": counts,
                "zip_size_human": format_size(zip_path.stat().st_size) if args.zip and zip_path.exists() else None,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
