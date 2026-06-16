"""Verify an incremental NSLT1000 Kaggle package folder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REQUIRED_FILES = (
    "README.md",
    "metadata.json",
    "scripts/materialize_regions_nslt1000_kaggle_manifests.py",
    "scripts/verify_incremental_package.py",
    "manifests/logical/nslt1000_train.csv",
    "manifests/logical/nslt1000_val.csv",
    "manifests/logical/nslt1000_test.csv",
    "manifests/missing/nslt1000_missing_train.csv",
    "manifests/missing/nslt1000_missing_val.csv",
    "manifests/missing/nslt1000_missing_test.csv",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify one incremental NSLT1000 package folder.")
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--summary-path", type=Path, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    package_root = args.package_root.resolve()
    missing = [path for path in REQUIRED_FILES if not (package_root / path).exists()]
    logical_counts = {}
    for split in ("train", "val", "test"):
        logical_path = package_root / "manifests" / "logical" / f"nslt1000_{split}.csv"
        if logical_path.exists():
            logical_counts[split] = int(len(pd.read_csv(logical_path)))

    status = "pass" if not missing else "fail"
    summary = {
        "status": status,
        "package_root": str(package_root).replace("\\", "/"),
        "missing_required_files": missing,
        "logical_counts": logical_counts,
    }
    summary_path = args.summary_path or (package_root / "verify" / "verify_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
