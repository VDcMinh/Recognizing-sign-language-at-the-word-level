"""Verify a packaged NSLT100 branch-inputs bundle."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify that a packaged WLASL nslt100 branch-inputs bundle is fusion-ready."
    )
    parser.add_argument("--package-root", type=Path, required=True, help="Path to packaging_outputs/wlasl-nslt100-branch-inputs.")
    parser.add_argument(
        "--skeleton-keypoint-set",
        type=str,
        default=None,
        help="Optional expected skeleton keypoint set, for example selected_31.",
    )
    parser.add_argument("--sample-checks", type=int, default=3, help="How many manifest rows to sanity-check per split.")
    return parser


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [{key: value or "" for key, value in row.items()} for row in reader]


def resolve_package_path(root: Path, value: str) -> Path:
    raw = Path(value)
    return raw if raw.is_absolute() else (root / raw)


def check_int(value: str, label: str) -> None:
    try:
        int(float(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not readable as int: {value!r}") from exc


def verify_skeleton(package_root: Path, metadata: dict[str, Any], sample_checks: int) -> dict[str, Any]:
    branch_root = package_root / "skeleton" / "rtmw_l"
    keypoint_set = str(metadata["keypoint_set"])
    expected_shape = tuple(int(value) for value in metadata.get("graph_tensor_shape") or [3, 150, 27, 1])
    splits = list(metadata["splits"])
    tensor_counts: dict[str, int] = {}
    examples: dict[str, list[str]] = {}

    for split in splits:
        manifest_path = branch_root / "manifests" / f"{metadata['subset']}_{keypoint_set}_{split}.csv"
        tensor_dir = branch_root / "tensors" / metadata["subset"] / split
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing skeleton manifest: {manifest_path.as_posix()}")
        if not tensor_dir.exists():
            raise FileNotFoundError(f"Missing skeleton tensor directory: {tensor_dir.as_posix()}")

        rows = read_manifest(manifest_path)
        files = sorted(tensor_dir.glob("*.npz"))
        tensor_counts[split] = len(files)
        if len(rows) != len(files):
            raise ValueError(
                f"Skeleton split={split} manifest rows ({len(rows)}) do not match tensor files ({len(files)})."
            )

        checked_ids: list[str] = []
        for row in rows[:sample_checks]:
            check_int(row.get("class_id", ""), "skeleton class_id")
            tensor_path = resolve_package_path(branch_root, row.get("graph_tensor_path", ""))
            if not tensor_path.exists():
                raise FileNotFoundError(f"Missing skeleton tensor referenced by manifest: {tensor_path.as_posix()}")
            with np.load(tensor_path, allow_pickle=False) as payload:
                array = payload["data"] if "data" in payload else payload["tensor"]
                if tuple(int(value) for value in array.shape) != expected_shape:
                    raise ValueError(
                        f"Skeleton tensor shape mismatch for {tensor_path.name}: "
                        f"expected {expected_shape}, got {tuple(array.shape)}"
                    )
            checked_ids.append(str(row.get("sample_id", "")))
        examples[split] = checked_ids

    return {
        "keypoint_set": keypoint_set,
        "expected_shape": list(expected_shape),
        "tensor_counts": tensor_counts,
        "sample_checks": examples,
    }


def verify_regions(package_root: Path, metadata: dict[str, Any], sample_checks: int) -> dict[str, Any]:
    branch_root = package_root / "regions" / "rtmw_l"
    expected_shape = tuple(int(value) for value in metadata.get("tensor_shape") or [3, 3, 64, 112, 112])
    splits = list(metadata["splits"])
    tensor_counts: dict[str, int] = {}
    examples: dict[str, list[str]] = {}

    for split in splits:
        manifest_path = branch_root / "manifests" / f"{metadata['subset']}_{split}.csv"
        tensor_dir = branch_root / "tensors" / metadata["subset"] / split
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing regions manifest: {manifest_path.as_posix()}")
        if not tensor_dir.exists():
            raise FileNotFoundError(f"Missing regions tensor directory: {tensor_dir.as_posix()}")

        rows = read_manifest(manifest_path)
        files = sorted(tensor_dir.glob("*.npz"))
        tensor_counts[split] = len(files)
        if len(rows) != len(files):
            raise ValueError(
                f"Regions split={split} manifest rows ({len(rows)}) do not match tensor files ({len(files)})."
            )

        checked_ids: list[str] = []
        for row in rows[:sample_checks]:
            check_int(row.get("class_id", ""), "regions class_id")
            tensor_path = resolve_package_path(branch_root, row.get("tensor_path", ""))
            if not tensor_path.exists():
                raise FileNotFoundError(f"Missing regions tensor referenced by manifest: {tensor_path.as_posix()}")
            with np.load(tensor_path, allow_pickle=False) as payload:
                array = payload["data"]
                if tuple(int(value) for value in array.shape) != expected_shape:
                    raise ValueError(
                        f"Regions tensor shape mismatch for {tensor_path.name}: "
                        f"expected {expected_shape}, got {tuple(array.shape)}"
                    )
            checked_ids.append(str(row.get("sample_id", "")))
        examples[split] = checked_ids

    return {
        "regions": list(metadata.get("regions", [])),
        "expected_shape": list(expected_shape),
        "tensor_counts": tensor_counts,
        "sample_checks": examples,
    }


def main() -> int:
    args = build_parser().parse_args()
    package_root = args.package_root.resolve()
    package_metadata_path = package_root / "metadata.json"
    if not package_metadata_path.exists():
        raise FileNotFoundError(f"Missing package metadata: {package_metadata_path.as_posix()}")

    with package_metadata_path.open("r", encoding="utf-8") as handle:
        package_metadata = json.load(handle)

    skeleton_metadata_path = package_root / "skeleton" / "rtmw_l" / "metadata.json"
    regions_metadata_path = package_root / "regions" / "rtmw_l" / "metadata.json"
    with skeleton_metadata_path.open("r", encoding="utf-8") as handle:
        skeleton_metadata = json.load(handle)
    with regions_metadata_path.open("r", encoding="utf-8") as handle:
        regions_metadata = json.load(handle)

    expected_keypoint_set = (
        str(args.skeleton_keypoint_set).strip() if args.skeleton_keypoint_set is not None else None
    )
    if expected_keypoint_set is not None and str(skeleton_metadata.get("keypoint_set", "")).strip() != expected_keypoint_set:
        raise ValueError(
            f"Skeleton keypoint set mismatch: package has {skeleton_metadata.get('keypoint_set')!r}, "
            f"expected {expected_keypoint_set!r}."
        )

    skeleton_result = verify_skeleton(package_root, skeleton_metadata, int(args.sample_checks))
    regions_result = verify_regions(package_root, regions_metadata, int(args.sample_checks))

    subset = str(package_metadata["subset"])
    package_name = str(package_metadata.get("package_name") or package_root.name)
    skeleton_keypoint_set = str(skeleton_result["keypoint_set"])
    kaggle_root = f"/kaggle/working/{package_name}"
    result = {
        "status": "pass",
        "package_root": package_root.as_posix(),
        "package_name": package_name,
        "skeleton": skeleton_result,
        "regions": regions_result,
        "kaggle_paths": {
            "SKELETON_DATA_ROOT": f"{kaggle_root}/skeleton/rtmw_l",
            "SKELETON_VAL_MANIFEST": f"{kaggle_root}/skeleton/rtmw_l/manifests/{subset}_{skeleton_keypoint_set}_val.csv",
            "SKELETON_TEST_MANIFEST": f"{kaggle_root}/skeleton/rtmw_l/manifests/{subset}_{skeleton_keypoint_set}_test.csv",
            "REGIONS_DATA_ROOT": f"{kaggle_root}/regions/rtmw_l",
            "REGIONS_VAL_MANIFEST": f"{kaggle_root}/regions/rtmw_l/manifests/{subset}_val.csv",
            "REGIONS_TEST_MANIFEST": f"{kaggle_root}/regions/rtmw_l/manifests/{subset}_test.csv",
        },
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
