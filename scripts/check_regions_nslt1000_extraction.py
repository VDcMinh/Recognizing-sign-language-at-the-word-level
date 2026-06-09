"""Verify nslt1000 regions extraction outputs across train/val/test."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader

from slr.branches.regions.dataset import RegionClipDataset, region_collate_fn
from slr.branches.regions.region_schema import REGION_NAMES


ALLOWED_SPLITS = ("train", "val", "test")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""

    parser = argparse.ArgumentParser(
        description="Verify nslt1000 region extraction manifests and tensors."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Root folder containing manifests/ and tensors/ for the regions branch.",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="nslt1000",
        help="Subset name to verify.",
    )
    parser.add_argument(
        "--expected-shape",
        type=str,
        default="3,3,64,112,112",
        help="Comma-separated expected tensor shape.",
    )
    parser.add_argument(
        "--active-regions",
        type=str,
        default="left_hand,right_hand,face",
        help="Comma-separated active region order.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=1000,
        help="Expected class count.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=4,
        help="Maximum number of samples per split to load during verification.",
    )
    return parser


def _parse_csv_list(value: str) -> list[str]:
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _parse_expected_shape(value: str) -> tuple[int, ...]:
    shape = tuple(int(part.strip()) for part in str(value).split(",") if part.strip())
    if len(shape) != 5:
        raise ValueError(f"expected_shape must contain 5 integers, got {shape}.")
    return shape


def _ensure(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _contains_other_subset_reference(frame: pd.DataFrame) -> bool:
    if frame.empty:
        return False
    text = frame.astype(str).agg(" ".join, axis=1)
    return bool(text.str.contains(r"nslt100\b|nslt300\b", case=False, regex=True).any())


def verify_split(
    *,
    data_root: Path,
    subset: str,
    split: str,
    expected_shape: tuple[int, ...],
    active_regions: list[str],
    num_classes: int,
    limit: int,
) -> dict[str, object]:
    """Verify one split manifest plus a few dataset loads."""

    manifest_path = data_root / "manifests" / f"{subset}_{split}.csv"
    tensor_dir = data_root / "tensors" / subset / split

    _ensure(manifest_path.exists(), f"Missing manifest: {manifest_path}")
    _ensure(tensor_dir.exists(), f"Missing tensor directory: {tensor_dir}")

    manifest = pd.read_csv(manifest_path)
    _ensure(not _contains_other_subset_reference(manifest), f"Unexpected nslt100/nslt300 reference in {manifest_path}")
    _ensure(set(manifest["split"].dropna().astype(str).str.lower().unique()).issubset({split}), f"Split mismatch inside {manifest_path}")

    class_ids = pd.to_numeric(manifest["class_id"], errors="coerce").dropna()
    _ensure(not class_ids.empty, f"No class_id values found in {manifest_path}")
    _ensure(int(class_ids.min()) >= 0, f"class_id min < 0 in {manifest_path}")
    _ensure(int(class_ids.max()) <= num_classes - 1, f"class_id max > {num_classes - 1} in {manifest_path}")

    dataset = RegionClipDataset(
        manifest_path=manifest_path,
        data_root=data_root,
        split=split,
        expected_shape=expected_shape,
        num_classes=num_classes,
        region_order=REGION_NAMES,
        active_regions=active_regions,
        return_metadata=True,
        strict_shape_check=True,
        limit=limit,
    )
    _ensure(len(dataset) > 0, f"No loadable ok samples found for {split}.")

    sample = dataset[0]
    _ensure(tuple(sample["data"].shape) == expected_shape, f"Unexpected sample shape for {split}: {tuple(sample['data'].shape)}")
    _ensure(sample.get("region_names") == active_regions, f"Unexpected region_names for {split}: {sample.get('region_names')}")
    _ensure(0 <= int(sample["class_id"]) <= num_classes - 1, f"Sample class_id out of range for {split}: {sample['class_id']}")

    loader = DataLoader(
        dataset,
        batch_size=min(max(1, limit), len(dataset)),
        shuffle=False,
        num_workers=0,
        collate_fn=region_collate_fn,
    )
    batch = next(iter(loader))
    batch_shape = tuple(batch["data"].shape)
    _ensure(batch_shape[1:] == expected_shape, f"Unexpected batch tail shape for {split}: {batch_shape}")

    ok_rows = manifest.loc[manifest["status"].astype(str).str.lower() == "ok"].copy()
    tensor_shape_values = sorted(set(ok_rows.get("tensor_shape", pd.Series(dtype="string")).dropna().astype(str).tolist()))
    return {
        "split": split,
        "manifest_path": str(manifest_path),
        "tensor_dir": str(tensor_dir),
        "rows": int(len(manifest)),
        "ok_rows": int(len(ok_rows)),
        "error_rows": int(len(manifest) - len(ok_rows)),
        "class_id_min": int(class_ids.min()),
        "class_id_max": int(class_ids.max()),
        "dataset_samples": int(len(dataset)),
        "sample_shape": tuple(sample["data"].shape),
        "batch_shape": batch_shape,
        "region_names": list(sample.get("region_names", [])),
        "tensor_shape_values": tensor_shape_values,
    }


def main() -> int:
    """CLI entrypoint."""

    parser = build_parser()
    args = parser.parse_args()

    active_regions = _parse_csv_list(args.active_regions)
    expected_shape = _parse_expected_shape(args.expected_shape)
    _ensure(active_regions == list(REGION_NAMES), f"Expected active_regions {list(REGION_NAMES)}, got {active_regions}")

    print("== Regions Extraction Verification ==")
    print(f"data_root: {args.data_root}")
    print(f"subset: {args.subset}")
    print(f"expected_shape: {expected_shape}")
    print(f"active_regions: {active_regions}")
    print()

    split_summaries: list[dict[str, object]] = []
    for split in ALLOWED_SPLITS:
        summary = verify_split(
            data_root=args.data_root,
            subset=args.subset,
            split=split,
            expected_shape=expected_shape,
            active_regions=active_regions,
            num_classes=args.num_classes,
            limit=args.limit,
        )
        split_summaries.append(summary)
        print(f"== {split} ==")
        print(f"manifest: {summary['manifest_path']}")
        print(f"tensor_dir: {summary['tensor_dir']}")
        print(f"rows: {summary['rows']}")
        print(f"ok_rows: {summary['ok_rows']}")
        print(f"error_rows: {summary['error_rows']}")
        print(f"class_id min/max: {summary['class_id_min']} / {summary['class_id_max']}")
        print(f"dataset_samples_loaded: {summary['dataset_samples']}")
        print(f"sample_shape: {summary['sample_shape']}")
        print(f"batch_shape: {summary['batch_shape']}")
        print(f"region_names: {summary['region_names']}")
        print(f"manifest tensor_shape values: {summary['tensor_shape_values']}")
        print()

    print("Sanity check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
