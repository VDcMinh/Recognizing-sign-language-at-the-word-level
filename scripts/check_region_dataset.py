"""Sanity-check the local-image region dataset loader."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader

from slr.branches.regions.dataset import RegionClipDataset, load_region_train_config, region_collate_fn
from slr.branches.regions.region_schema import (
    BBOX_SOURCE_BLACK_CROP_FAILED,
    BBOX_SOURCE_PREVIOUS_BBOX_FALLBACK,
    REGION_NAMES,
)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(
        description="Check region dataset loading, tensor shapes, and crop quality summaries."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional training config YAML. When provided, dataset settings are read from config.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Path to one region manifest CSV file.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Optional dataset root used to resolve relative tensor paths.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=4,
        help="Optional sample limit to keep the sanity check lightweight.",
    )
    parser.add_argument(
        "--show-low-quality",
        type=int,
        default=0,
        help="Print this many low-quality samples from the manifest ranking.",
    )
    parser.add_argument(
        "--active-regions",
        type=str,
        default=None,
        help="Optional comma-separated subset such as left_hand,right_hand.",
    )
    return parser


def _safe_mean(series: pd.Series) -> float:
    """Return a numeric mean with NaN-safe fallback."""

    numeric = pd.to_numeric(series, errors="coerce")
    return float(numeric.mean()) if not numeric.empty else 0.0


def _safe_min(series: pd.Series) -> int:
    """Return an integer min with fallback."""

    numeric = pd.to_numeric(series, errors="coerce").dropna()
    return int(numeric.min()) if not numeric.empty else 0


def _safe_max(series: pd.Series) -> int:
    """Return an integer max with fallback."""

    numeric = pd.to_numeric(series, errors="coerce").dropna()
    return int(numeric.max()) if not numeric.empty else 0


def main() -> int:
    """Run the region dataset sanity checks."""

    parser = build_parser()
    args = parser.parse_args()

    if args.config is None and args.manifest is None:
        raise ValueError("Either --config or --manifest must be provided.")

    if args.config is not None:
        resolved_config = load_region_train_config(args.config)
        if args.active_regions:
            resolved_config["dataset"]["active_regions"] = [
                part.strip() for part in str(args.active_regions).split(",") if part.strip()
            ]
        if args.data_root is not None:
            resolved_config["dataset"]["data_root"] = args.data_root
        manifest_path = Path(resolved_config["dataset"]["manifests"]["train"])
        dataset = RegionClipDataset.from_config(
            resolved_config,
            split="train",
            limit=args.limit,
            return_metadata=True,
        )
    else:
        active_regions = [part.strip() for part in str(args.active_regions).split(",") if part.strip()] if args.active_regions else None
        manifest_path = Path(args.manifest)
        dataset = RegionClipDataset(
            manifest_path=manifest_path,
            data_root=args.data_root,
            limit=args.limit,
            strict_shape_check=True,
            return_metadata=True,
            region_order=REGION_NAMES,
            active_regions=active_regions,
        )

    manifest = pd.read_csv(manifest_path)
    ok_manifest = manifest.loc[manifest["status"] == "ok"].copy()
    error_manifest = manifest.loc[manifest["status"] != "ok"].copy()

    print("== Dataset ==")
    print(f"manifest path: {manifest_path}")
    print(f"config path: {args.config or '<none>'}")
    print(f"data root: {args.data_root or '<none>'}")
    print(f"total samples: {len(manifest)}")
    print(f"ok samples: {len(ok_manifest)}")
    print(f"error samples: {len(error_manifest)}")
    print(f"resolved dataset samples: {len(dataset)}")
    print(f"expected_shape: {tuple(dataset.expected_shape)}")
    print(f"region_order: {list(dataset.region_order)}")
    print(f"active_regions: {list(dataset.active_regions)}")
    print()

    sample = dataset[0]
    print("== First Sample ==")
    print(f"sample_id: {sample['sample_id']}")
    print(f"gloss: {sample['gloss']}")
    print(f"label: {sample['label']}")
    print(f"data shape: {tuple(sample['data'].shape)}")
    print(f"region_names: {sample.get('region_names', [])}")
    if sample["valid_mask"] is not None:
        print(f"valid_mask shape: {tuple(sample['valid_mask'].shape)}")
    if sample.get("bbox_source") is not None:
        print(f"bbox_source shape: {tuple(sample['bbox_source'].shape)}")
        black_ratio = float((sample["bbox_source"] == BBOX_SOURCE_BLACK_CROP_FAILED).float().mean())
        previous_ratio = float((sample["bbox_source"] == BBOX_SOURCE_PREVIOUS_BBOX_FALLBACK).float().mean())
        print(f"bbox_source black ratio summary: {black_ratio:.6f}")
        print(f"bbox_source previous fallback summary: {previous_ratio:.6f}")
    print(f"preview path: {sample['preview_path'] or '<empty>'}")
    print()

    class_ids = ok_manifest["class_id"] if not ok_manifest.empty else pd.Series(dtype="int64")
    print("== Label Stats ==")
    print(f"class_id min/max: {_safe_min(class_ids)} / {_safe_max(class_ids)}")
    print(f"unique labels: {int(pd.to_numeric(class_ids, errors='coerce').nunique()) if not class_ids.empty else 0}")
    print()

    print("== Region Quality Summary ==")
    for region_name in dataset.active_regions:
        print(f"{region_name} valid ratio avg: {_safe_mean(ok_manifest[f'{region_name}_valid_ratio']):.6f}")
        print(
            f"{region_name} black crop ratio avg: "
            f"{_safe_mean(ok_manifest[f'{region_name}_black_crop_ratio']):.6f}"
        )
        print(
            f"{region_name} previous fallback ratio avg: "
            f"{_safe_mean(ok_manifest[f'{region_name}_previous_fallback_ratio']):.6f}"
        )
    print()

    loader = DataLoader(
        dataset,
        batch_size=min(max(1, args.limit), len(dataset)),
        shuffle=False,
        num_workers=0,
        collate_fn=region_collate_fn,
    )
    batch = next(iter(loader))
    print("== Batch ==")
    print(f"batch data shape: {tuple(batch['data'].shape)}")
    print(f"batch labels shape: {tuple(batch['labels'].shape)}")
    if "valid_mask" in batch:
        print(f"batch valid_mask shape: {tuple(batch['valid_mask'].shape)}")
    if "bbox_source" in batch:
        print(f"batch bbox_source shape: {tuple(batch['bbox_source'].shape)}")
    print()

    preview_paths = [path for path in ok_manifest.get("preview_path", pd.Series(dtype="string")).astype(str).tolist() if path]
    print("== Preview Paths ==")
    for preview_path in preview_paths[: min(5, len(preview_paths))]:
        print(preview_path)
    if not preview_paths:
        print("<none>")
    print()

    if args.show_low_quality > 0 and not manifest.empty:
        ranked = manifest.copy()
        ranked["_max_black_crop_ratio"] = ranked[
            ["left_hand_black_crop_ratio", "right_hand_black_crop_ratio", "face_black_crop_ratio"]
        ].apply(pd.to_numeric, errors="coerce").max(axis=1)
        ranked["_min_valid_ratio"] = ranked[
            ["left_hand_valid_ratio", "right_hand_valid_ratio", "face_valid_ratio"]
        ].apply(pd.to_numeric, errors="coerce").min(axis=1)
        ranked = ranked.sort_values(
            by=["status", "_max_black_crop_ratio", "_min_valid_ratio", "split", "sample_id"],
            ascending=[False, False, True, True, True],
            kind="stable",
        ).head(int(args.show_low_quality))

        print("== Low-Quality Samples ==")
        for _, row in ranked.iterrows():
            print(
                f"{row['sample_id']} split={row['split']} status={row['status']} "
                f"valid=({float(row['left_hand_valid_ratio']):.3f}, "
                f"{float(row['right_hand_valid_ratio']):.3f}, "
                f"{float(row['face_valid_ratio']):.3f}) "
                f"black=({float(row['left_hand_black_crop_ratio']):.3f}, "
                f"{float(row['right_hand_black_crop_ratio']):.3f}, "
                f"{float(row['face_black_crop_ratio']):.3f}) "
                f"preview={row.get('preview_path', '')}"
            )
        print()

    print("Sanity check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
