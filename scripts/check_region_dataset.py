"""Sanity-check the local-image region dataset loader."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader

from slr.branches.regions.dataset import RegionClipDataset, region_collate_fn


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(
        description="Check region dataset loading and tensor shapes."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to one region manifest CSV file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=4,
        help="Optional sample limit to keep the sanity check lightweight.",
    )
    return parser


def main() -> int:
    """Run the region dataset sanity checks."""

    parser = build_parser()
    args = parser.parse_args()

    dataset = RegionClipDataset(
        manifest_path=args.manifest,
        limit=args.limit,
        strict_shape_check=True,
        return_metadata=True,
    )
    manifest = pd.read_csv(args.manifest)

    print("== Dataset ==")
    print(f"manifest path: {args.manifest}")
    print(f"resolved samples: {len(dataset)}")
    print(f"expected_shape: {tuple(dataset.expected_shape)}")
    print()

    sample = dataset[0]
    print("== First Sample ==")
    print(f"sample_id: {sample['sample_id']}")
    print(f"gloss: {sample['gloss']}")
    print(f"label: {sample['label']}")
    print(f"data shape: {tuple(sample['data'].shape)}")
    if sample["valid_mask"] is not None:
        print(f"valid_mask shape: {tuple(sample['valid_mask'].shape)}")
        print(f"valid ratio summary: {float(sample['valid_mask'].float().mean()):.6f}")
    print(f"preview path: {sample['preview_path'] or '<empty>'}")
    print()

    class_ids = manifest.loc[manifest["status"] == "ok", "class_id"].astype(int)
    print("== Label Stats ==")
    print(f"class_id min/max/nunique: {int(class_ids.min())} / {int(class_ids.max())} / {int(class_ids.nunique())}")
    print()

    valid_ratio_columns = ["left_hand_valid_ratio", "right_hand_valid_ratio", "face_valid_ratio"]
    print("== Valid Ratio Summary ==")
    for column in valid_ratio_columns:
        values = pd.to_numeric(
            manifest.loc[manifest["status"] == "ok", column],
            errors="coerce",
        )
        print(f"{column}: mean={float(values.mean()):.6f} min={float(values.min()):.6f} max={float(values.max()):.6f}")
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
    print()

    print("Sanity check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
