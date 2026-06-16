"""Smoke test the gated feature fusion dataset and model wiring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from slr.branches.fusion import (
    PairedSkeletonRegionsDataset,
    build_gated_feature_fusion_from_config,
    load_gated_feature_fusion_config,
    paired_skeleton_regions_collate_fn,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check the first three setup steps for gated feature fusion."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Fusion config YAML.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Dataset split to inspect.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=4,
        help="Maximum matched samples to load from the chosen split.",
    )
    return parser


def _print_shape(label: str, value: torch.Tensor) -> None:
    print(f"{label}: {tuple(value.shape)}")


def main() -> int:
    args = build_parser().parse_args()
    resolved = load_gated_feature_fusion_config(args.config)
    dataset = PairedSkeletonRegionsDataset.from_config(
        resolved,
        split=args.split,
        limit=args.limit,
    )
    print("Alignment report:")
    print(json.dumps(dataset.get_alignment_report(), indent=2))

    dataloader_cfg = resolved.get("dataloader", {})
    loader = DataLoader(
        dataset,
        batch_size=min(int(dataloader_cfg.get("batch_size", 4)), len(dataset)),
        shuffle=False,
        num_workers=int(dataloader_cfg.get("num_workers", 0)),
        pin_memory=bool(dataloader_cfg.get("pin_memory", False)) and torch.cuda.is_available(),
        collate_fn=paired_skeleton_regions_collate_fn,
    )
    batch = next(iter(loader))
    _print_shape("skeleton", batch["skeleton"])
    _print_shape("regions", batch["regions"])
    if batch.get("regions_valid_mask") is not None:
        _print_shape("regions_valid_mask", batch["regions_valid_mask"])

    model, info = build_gated_feature_fusion_from_config(resolved)
    print("Branch/build info:")
    print(json.dumps(info, indent=2))

    device = next(model.parameters()).device
    skeleton_x = batch["skeleton"].to(device)
    regions_x = batch["regions"].to(device)
    regions_valid_mask = batch.get("regions_valid_mask")
    if regions_valid_mask is not None:
        regions_valid_mask = regions_valid_mask.to(device)

    model.eval()
    with torch.no_grad():
        logits, features = model(
            skeleton_x,
            regions_x,
            return_features=True,
            regions_valid_mask=regions_valid_mask,
        )

    _print_shape("skeleton_feature", features["skeleton_feature"])
    _print_shape("region_feature", features["region_feature"])
    _print_shape("gate", features["gate"])
    _print_shape("fused", features["fused"])
    _print_shape("logits", logits)

    expected_shape = (int(batch["labels"].shape[0]), int(resolved["dataset"]["num_classes"]))
    actual_shape = tuple(int(value) for value in logits.shape)
    if actual_shape != expected_shape:
        raise ValueError(
            f"logits shape mismatch: expected {expected_shape}, got {actual_shape}."
        )
    print(f"logits shape verified: {actual_shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
