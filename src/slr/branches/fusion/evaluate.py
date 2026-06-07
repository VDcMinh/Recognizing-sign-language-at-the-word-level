"""Evaluation entrypoint for gated skeleton-regions feature fusion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from slr.branches.fusion import build_gated_feature_fusion_from_config
from slr.branches.fusion.train import (
    load_fusion_checkpoint,
    resolve_training_config,
    run_one_epoch,
    select_device,
    build_paired_dataloaders,
    build_paired_datasets,
)
from slr.training.losses import build_loss_from_config
from slr.utils.io import ensure_dir, write_json
from slr.utils.logging import setup_logger


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for gated fusion evaluation."""

    parser = argparse.ArgumentParser(
        description="Evaluate a trained gated fusion checkpoint on train/val/test paired inputs."
    )
    parser.add_argument("--config", type=Path, required=True, help="Fusion training config YAML.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Fusion checkpoint to load.")
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate.",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Override dataloader.batch_size.")
    parser.add_argument("--device", type=str, default=None, help="Override train.device.")
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Limit the number of evaluated batches for quick checks.",
    )
    return parser


def _evaluation_output_path(checkpoint_path: Path, split: str) -> Path:
    return checkpoint_path.resolve().parent / f"eval_{split}.json"


def run_evaluation(
    *,
    config_path: Path,
    checkpoint_path: Path,
    split: str,
    batch_size: int | None = None,
    device_name: str | None = None,
    max_batches: int | None = None,
) -> dict[str, Any]:
    """Evaluate one gated fusion checkpoint on a selected split."""

    overrides: dict[str, Any] = {}
    if batch_size is not None:
        overrides["batch_size"] = int(batch_size)
    if device_name is not None:
        overrides["device"] = str(device_name)
    resolved_config = resolve_training_config(config_path, overrides)

    logger = setup_logger("slr.branches.fusion.evaluate")
    device = select_device(str(resolved_config["train"]["device"]))
    datasets = build_paired_datasets(resolved_config, splits=(split,))
    dataloaders = build_paired_dataloaders(resolved_config, datasets, device=device)
    model, build_info = build_gated_feature_fusion_from_config(resolved_config, device=device)
    criterion = build_loss_from_config(resolved_config)
    payload = load_fusion_checkpoint(checkpoint_path, model, map_location=device)

    metrics = run_one_epoch(
        model=model,
        loader=dataloaders[split],
        criterion=criterion,
        device=device,
        config=resolved_config,
        amp_enabled=bool(resolved_config["train"].get("amp", False)) and device.type == "cuda",
        max_batches=max_batches,
        collect_gate_stats=True,
    )
    result = {
        "split": str(split),
        "checkpoint": str(checkpoint_path.as_posix()),
        "epoch": int(payload.get("epoch", 0)),
        "num_samples": int(metrics["num_samples"]),
        "loss": float(metrics["loss"]),
        "top1": float(metrics["top1"]),
        "top5": float(metrics["top5"]),
        "top10": float(metrics["top10"]),
        "gate_mean": float(metrics["val/gate_mean"]),
        "gate_std": float(metrics["val/gate_std"]),
        "gate_min": float(metrics["val/gate_min"]),
        "gate_max": float(metrics["val/gate_max"]),
        "build_info": build_info,
    }

    output_path = _evaluation_output_path(checkpoint_path, split)
    ensure_dir(output_path.parent)
    write_json(result, output_path)
    logger.info(
        "Evaluation split=%s loss=%.4f top1=%.4f top5=%.4f top10=%.4f gate_mean=%.4f gate_std=%.4f",
        split,
        result["loss"],
        result["top1"],
        result["top5"],
        result["top10"],
        result["gate_mean"],
        result["gate_std"],
    )
    return result


def main() -> int:
    """CLI entrypoint for gated fusion evaluation."""

    parser = build_parser()
    args = parser.parse_args()
    result = run_evaluation(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        split=args.split,
        batch_size=args.batch_size,
        device_name=args.device,
        max_batches=args.max_batches,
    )
    print(json.dumps(result, indent=2))
    return 0


__all__ = ["build_parser", "main", "run_evaluation"]
