"""Checkpoint save/load helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from slr.utils.io import ensure_dir


def build_checkpoint_path(output_dir: str | Path, filename: str) -> Path:
    """Resolve a checkpoint filename under the given output directory."""

    return Path(output_dir) / filename


def save_checkpoint(
    path: str | Path,
    *,
    epoch: int,
    model,
    optimizer=None,
    scheduler=None,
    best_metric: float | None,
    last_metrics: dict[str, Any],
    config: dict[str, Any],
    keypoint_set: str,
    num_classes: int,
    num_nodes: int,
    model_name: str,
    class_id_to_gloss: dict[int, str] | None = None,
) -> Path:
    """Serialize a training checkpoint using ``state_dict`` payloads only."""

    checkpoint_path = Path(path)
    ensure_dir(checkpoint_path.parent)
    payload = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "best_metric": None if best_metric is None else float(best_metric),
        "last_metrics": dict(last_metrics),
        "config": config,
        "keypoint_set": str(keypoint_set),
        "num_classes": int(num_classes),
        "num_nodes": int(num_nodes),
        "model_name": str(model_name),
        "class_id_to_gloss": dict(class_id_to_gloss or {}),
    }
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def load_checkpoint(
    path: str | Path,
    model,
    optimizer=None,
    scheduler=None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint and restore any provided module state."""

    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file does not exist: {checkpoint_path}")

    payload = torch.load(checkpoint_path, map_location=map_location)
    if "model_state_dict" not in payload:
        raise KeyError(f"Checkpoint {checkpoint_path} is missing 'model_state_dict'.")

    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None and payload.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None and payload.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    return payload


__all__ = ["build_checkpoint_path", "load_checkpoint", "save_checkpoint"]
