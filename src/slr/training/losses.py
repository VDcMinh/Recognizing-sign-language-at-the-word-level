"""Training loss helpers."""

from __future__ import annotations

from typing import Any

import torch.nn as nn


def describe_loss(name: str) -> str:
    """Return a short human-readable description of the configured loss."""

    return f"Configured loss: {name}"


def build_loss(name: str, config: dict[str, Any] | None = None):
    """Build a training loss module."""

    del config
    loss_name = str(name).strip().lower()
    if loss_name == "cross_entropy":
        return nn.CrossEntropyLoss()
    raise ValueError(f"Unsupported loss {loss_name!r}. Expected 'cross_entropy'.")


__all__ = ["build_loss", "describe_loss"]
