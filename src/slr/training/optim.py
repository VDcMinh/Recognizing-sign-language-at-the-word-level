"""Optimizer and scheduler configuration helpers."""

from __future__ import annotations

from typing import Any

import torch


def describe_optimizer(name: str, learning_rate: float) -> str:
    """Return a short optimizer summary string."""

    return f"Optimizer(name={name}, learning_rate={learning_rate})"


def build_optimizer(parameters, config: dict[str, Any]):
    """Build an optimizer from one resolved training config."""

    optimizer_name = str(config.get("optimizer", "adamw")).strip().lower()
    learning_rate = float(config.get("learning_rate", 1e-3))
    weight_decay = float(config.get("weight_decay", 0.0))

    if optimizer_name == "adamw":
        return torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    if optimizer_name == "adam":
        return torch.optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
    if optimizer_name == "sgd":
        momentum = float(config.get("momentum", 0.9))
        nesterov = bool(config.get("nesterov", True))
        return torch.optim.SGD(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
        )
    raise ValueError(
        f"Unsupported optimizer {optimizer_name!r}. Expected one of: adamw, adam, sgd."
    )


def build_scheduler(optimizer, config: dict[str, Any], *, epochs: int):
    """Build an optional scheduler from one resolved training config."""

    if not bool(config.get("enabled", False)):
        return None

    scheduler_name = str(config.get("name", "cosine")).strip().lower()
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, int(epochs)),
            eta_min=float(config.get("min_lr", 1e-6)),
        )
    if scheduler_name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=max(1, int(config.get("step_size", 10))),
            gamma=float(config.get("gamma", 0.1)),
        )
    raise ValueError(
        f"Unsupported scheduler {scheduler_name!r}. Expected one of: cosine, step."
    )


__all__ = ["build_optimizer", "build_scheduler", "describe_optimizer"]
