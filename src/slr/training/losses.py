"""Training loss helpers."""

from __future__ import annotations

from typing import Any

import torch.nn as nn


def get_loss_name(cfg: dict[str, Any]) -> str:
    """Resolve the configured loss name from one training config."""

    train_cfg = cfg.get("train", {})
    loss_name = str(train_cfg.get("loss", "cross_entropy")).strip().lower()
    if not loss_name:
        return "cross_entropy"
    return loss_name


def get_label_smoothing_epsilon(cfg: dict[str, Any]) -> float:
    """Resolve and validate the label smoothing epsilon from one config."""

    loss_name = get_loss_name(cfg)
    if loss_name == "cross_entropy":
        return 0.0

    label_smoothing_cfg = cfg.get("label_smoothing", {})
    if "epsilon" not in label_smoothing_cfg:
        raise ValueError(
            "label_smoothing.epsilon must be provided when "
            "train.loss='standard_label_smoothing'."
        )

    epsilon_raw = label_smoothing_cfg.get("epsilon")
    try:
        epsilon = float(epsilon_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"label_smoothing.epsilon must be a float, got {epsilon_raw!r}."
        ) from exc

    if not 0.0 <= epsilon < 1.0:
        raise ValueError(
            f"label_smoothing.epsilon must satisfy 0 <= epsilon < 1, got {epsilon}."
        )
    return epsilon


def describe_loss(name: str) -> str:
    """Return a short human-readable description of the configured loss."""

    return f"Configured loss: {name}"


def build_loss_from_config(cfg: dict[str, Any]):
    """Build a training loss module from one resolved config."""

    loss_name = get_loss_name(cfg)
    if loss_name == "cross_entropy":
        return nn.CrossEntropyLoss()
    if loss_name == "standard_label_smoothing":
        epsilon = get_label_smoothing_epsilon(cfg)
        return nn.CrossEntropyLoss(label_smoothing=epsilon)
    raise ValueError(f"Unsupported loss type: {loss_name}")


def build_loss(name: str, config: dict[str, Any] | None = None):
    """Backward-compatible loss builder by explicit name."""

    resolved_config: dict[str, Any] = {"train": {"loss": str(name)}}
    if isinstance(config, dict):
        if "train" in config or "label_smoothing" in config:
            resolved_config.update(config)
        else:
            resolved_config["label_smoothing"] = dict(config)
    return build_loss_from_config(resolved_config)


__all__ = [
    "build_loss",
    "build_loss_from_config",
    "describe_loss",
    "get_label_smoothing_epsilon",
    "get_loss_name",
]
