"""Region branch model builders."""

from __future__ import annotations

from typing import Any

from torch import nn

from slr.branches.regions.models.region_cnn_gru import RegionCNNGRU, build_region_cnn_gru
from slr.branches.regions.models.region_resnet18_gru import (
    RegionResNet18GRU,
    build_region_resnet18_gru,
)


def build_region_model(cfg: dict[str, Any]) -> nn.Module:
    """Build one configured regions baseline model."""

    model_cfg = cfg.get("model", cfg) if isinstance(cfg, dict) else {}
    model_name = str(model_cfg.get("name", "region_resnet18_gru")).strip().lower()
    if model_name == "region_resnet18_gru":
        return build_region_resnet18_gru(model_cfg)
    if model_name == "region_cnn_gru":
        return build_region_cnn_gru(model_cfg)
    raise ValueError(
        "Unsupported regions model "
        f"{model_name!r}. Expected one of: 'region_resnet18_gru', 'region_cnn_gru'."
    )


__all__ = ["RegionCNNGRU", "RegionResNet18GRU", "build_region_model"]
