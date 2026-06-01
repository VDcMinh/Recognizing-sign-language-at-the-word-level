"""Region branch model builders."""

from __future__ import annotations

from typing import Any

from slr.branches.regions.models.region_cnn_gru import RegionCNNGRU, build_region_cnn_gru


def build_region_model(cfg: dict[str, Any]) -> RegionCNNGRU:
    """Build one configured regions baseline model."""

    model_cfg = cfg.get("model", cfg) if isinstance(cfg, dict) else {}
    model_name = str(model_cfg.get("name", "region_cnn_gru")).strip().lower()
    if model_name == "region_cnn_gru":
        return build_region_cnn_gru(model_cfg)
    raise ValueError(
        f"Unsupported regions model {model_name!r}. Expected one of: 'region_cnn_gru'."
    )


__all__ = ["RegionCNNGRU", "build_region_model"]
