"""Appearance branch model builders."""

from __future__ import annotations

from typing import Any

from torch import nn

from .fullbbox_i3d import FullBBoxI3D, build_fullbbox_i3d


def build_appearance_model(cfg: dict[str, Any]) -> nn.Module:
    """Build one configured appearance model."""

    model_cfg = cfg.get("model", cfg) if isinstance(cfg, dict) else {}
    model_name = str(model_cfg.get("name", "fullbbox_i3d")).strip().lower()
    if model_name in {"fullbbox_i3d", "appearance_i3d"}:
        return build_fullbbox_i3d(model_cfg)
    raise ValueError(
        "Unsupported appearance model "
        f"{model_name!r}. Expected one of: 'fullbbox_i3d', 'appearance_i3d'."
    )


__all__ = ["FullBBoxI3D", "build_appearance_model", "build_fullbbox_i3d"]
