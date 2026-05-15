"""Skeleton baseline model builders."""

from __future__ import annotations

from typing import Any

from slr.branches.skeleton.models.simple_stgcn import SimpleSTGCN
from slr.branches.skeleton.models.stgcnpp import STGCNPP


def build_skeleton_model(cfg: dict[str, Any], graph) -> SimpleSTGCN | STGCNPP:
    """Build one configured skeleton baseline model."""

    model_cfg = cfg.get("model", cfg) if isinstance(cfg, dict) else {}
    model_name = str(model_cfg.get("name", "simple_stgcn")).strip().lower()
    adjacency = graph.adjacency() if hasattr(graph, "adjacency") else graph
    if model_name == "simple_stgcn":
        return SimpleSTGCN(
            in_channels=int(model_cfg.get("in_channels", 3)),
            num_classes=int(model_cfg.get("num_classes", 100)),
            num_nodes=int(model_cfg.get("num_nodes")),
            adjacency=adjacency,
            hidden_channels=int(model_cfg.get("hidden_channels", 64)),
            dropout=float(model_cfg.get("dropout", 0.5)),
        )
    if model_name == "stgcnpp":
        return STGCNPP(
            in_channels=int(model_cfg.get("in_channels", 3)),
            num_classes=int(model_cfg.get("num_classes", 100)),
            num_nodes=int(model_cfg.get("num_nodes")),
            adjacency=adjacency,
            base_channels=int(model_cfg.get("base_channels", 64)),
            stage_channels=model_cfg.get("stage_channels"),
            temporal_strides=model_cfg.get("temporal_strides"),
            dropout=float(model_cfg.get("dropout", 0.5)),
        )
    raise ValueError(
        f"Unsupported skeleton model {model_name!r}. Expected one of: "
        "'simple_stgcn', 'stgcnpp'."
    )


__all__ = ["SimpleSTGCN", "STGCNPP", "build_skeleton_model"]
