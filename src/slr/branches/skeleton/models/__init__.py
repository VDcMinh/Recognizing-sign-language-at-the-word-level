"""Skeleton baseline model builders."""

from __future__ import annotations

from typing import Any

from slr.branches.skeleton.models.simple_stgcn import SimpleSTGCN


def build_skeleton_model(cfg: dict[str, Any], graph) -> SimpleSTGCN:
    """Build one configured skeleton baseline model."""

    model_cfg = cfg.get("model", cfg) if isinstance(cfg, dict) else {}
    model_name = str(model_cfg.get("name", "simple_stgcn")).strip().lower()
    if model_name != "simple_stgcn":
        raise ValueError(
            f"Unsupported skeleton model {model_name!r}. Expected 'simple_stgcn'."
        )

    adjacency = graph.adjacency() if hasattr(graph, "adjacency") else graph
    return SimpleSTGCN(
        in_channels=int(model_cfg.get("in_channels", 3)),
        num_classes=int(model_cfg.get("num_classes", 100)),
        num_nodes=int(model_cfg.get("num_nodes")),
        adjacency=adjacency,
        hidden_channels=int(model_cfg.get("hidden_channels", 64)),
        dropout=float(model_cfg.get("dropout", 0.5)),
    )


__all__ = ["SimpleSTGCN", "build_skeleton_model"]
