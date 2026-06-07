"""Builder helpers for gated feature fusion setup."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from slr.branches.fusion.dataset import load_paired_skeleton_regions_config
from slr.branches.fusion.models import GatedFeatureFusion
from slr.branches.regions.models import build_region_model
from slr.branches.skeleton.graph import SkeletonGraph
from slr.branches.skeleton.models import build_skeleton_model
from slr.training.checkpointing import load_checkpoint
from slr.utils.io import read_yaml


def _resolve_path(
    value: str | Path | None,
    *,
    project_root: Path,
) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if not str(path):
        return None
    return path if path.is_absolute() else (project_root / path)


def _select_device(device_name: str | None) -> torch.device:
    requested = str(device_name or "auto").strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable in this environment.")
    return torch.device(str(device_name))


def _load_reference_branch_config(
    *,
    config_path: Path | None,
    checkpoint_path: Path | None,
) -> tuple[dict[str, Any], str]:
    if config_path is not None:
        resolved_config_path = Path(config_path)
        if resolved_config_path.exists():
            return read_yaml(resolved_config_path), "config_path"

    if checkpoint_path is not None and Path(checkpoint_path).exists():
        payload = torch.load(Path(checkpoint_path), map_location="cpu")
        checkpoint_cfg = payload.get("config")
        if isinstance(checkpoint_cfg, dict):
            return checkpoint_cfg, "checkpoint"

    return {}, "inline"


def _resolve_branch_checkpoint_path(
    branch_cfg: dict[str, Any],
    *,
    default_path: str,
    project_root: Path,
) -> Path | None:
    return _resolve_path(
        branch_cfg.get("checkpoint_path", branch_cfg.get("checkpoint", default_path)),
        project_root=project_root,
    )


def load_gated_feature_fusion_config(
    config_path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Load and normalize one gated fusion config."""

    raw_config = read_yaml(config_path)
    paired_cfg = load_paired_skeleton_regions_config(
        config_path,
        project_root=project_root,
    )
    project_path = Path(project_root or paired_cfg["project_root"]).resolve()

    experiment_cfg = dict(raw_config.get("experiment", {}))
    skeleton_branch_cfg = dict(raw_config.get("skeleton_branch", {}))
    regions_branch_cfg = dict(raw_config.get("regions_branch", {}))
    fusion_model_cfg = dict(raw_config.get("fusion_model", {}))
    train_cfg = dict(raw_config.get("train", {}))
    scheduler_cfg = dict(raw_config.get("scheduler", {}))
    early_stopping_cfg = dict(raw_config.get("early_stopping", {}))
    logging_cfg = dict(raw_config.get("logging", {}))
    runtime_cfg = dict(raw_config.get("runtime", {}))

    dataset_cfg = paired_cfg["dataset"]
    dataloader_cfg = dict(paired_cfg["dataloader"])
    if "shuffle_train" not in dataloader_cfg:
        dataloader_cfg["shuffle_train"] = bool(
            raw_config.get("dataloader", {}).get("shuffle_train", dataloader_cfg.get("shuffle", True))
        )

    skeleton_dataset_cfg = dataset_cfg["skeleton"]
    regions_dataset_cfg = dataset_cfg["regions"]
    resolved_device = str(
        runtime_cfg.get("device", train_cfg.get("device", "auto"))
    ).strip() or "auto"

    return {
        "config_path": Path(config_path),
        "project_root": project_path,
        "experiment": experiment_cfg,
        "dataset": dataset_cfg,
        "dataloader": dataloader_cfg,
        "train": train_cfg,
        "scheduler": scheduler_cfg,
        "early_stopping": early_stopping_cfg,
        "logging": logging_cfg,
        "runtime": {
            "device": resolved_device,
        },
        "skeleton_branch": {
            "config_path": _resolve_path(
                skeleton_branch_cfg.get(
                    "config_path",
                    "configs/train/skeleton_selected_31_stgcnpp.yaml",
                ),
                project_root=project_path,
            ),
            "checkpoint_path": _resolve_branch_checkpoint_path(
                skeleton_branch_cfg,
                default_path="checkpoints/models/skeleton/best.pt",
                project_root=project_path,
            ),
            "graph": dict(
                skeleton_branch_cfg.get(
                    "graph",
                    {
                        "layout": str(skeleton_dataset_cfg.get("keypoint_set", "selected_31")),
                        "strategy": "spatial",
                        "add_self_links": True,
                        "normalize_adjacency": True,
                    },
                )
            ),
            "model": dict(skeleton_branch_cfg.get("model", {})),
        },
        "regions_branch": {
            "config_path": _resolve_path(
                regions_branch_cfg.get(
                    "config_path",
                    "configs/train/regions_resnet18_gru_nslt100.yaml",
                ),
                project_root=project_path,
            ),
            "checkpoint_path": _resolve_branch_checkpoint_path(
                regions_branch_cfg,
                default_path="checkpoints/models/regions/best.pt",
                project_root=project_path,
            ),
            "model": dict(regions_branch_cfg.get("model", {})),
        },
        "fusion_model": {
            "name": str(fusion_model_cfg.get("name", "gated_feature_fusion")),
            "hidden_dim": int(fusion_model_cfg.get("hidden_dim", 256)),
            "proj_dropout": float(fusion_model_cfg.get("proj_dropout", 0.2)),
            "classifier_dropout": float(fusion_model_cfg.get("classifier_dropout", 0.5)),
            "freeze_skeleton": bool(fusion_model_cfg.get("freeze_skeleton", True)),
            "freeze_regions": bool(fusion_model_cfg.get("freeze_regions", True)),
        },
    }


def _resolve_skeleton_branch_model_config(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    dataset_cfg = config["dataset"]["skeleton"]
    branch_cfg = config["skeleton_branch"]
    config_path = branch_cfg.get("config_path")
    checkpoint_path = branch_cfg.get("checkpoint_path", branch_cfg.get("checkpoint"))
    reference_cfg, config_source = _load_reference_branch_config(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
    )

    reference_dataset_cfg = dict(reference_cfg.get("dataset", {}))
    reference_graph_cfg = dict(reference_cfg.get("graph", {}))
    model_cfg = dict(reference_cfg.get("model", {}))
    model_cfg.update(branch_cfg.get("model", {}))
    model_cfg.setdefault("name", "stgcnpp")
    model_cfg.setdefault("in_channels", int(dataset_cfg.get("expected_shape", [3])[0]))
    model_cfg.setdefault("num_nodes", int(dataset_cfg.get("expected_shape", [3, 150, 31, 1])[2]))
    model_cfg.setdefault("num_classes", int(config["dataset"]["num_classes"]))
    model_cfg.setdefault("base_channels", 64)
    model_cfg.setdefault("dropout", 0.5)

    graph_cfg = dict(reference_graph_cfg)
    graph_cfg.update(branch_cfg.get("graph", {}))
    graph_cfg.setdefault("layout", str(reference_dataset_cfg.get("keypoint_set", dataset_cfg.get("keypoint_set", "selected_31"))))
    graph_cfg.setdefault("strategy", "spatial")
    graph_cfg.setdefault("add_self_links", True)
    graph_cfg.setdefault("normalize_adjacency", True)
    return model_cfg, graph_cfg, {"config_source": config_source}


def _resolve_regions_branch_model_config(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    dataset_cfg = config["dataset"]["regions"]
    branch_cfg = config["regions_branch"]
    config_path = branch_cfg.get("config_path")
    checkpoint_path = branch_cfg.get("checkpoint_path", branch_cfg.get("checkpoint"))
    reference_cfg, config_source = _load_reference_branch_config(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
    )

    reference_model_cfg = dict(reference_cfg.get("model", {}))
    model_cfg = dict(reference_model_cfg)
    model_cfg.update(branch_cfg.get("model", {}))
    expected_shape = list(dataset_cfg.get("expected_shape", [3, 3, 64, 112, 112]))
    model_cfg.setdefault("name", "region_resnet18_gru")
    model_cfg.setdefault("num_classes", int(config["dataset"]["num_classes"]))
    model_cfg.setdefault("num_regions", len(dataset_cfg.get("active_regions", ["left_hand", "right_hand", "face"])))
    model_cfg.setdefault("in_channels", int(expected_shape[1]))
    model_cfg.setdefault("clip_len", int(expected_shape[2]))
    model_cfg.setdefault("crop_size", int(expected_shape[3]))
    model_cfg.setdefault("pretrained", True)
    model_cfg.setdefault("freeze_encoder", True)
    model_cfg.setdefault("encoder_name", "resnet18")
    model_cfg.setdefault("encoder_feature_dim", 512)
    model_cfg.setdefault("gru_hidden_size", 128)
    model_cfg.setdefault("gru_num_layers", 1)
    model_cfg.setdefault("bidirectional", True)
    model_cfg.setdefault("dropout", 0.5)
    model_cfg.setdefault("fusion", "concat")
    model_cfg.setdefault("use_valid_mask", True)

    if checkpoint_path is not None and Path(checkpoint_path).exists():
        model_cfg["pretrained"] = False
    return model_cfg, {"config_source": config_source}


def build_skeleton_branch_model(
    config: str | Path | dict[str, Any],
    *,
    device: torch.device | None = None,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Build the skeleton branch model and load its checkpoint when available."""

    resolved = (
        load_gated_feature_fusion_config(config)
        if isinstance(config, (str, Path))
        else config
    )
    model_cfg, graph_cfg, meta = _resolve_skeleton_branch_model_config(resolved)
    device = device or _select_device(resolved.get("runtime", {}).get("device"))

    graph = SkeletonGraph(
        layout=str(graph_cfg["layout"]),
        strategy=str(graph_cfg["strategy"]),
        normalize=bool(graph_cfg["normalize_adjacency"]),
        add_self_links=bool(graph_cfg["add_self_links"]),
    )
    model = build_skeleton_model(model_cfg, graph).to(device)

    checkpoint_path = resolved["skeleton_branch"].get("checkpoint_path", resolved["skeleton_branch"].get("checkpoint"))
    checkpoint_loaded = False
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        load_checkpoint(checkpoint_path, model, map_location=device)
        checkpoint_loaded = True

    info = {
        "model_name": model.__class__.__name__,
        "feature_dim": int(getattr(model, "feature_dim", getattr(model, "output_dim"))),
        "checkpoint": None if checkpoint_path is None else str(Path(checkpoint_path).as_posix()),
        "checkpoint_loaded": checkpoint_loaded,
        "graph_layout": str(graph.layout),
        "graph_num_nodes": int(graph.num_nodes),
        **meta,
    }
    return model, info


def build_regions_branch_model(
    config: str | Path | dict[str, Any],
    *,
    device: torch.device | None = None,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Build the regions branch model and load its checkpoint when available."""

    resolved = (
        load_gated_feature_fusion_config(config)
        if isinstance(config, (str, Path))
        else config
    )
    model_cfg, meta = _resolve_regions_branch_model_config(resolved)
    device = device or _select_device(resolved.get("runtime", {}).get("device"))

    model = build_region_model(model_cfg).to(device)
    checkpoint_path = resolved["regions_branch"].get("checkpoint_path", resolved["regions_branch"].get("checkpoint"))
    checkpoint_loaded = False
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        load_checkpoint(checkpoint_path, model, map_location=device)
        checkpoint_loaded = True

    info = {
        "model_name": model.__class__.__name__,
        "feature_dim": int(getattr(model, "feature_dim", getattr(model, "output_dim"))),
        "checkpoint": None if checkpoint_path is None else str(Path(checkpoint_path).as_posix()),
        "checkpoint_loaded": checkpoint_loaded,
        **meta,
    }
    return model, info


def build_gated_feature_fusion_from_config(
    config: str | Path | dict[str, Any],
    *,
    device: str | torch.device | None = None,
) -> tuple[GatedFeatureFusion, dict[str, Any]]:
    """Build a full gated-fusion model bundle from one config."""

    resolved = (
        load_gated_feature_fusion_config(config)
        if isinstance(config, (str, Path))
        else config
    )
    resolved_device = (
        _select_device(resolved.get("runtime", {}).get("device"))
        if device is None
        else (device if isinstance(device, torch.device) else _select_device(str(device)))
    )

    skeleton_model, skeleton_info = build_skeleton_branch_model(
        resolved,
        device=resolved_device,
    )
    regions_model, regions_info = build_regions_branch_model(
        resolved,
        device=resolved_device,
    )

    fusion_cfg = resolved["fusion_model"]
    model = GatedFeatureFusion(
        skeleton_model=skeleton_model,
        regions_model=regions_model,
        skeleton_dim=int(skeleton_info["feature_dim"]),
        region_dim=int(regions_info["feature_dim"]),
        hidden_dim=int(fusion_cfg["hidden_dim"]),
        num_classes=int(resolved["dataset"]["num_classes"]),
        proj_dropout=float(fusion_cfg["proj_dropout"]),
        classifier_dropout=float(fusion_cfg["classifier_dropout"]),
        freeze_skeleton=bool(fusion_cfg["freeze_skeleton"]),
        freeze_regions=bool(fusion_cfg["freeze_regions"]),
    ).to(resolved_device)

    info = {
        "device": str(resolved_device),
        "num_classes": int(resolved["dataset"]["num_classes"]),
        "skeleton": skeleton_info,
        "regions": regions_info,
        "fusion_model": {
            "name": str(fusion_cfg["name"]),
            "hidden_dim": int(fusion_cfg["hidden_dim"]),
            "proj_dropout": float(fusion_cfg["proj_dropout"]),
            "classifier_dropout": float(fusion_cfg["classifier_dropout"]),
            "freeze_skeleton": bool(fusion_cfg["freeze_skeleton"]),
            "freeze_regions": bool(fusion_cfg["freeze_regions"]),
        },
    }
    return model, info


__all__ = [
    "build_gated_feature_fusion_from_config",
    "build_regions_branch_model",
    "build_skeleton_branch_model",
    "load_gated_feature_fusion_config",
]
