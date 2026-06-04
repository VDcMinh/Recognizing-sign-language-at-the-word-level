from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


UI_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = UI_ROOT.parent
DEFAULT_LABEL_SOURCE = REPO_ROOT / "data/datasets/WLASL/index/subsets/nslt100/train.csv"

MODEL_OPTIONS: dict[str, dict[str, Any]] = {
    "Skeleton": {
        "type": "skeleton",
        "checkpoint": "UI/checkpoints/skeleton/best.pt",
        "config": "UI/configs/skeleton/config_resolved.yaml",
        "fallback_train_config": "configs/train/skeleton_selected_31_stgcnpp.yaml",
        "label_source": "data/datasets/WLASL/index/subsets/nslt100/train.csv",
        "keypoint_set": "selected_31",
        "notes": "Single-branch skeleton recognition using an ST-GCN++ style checkpoint.",
    },
    "Skeleton + Fusion": {
        "type": "fusion",
        "checkpoint": "UI/checkpoints/fusion/best.pt",
        "config": "UI/configs/fusion/config_resolved.yaml",
        "fallback_train_config": "configs/fusion/nslt100_skeleton_regions_late_fusion.yaml",
        "label_source": "data/datasets/WLASL/index/subsets/nslt100/train.csv",
        "single_checkpoint_notes": "Use one fused checkpoint and one resolved config.",
        "late_fusion": {
            "skeleton_checkpoint": "UI/checkpoints/fusion/skeleton_best.pt",
            "regions_checkpoint": "UI/checkpoints/fusion/regions_best.pt",
            "skeleton_config": "UI/configs/fusion/skeleton_config_resolved.yaml",
            "regions_config": "UI/configs/fusion/regions_config_resolved.yaml",
            "fusion_config": "UI/configs/fusion/fusion_config.yaml",
        },
        "notes": "Supports a single fused checkpoint or late fusion from skeleton and regions branches.",
    },
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    model_type: str
    runtime_mode: str
    label_source: Path
    config_path: Path | None
    checkpoint_path: Path | None
    required_checkpoint_paths: tuple[Path, ...]
    required_config_paths: tuple[Path, ...]
    fallback_train_config: Path | None
    metadata: dict[str, Any]


def _repo_path(path_value: str | None) -> Path | None:
    if not path_value:
        return None
    return REPO_ROOT / Path(path_value)


def get_model_option_names() -> list[str]:
    return list(MODEL_OPTIONS.keys())


def _detect_fusion_runtime_mode(option: dict[str, Any]) -> str:
    late_fusion = option["late_fusion"]
    late_paths = [
        _repo_path(late_fusion["skeleton_checkpoint"]),
        _repo_path(late_fusion["regions_checkpoint"]),
        _repo_path(late_fusion["skeleton_config"]),
        _repo_path(late_fusion["regions_config"]),
        _repo_path(late_fusion["fusion_config"]),
    ]
    if all(path is not None and path.exists() for path in late_paths):
        return "late_fusion"
    return "single_checkpoint"


def get_model_spec(model_name: str) -> ModelSpec:
    if model_name not in MODEL_OPTIONS:
        available = ", ".join(MODEL_OPTIONS)
        raise KeyError(f"Unknown model {model_name!r}. Available options: {available}")

    option = MODEL_OPTIONS[model_name]
    model_type = str(option["type"])
    label_source = _repo_path(option.get("label_source")) or DEFAULT_LABEL_SOURCE
    fallback_train_config = _repo_path(option.get("fallback_train_config"))

    if model_type == "fusion":
        runtime_mode = _detect_fusion_runtime_mode(option)
        if runtime_mode == "late_fusion":
            late_fusion = option["late_fusion"]
            config_paths = (
                _repo_path(late_fusion["skeleton_config"]),
                _repo_path(late_fusion["regions_config"]),
                _repo_path(late_fusion["fusion_config"]),
            )
            checkpoint_paths = (
                _repo_path(late_fusion["skeleton_checkpoint"]),
                _repo_path(late_fusion["regions_checkpoint"]),
            )
            return ModelSpec(
                name=model_name,
                model_type=model_type,
                runtime_mode=runtime_mode,
                label_source=label_source,
                config_path=_repo_path(late_fusion["fusion_config"]),
                checkpoint_path=None,
                required_checkpoint_paths=tuple(path for path in checkpoint_paths if path is not None),
                required_config_paths=tuple(path for path in config_paths if path is not None),
                fallback_train_config=fallback_train_config,
                metadata=option,
            )

    config_path = _repo_path(option.get("config"))
    checkpoint_path = _repo_path(option.get("checkpoint"))
    return ModelSpec(
        name=model_name,
        model_type=model_type,
        runtime_mode="single_checkpoint",
        label_source=label_source,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        required_checkpoint_paths=tuple(path for path in [checkpoint_path] if path is not None),
        required_config_paths=tuple(path for path in [config_path] if path is not None),
        fallback_train_config=fallback_train_config,
        metadata=option,
    )


def describe_model_assets(model_name: str) -> dict[str, Any]:
    spec = get_model_spec(model_name)
    missing_checkpoints = [
        path.as_posix() for path in spec.required_checkpoint_paths if not path.exists()
    ]
    missing_configs = [
        path.as_posix() for path in spec.required_config_paths if not path.exists()
    ]
    return {
        "model_name": spec.name,
        "model_type": spec.model_type,
        "runtime_mode": spec.runtime_mode,
        "checkpoint_paths": [path.as_posix() for path in spec.required_checkpoint_paths],
        "config_paths": [path.as_posix() for path in spec.required_config_paths],
        "missing_checkpoints": missing_checkpoints,
        "missing_configs": missing_configs,
        "label_source": spec.label_source.as_posix(),
    }


def has_required_assets(model_name: str) -> bool:
    asset_report = describe_model_assets(model_name)
    return not asset_report["missing_checkpoints"] and not asset_report["missing_configs"]
