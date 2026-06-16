"""Schema objects for the model registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ALLOWED_BRANCHES = ("skeleton", "regions", "fusion")
ALLOWED_STATUSES = ("ready", "incomplete", "archived", "deprecated")


@dataclass(frozen=True)
class ArtifactRef:
    """One artifact reference with optional local and remote metadata."""

    local_path: str | None = None
    remote_type: str | None = None
    remote_uri: str | None = None
    sha256: str | None = None

    @classmethod
    def from_value(cls, value: Any) -> "ArtifactRef":
        if value is None:
            return cls()
        if isinstance(value, str):
            return cls(local_path=value.strip() or None)
        if not isinstance(value, dict):
            raise TypeError(f"Expected artifact mapping or string, got {type(value)!r}.")
        return cls(
            local_path=_optional_str(value.get("local_path")),
            remote_type=_optional_str(value.get("remote_type")),
            remote_uri=_optional_str(value.get("remote_uri")),
            sha256=_optional_str(value.get("sha256")),
        )


@dataclass(frozen=True)
class ArtifactBundle:
    """Artifact references associated with one model."""

    checkpoint: ArtifactRef = field(default_factory=ArtifactRef)
    class_map: ArtifactRef = field(default_factory=ArtifactRef)
    resolved_config: ArtifactRef = field(default_factory=ArtifactRef)

    @classmethod
    def from_dict(cls, value: Any) -> "ArtifactBundle":
        if not isinstance(value, dict):
            raise TypeError(f"Expected artifacts mapping, got {type(value)!r}.")
        return cls(
            checkpoint=ArtifactRef.from_value(value.get("checkpoint")),
            class_map=ArtifactRef.from_value(value.get("class_map")),
            resolved_config=ArtifactRef.from_value(value.get("resolved_config")),
        )


@dataclass(frozen=True)
class ModelIdentity:
    """Stable identity fields shared by all registered models."""

    id: str
    display_name: str
    description: str
    branch: str
    task: str
    subset: str
    num_classes: int
    status: str

    @classmethod
    def from_dict(cls, value: Any) -> "ModelIdentity":
        if not isinstance(value, dict):
            raise TypeError(f"Expected identity mapping, got {type(value)!r}.")
        return cls(
            id=_required_str(value, "id"),
            display_name=_required_str(value, "display_name"),
            description=_optional_str(value.get("description")) or "",
            branch=_required_str(value, "branch"),
            task=_required_str(value, "task"),
            subset=_required_str(value, "subset"),
            num_classes=int(value.get("num_classes")),
            status=_required_str(value, "status"),
        )


@dataclass(frozen=True)
class ModelRecord:
    """One full model metadata record loaded from ``model.yaml``."""

    schema_version: int
    identity: ModelIdentity
    model: dict[str, Any]
    input: dict[str, Any]
    artifacts: ArtifactBundle
    inference: dict[str, Any]
    ui: dict[str, Any]
    source_path: Path

    @classmethod
    def from_dict(cls, value: Any, *, source_path: Path) -> "ModelRecord":
        if not isinstance(value, dict):
            raise TypeError(f"Expected model record mapping, got {type(value)!r}.")
        model_cfg = value.get("model", {})
        input_cfg = value.get("input", {})
        inference_cfg = value.get("inference", {})
        ui_cfg = value.get("ui", {})
        if not isinstance(model_cfg, dict):
            raise TypeError("model must be a mapping.")
        if not isinstance(input_cfg, dict):
            raise TypeError("input must be a mapping.")
        if not isinstance(inference_cfg, dict):
            raise TypeError("inference must be a mapping.")
        if not isinstance(ui_cfg, dict):
            raise TypeError("ui must be a mapping.")
        return cls(
            schema_version=int(value.get("schema_version")),
            identity=ModelIdentity.from_dict(value.get("identity", {})),
            model=model_cfg,
            input=input_cfg,
            artifacts=ArtifactBundle.from_dict(value.get("artifacts", {})),
            inference=inference_cfg,
            ui=ui_cfg,
            source_path=Path(source_path),
        )


@dataclass(frozen=True)
class RegistryIndexEntry:
    """One summary entry from ``registry.yaml``."""

    id: str
    display_name: str
    branch: str
    model_type: str
    status: str
    registry_file: str

    @classmethod
    def from_dict(cls, value: Any) -> "RegistryIndexEntry":
        if not isinstance(value, dict):
            raise TypeError(f"Expected registry index entry mapping, got {type(value)!r}.")
        return cls(
            id=_required_str(value, "id"),
            display_name=_required_str(value, "display_name"),
            branch=_required_str(value, "branch"),
            model_type=_required_str(value, "model_type"),
            status=_required_str(value, "status"),
            registry_file=_required_str(value, "registry_file"),
        )


@dataclass(frozen=True)
class LoadedRegistry:
    """In-memory registry bundle ready for validation or UI use."""

    registry_path: Path
    registry_root: Path
    project_root: Path
    registry_version: int
    default_model: str | None
    entries: tuple[RegistryIndexEntry, ...]
    models: tuple[ModelRecord, ...]

    @property
    def models_by_id(self) -> dict[str, ModelRecord]:
        return {model.identity.id: model for model in self.models}


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _required_str(mapping: dict[str, Any], key: str) -> str:
    text = _optional_str(mapping.get(key))
    if text is None:
        raise ValueError(f"Missing required string field: {key}")
    return text
