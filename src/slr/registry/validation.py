"""Validation helpers for the model registry."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from slr.registry.schema import (
    ALLOWED_BRANCHES,
    ALLOWED_STATUSES,
    ArtifactRef,
    LoadedRegistry,
    ModelRecord,
    RegistryIndexEntry,
)


class RegistryValidationError(ValueError):
    """Raised when registry content is invalid."""


def import_class(class_path: str) -> type[Any]:
    """Import one class from its dotted path."""

    module_name, separator, attr_name = str(class_path).rpartition(".")
    if not module_name or not separator or not attr_name:
        raise RegistryValidationError(f"Invalid class_path: {class_path!r}")
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attr_name)
    except AttributeError as exc:
        raise RegistryValidationError(
            f"Class {attr_name!r} was not found in module {module_name!r}."
        ) from exc


def resolve_project_path(project_root: Path, raw_path: str | None) -> Path | None:
    """Resolve one possibly-relative path against the repo root."""

    if raw_path is None:
        return None
    path = Path(raw_path)
    return path if path.is_absolute() else (project_root / path)


def validate_registry_entry(entry: RegistryIndexEntry, *, registry_root: Path) -> None:
    """Validate one summary registry entry."""

    if entry.branch not in ALLOWED_BRANCHES:
        raise RegistryValidationError(
            f"Registry entry {entry.id!r} has invalid branch {entry.branch!r}."
        )
    if entry.status not in ALLOWED_STATUSES:
        raise RegistryValidationError(
            f"Registry entry {entry.id!r} has invalid status {entry.status!r}."
        )
    registry_file_path = registry_root / entry.registry_file
    if not registry_file_path.exists():
        raise RegistryValidationError(
            f"Registry file for {entry.id!r} does not exist: {entry.registry_file}"
        )


def validate_artifact_ref(
    artifact: ArtifactRef,
    *,
    label: str,
    project_root: Path,
    required: bool,
) -> None:
    """Validate one artifact reference."""

    local_path = resolve_project_path(project_root, artifact.local_path)
    has_remote = bool(artifact.remote_uri)
    if local_path is not None and not local_path.exists():
        raise RegistryValidationError(
            f"{label} local_path does not exist: {artifact.local_path}"
        )
    if required and local_path is None and not has_remote:
        raise RegistryValidationError(
            f"{label} must provide either local_path or remote_uri."
        )


def validate_model_record(record: ModelRecord, *, project_root: Path) -> None:
    """Validate one loaded ``model.yaml`` record."""

    if record.schema_version != 1:
        raise RegistryValidationError(
            f"Model {record.identity.id!r} must use schema_version=1."
        )
    if record.identity.branch not in ALLOWED_BRANCHES:
        raise RegistryValidationError(
            f"Model {record.identity.id!r} has invalid branch {record.identity.branch!r}."
        )
    if record.identity.status not in ALLOWED_STATUSES:
        raise RegistryValidationError(
            f"Model {record.identity.id!r} has invalid status {record.identity.status!r}."
        )

    class_path = str(record.model.get("class_path", "")).strip()
    if not class_path:
        raise RegistryValidationError(
            f"Model {record.identity.id!r} is missing model.class_path."
        )
    import_class(class_path)

    validate_artifact_ref(
        record.artifacts.resolved_config,
        label=f"{record.identity.id}.artifacts.resolved_config",
        project_root=project_root,
        required=True,
    )
    validate_artifact_ref(
        record.artifacts.class_map,
        label=f"{record.identity.id}.artifacts.class_map",
        project_root=project_root,
        required=False,
    )
    validate_artifact_ref(
        record.artifacts.checkpoint,
        label=f"{record.identity.id}.artifacts.checkpoint",
        project_root=project_root,
        required=record.identity.status == "ready",
    )
    validate_artifact_ref(
        record.artifacts.metrics,
        label=f"{record.identity.id}.artifacts.metrics",
        project_root=project_root,
        required=False,
    )
    validate_artifact_ref(
        record.artifacts.train_log,
        label=f"{record.identity.id}.artifacts.train_log",
        project_root=project_root,
        required=False,
    )

    input_type = str(record.input.get("type", "")).strip()
    if record.identity.branch == "skeleton":
        if input_type != "skeleton":
            raise RegistryValidationError(
                f"Skeleton model {record.identity.id!r} must declare input.type='skeleton'."
            )
        if not record.input.get("keypoint_set"):
            raise RegistryValidationError(
                f"Skeleton model {record.identity.id!r} is missing input.keypoint_set."
            )
    elif record.identity.branch == "regions":
        if input_type != "regions":
            raise RegistryValidationError(
                f"Regions model {record.identity.id!r} must declare input.type='regions'."
            )
        active_regions = record.input.get("active_regions")
        if not isinstance(active_regions, list) or not active_regions:
            raise RegistryValidationError(
                f"Regions model {record.identity.id!r} is missing input.active_regions."
            )
    elif record.identity.branch == "fusion":
        if input_type != "fusion":
            raise RegistryValidationError(
                f"Fusion model {record.identity.id!r} must declare input.type='fusion'."
            )
        if not record.model.get("skeleton_registry_id"):
            raise RegistryValidationError(
                f"Fusion model {record.identity.id!r} is missing model.skeleton_registry_id."
            )
        if not record.model.get("regions_registry_id"):
            raise RegistryValidationError(
                f"Fusion model {record.identity.id!r} is missing model.regions_registry_id."
            )


def validate_loaded_registry(loaded: LoadedRegistry) -> None:
    """Validate cross-file registry integrity."""

    if loaded.registry_version != 1:
        raise RegistryValidationError("registry.yaml must use registry_version=1.")

    seen_ids: set[str] = set()
    for entry in loaded.entries:
        if entry.id in seen_ids:
            raise RegistryValidationError(f"Duplicate registry entry id: {entry.id}")
        seen_ids.add(entry.id)
        validate_registry_entry(entry, registry_root=loaded.registry_root)

    model_ids: set[str] = set()
    for record in loaded.models:
        model_id = record.identity.id
        if model_id in model_ids:
            raise RegistryValidationError(f"Duplicate model identity id: {model_id}")
        model_ids.add(model_id)
        validate_model_record(record, project_root=loaded.project_root)

    entry_by_id = {entry.id: entry for entry in loaded.entries}
    for record in loaded.models:
        entry = entry_by_id.get(record.identity.id)
        if entry is None:
            raise RegistryValidationError(
                f"Model {record.identity.id!r} is not listed in registry.yaml."
            )
        if entry.branch != record.identity.branch:
            raise RegistryValidationError(
                f"Branch mismatch for {record.identity.id!r}: "
                f"registry.yaml={entry.branch!r}, model.yaml={record.identity.branch!r}."
            )
        if entry.status != record.identity.status:
            raise RegistryValidationError(
                f"Status mismatch for {record.identity.id!r}: "
                f"registry.yaml={entry.status!r}, model.yaml={record.identity.status!r}."
            )

    if loaded.default_model is not None and loaded.default_model not in model_ids:
        raise RegistryValidationError(
            f"default_model {loaded.default_model!r} is not present in registry.yaml."
        )

    models_by_id = loaded.models_by_id
    for record in loaded.models:
        if record.identity.branch != "fusion":
            continue
        skeleton_id = str(record.model.get("skeleton_registry_id", "")).strip()
        regions_id = str(record.model.get("regions_registry_id", "")).strip()
        if skeleton_id not in models_by_id:
            raise RegistryValidationError(
                f"Fusion model {record.identity.id!r} references missing skeleton model {skeleton_id!r}."
            )
        if regions_id not in models_by_id:
            raise RegistryValidationError(
                f"Fusion model {record.identity.id!r} references missing regions model {regions_id!r}."
            )
