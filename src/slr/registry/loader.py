"""Registry loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from slr.registry.schema import LoadedRegistry, ModelRecord, RegistryIndexEntry
from slr.registry.validation import validate_loaded_registry
from slr.utils.io import read_yaml


def load_registry(
    registry_path: str | Path = "model_registry/registry.yaml",
    *,
    validate: bool = True,
) -> LoadedRegistry:
    """Load the model registry index and all referenced model records."""

    registry_file = Path(registry_path).resolve()
    registry_root = registry_file.parent.resolve()
    project_root = registry_root.parent.resolve()
    payload = read_yaml(registry_file)

    registry_version = int(payload.get("registry_version"))
    default_model_value = payload.get("default_model")
    default_model = str(default_model_value).strip() if default_model_value is not None else None
    models_value = payload.get("models", [])
    if not isinstance(models_value, list):
        raise TypeError("registry.yaml field 'models' must be a list.")

    entries = tuple(RegistryIndexEntry.from_dict(item) for item in models_value)
    models: list[ModelRecord] = []
    for entry in entries:
        model_path = registry_root / entry.registry_file
        record_payload = read_yaml(model_path)
        models.append(ModelRecord.from_dict(record_payload, source_path=model_path))

    loaded = LoadedRegistry(
        registry_path=registry_file,
        registry_root=registry_root,
        project_root=project_root,
        registry_version=registry_version,
        default_model=default_model or None,
        entries=entries,
        models=tuple(models),
    )
    if validate:
        validate_loaded_registry(loaded)
    return loaded


def get_model_record(
    registry: LoadedRegistry,
    model_id: str,
) -> ModelRecord:
    """Return one model record by ID."""

    try:
        return registry.models_by_id[model_id]
    except KeyError as exc:
        raise KeyError(f"Unknown registry model id: {model_id}") from exc
