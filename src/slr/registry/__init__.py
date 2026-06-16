"""Public model-registry API."""

from slr.registry.loader import get_model_record, load_registry
from slr.registry.schema import (
    ALLOWED_BRANCHES,
    ALLOWED_STATUSES,
    ArtifactBundle,
    ArtifactRef,
    LoadedRegistry,
    ModelIdentity,
    ModelRecord,
    RegistryIndexEntry,
)
from slr.registry.validation import RegistryValidationError, import_class, validate_loaded_registry

__all__ = [
    "ALLOWED_BRANCHES",
    "ALLOWED_STATUSES",
    "ArtifactBundle",
    "ArtifactRef",
    "LoadedRegistry",
    "ModelIdentity",
    "ModelRecord",
    "RegistryIndexEntry",
    "RegistryValidationError",
    "get_model_record",
    "import_class",
    "load_registry",
    "validate_loaded_registry",
]
