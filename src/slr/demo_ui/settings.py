"""Manual settings for the local React demo UI."""

from __future__ import annotations

from pathlib import Path


ACTIVE_SUBSET = "nslt100"
SUPPORTED_SUBSETS = ("nslt100", "nslt300", "nslt1000")
SUPPORTED_BRANCHES = ("skeleton", "regions", "fusion")
SUPPORTED_VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")

REGISTRY_PATH = Path("model_registry/registry_serving.yaml")
RUNTIME_ROOT = Path("runtime/react_ui")

API_HOST = "127.0.0.1"
API_PORT = 8008
DEFAULT_TOP_K = 5


def validate_active_subset() -> str:
    """Return the configured subset after validation."""

    subset = str(ACTIVE_SUBSET).strip().lower()
    if subset not in SUPPORTED_SUBSETS:
        raise ValueError(
            f"Unsupported ACTIVE_SUBSET={ACTIVE_SUBSET!r}. "
            f"Expected one of {SUPPORTED_SUBSETS}."
        )
    return subset
