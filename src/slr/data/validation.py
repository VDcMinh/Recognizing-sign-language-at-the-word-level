"""Validation helpers for manifests and preprocessing outputs."""

from __future__ import annotations

from collections.abc import Iterable


def missing_columns(columns: Iterable[str], required: Iterable[str]) -> list[str]:
    """Return required columns that are not present."""

    available = set(columns)
    return [name for name in required if name not in available]
