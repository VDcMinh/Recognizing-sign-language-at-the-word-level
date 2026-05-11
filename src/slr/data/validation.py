"""Validation helpers for manifests and preprocessing outputs."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import pandas as pd


def missing_columns(columns: Iterable[str], required: Iterable[str]) -> list[str]:
    """Return required columns that are not present."""

    available = set(columns)
    return [name for name in required if name not in available]


def require_columns(
    frame: pd.DataFrame, required: Sequence[str], name: str = "DataFrame"
) -> None:
    """Raise if the dataframe does not contain the required columns."""

    missing = missing_columns(frame.columns, required)
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def validate_no_nulls_for_keys(
    frame: pd.DataFrame, keys: Sequence[str], name: str = "DataFrame"
) -> None:
    """Raise if any key column contains null values."""

    require_columns(frame, keys, name=name)
    null_columns = [column for column in keys if frame[column].isna().any()]
    if null_columns:
        raise ValueError(f"{name} contains null values in key columns: {null_columns}")


def validate_manifest_schema(
    frame: pd.DataFrame, columns: Sequence[str], name: str = "DataFrame"
) -> pd.DataFrame:
    """Validate the manifest schema and return the dataframe in schema order."""

    require_columns(frame, columns, name=name)
    return frame.loc[:, list(columns)]


def validate_split_values(
    values: Iterable[str | None],
    allowed: Sequence[str] = ("train", "val", "test"),
    context: str = "split values",
) -> None:
    """Raise if split values fall outside the allowed set."""

    allowed_set = set(allowed)
    invalid = sorted(
        {
            value
            for value in values
            if value is not None and not pd.isna(value) and value not in allowed_set
        }
    )
    if invalid:
        raise ValueError(f"Invalid {context}: {invalid}")
