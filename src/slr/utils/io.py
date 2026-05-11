"""File I/O helpers shared across scripts and modules."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not exist and return it as ``Path``."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def read_json(path: str | Path) -> Any:
    """Read a JSON file."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(data: Any, path: str | Path, indent: int = 2) -> None:
    """Write a JSON file."""

    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=indent, ensure_ascii=False)


def read_text_lines(path: str | Path, drop_empty: bool = False) -> list[str]:
    """Read a text file into a list of stripped lines."""

    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if drop_empty:
        return [line.strip() for line in lines if line.strip()]
    return [line.rstrip("\n\r") for line in lines]


def write_text(text: str, path: str | Path) -> None:
    """Write plain text to a file."""

    target = Path(path)
    ensure_dir(target.parent)
    target.write_text(text, encoding="utf-8")


def read_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML file into a dictionary."""

    import yaml

    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected YAML mapping at {path}, got {type(data)!r}.")
    return data


def write_yaml(data: dict[str, Any], path: str | Path) -> None:
    """Write a dictionary to a YAML file."""

    import yaml

    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)


def read_csv(path: str | Path, **kwargs):
    """Read a CSV file via pandas."""

    import pandas as pd

    return pd.read_csv(path, **kwargs)


def write_csv(frame, path: str | Path, **kwargs) -> None:
    """Write a pandas DataFrame as CSV."""

    target = Path(path)
    ensure_dir(target.parent)
    frame.to_csv(target, index=False, **kwargs)


def write_dataframe_csv(frame, path: str | Path, **kwargs) -> None:
    """Write a dataframe as UTF-8 CSV with a stable default configuration."""

    defaults = {"encoding": "utf-8"}
    defaults.update(kwargs)
    write_csv(frame, path, **defaults)
