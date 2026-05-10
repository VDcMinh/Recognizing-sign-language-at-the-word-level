"""Checkpoint path helpers."""

from __future__ import annotations

from pathlib import Path


def build_checkpoint_path(output_dir: str | Path, filename: str) -> Path:
    """Resolve a checkpoint filename under the given output directory."""

    return Path(output_dir) / filename
