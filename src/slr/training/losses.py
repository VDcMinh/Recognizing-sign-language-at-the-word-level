"""Training loss helpers."""

from __future__ import annotations


def describe_loss(name: str) -> str:
    """Return a short human-readable description of the configured loss."""

    return f"Configured loss: {name}"
