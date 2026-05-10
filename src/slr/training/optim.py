"""Optimizer configuration helpers."""

from __future__ import annotations


def describe_optimizer(name: str, learning_rate: float) -> str:
    """Return a short optimizer summary string."""

    return f"Optimizer(name={name}, learning_rate={learning_rate})"
