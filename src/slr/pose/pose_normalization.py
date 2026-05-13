"""Normalization helpers for graph-based pose models."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np


def center_keypoints(keypoints: np.ndarray, center_index: int) -> np.ndarray:
    """Center a keypoint sequence around a chosen landmark index."""

    centered = keypoints.copy()
    origin = centered[:, center_index : center_index + 1, :2]
    centered[:, :, :2] = centered[:, :, :2] - origin
    return centered


def normalize_xy_to_minus1_1(
    keypoints: np.ndarray,
    image_width: int,
    image_height: int,
    clip: bool = True,
) -> tuple[np.ndarray, dict[str, int]]:
    """Clip pixel coordinates to image bounds, then map them into ``[-1, 1]``."""

    if image_width <= 0 or image_height <= 0:
        raise ValueError("image_width and image_height must be positive.")

    normalized = np.asarray(keypoints, dtype=np.float32).copy()
    x = normalized[..., 0]
    y = normalized[..., 1]

    x_finite = np.isfinite(x)
    y_finite = np.isfinite(y)
    x_oob = x_finite & ((x < 0.0) | (x > float(image_width)))
    y_oob = y_finite & ((y < 0.0) | (y > float(image_height)))

    if clip:
        normalized[..., 0] = np.where(
            x_finite,
            np.clip(x, 0.0, float(image_width)),
            x,
        )
        normalized[..., 1] = np.where(
            y_finite,
            np.clip(y, 0.0, float(image_height)),
            y,
        )

    normalized[..., 0] = 2.0 * (normalized[..., 0] / float(image_width)) - 1.0
    normalized[..., 1] = 2.0 * (normalized[..., 1] / float(image_height)) - 1.0

    stats = {
        "x_out_of_bounds": int(x_oob.sum()),
        "y_out_of_bounds": int(y_oob.sum()),
        "xy_out_of_bounds": int(x_oob.sum() + y_oob.sum()),
    }
    return normalized, stats


def compute_confidence_scale(
    confidence_arrays: Iterable[np.ndarray],
    method: str = "percentile",
    percentile: float = 95.0,
) -> dict[str, Any]:
    """Fit one confidence normalization scale from a collection of arrays."""

    if method != "percentile":
        raise ValueError(f"Unsupported confidence normalization method: {method}")

    values: list[np.ndarray] = []
    num_arrays = 0
    for array in confidence_arrays:
        scores = np.asarray(array, dtype=np.float32).reshape(-1)
        scores = scores[np.isfinite(scores)]
        if scores.size == 0:
            continue
        values.append(scores)
        num_arrays += 1

    if not values:
        return {
            "scale": 1.0,
            "method": method,
            "percentile": float(percentile),
            "num_arrays": 0,
            "num_values": 0,
            "fallback_used": True,
            "warning": "No finite confidence values found. Falling back to scale=1.0.",
        }

    merged = np.concatenate(values, axis=0)
    scale = float(np.percentile(merged, percentile))
    fallback_used = not np.isfinite(scale) or scale <= 0.0
    warning = ""
    if fallback_used:
        scale = 1.0
        warning = "Computed confidence scale was non-finite or <= 0. Falling back to 1.0."

    return {
        "scale": float(scale),
        "method": method,
        "percentile": float(percentile),
        "num_arrays": int(num_arrays),
        "num_values": int(merged.size),
        "fallback_used": bool(fallback_used),
        "warning": warning,
    }


def normalize_confidence(
    keypoints: np.ndarray,
    confidence_scale: float,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
) -> np.ndarray:
    """Normalize raw confidence values by a fitted scale and clip them."""

    if not np.isfinite(confidence_scale) or confidence_scale <= 0.0:
        confidence_scale = 1.0

    normalized = np.asarray(keypoints, dtype=np.float32).copy()
    normalized[..., 2] = np.clip(
        normalized[..., 2] / float(confidence_scale),
        float(clip_min),
        float(clip_max),
    )
    return normalized


def sanitize_non_finite_keypoints(keypoints: np.ndarray) -> tuple[np.ndarray, int]:
    """Replace non-finite pose values with zeros after normalization."""

    sanitized = np.asarray(keypoints, dtype=np.float32).copy()
    invalid_mask = ~np.isfinite(sanitized)
    invalid_count = int(invalid_mask.sum())
    if invalid_count:
        sanitized[invalid_mask] = 0.0
    return sanitized, invalid_count
