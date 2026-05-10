"""Label smoothing helpers for skeleton branch experiments."""

from __future__ import annotations

import numpy as np


def standard_label_smoothing(
    class_index: int, num_classes: int, epsilon: float = 0.1
) -> np.ndarray:
    """Create a standard label-smoothed target distribution."""

    if not 0 <= class_index < num_classes:
        raise ValueError("class_index must be within [0, num_classes).")
    if num_classes <= 1:
        raise ValueError("num_classes must be greater than 1.")
    if not 0.0 <= epsilon < 1.0:
        raise ValueError("epsilon must be in [0, 1).")

    off_value = epsilon / (num_classes - 1)
    target = np.full(num_classes, off_value, dtype=np.float32)
    target[class_index] = 1.0 - epsilon
    return target


def language_label_smoothing(
    class_index: int,
    similarity_vector: np.ndarray,
    epsilon: float = 0.3,
) -> np.ndarray:
    """Blend a one-hot target with a language-informed similarity distribution.

    ``similarity_vector`` is expected to encode semantic neighborhood weights for
    all classes, for example from fastText or another gloss embedding space.
    """

    if similarity_vector.ndim != 1:
        raise ValueError("similarity_vector must be one-dimensional.")
    if not 0 <= class_index < similarity_vector.shape[0]:
        raise ValueError("class_index must be within the similarity vector range.")
    if not 0.0 <= epsilon < 1.0:
        raise ValueError("epsilon must be in [0, 1).")

    base = np.zeros_like(similarity_vector, dtype=np.float32)
    base[class_index] = 1.0

    weights = similarity_vector.astype(np.float32).copy()
    weights[class_index] = 0.0
    total = float(weights.sum())
    if total <= 0.0:
        return standard_label_smoothing(
            class_index=class_index,
            num_classes=similarity_vector.shape[0],
            epsilon=epsilon,
        )

    weights /= total
    return (1.0 - epsilon) * base + epsilon * weights
