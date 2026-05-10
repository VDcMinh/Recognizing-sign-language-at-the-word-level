"""Metric helpers for training and evaluation."""

from __future__ import annotations

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None


def _to_numpy(array):
    """Convert numpy arrays or torch tensors to ``np.ndarray``."""

    if torch is not None and isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def top_k_accuracy(logits, targets, topk: tuple[int, ...] = (1,)) -> dict[int, float]:
    """Compute top-k accuracy for batched class logits."""

    logits_np = _to_numpy(logits)
    targets_np = _to_numpy(targets).reshape(-1)

    if logits_np.ndim != 2:
        raise ValueError("logits must have shape (N, num_classes).")
    if logits_np.shape[0] != targets_np.shape[0]:
        raise ValueError("logits and targets must have the same batch dimension.")

    max_k = max(topk)
    ranking = np.argsort(logits_np, axis=1)[:, ::-1][:, :max_k]

    metrics: dict[int, float] = {}
    for k in topk:
        correct = (ranking[:, :k] == targets_np[:, None]).any(axis=1)
        metrics[k] = float(correct.mean())
    return metrics
