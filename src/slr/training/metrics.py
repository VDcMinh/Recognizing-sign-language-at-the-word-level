"""Metric helpers for training and evaluation."""

from __future__ import annotations

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None


class AverageMeter:
    """Track running weighted averages for scalar metrics."""

    def __init__(self, name: str) -> None:
        self.name = str(name)
        self.reset()

    def reset(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * int(n)
        self.count += int(n)

    @property
    def avg(self) -> float:
        if self.count <= 0:
            return 0.0
        return self.total / self.count


def _to_numpy(array):
    """Convert numpy arrays or torch tensors to ``np.ndarray``."""

    if torch is not None and isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def accuracy_topk(logits, targets, topk: tuple[int, ...] = (1,)) -> dict[str, float]:
    """Compute top-k accuracy as fractions in the range ``0..1``."""

    if not topk:
        raise ValueError("topk must not be empty.")
    normalized_topk = tuple(int(k) for k in topk)
    if any(k <= 0 for k in normalized_topk):
        raise ValueError("topk values must be positive integers.")

    if torch is not None and isinstance(logits, torch.Tensor):
        if logits.ndim != 2:
            raise ValueError("logits must have shape (N, num_classes).")
        if not isinstance(targets, torch.Tensor):
            targets = torch.as_tensor(targets, device=logits.device)
        targets = targets.reshape(-1).to(logits.device)
        if logits.shape[0] != targets.shape[0]:
            raise ValueError("logits and targets must have the same batch dimension.")

        max_k = min(max(normalized_topk), int(logits.shape[1]))
        predictions = logits.topk(max_k, dim=1).indices
        correct = predictions.eq(targets.unsqueeze(1))
        metrics: dict[str, float] = {}
        for k in normalized_topk:
            k_eff = min(int(k), int(logits.shape[1]))
            metrics[f"top{k}"] = float(correct[:, :k_eff].any(dim=1).float().mean().item())
        return metrics

    logits_np = _to_numpy(logits)
    targets_np = _to_numpy(targets).reshape(-1)

    if logits_np.ndim != 2:
        raise ValueError("logits must have shape (N, num_classes).")
    if logits_np.shape[0] != targets_np.shape[0]:
        raise ValueError("logits and targets must have the same batch dimension.")

    max_k = min(max(normalized_topk), int(logits_np.shape[1]))
    ranking = np.argsort(logits_np, axis=1)[:, ::-1][:, :max_k]

    metrics: dict[str, float] = {}
    for k in normalized_topk:
        k_eff = min(int(k), int(logits_np.shape[1]))
        correct = (ranking[:, :k_eff] == targets_np[:, None]).any(axis=1)
        metrics[f"top{k}"] = float(correct.mean())
    return metrics


def top_k_accuracy(logits, targets, topk: tuple[int, ...] = (1,)) -> dict[int, float]:
    """Backward-compatible ``top-k`` accuracy mapping keyed by integer ``k``."""

    metrics = accuracy_topk(logits, targets, topk=topk)
    return {int(k): float(metrics[f"top{k}"]) for k in tuple(int(value) for value in topk)}


__all__ = ["AverageMeter", "accuracy_topk", "top_k_accuracy"]
