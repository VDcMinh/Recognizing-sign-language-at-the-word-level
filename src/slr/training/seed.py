"""Training seed helpers with optional PyTorch determinism."""

from __future__ import annotations

import random

import numpy as np

from slr.utils.seed import seed_everything as _seed_everything

try:
    import torch
except ImportError:  # pragma: no cover - torch is optional at import time
    torch = None


def set_seed(seed: int, *, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch when available."""

    _seed_everything(int(seed))
    random.seed(int(seed))
    np.random.seed(int(seed))

    if torch is None:
        return

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
