"""Random seed helpers."""

from __future__ import annotations

import random

import numpy as np


def seed_everything(seed: int) -> None:
    """Seed Python and NumPy RNGs."""

    random.seed(seed)
    np.random.seed(seed)
