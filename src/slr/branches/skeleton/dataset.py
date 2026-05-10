"""Dataset interfaces for skeleton branch training."""

from __future__ import annotations


class SkeletonDataset:
    """Placeholder dataset for graph-based skeleton inputs."""

    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs

    def __len__(self) -> int:
        return 0

    def __getitem__(self, index: int):
        raise NotImplementedError("SkeletonDataset data loading is not implemented yet.")
