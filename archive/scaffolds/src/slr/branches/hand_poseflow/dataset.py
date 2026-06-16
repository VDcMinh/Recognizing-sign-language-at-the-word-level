"""Dataset interfaces for the hand sequence plus pose-flow branch."""

from __future__ import annotations


class HandPoseFlowDataset:
    """Placeholder dataset for two-stream hand and pose-flow inputs."""

    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs

    def __len__(self) -> int:
        return 0

    def __getitem__(self, index: int):
        raise NotImplementedError("HandPoseFlowDataset is not implemented yet.")
