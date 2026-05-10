"""Dataset interfaces for the region branch."""

from __future__ import annotations


class RegionSequenceDataset:
    """Placeholder dataset for face and hand crop sequences."""

    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs

    def __len__(self) -> int:
        return 0

    def __getitem__(self, index: int):
        raise NotImplementedError("RegionSequenceDataset is not implemented yet.")
