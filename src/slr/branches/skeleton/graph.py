"""Graph definitions for skeleton-based temporal models."""

from __future__ import annotations


class SkeletonGraph:
    """Placeholder graph container for ST-GCN++ / CTR-GCN adjacency design."""

    def __init__(self, keypoint_set: str = "selected_31") -> None:
        self.keypoint_set = keypoint_set

    def adjacency(self):
        raise NotImplementedError("Adjacency construction will be added later.")
