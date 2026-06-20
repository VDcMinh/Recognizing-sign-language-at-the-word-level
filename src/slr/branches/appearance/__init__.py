"""Appearance branch for full-bbox RGB video clips."""

from .dataset import AppearanceClipDataset, appearance_collate_fn

__all__ = ["AppearanceClipDataset", "appearance_collate_fn"]
