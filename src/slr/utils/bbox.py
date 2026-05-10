"""Bounding box helpers."""

from __future__ import annotations

from typing import NamedTuple


class BoundingBox(NamedTuple):
    """Axis-aligned bounding box in ``xyxy`` format."""

    x1: float
    y1: float
    x2: float
    y2: float


def clip_bbox(box: BoundingBox, width: int, height: int) -> BoundingBox:
    """Clip a bounding box to image bounds."""

    return BoundingBox(
        x1=max(0.0, min(box.x1, float(width))),
        y1=max(0.0, min(box.y1, float(height))),
        x2=max(0.0, min(box.x2, float(width))),
        y2=max(0.0, min(box.y2, float(height))),
    )
