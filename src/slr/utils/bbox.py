"""Bounding box helpers."""

from __future__ import annotations

import ast
from typing import NamedTuple


class BoundingBox(NamedTuple):
    """Axis-aligned bounding box in ``xyxy`` format."""

    x1: float
    y1: float
    x2: float
    y2: float


def parse_bbox(value) -> BoundingBox | None:
    """Parse a bbox from strings, sequences, or an existing ``BoundingBox``."""

    if value is None:
        return None
    if isinstance(value, BoundingBox):
        return value

    parts = None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            parsed = [item.strip() for item in text.split(",")]
        parts = parsed
    elif isinstance(value, (list, tuple)):
        parts = value

    if parts is None or len(parts) != 4:
        return None

    try:
        x1, y1, x2, y2 = (float(item) for item in parts)
    except (TypeError, ValueError):
        return None
    return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)


def is_valid_bbox(box: BoundingBox | None) -> bool:
    """Return ``True`` when the bbox has a positive area."""

    if box is None:
        return False
    return box.x2 > box.x1 and box.y2 > box.y1


def expand_bbox(
    box: BoundingBox,
    left: float = 0.0,
    right: float = 0.0,
    top: float = 0.0,
    bottom: float = 0.0,
) -> BoundingBox:
    """Expand a bbox by fractional margins relative to its width and height."""

    width = box.x2 - box.x1
    height = box.y2 - box.y1
    return BoundingBox(
        x1=box.x1 - left * width,
        y1=box.y1 - top * height,
        x2=box.x2 + right * width,
        y2=box.y2 + bottom * height,
    )


def clip_bbox(box: BoundingBox, width: int, height: int) -> BoundingBox:
    """Clip a bounding box to image bounds."""

    return BoundingBox(
        x1=max(0.0, min(box.x1, float(width))),
        y1=max(0.0, min(box.y1, float(height))),
        x2=max(0.0, min(box.x2, float(width))),
        y2=max(0.0, min(box.y2, float(height))),
    )


def bbox_to_int(box: BoundingBox) -> tuple[int, int, int, int]:
    """Convert a bbox to clipped integer pixel coordinates."""

    return (
        int(round(box.x1)),
        int(round(box.y1)),
        int(round(box.x2)),
        int(round(box.y2)),
    )


def bbox_to_string(box: BoundingBox | None) -> str:
    """Serialize a bbox as a stable ``[x1, y1, x2, y2]`` string."""

    if box is None:
        return ""
    return (
        f"[{int(round(box.x1))}, {int(round(box.y1))}, "
        f"{int(round(box.x2))}, {int(round(box.y2))}]"
    )
