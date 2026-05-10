"""Lightweight video-related helpers."""

from __future__ import annotations

from pathlib import Path

import cv2


def probe_video(path: str | Path) -> dict[str, float | int]:
    """Return basic video metadata using OpenCV."""

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")

    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        num_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()

    return {
        "fps": fps,
        "width": width,
        "height": height,
        "num_frames": num_frames,
    }
