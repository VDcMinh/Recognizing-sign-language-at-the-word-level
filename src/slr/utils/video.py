"""Lightweight video-related helpers."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


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


def probe_video_basic(path: str | Path) -> dict[str, float | int]:
    """Return basic video metadata using OpenCV."""

    return probe_video(path)


def read_frames(
    path: str | Path,
    start_frame: int = 0,
    end_frame: int | None = None,
) -> list[np.ndarray]:
    """Read a contiguous frame range from a video."""

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")

    frames: list[np.ndarray] = []
    try:
        if start_frame > 0:
            capture.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))

        frame_index = start_frame
        while True:
            if end_frame is not None and frame_index > end_frame:
                break

            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
            frame_index += 1
    finally:
        capture.release()

    return frames


def write_video_from_frames(
    frames: list[np.ndarray],
    path: str | Path,
    fps: float,
    codec: str = "mp4v",
) -> None:
    """Write frames to a video file."""

    if not frames:
        raise ValueError("Cannot write a video with zero frames.")

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(target), fourcc, float(fps), (width, height))
    if not writer.isOpened():
        raise OSError(f"Could not open video writer: {target}")

    try:
        for frame in frames:
            if frame.shape[:2] != (height, width):
                raise ValueError("All frames must share the same size.")
            writer.write(frame)
    finally:
        writer.release()
