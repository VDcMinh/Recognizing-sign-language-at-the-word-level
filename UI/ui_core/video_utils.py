from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import cv2


UI_ROOT = Path(__file__).resolve().parents[1]
UPLOADS_DIR = UI_ROOT / "outputs" / "uploads"
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}
MAX_VIDEO_SIZE_BYTES = 500 * 1024 * 1024


def validate_video_file(uploaded_file) -> tuple[bool, str]:
    if uploaded_file is None:
        return False, "No video uploaded yet."

    extension = Path(uploaded_file.name).suffix.lower()
    if extension not in ALLOWED_VIDEO_EXTENSIONS:
        return False, "Unsupported file format. Please upload MP4, MOV, AVI, or MKV."

    if int(uploaded_file.size) > MAX_VIDEO_SIZE_BYTES:
        return False, "File is too large. Maximum size is 500MB."

    return True, ""


def validate_saved_video_path(video_path: str | Path) -> tuple[bool, str]:
    path = Path(video_path)
    if not path.exists():
        return False, "Uploaded video could not be found on disk."

    if path.suffix.lower() not in ALLOWED_VIDEO_EXTENSIONS:
        return False, "Unsupported file format. Please upload MP4, MOV, AVI, or MKV."

    if path.stat().st_size > MAX_VIDEO_SIZE_BYTES:
        return False, "File is too large. Maximum size is 500MB."

    return True, ""


def _sanitize_stem(stem: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip("-._")
    return normalized or "video"


def save_uploaded_video(uploaded_file) -> str:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    extension = Path(uploaded_file.name).suffix.lower()
    sanitized_name = _sanitize_stem(Path(uploaded_file.name).stem)
    save_path = UPLOADS_DIR / f"{timestamp}_{sanitized_name}{extension}"
    save_path.write_bytes(uploaded_file.getbuffer())
    return save_path.as_posix()


def format_file_size(size_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    value = float(size_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{int(size_bytes)} B"


def format_duration(duration_seconds: float | None) -> str:
    if duration_seconds is None or duration_seconds <= 0:
        return "--:--"
    total_seconds = int(round(duration_seconds))
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def get_video_metadata(video_path: str | Path) -> dict[str, object]:
    path = Path(video_path)
    duration_seconds: float | None = None
    frame_count = 0
    fps = 0.0

    capture = cv2.VideoCapture(path.as_posix())
    if capture.isOpened():
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if fps > 0 and frame_count > 0:
            duration_seconds = frame_count / fps
    capture.release()

    file_size_bytes = path.stat().st_size
    extension = path.suffix.upper().replace(".", "")
    return {
        "filename": path.name,
        "path": path.as_posix(),
        "extension": extension,
        "file_size_bytes": file_size_bytes,
        "file_size_label": format_file_size(file_size_bytes),
        "duration_seconds": duration_seconds,
        "duration_label": format_duration(duration_seconds),
        "frame_count": frame_count,
        "fps": round(fps, 2) if fps > 0 else None,
    }
