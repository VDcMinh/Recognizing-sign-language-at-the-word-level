from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable

from PIL import Image


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


@dataclass(slots=True)
class ProbabilityScore:
    label: str
    probability: float


@dataclass(slots=True)
class PredictionResult:
    predicted_word: str
    confidence: float
    probabilities: list[ProbabilityScore] = field(default_factory=list)


@dataclass(slots=True)
class ModelOption:
    id: str
    label: str


@dataclass(slots=True)
class VideoItem:
    id: str
    path: Path
    name: str
    duration_seconds: float
    added_at: datetime
    thumbnail: Image.Image | None = None


def is_video_file(file_path: str | Path) -> bool:
    return Path(file_path).suffix.lower() in VIDEO_EXTENSIONS


def format_duration(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    minutes, remainder = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{remainder:02d}"
    return f"{minutes:02d}:{remainder:02d}"


def format_timestamp(timestamp: datetime) -> str:
    return timestamp.strftime("%b %d, %Y • %I:%M %p").replace(" 0", " ")


def normalize_percentage(value: float | int | None) -> float | None:
    if value is None:
        return None

    numeric = float(value)
    if numeric <= 1:
        numeric *= 100
    return max(0.0, min(100.0, numeric))


def build_prediction_result(
    predicted_word: str,
    confidence: float | int,
    probabilities: Iterable[tuple[str, float | int]],
) -> PredictionResult:
    normalized_items = [
        ProbabilityScore(label=label, probability=normalize_percentage(probability) or 0.0)
        for label, probability in probabilities
    ]
    return PredictionResult(
        predicted_word=predicted_word,
        confidence=normalize_percentage(confidence) or 0.0,
        probabilities=normalized_items,
    )
