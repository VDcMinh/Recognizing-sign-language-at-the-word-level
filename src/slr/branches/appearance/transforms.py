"""Spatial preprocessing and light augmentation for full-bbox RGB clips."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps


def _to_bool(value: Any, default: bool = False) -> bool:
    """Convert config values to bool with a stable default."""

    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def normalize_appearance_preprocessing_config(config: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize preprocessing config with safe defaults."""

    cfg = dict(config or {})
    train_aug = dict(cfg.get("train_augmentation", {}))
    color_jitter = train_aug.get("color_jitter", False)
    if isinstance(color_jitter, dict):
        color_jitter_cfg = dict(color_jitter)
        color_jitter_enabled = _to_bool(color_jitter_cfg.get("enabled", True), default=True)
    else:
        color_jitter_cfg = {}
        color_jitter_enabled = _to_bool(color_jitter, default=False)

    return {
        "resize_mode": str(cfg.get("resize_mode", "letterbox")).strip().lower(),
        "mean": [float(value) for value in cfg.get("mean", [0.45, 0.45, 0.45])],
        "std": [float(value) for value in cfg.get("std", [0.225, 0.225, 0.225])],
        "pad_value": int(cfg.get("pad_value", 0)),
        "train_augmentation": {
            "color_jitter": {
                "enabled": bool(color_jitter_enabled),
                "brightness": float(color_jitter_cfg.get("brightness", 0.1)),
                "contrast": float(color_jitter_cfg.get("contrast", 0.1)),
                "saturation": float(color_jitter_cfg.get("saturation", 0.1)),
                "probability": float(color_jitter_cfg.get("probability", 0.8)),
            },
            "horizontal_flip": _to_bool(train_aug.get("horizontal_flip", False), default=False),
            "horizontal_flip_prob": float(train_aug.get("horizontal_flip_prob", 0.5)),
        },
    }


def _letterbox_image(image: Image.Image, *, output_size: int, pad_value: int = 0) -> Image.Image:
    """Resize one RGB frame while preserving aspect ratio, then pad to a square."""

    if output_size <= 0:
        raise ValueError("output_size must be positive.")
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size: {image.size}")

    scale = min(float(output_size) / float(width), float(output_size) / float(height))
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = image.resize((new_width, new_height), resample=Image.BILINEAR)

    canvas = Image.new("RGB", (output_size, output_size), color=(pad_value, pad_value, pad_value))
    offset_x = (output_size - new_width) // 2
    offset_y = (output_size - new_height) // 2
    canvas.paste(resized, (offset_x, offset_y))
    return canvas


def _resize_frame(
    image: Image.Image,
    *,
    output_size: int,
    resize_mode: str,
    pad_value: int,
) -> Image.Image:
    """Apply deterministic spatial resizing without risky center crops."""

    mode = str(resize_mode).strip().lower()
    if mode not in {"letterbox", "resize"}:
        raise ValueError(f"Unsupported resize_mode {resize_mode!r}. Expected 'letterbox' or 'resize'.")
    if mode == "resize":
        return image.resize((output_size, output_size), resample=Image.BILINEAR)
    return _letterbox_image(image, output_size=output_size, pad_value=pad_value)


def _jitter_clip(frames: list[Image.Image], *, config: dict[str, Any]) -> list[Image.Image]:
    """Apply the same lightweight color jitter to every frame in a clip."""

    if not frames:
        return frames
    if not bool(config.get("enabled", False)):
        return frames
    if random.random() > float(config.get("probability", 0.8)):
        return frames

    brightness = float(config.get("brightness", 0.1))
    contrast = float(config.get("contrast", 0.1))
    saturation = float(config.get("saturation", 0.1))

    brightness_factor = 1.0 + random.uniform(-brightness, brightness)
    contrast_factor = 1.0 + random.uniform(-contrast, contrast)
    saturation_factor = 1.0 + random.uniform(-saturation, saturation)

    jittered: list[Image.Image] = []
    for frame in frames:
        current = ImageEnhance.Brightness(frame).enhance(brightness_factor)
        current = ImageEnhance.Contrast(current).enhance(contrast_factor)
        current = ImageEnhance.Color(current).enhance(saturation_factor)
        jittered.append(current)
    return jittered


def _maybe_horizontal_flip(frames: list[Image.Image], *, enabled: bool, probability: float) -> list[Image.Image]:
    """Optionally flip all frames in one clip together."""

    if not frames or not enabled:
        return frames
    if random.random() > float(probability):
        return frames
    return [ImageOps.mirror(frame) for frame in frames]


@dataclass
class AppearanceTransformPipeline:
    """Transform one clip of RGB PIL frames into a `C x T x H x W` tensor."""

    input_size: int
    transform_mode: str
    config: dict[str, Any]

    def __post_init__(self) -> None:
        self.transform_mode = str(self.transform_mode).strip().lower()
        if self.transform_mode not in {"train", "eval", "val", "test"}:
            raise ValueError(
                f"Unsupported transform_mode {self.transform_mode!r}. "
                "Expected one of: train, eval, val, test."
            )
        if int(self.input_size) <= 0:
            raise ValueError("input_size must be positive.")

    def __call__(self, frames: list[Image.Image]) -> torch.Tensor:
        if not frames:
            raise ValueError("frames must not be empty.")

        normalized_cfg = normalize_appearance_preprocessing_config(self.config)
        working = [frame.convert("RGB") for frame in frames]

        if self.transform_mode == "train":
            aug_cfg = normalized_cfg["train_augmentation"]
            working = _maybe_horizontal_flip(
                working,
                enabled=bool(aug_cfg["horizontal_flip"]),
                probability=float(aug_cfg["horizontal_flip_prob"]),
            )
            working = _jitter_clip(
                working,
                config=dict(aug_cfg["color_jitter"]),
            )

        resized = [
            _resize_frame(
                frame,
                output_size=int(self.input_size),
                resize_mode=str(normalized_cfg["resize_mode"]),
                pad_value=int(normalized_cfg["pad_value"]),
            )
            for frame in working
        ]

        tensors: list[torch.Tensor] = []
        mean = torch.tensor(normalized_cfg["mean"], dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor(normalized_cfg["std"], dtype=torch.float32).view(3, 1, 1)
        for frame in resized:
            array = np.asarray(frame, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
            tensor = (tensor - mean) / std
            tensors.append(tensor)

        clip = torch.stack(tensors, dim=0)  # T x C x H x W
        return clip.permute(1, 0, 2, 3).contiguous()  # C x T x H x W


def build_appearance_transform(
    *,
    input_size: int,
    transform_mode: str,
    config: dict[str, Any] | None,
) -> AppearanceTransformPipeline:
    """Build one appearance transform pipeline."""

    normalized = normalize_appearance_preprocessing_config(config)
    return AppearanceTransformPipeline(
        input_size=int(input_size),
        transform_mode=str(transform_mode),
        config=normalized,
    )


__all__ = [
    "AppearanceTransformPipeline",
    "build_appearance_transform",
    "normalize_appearance_preprocessing_config",
]
