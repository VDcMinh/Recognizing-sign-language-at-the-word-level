"""Transforms for face and hand crop sequences."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def normalize_uint8(image: np.ndarray) -> np.ndarray:
    """Scale uint8 images to the ``[0, 1]`` range."""

    return image.astype(np.float32) / 255.0


def normalize_region_clip_uint8(clip: np.ndarray) -> np.ndarray:
    """Scale a region clip tensor from ``uint8`` to ``float32`` in ``[0, 1]``."""

    return clip.astype(np.float32) / 255.0


def normalize_region_augmentation_config(config: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize one optional augmentation config block."""

    raw = dict(config or {})
    color_jitter = dict(raw.get("color_jitter", {}))
    random_resized_crop = dict(raw.get("random_resized_crop", {}))
    random_erasing = dict(raw.get("random_erasing", {}))
    temporal_dropout = dict(raw.get("temporal_dropout", {}))
    region_dropout = dict(raw.get("region_dropout", {}))
    crop_scale = random_resized_crop.get("scale", [0.85, 1.0])
    erase_scale = random_erasing.get("scale", [0.02, 0.12])
    erase_ratio = random_erasing.get("ratio", [0.3, 3.3])

    return {
        "enabled": bool(raw.get("enabled", False)),
        "color_jitter": {
            "enabled": bool(color_jitter.get("enabled", False)),
            "brightness": float(color_jitter.get("brightness", 0.0)),
            "contrast": float(color_jitter.get("contrast", 0.0)),
            "saturation": float(color_jitter.get("saturation", 0.0)),
            "hue": float(color_jitter.get("hue", 0.0)),
        },
        "random_resized_crop": {
            "enabled": bool(random_resized_crop.get("enabled", False)),
            "scale": [float(crop_scale[0]), float(crop_scale[1])],
        },
        "random_erasing": {
            "enabled": bool(random_erasing.get("enabled", False)),
            "p": float(random_erasing.get("p", 0.0)),
            "scale": [float(erase_scale[0]), float(erase_scale[1])],
            "ratio": [float(erase_ratio[0]), float(erase_ratio[1])],
            "value": float(random_erasing.get("value", 0.0)),
        },
        "temporal_dropout": {
            "enabled": bool(temporal_dropout.get("enabled", False)),
            "p": float(temporal_dropout.get("p", 0.0)),
        },
        "region_dropout": {
            "enabled": bool(region_dropout.get("enabled", False)),
            "p": float(region_dropout.get("p", 0.0)),
        },
    }


def apply_region_clip_augmentation(
    data: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
    config: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Apply train-time region augmentations while preserving ``(R, C, T, H, W)``."""

    normalized = normalize_region_augmentation_config(config)
    if not normalized["enabled"]:
        return data, valid_mask
    if data.ndim != 5:
        raise ValueError(f"Region clip augmentation expects (R, C, T, H, W), got {tuple(data.shape)}.")

    augmented = data.clone()
    augmented_mask = valid_mask.clone() if valid_mask is not None else None

    if normalized["random_resized_crop"]["enabled"]:
        augmented = _apply_random_resized_crop(augmented, normalized["random_resized_crop"])
    if normalized["color_jitter"]["enabled"]:
        augmented = _apply_color_jitter(augmented, normalized["color_jitter"])
    if normalized["random_erasing"]["enabled"]:
        augmented = _apply_random_erasing(augmented, normalized["random_erasing"])
    if normalized["temporal_dropout"]["enabled"]:
        augmented, augmented_mask = _apply_temporal_dropout(
            augmented,
            augmented_mask,
            normalized["temporal_dropout"],
        )
    if normalized["region_dropout"]["enabled"]:
        augmented, augmented_mask = _apply_region_dropout(
            augmented,
            augmented_mask,
            normalized["region_dropout"],
        )

    return augmented.clamp_(0.0, 1.0), augmented_mask


def _sample_uniform(low: float, high: float, *, device: torch.device) -> float:
    if high <= low:
        return float(low)
    return float(torch.empty((), device=device).uniform_(float(low), float(high)).item())


def _sample_int(low: int, high: int, *, device: torch.device) -> int:
    if high <= low:
        return int(low)
    return int(torch.randint(low, high + 1, size=(1,), device=device).item())


def _apply_random_resized_crop(data: torch.Tensor, config: dict[str, Any]) -> torch.Tensor:
    scale_min, scale_max = [float(value) for value in config.get("scale", [0.85, 1.0])]
    _, _, _, height, width = data.shape
    crop_scale = _sample_uniform(scale_min, scale_max, device=data.device)
    crop_height = max(1, min(height, int(round(height * crop_scale))))
    crop_width = max(1, min(width, int(round(width * crop_scale))))
    top = _sample_int(0, height - crop_height, device=data.device)
    left = _sample_int(0, width - crop_width, device=data.device)

    frames = data.permute(0, 2, 1, 3, 4).reshape(-1, data.shape[1], height, width)
    cropped = frames[:, :, top : top + crop_height, left : left + crop_width]
    resized = F.interpolate(cropped, size=(height, width), mode="bilinear", align_corners=False)
    return resized.reshape(data.shape[0], data.shape[2], data.shape[1], height, width).permute(0, 2, 1, 3, 4)


def _apply_color_jitter(data: torch.Tensor, config: dict[str, Any]) -> torch.Tensor:
    frames = data.permute(0, 2, 1, 3, 4).reshape(-1, data.shape[1], data.shape[3], data.shape[4])
    brightness = float(config.get("brightness", 0.0))
    contrast = float(config.get("contrast", 0.0))
    saturation = float(config.get("saturation", 0.0))
    hue = float(config.get("hue", 0.0))

    if brightness > 0.0:
        factor = _sample_uniform(max(0.0, 1.0 - brightness), 1.0 + brightness, device=data.device)
        frames = frames * factor
    if contrast > 0.0:
        factor = _sample_uniform(max(0.0, 1.0 - contrast), 1.0 + contrast, device=data.device)
        mean = frames.mean(dim=(1, 2, 3), keepdim=True)
        frames = (frames - mean) * factor + mean
    if saturation > 0.0 and frames.shape[1] == 3:
        factor = _sample_uniform(max(0.0, 1.0 - saturation), 1.0 + saturation, device=data.device)
        gray = _rgb_to_grayscale(frames)
        frames = (frames - gray) * factor + gray
    if hue > 0.0 and frames.shape[1] == 3:
        hue_delta = _sample_uniform(-hue, hue, device=data.device)
        frames = _adjust_hue(frames, hue_delta)

    frames = frames.clamp_(0.0, 1.0)
    return frames.reshape(data.shape[0], data.shape[2], data.shape[1], data.shape[3], data.shape[4]).permute(
        0, 2, 1, 3, 4
    )


def _rgb_to_grayscale(frames: torch.Tensor) -> torch.Tensor:
    weights = frames.new_tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1)
    return (frames * weights).sum(dim=1, keepdim=True)


def _apply_random_erasing(data: torch.Tensor, config: dict[str, Any]) -> torch.Tensor:
    probability = float(config.get("p", 0.0))
    if probability <= 0.0 or float(torch.rand((), device=data.device).item()) >= probability:
        return data

    scale_min, scale_max = [float(value) for value in config.get("scale", [0.02, 0.12])]
    ratio_min, ratio_max = [float(value) for value in config.get("ratio", [0.3, 3.3])]
    erase_value = float(config.get("value", 0.0))
    _, _, _, height, width = data.shape
    area = float(height * width)

    for _ in range(10):
        target_area = area * _sample_uniform(scale_min, scale_max, device=data.device)
        aspect_ratio = _sample_uniform(ratio_min, ratio_max, device=data.device)
        erase_height = int(round((target_area * aspect_ratio) ** 0.5))
        erase_width = int(round((target_area / aspect_ratio) ** 0.5))
        if 0 < erase_height < height and 0 < erase_width < width:
            top = _sample_int(0, height - erase_height, device=data.device)
            left = _sample_int(0, width - erase_width, device=data.device)
            data[:, :, :, top : top + erase_height, left : left + erase_width] = erase_value
            return data
    return data


def _apply_temporal_dropout(
    data: torch.Tensor,
    valid_mask: torch.Tensor | None,
    config: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    probability = float(config.get("p", 0.0))
    if probability <= 0.0:
        return data, valid_mask

    clip_len = int(data.shape[2])
    dropped = torch.rand((clip_len,), device=data.device) < probability
    if bool(dropped.any()):
        data[:, :, dropped, :, :] = 0.0
        if valid_mask is not None:
            valid_mask[:, dropped] = 0.0
    return data, valid_mask


def _apply_region_dropout(
    data: torch.Tensor,
    valid_mask: torch.Tensor | None,
    config: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    probability = float(config.get("p", 0.0))
    if probability <= 0.0:
        return data, valid_mask

    num_regions = int(data.shape[0])
    dropped = torch.rand((num_regions,), device=data.device) < probability
    if bool(dropped.any()):
        data[dropped, :, :, :, :] = 0.0
        if valid_mask is not None:
            valid_mask[dropped, :] = 0.0
    return data, valid_mask


def _adjust_hue(frames: torch.Tensor, hue_delta: float) -> torch.Tensor:
    hsv = _rgb_to_hsv(frames.clamp(0.0, 1.0))
    hsv[:, 0, :, :] = (hsv[:, 0, :, :] + float(hue_delta)) % 1.0
    return _hsv_to_rgb(hsv)


def _rgb_to_hsv(frames: torch.Tensor) -> torch.Tensor:
    red, green, blue = frames[:, 0, :, :], frames[:, 1, :, :], frames[:, 2, :, :]
    maxc, _ = frames.max(dim=1)
    minc, _ = frames.min(dim=1)
    delta = maxc - minc

    saturation = torch.where(maxc > 0, delta / maxc.clamp_min(1e-6), torch.zeros_like(delta))
    hue = torch.zeros_like(maxc)

    nonzero = delta > 1e-6
    red_mask = nonzero & (maxc == red)
    green_mask = nonzero & (maxc == green)
    blue_mask = nonzero & (maxc == blue)

    hue[red_mask] = ((green - blue)[red_mask] / delta[red_mask]) % 6.0
    hue[green_mask] = ((blue - red)[green_mask] / delta[green_mask]) + 2.0
    hue[blue_mask] = ((red - green)[blue_mask] / delta[blue_mask]) + 4.0
    hue = (hue / 6.0) % 1.0

    return torch.stack((hue, saturation, maxc), dim=1)


def _hsv_to_rgb(frames: torch.Tensor) -> torch.Tensor:
    hue = frames[:, 0, :, :] * 6.0
    saturation = frames[:, 1, :, :]
    value = frames[:, 2, :, :]

    chroma = value * saturation
    x_term = chroma * (1.0 - torch.abs(torch.remainder(hue, 2.0) - 1.0))
    zeros = torch.zeros_like(chroma)

    red = torch.zeros_like(chroma)
    green = torch.zeros_like(chroma)
    blue = torch.zeros_like(chroma)

    masks = [
        (0.0 <= hue) & (hue < 1.0),
        (1.0 <= hue) & (hue < 2.0),
        (2.0 <= hue) & (hue < 3.0),
        (3.0 <= hue) & (hue < 4.0),
        (4.0 <= hue) & (hue < 5.0),
        (5.0 <= hue) & (hue <= 6.0),
    ]
    values = [
        (chroma, x_term, zeros),
        (x_term, chroma, zeros),
        (zeros, chroma, x_term),
        (zeros, x_term, chroma),
        (x_term, zeros, chroma),
        (chroma, zeros, x_term),
    ]

    for mask, (r_value, g_value, b_value) in zip(masks, values):
        red = torch.where(mask, r_value, red)
        green = torch.where(mask, g_value, green)
        blue = torch.where(mask, b_value, blue)

    match = value - chroma
    return torch.stack((red + match, green + match, blue + match), dim=1)


__all__ = [
    "apply_region_clip_augmentation",
    "normalize_region_augmentation_config",
    "normalize_region_clip_uint8",
    "normalize_uint8",
]
