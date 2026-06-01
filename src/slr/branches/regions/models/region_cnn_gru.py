"""Lightweight CNN-GRU baseline for local region clip tensors."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class FrameCNNEncoder(nn.Module):
    """Encode one RGB crop frame into a compact feature vector."""

    def __init__(
        self,
        *,
        in_channels: int = 3,
        feature_dim: int = 256,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.projection = nn.Linear(128, feature_dim)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode ``(N, C, H, W)`` frames to ``(N, D)`` features."""

        features = self.features(frames)
        features = features.flatten(1)
        return self.projection(features)


class RegionTemporalEncoder(nn.Module):
    """Encode one region clip over time with a shared CNN and GRU."""

    def __init__(
        self,
        *,
        in_channels: int,
        cnn_feature_dim: int,
        gru_hidden_size: int,
        gru_num_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.3,
        use_valid_mask: bool = True,
    ) -> None:
        super().__init__()
        self.use_valid_mask = bool(use_valid_mask)
        self.frame_encoder = FrameCNNEncoder(
            in_channels=in_channels,
            feature_dim=cnn_feature_dim,
        )
        gru_dropout = float(dropout) if int(gru_num_layers) > 1 else 0.0
        self.temporal_encoder = nn.GRU(
            input_size=cnn_feature_dim,
            hidden_size=gru_hidden_size,
            num_layers=int(gru_num_layers),
            dropout=gru_dropout,
            bidirectional=bool(bidirectional),
            batch_first=True,
        )
        self.output_dim = int(gru_hidden_size) * (2 if bidirectional else 1)

    def forward(
        self,
        region_clip: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode one region clip.

        Args:
            region_clip: ``(N, C, T, H, W)``
            valid_mask: ``(N, T)`` with ``1`` for valid frames and ``0`` for black crops.

        Returns:
            region_feature: ``(N, F)``
            temporal_outputs: ``(N, T, F)``
        """

        if region_clip.ndim != 5:
            raise ValueError(
                "region_clip must have shape (N, C, T, H, W), "
                f"got {tuple(region_clip.shape)}."
            )

        batch_size, channels, clip_len, height, width = region_clip.shape
        frames = region_clip.permute(0, 2, 1, 3, 4).reshape(batch_size * clip_len, channels, height, width)
        frame_features = self.frame_encoder(frames).reshape(batch_size, clip_len, -1)
        temporal_outputs, _ = self.temporal_encoder(frame_features)

        if not self.use_valid_mask or valid_mask is None:
            region_feature = temporal_outputs.mean(dim=1)
            return region_feature, temporal_outputs

        if valid_mask.shape != (batch_size, clip_len):
            raise ValueError(
                "valid_mask must have shape (N, T), "
                f"got {tuple(valid_mask.shape)} for region clip shape {tuple(region_clip.shape)}."
            )

        mask = valid_mask.to(device=temporal_outputs.device, dtype=temporal_outputs.dtype)
        masked_outputs = temporal_outputs * mask.unsqueeze(-1)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        region_feature = masked_outputs.sum(dim=1) / denom
        return region_feature, temporal_outputs


class RegionCNNGRU(nn.Module):
    """Baseline CNN-GRU classifier over three local image regions."""

    def __init__(
        self,
        *,
        num_classes: int,
        num_regions: int = 3,
        in_channels: int = 3,
        clip_len: int = 64,
        crop_size: int = 112,
        cnn_feature_dim: int = 256,
        gru_hidden_size: int = 256,
        gru_num_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.3,
        fusion: str = "concat",
        use_valid_mask: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.num_regions = int(num_regions)
        self.in_channels = int(in_channels)
        self.clip_len = int(clip_len)
        self.crop_size = int(crop_size)
        self.cnn_feature_dim = int(cnn_feature_dim)
        self.gru_hidden_size = int(gru_hidden_size)
        self.gru_num_layers = int(gru_num_layers)
        self.bidirectional = bool(bidirectional)
        self.dropout_p = float(dropout)
        self.fusion = str(fusion).strip().lower()
        self.use_valid_mask = bool(use_valid_mask)

        if self.num_regions <= 0:
            raise ValueError("num_regions must be positive.")
        if self.num_classes <= 0:
            raise ValueError("num_classes must be positive.")
        if self.fusion not in {"concat", "average"}:
            raise ValueError("fusion must be one of: 'concat', 'average'.")

        self.region_encoder = RegionTemporalEncoder(
            in_channels=self.in_channels,
            cnn_feature_dim=self.cnn_feature_dim,
            gru_hidden_size=self.gru_hidden_size,
            gru_num_layers=self.gru_num_layers,
            bidirectional=self.bidirectional,
            dropout=self.dropout_p,
            use_valid_mask=self.use_valid_mask,
        )
        self.region_feature_dim = self.region_encoder.output_dim
        classifier_input_dim = (
            self.region_feature_dim * self.num_regions
            if self.fusion == "concat"
            else self.region_feature_dim
        )
        self.dropout = nn.Dropout(p=self.dropout_p)
        self.classifier = nn.Linear(classifier_input_dim, self.num_classes)

    def forward(
        self,
        data: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute classification logits.

        Args:
            data: ``(N, R, C, T, H, W)``
            valid_mask: ``(N, R, T)``
        """

        if data.ndim != 6:
            raise ValueError(
                "data must have shape (N, R, C, T, H, W), "
                f"got {tuple(data.shape)}."
            )

        batch_size, num_regions, channels, clip_len, height, width = data.shape
        if num_regions != self.num_regions:
            raise ValueError(
                f"Expected {self.num_regions} regions, got {num_regions}."
            )
        if channels != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} channels, got {channels}."
            )
        if clip_len != self.clip_len:
            raise ValueError(
                f"Expected clip_len={self.clip_len}, got {clip_len}."
            )

        if valid_mask is not None and valid_mask.shape != (batch_size, num_regions, clip_len):
            raise ValueError(
                "valid_mask must have shape (N, R, T), "
                f"got {tuple(valid_mask.shape)}."
            )

        region_features: list[torch.Tensor] = []
        for region_index in range(self.num_regions):
            region_clip = data[:, region_index, :, :, :, :]
            region_mask = valid_mask[:, region_index, :] if valid_mask is not None else None
            region_feature, _ = self.region_encoder(region_clip, valid_mask=region_mask)
            region_features.append(region_feature)

        stacked_features = torch.stack(region_features, dim=1)
        if self.fusion == "concat":
            fused = stacked_features.reshape(batch_size, self.num_regions * self.region_feature_dim)
        else:
            fused = stacked_features.mean(dim=1)
        fused = self.dropout(fused)
        return self.classifier(fused)

    @property
    def output_dim(self) -> int:
        """Return the fused feature dimension before the classifier."""

        if self.fusion == "concat":
            return self.region_feature_dim * self.num_regions
        return self.region_feature_dim


def build_region_cnn_gru(model_cfg: dict[str, Any]) -> RegionCNNGRU:
    """Build a :class:`RegionCNNGRU` model from one config dictionary."""

    return RegionCNNGRU(
        num_classes=int(model_cfg.get("num_classes", 100)),
        num_regions=int(model_cfg.get("num_regions", 3)),
        in_channels=int(model_cfg.get("in_channels", 3)),
        clip_len=int(model_cfg.get("clip_len", 64)),
        crop_size=int(model_cfg.get("crop_size", 112)),
        cnn_feature_dim=int(model_cfg.get("cnn_feature_dim", 256)),
        gru_hidden_size=int(model_cfg.get("gru_hidden_size", 256)),
        gru_num_layers=int(model_cfg.get("gru_num_layers", 1)),
        bidirectional=bool(model_cfg.get("bidirectional", True)),
        dropout=float(model_cfg.get("dropout", 0.3)),
        fusion=str(model_cfg.get("fusion", "concat")),
        use_valid_mask=bool(model_cfg.get("use_valid_mask", True)),
    )


__all__ = ["FrameCNNEncoder", "RegionTemporalEncoder", "RegionCNNGRU", "build_region_cnn_gru"]
