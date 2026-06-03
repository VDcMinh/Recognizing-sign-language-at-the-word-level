"""Pretrained ResNet18-GRU classifier for local region clip tensors."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


def _build_resnet18_backbone(*, pretrained: bool) -> tuple[nn.Module, int]:
    """Build a ResNet18 feature extractor without the classification head."""

    try:
        from torchvision.models import resnet18
    except ImportError as exc:
        raise RuntimeError("torchvision is required to build RegionResNet18GRU.") from exc

    try:
        from torchvision.models import ResNet18_Weights

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        try:
            backbone = resnet18(weights=weights)
        except TypeError:
            backbone = resnet18(pretrained=pretrained)
    except ImportError:
        backbone = resnet18(pretrained=pretrained)
    except Exception as exc:
        if pretrained:
            raise RuntimeError(
                "Failed to load pretrained ResNet18 weights. If you are running offline, "
                "set model.pretrained=false for smoke tests or cache the weights first."
            ) from exc
        raise

    encoder = nn.Sequential(*list(backbone.children())[:-1], nn.Flatten(1))
    return encoder, 512


def _freeze_module(module: nn.Module) -> None:
    """Freeze one module for feature extraction mode."""

    for parameter in module.parameters():
        parameter.requires_grad = False
    module.eval()


class RegionResNet18TemporalEncoder(nn.Module):
    """Encode one region clip over time with a ResNet18 frame encoder and GRU."""

    def __init__(
        self,
        *,
        in_channels: int,
        encoder_name: str = "resnet18",
        encoder_feature_dim: int = 512,
        pretrained: bool = True,
        freeze_encoder: bool = True,
        gru_hidden_size: int = 128,
        gru_num_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.5,
        use_valid_mask: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.encoder_name = str(encoder_name).strip().lower()
        self.pretrained = bool(pretrained)
        self.freeze_encoder = bool(freeze_encoder)
        self.use_valid_mask = bool(use_valid_mask)

        if self.in_channels != 3:
            raise ValueError("RegionResNet18TemporalEncoder expects in_channels=3.")
        if self.encoder_name != "resnet18":
            raise ValueError("RegionResNet18TemporalEncoder currently supports encoder_name='resnet18' only.")

        self.frame_encoder, actual_feature_dim = _build_resnet18_backbone(pretrained=self.pretrained)
        self.encoder_feature_dim = int(actual_feature_dim)
        if int(encoder_feature_dim) != self.encoder_feature_dim:
            raise ValueError(
                f"encoder_feature_dim must be {self.encoder_feature_dim} for ResNet18, "
                f"got {encoder_feature_dim}."
            )

        if self.freeze_encoder:
            _freeze_module(self.frame_encoder)

        gru_dropout = float(dropout) if int(gru_num_layers) > 1 else 0.0
        self.temporal_encoder = nn.GRU(
            input_size=self.encoder_feature_dim,
            hidden_size=int(gru_hidden_size),
            num_layers=int(gru_num_layers),
            dropout=gru_dropout,
            bidirectional=bool(bidirectional),
            batch_first=True,
        )
        self.output_dim = int(gru_hidden_size) * (2 if bidirectional else 1)

    def train(self, mode: bool = True) -> "RegionResNet18TemporalEncoder":
        """Keep the frozen encoder in eval mode while the GRU trains."""

        super().train(mode)
        if mode and self.freeze_encoder:
            self.frame_encoder.eval()
        return self

    def _encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        if self.freeze_encoder:
            with torch.no_grad():
                return self.frame_encoder(frames)
        return self.frame_encoder(frames)

    def forward(
        self,
        region_clip: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode one region clip.

        Args:
            region_clip: ``(N, C, T, H, W)``
            valid_mask: ``(N, T)`` with ``1`` for valid frames and ``0`` for invalid crops.

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
        if channels != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {channels}.")

        mask: torch.Tensor | None = None
        if valid_mask is not None:
            if valid_mask.shape != (batch_size, clip_len):
                raise ValueError(
                    "valid_mask must have shape (N, T), "
                    f"got {tuple(valid_mask.shape)} for region clip shape {tuple(region_clip.shape)}."
                )
            mask = valid_mask.to(device=region_clip.device, dtype=region_clip.dtype)

        if self.use_valid_mask and mask is not None:
            region_clip = region_clip * mask.view(batch_size, 1, clip_len, 1, 1)

        frames = region_clip.permute(0, 2, 1, 3, 4).reshape(batch_size * clip_len, channels, height, width)
        frame_features = self._encode_frames(frames).reshape(batch_size, clip_len, self.encoder_feature_dim)
        temporal_outputs, _ = self.temporal_encoder(frame_features)

        if not self.use_valid_mask or mask is None:
            region_feature = temporal_outputs.mean(dim=1)
            return region_feature, temporal_outputs

        mask = mask.to(dtype=temporal_outputs.dtype)
        masked_outputs = temporal_outputs * mask.unsqueeze(-1)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        region_feature = masked_outputs.sum(dim=1) / denom
        return region_feature, temporal_outputs


class RegionResNet18GRU(nn.Module):
    """Classifier over three local image regions using ResNet18 + GRU."""

    def __init__(
        self,
        *,
        num_classes: int,
        num_regions: int = 3,
        in_channels: int = 3,
        clip_len: int = 64,
        crop_size: int = 112,
        pretrained: bool = True,
        freeze_encoder: bool = True,
        encoder_name: str = "resnet18",
        encoder_feature_dim: int = 512,
        gru_hidden_size: int = 128,
        gru_num_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.5,
        fusion: str = "concat",
        use_valid_mask: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.num_regions = int(num_regions)
        self.in_channels = int(in_channels)
        self.clip_len = int(clip_len)
        self.crop_size = int(crop_size)
        self.pretrained = bool(pretrained)
        self.freeze_encoder = bool(freeze_encoder)
        self.encoder_name = str(encoder_name).strip().lower()
        self.encoder_feature_dim = int(encoder_feature_dim)
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

        self.region_encoder = RegionResNet18TemporalEncoder(
            in_channels=self.in_channels,
            encoder_name=self.encoder_name,
            encoder_feature_dim=self.encoder_feature_dim,
            pretrained=self.pretrained,
            freeze_encoder=self.freeze_encoder,
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

        batch_size, num_regions, channels, clip_len, _, _ = data.shape
        if num_regions != self.num_regions:
            raise ValueError(f"Expected {self.num_regions} regions, got {num_regions}.")
        if channels != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {channels}.")
        if clip_len != self.clip_len:
            raise ValueError(f"Expected clip_len={self.clip_len}, got {clip_len}.")

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


def build_region_resnet18_gru(model_cfg: dict[str, Any]) -> RegionResNet18GRU:
    """Build a :class:`RegionResNet18GRU` model from one config dictionary."""

    return RegionResNet18GRU(
        num_classes=int(model_cfg.get("num_classes", 100)),
        num_regions=int(model_cfg.get("num_regions", 3)),
        in_channels=int(model_cfg.get("in_channels", 3)),
        clip_len=int(model_cfg.get("clip_len", 64)),
        crop_size=int(model_cfg.get("crop_size", 112)),
        pretrained=bool(model_cfg.get("pretrained", True)),
        freeze_encoder=bool(model_cfg.get("freeze_encoder", True)),
        encoder_name=str(model_cfg.get("encoder_name", "resnet18")),
        encoder_feature_dim=int(model_cfg.get("encoder_feature_dim", 512)),
        gru_hidden_size=int(model_cfg.get("gru_hidden_size", 128)),
        gru_num_layers=int(model_cfg.get("gru_num_layers", 1)),
        bidirectional=bool(model_cfg.get("bidirectional", True)),
        dropout=float(model_cfg.get("dropout", 0.5)),
        fusion=str(model_cfg.get("fusion", "concat")),
        use_valid_mask=bool(model_cfg.get("use_valid_mask", True)),
    )


__all__ = [
    "RegionResNet18TemporalEncoder",
    "RegionResNet18GRU",
    "build_region_resnet18_gru",
]
