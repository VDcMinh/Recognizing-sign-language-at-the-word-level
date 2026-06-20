"""I3D-style RGB backbone for full-bbox appearance modeling.

This is a clean-room implementation inspired by the Inception-I3D design
described in "Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset"
by Carreira and Zisserman (2017). It follows the same high-level design ideas:

- spatio-temporal 3D convolutions
- Inception-style multi-branch 3D mixed blocks
- temporal/spatial pooling through the backbone
- global average pooling before the classifier
- dropout before logits

It is intentionally reported as an I3D-style / Inception3D-like RGB stream in
project reports rather than claimed to be a canonical reproduction with
pretrained Kinetics weights.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


def _triple(value: int | tuple[int, int, int]) -> tuple[int, int, int]:
    """Normalize scalar-or-triple kernel/stride arguments."""

    if isinstance(value, tuple):
        if len(value) != 3:
            raise ValueError(f"Expected a 3-tuple, got {value!r}.")
        return tuple(int(item) for item in value)
    scalar = int(value)
    return scalar, scalar, scalar


def _compute_same_padding(
    size: int,
    kernel_size: int,
    stride: int,
) -> tuple[int, int]:
    """Return left/right padding needed for TensorFlow-style SAME padding."""

    if stride <= 0:
        raise ValueError("stride must be positive.")
    output_size = (size + stride - 1) // stride
    pad_needed = max(0, (output_size - 1) * stride + kernel_size - size)
    pad_before = pad_needed // 2
    pad_after = pad_needed - pad_before
    return pad_before, pad_after


def _pad_same_3d(
    x: torch.Tensor,
    *,
    kernel_size: tuple[int, int, int],
    stride: tuple[int, int, int],
) -> torch.Tensor:
    """Pad one 5D tensor with TensorFlow-style SAME semantics."""

    _, _, time, height, width = x.shape
    pad_t = _compute_same_padding(time, kernel_size[0], stride[0])
    pad_h = _compute_same_padding(height, kernel_size[1], stride[1])
    pad_w = _compute_same_padding(width, kernel_size[2], stride[2])
    pad = (pad_w[0], pad_w[1], pad_h[0], pad_h[1], pad_t[0], pad_t[1])
    if any(value > 0 for value in pad):
        return F.pad(x, pad)
    return x


class Unit3D(nn.Module):
    """Basic I3D-style 3D conv block with SAME padding, BN, and ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int | tuple[int, int, int] = 1,
        stride: int | tuple[int, int, int] = 1,
        use_batch_norm: bool = True,
        activation: bool = True,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride)
        self.use_batch_norm = bool(use_batch_norm)
        self.use_activation = bool(activation)

        self.conv3d = nn.Conv3d(
            self.in_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=0,
            bias=bias,
        )
        self.batch_norm = nn.BatchNorm3d(self.out_channels) if self.use_batch_norm else None
        self.relu = nn.ReLU(inplace=True) if self.use_activation else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _pad_same_3d(x, kernel_size=self.kernel_size, stride=self.stride)
        x = self.conv3d(x)
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class MaxPool3dSamePadding(nn.Module):
    """3D max-pooling with SAME padding semantics."""

    def __init__(
        self,
        kernel_size: int | tuple[int, int, int],
        *,
        stride: int | tuple[int, int, int],
    ) -> None:
        super().__init__()
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride)
        self.pool = nn.MaxPool3d(self.kernel_size, stride=self.stride, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _pad_same_3d(x, kernel_size=self.kernel_size, stride=self.stride)
        return self.pool(x)


class InceptionModule3D(nn.Module):
    """Inception-style 3D mixed block used in I3D-like backbones."""

    def __init__(
        self,
        in_channels: int,
        branch0_out: int,
        branch1_reduce: int,
        branch1_out: int,
        branch2_reduce: int,
        branch2_out: int,
        branch3_out: int,
    ) -> None:
        super().__init__()
        self.branch0 = Unit3D(in_channels, branch0_out, kernel_size=1)
        self.branch1a = Unit3D(in_channels, branch1_reduce, kernel_size=1)
        self.branch1b = Unit3D(branch1_reduce, branch1_out, kernel_size=3)
        self.branch2a = Unit3D(in_channels, branch2_reduce, kernel_size=1)
        self.branch2b = Unit3D(branch2_reduce, branch2_out, kernel_size=3)
        self.branch3a = MaxPool3dSamePadding(kernel_size=3, stride=1)
        self.branch3b = Unit3D(in_channels, branch3_out, kernel_size=1)

    @property
    def output_channels(self) -> int:
        """Return the concatenated output channel count."""

        return (
            self.branch0.out_channels
            + self.branch1b.out_channels
            + self.branch2b.out_channels
            + self.branch3b.out_channels
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch0 = self.branch0(x)
        branch1 = self.branch1b(self.branch1a(x))
        branch2 = self.branch2b(self.branch2a(x))
        branch3 = self.branch3b(self.branch3a(x))
        return torch.cat((branch0, branch1, branch2, branch3), dim=1)


def _load_local_state_dict(path: str | None) -> dict[str, torch.Tensor] | None:
    """Load optional local pretrained weights without any network access."""

    if path is None:
        return None
    checkpoint_path = str(path).strip()
    if not checkpoint_path:
        return None

    payload = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(payload, dict):
        if "model_state_dict" in payload and isinstance(payload["model_state_dict"], dict):
            return payload["model_state_dict"]
        if "state_dict" in payload and isinstance(payload["state_dict"], dict):
            return payload["state_dict"]
        if all(isinstance(value, torch.Tensor) for value in payload.values()):
            return payload
    raise ValueError(
        "pretrained_path did not contain a readable state_dict payload. "
        f"path={checkpoint_path!r}"
    )


class FullBBoxI3D(nn.Module):
    """RGB-only I3D-style classifier for full-bbox clips."""

    def __init__(
        self,
        *,
        in_channels: int = 3,
        num_classes: int = 100,
        dropout: float = 0.5,
        feature_dim: int = 1024,
        pretrained_path: str | None = None,
        freeze_backbone: bool = False,
        return_features: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.dropout_p = float(dropout)
        self.feature_dim = int(feature_dim)
        self.pretrained_path = pretrained_path
        self.freeze_backbone = bool(freeze_backbone)
        self.default_return_features = bool(return_features)

        if self.in_channels != 3:
            raise ValueError("FullBBoxI3D currently supports RGB input with in_channels=3 only.")
        if self.num_classes <= 0:
            raise ValueError("num_classes must be positive.")

        self.conv3d_1a_7x7 = Unit3D(self.in_channels, 64, kernel_size=7, stride=(2, 2, 2))
        self.maxpool3d_2a_3x3 = MaxPool3dSamePadding(kernel_size=(1, 3, 3), stride=(1, 2, 2))
        self.conv3d_2b_1x1 = Unit3D(64, 64, kernel_size=1)
        self.conv3d_2c_3x3 = Unit3D(64, 192, kernel_size=3)
        self.maxpool3d_3a_3x3 = MaxPool3dSamePadding(kernel_size=(1, 3, 3), stride=(1, 2, 2))

        self.mixed_3b = InceptionModule3D(192, 64, 96, 128, 16, 32, 32)
        self.mixed_3c = InceptionModule3D(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3d_4a_3x3 = MaxPool3dSamePadding(kernel_size=3, stride=(2, 2, 2))

        self.mixed_4b = InceptionModule3D(480, 192, 96, 208, 16, 48, 64)
        self.mixed_4c = InceptionModule3D(512, 160, 112, 224, 24, 64, 64)
        self.mixed_4d = InceptionModule3D(512, 128, 128, 256, 24, 64, 64)
        self.mixed_4e = InceptionModule3D(512, 112, 144, 288, 32, 64, 64)
        self.mixed_4f = InceptionModule3D(528, 256, 160, 320, 32, 128, 128)
        self.maxpool3d_5a_2x2 = MaxPool3dSamePadding(kernel_size=2, stride=(2, 2, 2))

        self.mixed_5b = InceptionModule3D(832, 256, 160, 320, 32, 128, 128)
        self.mixed_5c = InceptionModule3D(832, 384, 192, 384, 48, 128, 128)

        if self.mixed_5c.output_channels != self.feature_dim:
            raise ValueError(
                f"feature_dim must match the final backbone channels ({self.mixed_5c.output_channels}), "
                f"got {self.feature_dim}."
            )

        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(p=self.dropout_p)
        self.classifier = nn.Linear(self.feature_dim, self.num_classes)

        if self.pretrained_path:
            state_dict = _load_local_state_dict(self.pretrained_path)
            if state_dict is not None:
                self.load_state_dict(state_dict, strict=False)

        if self.freeze_backbone:
            self._freeze_backbone()

    def _freeze_backbone(self) -> None:
        """Freeze the spatio-temporal backbone while keeping the classifier trainable."""

        for name, parameter in self.named_parameters():
            if not name.startswith("classifier"):
                parameter.requires_grad = False

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return pooled appearance features before the classifier."""

        if x.ndim != 5:
            raise ValueError(
                "FullBBoxI3D expects input shape (B, C, T, H, W), "
                f"got {tuple(x.shape)}."
            )
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {int(x.shape[1])}.")

        x = self.conv3d_1a_7x7(x)
        x = self.maxpool3d_2a_3x3(x)
        x = self.conv3d_2b_1x1(x)
        x = self.conv3d_2c_3x3(x)
        x = self.maxpool3d_3a_3x3(x)
        x = self.mixed_3b(x)
        x = self.mixed_3c(x)
        x = self.maxpool3d_4a_3x3(x)
        x = self.mixed_4b(x)
        x = self.mixed_4c(x)
        x = self.mixed_4d(x)
        x = self.mixed_4e(x)
        x = self.mixed_4f(x)
        x = self.maxpool3d_5a_2x2(x)
        x = self.mixed_5b(x)
        x = self.mixed_5c(x)
        x = self.avg_pool(x)
        return x.flatten(1)

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_features: bool | None = None,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Run one forward pass over RGB video clips."""

        features = self.extract_features(x)
        logits = self.classifier(self.dropout(features))
        use_dict = self.default_return_features if return_features is None else bool(return_features)
        if use_dict:
            return {"logits": logits, "features": features}
        return logits


def build_fullbbox_i3d(model_cfg: dict[str, Any]) -> FullBBoxI3D:
    """Build one I3D-style full-bbox classifier from config."""

    return FullBBoxI3D(
        in_channels=int(model_cfg.get("in_channels", 3)),
        num_classes=int(model_cfg.get("num_classes", 100)),
        dropout=float(model_cfg.get("dropout", 0.5)),
        feature_dim=int(model_cfg.get("feature_dim", 1024)),
        pretrained_path=model_cfg.get("pretrained_path"),
        freeze_backbone=bool(model_cfg.get("freeze_backbone", False)),
        return_features=bool(model_cfg.get("return_features", False)),
    )


__all__ = [
    "FullBBoxI3D",
    "InceptionModule3D",
    "MaxPool3dSamePadding",
    "Unit3D",
    "build_fullbbox_i3d",
]
