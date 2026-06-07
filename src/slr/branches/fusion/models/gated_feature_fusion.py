"""Gated feature fusion over skeleton and regions backbones."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


def _freeze_module(module: nn.Module) -> None:
    """Freeze one backbone in-place."""

    for parameter in module.parameters():
        parameter.requires_grad = False
    module.eval()


class GatedFeatureFusion(nn.Module):
    """Fuse pre-classifier skeleton and region features with a learned gate."""

    def __init__(
        self,
        *,
        skeleton_model: nn.Module,
        regions_model: nn.Module,
        skeleton_dim: int,
        region_dim: int,
        hidden_dim: int,
        num_classes: int,
        proj_dropout: float = 0.2,
        classifier_dropout: float = 0.5,
        freeze_skeleton: bool = True,
        freeze_regions: bool = True,
    ) -> None:
        super().__init__()
        self.skeleton_model = skeleton_model
        self.regions_model = regions_model
        self.skeleton_dim = int(skeleton_dim)
        self.region_dim = int(region_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        self.freeze_skeleton = bool(freeze_skeleton)
        self.freeze_regions = bool(freeze_regions)

        if not hasattr(self.skeleton_model, "extract_features"):
            raise TypeError("skeleton_model must expose extract_features(x).")
        if not hasattr(self.regions_model, "extract_features"):
            raise TypeError("regions_model must expose extract_features(x, valid_mask=None).")

        if self.freeze_skeleton:
            _freeze_module(self.skeleton_model)
        if self.freeze_regions:
            _freeze_module(self.regions_model)

        self.skeleton_proj = nn.Sequential(
            nn.Linear(self.skeleton_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(proj_dropout)),
        )
        self.region_proj = nn.Sequential(
            nn.Linear(self.region_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(proj_dropout)),
        )
        self.gate_network = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Sigmoid(),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(float(classifier_dropout)),
            nn.Linear(self.hidden_dim, self.num_classes),
        )

    def train(self, mode: bool = True) -> "GatedFeatureFusion":
        """Keep frozen backbones in eval mode while training the fusion head."""

        super().train(mode)
        if mode and self.freeze_skeleton:
            self.skeleton_model.eval()
        if mode and self.freeze_regions:
            self.regions_model.eval()
        return self

    def _extract_skeleton_features(self, skeleton_x: torch.Tensor) -> torch.Tensor:
        if self.freeze_skeleton:
            with torch.no_grad():
                return self.skeleton_model.extract_features(skeleton_x)
        return self.skeleton_model.extract_features(skeleton_x)

    def _extract_region_features(
        self,
        regions_x: torch.Tensor,
        *,
        regions_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.freeze_regions:
            with torch.no_grad():
                return self.regions_model.extract_features(
                    regions_x,
                    valid_mask=regions_valid_mask,
                )
        return self.regions_model.extract_features(
            regions_x,
            valid_mask=regions_valid_mask,
        )

    def forward(
        self,
        skeleton_x: torch.Tensor,
        regions_x: torch.Tensor,
        return_features: bool = False,
        regions_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        skeleton_feature = self._extract_skeleton_features(skeleton_x)
        region_feature = self._extract_region_features(
            regions_x,
            regions_valid_mask=regions_valid_mask,
        )

        skeleton_proj = self.skeleton_proj(skeleton_feature)
        region_proj = self.region_proj(region_feature)
        gate = self.gate_network(torch.cat([skeleton_proj, region_proj], dim=-1))
        fused = gate * skeleton_proj + (1.0 - gate) * region_proj
        logits = self.classifier(fused)

        if not return_features:
            return logits
        return logits, {
            "skeleton_feature": skeleton_feature,
            "region_feature": region_feature,
            "skeleton_proj": skeleton_proj,
            "region_proj": region_proj,
            "gate": gate,
            "fused": fused,
        }

    @property
    def output_dim(self) -> int:
        """Return the fusion hidden size consumed by the classifier."""

        return int(self.hidden_dim)

    def extra_repr(self) -> str:
        return (
            f"skeleton_dim={self.skeleton_dim}, region_dim={self.region_dim}, "
            f"hidden_dim={self.hidden_dim}, num_classes={self.num_classes}, "
            f"freeze_skeleton={self.freeze_skeleton}, freeze_regions={self.freeze_regions}"
        )


__all__ = ["GatedFeatureFusion"]
