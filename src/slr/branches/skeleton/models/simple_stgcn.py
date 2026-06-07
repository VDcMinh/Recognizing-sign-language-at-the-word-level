"""A small trainable ST-GCN-style baseline for skeleton graph tensors."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn


class GraphConv2d(nn.Module):
    """Apply a learnable projection followed by adjacency aggregation."""

    def __init__(self, in_channels: int, out_channels: int, num_subsets: int) -> None:
        super().__init__()
        self.out_channels = int(out_channels)
        self.num_subsets = int(num_subsets)
        self.proj = nn.Conv2d(
            in_channels,
            out_channels * num_subsets,
            kernel_size=1,
            bias=False,
        )

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected graph features with shape (N, C, T, V), got {tuple(x.shape)}.")
        if adjacency.ndim != 3:
            raise ValueError(
                f"Expected adjacency with shape (K, V, V), got {tuple(adjacency.shape)}."
            )

        projected = self.proj(x)
        batch_size, _, num_frames, num_nodes = projected.shape
        projected = projected.view(
            batch_size,
            self.num_subsets,
            self.out_channels,
            num_frames,
            num_nodes,
        )
        return torch.einsum("nkctv,kvw->nctw", projected, adjacency)


class STGCNBlock(nn.Module):
    """One lightweight spatial-temporal graph block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_subsets: int,
        *,
        stride: int = 1,
        dropout: float = 0.0,
        residual: bool = True,
    ) -> None:
        super().__init__()
        self.gcn = GraphConv2d(in_channels, out_channels, num_subsets)
        self.gcn_bn = nn.BatchNorm2d(out_channels)
        self.temporal = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(9, 1),
                stride=(stride, 1),
                padding=(4, 0),
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout),
        )

        if not residual:
            self.residual = None
        elif in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1),
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        residual = 0 if self.residual is None else self.residual(x)
        x = self.gcn(x, adjacency)
        x = self.gcn_bn(x)
        x = self.temporal(x)
        x = x + residual
        return self.relu(x)


class SimpleSTGCN(nn.Module):
    """A compact skeleton baseline that keeps the interface close to ST-GCN."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        num_nodes: int,
        adjacency: np.ndarray | torch.Tensor | Any,
        hidden_channels: int = 64,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        adjacency_tensor = torch.as_tensor(adjacency, dtype=torch.float32)
        if adjacency_tensor.ndim != 3:
            raise ValueError(
                f"Expected adjacency with shape (K, V, V), got {tuple(adjacency_tensor.shape)}."
            )
        if adjacency_tensor.shape[-1] != num_nodes or adjacency_tensor.shape[-2] != num_nodes:
            raise ValueError(
                f"Adjacency node dimensions {tuple(adjacency_tensor.shape[-2:])} do not match "
                f"num_nodes={num_nodes}."
            )

        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.num_nodes = int(num_nodes)
        self.hidden_channels = int(hidden_channels)
        self.dropout_rate = float(dropout)

        self.register_buffer("A", adjacency_tensor, persistent=True)
        num_subsets = int(adjacency_tensor.shape[0])

        self.input_bn = nn.BatchNorm1d(self.in_channels * self.num_nodes)
        self.blocks = nn.ModuleList(
            [
                STGCNBlock(
                    self.in_channels,
                    self.hidden_channels,
                    num_subsets,
                    residual=False,
                    dropout=self.dropout_rate * 0.5,
                ),
                STGCNBlock(
                    self.hidden_channels,
                    self.hidden_channels,
                    num_subsets,
                    dropout=self.dropout_rate,
                ),
                STGCNBlock(
                    self.hidden_channels,
                    self.hidden_channels * 2,
                    num_subsets,
                    stride=2,
                    dropout=self.dropout_rate,
                ),
                STGCNBlock(
                    self.hidden_channels * 2,
                    self.hidden_channels * 2,
                    num_subsets,
                    dropout=self.dropout_rate,
                ),
            ]
        )
        self.feature_dim = self.hidden_channels * 2
        self.feature_dropout = nn.Dropout(self.dropout_rate)
        self.classifier = nn.Linear(self.feature_dim, self.num_classes)

    def _forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(
                f"Expected input with shape (N, C, T, V, M), got {tuple(x.shape)}."
            )

        batch_size, in_channels, num_frames, num_nodes, num_persons = x.shape
        if in_channels != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {in_channels}."
            )
        if num_nodes != self.num_nodes:
            raise ValueError(f"Expected {self.num_nodes} graph nodes, got {num_nodes}.")
        if num_persons <= 0:
            raise ValueError("Input must contain at least one person dimension.")

        if num_persons == 1:
            x = x[..., 0]
        else:
            x = x.mean(dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous().view(batch_size, num_nodes * in_channels, num_frames)
        x = self.input_bn(x)
        x = x.view(batch_size, num_nodes, in_channels, num_frames).permute(0, 2, 3, 1).contiguous()

        for block in self.blocks:
            x = block(x, self.A)

        return x.mean(dim=(2, 3))

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return pooled skeleton features before the classifier."""

        return self._forward_backbone(x)

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        features = self.extract_features(x)
        logits = self.classifier(self.feature_dropout(features))
        if return_features:
            return logits, features
        return logits

    @property
    def output_dim(self) -> int:
        """Return the pooled feature dimension before the classifier."""

        return int(self.feature_dim)
