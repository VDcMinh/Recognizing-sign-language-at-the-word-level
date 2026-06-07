"""Repo-local ST-GCN++-compatible model for skeleton graph tensors.

This implementation is a clean-room PyTorch design inspired by the public
ST-GCN++ family of spatial-temporal graph models. It intentionally avoids
copying upstream MMAction2/PYSKL source so the training stack stays lightweight
and dependency-free for local and Kaggle use.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch import nn


def _as_int_list(values: Sequence[int] | None, default: Sequence[int]) -> list[int]:
    """Normalize one optional sequence of integers."""

    source = default if values is None else values
    return [int(value) for value in source]


class SpatialGraphConv(nn.Module):
    """Project node features, then aggregate them with a partitioned adjacency."""

    def __init__(self, in_channels: int, out_channels: int, num_subsets: int) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.num_subsets = int(num_subsets)
        self.proj = nn.Conv2d(
            self.in_channels,
            self.out_channels * self.num_subsets,
            kernel_size=1,
            bias=False,
        )
        self.edge_importance = nn.Parameter(torch.ones(self.num_subsets, 1, 1))

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"Expected graph features with shape (N, C, T, V), got {tuple(x.shape)}."
            )
        if adjacency.ndim != 3:
            raise ValueError(
                f"Expected adjacency with shape (K, V, V), got {tuple(adjacency.shape)}."
            )
        if adjacency.shape[0] != self.num_subsets:
            raise ValueError(
                f"Adjacency subset count {adjacency.shape[0]} does not match num_subsets="
                f"{self.num_subsets}."
            )
        if adjacency.shape[-1] != x.shape[-1] or adjacency.shape[-2] != x.shape[-1]:
            raise ValueError(
                f"Adjacency node dimensions {tuple(adjacency.shape[-2:])} do not match input "
                f"node count {x.shape[-1]}."
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
        effective_adjacency = adjacency * self.edge_importance
        return torch.einsum("nkctv,kvw->nctw", projected, effective_adjacency)


class MultiScaleTemporalConv(nn.Module):
    """Multi-branch temporal mixing similar to ST-GCN++-style TCN stages."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int = 1,
        dropout: float = 0.0,
        kernel_sizes: Sequence[int] = (3, 5),
        dilations: Sequence[int] = (1, 2),
    ) -> None:
        super().__init__()
        if len(kernel_sizes) != len(dilations):
            raise ValueError("kernel_sizes and dilations must have the same length.")

        num_branches = len(kernel_sizes) + 2
        if out_channels % num_branches != 0:
            raise ValueError(
                f"out_channels={out_channels} must be divisible by the number of temporal "
                f"branches ({num_branches})."
            )

        branch_channels = out_channels // num_branches
        branches: list[nn.Module] = []
        for kernel_size, dilation in zip(kernel_sizes, dilations, strict=True):
            padding = ((int(kernel_size) - 1) // 2) * int(dilation)
            branches.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(branch_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        branch_channels,
                        branch_channels,
                        kernel_size=(int(kernel_size), 1),
                        stride=(int(stride), 1),
                        padding=(padding, 0),
                        dilation=(int(dilation), 1),
                        bias=False,
                    ),
                    nn.BatchNorm2d(branch_channels),
                )
            )

        branches.append(
            nn.Sequential(
                nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(3, 1), stride=(int(stride), 1), padding=(1, 0)),
                nn.BatchNorm2d(branch_channels),
            )
        )
        branches.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    branch_channels,
                    kernel_size=1,
                    stride=(int(stride), 1),
                    bias=False,
                ),
                nn.BatchNorm2d(branch_channels),
            )
        )

        self.branches = nn.ModuleList(branches)
        self.project = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(float(dropout)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch_outputs = [branch(x) for branch in self.branches]
        return self.project(torch.cat(branch_outputs, dim=1))


class STGCNPPBlock(nn.Module):
    """One spatial-temporal residual block for the repo-local ST-GCN++ model."""

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
        self.gcn = SpatialGraphConv(in_channels, out_channels, num_subsets)
        self.gcn_bn = nn.BatchNorm2d(out_channels)
        self.tcn = MultiScaleTemporalConv(
            out_channels,
            out_channels,
            stride=stride,
            dropout=dropout,
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
                    stride=(int(stride), 1),
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        residual = 0 if self.residual is None else self.residual(x)
        x = self.gcn(x, adjacency)
        x = self.gcn_bn(x)
        x = self.relu(x)
        x = self.tcn(x)
        x = x + residual
        return self.relu(x)


class STGCNPP(nn.Module):
    """Dependency-free ST-GCN++-compatible classifier for precomputed graph tensors."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        num_nodes: int,
        adjacency: np.ndarray | torch.Tensor | Any,
        base_channels: int = 64,
        stage_channels: Sequence[int] | None = None,
        temporal_strides: Sequence[int] | None = None,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        adjacency_tensor = torch.as_tensor(adjacency, dtype=torch.float32)
        if adjacency_tensor.ndim != 3:
            raise ValueError(
                f"Expected adjacency with shape (K, V, V), got {tuple(adjacency_tensor.shape)}."
            )
        if adjacency_tensor.shape[-1] != int(num_nodes) or adjacency_tensor.shape[-2] != int(num_nodes):
            raise ValueError(
                f"Adjacency node dimensions {tuple(adjacency_tensor.shape[-2:])} do not match "
                f"num_nodes={int(num_nodes)}."
            )

        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.num_nodes = int(num_nodes)
        self.base_channels = int(base_channels)
        self.dropout_rate = float(dropout)

        default_stage_channels = (
            self.base_channels,
            self.base_channels,
            self.base_channels,
            self.base_channels * 2,
            self.base_channels * 2,
            self.base_channels * 4,
        )
        stage_channels_list = _as_int_list(stage_channels, default_stage_channels)
        temporal_stride_list = _as_int_list(temporal_strides, (1, 1, 1, 2, 1, 2))
        if len(stage_channels_list) != len(temporal_stride_list):
            raise ValueError("stage_channels and temporal_strides must have the same length.")

        self.register_buffer("A", adjacency_tensor, persistent=True)
        num_subsets = int(adjacency_tensor.shape[0])
        self.data_bn = nn.BatchNorm1d(self.in_channels * self.num_nodes)

        blocks: list[nn.Module] = []
        current_channels = self.in_channels
        for block_index, (out_channels, stride) in enumerate(
            zip(stage_channels_list, temporal_stride_list, strict=True)
        ):
            blocks.append(
                STGCNPPBlock(
                    current_channels,
                    int(out_channels),
                    num_subsets,
                    stride=int(stride),
                    dropout=self.dropout_rate,
                    residual=block_index != 0,
                )
            )
            current_channels = int(out_channels)
        self.blocks = nn.ModuleList(blocks)
        self.feature_dim = int(current_channels)
        self.feature_dropout = nn.Dropout(self.dropout_rate)
        self.classifier = nn.Linear(self.feature_dim, self.num_classes)

    def _forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(
                f"Expected input with shape (N, C, T, V, M), got {tuple(x.shape)}."
            )

        batch_size, in_channels, num_frames, num_nodes, num_persons = x.shape
        if in_channels != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {in_channels}.")
        if num_nodes != self.num_nodes:
            raise ValueError(f"Expected {self.num_nodes} graph nodes, got {num_nodes}.")
        if num_persons <= 0:
            raise ValueError("Input must contain at least one person dimension.")

        x = x.permute(0, 4, 1, 2, 3).contiguous().view(
            batch_size * num_persons,
            in_channels,
            num_frames,
            num_nodes,
        )
        x = x.permute(0, 3, 1, 2).contiguous().view(
            batch_size * num_persons,
            num_nodes * in_channels,
            num_frames,
        )
        x = self.data_bn(x)
        x = x.view(batch_size * num_persons, num_nodes, in_channels, num_frames).permute(
            0, 2, 3, 1
        ).contiguous()

        for block in self.blocks:
            x = block(x, self.A)

        x = x.mean(dim=(2, 3))
        x = x.view(batch_size, num_persons, -1).mean(dim=1)
        return x

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return pooled ST-GCN++ features before the classifier."""

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


__all__ = ["STGCNPP", "STGCNPPBlock", "SpatialGraphConv", "MultiScaleTemporalConv"]
