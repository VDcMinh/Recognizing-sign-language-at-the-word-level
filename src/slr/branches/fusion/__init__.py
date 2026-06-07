"""Fusion datasets and models for combining multiple branch features."""

from slr.branches.fusion.build import (
    build_gated_feature_fusion_from_config,
    load_gated_feature_fusion_config,
)
from slr.branches.fusion.dataset import (
    PairedSkeletonRegionsDataset,
    load_paired_skeleton_regions_config,
    paired_skeleton_regions_collate_fn,
)
from slr.branches.fusion.models import GatedFeatureFusion

__all__ = [
    "GatedFeatureFusion",
    "PairedSkeletonRegionsDataset",
    "build_gated_feature_fusion_from_config",
    "load_gated_feature_fusion_config",
    "load_paired_skeleton_regions_config",
    "paired_skeleton_regions_collate_fn",
]
