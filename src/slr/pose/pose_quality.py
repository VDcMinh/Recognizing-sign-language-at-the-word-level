"""Pose quality helpers for shared RTMW-l outputs."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from slr.pose.pose_schema import get_keypoint_region_indices


def confidence_coverage(keypoints: np.ndarray, threshold: float = 0.2) -> float:
    """Compute the fraction of pose points whose confidence exceeds ``threshold``."""

    if keypoints.size == 0:
        return 0.0
    scores = keypoints[..., 2]
    return float((scores >= threshold).mean())


def compute_mean_confidence(keypoints: np.ndarray) -> float:
    """Compute mean confidence over all frames and keypoints."""

    if keypoints.size == 0:
        return 0.0
    return float(np.nanmean(keypoints[..., 2]))


def compute_region_mean_confidence(keypoints: np.ndarray, region_name: str) -> float:
    """Compute mean confidence for one named whole-body region."""

    if keypoints.size == 0:
        return 0.0
    region_indices = list(get_keypoint_region_indices(region_name))
    return float(np.nanmean(keypoints[:, region_indices, 2]))


def compute_valid_frames(
    keypoints: np.ndarray,
    confidence_threshold: float = 0.01,
) -> tuple[int, float, int]:
    """Return valid frame count, valid frame ratio, and missing frame count."""

    if keypoints.size == 0 or keypoints.shape[0] == 0:
        return 0, 0.0, 0

    frame_confidence = np.nanmean(keypoints[..., 2], axis=1)
    valid_mask = frame_confidence > confidence_threshold
    valid_frames = int(valid_mask.sum())
    total_frames = int(keypoints.shape[0])
    missing_frames = total_frames - valid_frames
    valid_frames_ratio = float(valid_frames / total_frames) if total_frames else 0.0
    return valid_frames, valid_frames_ratio, missing_frames


def summarize_pose_manifest(
    manifest: pd.DataFrame,
    min_mean_confidence: float,
    min_valid_frames_ratio: float,
) -> dict[str, Any]:
    """Summarize pose-manifest quality metrics for reporting."""

    ok_manifest = manifest[manifest["status"] == "ok"].copy()
    status_counts = manifest["status"].value_counts().to_dict()
    summary = {
        "num_samples": int(len(manifest)),
        "num_ok": int(len(ok_manifest)),
        "num_errors": int(len(manifest) - len(ok_manifest)),
        "status_counts": status_counts,
        "total_frames_pose": int(ok_manifest["num_frames_pose"].fillna(0).sum())
        if not ok_manifest.empty
        else 0,
        "mean_confidence_avg": float(ok_manifest["mean_confidence"].mean())
        if not ok_manifest.empty
        else 0.0,
        "valid_frames_ratio_avg": float(ok_manifest["valid_frames_ratio"].mean())
        if not ok_manifest.empty
        else 0.0,
        "body_mean_confidence_avg": float(ok_manifest["body_mean_confidence"].mean())
        if not ok_manifest.empty
        else 0.0,
        "face_mean_confidence_avg": float(ok_manifest["face_mean_confidence"].mean())
        if not ok_manifest.empty
        else 0.0,
        "left_hand_mean_confidence_avg": float(ok_manifest["left_hand_mean_confidence"].mean())
        if not ok_manifest.empty
        else 0.0,
        "right_hand_mean_confidence_avg": float(ok_manifest["right_hand_mean_confidence"].mean())
        if not ok_manifest.empty
        else 0.0,
        "low_mean_confidence_samples": int(
            (ok_manifest["mean_confidence"] < min_mean_confidence).sum()
        )
        if not ok_manifest.empty
        else 0,
        "low_valid_frames_ratio_samples": int(
            (ok_manifest["valid_frames_ratio"] < min_valid_frames_ratio).sum()
        )
        if not ok_manifest.empty
        else 0,
    }
    return summary
