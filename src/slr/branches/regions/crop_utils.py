"""Cropping helpers for local-image region inputs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from slr.branches.regions.region_schema import FACE_68, LEFT_HAND_21, RIGHT_HAND_21
from slr.pose.pose_schema import BODY_FACE_ANCHORS, WHOLEBODY_133_NUM_KEYPOINTS
from slr.utils.bbox import BoundingBox, clip_bbox, is_valid_bbox, square_bbox
from slr.utils.image import crop_image, make_black_image, resize_letterbox


@dataclass(frozen=True)
class RegionBBoxResult:
    """Resolved bbox metadata for one region and frame."""

    box: BoundingBox | None
    mean_confidence: float
    num_valid_points: int
    source: str


def _bbox_side_length(box: BoundingBox) -> float:
    """Return the larger side length for one bbox."""

    return max(float(box.x2 - box.x1), float(box.y2 - box.y1))


def _validate_frame_keypoints(frame_keypoints: np.ndarray) -> np.ndarray:
    """Validate and normalize one wholebody_133 frame."""

    array = np.asarray(frame_keypoints, dtype=np.float32)
    if array.shape != (WHOLEBODY_133_NUM_KEYPOINTS, 3):
        raise ValueError(
            "Expected frame keypoints with shape "
            f"({WHOLEBODY_133_NUM_KEYPOINTS}, 3), got {array.shape}."
        )
    return array


def _point_mask(points: np.ndarray, conf_thr: float) -> np.ndarray:
    """Return the visibility mask for one point set."""

    return (
        np.isfinite(points[:, 0])
        & np.isfinite(points[:, 1])
        & np.isfinite(points[:, 2])
        & (points[:, 2] >= float(conf_thr))
    )


def _point_stats(points: np.ndarray, conf_thr: float) -> tuple[np.ndarray, int, float]:
    """Return valid points, count, and mean confidence for one region."""

    mask = _point_mask(points, conf_thr=conf_thr)
    valid = points[mask]
    count = int(valid.shape[0])
    mean_confidence = float(valid[:, 2].mean()) if count > 0 else 0.0
    return valid, count, mean_confidence


def bbox_from_points(
    points: np.ndarray,
    image_w: int,
    image_h: int,
    conf_thr: float,
    margin: float,
    min_points: int,
    min_bbox_size_px: float | None = None,
    max_bbox_size_ratio: float | None = None,
) -> BoundingBox | None:
    """Build a clipped square bbox from visible points."""

    point_array = np.asarray(points, dtype=np.float32)
    if point_array.ndim != 2 or point_array.shape[1] != 3:
        raise ValueError(f"Expected points with shape (N, 3), got {point_array.shape}.")
    if image_w <= 0 or image_h <= 0:
        raise ValueError("image_w and image_h must be positive.")

    valid_points, count, _ = _point_stats(point_array, conf_thr=conf_thr)
    if count < int(min_points):
        return None

    min_xy = valid_points[:, :2].min(axis=0)
    max_xy = valid_points[:, :2].max(axis=0)
    base_box = BoundingBox(
        x1=float(min_xy[0]),
        y1=float(min_xy[1]),
        x2=float(max_xy[0]),
        y2=float(max_xy[1]),
    )
    clipped = clip_bbox(square_bbox(base_box, scale=float(margin)), width=image_w, height=image_h)
    if not is_valid_bbox(clipped):
        return None

    side_length = _bbox_side_length(clipped)
    if min_bbox_size_px is not None and side_length < float(min_bbox_size_px):
        return None
    if max_bbox_size_ratio is not None:
        max_allowed_side = float(max_bbox_size_ratio) * float(min(image_w, image_h))
        if side_length > max_allowed_side:
            return None
    return clipped


def face_bbox_from_wholebody133(
    frame_keypoints: np.ndarray,
    image_w: int,
    image_h: int,
    conf_thr: float,
    primary_margin: float,
    fallback_margin: float,
    min_face_points: int = 8,
    min_anchor_points: int = 2,
    min_bbox_size_px: float | None = None,
    max_bbox_size_ratio: float | None = None,
) -> RegionBBoxResult:
    """Build a face bbox using face landmarks first and body anchors as fallback."""

    keypoints = _validate_frame_keypoints(frame_keypoints)

    face_points = keypoints[list(FACE_68)]
    face_valid, face_count, face_mean_conf = _point_stats(face_points, conf_thr=conf_thr)
    if face_count >= int(min_face_points):
        face_box = bbox_from_points(
            face_valid,
            image_w=image_w,
            image_h=image_h,
            conf_thr=conf_thr,
            margin=primary_margin,
            min_points=min_face_points,
            min_bbox_size_px=min_bbox_size_px,
            max_bbox_size_ratio=max_bbox_size_ratio,
        )
        if face_box is not None:
            return RegionBBoxResult(
                box=face_box,
                mean_confidence=face_mean_conf,
                num_valid_points=face_count,
                source="face_68",
            )

    anchor_points = keypoints[list(BODY_FACE_ANCHORS)]
    anchor_valid, anchor_count, anchor_mean_conf = _point_stats(anchor_points, conf_thr=conf_thr)
    if anchor_count >= int(min_anchor_points):
        anchor_box = bbox_from_points(
            anchor_valid,
            image_w=image_w,
            image_h=image_h,
            conf_thr=conf_thr,
            margin=fallback_margin,
            min_points=min_anchor_points,
            min_bbox_size_px=min_bbox_size_px,
            max_bbox_size_ratio=max_bbox_size_ratio,
        )
        if anchor_box is not None:
            return RegionBBoxResult(
                box=anchor_box,
                mean_confidence=anchor_mean_conf,
                num_valid_points=anchor_count,
                source="body_face_anchors",
            )

    return RegionBBoxResult(box=None, mean_confidence=0.0, num_valid_points=0, source="none")


def hand_bbox_from_wholebody133(
    frame_keypoints: np.ndarray,
    region_name: str,
    image_w: int,
    image_h: int,
    conf_thr: float,
    margin: float,
    min_points: int = 2,
    min_bbox_size_px: float | None = None,
    max_bbox_size_ratio: float | None = None,
) -> RegionBBoxResult:
    """Build one hand bbox from the wholebody_133 frame."""

    keypoints = _validate_frame_keypoints(frame_keypoints)
    if region_name == "left_hand":
        indices = LEFT_HAND_21
    elif region_name == "right_hand":
        indices = RIGHT_HAND_21
    else:
        raise ValueError(f"Unsupported hand region: {region_name!r}.")

    hand_points = keypoints[list(indices)]
    valid_points, count, mean_confidence = _point_stats(hand_points, conf_thr=conf_thr)
    if count < int(min_points):
        return RegionBBoxResult(box=None, mean_confidence=0.0, num_valid_points=count, source="none")

    hand_box = bbox_from_points(
        valid_points,
        image_w=image_w,
        image_h=image_h,
        conf_thr=conf_thr,
        margin=margin,
        min_points=min_points,
        min_bbox_size_px=min_bbox_size_px,
        max_bbox_size_ratio=max_bbox_size_ratio,
    )
    return RegionBBoxResult(
        box=hand_box,
        mean_confidence=mean_confidence if hand_box is not None else 0.0,
        num_valid_points=count if hand_box is not None else 0,
        source=region_name if hand_box is not None else "none",
    )


def black_fallback_crop(output_size: int) -> np.ndarray:
    """Return a black crop with the configured square output size."""

    return make_black_image(width=output_size, height=output_size)


def crop_and_resize(
    image: np.ndarray,
    box: BoundingBox | None,
    output_size: int,
) -> np.ndarray:
    """Crop one bbox from an image and resize it into a square canvas."""

    if box is None:
        return black_fallback_crop(output_size)
    crop = crop_image(image, box)
    if crop.size == 0:
        return black_fallback_crop(output_size)
    return resize_letterbox(crop, target_width=output_size, target_height=output_size)


__all__ = [
    "RegionBBoxResult",
    "bbox_from_points",
    "black_fallback_crop",
    "crop_and_resize",
    "face_bbox_from_wholebody133",
    "hand_bbox_from_wholebody133",
]
