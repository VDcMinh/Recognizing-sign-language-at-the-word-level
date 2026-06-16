"""Visualize reduced skeleton keypoints on standardized frames."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from slr.pose.pose_schema import get_keypoint_set_names
from slr.utils.image import read_image, save_image
from slr.utils.io import ensure_dir, write_text


BODY_COLOR = (255, 220, 0)
LEFT_HAND_COLOR = (0, 200, 0)
RIGHT_HAND_COLOR = (0, 140, 255)
MOUTH_COLOR = (255, 80, 120)
TEXT_COLOR = (240, 240, 240)
PANEL_BG = (24, 24, 24)
LEGEND_BG = (36, 36, 36)
FRAME_BORDER = (72, 72, 72)

BODY_INDICES = tuple(range(0, 7))
LEFT_HAND_INDICES = tuple(range(7, 17))
RIGHT_HAND_INDICES = tuple(range(17, 27))
MOUTH_INDICES_BY_SET = {
    "selected_27": (),
    "selected_31": tuple(range(27, 31)),
}
BODY_EDGES = ((0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6))
LEFT_HAND_EDGES = ((7, 8), (9, 10), (11, 12), (13, 14), (15, 16))
RIGHT_HAND_EDGES = ((17, 18), (19, 20), (21, 22), (23, 24), (25, 26))


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(
        description="Visualize selected_27 or selected_31 overlays on standardized frames."
    )
    parser.add_argument(
        "--keypoint-set",
        type=str,
        default="selected_31",
        choices=["selected_27", "selected_31"],
        help="Reduced keypoint set to visualize.",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="nslt100",
        help="Subset name to visualize.",
    )
    parser.add_argument(
        "--samples-per-split",
        type=int,
        default=2,
        help="How many samples to auto-select per split.",
    )
    parser.add_argument(
        "--sample-ids",
        nargs="*",
        default=None,
        help="Optional explicit sample IDs to visualize.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where visualization PNGs and report are written.",
    )
    return parser


def _read_csv(path: Path) -> pd.DataFrame:
    """Read one CSV file with stable string columns."""

    return pd.read_csv(path)


def load_manifests(keypoint_set: str, subset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load skeleton and pose manifests for one subset."""

    skeleton_manifest = _read_csv(
        Path("data/datasets/WLASL/branch_inputs/skeleton/rtmw_l/manifests")
        / f"{subset}_{keypoint_set}_all.csv"
    )
    pose_manifest = _read_csv(
        Path("data/datasets/WLASL/pose/rtmw_l/manifests")
        / f"{subset}_all.csv"
    )
    return skeleton_manifest, pose_manifest


def select_samples(
    skeleton_manifest: pd.DataFrame,
    pose_manifest: pd.DataFrame,
    samples_per_split: int,
    sample_ids: list[str] | None,
) -> pd.DataFrame:
    """Resolve which samples to visualize."""

    merged = skeleton_manifest.merge(
        pose_manifest[
            [
                "sample_id",
                "split",
                "frames_dir",
                "left_hand_mean_confidence",
                "right_hand_mean_confidence",
                "mean_confidence",
            ]
        ],
        on=["sample_id", "split"],
        how="left",
        suffixes=("", "_pose"),
    )
    merged["sample_id"] = merged["sample_id"].astype(str)
    merged["hand_score"] = (
        pd.to_numeric(merged["left_hand_mean_confidence"], errors="coerce").fillna(0.0)
        + pd.to_numeric(merged["right_hand_mean_confidence"], errors="coerce").fillna(0.0)
    )

    if sample_ids:
        selected = merged[merged["sample_id"].isin(sample_ids)].copy()
        return selected.sort_values(by=["split", "sample_id"]).reset_index(drop=True)

    selected_parts: list[pd.DataFrame] = []
    for split in ("train", "val", "test"):
        split_frame = merged[merged["split"] == split].copy()
        split_frame = split_frame.sort_values(
            by=["hand_score", "mean_confidence", "sample_id"],
            ascending=[False, False, True],
        )
        selected_parts.append(split_frame.head(samples_per_split))
    return pd.concat(selected_parts, ignore_index=True)


def resolve_frame_paths(frames_dir: Path) -> list[Path]:
    """Collect standardized frame paths for one sample."""

    return sorted(path for path in frames_dir.glob("*.jpg") if path.is_file())


def resolve_frames_dir(row: pd.Series) -> Path:
    """Resolve the local standardized-frames directory for one sample."""

    sample_id = str(row["sample_id"])
    split = str(row["split"])
    derived = (
        Path("data/datasets/WLASL/standardized/frames")
        / "nslt100"
        / split
        / sample_id
    )
    if derived.exists():
        return derived

    frames_text = str(row["frames_dir"])
    manifest_dir = Path(frames_text)
    if manifest_dir.exists():
        return manifest_dir

    normalized_text = frames_text.replace("\\", "/")
    marker = "/data/datasets/WLASL/standardized/frames/"
    if marker in normalized_text:
        suffix = normalized_text.split(marker, 1)[1]
        remapped = Path("data/datasets/WLASL/standardized/frames") / Path(suffix)
        if remapped.exists():
            return remapped

    return derived


def choose_frame_indices(num_frames: int) -> list[int]:
    """Choose three representative frame indices."""

    if num_frames <= 0:
        raise ValueError("No frames available.")
    if num_frames == 1:
        return [0, 0, 0]
    if num_frames == 2:
        return [0, 1, 1]
    return [0, num_frames // 2, num_frames - 1]


def get_group_specs(keypoint_set: str) -> list[dict[str, Any]]:
    """Return draw-group definitions for one reduced keypoint set."""

    specs = [
        {"indices": BODY_INDICES, "color": BODY_COLOR, "prefix": "B"},
        {"indices": LEFT_HAND_INDICES, "color": LEFT_HAND_COLOR, "prefix": "L"},
        {"indices": RIGHT_HAND_INDICES, "color": RIGHT_HAND_COLOR, "prefix": "R"},
    ]
    mouth_indices = MOUTH_INDICES_BY_SET.get(keypoint_set, ())
    if mouth_indices:
        specs.append({"indices": mouth_indices, "color": MOUTH_COLOR, "prefix": "M"})
    return specs


def build_legend_subtitle(keypoint_set: str) -> str:
    """Return the legend subtitle for one keypoint set."""

    subtitle = "Body = cyan | Left hand = green | Right hand = orange"
    if MOUTH_INDICES_BY_SET.get(keypoint_set):
        subtitle += " | Mouth = pink"
    return subtitle


def draw_keypoint_group(
    image: np.ndarray,
    keypoints: np.ndarray,
    indices: tuple[int, ...],
    color: tuple[int, int, int],
    label_prefix: str,
) -> None:
    """Draw one colored group of keypoints with tiny labels."""

    for global_index in indices:
        x, y, score = keypoints[global_index]
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        center = (int(round(float(x))), int(round(float(y))))
        radius = 4 if score > 0 else 3
        cv2.circle(image, center, radius, color, thickness=-1)
        cv2.circle(image, center, radius + 2, (0, 0, 0), thickness=1)
        cv2.putText(
            image,
            f"{label_prefix}{global_index}",
            (center[0] + 4, center[1] - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
            cv2.LINE_AA,
        )


def draw_edges(
    image: np.ndarray,
    keypoints: np.ndarray,
    edges: tuple[tuple[int, int], ...],
    color: tuple[int, int, int],
) -> None:
    """Draw lines between selected keypoint pairs."""

    for start, end in edges:
        x1, y1 = keypoints[start, :2]
        x2, y2 = keypoints[end, :2]
        if not (np.isfinite(x1) and np.isfinite(y1) and np.isfinite(x2) and np.isfinite(y2)):
            continue
        cv2.line(
            image,
            (int(round(float(x1))), int(round(float(y1)))),
            (int(round(float(x2))), int(round(float(y2)))),
            color,
            thickness=2,
            lineType=cv2.LINE_AA,
        )


def draw_overlay(
    image: np.ndarray,
    keypoints: np.ndarray,
    frame_label: str,
    keypoint_set: str,
) -> np.ndarray:
    """Draw the full selected-keypoint overlay onto one frame."""

    canvas = image.copy()
    draw_edges(canvas, keypoints, BODY_EDGES, BODY_COLOR)
    draw_edges(canvas, keypoints, LEFT_HAND_EDGES, LEFT_HAND_COLOR)
    draw_edges(canvas, keypoints, RIGHT_HAND_EDGES, RIGHT_HAND_COLOR)

    for spec in get_group_specs(keypoint_set):
        draw_keypoint_group(
            canvas,
            keypoints,
            spec["indices"],
            spec["color"],
            spec["prefix"],
        )

    cv2.putText(
        canvas,
        frame_label,
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        TEXT_COLOR,
        2,
        cv2.LINE_AA,
    )
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1] - 1, canvas.shape[0] - 1), FRAME_BORDER, 1)
    return canvas


def build_legend_panel(width: int, height: int, keypoint_set: str) -> np.ndarray:
    """Build a legend panel that explains index-to-name mapping."""

    panel = np.full((height, width, 3), LEGEND_BG, dtype=np.uint8)
    cv2.putText(panel, f"{keypoint_set} legend", (14, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2, cv2.LINE_AA)
    cv2.putText(
        panel,
        build_legend_subtitle(keypoint_set),
        (14, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        TEXT_COLOR,
        1,
        cv2.LINE_AA,
    )

    names = get_keypoint_set_names(keypoint_set)
    y = 78
    for index, name in enumerate(names):
        if index in BODY_INDICES:
            color = BODY_COLOR
            prefix = "B"
        elif index in LEFT_HAND_INDICES:
            color = LEFT_HAND_COLOR
            prefix = "L"
        elif index in MOUTH_INDICES_BY_SET.get(keypoint_set, ()):
            color = MOUTH_COLOR
            prefix = "M"
        else:
            color = RIGHT_HAND_COLOR
            prefix = "R"
        cv2.circle(panel, (18, y - 4), 5, color, thickness=-1)
        cv2.putText(
            panel,
            f"{prefix}{index:02d}: {name}",
            (30, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            TEXT_COLOR,
            1,
            cv2.LINE_AA,
        )
        y += 20
    return panel


def build_contact_sheet(
    sample_id: str,
    split: str,
    gloss: str,
    frame_images: list[np.ndarray],
    hand_score: float,
    frame_indices: list[int],
    keypoint_set: str,
) -> np.ndarray:
    """Assemble three overlays plus one legend panel into a contact sheet."""

    frame_height, frame_width = frame_images[0].shape[:2]
    legend_width = 340
    title_height = 58
    canvas = np.full(
        (title_height + frame_height, frame_width * 3 + legend_width, 3),
        PANEL_BG,
        dtype=np.uint8,
    )

    title = f"{split} | {sample_id} | {gloss} | {keypoint_set} | hand_score={hand_score:.3f}"
    cv2.putText(canvas, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, TEXT_COLOR, 2, cv2.LINE_AA)
    cv2.putText(canvas, f"Frames visualized: {frame_indices}", (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.48, TEXT_COLOR, 1, cv2.LINE_AA)

    for idx, frame_image in enumerate(frame_images):
        x0 = idx * frame_width
        canvas[title_height : title_height + frame_height, x0 : x0 + frame_width] = frame_image

    legend = build_legend_panel(legend_width, frame_height, keypoint_set)
    canvas[title_height : title_height + frame_height, frame_width * 3 : frame_width * 3 + legend_width] = legend
    return canvas


def visualize_one_sample(row: pd.Series, output_dir: Path, keypoint_set: str) -> Path:
    """Render one sample contact sheet and save it as PNG."""

    sample_id = str(row["sample_id"])
    split = str(row["split"])
    gloss = str(row["gloss"])
    frames_dir = resolve_frames_dir(row)
    selected_path = Path(str(row["selected_path"]))
    hand_score = float(row.get("hand_score", 0.0))

    frame_paths = resolve_frame_paths(frames_dir)
    if not frame_paths:
        raise FileNotFoundError(f"No frame files found for sample {sample_id}: {frames_dir}")

    with np.load(selected_path, allow_pickle=False) as payload:
        keypoints = payload["keypoints"].astype(np.float32)

    if keypoints.shape[0] != len(frame_paths):
        usable = min(keypoints.shape[0], len(frame_paths))
        keypoints = keypoints[:usable]
        frame_paths = frame_paths[:usable]

    frame_indices = choose_frame_indices(len(frame_paths))
    overlays: list[np.ndarray] = []
    for frame_index in frame_indices:
        frame_image = read_image(frame_paths[frame_index])
        overlay = draw_overlay(
            frame_image,
            keypoints[frame_index],
            f"frame {frame_index + 1}/{len(frame_paths)}",
            keypoint_set=keypoint_set,
        )
        overlays.append(overlay)

    sheet = build_contact_sheet(
        sample_id=sample_id,
        split=split,
        gloss=gloss,
        frame_images=overlays,
        hand_score=hand_score,
        frame_indices=frame_indices,
        keypoint_set=keypoint_set,
    )
    output_path = output_dir / f"{split}_{sample_id}_{gloss}.png"
    save_image(output_path, sheet)
    return output_path


def build_report(
    output_dir: Path,
    selected_samples: pd.DataFrame,
    image_paths: list[Path],
    keypoint_set: str,
) -> str:
    """Build a short Markdown index for the generated visualizations."""

    lines = [
        f"# {keypoint_set} Visualization Report",
        "",
        (
            "Muc tieu: kiem tra nhanh mapping body/left-hand/right-hand"
            + (" va mouth" if keypoint_set == "selected_31" else "")
            + f" cua {keypoint_set} bang overlay tren standardized frames."
        ),
        "",
        "## Samples",
        "",
    ]

    for _, row in selected_samples.iterrows():
        lines.append(
            f"- {row['split']} | {row['sample_id']} | {row['gloss']} | hand_score={float(row['hand_score']):.3f}"
        )

    lines.extend(["", "## Images", ""])
    for path in image_paths:
        lines.append(f"- `{path.as_posix()}`")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    """CLI entrypoint."""

    parser = build_parser()
    args = parser.parse_args()

    output_dir = ensure_dir(
        args.output_dir
        or Path("reports/preprocessing") / f"{args.keypoint_set}_visualization" / args.subset
    )
    skeleton_manifest, pose_manifest = load_manifests(args.keypoint_set, args.subset)
    selected_samples = select_samples(
        skeleton_manifest=skeleton_manifest,
        pose_manifest=pose_manifest,
        samples_per_split=args.samples_per_split,
        sample_ids=args.sample_ids,
    )

    image_paths: list[Path] = []
    for _, row in selected_samples.iterrows():
        image_paths.append(visualize_one_sample(row, output_dir, keypoint_set=args.keypoint_set))

    report_path = output_dir / "README.md"
    report_text = build_report(output_dir, selected_samples, image_paths, keypoint_set=args.keypoint_set)
    write_text(report_text, report_path)

    print(f"Generated {len(image_paths)} visualization(s) in {output_dir}")
    for path in image_paths:
        print(path.as_posix())
    print(report_path.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
