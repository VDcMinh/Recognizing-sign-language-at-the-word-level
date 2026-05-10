"""
Test thủ công RTMW-l / MMPose trên 1 video.

Cách chạy từ root project:

    python scripts/tools/test_rtmw_mmpose_video.py --video "data/datasets/WLASL/raw/videos/example.mp4"

Có thể chỉ định device:

    python scripts/tools/test_rtmw_mmpose_video.py --video "path/to/video.mp4" --device cuda:0
    python scripts/tools/test_rtmw_mmpose_video.py --video "path/to/video.mp4" --device cpu

Output test được ghi vào:

    scripts/tools/_rtmw_mmpose_video_test_output/

File này chỉ phục vụ smoke test. Có thể xóa cả file này và thư mục output nếu không cần nữa.
"""

from __future__ import annotations

import argparse
import json
import shutil
import traceback
from pathlib import Path
from typing import Any

import torch


TEST_OUTPUT_DIR = Path(__file__).resolve().parent / "_rtmw_mmpose_video_test_output"
EXPECTED_KEYPOINTS = 133


def find_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def find_rtmw_files(project_root: Path) -> tuple[Path, Path]:
    ckpt_dir = project_root / "checkpoints" / "pose" / "rtmw_l"

    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy checkpoint dir: {ckpt_dir}")

    config_candidates = sorted(ckpt_dir.glob("*rtmw*.py"))
    checkpoint_candidates = sorted(ckpt_dir.glob("*.pth"))

    if not config_candidates:
        raise FileNotFoundError(f"Không tìm thấy config RTMW-l .py trong: {ckpt_dir}")

    if not checkpoint_candidates:
        raise FileNotFoundError(f"Không tìm thấy checkpoint .pth trong: {ckpt_dir}")

    return config_candidates[0], checkpoint_candidates[0]


def as_posix_str(path: Path) -> str:
    return str(path.resolve())


def collect_files(root: Path) -> list[str]:
    if not root.exists():
        return []
    return [str(p.relative_to(TEST_OUTPUT_DIR)) for p in sorted(root.rglob("*")) if p.is_file()]


def first_keypoints_from_obj(obj: Any) -> Any | None:
    """Tìm field keypoints đầu tiên trong JSON output theo cách linh hoạt."""
    if isinstance(obj, dict):
        if "keypoints" in obj:
            return obj["keypoints"]

        for value in obj.values():
            found = first_keypoints_from_obj(value)
            if found is not None:
                return found

    elif isinstance(obj, list):
        for item in obj:
            found = first_keypoints_from_obj(item)
            if found is not None:
                return found

    return None


def infer_num_keypoints(keypoints: Any) -> int | None:
    """
    Hỗ trợ các dạng thường gặp:
    - (K, 2) hoặc (K, 3)
    - (M, K, 2) hoặc (M, K, 3)
    """
    if not isinstance(keypoints, list) or len(keypoints) == 0:
        return None

    first = keypoints[0]

    # Dạng (M, K, C)
    if isinstance(first, list) and len(first) > 0 and isinstance(first[0], list):
        return len(first)

    # Dạng (K, C)
    if isinstance(first, list):
        return len(keypoints)

    return None


def infer_num_frames_from_obj(obj: Any) -> int | None:
    if isinstance(obj, dict):
        for key in ("instance_info", "predictions", "frames"):
            value = obj.get(key)
            if isinstance(value, list):
                return len(value)

        for value in obj.values():
            found = infer_num_frames_from_obj(value)
            if found is not None:
                return found

    elif isinstance(obj, list):
        return len(obj)

    return None


def inspect_prediction_files(pred_dir: Path) -> dict[str, Any]:
    prediction_files = sorted(pred_dir.rglob("*.json"))

    result = {
        "num_prediction_files": len(prediction_files),
        "prediction_files": [str(p.relative_to(TEST_OUTPUT_DIR)) for p in prediction_files],
        "num_frames_processed": None,
        "num_keypoints_first_instance": None,
    }

    if not prediction_files:
        return result

    first_json = prediction_files[0]

    try:
        with first_json.open("r", encoding="utf-8") as f:
            obj = json.load(f)

        result["num_frames_processed"] = infer_num_frames_from_obj(obj)

        keypoints = first_keypoints_from_obj(obj)
        result["num_keypoints_first_instance"] = infer_num_keypoints(keypoints)

    except Exception as exc:
        result["prediction_parse_error"] = repr(exc)

    return result


def write_summary_and_report(summary: dict[str, Any]) -> None:
    TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_path = TEST_OUTPUT_DIR / "rtmw_test_summary.json"
    report_path = TEST_OUTPUT_DIR / "rtmw_test_report.md"

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    report_lines = [
        "# RTMW-l / MMPose video smoke test",
        "",
        f"- Input video: `{summary.get('input_video')}`",
        f"- Device: `{summary.get('device')}`",
        f"- Config: `{summary.get('config_path')}`",
        f"- Checkpoint: `{summary.get('checkpoint_path')}`",
        f"- Output dir: `{summary.get('output_dir')}`",
        f"- Success: `{summary.get('success')}`",
        f"- Prediction files: `{summary.get('num_prediction_files')}`",
        f"- Visualization files: `{len(summary.get('visualization_files', []))}`",
        f"- Frames processed: `{summary.get('num_frames_processed')}`",
        f"- First instance keypoints: `{summary.get('num_keypoints_first_instance')}`",
        f"- Expected keypoints: `{summary.get('expected_keypoints')}`",
        "",
    ]

    if summary.get("error"):
        report_lines += [
            "## Error",
            "",
            "```text",
            str(summary.get("error")),
            "```",
            "",
        ]

    with report_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))


def run_test(video_path: Path, device: str | None) -> dict[str, Any]:
    project_root = find_project_root()
    config_path, checkpoint_path = find_rtmw_files(project_root)

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    pred_dir = TEST_OUTPUT_DIR / "predictions"
    vis_dir = TEST_OUTPUT_DIR / "visualizations"

    if TEST_OUTPUT_DIR.exists():
        shutil.rmtree(TEST_OUTPUT_DIR)

    pred_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "input_video": as_posix_str(video_path),
        "device": device,
        "config_path": as_posix_str(config_path),
        "checkpoint_path": as_posix_str(checkpoint_path),
        "output_dir": as_posix_str(TEST_OUTPUT_DIR),
        "expected_keypoints": EXPECTED_KEYPOINTS,
        "success": False,
        "error": None,
    }

    try:
        if not video_path.exists():
            raise FileNotFoundError(f"Video không tồn tại: {video_path}")

        from mmpose.apis import MMPoseInferencer

        init_attempts = [
            {
                "pose2d": as_posix_str(config_path),
                "pose2d_weights": as_posix_str(checkpoint_path),
                "device": device,
                "det_model": "whole_image",
            },
            {
                "pose2d": as_posix_str(config_path),
                "pose2d_weights": as_posix_str(checkpoint_path),
                "device": device,
            },
        ]

        inferencer = None
        init_errors = []

        for kwargs in init_attempts:
            try:
                inferencer = MMPoseInferencer(**kwargs)
                summary["inferencer_init_kwargs"] = kwargs
                break
            except Exception as exc:
                init_errors.append(repr(exc))

        if inferencer is None:
            raise RuntimeError("Không khởi tạo được MMPoseInferencer: " + " | ".join(init_errors))

        result_generator = inferencer(
            as_posix_str(video_path),
            show=False,
            return_vis=False,
            pred_out_dir=as_posix_str(pred_dir),
            vis_out_dir=as_posix_str(vis_dir),
        )

        yielded_results = 0
        for _ in result_generator:
            yielded_results += 1

        summary["yielded_results"] = yielded_results

        pred_info = inspect_prediction_files(pred_dir)
        summary.update(pred_info)

        summary["visualization_files"] = collect_files(vis_dir)

        has_predictions = summary.get("num_prediction_files", 0) > 0
        has_keypoints = summary.get("num_keypoints_first_instance") is not None

        summary["success"] = bool(has_predictions and has_keypoints)

    except Exception:
        summary["success"] = False
        summary["error"] = traceback.format_exc()

    write_summary_and_report(summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path tới video .mp4 cần test.")
    parser.add_argument("--device", default=None, help="cuda:0 hoặc cpu. Mặc định tự chọn.")
    args = parser.parse_args()

    summary = run_test(Path(args.video), args.device)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print()
    print(f"Summary written to: {TEST_OUTPUT_DIR / 'rtmw_test_summary.json'}")
    print(f"Report written to: {TEST_OUTPUT_DIR / 'rtmw_test_report.md'}")

    return 0 if summary.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
