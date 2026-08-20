from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

import adaptive_arena_boundary as arena


def _record(frame: int, x: float, y: float) -> dict:
    points = np.array(
        [
            [x + 18.0, y],
            [x + 9.0, y - 8.0],
            [x + 9.0, y + 8.0],
            [x, y],
            [x - 10.0, y - 8.0],
            [x - 10.0, y + 8.0],
            [x - 24.0, y],
        ],
        dtype=float,
    )
    return {
        "frame": frame,
        "pose_detections": [
            {
                "keypoints_px": points,
                "keypoint_conf": np.full(7, 0.95, dtype=float),
                "bbox_xyxy": np.array([x - 30.0, y - 20.0, x + 25.0, y + 20.0]),
                "box_conf": 0.95,
                "pose_quality": 0.95,
            }
        ],
        "detector_boxes": [],
    }


def _large_sweeping_path() -> list[dict]:
    points: list[tuple[float, float]] = []
    for y in range(70, 311, 20):
        xs = range(90, 711, 20) if ((y - 70) // 20) % 2 == 0 else range(710, 89, -20)
        points.extend((float(x), float(y)) for x in xs)
    return [_record(frame, x, y) for frame, (x, y) in enumerate(points)]


def test_learns_and_reuses_boundary_for_the_same_video(tmp_path: Path) -> None:
    video = tmp_path / "video_a.avi"
    writer = cv2.VideoWriter(
        str(video),
        cv2.VideoWriter_fourcc(*"MJPG"),
        1.0,
        (800, 400),
    )
    if not writer.isOpened():
        pytest.skip("OpenCV MJPG writer is unavailable in this environment")
    writer.write(np.full((400, 800, 3), 80, dtype=np.uint8))
    writer.release()
    result, heatmap = arena.learn_from_yolo_records(
        _large_sweeping_path(),
        width=800,
        height=400,
        config={
            "min_motion_samples": 30,
            "long_track_min_frames": 3,
            "max_track_gap_frames": 1,
            "min_motion_px_per_frame": 1.0,
            "max_motion_px_per_frame": 40.0,
            "min_boundary_area_ratio": 0.10,
            "max_boundary_area_ratio": 0.95,
            "heatmap_cell_px": 20,
        },
        source_video=video,
    )
    assert result.source == "learned_motion_heatmap"
    assert result.accepted is True
    assert result.motion_sample_count >= 30
    assert len(result.polygon) == 4
    learned_polygon = np.asarray(result.polygon, dtype=float)
    assert len(np.unique(learned_polygon[:, 0])) == 2
    assert len(np.unique(learned_polygon[:, 1])) == 2
    assert result.expansion_ratio == pytest.approx(0.97)
    assert float(np.max(heatmap)) > 0.0

    json_path = tmp_path / "boundary.json"
    png_path = tmp_path / "boundary.png"
    comparison_path = tmp_path / "comparison.png"
    arena.save_boundary_artifacts(
        result,
        heatmap,
        json_path,
        png_path,
        comparison_path,
    )
    loaded = arena.load_boundary_json(
        json_path,
        width=800,
        height=400,
        source_video=video,
    )
    assert loaded.polygon == result.polygon
    assert json_path.exists()
    assert png_path.exists()
    assert comparison_path.exists()
    comparison = cv2.imread(str(comparison_path))
    assert comparison is not None
    assert comparison.shape[:2] == (400, 800)

    with pytest.raises(ValueError, match="其他视频|指纹"):
        arena.load_boundary_json(
            json_path,
            width=800,
            height=400,
            source_video=tmp_path / "video_b.mp4",
        )


def test_falls_back_when_motion_evidence_is_insufficient() -> None:
    result, heatmap = arena.learn_from_yolo_records(
        [_record(0, 320.0, 180.0)],
        width=640,
        height=360,
        config={"min_motion_samples": 30, "long_track_min_frames": 3},
    )
    assert result.source == "frame_fallback"
    assert result.accepted is False
    assert result.rejection_reason.startswith("insufficient_motion_samples")
    assert np.all(heatmap == 0.0)
