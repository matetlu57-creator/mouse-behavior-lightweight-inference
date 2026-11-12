"""Unit tests for modular 6-behavior benchmark evaluation and overlay rendering."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from mouse_behavior.behavior import InMemoryTrackStore, refine_chasing_segments
from mouse_behavior.evaluation.clip_benchmark import (
    BEHAVIOR_ORDER,
    combine_evaluation_reports,
    compute_temporal_iou,
    evaluate_six_behavior_clip,
)
from mouse_behavior.visualization.clip_overlay_renderer import (
    normalize_roles_map,
    render_annotated_original_video,
    render_cropped_comparison_video,
    video_frame_count,
)


def _write_synthetic_mp4(path: Path, frame_count: int = 35, fps: float = 30.0) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (160, 120))
    assert writer.isOpened()
    for f in range(frame_count):
        writer.write(np.full((120, 160, 3), (f * 3) % 255, dtype=np.uint8))
    writer.release()


def test_behavior_facade_exports_in_memory_store_and_chasing_refiner() -> None:
    store = InMemoryTrackStore([{"frame": i, "detections": []} for i in range(5)])
    assert [r["frame"] for r in store.rows(1, 4)] == [1, 2, 3]
    assert [r["frame"] for r in store.rows(2, 4, rebase=True)] == [0, 1]
    assert callable(refine_chasing_segments)


def test_compute_temporal_iou_edge_cases() -> None:
    assert compute_temporal_iou(None, (0, 10)) == 0.0
    assert compute_temporal_iou((5, 5), (0, 10)) == 0.0
    assert compute_temporal_iou((10, 5), (0, 10)) == 0.0
    assert compute_temporal_iou((0, 10), (10, 20)) == 0.0
    assert compute_temporal_iou((0, 20), (10, 30)) == pytest.approx(10 / 30)
    assert compute_temporal_iou((0, 20), (0, 20)) == 1.0


def test_normalize_roles_map_handles_lists_scalars_and_invalid() -> None:
    assert normalize_roles_map({"Attacker": [1], "Attacked": 2, "Skip": True, "Bad": "x"}) == {
        "Attacker": [1],
        "Attacked": [2],
    }
    assert normalize_roles_map(None) == {}


def test_combine_evaluation_reports_merges_all_six_behaviors(tmp_path: Path) -> None:
    rep_a = tmp_path / "a.json"
    rep_b = tmp_path / "b.json"
    rep_missing = tmp_path / "does_not_exist.json"

    rep_a.write_text(
        json.dumps({
            "per_clip_results": [
                {"clip_id": "c_att", "ground_truth_behavior": "Attack", "is_correct": True, "temporal_iou": 1.0},
                {"clip_id": "c_chs", "ground_truth_behavior": "Chasing", "is_correct": False, "temporal_iou": 0.0},
                {"clip_id": "c_iso", "ground_truth_behavior": "Isolation", "is_correct": True, "temporal_iou": 0.96},
            ]
        }),
        encoding="utf-8",
    )
    rep_b.write_text(
        json.dumps({
            "per_clip_results": [
                {"clip_id": "c_glo", "ground_truth_behavior": "Group locomotion", "is_correct": True, "temporal_iou": 1.0},
                {"clip_id": "c_sta", "ground_truth_behavior": "Static", "is_correct": True, "temporal_iou": 0.98},
                {"clip_id": "c_loc", "ground_truth_behavior": "Locomotion", "is_correct": True, "temporal_iou": 0.95},
            ]
        }),
        encoding="utf-8",
    )

    out = tmp_path / "combined.json"
    summary = combine_evaluation_reports([rep_a, rep_b, rep_missing], out)
    assert out.is_file()
    assert summary["total_clips"] == 6
    for beh in BEHAVIOR_ORDER:
        assert summary[f"{beh.lower().replace(' ', '_')}_clips"] == 1
    assert summary["overall_accuracy"] == pytest.approx(5 / 6, abs=1e-4)
    assert summary["iou_ge_95_rate"] == 1.0


def test_evaluate_six_behavior_clip_static_and_error_resilience(tmp_path: Path) -> None:
    missing_dir = tmp_path / "missing_clip"
    missing_dir.mkdir()
    err_res = evaluate_six_behavior_clip(missing_dir)
    assert err_res["status"] == "error"
    assert err_res["is_correct"] is False

    clip_dir = tmp_path / "static_clip"
    clip_dir.mkdir()
    fps = 30.0
    n_frames = 45
    _write_synthetic_mp4(clip_dir / "clip.mp4", frame_count=n_frames, fps=fps)
    tracks = [
        {
            "frame": f,
            "detections": [
                {
                    "track_id": 7,
                    "box": [40.0, 40.0, 80.0, 80.0],
                    "keypoints": [[60.0 + i * 0.1, 60.0, 0.9] for i in range(7)],
                }
            ],
        }
        for f in range(n_frames)
    ]
    (clip_dir / "tracks.json").write_text(json.dumps(tracks), encoding="utf-8")
    (clip_dir / "metadata.json").write_text(
        json.dumps({"clip": {"fps": fps, "frame_count": n_frames}}),
        encoding="utf-8",
    )
    (clip_dir / "annotation.json").write_text(
        json.dumps({
            "behavior": "Static",
            "mouse_ids": [7],
            "frame_range": {"start": 0, "end": n_frames - 1},
        }),
        encoding="utf-8",
    )

    crop_dir = tmp_path / "crops"
    orig_dir = tmp_path / "orig"
    res = evaluate_six_behavior_clip(
        clip_dir,
        crop_output_dir=crop_dir,
        orig_annotated_dir=orig_dir,
        render_videos=True,
    )
    assert res["status"] == "ok"
    assert res["predicted_behavior"] == "Static"
    assert res["is_correct"] is True
    assert res["temporal_iou"] >= 0.90
    assert Path(res["cropped_video_path"]).is_file()
    assert Path(res["annotated_orig_path"]).is_file()
    assert video_frame_count(Path(res["cropped_video_path"])) == (
        res["predicted_range_half_open"]["end"] - res["predicted_range_half_open"]["start"]
    )


def test_clip_overlay_renderer_outputs_valid_h264(tmp_path: Path) -> None:
    src = tmp_path / "src.mp4"
    _write_synthetic_mp4(src, frame_count=20, fps=10.0)
    tracks = [
        {
            "frame": f,
            "detections": [
                {"track_id": 1, "box": [20.0, 20.0, 50.0, 50.0]},
                {"track_id": 2, "box": [70.0, 30.0, 100.0, 60.0]},
            ],
        }
        for f in range(20)
    ]
    crop_out = tmp_path / "crop.mp4"
    orig_out = tmp_path / "orig.mp4"

    assert render_cropped_comparison_video(
        src,
        crop_out,
        5,
        15,
        10.0,
        tracks=tracks,
        target_mouse_ids=[1, 2],
        semantic_roles={"Attacker": [1], "Attacked": [2]},
        pred_behavior="Attack",
        gt_behavior="Attack",
        iou=1.0,
    )
    assert crop_out.is_file() and video_frame_count(crop_out) == 10

    assert render_annotated_original_video(
        src,
        orig_out,
        10.0,
        tracks=tracks,
        target_mouse_ids=[1, 2],
        semantic_roles={"Attacker": [1], "Attacked": [2]},
        behavior="Attack",
        show_banner=True,
    )
    assert orig_out.is_file() and video_frame_count(orig_out) == 20
