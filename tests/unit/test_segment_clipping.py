"""Unit and regression tests for modular behavior segment clipping and selection."""
from __future__ import annotations

import numpy as np
import pytest

from mouse_behavior.behavior import (
    endpoint_observations,
    endpoints_fully_observed,
    is_process_boundary_closed,
    refine_attack_segments,
    refine_behavior_segments,
    refine_dispersal_segments,
    refine_group_locomotion_segments,
    refine_rearing_segments,
    select_behavior_segments,
    update_segment_bounds,
)
from mouse_behavior.canonical_video_pipeline import infer_window_events
from mouse_behavior.evaluation.exported_dataset_models import HeuristicParameters


class _DummyStore:
    def __init__(self, frame_rows: list[dict]):
        self._rows = frame_rows

    def rows(self, start: int, end: int, *, rebase: bool = False):
        for idx in range(max(0, start), min(len(self._rows), end)):
            row = dict(self._rows[idx])
            if rebase:
                row["frame"] = idx - start
            yield row


def _make_detection(
    track_id: int,
    center: tuple[float, float] = (100.0, 100.0),
    box_size: tuple[float, float] = (20.0, 20.0),
    keypoints: np.ndarray | None = None,
) -> dict:
    cx, cy = center
    w, h = box_size
    box = [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0]
    if keypoints is None:
        # 0: nose, 1: right_ear, 2: left_ear, 3: neck, 4: left_hip, 5: right_hip, 6: tail_base
        kps = [
            [cx, cy - 10.0, 0.9],
            [cx + 3.0, cy - 6.0, 0.9],
            [cx - 3.0, cy - 6.0, 0.9],
            [cx, cy - 4.0, 0.9],
            [cx - 4.0, cy + 4.0, 0.9],
            [cx + 4.0, cy + 4.0, 0.9],
            [cx, cy + 12.0, 0.9],
        ]
    else:
        kps = keypoints.tolist()
    return {
        "track_id": int(track_id),
        "box": box,
        "keypoints": kps,
        "detection_confidence": 0.95,
        "class_id": 0,
    }


def test_endpoint_observations_and_boundary_checks():
    rows = []
    for f in range(20):
        dets = [_make_detection(1, (100.0, 100.0))]
        if 2 <= f < 18:
            dets.append(_make_detection(2, (115.0, 100.0)))
        rows.append({"frame": f, "source_frame": f, "time": f / 30.0, "detections": dets})
    store = _DummyStore(rows)
    info = {"fps": 30.0, "frame_count": 20}

    # Frames [2, 18) have both mice observed at start=2 and end-1=17
    ev_good = {"canonical_behavior": "chase", "mouse_ids": [1, 2], "start_frame": 2, "end_frame": 18}
    obs_good = endpoint_observations(store, ev_good, info)
    assert obs_good["start_observed"] and obs_good["end_observed"]
    assert endpoints_fully_observed(store, ev_good, info)

    # Frames [0, 18) miss mouse 2 at start=0
    ev_bad_start = {"canonical_behavior": "chase", "mouse_ids": [1, 2], "start_frame": 0, "end_frame": 18}
    assert not endpoints_fully_observed(store, ev_bad_start, info)

    # Boundary closure checks
    assert is_process_boundary_closed({"canonical_behavior": "attack"})
    assert not is_process_boundary_closed({
        "canonical_behavior": "approach",
        "process_boundary_status": {
            "requalified_on_source_store": True,
            "clipped_by_inference_window": True,
            "no_body_contact_throughout": True,
        },
    })
    assert is_process_boundary_closed({
        "canonical_behavior": "approach",
        "process_boundary_status": {
            "requalified_on_source_store": True,
            "clipped_by_inference_window": False,
            "no_body_contact_throughout": True,
        },
    })
    assert is_process_boundary_closed({
        "canonical_behavior": "huddle",
        "process_boundary_status": {"output_revision": "huddle-sliding-output-v37"},
    })
    assert not is_process_boundary_closed({
        "canonical_behavior": "huddle",
        "process_boundary_status": {"output_revision": "legacy"},
    })


def test_update_segment_bounds_clips_intervals_and_updates_metadata():
    ev = {
        "event_id": "attack_00000010_00000050_abcdef1234",
        "canonical_behavior": "attack",
        "behavior": "Attack",
        "mouse_ids": [1, 2],
        "start_frame": 10,
        "end_frame": 50,
        "canonical_support_frame_intervals": [[5, 25], [30, 55]],
        "observed_all_participants_frame_intervals": [[10, 50]],
        "observed_all_participants_frame_count": 40,
    }
    cropped = update_segment_bounds(ev, 15, 45, 30.0, suffix="test")
    assert cropped["start_frame"] == 15
    assert cropped["end_frame"] == 45
    assert cropped["frame_count"] == 30
    assert cropped["duration_s"] == pytest.approx(1.0)
    assert cropped["parent_candidate_event_id"] == ev["event_id"]
    assert cropped["canonical_support_frame_intervals"] == [[15, 25], [30, 45]]
    assert cropped["canonical_support_frame_count"] == 25
    assert cropped["observed_all_participants_frame_intervals"] == [[15, 45]]
    assert cropped["observed_all_participants_frame_count"] == 30


def test_refine_rearing_segments_standalone_vs_mounted_and_curled():
    fps = 30.0
    n_frames = 60
    # Upright standing keypoints: nose (0) forward of neck (3), tail (6) behind neck (3) -> cos_ang ~ -1.0
    upright_kps = np.array([
        [100.0, 92.0, 0.9],   # nose
        [102.0, 95.0, 0.9],   # right ear
        [98.0, 95.0, 0.9],    # left ear
        [100.0, 100.0, 0.9],  # neck
        [97.0, 108.0, 0.9],   # left hip
        [103.0, 108.0, 0.9],  # right hip
        [100.0, 115.0, 0.9],  # tail base
    ])
    rows_standalone = [
        {
            "frame": f,
            "source_frame": f,
            "time": f / fps,
            "detections": [
                _make_detection(1, (100.0, 104.0), (18.0, 24.0), upright_kps),
                _make_detection(2, (300.0, 300.0), (20.0, 20.0)),
            ],
        }
        for f in range(n_frames)
    ]
    ev = {
        "event_id": "rearing_00000000_00000060_1111111111",
        "canonical_behavior": "rearing",
        "behavior": "Rearing",
        "mouse_ids": [1],
        "start_frame": 0,
        "end_frame": n_frames,
    }
    res_ok = refine_rearing_segments([ev], _DummyStore(rows_standalone), {"source_video": {"fps": fps}})
    assert len(res_ok) == 1
    assert res_ok[0]["start_frame"] == 0
    assert res_ok[0]["end_frame"] == n_frames

    # Mounted case: mouse 2 box overlaps mouse 1 box on every frame -> rejected
    rows_mounted = [
        {
            "frame": f,
            "source_frame": f,
            "time": f / fps,
            "detections": [
                _make_detection(1, (100.0, 104.0), (18.0, 24.0), upright_kps),
                _make_detection(2, (101.0, 104.0), (18.0, 24.0)),
            ],
        }
        for f in range(n_frames)
    ]
    res_mounted = refine_rearing_segments([ev], _DummyStore(rows_mounted), {"source_video": {"fps": fps}})
    assert len(res_mounted) == 0

    # Curled dorsal grooming case: nose curled back toward tail (cos_ang > 0.15) -> rejected
    curled_kps = upright_kps.copy()
    curled_kps[0, :2] = [101.0, 112.0]  # nose near tail base on same side of neck
    rows_curled = [
        {
            "frame": f,
            "source_frame": f,
            "time": f / fps,
            "detections": [_make_detection(1, (100.0, 104.0), (18.0, 24.0), curled_kps)],
        }
        for f in range(n_frames)
    ]
    res_curled = refine_rearing_segments([ev], _DummyStore(rows_curled), {"source_video": {"fps": fps}})
    assert len(res_curled) == 0


def test_refine_attack_dispersal_and_group_locomotion():
    fps = 30.0
    # 1. Attack: 30 frames (1.0s > 0.7s), fast co-observed motion
    attack_rows = [
        {
            "frame": f,
            "source_frame": f,
            "time": f / fps,
            "detections": [
                _make_detection(1, (50.0 + f * 4.0, 100.0)),
                _make_detection(2, (60.0 + f * 4.0, 100.0)),
            ],
        }
        for f in range(30)
    ]
    attack_ev = {
        "event_id": "attack_00000000_00000030_aa",
        "canonical_behavior": "attack",
        "behavior": "Attack",
        "mouse_ids": [1, 2],
        "start_frame": 0,
        "end_frame": 30,
    }
    res_attack = refine_attack_segments([attack_ev], _DummyStore(attack_rows), {"source_video": {"fps": fps}})
    assert len(res_attack) == 1

    # 2. Dispersal: 180 frames (6.0s > 5.0s), starts compact, expands outward > 35px
    disp_rows = []
    for f in range(180):
        spread = min(80.0, max(0.0, (f - 10) * 1.2))
        disp_rows.append({
            "frame": f,
            "source_frame": f,
            "time": f / fps,
            "detections": [
                _make_detection(1, (200.0 - spread, 200.0)),
                _make_detection(2, (200.0 + spread, 200.0)),
                _make_detection(3, (200.0, 200.0 + spread)),
            ],
        })
    disp_ev = {
        "event_id": "dispersal_00000000_00000180_bb",
        "canonical_behavior": "dispersal",
        "behavior": "Dispersal",
        "mouse_ids": [1, 2, 3],
        "start_frame": 0,
        "end_frame": 180,
    }
    res_disp = refine_dispersal_segments([disp_ev], _DummyStore(disp_rows), {"source_video": {"fps": fps}})
    assert len(res_disp) == 1
    assert (res_disp[0]["end_frame"] - res_disp[0]["start_frame"]) / fps > 5.0

    # 3. Group Locomotion: 120 frames (4.0s > 3.0s), 3 mice moving together cohesively
    gloco_rows = [
        {
            "frame": f,
            "source_frame": f,
            "time": f / fps,
            "detections": [
                _make_detection(1, (50.0 + f * 1.5, 100.0)),
                _make_detection(2, (60.0 + f * 1.5, 105.0)),
                _make_detection(3, (55.0 + f * 1.5, 110.0)),
            ],
        }
        for f in range(120)
    ]
    gloco_ev = {
        "event_id": "group_locomotion_00000000_00000120_cc",
        "canonical_behavior": "group_locomotion",
        "behavior": "GroupLocomotion",
        "mouse_ids": [1, 2, 3],
        "start_frame": 0,
        "end_frame": 120,
        "canonical_support_frame_intervals": [[0, 120]],
        "canonical_support_frame_count": 120,
    }
    res_gloco, audit_gloco = refine_behavior_segments(
        [gloco_ev],
        _DummyStore(gloco_rows),
        {"source_video": {"fps": fps}},
        enable_huddle_sliding=False,
        enable_approach_requalification=False,
        enable_fine_cropping=True,
    )
    assert len(res_gloco) == 1
    assert audit_gloco["output_count"] == 1


def test_select_behavior_segments_has_no_default_count_cap():
    """Verify that >30 qualifying segments per behavior are all retained by default (no 30-clip cap)."""
    fps = 30.0
    events = []
    # Create 45 distinct non-overlapping nose_head_contact segments
    for i in range(45):
        s = i * 30
        e = s + 15  # 0.5s >= 0.25s minimum
        events.append({
            "event_id": f"nose_head_contact_{s:08d}_{e:08d}_{i:04d}",
            "canonical_behavior": "nose_head_contact",
            "behavior": "NoseHeadContact",
            "mouse_ids": [1, 2],
            "start_frame": s,
            "end_frame": e,
            "observed_all_participants_frame_count": 15,
            "canonical_support_frame_count": 15,
        })
    # Also add an overlapping duplicate on the same pair [1, 2] with weaker support
    events.append({
        "event_id": "nose_head_contact_overlap_weak",
        "canonical_behavior": "nose_head_contact",
        "behavior": "NoseHeadContact",
        "mouse_ids": [1, 2],
        "start_frame": 5,
        "end_frame": 20,
        "observed_all_participants_frame_count": 8,
        "canonical_support_frame_count": 8,
    })

    selected, summary = select_behavior_segments(
        events,
        binding={"source_video": {"fps": fps}},
        require_endpoint_visibility=False,
    )
    assert summary["max_per_behavior"] is None
    assert len(selected) == 45
    assert summary["counts_by_behavior"]["nose_head_contact"] == 45
    assert summary["rejected_counts"]["duplicate_or_overlapping"] == 1


def test_infer_window_events_calibrated_posture_mode():
    fps = 30.0
    rows = [
        {
            "frame": f,
            "source_frame": f,
            "time": f / fps,
            "detections": [_make_detection(1, (100.0, 100.0))],
        }
        for f in range(90)
    ]
    scale = {"cm_per_pixel": 0.1, "reference_body_cm": 7.918}
    events, audit = infer_window_events(rows, fps, scale, HeuristicParameters(), posture_mode="calibrated")
    assert audit["track_ids"] == [1]
    assert any(e["canonical_behavior"] == "stationary" for e in events)


def test_segment_clipping_edge_cases_and_error_paths():
    # Empty inputs
    selected_empty, summary_empty = select_behavior_segments([])
    assert selected_empty == []
    assert summary_empty["total_input_events"] == 0
    assert summary_empty["total_selected_events"] == 0

    store_empty = _DummyStore([])
    assert refine_rearing_segments([], store_empty) == []
    assert refine_attack_segments([], store_empty) == []
    assert refine_dispersal_segments([], store_empty) == []
    assert refine_group_locomotion_segments([], store_empty) == []

    # Invalid bounds in update_segment_bounds
    with pytest.raises(ValueError, match="invalid segment bounds"):
        update_segment_bounds({"start_frame": 10, "end_frame": 20}, 20, 20, 30.0)
    with pytest.raises(ValueError, match="invalid segment bounds"):
        update_segment_bounds({"start_frame": 10, "end_frame": 20}, 25, 20, 30.0)

    # Invalid max_per_behavior and time_bucket_seconds
    with pytest.raises(ValueError, match="max_per_behavior"):
        select_behavior_segments([], max_per_behavior=0)
    with pytest.raises(ValueError, match="max_per_behavior"):
        select_behavior_segments([], max_per_behavior=True)
    with pytest.raises(ValueError, match="time_bucket_seconds"):
        select_behavior_segments([], time_bucket_seconds=0.0)
    with pytest.raises(ValueError, match="time_bucket_seconds"):
        select_behavior_segments([], time_bucket_seconds=float("nan"))

    # Simultaneous segments on different mouse_ids survive strict_non_overlapping_same_ids=True
    ev_pair_a = {
        "event_id": "chase_pair_a",
        "canonical_behavior": "chase",
        "behavior": "Chase",
        "mouse_ids": [1, 2],
        "start_frame": 0,
        "end_frame": 40,
    }
    ev_pair_b = {
        "event_id": "chase_pair_b",
        "canonical_behavior": "chase",
        "behavior": "Chase",
        "mouse_ids": [3, 4],
        "start_frame": 0,
        "end_frame": 40,
    }
    selected_pairs, _ = select_behavior_segments(
        [ev_pair_a, ev_pair_b],
        binding={"source_video": {"fps": 30.0}},
        require_endpoint_visibility=False,
    )
    assert len(selected_pairs) == 2

