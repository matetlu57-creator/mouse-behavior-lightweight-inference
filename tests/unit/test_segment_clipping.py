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
    refine_isolation_segments,
    refine_locomotion_segments,
    refine_rearing_segments,
    refine_static_segments,
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


def test_exact_duration_boundaries_dispersal_and_group_locomotion():
    fps = 30.0
    # Qualified >5.0s dispersal (155 frames > 150 at 30 fps)
    disp_rows = []
    for f in range(155):
        spread = min(60.0, max(0.0, (f - 10) * 1.5))
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
        "event_id": "dispersal_qual_155",
        "canonical_behavior": "dispersal",
        "behavior": "Dispersal",
        "mouse_ids": [1, 2, 3],
        "start_frame": 0,
        "end_frame": 155,
    }
    res_disp = refine_dispersal_segments([disp_ev], _DummyStore(disp_rows), {"source_video": {"fps": fps}})
    assert len(res_disp) == 1
    assert (res_disp[0]["end_frame"] - res_disp[0]["start_frame"]) / fps > 5.0

    # Group locomotion with explicit 1.0s duration requirement override
    gloco_rows = [
        {
            "frame": f,
            "source_frame": f,
            "time": f / fps,
            "detections": [
                _make_detection(1, (50.0 + f * 2.0, 100.0)),
                _make_detection(2, (60.0 + f * 2.0, 105.0)),
                _make_detection(3, (55.0 + f * 2.0, 110.0)),
            ],
        }
        for f in range(35)
    ]
    gloco_ev = {
        "event_id": "group_loco_35",
        "canonical_behavior": "group_locomotion",
        "behavior": "GroupLocomotion",
        "mouse_ids": [1, 2, 3],
        "start_frame": 0,
        "end_frame": 35,
        "canonical_support_frame_intervals": [[0, 35]],
        "canonical_support_frame_count": 35,
    }
    binding_1s = {
        "source_video": {"fps": fps},
        "effective_parameters": {"group_locomotion_min_duration_s": 1.0},
    }
    res_gloco = refine_group_locomotion_segments(
        [gloco_ev],
        _DummyStore(gloco_rows),
        binding_1s,
        min_duration_s=1.0,
        min_support_frames=15,
    )
    assert len(res_gloco) == 1
    assert (res_gloco[0]["end_frame"] - res_gloco[0]["start_frame"]) / fps > 1.0


def test_symmetrical_mouse_ids_deduplication():
    fps = 30.0
    ev1 = {
        "event_id": "nhc_1",
        "canonical_behavior": "nose_head_contact",
        "behavior": "NoseHeadContact",
        "mouse_ids": [1, 2],
        "start_frame": 0,
        "end_frame": 30,
        "observed_all_participants_frame_count": 30,
        "canonical_support_frame_count": 30,
    }
    # ev2 has reversed mouse_ids [2, 1] and overlaps
    ev2 = {
        "event_id": "nhc_2",
        "canonical_behavior": "nose_head_contact",
        "behavior": "NoseHeadContact",
        "mouse_ids": [2, 1],
        "start_frame": 10,
        "end_frame": 40,
        "observed_all_participants_frame_count": 25,
        "canonical_support_frame_count": 25,
    }
    selected, summary = select_behavior_segments(
        [ev1, ev2],
        binding={"source_video": {"fps": fps}},
        require_endpoint_visibility=False,
    )
    assert len(selected) == 1
    assert summary["rejected_counts"]["duplicate_or_overlapping"] == 1


def test_update_segment_bounds_does_not_synthesize_canonical_support():
    fps = 30.0
    event_without_support = {
        "event_id": "huddle_test",
        "canonical_behavior": "huddle",
        "behavior": "Huddling",
        "mouse_ids": [1, 2, 3],
        "start_frame": 0,
        "end_frame": 150,
    }
    updated = update_segment_bounds(event_without_support, 10, 140, fps)
    assert "canonical_support_frame_intervals" not in updated
    assert "canonical_support_frame_count" not in updated


def test_refine_isolation_segments_clean():
    fps = 30.0
    px_per_cm = 10.0
    rows = []
    # 360 frames = 12.0s (> 10.0s isolation duration policy gate)
    for f in range(360):
        dets = [
            _make_detection(1, (500.0, 500.0)),
            _make_detection(2, (100.0, 100.0)),
            _make_detection(3, (120.0, 100.0)),
        ]
        rows.append({"frame": f, "source_frame": f, "time": f / fps, "detections": dets})

    store = _DummyStore(rows)
    ev = {
        "event_id": "iso_test_1",
        "canonical_behavior": "isolation",
        "behavior": "Isolation",
        "mouse_ids": [1],
        "target_mouse_id": 1,
        "start_frame": 0,
        "end_frame": 360,
    }
    binding = {
        "source_video": {"fps": fps, "scale_px_per_cm": px_per_cm},
    }

    res = refine_isolation_segments([ev], store, binding)
    assert len(res) == 1
    assert res[0]["target_mouse_id"] == 1
    assert res[0]["start_frame"] == 0
    assert res[0]["end_frame"] == 360
    assert res[0]["canonical_support_frame_count"] == 360
    assert res[0]["canonical_support_frame_intervals"] == [[0, 360]]


def test_refine_isolation_segments_rejects_non_isolated():
    fps = 30.0
    px_per_cm = 10.0
    rows = []
    for f in range(360):
        # Mouse 1 is at (105, 100), Mouse 2 is at (100, 100) (0.5 cm away -> not isolated)
        dets = [
            _make_detection(1, (105.0, 100.0)),
            _make_detection(2, (100.0, 100.0)),
        ]
        rows.append({"frame": f, "source_frame": f, "time": f / fps, "detections": dets})

    store = _DummyStore(rows)
    ev = {
        "event_id": "iso_test_fail",
        "canonical_behavior": "isolation",
        "behavior": "Isolation",
        "mouse_ids": [1],
        "target_mouse_id": 1,
        "start_frame": 0,
        "end_frame": 360,
    }
    binding = {
        "source_video": {"fps": fps, "scale_px_per_cm": px_per_cm},
    }

    res = refine_isolation_segments([ev], store, binding)
    assert len(res) == 0


def test_refine_isolation_segments_tolerates_brief_glitch():
    fps = 30.0
    px_per_cm = 10.0
    rows = []
    for f in range(360):
        # 350 frames isolated, 10 frames glitch
        pos1 = (105.0, 100.0) if 100 <= f < 110 else (500.0, 500.0)
        dets = [
            _make_detection(1, pos1),
            _make_detection(2, (100.0, 100.0)),
        ]
        rows.append({"frame": f, "source_frame": f, "time": f / fps, "detections": dets})

    store = _DummyStore(rows)
    ev = {
        "event_id": "iso_glitch_test",
        "canonical_behavior": "isolation",
        "behavior": "Isolation",
        "mouse_ids": [1],
        "target_mouse_id": 1,
        "start_frame": 0,
        "end_frame": 360,
    }
    binding = {
        "source_video": {"fps": fps, "scale_px_per_cm": px_per_cm},
    }

    res = refine_isolation_segments([ev], store, binding, min_isolated_fraction=0.85)
    assert len(res) == 1
    assert res[0]["canonical_support_frame_count"] >= 350


def test_refine_behavior_segments_isolation_dispatch():
    fps = 30.0
    px_per_cm = 10.0
    rows = []
    for f in range(360):
        dets = [
            _make_detection(1, (400.0, 400.0)),
            _make_detection(2, (100.0, 100.0)),
        ]
        rows.append({"frame": f, "source_frame": f, "time": f / fps, "detections": dets})

    store = _DummyStore(rows)
    ev = {
        "event_id": "iso_dispatch",
        "canonical_behavior": "isolation",
        "behavior": "Isolation",
        "mouse_ids": [1],
        "target_mouse_id": 1,
        "start_frame": 0,
        "end_frame": 360,
    }
    binding = {"source_video": {"fps": fps, "scale_px_per_cm": px_per_cm}}

    refined, audit = refine_behavior_segments([ev], store, binding, enable_fine_cropping=True)
    assert len(refined) == 1
    assert refined[0]["event_id"] == "iso_dispatch"
    assert audit["output_count"] == 1


def test_refine_isolation_segments_rejects_multi_mouse_without_target_id():
    fps = 30.0
    px_per_cm = 10.0
    rows = []
    for f in range(360):
        dets = [
            _make_detection(1, (500.0, 500.0)),
            _make_detection(2, (100.0, 100.0)),
            _make_detection(3, (120.0, 100.0)),
        ]
        rows.append({"frame": f, "source_frame": f, "time": f / fps, "detections": dets})

    store = _DummyStore(rows)
    ev = {
        "event_id": "iso_multi_fail",
        "canonical_behavior": "isolation",
        "behavior": "Isolation",
        "mouse_ids": [1, 2, 3],  # Multi-mouse without explicit target_mouse_id
        "start_frame": 0,
        "end_frame": 360,
    }
    binding = {"source_video": {"fps": fps, "scale_px_per_cm": px_per_cm}}

    res = refine_isolation_segments([ev], store, binding)
    assert len(res) == 0


def test_refine_isolation_segments_snaps_small_head_dropout():
    fps = 30.0
    px_per_cm = 10.0
    rows = []
    # Drop frames 0..4 (5 frames = 0.16s <= 2.0s)
    for f in range(360):
        dets = []
        if f >= 5:
            dets.append(_make_detection(1, (500.0, 500.0)))
        dets.append(_make_detection(2, (100.0, 100.0)))
        rows.append({"frame": f, "source_frame": f, "time": f / fps, "detections": dets})

    store = _DummyStore(rows)
    ev = {
        "event_id": "iso_snap_head",
        "canonical_behavior": "isolation",
        "behavior": "Isolation",
        "mouse_ids": [1],
        "target_mouse_id": 1,
        "start_frame": 0,
        "end_frame": 360,
    }
    binding = {"source_video": {"fps": fps, "scale_px_per_cm": px_per_cm}}

    res = refine_isolation_segments([ev], store, binding)
    assert len(res) == 1
    assert res[0]["start_frame"] == 0  # Snapped back to candidate start
    assert res[0]["end_frame"] == 360


def test_refine_group_locomotion_segments_snaps_small_head_dropout():
    fps = 30.0
    px_per_cm = 10.0
    rows = []
    # 3 mice moving together along x axis at 20 px/frame (> 12 px/s)
    # One mouse has a 3-frame dropout at the start (frames 0..2)
    for f in range(120):
        x = 100.0 + f * 2.0
        dets = []
        if f >= 3:
            dets.append(_make_detection(1, (x, 100.0)))
        dets.append(_make_detection(2, (x + 20.0, 100.0)))
        dets.append(_make_detection(3, (x + 40.0, 100.0)))
        rows.append({"frame": f, "source_frame": f, "time": f / fps, "detections": dets})

    store = _DummyStore(rows)
    ev = {
        "event_id": "gloco_snap_test",
        "canonical_behavior": "group_locomotion",
        "behavior": "Group locomotion",
        "mouse_ids": [1, 2, 3],
        "start_frame": 0,
        "end_frame": 120,
    }
    binding = {"source_video": {"fps": fps, "scale_px_per_cm": px_per_cm}}

    res = refine_group_locomotion_segments([ev], store, binding)
    assert len(res) == 1
    assert res[0]["start_frame"] == 0  # Symmetrically snapped to start
    assert res[0]["end_frame"] == 120


def test_refine_static_segments_clean():
    fps = 30.0
    px_per_cm = 10.0
    rows = []
    # 60 frames (2.0s >= 1.0s) of mouse staying virtually still at (100, 100)
    for f in range(60):
        # subtle sub-pixel jitter
        dx = 0.2 * np.sin(f)
        dy = 0.2 * np.cos(f)
        dets = [_make_detection(1, (100.0 + dx, 100.0 + dy))]
        rows.append({"frame": f, "source_frame": f, "time": f / fps, "detections": dets})

    store = _DummyStore(rows)
    ev = {
        "event_id": "static_clean",
        "canonical_behavior": "static",
        "behavior": "Static",
        "mouse_ids": [1],
        "start_frame": 0,
        "end_frame": 60,
    }
    binding = {"source_video": {"fps": fps, "scale_px_per_cm": px_per_cm}}

    res = refine_static_segments([ev], store, binding)
    assert len(res) == 1
    assert res[0]["start_frame"] == 0
    assert res[0]["end_frame"] == 60
    assert res[0]["canonical_support_frame_count"] == 60


def test_refine_static_segments_rejects_large_movement():
    fps = 30.0
    px_per_cm = 10.0
    rows = []
    # Mouse moves 10 cm away (> max_radius_cm = 4.0 cm)
    for f in range(60):
        dets = [_make_detection(1, (100.0 + f * 1.5, 100.0))]
        rows.append({"frame": f, "source_frame": f, "time": f / fps, "detections": dets})

    store = _DummyStore(rows)
    ev = {
        "event_id": "static_large_movement",
        "canonical_behavior": "static",
        "behavior": "Static",
        "mouse_ids": [1],
        "start_frame": 0,
        "end_frame": 60,
    }
    binding = {"source_video": {"fps": fps, "scale_px_per_cm": px_per_cm}}

    res = refine_static_segments([ev], store, binding)
    assert len(res) == 0


def test_refine_static_segments_rejects_excessive_speed():
    fps = 30.0
    px_per_cm = 10.0
    rows = []
    # Rapid back and forth oscillation within 3.5cm radius, but speed is high (> 6.0 cm/s)
    # 30 px = 3 cm oscillation at 30 fps
    for f in range(60):
        x = 100.0 + 30.0 if (f % 2 == 0) else 100.0 - 30.0
        dets = [_make_detection(1, (x, 100.0))]
        rows.append({"frame": f, "source_frame": f, "time": f / fps, "detections": dets})

    store = _DummyStore(rows)
    ev = {
        "event_id": "static_high_speed",
        "canonical_behavior": "static",
        "behavior": "Static",
        "mouse_ids": [1],
        "start_frame": 0,
        "end_frame": 60,
    }
    binding = {"source_video": {"fps": fps, "scale_px_per_cm": px_per_cm}}

    res = refine_static_segments([ev], store, binding)
    assert len(res) == 0


def test_refine_static_segments_snaps_dropout():
    fps = 30.0
    px_per_cm = 10.0
    rows = []
    # Missing first 4 frames (0.13s <= 2.0s)
    for f in range(60):
        dets = []
        if f >= 4:
            dets.append(_make_detection(1, (100.0, 100.0)))
        rows.append({"frame": f, "source_frame": f, "time": f / fps, "detections": dets})

    store = _DummyStore(rows)
    ev = {
        "event_id": "static_snap",
        "canonical_behavior": "static",
        "behavior": "Static",
        "mouse_ids": [1],
        "start_frame": 0,
        "end_frame": 60,
    }
    binding = {"source_video": {"fps": fps, "scale_px_per_cm": px_per_cm}}

    res = refine_static_segments([ev], store, binding)
    assert len(res) == 1
    assert res[0]["start_frame"] == 0  # Snapped back to 0
    assert res[0]["end_frame"] == 60


def test_refine_locomotion_segments_clean():
    fps = 30.0
    px_per_cm = 10.0
    rows = []
    # Mouse moves 10 cm in 30 frames (1.0s, speed = 10 cm/s >= 3.0 cm/s, displacement = 10 cm >= 3.0 cm)
    for f in range(30):
        dets = [_make_detection(1, (100.0 + f * 3.33, 100.0))]
        rows.append({"frame": f, "source_frame": f, "time": f / fps, "detections": dets})

    store = _DummyStore(rows)
    ev = {
        "event_id": "loco_clean",
        "canonical_behavior": "locomotion",
        "behavior": "Locomotion",
        "mouse_ids": [1],
        "start_frame": 0,
        "end_frame": 30,
    }
    binding = {"source_video": {"fps": fps, "scale_px_per_cm": px_per_cm}}

    res = refine_locomotion_segments([ev], store, binding)
    assert len(res) == 1
    assert res[0]["start_frame"] == 0
    assert res[0]["end_frame"] == 30
    assert res[0]["canonical_support_frame_count"] == 30


def test_refine_locomotion_segments_rejects_static():
    fps = 30.0
    px_per_cm = 10.0
    rows = []
    # Mouse is static (displacement = 0.5 cm < 3.0 cm, speed < 3.0 cm/s)
    for f in range(30):
        dets = [_make_detection(1, (100.0 + 0.1 * f, 100.0))]
        rows.append({"frame": f, "source_frame": f, "time": f / fps, "detections": dets})

    store = _DummyStore(rows)
    ev = {
        "event_id": "loco_reject_static",
        "canonical_behavior": "locomotion",
        "behavior": "Locomotion",
        "mouse_ids": [1],
        "start_frame": 0,
        "end_frame": 30,
    }
    binding = {"source_video": {"fps": fps, "scale_px_per_cm": px_per_cm}}

    res = refine_locomotion_segments([ev], store, binding)
    assert len(res) == 0


def test_refine_locomotion_segments_snaps_dropout():
    fps = 30.0
    px_per_cm = 10.0
    rows = []
    # Missing first 3 frames (0.1s <= 2.0s)
    for f in range(30):
        dets = []
        if f >= 3:
            dets.append(_make_detection(1, (100.0 + f * 3.33, 100.0)))
        rows.append({"frame": f, "source_frame": f, "time": f / fps, "detections": dets})

    store = _DummyStore(rows)
    ev = {
        "event_id": "loco_snap",
        "canonical_behavior": "locomotion",
        "behavior": "Locomotion",
        "mouse_ids": [1],
        "start_frame": 0,
        "end_frame": 30,
    }
    binding = {"source_video": {"fps": fps, "scale_px_per_cm": px_per_cm}}

    res = refine_locomotion_segments([ev], store, binding)
    assert len(res) == 1
    assert res[0]["start_frame"] == 0  # Snapped back to 0
    assert res[0]["end_frame"] == 30


def test_refine_behavior_segments_static_and_locomotion_dispatch():
    fps = 30.0
    px_per_cm = 10.0
    # Two mice: mouse 1 stationary at (100, 100), mouse 2 moving from 200 to 300
    rows = []
    for f in range(40):
        dets = [
            _make_detection(1, (100.0, 100.0)),
            _make_detection(2, (200.0 + f * 2.5, 200.0)),
        ]
        rows.append({"frame": f, "source_frame": f, "time": f / fps, "detections": dets})

    store = _DummyStore(rows)
    ev_static = {
        "event_id": "dispatch_static",
        "canonical_behavior": "static",
        "behavior": "Static",
        "mouse_ids": [1],
        "start_frame": 0,
        "end_frame": 40,
    }
    ev_loco = {
        "event_id": "dispatch_loco",
        "canonical_behavior": "locomotion",
        "behavior": "Locomotion",
        "mouse_ids": [2],
        "start_frame": 0,
        "end_frame": 40,
    }
    binding = {"source_video": {"fps": fps, "scale_px_per_cm": px_per_cm}}

    refined, audit = refine_behavior_segments(
        [ev_static, ev_loco],
        store,
        binding,
        enable_fine_cropping=True,
    )
    assert len(refined) == 2
    assert audit["output_count"] == 2
    ids = {r["event_id"] for r in refined}
    assert ids == {"dispatch_static", "dispatch_loco"}


def test_refine_static_segments_snaps_trailing_dropout():
    fps = 30.0
    px_per_cm = 10.0
    rows = []
    # Missing last 4 frames (0.13s <= 2.0s)
    for f in range(60):
        dets = []
        if f < 56:
            dets.append(_make_detection(1, (100.0, 100.0)))
        rows.append({"frame": f, "source_frame": f, "time": f / fps, "detections": dets})

    store = _DummyStore(rows)
    ev = {
        "event_id": "static_snap_tail",
        "canonical_behavior": "static",
        "behavior": "Static",
        "mouse_ids": [1],
        "start_frame": 0,
        "end_frame": 60,
    }
    binding = {"source_video": {"fps": fps, "scale_px_per_cm": px_per_cm}}

    res = refine_static_segments([ev], store, binding)
    assert len(res) == 1
    assert res[0]["start_frame"] == 0
    assert res[0]["end_frame"] == 60


def test_refine_locomotion_segments_slow_walking_not_swallowed():
    fps = 30.0
    px_per_cm = 10.0
    rows = []
    # Continuous slow walking at ~5.5 cm/s for 60 frames (total ~11 cm, 1s radius ~2.75 cm)
    for f in range(60):
        dets = [_make_detection(1, (100.0 + f * (5.5 * 10.0 / fps), 100.0))]
        rows.append({"frame": f, "source_frame": f, "time": f / fps, "detections": dets})

    store = _DummyStore(rows)
    ev = {
        "event_id": "loco_slow_walk",
        "canonical_behavior": "locomotion",
        "behavior": "Locomotion",
        "mouse_ids": [1],
        "start_frame": 0,
        "end_frame": 60,
    }
    binding = {"source_video": {"fps": fps, "scale_px_per_cm": px_per_cm}}

    res = refine_locomotion_segments([ev], store, binding)
    assert len(res) == 1
    assert res[0]["start_frame"] == 0
    assert res[0]["end_frame"] == 60


def test_refine_locomotion_segments_trims_static_prefix():
    fps = 30.0
    px_per_cm = 10.0
    rows = []
    # Frames 0..44 stationary (1.5s), frames 45..89 active locomotion
    for f in range(90):
        if f < 45:
            pos = (100.0, 100.0)
        else:
            pos = (100.0 + (f - 44) * 5.0, 100.0)
        rows.append({"frame": f, "source_frame": f, "time": f / fps, "detections": [_make_detection(1, pos)]})

    store = _DummyStore(rows)
    ev = {
        "event_id": "loco_trim_prefix",
        "canonical_behavior": "locomotion",
        "behavior": "Locomotion",
        "mouse_ids": [1],
        "start_frame": 0,
        "end_frame": 90,
    }
    binding = {"source_video": {"fps": fps, "scale_px_per_cm": px_per_cm}}

    res = refine_locomotion_segments([ev], store, binding)
    assert len(res) == 1
    assert res[0]["start_frame"] >= 40
    assert res[0]["end_frame"] == 90


def test_refine_attack_lookforward_is_bounded_on_long_videos():
    """Attack forward extension must stop at end + lookforward, not scan to frame_count."""
    fps = 30.0
    rows = []
    for f in range(1000):
        dx = (f % 2) * 15.0 if 100 <= f < 130 else 0.0
        rows.append({
            "frame": f,
            "source_frame": f,
            "time": f / fps,
            "detections": [
                _make_detection(1, (110.0, 100.0), (40.0, 20.0)),
                _make_detection(2, (150.0 + dx, 100.0), (40.0, 20.0)),
            ],
        })
    ev = {
        "event_id": "att_bounded",
        "canonical_behavior": "attack",
        "behavior": "Attack",
        "mouse_ids": [1, 2],
        "start_frame": 100,
        "end_frame": 130,
    }
    res = refine_attack_segments([ev], _DummyStore(rows), {"source_video": {"fps": fps, "frame_count": 1000}})
    assert len(res) == 1
    assert res[0]["end_frame"] <= 130 + int(round(6.0 * fps))


def test_refine_behavior_segments_forwards_scale_and_1080p_rearing():
    """refine_behavior_segments must forward `scale` to submodules and support 1080p rearing."""
    fps = 30.0
    # 1. Isolation with explicit scale={"cm_per_pixel": 0.05}: 220px = 11cm (< 15cm), must reject
    rows_iso = [
        {
            "frame": f,
            "source_frame": f,
            "time": f / fps,
            "detections": [
                {"track_id": 1, "box": [100, 100, 120, 120], "keypoints": []},
                {"track_id": 2, "box": [320, 100, 340, 120], "keypoints": []},
            ],
        }
        for f in range(360)
    ]
    ev_iso = {
        "event_id": "iso_scale_fwd",
        "canonical_behavior": "isolation",
        "behavior": "Isolation",
        "mouse_ids": [1],
        "target_mouse_id": 1,
        "start_frame": 0,
        "end_frame": 360,
    }
    res_iso, _ = refine_behavior_segments(
        [ev_iso],
        _DummyStore(rows_iso),
        {"source_video": {"fps": fps}},
        scale={"cm_per_pixel": 0.05},
        enable_huddle_sliding=False,
        enable_fine_cropping=True,
    )
    assert len(res_iso) == 0

    # 2. 1080p-scale rearing (ear_neck=22px, nose_tail=90px) retained when scale=0.06 cm/px
    rear_kps = np.array([
        [500.0, 410.0, 0.9],
        [515.0, 430.0, 0.9],
        [485.0, 430.0, 0.9],
        [500.0, 452.0, 0.9],
        [482.0, 480.0, 0.9],
        [518.0, 480.0, 0.9],
        [500.0, 500.0, 0.9],
    ])
    rows_rear = [
        {
            "frame": f,
            "source_frame": f,
            "time": f / fps,
            "detections": [_make_detection(1, (500.0, 455.0), (70.0, 100.0), rear_kps)],
        }
        for f in range(45)
    ]
    ev_rear = {
        "event_id": "rear_1080p",
        "canonical_behavior": "rearing",
        "behavior": "Rearing",
        "mouse_ids": [1],
        "start_frame": 0,
        "end_frame": 45,
    }
    res_rear = refine_rearing_segments(
        [ev_rear],
        _DummyStore(rows_rear),
        {"source_video": {"fps": fps}},
        scale={"cm_per_pixel": 0.06},
    )
    assert len(res_rear) == 1


def test_refine_group_locomotion_respects_15fps_support_frames():
    """At 15 fps, ~3.5s (52 frames) with 24 support frames (> 0.933s * 15 = 14) must pass."""
    fps = 15.0
    rows = [
        {
            "frame": f,
            "source_frame": f,
            "time": f / fps,
            "detections": [
                _make_detection(1, (50.0 + f * 3.0, 100.0)),
                _make_detection(2, (60.0 + f * 3.0, 105.0)),
                _make_detection(3, (55.0 + f * 3.0, 110.0)),
            ],
        }
        for f in range(52)
    ]
    ev = {
        "event_id": "gloco_15fps",
        "canonical_behavior": "group_locomotion",
        "behavior": "Group locomotion",
        "mouse_ids": [1, 2, 3],
        "start_frame": 0,
        "end_frame": 52,
        "canonical_support_frame_intervals": [[0, 24]],
        "canonical_support_frame_count": 24,
    }
    res = refine_group_locomotion_segments([ev], _DummyStore(rows), {"source_video": {"fps": fps}})
    assert len(res) == 1

