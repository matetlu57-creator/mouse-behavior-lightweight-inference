"""Unit tests for Stage 2 modular behavior segment clipping upgrades."""
from __future__ import annotations

import numpy as np

from mouse_behavior.behavior.segment_clipping import (
    MIN_PARTICIPANTS_BY_BEHAVIOR,
    min_participants_for_behavior,
    refine_behavior_segments,
    refine_contact_and_together_segments,
    refine_dispersal_segments,
    refine_grooming_segments,
    refine_group_locomotion_segments,
    refine_isolation_segments,
    refine_social_clustering_segments,
    select_behavior_segments,
)
from mouse_behavior.behavior.segment_clipping_common import InMemoryTrackStore
from mouse_behavior.evaluation.exported_dataset_models import HeuristicParameters
from mouse_behavior.evaluation.group_cropping_validation import (
    audit_group_segment_kinematics,
    refine_all_canonical_and_group_events,
    select_group_review_clips,
    summarize_group_cropping_validation,
)
from mouse_behavior.visualization.behavior_packages import huddle_output_events


def _det(
    track_id: int,
    center: tuple[float, float],
    *,
    box_size: tuple[float, float] = (20.0, 20.0),
    nose_offset: tuple[float, float] = (0.0, -10.0),
    tail_offset: tuple[float, float] = (0.0, 12.0),
    kps: np.ndarray | None = None,
) -> dict:
    cx, cy = center
    w, h = box_size
    box = [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0]
    if kps is None:
        nx, ny = cx + nose_offset[0], cy + nose_offset[1]
        tx, ty = cx + tail_offset[0], cy + tail_offset[1]
        pts = [
            [nx, ny, 0.9],
            [cx + 3.0, cy - 6.0, 0.9],
            [cx - 3.0, cy - 6.0, 0.9],
            [cx, cy - 4.0, 0.9],
            [cx - 4.0, cy + 4.0, 0.9],
            [cx + 4.0, cy + 4.0, 0.9],
            [tx, ty, 0.9],
        ]
    else:
        pts = kps.tolist()
    return {
        "track_id": int(track_id),
        "box": box,
        "keypoints": pts,
        "detection_confidence": 0.95,
        "class_id": 0,
    }


def _row(frame: int, fps: float, dets: list[dict]) -> dict:
    return {"frame": frame, "source_frame": frame, "time": frame / fps, "detections": dets}


def test_min_participants_by_behavior_mapping_and_enforcement() -> None:
    expected = {
        "huddle": 3,
        "social_clustering": 3,
        "dispersal": 3,
        "group_locomotion": 3,
        "approach": 2,
        "attack": 2,
        "chasing": 2,
        "head_contact": 2,
        "tail_contact": 2,
        "together": 2,
        "static": 1,
        "locomotion": 1,
        "grooming": 1,
        "rearing": 1,
        "isolation": 1,
    }
    assert MIN_PARTICIPANTS_BY_BEHAVIOR == expected
    assert min_participants_for_behavior("Group locomotion") == 3
    assert min_participants_for_behavior("chase") == 2
    assert min_participants_for_behavior("nose_head_contact") == 2
    assert min_participants_for_behavior("Snout-rear_contact") == 2
    assert min_participants_for_behavior("stationary") == 1

    fps = 30.0
    events = [
        {
            "event_id": "huddle_2mice",
            "canonical_behavior": "huddle",
            "behavior": "Huddling",
            "mouse_ids": [1, 2],
            "start_frame": 0,
            "end_frame": 180,
        },
        {
            "event_id": "sc_dup_id",
            "canonical_behavior": "social_clustering",
            "behavior": "Social clustering",
            "mouse_ids": [1, 1, 2],
            "start_frame": 0,
            "end_frame": 60,
        },
        {
            "event_id": "approach_1mouse",
            "canonical_behavior": "approach",
            "behavior": "Approach",
            "mouse_ids": [1],
            "start_frame": 0,
            "end_frame": 30,
        },
        {
            "event_id": "gloco_3mice",
            "canonical_behavior": "group_locomotion",
            "behavior": "Group locomotion",
            "mouse_ids": [1, 2, 3],
            "start_frame": 0,
            "end_frame": 120,
            "canonical_support_frame_intervals": [[0, 120]],
            "canonical_support_frame_count": 120,
        },
    ]

    refined, ref_audit = refine_behavior_segments(
        events,
        InMemoryTrackStore([]),
        {"source_video": {"fps": fps, "frame_count": 200}},
        enable_huddle_sliding=False,
        enable_fine_cropping=False,
    )
    assert [e["event_id"] for e in refined] == ["gloco_3mice"]
    assert ref_audit["rejected_insufficient_participants"] == 3

    selected, sel_summary = select_behavior_segments(
        events,
        binding={"source_video": {"fps": fps, "frame_count": 200}},
        require_closed_boundaries=False,
        require_endpoint_visibility=False,
    )
    assert [e["event_id"] for e in selected] == ["gloco_3mice"]
    assert sel_summary["rejected_counts"]["insufficient_participants"] == 3


def test_strict_non_overlapping_shared_mouse_selection() -> None:
    fps = 30.0
    events = [
        # Overlapping same-behavior clips sharing mouse 2 -> higher-quality one must win
        {
            "event_id": "sc_low_quality",
            "canonical_behavior": "social_clustering",
            "behavior": "Social clustering",
            "mouse_ids": [1, 2, 3],
            "start_frame": 0,
            "end_frame": 120,
            "observed_all_participants_frame_count": 60,
            "canonical_support_frame_count": 60,
        },
        {
            "event_id": "sc_high_quality",
            "canonical_behavior": "social_clustering",
            "behavior": "Social clustering",
            "mouse_ids": [2, 4, 5],
            "start_frame": 30,
            "end_frame": 140,
            "observed_all_participants_frame_count": 110,
            "canonical_support_frame_count": 110,
        },
        # Disjoint mouse set of the same behavior overlapping in time -> must coexist
        {
            "event_id": "sc_disjoint",
            "canonical_behavior": "social_clustering",
            "behavior": "Social clustering",
            "mouse_ids": [6, 7, 8],
            "start_frame": 20,
            "end_frame": 130,
            "observed_all_participants_frame_count": 110,
            "canonical_support_frame_count": 110,
        },
        # Different behavior sharing mouse 2 at the exact same time -> must coexist
        {
            "event_id": "together_shared_mouse2",
            "canonical_behavior": "together",
            "behavior": "Together",
            "mouse_ids": [2, 4],
            "start_frame": 30,
            "end_frame": 90,
            "observed_all_participants_frame_count": 60,
            "canonical_support_frame_count": 60,
        },
        # Overlapping alias behavior ("chase" vs "chasing") sharing mouse 1 -> higher priority wins
        {
            "event_id": "chase_bad_boundary",
            "canonical_behavior": "chase",
            "behavior": "Chasing",
            "mouse_ids": [1, 2],
            "start_frame": 100,
            "end_frame": 160,
            "observed_all_participants_frame_count": 60,
            "canonical_support_frame_count": 60,
            "kinematic_ok": False,
        },
        {
            "event_id": "chasing_good",
            "canonical_behavior": "chasing",
            "behavior": "Chasing",
            "mouse_ids": [1, 3],
            "start_frame": 120,
            "end_frame": 170,
            "observed_all_participants_frame_count": 50,
            "canonical_support_frame_count": 50,
            "kinematic_ok": True,
        },
    ]
    selected, summary = select_behavior_segments(
        events,
        binding={"source_video": {"fps": fps, "frame_count": 300}},
        require_closed_boundaries=False,
        require_endpoint_visibility=False,
        strict_non_overlapping_shared_mouse=True,
    )
    selected_ids = {e["event_id"] for e in selected}
    assert selected_ids == {
        "sc_high_quality",
        "sc_disjoint",
        "together_shared_mouse2",
        "chasing_good",
    }
    assert summary["rejected_counts"]["duplicate_or_overlapping"] == 2


def test_refine_isolation_segments_unchanged_behavior() -> None:
    fps = 30.0
    rows = [
        _row(
            f,
            fps,
            [
                _det(1, (500.0, 500.0)),
                _det(2, (100.0, 100.0)),
            ],
        )
        for f in range(360)
    ]
    ev = {
        "event_id": "iso_base",
        "canonical_behavior": "isolation",
        "behavior": "Isolation",
        "mouse_ids": [1],
        "target_mouse_id": 1,
        "start_frame": 0,
        "end_frame": 360,
    }
    res = refine_isolation_segments(
        [ev],
        InMemoryTrackStore(rows),
        {"source_video": {"fps": fps, "scale_px_per_cm": 10.0}},
    )
    assert len(res) == 1
    assert (res[0]["start_frame"], res[0]["end_frame"]) == (0, 360)
    assert res[0]["canonical_support_frame_count"] == 360


def test_refine_group_locomotion_extracts_sync_bouts_and_rejects_stationary_gaps() -> None:
    fps = 30.0
    scale = {"cm_per_pixel": 0.1}

    # Case A: 20s span (600 frames) with only 0.3s (9 frames) of sync motion -> must reject
    rows_dakuangchang = []
    for f in range(600):
        shift = 4.0 * min(f, 9)
        rows_dakuangchang.append(
            _row(
                f,
                fps,
                [
                    _det(1, (100.0 + shift, 100.0)),
                    _det(2, (120.0 + shift, 100.0)),
                    _det(3, (110.0 + shift, 120.0)),
                ],
            )
        )
    ev_bad = {
        "event_id": "gl_dakuangchang_like",
        "canonical_behavior": "group_locomotion",
        "behavior": "Group locomotion",
        "mouse_ids": [1, 2, 3],
        "start_frame": 0,
        "end_frame": 600,
    }
    res_bad = refine_group_locomotion_segments(
        [ev_bad],
        InMemoryTrackStore(rows_dakuangchang),
        {"source_video": {"fps": fps, "frame_count": 600}},
        scale=scale,
    )
    assert res_bad == []

    # Case B: 16s span with two real 4s synchronous bouts separated by a 4s stationary pause
    rows_two_bouts = []
    for f in range(480):
        if f < 120:
            x = 50.0 + f * 3.0
        elif f < 240:
            x = 50.0 + 119 * 3.0  # 4.0s stationary pause (> 1.5s)
        elif f < 360:
            x = 50.0 + 119 * 3.0 + (f - 240) * 3.0
        else:
            x = 50.0 + 119 * 3.0 + 119 * 3.0  # trailing 4.0s stationary pause
        rows_two_bouts.append(
            _row(
                f,
                fps,
                [
                    _det(1, (x, 100.0)),
                    _det(2, (x + 20.0, 105.0)),
                    _det(3, (x + 10.0, 120.0)),
                ],
            )
        )
    ev_split = {
        "event_id": "gl_split_bouts",
        "canonical_behavior": "group_locomotion",
        "behavior": "Group locomotion",
        "mouse_ids": [1, 2, 3],
        "start_frame": 0,
        "end_frame": 480,
    }
    res_split = refine_group_locomotion_segments(
        [ev_split],
        InMemoryTrackStore(rows_two_bouts),
        {"source_video": {"fps": fps, "frame_count": 480}},
        scale=scale,
    )
    assert len(res_split) == 2
    assert res_split[0]["end_frame"] <= 130
    assert 230 <= res_split[1]["start_frame"] <= 250
    assert res_split[1]["end_frame"] <= 370
    for seg in res_split:
        dur_frames = seg["end_frame"] - seg["start_frame"]
        assert seg["canonical_support_frame_count"] / dur_frames >= 0.30
        assert seg["canonical_support_frame_intervals"]


def test_refine_dispersal_recovers_origin_members_and_caps_duration() -> None:
    fps = 30.0
    scale = {"cm_per_pixel": 0.1}

    # Case A: 2 departing mice [1, 2] listed in event, mouse 3 stayed in compact origin -> recover [1, 2, 3]
    rows_rec = []
    for f in range(210):
        spread = min(120.0, max(0.0, (f - 15) * 1.5))
        rows_rec.append(
            _row(
                f,
                fps,
                [
                    _det(1, (300.0 - spread, 300.0)),
                    _det(2, (300.0 + spread, 300.0)),
                    _det(3, (305.0, 305.0)),
                ],
            )
        )
    ev_2mice = {
        "event_id": "disp_2mice_raw",
        "canonical_behavior": "dispersal",
        "behavior": "Dispersal",
        "mouse_ids": [1, 2],
        "start_frame": 0,
        "end_frame": 210,
    }
    res_rec = refine_dispersal_segments(
        [ev_2mice],
        InMemoryTrackStore(rows_rec),
        {"source_video": {"fps": fps, "frame_count": 210}},
        scale=scale,
    )
    assert len(res_rec) == 1
    assert res_rec[0]["mouse_ids"] == [1, 2, 3]
    assert res_rec[0]["departing_mouse_ids"] == [1, 2]

    # Case B: only 2 mice exist in the store -> must reject (< 3 observed origin members)
    rows_only2 = [
        _row(f, fps, [d for d in r["detections"] if d["track_id"] in {1, 2}])
        for f, r in enumerate(rows_rec)
    ]
    assert (
        refine_dispersal_segments(
            [ev_2mice],
            InMemoryTrackStore(rows_only2),
            {"source_video": {"fps": fps, "frame_count": 210}},
            scale=scale,
        )
        == []
    )

    # Case C: 36s runaway dispersal clip (1080 frames) -> capped at <= 15.0s
    rows_runaway = []
    for f in range(1080):
        spread = min(200.0, max(0.0, (f - 20) * 1.2))
        rows_runaway.append(
            _row(
                f,
                fps,
                [
                    _det(1, (300.0 - spread, 300.0)),
                    _det(2, (300.0 + spread, 300.0)),
                    _det(3, (300.0, 300.0 + spread)),
                ],
            )
        )
    ev_runaway = {
        "event_id": "disp_runaway",
        "canonical_behavior": "dispersal",
        "behavior": "Dispersal",
        "mouse_ids": [1, 2, 3],
        "start_frame": 0,
        "end_frame": 1080,
    }
    res_runaway = refine_dispersal_segments(
        [ev_runaway],
        InMemoryTrackStore(rows_runaway),
        {"source_video": {"fps": fps, "frame_count": 1080}},
        scale=scale,
    )
    assert len(res_runaway) == 1
    dur_s = (res_runaway[0]["end_frame"] - res_runaway[0]["start_frame"]) / fps
    assert 5.0 < dur_s <= 15.0 + 1e-9


def test_refine_social_clustering_scene_veto_occlusion_and_bounds() -> None:
    fps = 30.0
    scale = {"cm_per_pixel": 0.1}

    # Case A: Pre-existing 3-mouse cluster [1,2,3] + outsider 4 arriving at onset >= 0.5s -> VETO
    rows_veto = []
    for f in range(180):
        if f < 45:
            p4 = (560.0, 300.0)
        elif f < 96:
            t = (f - 45) / 51.0
            p4 = (560.0 - 230.0 * t, 310.0)
        else:
            p4 = (330.0, 310.0)
        rows_veto.append(
            _row(
                f,
                fps,
                [
                    _det(1, (300.0, 300.0)),
                    _det(2, (320.0, 300.0)),
                    _det(3, (300.0, 320.0)),
                    _det(4, p4),
                ],
            )
        )
    ev_veto = {
        "event_id": "sc_preexisting_veto",
        "canonical_behavior": "social_clustering",
        "behavior": "Social clustering",
        "mouse_ids": [1, 2, 3, 4],
        "start_frame": 45,
        "end_frame": 180,
    }
    assert (
        refine_social_clustering_segments(
            [ev_veto],
            InMemoryTrackStore(rows_veto),
            {"source_video": {"fps": fps, "frame_count": 180}},
            scale=scale,
        )
        == []
    )

    # Case B: Long pre-convergence idle prefix (120 frames = 4s) + convergence (30 frames) + 12s hold
    # Must trim idle prefix and cap post-arrival hold at <= 3.0s!
    rows_bounds = []
    for f in range(510):
        if f < 120:
            p3 = (600.0, 300.0)
        elif f < 150:
            t = (f - 120) / 30.0
            p3 = (600.0 - 280.0 * t, 300.0 + 20.0 * t)
        else:
            p3 = (320.0, 320.0)
        rows_bounds.append(
            _row(
                f,
                fps,
                [
                    _det(1, (300.0, 300.0)),
                    _det(2, (330.0, 300.0)),
                    _det(3, p3),
                ],
            )
        )
    ev_bounds = {
        "event_id": "sc_idle_and_long_hold",
        "canonical_behavior": "social_clustering",
        "behavior": "Social clustering",
        "mouse_ids": [1, 2, 3],
        "start_frame": 0,
        "end_frame": 510,
    }
    res_bounds = refine_social_clustering_segments(
        [ev_bounds],
        InMemoryTrackStore(rows_bounds),
        {"source_video": {"fps": fps, "frame_count": 510}},
        scale=scale,
    )
    assert len(res_bounds) == 1
    assert res_bounds[0]["start_frame"] >= 90  # idle prefix trimmed
    assert res_bounds[0]["end_frame"] <= 245   # arrival (~145) + <= 3.0s hold (90 frames)

    # Case C: Bounded top-down pile occlusion on k=4 settled pile without crediting missing frames
    rows_occ = []
    for f in range(150):
        frac = min(1.0, f / 70.0)
        c1 = (300.0 - 180.0 * (1.0 - frac), 300.0)
        c2 = (340.0 + 180.0 * (1.0 - frac), 300.0)
        c3 = (320.0, 280.0 - 180.0 * (1.0 - frac))
        c4 = (320.0, 320.0 + 180.0 * (1.0 - frac))
        dets = [_det(1, c1), _det(2, c2), _det(3, c3)]
        if f < 72 or f >= 105:
            dets.append(_det(4, c4))
        rows_occ.append(_row(f, fps, dets))
    ev_occ = {
        "event_id": "sc_pile_occ",
        "canonical_behavior": "social_clustering",
        "behavior": "Social clustering",
        "mouse_ids": [1, 2, 3, 4],
        "start_frame": 0,
        "end_frame": 150,
    }
    res_occ = refine_social_clustering_segments(
        [ev_occ],
        InMemoryTrackStore(rows_occ),
        {"source_video": {"fps": fps, "frame_count": 150}},
        scale=scale,
    )
    assert len(res_occ) == 1
    assert res_occ[0]["duration_s"] > 3.0
    assert res_occ[0]["end_frame"] <= 125
    assert res_occ[0]["canonical_support_frame_count"] <= (res_occ[0]["end_frame"] - res_occ[0]["start_frame"])


def test_huddle_output_events_min_participants_and_optional_max_duration() -> None:
    fps = 10.0
    length = 220  # 22.0s at 10fps
    rows = [
        _row(
            f,
            fps,
            [
                _det(1, (15.0, 5.0), box_size=(12.0, 10.0)),
                _det(2, (24.0, 5.0), box_size=(12.0, 10.0)),
                _det(3, (33.0, 5.0), box_size=(12.0, 10.0)),
            ],
        )
        for f in range(length)
    ]
    store = InMemoryTrackStore(rows)
    binding = {"source_video": {"fps": fps, "frame_count": length}}
    scale = {"cm_per_pixel": 0.1}

    ev_2 = {
        "event_id": "hud_2",
        "canonical_behavior": "huddle",
        "behavior": "Huddling",
        "mouse_ids": [1, 2],
        "start_frame": 0,
        "end_frame": length,
    }
    ev_dup = {
        "event_id": "hud_dup",
        "canonical_behavior": "huddle",
        "behavior": "Huddling",
        "mouse_ids": [1, 1, 2],
        "start_frame": 0,
        "end_frame": length,
    }
    out_rej, audit_rej = huddle_output_events([ev_2, ev_dup], store, binding, scale)
    assert out_rej == []
    assert all(a.get("reason") == "insufficient_participants" for a in audit_rej)

    ev_3 = {
        "event_id": "hud_long",
        "canonical_behavior": "huddle",
        "behavior": "Huddling",
        "mouse_ids": [1, 2, 3],
        "start_frame": 0,
        "end_frame": length,
    }
    out_unbounded, _ = huddle_output_events([ev_3], store, binding, scale)
    assert len(out_unbounded) == 1
    assert (out_unbounded[0]["end_frame"] - out_unbounded[0]["start_frame"]) / fps == 22.0

    out_capped, _ = huddle_output_events([ev_3], store, binding, scale, max_duration_s=15.0)
    assert len(out_capped) == 2
    assert all((e["end_frame"] - e["start_frame"]) / fps <= 15.0 + 1e-9 for e in out_capped)
    assert out_capped[0]["end_frame"] <= out_capped[1]["start_frame"]


def test_refine_grooming_and_contact_together_segments_and_dispatch() -> None:
    fps = 30.0
    scale = {"cm_per_pixel": 0.1}

    # 1. Grooming: frames 0..29 walking fast, frames 30..89 seated grooming (nose oscillating), 90..119 walking fast
    rows_groom = []
    for f in range(120):
        if f < 30:
            cx, cy = 50.0 + f * 5.0, 200.0
            nx, ny = cx, cy - 25.0
        elif f < 90:
            cx, cy = 200.0, 200.0
            osc = 14.0 if (f % 4 < 2) else -4.0
            nx, ny = cx, cy - 12.0 + osc
        else:
            cx, cy = 200.0 + (f - 89) * 5.0, 200.0
            nx, ny = cx, cy - 25.0
        kps = np.array(
            [
                [nx, ny, 0.9],
                [cx + 4.0, cy - 8.0, 0.9],
                [cx - 4.0, cy - 8.0, 0.9],
                [cx, cy - 5.0, 0.9],
                [cx - 5.0, cy + 12.0, 0.9],
                [cx + 5.0, cy + 12.0, 0.9],
                [cx, cy + 25.0, 0.9],
            ]
        )
        rows_groom.append(_row(f, fps, [_det(1, (cx, cy), box_size=(20.0, 50.0), kps=kps)]))
    ev_groom = {
        "event_id": "groom_padded",
        "canonical_behavior": "grooming",
        "behavior": "Grooming",
        "mouse_ids": [1],
        "start_frame": 0,
        "end_frame": 120,
    }
    res_groom = refine_grooming_segments(
        [ev_groom],
        InMemoryTrackStore(rows_groom),
        {"source_video": {"fps": fps, "frame_count": 120}},
        scale=scale,
    )
    assert len(res_groom) == 1
    assert res_groom[0]["start_frame"] >= 25
    assert res_groom[0]["end_frame"] <= 95

    # Pure walking tagged as grooming must be rejected
    rows_walk = rows_groom[:30]
    assert (
        refine_grooming_segments(
            [dict(ev_groom, end_frame=30)],
            InMemoryTrackStore(rows_walk),
            {"source_video": {"fps": fps, "frame_count": 30}},
            scale=scale,
        )
        == []
    )

    # 2. Head contact: frames 0..19 separated (15cm), 20..49 snout-to-head contact (<= 2cm), 50..69 separated
    rows_contact = []
    for f in range(70):
        sep = 150.0 if (f < 20 or f >= 50) else 15.0
        # Mouse 1 nose at (200, 200); mouse 2 head at (200 + sep, 200)
        kps1 = np.array(
            [
                [200.0, 200.0, 0.9],
                [190.0, 195.0, 0.9],
                [190.0, 205.0, 0.9],
                [180.0, 200.0, 0.9],
                [160.0, 195.0, 0.9],
                [160.0, 205.0, 0.9],
                [140.0, 200.0, 0.9],
            ]
        )
        bx2 = 200.0 + sep
        kps2 = np.array(
            [
                [bx2, 200.0, 0.9],
                [bx2 + 10.0, 195.0, 0.9],
                [bx2 + 10.0, 205.0, 0.9],
                [bx2 + 20.0, 200.0, 0.9],
                [bx2 + 40.0, 195.0, 0.9],
                [bx2 + 40.0, 205.0, 0.9],
                [bx2 + 60.0, 200.0, 0.9],
            ]
        )
        rows_contact.append(
            _row(
                f,
                fps,
                [
                    _det(1, (170.0, 200.0), kps=kps1),
                    _det(2, (bx2 + 30.0, 200.0), kps=kps2),
                ],
            )
        )
    ev_hc = {
        "event_id": "hc_padded",
        "canonical_behavior": "nose_head_contact",
        "behavior": "Snout-head_contact",
        "mouse_ids": [1, 2],
        "start_frame": 0,
        "end_frame": 70,
    }
    res_hc = refine_contact_and_together_segments(
        [ev_hc],
        InMemoryTrackStore(rows_contact),
        {"source_video": {"fps": fps, "frame_count": 70}},
        scale=scale,
    )
    assert len(res_hc) == 1
    assert res_hc[0]["start_frame"] == 20
    assert res_hc[0]["end_frame"] == 50

    # 3. Together: frames 0..19 slowly approaching outside Together radius (> 5.318cm at 1.5cm/s),
    # 20..64 quiet together (< 5.318cm and speed < 3.0cm/s), 65..84 separated
    rows_tog = []
    for f in range(85):
        if f < 20:
            sep = 63.0 - 0.5 * f
        elif f < 65:
            sep = max(30.0, 53.0 - 0.5 * (f - 20))
        else:
            sep = 150.0
        rows_tog.append(
            _row(
                f,
                fps,
                [
                    _det(1, (200.0, 200.0)),
                    _det(2, (200.0 + sep, 200.0)),
                ],
            )
        )
    ev_tog = {
        "event_id": "tog_padded",
        "canonical_behavior": "together",
        "behavior": "Together",
        "mouse_ids": [1, 2],
        "start_frame": 0,
        "end_frame": 85,
    }
    res_tog = refine_contact_and_together_segments(
        [ev_tog],
        InMemoryTrackStore(rows_tog),
        {"source_video": {"fps": fps, "frame_count": 85}},
        scale=scale,
    )
    assert len(res_tog) == 1
    assert res_tog[0]["start_frame"] == 20
    assert res_tog[0]["end_frame"] == 65

    # Fast fly-by within Together distance (> vel_together_max_cm_s) must be rejected
    rows_fast_pass = [
        _row(
            f,
            fps,
            [
                _det(1, (200.0, 200.0)),
                _det(2, (160.0 + f * 3.0, 220.0)),
            ],
        )
        for f in range(40)
    ]
    assert (
        refine_contact_and_together_segments(
            [dict(ev_tog, end_frame=40)],
            InMemoryTrackStore(rows_fast_pass),
            {"source_video": {"fps": fps, "frame_count": 40}},
            scale=scale,
        )
        == []
    )

    # Unified dispatch via refine_behavior_segments
    dispatched, _ = refine_behavior_segments(
        [ev_groom],
        InMemoryTrackStore(rows_groom),
        {"source_video": {"fps": fps, "frame_count": 120}},
        scale=scale,
        enable_huddle_sliding=False,
        enable_fine_cropping=True,
    )
    assert len(dispatched) == 1 and dispatched[0]["start_frame"] >= 25

    dispatched_tog, _ = refine_behavior_segments(
        [ev_tog],
        InMemoryTrackStore(rows_tog),
        {"source_video": {"fps": fps, "frame_count": 85}},
        scale=scale,
        enable_huddle_sliding=False,
        enable_fine_cropping=True,
    )
    assert len(dispatched_tog) == 1 and (dispatched_tog[0]["start_frame"], dispatched_tog[0]["end_frame"]) == (20, 65)


def test_group_cropping_validation_audits_participants_bounds_and_non_overlap() -> None:
    fps = 10.0
    scale = {"cm_per_pixel": 0.1, "reference_body_cm": 7.918}
    binding = {"source_video": {"fps": fps, "frame_count": 100}}
    params = HeuristicParameters()

    rows = []
    for f in range(100):
        d = max(0.0, f - 5) * 3.0
        rows.append(
            _row(
                f,
                fps,
                [
                    _det(1, (200.0 - d, 200.0)),
                    _det(2, (200.0 + d, 200.0)),
                    _det(3, (200.0, 200.0 + d)),
                ],
            )
        )
    store = InMemoryTrackStore(rows)

    bad_disp = {
        "event_id": "disp_2mice_bad",
        "behavior": "Dispersal",
        "canonical_behavior": "dispersal",
        "mouse_ids": [1, 2],
        "start_frame": 0,
        "end_frame": 75,
        "observed_all_participants_frame_count": 75,
    }
    kin_bad = audit_group_segment_kinematics(bad_disp, store, scale, fps, params)
    assert kin_bad["min_participants_ok"] is False
    assert kin_bad["full_crop_precision_ok"] is False

    good_disp = {
        "event_id": "disp_3mice_ok",
        "behavior": "Dispersal",
        "canonical_behavior": "dispersal",
        "mouse_ids": [1, 2, 3],
        "start_frame": 2,
        "end_frame": 70,
        "observed_all_participants_frame_count": 68,
        "canonical_support_frame_count": 68,
    }
    overlap_disp = {
        "event_id": "disp_3mice_overlap",
        "behavior": "Dispersal",
        "canonical_behavior": "dispersal",
        "mouse_ids": [1, 2, 3],
        "start_frame": 10,
        "end_frame": 80,
        "observed_all_participants_frame_count": 50,
        "canonical_support_frame_count": 50,
    }
    summary, audits = summarize_group_cropping_validation(
        [bad_disp],
        [good_disp, overlap_disp],
        store,
        binding,
        scale,
        params,
    )
    assert summary["aggregate"]["all_behaviors_shared_mouse_overlap_count"] == 1
    assert summary["aggregate"]["zero_shared_mouse_overlap_ok"] is False

    # select_group_review_clips must drop overlapping shared-mouse clips even under default max_per_behavior
    selected_review = select_group_review_clips(
        [good_disp, overlap_disp],
        store,
        binding,
        per_event_audit=audits,
        max_per_behavior=30,
    )
    assert len(selected_review) == 1

    # refine_all_canonical_and_group_events resolves overlaps and yields 0 shared-mouse overlaps
    refined_all, _ = refine_all_canonical_and_group_events(
        [good_disp, overlap_disp],
        store,
        binding,
        scale,
        params,
    )
    summary_clean, _ = summarize_group_cropping_validation(
        [good_disp],
        refined_all,
        store,
        binding,
        scale,
        params,
    )
    assert summary_clean["aggregate"]["zero_shared_mouse_overlap_ok"] is True
    assert summary_clean["aggregate"]["min_participants_pass_rate"] == 1.0


def test_refine_group_locomotion_caps_at_9s_and_isolation_caps_at_20s() -> None:
    fps = 30.0
    scale = {"cm_per_pixel": 0.1}
    params = HeuristicParameters()

    # 1. Group locomotion spanning 14.0s (420 frames) splits into <= 9.0s, > 3.0s non-overlapping clips
    rows_gloco = [
        _row(
            f,
            fps,
            [
                _det(1, (50.0 + f * 3.0, 100.0)),
                _det(2, (70.0 + f * 3.0, 105.0)),
                _det(3, (60.0 + f * 3.0, 120.0)),
            ],
        )
        for f in range(420)
    ]
    gloco_store = InMemoryTrackStore(rows_gloco)
    gloco_ev = {
        "event_id": "gloco_14s",
        "canonical_behavior": "group_locomotion",
        "behavior": "Group locomotion",
        "mouse_ids": [1, 2, 3],
        "start_frame": 0,
        "end_frame": 420,
    }
    res_gloco_capped = refine_group_locomotion_segments(
        [gloco_ev],
        gloco_store,
        {"source_video": {"fps": fps, "frame_count": 420}},
        scale=scale,
        parameters=params,
    )
    assert len(res_gloco_capped) == 2
    assert res_gloco_capped[0]["end_frame"] <= res_gloco_capped[1]["start_frame"]
    for seg in res_gloco_capped:
        dur_s = (seg["end_frame"] - seg["start_frame"]) / fps
        assert 3.0 < dur_s <= 9.0 + 1e-9

    res_gloco_unbounded = refine_group_locomotion_segments(
        [gloco_ev],
        gloco_store,
        {"source_video": {"fps": fps, "frame_count": 420}},
        scale=scale,
        parameters=params,
        max_duration_s=None,
    )
    assert len(res_gloco_unbounded) == 1
    assert (res_gloco_unbounded[0]["end_frame"] - res_gloco_unbounded[0]["start_frame"]) / fps == 14.0

    # 2. Isolation spanning 35.0s (1050 frames) splits into <= 20.0s, > 10.0s non-overlapping clips
    rows_iso = [
        _row(
            f,
            fps,
            [
                _det(1, (500.0, 500.0)),
                _det(2, (100.0, 100.0)),
            ],
        )
        for f in range(1050)
    ]
    iso_store = InMemoryTrackStore(rows_iso)
    iso_ev_35s = {
        "event_id": "iso_35s",
        "canonical_behavior": "isolation",
        "behavior": "Isolation",
        "mouse_ids": [1],
        "target_mouse_id": 1,
        "start_frame": 0,
        "end_frame": 1050,
    }
    res_iso_capped = refine_isolation_segments(
        [iso_ev_35s],
        iso_store,
        {"source_video": {"fps": fps, "frame_count": 1050}, "scale_px_per_cm": 10.0},
        parameters=params,
    )
    assert len(res_iso_capped) == 2
    assert (res_iso_capped[0]["start_frame"], res_iso_capped[0]["end_frame"]) == (0, 600)
    assert (res_iso_capped[1]["start_frame"], res_iso_capped[1]["end_frame"]) == (600, 1050)
    for seg in res_iso_capped:
        dur_s = (seg["end_frame"] - seg["start_frame"]) / fps
        assert 10.0 < dur_s <= 20.0 + 1e-9

    # 25.0s run keeps 20.0s first slice and drops the 5.0s remainder (below >10.0s minimum)
    iso_ev_25s = {**iso_ev_35s, "event_id": "iso_25s", "end_frame": 750}
    res_iso_25s = refine_isolation_segments(
        [iso_ev_25s],
        InMemoryTrackStore(rows_iso[:750]),
        {"source_video": {"fps": fps, "frame_count": 750}, "scale_px_per_cm": 10.0},
        parameters=params,
    )
    assert len(res_iso_25s) == 1
    assert (res_iso_25s[0]["start_frame"], res_iso_25s[0]["end_frame"]) == (0, 600)

    res_iso_unbounded = refine_isolation_segments(
        [iso_ev_35s],
        iso_store,
        {"source_video": {"fps": fps, "frame_count": 1050}, "scale_px_per_cm": 10.0},
        parameters=params,
        max_duration_s=None,
    )
    assert len(res_iso_unbounded) == 1
    assert (res_iso_unbounded[0]["start_frame"], res_iso_unbounded[0]["end_frame"]) == (0, 1050)

    # 3. Audit flags over-cap group_locomotion (>9s), over-cap isolation (>20s), and short social_clustering (<=3s)
    audit_gloco_over = audit_group_segment_kinematics(
        {
            "event_id": "gloco_10s_over",
            "canonical_behavior": "group_locomotion",
            "behavior": "Group locomotion",
            "mouse_ids": [1, 2, 3],
            "start_frame": 0,
            "end_frame": 300,
            "observed_all_participants_frame_count": 300,
            "canonical_support_frame_count": 300,
            "canonical_support_frame_intervals": [[0, 300]],
        },
        gloco_store,
        scale,
        fps,
        params,
    )
    assert audit_gloco_over["max_duration_s"] == 9.0
    assert audit_gloco_over["duration_bounds_ok"] is False
    assert audit_gloco_over["full_crop_precision_ok"] is False

    audit_iso_over = audit_group_segment_kinematics(
        {
            "event_id": "iso_22s_over",
            "canonical_behavior": "isolation",
            "behavior": "Isolation",
            "mouse_ids": [1],
            "target_mouse_id": 1,
            "start_frame": 0,
            "end_frame": 660,
            "observed_all_participants_frame_count": 660,
            "canonical_support_frame_count": 660,
            "canonical_support_frame_intervals": [[0, 660]],
        },
        iso_store,
        scale,
        fps,
        params,
    )
    assert audit_iso_over["max_duration_s"] == 20.0
    assert audit_iso_over["duration_bounds_ok"] is False
    assert audit_iso_over["full_crop_precision_ok"] is False

    audit_sc_short = audit_group_segment_kinematics(
        {
            "event_id": "sc_25s_short",
            "canonical_behavior": "social_clustering",
            "behavior": "Social clustering",
            "mouse_ids": [1, 2, 3],
            "start_frame": 0,
            "end_frame": 75,
            "observed_all_participants_frame_count": 75,
            "canonical_support_frame_count": 75,
        },
        iso_store,
        scale,
        fps,
        params,
    )
    assert audit_sc_short["duration_bounds_ok"] is False
    assert audit_sc_short["full_crop_precision_ok"] is False

    # Frames where no other mouse is detected must NOT count as confirmed isolation support
    rows_missing_other = []
    for f in range(360):
        dets = [_det(1, (500.0, 500.0))]
        if f >= 180:
            dets.append(_det(2, (100.0, 100.0)))
        rows_missing_other.append(_row(f, fps, dets))
    iso_partial_other = {
        **iso_ev_35s,
        "event_id": "iso_missing_other",
        "start_frame": 0,
        "end_frame": 360,
    }
    assert (
        refine_isolation_segments(
            [iso_partial_other],
            InMemoryTrackStore(rows_missing_other),
            {"source_video": {"fps": fps, "frame_count": 360}, "scale_px_per_cm": 10.0},
            parameters=params,
        )
        == []
    )
