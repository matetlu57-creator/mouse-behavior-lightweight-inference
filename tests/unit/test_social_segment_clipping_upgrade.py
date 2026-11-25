"""Regression and unit tests for modularized social segment clipping upgrades."""
from __future__ import annotations

from importlib import import_module

import numpy as np

from mouse_behavior import heuristics
from mouse_behavior.behavior import (
    segment_clipping,
    segment_clipping_social,
    segment_clipping_social_approach,
    segment_clipping_social_attack_chasing,
    segment_clipping_social_contact,
)
from mouse_behavior.behavior.segment_clipping_common import (
    InMemoryTrackStore,
    endpoints_fully_observed,
)
from mouse_behavior.behavior.duration_policy import qualify_event_duration
from mouse_behavior.canonical_video_pipeline import rule_execution_binding
from mouse_behavior.evaluation.exported_dataset_models import HeuristicParameters


def _det(
    track_id: int,
    center: tuple[float, float],
    *,
    dx_nose: float = 15.0,
    tail_dir: float = -1.0,
) -> dict:
    cx, cy = float(center[0]), float(center[1])
    nose = [cx + dx_nose, cy, 0.9]
    lear = [cx + 0.4 * dx_nose, cy - 6.0, 0.9]
    rear = [cx + 0.4 * dx_nose, cy + 6.0, 0.9]
    spine = [cx, cy, 0.9]
    tail = [cx + tail_dir * 20.0, cy, 0.9]
    return {
        "track_id": int(track_id),
        "box": [cx - 22.0, cy - 12.0, cx + 22.0, cy + 12.0],
        "keypoints": [
            nose,
            lear,
            rear,
            [cx, cy - 4.0, 0.9],
            spine,
            [cx, cy + 4.0, 0.9],
            tail,
        ],
    }


def _row(frame: int, fps: float, dets: list[dict]) -> dict:
    return {
        "frame": int(frame),
        "source_frame": int(frame),
        "time": float(frame / fps),
        "detections": dets,
    }


def test_social_submodules_facade_identity_and_pipeline_provenance() -> None:
    assert (
        segment_clipping_social.refine_approach_segments
        is segment_clipping_social_approach.refine_approach_segments
    )
    assert (
        segment_clipping_social.refine_attack_segments
        is segment_clipping_social_attack_chasing.refine_attack_segments
    )
    assert (
        segment_clipping_social.refine_chasing_segments
        is segment_clipping_social_attack_chasing.refine_chasing_segments
    )
    assert (
        segment_clipping_social.refine_contact_and_together_segments
        is segment_clipping_social_contact.refine_contact_and_together_segments
    )
    assert (
        segment_clipping_social.discover_fast_chasing_candidates
        is segment_clipping_social_attack_chasing.discover_fast_chasing_candidates
    )
    assert (
        segment_clipping_social.discover_fight_ball_attack_episodes
        is segment_clipping_social_attack_chasing.discover_fight_ball_attack_episodes
    )
    assert (
        segment_clipping_social.discover_id_switch_chasing_candidates
        is segment_clipping_social_attack_chasing.discover_id_switch_chasing_candidates
    )
    assert (
        segment_clipping_social.attach_social_roles
        is segment_clipping_social_contact.attach_social_roles
    )
    assert (
        segment_clipping_social.build_social_roles
        is segment_clipping_social_contact.build_social_roles
    )
    assert (
        segment_clipping.refine_approach_segments
        is segment_clipping_social_approach.refine_approach_segments
    )
    assert (
        segment_clipping.refine_attack_segments
        is segment_clipping_social_attack_chasing.refine_attack_segments
    )
    assert (
        heuristics.refine_contact_and_together_segments
        is segment_clipping_social_contact.refine_contact_and_together_segments
    )

    binding = rule_execution_binding(HeuristicParameters())
    source_hashes = binding["rule_source_sha256"]
    expected = {
        "mouse_behavior.behavior.segment_clipping_social",
        "mouse_behavior.behavior.segment_clipping_social_approach",
        "mouse_behavior.behavior.segment_clipping_social_attack_chasing",
        "mouse_behavior.behavior.segment_clipping_social_contact",
    }
    assert expected <= set(source_hashes)
    for mod_name in expected:
        assert import_module(mod_name) is not None


def test_social_clipping_strictly_rejects_sub_minimum_duration() -> None:
    fps = 30.0
    scale = {"cm_per_pixel": 0.1, "reference_body_cm": 7.918}
    binding = {"source_video": {"fps": fps, "frame_count": 120}}

    rows = []
    for f in range(120):
        # Continuous head contact / close pair & closing approach geometry across frames
        rows.append(_row(f, fps, [_det(1, (200.0, 200.0), dx_nose=15.0), _det(2, (235.0, 200.0), dx_nose=-15.0)]))
    store = InMemoryTrackStore(rows)

    # 1. Attack < 0.15 s (4 frames = 0.133 s) when separated immediately before/after -> rejected
    att_rows = [
        _row(
            f,
            fps,
            [
                _det(1, (100.0, 200.0)),
                _det(2, (230.0 if 20 <= f < 24 else 700.0, 200.0)),
            ],
        )
        for f in range(60)
    ]
    assert (
        segment_clipping_social_attack_chasing.refine_attack_segments(
            [{"event_id": "att_short", "canonical_behavior": "attack", "mouse_ids": [1, 2], "start_frame": 20, "end_frame": 24}],
            InMemoryTrackStore(att_rows),
            {"source_video": {"fps": fps, "frame_count": 60}},
            scale=scale,
        )
        == []
    )

    # 2. Chasing < 1.0 s (18 frames = 0.60 s) -> rejected (no dead-air padding)
    chase_rows = [
        _row(
            f,
            fps,
            [
                _det(1, (100.0 + f * 4.0, 200.0)) if 20 <= f < 38 else _det(1, (100.0, 200.0)),
                _det(2, (150.0 + f * 4.0, 200.0)) if 20 <= f < 38 else _det(2, (600.0, 200.0)),
            ],
        )
        for f in range(80)
    ]
    assert (
        segment_clipping_social_attack_chasing.refine_chasing_segments(
            [{"event_id": "chase_short", "canonical_behavior": "chase", "mouse_ids": [1, 2], "start_frame": 20, "end_frame": 38}],
            InMemoryTrackStore(chase_rows),
            {"source_video": {"fps": fps, "frame_count": 80}},
            scale=scale,
        )
        == []
    )

    # 3. Snout-head_contact and Snout-rear_contact <= 0.40s (12 frames @ 30fps) -> rejected;
    #    13 frames (0.433s > 0.40s) -> accepted.
    hc_rows = [
        _row(
            f,
            fps,
            [
                _det(1, (200.0, 200.0), dx_nose=15.0),
                _det(2, (235.0 if 20 <= f < 32 else 650.0, 200.0), dx_nose=-15.0),
            ],
        )
        for f in range(60)
    ]
    assert (
        segment_clipping_social_contact.refine_contact_and_together_segments(
            [{"event_id": "hc_exact_04", "canonical_behavior": "nose_head_contact", "mouse_ids": [1, 2], "start_frame": 20, "end_frame": 32}],
            InMemoryTrackStore(hc_rows),
            {"source_video": {"fps": fps, "frame_count": 60}},
            scale=scale,
        )
        == []
    )
    hc_rows_ok = [
        _row(
            f,
            fps,
            [
                _det(1, (200.0, 200.0), dx_nose=15.0),
                _det(2, (235.0 if 20 <= f < 33 else 650.0, 200.0), dx_nose=-15.0),
            ],
        )
        for f in range(60)
    ]
    res_hc_ok = segment_clipping_social_contact.refine_contact_and_together_segments(
        [{"event_id": "hc_over_04", "canonical_behavior": "nose_head_contact", "mouse_ids": [1, 2], "start_frame": 20, "end_frame": 33}],
        InMemoryTrackStore(hc_rows_ok),
        {"source_video": {"fps": fps, "frame_count": 60}},
        scale=scale,
    )
    assert len(res_hc_ok) == 1
    assert res_hc_ok[0]["duration_s"] > 0.4

    tc_rows = [
        _row(
            f,
            fps,
            [
                _det(1, (200.0, 200.0), dx_nose=15.0, tail_dir=-1.0),
                _det(2, (165.0 if 20 <= f < 32 else 650.0, 200.0), dx_nose=-15.0, tail_dir=1.0),
            ],
        )
        for f in range(60)
    ]
    assert (
        segment_clipping_social_contact.refine_contact_and_together_segments(
            [{"event_id": "tc_exact_04", "canonical_behavior": "nose_tail_contact", "mouse_ids": [1, 2], "start_frame": 20, "end_frame": 32}],
            InMemoryTrackStore(tc_rows),
            {"source_video": {"fps": fps, "frame_count": 60}},
            scale=scale,
        )
        == []
    )
    tc_rows_ok = [
        _row(
            f,
            fps,
            [
                _det(1, (200.0, 200.0), dx_nose=15.0, tail_dir=-1.0),
                _det(2, (165.0 if 20 <= f < 33 else 650.0, 200.0), dx_nose=-15.0, tail_dir=1.0),
            ],
        )
        for f in range(60)
    ]
    res_tc_ok = segment_clipping_social_contact.refine_contact_and_together_segments(
        [{"event_id": "tc_over_04", "canonical_behavior": "nose_tail_contact", "mouse_ids": [1, 2], "start_frame": 20, "end_frame": 33}],
        InMemoryTrackStore(tc_rows_ok),
        {"source_video": {"fps": fps, "frame_count": 60}},
        scale=scale,
    )
    assert len(res_tc_ok) == 1
    assert res_tc_ok[0]["duration_s"] > 0.4

    # 4. Together < 1.0 s (20 frames = 0.67 s) -> rejected
    tog_rows = [
        _row(
            f,
            fps,
            [
                _det(1, (200.0, 200.0)),
                _det(2, (235.0 if 20 <= f < 40 else 650.0, 200.0)),
            ],
        )
        for f in range(80)
    ]
    assert (
        segment_clipping_social_contact.refine_contact_and_together_segments(
            [{"event_id": "tog_short", "canonical_behavior": "together", "mouse_ids": [1, 2], "start_frame": 20, "end_frame": 40}],
            InMemoryTrackStore(tog_rows),
            {"source_video": {"fps": fps, "frame_count": 80}},
            scale=scale,
        )
        == []
    )

    # 5. Approach <= 0.3 s (9 frames = 0.30 s, strict '>' requires >= 10 frames at 30 fps) -> rejected
    app_short_rows = []
    for f in range(60):
        x1 = 100.0 + (f - 20) * 8.0 if 20 <= f < 29 else (50.0 if f < 20 else 700.0)
        app_short_rows.append(_row(f, fps, [_det(1, (x1, 200.0)), _det(2, (260.0, 200.0))]))
    refined_app, app_audit = segment_clipping_social_approach.refine_approach_segments(
        [{"event_id": "app_short", "canonical_behavior": "approach", "actor_id": 1, "target_id": 2, "mouse_ids": [1, 2], "start_frame": 20, "end_frame": 29}],
        InMemoryTrackStore(app_short_rows),
        {"source_video": {"fps": fps, "frame_count": 60}},
        scale=scale,
        requalify=False,
    )
    assert refined_app == []
    assert app_audit[0]["reason"] == "rejected_below_min_duration"
    assert isinstance(store, InMemoryTrackStore) and binding["source_video"]["fps"] == 30.0


def test_social_clipping_completes_truncated_bouts_and_observed_endpoints() -> None:
    fps = 10.0
    scale = {"cm_per_pixel": 0.1, "reference_body_cm": 7.918}
    binding = {"source_video": {"fps": fps, "frame_count": 80}}

    # Snout-head contact actually runs [15, 45), candidate truncated to [22, 35) -> must recover full [15, 45)
    hc_rows = []
    for f in range(80):
        c2x = 235.0 if 15 <= f < 45 else 600.0
        hc_rows.append(_row(f, fps, [_det(1, (200.0, 200.0), dx_nose=15.0), _det(2, (c2x, 200.0), dx_nose=-15.0)]))
    hc_store = InMemoryTrackStore(hc_rows)
    res_hc = segment_clipping_social_contact.refine_contact_and_together_segments(
        [{"event_id": "hc_trunc", "canonical_behavior": "nose_head_contact", "mouse_ids": [1, 2], "start_frame": 22, "end_frame": 35}],
        hc_store,
        binding,
        scale=scale,
    )
    assert len(res_hc) == 1
    # Contact (directed Snout_contacting->Head_contacted) & Together (undirected) role outputs
    assert res_hc[0]["role_status"] == "measured_directed_role"
    assert res_hc[0]["actor_id"] in (1, 2) and res_hc[0]["target_id"] in (1, 2)
    assert res_hc[0]["roles"]["initiator_role"] == "Snout_contacting"
    assert res_hc[0]["roles"]["target_role"] == "Head_contacted"
    assert [p["role_name"] for p in res_hc[0]["participants"]] == ["Snout_contacting", "Head_contacted"]

    assert (res_hc[0]["start_frame"], res_hc[0]["end_frame"]) == (15, 45)
    assert endpoints_fully_observed(hc_store, res_hc[0], binding["source_video"])

    # Together runs [10, 40), candidate truncated to [18, 30) with missing mouse 2 at 8..9 -> recovers [10, 40)
    tog_rows = []
    for f in range(80):
        dets = [_det(1, (200.0, 200.0))]
        if 10 <= f < 40:
            dets.append(_det(2, (235.0, 200.0)))
        elif f >= 40:
            dets.append(_det(2, (600.0, 200.0)))
        tog_rows.append(_row(f, fps, dets))
    tog_store = InMemoryTrackStore(tog_rows)
    res_tog = segment_clipping_social_contact.refine_contact_and_together_segments(
        [{"event_id": "tog_trunc", "canonical_behavior": "together", "mouse_ids": [1, 2], "start_frame": 18, "end_frame": 30}],
        tog_store,
        binding,
        scale=scale,
    )
    assert len(res_tog) == 1
    assert (res_tog[0]["start_frame"], res_tog[0]["end_frame"]) == (10, 40)
    assert res_tog[0]["role_status"] == "undirected_not_applicable"
    assert res_tog[0]["actor_id"] == -1
    assert res_tog[0]["target_id"] == -1
    assert res_tog[0]["roles"] is None
    assert res_tog[0]["participants"] == []
    assert endpoints_fully_observed(tog_store, res_tog[0], binding["source_video"])

    # Approach closing from frame 10..26 (d_start=21.0cm), boxes just touch at f=26 (end_bg<=0.15cm) -> ends cleanly at f=27
    app_rows = []
    for f in range(80):
        if 10 <= f <= 26:
            x1 = 100.0 + (f - 10) * 10.3  # f=26 -> x1=264.8 (right=286.8 vs x2 left=288.0 -> gap=0.12cm)
        elif f < 10:
            x1 = 100.0
        else:
            x1 = 278.0
        app_rows.append(_row(f, fps, [_det(1, (x1, 200.0)), _det(2, (310.0, 200.0))]))
    app_store = InMemoryTrackStore(app_rows)
    res_app, _ = segment_clipping.refine_behavior_segments(
        [{"event_id": "app_trunc", "canonical_behavior": "approach", "actor_id": 1, "target_id": 2, "mouse_ids": [1, 2], "start_frame": 16, "end_frame": 22}],
        app_store,
        binding,
        scale=scale,
        enable_huddle_sliding=False,
        enable_approach_requalification=False,
        enable_fine_cropping=True,
    )
    assert len(res_app) == 1
    assert res_app[0]["start_frame"] <= 16
    assert res_app[0]["end_frame"] == 27
    assert res_app[0]["duration_s"] >= 1.0
    assert res_app[0]["approach_end_box_gap_cm"] <= 0.15
    assert res_app[0]["min_pair_box_gap_cm"] > 0.0
    assert res_app[0]["role_status"] == "measured_directed_role"
    assert res_app[0]["roles"]["initiator_role"] == "Approaching"
    assert res_app[0]["roles"]["target_role"] == "Approached"
    assert [p["role_name"] for p in res_app[0]["participants"]] == ["Approaching", "Approached"]
    assert endpoints_fully_observed(app_store, res_app[0], binding["source_video"])

    # select_behavior_segments guarantees zero shared-mouse overlaps on overlapping refined social clips
    overlapping_pair = [
        {**res_hc[0], "event_id": "hc_a"},
        {**res_hc[0], "event_id": "hc_b", "mouse_ids": [2, 3], "start_frame": 20, "end_frame": 40, "canonical_support_frame_count": 10},
    ]
    hc_rows_3 = [
        _row(r["frame"], fps, r["detections"] + [_det(3, (240.0, 200.0), dx_nose=-15.0)])
        for r in hc_rows
    ]
    selected, _ = segment_clipping.select_behavior_segments(
        overlapping_pair,
        store=InMemoryTrackStore(hc_rows_3),
        binding=binding,
        strict_non_overlapping_shared_mouse=True,
    )
    assert len(selected) == 1
    assert selected[0]["event_id"] == "hc_a"

    # Negative actor_id/target_id sentinel (-1) on Attack must resolve mouse_ids cleanly,
    # reject stationary sitting and crowded corner huddles, and accept true >0.5s clinch+occlusion fight.
    sentinel_attack = {
        "event_id": "att_sentinel",
        "canonical_behavior": "attack",
        "actor_id": -1,
        "target_id": -1,
        "mouse_ids": [1, 2],
        "start_frame": 15,
        "end_frame": 25,
    }
    assert (
        segment_clipping_social_attack_chasing.refine_attack_segments(
            [sentinel_attack], hc_store, binding, scale=scale
        )
        == []
    )
    # Use 30 fps for >0.5s fight: 20 frames (f=10..29) with lunge, heavy box overlap + 4 clinch dropout frames, and breakout
    fps30 = 30.0
    attack_dyn_rows = []
    for f in range(60):
        if f < 10:
            x1, x2 = 160.0, 230.0
        elif f < 16:
            x1 = 160.0 + (f - 10) * 14.0
            x2 = 230.0 + (f - 10) * 4.0
        elif f < 28:
            # Close active tumbling clinch (gap <= 15px inside 44px box -> IoU >= 0.48), mid displacement > 12 cm
            x1 = 244.0 + (f - 16) * 11.0
            x2 = 252.0 + (f - 16) * 10.5
        elif f < 36:
            # Explosive recoil/pursuit
            x1 = 376.0 + (f - 28) * 12.0
            x2 = 378.0 + (f - 28) * 18.0
        else:
            x1, x2 = 472.0, 522.0
        dets = [_det(1, (x1, 200.0))]
        if f not in (20, 21, 22, 23):
            dets.append(_det(2, (x2, 200.0)))
        attack_dyn_rows.append(_row(f, fps30, dets))
    res_sentinel = segment_clipping_social_attack_chasing.refine_attack_segments(
        [{"event_id": "att_sentinel_30", "canonical_behavior": "attack", "actor_id": -1, "target_id": -1, "mouse_ids": [1, 2], "start_frame": 14, "end_frame": 28}],
        InMemoryTrackStore(attack_dyn_rows),
        {"source_video": {"fps": fps30, "frame_count": 60}},
        scale=scale,
    )
    assert len(res_sentinel) == 1
    assert set(res_sentinel[0]["mouse_ids"]) == {1, 2}
    assert res_sentinel[0]["duration_s"] > 0.5
    assert res_sentinel[0]["attack_max_iou"] >= 0.42
    assert res_sentinel[0]["attack_clinch_occlusion_frames"] >= 3
    assert res_sentinel[0]["role_status"] == "measured_directed_role"
    assert res_sentinel[0]["roles"]["initiator_role"] == "Attacker"
    assert res_sentinel[0]["roles"]["target_role"] == "Attacked"
    assert [p["role_name"] for p in res_sentinel[0]["participants"]] == ["Attacker", "Attacked"]

    # Crowded 3rd mouse right next to the clinch (>25% within 11 cm) -> rejected as corner huddle
    crowded_att_rows = [
        _row(r["frame"], fps30, r["detections"] + [_det(3, (275.0, 220.0))])
        for r in attack_dyn_rows
    ]
    assert (
        segment_clipping_social_attack_chasing.refine_attack_segments(
            [{"event_id": "att_crowd", "canonical_behavior": "attack", "mouse_ids": [1, 2], "start_frame": 14, "end_frame": 28}],
            InMemoryTrackStore(crowded_att_rows),
            {"source_video": {"fps": fps30, "frame_count": 60}},
            scale=scale,
        )
        == []
    )
    assert isinstance(np.zeros(1), np.ndarray)


def test_approach_zero_contact_outside_crowd_and_fast_chase() -> None:
    fps = 10.0
    scale = {"cm_per_pixel": 0.1, "reference_body_cm": 7.918}
    binding = {"source_video": {"fps": fps, "frame_count": 80}}

    # 1. Approach closes frames 10..25 from 22.4 cm without contact, then boxes just touch at f=25 (gap=0.10cm) -> stops at end_frame=26
    app_contact_rows = []
    for f in range(80):
        if f < 10:
            x1 = 80.0
        elif f <= 25:
            x1 = 86.0 + (f - 10) * 11.9333  # f=10 -> 86 (d=22.4cm), f=25 -> 265 (right=287 vs x2_left=288 -> gap=0.10cm)
        else:
            x1 = 278.0
        app_contact_rows.append(_row(f, fps, [_det(1, (x1, 200.0)), _det(2, (310.0, 200.0))]))
    res_app, _ = segment_clipping_social_approach.refine_approach_segments(
        [{"event_id": "app_with_contact", "canonical_behavior": "approach", "actor_id": 1, "target_id": 2, "mouse_ids": [1, 2], "start_frame": 12, "end_frame": 34}],
        InMemoryTrackStore(app_contact_rows),
        binding,
        scale=scale,
        requalify=False,
    )
    assert len(res_app) == 1
    assert res_app[0]["end_frame"] == 26
    assert res_app[0]["duration_s"] >= 1.0
    assert res_app[0]["approach_end_box_gap_cm"] <= 0.15
    assert res_app[0]["min_pair_box_gap_cm"] > 0.0
    assert res_app[0]["approach_start_dist_cm"] >= 20.0
    assert res_app[0]["approach_closing_cm"] >= 10.0
    assert res_app[0]["roles"]["summary_zh"] == "V1(主动接近者) → V2(被接近者)"

    # Target approaching a stationary actor -> dynamic orientation switches actor to active mouse 1,
    # unless stationary orientation forced, or if neither approaches -> rejected.
    still_rows = [
        _row(f, fps, [_det(1, (100.0, 200.0)), _det(2, (310.0, 200.0))])
        for f in range(40)
    ]
    res_still, still_audit = segment_clipping_social_approach.refine_approach_segments(
        [{"event_id": "app_still", "canonical_behavior": "approach", "actor_id": 2, "target_id": 1, "mouse_ids": [2, 1], "start_frame": 10, "end_frame": 25}],
        InMemoryTrackStore(still_rows),
        binding,
        scale=scale,
        requalify=False,
    )
    assert res_still == []
    assert still_audit[0]["reason"] in {"rejected_below_min_duration", "rejected_contact_or_crowd", "rejected_insufficient_closing"}

    # 2. Same closing approach inside a crowd (3rd mouse within 6cm of target) -> strictly rejected
    app_crowd_rows = [
        _row(r["frame"], fps, r["detections"] + [_det(3, (310.0, 260.0))])
        for r in app_contact_rows
    ]
    res_crowd, crowd_audit = segment_clipping_social_approach.refine_approach_segments(
        [{"event_id": "app_crowded", "canonical_behavior": "approach", "actor_id": 1, "target_id": 2, "mouse_ids": [1, 2], "start_frame": 12, "end_frame": 26}],
        InMemoryTrackStore(app_crowd_rows),
        binding,
        scale=scale,
        requalify=False,
    )
    assert res_crowd == []
    assert crowd_audit[0]["reason"] == "rejected_contact_or_crowd"

    # 3. Attack rejects 1-frame teleport artifact on motionless mice AND constant-velocity pass-by
    jump_rows = [
        _row(
            f,
            fps,
            [
                _det(1, (200.0 + (150.0 if f == 20 else 0.0), 200.0)),
                _det(2, (235.0, 200.0)),
            ],
        )
        for f in range(40)
    ]
    assert (
        segment_clipping_social_attack_chasing.refine_attack_segments(
            [{"event_id": "att_jump", "canonical_behavior": "attack", "mouse_ids": [1, 2], "start_frame": 15, "end_frame": 25}],
            InMemoryTrackStore(jump_rows),
            {"source_video": {"fps": fps, "frame_count": 40}},
            scale=scale,
        )
        == []
    )
    passby_rows = [
        _row(
            f,
            fps,
            [
                _det(1, (50.0 + f * 45.0, 200.0)),
                _det(2, (950.0 - f * 45.0, 200.0)),
            ],
        )
        for f in range(30)
    ]
    assert (
        segment_clipping_social_attack_chasing.refine_attack_segments(
            [{"event_id": "att_passby", "canonical_behavior": "attack", "mouse_ids": [1, 2], "start_frame": 9, "end_frame": 15}],
            InMemoryTrackStore(passby_rows),
            {"source_video": {"fps": fps, "frame_count": 30}},
            scale=scale,
        )
        == []
    )

    # 4. Chasing rejects slow wandering (20 cm/s), wide separation (30 cm), widening gap (>4.5 cm),
    # and intervening 3rd mouse; accepts >0.7s fast trail-locked pursuit + discovers missed pursuit.
    slow_chase_rows = [
        _row(f, fps, [_det(1, (100.0 + f * 20.0, 200.0)), _det(2, (160.0 + f * 20.0, 200.0))])
        for f in range(40)
    ]
    assert (
        segment_clipping_social_attack_chasing.refine_chasing_segments(
            [{"event_id": "chase_slow", "canonical_behavior": "chase", "mouse_ids": [1, 2], "start_frame": 5, "end_frame": 35}],
            InMemoryTrackStore(slow_chase_rows),
            {"source_video": {"fps": fps, "frame_count": 40}},
            scale=scale,
        )
        == []
    )
    wide_chase_rows = [
        _row(f, fps, [_det(1, (100.0 + f * 45.0, 200.0)), _det(2, (400.0 + f * 45.0, 200.0))])
        for f in range(40)
    ]
    assert (
        segment_clipping_social_attack_chasing.refine_chasing_segments(
            [{"event_id": "chase_wide", "canonical_behavior": "chase", "mouse_ids": [1, 2], "start_frame": 5, "end_frame": 35}],
            InMemoryTrackStore(wide_chase_rows),
            {"source_video": {"fps": fps, "frame_count": 40}},
            scale=scale,
        )
        == []
    )
    fast_chase_rows = [
        _row(f, fps, [_det(1, (100.0 + f * 45.0, 200.0)), _det(2, (180.0 + f * 44.0, 200.0))])
        for f in range(40)
    ]
    res_fast = segment_clipping_social_attack_chasing.refine_chasing_segments(
        [{"event_id": "chase_fast", "canonical_behavior": "chase", "mouse_ids": [1, 2], "start_frame": 5, "end_frame": 35}],
        InMemoryTrackStore(fast_chase_rows),
        {"source_video": {"fps": fps, "frame_count": 40}},
        scale=scale,
    )
    assert len(res_fast) == 1
    assert res_fast[0]["chase_mean_follower_speed_cm_s"] >= 33.0
    assert res_fast[0]["chase_median_dist_cm"] <= 15.0
    assert res_fast[0]["chase_trail_error_cm"] <= 3.2
    assert res_fast[0]["role_status"] == "measured_directed_role"
    assert res_fast[0]["roles"]["initiator_role"] == "Chasing"
    assert res_fast[0]["roles"]["target_role"] == "Chased"
    assert [p["role_name"] for p in res_fast[0]["participants"]] == ["Chasing", "Chased"]
    assert res_fast[0]["roles"]["summary_zh"] == "V1(追逐者) → V2(被追逐者)"

    # Intervening 3rd mouse right on the segment between chaser and chased -> rejected
    intervening_rows = [
        _row(
            f,
            fps,
            [
                _det(1, (100.0 + f * 45.0, 200.0)),
                _det(2, (200.0 + f * 44.0, 200.0)),
                _det(3, (150.0 + f * 44.5, 205.0)),
            ],
        )
        for f in range(40)
    ]
    assert (
        segment_clipping_social_attack_chasing.refine_chasing_segments(
            [{"event_id": "chase_inter", "canonical_behavior": "chase", "mouse_ids": [1, 2], "start_frame": 5, "end_frame": 35}],
            InMemoryTrackStore(intervening_rows),
            {"source_video": {"fps": fps, "frame_count": 40}},
            scale=scale,
        )
        == []
    )

    # discover_fast_chasing_candidates recovers 0.8s (24 frames @ 30fps) fast pursuit with "chase" & roles
    fps30 = 30.0
    disc_rows = [
        _row(
            f,
            fps30,
            [
                _det(7, (100.0 + f * 15.0, 200.0)),
                _det(8, (180.0 + f * 15.0, 200.0)),
            ],
        )
        for f in range(25)
    ]
    disc = segment_clipping_social_attack_chasing.discover_fast_chasing_candidates(
        InMemoryTrackStore(disc_rows),
        {"source_video": {"fps": fps30, "frame_count": 25}},
        scale=scale,
    )
    assert len(disc) == 1
    assert disc[0]["canonical_behavior"] == "chase"
    assert disc[0]["duration_s"] > 0.7
    assert disc[0]["chase_min_duration_policy_s"] == 0.7
    assert disc[0]["roles"]["summary_zh"] == "V7(追逐者) → V8(被追逐者)"


def _det_angled_head(
    track_id: int,
    center: tuple[float, float],
    heading_rad: float,
    *,
    box_w: float = 44.0,
    box_h: float = 24.0,
) -> dict:
    cx, cy = float(center[0]), float(center[1])
    ux, uy = float(np.cos(heading_rad)), float(np.sin(heading_rad))
    px, py = -uy, ux
    nose = [cx + 18.0 * ux, cy + 18.0 * uy, 0.9]
    lear = [cx + 7.0 * ux + 6.0 * px, cy + 7.0 * uy + 6.0 * py, 0.9]
    rear = [cx + 7.0 * ux - 6.0 * px, cy + 7.0 * uy - 6.0 * py, 0.9]
    neck = [cx, cy, 0.9]
    tail = [cx - 22.0 * ux, cy - 22.0 * uy, 0.9]
    return {
        "track_id": int(track_id),
        "box": [cx - box_w * 0.5, cy - box_h * 0.5, cx + box_w * 0.5, cy + box_h * 0.5],
        "keypoints": [
            nose,
            lear,
            rear,
            neck,
            [cx - 8.0 * ux + 4.0 * px, cy - 8.0 * uy + 4.0 * py, 0.9],
            [cx - 8.0 * ux - 4.0 * px, cy - 8.0 * uy - 4.0 * py, 0.9],
            tail,
        ],
    }


def test_approach_final_directionality_chasing_bystander_and_id_switch_discovery() -> None:
    fps = 30.0
    scale = {"cm_per_pixel": 0.1, "reference_body_cm": 7.918}
    binding = {"source_video": {"fps": fps, "frame_count": 120}}

    # 1. Oblique walk-past Approach (70 deg offset heading & trajectory) must be rejected cleanly
    oblique_rows = []
    for f in range(90):
        # Target sits at (400, 200); actor moves diagonally upward-right past target so closing >=10cm but end cos < 0.82
        if f < 10:
            ax, ay = 130.0, 200.0
        elif f <= 48:
            t = (f - 10) / 38.0
            ax = 130.0 + t * 238.0
            ay = 200.0 + t * 140.0 if f < 25 else 200.0 + 55.0 + (f - 25) * (-1.5)
        else:
            ax, ay = 368.0, 20.5
        oblique_rows.append(
            _row(
                f,
                fps,
                [
                    _det_angled_head(1, (ax, ay), heading_rad=float(np.deg2rad(-58.0))),
                    _det_angled_head(2, (400.0, 200.0), heading_rad=float(np.pi)),
                ],
            )
        )
    # Force an oblique pre-touch trajectory where velocity in last 12f is 50 deg off target vector
    oblique_walk_rows = []
    for f in range(80):
        # Target stationary at (420, 200), box [398, 188, 442, 212]. Actor arrives near (375, 175) moving sharply upward-right (-45 deg).
        if f <= 45:
            ax = 120.0 + f * 5.7
            ay = 320.0 - f * 3.18  # f=45 -> (376.5, 176.9), dx=21.5px vs x2_left=398 -> gap 0.05cm, cos~0.75
        else:
            ax, ay = 382.0, 175.0
        oblique_walk_rows.append(
            _row(
                f,
                fps,
                [
                    _det_angled_head(1, (ax, ay), heading_rad=float(np.deg2rad(-55.0))),
                    _det_angled_head(2, (420.0, 200.0), heading_rad=float(np.pi)),
                ],
            )
        )
    res_oblique, aud_oblique = segment_clipping_social_approach.refine_approach_segments(
        [{"event_id": "app_oblique", "canonical_behavior": "approach", "actor_id": 1, "target_id": 2, "mouse_ids": [1, 2], "start_frame": 20, "end_frame": 46}],
        InMemoryTrackStore(oblique_walk_rows),
        {"source_video": {"fps": fps, "frame_count": 80}},
        scale=scale,
        requalify=False,
    )
    assert res_oblique == []
    assert aud_oblique[0]["reason"] in {"rejected_weak_final_directionality", "rejected_contact_or_crowd", "rejected_insufficient_closing"}

    # Crisp direct pointing approach -> accepted with explicit final-phase directionality metrics
    direct_rows = []
    for f in range(80):
        ax = 120.0 + f * 6.0 if f <= 42 else 378.0  # f=42 -> 372, right=394 vs left=398? Wait: f=42 -> 375.0 -> right=397 (gap=0.1cm)
        if f <= 42:
            ax = 123.0 + f * 6.0
        else:
            ax = 382.0
        direct_rows.append(
            _row(
                f,
                fps,
                [
                    _det_angled_head(1, (ax, 200.0), heading_rad=0.0),
                    _det_angled_head(2, (420.0, 200.0), heading_rad=float(np.pi)),
                ],
            )
        )
    res_direct, _ = segment_clipping_social_approach.refine_approach_segments(
        [{"event_id": "app_direct", "canonical_behavior": "approach", "actor_id": 1, "target_id": 2, "mouse_ids": [1, 2], "start_frame": 15, "end_frame": 43}],
        InMemoryTrackStore(direct_rows),
        {"source_video": {"fps": fps, "frame_count": 80}},
        scale=scale,
        requalify=False,
    )
    assert len(res_direct) == 1
    assert res_direct[0]["approach_end_toward_cos"] >= 0.90
    assert res_direct[0]["approach_end_heading_cos"] >= 0.90
    assert res_direct[0]["approach_end_actor_dominance_cm_s"] >= 10.0

    # 2. 3-mouse wall train (bystander trailing directly behind chaser throughout) -> rejected
    train_rows = [
        _row(
            f,
            fps,
            [
                _det(1, (200.0 + f * 15.0, 200.0)),
                _det(2, (280.0 + f * 15.0, 200.0)),
                _det(3, (125.0 + f * 15.0, 200.0)),
            ],
        )
        for f in range(40)
    ]
    assert (
        segment_clipping_social_attack_chasing.refine_chasing_segments(
            [{"event_id": "chase_train", "canonical_behavior": "chase", "mouse_ids": [1, 2], "start_frame": 5, "end_frame": 35}],
            InMemoryTrackStore(train_rows),
            {"source_video": {"fps": fps, "frame_count": 40}},
            scale=scale,
        )
        == []
    )

    # 3. Pre-birth adjacent-only Attack loophole (miss1_inside == 0) must be rejected
    no_inside_occ_rows = []
    for f in range(60):
        x1 = 160.0 + f * 10.0 if 10 <= f <= 30 else 160.0
        x2 = 175.0 + f * 10.0 if 15 <= f <= 30 else 650.0  # newborn ID 2 appears at f=15
        dets = [_det(1, (x1, 200.0))]
        if 15 <= f <= 35:
            dets.append(_det(2, (x2, 200.0)))
        no_inside_occ_rows.append(_row(f, fps, dets))
    assert (
        segment_clipping_social_attack_chasing.refine_attack_segments(
            [{"event_id": "att_newborn_fp", "canonical_behavior": "attack", "mouse_ids": [1, 2], "start_frame": 16, "end_frame": 34}],
            InMemoryTrackStore(no_inside_occ_rows),
            {"source_video": {"fps": fps, "frame_count": 60}},
            scale=scale,
        )
        == []
    )

    # 4. ID-switched Attack fight ball discovery -> recovers fight with raw IDs preserved + phase endpoints
    fight_rows = []
    for f in range(90):
        # Baseline frames for median box area estimation (~44x24 = 1056 px^2)
        if f < 25:
            dets = [_det_angled_head(1, (100.0 + f * 4.0, 200.0), 0.0), _det_angled_head(2, (260.0 - f * 2.0, 200.0), float(np.pi))]
        elif f < 32:
            # Lunging convergence at >55 cm/s (18 px/f = 54 cm/s)
            x1 = 200.0 + (f - 25) * 18.0
            x2 = 210.0 + (f - 25) * 5.0
            dets = [_det_angled_head(1, (x1, 200.0), 0.0), _det_angled_head(2, (x2, 200.0), float(np.pi))]
        elif f < 44:
            # Merged huge tumbling box (3.2x area, ~50 cm/s tumbling) with track 1 & 2 merged into track 3!
            cx = 326.0 + (f - 32) * 16.0
            cy = 200.0 + (15.0 if f % 2 == 0 else -15.0)
            dets = [_det_angled_head(3, (cx, cy), 0.2, box_w=84.0, box_h=42.0)]
        elif f < 58:
            # Post-clinch breakout with switched new track IDs 4 & 5
            x4 = 518.0 + (f - 44) * 18.0
            x5 = 500.0 + (f - 44) * 9.0
            dets = [_det_angled_head(4, (x4, 200.0), 0.0), _det_angled_head(5, (x5, 210.0), 0.0)]
        else:
            dets = [_det_angled_head(4, (750.0, 200.0), 0.0), _det_angled_head(5, (630.0, 210.0), 0.0)]
        fight_rows.append(_row(f, fps, dets))
    fb_store = InMemoryTrackStore(fight_rows)
    fb_cands = segment_clipping_social_attack_chasing.discover_fight_ball_attack_episodes(
        fb_store, {"source_video": {"fps": fps, "frame_count": 90}}, scale=scale
    )
    assert len(fb_cands) == 1
    assert fb_cands[0]["has_visual_id_switch"] is True
    assert fb_cands[0]["duration_s"] > 0.5
    assert set(fb_cands[0]["mouse_ids"]) >= {1, 2, 3, 4}
    assert fb_cands[0]["roles"]["initiator_role"] == "Attacker"
    assert fb_cands[0]["roles"]["target_role"] == "Attacked"
    assert endpoints_fully_observed(fb_store, fb_cands[0], {"fps": fps, "frame_count": 90})

    # 5. ID-switched fast Chasing discovery -> recovers pursuit spanning track ID change
    idsw_chase_rows = []
    for f in range(45):
        dets = [_det(10, (80.0 + f * 15.0, 200.0))] if f < 22 else [_det(11, (80.0 + f * 15.0, 200.0))]
        dets.append(_det(20, (160.0 + f * 15.0, 200.0)) if f < 30 else _det(21, (160.0 + f * 15.0, 200.0)))
        idsw_chase_rows.append(_row(f, fps, dets))
    idsw_store = InMemoryTrackStore(idsw_chase_rows)
    idsw_chases = segment_clipping_social_attack_chasing.discover_id_switch_chasing_candidates(
        idsw_store, {"source_video": {"fps": fps, "frame_count": 45}}, scale=scale
    )
    assert len(idsw_chases) == 1
    assert idsw_chases[0]["has_visual_id_switch"] is True
    assert idsw_chases[0]["duration_s"] > 0.7
    assert qualify_event_duration(
        idsw_chases[0], fps,
    ) or idsw_chases[0]["chase_min_duration_policy_s"] == 0.7
    assert idsw_chases[0]["actor_id"] == 10
    assert idsw_chases[0]["target_id"] == 20
    assert idsw_chases[0]["roles"]["initiator_role"] == "Chasing"
    assert idsw_chases[0]["roles"]["target_role"] == "Chased"
    assert endpoints_fully_observed(idsw_store, idsw_chases[0], {"fps": fps, "frame_count": 45})
    assert isinstance(binding, dict)


def test_chasing_speed_and_directional_consistency_gates() -> None:
    fps = 30.0
    scale = {"cm_per_pixel": 0.1, "reference_body_cm": 7.918}

    # 1. Slow-moderate trotting (~29 cm/s) -> REJECTED
    moderate_rows = [
        _row(
            f,
            fps,
            [
                _det_angled_head(1, (100.0 + f * 9.6, 200.0), 0.0),
                _det_angled_head(2, (180.0 + f * 9.6, 200.0), 0.0),
            ],
        )
        for f in range(45)
    ]
    assert (
        segment_clipping_social_attack_chasing.refine_chasing_segments(
            [{"event_id": "chase_mod", "canonical_behavior": "chase", "mouse_ids": [1, 2], "start_frame": 5, "end_frame": 40}],
            InMemoryTrackStore(moderate_rows),
            {"source_video": {"fps": fps, "frame_count": 45}},
            scale=scale,
        )
        == []
    )

    # 2. Fast (~43 cm/s) but crisscrossing/misaligned first half + short (<0.7s) aligned tail -> REJECTED
    misalign_rows = []
    for f in range(36):
        if f < 10:
            y1 = 150.0 + f * 10.0
            y2 = 250.0 - f * 10.0
            x1, x2 = 100.0 + f * 10.0, 180.0 + f * 10.0
        elif f < 20:
            k = f - 10
            y1 = 250.0 - k * 10.0
            y2 = 150.0 + k * 10.0
            x1, x2 = 200.0 + k * 10.0, 280.0 + k * 10.0
        else:
            k = f - 20
            x1, y1 = 300.0 + k * 15.0, 200.0
            x2, y2 = 380.0 + k * 15.0, 200.0
        misalign_rows.append(
            _row(
                f,
                fps,
                [
                    _det_angled_head(1, (x1, y1), 0.0),
                    _det_angled_head(2, (x2, y2), 0.0),
                ],
            )
        )
    assert (
        segment_clipping_social_attack_chasing.refine_chasing_segments(
            [{"event_id": "chase_misalign", "canonical_behavior": "chase", "mouse_ids": [1, 2], "start_frame": 2, "end_frame": 34}],
            InMemoryTrackStore(misalign_rows),
            {"source_video": {"fps": fps, "frame_count": 36}},
            scale=scale,
        )
        == []
    )

    # 3. Slow/oblique lead-in followed by fast aligned sprint -> lead-in trimmed out
    leadin_rows = []
    for f in range(60):
        if f < 25:
            x1 = 100.0 + f * 6.0
            y1 = 120.0 + f * 3.2
            h1 = float(np.arctan2(3.2, 6.0))
            x2 = 185.0 + f * 6.6
            y2 = 200.0
        else:
            k = f - 25
            x1, y1, h1 = 250.0 + k * 15.0, 200.0, 0.0
            x2, y2 = 350.0 + k * 14.8, 200.0
        leadin_rows.append(
            _row(
                f,
                fps,
                [
                    _det_angled_head(1, (x1, y1), h1),
                    _det_angled_head(2, (x2, y2), 0.0),
                ],
            )
        )
    res_trimmed = segment_clipping_social_attack_chasing.refine_chasing_segments(
        [{"event_id": "chase_leadin", "canonical_behavior": "chase", "mouse_ids": [1, 2], "start_frame": 2, "end_frame": 58}],
        InMemoryTrackStore(leadin_rows),
        {"source_video": {"fps": fps, "frame_count": 60}},
        scale=scale,
    )
    assert len(res_trimmed) == 1
    assert res_trimmed[0]["start_frame"] >= 24
    assert res_trimmed[0]["end_frame"] >= 55
    assert res_trimmed[0]["chase_mean_vel_cos"] >= 0.90
    assert res_trimmed[0]["chase_both_p25_speed_cm_s"] >= 35.0
    assert res_trimmed[0]["chase_body_heading_cos"] >= 0.95

    # 4. Fast aligned sprint with 7-frame single-mouse mid-clip dropout -> RECOVERED
    dropout_rows = []
    for f in range(32):
        dets = [_det_angled_head(2, (180.0 + f * 15.5, 200.0), 0.0)]
        if not (12 <= f <= 18):
            dets.insert(0, _det_angled_head(1, (100.0 + f * 16.0, 200.0), 0.0))
        dropout_rows.append(_row(f, fps, dets))
    dropout_store = InMemoryTrackStore(dropout_rows)
    dropout_binding = {"source_video": {"fps": fps, "frame_count": 32}}
    res_dropout = segment_clipping_social_attack_chasing.refine_chasing_segments(
        [{"event_id": "chase_drop", "canonical_behavior": "chase", "mouse_ids": [1, 2], "start_frame": 0, "end_frame": 32}],
        dropout_store,
        dropout_binding,
        scale=scale,
    )
    assert len(res_dropout) == 1
    assert res_dropout[0]["start_frame"] == 0
    assert res_dropout[0]["end_frame"] == 32
    assert res_dropout[0]["roles"]["summary_zh"] == "V1(追逐者) → V2(被追逐者)"
    disc_dropout = segment_clipping_social_attack_chasing.discover_fast_chasing_candidates(
        dropout_store, dropout_binding, scale=scale
    )
    assert len(disc_dropout) == 1
    assert disc_dropout[0]["start_frame"] == 0
    assert disc_dropout[0]["end_frame"] == 32

    # 5. Fast centroid drift but body-heading opposite/perpendicular -> REJECTED
    bad_body_rows = [
        _row(
            f,
            fps,
            [
                _det_angled_head(1, (100.0 + f * 15.0, 200.0), float(np.pi)),
                _det_angled_head(2, (180.0 + f * 15.0, 200.0), 0.0),
            ],
        )
        for f in range(40)
    ]
    assert (
        segment_clipping_social_attack_chasing.refine_chasing_segments(
            [{"event_id": "chase_bad_body", "canonical_behavior": "chase", "mouse_ids": [1, 2], "start_frame": 5, "end_frame": 35}],
            InMemoryTrackStore(bad_body_rows),
            {"source_video": {"fps": fps, "frame_count": 40}},
            scale=scale,
        )
        == []
    )


def test_canonical_social_role_names_and_participants() -> None:
    expected = {
        "approach": ("Approaching", "Approached"),
        "attack": ("Attacker", "Attacked"),
        "chase": ("Chasing", "Chased"),
        "chasing": ("Chasing", "Chased"),
        "nose_head_contact": ("Snout_contacting", "Head_contacted"),
        "head_contact": ("Snout_contacting", "Head_contacted"),
        "nose_tail_contact": ("Snout_contacting", "Rear_contacted"),
        "tail_contact": ("Snout_contacting", "Rear_contacted"),
    }
    for beh, (init_role, tgt_role) in expected.items():
        built = segment_clipping_social_contact.build_social_roles(beh, 3, 7)
        assert built is not None
        assert built["initiator_role"] == init_role
        assert built["target_role"] == tgt_role

        ev = {"canonical_behavior": beh, "start_frame": 10, "end_frame": 40}
        segment_clipping_social_contact.attach_social_roles(ev, beh, 3, 7, [[10, 40]])
        assert ev["actor_id"] == 3
        assert ev["target_id"] == 7
        assert ev["role_status"] == "measured_directed_role"
        assert ev["roles"]["initiator_role"] == init_role
        assert ev["roles"]["target_role"] == tgt_role
        assert ev["participants"] == [
            {"role_name": init_role, "role_key": init_role.lower(), "track_ids": [3]},
            {"role_name": tgt_role, "role_key": tgt_role.lower(), "track_ids": [7]},
        ]

    assert segment_clipping_social_contact.build_social_roles("together", 3, 7) is None
    tog_ev = {"canonical_behavior": "together", "start_frame": 10, "end_frame": 40}
    segment_clipping_social_contact.attach_social_roles(tog_ev, "together", 3, 7)
    assert tog_ev["actor_id"] == -1
    assert tog_ev["target_id"] == -1
    assert tog_ev["role_status"] == "undirected_not_applicable"
    assert tog_ev["roles"] is None
    assert tog_ev["participants"] == []
