"""Behavior evidence tests for the experimental user-requested motion profile."""

from copy import deepcopy
import numpy as np
import pytest
from mouse_behavior.behavior.posture_motion_candidates import (
    _speed,
    motion_windows,
    grooming_window_mask,
    rearing_window_mask,
    window_candidate_mask,
)

G = dict(
    coverage=0.7,
    max_rear_speed=0.2,
    min_head_speed=0.1,
    min_amplitude=0.01,
    max_amplitude=0.2,
    min_reversals_s=2,
    min_coherence=0.5,
    min_back_fraction=0.2,
    max_triangle_area=0.02,
    max_triangle_ratio=0.75,
    front_enabled=True,
)
R = dict(
    coverage=0.7,
    max_rear_speed=0.2,
    min_rise_speed=0.3,
    min_forebody_ratio=2,
    min_neck_change=0.02,
    max_hold_speed=0.2,
    max_hold_to_rise_ratio=0.5,
    held_enabled=False,
    max_hold_radius=0.3,
    min_hold_alignment=0.7,
)


def clip(n=90):
    pose = np.array([[10, 0], [8, -1], [8, 1], [6, 0], [2, -1], [2, 1], [0, 0]], float)
    return dict(
        points=np.tile(pose, (n, 1, 1)),
        boxes=np.tile([0, -2, 11, 2], (n, 1)),
        observed=np.ones(n, bool),
        fps=30.0,
        track_id=7,
    )


def evidence(**updates):
    w = dict(
        coverage=1.0,
        rear_speed=0.01,
        relative_head_speed=0.4,
        body_relative_amplitude=0.08,
        reversals_s=4.0,
        lateral_reversals_s=4.0,
        coherence=0.8,
        back_fraction=0.0,
        triangle_coverage=1.0,
        head_triangle_area=0.03,
        head_triangle_ratio=1.0,
        before_forebody_speed=0.8,
        after_forebody_speed=0.05,
        after_coverage=1.0,
        after_window_complete=True,
        forebody_to_rear_speed=8.0,
        before_neck_radius_change=0.1,
        neck_rear_radius=0.5,
        spine_cosine=0.9,
        head_speed=0.1,
        neck_speed=0.1,
    )
    return dict(w, **updates)


def test_oscillating_head_passes_but_rigid_translation_does_not():
    c = clip()
    t = np.arange(90) / 30
    c["points"][:, :3, 1] += (0.4 * np.sin(2 * np.pi * 4 * t))[:, None]
    ws, _ = motion_windows(c, body_length_px=10)
    assert any(grooming_window_mask(w, G) for w in ws)
    rigid = clip()
    rigid["points"] += np.column_stack([t * 5, t * 0])[:, None, :]
    ws, _ = motion_windows(rigid, body_length_px=10)
    assert not any(grooming_window_mask(w, G) for w in ws)


def test_fast_head_neck_then_settle_is_rearing_evidence():
    c = clip()
    displacement = np.clip((np.arange(90) - 20) / 10, 0, 1) * -2
    c["points"][:, :4, 0] += displacement[:, None]
    ws, _ = motion_windows(c, body_length_px=10)
    assert any(rearing_window_mask(w, R) for w in ws)


def test_head_speed_alone_and_unsettled_movement_are_not_rearing():
    assert not rearing_window_mask(evidence(after_forebody_speed=0.7), dict(R, max_hold_speed=1.0))
    assert not rearing_window_mask(evidence(after_window_complete=False), R)
    assert not rearing_window_mask(evidence(before_forebody_speed=0), R)
    assert not rearing_window_mask(evidence(after_forebody_speed=1), R)
    assert not rearing_window_mask(evidence(rear_speed=1), R)


def test_back_and_abdomen_are_independent_of_front_coherence():
    w = evidence(coherence=0)
    assert not grooming_window_mask(w, G)
    assert grooming_window_mask(dict(w, back_fraction=0.4), G)
    assert grooming_window_mask(dict(w, head_triangle_area=0.01, head_triangle_ratio=0.5), G)
    assert not grooming_window_mask(
        dict(w, head_triangle_area=0.01, head_triangle_ratio=0.5, triangle_coverage=0.1), G
    )


def test_triangle_collapse_without_fast_repeated_motion_is_not_grooming():
    assert not grooming_window_mask(
        evidence(head_triangle_area=0.001, head_triangle_ratio=0.1, relative_head_speed=0), G
    )
    assert not grooming_window_mask(
        evidence(
            head_triangle_area=0.001, head_triangle_ratio=0.1, reversals_s=0, lateral_reversals_s=0
        ),
        G,
    )


def test_large_head_excursion_and_fast_posterior_are_rejected():
    assert not grooming_window_mask(evidence(body_relative_amplitude=0.5), G)
    assert grooming_window_mask(evidence(body_relative_amplitude=0.5, back_fraction=0.4), G)
    assert grooming_window_mask(
        evidence(body_relative_amplitude=0.5, head_triangle_area=0.01, head_triangle_ratio=0.5), G
    )
    assert not grooming_window_mask(evidence(rear_speed=1), G)


def test_still_held_pose_needs_compact_geometry():
    cfg = dict(R, held_enabled=True)
    w = evidence(before_forebody_speed=0, neck_rear_radius=0.2)
    assert rearing_window_mask(w, cfg)
    assert not rearing_window_mask(dict(w, neck_rear_radius=0.6), cfg)


def test_missing_intermediate_frame_never_creates_fast_motion():
    points = np.zeros((8, 2))
    points[4:, 0] = 10
    observed = np.ones(8, bool)
    observed[3:5] = False
    speed = _speed(points, observed, 30.0, 10.0)
    assert np.isnan(speed[3:7]).all()


def test_no_unobserved_frames_or_short_events_are_added():
    c = clip(30)
    c["observed"][10:20] = False
    c["points"][10:20] = np.nan
    ws, _ = motion_windows(c, body_length_px=10)
    mask = window_candidate_mask(c, ws, [True] * len(ws))
    assert not mask[10:20].any()
    short = clip(5)
    ws = [dict(start=0, end=5, center=2, coverage=1.0)] * 2
    assert not window_candidate_mask(short, ws, [True, True]).any()


def test_measurements_are_scale_invariant_and_do_not_mutate_input():
    c = clip()
    before = deepcopy(c)
    ws, _ = motion_windows(c, body_length_px=10)
    enlarged = deepcopy(c)
    enlarged["points"] *= 3
    enlarged["boxes"] *= 3
    other, _ = motion_windows(enlarged, body_length_px=30)
    for a, b in zip(ws, other):
        assert a["head_triangle_area"] == pytest.approx(b["head_triangle_area"])
        assert a["neck_rear_radius"] == pytest.approx(b["neck_rear_radius"])
    assert np.array_equal(c["points"], before["points"])


def test_changing_one_class_does_not_change_the_other_evidence():
    w = evidence()
    g = grooming_window_mask(w, G)
    r = rearing_window_mask(w, R)
    assert not grooming_window_mask(w, dict(G, min_head_speed=100))
    assert rearing_window_mask(w, R) == r
    assert not rearing_window_mask(w, dict(R, min_rise_speed=100))
    assert grooming_window_mask(w, G) == g
