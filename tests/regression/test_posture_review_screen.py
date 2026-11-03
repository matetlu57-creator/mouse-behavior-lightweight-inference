"""Trial screen rejects weak posture without modifying independent V3 evidence."""

from copy import deepcopy

import numpy as np
import pytest

from mouse_behavior.behavior.posture_review_screen import (
    PostureScreenConfig,
    screen_cached_windows,
    screen_posture_clip,
    screen_window_evidence,
)


def window(**updates):
    return dict(
        dict(
            start=0,
            end=30,
            center=15,
            coverage=1.0,
            body_relative_amplitude=0.1,
            body_relative_speed=0.8,
            reversals_s=3.0,
            lateral_reversals_s=0.0,
            coherence=0.8,
            posterior_excursion=0.01,
            projection_ratio=0.8,
            projection_change=0.2,
            relative_motion_fraction=0.8,
            spine_cosine=0.8,
            back_fraction=0,
            neck_rear_excursion=0.1,
            neck_length_change=0.05,
            comotion=0.8,
        ),
        **updates,
    )


def screen(w, **kwargs):
    return screen_cached_windows([w, dict(w, center=14)], np.ones(30, bool), 30, **kwargs)


def test_strong_grooming_and_base_v3_are_preserved_independently():
    w = window()
    before = deepcopy(w)
    result = screen(w)
    assert result["masks"]["Grooming"].all()
    assert result["base_v3_masks"]["Grooming"].all()
    assert not result["masks"]["PostureReview"].any()
    assert w == before


@pytest.mark.parametrize(
    "updates",
    [
        {"body_relative_amplitude": 0.05},
        {"body_relative_speed": 0.1},
        {"coherence": 0.5},
        {"reversals_s": 0},
    ],
)
def test_ordinary_or_weak_motion_does_not_become_review_from_v3_alone(updates):
    result = screen(window(projection_change=0, **updates))
    assert not any(mask.any() for mask in result["masks"].values())


def test_suspicious_active_motion_routes_to_review_instead_of_forcing_class():
    result = screen(window(projection_ratio=0.98))
    assert result["base_v3_masks"]["Grooming"].all()
    assert result["masks"]["PostureReview"].all()
    assert not result["masks"]["Grooming"].any()


def rear_window(**updates):
    return window(relative_motion_fraction=0.3, spine_cosine=0.99, projection_ratio=0.5, **updates)


def test_rearing_requires_actual_short_projection_without_window_smear():
    ratio = np.r_[np.full(15, 0.5), np.full(15, 0.8)]
    result = screen(rear_window(body_relative_speed=0), frame_projection_ratio=ratio)
    assert result["masks"]["Rearing"][:15].all()
    assert not result["masks"]["Rearing"][15:].any()
    assert result["base_v3_masks"]["Rearing"].all()
    absent = screen(rear_window(body_relative_speed=0))
    assert not absent["masks"]["Rearing"].any()
    assert absent["diagnostics"]["window_counts"]["Rearing"] == 2
    assert not absent["diagnostics"]["measured_projection_available"]


def test_conflict_and_social_probes_route_only_measured_activity_to_review():
    conflict = screen(rear_window(back_fraction=0.3), frame_projection_ratio=np.full(30, 0.5))
    assert conflict["masks"]["PostureReview"].all()
    assert not conflict["masks"]["Grooming"].any()
    assert not conflict["masks"]["Rearing"].any()
    social = screen(window(), social_probe_mask=np.ones(30, bool))
    assert social["masks"]["PostureReview"].all()
    quiet = screen(window(body_relative_speed=0), social_probe_mask=np.ones(30, bool))
    assert not quiet["masks"]["PostureReview"].any()


def test_quality_gaps_minimum_windows_and_duration_are_enforced():
    q = np.ones(30, bool)
    q[8:22] = False
    w = window()
    result = screen_cached_windows([w, w], q, 30)
    assert not result["masks"]["Grooming"].any()  # both observed runs are < .3 s
    one = screen_cached_windows([w], np.ones(30, bool), 30)
    assert not any(mask.any() for mask in one["masks"].values())
    low = screen(window(coverage=0.69))
    assert not any(mask.any() for mask in low["masks"].values())


def test_lateral_repetitions_do_not_bypass_independent_v3_gate():
    w = window(reversals_s=0, lateral_reversals_s=4)
    assert not screen_window_evidence(w)["Grooming"]
    assert screen_window_evidence(dict(w, back_fraction=0.3))["Grooming"]


def test_clip_api_does_not_mutate_or_fill_unobserved_input():
    pose = np.array([[10, 0], [8, -1], [8, 1], [6, 0], [2, -1], [2, 1], [0, 0]], float)
    clip = dict(
        points=np.tile(pose, (60, 1, 1)),
        boxes=np.tile([0, -2, 11, 2], (60, 1)),
        observed=np.zeros(60, bool),
        fps=30,
    )
    before = deepcopy(clip)
    result = screen_posture_clip(clip, body_length_px=10)
    assert not any(mask.any() for mask in result["masks"].values())
    for key in ("points", "boxes", "observed"):
        np.testing.assert_array_equal(clip[key], before[key])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fps": 0},
        {"fps": float("nan")},
        {"quality": [1] * 30},
        {"social_probe_mask": [True]},
        {"frame_projection_ratio": [0.5]},
        {"windows": [window(end=31)]},
    ],
)
def test_invalid_cached_inputs_rejected(kwargs):
    args = dict(windows=[window()], quality=np.ones(30, bool), fps=30)
    args.update(kwargs)
    with pytest.raises(ValueError):
        screen_cached_windows(**args)


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        PostureScreenConfig(minimum_positive_windows=1.5)
