"""Measurement parity and gap handling for the imported posture definitions."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from mouse_behavior.behavior.posture_measurements import (
    MeasurementConfig,
    extract_posture_windows,
    quality_frames,
    reversal_rate,
    smooth3,
)


def _stationary_clip(n=21, fps=10.0):
    pose = np.array([[10, 0], [8, -1], [8, 1], [6, 0], [2, -1], [2, 1], [0, 0]], float)
    return {
        "points": np.tile(pose, (n, 1, 1)),
        "boxes": np.tile([0, -2, 11, 2], (n, 1)),
        "observed": np.ones(n, bool),
        "fps": fps,
    }


def _moving_clip():
    t = np.arange(41, dtype=float)
    p = np.empty((41, 7, 2))
    p[:, 0] = np.column_stack([24 + 2 * np.sin(t * 0.7), 10 + 1.5 * np.cos(t * 0.4)])
    p[:, 1] = np.column_stack([19 + 1.3 * np.sin(t * 0.7 + 0.2), 8 + 0.8 * np.cos(t * 0.4)])
    p[:, 2] = np.column_stack([19 + 1.3 * np.sin(t * 0.7 - 0.2), 12 + 0.8 * np.cos(t * 0.4)])
    p[:, 3] = np.column_stack([15 + 0.9 * np.sin(t * 0.13), 10 + 0.6 * np.cos(t * 0.11)])
    p[:, 4] = np.column_stack([5 + 0.4 * np.sin(t * 0.09), 8.5 + 0.3 * np.cos(t * 0.17)])
    p[:, 5] = np.column_stack([5 + 0.4 * np.sin(t * 0.09), 11.5 + 0.3 * np.cos(t * 0.17)])
    p[:, 6] = np.column_stack([0.2 * np.sin(t * 0.09), 10 + 0.4 * np.cos(t * 0.17)])
    p[13:15] = np.nan
    p[23, 1] = np.nan
    observed = np.ones(41, bool)
    observed[13:15] = False
    return dict(points=p, boxes=np.tile([0, 0, 30, 20], (41, 1)), observed=observed, fps=20.0)


def test_all_descriptors_match_preserved_upstream_golden():
    # Captured from heuristics.extract_windows + probe_training.enrich in the
    # reviewed delivery, independently of the production module under test.
    expected = {
        "start": 8,
        "end": 25,
        "center": 16,
        "coverage": 0.8823529411764706,
        "activity": 0.5629396917450453,
        "amplitude": 0.11101087801683202,
        "lateral_amplitude": 0.09718029363595168,
        "reversals_s": 1.25,
        "lateral_reversals_s": 2.5,
        "coherence": 1.0,
        "posterior_excursion": 0.012778632778741769,
        "projection_ratio": 0.9269561678466938,
        "projection_change": 0.13535361986316263,
        "spine_cosine": 0.9944379067321707,
        "comotion": 0.02163665865579799,
        "back_fraction": 0.0,
        "head_excursion": 0.14383498216139368,
        "neck_excursion": 0.03488448514754896,
        "body_relative_amplitude": 0.1475378748296103,
        "body_relative_speed": 0.8792390403799011,
        "head_rear_speed": 0.852216065410016,
        "neck_rear_speed": 0.09060311708225197,
        "relative_motion_fraction": 1.0317090654197933,
        "neck_to_relative_excursion": 0.23644427024475348,
        "head_rear_excursion": 0.1438349821613937,
        "neck_rear_excursion": 0.034884485147548994,
        "neck_length_change": 0.028125153771341782,
    }
    windows, scale = extract_posture_windows(_moving_clip())
    assert scale == pytest.approx(25.81301595638665, abs=1e-12)
    assert len(windows) == 21
    assert windows[0]["start"] == 0
    assert windows[-1]["end"] == 41
    actual = next(w for w in windows if w["center"] == 16)
    assert actual.keys() == expected.keys()
    assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_stationary_pose_preserves_null_comotion_and_all_window_fields():
    windows, scale = extract_posture_windows(_stationary_clip())
    assert scale == 10.0
    assert windows
    for w in windows:
        assert w["coverage"] == 1.0
        assert w["projection_ratio"] == 1.0
        assert w["spine_cosine"] == 1.0
        assert w["comotion"] is None
        assert w["relative_motion_fraction"] is None
        assert w["activity"] == 0.0
        assert w["body_relative_speed"] == 0.0
        assert w["posterior_excursion"] == 0.0
        assert w["neck_length_change"] == 0.0


def test_lagged_derivatives_require_every_intervening_observation():
    clip = _stationary_clip(n=5, fps=30.0)
    clip["points"][2] = np.nan
    clip["observed"][2] = False
    clip["points"][3:, :3, 0] += 3.0
    config = MeasurementConfig(window_s=1.0, min_window_s=0.1)
    windows, scale = extract_posture_windows(clip, config, body_length_px=10.0)
    assert scale == 10.0
    assert windows
    for w in windows:
        assert w["activity"] == 0.0
        assert w["body_relative_speed"] is None
        assert w["head_rear_speed"] is None
        assert w["neck_rear_speed"] is None
        assert w["reversals_s"] == 0.0


def test_stale_unobserved_coordinates_do_not_affect_measurements_or_scale():
    clip = _moving_clip()
    reference = extract_posture_windows(clip)
    clip["points"][13:15] = 1e9 * np.arange(14).reshape(7, 2)
    assert extract_posture_windows(clip) == reference
    assert not quality_frames(clip)[13:15].any()


def test_input_arrays_are_not_modified():
    clip = _moving_clip()
    clip["points"][13] = np.inf
    before = {key: value.copy() for key, value in clip.items() if isinstance(value, np.ndarray)}
    extract_posture_windows(clip)
    quality_frames(clip)
    for key, expected in before.items():
        np.testing.assert_array_equal(clip[key], expected)


def test_quality_requires_nose_neck_one_ear_and_two_rear_points():
    clip = _stationary_clip(n=8)
    clip["points"][1, 0] = np.nan
    clip["points"][2, 3] = np.nan
    clip["points"][3, 1:3] = np.nan
    clip["points"][4, 4:6] = np.nan
    clip["points"][5, 1] = np.nan
    clip["points"][6, 4] = np.nan
    clip["observed"][7] = False
    np.testing.assert_array_equal(
        quality_frames(clip), [True, False, False, False, False, True, True, False]
    )


def test_smoothing_and_reversals_never_fill_or_bridge_missing_values():
    points = np.array([[0.0, 0.0], [1.0, 1.0], [np.nan, 2.0], [3.0, 3.0], [4.0, 4.0]])
    before = points.copy()
    result = smooth3(points)
    assert np.isnan(result[2, 0])
    np.testing.assert_array_equal(result[[0, -1]], points[[0, -1]])
    np.testing.assert_array_equal(points, before)
    assert reversal_rate(np.array([0.0, 1.0, np.nan, 1.0, 0.0]), 10.0, 0.006) == 0.0
    assert reversal_rate(np.array([0.0, 1.0, 0.0]), 10.0, 0.006) == 5.0


def test_no_usable_proxy_scale_returns_no_windows():
    clip = _stationary_clip()
    clip["points"][:] = np.nan
    assert extract_posture_windows(clip) == ([], None)
    assert not quality_frames(clip).any()


def test_config_is_frozen_and_external_scale_is_respected():
    config = MeasurementConfig()
    with pytest.raises(FrozenInstanceError):
        config.window_s = 1.0
    _, scale = extract_posture_windows(_stationary_clip(), config, body_length_px=20.0)
    assert scale == 20.0


@pytest.mark.parametrize(
    "field,value",
    [
        ("window_s", 0.0),
        ("stride_s", 0.0),
        ("min_window_s", 0.0),
        ("window_s", np.inf),
        ("window_s", np.nan),
        ("window_s", True),
        ("keypoint_confidence", 1.01),
        ("detection_confidence", -1.0),
        ("min_coverage", 1.01),
        ("step_deadzone_body", -0.1),
    ],
)
def test_invalid_config_rejected(field, value):
    with pytest.raises(ValueError, match=field):
        MeasurementConfig(**{field: value})


@pytest.mark.parametrize("value", [0.0, -1.0, np.inf, np.nan, True, "10"])
def test_invalid_external_scale_rejected(value):
    with pytest.raises(ValueError, match="body_length_px"):
        extract_posture_windows(_stationary_clip(), body_length_px=value)


@pytest.mark.parametrize("value", [0.0, -1.0, np.inf, np.nan, True, "30"])
@pytest.mark.parametrize("measure", [extract_posture_windows, quality_frames])
def test_invalid_fps_rejected(value, measure):
    clip = _stationary_clip()
    clip["fps"] = value
    with pytest.raises(ValueError, match="fps"):
        measure(clip)


@pytest.mark.parametrize(
    "field,replacement",
    [
        ("points", np.zeros((21, 6, 2))),
        ("points", np.zeros((21, 7, 3))),
        ("boxes", np.zeros((20, 4))),
        ("observed", np.ones((21, 1), bool)),
        ("observed", np.ones(21)),
    ],
)
@pytest.mark.parametrize("measure", [extract_posture_windows, quality_frames])
def test_invalid_array_shapes_and_observation_flags_rejected(field, replacement, measure):
    clip = _stationary_clip()
    clip[field] = replacement
    with pytest.raises(ValueError, match=field):
        measure(clip)
