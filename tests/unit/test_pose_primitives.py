"""Unit tests for the additive pose primitives used by the new behavior gates."""

from __future__ import annotations

import numpy as np
import pytest

from mouse_behavior.preprocessing import pose_primitives as pp


def _keypoints(frames: int, mice: int, points: np.ndarray) -> np.ndarray:
    """Broadcast one (K, 2) keypoint layout across every frame and mouse."""

    out = np.zeros((frames, mice, points.shape[0], 2), dtype=float)
    out[:, :, :, :] = points[None, None, :, :]
    return out


def test_keypoint_speed_reports_constant_translation():
    frames, fps = 5, 10.0
    points = np.zeros((7, 2), dtype=float)
    keypoints = _keypoints(frames, 1, points)
    for frame in range(frames):
        keypoints[frame, 0, :, 0] = frame * 1.0  # 1 cm per frame in +x
    valid = np.ones((frames, 1), dtype=bool)

    speeds = pp.keypoint_speed_cm_s(keypoints, valid, fps)

    assert np.all(np.isnan(speeds[0]))
    np.testing.assert_allclose(speeds[1:], 10.0)


def test_keypoint_speed_stays_nan_across_a_long_dropout():
    """A dropout must not degrade into a calm-looking zero speed sample."""

    frames, fps = 6, 10.0
    keypoints = _keypoints(frames, 1, np.zeros((7, 2), dtype=float))
    keypoints[:, 0, :, 0] = (np.arange(frames, dtype=float) * 2.0)[:, None]
    valid = np.zeros((frames, 1), dtype=bool)
    valid[0, 0] = True
    valid[5, 0] = True  # gap of five frames, beyond MAX_GAP_FRAMES

    speeds = pp.keypoint_speed_cm_s(keypoints, valid, fps)

    assert np.all(np.isnan(speeds))


def test_keypoint_speed_bridges_a_short_gap_with_the_true_dt():
    frames, fps = 3, 10.0
    keypoints = _keypoints(frames, 1, np.zeros((7, 2), dtype=float))
    keypoints[0, 0, :, 0] = 0.0
    keypoints[2, 0, :, 0] = 2.0
    valid = np.asarray([[True], [False], [True]])

    speeds = pp.keypoint_speed_cm_s(keypoints, valid, fps)

    assert np.all(np.isnan(speeds[0]))
    assert np.all(np.isnan(speeds[1]))
    np.testing.assert_allclose(speeds[2], 2.0 / 0.2)


def test_keypoint_acceleration_uses_absolute_speed_change():
    fps = 10.0
    speeds = np.asarray([[[0.0], [5.0], [10.0]]], dtype=float).reshape(3, 1, 1)
    valid = np.ones((3, 1), dtype=bool)

    acceleration = pp.keypoint_acceleration_cm_s2(speeds, valid, fps)

    assert np.isnan(acceleration[0, 0, 0])
    np.testing.assert_allclose(acceleration[1:, 0, 0], 50.0)


def test_group_speed_honours_min_valid():
    speeds = np.asarray([[[1.0, 3.0, np.nan]]], dtype=float)

    lenient = pp.group_speed_cm_s(speeds, (0, 1, 2), min_valid=1)
    strict = pp.group_speed_cm_s(speeds, (0, 1, 2), min_valid=3)

    np.testing.assert_allclose(lenient[0, 0], 2.0)
    assert np.isnan(strict[0, 0])


def test_head_and_hind_speeds_select_the_expected_keypoints():
    speeds = np.zeros((1, 1, 7), dtype=float)
    speeds[0, 0, pp.KP_NOSE] = 1.0
    speeds[0, 0, pp.KP_LEFT_EAR] = 3.0
    speeds[0, 0, pp.KP_RIGHT_EAR] = 3.0
    speeds[0, 0, pp.KP_LEFT_HIP] = 0.0
    speeds[0, 0, pp.KP_RIGHT_HIP] = 0.0
    speeds[0, 0, pp.KP_TAIL] = 0.0

    np.testing.assert_allclose(pp.head_speed_cm_s(speeds)[0, 0], 7.0 / 3.0)
    np.testing.assert_allclose(pp.hind_speed_cm_s(speeds)[0, 0], 0.0)


def test_ratio_or_nan_floors_small_denominators():
    ratio = pp.ratio_or_nan(np.asarray([4.0, 4.0]), np.asarray([2.0, 1e-12]))

    np.testing.assert_allclose(ratio[0], 2.0)
    assert np.isnan(ratio[1])


def test_body_length_px_is_nan_when_the_tail_is_missing():
    points = np.zeros((7, 2), dtype=float)
    points[pp.KP_NOSE] = (100.0, 0.0)
    points[pp.KP_TAIL] = (0.0, 0.0)
    keypoints = _keypoints(1, 1, points)
    valid = np.ones((1, 1), dtype=bool)

    np.testing.assert_allclose(pp.body_length_px(keypoints, valid)[0, 0], 100.0)

    keypoints[0, 0, pp.KP_TAIL] = np.nan
    assert np.isnan(pp.body_length_px(keypoints, valid)[0, 0])


def test_foreshorten_ratio_flags_a_rearing_posture():
    frames = 30
    body_px = np.full((frames, 1), 100.0, dtype=float)
    body_px[20:, 0] = 60.0
    valid = np.ones((frames, 1), dtype=bool)

    ratio = pp.foreshorten_ratio(body_px, valid, window=10)

    np.testing.assert_allclose(ratio[:20, 0], 1.0, atol=1e-9)
    np.testing.assert_allclose(ratio[20:, 0], 0.6, atol=1e-9)


def test_foreshorten_base_resists_a_mostly_rearing_track():
    """A short window full of rearing frames must not redefine the baseline."""

    frames = 35
    body_px = np.full((frames, 1), 100.0, dtype=float)
    body_px[30:, 0] = 60.0  # only the last five frames are short
    valid = np.ones((frames, 1), dtype=bool)

    ratio = pp.foreshorten_ratio(body_px, valid, window=5)

    np.testing.assert_allclose(ratio[30:, 0], 0.6, atol=1e-9)


def test_rolling_iqr_is_zero_for_a_constant_signal():
    values = np.full((10, 1), 3.0, dtype=float)

    np.testing.assert_allclose(pp.rolling_iqr(values, 5)[:, 0], 0.0, atol=1e-12)


def test_deformation_burst_ratio_separates_grooming_from_a_bite():
    frames, window = 40, 20
    valid = np.ones((frames, 1), dtype=bool)

    grooming = np.full((frames, 1), 0.50, dtype=float)
    # A bite is a short burst, not a single sample: six explosive frames inside
    # the trailing window have to lift the q0.95 tail while the median stays at
    # the calm baseline.
    biting = np.full((frames, 1), 0.20, dtype=float)
    biting[20:26, 0] = 6.0

    calm = pp.deformation_burst_ratio(grooming, valid, window=window)
    bursty = pp.deformation_burst_ratio(biting, valid, window=window)

    np.testing.assert_allclose(calm[39, 0], 1.0, atol=1e-9)
    assert bursty[39, 0] > 5.0


def _torso_frame(nose_xy: tuple[float, float]) -> np.ndarray:
    points = np.zeros((7, 2), dtype=float)
    points[pp.KP_NOSE] = nose_xy
    points[pp.KP_NECK] = (0.0, 0.0)
    points[pp.KP_LEFT_HIP] = (-2.0, -1.0)
    points[pp.KP_RIGHT_HIP] = (-2.0, 1.0)
    points[pp.KP_TAIL] = (-4.0, 0.0)
    return points


def test_nose_in_torso_accepts_an_inside_nose_only():
    valid = np.ones((1, 1), dtype=bool)
    inside = _keypoints(1, 1, _torso_frame((-1.0, 0.0)))
    outside = _keypoints(1, 1, _torso_frame((5.0, 0.0)))

    assert pp.nose_in_torso(inside, valid)[0, 0]
    assert not pp.nose_in_torso(outside, valid)[0, 0]


def test_nose_in_torso_inflation_widens_the_region():
    valid = np.ones((1, 1), dtype=bool)
    edge = _keypoints(1, 1, _torso_frame((-5.5, 0.0)))

    assert not pp.nose_in_torso(edge, valid, inflate_cm=0.0)[0, 0]
    assert pp.nose_in_torso(edge, valid, inflate_cm=2.0)[0, 0]


def test_nose_in_torso_reports_undecidable_frames_as_false():
    valid = np.ones((1, 1), dtype=bool)
    degraded = _keypoints(1, 1, _torso_frame((-1.0, 0.0)))
    degraded[0, 0, pp.KP_LEFT_HIP] = np.nan
    degraded[0, 0, pp.KP_RIGHT_HIP] = np.nan

    missing_nose = _keypoints(1, 1, _torso_frame((-1.0, 0.0)))
    missing_nose[0, 0, pp.KP_NOSE] = np.nan

    assert not pp.nose_in_torso(degraded, valid)[0, 0]
    assert not pp.nose_in_torso(missing_nose, valid)[0, 0]
    assert not pp.nose_in_torso(
        _keypoints(1, 1, _torso_frame((-1.0, 0.0))), np.zeros((1, 1), dtype=bool)
    )[0, 0]


def test_nose_to_nearest_keypoint_picks_the_minimum_distance():
    target = np.zeros((1, 7, 2), dtype=float)
    target[0, :, 0] = np.asarray([5.0, 2.0, 9.0, 7.0, 7.0, 7.0, 7.0])
    actor = np.zeros((1, 2), dtype=float)

    distance = pp.nose_to_nearest_keypoint_cm(actor, target)

    np.testing.assert_allclose(distance[0], 2.0)

    excluded = pp.nose_to_nearest_keypoint_cm(actor, target, exclude_indices=(1,))
    np.testing.assert_allclose(excluded[0], 5.0)


def test_nose_to_nearest_keypoint_is_nan_without_an_actor_nose():
    target = np.zeros((1, 7, 2), dtype=float)
    target[0, :, 0] = 3.0
    actor = np.full((1, 2), np.nan, dtype=float)

    assert np.isnan(pp.nose_to_nearest_keypoint_cm(actor, target)[0])


def test_nose_to_nearest_keypoint_matrix_leaves_the_diagonal_undecidable():
    frames, mice = 2, 2
    actor = np.zeros((frames, mice, 2), dtype=float)
    actor[:, 1, 0] = 8.0
    target = np.zeros((frames, mice, 7, 2), dtype=float)
    target[:, 1, :, 0] = 8.0

    matrix = pp.nose_to_nearest_keypoint_matrix(actor, target)

    assert matrix.shape == (frames, mice, mice)
    np.testing.assert_allclose(matrix[:, 0, 1], 8.0)
    np.testing.assert_allclose(matrix[:, 1, 0], 8.0)
    assert np.all(np.isnan(matrix[:, 0, 0]))
    assert np.all(np.isnan(matrix[:, 1, 1]))


def test_nose_matrix_chunking_matches_the_single_pass(monkeypatch):
    frames, mice = 6, 2
    rng = np.random.default_rng(20261004)
    actor = rng.normal(size=(frames, mice, 2)) * 5.0
    target = rng.normal(size=(frames, mice, 7, 2)) * 5.0
    valid = np.ones((frames, mice, 7), dtype=bool)
    valid[2, 0, :] = False

    reference = pp.nose_to_nearest_keypoint_matrix(actor, target, valid)
    monkeypatch.setattr(pp, "_MAX_MATRIX_ELEMENTS", mice * mice * 7)
    chunked = pp.nose_to_nearest_keypoint_matrix(actor, target, valid)

    np.testing.assert_allclose(chunked, reference, equal_nan=True)


@pytest.mark.parametrize(
    "call",
    [
        lambda: pp.keypoint_speed_cm_s(np.zeros((2, 1, 7)), np.ones((2, 1), bool), 10.0),
        lambda: pp.keypoint_speed_cm_s(np.zeros((2, 1, 7, 2)), np.ones((2, 2), bool), 10.0),
        lambda: pp.keypoint_speed_cm_s(np.zeros((2, 1, 7, 2)), np.ones((2, 1), bool), 0.0),
        lambda: pp.keypoint_speed_cm_s(np.zeros((2, 1, 7, 2)), np.ones((2, 1), bool), 10.0, 0),
        lambda: pp.group_speed_cm_s(np.zeros((1, 1, 7)), ()),
        lambda: pp.group_speed_cm_s(np.zeros((1, 1, 7)), (9,)),
        lambda: pp.group_speed_cm_s(np.zeros((1, 1, 7)), (0, 1), min_valid=3),
        lambda: pp.foreshorten_ratio(np.zeros((2, 1)), np.ones((2, 2), bool), window=2),
        lambda: pp.nose_in_torso(np.zeros((1, 1, 7, 2)), np.ones((1, 1), bool), inflate_cm=-1.0),
        lambda: pp.nose_to_nearest_keypoint_cm(np.zeros((3, 2)), np.zeros((2, 7, 2))),
        lambda: pp.nose_to_nearest_keypoint_cm(
            np.zeros((1, 2)), np.zeros((1, 7, 2)), exclude_indices=(7,)
        ),
    ],
)
def test_primitives_reject_malformed_inputs(call):
    with pytest.raises(ValueError):
        call()