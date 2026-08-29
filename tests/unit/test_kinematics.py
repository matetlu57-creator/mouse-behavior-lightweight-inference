from __future__ import annotations

import numpy as np
import pytest

from mouse_behavior.preprocessing.kinematics import (
    _kinematics,
    _reference_body_pixel_length,
)


def test_reference_body_pixel_length_gives_each_track_equal_weight() -> None:
    body_lengths = np.asarray(
        [
            [120.0, 60.0, 60.0, 60.0],
            [120.0, 60.0, 60.0, 60.0],
            [120.0, 60.0, 60.0, 60.0],
            [120.0, 60.0, 60.0, 60.0],
        ],
        dtype=float,
    )
    valid = np.ones_like(body_lengths, dtype=bool)

    reference = _reference_body_pixel_length(body_lengths, valid)

    # The large mouse has many pixels, but it must not outweigh the three
    # ordinary tracks when the shared camera scale is estimated.
    assert reference == pytest.approx(60.0)


def test_reference_body_pixel_length_ignores_invalid_track_observations() -> None:
    body_lengths = np.asarray(
        [
            [120.0, 60.0, 62.0],
            [120.0, 60.0, 61.0],
            [120.0, 60.0, 60.0],
        ],
        dtype=float,
    )
    valid = np.asarray(
        [
            [False, True, True],
            [False, True, True],
            [False, True, True],
        ],
        dtype=bool,
    )

    reference = _reference_body_pixel_length(body_lengths, valid)

    assert reference == pytest.approx(60.5)


def test_kinematics_accepts_fixed_shared_camera_scale() -> None:
    frames = 2
    mice = 1
    keypoints = np.zeros((frames, mice, 7, 2), dtype=float)
    keypoints[:, 0, 0] = [20.0, 0.0]
    keypoints[:, 0, 3] = [10.0, 0.0]
    keypoints[:, 0, 4] = [0.0, 0.0]
    keypoints[:, 0, 5] = [0.0, 10.0]
    keypoints[:, 0, 6] = [-20.0, 0.0]
    tracks = {
        "valid": np.ones((frames, mice), dtype=bool),
        "pose_quality": np.ones((frames, mice), dtype=float),
        "keypoints_px": keypoints,
        "centers_px": np.zeros((frames, mice, 2), dtype=float),
        "body_lengths_px": np.full((frames, mice), 40.0, dtype=float),
    }

    result = _kinematics(tracks, fps=10.0, cm_per_pixel=0.25)

    assert result["cm_per_pixel"] == pytest.approx(0.25)
    assert result["reference_body_px"] == pytest.approx(40.0)
    assert result["body_cm"][0, 0] == pytest.approx(10.0)


def test_kinematics_rejects_non_positive_fixed_shared_camera_scale() -> None:
    tracks = {
        "valid": np.ones((1, 1), dtype=bool),
        "pose_quality": np.ones((1, 1), dtype=float),
        "keypoints_px": np.zeros((1, 1, 7, 2), dtype=float),
        "centers_px": np.zeros((1, 1, 2), dtype=float),
        "body_lengths_px": np.ones((1, 1), dtype=float) * 40.0,
    }

    with pytest.raises(ValueError, match="cm_per_pixel"):
        _kinematics(tracks, fps=10.0, cm_per_pixel=0.0)
