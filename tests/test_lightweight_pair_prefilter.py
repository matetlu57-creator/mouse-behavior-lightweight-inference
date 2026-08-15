from __future__ import annotations

import numpy as np

from mouse_behavior import lightweight_behavior_inference as lightweight


def _three_mouse_kinematics() -> dict[str, np.ndarray]:
    frames = 3
    centers = np.zeros((frames, 3, 2), dtype=float)
    centers[:, 0] = (0.0, 0.0)
    centers[:, 1] = (8.0, 0.0)
    centers[:, 2] = (0.0, 15.0)
    heading = np.zeros((frames, 3, 2), dtype=float)
    heading[:, 0] = (0.0, 1.0)
    heading[:, 1] = (1.0, 0.0)
    heading[:, 2] = (0.0, 1.0)
    return {
        "centers_cm": centers,
        "heading": heading,
        "valid": np.ones((frames, 3), dtype=bool),
    }


def test_pair_prefilter_keeps_close_pairs_and_facing_wider_pairs():
    kin = _three_mouse_kinematics()
    result, pair_i, pair_j = lightweight._pair_prefilter(
        kin,
        {
            "lightweight_behavior_inference": {
                "pair_prefilter": {
                    "enabled": True,
                    "close_distance_cm": 10.0,
                    "min_heading_cosine": 0.0,
                }
            }
        },
        interaction_radius=17.0,
    )

    # Stable triu order for three mice is (0,1), (0,2), (1,2).
    assert list(zip(pair_i.tolist(), pair_j.tolist())) == [(0, 1), (0, 2), (1, 2)]
    assert result["candidate_pair_mask"].tolist() == [True, True, False]
    assert result["valuable_frame"].shape == (3, 3)


def test_pair_prefilter_can_restore_distance_only_compatibility_mode():
    kin = _three_mouse_kinematics()
    result, _, _ = lightweight._pair_prefilter(
        kin,
        {
            "lightweight_behavior_inference": {
                "pair_prefilter": {"enabled": False}
            }
        },
        interaction_radius=17.0,
    )

    assert result["candidate_pair_mask"].tolist() == [True, True, True]
    assert np.array_equal(result["valuable_frame"], result["valid_pair"] & (result["distance"] <= 17.0))


def test_pair_metrics_subset_preserves_selected_full_pair_columns():
    rng = np.random.default_rng(7)
    frames, mice = 8, 4
    kin = {
        "centers_cm": rng.normal(size=(frames, mice, 2)).cumsum(axis=0),
        "head_cm": rng.normal(size=(frames, mice, 2)),
        "keypoints_cm": rng.normal(size=(frames, mice, lightweight.KEYPOINTS, 2)),
        "heading": rng.normal(size=(frames, mice, 2)),
        "velocity": rng.normal(size=(frames, mice, 2)),
        "valid": np.ones((frames, mice), dtype=bool),
        "speed": np.abs(rng.normal(size=(frames, mice))),
        "nose_speed": np.abs(rng.normal(size=(frames, mice))),
        "acceleration": rng.normal(size=(frames, mice)),
        "angular_speed": rng.normal(size=(frames, mice)),
        "pose_quality": np.ones((frames, mice), dtype=float),
        "behavior_speed": np.abs(rng.normal(size=(frames, mice))),
        "pose_deformation": np.abs(rng.normal(size=(frames, mice))),
    }
    full, full_i, full_j = lightweight._pair_metrics(kin, 10.0)
    selected_indices = np.array([1, 4], dtype=int)
    subset, subset_i, subset_j = lightweight._pair_metrics(
        kin,
        10.0,
        pair_indices=selected_indices,
    )

    assert np.array_equal(subset_i, full_i[selected_indices])
    assert np.array_equal(subset_j, full_j[selected_indices])
    for name, value in full.items():
        if not isinstance(value, np.ndarray) or value.ndim < 2:
            continue
        if value.shape[1] != len(full_i):
            continue
        np.testing.assert_allclose(
            subset[name],
            value[:, selected_indices],
            equal_nan=True,
        )
