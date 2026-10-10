import numpy as np

from mouse_behavior.behavior.static_rule import (
    bounded_stationary_mask,
    bridge_gaps_outside_static,
)


def test_static_bridges_only_short_excursion_within_spatial_limit():
    mask = np.ones(40, dtype=bool)
    mask[15:20] = False
    centers = np.zeros((40, 2), dtype=float)
    centers[15:20, 0] = [0.2, 0.5, 0.9, 0.5, 0.2]

    result = bounded_stationary_mask(
        mask[:, None], centers[:, None, :], np.ones((40, 1), dtype=bool),
        fps=25., max_gap_s=.2, max_transient_displacement_cm=1.3375,
        max_center_radius_cm=1.3375,
    )

    assert result[:, 0].all()


def test_static_does_not_bridge_long_or_large_excursion():
    mask = np.ones(50, dtype=bool)
    mask[10:17] = False  # 0.28 s at 25 FPS
    mask[30:34] = False
    centers = np.zeros((50, 2), dtype=float)
    centers[30:34, 0] = [0.5, 1.2, 1.8, 0.5]

    result = bounded_stationary_mask(
        mask[:, None], centers[:, None, :], np.ones((50, 1), dtype=bool),
        fps=25., max_gap_s=.2, max_transient_displacement_cm=1.3375,
        max_center_radius_cm=1.3375,
    )

    assert not result[10:17, 0].any()
    assert not result[30:34, 0].any()


def test_static_rejects_continuous_drift_beyond_center_radius():
    mask = np.ones(30, dtype=bool)
    centers = np.zeros((30, 2), dtype=float)
    centers[:, 0] = np.linspace(0., 2., len(centers))

    result = bounded_stationary_mask(
        mask[:, None], centers[:, None, :], np.ones((30, 1), dtype=bool),
        fps=30., max_gap_s=.2, max_transient_displacement_cm=1.3375,
        max_center_radius_cm=1.3375,
    )

    assert not result[:, 0].any()


def test_locomotion_gap_recovery_never_overlaps_static_frames():
    moving = np.zeros((30, 1), dtype=bool)
    moving[2:7, 0] = True
    moving[8:12, 0] = True
    moving[15:18, 0] = True
    moving[20:23, 0] = True
    moving[25:28, 0] = True
    stationary = np.zeros_like(moving)
    stationary[7, 0] = True
    stationary[18:20, 0] = True

    result = bridge_gaps_outside_static(moving, stationary, max_gap_frames=2)

    assert not result[12:15, 0].any()
    assert not result[7, 0]
    assert not result[18:20, 0].any()
    assert result[23:25, 0].all()
