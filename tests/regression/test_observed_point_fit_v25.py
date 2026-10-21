"""Real observed windows must not inherit gaps or jumps from a different run."""
import numpy as np
import pytest

from mouse_behavior.evaluation import confirmed_rules as rules


def fit(points, observed, baseline=None):
    return rules._observed_run_point_kinematics(
        points, observed, 30., half_window_s=.125,
        max_jump_speed_cm_s=120., baseline=baseline)


def quadratic(n):
    time = np.arange(n) / 30.
    return np.column_stack((12 * time + 60 * time**2, np.zeros(n)))


def test_observed_clip_edges_have_fits_and_legacy_interior_is_unchanged():
    points = quadratic(9)
    observed = np.ones(9, bool)
    old = rules._local_quadratic_point_kinematics(
        points, observed, 30., half_window_s=.125, max_jump_speed_cm_s=120.)
    assert not np.isfinite(old[0][[0, 1, 7, 8]]).any()
    for array in old:
        array.setflags(write=False)
    velocity, acceleration = fit(points, observed, old)
    np.testing.assert_allclose(velocity[:, 0], 12 + 120 * np.arange(9) / 30., atol=1e-9)
    np.testing.assert_allclose(acceleration[:, 0], 120., atol=1e-9)
    existing = np.isfinite(old[0]).all(axis=1)
    assert np.array_equal(velocity[existing], old[0][existing])
    assert np.array_equal(acceleration[existing], old[1][existing])


def test_five_observed_points_between_gaps_can_be_fitted_without_filling_gaps():
    points = quadratic(15)
    observed = np.zeros(15, bool)
    observed[5:10] = True
    source = points.copy()
    velocity, acceleration = fit(points, observed)
    assert np.isfinite(velocity[observed]).all()
    assert not np.isfinite(velocity[~observed]).any()
    assert not np.isfinite(acceleration[~observed]).any()
    points[~observed] = 1e6
    changed = fit(points, observed)
    np.testing.assert_array_equal(velocity, changed[0])
    np.testing.assert_array_equal(acceleration, changed[1])
    # The unperturbed input itself was not used as a writable workspace.
    assert np.array_equal(source[observed], points[observed])


def test_tracker_jump_splits_fit_windows_and_cannot_create_an_acceleration_spike():
    points = quadratic(14)
    points[7:, 0] += 10000.
    observed = np.ones(14, bool)
    velocity, acceleration = fit(points, observed)
    np.testing.assert_allclose(velocity[:, 0], 12 + 120 * np.arange(14) / 30., atol=1e-7)
    np.testing.assert_allclose(acceleration[:, 0], 120., atol=1e-6)
    altered = points.copy()
    altered[:7] -= 1000.
    other = fit(altered, observed)
    np.testing.assert_array_equal(velocity[7:], other[0][7:])
    np.testing.assert_array_equal(acceleration[7:], other[1][7:])


@pytest.mark.parametrize('count', [1, 2, 3, 4])
def test_fewer_than_five_real_points_never_produces_kinematics(count):
    points = quadratic(15)
    observed = np.zeros(15, bool)
    observed[5:5 + count] = True
    velocity, acceleration = fit(points, observed)
    assert not np.isfinite(velocity).any()
    assert not np.isfinite(acceleration).any()


def test_unknown_point_stays_unknown_and_inputs_are_not_mutated():
    points = quadratic(15)
    points[7] = np.nan
    observed = np.ones(15, bool)
    original = points.copy()
    velocity, acceleration = fit(points, observed)
    assert np.isnan(velocity[7]).all() and np.isnan(acceleration[7]).all()
    np.testing.assert_array_equal(points, original)
    assert observed.all()
