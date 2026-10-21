"""An established approach must survive unrelated unavailable crop boundaries."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from tests.regression import test_heuristics_v16_audit as audit
from mouse_behavior.evaluation.confirmed_rules import _open_approach_mask
from mouse_behavior.evaluation.exported_dataset_config import config_seed_parameters
from mouse_behavior.evaluation.social_heuristic_config import DEFAULT_SOCIAL_HEURISTIC_CONFIG


def _closing_pair():
    f = audit.features(60, 2, reference_body_cm=7.918)
    f.centers_cm[:, 0, 0] = np.arange(60) * .2
    f.centers_cm[:, 1, 0] = 18.
    audit.sync(f)
    audit.resync_kinematics(f)
    shape = np.array([[3.959, 0.], [2.5, .8], [2.5, -.8], [1., 0.],
                      [-2., 1.], [-2., -1.], [-3.959, 0.]])
    f.collective_keypoints_cm = f.centers_cm[:, :, None, :] + shape
    return f


def test_unmeasured_first_velocity_does_not_erase_observed_approach():
    measured = _closing_pair()
    control = _open_approach_mask(measured, 0, config_seed_parameters({}))
    assert control.any(), "control needs real directed closing and endpoint geometry"
    loaded = deepcopy(measured)
    # Production loader has no adjacent observation at frame0: velocity0 is
    # an unmeasured sentinel. Frame1 onward remains genuine directed motion.
    loaded.velocity_cm_s[0] = 0.
    loaded.speed_cm_s[0] = 0.
    result = _open_approach_mask(loaded, 0, config_seed_parameters({}))
    assert result[1:35].all()
    assert not result[0]


def test_later_missing_pose_does_not_erase_a_completed_earlier_approach():
    complete = _closing_pair()
    cropped = deepcopy(complete)
    cropped.collective_keypoints_cm[40:, :, 0] = np.nan
    # The measured prefix itself already meets unchanged0.3s and endpoint
    # gates. Later unavailable snouts cannot retroactively remove it.
    before = deepcopy(complete)
    for name, value in vars(before).items():
        if isinstance(value, np.ndarray) and value.ndim and value.shape[0] == 60:
            setattr(before, name, value[:40].copy())
    expected = _open_approach_mask(before, 0, config_seed_parameters({}))
    assert expected.any()
    result = _open_approach_mask(cropped, 0, config_seed_parameters({}))
    assert np.array_equal(result[:40], expected)
    assert not result[40:].any()


def test_measured_target_departure_after_a_completed_prefix_does_not_erase_prefix():
    f = _closing_pair()
    f.centers_cm[35:, 1, 0] += np.arange(25) * .1
    audit.sync(f)
    audit.resync_kinematics(f)
    shape = f.collective_keypoints_cm - _closing_pair().centers_cm[:, :, None, :]
    f.collective_keypoints_cm = f.centers_cm[:, :, None, :] + shape
    result = _open_approach_mask(f, 0, config_seed_parameters({}))
    assert result[:35].all()


def test_receding_target_is_allowed_when_actor_closes_toward_it():
    f = _closing_pair()
    f.centers_cm[:, 0, 0] = np.arange(60) * (6.5 / 30.)
    f.centers_cm[:, 1, 0] = 14. + np.arange(60) * (5.2 / 30.)
    audit.sync(f)
    audit.resync_kinematics(f)
    shape = _closing_pair().collective_keypoints_cm - _closing_pair().centers_cm[:, :, None, :]
    f.collective_keypoints_cm = f.centers_cm[:, :, None, :] + shape
    assert _open_approach_mask(f, 0, config_seed_parameters({})).any()


def test_missing_all_snouts_cannot_supply_a_fabricated_approach():
    f = _closing_pair()
    f.collective_keypoints_cm[:, :, 0] = np.nan
    assert not _open_approach_mask(f, 0, config_seed_parameters({})).any()


def test_visible_receding_lookback_rows_are_not_painted_as_approach():
    f = _closing_pair()
    shape = f.collective_keypoints_cm - f.centers_cm[:, :, None, :]
    f.centers_cm[:5, 0, 0] = -np.arange(5) * .2
    f.centers_cm[5:, 0, 0] = -.8 + np.arange(55) * .2
    audit.sync(f)
    audit.resync_kinematics(f)
    f.collective_keypoints_cm = f.centers_cm[:, :, None, :] + shape
    assert (f.velocity_cm_s[:5, 0, 0] < 0).all()
    result = _open_approach_mask(f, 0, config_seed_parameters({}))
    assert not result[:6].any()
    assert result[6:35].all()


def test_target_approaching_actor_cannot_create_approach_by_actor_moving_away():
    f = _closing_pair()
    shape = f.collective_keypoints_cm - f.centers_cm[:, :, None, :]
    f.centers_cm[:, 0, 0] = -np.arange(60) * (2. / 30.)
    f.centers_cm[:, 1, 0] = 18. - np.arange(60) * (7. / 30.)
    audit.sync(f)
    audit.resync_kinematics(f)
    f.collective_keypoints_cm = f.centers_cm[:, :, None, :] + shape
    assert f.pair_distance_cm[-1, 0] < f.pair_distance_cm[0, 0]
    assert not _open_approach_mask(f, 0, config_seed_parameters({})).any()


def test_actor_motion_toward_target_without_actual_closing_is_not_approach():
    f = _closing_pair()
    shape = f.collective_keypoints_cm - f.centers_cm[:, :, None, :]
    f.centers_cm[:, 1, 0] = 14. + np.arange(60) * (7. / 30.)
    audit.sync(f)
    audit.resync_kinematics(f)
    f.collective_keypoints_cm = f.centers_cm[:, :, None, :] + shape
    assert f.pair_distance_cm[-1, 0] > f.pair_distance_cm[0, 0]
    assert not _open_approach_mask(f, 0, config_seed_parameters({})).any()


def test_laterally_moving_target_does_not_require_toward_velocity_or_small_path():
    f = _closing_pair()
    shape = f.collective_keypoints_cm - f.centers_cm[:, :, None, :]
    f.centers_cm[:, 1, 1] = np.arange(60) * (1. / 30.)
    audit.sync(f)
    audit.resync_kinematics(f)
    f.collective_keypoints_cm = f.centers_cm[:, :, None, :] + shape
    assert np.linalg.norm(f.centers_cm[-1, 1] - f.centers_cm[0, 1]) > 1.3375
    assert _open_approach_mask(f, 0, config_seed_parameters({})).any()


def test_legacy_target_motion_settings_warn_and_do_not_change_detection():
    f = _closing_pair()
    p = config_seed_parameters({})
    expected = _open_approach_mask(f, 0, p)
    assert expected.any()
    with pytest.warns(RuntimeWarning, match="Retired social settings"):
        legacy = replace(DEFAULT_SOCIAL_HEURISTIC_CONFIG,
                         approach_stationary_target_max_path_cm=.00001,
                         vel_approach_actor_target_ratio_min=999.)
    assert np.array_equal(_open_approach_mask(f, 0, p, legacy), expected)


def test_lateral_target_faster_than_retired_relative_speed_limit_is_allowed():
    f = _closing_pair()
    shape = f.collective_keypoints_cm - f.centers_cm[:, :, None, :]
    f.centers_cm[:, 0, 0] = np.arange(60) * (6.5 / 30.)
    f.centers_cm[:, 1, 0] = 14.
    f.centers_cm[:, 1, 1] = np.arange(60) * (6. / 30.)
    audit.sync(f)
    audit.resync_kinematics(f)
    f.collective_keypoints_cm = f.centers_cm[:, :, None, :] + shape
    assert np.all(f.speed_cm_s[:, 0] < 1.2 * f.speed_cm_s[:, 1])
    assert np.linalg.norm(f.centers_cm[-1, 1] - f.centers_cm[0, 1]) > 1.3375
    assert f.pair_distance_cm[-1, 0] < f.pair_distance_cm[0, 0]
    assert _open_approach_mask(f, 0, config_seed_parameters({})).any()
