"""Observable head/body directions and stable actor-target pursuit regressions."""
from copy import deepcopy

import numpy as np
import pytest

from tests.regression import test_heuristics_v16_audit as audit
from mouse_behavior.evaluation import confirmed_rules as rules
from mouse_behavior.evaluation.exported_dataset_config import config_seed_parameters


def _pose(f):
    forward = np.zeros_like(f.centers_cm)
    forward[..., 0] = 1.
    speed = np.linalg.norm(f.velocity_cm_s, axis=-1)
    moving = speed > 1e-8
    forward[moving] = f.velocity_cm_s[moving] / speed[moving, None]
    side = np.stack((-forward[..., 1], forward[..., 0]), axis=-1)
    shape = ((3.959, 0.), (2.5, .8), (2.5, -.8), (1., 0.),
             (-2., 1.), (-2., -1.), (-3.959, 0.))
    f.collective_keypoints_cm = np.stack([
        f.centers_cm + x * forward + y * side for x, y in shape
    ], axis=2)
    return f


def _approach(target_motion=(0., 0.)):
    f = audit.features(60, 2, reference_body_cm=7.918)
    time = np.arange(60) / f.fps
    f.centers_cm[:, 0, 0] = 6.5 * time
    f.centers_cm[:, 1, 0] = 18. + target_motion[0] * time
    f.centers_cm[:, 1, 1] = target_motion[1] * time
    audit.sync(f)
    audit.resync_kinematics(f)
    return _pose(f)


def _chase(turn=False):
    f = audit.features(150, 2, reference_body_cm=7.918)
    time = np.arange(150) / f.fps
    for slot, delay in ((0, .3), (1, 0.)):
        shifted = time - delay
        f.centers_cm[:, slot, 0] = 60 * (np.minimum(shifted, 2.) if turn else shifted)
        f.centers_cm[:, slot, 1] = 60 * np.maximum(shifted - 2., 0.) if turn else 0.
    audit.sync(f)
    audit.resync_kinematics(f)
    return _pose(f)


def _roles(f):
    roles = {}
    masks = rules.pair_rule_masks(f, config_seed_parameters({}), roles=roles)
    return masks, roles


def test_measured_backward_head_is_not_replaced_by_forward_velocity():
    f = _approach()
    assert rules._open_approach_mask(f, 0, config_seed_parameters({})).any()
    f.collective_keypoints_cm[:, 0, 0] = (
        2 * f.collective_keypoints_cm[:, 0, 3] - f.collective_keypoints_cm[:, 0, 0])
    assert not rules._open_approach_mask(f, 0, config_seed_parameters({})).any()


@pytest.mark.parametrize("target_motion", [(0., 0.), (0., 1.), (1., 0.)])
def test_approach_roles_use_visual_ids_for_stationary_lateral_or_receding_target(target_motion):
    f = _approach(target_motion)
    masks, roles = _roles(f)
    selected = masks["approach"][:, 0]
    assert selected.any()
    assert (roles["approach_actor"][selected, 0] == 100).all()
    assert (roles["approach_target"][selected, 0] == 101).all()
    assert not masks["chasing"].any(), "target motion alone is not pursuit"


def test_missing_neck_does_not_promote_velocity_fallback_to_measured_head_direction():
    f = _approach()
    f.collective_keypoints_cm[:, 0, 3] = np.nan
    assert np.isfinite(f.heading_unit[:, 0]).all(), "legacy heading has velocity fallback"
    masks, roles = _roles(f)
    assert masks["approach"].any(), "measured closing can remain an undirected candidate"
    assert (roles["approach_actor"] == -1).all()
    assert (roles["approach_target"] == -1).all()


def test_fully_backward_pose_cannot_establish_directed_chase():
    f = _chase()
    f.collective_keypoints_cm[:, 0] = (
        2 * f.centers_cm[:, 0, None] - f.collective_keypoints_cm[:, 0])
    assert not rules._chase_mask(f, 0).any()


def test_parallel_side_by_side_is_not_chase_even_when_both_heads_point_forward():
    f = _chase()
    f.centers_cm[:, 0, 0] += 18.  # Cancel .3s follower lag at the new60cm/s fixture speed.
    f.centers_cm[:, 1, 1] = 5.
    audit.sync(f)
    audit.resync_kinematics(f)
    _pose(f)
    assert not rules._chase_mask(f, 0).any()


def test_real_leader_turn_retains_same_chaser_id():
    f = _chase(turn=True)
    masks, roles = _roles(f)
    time = np.arange(len(f.valid)) / f.fps
    corner = (time > 2.05) & (time < 2.25)
    assert masks["chasing"][corner, 0].all()
    assigned = roles["chasing_actor"][:, 0]
    assert set(assigned[assigned >= 0]) == {100}
    assert not (roles["chasing_actor"][corner, 0] == 101).any()


def test_short_actor_look_back_never_swaps_roles_and_missing_frames_stay_unknown():
    f = _chase()
    f.collective_keypoints_cm[65:70, 0, 0] = (
        2 * f.collective_keypoints_cm[65:70, 0, 3] - f.collective_keypoints_cm[65:70, 0, 0])
    f.valid[90:94] = False
    masks, roles = _roles(f)
    assert masks["chasing"][65:70, 0].all(), "a short look-back does not end ongoing pursuit"
    assert not masks["chasing"][90:94].any()
    assert (roles["chasing_actor"][90:94] == -1).all()
    assigned = roles["chasing_actor"][:, 0]
    assert set(assigned[assigned >= 0]) == {100}


def test_sustained_chase_overlap_is_not_swallowed_by_approach_priority():
    f = audit.features(90, 2, reference_body_cm=7.918)
    time = np.arange(90) / f.fps
    f.centers_cm[:, 0, 0] = 60 * time
    f.centers_cm[:, 1, 0] = 18. + 57 * time
    audit.sync(f)
    audit.resync_kinematics(f)
    _pose(f)
    raw_approach = rules._open_approach_mask(f, 0, config_seed_parameters({}))
    raw_chase = rules._chase_mask(f, 0)
    assert (raw_approach & raw_chase).any(), "fixture must expose actual candidate overlap"
    masks, roles = _roles(f)
    assert masks["chasing"].any()
    assert not (masks["approach"] & masks["chasing"]).any()
    selected = masks["chasing"][:, 0]
    assert (roles["chasing_actor"][selected, 0] == 100).all()


def test_role_collection_is_optional_and_does_not_change_any_mask():
    f = _approach()
    old_values = deepcopy(vars(f))
    masks, roles = _roles(f)
    baseline = rules.pair_rule_masks(f, config_seed_parameters({}))
    assert masks.keys() == baseline.keys()
    for key in masks:
        np.testing.assert_array_equal(masks[key], baseline[key])
    assert set(roles) == {"approach_actor", "approach_target", "chasing_actor", "chasing_target"}
    for key, before in old_values.items():
        if isinstance(before, np.ndarray):
            np.testing.assert_array_equal(getattr(f, key), before)


def test_observed_head_and_body_vectors_use_separate_real_anatomical_endpoints():
    from mouse_behavior.evaluation.pair_roles import observed_orientations

    f = _approach()
    f.heading_unit[:] = -1.
    measured = observed_orientations(f)
    np.testing.assert_allclose(measured.head_unit[:, 0], np.tile([1., 0.], (60, 1)))
    np.testing.assert_allclose(measured.body_unit[:, 0], np.tile([1., 0.], (60, 1)))
    f.collective_keypoints_cm[20, 0, 0] = f.collective_keypoints_cm[20, 0, 3]
    measured = observed_orientations(f)
    assert np.isnan(measured.head_unit[20, 0]).all(), "zero-length head is unknown"
    assert np.isfinite(measured.body_unit[20, 0]).all(), "body and head have separate evidence"


def test_single_missing_measured_heading_is_not_filled_from_neighboring_roles():
    f = _approach()
    f.collective_keypoints_cm[20, 0, 3] = np.nan
    masks, roles = _roles(f)
    assert masks["approach"][20, 0], "closing behavior may continue with unknown direction"
    assert roles["approach_actor"][20, 0] == -1
    assert roles["approach_target"][20, 0] == -1
    assert roles["approach_actor"][19, 0] == roles["approach_actor"][21, 0] == 100


def test_roles_can_be_added_to_existing_masks_without_rerunning_attack(monkeypatch):
    f = _approach()
    masks, expected = _roles(f)

    def no_attack(*args, **kwargs):
        raise AssertionError("existing mask role annotation must not rerun Attack")

    monkeypatch.setattr(rules, "_attack_action_mask", no_attack)
    actual = rules.pair_rule_roles(f, config_seed_parameters({}), masks=masks)
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key])
