"""Observed membership, degree-one endpoint geometry and channel-specific evidence."""
import numpy as np
import pytest

from tests.regression import test_heuristics_v16_audit as audit
from mouse_behavior.evaluation import confirmed_rules as social
from mouse_behavior.evaluation import group_event_rules as group


def departure(frames=330, members=4):
    f = audit.features(frames, members, reference_body_cm=7.918)
    time = np.arange(frames) / f.fps
    for slot in range(members):
        join = slot * .6
        f.valid[time < join, slot] = False
        f.centers_cm[:, slot, 0] = (-1 if slot % 2 == 0 else 1) * (
            1. + 1.5 * np.maximum(np.minimum(time, 8.5) - join, 0.))
        f.centers_cm[:, slot, 1] = .5 * slot
    audit.sync(f)
    audit.resync_kinematics(f)
    return f


def test_disperse_accepts_continuously_appearing_ids_with_individual_origin_observations():
    f = departure()
    candidates = group.group_rule_candidates(f, audit.confirmed_params(), (), behaviors=('dispersal',))
    assert any(c.target_ids == tuple(map(int, f.track_ids)) for c in candidates)
    assert all(c.evidence > 5. for c in candidates)


@pytest.mark.parametrize('points,expected', [
    ([[0, 0], [2, 0], [10, 0], [12, 0]], True),
    ([[0, 0], [2, 0], [4, 0], [12, 0]], False),
    ([[0, 0], [3, 0], [6, 0]], True),
    ([[0, 0], [1, 0], [2, 0]], False),
])
def test_departure_final_close_pairs_form_a_matching(points, expected):
    assert group._dispersal_endpoint_separated(np.asarray(points, float)) is expected


def test_late_id_first_observed_far_from_origin_cannot_join_departure():
    f = departure()
    f.centers_cm[:, 3, 1] += 40.
    audit.sync(f)
    audit.resync_kinematics(f)
    candidates = group.group_rule_candidates(f, audit.confirmed_params(), (), behaviors=('dispersal',))
    assert all(int(f.track_ids[3]) not in c.target_ids for c in candidates)


def test_already_present_dense_group_without_arrival_is_not_clustering():
    f = audit.features(60, 3)
    audit.set_centers(f, [[0, 0], [2, 0], [4, 0]])
    audit.sync(f)
    near = f.pair_distance_cm <= 10.
    result = group._clustering_process(
        f, (0, 1, 2), near, np.full_like(f.velocity_cm_s, np.nan),
        np.full_like(f.speed_cm_s, np.nan), group.DEFAULT_GROUP_HEURISTIC_CONFIG,
        neighbor_cm=10., max_gap_s=5.)
    assert result.evidence == 0.
    assert result.frames == ()


def test_cluster_does_not_hold_an_unobserved_member_as_spatial_evidence():
    f = audit.features(60, 3)
    audit.set_centers(f, [[0, 0], [2, 0], [4, 0]])
    f.valid[10:50, 2] = False
    audit.sync(f)
    result = group._clustering_process(
        f, (0, 1, 2), f.pair_distance_cm <= 10., f.velocity_cm_s,
        f.speed_cm_s, group.DEFAULT_GROUP_HEURISTIC_CONFIG, neighbor_cm=10., max_gap_s=5.)
    assert result.evidence == 0., 'each visible stable run is shorter than 0.5s'


def test_cluster_public_api_rejects_a_long_connected_chain_without_shared_density():
    f = audit.features(60, 4)
    audit.set_centers(f, [[0, 0], [10, 0], [20, 0], [30, 0]])
    audit.sync(f)
    candidates = group.group_rule_candidates(f, audit.confirmed_params(), (), behaviors=('social_clustering',))
    assert not any(c.target_ids == tuple(map(int, f.track_ids)) for c in candidates)
    audit.set_centers(f, [[0, 0], [5, 0], [10, 0], [15, 0]])
    audit.sync(f)
    candidates = group.group_rule_candidates(f, audit.confirmed_params(), (), behaviors=('social_clustering',))
    assert not candidates, "a pre-existing chain has no measured arriving actor"
    assert np.ptp(f.centers_cm[0, :, 0]) > 10., 'neighbor radius is not the group diameter'


def test_light_attack_uses_directed_acceleration_without_point_speed_floor(monkeypatch):
    f = audit.features(60, 2)
    audit.set_centers(f, [[0, 0], [6, 0]])
    audit.sync(f)
    f.collective_keypoints_cm = np.repeat(f.centers_cm[:, :, None, :], 7, axis=2)
    def measured(positions, *args, **kwargs):
        velocity, acceleration = np.zeros((60, 2)), np.zeros((60, 2))
        if positions[0, 0] < 3:
            velocity[10:20, 0] = .5
            acceleration[10:20, 0] = 150.
        return velocity, acceleration
    monkeypatch.setattr(social, '_local_quadratic_point_kinematics', measured)
    assert social._attack_action_mask(f, 0)[10:20].all()


def test_heavy_attack_needs_repeated_target_box_cycles_with_directed_actor_anchors():
    f = audit.features(90, 2)
    time = np.arange(90) / f.fps
    for slot in range(2):
        f.centers_cm[:, slot, 0] = 20. * time + slot * 6.
    audit.sync(f)
    audit.resync_kinematics(f)
    f.collective_boxes_cm = np.ones((90, 2, 4))
    f.collective_boxes_cm[20:25, 1] = np.nan
    f.collective_boxes_cm[32:37, 1] = np.nan
    channels = {}
    positive = social._attack_action_mask(f, 0, channels=channels)
    assert positive[19:38].all()
    assert channels['heavy_repeated_target_boxes'][19:38].all()
    assert not channels['light_observed_acceleration'].any()
    f.collective_boxes_cm[32:37, 1] = 1.
    assert not social._attack_action_mask(f, 0).any(), 'one cycle is not repeated disappearance'
    f.collective_boxes_cm[32:37, 1] = np.nan
    f.velocity_cm_s[:, 0] *= -1.
    f.velocity_cm_s[:, 1] = 0.
    f.speed_cm_s[:, 1] = 0.
    assert not social._attack_action_mask(f, 0).any(), 'bbox flicker alone has no attack direction'


def test_alternating_one_cycle_per_target_is_not_two_cycles_for_one_target():
    f = audit.features(90, 2)
    for slot in range(2):
        f.centers_cm[:, slot, 0] = 20. * np.arange(90) / f.fps + slot * 6.
    audit.sync(f)
    audit.resync_kinematics(f)
    f.collective_boxes_cm = np.ones((90, 2, 4))
    f.collective_boxes_cm[20:25, 0] = np.nan
    f.collective_boxes_cm[32:37, 1] = np.nan
    assert not social._attack_action_mask(f, 0).any()


def test_heavy_target_box_cycles_allow_actor_occlusion_inside_same_id_observed_anchors():
    f = audit.features(90, 2)
    for slot in range(2):
        f.centers_cm[:, slot, 0] = 20. * np.arange(90) / f.fps + slot * 6.
    audit.sync(f)
    audit.resync_kinematics(f)
    f.collective_boxes_cm = np.ones((90, 2, 4))
    for low, high in ((20, 25), (32, 37)):
        f.collective_boxes_cm[low:high] = np.nan
        f.valid[low:high] = False
    channels = {}
    result = social._attack_action_mask(f, 0, channels=channels)
    assert result[19:38].all()
    assert channels['heavy_repeated_target_boxes'][19:38].all()
    assert not channels['light_observed_acceleration'].any()


def test_chase_counts_only_jointly_visible_motion_and_no_cross_gap_travel():
    f = audit.features(150, 2)
    for slot in range(2):
        f.centers_cm[:, slot, 0] = 60. * np.arange(150) / f.fps + slot * 20.
    audit.sync(f)
    audit.resync_kinematics(f)
    f.valid[40:46] = False
    mask = social._chase_mask(f, 0)
    assert mask[:40].any() and mask[46:].any()
    assert not mask[40:46].any()
    # Only 0.6s actually observed: no one-second support from a missing interval.
    f.valid[:] = False
    f.valid[:9] = f.valid[99:108] = True
    assert not social._chase_mask(f, 0).any()
    # Two one-second runs, but all 40cm travel is inside the missing interval.
    for slot in range(2):
        f.centers_cm[:40, slot, 0] = slot * 20.
        f.centers_cm[46:, slot, 0] = slot * 20. + 40.
    f.valid[:] = True
    f.valid[40:46] = False
    assert not social._chase_mask(f, 0).any()


def test_chase_visible_fragments_can_sum_support_without_labeling_unknowns():
    f = audit.features(90, 2)
    for slot in range(2):
        f.centers_cm[:, slot, 0] = 60. * np.arange(90) / f.fps + slot * 20.
    audit.sync(f)
    audit.resync_kinematics(f)
    f.valid[12:21] = False
    f.valid[33:42] = False
    f.valid[54:] = False
    mask = social._chase_mask(f, 0)
    assert mask.sum() >= 30
    assert not mask[~f.valid.all(axis=1)].any()


def test_pose_chase_smoothing_and_lag_do_not_read_across_a_joint_observation_gap():
    f = audit.features(120, 2)
    for slot in range(2):
        f.centers_cm[:, slot, 0] = np.arange(120) / f.fps * 20. + slot * 20.
    audit.sync(f)
    f.collective_keypoints_cm = np.repeat(f.centers_cm[:, :, None, :], 7, axis=2)
    f.valid[40:46, 1] = False
    before = social._coobserved_chase_motion(f, 0, social.body_motion(f, social.CONFIRMED_POLICY))
    f.centers_cm[:40, :, 0] -= 1000.
    after = social._coobserved_chase_motion(f, 0, social.body_motion(f, social.CONFIRMED_POLICY))
    for first, second in zip(before, after):
        np.testing.assert_allclose(first[46:], second[46:], equal_nan=True)
    assert not np.isfinite(after[2][40:46]).any()


def test_departure_smoothed_velocity_cannot_read_across_a_members_missing_interval():
    f = departure()
    f.valid[100:106, 0] = False
    before = group._observed_departure_velocity(f)
    f.centers_cm[:100, 0] += 1000.
    after = group._observed_departure_velocity(f)
    np.testing.assert_allclose(before[106:, 0], after[106:, 0], equal_nan=True)
    assert not np.isfinite(after[100:106, 0]).any()


def test_short_observed_runs_supply_real_motion_but_singletons_remain_unknown():
    from mouse_behavior.evaluation.screening_signals import observed_run_motion
    for frames in (2, 3, 6, 10, 30):
        positions = np.c_[np.arange(frames) * .5, np.zeros(frames)]
        _, velocity, speed = observed_run_motion(positions, 30., social.CONFIRMED_POLICY)
        assert np.isfinite(speed).all() and (speed > 0).all()
        assert (velocity[:, 0] > 0).all() and (velocity[:, 1] == 0).all()
    _, velocity, speed = observed_run_motion(np.array([[1., 0.]]), 30., social.CONFIRMED_POLICY)
    assert np.isnan(velocity).all() and np.isnan(speed).all()


def test_pure_translation_of_an_existing_group_is_not_an_arrival_process():
    f = audit.features(90, 3)
    audit.set_centers(f, [[0, 0], [2, 0], [4, 0]])
    for frame in range(90):
        f.centers_cm[frame] += [frame * 2., frame * .5]
    audit.sync(f)
    audit.resync_kinematics(f)
    candidates = group.group_rule_candidates(f, audit.confirmed_params(), (), behaviors=('social_clustering',))
    assert not candidates, "translation alone has no seed-relative distance drop"
