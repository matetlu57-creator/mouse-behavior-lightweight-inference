"""Behavior counterexamples for deceleration, departure duration and spatial density."""

import numpy as np
import pytest

from tests.regression import test_heuristics_v16_audit as audit
from tests.regression import test_heuristics_v18_behavior_specific as v18
from tests.regression import test_group_evidence_seconds_v17 as fixtures
from mouse_behavior.evaluation import confirmed_rules as social
from mouse_behavior.evaluation import group_event_rules as group
from mouse_behavior.behavior.individual_rules import (
    DEFAULT_INDIVIDUAL_CONFIG, _positive_motion_dominance,
)


@pytest.mark.parametrize("long_seed", [False, True])
def test_attack_trigger_can_slow_below_trigger_speed_and_a_long_trigger_is_not_discarded(monkeypatch, long_seed):
    f = audit.features(90, 2)
    audit.set_centers(f, [[0., 0.], [6., 0.]])
    audit.sync(f)
    f.collective_keypoints_cm = np.repeat(f.centers_cm[:, :, None, :], 7, axis=2)
    stop_seed = 40 if long_seed else 11
    stop = stop_seed + 15

    def measured(positions, *args, **kwargs):
        velocity = np.zeros((90, 2))
        acceleration = np.zeros((90, 2))
        if positions[0, 0] < 3:
            velocity[5:stop_seed, 0] = 20.
            acceleration[5:stop_seed, 0] = 150.
            velocity[stop_seed:stop, 0] = np.linspace(9., 1., stop - stop_seed)
            acceleration[stop_seed:stop, 0] = -20.
        return velocity, acceleration

    monkeypatch.setattr(social, "_local_quadratic_point_kinematics", measured)
    mask = social._attack_action_mask(f, 0)
    assert mask[5:stop].all()
    assert not mask[:5].any() and not mask[stop:].any(), "an observed stop ends the event"


@pytest.mark.parametrize("stop_s,expected", [(4., False), (5., False), (5.4, True)])
def test_departure_actual_motion_must_be_strictly_longer_than_five_seconds(stop_s, expected):
    f = v18._departure(stop_s=stop_s)
    phase = group.dispersal_phase(
        f.pair_distance_cm, np.ones(len(f.valid), bool), np.ones(len(f.valid), bool),
        audit.confirmed_params(), f.fps, centers_cm=f.centers_cm, member_valid=f.valid,
    )
    assert bool(phase.any()) == expected


def test_departure_can_be_confirmed_later_than_the_exact_five_second_frame():
    f = audit.features(300, 3, reference_body_cm=7.918)
    t = np.arange(300) / f.fps
    for slot, x in [(0, -1.), (1, 1.)]:
        f.centers_cm[:, slot, 0] = x + np.sign(x) * np.maximum(np.minimum(t, 7.) - 5., 0.)
        f.centers_cm[:, slot, 1] = 2. * np.minimum(t, 7.)
    audit.sync(f)
    audit.resync_kinematics(f)
    candidates = group.group_rule_candidates(f, audit.confirmed_params(), (), behaviors=("dispersal",))
    assert candidates, "two departures remain close at +5 s, then separate while moving"
    assert all(c.target_ids == tuple(map(int, f.track_ids[:2])) for c in candidates)


def test_cluster_uses_half_second_spatial_stability_without_instantaneous_quiet_speed():
    f = fixtures._clustering(42)
    near = np.isfinite(f.pair_distance_cm) & (f.pair_distance_cm <= 10.)
    speed = np.full(f.valid.shape, 50.)
    result = group._clustering_process(
        f, (0, 1, 2), near, f.velocity_cm_s, speed,
        group.DEFAULT_GROUP_HEURISTIC_CONFIG, neighbor_cm=10., max_gap_s=5.,
    )
    assert result.evidence > 0, "spatially stable dense positions survive noisy instantaneous speeds"
    assert group.DEFAULT_GROUP_HEURISTIC_CONFIG.clustering_final_stability_s == .5


def test_cluster_still_rejects_spatially_unstable_formation():
    f = fixtures._clustering(100)
    f.centers_cm[:, 1, 0] = 2. + np.arange(100) % 2 * 40.
    audit.sync(f)
    near = np.isfinite(f.pair_distance_cm) & (f.pair_distance_cm <= 10.)
    result = group._clustering_process(
        f, (0, 1, 2), near, f.velocity_cm_s, np.full(f.valid.shape, 50.),
        group.DEFAULT_GROUP_HEURISTIC_CONFIG, neighbor_cm=10., max_gap_s=5.,
    )
    assert result.evidence == 0, "observed departures prevent a persistent dense state"


@pytest.mark.parametrize("fast_slot", [0, 1])
def test_together_speed_gate_constrains_each_participant(monkeypatch, fast_slot):
    f = audit.features(90, 2)
    audit.set_centers(f, [[0., 0.], [3., 0.]])
    audit.sync(f)
    cfg = social.DEFAULT_SOCIAL_HEURISTIC_CONFIG
    speeds = np.zeros(f.valid.shape)
    speeds[:, fast_slot] = cfg.vel_together_max_cm_s + .1
    monkeypatch.setattr(social, "body_motion", lambda *a, **k: (f.centers_cm, f.velocity_cm_s, speeds))
    for name in ("_open_approach_mask", "_attack_action_mask", "_chase_mask"):
        monkeypatch.setattr(social, name, lambda *a, **k: np.zeros(len(f.valid), bool))
    assert not social.pair_rule_masks(f, audit.confirmed_params())["together"].any()


def test_offline_head_motion_can_dominate_a_truly_still_rear_without_accepting_unknowns():
    numerator = np.array([6., 0., np.nan, 6., 6., 2.])
    denominator = np.array([0., 0., 0., np.nan, 2., 1.])
    np.testing.assert_array_equal(
        _positive_motion_dominance(numerator, denominator, 3.),
        [True, False, False, False, True, False],
    )
    assert not DEFAULT_INDIVIDUAL_CONFIG.enable_posture_screen


def test_return_before_confirmation_splits_departure_and_cannot_be_pause_bridged():
    f = v18._departure(frames=241, stop_s=7.)
    f.centers_cm[90, :2] = f.centers_cm[0, :2]
    audit.sync(f)
    audit.resync_kinematics(f)
    phase = group._departure_process_mask(
        f.centers_cm, f.valid, 0, 180, (0, 1), f.fps,
        minimum_outward_speed_cm_s=.11536234840750695,
        pause_max_s=.3998359188212399, minimum_duration_s=5.,
        velocity_cm_s=f.velocity_cm_s,
    )
    assert not phase.any(), "two episodes shorter than5s cannot be combined across a real return"


def test_departure_onset_can_precede_the_third_members_first_in_origin_observation():
    f = v18._departure()
    f.valid[:5, 2] = False
    audit.sync(f)
    audit.resync_kinematics(f)
    near = np.isfinite(f.pair_distance_cm) & (f.pair_distance_cm <= 10.)
    compact, enough, observed = group._geometry(f, (0, 1, 2), near)
    phase, leaving, checkpoint = group._dispersal_process(
        f.pair_distance_cm, enough, compact, audit.confirmed_params(), f.fps,
        centers_cm=f.centers_cm, member_valid=observed,
        member_velocity_cm_s=f.velocity_cm_s,
    )
    assert phase.any() and leaving == (0, 1)
    assert 5 * f.fps < checkpoint <= 5 + 5 * f.fps


def test_attack_continuation_preserves_existing_fast_forward_evidence(monkeypatch):
    f = audit.features(60, 2)
    audit.set_centers(f, [[0., 0.], [6., 0.]])
    audit.sync(f)
    f.collective_keypoints_cm = np.repeat(f.centers_cm[:, :, None, :], 7, axis=2)

    def measured(positions, *args, **kwargs):
        velocity = np.zeros((60, 2))
        acceleration = np.zeros((60, 2))
        if positions[0, 0] < 3:
            velocity[5:26, 0] = 20.
            acceleration[5:11, 0] = 150.
            acceleration[11:26, 0] = 5.
        return velocity, acceleration

    monkeypatch.setattr(social, "_local_quadratic_point_kinematics", measured)
    assert social._attack_action_mask(f, 0)[5:26].all()


def test_subthreshold_constant_speed_without_measured_slowing_does_not_extend_attack(monkeypatch):
    f = audit.features(60, 2)
    audit.set_centers(f, [[0., 0.], [6., 0.]])
    audit.sync(f)
    f.collective_keypoints_cm = np.repeat(f.centers_cm[:, :, None, :], 7, axis=2)

    def measured(positions, *args, **kwargs):
        velocity = np.zeros((60, 2))
        acceleration = np.zeros((60, 2))
        if positions[0, 0] < 3:
            velocity[5:11, 0] = 20.
            acceleration[5:11, 0] = 150.
            velocity[11:20, 0] = 1.
        return velocity, acceleration

    monkeypatch.setattr(social, "_local_quadratic_point_kinematics", measured)
    mask = social._attack_action_mask(f, 0)
    assert mask[5:11].all() and not mask[11:].any()
