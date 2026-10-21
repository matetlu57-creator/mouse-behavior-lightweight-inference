"""Counterexamples for per-process thresholds and checkpoint/event boundaries."""
from dataclasses import replace

import numpy as np
import pytest

from tests.regression import test_heuristics_v16_audit as audit
from mouse_behavior.evaluation import confirmed_rules as social
from mouse_behavior.evaluation import group_event_rules as group
from mouse_behavior.evaluation.group_heuristic_config import DEFAULT_GROUP_HEURISTIC_CONFIG as config
from mouse_behavior.evaluation.retrospective_thresholds_v18 import (
    ACTIVE_DATASET_SAMPLE_COUNT, INDEPENDENT_VALIDATION, SOCIAL_FITTED,
)
from mouse_behavior.evaluation.social_heuristic_config import DEFAULT_SOCIAL_HEURISTIC_CONFIG
from mouse_behavior.evaluation.retrospective_thresholds_v20 import SOCIAL_FITTED as V20_SOCIAL_FITTED


def test_profile_is_retrospective_and_posture_remains_disabled():
    from mouse_behavior.behavior.individual_rules import DEFAULT_INDIVIDUAL_CONFIG
    receipt = social.policy_receipt()
    assert ACTIVE_DATASET_SAMPLE_COUNT == 755 and not INDEPENDENT_VALIDATION
    assert not DEFAULT_INDIVIDUAL_CONFIG.enable_posture_screen
    assert receipt["fit_scope"] == "full_dataset_retrospective_with_user_requested_v28_overrides"
    assert receipt["exclusive_pair_priority"] == ["approach", "together", "attack", "chasing", "head_contact", "tail_contact"]
    assert all(getattr(DEFAULT_SOCIAL_HEURISTIC_CONFIG, name) == value
               for name, value in {**SOCIAL_FITTED, **V20_SOCIAL_FITTED}.items()
               if name != "vel_together_max_cm_s")


@pytest.mark.parametrize("distance_offset,speed_offset,expected", [
    (-.01, -.1, True), (0., -.1, False), (-.01, 0., False),
])
def test_current_together_limits_are_strict(monkeypatch, distance_offset, speed_offset, expected):
    cfg = DEFAULT_SOCIAL_HEURISTIC_CONFIG
    f = audit.features(90, 2)
    speed = cfg.vel_together_max_cm_s + speed_offset
    f.centers_cm[:, 0, 0] = np.arange(90) / f.fps * speed
    f.centers_cm[:, 1, 0] = f.centers_cm[:, 0, 0] + cfg.dist_together_max_cm + distance_offset
    audit.sync(f)
    f.pair_distance_cm[:] = cfg.dist_together_max_cm + distance_offset
    f.speed_cm_s[:] = speed
    for name in ("_open_approach_mask", "_chase_mask", "_attack_action_mask"):
        monkeypatch.setattr(social, name, lambda features, *args, **kwargs: np.zeros(len(features.valid), bool))
    assert bool(social.pair_rule_masks(f, audit.confirmed_params())["together"].any()) == expected


def _departure(frames=541, stop_s=8.0, late_third=False):
    f = audit.features(frames, 3, reference_body_cm=7.918)
    time = np.arange(frames) / f.fps
    motion = np.minimum(time, stop_s)
    f.centers_cm[:, 0, 0] = -1.0 - 2.0 * motion
    f.centers_cm[:, 1, 0] = 1.0 + 2.0 * motion
    if late_third:
        f.centers_cm[:, 2, 1] = np.maximum(time - 6.0, 0.0) * 3.0
    audit.sync(f)
    audit.resync_kinematics(f)
    return f


def test_disperse_confirmation_is_not_the_motion_end_or_the_clip_end():
    f = _departure()
    phase = group.dispersal_phase(
        f.pair_distance_cm, np.ones(len(f.valid), bool), np.ones(len(f.valid), bool),
        audit.confirmed_params(), f.fps, centers_cm=f.centers_cm, member_valid=f.valid,
    )
    support = np.flatnonzero(phase)
    assert len(support) > 5 * f.fps
    assert support[-1] / f.fps == pytest.approx(8.0, abs=1 / f.fps)
    assert not phase[round(9 * f.fps):].any(), "quiet tail cannot inflate the event"


def test_disperse_each_output_set_is_fixed_at_its_own_confirmation():
    f = _departure(late_third=True)
    confirmations = []
    _, first_members, first_checkpoint = group._dispersal_process(
        f.pair_distance_cm, np.ones(len(f.valid), bool), np.ones(len(f.valid), bool),
        audit.confirmed_params(), f.fps, centers_cm=f.centers_cm, member_valid=f.valid,
        member_velocity_cm_s=group._observed_departure_velocity(f),
        confirmations=confirmations,
    )
    assert first_members == (0, 1), 'third member has not departed at first confirmation'
    assert first_checkpoint / f.fps < 6., 'the third member is still stationary then'
    complete = [(phase, checkpoint) for phase, members, checkpoint in confirmations if members == (0, 1, 2)]
    assert complete, 'a later independent checkpoint may admit a real late departure'
    assert all(checkpoint / f.fps >= 6. + 4. / 3. for _, checkpoint in complete)
    # Compatibility callers that request one process still get the first set.
    _, legacy_members, legacy_checkpoint = group._dispersal_process(
        f.pair_distance_cm, np.ones(len(f.valid), bool), np.ones(len(f.valid), bool),
        audit.confirmed_params(), f.fps, centers_cm=f.centers_cm, member_valid=f.valid,
        member_velocity_cm_s=group._observed_departure_velocity(f),
    )
    assert (legacy_members, legacy_checkpoint) == (first_members, first_checkpoint)
    actual = group.group_rule_candidates(f, audit.confirmed_params(), (), behaviors=("dispersal",))
    sets = {c.target_ids for c in actual}
    assert tuple(map(int, f.track_ids[:2])) in sets
    assert tuple(map(int, f.track_ids)) in sets
    assert all(c.evidence > 5.0 for c in actual)


def test_checkpoint_mode_is_explicitly_reproducible():
    f = _departure()
    old = group.dispersal_phase(
        f.pair_distance_cm, np.ones(len(f.valid), bool), np.ones(len(f.valid), bool),
        audit.confirmed_params(), f.fps, centers_cm=f.centers_cm, member_valid=f.valid,
        extend_actual_motion=False,
    )
    assert np.flatnonzero(old)[-1] == int(np.ceil(5 * f.fps))


def test_missing_pose_cannot_erase_observed_departure_motion():
    f = _departure()
    f.collective_keypoints_cm = None
    f.velocity_cm_s[:] = 0.0
    f.speed_cm_s[:] = 0.0
    actual = group.group_rule_candidates(f, audit.confirmed_params(), (), behaviors=("dispersal",))
    assert actual and all(c.evidence > 5.0 for c in actual)


def test_low_speed_coherence_cannot_replace_real_group_travel():
    f = audit.features(180, 3)
    for slot in range(3):
        f.centers_cm[:, slot, 0] = np.arange(180) / f.fps * .2 + 2 * slot
    audit.sync(f)
    velocity = np.zeros_like(f.centers_cm)
    velocity[..., 0] = .2
    speed = np.full(f.valid.shape, .2)
    params = replace(audit.confirmed_params(),
                     collective_active_speed_cm_s=config.locomotion_active_speed_cm_s,
                     group_locomotion_min_direction_similarity=config.locomotion_min_direction_similarity,
                     group_locomotion_max_distance_cm=config.locomotion_max_pair_distance_cm)
    assert group._synchronous_event(f, (0, 1, 2), velocity, speed, params,
                                   replace(config, locomotion_min_net_displacement_cm=0.0)) > 3.0
    assert group._synchronous_event(f, (0, 1, 2), velocity, speed, params, config) == 0.0


def test_group_parameter_changes_are_scoped_to_their_process(monkeypatch):
    f = audit.features(60, 3)
    audit.set_centers(f, [[0, 0], [12, 0], [24, 0]])
    audit.sync(f)
    seen = []
    def proposals(features, near, velocity, speed, params, incoming):
        seen.append((near.copy(), params.collective_active_speed_cm_s))
        return []
    monkeypatch.setattr(group, "_proposal_sets", proposals)
    params = audit.confirmed_params()
    group.group_rule_candidates(f, params, (), behaviors=("huddle",))
    group.group_rule_candidates(f, params, (), behaviors=("social_clustering",))
    group.group_rule_candidates(f, params, (), behaviors=("group_locomotion",))
    assert seen[0][0].any() and not seen[1][0].any()
    assert seen[0][1] == seen[1][1] == params.collective_active_speed_cm_s
    assert seen[2][1] == config.locomotion_active_speed_cm_s
    with pytest.raises(ValueError, match="Unknown group"):
        group.group_rule_candidates(f, params, (), behaviors=("invented",))


def test_point_kinematics_are_cached_per_prediction_and_never_across_clips(monkeypatch):
    f = audit.features(30, 3)
    audit.set_centers(f, [[0, 0], [2, 0], [4, 0]])
    f.collective_keypoints_cm = np.repeat(f.centers_cm[:, :, None, :], 7, axis=2)
    calls = []
    original = social._local_quadratic_point_kinematics
    def measured(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(social, "_local_quadratic_point_kinematics", measured)
    social.pair_rule_masks(f, audit.confirmed_params())
    first = len(calls)
    assert first == 2 * len(f.track_ids), "each actor's nose/head fit is partner-independent"
    social.pair_rule_masks(f, audit.confirmed_params())
    assert len(calls) == 2 * first, "a later clip must not reuse an earlier clip's fit"
