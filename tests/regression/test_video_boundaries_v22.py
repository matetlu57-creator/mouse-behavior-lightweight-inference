"""Video-audited dropout anchors and progressive departure confirmation."""
from dataclasses import replace

import numpy as np

from tests.regression import test_heuristics_v16_audit as audit
from mouse_behavior.evaluation import confirmed_rules as social
from mouse_behavior.evaluation import group_event_rules as group


def test_video_audited_gap_limit_is_explicit_and_retains_two_same_target_cycles():
    cfg = social.DEFAULT_SOCIAL_HEURISTIC_CONFIG
    assert cfg.attack_box_gap_max_s == 28. / 29.967457227456368
    assert cfg.attack_box_reappearance_min_cycles == 2
    assert social.policy_receipt()['boundary_parameter_revision'] == '2026-10-18-v22-video-audited-loss-cohort'


def heavy_fixture():
    f = audit.features(90, 2)
    for slot in range(2):
        f.centers_cm[:, slot, 0] = 20. * np.arange(90) / f.fps + slot * 6.
    audit.sync(f)
    audit.resync_kinematics(f)
    f.collective_boxes_cm = np.ones((90, 2, 4))
    for low, high in ((5, 33), (35, 39)):
        f.collective_boxes_cm[low:high, 1] = np.nan
        f.valid[low:high, 1] = False
    f.valid[39, 0] = False
    return f


def test_repeated_same_target_cycles_use_bounded_joint_anchors_not_adjacent_valid_rows():
    f = heavy_fixture()
    channels = {}
    mask = social._attack_action_mask(f, 0, channels=channels)
    assert mask[4:41].all(), 'measured pair anchors at 4,33,34,40 enclose two target dropouts'
    assert not mask[:4].any() and not mask[41:].any(), 'no whole-clip padding'
    assert not channels['light_observed_acceleration'].any()
    assert channels['heavy_repeated_target_boxes'][~f.valid.all(axis=1)].any()


def test_one_long_dropout_stays_unknown_and_cycle_cap_is_enforced():
    f = heavy_fixture()
    f.collective_boxes_cm[35:39, 1] = 1.
    assert not social._attack_action_mask(f, 0).any(), 'single-gap route remains retired'
    f = heavy_fixture()
    strict = replace(social.DEFAULT_SOCIAL_HEURISTIC_CONFIG, attack_box_gap_max_s=.5)
    assert not social._attack_action_mask(f, 0, strict).any()


def test_window_anchors_cannot_jump_over_observed_separation():
    f = heavy_fixture()
    # Both IDs are observed and far at 34, contradicting the repeated episode.
    f.pair_distance_cm[34, 0] = 20.
    assert not social._attack_action_mask(f, 0).any()


def test_departure_keeps_later_confirmed_members_without_union_of_incompatible_events():
    f = audit.features(300, 3, reference_body_cm=7.918)
    time = np.minimum(np.arange(300) / f.fps, 9.)
    f.centers_cm[:, 0, 0] = -1. - 2. * time
    f.centers_cm[:, 1, 0] = 1. + 2. * time
    f.centers_cm[:, 2, 1] = 1. + .5 * time
    audit.sync(f)
    audit.resync_kinematics(f)
    candidates = group.group_rule_candidates(f, audit.confirmed_params(), (), behaviors=('dispersal',))
    sets = {c.target_ids for c in candidates}
    assert tuple(map(int, f.track_ids[:2])) in sets, 'earlier departure remains a candidate'
    assert tuple(map(int, f.track_ids)) in sets, 'later real departure must not be lost to first-success return'
    assert len({(c.target_ids, c.frame_intervals) for c in candidates}) == len(candidates)
    # A stationary in-origin member may not be copied into a departing subset.
    f.centers_cm[:, 2, 1] = 1.
    audit.sync(f)
    audit.resync_kinematics(f)
    candidates = group.group_rule_candidates(f, audit.confirmed_params(), (), behaviors=('dispersal',))
    assert candidates and all(int(f.track_ids[2]) not in c.target_ids for c in candidates)
