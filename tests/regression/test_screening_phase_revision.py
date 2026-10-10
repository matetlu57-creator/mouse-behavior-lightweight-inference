"""The user's visual corrections constrain phases, not clip-wide label scores."""

import numpy as np
import pytest

from mouse_behavior.evaluation.exported_dataset import HeuristicParameters
from mouse_behavior.evaluation.screening_signals import ScreenPolicy, pair_phases, posture_evidence, supported_episodes, pursuit_similarity
from mouse_behavior.evaluation.temporal_screen import group_phase_masks, screen_record
from tests.regression.test_temporal_screen import features, record


def test_approach_checks_distance_at_exported_first_frame_not_earlier_lookback():
    f = features(n=40, fps=20.)
    f.speed_cm_s[:] = 10
    nearest = np.linspace(7.7, 0., 40)[:, None]
    f.pair_nose_nearest_left_to_right_cm = nearest
    f.pair_nose_nearest_right_to_left_cm = np.full_like(nearest, 100.)
    plans, _ = screen_record(record('Approach', f), f, HeuristicParameters())
    approaches = [w for w in plans if w.behavior == 'Approach']
    assert approaches
    assert all(nearest[w.start_frame, 0] >= 6.6875 for w in approaches)


def test_far_but_unobserved_prefix_cannot_qualify_a_near_approach_start():
    f = features(n=40, fps=20.)
    f.speed_cm_s[:] = 10
    nearest = np.linspace(8., 0., 40)[:, None]
    f.pair_nose_nearest_left_to_right_cm = nearest
    f.pair_nose_nearest_right_to_left_cm = np.full_like(nearest, 100.)
    f.valid[:8] = False
    plans, _ = screen_record(record('Approach', f), f, HeuristicParameters())
    assert not any(w.behavior == 'Approach' for w in plans)


@pytest.mark.parametrize('interruption', ['paused', 'departing'])
def test_approach_onset_recovery_cannot_cross_a_pause_or_departure(interruption):
    f = features(n=40, fps=20.)
    f.speed_cm_s[:] = 10
    nearest = np.linspace(7.7, 0., 40)[:, None]
    f.pair_nose_nearest_left_to_right_cm = nearest
    f.pair_nose_nearest_right_to_left_cm = np.full_like(nearest, 100.)
    if interruption == 'paused':
        f.speed_cm_s[5] = 0
    else:
        nearest[5] = 6.0  # The next frame moved away, not toward the target.
    phases = pair_phases(f, 0, HeuristicParameters(), ScreenPolicy())
    assert not phases['approach'].any()


def pose_features(n=120, fps=30.):
    f = features(n=n, mice=2, fps=fps)
    f.reference_body_cm = 10.7
    f.collective_keypoints_cm = np.broadcast_to(np.array([
        [10.7, 0], [9, 1], [9, -1], [7, 0], [3, 2], [3, -2], [0, 0]
    ]), (n, 2, 7, 2)).copy()
    f.collective_keypoints_cm[:, 1, :, 1] += 20
    f.centers_cm[:, :, :] = f.collective_keypoints_cm[:, :, 3]
    return f


def test_approach_must_stop_at_nose_to_torso_contact_not_only_head_or_tail():
    f = pose_features()
    f.collective_keypoints_cm[:, 1] = f.collective_keypoints_cm[:, 0] + [15, 0]
    # The other nose enters the hip/neck polygon, far from both head and tail.
    f.collective_keypoints_cm[60:, 1, 0] = [5., 0.]
    f.centers_cm[:, 0, 0] = np.linspace(-20, 20, len(f.valid))
    f.pair_distance_cm[:, 0] = np.linspace(20, 5, len(f.valid))
    nearest = np.linspace(15, 4, len(f.valid))[:, None]
    f.pair_nose_nearest_left_to_right_cm = nearest
    f.pair_nose_nearest_right_to_left_cm = nearest.copy()
    f.speed_cm_s[:] = 15
    phases = pair_phases(f, 0, HeuristicParameters(), ScreenPolicy())
    assert phases['contact'][60:].all()
    candidates, _ = screen_record(record('Approach', f), f, HeuristicParameters())
    assert not any(p.behavior == 'Approach' and p.end_frame > 60 for p in candidates)


def test_together_excludes_active_touch_even_when_attack_speed_not_reached():
    f = pose_features()
    f.collective_keypoints_cm[:, 1] = f.collective_keypoints_cm[:, 0] + [6, 0]
    f.centers_cm[:] = f.collective_keypoints_cm[:, :, 3]
    f.pair_distance_cm[:] = 6
    f.nose_speed_cm_s = np.full_like(f.speed_cm_s, 6.)
    original = f.collective_keypoints_cm.copy()
    wave = np.sin(np.arange(len(f.valid)) / f.fps * 2 * np.pi * 2)
    f.collective_keypoints_cm[:, :, :3, 0] += wave[:, None, None]
    result, _ = screen_record(record('Together', f), f, HeuristicParameters())
    assert result == []
    f.nose_speed_cm_s[:] = 0
    f.collective_keypoints_cm[:] = original
    result, _ = screen_record(record('Together', f), f, HeuristicParameters())
    assert result and result[0].behavior == 'Together'


def test_half_standing_rhythmic_head_grooming_overrides_rearing():
    f = pose_features()
    # Shortened body plus coherent repetitive forward/back head movement.
    f.collective_keypoints_cm[:, 0, :4, 0] -= 3.2
    wave = np.sin(np.arange(len(f.valid)) / f.fps * 2 * np.pi * 3) * .8
    f.collective_keypoints_cm[:, 0, :3, 0] += wave[:, None]
    f.centers_cm[:] = 0
    masks, _ = posture_evidence(f, ScreenPolicy())
    assert masks['grooming'][20:-20, 0].any()
    assert not (masks['grooming'][:, 0] & masks['rearing'][:, 0]).any()


def test_rearing_has_small_axial_head_motion_and_static_head_turn_is_not_grooming():
    f = pose_features()
    f.collective_keypoints_cm[:, 0, :4, 0] -= 3.2
    f.centers_cm[:] = 0
    masks, _ = posture_evidence(f, ScreenPolicy())
    assert not masks['rearing'][:, 0].any()
    f.collective_keypoints_cm[:15, 0, :4, 0] += 3.2
    f.collective_keypoints_cm[-15:, 0, :4, 0] += 3.2
    masks, _ = posture_evidence(f, ScreenPolicy())
    assert masks['rearing'][35:-35, 0].all()
    assert not masks['grooming'][:, 0].any()
    f = pose_features()
    f.collective_keypoints_cm[:, 0, :3, 1] += (np.sin(np.arange(len(f.valid)) / 8) * 2)[:, None]
    f.centers_cm[:] = 0
    masks, _ = posture_evidence(f, ScreenPolicy())
    assert not masks['grooming'][:, 0].any()
    assert not masks['rearing'][:, 0].any()


def test_back_grooming_requires_repeated_motion_in_back_region():
    f = pose_features()
    f.collective_keypoints_cm[:, 0, 0] = [3, .5]
    f.collective_keypoints_cm[:, 0, 1:3, 0] = 4
    f.collective_keypoints_cm[:, 0, :3, 1] += (np.sin(np.arange(len(f.valid)) / 2) * .8)[:, None]
    f.collective_keypoints_cm[:, 0, :3, 0] += (np.sin(np.arange(len(f.valid)) / 2) * .5)[:, None]
    f.centers_cm[:] = 0
    masks, _ = posture_evidence(f, ScreenPolicy())
    assert masks['grooming'][20:-20, 0].any()
    f.collective_keypoints_cm[:, 0] = f.collective_keypoints_cm[60, 0]
    masks, _ = posture_evidence(f, ScreenPolicy())
    assert not masks['grooming'][:, 0].any()


def test_plain_contact_is_not_eliminated_by_nose_acceleration_jitter():
    f = features(n=40)
    f.pair_nose_head_cm[:] = 1
    f.nose_acceleration_cm_s2[:] = 1000
    candidates, _ = screen_record(record('Snout-head_contact', f), f, HeuristicParameters())
    assert candidates and all(p.behavior == 'Snout-head_contact' for p in candidates)


def test_event_support_recovers_between_action_seeds_without_calm_lead_or_tail():
    seed = np.zeros(60, bool)
    seed[[10, 20, 30, 40]] = True
    support = np.zeros(60, bool)
    support[5:45] = True
    mask = supported_episodes(seed, support, 3)
    assert np.array_equal(np.flatnonzero(mask), np.arange(10, 41))
    support[24:27] = False  # a real unsupported interruption must split events
    mask = supported_episodes(seed, support, 3)
    assert not mask[24:27].any()


def test_approach_stops_before_visible_clearance_and_never_contains_departure():
    f = features(n=60)
    f.mouse_width_cm = 4.15
    f.speed_cm_s[:] = 10
    f.pair_nose_head_cm[:] = 100
    nearest = np.r_[np.linspace(12, 2.8, 30), np.linspace(2.8, 10, 30)][:, None]
    f.pair_nose_nearest_left_to_right_cm = nearest
    f.pair_nose_nearest_right_to_left_cm = np.full_like(nearest, 100)
    plans, _ = screen_record(record('Approach', f), f, HeuristicParameters(),
                             policy=ScreenPolicy(approach_stop_distance_cm=3.))
    approaches = [p for p in plans if p.behavior == 'Approach']
    assert approaches and max(p.end_frame for p in approaches) <= 30
    assert 27 <= max(p.end_frame for p in approaches) <= 30


def test_contact_mined_from_approach_source_starts_after_approach_and_touch_onset():
    f = features(n=60)
    f.speed_cm_s[:20, 0] = 10
    nearest = np.r_[np.linspace(12, 3, 20), np.ones(40)][:, None]
    f.pair_nose_nearest_left_to_right_cm = nearest
    f.pair_nose_nearest_right_to_left_cm = np.full_like(nearest, 100)
    f.pair_nose_head_cm[20:] = 1
    plans, _ = screen_record(record('Approach', f), f, HeuristicParameters())
    approaches = [p for p in plans if p.behavior == 'Approach']
    contacts = [p for p in plans if p.behavior == 'Snout-head_contact']
    assert approaches and contacts
    assert min(p.start_frame for p in contacts) >= 20
    assert min(p.start_frame for p in contacts) >= max(p.end_frame for p in approaches)


def test_observed_group_disconnection_is_never_filled_as_compact():
    f = features(n=60, mice=3)
    f.pair_distance_cm[:] = 1
    f.pair_distance_cm[25:28] = 100
    compact, _ = group_phase_masks(f, [0, 1, 2], HeuristicParameters())
    assert not compact[25:28].any()
    assert compact[:25].all() and compact[28:].all()


def test_group_actor_moving_away_is_a_hard_formation_boundary():
    f = features(n=60, mice=3)
    f.pair_distance_cm[:] = 100
    f.centers_cm[:, 0, 0] = 20
    f.centers_cm[:, 2, 0] = 1
    f.velocity_cm_s[:, 0, 0] = -10
    f.velocity_cm_s[25:28, 0, 0] = 10
    f.speed_cm_s[:, 0] = 10
    _, formation = group_phase_masks(f, [0, 1, 2], HeuristicParameters())
    assert not formation[25:28].any()
    assert formation[:25].all() and formation[28:].all()


def test_missing_motion_cannot_become_quiet_together_or_attack():
    f = features(n=40)
    f.speed_cm_s[:] = np.nan
    f.pair_nose_head_cm[:] = 1
    for label in ('Together', 'Attack'):
        plans, _ = screen_record(record(label, f), f, HeuristicParameters())
        assert plans == []


def test_surface_gap_two_cm_is_not_contact_but_is_beyond_approach_cutoff():
    f = pose_features(n=120)
    f.mouse_width_cm = 4.15
    f.collective_keypoints_cm[:, 1] = f.collective_keypoints_cm[:, 0] + [12.7, 0]
    f.centers_cm[:, 0, 0] = np.linspace(-20, 20, len(f.valid))
    phases = pair_phases(f, 0, HeuristicParameters(), ScreenPolicy())
    assert np.allclose(phases['nearest_surface_cm'], 2.)
    assert not phases['contact'].any()
    assert not phases['guarded_contact'].any()
    assert not phases['approach'].any()


def test_whole_forebody_sway_is_not_head_grooming():
    f = pose_features()
    f.centers_cm[:] = 0
    wave = np.sin(np.arange(len(f.valid)) / f.fps * 2 * np.pi * 3) * .8
    f.collective_keypoints_cm[:, 0, :4, 0] += wave[:, None]
    masks, _ = posture_evidence(f, ScreenPolicy())
    assert not masks['grooming'][:, 0].any()


def test_constant_curled_rest_has_no_upright_transition_evidence():
    f = pose_features()
    f.collective_keypoints_cm[:, 0, :4, 0] -= 3.2
    f.centers_cm[:] = 0
    masks, _ = posture_evidence(f, ScreenPolicy())
    assert not masks['rearing'][:, 0].any()


def test_screen_policy_rejects_invalid_boundary_configuration():
    import pytest
    for settings in ({'contact_guard_s': float('nan')}, {'surface_contact_width_fraction': -1},
                     {'rear_transition_ratio': 1}, {'motion_window_s': 0},
                     {'chase_support_speed_fraction': 0}, {'chase_support_speed_fraction': 1.1},
                     {'chase_ahead_cosine': 2}, {'chase_path_width_fraction': 100},
                     {'chase_heading_lag_s': .6}, {'approach_stop_distance_cm': 0}):
        with pytest.raises(ValueError):
            ScreenPolicy(**settings)


def test_pursuit_follows_the_same_corner_with_a_short_heading_delay():
    n, delay, fps = 80, 8, 20.
    leader_velocity = np.tile([30., 0.], (n, 1))
    leader_velocity[30:] = [0., 30.]
    leader_position = np.cumsum(leader_velocity/fps, axis=0)
    previous = np.maximum(0, np.arange(n)-delay)
    follower_velocity = leader_velocity[previous]
    follower_position = leader_position[previous].copy()
    follower_position[:delay, 0] -= (delay-np.arange(delay))*30/fps
    centers = np.stack((follower_position, leader_position), axis=1)
    velocity = np.stack((follower_velocity, leader_velocity), axis=1)
    speed = np.linalg.norm(velocity, axis=2)
    synchronous = (velocity[:, 0]*velocity[:, 1]).sum(axis=1)/900
    assert (synchronous[30:38] == 0).all()
    similarity = pursuit_similarity(centers, velocity, speed, 0, 1, fps, ScreenPolicy(), 4.15)
    assert (similarity[30:38] >= .707).all()
    # A pair traveling side by side has no leader ahead on the movement axis.
    centers[:, 0, 0] = np.arange(n)*30/fps
    centers[:, 0, 1] = 0
    centers[:, 1] = centers[:, 0] + [0., 12.]
    velocity[:] = [30., 0.]
    similarity = pursuit_similarity(centers, velocity, speed, 0, 1, fps, ScreenPolicy(), 4.15)
    assert not (similarity >= .707).any()


def test_default_formal_posture_exports_require_a_human_decision():
    f = features(mice=2)
    episodes = [{"track_id": 101, "candidate": "rearing", "start_frame": 0, "end_frame": 40,
                 "body_static_verified": True}]
    plans, _ = screen_record(record("Static", f, (100,)), f, HeuristicParameters(), posture_episodes=episodes)
    assert not any(w.behavior in {"Grooming", "Rearing"} for w in plans)
    audits = [{"track_id": "101", "observed_behavior": "rearing", "decision": "confirmed",
               "start_frame": "0", "end_frame": "40", "review_id": "known"}]
    plans, _ = screen_record(record("Static", f, (100,)), f, HeuristicParameters(),
                             posture_episodes=episodes, audit_rows=audits)
    assert any(w.behavior == "Rearing" and w.status == "human_confirmed" for w in plans)


@pytest.mark.parametrize("cutoff", [3., 3.5])
def test_approach_cutoff_has_no_extra_early_padding(cutoff):
    f = features(n=60, fps=20.)
    f.mouse_width_cm = 4.15
    f.speed_cm_s[:] = 10
    distance = np.linspace(12., 0., 60)[:, None]
    f.pair_nose_nearest_left_to_right_cm = distance
    f.pair_nose_nearest_right_to_left_cm = np.full_like(distance, 100.)
    phases = pair_phases(f, 0, HeuristicParameters(), ScreenPolicy(approach_stop_distance_cm=cutoff))
    cut = np.flatnonzero(distance[:, 0] <= cutoff)[0]
    last = min((cut-1, cut), key=lambda i: abs(distance[i, 0]-cutoff))
    assert phases['approach'][last]
    assert not phases['approach'][last+1:].any()
    assert abs(phases['nearest_surface_cm'][last]-cutoff) < .11


def test_chase_bridge_cannot_fill_a_real_stop_between_fast_seeds():
    f = features(n=60)
    f.speed_cm_s[:] = 30
    f.speed_cm_s[28:31] = 0
    phases = pair_phases(f, 0, HeuristicParameters(), ScreenPolicy())
    assert phases['chasing'][:28].all() and phases['chasing'][31:].all()
    assert not phases['chasing'][28:31].any()


def test_approach_that_never_reaches_three_cm_is_not_a_completed_export():
    f = features(n=40, fps=30.)
    f.speed_cm_s[:] = 10
    distance = np.linspace(12., 5., 40)[:, None]
    f.pair_nose_nearest_left_to_right_cm = distance
    f.pair_nose_nearest_right_to_left_cm = np.full_like(distance, 100.)
    plans, _ = screen_record(record("Approach", f), f, HeuristicParameters())
    assert not any(w.behavior == "Approach" for w in plans)


@pytest.mark.parametrize("cutoff", [3., 3.5])
def test_fast_approach_chooses_real_frame_nearest_cutoff_not_far_predecessor(cutoff):
    f = features(n=60, fps=30.)
    f.speed_cm_s[:] = 15
    distance = np.r_[np.linspace(15, 5.1, 25), cutoff-.16, np.linspace(2.7, 0, 34)][:, None]
    f.pair_nose_nearest_left_to_right_cm = distance
    f.pair_nose_nearest_right_to_left_cm = np.full_like(distance, 100.)
    plans, _ = screen_record(record("Approach", f), f, HeuristicParameters(),
                             policy=ScreenPolicy(approach_stop_distance_cm=cutoff))
    approaches = [w for w in plans if w.behavior == "Approach"]
    assert approaches and approaches[0].end_frame == 26
    assert abs(distance[approaches[0].end_frame-1, 0]-cutoff) < .5
    distance[25:, 0] = 1.
    plans, _ = screen_record(record("Approach", f), f, HeuristicParameters(),
                             policy=ScreenPolicy(approach_stop_distance_cm=cutoff))
    assert not any(w.behavior == "Approach" for w in plans)


def test_default_approach_cutoff_increases_distance_by_half_a_cm():
    assert ScreenPolicy().approach_stop_distance_cm == 3.5
    assert ScreenPolicy().approach_endpoint_tolerance_cm == .5
    f = features(n=60, fps=20.)
    f.speed_cm_s[:] = 10
    distance = np.linspace(12., 0., 60)[:, None]
    f.pair_nose_nearest_left_to_right_cm = distance
    f.pair_nose_nearest_right_to_left_cm = np.full_like(distance, 100.)
    current, _ = screen_record(record("Approach", f), f, HeuristicParameters())
    prior, _ = screen_record(record("Approach", f), f, HeuristicParameters(),
                             policy=ScreenPolicy(approach_stop_distance_cm=3.))
    current = next(w for w in current if w.behavior == "Approach")
    prior = next(w for w in prior if w.behavior == "Approach")
    assert current.end_frame < prior.end_frame
    assert 3. <= distance[current.end_frame-1, 0] <= 4.
    assert distance[current.end_frame-1, 0] > distance[prior.end_frame-1, 0]
