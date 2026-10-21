"""User-directed v17 boundaries, independent candidate masks and missing IDs."""

from dataclasses import replace
import numpy as np
import pytest

from tests.regression import test_heuristics_v16_audit as audit
from mouse_behavior.behavior.individual_rules import (
    DEFAULT_INDIVIDUAL_CONFIG, IndividualHeuristicConfig, individual_rule_masks,
)
from mouse_behavior.evaluation import confirmed_rules as social
from mouse_behavior.evaluation import group_event_rules as group
from mouse_behavior.evaluation.group_heuristic_config import DEFAULT_GROUP_HEURISTIC_CONFIG
from mouse_behavior.evaluation.social_heuristic_config import DEFAULT_SOCIAL_HEURISTIC_CONFIG


def _historical_dispersal_phase(*args, **kwargs):
    return group.dispersal_phase(*args, **kwargs, extend_actual_motion=False)


def _pair(speed=0.0, distance=4.0, frames=60):
    f = audit.features(frames, 2, reference_body_cm=7.918)
    f.centers_cm[:, 0, 0] = np.arange(frames) * speed / f.fps
    f.centers_cm[:, 1, 0] = f.centers_cm[:, 0, 0] + distance
    audit.sync(f)
    audit.resync_kinematics(f)
    f.speed_cm_s[:] = speed
    return f


def _raw_masks(monkeypatch, positive):
    for name, label in (("_open_approach_mask", "approach"),
                        ("_attack_action_mask", "attack"), ("_chase_mask", "chasing")):
        monkeypatch.setattr(social, name,
                            lambda f, *args, label=label, **kwargs:
                            np.full(len(f.valid), label in positive))


@pytest.mark.parametrize("positive,speed,winner", [
    ({"approach", "attack", "chasing"}, 0.0, "approach"),
    ({"attack", "chasing"}, 0.0, "together"),
    ({"attack", "chasing"}, 20.0, "attack"),
    ({"chasing"}, 60.0, "chasing"),
    (set(), 20.0, "head_contact"),
])
def test_six_way_priority_is_applied_after_independent_evidence(monkeypatch, positive, speed, winner):
    f = _pair(speed=speed)
    # Both contact subtypes qualify; the priority, rather than raw suppression
    # or the longer-tail exception, must decide the selected label.
    f.pair_nose_head_left_to_right_cm[:] = 0.0
    f.pair_nose_tail_left_to_right_cm[:] = 0.0
    _raw_masks(monkeypatch, positive)
    masks = social.pair_rule_masks(f, audit.confirmed_params())
    labels = ("approach", "together", "attack", "chasing", "head_contact", "tail_contact")
    assert masks[winner].all()
    assert all(not masks[name].any() for name in labels if name != winner)
    assert DEFAULT_SOCIAL_HEURISTIC_CONFIG.priority_order == labels


@pytest.mark.parametrize("distance,speed,expected", [(6.9, 3.9, True), (7.0, 3.9, False), (6.9, 4.0, False)])
def test_together_uses_strict_seven_cm_four_cm_s_limits(monkeypatch, distance, speed, expected):
    _raw_masks(monkeypatch, set())
    f = _pair(speed=speed, distance=distance)
    # Historical v17 policy is still explicitly reproducible after v18 fitting.
    legacy = replace(DEFAULT_SOCIAL_HEURISTIC_CONFIG, dist_together_max_cm=7.0,
                     vel_together_max_cm_s=4.0)
    masks = social.pair_rule_masks(f, audit.confirmed_params(), legacy)
    assert bool(masks["together"].any()) == expected


def test_static_configuration_and_one_second_excursion():
    assert DEFAULT_INDIVIDUAL_CONFIG.static_max_radius_cm == 4.0
    assert DEFAULT_INDIVIDUAL_CONFIG.static_transient_max_displacement_cm == 4.0
    assert DEFAULT_INDIVIDUAL_CONFIG.static_transient_max_s == 1.0
    # Historical positional construction still binds its first argument to
    # motion speed, rather than silently enabling a posture screen.
    assert IndividualHeuristicConfig(3.0).motion_speed_threshold_cm_s == 3.0
    assert not IndividualHeuristicConfig(3.0).enable_posture_screen
    centers = np.zeros((150, 1, 2))
    centers[60:90, 0, 0] = 1.8
    valid = np.ones((150, 1), bool)
    masks = individual_rule_masks(centers, valid, 30.0).masks
    assert masks["stationary"][45:105].all()
    assert not masks["locomotion"][45:105].any()


def test_chase_has_no_ten_cm_candidate_or_reacquisition_distance_gate():
    f = _pair(speed=40.0, distance=20.0, frames=90)
    f.heading_unit = np.zeros((90, 2, 2))
    f.heading_unit[..., 0] = 1.0
    template = np.array([[3.959, 0], [2.5, .8], [2.5, -.8], [1, 0],
                         [-2, 1], [-2, -1], [-3.959, 0]])
    f.collective_keypoints_cm = f.centers_cm[:, :, None, :] + template
    assert social._chase_mask(f, 0).any()


def test_huddle_forward_hold_does_not_require_reappearance_and_is_bounded():
    f = audit.features(300, 3, reference_body_cm=7.918)
    audit.set_centers(f, [[0, 0], [2, 0], [0, 2]])
    f.valid[60:, 2] = False
    f.centers_cm[60:, 2] = np.nan
    audit.sync(f)
    near = np.isfinite(f.pair_distance_cm) & (f.pair_distance_cm <= 10.0)
    geometry = group._huddle_geometry(
        f, (0, 1, 2), near, neighbor_cm=10.0, max_gap_s=5.0,
        speed=np.zeros((300, 3)), quiet_speed_cm_s=2.0,
    )
    held = geometry[4]
    assert held[60:210, 2].all()
    assert not held[210:, 2].any()
    assert not f.valid[60:, 2].any(), "holding must never overwrite raw observations"


def test_huddle_does_not_hold_an_observed_departure():
    f = audit.features(90, 3)
    audit.set_centers(f, [[0, 0], [2, 0], [0, 2]])
    f.centers_cm[30:40, 2] = (50, 50)
    f.valid[40:, 2] = False
    f.centers_cm[40:, 2] = np.nan
    audit.sync(f)
    near = np.isfinite(f.pair_distance_cm) & (f.pair_distance_cm <= 10.0)
    held = group._huddle_geometry(
        f, (0, 1, 2), near, neighbor_cm=10.0, max_gap_s=5.0,
        speed=np.zeros((90, 3)), quiet_speed_cm_s=2.0,
    )[4]
    assert not held[40:, 2].any()


def test_group_defaults_keep_neighbor_and_quiet_limits_with_fitted_sync_support():
    config = DEFAULT_GROUP_HEURISTIC_CONFIG
    assert config.neighbor_distance_cm == 10.0
    assert config.clustering_quiet_speed_cm_s == 4.0
    assert config.locomotion_simultaneous_min_s == pytest.approx(0.9329504772495598)
    assert config.huddle_hold_max_speed_cm_s == 2.0


def test_group_motion_accepts_exactly_one_second_of_synchrony():
    f = audit.features(150, 3)
    for slot in range(3):
        f.centers_cm[:, slot, 0] = np.arange(150) / 30.0 * 10.0 + slot * 2.0
    audit.sync(f)
    velocity = np.zeros((150, 3, 2))
    speed = np.zeros((150, 3))
    velocity[:30, :, 0] = 10.0
    speed[:30] = 10.0
    # Later one-mouse movement extends the process beyond the unchanged 3 s
    # span gate without adding simultaneous support.
    velocity[120:150, 0, 0] = 10.0
    speed[120:150, 0] = 10.0
    legacy = replace(DEFAULT_GROUP_HEURISTIC_CONFIG, locomotion_min_net_displacement_cm=0.0,
                     locomotion_simultaneous_min_s=1.0)
    duration = group._synchronous_event(f, (0, 1, 2), velocity, speed,
                                        audit.confirmed_params(), legacy)
    assert duration > 3.0
    speed[29] = 0.0
    velocity[29] = 0.0
    assert group._synchronous_event(f, (0, 1, 2), velocity, speed,
                                    audit.confirmed_params(), legacy) == 0.0


def _dispersing_fixture(final_x=(0.0, 12.0, -12.0), frames=181):
    f = audit.features(frames, 3, reference_body_cm=7.918)
    initial = np.array([[0., 0.], [1., 0.], [-1., 0.]])
    final = initial.copy()
    final[:, 0] = final_x
    fraction = np.minimum(np.arange(frames) / 150., 1.)
    f.centers_cm[:] = initial[None] + fraction[:, None, None] * (final - initial)[None]
    audit.sync(f)
    return f


def test_dispersal_uses_absolute_departed_pair_distance_at_exact_plus_five_seconds():
    f = _dispersing_fixture()
    compact = np.zeros(len(f.valid), bool)
    compact[0] = True
    mask = _historical_dispersal_phase(f.pair_distance_cm, f.valid.all(axis=1), compact,
                                audit.confirmed_params(), f.fps,
                                centers_cm=f.centers_cm, member_valid=f.valid)
    assert mask[:151].all() and not mask[151:].any()
    # The initial compact cohort contains ID0, but only1 and2 actually leave.
    leaving = group._departing_member_slots(f.centers_cm, f.valid, 0, 150,
                                            minimum_outward_displacement_cm=4.,
                                            minimum_visible_fraction=.6)
    assert leaving == (1, 2)
    f.valid[150, 2] = False
    assert not group.dispersal_phase(f.pair_distance_cm, f.valid.all(axis=1), compact,
                                    audit.confirmed_params(), f.fps,
                                    centers_cm=f.centers_cm, member_valid=f.valid).any()


def test_dispersal_rejects_short_clip_and_leavers_still_within_three_cm():
    for f in (_dispersing_fixture(frames=150), _dispersing_fixture((0., 10., 12.))):
        compact = np.zeros(len(f.valid), bool)
        compact[0] = True
        assert not group.dispersal_phase(f.pair_distance_cm, f.valid.all(axis=1), compact,
                                        audit.confirmed_params(), f.fps,
                                        centers_cm=f.centers_cm, member_valid=f.valid).any()


def test_public_seed_runs_current_social_and_group_rules(monkeypatch):
    from mouse_behavior.evaluation.exported_dataset_config import config_seed_parameters
    from mouse_behavior.evaluation.exported_dataset_prediction import predict_features
    from mouse_behavior.evaluation.blind_top1 import Candidate
    parameters = config_seed_parameters({})
    assert parameters.confirmed_rules_v1 == 1
    assert parameters.group_core_candidate_mode == 1
    assert parameters.isolation_distance_cm == 20.0
    observed = []
    def candidates(f, p, proposals):
        observed.append(True)
        return (Candidate("dispersal", (int(f.track_ids[1]), int(f.track_ids[2])), 5.),)
    monkeypatch.setattr(group, "group_rule_candidates", candidates)
    f = _dispersing_fixture()
    table = predict_features(f, parameters)
    assert observed
    assert table["group_events"][-1]["member_ids"] == list(map(int, f.track_ids[1:]))
    assert table["identity_scores"][str(int(f.track_ids[0]))]["dispersal"] == 0.
    assert table["identity_scores"][str(int(f.track_ids[1]))]["dispersal"] > 0.


def test_retrospective_profile_and_receipt_use_matching_metric_roles():
    config = DEFAULT_SOCIAL_HEURISTIC_CONFIG
    assert config.acc_attack_min_cm_s2 == 99.59917210877003
    assert config.vel_chasing_min_cm_s == .41903117299079895
    assert config.vel_chasing_target_min_cm_s == .5729907751083374
    assert config.dist_nose_head_contact_max_cm == 4.380314658914379
    assert config.dist_nose_tail_contact_max_cm == 12.985665630296548
    assert config.dist_approach_end_min_cm == 4.456491470336914
    assert config.dist_approach_end_max_cm == 19.76601791381836
    assert config.approach_surface_endpoint_cm - config.approach_surface_endpoint_tolerance_cm == pytest.approx(.7240091693751944)
    receipt = social.policy_receipt()
    assert receipt["fit_scope"] == "full_dataset_retrospective_with_user_requested_v28_overrides"
    assert receipt["independent_validation"] is False
    assert "disabled_canonically" in receipt["grooming_rearing"]
    assert "dist_chasing_max_cm" in receipt["retired_social_config_fields"]


def test_current_profile_is_not_changed_by_legacy_training_sweeps():
    from mouse_behavior.evaluation.exported_dataset_config import config_seed_parameters
    from mouse_behavior.evaluation.calibrated_rule_ranker import rule_profiles
    parameters = config_seed_parameters({})
    assert rule_profiles(parameters) == (parameters,)


def test_threshold_dependency_is_in_all_applicable_validation_receipts(monkeypatch):
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    import validate_social_heuristics as social_validation
    import validate_group_heuristics as group_validation
    from mouse_behavior.evaluation.exported_dataset import EXPORTED_DATASET_SOURCE_FILES
    path = "src/mouse_behavior/evaluation/retrospective_thresholds_v17.py"
    assert path in social_validation.CODE_FILES
    assert path in group_validation.CODE_FILES
    assert path in EXPORTED_DATASET_SOURCE_FILES
    current = "src/mouse_behavior/evaluation/retrospective_thresholds_v18.py"
    assert current in social_validation.CODE_FILES
    assert current in group_validation.CODE_FILES
    assert current in EXPORTED_DATASET_SOURCE_FILES
    requested = "src/mouse_behavior/evaluation/requested_thresholds_v27.py"
    assert requested in social_validation.CODE_FILES
    assert requested in group_validation.CODE_FILES
    assert requested in EXPORTED_DATASET_SOURCE_FILES
    current_request = "src/mouse_behavior/evaluation/requested_thresholds_v28.py"
    assert current_request in social_validation.CODE_FILES
    assert current_request in group_validation.CODE_FILES
    assert current_request in EXPORTED_DATASET_SOURCE_FILES


def test_missing_approach_distances_never_require_inf_minus_inf_arithmetic():
    f = _pair(speed=5.0)
    f.pair_distance_cm[10:30] = np.inf
    with np.errstate(invalid="raise"):
        mask = social._open_approach_mask(f, 0, audit.confirmed_params())
    assert not mask[10:30].any()


def test_dispersal_noninteger_fps_uses_fixed_future_grid_frame_not_next_observation():
    f = _dispersing_fixture()
    f.fps = 30.012442789554914
    compact = np.zeros(len(f.valid), bool)
    compact[0] = True
    mask = _historical_dispersal_phase(f.pair_distance_cm, f.valid.all(axis=1), compact,
                                audit.confirmed_params(), f.fps,
                                centers_cm=f.centers_cm, member_valid=f.valid)
    assert np.flatnonzero(mask)[-1] == 151
    f.valid[151, 2] = False
    assert f.valid[150, 2] and f.valid[152, 2]
    assert not group.dispersal_phase(f.pair_distance_cm, f.valid.all(axis=1), compact,
                                    audit.confirmed_params(), f.fps,
                                    centers_cm=f.centers_cm, member_valid=f.valid).any()


def test_reviewed_isolation_fit_receipt_keeps_raw_identity_separate():
    from mouse_behavior.evaluation.retrospective_thresholds_v17 import ISOLATION_DISTANCE_CM
    assert ISOLATION_DISTANCE_CM == 5.520760536193848
    receipt = social.policy_receipt()
    assert receipt["group_rules"]["isolation_distance_cm"] == 20.0
    assert len(receipt["isolation_statistics_sha256"]) == 64
    assert len(receipt["isolation_exclusions_sha256"]) == 64
    assert len(receipt["isolation_user_decision_sha256"]) == 64
    assert len(receipt["isolation_authorization_binding_sha256"]) == 64
    assert "8_pairs_20_rows_5_samples" in receipt["isolation_threshold_fit_policy"]
