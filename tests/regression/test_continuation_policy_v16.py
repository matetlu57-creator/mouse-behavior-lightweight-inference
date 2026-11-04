"""Runtime safeguards for the human decisions recorded on 2026-10-16."""

from dataclasses import replace

import numpy as np
import pytest

from tests.regression import test_heuristics_v16_audit as audit
from mouse_behavior.behavior import individual_rules
from mouse_behavior.behavior.individual_rules import IndividualRuleResult
from mouse_behavior.evaluation import calibrated_rule_ranker as ranker
from mouse_behavior.evaluation import confirmed_rules, group_event_rules
from mouse_behavior.evaluation import exported_dataset_prediction as prediction
from mouse_behavior.evaluation.blind_top1 import Candidate
from mouse_behavior.evaluation.social_heuristic_config import DEFAULT_SOCIAL_HEURISTIC_CONFIG


def _static_features():
    f = audit.features(60, 1, reference_body_cm=7.918)
    f.collective_keypoints_cm = np.zeros((60, 1, 7, 2))
    return f


def test_default_posture_screen_is_disabled_without_suppressing_stationary(monkeypatch):
    f = _static_features()

    def forbidden(*args, **kwargs):
        raise AssertionError("unreviewed posture screen ran on the default path")

    monkeypatch.setattr(individual_rules, "_robust_posterior_and_head", forbidden)
    result = individual_rules.individual_rule_masks(
        f.centers_cm, f.valid, f.fps, keypoints_cm=f.collective_keypoints_cm,
    )
    assert not individual_rules.DEFAULT_INDIVIDUAL_CONFIG.enable_posture_screen
    assert result.masks["stationary"].any()
    for name in ("grooming", "rearing"):
        assert not result.masks[name].any()


def test_offline_posture_screen_requires_explicit_opt_in(monkeypatch):
    f = _static_features()
    original = individual_rules._robust_posterior_and_head
    calls = []

    def observed(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(individual_rules, "_robust_posterior_and_head", observed)
    individual_rules.individual_rule_masks(
        f.centers_cm, f.valid, f.fps, keypoints_cm=f.collective_keypoints_cm,
        config=replace(individual_rules.DEFAULT_INDIVIDUAL_CONFIG, enable_posture_screen=True),
    )
    assert calls, "the audit-only screen must remain available for manual review"


def _unreviewed_result(centers, valid, fps, **kwargs):
    return IndividualRuleResult({
        "stationary": np.zeros_like(valid), "locomotion": np.zeros_like(valid),
        "grooming": np.ones_like(valid), "rearing": np.ones_like(valid),
    }, {})


def test_export_prediction_rejects_positive_unreviewed_posture_evidence(monkeypatch):
    monkeypatch.setattr(prediction, "individual_rule_masks", _unreviewed_result)
    result = prediction.predict_features(_static_features(), audit.confirmed_params())
    for name in ("grooming", "rearing"):
        assert name not in result["detected"]
        assert result["scores"][name] == 0.0
        assert all(row[name] == 0.0 for row in result["identity_scores"].values())


@pytest.mark.parametrize("subset", [None, ("locomotion", "stationary", "grooming", "rearing"), ("grooming",)])
def test_ranker_never_seeds_or_promotes_suspended_posture(monkeypatch, subset):
    monkeypatch.setattr(individual_rules, "individual_rule_masks", _unreviewed_result)
    monkeypatch.setattr(prediction, "individual_rule_masks", _unreviewed_result)
    table = ranker.build_candidate_table(
        _static_features(), [audit.confirmed_params()], behavior_subset=subset,
    )
    assert all(c.behavior not in {"grooming", "rearing"} for c in table.candidates)
    if subset == ("grooming",):
        assert table.values.shape[0] == 0
        assert ranker.predict_table(table, {"feature_names": table.feature_names}).behavior is None


def test_frozen_candidate_table_cannot_restore_suspended_predictions():
    table = ranker.CandidateTable(
        (Candidate("rearing", (100,), 2.0), Candidate("stationary", (100,), 1.0)),
        np.array([[100.0], [1.0]], dtype=np.float32), ("test_evidence",),
    )
    model = {"feature_names": table.feature_names,
             "coefficients": {"rearing": [1.0], "stationary": [1.0]}}
    result = ranker.predict_table(table, model)
    assert result.behavior == "stationary"
    assert result.target_ids == (100,)


def test_dispersal_facade_forwards_every_hardened_argument(monkeypatch):
    received = {}

    def phase(*args, **kwargs):
        received.update(kwargs)
        return np.ones(10, dtype=bool)

    monkeypatch.setattr(group_event_rules, "dispersal_phase", phase)
    centers = np.zeros((10, 3, 2))
    valid = np.ones((10, 3), bool)
    result = confirmed_rules.dispersal_phase(
        np.zeros((10, 3)), np.ones(10, bool), np.ones(10, bool),
        audit.confirmed_params(), 30.0, centers_cm=centers, member_valid=valid,
        initial_group_radius_cm=9.8975, minimum_outward_members=3,
        minimum_outward_displacement_cm=5.0, minimum_visible_fraction=0.8,
    )
    assert result.all()
    assert received["centers_cm"] is centers
    assert received["member_valid"] is valid
    assert received["initial_group_radius_cm"] == 9.8975
    assert received["minimum_outward_members"] == 3
    assert received["minimum_outward_displacement_cm"] == 5.0
    assert received["minimum_visible_fraction"] == 0.8


@pytest.mark.parametrize("field,value", [
    ("chasing_track_gap_max_s", 0.6), ("nose_contact_gap_grace_s", 0.12),
    ("nose_contact_exit_distance_cm", 2.2), ("vel_nose_probing_min_cm_s", 0.3),
    ("angle_nose_probing_max_deg", 55.0), ("contact_target_margin_cm", 0.6),
])
def test_retired_social_settings_cannot_be_tuned_silently(field, value):
    assert value != getattr(DEFAULT_SOCIAL_HEURISTIC_CONFIG, field)
    with pytest.warns(RuntimeWarning, match=f"Retired social settings.*{field}"):
        config = replace(DEFAULT_SOCIAL_HEURISTIC_CONFIG, **{field: value})
    assert getattr(config, field) == value  # historical receipts remain readable


def test_stale_social_parameter_mirrors_warn_and_do_not_override_the_active_policy():
    f = _approaching_pair()
    parameters = audit.confirmed_params()
    control = confirmed_rules._open_approach_mask(f, 0, parameters)
    stale = replace(parameters, approach_min_actor_speed_cm_s=1000.0)
    confirmed_rules._warn_social_mirror_drift.cache_clear()
    with pytest.warns(RuntimeWarning, match="social_config alone"):
        actual = confirmed_rules._open_approach_mask(f, 0, stale)
    assert np.array_equal(actual, control)


def _approaching_pair():
    f = audit.features(60, 2, reference_body_cm=7.918)
    f.centers_cm[:, 0, 0] = np.linspace(-30.0, -9.918, 60)
    f.centers_cm[:, 1, :] = 0.0
    audit.sync(f)
    audit.resync_kinematics(f)
    f.heading_unit = np.zeros((60, 2, 2))
    f.heading_unit[:, 0, 0] = 1.0
    f.heading_unit[:, 1, 0] = -1.0
    local = np.array([
        [3.959, 0], [2.5, .8], [2.5, -.8], [1, 0],
        [-2, 1], [-2, -1], [-3.959, 0],
    ])
    f.collective_keypoints_cm = np.empty((60, 2, 7, 2))
    f.collective_keypoints_cm[:, 0] = f.centers_cm[:, 0, None, :] + local
    f.collective_keypoints_cm[:, 1] = local * np.array([-1, 1])
    return f


@pytest.mark.parametrize("missing_detection", [False, True])
def test_real_geometry_approach_survives_one_unknown_frame(missing_detection):
    f = _approaching_pair()
    control = confirmed_rules._open_approach_mask(f, 0, audit.confirmed_params())
    assert control.sum() == 60, "a negative-only fixture cannot verify the bridge"
    f.heading_unit[30, 0] = np.nan
    if missing_detection:
        f.valid[30] = False
        f.pair_distance_cm[30] = np.nan
        f.collective_keypoints_cm[30] = np.nan
    bridged = confirmed_rules._open_approach_mask(f, 0, audit.confirmed_params())
    assert np.array_equal(control, bridged)


def test_real_geometry_approach_rejects_a_gap_above_the_ratio_and_duration_limits():
    f = _approaching_pair()
    assert confirmed_rules._open_approach_mask(f, 0, audit.confirmed_params()).any()
    f.valid[15:45] = False
    f.pair_distance_cm[15:45] = np.nan
    f.collective_keypoints_cm[15:45] = np.nan
    mask = confirmed_rules._open_approach_mask(f, 0, audit.confirmed_params())
    # A later, independently qualifying approach can still exist. The large
    # unknown interval itself must never become claimed approach support.
    assert not mask[15:45].any()


def test_group_evidence_and_ranker_fractions_stay_within_clip_duration():
    f = audit._static_pile(180, 4)
    audit.set_centers(f, [[0, 0], [3, 0], [0, 3], [3, 3]])
    audit.resync_kinematics(f)
    params = audit.confirmed_params()
    duration = len(f.valid) / f.fps
    candidates = group_event_rules.group_rule_candidates(f, params, ())
    assert any(c.behavior == "huddle" and c.evidence > 0 for c in candidates)
    assert all(0 <= c.evidence <= duration for c in candidates)
    table = ranker.build_candidate_table(f, [params], behavior_subset=audit.GROUP_BEHAVIORS)
    fractions = [i for i, name in enumerate(table.feature_names)
                 if name == "evidence_fraction" or (name.startswith("rule_") and name.endswith("_fraction"))]
    assert fractions
    assert np.all(table.values[:, fractions] >= 0)
    assert np.all(table.values[:, fractions] <= 1.0 + 1e-6)
