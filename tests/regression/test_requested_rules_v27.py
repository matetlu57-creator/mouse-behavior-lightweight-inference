"""Requested v27 limits and counterexamples, independent of frozen review data."""
from dataclasses import replace

import numpy as np
import pytest

from tests.regression import test_heuristics_v16_audit as audit
from mouse_behavior.behavior.individual_rules import DEFAULT_INDIVIDUAL_CONFIG
from mouse_behavior.behavior.static_rule import bounded_stationary_mask
from mouse_behavior.evaluation import confirmed_rules as social
from mouse_behavior.evaluation import group_event_rules as group
from mouse_behavior.evaluation.social_heuristic_config import DEFAULT_SOCIAL_HEURISTIC_CONFIG as cfg


def test_requested_defaults_and_profile_mirrors():
    from mouse_behavior.config import load_config
    from mouse_behavior.evaluation.exported_dataset_config import config_seed_parameters
    from pathlib import Path
    params = config_seed_parameters(load_config(Path(__file__).resolve().parents[2] / 'configs/default.yaml'))
    assert DEFAULT_INDIVIDUAL_CONFIG.static_max_radius_cm == 4.
    assert DEFAULT_INDIVIDUAL_CONFIG.static_transient_max_displacement_cm == 4.
    assert params.stationary_max_center_radius_cm == 4.
    assert params.stationary_transient_max_displacement_cm == 4.
    assert cfg.dist_approach_end_min_cm == pytest.approx(4.456491470336914)
    assert cfg.dist_approach_end_max_cm == pytest.approx(19.76601791381836)
    assert cfg.approach_surface_endpoint_cm - cfg.approach_surface_endpoint_tolerance_cm == pytest.approx(.7240091693751944)
    assert cfg.approach_surface_endpoint_cm + cfg.approach_surface_endpoint_tolerance_cm == pytest.approx(6.3939353708899365)
    assert cfg.vel_together_max_cm_s == 3.
    assert cfg.nose_contact_min_duration_s == .1
    assert cfg.contact_head_extension_min_duration_s == .1
    assert cfg.contact_tail_extension_min_duration_s == .15
    assert cfg.chasing_min_mean_speed_cm_s == 15.
    assert params.isolation_distance_cm == 20.


def test_short_observed_tail_cannot_erase_a_long_bounded_stationary_prefix():
    # Explicit historical 2cm bound isolates the defect from the new 4cm setting.
    centers = np.zeros((180, 1, 2))
    centers[156:, 0, 0] = np.linspace(.1, 3., 24)
    output = bounded_stationary_mask(
        np.ones((180, 1), bool), centers, np.ones((180, 1), bool),
        fps=30., max_gap_s=1., max_transient_displacement_cm=2., max_center_radius_cm=2.)
    assert output[:155, 0].all()
    assert not output[-1, 0]


def test_tail_recovery_cannot_turn_a_long_slow_drift_into_stationary():
    centers = np.zeros((600, 1, 2))
    centers[:, 0, 0] = np.arange(600) / 30.
    output = bounded_stationary_mask(
        np.ones((600, 1), bool), centers, np.ones((600, 1), bool),
        fps=30., max_gap_s=1., max_transient_displacement_cm=4., max_center_radius_cm=4.)
    assert not output.any()


@pytest.mark.parametrize('name', ['head_contact', 'tail_contact'])
def test_contact_subtype_short_minimum_is_real_measured_support(monkeypatch, name):
    f = audit.features(12, 2)
    audit.set_centers(f, [[0., 0.], [5., 0.]])
    audit.sync(f)
    for key in ('pair_nose_head_left_to_right_cm', 'pair_nose_head_right_to_left_cm',
                'pair_nose_tail_left_to_right_cm', 'pair_nose_tail_right_to_left_cm'):
        setattr(f, key, np.full((12, 1), 30.))
    field = 'pair_nose_head_left_to_right_cm' if name == 'head_contact' else 'pair_nose_tail_left_to_right_cm'
    end = 6 if name == 'head_contact' else 8  # .1 head / .15 tail at30fps
    getattr(f, field)[3:end] = .5
    for helper in ('_open_approach_mask', '_chase_mask', '_attack_action_mask'):
        monkeypatch.setattr(social, helper, lambda features, *a, **k: np.zeros(len(features.valid), bool))
    masks = social.pair_rule_masks(f, audit.confirmed_params())
    assert masks[name][3:end, 0].all()
    getattr(f, field)[end - 1] = 30.
    assert not social.pair_rule_masks(f, audit.confirmed_params())[name].any()


def test_contact_missing_gap_does_not_manufacture_minimum_duration():
    strict = np.array([True, False, True])
    qualified, _ = social._bridged_contact_bout_mask(
        strict=strict, measured=np.array([.5, np.nan, .5]), center=np.full(3, 5.),
        body_cm=7.918, fps=30., social_config=cfg)
    assert not qualified.any(), 'two measured frames are less than .1 seconds'


@pytest.mark.parametrize('extension', [False, True])
def test_contact_after_attack_priority_rechecks_real_measured_minimum(monkeypatch, extension):
    f = audit.features(12, 2)
    audit.set_centers(f, [[0., 0.], [5., 0.]])
    audit.sync(f)
    for key in ('pair_nose_head_left_to_right_cm', 'pair_nose_head_right_to_left_cm',
                'pair_nose_tail_left_to_right_cm', 'pair_nose_tail_right_to_left_cm'):
        setattr(f, key, np.full((12, 1), 30.))
    f.pair_nose_head_left_to_right_cm[[4, 5, 7], 0] = .5
    f.pair_nose_head_left_to_right_cm[6, 0] = np.nan
    for helper in ('_open_approach_mask', '_chase_mask'):
        monkeypatch.setattr(social, helper, lambda features, *a, **k: np.zeros(len(features.valid), bool))
    monkeypatch.setattr(social, '_attack_action_mask',
                        lambda features, *a, **k: np.arange(len(features.valid)) < 5)
    masks = social.pair_rule_masks(f, audit.confirmed_params(), allow_contact_extension=extension)
    assert masks['attack'][:5, 0].all()
    assert not masks['head_contact'].any(), 'priority left only 2 real frames plus1 unknown'


@pytest.mark.parametrize('speed,expected', [(39.99, False), (40., True), (41., True)])
def test_chase_mean_speed_uses_adjacent_observed_path(speed, expected):
    centers = np.zeros((90, 2, 2))
    centers[:, :, 0] = np.arange(90)[:, None] / 30. * speed
    centers[:, 1, 0] += 5.
    known = np.ones(90, bool)
    result = social._qualify_observed_chase(known, known, centers, 0, 1, 30.,
                                          replace(cfg, chasing_min_mean_speed_cm_s=40.))
    assert bool(result.any()) is expected


def test_chase_mean_is_required_for_both_and_does_not_count_cross_gap_travel():
    centers = np.zeros((120, 2, 2))
    centers[:, 0, 0] = np.arange(120) / 30. * 60.
    centers[:, 1, 0] = 5 + np.arange(120) / 30. * 20.
    known = np.ones(120, bool)
    historical = replace(cfg, chasing_min_mean_speed_cm_s=40.)
    assert not social._qualify_observed_chase(known, known, centers, 0, 1, 30., historical).any()
    centers[:, :, 0] = np.arange(120)[:, None] / 30. * 20.
    centers[60:, :, 0] += 1000.
    known[45:60] = False
    result = social._qualify_observed_chase(known, known, centers, 0, 1, 30., historical)
    assert not result.any()


@pytest.mark.parametrize('distance,expected', [(19.999, False), (20., False), (20.001, True)])
def test_isolation_strict_twenty_cm(distance, expected):
    f = audit.features(360, 2)
    audit.set_centers(f, [[0., 0.], [distance, 0.]])
    audit.sync(f)
    result = group.isolation_candidates(f, replace(audit.confirmed_params(), isolation_distance_cm=20.))
    assert bool(result) is expected


def test_default_library_and_ranker_parameter_paths_share_twenty_cm_policy():
    from mouse_behavior.evaluation.exported_dataset_models import HeuristicParameters
    from mouse_behavior.evaluation.calibrated_rule_ranker import build_candidate_table
    f = audit.features(360, 2)
    audit.set_centers(f, [[0., 0.], [10., 0.]])
    audit.sync(f)
    assert HeuristicParameters().isolation_distance_cm == 20.
    assert not group.isolation_candidates(f, HeuristicParameters())
    table = build_candidate_table(f, [HeuristicParameters(confirmed_rules_v1=1.)],
                                  behavior_subset=('isolation',))
    assert not any(c.behavior == 'isolation' and c.evidence > 0 for c in table.candidates)


def test_old_blind_cli_cannot_silently_present_a_historical_model_as_v27(monkeypatch):
    import sys
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / 'scripts'))
    import predict_exported_blind as blind
    argv = ['predict_exported_blind.py', '--blind-manifest', 'unread.csv',
            '--model-dir', 'unread-model', '--output-dir', 'unused-output']
    monkeypatch.setattr(sys, 'argv', argv)
    with pytest.raises(SystemExit) as exc:
        blind.parse_args()
    assert exc.value.code == 2
    monkeypatch.setattr(sys, 'argv', argv + ['--historical-model'])
    assert blind.parse_args().historical_model


@pytest.mark.parametrize('distance,expected', [(20., False), (20.001, True)])
def test_legacy_facade_also_respects_strict_isolation_boundary(distance, expected):
    from mouse_behavior.evaluation.exported_dataset_models import HeuristicParameters
    from mouse_behavior.evaluation.exported_dataset_prediction import predict_features
    f = audit.features(360, 2)
    audit.set_centers(f, [[0., 0.], [distance, 0.]])
    audit.sync(f)
    output = predict_features(f, HeuristicParameters())
    assert bool(output['identity_scores']['100']['isolation'] > 0.) is expected


def formation(frames=150, translate=False):
    f = audit.features(frames, 3)
    time = np.arange(frames) / f.fps
    f.centers_cm[:, 0, 0] = np.maximum(3., 22. - 7.*time)
    f.centers_cm[:, 1] = [0., 0.]
    f.centers_cm[:, 2] = [0., 2.]
    if translate:
        f.centers_cm[:, :, 1] += 20. * time[:, None]
    audit.sync(f)
    audit.resync_kinematics(f)
    return f


@pytest.mark.parametrize('translate', [False, True])
def test_clustering_actor_arrives_into_observed_pair_without_recipient_speed_cap(translate):
    f = formation(translate=translate)
    near = np.isfinite(f.pair_distance_cm) & (f.pair_distance_cm <= 10.)
    process = group._clustering_process(f, (0, 1, 2), near, f.velocity_cm_s, f.speed_cm_s,
                                        group.DEFAULT_GROUP_HEURISTIC_CONFIG, neighbor_cm=10., max_gap_s=5.)
    assert process.evidence > 0
    assert process.frames[0][0] == 0, 'V29 includes actual convergence before final density'
    assert process.frames[-1][1] == len(f.valid)


@pytest.mark.parametrize('translate', [False, True])
def test_clustering_static_pile_and_pure_group_translation_are_not_formation(translate):
    f = formation()
    f.centers_cm[:, 0] = [3., 0.]
    if translate:
        f.centers_cm[:, :, 1] += np.arange(len(f.valid))[:, None] / f.fps * 20.
    audit.sync(f)
    audit.resync_kinematics(f)
    near = np.isfinite(f.pair_distance_cm) & (f.pair_distance_cm <= 10.)
    process = group._clustering_process(f, (0, 1, 2), near, f.velocity_cm_s, f.speed_cm_s,
                                        group.DEFAULT_GROUP_HEURISTIC_CONFIG, neighbor_cm=10., max_gap_s=5.)
    assert process.evidence == 0


def test_clustering_never_borrows_displacement_across_an_unobserved_actor_gap():
    f = formation()
    f.centers_cm[:50, 0] = [22., 0.]
    f.centers_cm[50:, 0] = [3., 0.]
    f.valid[45:55, 0] = False
    audit.sync(f)
    audit.resync_kinematics(f)
    near = np.isfinite(f.pair_distance_cm) & (f.pair_distance_cm <= 10.)
    process = group._clustering_process(f, (0, 1, 2), near, f.velocity_cm_s, f.speed_cm_s,
                                        group.DEFAULT_GROUP_HEURISTIC_CONFIG, neighbor_cm=10., max_gap_s=5.)
    assert process.evidence == 0
