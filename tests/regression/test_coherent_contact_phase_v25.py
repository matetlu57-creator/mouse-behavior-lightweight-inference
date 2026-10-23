"""A center pulse may continue through measured contact, never through an epoch gap."""
import numpy as np

from tests.regression import test_heuristics_v16_audit as audit
from tests.regression.test_coherent_attack_v24 import impulse_fixture
from mouse_behavior.evaluation import confirmed_rules as rules


def test_v25_provenance_and_unchanged_acceptance_and_numerical_gates():
    receipt = rules.policy_receipt()
    cfg = rules.DEFAULT_SOCIAL_HEURISTIC_CONFIG
    assert receipt['version'] == 'confirmed_rules_v30'
    assert receipt['attack_acceptance_default_coverage'] == .30
    assert receipt['other_behavior_acceptance_default_coverage'] == .50
    assert cfg.acc_attack_min_cm_s2 == 99.59917210877003
    assert cfg.vel_attack_min_cm_s == 10.231697660707422
    assert cfg.dist_attack_max_cm == 7.321547985076904
    assert cfg.attack_min_duration_s == .15
    assert cfg.attack_deceleration_max_s == .75
    assert receipt['v25_fit_artifact_sha256'] == {
        'outputs/heuristics_v25_attack_misses_20261018/fit_decision_target_confirmed.json':
        'a280985ea95e5f3ff7de1399ac50bc388f03a9c7b0823670bccdba65f5674af5'}


def test_center_near_pulse_can_continue_while_measured_body_contact_remains(monkeypatch):
    f = impulse_fixture(monkeypatch)
    f.centers_cm[11:, 1, 0] = 9.
    audit.sync(f)
    channels = {}
    actual = rules._attack_action_mask(f, 0, channels=channels)
    assert actual[6:18].all()
    assert not actual[:6].any() and not actual[18:].any()
    assert not channels['center_only_raw'].any()


def test_contact_cannot_backdate_an_action_before_its_center_near_seed(monkeypatch):
    f = impulse_fixture(monkeypatch)
    f.centers_cm[:, 1, 0] = 9.
    f.centers_cm[10, 1, 0] = 6.
    audit.sync(f)
    actual = rules._attack_action_mask(f, 0)
    assert actual[10:18].all()
    assert not actual[:10].any()


def test_unknown_splits_contact_continuation_and_next_run_needs_its_own_seed(monkeypatch):
    f = impulse_fixture(monkeypatch)
    f.centers_cm[11:, 1, 0] = 9.
    f.valid[13, 1] = False
    audit.sync(f)
    actual = rules._attack_action_mask(f, 0)
    assert actual[6:13].all()
    assert not actual[13:].any()


def test_measured_separation_stops_contact_continuation(monkeypatch):
    f = impulse_fixture(monkeypatch)
    f.centers_cm[11:, 1, 0] = 9.
    f.collective_keypoints_cm[15:, 1, :6, 0] += 10.
    audit.sync(f)
    actual = rules._attack_action_mask(f, 0)
    assert actual[6:15].all()
    assert not actual[15:].any()


def test_new_motion_growth_stays_on_the_same_side_of_a_pose_jump(monkeypatch):
    f = impulse_fixture(monkeypatch)
    f.collective_keypoints_cm[9:, 0, :3, 0] -= 5.
    # Isolate jump segmentation from independently tested target geometry.
    monkeypatch.setattr(rules, 'directed_snout_body_footprint_distances',
                        lambda *a, **k: np.zeros(len(f.valid)))
    actual = rules._coherent_observed_attack_action(
        f, 0, rules.DEFAULT_SOCIAL_HEURISTIC_CONFIG, allow_observed_run_extension=True)
    assert actual[9:18].all()
    assert not actual[:9].any()


def test_subminimum_fragments_on_either_side_of_jump_cannot_qualify_by_union(monkeypatch):
    f = impulse_fixture(monkeypatch)
    f.collective_keypoints_cm[10:, 0, :3, 0] -= 5.
    monkeypatch.setattr(rules, 'directed_snout_body_footprint_distances',
                        lambda *a, **k: np.zeros(len(f.valid)))

    def measured(points, *args, **kwargs):
        velocity = np.zeros((len(points), 2))
        acceleration = np.zeros_like(velocity)
        if points[0, 0] < 4.3:
            velocity[8:13, 0] = 14.
            acceleration[[9, 11], 0] = 150.
        return velocity, acceleration

    monkeypatch.setattr(rules, '_local_quadratic_point_kinematics', measured)
    actual = rules._coherent_observed_attack_action(
        f, 0, rules.DEFAULT_SOCIAL_HEURISTIC_CONFIG, allow_observed_run_extension=True)
    assert not actual.any()
