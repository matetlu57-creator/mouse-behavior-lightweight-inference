"""A brief coherent impulse may establish measured action, never an unknown span."""
import numpy as np

from tests.regression import test_heuristics_v16_audit as audit
from tests.regression.test_observed_attack_contact_v23 import contact_fixture
from mouse_behavior.evaluation import confirmed_rules as rules


def impulse_fixture(monkeypatch, *, both_points=True):
    f = contact_fixture()
    f.centers_cm[:, 1, 0] = 6.
    audit.sync(f)

    def measured(points, *args, **kwargs):
        velocity = np.zeros((len(points), 2))
        acceleration = np.zeros_like(velocity)
        if points[0, 0] < 4.3:
            velocity[6:18, 0] = 14.
            velocity[11:18, 0] = 2.
            acceleration[11:18, 0] = -100.
            if both_points or points[0, 0] > 3.5:
                acceleration[10, 0] = 150.
        return velocity, acceleration

    monkeypatch.setattr(rules, '_local_quadratic_point_kinematics', measured)
    return f


def test_one_coherent_impulse_qualifies_measured_action_and_slowdown(monkeypatch):
    f = impulse_fixture(monkeypatch)
    channels = {}
    mask = rules._attack_action_mask(f, 0, channels=channels)
    assert mask[6:18].all()
    assert not mask[:6].any() and not mask[18:].any()
    assert not channels['center_only_raw'].any()
    assert np.array_equal(mask, channels['light_observed_acceleration'])
    assert not channels['heavy_repeated_target_boxes'].any()


def test_nose_swing_without_complete_head_impulse_does_not_trigger(monkeypatch):
    f = impulse_fixture(monkeypatch, both_points=False)
    assert not rules._attack_action_mask(f, 0).any()


def test_unknown_row_splits_new_measured_action_and_cannot_be_painted(monkeypatch):
    f = impulse_fixture(monkeypatch)
    f.valid[13, 1] = False
    mask = rules._attack_action_mask(f, 0)
    assert mask[6:13].all()
    assert not mask[13:].any()


def test_surface_proximity_cannot_create_a_coherent_light_onset(monkeypatch):
    f = impulse_fixture(monkeypatch)
    f.centers_cm[:, 1, 0] = 9.
    audit.sync(f)
    assert not rules._attack_action_mask(f, 0).any()
