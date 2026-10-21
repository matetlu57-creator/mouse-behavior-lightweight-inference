"""New action windows need measured approach to their own target's body."""
import numpy as np
import pytest
from dataclasses import replace

from tests.regression.test_observed_attack_contact_v23 import contact_fixture
from tests.regression.test_coherent_attack_v24 import impulse_fixture
from mouse_behavior.evaluation import confirmed_rules as rules
from mouse_behavior.evaluation import social_event_geometry as geometry


def test_directed_nose_body_gap_ignores_tail_and_is_target_specific():
    f = contact_fixture()
    gap = geometry.directed_snout_body_footprint_distances(f, 0, actor=0, max_body_lengths=2.)
    np.testing.assert_allclose(gap, .5, atol=1e-5)
    f.collective_keypoints_cm[:,1,6] = f.collective_keypoints_cm[:,0,0]
    np.testing.assert_allclose(geometry.directed_snout_body_footprint_distances(
        f, 0, actor=0, max_body_lengths=2.),gap)
    f.collective_keypoints_cm[:,0,0,0] -= 4.
    f.collective_keypoints_cm[:,0,3:6,0] += 9.
    actual = geometry.directed_snout_body_footprint_distances(f, 0, actor=0, max_body_lengths=2.)
    assert (actual > 4.).all()
    # Rear/body overlap alone does not imply this actor's nose reaches target.
    assert (geometry.body_footprint_distances(f, 0, max_body_lengths=2.) <= .5).all()


def test_target_surface_rejects_unknown_sparse_degenerate_large_or_unselected_geometry():
    f = contact_fixture()
    f.valid[0,1] = False
    f.collective_keypoints_cm[1,0,0] = np.nan
    f.collective_keypoints_cm[2,1,3:6] = np.nan
    f.collective_keypoints_cm[3,1,:6,1] = 0.
    f.collective_keypoints_cm[4,1,0,0] = 1000.
    selected = np.ones(60,bool)
    selected[5] = False
    actual = geometry.directed_snout_body_footprint_distances(
        f, 0, actor=0, max_body_lengths=2., frame_mask=selected)
    assert np.isnan(actual[:6]).all()
    assert np.isfinite(actual[6:]).all()
    with pytest.raises(ValueError):
        geometry.directed_snout_body_footprint_distances(f, 0, actor=3, max_body_lengths=2.)
    with pytest.raises(ValueError):
        geometry.directed_snout_body_footprint_distances(f, 0, actor=0, max_body_lengths=0.)


def epoch(confirmation, *, seed_frames=(3,), unknown=()):
    n = 24
    seed, active, joint = np.zeros(n,bool), np.ones(n,bool), np.ones(n,bool)
    seed[list(seed_frames)] = True
    joint[list(unknown)] = False
    points = np.zeros((n,2))
    cfg = rules.DEFAULT_SOCIAL_HEURISTIC_CONFIG
    return rules._coherent_epoch_action(seed,active,joint,joint,points,points,
                                         np.ones(n,bool),30.,cfg,confirmation=confirmation)


def test_new_window_without_target_reach_cannot_be_qualified_by_direction_alone():
    assert not epoch(np.zeros(24,bool)).any()


def test_confirmation_must_be_in_the_same_observed_epoch_and_window():
    confirmation = np.zeros(24,bool)
    confirmation[15] = True
    assert not epoch(confirmation,unknown=(10,)).any()
    confirmation[7] = True
    assert epoch(confirmation,unknown=(10,))[:10].all()
    assert not epoch(confirmation,unknown=(10,))[10:].any()


def test_unknown_confirmation_cannot_validate_a_visible_action():
    confirmation = np.zeros(24,bool)
    confirmation[10] = True
    assert not epoch(confirmation,unknown=(10,)).any()


def test_window_cannot_borrow_confirmation_from_later_in_the_clip():
    n = 60
    seed,active = np.zeros(n,bool),np.zeros(n,bool)
    seed[10] = True
    active[8:14] = True
    confirmation = np.zeros(n,bool)
    confirmation[40] = True
    points=np.zeros((n,2))
    observed=np.ones(n,bool)
    result=rules._coherent_epoch_action(seed,active,observed,observed,points,points,
        observed,30.,rules.DEFAULT_SOCIAL_HEURISTIC_CONFIG,confirmation=confirmation)
    assert not result.any()


def test_target_surface_parameter_is_explicit_and_rejects_invalid_values():
    cfg=rules.DEFAULT_SOCIAL_HEURISTIC_CONFIG
    assert cfg.attack_new_action_target_surface_max_gap_cm == 1.9986556302093281
    for value in (0., -1., np.nan):
        with pytest.raises(ValueError):
            replace(cfg,attack_new_action_target_surface_max_gap_cm=value)


def test_public_new_action_rejects_body_overlap_while_nose_turns_away_from_assigned_target(monkeypatch):
    f=impulse_fixture(monkeypatch)
    f.collective_keypoints_cm[:,1,:6,0] += 4.
    f.collective_keypoints_cm[:,0,3:6,0] += 12.

    def unavailable(points,*args,**kwargs):
        return np.full_like(points,np.nan),np.full_like(points,np.nan)

    def measured(points,*args,**kwargs):
        velocity,acceleration=np.zeros_like(points),np.zeros_like(points)
        if points[0,0] < 4.3:
            velocity[6:18,0]=14.
            velocity[11:18,0]=2.
            acceleration[10,0]=150.
            acceleration[11:18,0]=-100.
        return velocity,acceleration

    monkeypatch.setattr(rules,'_local_quadratic_point_kinematics',unavailable)
    monkeypatch.setattr(rules,'_observed_run_point_kinematics',measured)
    np.testing.assert_allclose(geometry.body_footprint_distances(f,0,max_body_lengths=2.),0.)
    assert not rules._attack_action_mask(f,0).any()
    # Same pulse and center distance, now the assigned target is actually
    # reached by the actor's nose. This is the independently controlled gate.
    f.collective_keypoints_cm[:,1,:6,0] -= 4.
    actual=rules._attack_action_mask(f,0)
    assert actual[6:18].all()
    assert not actual[:6].any() and not actual[18:].any()
