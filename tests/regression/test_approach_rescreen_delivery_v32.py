"""Derived delivery preserves actor direction and clips upstream support."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

scripts = Path(__file__).resolve().parents[2] / 'scripts'
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location('rescreen_delivery_v32', scripts/'rescreen_approach_v32.py')
delivery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(delivery)


def test_opposite_direction_cannot_lend_candidate_frames(monkeypatch):
    f = SimpleNamespace(track_ids=[1, 3], pair_i=[0], pair_j=[1])
    def fake_mask(features, edge, params, *, direction_evidence):
        direction_evidence['candidates'] = np.array([[True,True,False], [False,False,True]])
        return np.ones(3, bool)
    monkeypatch.setattr(delivery, '_open_approach_mask', fake_mask)
    mask, edge, actor, target = delivery.actor_candidate_mask(f, {'actor_id':3,'target_id':1}, None)
    assert mask.tolist() == [False,False,True]
    assert (edge,actor,target) == (0,1,0)


def test_retained_prefix_clips_every_parent_support_ledger_and_recounts():
    parent = dict(event_id='parent', start_frame=100, end_frame=170,
        observed_all_participants_frame_intervals=[[100,120],[125,170]],
        observed_all_participants_frame_count=65,
        role_support_frame_intervals=[[100,110],[115,160]],role_support_frame_count=55,
        role_unknown_frame_intervals=[[110,115],[160,170]],role_unknown_frame_count=15)
    child = delivery.clipped_child(parent,105,140)
    assert child['observed_all_participants_frame_intervals'] == [(105,120),(125,140)]
    assert child['observed_all_participants_frame_count'] == 30
    assert child['inferred_or_partially_unobserved_frame_count'] == 5
    assert child['role_support_frame_intervals'] == [(105,110),(115,140)]
    assert child['role_support_frame_count'] == 30
    assert child['role_unknown_frame_intervals'] == [(110,115)]
    assert child['role_unknown_frame_count'] == 5
    assert parent['role_support_frame_count'] == 55
