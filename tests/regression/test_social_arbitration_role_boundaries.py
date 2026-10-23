"""Full inference must preserve qualified geometry and sustained role changes."""
import json

import numpy as np
import pytest

from mouse_behavior.canonical_video_pipeline import infer_window_events
from mouse_behavior.evaluation import confirmed_rules as rules
from mouse_behavior.evaluation.exported_dataset_config import config_seed_parameters
from mouse_behavior.evaluation.exported_dataset_features import load_exported_features
from mouse_behavior.evaluation.pair_roles import assign_event_roles
from mouse_behavior.evaluation.screening_signals import runs
from mouse_behavior.evaluation.social_event_geometry import snout_surface_distances
from mouse_behavior.evaluation.social_heuristic_config import DEFAULT_SOCIAL_HEURISTIC_CONFIG


FPS = 30.
SCALE = dict(cm_per_pixel=.1, reference_body_cm=7.918)
BODY = np.array([[3.959, 0.], [2.5, .8], [2.5, -.8], [1., 0.],
                 [-2., 1.], [-2., -1.], [-3.959, 0.]])


def _rows(actor_x, target_x, target_y=0.):
    rows = []
    for frame, (left, right) in enumerate(zip(actor_x, target_x, strict=True)):
        detections = []
        for track, x, y in ((100, left, 0.), (101, right, target_y)):
            points = (BODY + [x, y]) / SCALE["cm_per_pixel"]
            keypoints = np.column_stack((points, np.ones(7)))[[0, 2, 1, 3, 4, 5, 6]]
            low, high = points.min(axis=0), points.max(axis=0)
            detections.append(dict(track_id=track, keypoints=keypoints.tolist(),
                                   box=[*low, *high], detection_confidence=1.))
        rows.append(dict(frame=frame, detections=detections))
    return rows


def _features(rows, tmp_path):
    path = tmp_path / "tracks.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    return load_exported_features(path, fps=FPS, scale_config=dict(
        mode="fixed", cm_per_pixel=SCALE["cm_per_pixel"],
        assumed_mouse_body_length_cm=SCALE["reference_body_cm"]))


def test_restored_chase_cannot_leave_approach_using_its_excluded_endpoint(tmp_path):
    time = np.arange(66) / FPS
    rows = _rows(30 * time, 35 + 20 * np.maximum(time - .4, 0))
    features = _features(rows, tmp_path)
    params = config_seed_parameters({})
    approach = rules._open_approach_mask(features, 0, params)
    chase = rules._chase_mask(features, 0)
    assert (approach & chase).any(), "the original valid prefix must overlap pursuit"
    assert runs(chase) == [(13, 66)]
    config = DEFAULT_SOCIAL_HEURISTIC_CONFIG
    assert features.pair_distance_cm[12, 0] > config.dist_approach_end_max_cm
    assert snout_surface_distances(features, 0)[0][12] > (
        config.approach_surface_endpoint_cm + config.approach_surface_endpoint_tolerance_cm)
    events, _ = infer_window_events(rows, FPS, SCALE, params)
    assert not [event for event in events if event["behavior"] == "Approach"]
    assert [(event["start_frame"], event["end_frame"]) for event in events
            if event["behavior"] == "Chasing"] == [(13, 66)]


@pytest.mark.parametrize("departure_s, initial_cm", [(.4, 25.), (.3, 22.)])
def test_restored_chase_preserves_independently_valid_approach_prefix(
    tmp_path, departure_s, initial_cm,
):
    time = np.arange(66) / FPS
    rows = _rows(30 * time, initial_cm + 20 * np.maximum(time - departure_s, 0))
    features = _features(rows, tmp_path)
    params = config_seed_parameters({})
    assert (rules._open_approach_mask(features, 0, params)
            & rules._chase_mask(features, 0)).any()
    events, _ = infer_window_events(rows, FPS, SCALE, params)
    approaches = [event for event in events if event["behavior"] == "Approach"]
    assert [(event["start_frame"], event["end_frame"], event["actor_id"])
            for event in approaches] == [(1, round(departure_s * FPS) + 1, 100)]
    assert any(event["behavior"] == "Chasing" for event in events)


def test_qualified_pursuit_reversal_emits_both_directions(tmp_path):
    time = np.arange(75) / FPS
    rows = _rows(30 * time, 16.25 + 15 * time, target_y=.1)
    features = _features(rows, tmp_path)
    evidence = {}
    assert rules._chase_mask(features, 0, direction_evidence=evidence).all()
    assert [runs(direction) for direction in evidence["candidates"]] == [
        [(0, 33)], [(33, 75)]]
    events, _ = infer_window_events(rows, FPS, SCALE, config_seed_parameters({}))
    pursuits = [event for event in events if event["behavior"] == "Chasing"]
    assert [(event["start_frame"], event["end_frame"], event["actor_id"], event["target_id"])
            for event in pursuits] == [(0, 33, 100, 101), (33, 75, 101, 100)]
    assert [event["role_support_frame_intervals"] for event in pursuits] == [
        [(0, 33)], [(33, 75)]]
    assert all(not event["role_unknown_frame_intervals"] for event in pursuits)


def test_roles_require_unique_measured_direction_on_each_selected_frame():
    selected = np.array([True, True, True, True, True, False])
    directed = np.array([[True, True, True, False, False, True],
                         [False, True, False, True, True, False]])
    facing = np.ones_like(directed)
    facing[0, 2] = False
    actor, target = assign_event_roles(selected, directed, facing, np.array([100, 101]), 0, 1)
    np.testing.assert_array_equal(actor, [100, -1, -1, 101, 101, -1])
    np.testing.assert_array_equal(target, [101, -1, -1, 100, 100, -1])


def test_retained_approach_cannot_borrow_an_excluded_start_anchor(tmp_path):
    time = np.arange(60) / FPS
    features = _features(_rows(6.5 * time, np.full(60, 18.)), tmp_path)
    params = config_seed_parameters({})
    original = rules._open_approach_mask(features, 0, params)
    retained = original & (np.arange(60) >= 30)
    assert retained.sum() >= 9
    assert features.pair_distance_cm[30, 0] < DEFAULT_SOCIAL_HEURISTIC_CONFIG.dist_approach_start_min_cm
    assert not rules._open_approach_mask(features, 0, params, retained_mask=retained).any()


def test_retained_approach_rechecks_its_own_observed_fraction(tmp_path):
    time = np.arange(60) / FPS
    features = _features(_rows(6.5 * time, np.full(60, 18.)), tmp_path)
    features.valid[5:9] = False
    params = config_seed_parameters({})
    original = rules._open_approach_mask(features, 0, params)
    assert original[1:19].all(), "the longer original event qualifies around a bounded gap"
    retained = original & (np.arange(60) < 19)
    assert features.valid[retained].all(axis=1).mean() < .8
    assert not rules._open_approach_mask(features, 0, params, retained_mask=retained).any()
