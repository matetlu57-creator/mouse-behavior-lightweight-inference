"""Real pose-loaded departures keep their own observed formation origins."""
from __future__ import annotations

import json
import sqlite3

import numpy as np
import pytest

from mouse_behavior.canonical_video_pipeline import TrackStore, infer_video_events
from mouse_behavior.evaluation import group_event_rules as group
from mouse_behavior.evaluation.exported_dataset_config import config_seed_parameters
from mouse_behavior.evaluation.exported_dataset_features import load_exported_features


FPS = 30.
SCALE = dict(reference_body_cm=7.918, cm_per_pixel=.1)
BODY = np.array([[3.959, 0], [2.5, .8], [2.5, -.8], [1, 0],
                 [-2, 1], [-2, -1], [-3.959, 0]])


def _smooth(value):
    value = np.clip(value, 0., 1.)
    return value * value * (3. - 2. * value)


def _rows(*, seconds=60., shift=0., departure_seconds=8., second=False,
          never_compact=False):
    time = np.arange(round(seconds * FPS)) / FPS - shift
    radius = 24. - 21. * _smooth((time - 20.) / 6.)
    radius += 16. * _smooth((time - 28.) / departure_seconds)
    if second:
        radius -= 16. * _smooth((time - 40.) / 6.)
        radius += 16. * _smooth((time - 48.) / 8.)
    if never_compact:
        radius = 24. + 16. * _smooth((time - 28.) / departure_seconds)
    angles = np.arange(3) * 2. * np.pi / 3.
    directions = np.c_[np.cos(angles), np.sin(angles)]
    return _pose_rows(np.array([37., 37.]) + radius[:, None, None] * directions)


def _pose_rows(centers):
    rows = []
    for frame, positions in enumerate(centers):
        detections = []
        for slot, position in enumerate(positions):
            points = (position + BODY) * 10.
            detections.append(dict(track_id=100 + slot,
                box=[*points.min(axis=0), *points.max(axis=0)],
                keypoints=[[*points[index], 1.] for index in (0, 2, 1, 3, 4, 5, 6)],
                detection_confidence=1., class_id=0))
        rows.append(dict(frame=frame, time=frame / FPS, source_frame=frame,
                         source_fps=FPS, detections=detections))
    return rows


def _features(tmp_path, rows):
    path = tmp_path / "tracks.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    tracks = {detection['track_id'] for row in rows for detection in row['detections']}
    return load_exported_features(path, fps=FPS, max_tracks=max(3, len(tracks)),
        scale_config=dict(mode="fixed", cm_per_pixel=SCALE["cm_per_pixel"],
                          assumed_mouse_body_length_cm=SCALE["reference_body_cm"]))


def _departures(tmp_path, rows):
    return group.group_rule_candidates(_features(tmp_path, rows), config_seed_parameters({}),
                                       (), behaviors=("dispersal",))


def _intervals(candidates):
    return sorted({interval for c in candidates for interval in c.frame_intervals})


def test_default_video_context_detects_group_formed_after_window_start(tmp_path):
    rows = _rows()
    database = tmp_path / "tracks.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE frames(frame INTEGER PRIMARY KEY,payload TEXT)")
        connection.executemany("INSERT INTO frames VALUES(?,?)",
                               ((row["frame"], json.dumps(row)) for row in rows))
    binding = dict(source_video=dict(fps=FPS, frame_count=len(rows)))
    events, _ = infer_video_events(TrackStore(database), binding, SCALE,
                                   config_seed_parameters({}))
    departures = [event for event in events if event["canonical_behavior"] == "dispersal"]
    assert departures, "the default 40-second context loses a later real formation"
    assert all(event["mouse_ids"] == [100, 101, 102] for event in departures)
    assert len(departures) == 1
    assert 28. <= departures[0]["start_frame"] / FPS < 29.
    assert 36. <= departures[0]["end_frame"] / FPS < 36.3


def test_departure_is_invariant_to_observed_quiet_prefix_padding(tmp_path):
    first = _departures(tmp_path, _rows(seconds=45.))
    shifted = _departures(tmp_path, _rows(seconds=52., shift=7.))
    assert first and shifted
    assert _intervals(first) == [(low - 210, high - 210)
                                 for low, high in _intervals(shifted)]
    assert {c.target_ids for c in first} == {c.target_ids for c in shifted}
    assert {c.evidence for c in first} == {c.evidence for c in shifted}


def test_missing_prefix_is_not_filled_or_used_as_an_origin(tmp_path):
    rows = _rows(seconds=45.)
    first = _departures(tmp_path, rows)
    shifted = _rows(seconds=48., shift=3.)
    for row in shifted[:90]:
        row["detections"] = []
    later = _departures(tmp_path, shifted)
    assert first and later
    assert _intervals(first) == [(low - 90, high - 90) for low, high in _intervals(later)]
    assert all(low >= 90 for low, _ in _intervals(later))


def test_second_formation_has_its_own_departure_without_episode_union(tmp_path):
    candidates = _departures(tmp_path, _rows(second=True))
    intervals = _intervals(candidates)
    assert len(intervals) == 2, "each observed formation needs an independent origin"
    assert 28. <= intervals[0][0] / FPS < 29.
    assert 36. <= intervals[0][1] / FPS < 36.3
    assert 48. <= intervals[1][0] / FPS < 49.
    assert 56. <= intervals[1][1] / FPS < 56.3
    assert all(c.target_ids == (100, 101, 102) for c in candidates)
    assert all(len(c.frame_intervals) == 1 and c.evidence > 5. for c in candidates)


def test_short_first_departure_cannot_borrow_motion_from_second_formation(tmp_path):
    candidates = _departures(tmp_path, _rows(second=True, departure_seconds=4.))
    intervals = _intervals(candidates)
    assert len(intervals) == 1
    assert 48. <= intervals[0][0] / FPS < 49.
    assert 56. <= intervals[0][1] / FPS < 56.3


@pytest.mark.parametrize("stationary_members", [1, 2])
def test_neighbor_pair_or_triple_reconnection_cannot_cut_ongoing_departure(tmp_path, stationary_members):
    time = np.arange(360) / FPS
    distance = 1. + 2. * np.minimum(time, 8.)
    centers = np.zeros((len(time), 2 + stationary_members, 2))
    centers[:, 0, 0], centers[:, 1, 0] = -distance, distance
    centers[:, 2:, 1] = np.arange(1, stationary_members + 1) * .5
    # A measured small inward movement reconnects a neighbor edge at 4.53 s.
    # It neither returns to the original hull nor ends the other mouse's
    # continuous outward bout. The regrouped subset is not a new episode.
    centers[134:139, 0, 0] = [-10.2, -10.2, -9.8, -10.2, -10.2]
    candidates = _departures(tmp_path, _pose_rows(centers))
    assert candidates, "a transient graph edge before5s erased the actual8s departure"
    assert all(c.target_ids == (100, 101) for c in candidates)
    assert all(c.evidence > 7.5 for c in candidates)


def test_genuine_reformation_at_another_location_owns_the_second_origin(tmp_path):
    rows = _rows(second=True)
    # Relocation occurs during re-formation, after the first measured outward
    # bout. The second group never returns to the first origin's spatial hull.
    for frame, row in enumerate(rows):
        shift_px = 400. * _smooth((frame / FPS - 40.) / 6.)
        for detection in row['detections']:
            detection['box'][0] += shift_px
            detection['box'][2] += shift_px
            for point in detection['keypoints']:
                point[0] += shift_px
    intervals = _intervals(_departures(tmp_path, rows))
    assert len(intervals) == 2
    assert 28. <= intervals[0][0] / FPS < 29. and intervals[0][1] / FPS < 36.3
    assert 48. <= intervals[1][0] / FPS < 49. and 56. <= intervals[1][1] / FPS < 56.3


def test_partial_reformation_cannot_cancel_other_original_departing_members(tmp_path):
    time = np.arange(480) / FPS
    early = _smooth(time / 2.)
    later = 16. * _smooth((time - 6.) / 8.)
    centers = np.zeros((len(time), 5, 2))
    centers[:, 0, 0] = -1. + 19. * early + later
    centers[:, 1, 0] = 1. + 21. * early + later
    centers[:, 2, 1] = 1. + 19. * early + later
    centers[:, 3, 1] = -1. - 19. * early - later
    centers[:, 4, 0] = 50. - 30. * _smooth((time - 3.) / 1.)
    centers[:, 4, 1] = 1.
    # After a short outward bout, 100/101/104 make another local group.
    # That partial regrouping says nothing about 102/103's ongoing departure
    # from the first four-ID origin. The early far ID104 cannot be backfilled.
    candidates = _departures(tmp_path, _pose_rows(centers))
    original = [c for c in candidates if c.target_ids == (100, 101, 102, 103)]
    assert original, "a partial three-ID formation cancelled the original four-ID process"
    assert all(c.evidence > 7. for c in original)
    assert all(6. <= low / FPS < 7. and 14. <= high / FPS < 14.3
               for c in original for low, high in c.frame_intervals)


def test_unknown_gap_after_outward_motion_cannot_reset_first_observation_admission(tmp_path):
    time = np.arange(450) / FPS
    radius = 1. + 4.7 * np.minimum(time / 3., 1.) + 16. * _smooth((time - 5.) / 8.)
    angles = np.array([0., 2. * np.pi / 3., 4. * np.pi / 3., 1.5 * np.pi])
    directions = np.c_[np.cos(angles), np.sin(angles)]
    centers = radius[:, None, None] * directions
    centers[time < 5., 3] = [40., 0.]
    rows = _pose_rows(centers)
    for row in rows[90:150]:
        row['detections'] = []
    candidates = _departures(tmp_path, rows)
    assert candidates
    assert all(103 not in c.target_ids for c in candidates), (
        "an unobserved bout end cannot reset the initially far ID's admission")


@pytest.mark.parametrize("translated", [False, True])
def test_observed_reformation_after_unknown_gap_can_close_previous_origin(tmp_path, translated):
    rows = _rows(second=True)
    if translated:
        for frame, row in enumerate(rows):
            shift_px = 400. * _smooth((frame / FPS - 40.) / 6.)
            for detection in row['detections']:
                detection['box'][0] += shift_px
                detection['box'][2] += shift_px
                for point in detection['keypoints']:
                    point[0] += shift_px
    def origins(values):
        features = _features(tmp_path, values)
        near = np.isfinite(features.pair_distance_cm) & (features.pair_distance_cm <= 10.)
        return group._compact_origin_episodes(features, near, 10.)
    uninterrupted = origins(rows)
    for row in rows[900:1050]:
        row['detections'] = []
    assert origins(rows) == uninterrupted, (
        "unknown motion end must not prevent a later observed reformation from owning its origin")
    candidates = _departures(tmp_path, rows)
    assert len(_intervals(candidates)) == 1
    assert all(48. <= low / FPS < 49. and 56. <= high / FPS < 56.3
               for low, high in _intervals(candidates))


def test_observed_pause_after_gap_can_reform_without_another_new_neighbor_edge(tmp_path):
    time = np.arange(510) / FPS
    radius = 1. + 4.3 * np.minimum(time / 3., 1.)
    radius += .2 * np.clip((time - 5.) / .5, 0., 1.)
    radius += 16. * _smooth((time - 7.) / 8.)
    angles = np.array([0., 2. * np.pi / 3., 4. * np.pi / 3., 1.5 * np.pi])
    centers = radius[:, None, None] * np.c_[np.cos(angles), np.sin(angles)]
    centers[time < 5., 3] = [40., 0.]
    rows = _pose_rows(centers)
    for row in rows[90:150]:
        row['detections'] = []
    candidates = _departures(tmp_path, rows)
    assert candidates and any(103 in candidate.target_ids for candidate in candidates), (
        "a measured long pause after reappearance must allow a genuinely new admission clock")


@pytest.mark.parametrize("first_seen_far", [False, True])
@pytest.mark.parametrize("unknown_gap", [False, True])
def test_late_member_admission_uses_first_observation_in_same_origin(tmp_path, first_seen_far, unknown_gap):
    time = np.arange(600) / FPS
    distance = 1. + 16. * _smooth((time - 4.) / 8.)
    centers = np.zeros((len(time), 3, 2))
    centers[:, 0, 0] = -distance
    centers[:, 1, 0] = distance
    centers[:, 2, 1] = np.where(time < 2., 30., 1.)
    rows = _pose_rows(centers)
    if not first_seen_far:
        for row in rows[:60]:
            row["detections"] = row["detections"][:2]
    if unknown_gap:
        for row in rows[30:35]:
            row["detections"] = []
    candidates = _departures(tmp_path, rows)
    if first_seen_far:
        assert not candidates, "entering later cannot reset the original first-observation gate"
    else:
        assert candidates and all(c.target_ids == (100, 101) for c in candidates)


@pytest.mark.parametrize("options", [dict(never_compact=True), dict(departure_seconds=4.)])
def test_departure_requires_real_origin_and_more_than_five_seconds(tmp_path, options):
    assert not _departures(tmp_path, _rows(seconds=45., **options))
