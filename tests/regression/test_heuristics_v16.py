"""Synthetic regressions for the 2026-10-15 behavior-rule repairs."""

from __future__ import annotations

import itertools
from dataclasses import replace

import numpy as np

from mouse_behavior.behavior.individual_rules import individual_rule_masks
from mouse_behavior.evaluation.calibrated_rule_ranker import (
    CandidateTable,
    build_candidate_table,
    predict_table,
)
from mouse_behavior.evaluation.confirmed_rules import group_rule_candidates, pair_rule_masks
from mouse_behavior.evaluation.exported_dataset import (
    CANONICAL_BEHAVIORS,
    GROUP_BEHAVIORS,
    ExportedFeatures,
    HeuristicParameters,
)
from mouse_behavior.evaluation.group_event_rules import _initial_compact_components
from mouse_behavior.evaluation.social_heuristic_config import SocialHeuristicConfig


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.r_[False, np.asarray(mask, dtype=bool), False].astype(np.int8)
    changes = np.diff(padded)
    return list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))


def _features(frames: int, mice: int, fps: float = 30.0) -> ExportedFeatures:
    track_ids = np.arange(100, 100 + mice, dtype=int)
    pairs = list(itertools.combinations(range(mice), 2))
    pair_i = np.asarray([left for left, _ in pairs], dtype=int)
    pair_j = np.asarray([right for _, right in pairs], dtype=int)
    valid = np.ones((frames, mice), dtype=bool)
    centers = np.zeros((frames, mice, 2), dtype=float)
    pair_distance = np.zeros((frames, len(pairs)), dtype=float)
    nearest = np.full((frames, mice), np.nan, dtype=float)
    for edge, (left, right) in enumerate(pairs):
        distance = np.linalg.norm(centers[:, left] - centers[:, right], axis=1)
        pair_distance[:, edge] = distance
    speed = np.zeros((frames, mice), dtype=float)
    velocity = np.zeros((frames, mice, 2), dtype=float)
    far = np.full((frames, len(pairs)), 10.0, dtype=float)
    boxes = np.zeros((frames, mice, 4), dtype=float)
    result = ExportedFeatures(
        fps=fps,
        track_ids=track_ids,
        valid=valid,
        centers_cm=centers,
        speed_cm_s=speed,
        velocity_cm_s=velocity,
        pair_i=pair_i,
        pair_j=pair_j,
        pair_distance_cm=pair_distance,
        pair_closing_speed_cm_s=np.zeros_like(pair_distance),
        pair_direction_similarity=np.zeros_like(pair_distance),
        pair_nose_head_cm=far.copy(),
        pair_nose_tail_cm=far.copy(),
        nearest_distance_cm=nearest,
        mean_nearest_distance_cm=nearest.copy(),
        cm_per_pixel=1.0,
        track_selection_truncated=False,
        selected_track_count=mice,
        reference_body_cm=7.918,
        pair_nose_head_left_to_right_cm=far.copy(),
        pair_nose_head_right_to_left_cm=far.copy(),
        pair_nose_tail_left_to_right_cm=far.copy(),
        pair_nose_tail_right_to_left_cm=far.copy(),
        collective_boxes_cm=boxes,
    )
    return result


def _refresh_pair_geometry(features: ExportedFeatures) -> None:
    frames, mice = features.valid.shape
    for edge, (left, right) in enumerate(zip(features.pair_i, features.pair_j, strict=True)):
        observed = features.valid[:, left] & features.valid[:, right]
        values = np.linalg.norm(
            features.centers_cm[:, left] - features.centers_cm[:, right], axis=1,
        )
        features.pair_distance_cm[:, edge] = np.where(observed, values, np.nan)
    features.nearest_distance_cm[:] = np.nan
    for frame in range(frames):
        for mouse in range(mice):
            others = [slot for slot in range(mice) if slot != mouse and features.valid[frame, slot]]
            if features.valid[frame, mouse] and others:
                distances = [np.linalg.norm(features.centers_cm[frame, mouse]
                                            - features.centers_cm[frame, slot]) for slot in others]
                features.nearest_distance_cm[frame, mouse] = min(distances)
    features.mean_nearest_distance_cm[:] = features.nearest_distance_cm


def _contact_features(frames: int = 60, fps: float = 30.0) -> ExportedFeatures:
    features = _features(frames, 2, fps)
    features.centers_cm[:, 1, 0] = 4.0
    _refresh_pair_geometry(features)
    return features


def test_locomotion_connects_point_8_second_pause_with_two_missing_frames():
    fps = 30.0
    frame_count = 120
    centers = np.zeros((frame_count, 1, 2), dtype=float)
    valid = np.ones((frame_count, 1), dtype=bool)
    pause_start, pause_end = 36, 60
    first_distance = np.arange(pause_start, dtype=float) * (10.0 / fps)
    centers[:pause_start, 0, 0] = first_distance
    centers[pause_start:pause_end, 0, 0] = first_distance[-1]
    centers[pause_end:, 0, 0] = first_distance[-1] + np.arange(
        1, frame_count - pause_end + 1, dtype=float,
    ) * (10.0 / fps)
    valid[47:49, 0] = False
    result = individual_rule_masks(centers, valid, fps)
    bouts = _runs(result.masks["locomotion"][:, 0])
    assert len(bouts) == 1
    assert bouts[0][0] < pause_start
    assert bouts[0][1] > pause_end


def test_short_foreshortened_mouse_remains_stationary_through_three_missing_frames():
    fps = 30.0
    frame_count = 90
    centers = np.zeros((frame_count, 1, 2), dtype=float)
    valid = np.ones((frame_count, 1), dtype=bool)
    valid[40:43, 0] = False
    keypoints = np.zeros((frame_count, 1, 7, 2), dtype=float)
    keypoints[:, 0, 0, 0] = 4.5
    keypoints[:, 0, 6, 0] = 0.0
    keypoints[~valid] = np.nan
    result = individual_rule_masks(centers, valid, fps, keypoints_cm=keypoints)
    assert result.masks["stationary"][40:43, 0].all()
    assert not result.masks["locomotion"][:, 0].any()


def test_single_long_id_dropout_is_unknown_not_heavy_attack():
    fps = 10.0
    features = _contact_features(frames=50, fps=fps)
    features.valid[12:24, 1] = False
    features.collective_boxes_cm[12:24, 1] = np.nan
    features.centers_cm[:, 0, 0] = np.arange(50) * 0.1
    features.centers_cm[:, 1, 0] = features.centers_cm[:, 0, 0] + 3.0
    features.speed_cm_s[8:11, 0] = 6.0
    features.speed_cm_s[24:27, 1] = 6.0
    _refresh_pair_geometry(features)
    # Exercise reacquisition independently of v17 Together winning this quiet pair.
    config = SocialHeuristicConfig(attack_priority=2, together_priority=3,
                                   vel_attack_min_cm_s=4.0)
    masks = pair_rule_masks(features, HeuristicParameters(confirmed_rules_v1=1.0), config)
    assert not masks["attack"][:, 0].any()
    assert not masks["attack"][12:24, 0].any()
    assert not masks["approach"][:, 0].any()


def test_head_contact_bridges_two_missing_keypoint_frames():
    features = _contact_features(frames=30, fps=30.0)
    features.centers_cm[:, 1, 0] = 7.5  # Outside Together; isolate contact bridging.
    _refresh_pair_geometry(features)
    directed = features.pair_nose_head_left_to_right_cm[:, 0]
    directed[:] = 10.0
    directed[:24] = 1.0
    directed[9:11] = np.nan
    # Historical two-frame bridge is explicit; the v27 .1s contact minimum
    # uses a shorter .05s gap and is exercised separately.
    config = SocialHeuristicConfig(nose_contact_min_duration_s=.5,
                                  contact_bridge_max_gap_s=.15,
                                  together_contact_veto_min_duration_s=.15)
    masks = pair_rule_masks(features, HeuristicParameters(confirmed_rules_v1=1.0), config)
    assert masks["head_contact"][:, 0].sum() >= 24
    assert not masks["approach"][:, 0].any()


def test_head_contact_priority_outranks_a_longer_tail_contact():
    features = _contact_features(frames=60, fps=30.0)
    features.centers_cm[:, 1, 0] = 7.5
    _refresh_pair_geometry(features)
    features.pair_nose_tail_left_to_right_cm[:, 0] = 10.0
    features.pair_nose_tail_left_to_right_cm[:45, 0] = 1.0
    features.pair_nose_head_left_to_right_cm[:, 0] = 10.0
    features.pair_nose_head_left_to_right_cm[15:30, 0] = 1.0
    # Historical 1.5 cm endpoints keep this semantic fixture independent of fitting.
    config = SocialHeuristicConfig(dist_nose_head_contact_max_cm=1.5,
                                   dist_nose_tail_contact_max_cm=1.5)
    masks = pair_rule_masks(features, HeuristicParameters(confirmed_rules_v1=1.0), config)
    assert masks["head_contact"][15:30, 0].all()
    assert masks["head_contact"][:, 0].sum() == 15
    assert not masks["tail_contact"][15:30, 0].any()
    assert masks["tail_contact"][:15, 0].all()
    assert masks["tail_contact"][30:45, 0].all()
    assert masks["tail_contact"][:, 0].sum() == 30


def test_huddle_keeps_late_members_and_suppresses_background_isolation_top1():
    fps = 30.0
    frames = 360
    features = _features(frames, 5, fps)
    features.valid[:, 2:4] = False
    features.valid[30:, 2:4] = True
    features.valid[90:135, 1] = False
    compact_positions = np.asarray([
        [0.0, 0.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5], [25.0, 0.0],
    ])
    features.centers_cm[:] = compact_positions
    features.centers_cm[210:, 2] = (-15.0, 0.0)
    features.centers_cm[210:, 3] = (-14.5, 0.5)
    features.centers_cm[~features.valid] = np.nan
    _refresh_pair_geometry(features)
    parameters = replace(HeuristicParameters(), confirmed_rules_v1=1.0)
    table = build_candidate_table(features, [parameters], behavior_subset=GROUP_BEHAVIORS)
    target_ids = (100, 101, 102, 103)
    huddle_index = next(
        index for index, candidate in enumerate(table.candidates)
        if candidate.behavior == "huddle" and candidate.target_ids == target_ids
        and candidate.evidence > 0
    )
    isolation_index = next(
        index for index, candidate in enumerate(table.candidates)
        if candidate.behavior == "isolation" and candidate.target_ids == (104,)
    )
    assert table.candidates[isolation_index].evidence == 0.0
    focused = CandidateTable(
        candidates=(table.candidates[huddle_index], table.candidates[isolation_index]),
        values=table.values[[huddle_index, isolation_index]],
        feature_names=table.feature_names,
    )
    model = {
        "feature_names": list(focused.feature_names),
        "coefficients": {behavior: [0.0] * len(focused.feature_names)
                         for behavior in CANONICAL_BEHAVIORS},
        "require_positive_evidence": True,
    }
    prediction = predict_table(focused, model)
    assert prediction.behavior == "huddle"
    assert prediction.target_ids == target_ids


def test_dispersal_backfill_preserves_initial_group_but_requires_departure_boundary():
    fps = 30.0
    frames = 210
    features = _features(frames, 4, fps)
    features.valid[:3, 3] = False
    features.centers_cm[:] = np.asarray([
        [-0.5, 0.0], [0.5, 0.0], [0.0, 0.5], [0.0, -0.5],
    ])
    features.centers_cm[30:, 0, 0] = np.linspace(-0.5, -14.0, frames - 30)
    features.centers_cm[45:, 1, 0] = np.linspace(0.5, 14.0, frames - 45)
    features.centers_cm[45:, 3, 1] = np.linspace(-0.5, -14.0, frames - 45)
    features.centers_cm[~features.valid] = np.nan
    _refresh_pair_geometry(features)
    params = replace(HeuristicParameters(), confirmed_rules_v1=1.0)
    assert (0, 1, 2, 3) in _initial_compact_components(
        features,
        np.isfinite(features.pair_distance_cm) & (features.pair_distance_cm <= 9.8975),
        9.8975,
    )
    candidates = group_rule_candidates(features, params, ())
    # V21 admits ID103 from its own observed in-origin boundary;
    # ID102 remains in the pile and cannot become a departed target.
    assert any(candidate.behavior == "dispersal"
               and candidate.target_ids == (100, 101)
               for candidate in candidates)
    assert any(candidate.behavior == "dispersal"
               and candidate.target_ids == (100, 101, 103) for candidate in candidates)
    assert not any(candidate.behavior == "dispersal" and 102 in candidate.target_ids
                   for candidate in candidates)


