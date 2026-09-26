from __future__ import annotations

import json
from dataclasses import replace

import numpy as np

from mouse_behavior.evaluation.exported_dataset import (
    ExportedFeatures,
    HeuristicParameters,
    SampleRecord,
    _group_resting_at_onset,
    _durations_by_column,
    _prioritize_attack_over_approach,
    calibration_rank,
    config_seed_parameters,
    evaluate_predictions,
    filter_behaviors,
    group_rule_calibration_rank,
    group_rule_parameter_tie_break,
    load_exported_features,
    minimum_behavior_accuracy,
    predict_features,
    split_by_recording_session,
    strict_top1_regressions,
    training_target_reached,
)


def _record(
    sample_id: str,
    behavior: str,
    session: str,
    *,
    mouse_ids: tuple[int, ...] = (1, 2),
) -> SampleRecord:
    return SampleRecord(
        sample_id=sample_id,
        behavior=behavior,
        canonical_behavior={
            "Huddling": "huddle",
            "Social clustering": "social_clustering",
            "Static": "stationary",
            "Together": "together",
        }.get(behavior, behavior.lower()),
        sample_dir="sample",
        annotation_path="annotation.json",
        metadata_path="metadata.json",
        tracks_path="tracks.json",
        clip_path="clip.mp4",
        source_video=session,
        recording_session=session,
        fps=30.0,
        frame_count=4,
        mouse_ids=mouse_ids,
        confidence="certain",
        frame_start=0,
        frame_end=3,
        time_start=0.0,
        time_end=0.1,
    )


def test_filter_and_session_split_preserve_behavior_coverage() -> None:
    records = [
        _record("a1", "Static", "train.mp4"),
        _record("a2", "Static", "val.mp4"),
        _record("b1", "Together", "train.mp4"),
        _record("b2", "Together", "val.mp4"),
        _record("excluded", "Rare", "train.mp4"),
    ]
    kept, counts, excluded = filter_behaviors(records, minimum_count=2)
    assert [record.sample_id for record in kept] == ["a1", "a2", "b1", "b2"]
    assert counts == {"Rare": 1, "Static": 2, "Together": 2}
    assert excluded == {"Rare": 1}
    train, validation = split_by_recording_session(kept, validation_sessions={"val.mp4"})
    assert {record.sample_id for record in train} == {"a1", "b1"}
    assert {record.sample_id for record in validation} == {"a2", "b2"}


def test_predict_and_metrics_use_clip_local_features() -> None:
    frames = 4
    valid = np.ones((frames, 2), dtype=bool)
    speed = np.zeros((frames, 2), dtype=np.float32)
    pair_distance = np.full((frames, 1), 5.0, dtype=np.float32)
    features = ExportedFeatures(
        fps=30.0,
        track_ids=np.asarray([1, 2]),
        valid=valid,
        centers_cm=np.zeros((frames, 2, 2), dtype=np.float32),
        speed_cm_s=speed,
        velocity_cm_s=np.zeros((frames, 2, 2), dtype=np.float32),
        pair_i=np.asarray([0]),
        pair_j=np.asarray([1]),
        pair_distance_cm=pair_distance,
        pair_closing_speed_cm_s=np.zeros_like(pair_distance),
        pair_direction_similarity=np.ones_like(pair_distance),
        pair_nose_head_cm=np.full_like(pair_distance, 1.9),
        pair_nose_tail_cm=np.full_like(pair_distance, 9.0),
        nearest_distance_cm=np.full((frames, 2), 5.0, dtype=np.float32),
        mean_nearest_distance_cm=np.full(frames, 5.0, dtype=np.float32),
        cm_per_pixel=0.1,
        track_selection_truncated=False,
        selected_track_count=2,
    )
    prediction = predict_features(
        features,
        parameters=HeuristicParameters(
            stationary_min_duration_s=0.05,
            pair_min_duration_s=0.05,
            together_min_duration_s=0.05,
            contact_min_cumulative_seconds=0.1,
            huddle_distance_cm=8.0,
            huddle_width_multiplier=3.0,
        ),
    )
    assert "stationary" in prediction["detected"]
    assert "together" in prediction["detected"]
    assert "nose_head_contact" in prediction["detected"]
    assert prediction["identity_scores"]["1"]["stationary"] > 0.0
    assert prediction["pair_scores"]["1,2"]["together"] > 0.0

    moving_pair = predict_features(
        replace(features, speed_cm_s=np.full_like(features.speed_cm_s, 12.0)),
        parameters=HeuristicParameters(
            stationary_max_speed_cm_s=10.0,
            together_max_individual_speed_cm_s=10.0,
            together_min_duration_s=0.05,
            huddle_distance_cm=8.0,
            huddle_width_multiplier=3.0,
        ),
    )
    assert moving_pair.get("pair_scores", {}).get("1,2", {}).get("together", 0.0) == 0.0
    rows = [
        (_record("a", "Static", "train.mp4", mouse_ids=(1,)), prediction),
        (_record("b", "Together", "train.mp4"), prediction),
    ]
    metrics = evaluate_predictions(rows)
    assert metrics["n_samples"] == 2
    assert metrics["per_behavior"]["stationary"]["support"] == 1
    assert 0.0 <= metrics["target_id_accuracy"] <= 1.0


def test_contact_rule_uses_absolute_head_or_tail_distance() -> None:
    frames = 6
    valid = np.ones((frames, 2), dtype=bool)
    distance = np.full((frames, 1), 3.0, dtype=np.float32)
    features = ExportedFeatures(
        fps=10.0,
        track_ids=np.asarray([1, 2]),
        valid=valid,
        centers_cm=np.zeros((frames, 2, 2), dtype=np.float32),
        speed_cm_s=np.zeros((frames, 2), dtype=np.float32),
        velocity_cm_s=np.zeros((frames, 2, 2), dtype=np.float32),
        pair_i=np.asarray([0]),
        pair_j=np.asarray([1]),
        pair_distance_cm=distance,
        pair_closing_speed_cm_s=np.zeros_like(distance),
        pair_direction_similarity=np.zeros_like(distance),
        pair_nose_head_cm=np.full_like(distance, 1.9),
        pair_nose_tail_cm=np.full_like(distance, 1.8),
        nearest_distance_cm=np.full((frames, 2), 3.0, dtype=np.float32),
        mean_nearest_distance_cm=np.full(frames, 3.0, dtype=np.float32),
        cm_per_pixel=0.1,
        track_selection_truncated=False,
        selected_track_count=2,
    )
    ambiguous = predict_features(features, HeuristicParameters(pair_min_duration_s=0.1))
    assert ambiguous["pair_scores"]["1,2"]["nose_head_contact"] > 0.0
    assert ambiguous["pair_scores"]["1,2"]["nose_tail_contact"] > 0.0


def test_target_id_accuracy_does_not_credit_the_wrong_mouse() -> None:
    prediction = {
        "identity_scores": {
            "1": {"stationary": 0.0, "walking": 2.0, "isolation": 0.0},
            "2": {"stationary": 3.0, "walking": 0.0, "isolation": 0.0},
        },
        "pair_scores": {},
    }

    metrics = evaluate_predictions(
        [(_record("static-target-1", "Static", "val.mp4", mouse_ids=(1,)), prediction)]
    )

    assert metrics["target_id_accuracy"] == 0.0
    assert metrics["per_behavior"]["stationary"]["correct"] == 0
    assert metrics["confusion"] == {"stationary->walking": 1}


def test_target_id_accuracy_is_behavior_recognition_not_strict_top1() -> None:
    prediction = {
        "identity_scores": {
            "1": {"stationary": 1.0, "walking": 2.0, "isolation": 0.0},
        },
        "pair_scores": {},
    }

    metrics = evaluate_predictions(
        [(_record("static-target-1", "Static", "val.mp4", mouse_ids=(1,)), prediction)]
    )

    assert metrics["target_id_accuracy"] == 1.0
    assert metrics["strict_target_id_accuracy"] == 0.0
    assert metrics["per_behavior"]["stationary"]["correct"] == 1
    assert metrics["per_behavior"]["stationary"]["strict_top1_correct"] == 0


def test_pair_behavior_requires_the_labeled_id_pair() -> None:
    prediction = {
        "identity_scores": {},
        "pair_scores": {
            "1,2": {"approach": 1.5, "together": 0.0},
            "1,3": {"approach": 0.0, "together": 4.0},
        },
    }

    metrics = evaluate_predictions(
        [(_record("together-1-2", "Together", "val.mp4", mouse_ids=(1, 2)), prediction)]
    )

    assert metrics["target_id_accuracy"] == 0.0
    assert metrics["confusion"] == {"together->approach": 1}


def test_group_behavior_requires_evidence_for_every_labeled_id() -> None:
    prediction = {
        "identity_scores": {
            "1": {"huddle": 3.0, "social_clustering": 1.0},
            "2": {"huddle": 3.0, "social_clustering": 1.0},
            "3": {"huddle": 0.0, "social_clustering": 1.0},
        },
        "pair_scores": {},
    }

    metrics = evaluate_predictions(
        [(_record("huddle-1-2-3", "Huddling", "val.mp4", mouse_ids=(1, 2, 3)), prediction)]
    )

    assert metrics["target_id_accuracy"] == 0.0
    assert metrics["confusion"] == {"huddle->social_clustering": 1}


def test_group_scoring_requires_one_event_containing_all_annotated_mice() -> None:
    prediction = {
        "track_ids": [1, 2, 3],
        "identity_scores": {
            "1": {"huddle": 4.0},
            "2": {"huddle": 4.0},
            "3": {"huddle": 4.0},
        },
        "group_scores": {
            "1,2": {"huddle": 4.0},
            "1,3": {"huddle": 4.0},
            "2,3": {"huddle": 4.0},
        },
        "pair_scores": {},
    }

    metrics = evaluate_predictions(
        [(_record("huddle-split-events", "Huddling", "val.mp4", mouse_ids=(1, 2, 3)), prediction)]
    )

    assert metrics["target_id_accuracy"] == 0.0
    assert metrics["strict_target_id_accuracy"] == 0.0
    assert metrics["confusion"] == {"huddle-><none>": 1}


def test_strict_top1_rejects_extra_group_member_ids() -> None:
    prediction = {
        "track_ids": [1, 2, 3, 4],
        "identity_scores": {str(track_id): {} for track_id in (1, 2, 3, 4)},
        "group_scores": {"1,2,3,4": {"huddle": 2.0}},
        "pair_scores": {},
    }

    metrics = evaluate_predictions(
        [(_record("huddle-extra-member", "Huddling", "val.mp4", mouse_ids=(1, 2, 3)), prediction)]
    )

    assert metrics["target_id_accuracy"] == 1.0
    assert metrics["strict_target_id_accuracy"] == 0.0
    assert metrics["per_behavior"]["huddle"]["strict_top1_correct"] == 0
    assert metrics["per_behavior"]["huddle"]["exact_id_set_correct"] == 0


def test_isolation_loses_for_same_target_if_it_is_in_a_group_event() -> None:
    prediction = {
        "track_ids": [1, 2, 3],
        "identity_scores": {
            "1": {"isolation": 10.0},
            "2": {},
            "3": {},
        },
        "group_scores": {"1,2,3": {"huddle": 2.0}},
        "pair_scores": {},
    }

    metrics = evaluate_predictions(
        [
            (
                _record("isolation-target-in-huddle", "Isolation", "val.mp4", mouse_ids=(1,)),
                prediction,
            )
        ]
    )

    assert metrics["target_id_accuracy"] == 0.0
    assert metrics["confusion"] == {"isolation->huddle": 1}


def _group_formation_features(*, moving_members: int) -> ExportedFeatures:
    frames = 12
    pair_i = np.asarray([0, 0, 1], dtype=np.int64)
    pair_j = np.asarray([1, 2, 2], dtype=np.int64)
    pair_distance = np.full((frames, 3), 28.0, dtype=np.float32)
    pair_distance[3:, :] = 24.0
    speed = np.zeros((frames, 3), dtype=np.float32)
    speed[3:9, :moving_members] = 6.0
    nearest = np.full((frames, 3), 28.0, dtype=np.float32)
    nearest[3:, :] = 24.0
    return ExportedFeatures(
        fps=1.0,
        track_ids=np.asarray([1, 2, 3]),
        valid=np.ones((frames, 3), dtype=bool),
        centers_cm=np.zeros((frames, 3, 2), dtype=np.float32),
        speed_cm_s=speed,
        velocity_cm_s=np.zeros((frames, 3, 2), dtype=np.float32),
        pair_i=pair_i,
        pair_j=pair_j,
        pair_distance_cm=pair_distance,
        pair_closing_speed_cm_s=np.zeros_like(pair_distance),
        pair_direction_similarity=np.ones_like(pair_distance),
        pair_nose_head_cm=np.full_like(pair_distance, 100.0),
        pair_nose_tail_cm=np.full_like(pair_distance, 100.0),
        nearest_distance_cm=nearest,
        mean_nearest_distance_cm=np.full(frames, 24.0, dtype=np.float32),
        cm_per_pixel=0.1,
        track_selection_truncated=False,
        selected_track_count=3,
    )


def test_social_clustering_needs_multiple_moving_members_and_near_start() -> None:
    parameters = HeuristicParameters(
        isolation_distance_cm=100.0,
        clustering_max_distance_cm=30.0,
        clustering_initial_max_distance_cm=30.0,
        clustering_min_mean_speed_cm_s=5.0,
        clustering_min_moving_member_fraction=2.0 / 3.0,
        clustering_window_s=2.0,
        clustering_min_nearest_neighbor_drop_cm=2.0,
        clustering_min_duration_s=5.0,
    )

    one_mouse_moves = predict_features(_group_formation_features(moving_members=1), parameters)
    two_mice_move = predict_features(_group_formation_features(moving_members=2), parameters)
    distant_start = predict_features(
        _group_formation_features(moving_members=3),
        replace(parameters, clustering_initial_max_distance_cm=20.0),
    )
    huddle_member = predict_features(
        _group_formation_features(moving_members=3),
        replace(parameters, isolation_distance_cm=20.0),
    )

    assert "1,2,3" not in one_mouse_moves["group_scores"]
    assert two_mice_move["group_scores"]["1,2,3"]["social_clustering"] > 0.0
    assert "1,2,3" not in distant_start["group_scores"]
    assert huddle_member["group_scores"]["1,2,3"]["social_clustering"] > 0.0
    assert all(values["isolation"] == 0.0 for values in huddle_member["identity_scores"].values())


def test_social_clustering_member_support_excludes_one_frame_bystander() -> None:
    features = _group_formation_features(moving_members=3)
    valid = np.column_stack((features.valid, np.arange(12) == 4))
    fourth_pair = np.full((12, 1), np.inf, dtype=np.float32)
    fourth_pair[4, 0] = 24.0
    distances = np.column_stack((features.pair_distance_cm, fourth_pair))
    nose = np.full_like(distances, 100.0)
    features = replace(
        features,
        track_ids=np.asarray([1, 2, 3, 4]),
        valid=valid,
        centers_cm=np.zeros((12, 4, 2), dtype=np.float32),
        speed_cm_s=np.column_stack((features.speed_cm_s, np.zeros(12))),
        velocity_cm_s=np.zeros((12, 4, 2), dtype=np.float32),
        pair_i=np.asarray([0, 0, 1, 2]),
        pair_j=np.asarray([1, 2, 2, 3]),
        pair_distance_cm=distances,
        pair_closing_speed_cm_s=np.zeros_like(distances),
        pair_direction_similarity=np.ones_like(distances),
        pair_nose_head_cm=nose,
        pair_nose_tail_cm=nose,
        nearest_distance_cm=np.column_stack((features.nearest_distance_cm, np.full(12, 24.0))),
        selected_track_count=4,
    )
    baseline = HeuristicParameters(
        clustering_initial_max_distance_cm=30.0,
        clustering_min_mean_speed_cm_s=5.0,
        clustering_min_duration_s=5.0,
    )
    pruned = replace(baseline, clustering_min_member_support_fraction=0.5)

    baseline_groups = predict_features(features, baseline)["group_scores"]
    pruned_groups = predict_features(features, pruned)["group_scores"]

    assert baseline_groups["1,2,3,4"]["social_clustering"] > 0
    assert pruned_groups["1,2,3"]["social_clustering"] > 0
    assert "1,2,3,4" not in pruned_groups


def test_already_close_moving_group_can_be_social_clustering_without_distance_drop() -> None:
    features = _group_formation_features(moving_members=3)
    frames = len(features.valid)
    centers = np.zeros_like(features.centers_cm)
    centers[:, :, 0] = np.arange(frames)[:, None] * (8.0 / (frames - 1))
    features = replace(
        features,
        centers_cm=centers,
        speed_cm_s=np.full_like(features.speed_cm_s, 3.0),
        pair_distance_cm=np.full_like(features.pair_distance_cm, 12.0),
        nearest_distance_cm=np.full_like(features.nearest_distance_cm, 12.0),
    )
    parameters = HeuristicParameters(
        clustering_min_duration_s=5.0,
        clustering_motion_max_distance_cm=15.0,
        clustering_motion_min_displacement_cm=6.0,
    )

    assert predict_features(features, parameters)["group_scores"]["1,2,3"]["social_clustering"] > 0
    stable = replace(features, centers_cm=np.zeros_like(centers))
    assert "social_clustering" not in predict_features(stable, parameters)["group_scores"].get(
        "1,2,3", {}
    )
    short = replace(
        features,
        valid=features.valid[:4],
        centers_cm=features.centers_cm[:4],
        speed_cm_s=features.speed_cm_s[:4],
        velocity_cm_s=features.velocity_cm_s[:4],
        pair_distance_cm=features.pair_distance_cm[:4],
        pair_closing_speed_cm_s=features.pair_closing_speed_cm_s[:4],
        pair_direction_similarity=features.pair_direction_similarity[:4],
        pair_nose_head_cm=features.pair_nose_head_cm[:4],
        pair_nose_tail_cm=features.pair_nose_tail_cm[:4],
        nearest_distance_cm=features.nearest_distance_cm[:4],
        mean_nearest_distance_cm=features.mean_nearest_distance_cm[:4],
    )
    assert "social_clustering" not in predict_features(short, parameters)["group_scores"].get(
        "1,2,3", {}
    )


def test_one_track_displacement_outlier_does_not_override_tight_huddle() -> None:
    features = _group_formation_features(moving_members=3)
    frames = len(features.valid)
    centers = np.zeros_like(features.centers_cm)
    centers[:, 1, 0] = np.arange(frames) * (22.0 / (frames - 1))
    features = replace(
        features,
        centers_cm=centers,
        speed_cm_s=np.full_like(features.speed_cm_s, 3.0),
        pair_distance_cm=np.full_like(features.pair_distance_cm, 9.0),
        nearest_distance_cm=np.full_like(features.nearest_distance_cm, 9.0),
    )
    parameters = HeuristicParameters(
        clustering_motion_max_distance_cm=15.0,
        clustering_motion_min_displacement_cm=6.0,
    )

    assert "social_clustering" not in predict_features(features, parameters)["group_scores"].get(
        "1,2,3", {}
    )
    loose = replace(
        features,
        pair_distance_cm=np.full_like(features.pair_distance_cm, 10.0),
        nearest_distance_cm=np.full_like(features.nearest_distance_cm, 10.0),
    )
    assert predict_features(loose, parameters)["group_scores"]["1,2,3"]["social_clustering"] > 0


def test_huddling_keeps_a_reacquired_member_with_low_visibility() -> None:
    frames = 6
    pair_i = np.asarray([0, 0, 0, 1, 1, 2], dtype=np.int64)
    pair_j = np.asarray([1, 2, 3, 2, 3, 3], dtype=np.int64)
    pair_distance = np.full((frames, 6), 2.0, dtype=np.float32)
    valid = np.ones((frames, 4), dtype=bool)
    valid[1:5, 3] = False
    for pair_index, (left, right) in enumerate(zip(pair_i, pair_j)):
        pair_distance[~(valid[:, left] & valid[:, right]), pair_index] = np.nan
    nearest = np.full((frames, 4), 2.0, dtype=np.float32)
    nearest[1:5, 3] = np.inf
    features = ExportedFeatures(
        fps=1.0,
        track_ids=np.asarray([1, 2, 3, 4]),
        valid=valid,
        centers_cm=np.zeros((frames, 4, 2), dtype=np.float32),
        speed_cm_s=np.zeros((frames, 4), dtype=np.float32),
        velocity_cm_s=np.zeros((frames, 4, 2), dtype=np.float32),
        pair_i=pair_i,
        pair_j=pair_j,
        pair_distance_cm=pair_distance,
        pair_closing_speed_cm_s=np.zeros_like(pair_distance),
        pair_direction_similarity=np.ones_like(pair_distance),
        pair_nose_head_cm=np.full_like(pair_distance, 100.0),
        pair_nose_tail_cm=np.full_like(pair_distance, 100.0),
        nearest_distance_cm=nearest,
        mean_nearest_distance_cm=np.full(frames, 2.0, dtype=np.float32),
        cm_per_pixel=0.1,
        track_selection_truncated=False,
        selected_track_count=4,
    )
    base = HeuristicParameters(
        huddle_width_multiplier=3.0,
        huddle_min_duration_s=1.0,
        isolation_distance_cm=100.0,
    )

    strict_support = predict_features(
        features, replace(base, huddle_min_member_support_fraction=0.75)
    )
    dropout_tolerant = predict_features(
        features, replace(base, huddle_min_member_support_fraction=0.25)
    )

    assert "1,2,3" in strict_support["group_scores"]
    assert "1,2,3,4" not in strict_support["group_scores"]
    assert dropout_tolerant["group_scores"]["1,2,3,4"]["huddle"] > 0.0


def test_group_rule_tuning_penalizes_false_positives_before_top1_recall() -> None:
    broad_rule = {
        "per_behavior": {
            "huddle": {"support": 1, "strict_top1_accuracy": 0.9, "f1": 0.3},
            "isolation": {"support": 1, "strict_top1_accuracy": 0.9, "f1": 0.3},
            "social_clustering": {"support": 1, "strict_top1_accuracy": 0.9, "f1": 0.3},
        },
        "strict_target_id_accuracy": 0.9,
        "macro_f1": 0.3,
    }
    specific_rule = {
        "per_behavior": {
            "huddle": {"support": 1, "strict_top1_accuracy": 0.7, "f1": 0.8},
            "isolation": {"support": 1, "strict_top1_accuracy": 0.7, "f1": 0.8},
            "social_clustering": {"support": 1, "strict_top1_accuracy": 0.7, "f1": 0.8},
        },
        "strict_target_id_accuracy": 0.7,
        "macro_f1": 0.8,
    }

    assert group_rule_calibration_rank(specific_rule) > group_rule_calibration_rank(broad_rule)


def test_group_rule_ties_prefer_narrow_clustering_and_huddle_visibility_floor() -> None:
    assert group_rule_parameter_tie_break(
        "clustering_initial_max_distance_cm", 24.0, 6
    ) > group_rule_parameter_tie_break("clustering_initial_max_distance_cm", 30.0, 0)
    assert group_rule_parameter_tie_break(
        "huddle_min_member_support_fraction", 0.25, 3
    ) > group_rule_parameter_tie_break("huddle_min_member_support_fraction", 0.1, 1)


def test_walking_threshold_is_independent_from_stationary_threshold() -> None:
    frames = 4
    valid = np.ones((frames, 1), dtype=bool)
    speed = np.full((frames, 1), 3.0, dtype=np.float32)
    features = ExportedFeatures(
        fps=30.0,
        track_ids=np.asarray([1]),
        valid=valid,
        centers_cm=np.zeros((frames, 1, 2), dtype=np.float32),
        speed_cm_s=speed,
        velocity_cm_s=np.zeros((frames, 1, 2), dtype=np.float32),
        pair_i=np.asarray([], dtype=np.int64),
        pair_j=np.asarray([], dtype=np.int64),
        pair_distance_cm=np.empty((frames, 0), dtype=np.float32),
        pair_closing_speed_cm_s=np.empty((frames, 0), dtype=np.float32),
        pair_direction_similarity=np.empty((frames, 0), dtype=np.float32),
        pair_nose_head_cm=np.empty((frames, 0), dtype=np.float32),
        pair_nose_tail_cm=np.empty((frames, 0), dtype=np.float32),
        nearest_distance_cm=np.full((frames, 1), np.inf, dtype=np.float32),
        mean_nearest_distance_cm=np.full(frames, np.inf, dtype=np.float32),
        cm_per_pixel=0.1,
        track_selection_truncated=False,
        selected_track_count=1,
    )

    prediction = predict_features(
        features,
        HeuristicParameters(
            stationary_max_speed_cm_s=4.0,
            walking_min_speed_cm_s=2.0,
            walking_max_speed_cm_s=10.0,
            stationary_min_duration_s=0.05,
            walking_min_duration_s=0.05,
        ),
    )

    assert prediction["identity_scores"]["1"]["stationary"] > 0.0
    assert prediction["identity_scores"]["1"]["walking"] > 0.0


def test_default_static_walking_boundary_uses_calibrated_limits() -> None:
    parameters = HeuristicParameters()

    assert parameters.stationary_max_speed_cm_s == 10.0
    assert parameters.walking_min_speed_cm_s == 10.0
    assert parameters.walking_max_speed_cm_s == 85.0
    assert parameters.stationary_min_duration_s == 1.0
    assert parameters.walking_min_duration_s == 1.0
    assert parameters.huddle_min_duration_s == 1.0
    assert parameters.huddle_max_gap_s == 5.0
    assert parameters.huddle_max_mean_speed_cm_s == 10.0
    assert parameters.together_max_distance_cm == 8.0
    assert parameters.together_max_individual_speed_cm_s == 16.0
    assert parameters.approach_terminal_distance_cm == 17.0
    assert parameters.isolation_distance_cm == 8.0
    assert parameters.pair_min_duration_s == 0.1
    assert parameters.group_min_duration_s == 0.3
    assert parameters.isolation_min_duration_s == 10.0
    assert parameters.clustering_min_duration_s == 5.0
    assert parameters.clustering_initial_max_distance_cm == 24.0
    assert parameters.clustering_min_moving_member_fraction == 0.5
    assert parameters.huddle_width_multiplier == 2.0
    assert parameters.huddle_min_member_support_fraction == 0.25
    assert parameters.clustering_min_mean_speed_cm_s == 3.0


def test_huddle_onset_motion_gate_tolerates_missing_group_detections() -> None:
    grouped = np.asarray([True, True, *([False] * 8)])
    valid = np.zeros((10, 3), dtype=bool)
    valid[:2] = True
    speed = np.zeros((10, 3), dtype=float)

    assert _group_resting_at_onset(
        grouped,
        speed,
        valid,
        (0, 1, 2),
        fps=10.0,
        minimum_seconds=1.0,
        speed_limit_cm_s=10.0,
    )


def test_group_calibration_parameters_load_from_project_config_shape() -> None:
    parameters = config_seed_parameters(
        {
            "extended_behavior": {
                "social": {
                    "approach_terminal_distance_cm": 17.0,
                    "together_max_individual_speed_cm_s": 16.0,
                    "attack_vs_approach_multiplier": 1.5,
                },
                "group": {
                    "huddle_width_multiplier": 2.0,
                    "huddle_min_member_support_fraction": 0.25,
                    "isolation_distance_cm": 8.0,
                    "social_clustering": {
                        "initial_max_neighbor_distance_cm": 24.0,
                        "min_mean_speed_cm_s": 3.0,
                    },
                },
            }
        }
    )

    assert parameters.huddle_width_multiplier == 2.0
    assert parameters.huddle_min_member_support_fraction == 0.25
    assert parameters.isolation_distance_cm == 8.0
    assert parameters.clustering_initial_max_distance_cm == 24.0
    assert parameters.clustering_min_mean_speed_cm_s == 3.0
    assert parameters.approach_terminal_distance_cm == 17.0
    assert parameters.together_max_individual_speed_cm_s == 16.0
    assert parameters.attack_vs_approach_multiplier == 1.5


def test_strict_top1_guard_rejects_improving_huddle_at_social_cluster_cost():
    baseline = {
        "per_behavior": {
            "huddle": {"support": 10, "strict_top1_correct": 4},
            "social_clustering": {"support": 10, "strict_top1_correct": 5},
            "isolation": {"support": 10, "strict_top1_correct": 3},
            "together": {"support": 10, "strict_top1_correct": 2},
        }
    }
    candidate = {
        "per_behavior": {
            "huddle": {"strict_top1_correct": 6},
            "social_clustering": {"strict_top1_correct": 4},
            "isolation": {"strict_top1_correct": 3},
            "together": {"strict_top1_correct": 2},
        }
    }

    assert strict_top1_regressions(candidate, baseline) == ["social_clustering"]
    assert (
        strict_top1_regressions(candidate, baseline, excluded_behaviors={"social_clustering"}) == []
    )


def test_running_requires_directional_fast_motion_for_half_a_second():
    frames = 10
    velocity = np.zeros((frames, 1, 2), dtype=np.float32)
    velocity[:, 0, 0] = 100.0
    features = ExportedFeatures(
        fps=10.0,
        track_ids=np.asarray([5]),
        valid=np.ones((frames, 1), dtype=bool),
        centers_cm=np.zeros((frames, 1, 2), dtype=np.float32),
        speed_cm_s=np.full((frames, 1), 100.0, dtype=np.float32),
        velocity_cm_s=velocity,
        pair_i=np.empty(0, dtype=np.int64),
        pair_j=np.empty(0, dtype=np.int64),
        pair_distance_cm=np.empty((frames, 0), dtype=np.float32),
        pair_closing_speed_cm_s=np.empty((frames, 0), dtype=np.float32),
        pair_direction_similarity=np.empty((frames, 0), dtype=np.float32),
        pair_nose_head_cm=np.empty((frames, 0), dtype=np.float32),
        pair_nose_tail_cm=np.empty((frames, 0), dtype=np.float32),
        nearest_distance_cm=np.full((frames, 1), np.inf, dtype=np.float32),
        mean_nearest_distance_cm=np.full(frames, np.inf, dtype=np.float32),
        cm_per_pixel=0.8,
        track_selection_truncated=False,
        selected_track_count=1,
    )
    prediction = predict_features(features, HeuristicParameters())
    assert prediction["identity_scores"]["5"]["running"] == 1.0
    assert prediction["identity_scores"]["5"]["walking"] == 0.0


def test_loader_uses_eight_centimeter_body_length_and_estimates_mouse_width(tmp_path):
    detections = []
    for track_id, x in ((1, 0), (2, 100)):
        points = [
            [x + dx, dy, 0.99]
            for dx, dy in ((0, 0), (1, 1), (1, -1), (3, 0), (5, 2), (5, -2), (10, 0))
        ]
        detections.append({"track_id": track_id, "keypoints": points, "box": [x, -1, x + 10, 1]})
    path = tmp_path / "tracks.json"
    path.write_text(json.dumps([{"frame": 0, "detections": detections}]), encoding="utf-8")

    features = load_exported_features(path, fps=30.0)

    assert features.cm_per_pixel == 0.8
    assert np.isclose(features.mouse_width_cm, 3.2)


def test_loader_uses_partial_keypoints_and_never_uses_box_geometry(tmp_path):
    points = [
        [10.0, 10.0, 0.99],
        [20.0, 20.0, 0.0],
        [30.0, 30.0, 0.99],
        [40.0, 40.0, 0.99],
        [50.0, 50.0, 0.99],
        [60.0, 60.0, 0.0],
        [70.0, 70.0, 0.0],
    ]
    path = tmp_path / "partial_tracks.json"
    path.write_text(
        json.dumps(
            [
                {
                    "frame": 0,
                    "detections": [
                        {"track_id": 7, "keypoints": points, "box": [1000, 1000, 1100, 1100]}
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    features = load_exported_features(path, fps=30.0)

    assert features.valid[0, 0]
    assert np.isfinite(features.centers_cm[0, 0]).all()
    assert features.centers_cm[0, 0, 0] < 100.0


def test_huddling_gap_default_and_profile_override_are_five_seconds() -> None:
    assert HeuristicParameters().huddle_max_gap_s == 5.0
    parameters = config_seed_parameters(
        {"extended_behavior": {"group": {"huddle_fill_gap_seconds": 5.0}}}
    )

    assert parameters.huddle_min_duration_s == 1.0
    assert parameters.huddle_max_gap_s == 5.0


def test_isolation_uses_nearest_neighbor_distance_for_ten_seconds() -> None:
    frames = 10
    pair_distances = np.tile(np.asarray([[10.0, 100.0, 100.0]], dtype=np.float32), (frames, 1))
    features = ExportedFeatures(
        fps=1.0,
        track_ids=np.asarray([1, 2, 3]),
        valid=np.ones((frames, 3), dtype=bool),
        centers_cm=np.zeros((frames, 3, 2), dtype=np.float32),
        speed_cm_s=np.zeros((frames, 3), dtype=np.float32),
        velocity_cm_s=np.zeros((frames, 3, 2), dtype=np.float32),
        pair_i=np.asarray([0, 0, 1]),
        pair_j=np.asarray([1, 2, 2]),
        pair_distance_cm=pair_distances,
        pair_closing_speed_cm_s=np.zeros_like(pair_distances),
        pair_direction_similarity=np.ones_like(pair_distances),
        pair_nose_head_cm=np.full_like(pair_distances, 100.0),
        pair_nose_tail_cm=np.full_like(pair_distances, 100.0),
        nearest_distance_cm=np.tile(
            np.asarray([[10.0, 10.0, 100.0]], dtype=np.float32), (frames, 1)
        ),
        mean_nearest_distance_cm=np.full(frames, 40.0, dtype=np.float32),
        cm_per_pixel=0.1,
        track_selection_truncated=False,
        selected_track_count=3,
    )

    prediction = predict_features(
        features,
        HeuristicParameters(isolation_distance_cm=20.0),
    )

    # Mouse 1 is close to one companion even though its distance to the other
    # is large; isolation is determined by the nearest companion, not a mean.
    assert prediction["identity_scores"]["1"]["isolation"] == 0.0
    assert prediction["identity_scores"]["3"]["isolation"] > 0.0


def test_together_distance_cutoff_is_independent_of_huddling() -> None:
    frames = 4
    pair_distance = np.full((frames, 1), 5.0, dtype=np.float32)
    features = ExportedFeatures(
        fps=30.0,
        track_ids=np.asarray([1, 2]),
        valid=np.ones((frames, 2), dtype=bool),
        centers_cm=np.zeros((frames, 2, 2), dtype=np.float32),
        speed_cm_s=np.zeros((frames, 2), dtype=np.float32),
        velocity_cm_s=np.zeros((frames, 2, 2), dtype=np.float32),
        pair_i=np.asarray([0]),
        pair_j=np.asarray([1]),
        pair_distance_cm=pair_distance,
        pair_closing_speed_cm_s=np.zeros_like(pair_distance),
        pair_direction_similarity=np.ones_like(pair_distance),
        pair_nose_head_cm=np.full_like(pair_distance, 20.0),
        pair_nose_tail_cm=np.full_like(pair_distance, 20.0),
        nearest_distance_cm=np.full((frames, 2), 5.0, dtype=np.float32),
        mean_nearest_distance_cm=np.full(frames, 5.0, dtype=np.float32),
        cm_per_pixel=0.1,
        track_selection_truncated=False,
        selected_track_count=2,
        mouse_width_cm=2.4,
    )
    parameters = HeuristicParameters(
        together_max_distance_cm=8.0,
        huddle_distance_cm=4.0,
        huddle_width_multiplier=1.0,
        together_min_duration_s=0.05,
    )

    prediction = predict_features(features, parameters)

    assert prediction.get("pair_scores", {}).get("1,2", {}).get("together", 0.0) > 0.0


def test_v7_together_speed_limit_allows_pair_motion_below_16_cm_s() -> None:
    frames = 31
    pair_distance = np.full((frames, 1), 6.0, dtype=np.float32)
    features = ExportedFeatures(
        fps=30.0,
        track_ids=np.asarray([1, 2]),
        valid=np.ones((frames, 2), dtype=bool),
        centers_cm=np.zeros((frames, 2, 2), dtype=np.float32),
        speed_cm_s=np.full((frames, 2), 14.0, dtype=np.float32),
        velocity_cm_s=np.zeros((frames, 2, 2), dtype=np.float32),
        pair_i=np.asarray([0]),
        pair_j=np.asarray([1]),
        pair_distance_cm=pair_distance,
        pair_closing_speed_cm_s=np.zeros_like(pair_distance),
        pair_direction_similarity=np.zeros_like(pair_distance),
        pair_nose_head_cm=np.full_like(pair_distance, 20.0),
        pair_nose_tail_cm=np.full_like(pair_distance, 20.0),
        nearest_distance_cm=np.full((frames, 2), 6.0, dtype=np.float32),
        mean_nearest_distance_cm=np.full(frames, 6.0, dtype=np.float32),
        cm_per_pixel=0.1,
        track_selection_truncated=False,
        selected_track_count=2,
    )
    parameters = HeuristicParameters(
        stationary_max_speed_cm_s=10.0,
        together_min_duration_s=1.0,
    )

    prediction = predict_features(features, parameters)

    assert prediction["pair_scores"].get("1,2", {}).get("together", 0.0) > 0.0
    assert prediction["identity_scores"]["1"]["stationary"] == 0.0
    assert prediction["identity_scores"]["2"]["stationary"] == 0.0


def test_attack_approach_priority_preserves_contact_top1() -> None:
    contest = {"approach": 0.20, "attack": 0.14, "nose_tail_contact": 0.0}
    _prioritize_attack_over_approach(contest, 1.5)
    assert contest["attack"] > contest["approach"]

    contact = {"approach": 0.20, "attack": 0.14, "nose_tail_contact": 0.25}
    _prioritize_attack_over_approach(contact, 1.5)
    assert contact["attack"] == 0.14
    assert contact["nose_tail_contact"] > contact["approach"]


def test_calibration_rank_enforces_every_behavior_target_before_f1() -> None:
    feasible = {
        "target_id_accuracy": 0.96,
        "macro_target_id_accuracy": 0.955,
        "strict_target_id_accuracy": 0.40,
        "macro_f1": 0.35,
        "per_behavior": {
            "walking": {"support": 10, "accuracy": 0.95},
            "stationary": {"support": 10, "accuracy": 0.96},
        },
    }
    high_f1_but_failed_class = {
        "target_id_accuracy": 0.97,
        "macro_target_id_accuracy": 0.95,
        "strict_target_id_accuracy": 0.80,
        "macro_f1": 0.85,
        "per_behavior": {
            "walking": {"support": 10, "accuracy": 1.0},
            "stationary": {"support": 10, "accuracy": 0.90},
        },
    }

    assert minimum_behavior_accuracy(feasible) == 0.95
    assert training_target_reached(feasible, 0.95)
    assert not training_target_reached(high_f1_but_failed_class, 0.95)
    assert calibration_rank(feasible, 0.95) > calibration_rank(high_f1_but_failed_class, 0.95)


def test_duration_vectorization_preserves_longest_run_per_column() -> None:
    mask = np.asarray(
        [
            [True, False, True, False],
            [True, True, False, False],
            [False, True, True, False],
            [True, True, True, False],
            [True, False, True, False],
        ],
        dtype=bool,
    )

    np.testing.assert_allclose(
        _durations_by_column(mask, fps=2.0),
        np.asarray([1.0, 1.5, 1.5, 0.0]),
    )
