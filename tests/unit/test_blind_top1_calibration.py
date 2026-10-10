from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import replace

import pytest
import numpy as np

from mouse_behavior.evaluation.blind_top1 import (
    Candidate,
    LabeledClip,
    Top1Prediction,
    candidates_from_evidence,
    evaluate_top1,
    hierarchical_candidate_indices,
    load_locked_predictions,
    load_training_manifest,
    load_unlabeled_manifest,
    select_top1,
    target_layer_candidate_diagnostics,
)
from mouse_behavior.evaluation.exported_dataset import (
    CANONICAL_BEHAVIORS,
    GROUP_BEHAVIORS,
    INDIVIDUAL_BEHAVIORS,
    LABEL_TO_CANONICAL,
    PAIR_BEHAVIORS,
    ExportedFeatures,
    HeuristicParameters,
    _bridge_short_false_gaps,
    _group_episode_scores,
    predict_features,
)
from mouse_behavior.evaluation.train_only_tuning import (
    candidate_oracle_metrics,
    fit_weights,
    session_folds,
)


def _clip(behavior: str, target_ids: tuple[int, ...]) -> LabeledClip:
    return LabeledClip(
        sample_id="opaque-1",
        tracks_path="tracks.json",
        fps=30.0,
        recording_session="session-a",
        behavior=behavior,
        target_ids=target_ids,
    )


def test_top1_is_selected_without_ground_truth_ids() -> None:
    evidence = {
        "identity_scores": {
            "1": {"stationary": 1.0},
            "2": {"stationary": 2.0},
        },
        "pair_scores": {"1,2": {"attack": 0.5}},
        "track_ids": [1, 2],
    }

    candidates = candidates_from_evidence(evidence)
    prediction = select_top1(candidates)

    assert prediction == Top1Prediction("stationary", (2,), 2.0)
    metrics = evaluate_top1([(_clip("stationary", (2,)), prediction)])
    assert metrics["joint_top1_accuracy"] == 1.0
    assert metrics["behavior_top1_accuracy"] == 1.0
    assert metrics["id_top1_accuracy"] == 1.0


def test_exact_pair_id_match_is_unordered_but_behavior_must_match() -> None:
    correct = Top1Prediction("attack", (2, 1), 1.0)
    wrong_behavior = Top1Prediction("approach", (1, 2), 1.0)

    metrics = evaluate_top1(
        [
            (_clip("attack", (1, 2)), correct),
            (_clip("attack", (1, 2)), wrong_behavior),
        ]
    )

    assert metrics["joint_top1_accuracy"] == 0.5
    assert metrics["id_top1_accuracy"] == 1.0
    assert metrics["behavior_top1_accuracy"] == 0.5
    assert metrics["per_behavior"]["attack"]["joint_correct"] == 1


def test_blind_compatible_top1_mirrors_strict_after_dual_removal() -> None:
    # [SUSPENDED-STEP3 2026-10-04] dual-label credit removed: Together now allows contact without interaction
    rows = [
        (
            _clip("together", (1, 2)),
            Top1Prediction("nose_head_contact", (1, 2), 2.0),
        ),
        (
            _clip("nose_tail_contact", (1, 2)),
            Top1Prediction("nose_head_contact", (1, 2), 2.0, ("together",)),
        ),
        (
            _clip("together", (1, 2)),
            Top1Prediction("nose_tail_contact", (1, 3), 2.0, ("together",)),
        ),
    ]
    metrics = evaluate_top1(rows)
    assert metrics["joint_top1_accuracy"] == 0.0
    assert metrics["compatible_exact_top1_accuracy"] == 0.0
    assert metrics["per_behavior"]["together"]["compatible_exact_correct"] == 0


def test_target_behavior_top1_allows_extra_group_members_but_requires_labeled_ids() -> None:
    label = _clip("huddle", (1, 2, 3))
    prediction = Top1Prediction("huddle", (1, 2, 3, 4), 1.0)
    metrics = evaluate_top1([(label, prediction)])

    assert metrics["joint_top1_accuracy"] == 1.0
    assert metrics["id_top1_accuracy"] == 1.0
    assert metrics["exact_id_set_accuracy"] == 0.0

    missing_target = Top1Prediction("huddle", (1, 2, 4), 1.0)
    missed = evaluate_top1([(label, missing_target)])
    assert missed["joint_top1_accuracy"] == 0.0


def test_group_candidate_requires_at_least_three_positive_members() -> None:
    evidence = {
        "identity_scores": {
            "1": {"huddle": 2.0},
            "2": {"huddle": 1.0},
            "3": {"huddle": 0.5},
            "4": {"huddle": 0.0},
        },
        "pair_scores": {},
    }
    candidates = candidates_from_evidence(evidence)

    assert [(item.behavior, item.target_ids) for item in candidates] == [
        ("huddle", (1, 2, 3))
    ]


def test_no_evidence_produces_no_top1_prediction() -> None:
    assert select_top1(candidates_from_evidence({"identity_scores": {}, "pair_scores": {}})) == (
        Top1Prediction(None, (), 0.0)
    )


def test_layer_selector_preserves_cooccurring_group_social_and_individual_candidates():
    candidates = (
        Candidate("huddle", (1, 2, 3), 3.0),
        Candidate("social_clustering", (1, 2, 3), 2.0),
        Candidate("isolation", (1,), 1.0),
        Candidate("approach", (1, 4), 4.0),
        Candidate("stationary", (1,), 0.5),
    )

    eligible = hierarchical_candidate_indices(candidates)

    assert {candidates[index].behavior for index in eligible} == {
        "huddle", "social_clustering", "isolation", "approach", "stationary"
    }


def test_target_layer_candidate_diagnostic_counts_target_intersections():
    label = _clip("huddle", (1, 2, 3))
    candidates = (
        Candidate("locomotion", (1,), 1.0),
        Candidate("approach", (2, 4), 2.0),
        Candidate("huddle", (2, 3, 4), 3.0),
        Candidate("attack", (5, 6), 4.0),
    )

    result = target_layer_candidate_diagnostics([(label, candidates)])
    huddle = result["per_labeled_behavior"]["huddle"]

    assert result["n_samples"] == 1
    assert huddle["individual"]["target_mouse_instances_with_candidate"] == 1
    assert huddle["social"]["target_mouse_instances_with_candidate"] == 1
    assert huddle["group"]["target_mouse_instances_with_candidate"] == 2


def test_training_manifest_rejects_validation_rows(tmp_path) -> None:
    path = tmp_path / "train.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "sample_id",
                "tracks_path",
                "fps",
                "recording_session",
                "canonical_behavior",
                "mouse_ids",
                "split",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": "x",
                "tracks_path": "never-open.json",
                "fps": "30",
                "recording_session": "session-a",
                "canonical_behavior": "stationary",
                "mouse_ids": "1",
                "split": "validation",
            }
        )

    with pytest.raises(ValueError, match="non-training split"):
        load_training_manifest(path)


def test_hierarchical_behavior_schema_contains_all_active_labels_and_layers():
    # [SUSPENDED-STEP3 2026-10-04] Following/Avoiding 暂停（用户决策），保留待恢复：活跃标签 16 -> 14。
    # [STEP4 2026-10-04] 个体层合并：Walking+Running -> Locomotion
    # Step 4 merges the two annotated motion labels into one, so the annotatable
    # label count goes 16 -> 14 (Step 3) -> 13. Grooming/Rearing are individual
    # *slots* in INDIVIDUAL_BEHAVIORS (detectors off, score always 0) and are
    # deliberately not annotation labels, so they stay out of CANONICAL_BEHAVIORS.
    assert len(LABEL_TO_CANONICAL) == len(CANONICAL_BEHAVIORS) == 13
    assert set(CANONICAL_BEHAVIORS) == {
        "locomotion", "stationary", "isolation",
        "approach", "together", "chase", "attack",
        "nose_head_contact", "nose_tail_contact",
        "huddle", "social_clustering", "dispersal", "group_locomotion",
    }
    assert LABEL_TO_CANONICAL["Locomotion"] == "locomotion"
    for legacy in ("Walking", "Running"):
        assert legacy not in LABEL_TO_CANONICAL
    assert set(INDIVIDUAL_BEHAVIORS) == {
        "locomotion", "stationary", "grooming", "rearing",
    }
    assert set(PAIR_BEHAVIORS) == {
        "approach", "together", "chase", "attack",
        "nose_head_contact", "nose_tail_contact",
    }
    assert set(GROUP_BEHAVIORS) == {
        "huddle", "social_clustering", "dispersal", "group_locomotion", "isolation",
    }


def test_isolation_group_candidate_is_a_single_target_mouse():
    candidates = candidates_from_evidence(
        {
            "identity_scores": {"7": {"locomotion": 2.0, "isolation": 4.0}},
            "pair_scores": {},
            "group_scores": {"7": {"isolation": 4.0}},
        }
    )
    assert Candidate("isolation", (7,), 4.0) in candidates
    label = _clip("isolation", (7,))
    assert evaluate_top1([(label, Top1Prediction("isolation", (7,), 1.0))])["joint_top1_accuracy"] == 1.0


def test_blind_manifest_rejects_label_columns_without_opening_tracks(tmp_path) -> None:
    path = tmp_path / "blind.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("sample_id", "tracks_path", "fps", "mouse_ids")
        )
        writer.writeheader()
        writer.writerow(
            {"sample_id": "opaque-1", "tracks_path": "missing.json", "fps": "30", "mouse_ids": "1"}
        )

    with pytest.raises(ValueError, match="exactly"):
        load_unlabeled_manifest(path)


def test_blind_manifest_rejects_hidden_extra_csv_cell(tmp_path) -> None:
    path = tmp_path / "blind.csv"
    path.write_text(
        "sample_id,tracks_path,fps\nopaque-1,never-open.json,30,secret-label\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="extra"):
        load_unlabeled_manifest(path)


def test_session_folds_never_mix_recordings() -> None:
    records = [
        _clip("stationary", (1,)),
        _clip("locomotion", (2,)),
        LabeledClip("opaque-2", "other.json", 30.0, "session-b", "locomotion", (2,)),
    ]

    folds = session_folds(records)

    assert len(folds) == 2
    assert folds[0] == ("session-a", [2], [0, 1])
    assert folds[1] == ("session-b", [0, 1], [2])


def test_ranking_weights_are_fitted_from_training_labels() -> None:
    rows = [
        (
            _clip("attack", (1, 2)),
            (
                Candidate("stationary", (3,), 3.0),
                Candidate("attack", (1, 2), 1.0),
            ),
        ),
        (
            _clip("stationary", (3,)),
            (Candidate("stationary", (3,), 2.0),),
        ),
    ]

    weights, metrics = fit_weights(rows)

    assert weights["attack"] > 3.0
    assert metrics["joint_top1_accuracy"] == 1.0


def test_candidate_oracle_is_an_upper_bound_for_fixed_rules() -> None:
    rows = [
        (_clip("attack", (1, 2)), (Candidate("attack", (1, 2), 0.2),)),
        (_clip("stationary", (3,)), (Candidate("stationary", (4,), 3.0),)),
    ]

    oracle = candidate_oracle_metrics(rows)

    assert oracle["target_candidate_rate"] == 0.5
    assert oracle["per_behavior"]["stationary"]["behavior_candidate"] == 1
    assert oracle["per_behavior"]["stationary"]["target_candidate"] == 0


def test_blind_scorer_refuses_predictions_changed_after_receipt(tmp_path) -> None:
    predictions = tmp_path / "predictions.csv"
    predictions.write_text(
        "sample_id,predicted_behavior,predicted_target_ids,score\n"
        'opaque-1,attack,"1,2",1.0\n',
        encoding="utf-8",
    )
    digest = hashlib.sha256(predictions.read_bytes()).hexdigest()
    (tmp_path / "prediction_receipt.json").write_text(
        json.dumps({"predictions_sha256": digest, "prediction_count": 1}), encoding="utf-8"
    )

    predictions.write_text(predictions.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="changed"):
        load_locked_predictions(tmp_path)


def test_locked_predictions_preserve_behavior_and_exact_ids(tmp_path) -> None:
    predictions = tmp_path / "predictions.csv"
    predictions.write_text(
        'sample_id,predicted_behavior,predicted_target_ids,score\n'
        'opaque-1,attack,"1,2",0.75\n',
        encoding="utf-8",
    )
    (tmp_path / "prediction_receipt.json").write_text(
        json.dumps(
            {
                "predictions_sha256": hashlib.sha256(predictions.read_bytes()).hexdigest(),
                "prediction_count": 1,
            }
        ),
        encoding="utf-8",
    )

    assert load_locked_predictions(tmp_path) == {
        "opaque-1": Top1Prediction("attack", (1, 2), 0.75)
    }


def test_huddle_is_not_suppressed_by_unrelated_fast_mouse() -> None:
    # [STEP6 2026-10-05] Huddling ≥ 5 s 且无任何速度门：合成片段拉长到 6 s，
    # 无关高速鼠与成员高速都不得抑制 Huddling。
    frames = 60
    pair_i, pair_j = np.triu_indices(4, k=1)
    distances = np.full((frames, 6), 100.0, dtype=np.float32)
    distances[:, [0, 1, 3]] = 2.0
    speed = np.zeros((frames, 4), dtype=np.float32)
    speed[:, 3] = 100.0
    features = ExportedFeatures(
        fps=10.0,
        track_ids=np.asarray([1, 2, 3, 4]),
        valid=np.ones((frames, 4), dtype=bool),
        centers_cm=np.zeros((frames, 4, 2), dtype=np.float32),
        speed_cm_s=speed,
        velocity_cm_s=np.zeros((frames, 4, 2), dtype=np.float32),
        pair_i=pair_i,
        pair_j=pair_j,
        pair_distance_cm=distances,
        pair_closing_speed_cm_s=np.zeros_like(distances),
        pair_direction_similarity=np.ones_like(distances),
        pair_nose_head_cm=np.full_like(distances, 100.0),
        pair_nose_tail_cm=np.full_like(distances, 100.0),
        nearest_distance_cm=np.asarray([[2.0, 2.0, 2.0, 100.0]] * frames),
        mean_nearest_distance_cm=np.full(frames, 26.5),
        cm_per_pixel=0.1,
        track_selection_truncated=False,
        selected_track_count=4,
    )

    # [STEP6 2026-10-05] Huddling 文档门为 ≥ 5 s；数据类默认仍是 1.0（5 s 经
    # configs/default.yaml group.huddle_min_duration_seconds 下发），所以这里显式钉住 5 s，
    # 用 6 s 合成片段验证新门下界。
    parameters = HeuristicParameters(group_min_duration_s=0.5, huddle_min_duration_s=5.0)
    evidence = predict_features(features, parameters)
    huddle_score = evidence["group_scores"]["1,2,3"]["huddle"]
    assert huddle_score > 0
    # 参与者集合必须是观测到的精确 visual-ID 三元组，不得扩到无关快鼠 4。
    assert Candidate("huddle", (1, 2, 3), huddle_score) in candidates_from_evidence(evidence)

    # 门不存在回归断言：成员高速运动（20 cm/s）时 Huddling 仍然成立。
    moving_member = speed.copy()
    moving_member[:, 1] = 20.0
    moving_evidence = predict_features(
        replace(features, speed_cm_s=moving_member),
        parameters,
    )
    assert moving_evidence["group_scores"]["1,2,3"]["huddle"] > 0
    assert any(
        candidate.behavior == "huddle" and candidate.target_ids == (1, 2, 3)
        for candidate in candidates_from_evidence(moving_evidence)
    )


def test_group_episodes_do_not_merge_disjoint_clusters() -> None:
    pair_i, pair_j = np.triu_indices(6, k=1)
    mask = np.zeros((5, len(pair_i)), dtype=bool)
    for edge_index, (left, right) in enumerate(zip(pair_i, pair_j)):
        if (left < 3 and right < 3) or (left >= 3 and right >= 3):
            mask[:, edge_index] = True

    groups = _group_episode_scores(
        mask, pair_i, pair_j, [1, 2, 3, 4, 5, 6], fps=5.0, minimum_seconds=0.5
    )

    assert groups == {"1,2,3": 1.0, "4,5,6": 1.0}


def test_group_episode_retains_visual_tracklets_across_member_switch() -> None:
    pair_i, pair_j = np.triu_indices(4, k=1)
    mask = np.zeros((4, len(pair_i)), dtype=bool)
    for frame in range(4):
        members = {0, 1, 2} if frame < 2 else {0, 1, 3}
        for edge_index, (left, right) in enumerate(zip(pair_i, pair_j)):
            mask[frame, edge_index] = left in members and right in members

    groups = _group_episode_scores(
        mask, pair_i, pair_j, [1, 2, 3, 4], fps=4.0, minimum_seconds=0.5
    )

    assert groups == {"1,2,3,4": 1.0}


def test_huddle_episode_bridges_short_detector_gap_when_requested() -> None:
    pair_i, pair_j = np.triu_indices(3, k=1)
    edges = np.ones((6, len(pair_i)), dtype=bool)
    edges[2, :] = False

    groups = _group_episode_scores(
        edges,
        pair_i,
        pair_j,
        [1, 2, 3],
        fps=10.0,
        minimum_seconds=0.5,
        max_gap_seconds=0.2,
    )

    assert groups == {"1,2,3": 0.6}


def test_locomotion_bridge_allows_only_interior_stop_gap() -> None:
    # [STEP4 2026-10-04] 个体层合并：Walking+Running -> Locomotion: the stop-gap bridge now belongs to the merged mask.
    assert _bridge_short_false_gaps(np.asarray([True, False, True]), 1).tolist() == [
        True, True, True
    ]
    assert _bridge_short_false_gaps(np.asarray([False, True, False]), 1).tolist() == [
        False, True, False
    ]


def test_huddle_episode_can_reacquire_after_long_missing_gap() -> None:
    pair_i, pair_j = np.triu_indices(3, k=1)
    edges = np.zeros((52, len(pair_i)), dtype=bool)
    edges[0, :] = True
    edges[-1, :] = True

    groups = _group_episode_scores(
        edges,
        pair_i,
        pair_j,
        [1, 2, 3],
        fps=10.0,
        minimum_seconds=0.75,
        max_gap_seconds=5.0,
    )

    assert groups == {"1,2,3": 5.2}


def test_attack_accepts_short_contact_with_later_nearby_reacquisition() -> None:
    frames = 5
    valid = np.asarray(
        [[True, True], [False, False], [False, False], [False, False], [True, True]]
    )
    distance = np.asarray([[3.0], [np.inf], [np.inf], [np.inf], [10.0]])
    features = ExportedFeatures(
        fps=10.0,
        track_ids=np.asarray([1, 2]),
        valid=valid,
        centers_cm=np.zeros((frames, 2, 2), dtype=np.float32),
        speed_cm_s=np.asarray([[5.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
        velocity_cm_s=np.zeros((frames, 2, 2), dtype=np.float32),
        pair_i=np.asarray([0]), pair_j=np.asarray([1]),
        pair_distance_cm=distance,
        pair_closing_speed_cm_s=np.zeros_like(distance),
        pair_direction_similarity=np.zeros_like(distance),
        pair_nose_head_cm=np.asarray([[1.9], [np.inf], [np.inf], [np.inf], [10.0]]),
        pair_nose_tail_cm=np.full_like(distance, 20.0),
        nearest_distance_cm=np.asarray([[3.0, 3.0], [np.inf, np.inf], [np.inf, np.inf], [np.inf, np.inf], [10.0, 10.0]]),
        mean_nearest_distance_cm=np.asarray([3.0, np.nan, np.nan, np.nan, 10.0]),
        cm_per_pixel=0.1, track_selection_truncated=False, selected_track_count=2,
    )

    prediction = predict_features(
        features,
        HeuristicParameters(
            pair_min_duration_s=0.2,
            attack_min_speed_cm_s=4.0,
            attack_endpoint_distance_cm=12.0,
            attack_reacquisition_max_gap_s=5.0,
        ),
    )

    assert prediction["pair_scores"]["1,2"]["attack"] > 0.0


def test_social_cluster_has_no_minimum_duration_gate() -> None:
    # [STEP6 2026-10-05] Social clustering 无最小时长门（clustering_min_duration_s=0）。
    assert HeuristicParameters().clustering_min_duration_s == 0.0

    pair_i, pair_j = np.triu_indices(3, k=1)
    edges = np.ones((4, len(pair_i)), dtype=bool)
    drop = np.full((4, 3), 3.0)

    # 默认零门：4 帧 @1 fps 的短组也能成立。
    ungated = _group_episode_scores(
        edges, pair_i, pair_j, [1, 2, 3], fps=1.0,
        minimum_seconds=HeuristicParameters().clustering_min_duration_s,
        nearest_drop_cm=drop, min_nearest_drop_cm=2.0,
    )
    assert ungated == {"1,2,3": 4.0}

    # 显式配置 5 s 门时旧行为仍可复现：4 s 太短，5 s 成立。
    too_short = _group_episode_scores(
        edges, pair_i, pair_j, [1, 2, 3], fps=1.0, minimum_seconds=5.0,
        nearest_drop_cm=drop, min_nearest_drop_cm=2.0,
    )
    long_enough = _group_episode_scores(
        np.ones((5, len(pair_i)), dtype=bool), pair_i, pair_j, [1, 2, 3],
        fps=1.0, minimum_seconds=5.0, nearest_drop_cm=np.full((5, 3), 3.0),
        min_nearest_drop_cm=2.0,
    )

    assert too_short == {}
    assert long_enough == {"1,2,3": 5.0}


def test_spatial_group_scoring_does_not_use_absolute_speed_gate() -> None:
    pair_i, pair_j = np.triu_indices(3, k=1)
    edges = np.ones((5, len(pair_i)), dtype=bool)

    slow = _group_episode_scores(
        edges,
        pair_i,
        pair_j,
        [1, 2, 3],
        fps=1.0,
        minimum_seconds=5.0,
    )
    fast = _group_episode_scores(
        edges,
        pair_i,
        pair_j,
        [1, 2, 3],
        fps=1.0,
        minimum_seconds=5.0,
    )

    assert slow == fast == {"1,2,3": 5.0}
