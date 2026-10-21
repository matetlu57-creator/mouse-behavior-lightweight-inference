from __future__ import annotations

import json
from dataclasses import replace
import gc
import weakref

import numpy as np
import pytest

from mouse_behavior.evaluation.exported_dataset import (
    CANONICAL_BEHAVIORS,
    INDIVIDUAL_BEHAVIORS,
    ExportedFeatures,
    HeuristicParameters,
    SampleRecord,
    _group_resting_at_onset,
    _group_episode_scores,
    _group_formation_geometry_support,
    _COLLECTIVE_WINDOW_CACHE,
    _cached_collective_windows,
    collective_track_quality_failures,
    collective_track_quality_metrics,
    _isolation_cumulative_durations,
    _windowed_body_speed,
    _durations_by_column,
    _directional_walker_in_group,
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
from mouse_behavior.evaluation.handcrafted_features import behavior_feature_vector


def test_collective_window_cache_rejects_recycled_feature_identity() -> None:
    class CacheOwner:
        pass

    current_owner = CacheOwner()
    previous_owner = CacheOwner()
    cache_key = (id(current_owner), (1, 2, 3))
    _COLLECTIVE_WINDOW_CACHE[cache_key] = (
        weakref.ref(previous_owner),
        ("windows from a prior clip",),
    )
    try:
        windows = _cached_collective_windows(
            cache_key,
            current_owner,
            lambda: ("windows from this clip",),
        )
        assert windows == ("windows from this clip",)
        assert _COLLECTIVE_WINDOW_CACHE[cache_key][0]() is current_owner
    finally:
        _COLLECTIVE_WINDOW_CACHE.pop(cache_key, None)


def test_collective_window_cache_releases_windows_when_owner_is_collected() -> None:
    class CacheOwner:
        pass

    owner = CacheOwner()
    cache_key = (id(owner), "release-on-owner-collection")
    _cached_collective_windows(cache_key, owner, lambda: ("clip windows",))
    owner_ref = weakref.ref(owner)

    del owner
    gc.collect()

    assert owner_ref() is None
    assert cache_key not in _COLLECTIVE_WINDOW_CACHE


def test_collective_track_quality_rejects_short_fragments_and_exact_duplicates() -> None:
    features = _group_formation_features(moving_members=3)
    valid = features.valid.copy()
    valid[:10, 2] = False
    confidence = np.full_like(features.speed_cm_s, 0.8)
    coverage = np.full_like(features.speed_cm_s, 0.7)
    overlap = np.full_like(features.pair_distance_cm, np.nan)
    overlap[:, 0] = 0.99
    distance = features.pair_distance_cm.copy()
    distance[:, 0] = 0.2
    features = replace(
        features,
        valid=valid,
        detection_confidence=confidence,
        keypoint_coverage=coverage,
        pair_box_iou=overlap,
        pair_distance_cm=distance,
        track_presence_fraction=valid.mean(axis=0),
    )

    quality = collective_track_quality_metrics(features)
    by_id = {int(row["track_id"]): row for row in quality}
    parameters = HeuristicParameters(collective_track_min_presence_ratio=0.20)

    assert "track_presence_ratio" in collective_track_quality_failures(by_id[3], parameters)
    assert by_id[2]["spatial_duplicate_score"] >= parameters.collective_track_max_spatial_duplicate_score
    assert "spatial_duplicate_score" in collective_track_quality_failures(by_id[2], parameters)
    assert collective_track_quality_failures(by_id[1], parameters) == []


def test_collective_quality_gate_does_not_change_other_behavior_scores() -> None:
    features = _group_formation_features(moving_members=3)
    base = HeuristicParameters(collective_feature_rules_enabled=1.0)
    filtered = replace(base, collective_track_min_presence_ratio=1.01)

    before = predict_features(features, base)
    after = predict_features(features, filtered)
    group_only = {
        "huddle",
        "social_clustering",
        "dispersal",
        "group_locomotion",
    }

    assert any(row["eligible_for_collective_graph"] for row in before["collective_track_quality"])
    assert not any(row["eligible_for_collective_graph"] for row in after["collective_track_quality"])
    assert all(
        before["scores"][behavior] == after["scores"][behavior]
        for behavior in before["scores"]
        if behavior not in group_only
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
            "Snout-head_contact": "nose_head_contact",
            "Snout-rear_contact": "nose_tail_contact",
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

    # [STEP6 2026-10-05] Together 删除速度门：12 cm/s（高于个体 Static 上限 10 cm/s）
    # 的移动对只要不处于 interaction 帧，仍然计入 Together。
    moving_parameters = HeuristicParameters(
        stationary_max_speed_cm_s=10.0,
        together_min_duration_s=0.05,
        huddle_distance_cm=8.0,
        huddle_width_multiplier=3.0,
    )
    moving_pair = predict_features(
        replace(
            features,
            speed_cm_s=np.full_like(features.speed_cm_s, 12.0),
            # 抬高鼻-头距离以避免接触，从而不触发 Attack 互动帧。
            pair_nose_head_cm=np.full_like(pair_distance, 9.0),
        ),
        parameters=moving_parameters,
    )
    assert moving_pair["pair_scores"]["1,2"]["together"] == pytest.approx(4 / 30)
    # 个体层仍把 12 cm/s 判为非 Static：Together 不复用个体静止边界。
    assert moving_pair["identity_scores"]["1"].get("stationary", 0.0) == 0.0

    # interaction 帧硬排除：同样的 12 cm/s，一旦接触触发 Attack，Together 归零。
    interacting_pair = predict_features(
        replace(features, speed_cm_s=np.full_like(features.speed_cm_s, 12.0)),
        parameters=moving_parameters,
    )
    assert interacting_pair["pair_scores"]["1,2"]["attack"] > 0.0
    assert interacting_pair["pair_scores"]["1,2"].get("together", 0.0) == 0.0
    rows = [
        (_record("a", "Static", "train.mp4", mouse_ids=(1,)), prediction),
        (_record("b", "Together", "train.mp4"), prediction),
    ]
    metrics = evaluate_predictions(rows)
    assert metrics["n_samples"] == 2
    assert metrics["per_behavior"]["stationary"]["support"] == 1
    assert 0.0 <= metrics["target_id_accuracy"] <= 1.0


def test_collective_feature_toggle_only_replaces_huddle_and_clustering_scores() -> None:
    # [STEP6 2026-10-05] Huddling ≥ 5 s：合成片段拉长到 6.1 s 才能过新时长门。
    frames = 61
    centers = np.repeat(
        np.asarray([[[0.0, 0.0], [2.0, 0.0], [1.0, 1.5]]], dtype=np.float32),
        frames,
        axis=0,
    )
    points = np.full((frames, 3, 7, 2), np.nan, dtype=np.float32)
    offsets = np.asarray(
        [[3, 0], [2, -0.4], [2, 0.4], [1, 0], [-1, -0.5], [-1, 0.5], [-3, 0]],
        dtype=np.float32,
    )
    points[:] = centers[:, :, None, :] + offsets[None, None]
    boxes = np.empty((frames, 3, 4), dtype=np.float32)
    boxes[:, :, 0] = centers[:, :, 0] - 2
    boxes[:, :, 1] = centers[:, :, 1] - 1.5
    boxes[:, :, 2] = centers[:, :, 0] + 2
    boxes[:, :, 3] = centers[:, :, 1] + 1.5
    pair_i = np.asarray([0, 0, 1])
    pair_j = np.asarray([1, 2, 2])
    distance = np.linalg.norm(centers[:, pair_i] - centers[:, pair_j], axis=2)
    features = ExportedFeatures(
        fps=10.0,
        track_ids=np.asarray([1, 2, 3]),
        valid=np.ones((frames, 3), dtype=bool),
        centers_cm=centers,
        speed_cm_s=np.zeros((frames, 3), dtype=np.float32),
        velocity_cm_s=np.zeros((frames, 3, 2), dtype=np.float32),
        pair_i=pair_i,
        pair_j=pair_j,
        pair_distance_cm=distance,
        pair_closing_speed_cm_s=np.zeros_like(distance),
        pair_direction_similarity=np.zeros_like(distance),
        pair_nose_head_cm=np.full_like(distance, 10.0),
        pair_nose_tail_cm=np.full_like(distance, 10.0),
        nearest_distance_cm=np.asarray(
            [[2.0, 1.8, 1.8]] * frames, dtype=np.float32
        ),
        mean_nearest_distance_cm=np.full(frames, 1.8, dtype=np.float32),
        cm_per_pixel=1.0,
        track_selection_truncated=False,
        selected_track_count=3,
        mouse_width_cm=1.0,
        collective_centers_cm=centers,
        collective_valid=np.ones((frames, 3), dtype=bool),
        collective_velocity_cm_s=np.zeros((frames, 3, 2), dtype=np.float32),
        collective_speed_cm_s=np.zeros((frames, 3), dtype=np.float32),
        collective_keypoints_cm=points,
        collective_boxes_cm=boxes,
        body_length_ref_cm=np.full(3, 6.0, dtype=np.float32),
        body_width_ref_cm=np.ones(3, dtype=np.float32),
    )
    baseline = predict_features(features, HeuristicParameters())
    updated = predict_features(
        features,
        HeuristicParameters(collective_feature_rules_enabled=1.0),
    )
    protected = {
        # [STEP4 2026-10-04] 个体层合并：Walking+Running -> Locomotion
        "locomotion",
        "stationary",
        "isolation",
        "dispersal",
        "group_locomotion",
        "approach",
        "together",
        "chase",
        "attack",
        "nose_head_contact",
        "nose_tail_contact",
    }
    assert {name: baseline["scores"][name] for name in protected} == {
        name: updated["scores"][name] for name in protected
    }
    assert updated["scores"]["huddle"] > 0
    assert updated["scores"]["social_clustering"] == 0
    assert updated["collective_feature_debug"]


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
            "1": {"stationary": 0.0, "locomotion": 2.0, "isolation": 0.0},
            "2": {"stationary": 3.0, "locomotion": 0.0, "isolation": 0.0},
        },
        "pair_scores": {},
    }

    metrics = evaluate_predictions(
        [(_record("static-target-1", "Static", "val.mp4", mouse_ids=(1,)), prediction)]
    )

    assert metrics["target_id_accuracy"] == 0.0
    assert metrics["per_behavior"]["stationary"]["correct"] == 0
    assert metrics["confusion"] == {"stationary->locomotion": 1}


def test_target_id_accuracy_is_behavior_recognition_not_strict_top1() -> None:
    prediction = {
        "identity_scores": {
            "1": {"stationary": 1.0, "locomotion": 2.0, "isolation": 0.0},
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


def test_pair_dual_label_credit_removed_and_compatible_mirrors_strict() -> None:
    # [SUSPENDED-STEP3 2026-10-04] dual-label credit removed: Together now allows contact without interaction。
    prediction = {
        "track_ids": [1, 2, 3],
        "identity_scores": {},
        "pair_scores": {
            "1,2": {"together": 1.0, "nose_head_contact": 2.0},
            "1,3": {"together": 3.0, "nose_tail_contact": 2.0},
        },
        "pair_compatible_labels": {"1,2": ["together", "nose_head_contact"]},
    }
    rows = [
        (_record("together", "Together", "val.mp4", mouse_ids=(1, 2)), prediction),
        (
            _record("head", "Snout-head_contact", "val.mp4", mouse_ids=(1, 2)),
            prediction,
        ),
        (
            _record("rear", "Snout-rear_contact", "val.mp4", mouse_ids=(1, 3)),
            prediction,
        ),
    ]
    metrics = evaluate_predictions(rows)

    assert metrics["strict_target_id_accuracy"] == 1 / 3
    assert metrics["compatible_top1_accuracy"] == 1 / 3
    assert metrics["per_behavior"]["together"]["compatible_top1_correct"] == 0
    assert metrics["per_behavior"]["nose_head_contact"]["compatible_top1_correct"] == 1
    assert metrics["per_behavior"]["nose_tail_contact"]["compatible_top1_correct"] == 0


def test_together_label_accepts_qualified_contact_without_together_bout() -> None:
    prediction = {
        "track_ids": [1, 2],
        "identity_scores": {},
        "pair_scores": {"1,2": {"nose_head_contact": 1.2}},
        "pair_compatible_labels": {},
    }
    result = evaluate_predictions(
        [(_record("quiet-pair-with-contact", "Together", "val.mp4"), prediction)]
    )
    assert result["strict_target_id_accuracy"] == 0.0
    assert result["compatible_top1_accuracy"] == 0.0


def test_contact_labels_never_equated_and_no_dual_credit() -> None:
    # [SUSPENDED-STEP3 2026-10-04] dual-label credit removed: Together now allows contact without interaction
    prediction = {
        "track_ids": [1, 2],
        "identity_scores": {},
        "pair_scores": {
            "1,2": {
                "together": 1.0,
                "nose_head_contact": 2.0,
                "nose_tail_contact": 3.0,
            }
        },
        "pair_compatible_labels": {"1,2": ["together", "nose_head_contact"]},
    }
    metrics = evaluate_predictions(
        [(_record("head", "Snout-head_contact", "val.mp4"), prediction)]
    )
    assert metrics["strict_target_id_accuracy"] == 0.0
    assert metrics["compatible_top1_accuracy"] == 0.0


def test_pair_compatible_labels_removed_from_predictions() -> None:
    # [SUSPENDED-STEP3 2026-10-04] dual-label credit removed: Together now allows contact without interaction
    frames = 20
    pair_distance = np.full((frames, 1), 6.0, dtype=np.float32)
    head_distance = np.full((frames, 1), 20.0, dtype=np.float32)
    head_distance[:6] = 1.0
    features = ExportedFeatures(
        fps=10.0,
        track_ids=np.asarray([1, 2]),
        valid=np.ones((frames, 2), dtype=bool),
        centers_cm=np.zeros((frames, 2, 2), dtype=np.float32),
        speed_cm_s=np.zeros((frames, 2), dtype=np.float32),
        velocity_cm_s=np.zeros((frames, 2, 2), dtype=np.float32),
        pair_i=np.asarray([0]),
        pair_j=np.asarray([1]),
        pair_distance_cm=pair_distance,
        pair_closing_speed_cm_s=np.zeros_like(pair_distance),
        pair_direction_similarity=np.zeros_like(pair_distance),
        pair_nose_head_cm=head_distance,
        pair_nose_tail_cm=np.full_like(pair_distance, 20.0),
        nearest_distance_cm=np.full((frames, 2), 6.0, dtype=np.float32),
        mean_nearest_distance_cm=np.full(frames, 6.0, dtype=np.float32),
        cm_per_pixel=0.1,
        track_selection_truncated=False,
        selected_track_count=2,
    )
    parameters = HeuristicParameters()
    overlapping = predict_features(features, parameters)
    disjoint = predict_features(
        replace(
            features,
            speed_cm_s=np.r_[np.zeros((10, 2)), np.full((10, 2), 30.0)],
            pair_nose_head_cm=np.r_[np.full((10, 1), 20.0), head_distance[:10]],
        ),
        parameters,
    )

    assert "pair_compatible_labels" not in overlapping
    assert disjoint["pair_scores"]["1,2"]["nose_head_contact"] > 0
    assert "pair_compatible_labels" not in disjoint
    left_head = np.full((frames, 1), 20.0, dtype=np.float32)
    right_head = left_head.copy()
    left_head[:3] = 1.0
    right_head[3:6] = 1.0
    alternating = predict_features(
        replace(
            features,
            pair_nose_head_cm=np.minimum(left_head, right_head),
            pair_nose_head_left_to_right_cm=left_head,
            pair_nose_head_right_to_left_cm=right_head,
        ),
        parameters,
    )
    # Each direction independently sustains 0.3 s, above the current 0.2 s
    # contact minimum. This earns contact evidence without compatibility credit.
    assert alternating["pair_scores"]["1,2"]["nose_head_contact"] == pytest.approx(0.3)
    assert "pair_compatible_labels" not in alternating


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


def test_isolation_is_not_erased_by_separate_group_movement_phase() -> None:
    prediction = {
        "track_ids": [1, 2, 3],
        "identity_scores": {"1": {"isolation": 10.0}, "2": {}, "3": {}},
        "group_scores": {"1,2,3": {"group_locomotion": 3.0}},
        "pair_scores": {},
    }
    metrics = evaluate_predictions(
        [(_record("isolation-after-travel", "Isolation", "val.mp4", mouse_ids=(1,)), prediction)]
    )
    assert metrics["strict_target_id_accuracy"] == 1.0


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


def test_all_unsuspended_behaviors_have_distinct_auditable_feature_vectors() -> None:
    # Walking/Running share Locomotion. Grooming/Rearing remain known taxonomy
    # labels, but their independent review engine is excluded from this
    # default-inference handcrafted vector contract.
    features = _group_formation_features(moving_members=2)
    categories = {
        (1,): ("locomotion", "stationary", "isolation"),
        (1, 2): (
            "approach", "together", "chase", "attack",
            "nose_head_contact", "nose_tail_contact",
        ),
        (1, 2, 3): ("huddle", "social_clustering", "dispersal", "group_locomotion"),
    }
    vectors = {
        behavior: behavior_feature_vector(features, behavior, ids)
        for ids, names in categories.items()
        for behavior in names
    }

    assert len(vectors) == 13
    assert set(vectors) == set(CANONICAL_BEHAVIORS) - {"grooming", "rearing"}
    assert vectors["locomotion"]["straightness"] is not None
    for merged_key in ("speed_p25_cm_s", "speed_p90_cm_s", "active_speed_median_cm_s",
                       "stop_fraction", "net_displacement_cm", "straightness"):
        assert merged_key in vectors["locomotion"]
    assert vectors["together"]["head_contact_s"] == 0.0
    assert vectors["social_clustering"]["pair_distance_end_change_cm"] < 0.0
    assert vectors["dispersal"]["pair_distance_end_change_cm"] < 0.0
    assert "box_iou_max" in vectors["attack"]
    assert "contact_nose_acceleration_p90_cm_s2" in vectors["attack"]


def _measured_group_features(centers: np.ndarray, fps: float = 10.0) -> ExportedFeatures:
    """Derive all motion/distance channels from the same observed trajectory."""
    frames, mice, _ = centers.shape
    pair_i, pair_j = np.triu_indices(mice, 1)
    velocity = np.gradient(centers, axis=0) * fps
    speed = np.linalg.norm(velocity, axis=2)
    distances = np.linalg.norm(centers[:, pair_i] - centers[:, pair_j], axis=2)
    unit = velocity / np.maximum(speed[:, :, None], 1e-9)
    nearest = np.full((frames, mice), np.inf)
    for edge, (left, right) in enumerate(zip(pair_i, pair_j)):
        nearest[:, left] = np.minimum(nearest[:, left], distances[:, edge])
        nearest[:, right] = np.minimum(nearest[:, right], distances[:, edge])
    return ExportedFeatures(
        fps=fps, track_ids=np.arange(1, mice + 1), valid=np.ones((frames, mice), bool),
        centers_cm=centers, speed_cm_s=speed, velocity_cm_s=velocity,
        pair_i=pair_i, pair_j=pair_j, pair_distance_cm=distances,
        pair_closing_speed_cm_s=-np.gradient(distances, axis=0) * fps,
        pair_direction_similarity=np.sum(unit[:, pair_i] * unit[:, pair_j], axis=2),
        pair_nose_head_cm=np.full_like(distances, np.inf),
        pair_nose_tail_cm=np.full_like(distances, np.inf),
        nearest_distance_cm=nearest, mean_nearest_distance_cm=nearest.mean(axis=1),
        cm_per_pixel=.1, track_selection_truncated=False, selected_track_count=mice,
        reference_body_cm=7.918,
    )


@pytest.mark.parametrize("frames, expected", [(29, False), (30, False), (31, True)])
def test_group_locomotion_requires_aligned_moving_triple_for_more_than_three_seconds(
    frames: int, expected: bool,
) -> None:
    centers = np.zeros((frames, 3, 2))
    centers[:, :, 0] = np.arange(frames)[:, None] * .6
    centers[:, :, 1] = [0., 2., 4.]
    parameters = config_seed_parameters({})
    aligned = predict_features(_measured_group_features(centers), parameters)
    score = aligned["group_scores"].get("1,2,3", {}).get("group_locomotion", 0.)
    assert (score > 0.) is expected
    if expected:
        assert score == pytest.approx(frames / 10.)
    # Reverse one actual trajectory: a cached direction-similarity flag cannot
    # manufacture coherent common displacement for this group.
    centers[:, 2, 0] *= -1.
    opposed = predict_features(_measured_group_features(centers), parameters)
    assert not any(scores.get("group_locomotion", 0) > 0
                   for scores in opposed["group_scores"].values())


@pytest.mark.parametrize("frames, expected", [(50, False), (51, False), (52, True)])
def test_dispersal_requires_initial_group_and_sustained_separation(
    frames: int, expected: bool,
) -> None:
    origins = np.array([[-1., 0.], [1., 0.], [0., 1.]])
    velocity = np.array([[-2., 0.], [2., 0.], [0., 2.]])
    centers = origins[None] + np.arange(frames)[:, None, None] / 10. * velocity[None]
    parameters = config_seed_parameters({})
    dispersed = predict_features(_measured_group_features(centers), parameters)
    score = dispersed["group_scores"].get("1,2,3", {}).get("dispersal", 0.)
    assert (score > 0.) is expected
    if expected:
        assert score > 5.0
    unchanged_centers = np.broadcast_to(origins, centers.shape).copy()
    unchanged = predict_features(_measured_group_features(unchanged_centers), parameters)
    assert not any(scores.get("dispersal", 0) > 0
                   for scores in unchanged["group_scores"].values())


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


def test_directional_group_movement_requires_a_walker_approaching_group_location() -> None:
    frames = 20
    centers = np.zeros((frames, 3, 2), dtype=np.float32)
    centers[:, 0, 0] = np.linspace(8.0, 0.0, frames)
    velocity = np.zeros_like(centers)
    velocity[1:, 0, 0] = np.diff(centers[:, 0, 0]) * 10.0
    features = ExportedFeatures(
        fps=10.0,
        track_ids=np.asarray([1, 2, 3]),
        valid=np.ones((frames, 3), dtype=bool),
        centers_cm=centers,
        speed_cm_s=np.linalg.norm(velocity, axis=2),
        velocity_cm_s=velocity,
        pair_i=np.asarray([0, 0, 1]),
        pair_j=np.asarray([1, 2, 2]),
        pair_distance_cm=np.full((frames, 3), 5.0, dtype=np.float32),
        pair_closing_speed_cm_s=np.zeros((frames, 3), dtype=np.float32),
        pair_direction_similarity=np.ones((frames, 3), dtype=np.float32),
        pair_nose_head_cm=np.full((frames, 3), 100.0, dtype=np.float32),
        pair_nose_tail_cm=np.full((frames, 3), 100.0, dtype=np.float32),
        nearest_distance_cm=np.full((frames, 3), 5.0, dtype=np.float32),
        mean_nearest_distance_cm=np.full(frames, 5.0, dtype=np.float32),
        cm_per_pixel=0.1,
        track_selection_truncated=False,
        selected_track_count=3,
        motion_valid=np.ones((frames, 3), dtype=bool),
    )
    parameters = HeuristicParameters(
        # [STEP4 2026-10-04] 个体层合并：Walking+Running -> Locomotion
        # The directional gate now reads the merged mask, so the static ceiling
        # replaces the deleted walking speed band (this mouse moves ~4.2 cm/s).
        stationary_max_speed_cm_s=2.0,
        clustering_directional_walk_min_seconds=0.3,
        clustering_directional_min_distance_drop_cm=2.0,
    )

    assert _directional_walker_in_group(features, (1, 2, 3), parameters)
    assert not _directional_walker_in_group(
        features, (1, 2, 3), parameters, start_frame=0, end_frame=3
    )


def test_formation_geometry_uses_body_translation_spread_and_box_overlap() -> None:
    base = _group_formation_features(moving_members=0)
    centers = np.zeros_like(base.centers_cm)
    centers[:, 0, 0] = np.linspace(12.0, 0.0, len(centers))
    centers[:, 1, 0] = 3.0
    centers[:, 2, 1] = 3.0
    moving = replace(
        base, centers_cm=centers,
        pair_box_iou=np.zeros_like(base.pair_distance_cm),
    )
    resting = replace(
        moving,
        centers_cm=np.repeat(centers[-1:], len(centers), axis=0),
        pair_box_iou=np.ones_like(base.pair_distance_cm),
    )

    assert _group_formation_geometry_support(
        moving, (1, 2, 3), start_frame=0, end_frame=11
    ) > 0.65
    assert _group_formation_geometry_support(
        resting, (1, 2, 3), start_frame=0, end_frame=11
    ) < 0.2


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
    centers[:, 1, 0] = np.arange(frames) * (29.425 / (frames - 1))
    features = replace(
        features,
        centers_cm=centers,
        speed_cm_s=np.full_like(features.speed_cm_s, 4.0125),
        pair_distance_cm=np.full_like(features.pair_distance_cm, 12.0375),
        nearest_distance_cm=np.full_like(features.nearest_distance_cm, 12.0375),
    )
    parameters = HeuristicParameters(
        clustering_motion_max_distance_cm=20.0625,
        clustering_motion_min_displacement_cm=8.025,
    )

    assert "social_clustering" not in predict_features(features, parameters)["group_scores"].get(
        "1,2,3", {}
    )
    loose = replace(
        features,
        pair_distance_cm=np.full_like(features.pair_distance_cm, 13.375),
        nearest_distance_cm=np.full_like(features.nearest_distance_cm, 13.375),
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


def test_group_windowed_motion_keeps_missing_id_unfilled_and_huddle_has_no_speed_gate() -> None:
    # [STEP6 2026-10-05] Huddling ≥ 5 s：合成片段拉长到 6 s 才能过新时长门。
    frames = 60
    valid = np.ones((frames, 3), dtype=bool)
    pair_distance = np.full((frames, 3), 2.0, dtype=np.float32)
    centers = np.zeros((frames, 3, 2), dtype=np.float32)
    centers[:, :, 0] = np.asarray([0.0, 0.2] * 30)[:, None]
    features = ExportedFeatures(
        fps=10.0,
        track_ids=np.asarray([1, 2, 3]),
        valid=valid,
        centers_cm=centers,
        speed_cm_s=np.full((frames, 3), 12.0, dtype=np.float32),
        velocity_cm_s=np.zeros((frames, 3, 2), dtype=np.float32),
        pair_i=np.asarray([0, 0, 1]),
        pair_j=np.asarray([1, 2, 2]),
        pair_distance_cm=pair_distance,
        pair_closing_speed_cm_s=np.zeros_like(pair_distance),
        pair_direction_similarity=np.zeros_like(pair_distance),
        pair_nose_head_cm=np.full_like(pair_distance, 20.0),
        pair_nose_tail_cm=np.full_like(pair_distance, 20.0),
        nearest_distance_cm=np.full((frames, 3), 2.0, dtype=np.float32),
        mean_nearest_distance_cm=np.full(frames, 2.0, dtype=np.float32),
        cm_per_pixel=0.1,
        track_selection_truncated=False,
        selected_track_count=3,
    )
    parameters = HeuristicParameters(
        huddle_distance_cm=8.0,
        huddle_width_multiplier=3.0,
        clustering_motion_max_distance_cm=0.0,
    )
    # [STEP6 2026-10-05] Huddling 删除全部速度门：姿态抖动带来的 12 cm/s 伪速度
    # 不再需要 group_motion_window_s 来压制，huddle 候选直接成立。
    assert predict_features(features, parameters)["group_scores"]["1,2,3"]["huddle"] > 0.0
    windowed = replace(parameters, group_motion_window_s=0.5)
    assert predict_features(features, windowed)["group_scores"]["1,2,3"]["huddle"] > 0.0

    interrupted = replace(features, valid=np.where(np.arange(frames)[:, None] == 15, False, valid))
    robust_speed = _windowed_body_speed(interrupted, 0.5)
    assert np.isnan(robust_speed[15]).all()
    assert np.isnan(robust_speed[14]).all()


def test_group_core_keeps_reacquired_ids_and_excludes_brief_visitor() -> None:
    pair_i = np.asarray([0, 0, 0, 1, 1, 2])
    pair_j = np.asarray([1, 2, 3, 2, 3, 3])
    close = np.zeros((20, 6), dtype=bool)
    close[:, [0, 1, 3]] = True
    close[8:12, [1, 3]] = False  # ID 3 is temporarily missing.
    close[5:7, [2, 4, 5]] = True  # ID 4 briefly joins the same component.
    events: list[dict] = []

    scores = _group_episode_scores(
        close,
        pair_i,
        pair_j,
        (1, 2, 3, 4),
        fps=10.0,
        minimum_seconds=1.0,
        max_gap_seconds=0.5,
        min_member_support_fraction=0.0,
        emit_core_candidates=True,
        event_behavior="huddle",
        event_output=events,
    )

    assert scores["1,2,3"] > 0.0
    assert "1,2,3,4" not in scores
    assert any(
        event["member_ids"] == (1, 2, 3)
        and event["start_frame"] == 0
        and event["end_frame"] == 19
        for event in events
    )


def test_group_core_does_not_treat_new_visual_id_as_reacquired_animal() -> None:
    pair_i = np.asarray([0, 0, 0, 1, 1, 2])
    pair_j = np.asarray([1, 2, 3, 2, 3, 3])
    close = np.zeros((20, 6), dtype=bool)
    close[:5, [0, 1, 3]] = True
    close[5:, [0, 2, 4]] = True

    scores = _group_episode_scores(
        close, pair_i, pair_j, (1, 2, 3, 4), fps=10.0,
        minimum_seconds=1.0, max_gap_seconds=0.5,
        emit_core_candidates=True,
    )

    assert "1,2,4" in scores
    assert "1,2,3" not in scores
    assert "1,2,3,4" not in scores


def test_group_core_can_confirm_alternating_visible_members_without_full_frame() -> None:
    pair_i = np.asarray([0, 0, 0, 0, 1, 1, 1, 2, 2, 3])
    pair_j = np.asarray([1, 2, 3, 4, 2, 3, 4, 3, 4, 4])
    close = np.zeros((20, len(pair_i)), dtype=bool)
    close[:, [0, 1, 4]] = True
    close[:10, [2, 5, 7]] = True
    close[10:, [3, 6, 8]] = True

    scores = _group_episode_scores(
        close, pair_i, pair_j, (1, 2, 3, 4, 5), fps=10.0,
        minimum_seconds=1.0, emit_core_candidates=True,
    )

    assert scores["1,2,3,4,5"] > scores["1,2,3"]


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


def test_locomotion_and_stationary_use_measured_body_translation() -> None:
    # Cached per-frame speed cannot turn a fixed body center into locomotion.

    frames = 40
    valid = np.ones((frames, 1), dtype=bool)
    speed = np.full((frames, 1), 3.0, dtype=np.float32)
    features = ExportedFeatures(
        fps=10.0,
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

    slow = HeuristicParameters(
        stationary_max_speed_cm_s=4.0,
        stationary_min_duration_s=0.05,
        locomotion_min_duration_s=0.05,
    )
    prediction = predict_features(features, slow)

    assert prediction["identity_scores"]["1"]["stationary"] > 0.0
    assert prediction["identity_scores"]["1"]["locomotion"] == 0.0

    faster = predict_features(
        replace(features, speed_cm_s=np.full((frames, 1), 16.0, dtype=np.float32)),
        slow,
    )
    assert faster["identity_scores"] == prediction["identity_scores"]
    translated_centers = features.centers_cm.copy()
    translated_centers[:, 0, 0] = np.arange(frames) * 2.0
    translated = predict_features(replace(features, centers_cm=translated_centers), slow)
    assert translated["identity_scores"]["1"]["stationary"] == 0.0
    assert translated["identity_scores"]["1"]["locomotion"] > 0.0

    # The disabled Step 5-A detectors keep their score slots at exactly zero.
    assert prediction["identity_scores"]["1"]["grooming"] == 0.0
    assert prediction["identity_scores"]["1"]["rearing"] == 0.0
    assert set(INDIVIDUAL_BEHAVIORS) <= set(prediction["identity_scores"]["1"])


def test_default_static_locomotion_boundary_uses_calibrated_limits() -> None:
    # [STEP4 2026-10-04] 个体层合并：Walking+Running -> Locomotion
    parameters = HeuristicParameters()

    # The historical group-speed mirror is separate from current individual
    # body-center motion and its bounded Static geometry.
    assert parameters.stationary_max_speed_cm_s == 0.66875
    assert parameters.individual_motion_speed_threshold_cm_s == 4.0
    assert parameters.stationary_max_center_radius_cm == 4.0
    assert parameters.stationary_transient_max_duration_s == 1.0
    assert parameters.locomotion_min_duration_s == 0.5
    assert parameters.locomotion_stop_gap_s == 0.5
    # Retained as the Chasing high-speed floor and the locomotion score normaliser.
    assert parameters.running_min_speed_cm_s == 113.6875
    assert parameters.stationary_min_duration_s == 1.0
    for deleted in (
        "walking_min_speed_cm_s", "walking_max_speed_cm_s", "walking_max_stop_gap_s",
        "walking_min_duration_s", "running_min_directionality", "running_min_duration_s",
    ):
        assert not hasattr(parameters, deleted), f"{deleted} must stay deleted"
    # [STEP6 2026-10-05] 社交/群体新默认值。
    assert parameters.together_max_distance_cm == 7.0
    assert parameters.together_min_distance_cm == 4.0125
    assert parameters.together_min_duration_s == 1.0
    assert parameters.approach_min_start_distance_cm == 6.6875
    assert parameters.approach_min_end_distance_cm == 2.675
    assert parameters.approach_min_actor_speed_cm_s == 4.0125
    assert parameters.approach_min_closing_speed_cm_s == 2.675
    # [STEP6 2026-10-05] approach_terminal_distance_cm 已改为 Approach 接触截断阈值，
    # 与 approach_min_end_distance_cm 同义（旧值 17.0 不再是默认）。
    assert parameters.approach_terminal_distance_cm == 2.675
    assert parameters.contact_distance_cm == 2.675
    assert parameters.chase_max_distance_cm == 16.05
    assert parameters.chase_min_speed_cm_s == 26.75
    assert parameters.chase_min_duration_s == 1.0
    assert parameters.chase_min_direction_similarity == 0.707
    assert parameters.chase_fill_gap_s == 0.5
    assert parameters.attack_min_speed_cm_s == 10.7
    assert parameters.attack_mild_nose_accel_cm_s2 == 66.875
    # These are compatibility scalar defaults. Current group geometry is
    # selected through config_seed_parameters and GroupHeuristicConfig.
    assert parameters.huddle_distance_cm == 10.0
    assert parameters.huddle_nose_distance_cm == 4.0125
    assert parameters.huddle_min_duration_s == 5.0
    assert parameters.huddle_max_gap_s == 5.0
    assert parameters.huddle_width_multiplier == 2.0
    assert parameters.huddle_min_member_support_fraction == 0.25
    # Social clustering 无最小时长门；Dispersal ≥ 5 s；Isolation ≥ 20 cm / ≥ 10 s。
    assert parameters.clustering_min_duration_s == 0.0
    assert parameters.clustering_initial_max_distance_cm == 32.1
    assert parameters.clustering_min_moving_member_fraction == 0.5
    assert parameters.clustering_min_mean_speed_cm_s == 4.0125
    assert parameters.dispersal_min_duration_s == 5.0
    assert parameters.group_locomotion_max_distance_cm == 30.0
    assert parameters.group_locomotion_min_duration_s == 3.0
    assert parameters.isolation_distance_cm == 20.0
    assert parameters.isolation_min_duration_s == 10.0
    assert parameters.isolation_max_gap_s == 0.2
    assert parameters.pair_min_duration_s == 0.1
    assert parameters.group_min_duration_s == 0.3
    # Compatibility speed fields do not control the current independent
    # SocialHeuristicConfig/GroupHeuristicConfig limits.
    assert parameters.together_max_individual_speed_cm_s == 16.0
    assert parameters.together_max_combined_speed_cm_s == 28.0
    assert parameters.huddle_max_mean_speed_cm_s == 10.0


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
    # Confirmed profiles fix the current physical policy after reading legacy
    # configuration fields; historical scalar overrides do not change it.
    assert parameters.isolation_distance_cm == 20.0
    assert parameters.clustering_initial_max_distance_cm == 24.0
    assert parameters.clustering_min_mean_speed_cm_s == 3.0
    assert parameters.approach_terminal_distance_cm == pytest.approx(4.456491470336914)
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


def test_fast_motion_is_locomotion_without_a_directionality_gate():
    # [STEP4 2026-10-04] 个体层合并：Walking+Running -> Locomotion
    # Regression: the deleted Running directionality gate must not come back. A
    # straight run and a broad back-and-forth path can both be locomotion.
    # Motion must be present in centers, not merely in a cached speed array.

    frames = 20
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
    centers = features.centers_cm.copy()
    centers[:, 0, 0] = np.arange(frames) * 10.0
    features = replace(features, centers_cm=centers)
    prediction = predict_features(features, HeuristicParameters())
    assert prediction["identity_scores"]["5"]["locomotion"] == 2.0
    assert prediction["identity_scores"]["5"]["stationary"] == 0.0

    alternation = np.where(np.arange(frames) % 2 == 0, 1.0, -1.0).astype(np.float32)
    turning_centers = centers.copy()
    turning_centers[:, 0, 0] = alternation * 10.0
    oscillating = replace(features, centers_cm=turning_centers,
                          velocity_cm_s=velocity * alternation[:, None, None])
    turned = predict_features(oscillating, HeuristicParameters())
    assert turned["identity_scores"]["5"]["locomotion"] == 2.0

    for deleted in ("running", "walking"):
        assert deleted not in prediction["identity_scores"]["5"]


def test_loader_supports_explicit_body_length_scale_and_unified_per_video_default(tmp_path):
    detections = []
    for track_id, x in ((1, 0), (2, 100)):
        points = [
            [x + dx, dy, 0.99]
            for dx, dy in ((0, 0), (1, 1), (1, -1), (3, 0), (5, 2), (5, -2), (10, 0))
        ]
        detections.append({"track_id": track_id, "keypoints": points, "box": [x, -1, x + 10, 1]})
    path = tmp_path / "tracks.json"
    path.write_text(json.dumps([{"frame": 0, "detections": detections}]), encoding="utf-8")

    legacy = load_exported_features(
        path,
        fps=30.0,
        scale_config={"mode": "body_length", "assumed_mouse_body_length_cm": 8.0},
    )
    assert legacy.cm_per_pixel == 0.8
    assert np.isclose(legacy.mouse_width_cm, 3.2)

    unified = load_exported_features(path, fps=30.0)
    # Nose-to-tail base is 10 px, so the current 7.918 cm ruler gives .7918 cm/px.
    assert unified.cm_per_pixel == pytest.approx(7.918 / 10.0)
    assert unified.reference_body_cm == pytest.approx(7.918)


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


def test_loader_preserves_box_and_confidence_features_without_replacing_pose_center(tmp_path):
    def detection(track_id: int, x: float, confidence: float):
        points = [[x + dx, 0.0, 0.99] for dx in (0, 1, 2, 3, 4, 5, 10)]
        return {
            "track_id": track_id,
            "keypoints": points,
            "box": [x, -2, x + 12, 2],
            "detection_confidence": confidence,
        }

    path = tmp_path / "tracks.json"
    path.write_text(
        json.dumps(
            [
                {"frame": 0, "detections": [detection(1, 0, 0.8), detection(2, 8, 0.6)]},
                {"frame": 1, "detections": [detection(1, 1, 0.7)]},
                {"frame": 2, "detections": [detection(1, 2, 0.9), detection(2, 10, 0.5)]},
            ]
        ),
        encoding="utf-8",
    )
    features = load_exported_features(path, fps=10.0)

    assert np.isclose(features.track_presence_fraction[1], 2 / 3)
    assert np.isclose(features.detection_confidence[0, 0], 0.8)
    assert np.isnan(features.detection_confidence[1, 1])
    assert np.isclose(features.pair_box_iou[0, 0], 4 / 20)
    assert np.isnan(features.pair_box_iou[1, 0])
    assert features.motion_valid[2, 0]
    assert not features.motion_valid[2, 1]
    assert features.centers_cm[0, 0, 0] < 10.0
    prediction = predict_features(
        features, HeuristicParameters(stationary_min_duration_s=0.1)
    )
    assert prediction["identity_scores"]["2"]["stationary"] == 0.0


def test_head_contact_uses_nearest_nose_or_ear_not_head_centroid(tmp_path):
    def points(nose_x: float):
        return [
            [nose_x, 0, 0.99],
            [nose_x + 20, 10, 0.99],
            [nose_x + 20, -10, 0.99],
            [nose_x + 3, 0, 0.99],
            [nose_x + 5, 2, 0.99],
            [nose_x + 5, -2, 0.99],
            [nose_x + 10, 0, 0.99],
        ]

    path = tmp_path / "head_region_tracks.json"
    path.write_text(
        json.dumps(
            [{"frame": 0, "detections": [
                {"track_id": 1, "keypoints": points(0)},
                {"track_id": 2, "keypoints": points(2)},
            ]}]
        ),
        encoding="utf-8",
    )
    features = load_exported_features(
        path,
        fps=10.0,
        scale_config={"mode": "body_length", "assumed_mouse_body_length_cm": 8.0},
    )

    assert np.isclose(features.pair_nose_head_left_to_right_cm[0, 0], 1.6)
    assert np.isclose(features.pair_nose_head_cm[0, 0], 1.6)


def test_huddling_gap_default_and_profile_override_are_five_seconds() -> None:
    assert HeuristicParameters().huddle_max_gap_s == 5.0
    parameters = config_seed_parameters(
        {"extended_behavior": {"group": {"huddle_fill_gap_seconds": 5.0}}}
    )

    assert parameters.huddle_min_duration_s == 5.0
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


def test_isolation_accumulates_observed_frames_across_only_brief_id_loss() -> None:
    evidence = np.zeros((53, 1), dtype=bool)
    evidence[:25] = True
    evidence[28:] = True
    unknown = np.zeros_like(evidence)
    unknown[25:28] = True

    assert _isolation_cumulative_durations(evidence, unknown, 10.0, 0.3)[0] == 5.0
    assert _isolation_cumulative_durations(evidence, unknown, 10.0, 0.2)[0] == 2.5

    # A measured nearby mouse ends the episode even if the interruption is short.
    unknown[26] = False
    assert _isolation_cumulative_durations(evidence, unknown, 10.0, 0.3)[0] == 2.5


def test_isolation_does_not_count_missing_frames_toward_ten_seconds() -> None:
    evidence = np.ones((100, 1), dtype=bool)
    evidence[24:26] = False
    unknown = ~evidence

    assert _isolation_cumulative_durations(evidence, unknown, 10.0, 0.2)[0] == 9.8


def test_isolation_prediction_recovers_same_id_after_short_dropout() -> None:
    frames = 102
    valid = np.ones((frames, 2), dtype=bool)
    valid[24:26, 0] = False
    pair_distance = np.full((frames, 1), 30.0, dtype=np.float32)
    features = ExportedFeatures(
        fps=10.0,
        track_ids=np.asarray([1, 2]),
        valid=valid,
        centers_cm=np.zeros((frames, 2, 2), dtype=np.float32),
        speed_cm_s=np.zeros((frames, 2), dtype=np.float32),
        velocity_cm_s=np.zeros((frames, 2, 2), dtype=np.float32),
        pair_i=np.asarray([0]),
        pair_j=np.asarray([1]),
        pair_distance_cm=pair_distance,
        pair_closing_speed_cm_s=np.zeros_like(pair_distance),
        pair_direction_similarity=np.zeros_like(pair_distance),
        pair_nose_head_cm=np.full_like(pair_distance, 30.0),
        pair_nose_tail_cm=np.full_like(pair_distance, 30.0),
        nearest_distance_cm=np.full((frames, 2), 30.0, dtype=np.float32),
        mean_nearest_distance_cm=np.full(frames, 30.0, dtype=np.float32),
        cm_per_pixel=0.1,
        track_selection_truncated=False,
        selected_track_count=2,
    )

    prediction = predict_features(features, HeuristicParameters(isolation_distance_cm=20.0))
    assert prediction["identity_scores"]["1"]["isolation"] == 10.0

    observed_near = valid.copy()
    observed_near[25, 0] = True
    nearest = features.nearest_distance_cm.copy()
    nearest[25, 0] = 5.0
    prediction = predict_features(
        replace(features, valid=observed_near, nearest_distance_cm=nearest),
        HeuristicParameters(isolation_distance_cm=20.0),
    )
    assert prediction["identity_scores"]["1"]["isolation"] == 0.0


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


def test_together_clip_summary_allows_brief_motion_when_pair_stays_close() -> None:
    frames = 30
    speed = np.zeros((frames, 2), dtype=np.float32)
    speed[[4, 9, 14, 19, 24], :] = 18.0
    pair_distance = np.full((frames, 1), 6.0, dtype=np.float32)
    features = ExportedFeatures(
        fps=10.0,
        track_ids=np.asarray([1, 2]),
        valid=np.ones((frames, 2), dtype=bool),
        centers_cm=np.zeros((frames, 2, 2), dtype=np.float32),
        speed_cm_s=speed,
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
        motion_valid=np.ones((frames, 2), dtype=bool),
    )
    parameters = HeuristicParameters(
        together_max_individual_speed_cm_s=16.0,
        together_max_combined_speed_cm_s=28.0,
        together_initial_max_distance_cm=12.0,
        together_median_max_distance_cm=10.0,
        together_mean_max_speed_cm_s=12.0,
        together_max_distance_std_cm=3.0,
        together_min_observed_seconds=1.0,
        together_min_duration_s=1.0,
    )

    prediction = predict_features(features, parameters)

    assert prediction["pair_scores"]["1,2"]["together"] >= 1.5


def test_attack_contact_keypoint_acceleration_boosts_attack_over_contact() -> None:
    frames = 20
    pair_distance = np.full((frames, 1), 5.0, dtype=np.float32)
    features = ExportedFeatures(
        fps=10.0,
        track_ids=np.asarray([1, 2]),
        valid=np.ones((frames, 2), dtype=bool),
        centers_cm=np.zeros((frames, 2, 2), dtype=np.float32),
        speed_cm_s=np.full((frames, 2), 30.0, dtype=np.float32),
        velocity_cm_s=np.zeros((frames, 2, 2), dtype=np.float32),
        pair_i=np.asarray([0]),
        pair_j=np.asarray([1]),
        pair_distance_cm=pair_distance,
        pair_closing_speed_cm_s=np.zeros_like(pair_distance),
        pair_direction_similarity=np.zeros_like(pair_distance),
        pair_nose_head_cm=np.full_like(pair_distance, 1.0),
        pair_nose_tail_cm=np.full_like(pair_distance, 20.0),
        nearest_distance_cm=np.full((frames, 2), 5.0, dtype=np.float32),
        mean_nearest_distance_cm=np.full(frames, 5.0, dtype=np.float32),
        cm_per_pixel=0.1,
        track_selection_truncated=False,
        selected_track_count=2,
        motion_valid=np.ones((frames, 2), dtype=bool),
        nose_speed_cm_s=np.full((frames, 2), 30.0, dtype=np.float32),
        nose_acceleration_cm_s2=np.full((frames, 2), 800.0, dtype=np.float32),
    )
    parameters = HeuristicParameters(
        attack_contact_nose_min_speed_cm_s=25.0,
        attack_contact_nose_min_acceleration_cm_s2=700.0,
        attack_contact_body_min_speed_cm_s=25.0,
        attack_contact_score_multiplier=1.5,
    )

    score = predict_features(features, parameters)["pair_scores"]["1,2"]

    assert score["attack"] > score["nose_head_contact"]


def test_brief_contact_and_generic_speed_do_not_establish_measured_attack() -> None:
    frames = 20
    pair_distance = np.full((frames, 1), 5.0, dtype=np.float32)
    nose_head = np.full_like(pair_distance, 20.0)
    nose_head[5:8] = 1.0  # 0.3 s exceeds the current standalone contact minimum.
    speed = np.zeros((frames, 2), dtype=np.float32)
    speed[8:10] = 30.0  # The body accelerates after contact, not during it.
    features = ExportedFeatures(
        fps=10.0,
        track_ids=np.asarray([1, 2]),
        valid=np.ones((frames, 2), dtype=bool),
        centers_cm=np.zeros((frames, 2, 2), dtype=np.float32),
        speed_cm_s=speed,
        velocity_cm_s=np.zeros((frames, 2, 2), dtype=np.float32),
        pair_i=np.asarray([0]),
        pair_j=np.asarray([1]),
        pair_distance_cm=pair_distance,
        pair_closing_speed_cm_s=np.zeros_like(pair_distance),
        pair_direction_similarity=np.zeros_like(pair_distance),
        pair_nose_head_cm=nose_head,
        pair_nose_tail_cm=np.full_like(pair_distance, 20.0),
        nearest_distance_cm=np.full((frames, 2), 5.0, dtype=np.float32),
        mean_nearest_distance_cm=np.full(frames, 5.0, dtype=np.float32),
        cm_per_pixel=0.1,
        track_selection_truncated=False,
        selected_track_count=2,
        motion_valid=np.ones((frames, 2), dtype=bool),
        nose_speed_cm_s=np.full((frames, 2), 30.0, dtype=np.float32),
        nose_acceleration_cm_s2=np.full((frames, 2), 800.0, dtype=np.float32),
    )
    features.pair_nose_head_left_to_right_cm = nose_head.copy()
    features.pair_nose_head_right_to_left_cm = np.full_like(nose_head, 20.)
    features.pair_nose_tail_left_to_right_cm = np.full_like(nose_head, 20.)
    features.pair_nose_tail_right_to_left_cm = np.full_like(nose_head, 20.)
    parameters = replace(config_seed_parameters({}),
        attack_contact_nose_min_speed_cm_s=25.0,
        attack_contact_nose_min_acceleration_cm_s2=700.0,
        attack_contact_body_min_speed_cm_s=25.0,
        attack_contact_score_multiplier=1.5,
        attack_contact_post_window_s=0.2,
    )

    pair_score = predict_features(features, parameters)["pair_scores"]["1,2"]

    assert pair_score["nose_head_contact"] == pytest.approx(0.3)
    # Generic scalar speed/acceleration do not prove partner-directed action.
    assert pair_score["attack"] == 0.0


def test_attack_mild_gate_uses_two_mouse_max_and_boost_uses_contacting_nose() -> None:
    frames = 20
    pair_distance = np.full((frames, 1), 5.0, dtype=np.float32)
    nose_head = np.full_like(pair_distance, 20.0)
    nose_head[5:8] = 1.0
    directed_left = np.full_like(pair_distance, 20.0)
    directed_left[5:8] = 1.0
    speed = np.zeros((frames, 2), dtype=np.float32)
    speed[8:10, 1] = 30.0
    features = ExportedFeatures(
        fps=10.0,
        track_ids=np.asarray([1, 2]),
        valid=np.ones((frames, 2), dtype=bool),
        centers_cm=np.zeros((frames, 2, 2), dtype=np.float32),
        speed_cm_s=speed,
        velocity_cm_s=np.zeros((frames, 2, 2), dtype=np.float32),
        pair_i=np.asarray([0]),
        pair_j=np.asarray([1]),
        pair_distance_cm=pair_distance,
        pair_closing_speed_cm_s=np.zeros_like(pair_distance),
        pair_direction_similarity=np.zeros_like(pair_distance),
        pair_nose_head_cm=nose_head,
        pair_nose_tail_cm=np.full_like(pair_distance, 20.0),
        nearest_distance_cm=np.full((frames, 2), 5.0, dtype=np.float32),
        mean_nearest_distance_cm=np.full(frames, 5.0, dtype=np.float32),
        cm_per_pixel=0.1,
        track_selection_truncated=False,
        selected_track_count=2,
        motion_valid=np.ones((frames, 2), dtype=bool),
        nose_speed_cm_s=np.column_stack(
            (np.zeros(frames, dtype=np.float32), np.full(frames, 30.0, dtype=np.float32))
        ),
        nose_acceleration_cm_s2=np.column_stack(
            (np.zeros(frames, dtype=np.float32), np.full(frames, 2000.0, dtype=np.float32))
        ),
        pair_nose_head_left_to_right_cm=directed_left,
        pair_nose_head_right_to_left_cm=np.full_like(pair_distance, 20.0),
        pair_nose_tail_left_to_right_cm=np.full_like(pair_distance, 20.0),
        pair_nose_tail_right_to_left_cm=np.full_like(pair_distance, 20.0),
    )
    parameters = HeuristicParameters(
        attack_contact_nose_min_speed_cm_s=25.0,
        attack_contact_nose_min_acceleration_cm_s2=700.0,
        attack_contact_body_min_speed_cm_s=25.0,
        attack_contact_score_multiplier=1.5,
    )

    pair_score = predict_features(features, parameters)["pair_scores"]["1,2"]

    # [STEP6 2026-10-05] Attack = contact & (双鼠最大速度≥ 8 cm/s OR 双鼠鼻加速度最大
    # ≥ 50 cm/s2，轻度攻击=攻击下限）。mouse2 的 2000 cm/s2 命中 OR 门，
    # 接触 3 帧 @10 fps = 0.3 s。
    assert HeuristicParameters().attack_mild_nose_accel_cm_s2 == 66.875
    assert pair_score["attack"] == pytest.approx(0.3)
    # 保留原意图：接触鼻加分路径只读“接触鼠”(mouse1) 的定向鼻证据。mouse1
    # 鼻速/鼻加速度为 0，所以 1.5x 加分不得触发；若借用 mouse2 的 30 cm/s /
    # 2000 cm/s2 会得到 0.45，下方正向对照证明该加分只对接触鼠生效。

    # 正向对照：把快速鼻运动放到接触鼠 mouse1 上，1.5x 加分立即生效。
    contacting_speed = speed.copy()
    contacting_speed[5:10, 0] = 30.0
    boosted = predict_features(
        replace(
            features,
            speed_cm_s=contacting_speed,
            nose_speed_cm_s=np.full((frames, 2), 30.0, dtype=np.float32),
            nose_acceleration_cm_s2=np.full((frames, 2), 800.0, dtype=np.float32),
        ),
        parameters,
    )["pair_scores"]["1,2"]
    assert boosted["attack"] == pytest.approx(0.3 * 1.5)
    assert boosted["attack"] > pair_score["attack"]

    # OR 门下界回归：双鼠鼻加速度 49 cm/s2 且无速度证据时不得判为 Attack。
    below_gate = predict_features(
        replace(
            features,
            speed_cm_s=np.zeros((frames, 2), dtype=np.float32),
            nose_acceleration_cm_s2=np.full((frames, 2), 49.0, dtype=np.float32),
        ),
        HeuristicParameters(),
    )["pair_scores"]["1,2"]
    assert below_gate.get("attack", 0.0) == 0.0


@pytest.mark.parametrize("speed_cm_s, expected", [(2.9, True), (3.0, False), (14.0, False)])
def test_together_uses_its_own_strict_three_cm_per_second_speed_limit(
    speed_cm_s: float, expected: bool,
) -> None:
    # Current Together requires both participants to move strictly below 3 cm/s;
    # the historical individual Static scalar cannot loosen that social gate.
    frames = 31
    pair_distance = np.full((frames, 1), 3.0, dtype=np.float32)
    features = ExportedFeatures(
        fps=30.0,
        track_ids=np.asarray([1, 2]),
        valid=np.ones((frames, 2), dtype=bool),
        centers_cm=np.zeros((frames, 2, 2), dtype=np.float32),
        speed_cm_s=np.full((frames, 2), speed_cm_s, dtype=np.float32),
        velocity_cm_s=np.zeros((frames, 2, 2), dtype=np.float32),
        pair_i=np.asarray([0]),
        pair_j=np.asarray([1]),
        pair_distance_cm=pair_distance,
        pair_closing_speed_cm_s=np.zeros_like(pair_distance),
        pair_direction_similarity=np.ones_like(pair_distance),
        pair_nose_head_cm=np.full_like(pair_distance, 20.0),
        pair_nose_tail_cm=np.full_like(pair_distance, 20.0),
        nearest_distance_cm=np.full((frames, 2), 3.0, dtype=np.float32),
        mean_nearest_distance_cm=np.full(frames, 3.0, dtype=np.float32),
        cm_per_pixel=0.1,
        track_selection_truncated=False,
        selected_track_count=2,
    )
    parameters = replace(config_seed_parameters({}),
        stationary_max_speed_cm_s=10.0,
        together_min_duration_s=1.0,
    )

    features.centers_cm[:, :, 0] = np.arange(frames)[:, None] * speed_cm_s / 30.0
    features.centers_cm[:, 1, 1] = 3.0
    features.velocity_cm_s[:, :, 0] = speed_cm_s

    prediction = predict_features(features, parameters)

    together = prediction["pair_scores"].get("1,2", {}).get("together", 0.0)
    assert (together > 0.) is expected
    if speed_cm_s == 14.0:
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
            "locomotion": {"support": 10, "accuracy": 0.95},
            "stationary": {"support": 10, "accuracy": 0.96},
        },
    }
    high_f1_but_failed_class = {
        "target_id_accuracy": 0.97,
        "macro_target_id_accuracy": 0.95,
        "strict_target_id_accuracy": 0.80,
        "macro_f1": 0.85,
        "per_behavior": {
            "locomotion": {"support": 10, "accuracy": 1.0},
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
