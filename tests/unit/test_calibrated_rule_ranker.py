"""Regression coverage for train-only candidate-score calibration."""

from dataclasses import replace
import json

import numpy as np

from mouse_behavior.evaluation.blind_top1 import Candidate, LabeledClip
from mouse_behavior.evaluation.calibrated_rule_ranker import (
    CandidateTable,
    _physical_features,
    average_ranker_models,
    fit_ranker,
    predict_table,
    build_candidate_table,
    rule_profiles,
)
from mouse_behavior.evaluation.exported_dataset import (
    CANONICAL_BEHAVIORS,
    HeuristicParameters,
    load_exported_features,
)


def test_scoring_can_choose_short_salient_event_over_long_background():
    # A single duration multiplier can never pick ID 2 here. A calibrated
    # trajectory feature must be able to prefer salient motion over duration.
    table = CandidateTable(
        (Candidate("locomotion", (1,), 8.0), Candidate("locomotion", (2,), 2.0)),
        np.asarray([[1.0, 0.8, 0.0], [1.0, 0.2, 1.0]], dtype=np.float32),
        ("bias", "duration_fraction", "motion_salience"),
    )
    label = LabeledClip("opaque", "unused", 30.0, "session", "locomotion", (2,))
    model, _ = fit_ranker([table], [label], steps=100, regularization=0.01)
    prediction = predict_table(table, model)
    assert prediction.behavior == "locomotion"
    assert prediction.target_ids == (2,)
    # File names, session names and opaque sample IDs are not scoring inputs.
    renamed = replace(label, sample_id="attack_ID1", tracks_path="attack/1.json")
    renamed_model, _ = fit_ranker([table], [renamed], steps=100, regularization=0.01)
    assert renamed_model == model


def test_missing_exact_group_candidate_is_not_credited_or_added_by_training():
    table = CandidateTable(
        (Candidate("huddle", (1, 2, 3), 5.0),),
        np.asarray([[1.0, 1.0]], dtype=np.float32), ("bias", "evidence"),
    )
    label = LabeledClip("opaque", "unused", 30.0, "s", "huddle", (1, 2, 3, 4))
    model, metrics = fit_ranker([table], [label], steps=10, regularization=0.01)
    assert metrics["joint_top1_accuracy"] == 0.0
    assert predict_table(table, model).target_ids == (1, 2, 3)


def test_training_gold_candidate_matches_all_labeled_targets_without_exact_group_set():
    table = CandidateTable(
        (Candidate("huddle", (1, 2, 3, 4), 5.0),),
        np.asarray([[1.0, 1.0]], dtype=np.float32), ("bias", "evidence"),
    )
    label = LabeledClip("opaque", "unused", 30.0, "s", "huddle", (1, 2, 3))

    model, metrics = fit_ranker([table], [label], steps=5, regularization=0.01)

    assert metrics["joint_top1_accuracy"] == 1.0
    assert metrics["exact_id_set_accuracy"] == 0.0


def test_empty_candidates_produce_abstention():
    table = CandidateTable((), np.empty((0, 1), dtype=np.float32), ("bias",))
    label = LabeledClip("opaque", "unused", 30.0, "s", "locomotion", (1,))
    model, metrics = fit_ranker([table], [label], steps=2, regularization=0.01)
    assert predict_table(table, model).behavior is None
    assert metrics["joint_top1_accuracy"] == 0.0


def test_top1_ranker_can_choose_social_behavior_during_cooccurring_group_event():
    table = CandidateTable(
        (
            Candidate("together", (1, 2), 2.0),
            Candidate("social_clustering", (1, 2, 3), 1.0),
        ),
        np.asarray([[1.0], [1.0]], dtype=np.float32),
        ("bias",),
    )
    coefficients = {behavior: [0.1] for behavior in CANONICAL_BEHAVIORS}
    coefficients["together"] = [100.0]
    model = {"feature_names": ["bias"], "coefficients": coefficients}

    prediction = predict_table(table, model)

    assert prediction.behavior == "together"
    assert prediction.target_ids == (1, 2)


def test_group_hierarchy_isolation_then_clustering_then_huddling():
    candidates = (
        Candidate("huddle", (1, 2, 3), 3.0),
        Candidate("social_clustering", (1, 2, 3), 2.0),
        Candidate("isolation", (1,), 1.0),
    )
    table = CandidateTable(candidates, np.ones((3, 1), dtype=np.float32), ("bias",))
    coefficients = {behavior: [1.0] for behavior in CANONICAL_BEHAVIORS}
    prediction = predict_table(table, {"feature_names": ["bias"], "coefficients": coefficients})

    assert prediction.behavior == "huddle"
    assert prediction.target_ids == (1, 2, 3)


def test_unrelated_group_does_not_override_social_candidate():
    candidates = (
        Candidate("huddle", (7, 8, 9), 2.0),
        Candidate("approach", (1, 2), 3.0),
    )
    table = CandidateTable(candidates, np.ones((2, 1), dtype=np.float32), ("bias",))
    coefficients = {behavior: [1.0] for behavior in CANONICAL_BEHAVIORS}
    coefficients["approach"] = [2.0]

    prediction = predict_table(table, {"feature_names": ["bias"], "coefficients": coefficients})

    assert prediction.behavior == "approach"


def test_hard_margin_objective_is_explicit_and_reproducible():
    table = CandidateTable(
        (Candidate("locomotion", (1,), 1.0), Candidate("locomotion", (2,), 1.0)),
        np.asarray([[1.0, 0.9], [1.0, 0.1]], dtype=np.float32),
        ("bias", "salience"),
    )
    label = LabeledClip("opaque", "unused", 30.0, "session", "locomotion", (2,))
    first, _ = fit_ranker([table], [label], steps=20, regularization=0.01,
                           learning_rate=0.03, objective="hard_margin")
    second, _ = fit_ranker([table], [label], steps=20, regularization=0.01,
                            learning_rate=0.03, objective="hard_margin")
    assert first == second
    assert first["objective"] == "hard_margin"
    assert first["learning_rate"] == 0.03


def test_session_model_average_preserves_schema_and_averages_coefficients():
    first = {
        "feature_names": ["bias"],
        "coefficients": {behavior: [1.0] for behavior in CANONICAL_BEHAVIORS},
    }
    second = {
        "feature_names": ["bias"],
        "coefficients": {behavior: [3.0] for behavior in first["coefficients"]},
    }
    averaged = average_ranker_models((first, second))
    assert averaged["ensemble_type"] == "session_mean"
    assert averaged["ensemble_size"] == 2
    assert all(values == [2.0] for values in averaged["coefficients"].values())


_LEGACY_AUTO_SCALE = {"mode": "body_length", "assumed_mouse_body_length_cm": 8.0}


def _group_features(tmp_path, frames):
    data = []
    for frame in range(frames):
        detections = []
        for track_id, x in ((1, 0), (2, 2), (3, 4), (4, 100), (5, 102), (6, 104)):
            points = [[x + dx, dy, 0.99] for dx, dy in ((0, 0), (1, 1), (1, -1), (3, 0), (5, 1), (5, -1), (10, 0))]
            detections.append({"track_id": track_id, "keypoints": points, "box": [x, -1, x + 10, 1]})
        data.append({"frame": frame, "detections": detections})
    path = tmp_path / "tracks.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return load_exported_features(path, fps=1, scale_config=_LEGACY_AUTO_SCALE)


def _forming_groups(features):
    distances = features.pair_distance_cm.copy()
    leading_pairs = ((features.pair_i == 0) & np.isin(features.pair_j, (1, 2)))
    leading_pairs |= ((features.pair_i == 3) & np.isin(features.pair_j, (4, 5)))
    distances[0, leading_pairs] = 10.0
    speed = features.speed_cm_s.copy()
    speed[: min(3, len(speed)), :] = 8.0
    return replace(features, pair_distance_cm=distances, speed_cm_s=speed)


def test_expanded_groups_keep_disjoint_clusters_separate_and_obey_five_seconds(tmp_path):
    profiles = rule_profiles(HeuristicParameters())
    short = build_candidate_table(_group_features(tmp_path, 4), profiles)
    assert not any(c.behavior == "social_clustering" for c in short.candidates)
    stable_cluster = build_candidate_table(_group_features(tmp_path, 9), profiles)
    assert not any(c.behavior == "social_clustering" for c in stable_cluster.candidates)
    assert any(c.behavior == "huddle" for c in stable_cluster.candidates)
    long = build_candidate_table(_forming_groups(_group_features(tmp_path, 9)), profiles)
    cluster_sets = {c.target_ids for c in long.candidates if c.behavior == "social_clustering"}
    assert (1, 2, 3) in cluster_sets
    assert (4, 5, 6) in cluster_sets
    assert (1, 2, 3, 4, 5, 6) not in cluster_sets
    assert all(np.isfinite(long.values).flat)


def test_social_clustering_counts_observed_formation_inside_five_second_event(tmp_path):
    features = _forming_groups(_group_features(tmp_path, 5))
    table = build_candidate_table(features, rule_profiles(HeuristicParameters()))
    assert any(
        candidate.behavior == "social_clustering" and candidate.target_ids == (1, 2, 3)
        for candidate in table.candidates
    )


def test_social_clustering_duration_and_motion_include_formation_onset(tmp_path):
    features = _forming_groups(_group_features(tmp_path, 5))
    speed = np.zeros_like(features.speed_cm_s)
    speed[:3, :] = 8.0
    features = replace(features, speed_cm_s=speed)

    profile = HeuristicParameters(
        clustering_min_duration_s=5.0,
        clustering_min_mean_speed_cm_s=5.0,
        clustering_min_nearest_neighbor_drop_cm=2.0,
        clustering_window_s=2.0,
    )
    table = build_candidate_table(features, (profile,))

    assert any(
        candidate.behavior == "social_clustering" and candidate.target_ids == (1, 2, 3)
        for candidate in table.candidates
    )


def test_social_clustering_requires_multiple_moving_members(tmp_path):
    features = _forming_groups(_group_features(tmp_path, 9))
    speed = np.zeros_like(features.speed_cm_s)
    speed[:3, 0] = 12.0
    features = replace(features, speed_cm_s=speed)
    profile = HeuristicParameters(
        clustering_min_mean_speed_cm_s=3.0,
        clustering_min_moving_member_fraction=0.5,
    )

    table = build_candidate_table(features, (profile,))

    assert not any(
        candidate.behavior == "social_clustering" and candidate.target_ids == (1, 2, 3)
        for candidate in table.candidates
    )


def test_social_clustering_gap_respects_configured_frame_limit(tmp_path):
    features = _forming_groups(_group_features(tmp_path, 180))
    speed = np.zeros_like(features.speed_cm_s)
    speed[:61, :] = 8.0
    features = replace(features, fps=30.0, speed_cm_s=speed)
    # [STEP6 2026-10-05] clustering_min_duration_s 新默认为 0（无最小时长门）；要让
    # “帧间隔上限”重新可观测，必须显式钉住时长门，否则任一分段都单独成立。
    profile = HeuristicParameters(clustering_max_gap_s=0.25, clustering_min_duration_s=5.0)

    def has_group(gap_frames):
        valid = features.valid.copy()
        distances = features.pair_distance_cm.copy()
        valid[90 : 90 + gap_frames, :] = False
        distances[90 : 90 + gap_frames, :] = np.inf
        table = build_candidate_table(
            replace(features, valid=valid, pair_distance_cm=distances), (profile,)
        )
        return any(
            candidate.behavior == "social_clustering" and candidate.target_ids == (1, 2, 3)
            for candidate in table.candidates
        )

    assert has_group(7)
    assert not has_group(8)


def test_huddling_has_no_speed_gate_while_clustering_requires_formation_motion(tmp_path):
    # [STEP6 2026-10-05] Huddling 删除全部速度门（门不存在回归断言）：全员
    # 120 cm/s 的高速群体仍然提供 huddle 候选；Social clustering 仍要求成形运动证据。
    features = _forming_groups(_group_features(tmp_path, 9))
    fast = replace(features, speed_cm_s=np.full_like(features.speed_cm_s, 120.0))
    table = build_candidate_table(fast, rule_profiles(HeuristicParameters()))

    assert any(c.behavior == "huddle" and c.target_ids == (1, 2, 3) for c in table.candidates)
    assert any(
        c.behavior == "social_clustering" and c.target_ids == (1, 2, 3)
        for c in table.candidates
    )


def test_huddling_onset_resting_gate_is_deleted(tmp_path):
    # [STEP6 2026-10-05] Huddling onset resting 门已删除（门不存在回归断言）：
    # 群组起点的一次短暂运动爆发不再取消 huddle 候选。
    features = _group_features(tmp_path, 9)
    speed = features.speed_cm_s.copy()
    speed[0, :3] = 12.0
    burst = build_candidate_table(
        replace(features, speed_cm_s=speed), rule_profiles(HeuristicParameters())
    )
    resting = build_candidate_table(features, rule_profiles(HeuristicParameters()))

    def has_huddle(table):
        return any(
            candidate.behavior == "huddle" and candidate.target_ids == (1, 2, 3)
            for candidate in table.candidates
        )

    # 门已删除 => 起点爆发与完全静止得到相同的 huddle 候选。
    assert has_huddle(burst)
    assert has_huddle(resting)


def test_together_distance_profiles_stay_within_the_calibrated_band():
    # [STEP6 2026-10-05] Together 上限新默认 6 cm；训练可选 profile bank 只在
    # [6,8] cm 内探索，且下限 together_min_distance_cm 固定 3 cm，与上限探索解耦。
    profiles = rule_profiles(HeuristicParameters())

    assert profiles[0].together_max_distance_cm == 8.025
    assert {profile.together_max_distance_cm for profile in profiles} <= {8.025, 10.7}
    assert all(profile.together_min_distance_cm == 4.0125 for profile in profiles)


def test_group_candidates_retain_core_despite_temporary_extra_track(tmp_path):
    features = _group_features(tmp_path, 6)
    # Add a spatially close fourth member for two frames without claiming that
    # its visual ID is another animal's stable RFID identity.
    distances = features.pair_distance_cm.copy()
    near_four = ((features.pair_i == 3) & (features.pair_j < 3)) | ((features.pair_j == 3) & (features.pair_i < 3))
    distances[:2, near_four] = 2
    features = replace(features, pair_distance_cm=distances)
    table = build_candidate_table(features, rule_profiles(HeuristicParameters()))
    assert any(c.behavior == "huddle" and c.target_ids == (1, 2, 3) for c in table.candidates)


def test_huddle_offers_sparse_observed_member_without_inventing_identity(tmp_path):
    features = _group_features(tmp_path, 9)
    valid = features.valid.copy()
    valid[:6, 3] = False
    distances = features.pair_distance_cm.copy()
    fourth_edges = (features.pair_i == 3) | (features.pair_j == 3)
    distances[:, fourth_edges] = np.inf
    near_core = fourth_edges & (np.isin(features.pair_i, (0, 1, 2, 3))) & (
        np.isin(features.pair_j, (0, 1, 2, 3))
    )
    distances[6:, near_core] = 0.8
    table = build_candidate_table(replace(features, valid=valid, pair_distance_cm=distances),
                                  rule_profiles(HeuristicParameters()))
    huddle_ids = {candidate.target_ids for candidate in table.candidates
                  if candidate.behavior == "huddle"}
    assert (1, 2, 3) in huddle_ids
    assert (1, 2, 3, 4) in huddle_ids
    assert all(set(ids).issubset(set(features.track_ids)) for ids in huddle_ids)


def test_huddle_center_distance_uses_measured_body_parameter_basis(tmp_path):
    features = _group_features(tmp_path, 9)
    distances = np.full_like(features.pair_distance_cm, np.inf)
    core = np.isin(features.pair_i, (0, 1, 2)) & np.isin(features.pair_j, (0, 1, 2))
    distances[:, core] = 10.7
    profiles = rule_profiles(HeuristicParameters())
    assert profiles[-1].huddle_distance_cm == 13.375
    table = build_candidate_table(replace(features, pair_distance_cm=distances,
                                          mouse_width_cm=5.35), profiles)
    assert any(candidate.behavior == "huddle" and candidate.target_ids == (1, 2, 3)
               for candidate in table.candidates)


def test_group_candidates_offer_dense_core_inside_bridged_component(tmp_path):
    features = _group_features(tmp_path, 6)
    distances = np.full_like(features.pair_distance_cm, 100.0)
    core = {0, 1, 2, 3}
    for edge, (left, right) in enumerate(zip(features.pair_i, features.pair_j)):
        if {int(left), int(right)} <= core or {int(left), int(right)} == {0, 4}:
            distances[:, edge] = 2.0
    table = build_candidate_table(
        replace(features, pair_distance_cm=distances, mouse_width_cm=2.1),
        rule_profiles(HeuristicParameters()),
    )
    assert any(
        candidate.behavior == "huddle" and candidate.target_ids == (1, 2, 3, 4)
        for candidate in table.candidates
    )


def test_spatial_group_survives_one_missing_detection_frame(tmp_path):
    features = _forming_groups(_group_features(tmp_path, 9))
    # One missing video detection within a nine-second spatial episode should
    # not split it into two episodes below the five-second duration gate.
    valid = features.valid.copy()
    valid[3, :] = False
    distances = features.pair_distance_cm.copy()
    distances[3, :] = np.inf
    features = replace(features, valid=valid, pair_distance_cm=distances)
    # At this fixture's 1 fps, one missing frame is one second; explicitly
    # allow that gap instead of relying on the 0.25-second production default.
    table = build_candidate_table(
        features, rule_profiles(HeuristicParameters(clustering_max_gap_s=1.0))
    )
    assert any(c.behavior == "social_clustering" and c.target_ids == (1, 2, 3) for c in table.candidates)


def test_pair_temporal_features_distinguish_sustained_contact_from_brief_contact(tmp_path):
    features = _group_features(tmp_path, 12)
    pair = (features.pair_i == 0) & (features.pair_j == 1)
    sustained = features.pair_distance_cm.copy()
    sustained[:, pair] = np.r_[np.full(3, 20.0), np.full(9, 2.0)][:, None]
    brief = features.pair_distance_cm.copy()
    brief[:, pair] = np.r_[np.full(9, 20.0), np.full(3, 2.0)][:, None]
    long_event = _physical_features(replace(features, pair_distance_cm=sustained), (1, 2))
    short_event = _physical_features(replace(features, pair_distance_cm=brief), (1, 2))
    assert long_event["distance_close_5_longest_fraction"] > short_event["distance_close_5_longest_fraction"]
    assert long_event["distance_late_q50"] < short_event["distance_early_q50"]
    assert all(np.isfinite(value) for value in long_event.values())


def test_visible_tracks_and_pairs_remain_candidates_when_threshold_evidence_is_zero(tmp_path):
    features = _group_features(tmp_path, 3)
    impossible = HeuristicParameters(
        stationary_max_speed_cm_s=-1,
        # [STEP4 2026-10-04] 个体层合并：Walking+Running -> Locomotion
        # The merged mask has no speed band any more, so an unreachable duration
        # floor is what zeroes its evidence now.
        locomotion_min_duration_s=1_000,
        together_max_distance_cm=0,
        pair_max_distance_cm=0,
        contact_distance_cm=0,
        huddle_distance_cm=0,
        isolation_distance_cm=1_000,
        clustering_max_distance_cm=0,
    )
    table = build_candidate_table(features, (impossible,))
    keys = {(candidate.behavior, candidate.target_ids) for candidate in table.candidates}
    for track_id in features.track_ids:
        # [STEP4 2026-10-04] 个体层合并：Walking+Running -> Locomotion。
        # grooming/rearing 是个体评分槽位，但 Phase B 进入标注 schema 前不作 ranker 候选。
        for behavior in ("locomotion", "stationary", "isolation"):
            assert (behavior, (int(track_id),)) in keys
    for left, right in zip(features.pair_i, features.pair_j):
        ids = tuple(sorted((int(features.track_ids[left]), int(features.track_ids[right]))))
        for behavior in CANONICAL_BEHAVIORS:
            if behavior not in {"approach", "attack", "nose_head_contact", "nose_tail_contact",
                                "together", "chase"}:
                continue
            assert (behavior, ids) in keys
