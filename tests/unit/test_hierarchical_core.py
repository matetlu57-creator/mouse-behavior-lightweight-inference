import numpy as np
import pytest
import yaml
from pathlib import Path

from mouse_behavior.behavior import hierarchical_core as h


def _pair(**overrides):
    values = dict(
        actor=1, target=2, duration_s=1.0, center_distance_cm=3.0,
        distance_slope_cm_s=-1.0, heading_similarity=0.8,
        actor_toward_target_cos=0.9, target_escape_cos=0.0,
        actor_speed_cm_s=5.0, target_speed_cm_s=0.2,
        actor_motion=h.MotionLabel.LOCOMOTION.value,
        target_motion=h.MotionLabel.STATIC.value, pair_contact=True,
    )
    values.update(overrides)
    return h.PairWindowStats(**values)


def _group(**overrides):
    values = dict(
        members=(1, 2, 3), duration_s=6.0,
        median_neighbor_distance_cm=3.0, median_neighbor_width_ratio=0.8,
        nnd_slope_cm_s=-0.3, group_radius_slope_cm_s=-0.3,
        moving_fraction=0.1, inward_motion_fraction=0.8,
        high_motion_fraction=0.2, direction_coherence=0.8,
        spacing_cv=0.2, member_recovery_near_group=1.0,
    )
    values.update(overrides)
    return h.GroupWindowStats(**values)


def test_missing_keypoints_remain_nan_and_geometry_uses_valid_points():
    points = np.array([
        [10., 10.], [11., 10.], [9., 10.], [10., 12.],
        [9., 15.], [11., 15.], [10., 18.],
    ])
    points[1] = np.nan
    geometry = h.compute_individual_geometry(h.PoseObservation(1, 0, 0., points), 10.)
    assert geometry.quality.completeness == pytest.approx(6 / 7)
    assert np.isnan(geometry.intra_dist_cm[1]).all()
    assert np.isfinite(geometry.center_cm).all()
    assert geometry.mouse_width_cm == pytest.approx(0.2)


def test_pair_geometry_coverage_uses_valid_keypoint_pairs():
    left = np.arange(14, dtype=float).reshape(7, 2)
    right = left + 10
    left[0] = np.nan
    right[6] = np.nan
    obs_left = h.PoseObservation(1, 0, 0., left)
    obs_right = h.PoseObservation(2, 0, 0., right)
    geom_left = h.compute_individual_geometry(obs_left, 1.)
    geom_right = h.compute_individual_geometry(obs_right, 1.)
    pair = h.compute_pair_geometry(obs_left, obs_right, geom_left, geom_right, 1.)
    assert pair.inter_valid.sum() == 36
    assert pair.coverage == pytest.approx(36 / 49)
    with pytest.raises(ValueError, match="geometry IDs"):
        h.compute_pair_geometry(obs_left, obs_right, geom_right, geom_left, 1.)


def test_individual_states_are_unranked_disjoint_and_allow_locomotion_stop():
    # [STEP4 2026-10-04] 个体层合并：Walking+Running -> Locomotion
    # Locomotion is the union of the former Running and Walking speed bands: one
    # label for every speed at or above the moving floor, including the old
    # high-speed Running range and the old low-speed Walking range.
    cfg = h.MotionThresholds()
    fast = h.detect_individual_motion(h.MotionWindowStats(1, .5, 45, 30, 25, 1., 2.), cfg)
    slow = h.detect_individual_motion(h.MotionWindowStats(1, 1.2, 5, 6, 4, .45, 2.), cfg)
    static = h.detect_individual_motion(h.MotionWindowStats(1, 1.2, .2, .2, .05, .1, .2), cfg)
    assert (fast.label, slow.label, static.label) == (
        h.MotionLabel.LOCOMOTION.value,
        h.MotionLabel.LOCOMOTION.value,
        h.MotionLabel.STATIC.value,
    )
    assert fast.evidence["speed_cm_s"] == pytest.approx(45.)
    assert cfg.moving_min_speed_cm_s == 0.66875
    assert cfg.locomotion_min_duration_s == .5
    # Overlapping Static/Locomotion bands are rejected by the config guard. The
    # in-detector RuntimeError is a defensive backstop that a validate()-passing
    # config can never reach, so it is asserted through the reachable guard.
    with pytest.raises(ValueError, match="overlap"):
        h.MotionThresholds(static_max_speed_cm_s=2, moving_min_speed_cm_s=1).validate()


def test_locomotion_has_no_directionality_or_active_fraction_gate():
    # [STEP4 2026-10-04] 个体层合并：Walking+Running -> Locomotion
    # Regression: the deleted Running directionality gate and the deleted
    # Walking active-fraction gate must not come back. A long, meandering,
    # mostly-paused track that still averages above the moving floor is
    # Locomotion, even though both removed gates would have rejected it.
    cfg = h.MotionThresholds()
    stats = h.MotionWindowStats(1, 1.2, 5., 60., 1., .05, 2.)
    assert stats.directionality == pytest.approx(1. / 60.)
    assert stats.active_fraction == pytest.approx(.05)
    decision = h.detect_individual_motion(stats, cfg)
    assert decision.label == h.MotionLabel.LOCOMOTION.value
    for deleted in (
        "running_min_directionality", "walking_min_active_fraction",
        "walking_min_speed_cm_s", "walking_max_speed_cm_s",
        "running_min_duration_s", "walking_min_duration_s",
    ):
        assert not hasattr(cfg, deleted), f"{deleted} must stay deleted"
    assert h.MotionLabel.LOCOMOTION.value == "Locomotion"
    for deleted_label in ("RUNNING", "WALKING"):
        assert not hasattr(h.MotionLabel, deleted_label)


def test_locomotion_duration_floor_replaces_the_two_deleted_floors():
    # [STEP4 2026-10-04] 个体层合并：Walking+Running -> Locomotion
    cfg = h.MotionThresholds()
    moving = dict(mean_speed_cm_s=5., path_length_cm=6., net_displacement_cm=4.,
                  active_fraction=.45, center_radius_cm=2.)
    too_short = h.detect_individual_motion(
        h.MotionWindowStats(1, cfg.locomotion_min_duration_s - .01, **moving), cfg)
    long_enough = h.detect_individual_motion(
        h.MotionWindowStats(1, cfg.locomotion_min_duration_s, **moving), cfg)
    assert too_short.label == h.MotionLabel.UNKNOWN.value
    assert long_enough.label == h.MotionLabel.LOCOMOTION.value


def test_rearing_and_grooming_evidence_outrank_the_speed_gate():
    # [STEP4 2026-10-04] 个体层合并：Walking+Running -> Locomotion
    # Negative definition: posture evidence suppresses the speed-based labels and
    # Rearing outranks Grooming. Both detectors ship disabled, so their scores are
    # only reachable through explicit evidence flags.
    cfg = h.MotionThresholds()
    base = dict(duration_s=1.2, mean_speed_cm_s=5., path_length_cm=6.,
                net_displacement_cm=4., active_fraction=.45, center_radius_cm=2.)
    assert h.detect_individual_motion(
        h.MotionWindowStats(1, rearing_evidence=True, grooming_evidence=True, **base),
        cfg).label == h.MotionLabel.REARING.value
    assert h.detect_individual_motion(
        h.MotionWindowStats(1, grooming_evidence=True, **base),
        cfg).label == h.MotionLabel.GROOMING.value
    assert h.detect_individual_motion(
        h.MotionWindowStats(1, **base), cfg).label == h.MotionLabel.LOCOMOTION.value
    assert h.MotionLabel.GROOMING.value == "Grooming"
    assert h.MotionLabel.REARING.value == "Rearing"


def test_static_accepts_only_bounded_brief_body_displacement():
    cfg = h.MotionThresholds()
    allowed = h.MotionWindowStats(
        1, 1.2, 2.0, 2.4, 0.8, 0.2, 0.8,
        transient_motion_duration_s=0.2,
        transient_displacement_cm=1.3375,
    )
    too_long = h.MotionWindowStats(
        1, 1.2, 2.0, 2.4, 0.8, 0.2, 0.8,
        transient_motion_duration_s=0.21,
        transient_displacement_cm=1.0,
    )
    too_far = h.MotionWindowStats(
        1, 1.2, 2.0, 2.4, 0.8, 0.2, 0.8,
        transient_motion_duration_s=0.2,
        transient_displacement_cm=1.34,
    )

    assert h.detect_individual_motion(allowed, cfg).label == h.MotionLabel.STATIC.value
    assert h.detect_individual_motion(too_long, cfg).label == h.MotionLabel.LOCOMOTION.value
    assert h.detect_individual_motion(too_far, cfg).label == h.MotionLabel.LOCOMOTION.value


def test_brief_contact_stays_approach_or_together_but_not_contact():
    # [STEP6 2026-10-05] Approach: closing≥2 cm/s + center∈[2,5] cm + actor LOCOMOTION 才计分。
    stats = _pair(snout_head_contact_duration_s=.3, nose_to_target_head_cm=1.,
                  center_distance_cm=3., distance_slope_cm_s=-3.)
    candidates = h.social_candidates_from_pair(stats, h.SocialThresholds())
    labels = {item.label for item in candidates}
    # 0.3 s < min_contact_duration_s(0.5)：短暂接触不得升级为接触标签。
    assert h.SocialLabel.SNOUT_HEAD.value not in labels
    assert h.SocialLabel.APPROACH.value in labels
    # [STEP6 2026-10-05] Together 无速度门：同一帧组也是 Together 候选，由优先级让位给 Approach。
    assert h.SocialLabel.TOGETHER.value in labels
    assert h.resolve_social_candidates(candidates).label == h.SocialLabel.APPROACH.value


def test_together_uses_distance_band_without_motion_gate():
    # [STEP6 2026-10-05] Together: 中心距离带 [3,6] cm、≥ 1 s、无速度门/无静止门；
    # 接触帧（pair_contact）容忍，interaction 帧由优先级层处理。
    cfg = h.SocialThresholds()
    assert cfg.together_max_distance_cm == 8.025
    assert cfg.together_min_distance_cm == 4.0125
    assert cfg.together_min_duration_s == 1.0
    for deleted in ("together_max_individual_speed_cm_s", "together_max_combined_speed_cm_s"):
        assert not hasattr(cfg, deleted), f"{deleted} must stay deleted"

    def together(**overrides):
        return h.SocialLabel.TOGETHER.value in {
            item.label for item in h.social_candidates_from_pair(_pair(**overrides), cfg)}

    moving = dict(actor_motion=h.MotionLabel.LOCOMOTION.value,
                  target_motion=h.MotionLabel.STATIC.value,
                  actor_speed_cm_s=5.0, target_speed_cm_s=0.2,
                  center_distance_cm=6.0, pair_contact=False, distance_slope_cm_s=0.0)
    resting = dict(actor_motion=h.MotionLabel.STATIC.value,
                   target_motion=h.MotionLabel.STATIC.value,
                   actor_speed_cm_s=0.2, target_speed_cm_s=0.3,
                   center_distance_cm=6.0, pair_contact=False, distance_slope_cm_s=0.0)
    # 速度门/静止门已删除：移动对与静止对同样成立，高速对也不得被拒绝。
    assert together(**moving)
    assert together(**resting)
    assert together(**{**moving, "actor_speed_cm_s": 40.0, "target_speed_cm_s": 35.0})
    # 上边界：6 cm 含，6.1/7 cm 不含。
    assert together(**{**resting, "center_distance_cm": cfg.together_max_distance_cm})
    assert not together(**{**resting, "center_distance_cm": cfg.together_max_distance_cm + .1})
    assert not together(**{**resting, "center_distance_cm": cfg.together_max_distance_cm + 1.})
    # 下边界：3 cm 含；2.9 cm 无接触不含，接触帧容忍。
    assert together(**{**resting, "center_distance_cm": cfg.together_min_distance_cm})
    assert not together(**{**resting, "center_distance_cm": cfg.together_min_distance_cm - .1})
    assert together(**{**resting, "center_distance_cm": cfg.together_min_distance_cm - .1, "pair_contact": True})
    # 时长门仍为 1 s。
    assert not together(**{**resting, "duration_s": 0.9})


def test_sustained_snout_contact_outranks_approach_and_together():
    stats = _pair(snout_head_contact_duration_s=.5, nose_to_target_head_cm=1.)
    chosen = h.resolve_social_candidates(h.social_candidates_from_pair(stats, h.SocialThresholds()))
    assert chosen.label == h.SocialLabel.SNOUT_HEAD.value


def test_chasing_requires_two_fast_mice_no_contact_and_duration():
    # [STEP4 2026-10-04] 个体层合并：Walking+Running -> Locomotion
    # [STEP6 2026-10-05] Chasing: center≤12 cm、双鼠≥ 20 cm/s、heading cos≥0.707、
    # ≥ 1 s、逐帧零接触；两鼠都必须 Locomotion 且达到速度下限。
    cfg = h.SocialThresholds()
    assert cfg.chasing_min_speed_cm_s == 26.75
    assert cfg.chasing_max_distance_cm == 16.05
    assert cfg.chasing_min_duration_s == 1.0
    assert cfg.min_heading_similarity == 0.707
    base = dict(duration_s=2.5, center_distance_cm=10., distance_slope_cm_s=-.2,
        heading_similarity=.9, actor_toward_target_cos=.8, target_escape_cos=.8,
        actor_speed_cm_s=45., target_speed_cm_s=44.,
        actor_motion=h.MotionLabel.LOCOMOTION.value,
        target_motion=h.MotionLabel.LOCOMOTION.value,
        pair_contact=False)

    def chasing(values):
        return any(item.label == h.SocialLabel.CHASING.value for item in
                   h.social_candidates_from_pair(_pair(**values), cfg))

    assert h.resolve_social_candidates(h.social_candidates_from_pair(
        _pair(**base), cfg)).label == h.SocialLabel.CHASING.value
    assert not chasing({**base, "pair_contact": True})
    assert not chasing({**base, "duration_s": 0.9})
    assert not chasing({**base, "center_distance_cm": cfg.chasing_max_distance_cm + 1.})
    assert not chasing({**base, "heading_similarity": cfg.min_heading_similarity - .01})
    assert not chasing({**base, "actor_speed_cm_s": cfg.chasing_min_speed_cm_s - 1.})
    assert not chasing({**base, "target_speed_cm_s": cfg.chasing_min_speed_cm_s - 1.})
    # A slow Locomotion pair is still Locomotion individually but never Chasing.
    assert not chasing({**base, "actor_speed_cm_s": 5., "target_speed_cm_s": 5.})


def test_following_suspended_and_never_emitted():
    # [SUSPENDED-STEP3 2026-10-04] Following/Avoiding 暂停（用户决策），保留待恢复。
    stats = _pair(duration_s=3.5, center_distance_cm=15., distance_slope_cm_s=0.,
        target_escape_cos=-.1, actor_speed_cm_s=10., target_speed_cm_s=9.,
        actor_motion=h.MotionLabel.LOCOMOTION.value,
        target_motion=h.MotionLabel.LOCOMOTION.value,
        pair_contact=False)
    labels = {item.label for item in h.social_candidates_from_pair(stats, h.SocialThresholds())}
    assert "Following" not in labels
    with pytest.raises(AttributeError):
        h.SocialLabel.FOLLOWING


def test_avoiding_suspended_and_never_emitted():
    # [SUSPENDED-STEP3 2026-10-04] Following/Avoiding 暂停（用户决策），保留待恢复。
    values = dict(duration_s=.5, center_distance_cm=7., distance_slope_cm_s=2.,
        heading_similarity=.2, actor_toward_target_cos=.2, target_escape_cos=.8,
        actor_speed_cm_s=3., target_speed_cm_s=15.,
        actor_motion=h.MotionLabel.LOCOMOTION.value,
        target_motion=h.MotionLabel.LOCOMOTION.value,
        pair_contact=False, target_turn_deg=70.,
        approach_recent=True)
    labels = {item.label for item in
              h.social_candidates_from_pair(_pair(**values), h.SocialThresholds())}
    assert "Avoiding" not in labels
    with pytest.raises(AttributeError):
        h.SocialLabel.AVOIDING


def test_attack_requires_body_contact_and_uses_configured_weights():
    evidence = dict(actor=1, target=2, duration_s=.8, rapid_closing=.8,
        pounce_or_bite=.9, entanglement=.8, local_motion_energy=.8,
        occlusion_or_observation_loss=.8, recovery_escape=.7)
    assert h.detect_attack(h.ConflictWindowStats(body_contact=False, **evidence)) is None
    cfg = h.AttackThresholds(min_conflict_score=.8, rapid_closing_weight=0,
        pounce_or_bite_weight=1, entanglement_weight=0, local_motion_weight=0,
        observation_loss_weight=0, recovery_escape_weight=0)
    decision = h.detect_attack(h.ConflictWindowStats(body_contact=True, **evidence), cfg)
    assert decision is not None and decision.label == h.SocialLabel.ATTACK.value
    assert decision.confidence == pytest.approx(.9)


def test_huddling_wins_collective_conflict_and_can_use_recovery_support():
    stats = _group()
    candidates = h.collective_candidates_from_group(stats, h.CollectiveThresholds())
    labels = {item.label for item in candidates}
    assert h.CollectiveLabel.HUDDLING.value in labels
    assert h.CollectiveLabel.SOCIAL_CLUSTERING.value not in labels
    assert h.resolve_collective_candidates(candidates).label == h.CollectiveLabel.HUDDLING.value
    recovered = _group(member_continuity_fraction=.2, member_recovery_near_group=.8)
    assert h.CollectiveLabel.HUDDLING.value in {
        item.label for item in h.collective_candidates_from_group(recovered, h.CollectiveThresholds())
    }


def test_moving_group_is_clustering_and_huddling_has_no_speed_gate():
    # [STEP6 2026-10-05] Huddling 删除全部速度门（含 onset resting 门）：移动群体
    # 既是 Social clustering 候选，也仍然是 Huddling 候选；冲突由优先级解决。
    cfg = h.CollectiveThresholds()
    assert not hasattr(cfg, "huddle_max_mean_speed_cm_s"), "huddle speed gate must stay deleted"
    stats = _group(moving_fraction=0.6, high_motion_fraction=0.2)
    candidates = h.collective_candidates_from_group(stats, cfg)
    labels = {item.label for item in candidates}
    assert h.CollectiveLabel.SOCIAL_CLUSTERING.value in labels
    assert h.CollectiveLabel.HUDDLING.value in labels
    assert h.resolve_collective_candidates(candidates).label == h.CollectiveLabel.HUDDLING.value
    # 门不存在 => 全员高速运动时 Huddling 也不得被拒绝。
    all_moving = _group(moving_fraction=1.0, high_motion_fraction=1.0)
    assert h.CollectiveLabel.HUDDLING.value in {
        item.label for item in h.collective_candidates_from_group(all_moving, cfg)}


def test_huddle_recovery_accepts_zero_distance_only_for_prior_member():
    assert h.accept_huddle_recovery(previous_members=(1, 2, 3), recovered_mouse_id=2,
        distance_to_current_group_cm=0., reference_mouse_width_cm=3.)
    assert not h.accept_huddle_recovery(previous_members=(1, 2, 3), recovered_mouse_id=4,
        distance_to_current_group_cm=0., reference_mouse_width_cm=3.)


def test_dispersal_requires_previous_cluster_and_group_motion_requires_stable_spacing():
    values = dict(duration_s=12., median_neighbor_distance_cm=10.,
        median_neighbor_width_ratio=2., nnd_slope_cm_s=.5, group_radius_slope_cm_s=.5,
        moving_fraction=.8, inward_motion_fraction=.1, high_motion_fraction=.5,
        direction_coherence=.3, spacing_cv=.8)
    no_history = h.collective_candidates_from_group(_group(**values), h.CollectiveThresholds())
    prior = h.collective_candidates_from_group(_group(**{**values,
        "previous_collective_label": h.CollectiveLabel.HUDDLING.value}), h.CollectiveThresholds())
    assert h.CollectiveLabel.DISPERSAL.value not in {item.label for item in no_history}
    assert h.CollectiveLabel.DISPERSAL.value in {item.label for item in prior}
    stable = _group(duration_s=3.5, median_neighbor_width_ratio=2.,
        median_neighbor_distance_cm=10., nnd_slope_cm_s=0., group_radius_slope_cm_s=0.,
        moving_fraction=.8, inward_motion_fraction=.1, high_motion_fraction=.8,
        direction_coherence=.9, spacing_cv=.2)
    assert h.CollectiveLabel.GROUP_LOCOMOTION.value in {
        item.label for item in h.collective_candidates_from_group(stable, h.CollectiveThresholds())
    }


def test_isolation_is_collective_single_mouse_state():
    cfg = h.CollectiveThresholds(isolation_min_distance_cm=20.0)
    decision = h.detect_isolation(h.IsolationWindowStats(5, 10., 25.), cfg)
    assert decision is not None
    assert decision.level == h.Level.COLLECTIVE
    assert decision.members == (5,)
    assert decision.evidence["min_distance_to_any_cm"] == 25.0
    assert h.detect_isolation(h.IsolationWindowStats(5, 9.9, 25.), cfg) is None
    assert h.detect_isolation(h.IsolationWindowStats(5, 10., 19.), cfg) is None


def test_hierarchy_preserves_all_layers_and_primary_is_highest():
    individual = {
        1: h.BehaviorDecision(h.Level.INDIVIDUAL, h.MotionLabel.LOCOMOTION.value, (1,), 1.),
        2: h.BehaviorDecision(h.Level.INDIVIDUAL, h.MotionLabel.LOCOMOTION.value, (2,), 1.),
        3: h.BehaviorDecision(h.Level.INDIVIDUAL, h.MotionLabel.STATIC.value, (3,), 1.),
    }
    social = h.BehaviorDecision(h.Level.SOCIAL, h.SocialLabel.TOGETHER.value,
        (1, 2), 1., h.SOCIAL_PRIORITY[h.SocialLabel.TOGETHER.value])
    collective = h.BehaviorDecision(h.Level.COLLECTIVE, h.CollectiveLabel.HUDDLING.value,
        (1, 2, 3), 1., h.COLLECTIVE_PRIORITY[h.CollectiveLabel.HUDDLING.value])
    snapshot = h.resolve_behavior_hierarchy(individual_by_mouse=individual,
        social_events=(social,), collective_events=(collective,))
    assert snapshot.individual_by_mouse[1].label == h.MotionLabel.LOCOMOTION.value
    assert snapshot.social_events[0].label == h.SocialLabel.TOGETHER.value
    assert snapshot.primary_by_mouse[1].label == h.CollectiveLabel.HUDDLING.value
    assert snapshot.primary_by_mouse[3].label == h.CollectiveLabel.HUDDLING.value


def test_valid_social_event_overrides_static_primary_but_keeps_individual_layer():
    individual = {
        mouse_id: h.BehaviorDecision(
            h.Level.INDIVIDUAL, h.MotionLabel.STATIC.value, (mouse_id,), 1.0
        )
        for mouse_id in (1, 2)
    }
    social = h.BehaviorDecision(
        h.Level.SOCIAL,
        h.SocialLabel.TOGETHER.value,
        (1, 2),
        1.0,
        h.SOCIAL_PRIORITY[h.SocialLabel.TOGETHER.value],
    )
    snapshot = h.resolve_behavior_hierarchy(
        individual_by_mouse=individual, social_events=(social,)
    )
    assert snapshot.individual_by_mouse[1].label == h.MotionLabel.STATIC.value
    assert snapshot.primary_by_mouse[1].label == h.SocialLabel.TOGETHER.value


def test_uncertain_higher_layer_lock_does_not_fall_through_to_lower_layer():
    individual = {1: h.BehaviorDecision(h.Level.INDIVIDUAL,
        h.MotionLabel.LOCOMOTION.value, (1,), 1.)}
    snapshot = h.resolve_behavior_hierarchy(individual_by_mouse=individual,
        social_locks={1})
    assert snapshot.primary_by_mouse[1].label == h.SocialLabel.UNKNOWN.value
    snapshot = h.resolve_behavior_hierarchy(individual_by_mouse=individual,
        collective_locks={1})
    assert snapshot.primary_by_mouse[1].label == h.CollectiveLabel.UNKNOWN.value


def test_velocity_coherence_and_group_components():
    assert h.velocity_coherence(np.array([[1., 0.], [2., 0.], [3., 0.]])) == pytest.approx(1.)
    matrix = np.array([[0., 2., 20.], [2., 0., 20.], [20., 20., 0.]])
    assert h.connected_components_from_distance(matrix, 3.) == ((0, 1), (2,))


def test_default_config_uses_core_schema_and_hard_duration_floors():
    config_path = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    thresholds = h.thresholds_from_config(config)
    # [STEP4 2026-10-04] 个体层合并：Walking+Running -> Locomotion
    assert thresholds.motion.locomotion_min_duration_s == .5
    assert thresholds.motion.moving_min_speed_cm_s == 0.66875
    assert thresholds.motion.running_min_speed_cm_s == 53.5
    assert thresholds.social.chasing_min_speed_cm_s == 26.75
    assert thresholds.collective.clustering_min_moving_fraction == .5
    assert thresholds.collective.dispersal_min_moving_fraction == .3
    assert thresholds.collective.group_locomotion_min_moving_fraction == .6
    for deleted in ("running_min_duration_s", "walking_min_duration_s",
                    "walking_min_speed_cm_s", "walking_max_speed_cm_s",
                    "running_min_directionality"):
        assert not hasattr(thresholds.motion, deleted), f"{deleted} must stay deleted"
    assert thresholds.social.min_contact_duration_s == .5
    assert thresholds.social.chasing_min_duration_s == 1.
    assert thresholds.social.chasing_max_distance_cm == 16.05
    assert thresholds.social.min_heading_similarity == .707
    # [STEP6 2026-10-05] Together [3,6] cm 带；Approach 终点 [2,5] cm + closing≥2 cm/s。
    assert thresholds.social.together_max_distance_cm == 8.025
    assert thresholds.social.together_min_distance_cm == 4.0125
    assert thresholds.social.together_min_duration_s == 1.
    assert thresholds.social.approach_final_distance_cm == 6.6875
    assert thresholds.social.approach_min_end_distance_cm == 2.675
    assert thresholds.social.approach_min_closing_speed_cm_s == 2.675
    assert thresholds.collective.huddling_min_duration_s == 5.
    assert thresholds.collective.social_clustering_min_duration_s == 0.
    assert thresholds.collective.dispersal_min_duration_s == 5.
    assert thresholds.collective.group_locomotion_min_duration_s == 3.
    assert thresholds.collective.group_locomotion_max_neighbor_distance_cm == 33.4375
    assert thresholds.collective.group_locomotion_min_coherence == .7
    assert thresholds.collective.isolation_min_duration_s == 10.
    assert thresholds.collective.isolation_min_distance_cm == 26.75
    # [STEP6 2026-10-05] 已删字段不得回归（Together/Huddling 速度门）。
    for deleted in ("together_max_individual_speed_cm_s", "together_max_combined_speed_cm_s"):
        assert not hasattr(thresholds.social, deleted), f"{deleted} must stay deleted"
    assert not hasattr(thresholds.collective, "huddle_max_mean_speed_cm_s"), \
        "huddle_max_mean_speed_cm_s must stay deleted"
    with pytest.raises(ValueError, match="unknown social thresholds"):
        h.thresholds_from_config({"social": {"contact_min_duratoin_s": .5}})
