"""Loading heuristic parameter values from project configuration."""

from __future__ import annotations

from mouse_behavior.evaluation.exported_dataset_models import (
    HeuristicParameters,
)

from typing import Any, Mapping
from .requested_thresholds_v27 import ISOLATION_DISTANCE_CM

def config_seed_parameters(config: Mapping[str, Any]) -> HeuristicParameters:
    """Map the repository YAML defaults to the exported-data parameter set."""

    extended = dict(config.get("extended_behavior", {}))
    individual = dict(extended.get("individual", {}))
    social = dict(extended.get("social", {}))
    group = dict(extended.get("group", {}))
    clustering = dict(group.get("social_clustering", {}))
    stationary_max_speed_cm_s = float(individual.get("stationary_max_speed_cm_s", 0.66875))
    individual_motion_speed_threshold_cm_s = float(individual.get("motion_speed_threshold_cm_s", 4.0))
    parameters = HeuristicParameters(
        stationary_max_speed_cm_s=stationary_max_speed_cm_s,
        individual_motion_speed_threshold_cm_s=individual_motion_speed_threshold_cm_s,
        stationary_transient_max_duration_s=float(
            individual.get("stationary_transient_max_duration_seconds", 1.0)
        ),
        stationary_transient_max_displacement_cm=float(
            individual.get("stationary_transient_max_displacement_cm", 4.0)
        ),
        stationary_max_center_radius_cm=float(
            individual.get("stationary_max_center_radius_cm", 4.0)
        ),
        stationary_min_projected_body_length_ratio=float(
            individual.get("stationary_min_projected_body_length_ratio", 0.55)
        ),
        # [STEP4 2026-10-04] 个体层合并：Walking+Running -> Locomotion（用户决策/文档）
        # walking_min_speed_cm_s=float(
        #     individual.get("walking_min_speed_cm_s", stationary_max_speed_cm_s)
        # ),
        # [STEP4 2026-10-04] 个体层合并：Walking+Running -> Locomotion（用户决策/文档）
        # walking_max_speed_cm_s=float(individual.get("walking_max_speed_cm_s", 85.0)),
        locomotion_stop_gap_s=float(individual.get("locomotion_stop_gap_seconds", 1.0)),
        running_min_speed_cm_s=float(individual.get("running_min_speed_cm_s", 113.6875)),
        together_max_distance_cm=float(social.get("together_max_distance_cm", 7.0)),
        together_min_distance_cm=float(social.get("together_min_distance_cm", 4.0125)),
        # [STEP6 2026-10-05] Together: 速度门删除（不再参与掩码）；键仍读取以兼容旧配置/断言。
        together_max_individual_speed_cm_s=float(
            social.get("together_max_individual_speed_cm_s", 16.0)
        ),
        together_max_combined_speed_cm_s=float(
            social.get("together_max_combined_speed_cm_s", 28.0)
        ),
        pair_max_distance_cm=float(social.get("pair_max_distance_cm", 6.6875)),
        approach_min_distance_drop_cm=float(social.get("approach_min_distance_drop_cm", 1.5)),
        approach_min_closing_speed_cm_s=float(social.get("approach_min_closing_speed_cm_s", 2.675)),
        # [STEP6 2026-10-05] Approach: terminal 语义 = 接触截断（nose_nearest > 2.0 cm）；优先读新 yaml 键
        # approach_min_end_distance_cm（主代理约定），缺失时回退旧键 approach_terminal_distance_cm，默认 2.0。
        approach_terminal_distance_cm=float(
            social.get(
                "approach_min_end_distance_cm",
                social.get("approach_terminal_distance_cm", 2.675),
            )
        ),
        contact_distance_cm=float(
            config.get("contact_detection", {}).get("nose_head_distance_cm", 4.0125)
        ),
        contact_min_cumulative_seconds=max(
            float(config.get("contact_detection", {}).get(
                "nose_head_min_cumulative_duration_seconds", 0.2
            )),
            0.2,
        ),
        # [STEP4 2026-10-04] 个体层合并：Walking+Running -> Locomotion（用户决策/文档）
        # running_min_duration_s=max(float(individual.get("running_min_duration_seconds", 0.5)), 0.5),
        locomotion_min_duration_s=max(float(individual.get("locomotion_min_duration_seconds", 0.5)), 0.5),
        attack_min_speed_cm_s=float(
            dict(social.get("attack_fallback", {})).get("min_raw_actor_speed_cm_s", 10.7)
        ),
        attack_vs_approach_multiplier=float(social.get("attack_vs_approach_multiplier", 1.5)),
        attack_endpoint_distance_cm=float(
            dict(social.get("attack_fallback", {})).get("endpoint_distance_cm", 16.05)
        ),
        attack_reacquisition_max_gap_s=float(
            dict(social.get("attack_fallback", {})).get("reacquisition_max_gap_seconds", 5.0)
        ),
        # [STEP6 2026-10-05] Huddling: 中心距离边默认 5→6 cm。
        huddle_distance_cm=float(group.get("huddle_distance_cm", 10.0)),
        huddle_width_multiplier=float(group.get("huddle_width_multiplier", 1.0)),
        # # [STEP6 2026-10-05] 社交/群体参数按文档 v4 + 用户覆盖落地
        # [STEP6 2026-10-05] Huddling: 速度门删除；键仍读取以兼容旧配置与外部 ranker，不再参与判定。
        huddle_max_mean_speed_cm_s=float(group.get("huddle_max_mean_speed_cm_s", 10.0)),
        huddle_nose_distance_cm=float(group.get("huddle_nose_distance_cm", 4.0125)),
        huddle_min_duration_s=float(
            max(
                float(group.get("huddle_min_duration_seconds", group.get("confirm_seconds", 1.0))),
                1.0,
            )
        ),
        huddle_max_gap_s=float(
            group.get("huddle_fill_gap_seconds", group.get("fill_gap_seconds", 5.0))
        ),
        huddle_min_member_support_fraction=float(
            group.get("huddle_min_member_support_fraction", 0.25)
        ),
        isolation_distance_cm=float(group.get("isolation_distance_cm", ISOLATION_DISTANCE_CM)),
        isolation_min_duration_s=max(
            float(group.get("isolation_min_duration_seconds", 10.0)), 10.0
        ),
        isolation_max_gap_s=max(float(group.get("isolation_fill_gap_seconds", 0.2)), 0.0),
        clustering_max_distance_cm=float(clustering.get("max_neighbor_distance_cm", 40.125)),
        clustering_initial_max_distance_cm=float(
            clustering.get("initial_max_neighbor_distance_cm", 32.1)
        ),
        clustering_min_nearest_neighbor_drop_cm=float(
            clustering.get("min_nearest_neighbor_drop_cm", 2.675)
        ),
        stationary_min_duration_s=max(
            float(individual.get("stationary_min_duration_seconds", 1.0)), 1.0
        ),
        # [STEP4 2026-10-04] 个体层合并：Walking+Running -> Locomotion（用户决策/文档）
        # walking_min_duration_s=max(float(individual.get("walking_min_duration_seconds", 1.0)), 1.0),
        pair_min_duration_s=float(social.get("approach_min_duration_seconds", 0.1)),
        together_min_duration_s=float(social.get("together_min_duration_seconds", 1.0)),
        group_min_duration_s=float(group.get("confirm_seconds", 0.3)),
        approach_window_s=0.3,
        clustering_window_s=float(clustering.get("formation_window_seconds", 2.0)),
        # [STEP6 2026-10-05] Social clustering: 最小时长门删除——默认 0.0，不再强制 ≥5 s 下限（仅 >0 时生效）。
        clustering_min_duration_s=max(float(clustering.get("min_duration_seconds", 0.0)), 0.0),
        clustering_max_gap_s=float(
            clustering.get("fill_gap_seconds", group.get("fill_gap_seconds", 0.25))
        ),
        clustering_formation_score_bonus=float(clustering.get("formation_score_bonus", 1.0)),
        clustering_min_mean_speed_cm_s=float(clustering.get("min_mean_speed_cm_s", 4.0125)),
        clustering_min_moving_member_fraction=float(
            clustering.get("min_moving_member_fraction", 0.5)
        ),
        chase_min_duration_s=max(float(social.get("chase_min_duration_seconds", 1.0)), 1.0),
        chase_min_speed_cm_s=float(social.get("chase_min_speed_cm_s", 26.75)),
        chase_fill_gap_s=float(social.get("chase_fill_gap_seconds", 0.5)),
        attack_mild_nose_accel_cm_s2=float(social.get("attack_mild_nose_accel_cm_s2", 66.875)),
        approach_min_start_distance_cm=float(social.get("approach_min_start_distance_cm", 6.6875)),
        approach_min_end_distance_cm=float(social.get("approach_min_end_distance_cm", 2.675)),
        approach_min_actor_speed_cm_s=float(social.get("approach_min_actor_speed_cm_s", 4.0125)),
        # [STEP6 2026-10-05] Chasing: 距离上限默认 30→12 cm。
        chase_max_distance_cm=float(social.get("chasing_max_distance_cm", 16.05)),
        approach_min_facing_cosine=float(social.get("approach_min_facing_cosine", 0.5)),
        approach_min_motion_toward_cosine=float(
            social.get("approach_min_motion_toward_cosine", 0.0)
        ),
        chase_min_chaser_speed_cm_s=float(
            social.get("chase_min_chaser_speed_cm_s", 53.5)
        ),
        chase_min_chaser_facing_cosine=float(
            social.get("chase_min_chaser_facing_cosine", 0.5)
        ),
        chase_min_chaser_toward_cosine=float(
            social.get("chase_min_chaser_toward_cosine", 0.5)
        ),
        chase_min_net_displacement_cm=float(
            social.get("chase_min_net_displacement_cm", 10.7)
        ),
        chase_min_directed_support_s=float(
            social.get("chase_min_directed_support_seconds", 0.2)
        ),
        attack_min_nose_acceleration_toward_cm_s2=float(
            social.get("attack_min_nose_acceleration_toward_cm_s2", 500.0)
        ),
        attack_min_box_reacquisitions_per_s=float(
            social.get("attack_min_box_reacquisitions_per_s", 0.5)
        ),
        attack_box_max_gap_s=float(social.get("attack_box_max_gap_s", 0.5)),
        attack_reacquisition_context_s=float(
            social.get("attack_reacquisition_context_s", 0.5)
        ),
        attack_min_event_duration_s=float(
            social.get("attack_min_event_duration_seconds", 0.033)
        ),
        # [SUSPENDED-STEP3 2026-10-04] Following/Avoiding 暂停（用户决策），保留待恢复
        # following_min_duration_s=max(float(social.get("following_min_duration_seconds", 3.0)), 3.0),
        # [STEP6 2026-10-05] Group locomotion: 邻近距离上限默认 30→25 cm。
        group_locomotion_max_distance_cm=float(
            dict(group.get("group_locomotion", {})).get("max_neighbor_distance_cm", 30.0)
        ),
        group_locomotion_min_duration_s=max(
            float(dict(group.get("group_locomotion", {})).get("min_duration_seconds", 3.0)), 3.0
        ),
        # [STEP6 2026-10-05] Dispersal: 时长下限 10→5 s。
        dispersal_min_duration_s=max(
            float(dict(group.get("dispersal", {})).get("min_duration_seconds", 5.0)), 5.0
        ),
    )


    # The public YAML loader selects the current confirmed rules. Historical
    # serialized HeuristicParameters remain explicitly reproducible separately.
    from dataclasses import replace
    from .confirmed_rules import confirmed_profiles, social_profiles
    return replace(social_profiles(confirmed_profiles(parameters)[0])[0],
                   group_core_candidate_mode=1.0)
