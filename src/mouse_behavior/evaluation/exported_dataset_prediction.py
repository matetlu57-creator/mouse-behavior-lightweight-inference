"""Per-track, pair, and behavior prediction from exported features."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from mouse_behavior.behavior.individual_rules import (
    DEFAULT_INDIVIDUAL_CONFIG,
    SUSPENDED_POSTURE_BEHAVIORS,
    individual_rule_masks,
)
from mouse_behavior.evaluation.exported_dataset_collective import (
    _directional_walker_in_group,
    _dispersal_scores,
    _finite_quantile,
    _group_episode_scores,
    _group_formation_geometry_support,
    _recent_contact_mask,
    _replace_collective_group_scores_with_features,
    _rolling_drop,
    _windowed_body_speed,
)
from mouse_behavior.evaluation.exported_dataset_models import (
    CANONICAL_BEHAVIORS,
    ExportedFeatures,
    GROUP_BEHAVIORS,
    HeuristicParameters,
    INDIVIDUAL_BEHAVIORS,
)
from mouse_behavior.evaluation.social_heuristic_config import (
    DEFAULT_SOCIAL_HEURISTIC_CONFIG,
    SocialHeuristicConfig,
)
from mouse_behavior.evaluation.exported_dataset_temporal import (
    _durations_by_column,
    _isolation_cumulative_durations,
    _pair_key,
    _prioritize_attack_over_approach,
    _qualifying_score,
)

def predict_features(
    features: ExportedFeatures,
    parameters: HeuristicParameters,
    *,
    temporal_only: bool = False,
    social_config: SocialHeuristicConfig = DEFAULT_SOCIAL_HEURISTIC_CONFIG,
) -> dict[str, Any]:
    """Infer behavior evidence for every visible ID without consulting annotations.

    Identity scores are used for single-mouse and group-membership behaviors.
    Pair scores are keyed by the exact unordered visual-ID pair.  Keeping those
    scores separate prevents a behavior performed by unrelated mice elsewhere
    in the clip from being credited to the annotated target IDs later.
    """

    fps = features.fps
    valid = features.valid
    motion_valid = features.motion_valid if features.motion_valid is not None else valid
    speed = features.speed_cm_s
    group_speed = _windowed_body_speed(features, parameters.group_motion_window_s)
    pair_distance = features.pair_distance_cm
    pair_valid = np.isfinite(pair_distance)
    track_ids = [int(value) for value in features.track_ids.tolist()]
    identity_scores: dict[str, dict[str, float]] = {
        str(track_id): {behavior: 0.0 for behavior in (*INDIVIDUAL_BEHAVIORS, *GROUP_BEHAVIORS)}
        for track_id in track_ids
    }
    pair_scores: dict[str, dict[str, float]] = {}
    # [SUSPENDED-STEP3 2026-10-04] dual-label credit removed: Together now allows contact without interaction
    # pair_compatible_labels: dict[str, list[str]] = {}
    group_events: list[dict[str, Any]] = []
    core_mode = parameters.group_core_candidate_mode > 0.0

    individual_config = replace(
        DEFAULT_INDIVIDUAL_CONFIG,
        motion_speed_threshold_cm_s=parameters.individual_motion_speed_threshold_cm_s,
        static_min_duration_s=parameters.stationary_min_duration_s,
        locomotion_min_duration_s=parameters.locomotion_min_duration_s,
        static_max_radius_cm=parameters.stationary_max_center_radius_cm,
        static_transient_max_s=parameters.stationary_transient_max_duration_s,
        static_transient_max_displacement_cm=parameters.stationary_transient_max_displacement_cm,
        locomotion_pause_max_s=parameters.locomotion_stop_gap_s,
    )
    individual = individual_rule_masks(
        features.centers_cm, valid, fps,
        keypoints_cm=features.collective_keypoints_cm,
        reference_body_cm=features.reference_body_cm,
        config=individual_config,
    )
    locomotion = individual.masks['locomotion']
    for behavior, mask in individual.masks.items():
        if behavior in SUSPENDED_POSTURE_BEHAVIORS:
            continue
        durations = _durations_by_column(mask, fps)
        minimum = getattr(individual_config, {
            'stationary': 'static_min_duration_s',
            'locomotion': 'locomotion_min_duration_s',
            'grooming': 'grooming_min_duration_s',
            'rearing': 'rearing_min_duration_s',
        }[behavior])
        for slot, track_id in enumerate(track_ids):
            identity_scores[str(track_id)][behavior] = _qualifying_score(durations[slot], minimum, fps)

    # # [STEP6 2026-10-05] 社交/群体参数按文档 v4 + 用户覆盖落地：Together 速度门删除，pair_*_speed 不再使用。
    # pair_left_speed = speed[:, features.pair_i]
    # pair_right_speed = speed[:, features.pair_j]
    pair_members_observed = valid[:, features.pair_i] & valid[:, features.pair_j]
    # # [STEP6 2026-10-05] 社交/群体参数按文档 v4 + 用户覆盖落地：宽度比例 cutoff 由中心/鼻-鼻边规则取代。
    # huddle_distance = _huddle_pair_distance_cm(features, parameters)
    # # [STEP6 2026-10-05] 社交/群体参数按文档 v4 + 用户覆盖落地：Together 掩码移至 attack/chasing 之后（需要 interaction 证据）。
    # qualifying_together = _qualified_bout_mask(
        # together, max(1, int(np.ceil(parameters.together_min_duration_s * fps)))
    # )

    # # [STEP6 2026-10-05] 社交/群体参数按文档 v4 + 用户覆盖落地：Approach 度量 = 接近鼠鼻头到被接近鼠最近关键点；起点≥5 cm、
    # 终点>2 cm（接触截断）、actor 速度>3 cm/s、closing≥2 cm/s。
    # [STEP6 2026-10-05] Approach: 旧 center-distance 掩码（已废弃，按要求保留注释）：
    # distance_drop = np.zeros_like(pair_distance, dtype=float)
    # for pair in range(pair_distance.shape[1]):
    #     distance_drop[:, pair] = _rolling_drop(pair_distance[:, pair], window)
    # approach = pair_valid & (pair_distance < parameters.approach_terminal_distance_cm)
    # approach &= distance_drop >= parameters.approach_min_distance_drop_cm
    # approach &= features.pair_closing_speed_cm_s >= parameters.approach_min_closing_speed_cm_s
    # approach_durations = _durations_by_column(approach, fps)
    nn_ltr = features.pair_nose_nearest_left_to_right_cm
    nn_rtl = features.pair_nose_nearest_right_to_left_cm
    if nn_ltr is None or nn_rtl is None:
        nn_ltr = nn_rtl = pair_distance
    nn_ltr = np.where(np.isfinite(nn_ltr), nn_ltr, np.inf)
    nn_rtl = np.where(np.isfinite(nn_rtl), nn_rtl, np.inf)
    nose_nearest = np.minimum(nn_ltr, nn_rtl)
    actor_is_left = nn_ltr <= nn_rtl
    approach_actor_speed = np.where(
        actor_is_left, speed[:, features.pair_i], speed[:, features.pair_j]
    )
    window = max(int(round(parameters.approach_window_s * fps)), 1)
    window_seconds = window / max(fps, 1e-6)
    nose_drop = np.zeros_like(nose_nearest, dtype=float)
    for pair in range(nose_nearest.shape[1]):
        column = np.where(np.isfinite(nose_nearest[:, pair]), nose_nearest[:, pair], np.inf)
        nose_drop[:, pair] = _rolling_drop(column, window)
    approach = pair_valid & np.isfinite(nose_nearest)
    # [STEP6 2026-10-05] Approach: 接触截断阈值按规范用 approach_terminal_distance_cm（默认 2.0；yaml 键 approach_min_end_distance_cm 同义）。
    approach &= nose_nearest > parameters.approach_terminal_distance_cm
    approach &= approach_actor_speed > parameters.approach_min_actor_speed_cm_s
    approach &= nose_drop / max(window_seconds, 1e-6) >= parameters.approach_min_closing_speed_cm_s
    approach_durations = np.zeros(pair_distance.shape[1], dtype=float)
    qualifying_approach = np.zeros_like(approach) if temporal_only else None
    for pair in range(pair_distance.shape[1]):
        column = approach[:, pair]
        padded = np.r_[False, column, False].astype(np.int8)
        changes = np.diff(padded)
        best = 0.0
        for start, end in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)):
            if nose_nearest[start, pair] >= parameters.approach_min_start_distance_cm:
                best = max(best, (end - start) / max(fps, 1e-6))
                if qualifying_approach is not None:
                    qualifying_approach[start:end, pair] = True
        approach_durations[pair] = best

    contact = pair_valid & (
        (features.pair_nose_head_cm < parameters.contact_distance_cm)
        | (features.pair_nose_tail_cm < parameters.contact_distance_cm)
    )
    # # [STEP6 2026-10-05] 社交/群体参数按文档 v4 + 用户覆盖落地：轻度攻击=攻击下限：速度门 OR 鼻加速度门。
    attack_intensity = (
        np.maximum(speed[:, features.pair_i], speed[:, features.pair_j])
        >= parameters.attack_min_speed_cm_s
    )
    if features.nose_acceleration_cm_s2 is not None:
        attack_intensity = attack_intensity | (
            np.maximum(
                features.nose_acceleration_cm_s2[:, features.pair_i],
                features.nose_acceleration_cm_s2[:, features.pair_j],
            )
            >= parameters.attack_mild_nose_accel_cm_s2
        )
    attack = contact & attack_intensity & (
        motion_valid[:, features.pair_i] | motion_valid[:, features.pair_j]
    )
    attack_durations = _durations_by_column(attack, fps)
    head_contact = pair_valid & (features.pair_nose_head_cm < parameters.contact_distance_cm)
    tail_contact = pair_valid & (features.pair_nose_tail_cm < parameters.contact_distance_cm)
    # qualified_head_contact = head_contact
    # qualified_tail_contact = tail_contact
    nose_head_cumulative = np.sum(head_contact, axis=0, dtype=np.int64) / max(fps, 1e-6)
    nose_tail_cumulative = np.sum(tail_contact, axis=0, dtype=np.int64) / max(fps, 1e-6)
    for directed, name in (
        (
            (
                features.pair_nose_head_left_to_right_cm,
                features.pair_nose_head_right_to_left_cm,
            ),
            "head",
        ),
        (
            (
                features.pair_nose_tail_left_to_right_cm,
                features.pair_nose_tail_right_to_left_cm,
            ),
            "tail",
        ),
    ):
        if directed[0] is None or directed[1] is None:
            continue
        left_contact = pair_valid & (directed[0] < parameters.contact_distance_cm)
        right_contact = pair_valid & (directed[1] < parameters.contact_distance_cm)
        left_seconds = np.sum(left_contact, axis=0) / fps
        right_seconds = np.sum(right_contact, axis=0) / fps
        qualifying = (
            left_contact
            & (left_seconds[None, :] + 1e-12 >= parameters.contact_min_cumulative_seconds)
        ) | (
            right_contact
            & (right_seconds[None, :] + 1e-12 >= parameters.contact_min_cumulative_seconds)
        )
        if name == "head":
            nose_head_cumulative = np.maximum(left_seconds, right_seconds)
            # qualified_head_contact = qualifying
        else:
            nose_tail_cumulative = np.maximum(left_seconds, right_seconds)
            # qualified_tail_contact = qualifying
    no_contact_frames = max(1, int(np.ceil(parameters.pair_no_contact_seconds * fps)))
    chased = (
        pair_valid
        & (pair_distance >= parameters.contact_distance_cm)
        & (pair_distance <= parameters.chase_max_distance_cm)
        & (features.pair_direction_similarity >= parameters.chase_min_direction_similarity)
        & (speed[:, features.pair_i] >= parameters.chase_min_speed_cm_s)
        & (speed[:, features.pair_j] >= parameters.chase_min_speed_cm_s)
    )
    # walking_pair = walking[:, features.pair_i] & walking[:, features.pair_j]
    # [SUSPENDED-STEP3 2026-10-04] Following/Avoiding 暂停（用户决策），保留待恢复
    # aligned_follow = (
    #     pair_valid
    #     & (pair_distance >= parameters.following_min_distance_cm)
    #     & (pair_distance <= parameters.following_max_distance_cm)
    #     & (features.pair_direction_similarity >= parameters.following_min_direction_similarity)
    #     & walking_pair
    # )
    contact_or_dropout = contact | ~pair_valid
    no_contact = np.zeros_like(pair_valid)
    for pair in range(pair_distance.shape[1]):
        clean = ~contact_or_dropout[:, pair]
        padded = np.r_[False, clean, False].astype(np.int8)
        changes = np.diff(padded)
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1)
        for start, end in zip(starts, ends):
            if end - start >= no_contact_frames:
                no_contact[start:end, pair] = True
    # Brief nose touches are intentionally not elevated to separate contact
    # events; Chasing requires at least one clean 0.5 s interval.
    # [SUSPENDED-STEP3 2026-10-04] Following/Avoiding 暂停（用户决策），保留待恢复
    # following = aligned_follow & no_contact
    # [STEP6 2026-10-05] Chasing: chasing = chased & no_contact 保持（逐帧零接触即全程零接触）。
    chasing = chased & no_contact
    # # [STEP6 2026-10-05] 社交/群体参数按文档 v4 + 用户覆盖落地：仅桥接真 dropout（间隔帧全部 pair 不可见）且 ≤ chase_fill_gap_s。
    chase_bridge = max(1, int(round(parameters.chase_fill_gap_s * fps)))
    for pair in range(chasing.shape[1]):
        column = chasing[:, pair].copy()
        dropout = ~pair_valid[:, pair]
        padded = np.r_[False, column, False].astype(np.int8)
        changes = np.diff(padded)
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1)
        for left_end, right_start in zip(ends[:-1], starts[1:]):
            if right_start - left_end <= chase_bridge and dropout[left_end:right_start].all():
                column[left_end:right_start] = True
        chasing[:, pair] = column
    chasing_durations = _durations_by_column(chasing, fps)
    # following_durations = _durations_by_column(following, fps)
    # # [STEP6 2026-10-05] 社交/群体参数按文档 v4 + 用户覆盖落地：Together = [3,6] cm 带、≥1 s、无速度门；接触帧容忍、互动帧排除。
    # [STEP6 2026-10-05] Together: 帧掩码 = pair_valid & 双鼠可见 & (center ≤ together_max_distance_cm)
    # & ((center ≥ together_min_distance_cm) | ~interaction)；interaction = attack帧 | chasing帧——
    # [3,6] cm 带无条件 Together，<3 cm 接触区仅在非互动帧计入（接触允许、互动不允许）；速度门全部删除。
    interaction = attack | chasing
    # # [STEP6 2026-10-05] 社交/群体参数按文档 v4 + 用户覆盖落地：用户规则"不能有互动"→ interaction 帧硬排除；接触帧(d<2)容忍。
    together = (
        pair_valid
        & pair_members_observed
        & (pair_distance <= parameters.together_max_distance_cm)
        & (
            (pair_distance >= parameters.together_min_distance_cm)
            | (pair_distance < parameters.contact_distance_cm)
        )
        & ~interaction
    )
    together_durations = _durations_by_column(together, fps)
    if parameters.confirmed_rules_v1 > 0:
        from .confirmed_rules import pair_rule_masks
        masks = pair_rule_masks(features, parameters, social_config)
        approach, attack = masks["approach"], masks["attack"]
        chasing, together = masks["chasing"], masks["together"]
        head_contact, tail_contact = masks["head_contact"], masks["tail_contact"]
        qualifying_approach = approach
        approach_durations = _durations_by_column(approach, fps)
        attack_durations = _durations_by_column(attack, fps)
        chasing_durations = (
            np.count_nonzero(chasing, axis=0).astype(float) / max(fps, 1e-9)
            if parameters.confirmed_rules_v1 > 0
            else _durations_by_column(chasing, fps)
        )
        together_durations = _durations_by_column(together, fps)
        nose_head_cumulative = masks["head_contact_seconds"]
        nose_tail_cumulative = masks["tail_contact_seconds"]
    if temporal_only:
        # Initial screening needs boundaries, not clip-wide ranking. These
        # masks reuse the predictor's exact geometry and preserve visual IDs.
        # Return before score recovery/boosts so nonbehavior lead-ins cannot
        # borrow a contact or attack label from a later frame.
        return {
            "track_ids": track_ids,
            "pair_ids": [tuple(sorted((track_ids[int(i)], track_ids[int(j)])))
                         for i, j in zip(features.pair_i, features.pair_j)],
            "individual_masks": individual.masks,
            "individual_diagnostics": individual.diagnostics,
            "pair_masks": {"approach": qualifying_approach, "approach_raw": approach,
                           "attack": attack, "chasing": (chasing if parameters.confirmed_rules_v1 > 0
                                                          else chasing & ~contact & pair_valid),
                           "together": together, "contact": contact,
                           "nose_head_contact": head_contact, "nose_tail_contact": tail_contact},
        }
    # Avoiding is a response after a prior close approach followed by sustained
    # opening distance and a fast change in the pair's relative motion.
    # opening = pair_valid & (
        # features.pair_closing_speed_cm_s <= -parameters.approach_min_closing_speed_cm_s
    # )
    # opening &= -distance_drop >= parameters.avoiding_min_distance_increase_cm
    # avoid_scores = np.zeros(pair_distance.shape[1], dtype=float)
    for pair_index, (left_slot, right_slot) in enumerate(zip(features.pair_i, features.pair_j)):
        key = _pair_key(track_ids[int(left_slot)], track_ids[int(right_slot)])
        attack_score = _qualifying_score(
            attack_durations[pair_index],
            (parameters.attack_min_event_duration_s
             if parameters.confirmed_rules_v1 > 0
             else parameters.pair_min_duration_s),
            fps,
        )
        if attack_score <= 0.0 and parameters.confirmed_rules_v1 <= 0:
            contact_frames = np.flatnonzero(attack[:, pair_index])
            endpoint = pair_valid[:, pair_index] & (
                pair_distance[:, pair_index] <= parameters.attack_endpoint_distance_cm
            )
            endpoint_frames = np.flatnonzero(endpoint)
            if contact_frames.size and endpoint_frames.size:
                last_contact = int(contact_frames[-1])
                # Recovery is valid only if this exact visual-ID pair was
                # genuinely unobserved and the first re-observation is nearby.
                # A continuously visible non-contact interval cannot be used
                # to manufacture an attack from an old contact sample.
                later_endpoint = endpoint_frames[endpoint_frames > last_contact]
                if later_endpoint.size:
                    first_reacquired = int(later_endpoint[0])
                    missing_frames = first_reacquired - last_contact - 1
                    gap_seconds = missing_frames / max(fps, 1e-6)
                    missing_pair = not np.all(
                        pair_valid[last_contact + 1 : first_reacquired, pair_index]
                    )
                    if (
                        missing_frames > 0
                        and missing_pair
                        and gap_seconds <= parameters.attack_reacquisition_max_gap_s
                        and pair_distance[first_reacquired, pair_index]
                        <= parameters.attack_endpoint_distance_cm
                    ):
                        attack_score = max(
                            attack_score,
                            (first_reacquired - int(contact_frames[0]) + 1) / max(fps, 1e-6),
                        )
        values = {
            "approach": _qualifying_score(
                approach_durations[pair_index], parameters.pair_min_duration_s, fps
            ),
            "attack": attack_score,
            "nose_head_contact": _qualifying_score(
                nose_head_cumulative[pair_index], parameters.contact_min_cumulative_seconds, fps
            ),
            "nose_tail_contact": _qualifying_score(
                nose_tail_cumulative[pair_index], parameters.contact_min_cumulative_seconds, fps
            ),
            "together": _qualifying_score(
                together_durations[pair_index], parameters.together_min_duration_s, fps
            ),
            "chase": _qualifying_score(
                chasing_durations[pair_index],
                (parameters.chase_min_directed_support_s
                 if parameters.confirmed_rules_v1 > 0
                 else parameters.chase_min_duration_s),
                fps,
            ),
            # [SUSPENDED-STEP3 2026-10-04] Following/Avoiding 暂停（用户决策），保留待恢复
            # "following": _qualifying_score(
            #     following_durations[pair_index], parameters.following_min_duration_s, fps
            # ),
        }
        # [STEP6 2026-10-05] Together: recovery 块 mean_speed 条件删除（启用门不再要求 together_mean_max_speed_cm_s），距离条件保留。
        if (
            values["together"] <= 0.0
            and parameters.confirmed_rules_v1 <= 0
            and values["nose_head_contact"] <= 0.0
            and values["nose_tail_contact"] <= 0.0
            and parameters.together_initial_max_distance_cm > 0.0
            and parameters.together_median_max_distance_cm > 0.0
        ):
            # A low-motion pair may pause and move briefly, splitting the
            # framewise Together bout. Summarize only a pair observed near the
            # clip start so an arriving Approach is not promoted to Together.
            observed = pair_valid[:, pair_index] & pair_members_observed[:, pair_index]
            distances = pair_distance[observed, pair_index]
            if (
                len(distances) > 0
                and len(distances) / fps >= parameters.together_min_observed_seconds
                and np.flatnonzero(observed)[0] <= max(1, int(0.3 * len(observed)))
            ):
                beginning = max(1, int(np.ceil(0.2 * len(distances))))
                # [STEP6 2026-10-05] Together: recovery 块 mean_speed 条件删除，距离条件保留。
                # speeds = np.column_stack(
                #     (
                #         np.where(motion_valid[:, left_slot], speed[:, left_slot], np.nan),
                #         np.where(motion_valid[:, right_slot], speed[:, right_slot], np.nan),
                #     )
                # )[observed]
                # mean_speed = _safe_mean(speeds)
                if (
                    np.median(distances[:beginning])
                    <= parameters.together_initial_max_distance_cm
                    and np.median(distances) <= parameters.together_median_max_distance_cm
                    and (
                        parameters.together_max_distance_std_cm <= 0.0
                        or np.std(distances) <= parameters.together_max_distance_std_cm
                    )
                ):
                    values["together"] = min(len(distances) / fps, 1.5)
        if (
            parameters.attack_contact_score_multiplier > 1.0
            and parameters.confirmed_rules_v1 <= 0
            and features.nose_speed_cm_s is not None
            and features.nose_acceleration_cm_s2 is not None
        ):
            directed_contact = (
                features.pair_nose_head_left_to_right_cm,
                features.pair_nose_head_right_to_left_cm,
                features.pair_nose_tail_left_to_right_cm,
                features.pair_nose_tail_right_to_left_cm,
            )
            if all(value is not None for value in directed_contact):
                head_ltr, head_rtl, tail_ltr, tail_rtl = directed_contact
                contacting_sides = (
                    (
                        int(left_slot),
                        pair_valid[:, pair_index]
                        & (
                            (head_ltr[:, pair_index] < parameters.contact_distance_cm)
                            | (tail_ltr[:, pair_index] < parameters.contact_distance_cm)
                        ),
                    ),
                    (
                        int(right_slot),
                        pair_valid[:, pair_index]
                        & (
                            (head_rtl[:, pair_index] < parameters.contact_distance_cm)
                            | (tail_rtl[:, pair_index] < parameters.contact_distance_cm)
                        ),
                    ),
                )
            else:
                # Older feature caches lack direction-specific contact points;
                # retain their pair-level fallback for backward compatibility.
                contacting_sides = (
                    (int(left_slot), contact[:, pair_index]),
                    (int(right_slot), contact[:, pair_index]),
                )

            for contacting_slot, side_contact in contacting_sides:
                if not np.any(side_contact):
                    continue
                after_side_contact = _recent_contact_mask(
                    side_contact,
                    int(np.ceil(parameters.attack_contact_post_window_s * fps)),
                )
                if (
                    _finite_quantile(
                        features.nose_speed_cm_s[after_side_contact, contacting_slot], 0.9
                    )
                    >= parameters.attack_contact_nose_min_speed_cm_s
                    and _finite_quantile(
                        features.nose_acceleration_cm_s2[
                            after_side_contact, contacting_slot
                        ],
                        0.9,
                    )
                    >= parameters.attack_contact_nose_min_acceleration_cm_s2
                    and _finite_quantile(speed[after_side_contact, contacting_slot], 0.9)
                    >= parameters.attack_contact_body_min_speed_cm_s
                ):
                    # Attack may contain only brief contact, unlike standalone
                    # nose-contact labels which require 0.5 s cumulative contact.
                    # Use the contacting mouse's raw contact frames, then demand
                    # strong post-contact nose-point kinematics.
                    contact_seconds = np.count_nonzero(side_contact) / max(fps, 1e-6)
                    contact_score = max(contact_seconds, 1.0 / max(fps, 1e-6))
                    values["attack"] = max(
                        values["attack"],
                        contact_score * parameters.attack_contact_score_multiplier,
                    )
        # [SUSPENDED-STEP3 2026-10-04] Following/Avoiding 暂停（用户决策），保留待恢复
        # near_before = np.flatnonzero(
        #     pair_valid[: max(0, len(pair_distance) - 1), pair_index]
        #     & (
        #         pair_distance[: max(0, len(pair_distance) - 1), pair_index]
        #         <= parameters.approach_terminal_distance_cm
        #     )
        # )
        # if near_before.size:
        #     first_near = int(near_before[0])
        #     fast_escape = (
        #         np.maximum(speed[:, left_slot], speed[:, right_slot])
        #         >= parameters.walking_min_speed_cm_s
        #     )
        #     subsequent_opening = opening[first_near:, pair_index] & fast_escape[first_near:]
        #     avoid_duration = _longest_run(subsequent_opening) / max(fps, 1e-6)
        #     avoid_scores[pair_index] = _qualifying_score(
        #         avoid_duration, parameters.avoiding_min_duration_s, fps
        #     )
        # values["avoidance"] = float(avoid_scores[pair_index])
        _prioritize_attack_over_approach(values, parameters.attack_vs_approach_multiplier)
        # [SUSPENDED-STEP3 2026-10-04] dual-label credit removed: Together now allows contact without interaction
        # for contact_name, contact_mask in (
        #     ("nose_head_contact", qualified_head_contact),
        #     ("nose_tail_contact", qualified_tail_contact),
        # ):
        #     overlap_s = (
        #         np.count_nonzero(contact_mask[:, pair_index] & qualifying_together[:, pair_index])
        #         / max(fps, 1e-6)
        #     )
        #     if (
        #         values["together"] > 0.0
        #         and values[contact_name] > 0.0
        #         and overlap_s + 1e-12 >= parameters.contact_min_cumulative_seconds
        #     ):
        #         pair_compatible_labels.setdefault(key, ["together"]).append(contact_name)
        # if len(pair_compatible_labels.get(key, ())) > 2:
        #     pair_compatible_labels.pop(key)
        if any(score > 0.0 for score in values.values()):
            pair_scores[key] = values

    valid_count = valid.sum(axis=1)
    # [STEP6 2026-10-05] Huddling: 边规则 = (pair_nose_distance_cm ≤ huddle_nose_distance_cm) | (center ≤ huddle_distance_cm)；删除全部速度门；nose 字段缺失时退化为仅中心距离。
    nose_pair = features.pair_nose_distance_cm
    if nose_pair is None:
        nose_pair = np.full_like(pair_distance, np.inf)
    nose_pair = np.where(np.isfinite(nose_pair), nose_pair, np.inf)
    huddle_pairs = (
        pair_valid
        & pair_members_observed
        & (
            (pair_distance <= parameters.huddle_distance_cm)
            | (nose_pair <= parameters.huddle_nose_distance_cm)
        )
    )
    huddle_groups = _group_episode_scores(
        huddle_pairs,
        features.pair_i,
        features.pair_j,
        track_ids,
        fps=fps,
        minimum_seconds=parameters.huddle_min_duration_s,
        max_gap_seconds=parameters.huddle_max_gap_s,
        min_member_support_fraction=parameters.huddle_min_member_support_fraction,
        emit_core_candidates=core_mode,
        event_behavior="huddle",
        event_output=group_events if core_mode else None,
    )
    # Huddling is a resting state, not a moving formation followed by a pause.
    # Inspect spatial connectivity independently of the speed-gated huddle mask
    # so motion at the start of the close-group episode cannot be averaged away.
    track_slot = {track_id: slot for slot, track_id in enumerate(track_ids)}
    for key in tuple(huddle_groups):
        member_slots = tuple(track_slot[int(value)] for value in key.split(","))
        internal_edges = np.isin(features.pair_i, member_slots) & np.isin(
            features.pair_j, member_slots
        )
        # # [STEP6 2026-10-05] 社交/群体参数按文档 v4 + 用户覆盖落地：Huddling  onset 速度门删除（用户决策：扎堆不要求速度条件）。
        episodes = [
            event for event in group_events
            if event["behavior"] == "huddle"
            and ",".join(map(str, event["member_ids"])) == key
        ] if core_mode else []
        qualifying = episodes
        accepted = bool(episodes) if core_mode else True
        if not accepted:
            huddle_groups.pop(key)
        elif core_mode:
            huddle_groups[key] = max(float(event["score"]) for event in qualifying)
            group_events[:] = [
                event for event in group_events
                if event["behavior"] != "huddle"
                or ",".join(map(str, event["member_ids"])) != key
                or event in qualifying
            ]
    if core_mode:
        group_events[:] = [
            event for event in group_events
            if event["behavior"] != "huddle"
            or ",".join(map(str, event["member_ids"])) in huddle_groups
        ]

    isolation_distance = (np.isfinite(features.nearest_distance_cm)
                          & (features.nearest_distance_cm > parameters.isolation_distance_cm))
    isolated = valid & (valid_count >= 2)[:, None] & isolation_distance
    if parameters.confirmed_rules_v1 > 0:
        # Isolation describes the target's spatial relation for the annotated
        # event. Unknown frames do not count either way, but one observed
        # nearest-neighbour distance below the fitted minimum contradicts Isolation
        # for that target in this clip. Individual actions do not change it.
        comparison_observed = (
            valid
            & (valid_count >= 2)[:, None]
            & np.isfinite(features.nearest_distance_cm)
        )
        observed_near = comparison_observed & ~isolation_distance
        target_contradicted = observed_near.any(axis=0)
        isolated[:, target_contradicted] = False
    isolation_unknown = ~valid | (valid_count < 2)[:, None]
    isolation_durations = _isolation_cumulative_durations(
        isolated, isolation_unknown, fps, parameters.isolation_max_gap_s
    )

    cluster_window = max(int(round(parameters.clustering_window_s * fps)), 1)
    cluster_pairs = pair_valid & (pair_distance < parameters.clustering_max_distance_cm)
    # # [STEP6 2026-10-05] 社交/群体参数按文档 v4 + 用户覆盖落地：到达扎堆状态即结束 clustering（逐对以 huddle run 起点截断）。
    huddle_min_frames = max(1, int(round(parameters.huddle_min_duration_s * fps)))
    for pair in range(cluster_pairs.shape[1]):
        hcol = huddle_pairs[:, pair]
        if not np.any(hcol):
            continue
        padded = np.r_[False, hcol, False].astype(np.int8)
        changes = np.diff(padded)
        for start, end in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)):
            if end - start >= huddle_min_frames:
                cluster_pairs[start:, pair] = False
                break
    cluster_drop = np.zeros_like(features.nearest_distance_cm, dtype=float)
    for slot in range(cluster_drop.shape[1]):
        cluster_drop[:, slot] = _rolling_drop(
            np.asarray(features.nearest_distance_cm[:, slot], dtype=float), cluster_window
        )
    initial_neighbor_distance = np.full_like(features.nearest_distance_cm, np.inf)
    if cluster_window < len(features.nearest_distance_cm):
        initial_neighbor_distance[cluster_window:] = features.nearest_distance_cm[:-cluster_window]
    formation_evidence = (
        np.isfinite(initial_neighbor_distance)
        & (initial_neighbor_distance <= parameters.clustering_initial_max_distance_cm)
        & (cluster_drop >= parameters.clustering_min_nearest_neighbor_drop_cm)
        & np.isfinite(group_speed)
        & (group_speed >= parameters.clustering_min_mean_speed_cm_s)
    )
    cluster_groups = _group_episode_scores(
        cluster_pairs,
        features.pair_i,
        features.pair_j,
        track_ids,
        fps=fps,
        minimum_seconds=parameters.clustering_min_duration_s,
        nearest_drop_cm=cluster_drop,
        min_nearest_drop_cm=parameters.clustering_min_nearest_neighbor_drop_cm,
        max_gap_seconds=parameters.clustering_max_gap_s,
        score_multiplier=parameters.clustering_formation_score_bonus,
        formation_evidence=formation_evidence,
        min_formation_member_fraction=parameters.clustering_min_moving_member_fraction,
        motion_speed_cm_s=group_speed,
        minimum_motion_speed_cm_s=parameters.clustering_min_mean_speed_cm_s,
        minimum_moving_member_fraction=parameters.clustering_min_moving_member_fraction,
        min_member_support_fraction=parameters.clustering_min_member_support_fraction,
        emit_core_candidates=core_mode,
        event_behavior="social_clustering",
        event_output=group_events if core_mode else None,
    )
    # Collective travel requires an observed, connected walking triple with
    # aligned directions. Brief within-walk stops use the existing gap policy.
    traveling_pairs = (
        pair_valid
        & (pair_distance < parameters.group_locomotion_max_distance_cm)
        & (features.pair_direction_similarity >= parameters.group_locomotion_min_direction_similarity)
        & locomotion[:, features.pair_i]
        & locomotion[:, features.pair_j]
    )
    locomotion_groups = _group_episode_scores(
        traveling_pairs,
        features.pair_i,
        features.pair_j,
        track_ids,
        fps=fps,
        minimum_seconds=parameters.group_locomotion_min_duration_s,
            max_gap_seconds=parameters.locomotion_stop_gap_s,
            min_member_support_fraction=parameters.group_locomotion_min_moving_fraction,
    )
    dispersal_groups = _dispersal_scores(features, parameters, locomotion)
    if parameters.clustering_motion_max_distance_cm > 0.0:
        # Formation may precede the exported clip. A sustained, spatially
        # connected group with net movement is still a clustering candidate;
        # resting Huddles are rejected by the displacement gate. Membership
        # comes only from observed visual IDs, never from inferred identity.
        moving_events: list[dict[str, Any]] = []
        moving_groups = _group_episode_scores(
            pair_valid & (pair_distance < parameters.clustering_motion_max_distance_cm),
            features.pair_i,
            features.pair_j,
            track_ids,
            fps=fps,
            minimum_seconds=parameters.clustering_min_duration_s,
            max_gap_seconds=parameters.clustering_max_gap_s,
            min_member_support_fraction=0.5,
            emit_core_candidates=core_mode,
            event_behavior="social_clustering" if core_mode else None,
            event_output=moving_events if core_mode else None,
        )
        for key, episode_score in moving_groups.items():
            member_ids = tuple(int(value) for value in key.split(","))
            member_slots = {track_slot[track_id] for track_id in member_ids}
            member_displacements = []
            for track_id in member_ids:
                slot = track_slot[track_id]
                points = features.centers_cm[valid[:, slot], slot]
                if len(points) < 2:
                    break
                member_displacements.append(float(np.linalg.norm(points[-1] - points[0])))
            internal_edges = np.isin(features.pair_i, list(member_slots)) & np.isin(
                features.pair_j, list(member_slots)
            )
            internal_distances = pair_distance[:, internal_edges]
            observed_distances = internal_distances[np.isfinite(internal_distances)]
            loose_group = bool(
                observed_distances.size
                and np.median(observed_distances)
                >= parameters.clustering_motion_loose_pair_distance_cm
            )
            if (
                len(member_displacements) >= 3
                and np.mean(member_displacements)
                >= parameters.clustering_motion_min_displacement_cm
                and (
                    sum(
                        displacement >= parameters.clustering_motion_min_displacement_cm
                        for displacement in member_displacements
                    )
                    >= 2
                    or loose_group
                )
            ):
                cluster_groups[key] = max(cluster_groups.get(key, 0.0), episode_score * 1.25)
                if core_mode:
                    for event in moving_events:
                        if ",".join(map(str, event["member_ids"])) == key:
                            group_events.append(
                                {**event, "score": float(event["score"]) * 1.25}
                            )

    if core_mode and parameters.clustering_formation_rescue_min_geometry_support > 0.0:
        for event in tuple(group_events):
            if event["behavior"] != "huddle":
                continue
            member_ids = tuple(event["member_ids"])
            key = ",".join(map(str, member_ids))
            # Geometry may disambiguate a genuine formation candidate, but it
            # must never manufacture a new ID set from an Huddling candidate.
            if key not in cluster_groups:
                continue
            if (
                parameters.clustering_min_duration_s > 0.0
                and (int(event["end_frame"]) + 1) / fps < parameters.clustering_min_duration_s
            ):
                continue
            geometry_support = _group_formation_geometry_support(
                features, member_ids, start_frame=0, end_frame=int(event["end_frame"])
            )
            if geometry_support < parameters.clustering_formation_rescue_min_geometry_support:
                continue
            if not _directional_walker_in_group(features, member_ids, parameters):
                continue
            score = max(
                cluster_groups.get(key, 0.0),
                float(event["score"]) * (1.0 + geometry_support),
            )
            cluster_groups[key] = score
            group_events.append(
                {
                    "behavior": "social_clustering",
                    "member_ids": member_ids,
                    "start_frame": 0,
                    "end_frame": int(event["end_frame"]),
                    "score": score,
                    "formation_geometry_support": geometry_support,
                }
            )

    if parameters.clustering_directional_walk_min_seconds > 0.0:
        for key in tuple(cluster_groups):
            member_ids = tuple(int(value) for value in key.split(","))
            qualifying_events = [
                event for event in group_events
                if event["behavior"] == "social_clustering"
                and event["member_ids"] == member_ids
            ]
            if core_mode and qualifying_events:
                directional = any(
                    _directional_walker_in_group(
                        features,
                        member_ids,
                        parameters,
                        start_frame=event["start_frame"],
                        end_frame=event["end_frame"],
                    )
                    for event in qualifying_events
                )
            else:
                directional = _directional_walker_in_group(features, member_ids, parameters)
            if not directional:
                cluster_groups.pop(key)
    # [STEP6 2026-10-05] Social clustering: episode 截断后处理——同成员集合的 huddle episode 开始时，
    # clustering episode 的结束帧截到 huddle 开始帧，并按帧跨度等比缩减分数；huddle 开始不晚于
    # clustering 起点时该 clustering episode 整体被 huddle 取代而移除。与上方逐对边掩码预截断互补；
    # 仅 core 模式产出带帧区间的事件，非 core 模式的分数已由预截断约束。
    if core_mode:
        huddle_onset_by_members: dict[tuple[int, ...], int] = {}
        for event in group_events:
            if event.get("behavior") != "huddle":
                continue
            member_key = tuple(int(value) for value in event.get("member_ids", ()))
            onset = int(event.get("start_frame", 0))
            previous_onset = huddle_onset_by_members.get(member_key)
            if previous_onset is None or onset < previous_onset:
                huddle_onset_by_members[member_key] = onset
        kept_events: list[dict[str, Any]] = []
        truncated_cluster_scores: dict[str, float] = {}
        for event in group_events:
            if event.get("behavior") != "social_clustering":
                kept_events.append(event)
                continue
            member_key = tuple(int(value) for value in event.get("member_ids", ()))
            event_key = ",".join(map(str, member_key))
            onset = huddle_onset_by_members.get(member_key)
            start_frame = int(event.get("start_frame", 0))
            end_frame = int(event.get("end_frame", start_frame))
            if onset is not None and onset <= end_frame:
                if onset <= start_frame:
                    continue
                original_span = max(end_frame - start_frame + 1, 1)
                truncated_span = max(onset - start_frame + 1, 1)
                event = {
                    **event,
                    "end_frame": onset,
                    "score": float(event.get("score", 0.0)) * truncated_span / original_span,
                }
            kept_events.append(event)
            if event_key in cluster_groups:
                truncated_cluster_scores[event_key] = max(
                    truncated_cluster_scores.get(event_key, 0.0),
                    float(event.get("score", 0.0)),
                )
        group_events[:] = kept_events
        for event_key in tuple(cluster_groups):
            if event_key in truncated_cluster_scores:
                cluster_groups[event_key] = truncated_cluster_scores[event_key]
            else:
                cluster_groups.pop(event_key)
    if core_mode:
        group_events[:] = [
            event for event in group_events
            if event["behavior"] != "social_clustering"
            or ",".join(map(str, event["member_ids"])) in cluster_groups
        ]

    # Keep the legacy group membership snapshot for Isolation's historical
    # suppression rule. The feature refactor changes only Huddling and Social
    # clustering evidence and cannot alter any other behavior's rule path.
    legacy_collective_member_ids = {
        int(track_id)
        for key in (*huddle_groups, *cluster_groups)
        for track_id in key.split(",")
    }
    collective_debug: list[dict[str, Any]] = []
    collective_track_quality: list[dict[str, Any]] = []
    if parameters.collective_feature_rules_enabled > 0.0:
        group_events[:] = [
            event
            for event in group_events
            if event.get("behavior") not in {"huddle", "social_clustering"}
        ]
        huddle_groups, cluster_groups, collective_debug, collective_track_quality = (
            _replace_collective_group_scores_with_features(
                features,
                parameters,
                huddle_groups=huddle_groups,
                cluster_groups=cluster_groups,
                group_events=group_events,
            )
        )

    if parameters.confirmed_rules_v1 > 0:
        # The public current profile must execute the same group rules as the
        # direct audit/scorer. Historical legacy masks above cannot override it.
        from .group_event_rules import group_rule_candidates
        current = group_rule_candidates(features, parameters, ())
        grouped = {name: {} for name in ("huddle", "social_clustering", "group_locomotion", "dispersal")}
        group_events = []
        for candidate in current:
            key = ",".join(map(str, candidate.target_ids))
            grouped[candidate.behavior][key] = max(
                grouped[candidate.behavior].get(key, 0.0), float(candidate.evidence))
            group_events.append({"behavior": candidate.behavior,
                                 "member_ids": list(candidate.target_ids),
                                 "score": float(candidate.evidence),
                                 "evidence_seconds": float(candidate.evidence),
                                 "temporal_support": "clip_local_half_open_event_frame_intervals",
                                 "frame_intervals": candidate.frame_intervals,
                                 "support_frame_intervals": candidate.support_frame_intervals})
        huddle_groups = grouped["huddle"]
        cluster_groups = grouped["social_clustering"]
        locomotion_groups = grouped["group_locomotion"]
        dispersal_groups = grouped["dispersal"]
        for values in identity_scores.values():
            for name in grouped:
                values[name] = 0.0

    # The new movement-phase group rules may occur before/after a sustained
    # isolation interval in one clip. They must not erase the Isolation score.
    for slot, track_id in enumerate(track_ids):
        values = identity_scores[str(track_id)]
        minimum_isolation_seconds = max(parameters.isolation_min_duration_s, 10.0)
        values["isolation"] = _qualifying_score(
            isolation_durations[slot], minimum_isolation_seconds, fps
        )
        if parameters.confirmed_rules_v1 > 0 and isolation_durations[slot] <= minimum_isolation_seconds:
            values["isolation"] = 0.0
        if parameters.confirmed_rules_v1 <= 0 and track_id in legacy_collective_member_ids:
            values["isolation"] = 0.0
    group_scores: dict[str, dict[str, float]] = {}
    for slot, track_id in enumerate(track_ids):
        isolation_score = identity_scores[str(track_id)]["isolation"]
        if isolation_score > 0.0:
            group_scores[str(track_id)] = {"isolation": isolation_score}
    for behavior, episodes in (
        ("huddle", huddle_groups),
        ("social_clustering", cluster_groups),
        ("group_locomotion", locomotion_groups),
        ("dispersal", dispersal_groups),
    ):
        for key, score in episodes.items():
            group_scores.setdefault(key, {})[behavior] = score
            for track_id in key.split(","):
                values = identity_scores[track_id]
                values[behavior] = max(values[behavior], score)

    # Keep taxonomy slots for historical readers; suspended posture scores stay
    # zero and cannot become detected labels before manual review/calibration.
    scores: dict[str, float] = {
        behavior: 0.0 for behavior in CANONICAL_BEHAVIORS
    }
    for values in identity_scores.values():
        for behavior, score in values.items():
            scores[behavior] = max(scores[behavior], float(score))
    for values in pair_scores.values():
        for behavior, score in values.items():
            scores[behavior] = max(scores[behavior], float(score))
    detected = sorted(behavior for behavior, score in scores.items() if score > 0.0)

    return {
        "suspended_posture_behaviors": sorted(SUSPENDED_POSTURE_BEHAVIORS),
        "detected": detected,
        "scores": {key: float(value) for key, value in scores.items()},
        "track_ids": track_ids,
        "identity_scores": identity_scores,
        "pair_scores": pair_scores,
        # [SUSPENDED-STEP3 2026-10-04] dual-label credit removed: Together now allows contact without interaction
        # "pair_compatible_labels": pair_compatible_labels,
        "group_scores": group_scores,
        "group_events": group_events,
        "collective_feature_debug": collective_debug,
        "collective_track_quality": collective_track_quality,
        "track_selection_truncated": features.track_selection_truncated,
        "selected_track_count": features.selected_track_count,
    }
