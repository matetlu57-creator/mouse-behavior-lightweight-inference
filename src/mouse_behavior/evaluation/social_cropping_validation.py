"""Quantitative validation and audit helpers for modular social-behavior cropping.

Evaluates all six pair/social canonical behaviors across full-length recordings:
- Approach (``approach``)
- Attack (``attack``)
- Chasing (``chase``)
- Snout-head contact (``nose_head_contact``)
- Snout-rear contact (``nose_tail_contact``)
- Together (``together``)
"""
from __future__ import annotations

import copy
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from mouse_behavior.behavior.event_boundaries import EventBoundary
from mouse_behavior.behavior.segment_clipping import (
    refine_behavior_segments,
    select_behavior_segments,
)
from mouse_behavior.behavior.segment_clipping_social_attack_chasing import (
    _build_full_store_track_index,
)
from mouse_behavior.behavior.segment_clipping_social import (
    discover_fast_chasing_candidates,
    discover_fight_ball_attack_episodes,
    discover_id_switch_chasing_candidates,
)
from mouse_behavior.behavior.segment_clipping_common import (
    _observed_support_intervals,
    endpoints_fully_observed,
    event_participant_ids,
    min_clip_duration_policy_for_behavior,
    min_participants_for_behavior,
    normalize_behavior_key,
    qualify_clip_duration,
)
from mouse_behavior.evaluation.exported_dataset_models import HeuristicParameters
from mouse_behavior.evaluation.group_cropping_validation import (
    CachedTrackStore,
    _count_shared_mouse_overlaps,
    _interval_iou,
    write_pose_cache_source_receipt,
)

SOCIAL_BEHAVIOR_KEYS: tuple[str, ...] = (
    "approach",
    "attack",
    "chase",
    "nose_head_contact",
    "nose_tail_contact",
    "together",
)

SOCIAL_BEHAVIOR_LABELS: dict[str, str] = {
    "approach": "接近 (Approach)",
    "attack": "攻击 (Attack)",
    "chase": "追逐 (Chasing)",
    "nose_head_contact": "鼻吻/头接触 (Snout-head contact)",
    "nose_tail_contact": "尾嗅/尾接触 (Snout-rear contact)",
    "together": "伴随共处 (Together)",
}

__all__ = [
    "CachedTrackStore",
    "SOCIAL_BEHAVIOR_KEYS",
    "SOCIAL_BEHAVIOR_LABELS",
    "audit_social_segment",
    "refine_all_social_events",
    "select_social_review_clips",
    "summarize_social_cropping_validation",
    "write_pose_cache_source_receipt",
]


def _normalize_social_canonical(beh: str) -> str:
    key = normalize_behavior_key(beh)
    return {
        "chasing": "chase",
        "head_contact": "nose_head_contact",
        "tail_contact": "nose_tail_contact",
    }.get(key, key)


def refine_all_social_events(
    pre_events: Sequence[Mapping[str, Any]],
    store: Any,
    binding: Mapping[str, Any],
    scale: Mapping[str, Any],
    parameters: HeuristicParameters,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run modular social refinement, endpoint observation accounting, and overlap selection."""
    fps = float(binding["source_video"]["fps"])
    total = int(binding["source_video"]["frame_count"])
    social_inputs = [
        copy.deepcopy(dict(ev))
        for ev in pre_events
        if _normalize_social_canonical(str(ev.get("canonical_behavior") or ev.get("behavior") or ""))
        in SOCIAL_BEHAVIOR_KEYS
    ]

    refined, refinement_audit = refine_behavior_segments(
        social_inputs,
        store,
        binding,
        scale=scale,
        parameters=parameters,
        enable_huddle_sliding=False,
        enable_approach_requalification=False,
        enable_fine_cropping=True,
    )

    shared_index = _build_full_store_track_index(store, binding, scale, max_speed_cm_s=240.0)
    discovered_attacks = discover_fight_ball_attack_episodes(
        store,
        binding,
        scale=scale,
        parameters=parameters,
        existing_events=refined,
        _index=shared_index,
    )
    existing_attack_spans: list[tuple[int, int]] = [
        (int(ev["start_frame"]), int(ev["end_frame"]))
        for ev in refined
        if _normalize_social_canonical(str(ev.get("canonical_behavior") or ev.get("behavior") or "")) == "attack"
    ]
    added_attacks = 0
    for d_ev in discovered_attacks:
        ds, de = int(d_ev["start_frame"]), int(d_ev["end_frame"])
        if any(not (de <= s0 or ds >= e0) for s0, e0 in existing_attack_spans):
            continue
        existing_attack_spans.append((ds, de))
        refined.append(d_ev)
        added_attacks += 1
    refinement_audit["discovered_attack_count"] = added_attacks

    discovered_chases = discover_fast_chasing_candidates(
        store,
        binding,
        scale=scale,
        parameters=parameters,
        existing_events=refined,
    )
    existing_chase_spans: list[tuple[frozenset[int], int, int]] = [
        (
            frozenset(int(m) for m in ev.get("mouse_ids", ())[:2]),
            int(ev["start_frame"]),
            int(ev["end_frame"]),
        )
        for ev in refined
        if _normalize_social_canonical(str(ev.get("canonical_behavior") or ev.get("behavior") or "")) == "chase"
        and len(ev.get("mouse_ids", ())) >= 2
    ]
    added_discovered = 0
    for d_ev in discovered_chases:
        mids_pair = frozenset(int(m) for m in d_ev.get("mouse_ids", ())[:2])
        ds, de = int(d_ev["start_frame"]), int(d_ev["end_frame"])
        if any(p == mids_pair and not (de <= s0 or ds >= e0) for p, s0, e0 in existing_chase_spans):
            continue
        existing_chase_spans.append((mids_pair, ds, de))
        refined.append(d_ev)
        added_discovered += 1
    refinement_audit["discovered_chasing_count"] = added_discovered

    discovered_idsw_chases = discover_id_switch_chasing_candidates(
        store,
        binding,
        scale=scale,
        parameters=parameters,
        existing_events=refined,
        _index=shared_index,
    )
    added_idsw_chases = 0
    for d_ev in discovered_idsw_chases:
        ds, de = int(d_ev["start_frame"]), int(d_ev["end_frame"])
        if any(not (de <= s0 or ds >= e0) for s0, e0 in existing_attack_spans):
            continue
        if any(not (de <= s0 or ds >= e0) for _, s0, e0 in existing_chase_spans):
            continue
        existing_chase_spans.append((frozenset(int(m) for m in d_ev.get("mouse_ids", ())[:2]), ds, de))
        refined.append(d_ev)
        added_idsw_chases += 1
    refinement_audit["discovered_id_switch_chasing_count"] = added_idsw_chases

    for event in refined:
        s_ev, e_ev = int(event["start_frame"]), int(event["end_frame"])
        mids_ev = {int(m) for m in event.get("mouse_ids", ())}
        is_idsw = bool(event.get("has_visual_id_switch"))
        obs_mask = np.zeros(max(0, e_ev - s_ev), dtype=bool)
        for row in store.rows(s_ev, e_ev):
            fn = int(row.get("source_frame", row.get("frame", -1)))
            if s_ev <= fn < e_ev:
                present = {int(d["track_id"]) for d in row.get("detections", ()) if "track_id" in d}
                obs_mask[fn - s_ev] = bool(mids_ev & present) if is_idsw else mids_ev.issubset(present)
        obs_intervals = _observed_support_intervals(obs_mask, s_ev)
        obs_count = int(obs_mask.sum())
        event["observed_all_participants_frame_intervals"] = obs_intervals
        event["observed_all_participants_frame_count"] = obs_count
        event["inferred_or_partially_unobserved_frame_count"] = max(0, (e_ev - s_ev) - obs_count)
        if not event.get("canonical_support_frame_intervals"):
            event["canonical_support_frame_intervals"] = obs_intervals
            event["canonical_support_frame_count"] = obs_count
        event["boundary"] = EventBoundary(s_ev, e_ev, s_ev, e_ev, fps).as_dict()

        contexts = event.get("inference_parent_contexts", [])
        if contexts and "process_boundary_status" not in event:
            event["process_boundary_status"] = {
                "start_closed_in_context": any(c["context_start"] < s_ev < c["context_end"] for c in contexts),
                "end_closed_in_context": any(c["context_start"] < e_ev < c["context_end"] for c in contexts),
                "starts_at_source_boundary": s_ev == 0,
                "ends_at_source_boundary": e_ev == total,
                "minimum_duration_used_as_crop_length": False,
                "meaning": "Modular fine-cropped social segment boundaries",
            }

    selected, selection_audit = select_behavior_segments(
        refined,
        store=store,
        binding=binding,
        max_per_behavior=None,
        require_closed_boundaries=False,
        require_endpoint_visibility=True,
        strict_non_overlapping_shared_mouse=True,
    )
    return selected, {"refinement_audit": refinement_audit, "selection_audit": selection_audit}


def audit_social_segment(
    event: Mapping[str, Any],
    store: Any,
    binding: Mapping[str, Any] | None,
    fps: float,
    parameters: HeuristicParameters,
) -> dict[str, Any]:
    """Audit participant count, strict minimum duration, observed endpoints, and completeness."""
    beh_raw = str(event.get("canonical_behavior") or event.get("behavior") or "")
    beh = _normalize_social_canonical(beh_raw)
    start, end = int(event["start_frame"]), int(event["end_frame"])
    span = max(1, end - start)
    duration_s = span / fps

    unique_mids = event_participant_ids(event)
    min_req_participants = min_participants_for_behavior(beh)
    min_participants_ok = len(unique_mids) >= min_req_participants

    eff_req_s, eff_comp = min_clip_duration_policy_for_behavior(beh, parameters=parameters)
    dur_diag = qualify_clip_duration(event, fps, parameters=parameters)
    if dur_diag.get("required_s") is not None:
        eff_req_s = float(dur_diag["required_s"])
        eff_comp = str(dur_diag["comparator"])
    duration_ok = bool(dur_diag["ok"])
    endpoint_obs = endpoints_fully_observed(store, event, binding)

    obs_cnt = int(event.get("observed_all_participants_frame_count", 0))
    supp_cnt = int(event.get("canonical_support_frame_count", obs_cnt))
    obs_ratio = obs_cnt / span
    supp_ratio = supp_cnt / span
    is_idsw = bool(event.get("has_visual_id_switch"))
    if beh == "attack":
        min_obs_ratio = 0.50 if is_idsw else 0.55
    elif beh == "chase":
        min_obs_ratio = 0.55 if is_idsw else 0.65
    else:
        min_obs_ratio = 0.70
    continuity_ok = obs_ratio >= min_obs_ratio

    kin_ok = True
    extra_metrics: dict[str, Any] = {}
    notes: list[str] = []
    if beh == "approach":
        box_gap = event.get("min_pair_box_gap_cm")
        end_box_gap = event.get("approach_end_box_gap_cm")
        other_dist = event.get("min_other_mouse_dist_cm")
        d_start = event.get("approach_start_dist_cm")
        d_end = event.get("approach_end_dist_cm")
        closing = event.get("approach_closing_cm")
        actor_sp = event.get("approach_actor_speed_cm_s")
        target_sp = event.get("approach_target_speed_cm_s")
        toward_cos = event.get("approach_toward_cos")
        end_proj = event.get("approach_end_proj_toward_cm_s")
        end_dom = event.get("approach_end_actor_dominance_cm_s")
        end_tcos = event.get("approach_end_toward_cos")
        end_hcos = event.get("approach_end_heading_cos")
        end_baxis = event.get("approach_end_body_axis_cos")
        end_cl = event.get("approach_end_closing_rate_cm_s")
        end_adv = event.get("approach_end_snout_tail_adv_cm")
        extra_metrics["approach_start_dist_cm"] = d_start
        extra_metrics["approach_end_dist_cm"] = d_end
        extra_metrics["approach_closing_cm"] = closing
        extra_metrics["approach_end_box_gap_cm"] = end_box_gap
        extra_metrics["approach_pre_touch_min_box_gap_cm"] = event.get("approach_pre_touch_min_box_gap_cm")
        extra_metrics["min_pair_box_gap_cm"] = box_gap
        extra_metrics["min_other_mouse_dist_cm"] = other_dist
        extra_metrics["approach_actor_speed_cm_s"] = actor_sp
        extra_metrics["approach_target_speed_cm_s"] = target_sp
        extra_metrics["approach_toward_cos"] = toward_cos
        extra_metrics["approach_end_proj_toward_cm_s"] = end_proj
        extra_metrics["approach_end_actor_dominance_cm_s"] = end_dom
        extra_metrics["approach_end_toward_cos"] = end_tcos
        extra_metrics["approach_end_heading_cos"] = end_hcos
        extra_metrics["approach_end_body_axis_cos"] = end_baxis
        extra_metrics["approach_end_closing_rate_cm_s"] = end_cl
        extra_metrics["approach_end_snout_tail_adv_cm"] = end_adv
        if box_gap is not None and float(box_gap) <= 0.0:
            kin_ok = False
            notes.append(f"接近过程接框前发生重叠接触(pre_gap={float(box_gap):.2f}cm)")
        if end_box_gap is not None and float(end_box_gap) > 0.15 + 1e-9:
            kin_ok = False
            notes.append(f"接近终止帧未贴边接触(end_gap={float(end_box_gap):.2f}cm>0.15cm)")
        if other_dist is not None and float(other_dist) < 10.0:
            kin_ok = False
            notes.append(f"接近过程处于他鼠拥挤区({float(other_dist):.2f}cm<10.0cm)")
        if d_start is not None and float(d_start) < 20.0:
            kin_ok = False
            notes.append(f"接近起点距离偏近({float(d_start):.1f}cm<20.0cm)")
        if closing is not None and float(closing) < 10.0:
            kin_ok = False
            notes.append(f"接近缩短幅度不足({float(closing):.1f}cm<10.0cm)")
        if actor_sp is not None and float(actor_sp) < 8.0:
            kin_ok = False
            notes.append(f"主动接近方速度偏低({float(actor_sp):.1f}cm/s<8.0cm/s)")
        if toward_cos is not None and float(toward_cos) < 0.40:
            kin_ok = False
            notes.append(f"接近朝向投影不足({float(toward_cos):.2f}<0.40)")
        if (
            (end_proj is not None and float(end_proj) < 7.0)
            or (end_tcos is not None and float(end_tcos) < 0.82)
            or (end_cl is not None and float(end_cl) < 7.0)
            or (end_dom is not None and float(end_dom) < 3.0)
            or (end_hcos is not None and float(end_hcos) < 0.82)
            or (end_baxis is not None and float(end_baxis) < 0.72)
            or (end_adv is not None and float(end_adv) < 3.0)
        ):
            kin_ok = False
            notes.append(
                f"末端接近指向性不足(proj={end_proj}, cos={end_tcos}, hcos={end_hcos}, dom={end_dom}, cl={end_cl})"
            )
    elif beh == "attack":
        pk = float(event.get("attack_peak_speed_cm_s", 0.0))
        bpk = float(event.get("attack_both_peak_speed_cm_s", 0.0))
        close_mn = float(event.get("attack_close_mean_speed_cm_s", 0.0))
        max_iou = float(event.get("attack_max_iou", 0.0))
        m1_in = int(event.get("attack_clinch_occlusion_frames", 0))
        m1_adj = int(event.get("attack_adjacent_occlusion_frames", 0))
        eng = int(event.get("attack_engaged_frames", 0))
        mid_path = float(event.get("attack_mid_path_cm", 0.0))
        crowd_frac = float(event.get("attack_crowd_fraction", 0.0))
        md = float(event.get("attack_min_dist_cm", 99.0))
        r1_max = float(event.get("attack_max_raw1f_speed_cm_s", 0.0))
        max_ar = float(event.get("attack_max_area_ratio", 0.0))
        n_huge = int(event.get("attack_huge_box_frames", 0))
        id_churn = int(event.get("attack_id_churn", 0))
        stat_by = int(event.get("attack_stationary_bystanders", 0))
        extra_metrics["has_visual_id_switch"] = is_idsw
        extra_metrics["attack_max_iou"] = max_iou
        extra_metrics["attack_max_area_ratio"] = max_ar
        extra_metrics["attack_huge_box_frames"] = n_huge
        extra_metrics["attack_id_churn"] = id_churn
        extra_metrics["attack_stationary_bystanders"] = stat_by
        extra_metrics["attack_tumble_path_cm"] = event.get("attack_tumble_path_cm", mid_path)
        extra_metrics["attack_clinch_occlusion_frames"] = m1_in
        extra_metrics["attack_adjacent_occlusion_frames"] = m1_adj
        extra_metrics["attack_engaged_frames"] = eng
        extra_metrics["attack_close_frames"] = event.get("attack_close_frames")
        extra_metrics["attack_peak_speed_cm_s"] = pk
        extra_metrics["attack_both_peak_speed_cm_s"] = bpk
        extra_metrics["attack_mean_speed_cm_s"] = event.get("attack_mean_speed_cm_s")
        extra_metrics["attack_close_mean_speed_cm_s"] = close_mn
        extra_metrics["attack_mid_path_cm"] = mid_path
        extra_metrics["attack_crowd_fraction"] = crowd_frac
        extra_metrics["attack_min_dist_cm"] = md
        extra_metrics["attack_min_box_gap_cm"] = event.get("attack_min_box_gap_cm")
        extra_metrics["attack_max_raw1f_speed_cm_s"] = r1_max
        extra_metrics["attack_active_frames"] = event.get("attack_active_frames")
        if is_idsw:
            if (
                max_ar < 2.15
                or n_huge < 4
                or pk < 48.0
                or id_churn < 4
                or stat_by > 1
                or crowd_frac > 0.25 + 1e-9
            ):
                kin_ok = False
                notes.append(
                    f"纠缠滚打攻击校验未达标(area={max_ar:.2f}x, nH={n_huge}, pk={pk:.1f}, churn={id_churn}, stBy={stat_by})"
                )
        else:
            occ_ok = m1_in >= 3
            if (
                max_iou < 0.42
                or pk < 35.0
                or bpk < 22.0
                or close_mn < 22.0
                or eng < 14
                or mid_path < 11.0
                or crowd_frac > 0.25 + 1e-9
                or r1_max > 125.0
                or not occ_ok
            ):
                kin_ok = False
                notes.append(
                    f"攻击缠斗/重叠校验未达标(iou={max_iou:.2f}, m1in={m1_in}, eng={eng}, pk={pk:.1f}, close_mn={close_mn:.1f}, crowd={crowd_frac:.2f})"
                )
    elif beh == "chase":
        spf = float(event.get("chase_mean_follower_speed_cm_s", 0.0))
        spl = float(event.get("chase_mean_leader_speed_cm_s", 0.0))
        p25_sp = event.get("chase_both_p25_speed_cm_s")
        fast_frac = event.get("chase_both_fast_fraction")
        pkf = float(event.get("chase_peak_follower_speed_cm_s", 0.0))
        pkl = float(event.get("chase_peak_leader_speed_cm_s", 0.0))
        med_d = float(event.get("chase_median_dist_cm", 99.0))
        max_d = float(event.get("chase_max_dist_cm", 99.0))
        gap_widen = float(event.get("chase_gap_widening_cm", 0.0))
        trail_err = float(event.get("chase_trail_error_cm", 99.0))
        inter_frac = float(event.get("chase_intervening_fraction", 0.0))
        trail_byst = float(event.get("chase_trailing_bystander_fraction", 0.0))
        vcos = float(event.get("chase_mean_vel_cos", 0.0))
        p25_vcos = event.get("chase_vel_cos_p25")
        vel_al_frac = event.get("chase_vel_aligned_fraction")
        mcos = float(event.get("chase_mean_cos", 0.0))
        p25_pcos = event.get("chase_pursuit_cos_p25")
        body_cos = event.get("chase_body_heading_cos")
        min_net = float(event.get("chase_min_net_disp_cm", 0.0))
        extra_metrics["has_visual_id_switch"] = is_idsw
        extra_metrics["chase_mean_follower_speed_cm_s"] = spf
        extra_metrics["chase_mean_leader_speed_cm_s"] = spl
        extra_metrics["chase_both_p25_speed_cm_s"] = p25_sp
        extra_metrics["chase_both_fast_fraction"] = fast_frac
        extra_metrics["chase_peak_follower_speed_cm_s"] = pkf
        extra_metrics["chase_peak_leader_speed_cm_s"] = pkl
        extra_metrics["chase_median_dist_cm"] = med_d
        extra_metrics["chase_max_dist_cm"] = max_d
        extra_metrics["chase_gap_widening_cm"] = gap_widen
        extra_metrics["chase_trail_error_cm"] = trail_err
        extra_metrics["chase_intervening_fraction"] = inter_frac
        extra_metrics["chase_trailing_bystander_fraction"] = trail_byst
        extra_metrics["chase_mean_vel_cos"] = vcos
        extra_metrics["chase_vel_cos_p25"] = p25_vcos
        extra_metrics["chase_vel_aligned_fraction"] = vel_al_frac
        extra_metrics["chase_mean_cos"] = mcos
        extra_metrics["chase_pursuit_cos_p25"] = p25_pcos
        extra_metrics["chase_body_heading_cos"] = body_cos
        extra_metrics["chase_min_net_disp_cm"] = min_net
        sustained_bad = (
            (p25_sp is not None and float(p25_sp) < 28.0)
            or (fast_frac is not None and float(fast_frac) < 0.70 - 1e-9)
            or (p25_vcos is not None and float(p25_vcos) < 0.78)
            or (vel_al_frac is not None and float(vel_al_frac) < 0.80 - 1e-9)
            or (p25_pcos is not None and float(p25_pcos) < 0.65)
            or (body_cos is not None and float(body_cos) < 0.75)
        )
        if is_idsw:
            if (
                min(spf, spl) < 36.0
                or min(pkf, pkl) < 43.5
                or med_d > 14.5
                or gap_widen > 4.0 + 1e-9
                or min_net < 25.0
                or vcos < 0.85
                or mcos < 0.78
                or sustained_bad
                or trail_byst > 0.35 + 1e-9
            ):
                kin_ok = False
                notes.append(
                    f"ID切换高速同向追逐校验未达标(追={spf:.1f}, 逃={spl:.1f}, p25={p25_sp}, vcos={vcos:.3f}, mcos={mcos:.3f}, net={min_net:.1f})"
                )
        elif (
            spf < 36.0
            or spl < 36.0
            or min(pkf, pkl) < 43.5
            or med_d > 14.5
            or max_d > 20.0
            or gap_widen > 4.0 + 1e-9
            or trail_err > 2.6 + 1e-9
            or inter_frac > 0.12 + 1e-9
            or trail_byst > 0.35 + 1e-9
            or vcos < 0.85
            or mcos < 0.78
            or min_net < 22.0
            or sustained_bad
        ):
            kin_ok = False
            notes.append(
                f"追逐高速/方向一致校验未达标(追={spf:.1f}, 逃={spl:.1f}, p25={p25_sp}, vcos={vcos:.3f}, pcos={mcos:.3f}, body={body_cos}, trail={trail_err:.2f})"
            )

    aid = event.get("actor_id", -1)
    tid = event.get("target_id", -1)
    roles_obj = event.get("roles")
    role_status = str(event.get("role_status", ""))
    if beh == "together":
        role_ok = bool(
            role_status == "undirected_not_applicable"
            and (aid is None or int(aid) == -1)
            and (tid is None or int(tid) == -1)
            and roles_obj is None
        )
        role_summary_zh = "无方向（共同伴随）"
        if not role_ok:
            notes.append("Together 应标记为无方向行为(actor_id=-1,target_id=-1,roles=null)")
    else:
        role_ok = bool(
            role_status == "measured_directed_role"
            and aid is not None
            and not isinstance(aid, bool)
            and int(aid) >= 0
            and tid is not None
            and not isinstance(tid, bool)
            and int(tid) >= 0
            and int(aid) != int(tid)
            and isinstance(roles_obj, Mapping)
            and bool(roles_obj.get("summary_zh"))
        )
        role_summary_zh = str(roles_obj.get("summary_zh", "")) if isinstance(roles_obj, Mapping) else "缺失角色信息"
        if not role_ok:
            notes.append("缺失有向社交行为角色归因")

    full_ok = bool(min_participants_ok and duration_ok and endpoint_obs and continuity_ok and kin_ok and role_ok)

    if not min_participants_ok:
        notes.append(f"参与目标数不足({len(unique_mids)}<{min_req_participants})")
    if not duration_ok:
        notes.append(f"未达最低时长要求({duration_s:.3f}s vs {eff_comp}{eff_req_s:.2f}s)")
    if not endpoint_obs:
        notes.append("起止端点存在目标缺框")
    if not continuity_ok:
        notes.append(f"双鼠共现连续率偏低({obs_ratio:.1%})")
    if not notes:
        notes.append(
            f"满足最低时长({eff_comp}{eff_req_s:.2f}s)、端点双鼠完整可见、角色归因({role_summary_zh})及高质量运动/场景条件"
        )

    return {
        "event_id": str(event.get("event_id", "")),
        "behavior": beh,
        "duration_s": round(duration_s, 3),
        "required_duration_s": round(eff_req_s, 3),
        "duration_comparator": eff_comp,
        "participant_count": len(unique_mids),
        "min_required_participants": min_req_participants,
        "min_participants_ok": min_participants_ok,
        "duration_ok": duration_ok,
        "endpoint_observed_ok": endpoint_obs,
        "observed_all_ratio": round(obs_ratio, 4),
        "support_ratio": round(supp_ratio, 4),
        "continuity_ok": continuity_ok,
        "role_ok": role_ok,
        "role_status": role_status,
        "role_summary_zh": role_summary_zh,
        **extra_metrics,
        "full_crop_precision_ok": full_ok,
        "audit_notes": notes,
    }


def summarize_social_cropping_validation(
    pre_events: Sequence[Mapping[str, Any]],
    post_events: Sequence[Mapping[str, Any]],
    store: Any,
    binding: Mapping[str, Any],
    parameters: HeuristicParameters,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Compare raw rule social candidates vs modular fine-cropped social segments."""
    fps = float(binding["source_video"]["fps"])
    pre_by_id: dict[str, dict[str, Any]] = {}
    pre_by_beh: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for e in pre_events:
        beh = _normalize_social_canonical(str(e.get("canonical_behavior") or e.get("behavior") or ""))
        if beh in SOCIAL_BEHAVIOR_KEYS:
            item = dict(e)
            pre_by_beh[beh].append(item)
            if "event_id" in item:
                pre_by_id[str(item["event_id"])] = item

    post_by_beh: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for e in post_events:
        beh = _normalize_social_canonical(str(e.get("canonical_behavior") or e.get("behavior") or ""))
        if beh in SOCIAL_BEHAVIOR_KEYS:
            post_by_beh[beh].append(dict(e))

    all_overlaps, overlaps_by_beh = _count_shared_mouse_overlaps(post_events)
    per_event_audit: dict[str, dict[str, Any]] = {}
    behavior_summaries: dict[str, dict[str, Any]] = {}

    for beh in SOCIAL_BEHAVIOR_KEYS:
        raw_list = pre_by_beh.get(beh, [])
        crop_list = post_by_beh.get(beh, [])

        parent_to_children: dict[str, list[dict[str, Any]]] = defaultdict(list)
        start_shifts_s: list[float] = []
        end_trims_s: list[float] = []
        changed_count = 0
        unchanged_count = 0
        extended_count = 0
        trimmed_count = 0

        for ev in crop_list:
            eid = str(ev["event_id"])
            kin = audit_social_segment(ev, store, binding, fps, parameters)
            pid = str(ev.get("parent_candidate_event_id", ""))
            parent = pre_by_id.get(pid)
            if parent is None:
                mids_t = tuple(sorted(event_participant_ids(ev)))
                overlapping = [
                    r
                    for r in raw_list
                    if tuple(sorted(event_participant_ids(r))) == mids_t
                    and max(int(r["start_frame"]), int(ev["start_frame"]))
                    < min(int(r["end_frame"]), int(ev["end_frame"]))
                ]
                if overlapping:
                    parent = max(
                        overlapping,
                        key=lambda r: _interval_iou(
                            int(r["start_frame"]),
                            int(r["end_frame"]),
                            int(ev["start_frame"]),
                            int(ev["end_frame"]),
                        ),
                    )
            if parent is not None:
                parent_to_children[str(parent["event_id"])].append(ev)
                ds = (int(ev["start_frame"]) - int(parent["start_frame"])) / fps
                de = (int(parent["end_frame"]) - int(ev["end_frame"])) / fps
                start_shifts_s.append(ds)
                end_trims_s.append(de)
                changed = (int(ev["start_frame"]) != int(parent["start_frame"])) or (
                    int(ev["end_frame"]) != int(parent["end_frame"])
                )
                raw_len = int(parent["end_frame"]) - int(parent["start_frame"])
                new_len = int(ev["end_frame"]) - int(ev["start_frame"])
                if changed:
                    changed_count += 1
                    if new_len > raw_len:
                        extended_count += 1
                    else:
                        trimmed_count += 1
                else:
                    unchanged_count += 1
                kin["raw_parent_event_id"] = str(parent["event_id"])
                kin["raw_start_frame"] = int(parent["start_frame"])
                kin["raw_end_frame"] = int(parent["end_frame"])
                kin["raw_duration_s"] = round(raw_len / fps, 3)
                kin["boundary_delta"] = {
                    "changed": changed,
                    "start_shift_s": round(ds, 3),
                    "end_trim_s": round(de, 3),
                    "classification": "extended" if (changed and new_len > raw_len) else ("trimmed" if changed else "unchanged"),
                }
            else:
                kin["boundary_delta"] = {
                    "changed": False,
                    "start_shift_s": 0.0,
                    "end_trim_s": 0.0,
                    "classification": "unmatched",
                }
            per_event_audit[eid] = kin

        split_parents = sum(1 for ch in parent_to_children.values() if len(ch) > 1)
        rejected_raw = sum(1 for r in raw_list if str(r.get("event_id", "")) not in parent_to_children)

        n_post = len(crop_list)
        min_part_cnt = sum(1 for e in crop_list if per_event_audit[str(e["event_id"])]["min_participants_ok"])
        dur_ok_cnt = sum(1 for e in crop_list if per_event_audit[str(e["event_id"])]["duration_ok"])
        end_ok_cnt = sum(1 for e in crop_list if per_event_audit[str(e["event_id"])]["endpoint_observed_ok"])
        full_ok_cnt = sum(1 for e in crop_list if per_event_audit[str(e["event_id"])]["full_crop_precision_ok"])

        raw_durs = [(int(e["end_frame"]) - int(e["start_frame"])) / fps for e in raw_list]
        post_durs = [(int(e["end_frame"]) - int(e["start_frame"])) / fps for e in crop_list]

        eff_req_s, eff_comp = min_clip_duration_policy_for_behavior(beh, parameters=parameters)

        behavior_summaries[beh] = {
            "behavior": beh,
            "label": SOCIAL_BEHAVIOR_LABELS[beh],
            "required_duration_s": round(eff_req_s, 3),
            "duration_comparator": eff_comp,
            "raw_candidate_count": len(raw_list),
            "post_crop_count": n_post,
            "changed_bounds_count": changed_count,
            "unchanged_count": unchanged_count,
            "extended_count": extended_count,
            "trimmed_count": trimmed_count,
            "split_parent_count": split_parents,
            "rejected_by_fine_crop_count": rejected_raw,
            "min_raw_duration_s": round(float(min(raw_durs)), 3) if raw_durs else 0.0,
            "mean_raw_duration_s": round(float(np.mean(raw_durs)), 3) if raw_durs else 0.0,
            "min_post_duration_s": round(float(min(post_durs)), 3) if post_durs else 0.0,
            "mean_post_duration_s": round(float(np.mean(post_durs)), 3) if post_durs else 0.0,
            "max_post_duration_s": round(float(max(post_durs)), 3) if post_durs else 0.0,
            "mean_start_shift_s": round(float(np.mean(start_shifts_s)), 3) if start_shifts_s else 0.0,
            "mean_end_trim_s": round(float(np.mean(end_trims_s)), 3) if end_trims_s else 0.0,
            "min_participants_pass_count": min_part_cnt,
            "duration_pass_count": dur_ok_cnt,
            "endpoint_observed_count": end_ok_cnt,
            "full_crop_precision_count": full_ok_cnt,
            "shared_mouse_overlap_count": int(overlaps_by_beh.get(normalize_behavior_key(beh), 0)),
            "min_participants_pass_rate": round(min_part_cnt / n_post, 4) if n_post else None,
            "duration_pass_rate": round(dur_ok_cnt / n_post, 4) if n_post else None,
            "endpoint_observed_rate": round(end_ok_cnt / n_post, 4) if n_post else None,
            "full_crop_precision_rate": round(full_ok_cnt / n_post, 4) if n_post else None,
        }

    total_raw = sum(v["raw_candidate_count"] for v in behavior_summaries.values())
    total_post = sum(v["post_crop_count"] for v in behavior_summaries.values())
    total_min_part = sum(v["min_participants_pass_count"] for v in behavior_summaries.values())
    total_dur_ok = sum(v["duration_pass_count"] for v in behavior_summaries.values())
    total_end_ok = sum(v["endpoint_observed_count"] for v in behavior_summaries.values())
    total_full_ok = sum(v["full_crop_precision_count"] for v in behavior_summaries.values())
    social_overlaps = sum(v["shared_mouse_overlap_count"] for v in behavior_summaries.values())

    aggregate = {
        "total_raw_social_candidates": total_raw,
        "total_post_crop_social_segments": total_post,
        "total_rejected_by_fine_crop": sum(v["rejected_by_fine_crop_count"] for v in behavior_summaries.values()),
        "total_boundary_adjusted": sum(v["changed_bounds_count"] for v in behavior_summaries.values()),
        "min_participants_pass_rate": round(total_min_part / total_post, 4) if total_post else None,
        "duration_pass_rate": round(total_dur_ok / total_post, 4) if total_post else None,
        "endpoint_observed_rate": round(total_end_ok / total_post, 4) if total_post else None,
        "full_crop_precision_rate": round(total_full_ok / total_post, 4) if total_post else None,
        "social_shared_mouse_overlap_count": int(social_overlaps),
        "all_behaviors_shared_mouse_overlap_count": int(all_overlaps),
        "shared_mouse_overlap_by_behavior": {k: v for k, v in sorted(overlaps_by_beh.items()) if v > 0},
        "zero_shared_mouse_overlap_ok": (all_overlaps == 0),
    }

    return {"aggregate": aggregate, "by_behavior": behavior_summaries}, per_event_audit


def select_social_review_clips(
    post_events: Sequence[Mapping[str, Any]],
    store: Any,
    binding: Mapping[str, Any],
    *,
    per_event_audit: Mapping[str, Mapping[str, Any]] | None = None,
    max_per_behavior: int | None = None,
) -> list[dict[str, Any]]:
    """Select diverse endpoint-observed, non-overlapping social segments across timestamps."""
    fps = float(binding["source_video"]["fps"])
    selected: list[dict[str, Any]] = []
    for beh in SOCIAL_BEHAVIOR_KEYS:
        min_req = min_participants_for_behavior(beh)
        candidates = [
            copy.deepcopy(dict(e))
            for e in post_events
            if _normalize_social_canonical(str(e.get("canonical_behavior") or e.get("behavior") or "")) == beh
            and len(event_participant_ids(e)) >= min_req
        ]
        eligible = [e for e in candidates if endpoints_fully_observed(store, e, binding.get("source_video"))]
        pool = eligible if eligible else candidates

        def _sort_key(e: Mapping[str, Any]) -> tuple[Any, ...]:
            eid = str(e["event_id"])
            audit = per_event_audit.get(eid, {}) if per_event_audit else {}
            bd = audit.get("boundary_delta", {})
            full_ok = 1 if audit.get("full_crop_precision_ok") else 0
            dur = (int(e["end_frame"]) - int(e["start_frame"])) / fps
            if beh == "approach":
                end_hcos = float(e.get("approach_end_heading_cos", e.get("approach_end_toward_cos", 0.0)))
                end_tcos = float(e.get("approach_end_toward_cos", e.get("approach_toward_cos", 0.0)))
                end_dom = min(float(e.get("approach_end_actor_dominance_cm_s", 0.0)), 40.0)
                other_d = min(float(e.get("min_other_mouse_dist_cm", 0.0)), 25.0)
                end_bg = float(e.get("approach_end_box_gap_cm", 9.0))
                closing_cm = min(float(e.get("approach_closing_cm", 0.0)), 35.0)
                return (-full_ok, -round(end_hcos, 2), -round(end_tcos, 2), -end_dom, -other_d, end_bg, -closing_cm, int(e["start_frame"]))
            if beh == "attack":
                max_iou = float(e.get("attack_max_iou", 0.0))
                area_score = float(e.get("attack_max_area_ratio", 0.0)) / 5.0
                occ = float(e.get("attack_clinch_occlusion_frames", 0))
                pk = float(e.get("attack_peak_speed_cm_s", 0.0))
                return (-full_ok, -max(max_iou, area_score), -occ, -pk, int(e["start_frame"]))
            if beh == "chase":
                min_sp = min(
                    float(e.get("chase_mean_follower_speed_cm_s", 0.0)),
                    float(e.get("chase_mean_leader_speed_cm_s", 0.0)),
                )
                vcos = float(e.get("chase_mean_vel_cos", 0.0))
                trail = float(e.get("chase_trail_error_cm", 99.0))
                inter = float(e.get("chase_intervening_fraction", 1.0))
                score = round(min_sp * (0.5 + 0.5 * vcos), 2)
                return (-full_ok, -score, -vcos, trail, inter, int(e["start_frame"]))
            changed = 1 if bd.get("changed") else 0
            return (-full_ok, -changed, -dur, int(e["start_frame"]))

        ranked = sorted(pool, key=_sort_key)
        chosen: list[dict[str, Any]] = []
        used_buckets: set[int] = set()
        limit = len(ranked) if (max_per_behavior is None or max_per_behavior <= 0) else int(max_per_behavior)

        def _overlaps_shared_mouse(ev: Mapping[str, Any]) -> bool:
            mids_set = set(event_participant_ids(ev))
            s_e, e_e = int(ev["start_frame"]), int(ev["end_frame"])
            for old in chosen:
                if not (mids_set & set(event_participant_ids(old))):
                    continue
                if not (e_e <= int(old["start_frame"]) or s_e >= int(old["end_frame"])):
                    return True
            return False

        passes = (True, False) if len(ranked) > limit else (False,)
        for diversify in passes:
            for ev in ranked:
                if ev in chosen or _overlaps_shared_mouse(ev):
                    continue
                bucket = int(int(ev["start_frame"]) / fps / 45.0)
                if diversify and bucket in used_buckets:
                    continue
                chosen.append(ev)
                used_buckets.add(bucket)
                if len(chosen) >= limit:
                    break
            if len(chosen) >= limit:
                break
        selected.extend(sorted(chosen, key=lambda e: int(e["start_frame"])))
    return selected
