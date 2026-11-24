"""Attack and Chasing segment clipping with active intensity, fast pursuit, and directed roles."""
from __future__ import annotations

import copy
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from mouse_behavior.behavior.duration_policy import qualify_event_duration, resolve_duration_policy
from mouse_behavior.behavior.segment_clipping_common import (
    _detection_map,
    _extract_box_px,
    _extract_center_px,
    _extract_keypoints_px,
    _has_valid_measurement,
    _observed_support_intervals,
    _resolve_fps,
    _resolve_parameters,
    _resolve_scale_cm_per_px,
    _resolve_video_info,
    endpoints_fully_observed,
    normalize_behavior_key,
    update_segment_bounds,
)
from mouse_behavior.behavior.segment_clipping_social_approach import _smoothed_track_kinematics
from mouse_behavior.behavior.segment_clipping_social_contact import (
    _box_gap_cm,
    _box_iou,
    attach_social_roles,
)
from mouse_behavior.evaluation.exported_dataset_models import HeuristicParameters

__all__ = [
    "discover_fast_chasing_candidates",
    "discover_fight_ball_attack_episodes",
    "discover_id_switch_chasing_candidates",
    "refine_attack_segments",
    "refine_chasing_segments",
]


def _ordered_pair_ids(event: Mapping[str, Any]) -> list[int]:
    raw: list[int] = []
    for key in ("actor_id", "target_id"):
        val = event.get(key)
        if val is not None and not isinstance(val, bool) and int(val) >= 0:
            raw.append(int(val))
    for mid in event.get("mouse_ids", ()):
        if not isinstance(mid, bool) and int(mid) >= 0:
            raw.append(int(mid))
    return list(dict.fromkeys(raw))


def _point_to_segment_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-9:
        return float(np.linalg.norm(p - a))
    t = max(0.0, min(1.0, float(np.dot(p - a, ab)) / denom))
    return float(np.linalg.norm(p - (a + t * ab)))


def _pair_body_heading_cos(
    kp_follower: np.ndarray | None,
    kp_leader: np.ndarray | None,
) -> float | None:
    """Return cosine between chaser and chased tail-to-nose body axes when both are finite."""
    if kp_follower is None or kp_leader is None:
        return None
    if not (
        np.isfinite(kp_follower[0]).all()
        and np.isfinite(kp_follower[6]).all()
        and np.isfinite(kp_leader[0]).all()
        and np.isfinite(kp_leader[6]).all()
    ):
        return None
    bf = kp_follower[0] - kp_follower[6]
    bl = kp_leader[0] - kp_leader[6]
    nbf = float(np.linalg.norm(bf))
    nbl = float(np.linalg.norm(bl))
    if nbf <= 1e-3 or nbl <= 1e-3:
        return None
    return float(np.dot(bf, bl) / (nbf * nbl))


def _extract_pair_smoothed_kinematics(
    rows: Sequence[Mapping[str, Any]],
    mid_a: int,
    mid_b: int,
    fps: float,
    cm_per_px: float,
    *,
    half_w_s: float = 0.10,
    crowd_radius_cm: float = 11.0,
    intervening_corridor_cm: float = 4.5,
) -> dict[str, dict[int, Any]]:
    """Compute pair kinematics, box IoU/gap, single-mouse clinch occlusion, crowd & intervening flags."""
    frame_map = {int(r.get("frame", r.get("source_frame", 0))): r for r in rows}
    all_frames = sorted(frame_map.keys())

    pos_a: dict[int, np.ndarray | None] = {}
    pos_b: dict[int, np.ndarray | None] = {}
    kp_a_map: dict[int, np.ndarray | None] = {}
    kp_b_map: dict[int, np.ndarray | None] = {}
    dist_cm: dict[int, float] = {}
    box_gap_cm: dict[int, float] = {}
    iou_map: dict[int, float] = {}
    co_obs: dict[int, bool] = {}
    obs_a: dict[int, bool] = {}
    obs_b: dict[int, bool] = {}
    crowd_flag: dict[int, bool] = {}
    intervening_flag: dict[int, bool] = {}
    others_map: dict[int, list[np.ndarray]] = {}
    single_pos: dict[int, np.ndarray] = {}

    for f in all_frames:
        dmap = _detection_map(frame_map[f])
        da = dmap.get(mid_a)
        db = dmap.get(mid_b)
        ca = _extract_center_px(da) if da is not None and _has_valid_measurement(da)[0] else None
        cb = _extract_center_px(db) if db is not None and _has_valid_measurement(db)[0] else None
        pos_a[f] = ca
        pos_b[f] = cb
        obs_a[f] = ca is not None
        obs_b[f] = cb is not None
        kp_a_map[f] = _extract_keypoints_px(da) if (da is not None and ca is not None) else None
        kp_b_map[f] = _extract_keypoints_px(db) if (db is not None and cb is not None) else None

        others: list[np.ndarray] = []
        for oid, odet in dmap.items():
            if oid in (mid_a, mid_b) or oid < 0 or not _has_valid_measurement(odet)[0]:
                continue
            co = _extract_center_px(odet)
            if co is not None:
                others.append(co)
        if others:
            others_map[f] = others

        if ca is not None and cb is not None and da is not None and db is not None:
            ca0, ca1 = float(ca[0]), float(ca[1])
            cb0, cb1 = float(cb[0]), float(cb[1])
            d_ab = math.hypot(ca0 - cb0, ca1 - cb1) * cm_per_px
            dist_cm[f] = d_ab
            ba = _extract_box_px(da)
            bb = _extract_box_px(db)
            if ba is not None and bb is not None:
                box_gap_cm[f] = _box_gap_cm(ba, bb, cm_per_px)
                iou_map[f] = _box_iou(ba, bb)
            else:
                box_gap_cm[f] = max(0.0, d_ab - 4.0)
                iou_map[f] = 0.0
            co_obs[f] = True

            crowd_hit = False
            inter_hit = False
            for co in others:
                ox, oy = float(co[0]), float(co[1])
                do_cm = min(
                    math.hypot(ox - ca0, oy - ca1) * cm_per_px,
                    math.hypot(ox - cb0, oy - cb1) * cm_per_px,
                )
                if do_cm <= crowd_radius_cm:
                    crowd_hit = True
                if _point_to_segment_dist(co, ca, cb) * cm_per_px <= intervening_corridor_cm:
                    inter_hit = True
            crowd_flag[f] = crowd_hit
            intervening_flag[f] = inter_hit
        else:
            co_obs[f] = False
            crowd_flag[f] = False
            intervening_flag[f] = False
            if (ca is not None) != (cb is not None):
                single_pos[f] = ca if ca is not None else cb  # type: ignore[assignment]

    # Mark single-mouse clinch occlusion only when the visible mouse stays near a known close pair site.
    anchor_midpoints = {
        f: 0.5 * (pos_a[f] + pos_b[f])  # type: ignore[operator]
        for f in all_frames
        if co_obs.get(f, False) and dist_cm.get(f, 99.0) <= 10.0 and pos_a[f] is not None and pos_b[f] is not None
    }
    sorted_anchors = sorted(anchor_midpoints.keys())
    single_obs: dict[int, bool] = {}
    if sorted_anchors:
        import bisect

        for f, sp in single_pos.items():
            idx = bisect.bisect_left(sorted_anchors, f)
            near = False
            for j in (idx - 1, idx):
                if 0 <= j < len(sorted_anchors):
                    af = sorted_anchors[j]
                    if abs(af - f) <= int(round(0.6 * fps)) + 4:
                        if float(np.linalg.norm(sp - anchor_midpoints[af])) * cm_per_px <= 11.0:
                            near = True
                            break
            single_obs[f] = near

    vel_a, sm_a, raw1f_a = _smoothed_track_kinematics(pos_a, all_frames, fps, cm_per_px, half_w_s=half_w_s)
    vel_b, sm_b, raw1f_b = _smoothed_track_kinematics(pos_b, all_frames, fps, cm_per_px, half_w_s=half_w_s)
    return {
        "pos_a": pos_a,
        "pos_b": pos_b,
        "kp_a": kp_a_map,
        "kp_b": kp_b_map,
        "dist_cm": dist_cm,
        "box_gap_cm": box_gap_cm,
        "iou_map": iou_map,
        "vel_a": vel_a,
        "vel_b": vel_b,
        "sm_a": sm_a,
        "sm_b": sm_b,
        "raw1f_a": raw1f_a,
        "raw1f_b": raw1f_b,
        "co_obs": co_obs,
        "single_obs": single_obs,
        "crowd_flag": crowd_flag,
        "intervening_flag": intervening_flag,
        "others_map": others_map,
    }


def _trim_to_observed_endpoints(
    cand_s: int,
    cand_e: int,
    co_obs: Mapping[int, bool],
) -> tuple[int, int] | None:
    obs_frames = [f for f in range(cand_s, cand_e) if co_obs.get(f, False)]
    if not obs_frames:
        return None
    return obs_frames[0], obs_frames[-1] + 1


def _finalize_pair_candidate(
    event: Mapping[str, Any],
    cand_s: int,
    cand_e: int,
    orig_start: int,
    orig_end: int,
    co_obs: Mapping[int, bool],
    store: Any,
    video_info: Mapping[str, Any],
    fps: float,
    params: HeuristicParameters,
    *,
    max_dropout_fraction: float,
    req_s: float,
    suffix_crop: str,
    suffix_fallback: str,
    span_validator: Any = None,
    allow_fallback: bool = False,
    min_req_override_s: float | None = None,
    strict_gt: bool = False,
) -> dict[str, Any] | None:
    """Enforce observed endpoints, dropout ceiling, kinematics, and minimum duration gate."""
    policy_req_s = float(req_s)
    eff_req_s = float(min_req_override_s) if min_req_override_s is not None else policy_req_s

    def _try_span(s_val: int, e_val: int, suffix: str) -> dict[str, Any] | None:
        trimmed = _trim_to_observed_endpoints(s_val, e_val, co_obs)
        if trimmed is None:
            return None
        ts, te = trimmed
        span = te - ts
        if span <= 0:
            return None
        obs_mask = np.array([co_obs.get(f, False) for f in range(ts, te)], dtype=bool)
        both_in_span = int(obs_mask.sum())
        if (1.0 - both_in_span / max(1, span)) > float(max_dropout_fraction) + 1e-9:
            return None
        dur_s = span / fps
        if strict_gt:
            if dur_s <= eff_req_s + 1e-9:
                return None
        elif dur_s < eff_req_s - 1e-9:
            return None

        cropped = update_segment_bounds(event, ts, te, fps, suffix=suffix)
        supp_intervals = _observed_support_intervals(obs_mask, ts)
        if not supp_intervals:
            return None
        cropped["canonical_support_frame_intervals"] = supp_intervals
        cropped["canonical_support_frame_count"] = int(sum(b - a for a, b in supp_intervals))

        if min_req_override_s is None or float(min_req_override_s) >= policy_req_s - 1e-9:
            if not qualify_event_duration(cropped, fps, parameters=params)["ok"]:
                return None
        elif normalize_behavior_key(str(cropped.get("canonical_behavior") or "")) == "chasing":
            cropped["chase_min_duration_policy_s"] = round(float(eff_req_s), 3)

        if not endpoints_fully_observed(store, cropped, video_info):
            return None
        if span_validator is not None and not span_validator(cropped, ts, te):
            return None
        return cropped

    result = _try_span(cand_s, cand_e, suffix_crop)
    if result is not None or not allow_fallback:
        return result
    return _try_span(orig_start, orig_end, suffix_fallback)


def _select_attack_roles(
    mid_a: int,
    mid_b: int,
    ts: int,
    te: int,
    f_min: int,
    scan_s: int,
    fps: float,
    cm_per_px: float,
    pos_a: Mapping[int, np.ndarray | None],
    pos_b: Mapping[int, np.ndarray | None],
    kp_a: Mapping[int, np.ndarray | None],
    kp_b: Mapping[int, np.ndarray | None],
    vel_a: Mapping[int, np.ndarray],
    vel_b: Mapping[int, np.ndarray],
    dist_cm: Mapping[int, float],
) -> tuple[int, int]:
    """Identify attacker vs victim from pre-clash lunge, snout-to-target orientation, and breakout pursuit."""
    pre_lo = max(scan_s, min(ts, f_min - max(3, int(round(0.50 * fps)))))
    early_hi = min(te, f_min + max(2, int(round(0.10 * fps))))

    score_a = 0.0
    score_b = 0.0

    for f in range(pre_lo, early_hi):
        pa, pb = pos_a.get(f), pos_b.get(f)
        va, vb = vel_a.get(f), vel_b.get(f)
        d = dist_cm.get(f, 99.0)
        if pa is None or pb is None or d > 22.0:
            continue
        rab = pb - pa
        nr = float(np.linalg.norm(rab))
        if nr > 1.0:
            uab = rab / nr
            w = 1.5 if f <= f_min else 1.0
            if va is not None:
                score_a += w * float(np.dot(va, uab))
            if vb is not None:
                score_b += w * float(np.dot(vb, -uab))

        ka, kb = kp_a.get(f), kp_b.get(f)
        if ka is not None and np.isfinite(ka[0]).all():
            da_snout = float(np.linalg.norm(ka[0] - pb)) * cm_per_px
            if da_snout <= 5.5:
                score_a += 10.0 * max(0.0, 5.5 - da_snout)
        if kb is not None and np.isfinite(kb[0]).all():
            db_snout = float(np.linalg.norm(kb[0] - pa)) * cm_per_px
            if db_snout <= 5.5:
                score_b += 10.0 * max(0.0, 5.5 - db_snout)

    for f in range(early_hi, te):
        pa, pb = pos_a.get(f), pos_b.get(f)
        va, vb = vel_a.get(f), vel_b.get(f)
        if pa is None or pb is None or va is None or vb is None:
            continue
        rab = pb - pa
        nr = float(np.linalg.norm(rab))
        if nr > 1.0:
            uab = rab / nr
            score_a += 0.5 * float(np.dot(va, uab))
            score_b += 0.5 * float(np.dot(vb, -uab))

    return (mid_a, mid_b) if score_a >= score_b else (mid_b, mid_a)


def refine_attack_segments(
    events: Sequence[Mapping[str, Any]],
    store: Any,
    binding: Mapping[str, Any] | None = None,
    *,
    scale: Mapping[str, Any] | None = None,
    parameters: HeuristicParameters | None = None,
    min_duration_s: float | None = 0.5,
    max_dropout_fraction: float = 0.45,
    min_path_px: float = 60.0,
    min_speed_px_s: float = 40.0,
    surge_dist_cm: float = 16.0,
    surge_speed_cm_s: float = 12.0,
    grapple_dist_cm: float = 8.5,
    min_peak_speed_cm_s: float | None = None,
    max_min_dist_cm: float | None = None,
    max_contact_dist_cm: float = 7.5,
    max_contact_box_gap_cm: float = 0.35,
    max_jump_speed_cm_s: float = 125.0,
    min_sm_peak_speed_cm_s: float = 35.0,
    min_sm_both_peak_cm_s: float = 22.0,
    min_close_mean_speed_cm_s: float = 22.0,
    min_mid_path_cm: float = 11.0,
    min_max_iou: float = 0.42,
    min_clinch_occlusion_frames: int = 3,
    min_engaged_frames: int = 14,
    max_crowd_fraction: float = 0.25,
    crowd_radius_cm: float = 11.0,
    require_clinch_occlusion: bool = True,
    min_burst_speed_cm_s: float = 15.0,
    min_vel_change_cm_s: float = 15.0,
    min_active_speed_cm_s: float = 20.0,
    max_gap_frames: int | None = None,
) -> list[dict[str, Any]]:
    """Validate and crop real Attack segments (>0.5s) using non-crowded clinch IoU/occlusion, struggle kinematics, and roles."""
    _unused_compat = (min_path_px, min_speed_px_s, min_burst_speed_cm_s, min_vel_change_cm_s)
    del _unused_compat

    fps = _resolve_fps(binding)
    params = _resolve_parameters(binding, parameters)
    policy_req_s = float(resolve_duration_policy("attack", parameters=params).required_s)
    eff_min_dur = float(min_duration_s) if min_duration_s is not None else 0.5
    req_s = max(policy_req_s, eff_min_dur)
    info = _resolve_video_info(binding)
    frame_count = info.get("frame_count")
    gap_limit = max_gap_frames if max_gap_frames is not None else max(2, int(round(0.10 * fps)))

    eff_contact_dist = float(max_min_dist_cm if max_min_dist_cm is not None else max_contact_dist_cm)
    eff_peak_speed = float(min_peak_speed_cm_s if min_peak_speed_cm_s is not None else min_sm_peak_speed_cm_s)
    eff_min_active = max(4, int(round(0.15 * fps)))

    refined: list[dict[str, Any]] = []
    for event in events:
        if normalize_behavior_key(str(event.get("canonical_behavior") or event.get("behavior") or "")) != "attack":
            refined.append(copy.deepcopy(dict(event)))
            continue
        start = int(event["start_frame"])
        end = int(event["end_frame"])
        mids = _ordered_pair_ids(event)
        if len(mids) < 2 or end <= start:
            continue

        mid_a, mid_b = mids[0], mids[1]
        lookback = int(round(1.0 * fps))
        lookforward = int(round(1.2 * fps))
        scan_s = max(0, start - lookback)
        scan_e = min(int(frame_count), end + lookforward) if frame_count is not None else end + lookforward

        rows = list(store.rows(scan_s, scan_e))
        if not rows:
            continue

        cm_per_px = _resolve_scale_cm_per_px(rows, mids[:2], binding, scale=scale)
        kin = _extract_pair_smoothed_kinematics(
            rows,
            mid_a,
            mid_b,
            fps,
            cm_per_px,
            half_w_s=0.10,
            crowd_radius_cm=crowd_radius_cm,
        )
        pos_a: dict[int, np.ndarray | None] = kin["pos_a"]
        pos_b: dict[int, np.ndarray | None] = kin["pos_b"]
        kp_a_map: dict[int, np.ndarray | None] = kin["kp_a"]
        kp_b_map: dict[int, np.ndarray | None] = kin["kp_b"]
        dist_cm: dict[int, float] = kin["dist_cm"]
        box_gap_cm: dict[int, float] = kin["box_gap_cm"]
        iou_map: dict[int, float] = kin["iou_map"]
        vel_a: dict[int, np.ndarray] = kin["vel_a"]
        vel_b: dict[int, np.ndarray] = kin["vel_b"]
        sm_a: dict[int, float] = kin["sm_a"]
        sm_b: dict[int, float] = kin["sm_b"]
        r1a: dict[int, float] = kin["raw1f_a"]
        r1b: dict[int, float] = kin["raw1f_b"]
        co_obs: dict[int, bool] = kin["co_obs"]
        single_obs: dict[int, bool] = kin["single_obs"]
        crowd_flag: dict[int, bool] = kin["crowd_flag"]

        core_lo = max(scan_s, start - 5)
        core_hi = min(scan_e, end + 6)
        core_fs = [f for f in range(core_lo, core_hi) if co_obs.get(f, False) and f in dist_cm]
        if not core_fs:
            continue
        f_min = max(core_fs, key=lambda f: iou_map.get(f, 0.0) - 0.05 * dist_cm[f])
        if dist_cm[f_min] > eff_contact_dist:
            f_min = min(core_fs, key=lambda f: dist_cm[f])
        if dist_cm[f_min] > eff_contact_dist or min(box_gap_cm.get(f, 99.0) for f in core_fs) > max_contact_box_gap_cm:
            continue

        def _jump_free(fn: int) -> bool:
            return r1a.get(fn, 0.0) <= max_jump_speed_cm_s and r1b.get(fn, 0.0) <= max_jump_speed_cm_s

        pre_limit = max(scan_s, f_min - int(round(0.65 * fps)))
        cand_s = f_min
        gap = 0
        clinch_occ_run = 0
        for f in range(f_min - 1, pre_limit - 1, -1):
            d = dist_cm.get(f)
            max_sp = max(sm_a.get(f, 0.0), sm_b.get(f, 0.0))
            if (
                co_obs.get(f, False)
                and _jump_free(f)
                and d is not None
                and d <= surge_dist_cm
                and (max_sp >= surge_speed_cm_s or d <= grapple_dist_cm)
            ):
                cand_s = f
                gap = 0
                clinch_occ_run = 0
            elif single_obs.get(f, False) and clinch_occ_run < 8:
                cand_s = f
                clinch_occ_run += 1
            else:
                gap += 1
                if gap > min(2, gap_limit):
                    break

        post_limit = min(scan_e, f_min + int(round(1.10 * fps)))
        cand_e = f_min + 1
        gap_e = 0
        clinch_occ_post = 0
        for f in range(f_min + 1, post_limit):
            d = dist_cm.get(f)
            max_sp = max(sm_a.get(f, 0.0), sm_b.get(f, 0.0))
            active_post = d is not None and (
                (d <= grapple_dist_cm and (max_sp >= 10.0 or iou_map.get(f, 0.0) >= 0.30))
                or (d <= 22.0 and max_sp >= 24.0)
            )
            if co_obs.get(f, False) and _jump_free(f) and active_post:
                cand_e = f + 1
                gap_e = 0
                clinch_occ_post = 0
            elif single_obs.get(f, False) and clinch_occ_post < 8:
                cand_e = f + 1
                clinch_occ_post += 1
            else:
                gap_e += 1
                if gap_e > min(3, gap_limit + 1):
                    break

        trim_floor = max(cand_s, f_min - int(round(0.25 * fps)))
        while cand_s < trim_floor:
            max_sp = max(sm_a.get(cand_s, 0.0), sm_b.get(cand_s, 0.0))
            if not co_obs.get(cand_s, False) or max_sp < 12.0:
                cand_s += 1
            else:
                break

        def _validate_attack_span(cropped: dict[str, Any], ts: int, te: int) -> bool:
            fs = [f for f in range(ts, te) if co_obs.get(f, False) and f in dist_cm]
            if len(fs) < 4:
                return False

            check_jump_fs = sorted(set(range(ts + 1, te)) | set(range(max(scan_s + 1, f_min - 3), min(scan_e, f_min + 4))))
            max_r1 = max((max(r1a.get(f, 0.0), r1b.get(f, 0.0)) for f in check_jump_fs), default=0.0)
            if max_r1 > max_jump_speed_cm_s:
                return False

            crowd_frac = sum(1 for f in fs if crowd_flag.get(f, False)) / max(1, len(fs))
            if crowd_frac > max_crowd_fraction + 1e-9:
                return False

            min_d = float(min(dist_cm[f] for f in fs))
            min_bg = float(min(box_gap_cm.get(f, 99.0) for f in fs))
            max_iou = float(max(iou_map.get(f, 0.0) for f in fs))
            if min_d > eff_contact_dist or min_bg > max_contact_box_gap_cm or max_iou < min_max_iou:
                return False

            miss1_inside = sum(1 for f in range(ts, te) if single_obs.get(f, False))
            miss1_adj_pre = 0
            for f in range(ts - 1, max(scan_s - 1, ts - 13), -1):
                if single_obs.get(f, False):
                    miss1_adj_pre += 1
                else:
                    break
            miss1_adj_post = 0
            for f in range(te, min(scan_e, te + 12)):
                if single_obs.get(f, False):
                    miss1_adj_post += 1
                else:
                    break
            miss1_adj = max(miss1_adj_pre, miss1_adj_post)

            sm_max_list = [max(sm_a.get(f, 0.0), sm_b.get(f, 0.0)) for f in fs]
            sm_min_list = [min(sm_a.get(f, 0.0), sm_b.get(f, 0.0)) for f in fs]
            peak_max = float(max(sm_max_list))
            peak_both = float(max(sm_min_list))
            mean_sp = float(np.mean(sm_max_list))
            act_frames = sum(1 for sp in sm_max_list if sp >= min_active_speed_cm_s)

            close_fs = [f for f in fs if dist_cm[f] <= 7.8]
            if not close_fs:
                return False
            close_mn = float(np.mean([max(sm_a.get(f, 0.0), sm_b.get(f, 0.0)) for f in close_fs]))
            engaged_frames = len(close_fs) + miss1_inside + (miss1_adj_pre if miss1_inside < min_clinch_occlusion_frames else 0)

            if (
                peak_max < eff_peak_speed
                or peak_both < min_sm_both_peak_cm_s
                or close_mn < min_close_mean_speed_cm_s
                or act_frames < eff_min_active
                or engaged_frames < min_engaged_frames
                or len(close_fs) < min(len(close_fs) + miss1_inside, 10)
            ):
                return False

            if require_clinch_occlusion and miss1_inside < min_clinch_occlusion_frames:
                return False

            mids_xy = [0.5 * (pos_a[f] + pos_b[f]) for f in fs if pos_a[f] is not None and pos_b[f] is not None]  # type: ignore[operator]
            mid_path = (
                sum(float(np.linalg.norm(mids_xy[i] - mids_xy[i - 1])) * cm_per_px for i in range(1, len(mids_xy)))
                if len(mids_xy) >= 2
                else 0.0
            )
            if mid_path < min_mid_path_cm:
                return False

            sep_fs = [f for f in fs if dist_cm[f] > 12.0]
            if len(sep_fs) >= 4 and len(close_fs) + miss1_inside < 10:
                sep_cos = []
                for f in sep_fs:
                    va, vb = vel_a.get(f), vel_b.get(f)
                    sa, sb = sm_a.get(f, 0.0), sm_b.get(f, 0.0)
                    if va is not None and vb is not None and sa > 1e-4 and sb > 1e-4:
                        sep_cos.append(float(np.dot(va, vb) / (sa * sb)))
                if sep_cos and float(np.mean(sep_cos)) < -0.5:
                    return False

            attacker_id, victim_id = _select_attack_roles(
                mid_a,
                mid_b,
                ts,
                te,
                f_min,
                scan_s,
                fps,
                cm_per_px,
                pos_a,
                pos_b,
                kp_a_map,
                kp_b_map,
                vel_a,
                vel_b,
                dist_cm,
            )
            cropped["mouse_ids"] = [attacker_id, victim_id]
            attach_social_roles(
                cropped,
                "attack",
                attacker_id,
                victim_id,
                cropped.get("canonical_support_frame_intervals"),
            )

            cropped["attack_peak_speed_cm_s"] = round(peak_max, 2)
            cropped["attack_both_peak_speed_cm_s"] = round(peak_both, 2)
            cropped["attack_mean_speed_cm_s"] = round(mean_sp, 2)
            cropped["attack_close_mean_speed_cm_s"] = round(close_mn, 2)
            cropped["attack_max_iou"] = round(max_iou, 3)
            cropped["attack_clinch_occlusion_frames"] = int(miss1_inside)
            cropped["attack_adjacent_occlusion_frames"] = int(miss1_adj)
            cropped["attack_engaged_frames"] = int(engaged_frames)
            cropped["attack_close_frames"] = int(len(close_fs))
            cropped["attack_mid_path_cm"] = round(mid_path, 2)
            cropped["attack_crowd_fraction"] = round(crowd_frac, 3)
            cropped["attack_min_dist_cm"] = round(min_d, 3)
            cropped["attack_min_box_gap_cm"] = round(min_bg, 3)
            cropped["attack_max_raw1f_speed_cm_s"] = round(float(max_r1), 2)
            cropped["attack_active_frames"] = int(act_frames)
            return True

        finalized = _finalize_pair_candidate(
            event,
            cand_s,
            cand_e,
            start,
            end,
            co_obs,
            store,
            info,
            fps,
            params,
            max_dropout_fraction=max_dropout_fraction,
            req_s=policy_req_s,
            suffix_crop="attack_crop",
            suffix_fallback="attack_tight",
            span_validator=_validate_attack_span,
            allow_fallback=False,
            min_req_override_s=req_s,
            strict_gt=True,
        )
        if finalized is not None:
            refined.append(finalized)

    return refined


def refine_chasing_segments(
    events: Sequence[Mapping[str, Any]],
    store: Any,
    binding: Mapping[str, Any] | None = None,
    *,
    scale: Mapping[str, Any] | None = None,
    parameters: HeuristicParameters | None = None,
    min_duration_s: float | None = 0.7,
    max_separation_cm: float = 20.0,
    max_median_separation_cm: float = 14.5,
    max_gap_widening_cm: float = 4.0,
    min_follower_speed_cm_s: float = 36.0,
    min_leader_speed_cm_s: float = 36.0,
    min_mean_speed_cm_s: float = 36.0,
    min_both_p25_speed_cm_s: float = 28.0,
    min_both_fast_fraction: float = 0.70,
    both_fast_frame_speed_cm_s: float = 30.0,
    min_both_peak_speed_cm_s: float = 43.5,
    max_trail_error_cm: float = 2.6,
    max_intervening_mouse_fraction: float = 0.12,
    max_trailing_bystander_fraction: float = 0.35,
    min_net_displacement_cm: float = 22.0,
    min_vel_alignment_cos: float = 0.85,
    min_vel_alignment_p25_cos: float = 0.78,
    min_vel_aligned_fraction: float = 0.80,
    vel_aligned_frame_cos: float = 0.78,
    min_pursuit_cos: float = 0.78,
    min_pursuit_p25_cos: float = 0.65,
    min_body_heading_cos: float = 0.75,
    max_jump_speed_cm_s: float = 280.0,
    max_gap_frames: int | None = None,
    max_dropout_fraction: float = 0.35,
) -> list[dict[str, Any]]:
    """Fine-crop fast (>0.7s) trail-locked, direction-aligned Chasing segments with roles."""
    fps = _resolve_fps(binding)
    params = _resolve_parameters(binding, parameters)
    policy_req_s = float(resolve_duration_policy("chasing", parameters=params).required_s)
    eff_req_s = float(min_duration_s) if min_duration_s is not None else 0.7
    req_frames = max(2, int(np.floor(eff_req_s * fps + 1e-9)) + 1)
    info = _resolve_video_info(binding)
    frame_count = info.get("frame_count")
    gap_limit = max_gap_frames if max_gap_frames is not None else max(4, int(round(0.30 * fps)))
    eff_follower_floor = max(float(min_follower_speed_cm_s), float(min_mean_speed_cm_s))
    eff_leader_floor = max(float(min_leader_speed_cm_s), float(min_mean_speed_cm_s))

    refined: list[dict[str, Any]] = []
    for event in events:
        if normalize_behavior_key(str(event.get("canonical_behavior") or event.get("behavior") or "")) != "chasing":
            refined.append(copy.deepcopy(dict(event)))
            continue

        start = int(event["start_frame"])
        end = int(event["end_frame"])
        mids = _ordered_pair_ids(event)
        if len(mids) < 2 or end <= start:
            continue

        mid_a, mid_b = mids[0], mids[1]
        lookback = int(round(1.5 * fps))
        lookforward = int(round(1.5 * fps))
        scan_s = max(0, start - lookback)
        scan_e = min(int(frame_count), end + lookforward) if frame_count is not None else end + lookforward

        rows = list(store.rows(scan_s, scan_e))
        if not rows:
            continue

        cm_per_px = _resolve_scale_cm_per_px(rows, mids[:2], binding, scale=scale)
        kin = _extract_pair_smoothed_kinematics(
            rows,
            mid_a,
            mid_b,
            fps,
            cm_per_px,
            half_w_s=0.10,
            intervening_corridor_cm=4.5,
        )
        pos_a: dict[int, np.ndarray | None] = kin["pos_a"]
        pos_b: dict[int, np.ndarray | None] = kin["pos_b"]
        kp_a: dict[int, np.ndarray | None] = kin["kp_a"]
        kp_b: dict[int, np.ndarray | None] = kin["kp_b"]
        dist_cm: dict[int, float] = kin["dist_cm"]
        vel_a: dict[int, np.ndarray] = kin["vel_a"]
        vel_b: dict[int, np.ndarray] = kin["vel_b"]
        sm_a: dict[int, float] = kin["sm_a"]
        sm_b: dict[int, float] = kin["sm_b"]
        r1a: dict[int, float] = kin["raw1f_a"]
        r1b: dict[int, float] = kin["raw1f_b"]
        co_obs: dict[int, bool] = kin["co_obs"]
        intervening_flag: dict[int, bool] = kin["intervening_flag"]
        crowd_flag: dict[int, bool] = kin["crowd_flag"]
        others_map: dict[int, list[np.ndarray]] = kin["others_map"]

        def _trail_error_med(
            fol_map: Mapping[int, np.ndarray | None],
            lead_map: Mapping[int, np.ndarray | None],
            ts: int,
            fs: Sequence[int],
        ) -> float:
            trail_look = int(round(0.60 * fps))
            win_lo = max(scan_s, ts - trail_look)
            if not fs:
                return 99.0
            win_hi = int(fs[-1])
            l_frames: list[int] = []
            l_pts_list: list[np.ndarray] = []
            for k in range(win_lo, win_hi + 1):
                lp = lead_map.get(k)
                if lp is not None:
                    l_frames.append(k)
                    l_pts_list.append(lp)
            if not l_pts_list:
                return 99.0
            l_pts_arr = np.vstack(l_pts_list)
            l_f_arr = np.asarray(l_frames, dtype=np.int64)
            segs_a = l_pts_arr[:-1]
            segs_ab = l_pts_arr[1:] - segs_a
            segs_denom = np.einsum("ij,ij->i", segs_ab, segs_ab) if len(segs_a) else np.empty(0, dtype=float)
            errs: list[float] = []
            for fn in fs:
                pf = fol_map.get(fn)
                if pf is None:
                    continue
                n_valid = int(np.searchsorted(l_f_arr, fn, side="right"))
                if n_valid <= 0:
                    continue
                if n_valid == 1:
                    d_px = math.hypot(float(pf[0] - l_pts_arr[0, 0]), float(pf[1] - l_pts_arr[0, 1]))
                else:
                    m_seg = n_valid - 1
                    sa = segs_a[:m_seg]
                    ab = segs_ab[:m_seg]
                    denom = segs_denom[:m_seg]
                    ap = pf - sa
                    dot = np.einsum("ij,ij->i", ap, ab)
                    t = np.where(denom > 1e-9, dot / np.maximum(denom, 1e-9), 0.0)
                    t = np.clip(t, 0.0, 1.0)
                    proj = sa + t[:, None] * ab
                    diff = pf - proj
                    d_px = float(np.sqrt(np.min(diff[:, 0] ** 2 + diff[:, 1] ** 2)))
                errs.append(d_px * cm_per_px)
            return float(np.median(errs)) if errs else 99.0

        def _determine_follower(frame_iter: Sequence[int] | range) -> bool:
            score_a = 0.0
            score_b = 0.0
            f_list = list(frame_iter)
            for fn in f_list:
                pa_c, pb_c = pos_a.get(fn), pos_b.get(fn)
                va_f, vb_f = vel_a.get(fn), vel_b.get(fn)
                if pa_c is not None and pb_c is not None and va_f is not None and vb_f is not None:
                    rab = pb_c - pa_c
                    norm_r = float(np.linalg.norm(rab))
                    if norm_r > 1.0:
                        proj_a = float(np.dot(va_f, rab)) / norm_r
                        proj_b = float(np.dot(vb_f, -rab)) / norm_r
                        if proj_a > proj_b:
                            score_a += 1.0
                        elif proj_b > proj_a:
                            score_b += 1.0
            if abs(score_a - score_b) < 2.0 and f_list:
                co_fs = [f for f in f_list if co_obs.get(f, False)]
                if co_fs:
                    ea = _trail_error_med(pos_a, pos_b, co_fs[0], co_fs)
                    eb = _trail_error_med(pos_b, pos_a, co_fs[0], co_fs)
                    if ea + 0.5 < eb:
                        score_a += 3.0
                    elif eb + 0.5 < ea:
                        score_b += 3.0
            return score_a >= score_b

        def _frame_kin(fn: int, *, fol_a: bool) -> tuple[float | None, float, float, float, float, bool]:
            d = dist_cm.get(fn)
            spf = sm_a.get(fn, 0.0) if fol_a else sm_b.get(fn, 0.0)
            spl = sm_b.get(fn, 0.0) if fol_a else sm_a.get(fn, 0.0)
            vf = vel_a.get(fn) if fol_a else vel_b.get(fn)
            vl = vel_b.get(fn) if fol_a else vel_a.get(fn)
            cf = pos_a.get(fn) if fol_a else pos_b.get(fn)
            cl = pos_b.get(fn) if fol_a else pos_a.get(fn)
            jump_ok = (
                r1a.get(fn, 0.0) <= max_jump_speed_cm_s
                and r1b.get(fn, 0.0) <= max_jump_speed_cm_s
                and sm_a.get(fn, 0.0) <= 165.0
                and sm_b.get(fn, 0.0) <= 165.0
            )

            cos_vel = 0.0
            cos_pur = 0.0
            if vf is not None and vl is not None and spf > 1e-4 and spl > 1e-4:
                cos_vel = float(np.dot(vf, vl) / (spf * spl))
            if cf is not None and cl is not None and vf is not None and spf > 1e-4:
                rel = cl - cf
                nr = float(np.linalg.norm(rel))
                if nr > 1e-4:
                    cos_pur = float(np.dot(vf, rel) / (spf * nr))
            return d, spf, spl, cos_vel, cos_pur, jump_ok

        runs_idx: list[tuple[int, int]] = []
        cur_s: int | None = None
        last_close = -1
        for f in range(scan_s, scan_e):
            d_f = dist_cm.get(f)
            jump_ok_f = (
                r1a.get(f, 0.0) <= max_jump_speed_cm_s
                and r1b.get(f, 0.0) <= max_jump_speed_cm_s
                and sm_a.get(f, 0.0) <= 165.0
                and sm_b.get(f, 0.0) <= 165.0
            )
            close_obs = co_obs.get(f, False) and d_f is not None and d_f <= max_separation_cm
            if close_obs and (cur_s is None or jump_ok_f):
                if cur_s is None:
                    cur_s = f
                last_close = f
            elif close_obs and not jump_ok_f and cur_s is not None:
                runs_idx.append((cur_s, last_close + 1))
                cur_s = f
                last_close = f
            elif cur_s is not None:
                if co_obs.get(f, False) or (f - last_close) > gap_limit:
                    runs_idx.append((cur_s, last_close + 1))
                    cur_s = None
        if cur_s is not None:
            runs_idx.append((cur_s, last_close + 1))

        candidate_runs = [r for r in runs_idx if r[1] > start and r[0] < end and (r[1] - r[0]) >= req_frames]
        if not candidate_runs:
            continue

        def _span_metrics(ts: int, te: int, *, compute_trail: bool = True) -> dict[str, Any] | None:
            fs = [f for f in range(ts, te) if co_obs.get(f, False)]
            if len(fs) < 2:
                return None
            if (1.0 - len(fs) / max(1, te - ts)) > float(max_dropout_fraction) + 1e-9:
                return None
            if any(
                r1a.get(f, 0.0) > max_jump_speed_cm_s
                or r1b.get(f, 0.0) > max_jump_speed_cm_s
                or sm_a.get(f, 0.0) > 165.0
                or sm_b.get(f, 0.0) > 165.0
                for f in range(ts + 1, te)
            ):
                return None
            fol_a = _determine_follower(fs)
            kins = [_frame_kin(f, fol_a=fol_a) for f in fs]
            ds = [k[0] for k in kins if k[0] is not None]
            if len(ds) < 2:
                return None
            spfs = [k[1] for k in kins]
            spls = [k[2] for k in kins]
            cvs = [k[3] for k in kins]
            cps = [k[4] for k in kins]
            mins = [min(sf, sl) for sf, sl in zip(spfs, spls)]
            mean_spf = float(np.mean(spfs))
            mean_spl = float(np.mean(spls))
            p25_both_sp = float(np.percentile(mins, 25))
            both_fast_frac = sum(1 for v in mins if v >= both_fast_frame_speed_cm_s) / max(1, len(mins))
            pk_spf = float(max(spfs))
            pk_spl = float(max(spls))
            min_pk = float(min(pk_spf, pk_spl))
            med_d = float(np.median(ds))
            max_d = float(max(ds))
            gap_widening = float(ds[-1] - ds[0])
            mean_cv = float(np.mean(cvs))
            p25_cv = float(np.percentile(cvs, 25))
            vel_aligned_frac = sum(1 for v in cvs if v >= vel_aligned_frame_cos) / max(1, len(cvs))
            mean_cp = float(np.mean(cps))
            p25_cp = float(np.percentile(cps, 25))
            body_cos_vals: list[float] = []
            for fn in fs:
                kpf = kp_a.get(fn) if fol_a else kp_b.get(fn)
                kpl = kp_b.get(fn) if fol_a else kp_a.get(fn)
                bcos = _pair_body_heading_cos(kpf, kpl)
                if bcos is not None:
                    body_cos_vals.append(bcos)
            body_heading_cos: float | None = float(np.median(body_cos_vals)) if len(body_cos_vals) >= 3 else None

            pa0 = pos_a.get(fs[0])
            pb0 = pos_b.get(fs[0])
            net_a = (
                max(
                    (
                        math.hypot(float(pos_a[f][0] - pa0[0]), float(pos_a[f][1] - pa0[1])) * cm_per_px  # type: ignore[index]
                        for f in fs
                        if pos_a.get(f) is not None
                    ),
                    default=0.0,
                )
                if pa0 is not None
                else 0.0
            )
            net_b = (
                max(
                    (
                        math.hypot(float(pos_b[f][0] - pb0[0]), float(pos_b[f][1] - pb0[1])) * cm_per_px  # type: ignore[index]
                        for f in fs
                        if pos_b.get(f) is not None
                    ),
                    default=0.0,
                )
                if pb0 is not None
                else 0.0
            )
            min_net = float(min(net_a, net_b))
            inter_frac = sum(1 for f in fs if intervening_flag.get(f, False)) / max(1, len(fs))
            crowd_frac = sum(1 for f in fs if crowd_flag.get(f, False)) / max(1, len(fs))
            behind_hits = 0
            for f in fs:
                cf = pos_a.get(f) if fol_a else pos_b.get(f)
                cl = pos_b.get(f) if fol_a else pos_a.get(f)
                if cf is None or cl is None:
                    continue
                v_fl = cl - cf
                nv = float(np.linalg.norm(v_fl))
                if nv <= 1e-3:
                    continue
                ux, uy = float(v_fl[0] / nv), float(v_fl[1] / nv)
                for co in others_map.get(f, ()):
                    dx_cm = (float(co[0]) - float(cf[0])) * cm_per_px
                    dy_cm = (float(co[1]) - float(cf[1])) * cm_per_px
                    d_oc = math.hypot(dx_cm, dy_cm)
                    proj = dx_cm * ux + dy_cm * uy
                    perp = abs(dx_cm * (-uy) + dy_cm * ux)
                    if -13.0 <= proj <= 1.0 and perp <= 5.5 and d_oc <= 13.5:
                        behind_hits += 1
                        break
            trailing_bystander_frac = behind_hits / max(1, len(fs))

            cheap_ok = (
                max_d <= max_separation_cm
                and med_d <= max_median_separation_cm
                and gap_widening <= max_gap_widening_cm + 1e-9
                and mean_spf >= eff_follower_floor
                and mean_spl >= eff_leader_floor
                and p25_both_sp >= min_both_p25_speed_cm_s
                and both_fast_frac >= min_both_fast_fraction - 1e-9
                and min_pk >= min_both_peak_speed_cm_s
                and inter_frac <= max_intervening_mouse_fraction + 1e-9
                and trailing_bystander_frac <= max_trailing_bystander_fraction + 1e-9
                and min_net >= min_net_displacement_cm
                and mean_cv >= min_vel_alignment_cos
                and p25_cv >= min_vel_alignment_p25_cos
                and vel_aligned_frac >= min_vel_aligned_fraction - 1e-9
                and mean_cp >= min_pursuit_cos
                and p25_cp >= min_pursuit_p25_cos
                and (body_heading_cos is None or body_heading_cos >= min_body_heading_cos)
            )
            if compute_trail and cheap_ok:
                fol_map, lead_map = (pos_a, pos_b) if fol_a else (pos_b, pos_a)
                trail_err = _trail_error_med(fol_map, lead_map, ts, fs)
            else:
                trail_err = 99.0

            return {
                "fol_a": fol_a,
                "mean_spf": mean_spf,
                "mean_spl": mean_spl,
                "min_mean_sp": float(min(mean_spf, mean_spl)),
                "p25_both_sp": p25_both_sp,
                "both_fast_frac": float(both_fast_frac),
                "pk_spf": pk_spf,
                "pk_spl": pk_spl,
                "min_pk": min_pk,
                "med_d": med_d,
                "max_d": max_d,
                "gap_widening": gap_widening,
                "trail_err": trail_err,
                "inter_frac": float(inter_frac),
                "trailing_bystander_frac": float(trailing_bystander_frac),
                "crowd_frac": float(crowd_frac),
                "mean_cv": mean_cv,
                "p25_cv": p25_cv,
                "vel_aligned_frac": float(vel_aligned_frac),
                "mean_cp": mean_cp,
                "p25_cp": p25_cp,
                "body_heading_cos": body_heading_cos,
                "min_net": min_net,
            }

        def _passes_chase(m: Mapping[str, Any] | None) -> bool:
            if m is None:
                return False
            bcos = m.get("body_heading_cos")
            return bool(
                m["max_d"] <= max_separation_cm
                and m["med_d"] <= max_median_separation_cm
                and m["gap_widening"] <= max_gap_widening_cm + 1e-9
                and m["mean_spf"] >= eff_follower_floor
                and m["mean_spl"] >= eff_leader_floor
                and m["p25_both_sp"] >= min_both_p25_speed_cm_s
                and m["both_fast_frac"] >= min_both_fast_fraction - 1e-9
                and m["min_pk"] >= min_both_peak_speed_cm_s
                and m["trail_err"] <= max_trail_error_cm + 1e-9
                and m["inter_frac"] <= max_intervening_mouse_fraction + 1e-9
                and m["trailing_bystander_frac"] <= max_trailing_bystander_fraction + 1e-9
                and m["min_net"] >= min_net_displacement_cm
                and m["mean_cv"] >= min_vel_alignment_cos
                and m["p25_cv"] >= min_vel_alignment_p25_cos
                and m["vel_aligned_frac"] >= min_vel_aligned_fraction - 1e-9
                and m["mean_cp"] >= min_pursuit_cos
                and m["p25_cp"] >= min_pursuit_p25_cos
                and (bcos is None or float(bcos) >= min_body_heading_cos)
            )

        best_choice: tuple[float, int, int, int] | None = None
        for r_s, r_e in candidate_runs:
            cs, ce = r_s, r_e
            fol_run = _determine_follower(range(cs, ce))
            while ce - cs > req_frames:
                _, spf, spl, cv, cp, jump_ok = _frame_kin(cs, fol_a=fol_run)
                if not jump_ok or not co_obs.get(cs, False) or spf < 28.0 or spl < 28.0 or cv < 0.72 or cp < 0.65:
                    cs += 1
                else:
                    break
            while ce - cs > req_frames:
                d_end, spf, spl, cv, cp, jump_ok = _frame_kin(ce - 1, fol_a=fol_run)
                if (
                    not jump_ok
                    or not co_obs.get(ce - 1, False)
                    or spf < 28.0
                    or spl < 28.0
                    or cv < 0.72
                    or cp < 0.65
                    or (d_end is not None and d_end > max_separation_cm)
                ):
                    ce -= 1
                else:
                    break

            m_full = _span_metrics(cs, ce)
            if _passes_chase(m_full) and m_full is not None:
                sc_full = round(float(m_full["min_mean_sp"]) * (0.5 + 0.5 * float(m_full["mean_cv"])), 2)
                key = (sc_full, ce - cs, cs, ce)
                if best_choice is None or key[:2] > best_choice[:2]:
                    best_choice = key
                continue

            cand_starts = [ws for ws in range(r_s, r_e - req_frames + 1) if co_obs.get(ws, False)]
            if (r_e - r_s) > 180 and cand_starts:
                fast_aligned_near: set[int] = set()
                for fn in range(r_s, r_e):
                    if not co_obs.get(fn, False):
                        continue
                    _, f_spf, f_spl, f_cv, _, _ = _frame_kin(fn, fol_a=fol_run)
                    if min(f_spf, f_spl) >= 26.0 and f_cv >= 0.72:
                        fast_aligned_near.add(fn)
                cand_starts = [ws for ws in cand_starts if any(abs(ws - fn) <= 5 for fn in fast_aligned_near)]

            for ws in cand_starts:
                for we in range(r_e, ws + req_frames - 1, -1):
                    if not co_obs.get(we - 1, False):
                        continue
                    m_sub = _span_metrics(ws, we, compute_trail=True)
                    if _passes_chase(m_sub) and m_sub is not None:
                        sc_sub = round(float(m_sub["min_mean_sp"]) * (0.5 + 0.5 * float(m_sub["mean_cv"])), 2)
                        key = (sc_sub, we - ws, ws, we)
                        if best_choice is None or key[:2] > best_choice[:2]:
                            best_choice = key
                        break

        if best_choice is None:
            continue
        _, _, cand_s, cand_e = best_choice

        def _validate_chase_span(cropped: dict[str, Any], ts: int, te: int) -> bool:
            m = _span_metrics(ts, te)
            if not _passes_chase(m) or m is None:
                return False
            fol_a = bool(m["fol_a"])
            chaser_id = mid_a if fol_a else mid_b
            chased_id = mid_b if fol_a else mid_a
            cropped["mouse_ids"] = [chaser_id, chased_id]
            attach_social_roles(
                cropped,
                "chase",
                chaser_id,
                chased_id,
                cropped.get("canonical_support_frame_intervals"),
            )
            cropped["chase_mean_follower_speed_cm_s"] = round(float(m["mean_spf"]), 2)
            cropped["chase_mean_leader_speed_cm_s"] = round(float(m["mean_spl"]), 2)
            cropped["chase_min_mean_speed_cm_s"] = round(float(m["min_mean_sp"]), 2)
            cropped["chase_both_p25_speed_cm_s"] = round(float(m["p25_both_sp"]), 2)
            cropped["chase_both_fast_fraction"] = round(float(m["both_fast_frac"]), 3)
            cropped["chase_peak_follower_speed_cm_s"] = round(float(m["pk_spf"]), 2)
            cropped["chase_peak_leader_speed_cm_s"] = round(float(m["pk_spl"]), 2)
            cropped["chase_peak_speed_cm_s"] = round(float(max(m["pk_spf"], m["pk_spl"])), 2)
            cropped["chase_median_dist_cm"] = round(float(m["med_d"]), 2)
            cropped["chase_max_dist_cm"] = round(float(m["max_d"]), 2)
            cropped["chase_gap_widening_cm"] = round(float(m["gap_widening"]), 2)
            cropped["chase_trail_error_cm"] = round(float(m["trail_err"]), 2)
            cropped["chase_intervening_fraction"] = round(float(m["inter_frac"]), 3)
            cropped["chase_trailing_bystander_fraction"] = round(float(m["trailing_bystander_frac"]), 3)
            cropped["chase_crowd_fraction"] = round(float(m["crowd_frac"]), 3)
            cropped["chase_mean_vel_cos"] = round(float(m["mean_cv"]), 3)
            cropped["chase_vel_cos_p25"] = round(float(m["p25_cv"]), 3)
            cropped["chase_vel_aligned_fraction"] = round(float(m["vel_aligned_frac"]), 3)
            cropped["chase_mean_cos"] = round(float(m["mean_cp"]), 3)
            cropped["chase_pursuit_cos_p25"] = round(float(m["p25_cp"]), 3)
            bcos = m.get("body_heading_cos")
            cropped["chase_body_heading_cos"] = round(float(bcos), 3) if bcos is not None else None
            cropped["chase_min_net_disp_cm"] = round(float(m["min_net"]), 2)
            return True

        finalized = _finalize_pair_candidate(
            event,
            cand_s,
            cand_e,
            start,
            end,
            co_obs,
            store,
            info,
            fps,
            params,
            max_dropout_fraction=max_dropout_fraction,
            req_s=policy_req_s,
            suffix_crop="chasing_crop",
            suffix_fallback="chasing_tight",
            span_validator=_validate_chase_span,
            allow_fallback=False,
            min_req_override_s=eff_req_s,
            strict_gt=True,
        )
        if finalized is not None:
            refined.append(finalized)

    return refined


def discover_fast_chasing_candidates(
    store: Any,
    binding: Mapping[str, Any] | None = None,
    scale: Mapping[str, Any] | None = None,
    *,
    parameters: HeuristicParameters | None = None,
    existing_events: Sequence[Mapping[str, Any]] = (),
    min_duration_s: float = 0.7,
    max_separation_cm: float = 20.0,
    min_pair_speed_cm_s: float = 28.0,
    refine_kwargs: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Discover missed fast-pursuit Chasing bouts directly from full trajectories and validate them."""
    info = _resolve_video_info(binding)
    fps = _resolve_fps(binding)
    params = _resolve_parameters(binding, parameters)
    frame_count = info.get("frame_count")
    scan_end = int(frame_count) if frame_count is not None else int(getattr(store, "frame_count", 0) or 0)

    if scan_end <= 0:
        rows = list(store.rows(0, 10**9))
    else:
        rows = list(store.rows(0, scan_end))
    if len(rows) < 4:
        return []

    track_frames: dict[int, list[int]] = defaultdict(list)
    track_coords: dict[int, list[tuple[float, float]]] = defaultdict(list)
    frame_centers: dict[int, dict[int, tuple[float, float]]] = {}
    sample_mids: list[int] = []

    for row in rows:
        f = int(row.get("frame", row.get("source_frame", 0)))
        dmap = _detection_map(row)
        fc: dict[int, tuple[float, float]] = {}
        for tid, det in dmap.items():
            if tid < 0 or not _has_valid_measurement(det)[0]:
                continue
            c = _extract_center_px(det)
            if c is None:
                continue
            pt = (float(c[0]), float(c[1]))
            fc[tid] = pt
            track_frames[tid].append(f)
            track_coords[tid].append(pt)
            if len(sample_mids) < 8 and tid not in sample_mids:
                sample_mids.append(tid)
        if fc:
            frame_centers[f] = fc

    if len(track_frames) < 2:
        return []

    cm_per_px = _resolve_scale_cm_per_px(rows[: min(len(rows), 200)], sample_mids[:2], binding, scale=scale)
    half_w = max(2, int(round(0.10 * fps)))

    fast_frames_by_track: dict[int, set[int]] = {}
    for tid, f_list in track_frames.items():
        n = len(f_list)
        if n < 4:
            continue
        coords = track_coords[tid]
        fast_set: set[int] = set()
        j0 = 0
        j1 = 0
        for idx in range(n):
            f_cur = f_list[idx]
            while j0 < idx and f_cur - f_list[j0] > half_w:
                j0 += 1
            while j1 + 1 < n and f_list[j1 + 1] - f_cur <= half_w:
                j1 += 1
            df = f_list[j1] - f_list[j0]
            if df <= 0:
                continue
            x0, y0 = coords[j0]
            x1, y1 = coords[j1]
            sp = math.hypot(x1 - x0, y1 - y0) * (fps / df) * cm_per_px
            if sp >= min_pair_speed_cm_s:
                fast_set.add(f_cur)
        if fast_set:
            fast_frames_by_track[tid] = fast_set

    pair_frames: dict[tuple[int, int], list[int]] = defaultdict(list)
    sep_limit_px = (max_separation_cm + 2.0) / max(1e-9, cm_per_px)
    for f, fc in frame_centers.items():
        fast_ids = sorted(tid for tid in fc if tid in fast_frames_by_track and f in fast_frames_by_track[tid])
        if len(fast_ids) < 2:
            continue
        for i in range(len(fast_ids)):
            a = fast_ids[i]
            ax, ay = fc[a]
            for j in range(i + 1, len(fast_ids)):
                b = fast_ids[j]
                bx, by = fc[b]
                if math.hypot(ax - bx, ay - by) <= sep_limit_px:
                    pair_frames[(a, b)].append(f)

    existing_chase_spans: list[tuple[frozenset[int], int, int]] = []
    for ev in existing_events:
        if normalize_behavior_key(str(ev.get("canonical_behavior") or ev.get("behavior") or "")) != "chasing":
            continue
        mids = _ordered_pair_ids(ev)
        if len(mids) >= 2:
            existing_chase_spans.append((frozenset(mids[:2]), int(ev["start_frame"]), int(ev["end_frame"])))

    min_seed_frames = max(6, int(round(0.35 * min_duration_s * fps)))
    raw_candidates: list[dict[str, Any]] = []
    max_bridge = max(6, int(round(0.35 * fps)))
    pad_existing = int(round(1.0 * fps))

    for (a, b), flist in sorted(pair_frames.items()):
        if len(flist) < min_seed_frames:
            continue
        flist.sort()
        s = flist[0]
        e = flist[0] + 1
        merged_runs: list[tuple[int, int]] = []
        for fn in flist[1:]:
            if fn - (e - 1) <= max_bridge:
                e = fn + 1
            else:
                merged_runs.append((s, e))
                s = fn
                e = fn + 1
        merged_runs.append((s, e))

        pair_set = frozenset((a, b))
        for rs, re in merged_runs:
            if re - rs < min_seed_frames:
                continue
            overlaps_existing = any(
                ps == pair_set and not (re <= es - pad_existing or rs >= ee + pad_existing)
                for ps, es, ee in existing_chase_spans
            )
            if overlaps_existing:
                continue
            raw_candidates.append(
                {
                    "event_id": f"chase_disc_{a}_{b}_{rs:08d}_{re:08d}",
                    "canonical_behavior": "chase",
                    "behavior": "Chasing",
                    "mouse_ids": [a, b],
                    "start_frame": int(rs),
                    "end_frame": int(re),
                    "discovery_source": "fast_pursuit_scan",
                }
            )

    if not raw_candidates:
        return []

    kwargs = dict(refine_kwargs or {})
    kwargs.setdefault("min_duration_s", min_duration_s)
    kwargs.setdefault("max_separation_cm", max_separation_cm)
    validated = refine_chasing_segments(
        raw_candidates,
        store,
        binding,
        scale=scale,
        parameters=params,
        **kwargs,
    )

    deduped: list[dict[str, Any]] = []
    seen_spans: list[tuple[frozenset[int], int, int]] = list(existing_chase_spans)
    for ev in validated:
        mids = _ordered_pair_ids(ev)
        if len(mids) < 2:
            continue
        ps = frozenset(mids[:2])
        s_ev, e_ev = int(ev["start_frame"]), int(ev["end_frame"])
        if any(p == ps and not (e_ev <= s0 or s_ev >= e0) for p, s0, e0 in seen_spans):
            continue
        seen_spans.append((ps, s_ev, e_ev))
        deduped.append(ev)
    return deduped


def _build_full_store_track_index(
    store: Any,
    binding: Mapping[str, Any] | None,
    scale: Mapping[str, Any] | None,
    *,
    max_speed_cm_s: float = 220.0,
    half_w_s: float = 0.10,
) -> dict[str, Any] | None:
    """Scan full track store once to index centers, boxes, smoothed velocities, and median area."""
    info = _resolve_video_info(binding)
    fps = _resolve_fps(binding)
    frame_count = info.get("frame_count")
    scan_end = int(frame_count) if frame_count is not None else int(getattr(store, "frame_count", 0) or 0)
    rows = list(store.rows(0, 10**9 if scan_end <= 0 else scan_end))
    if len(rows) < 4:
        return None

    fmap: dict[int, dict[int, tuple[float, float, float, float, float, float, float]]] = {}
    t_seq: dict[int, list[tuple[int, float, float, float]]] = defaultdict(list)
    sample_areas: list[float] = []
    sample_mids: list[int] = []
    rows_by_frame: dict[int, Mapping[str, Any]] = {}

    for row in rows:
        f = int(row.get("source_frame", row.get("frame", 0)))
        rows_by_frame[f] = row
        d: dict[int, tuple[float, float, float, float, float, float, float]] = {}
        for tid, det in _detection_map(row).items():
            if tid < 0 or not _has_valid_measurement(det)[0]:
                continue
            c = _extract_center_px(det)
            b = _extract_box_px(det)
            if c is None or b is None:
                continue
            cx, cy = float(c[0]), float(c[1])
            b0, b1, b2, b3 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
            area = max(0.0, (b2 - b0) * (b3 - b1))
            d[tid] = (cx, cy, b0, b1, b2, b3, area)
            t_seq[tid].append((f, cx, cy, area))
            if f % 30 == 0 or len(sample_areas) < 50:
                sample_areas.append(area)
            if len(sample_mids) < 8 and tid not in sample_mids:
                sample_mids.append(tid)
        if d:
            fmap[f] = d

    if len(sample_areas) < 20 or len(t_seq) < 2:
        return None

    cm_per_px = _resolve_scale_cm_per_px(rows[: min(len(rows), 200)], sample_mids[:2], binding, scale=scale)
    med_area = float(np.median(sample_areas))
    if med_area <= 1.0:
        return None

    half_w = max(2, int(round(half_w_s * fps)))
    t_first: dict[int, int] = {}
    t_last: dict[int, int] = {}
    vel: dict[tuple[int, int], tuple[float, float, float]] = {}
    for tid, seq in t_seq.items():
        t_first[tid] = seq[0][0]
        t_last[tid] = seq[-1][0]
        n = len(seq)
        j0 = 0
        j1 = 0
        for i in range(n):
            fc = seq[i][0]
            while j0 < i and fc - seq[j0][0] > half_w:
                j0 += 1
            while j1 + 1 < n and seq[j1 + 1][0] - fc <= half_w:
                j1 += 1
            df = seq[j1][0] - seq[j0][0]
            if df <= 0:
                continue
            vx = (seq[j1][1] - seq[j0][1]) * cm_per_px * (fps / df)
            vy = (seq[j1][2] - seq[j0][2]) * cm_per_px * (fps / df)
            sp = math.hypot(vx, vy)
            if sp <= max_speed_cm_s:
                vel[(fc, tid)] = (vx, vy, sp)

    all_frames = sorted(fmap.keys())
    max_frame = int(frame_count) if frame_count is not None else (all_frames[-1] + 1)
    return {
        "info": info,
        "fps": fps,
        "cm_per_px": cm_per_px,
        "frame_count": max_frame,
        "med_area": med_area,
        "fmap": fmap,
        "t_seq": t_seq,
        "t_first": t_first,
        "t_last": t_last,
        "vel": vel,
        "all_frames": all_frames,
        "rows_by_frame": rows_by_frame,
    }


def _track_overlaps_intervals(
    s: int,
    e: int,
    events: Sequence[Mapping[str, Any]],
    behaviors: set[str],
) -> bool:
    for ev in events:
        beh = normalize_behavior_key(str(ev.get("canonical_behavior") or ev.get("behavior") or ""))
        if beh not in behaviors:
            continue
        es, ee = int(ev["start_frame"]), int(ev["end_frame"])
        if not (e <= es or s >= ee):
            return True
    return False


def discover_id_switch_chasing_candidates(
    store: Any,
    binding: Mapping[str, Any] | None = None,
    scale: Mapping[str, Any] | None = None,
    *,
    parameters: HeuristicParameters | None = None,
    existing_events: Sequence[Mapping[str, Any]] = (),
    min_duration_s: float = 0.7,
    max_separation_cm: float = 18.0,
    max_median_separation_cm: float = 14.5,
    max_gap_widening_cm: float = 4.0,
    min_pair_speed_cm_s: float = 30.0,
    min_mean_speed_cm_s: float = 36.0,
    min_both_p25_speed_cm_s: float = 28.0,
    min_both_fast_fraction: float = 0.70,
    both_fast_frame_speed_cm_s: float = 30.0,
    min_both_peak_speed_cm_s: float = 43.5,
    min_net_displacement_cm: float = 25.0,
    min_vel_alignment_cos: float = 0.85,
    min_vel_alignment_p25_cos: float = 0.78,
    min_vel_aligned_fraction: float = 0.80,
    vel_aligned_frame_cos: float = 0.78,
    min_pursuit_cos: float = 0.78,
    min_pursuit_p25_cos: float = 0.65,
    max_trailing_bystander_fraction: float = 0.35,
    _index: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Discover >0.7s fast aligned pursuit bouts spanning YOLO ID switches without relabeling raw IDs."""
    idx = _index or _build_full_store_track_index(store, binding, scale, max_speed_cm_s=175.0)
    if idx is None:
        return []
    info = idx["info"]
    fps = float(idx["fps"])
    cm_per_px = float(idx["cm_per_px"])
    med_area = float(idx["med_area"])
    fmap = idx["fmap"]
    vel = idx["vel"]
    all_frames = idx["all_frames"]
    del parameters
    eff_req_s = max(float(min_duration_s), 0.7)
    frame_min_cv = min(0.78, float(min_vel_alignment_cos))
    frame_min_cp = min(0.65, float(min_pursuit_cos))

    # Collect ALL qualifying fast-pursuit pair-frames (never collapse to 1 pair per frame).
    HitTuple = tuple[int, int, int, float, float, float, float, float, float, float, float, float, float]
    frame_hits: dict[int, list[HitTuple]] = {}
    for f in all_frames:
        fd = fmap[f]
        fast = [
            (t, fd[t], vel[(f, t)])
            for t in fd
            if (f, t) in vel and min_pair_speed_cm_s <= vel[(f, t)][2] <= 170.0 and fd[t][6] < 2.4 * med_area
        ]
        if len(fast) < 2:
            continue
        hits_f: list[HitTuple] = []
        for i in range(len(fast)):
            t1, p1, v1 = fast[i]
            for j in range(i + 1, len(fast)):
                t2, p2, v2 = fast[j]
                dx, dy = p2[0] - p1[0], p2[1] - p1[1]
                dist_px = math.hypot(dx, dy)
                d_cm = dist_px * cm_per_px
                if not (1.0 <= d_cm <= max_separation_cm) or dist_px <= 1e-6:
                    continue
                cv = (v1[0] * v2[0] + v1[1] * v2[1]) / (v1[2] * v2[2])
                if cv < frame_min_cv:
                    continue
                ux, uy = dx / dist_px, dy / dist_px
                cp1 = (v1[0] * ux + v1[1] * uy) / v1[2]
                cp2 = -(v2[0] * ux + v2[1] * uy) / v2[2]
                if cp1 >= cp2:
                    ch, ld, cp, spf, spl = t1, t2, cp1, v1[2], v2[2]
                    cf, cl = p1, p2
                    dvx, dvy = (v1[0] / v1[2] + v2[0] / v2[2]) * 0.5, (v1[1] / v1[2] + v2[1] / v2[2]) * 0.5
                else:
                    ch, ld, cp, spf, spl = t2, t1, cp2, v2[2], v1[2]
                    cf, cl = p2, p1
                    dvx, dvy = (v1[0] / v1[2] + v2[0] / v2[2]) * 0.5, (v1[1] / v1[2] + v2[1] / v2[2]) * 0.5
                if cp < frame_min_cp:
                    continue
                # Intervening 3rd mouse check
                ab2 = dx * dx + dy * dy
                inter = False
                trailing_hit = False
                ux_fl = (cl[0] - cf[0]) / max(1e-6, math.hypot(cl[0] - cf[0], cl[1] - cf[1]))
                uy_fl = (cl[1] - cf[1]) / max(1e-6, math.hypot(cl[0] - cf[0], cl[1] - cf[1]))
                for ot, ov in fd.items():
                    if ot in (ch, ld):
                        continue
                    t_proj = max(0.0, min(1.0, ((ov[0] - p1[0]) * dx + (ov[1] - p1[1]) * dy) / max(1e-6, ab2)))
                    if math.hypot(ov[0] - (p1[0] + t_proj * dx), ov[1] - (p1[1] + t_proj * dy)) * cm_per_px <= 4.0:
                        inter = True
                        break
                    dx_cm = (ov[0] - cf[0]) * cm_per_px
                    dy_cm = (ov[1] - cf[1]) * cm_per_px
                    d_oc = math.hypot(dx_cm, dy_cm)
                    proj_b = dx_cm * ux_fl + dy_cm * uy_fl
                    perp_b = abs(dx_cm * (-uy_fl) + dy_cm * ux_fl)
                    if -13.0 <= proj_b <= 1.0 and perp_b <= 5.5 and d_oc <= 13.5:
                        trailing_hit = True
                if inter:
                    continue
                dn = max(1e-6, math.hypot(dvx, dvy))
                hits_f.append(
                    (
                        f,
                        ch,
                        ld,
                        0.5 * (p1[0] + p2[0]),
                        0.5 * (p1[1] + p2[1]),
                        spf,
                        spl,
                        d_cm,
                        cv,
                        cp,
                        dvx / dn,
                        dvy / dn,
                        1.0 if trailing_hit else 0.0,
                    )
                )
        if hits_f:
            frame_hits[f] = hits_f

    # Chain compatible hits across frames into continuous ID-switch pursuit trajectories
    active_chains: list[list[HitTuple]] = []
    finished_chains: list[list[HitTuple]] = []
    max_gap = max(4, int(round(0.20 * fps)))
    for f in sorted(frame_hits.keys()):
        still_active = []
        for ch_list in active_chains:
            if f - ch_list[-1][0] <= max_gap:
                still_active.append(ch_list)
            else:
                finished_chains.append(ch_list)
        active_chains = still_active
        used_chains: set[int] = set()
        for h in sorted(frame_hits[f], key=lambda item: min(item[5], item[6]), reverse=True):
            best_c_idx = None
            best_score = -1e9
            for ci, ch_list in enumerate(active_chains):
                if ci in used_chains:
                    continue
                lh = ch_list[-1]
                df = f - lh[0]
                if not (0 < df <= max_gap):
                    continue
                step_cm = math.hypot(h[3] - lh[3], h[4] - lh[4]) * cm_per_px
                if step_cm > 20.0:
                    continue
                dir_cos = h[10] * lh[10] + h[11] * lh[11]
                if dir_cos < 0.30:
                    continue
                same_pair_bonus = 50.0 if (h[1], h[2]) == (lh[1], lh[2]) else 0.0
                score = same_pair_bonus - step_cm + 10.0 * dir_cos
                if score > best_score:
                    best_score = score
                    best_c_idx = ci
            if best_c_idx is not None:
                active_chains[best_c_idx].append(h)
                used_chains.add(best_c_idx)
            else:
                active_chains.append([h])
    finished_chains.extend(active_chains)

    out: list[dict[str, Any]] = []
    min_req_frames = max(2, int(np.floor(eff_req_s * fps + 1e-9)) + 1)
    for chain in finished_chains:
        ts = int(chain[0][0])
        te = int(chain[-1][0]) + 1
        span = te - ts
        if span < min_req_frames or (span / fps) <= eff_req_s + 1e-9:
            continue
        if len(chain) < max(14, int(round(0.45 * span))):
            continue
        pairs = {(h[1], h[2]) for h in chain}
        if len(pairs) < 2:
            continue

        spfs = [h[5] for h in chain]
        spls = [h[6] for h in chain]
        mins = [min(sf, sl) for sf, sl in zip(spfs, spls)]
        mean_spf = float(np.mean(spfs))
        mean_spl = float(np.mean(spls))
        p25_both_sp = float(np.percentile(mins, 25))
        both_fast_frac = sum(1 for v in mins if v >= both_fast_frame_speed_cm_s) / max(1, len(mins))
        pk_spf = float(max(spfs))
        pk_spl = float(max(spls))
        ds = [h[7] for h in chain]
        med_d = float(np.median(ds))
        max_d = float(max(ds))
        gap_widen = float(ds[-1] - ds[0])
        cvs = [h[8] for h in chain]
        cps = [h[9] for h in chain]
        mean_cv = float(np.mean(cvs))
        p25_cv = float(np.percentile(cvs, 25))
        vel_aligned_frac = sum(1 for v in cvs if v >= vel_aligned_frame_cos) / max(1, len(cvs))
        mean_cp = float(np.mean(cps))
        p25_cp = float(np.percentile(cps, 25))
        tail_hits = chain[-min(3, len(chain)) :]
        tail_cv = float(np.mean([h[8] for h in tail_hits]))
        tail_cp = float(np.mean([h[9] for h in tail_hits]))
        net_disp = math.hypot(chain[-1][3] - chain[0][3], chain[-1][4] - chain[0][4]) * cm_per_px
        trailing_frac = float(np.mean([h[12] for h in chain]))
        if (
            min(mean_spf, mean_spl) < min_mean_speed_cm_s
            or p25_both_sp < min_both_p25_speed_cm_s
            or both_fast_frac < min_both_fast_fraction - 1e-9
            or min(pk_spf, pk_spl) < min_both_peak_speed_cm_s
            or med_d > max_median_separation_cm
            or gap_widen > max_gap_widening_cm + 1e-9
            or mean_cv < min_vel_alignment_cos
            or p25_cv < min_vel_alignment_p25_cos
            or vel_aligned_frac < min_vel_aligned_fraction - 1e-9
            or mean_cp < min_pursuit_cos
            or p25_cp < min_pursuit_p25_cos
            or tail_cv < 0.75
            or tail_cp < 0.65
            or net_disp < min_net_displacement_cm
            or trailing_frac > max_trailing_bystander_fraction + 1e-9
        ):
            continue

        # Exclude bouts overlapping accepted chases or attacks
        if _track_overlaps_intervals(ts, te, existing_events, {"chasing", "chase", "attack"}):
            continue
        if any(not (te <= int(ev["start_frame"]) or ts >= int(ev["end_frame"])) for ev in out):
            continue

        onset_chaser, onset_chased = int(chain[0][1]), int(chain[0][2])
        end_chaser, end_chased = int(chain[-1][1]), int(chain[-1][2])
        ordered_mids: list[int] = []
        for tid in (onset_chaser, onset_chased, end_chaser, end_chased):
            if tid not in ordered_mids:
                ordered_mids.append(tid)
        for h in chain:
            for tid in (int(h[1]), int(h[2])):
                if tid not in ordered_mids and len(ordered_mids) < 6:
                    ordered_mids.append(tid)
        if len(set(ordered_mids)) < 2:
            continue

        ev: dict[str, Any] = {
            "event_id": f"chase_idsw_{onset_chaser}_{onset_chased}_{ts:08d}_{te:08d}",
            "canonical_behavior": "chase",
            "behavior": "Chasing",
            "mouse_ids": ordered_mids,
            "start_frame": ts,
            "end_frame": te,
            "frame_count": span,
            "duration_s": float(span / fps),
            "start_time_s": float(ts / fps),
            "end_time_s": float(te / fps),
            "chase_min_duration_policy_s": round(eff_req_s, 3),
            "has_visual_id_switch": True,
            "endpoint_observed_track_ids": {
                "start": [onset_chaser, onset_chased],
                "end": [end_chaser, end_chased],
            },
            "observed_onset_pair": [onset_chaser, onset_chased],
            "id_switch_note": "YOLO visual_track_id switched during fast pursuit; raw tracker IDs preserved without re-identification.",
            "discovery_source": "id_switch_fast_pursuit_scan",
            "chase_mean_follower_speed_cm_s": round(mean_spf, 2),
            "chase_mean_leader_speed_cm_s": round(mean_spl, 2),
            "chase_min_mean_speed_cm_s": round(min(mean_spf, mean_spl), 2),
            "chase_both_p25_speed_cm_s": round(p25_both_sp, 2),
            "chase_both_fast_fraction": round(both_fast_frac, 3),
            "chase_peak_follower_speed_cm_s": round(pk_spf, 2),
            "chase_peak_leader_speed_cm_s": round(pk_spl, 2),
            "chase_peak_speed_cm_s": round(max(pk_spf, pk_spl), 2),
            "chase_median_dist_cm": round(med_d, 2),
            "chase_max_dist_cm": round(max_d, 2),
            "chase_gap_widening_cm": round(gap_widen, 2),
            "chase_trail_error_cm": 0.0,
            "chase_intervening_fraction": 0.0,
            "chase_trailing_bystander_fraction": round(trailing_frac, 3),
            "chase_crowd_fraction": 0.0,
            "chase_mean_vel_cos": round(mean_cv, 3),
            "chase_vel_cos_p25": round(p25_cv, 3),
            "chase_vel_aligned_fraction": round(vel_aligned_frac, 3),
            "chase_mean_cos": round(mean_cp, 3),
            "chase_pursuit_cos_p25": round(p25_cp, 3),
            "chase_body_heading_cos": None,
            "chase_min_net_disp_cm": round(net_disp, 2),
            "chase_hit_frames": len(chain),
        }
        onset_fs = {
            int(h[0]) for h in chain if (int(h[1]), int(h[2])) == (onset_chaser, onset_chased)
        }
        onset_supp = _observed_support_intervals(
            np.array([f in onset_fs for f in range(ts, te)], dtype=bool),
            ts,
        ) or [[ts, ts + 1]]
        attach_social_roles(ev, "chase", onset_chaser, onset_chased, onset_supp)
        if not endpoints_fully_observed(store, ev, info):
            continue
        out.append(ev)

    return sorted(out, key=lambda e: int(e["start_frame"]))


def discover_fight_ball_attack_episodes(
    store: Any,
    binding: Mapping[str, Any] | None = None,
    scale: Mapping[str, Any] | None = None,
    *,
    parameters: HeuristicParameters | None = None,
    existing_events: Sequence[Mapping[str, Any]] = (),
    min_duration_s: float = 0.5,
    min_area_ratio: float = 2.15,
    min_huge_box_frames: int = 4,
    min_huge_peak_speed_cm_s: float = 42.0,
    min_id_churn: int = 4,
    max_duration_s: float = 4.5,
    _index: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Discover >0.5s merged-box/ID-switched Attack fights while preserving raw visual_track_ids."""
    idx = _index or _build_full_store_track_index(store, binding, scale, max_speed_cm_s=240.0)
    if idx is None:
        return []
    info = idx["info"]
    fps = float(idx["fps"])
    cm_per_px = float(idx["cm_per_px"])
    n_f = int(idx["frame_count"])
    med_area = float(idx["med_area"])
    fmap = idx["fmap"]
    t_seq = idx["t_seq"]
    t_first = idx["t_first"]
    t_last = idx["t_last"]
    vel = idx["vel"]
    all_frames = idx["all_frames"]
    rows_by_frame = idx["rows_by_frame"]
    params = _resolve_parameters(binding, parameters)
    eff_req_s = max(float(min_duration_s), float(resolve_duration_policy("attack", parameters=params).required_s))
    max_span_frames = max(int(round(eff_req_s * fps)) + 4, int(round(max_duration_s * fps)))

    # 1. Seed frames where a merged detection expands to >= min_area_ratio * median single-mouse box area.
    clusters: list[dict[str, Any]] = []
    for f in all_frames:
        fd = fmap[f]
        for tid, v in fd.items():
            ar_rat = v[6] / med_area
            if ar_rat < min_area_ratio:
                continue
            x, y = v[0], v[1]
            sp = vel.get((f, tid), (0.0, 0.0, 0.0))[2]
            merged = False
            for cl in reversed(clusters[-12:]):
                if f - cl["f_max"] <= 25 and math.hypot(x - cl["cx"], y - cl["cy"]) * cm_per_px <= 16.0:
                    cl["hits"].append((f, tid, x, y, ar_rat, sp))
                    cl["f_max"] = max(cl["f_max"], f)
                    cl["cx"] = 0.75 * cl["cx"] + 0.25 * x
                    cl["cy"] = 0.75 * cl["cy"] + 0.25 * y
                    merged = True
                    break
            if not merged:
                clusters.append({"f_min": f, "f_max": f, "cx": x, "cy": y, "hits": [(f, tid, x, y, ar_rat, sp)]})

    raw_scored: list[tuple[float, dict[str, Any], float, float]] = []
    pre_pad = max(6, int(round(0.60 * fps)))
    post_pad = max(6, int(round(0.60 * fps)))
    search_pad = max(8, int(round(0.80 * fps)))

    for cl in clusters:
        hits = cl["hits"]
        n_huge = len(hits)
        if n_huge < min_huge_box_frames:
            continue
        max_ar = max(h[4] for h in hits)
        h_pk = max(h[5] for h in hits)
        if h_pk < min_huge_peak_speed_cm_s:
            continue

        s0 = max(0, int(cl["f_min"]) - pre_pad)
        e0 = min(n_f - 1, int(cl["f_max"]) + post_pad)
        if (e0 - s0 + 1) / fps <= eff_req_s + 1e-9:
            continue

        cx = float(np.median([h[2] for h in hits]))
        cy = float(np.median([h[3] for h in hits]))
        hit_pts = [(h[0], h[2], h[3]) for h in hits]

        # Count stationary corner/sleep bystanders within 10.5 cm.
        cand_tids = {t for f in range(s0, e0 + 1, 2) for t in fmap.get(f, ())}
        stat_by = 0
        for tid in cand_tids:
            in_w = [p for p in t_seq.get(tid, ()) if s0 <= p[0] <= e0]
            if len(in_w) < max(6, int(0.45 * (e0 - s0 + 1))):
                continue
            d_med = float(np.median([math.hypot(p[1] - cx, p[2] - cy) * cm_per_px for p in in_w]))
            sp_med = float(np.median([vel.get((p[0], tid), (0.0, 0.0, 0.0))[2] for p in in_w]))
            if d_med <= 10.5 and sp_med < 6.0:
                stat_by += 1

        # Gather core fight tracks entering <= 9.5 cm of the time-local merged ball.
        core_tids: set[int] = {h[1] for h in hits}
        w_lo = max(0, int(cl["f_min"]) - search_pad)
        w_hi = min(n_f - 1, int(cl["f_max"]) + search_pad)
        for f in range(w_lo, w_hi + 1):
            _, hx, hy = min(hit_pts, key=lambda p: abs(p[0] - f))
            for tid, v in fmap.get(f, {}).items():
                if math.hypot(v[0] - hx, v[1] - hy) * cm_per_px <= 9.5:
                    core_tids.add(tid)

        active_core = {
            tid
            for tid in core_tids
            if any(
                vel.get((p[0], tid), (0.0, 0.0, 0.0))[2] >= 14.0 or (p[3] / med_area) >= 2.0
                for p in t_seq.get(tid, ())
                if max(0, s0 - 6) <= p[0] <= min(n_f - 1, e0 + 6)
            )
        }
        if len(active_core) < 3:
            continue

        births = sum(1 for t in active_core if (s0 - 4) <= t_first[t] <= (e0 + 4))
        deaths = sum(1 for t in active_core if (s0 - 4) <= t_last[t] <= (e0 + 4))
        churn = births + deaths
        if churn < min_id_churn:
            continue

        huddle_ok = (stat_by == 0) or (stat_by == 1 and max_ar >= 3.5 and churn >= 6)
        if not huddle_ok:
            continue

        # Measure core speed, centroid tumble path, box IoU, and find onset/breakout simultaneous pairs.
        c_max_sps: list[float] = []
        core_cent: list[tuple[float, float]] = []
        max_iou = 0.0
        sim_pairs: list[tuple[int, float, int, int]] = []
        any_obs_frames: list[int] = []
        occ_frames = 0
        crowd_frames = 0

        for f in range(max(0, s0 - 6), min(n_f, e0 + 7)):
            fd = fmap.get(f, {})
            present = [t for t in active_core if t in fd]
            if s0 <= f <= e0:
                if present:
                    any_obs_frames.append(f)
                    c_max_sps.append(max(vel.get((f, t), (0.0, 0.0, 0.0))[2] for t in present))
                    core_cent.append((sum(fd[t][0] for t in present) / len(present), sum(fd[t][1] for t in present) / len(present)))
                if len(present) <= 1 and (int(cl["f_min"]) - 6) <= f <= (int(cl["f_max"]) + 6):
                    occ_frames += 1
                # Crowding inside 11 cm of core centroid
                if present:
                    mx_f = sum(fd[t][0] for t in present) / len(present)
                    my_f = sum(fd[t][1] for t in present) / len(present)
                    if sum(1 for ot, ov in fd.items() if ot not in active_core and math.hypot(ov[0] - mx_f, ov[1] - my_f) * cm_per_px <= 11.0) >= 2:
                        crowd_frames += 1
            if len(present) >= 2:
                best_p = min(
                    (
                        (math.hypot(fd[a][0] - fd[b][0], fd[a][1] - fd[b][1]) * cm_per_px, a, b)
                        for i, a in enumerate(present)
                        for b in present[i + 1 :]
                    ),
                    key=lambda x: x[0],
                )
                if best_p[0] <= 18.0:
                    sim_pairs.append((f, best_p[0], best_p[1], best_p[2]))
                for i, a in enumerate(present):
                    for b in present[i + 1 :]:
                        max_iou = max(max_iou, _box_iou(np.array(fd[a][2:6]), np.array(fd[b][2:6])))

        if not sim_pairs or len(any_obs_frames) < 4:
            continue

        c_pk = max(c_max_sps) if c_max_sps else 0.0
        c_mean = float(np.mean(c_max_sps)) if c_max_sps else 0.0
        t_path = sum(
            math.hypot(core_cent[i][0] - core_cent[i - 1][0], core_cent[i][1] - core_cent[i - 1][1]) * cm_per_px
            for i in range(1, len(core_cent))
        )

        kinetic_ok = (
            max_ar >= 3.2 and h_pk >= min_huge_peak_speed_cm_s and c_pk >= 48.0 and c_mean >= 20.0 and t_path >= 35.0
        ) or (
            max_ar >= min_area_ratio
            and max(h_pk, c_pk) >= 64.0
            and h_pk >= min_huge_peak_speed_cm_s
            and c_mean >= 22.0
            and t_path >= 35.0
        )
        if not kinetic_ok:
            continue

        # Tighten [ts, te) around [f_min, f_max] with observed onset & breakout endpoints.
        onset_candidates = [sp for sp in sim_pairs if max(0, int(cl["f_min"]) - pre_pad) <= sp[0] <= int(cl["f_min"]) + 6]
        if onset_candidates:
            onset_rec = onset_candidates[0]
        else:
            onset_rec = min(sim_pairs, key=lambda sp: abs(sp[0] - int(cl["f_min"])))
        ts = min(onset_rec[0], int(cl["f_min"]))
        if ts < any_obs_frames[0]:
            ts = any_obs_frames[0]

        end_candidates = [sp for sp in sim_pairs if int(cl["f_max"]) <= sp[0] <= min(n_f - 1, int(cl["f_max"]) + post_pad)]
        te = (end_candidates[-1][0] + 1) if end_candidates else (any_obs_frames[-1] + 1)
        if te - ts > max_span_frames:
            mid_core = int(round(0.5 * (int(cl["f_min"]) + int(cl["f_max"]))))
            ts = max(ts, mid_core - max_span_frames // 2)
            te = min(te, ts + max_span_frames)
            obs_sub = [f for f in any_obs_frames if ts <= f < te]
            if len(obs_sub) < 4:
                continue
            ts, te = obs_sub[0], obs_sub[-1] + 1

        span = te - ts
        if (span / fps) <= eff_req_s + 1e-9:
            continue

        start_present = [t for t in active_core if t in fmap.get(ts, {})]
        end_present = [t for t in active_core if t in fmap.get(te - 1, {})]
        if not start_present or not end_present:
            continue
        start_present.sort(key=lambda t: math.hypot(fmap[ts][t][0] - cx, fmap[ts][t][1] - cy))
        end_present.sort(key=lambda t: math.hypot(fmap[te - 1][t][0] - cx, fmap[te - 1][t][1] - cy))
        start_obs_ids = start_present[:2]
        end_obs_ids = end_present[:2]

        # Order participants by frequency inside [ts, te) with onset pair first.
        onset_f, pair_a, pair_b = onset_rec[0], int(onset_rec[2]), int(onset_rec[3])
        if onset_f > ts and len(start_present) >= 2:
            pair_a, pair_b = start_present[0], start_present[1]
            onset_f = ts
        score_a = 0.0
        score_b = 0.0
        for f in range(max(0, onset_f - 6), min(n_f, max(onset_f + 8, int(cl["f_min"]) + 8))):
            pa_tup = fmap.get(f, {}).get(pair_a)
            pb_tup = fmap.get(f, {}).get(pair_b)
            if pa_tup is None or pb_tup is None:
                continue
            dx, dy = pb_tup[0] - pa_tup[0], pb_tup[1] - pa_tup[1]
            nr = math.hypot(dx, dy)
            if nr > 1.0:
                ux, uy = dx / nr, dy / nr
                va = vel.get((f, pair_a))
                vb = vel.get((f, pair_b))
                if va is not None:
                    score_a += float(va[0] * ux + va[1] * uy)
                if vb is not None:
                    score_b += float(vb[0] * (-ux) + vb[1] * (-uy))
            row = rows_by_frame.get(f)
            if row is not None:
                dm = _detection_map(row)
                ka = _extract_keypoints_px(dm[pair_a]) if pair_a in dm else None
                kb = _extract_keypoints_px(dm[pair_b]) if pair_b in dm else None
                if ka is not None and np.isfinite(ka[0]).all():
                    da_sn = math.hypot(float(ka[0][0]) - pb_tup[0], float(ka[0][1]) - pb_tup[1]) * cm_per_px
                    if da_sn <= 5.5:
                        score_a += 10.0 * max(0.0, 5.5 - da_sn)
                if kb is not None and np.isfinite(kb[0]).all():
                    db_sn = math.hypot(float(kb[0][0]) - pa_tup[0], float(kb[0][1]) - pa_tup[1]) * cm_per_px
                    if db_sn <= 5.5:
                        score_b += 10.0 * max(0.0, 5.5 - db_sn)

        attacker_id, victim_id = (pair_a, pair_b) if score_a >= score_b else (pair_b, pair_a)
        freq_counts = {t: sum(1 for f in range(ts, te) if t in fmap.get(f, ())) for t in active_core}
        ordered_mids = []
        for t in (attacker_id, victim_id, *start_obs_ids, *end_obs_ids):
            if t not in ordered_mids:
                ordered_mids.append(t)
        for t in sorted(active_core, key=lambda x: (-freq_counts[x], x)):
            if t not in ordered_mids and len(ordered_mids) < 6:
                ordered_mids.append(t)
        if len(set(ordered_mids)) < 2:
            continue

        if _track_overlaps_intervals(ts, te, existing_events, {"attack"}):
            continue

        onset_supp_mask = np.array(
            [1 if (attacker_id in fmap.get(f, ()) and victim_id in fmap.get(f, ())) else 0 for f in range(ts, te)],
            dtype=bool,
        )
        clamped_f = max(ts, min(te - 1, onset_f))
        onset_supp = _observed_support_intervals(onset_supp_mask, ts) or [[clamped_f, clamped_f + 1]]

        ev: dict[str, Any] = {
            "event_id": f"attack_fightball_{attacker_id}_{victim_id}_{ts:08d}_{te:08d}",
            "canonical_behavior": "attack",
            "behavior": "Attack",
            "mouse_ids": ordered_mids,
            "start_frame": int(ts),
            "end_frame": int(te),
            "frame_count": int(span),
            "duration_s": float(span / fps),
            "start_time_s": float(ts / fps),
            "end_time_s": float(te / fps),
            "has_visual_id_switch": True,
            "endpoint_observed_track_ids": {
                "start": start_obs_ids,
                "end": end_obs_ids,
            },
            "observed_onset_pair": [attacker_id, victim_id],
            "id_switch_note": "YOLO visual_track_id switched/merged during clinch; raw tracker IDs preserved without re-identification.",
            "discovery_source": "fight_ball_id_switch_scan",
            "attack_peak_speed_cm_s": round(float(max(h_pk, c_pk)), 2),
            "attack_both_peak_speed_cm_s": round(float(c_pk), 2),
            "attack_mean_speed_cm_s": round(float(c_mean), 2),
            "attack_close_mean_speed_cm_s": round(float(c_mean), 2),
            "attack_max_area_ratio": round(float(max_ar), 2),
            "attack_huge_box_frames": int(n_huge),
            "attack_id_churn": int(churn),
            "attack_stationary_bystanders": int(stat_by),
            "attack_tumble_path_cm": round(float(t_path), 1),
            "attack_max_iou": round(float(max_iou), 3),
            "attack_clinch_occlusion_frames": int(occ_frames),
            "attack_adjacent_occlusion_frames": 0,
            "attack_engaged_frames": int(len(any_obs_frames)),
            "attack_close_frames": int(len(any_obs_frames)),
            "attack_mid_path_cm": round(float(t_path), 2),
            "attack_crowd_fraction": round(float(crowd_frames / max(1, span)), 3),
            "attack_min_dist_cm": 0.0,
            "attack_min_box_gap_cm": 0.0,
            "attack_max_raw1f_speed_cm_s": round(float(max(h_pk, c_pk)), 2),
            "attack_active_frames": int(len(any_obs_frames)),
        }
        attach_social_roles(ev, "attack", attacker_id, victim_id, onset_supp)
        if not endpoints_fully_observed(store, ev, info):
            continue
        qual = max_ar * 12.0 + n_huge * 2.0 + max(h_pk, c_pk) * 0.25 + churn * 3.0
        raw_scored.append((qual, ev, cx, cy))

    # Deduplicate overlapping fight-ball discoveries nearby in space, keeping higher score first.
    raw_scored.sort(key=lambda item: item[0], reverse=True)
    accepted: list[tuple[dict[str, Any], float, float]] = []
    for _, ev, cx, cy in raw_scored:
        s_ev, e_ev = int(ev["start_frame"]), int(ev["end_frame"])
        clash_dup = any(
            not (e_ev <= int(old["start_frame"]) or s_ev >= int(old["end_frame"]))
            and math.hypot(cx - ox, cy - oy) * cm_per_px <= 25.0
            for old, ox, oy in accepted
        )
        if not clash_dup:
            accepted.append((ev, cx, cy))

    return sorted([item[0] for item in accepted], key=lambda e: int(e["start_frame"]))

