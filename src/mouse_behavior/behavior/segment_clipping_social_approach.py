"""Approach behavior segment clipping, zero-contact & outside-crowd filtering."""
from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from mouse_behavior.behavior.duration_policy import qualify_event_duration
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
    event_participant_ids,
    normalize_behavior_key,
    update_segment_bounds,
)
from mouse_behavior.behavior.segment_clipping_social_contact import (
    _box_gap_cm,
    _directed_head_distance_cm,
    attach_social_roles,
)
from mouse_behavior.evaluation.exported_dataset_models import HeuristicParameters

__all__ = ["_smoothed_track_kinematics", "refine_approach_segments"]


class _StoreWindowAdapter:
    """Adapt `.rows(..., rebase=True)` stores to provide `.window(start, end)`."""

    def __init__(self, store: Any):
        self._store = store

    def rows(self, start: int, end: int, *, rebase: bool = False):
        return self._store.rows(start, end, rebase=rebase)

    def window(self, start: int, end: int) -> list[dict[str, Any]]:
        return list(self._store.rows(start, end, rebase=True))


def _ordered_pair_ids(event: Mapping[str, Any]) -> list[int]:
    """Return ordered distinct `[actor_or_first, target_or_second]` participant IDs."""
    raw: list[int] = []
    for key in ("actor_id", "target_id"):
        val = event.get(key)
        if val is not None and not isinstance(val, bool) and int(val) >= 0:
            raw.append(int(val))
    for mid in event.get("mouse_ids", ()):
        if not isinstance(mid, bool) and int(mid) >= 0:
            raw.append(int(mid))
    return list(dict.fromkeys(raw))


def _pair_snout_contact_cm(
    kp_a_cm: np.ndarray | None,
    kp_b_cm: np.ndarray | None,
) -> float | None:
    """Return smallest directed nose-to-head/tail distance in cm when measurable."""
    if kp_a_cm is None or kp_b_cm is None:
        return None
    dists: list[float] = []
    for actor_kp, target_kp in ((kp_a_cm, kp_b_cm), (kp_b_cm, kp_a_cm)):
        d_head = _directed_head_distance_cm(actor_kp, target_kp)
        if np.isfinite(d_head):
            dists.append(float(d_head))
        if np.isfinite(actor_kp[0]).all() and np.isfinite(target_kp[6]).all():
            dists.append(float(np.linalg.norm(actor_kp[0] - target_kp[6])))
    return min(dists) if dists else None


def _smoothed_track_kinematics(
    pos: Mapping[int, np.ndarray | None],
    frames: Sequence[int],
    fps: float,
    cm_per_px: float,
    *,
    half_w_s: float = 0.10,
) -> tuple[dict[int, np.ndarray], dict[int, float], dict[int, float]]:
    """Compute centered smoothed velocities (cm/s) and raw 1-frame step speeds (cm/s)."""
    half_w = max(2, int(round(half_w_s * fps)))
    vel: dict[int, np.ndarray] = {}
    sm_speed: dict[int, float] = {}
    raw1f: dict[int, float] = {}
    for f in frames:
        p = pos.get(f)
        if p is None:
            raw1f[f] = 0.0
            sm_speed[f] = 0.0
            continue
        pp = pos.get(f - 1)
        if pp is not None:
            raw1f[f] = float(np.linalg.norm(p - pp)) * fps * cm_per_px
        elif f == 0 and pos.get(1) is not None:
            raw1f[0] = float(np.linalg.norm(pos[1] - p)) * fps * cm_per_px
        else:
            raw1f[f] = 0.0

        j0, j1 = f - half_w, f + half_w
        p0, p1 = pos.get(j0), pos.get(j1)
        if p0 is None:
            for k in range(f - half_w, f + 1):
                if pos.get(k) is not None:
                    j0, p0 = k, pos[k]
                    break
        if p1 is None:
            for k in range(f + half_w, f - 1, -1):
                if pos.get(k) is not None:
                    j1, p1 = k, pos[k]
                    break
        if p0 is not None and p1 is not None and j1 > j0:
            v = (p1 - p0) * (fps / (j1 - j0)) * cm_per_px
            vel[f] = v
            sm_speed[f] = float(np.linalg.norm(v))
        else:
            sm_speed[f] = 0.0
    return vel, sm_speed, raw1f


def _analyze_approach_scene(
    rows: Sequence[Mapping[str, Any]],
    mid_a: int,
    mid_b: int,
    fps: float,
    cm_per_px: float,
    *,
    min_pair_box_gap_cm: float = 0.35,
    min_pair_center_dist_cm: float = 5.0,
    min_kp_contact_gap_cm: float = 1.6,
    min_other_box_gap_cm: float = 1.0,
    single_neighbor_min_dist_cm: float = 10.0,
    crowd_radius_cm: float = 15.0,
    crowd_max_neighbors: int = 1,
) -> tuple[
    dict[int, np.ndarray | None],
    dict[int, np.ndarray | None],
    dict[int, float],
    dict[int, np.ndarray],
    dict[int, float],
    dict[int, float],
    dict[int, float],
    dict[int, float],
    dict[int, float],
    dict[int, float],
    dict[int, bool],
    dict[int, bool],
    dict[int, bool],
]:
    """Extract pair smoothed kinematics alongside strict zero-contact and outside-crowd scene checks."""
    frame_map = {int(r.get("frame", r.get("source_frame", 0))): r for r in rows}
    all_frames = sorted(frame_map)

    pos_a: dict[int, np.ndarray | None] = {}
    pos_b: dict[int, np.ndarray | None] = {}
    dist_cm: dict[int, float] = {}
    pair_box_gap_cm: dict[int, float] = {}
    min_other_dist_cm: dict[int, float] = {}
    co_obs: dict[int, bool] = {}
    clean_frame: dict[int, bool] = {}
    crowd_ok_map: dict[int, bool] = {}

    for f in all_frames:
        dmap = _detection_map(frame_map[f])
        da = dmap.get(mid_a)
        db = dmap.get(mid_b)
        ca = _extract_center_px(da) if da is not None and _has_valid_measurement(da)[0] else None
        cb = _extract_center_px(db) if db is not None and _has_valid_measurement(db)[0] else None
        pos_a[f] = ca
        pos_b[f] = cb
        if ca is None or cb is None or da is None or db is None:
            co_obs[f] = False
            clean_frame[f] = False
            crowd_ok_map[f] = False
            continue

        co_obs[f] = True
        d_ab = math.hypot(float(ca[0] - cb[0]), float(ca[1] - cb[1])) * cm_per_px
        dist_cm[f] = d_ab

        ba = _extract_box_px(da)
        bb = _extract_box_px(db)
        if ba is not None and bb is not None:
            bgap = _box_gap_cm(ba, bb, cm_per_px)
        else:
            bgap = max(0.0, d_ab - 4.0)
        pair_box_gap_cm[f] = bgap

        kpa = _extract_keypoints_px(da)
        kpb = _extract_keypoints_px(db)
        kp_min = _pair_snout_contact_cm(
            kpa * cm_per_px if kpa is not None else None,
            kpb * cm_per_px if kpb is not None else None,
        )

        contact_ok = (bgap >= min_pair_box_gap_cm) and (d_ab >= min_pair_center_dist_cm)
        if kp_min is not None and kp_min < min_kp_contact_gap_cm and d_ab < 8.5:
            contact_ok = False

        # Outside-crowd and third-mouse non-contact guard across all other mice in the frame.
        other_min_d = 999.0
        other_touch = False
        near_a = 0
        near_b = 0
        ca0, ca1 = float(ca[0]), float(ca[1])
        cb0, cb1 = float(cb[0]), float(cb[1])
        for oid, det_o in dmap.items():
            if oid in (mid_a, mid_b):
                continue
            co_o = _extract_center_px(det_o)
            if co_o is None:
                continue
            ox, oy = float(co_o[0]), float(co_o[1])
            d_ao = math.hypot(ca0 - ox, ca1 - oy) * cm_per_px
            d_bo = math.hypot(cb0 - ox, cb1 - oy) * cm_per_px
            other_min_d = min(other_min_d, d_ao, d_bo)
            if d_ao <= crowd_radius_cm:
                near_a += 1
            if d_bo <= crowd_radius_cm:
                near_b += 1
            bo = _extract_box_px(det_o)
            if bo is not None:
                if ba is not None and _box_gap_cm(ba, bo, cm_per_px) < min_other_box_gap_cm:
                    other_touch = True
                if bb is not None and _box_gap_cm(bb, bo, cm_per_px) < min_other_box_gap_cm:
                    other_touch = True
            if d_ao < single_neighbor_min_dist_cm or d_bo < single_neighbor_min_dist_cm:
                other_touch = True

        min_other_dist_cm[f] = other_min_d
        crowd_ok = (not other_touch) and (near_a <= crowd_max_neighbors) and (near_b <= crowd_max_neighbors)
        crowd_ok_map[f] = crowd_ok
        clean_frame[f] = bool(contact_ok and crowd_ok)

    vel_a, sm_a, raw1f_a = _smoothed_track_kinematics(pos_a, all_frames, fps, cm_per_px, half_w_s=0.10)
    vel_b, sm_b, raw1f_b = _smoothed_track_kinematics(pos_b, all_frames, fps, cm_per_px, half_w_s=0.10)
    return (
        pos_a,
        pos_b,
        dist_cm,
        vel_a,
        vel_b,
        sm_a,
        sm_b,
        raw1f_a,
        raw1f_b,
        pair_box_gap_cm,
        min_other_dist_cm,
        co_obs,
        clean_frame,
        crowd_ok_map,
    )


def _pair_crowd_flags(
    rows: Sequence[Mapping[str, Any]],
    mid_a: int,
    mid_b: int,
    cm_per_px: float,
    *,
    min_other_box_gap_cm: float = 1.0,
    single_neighbor_min_dist_cm: float = 10.0,
    crowd_radius_cm: float = 15.0,
    crowd_max_neighbors: int = 1,
) -> tuple[dict[int, bool], dict[int, float]]:
    """Return outside-crowd suitability per frame separately from pair box touch."""
    crowd_ok: dict[int, bool] = {}
    min_other: dict[int, float] = {}
    for row in rows:
        f = int(row.get("frame", row.get("source_frame", 0)))
        dmap = _detection_map(row)
        da = dmap.get(mid_a)
        db = dmap.get(mid_b)
        if da is None or db is None:
            crowd_ok[f] = False
            continue
        ca = _extract_center_px(da) if _has_valid_measurement(da)[0] else None
        cb = _extract_center_px(db) if _has_valid_measurement(db)[0] else None
        if ca is None or cb is None:
            crowd_ok[f] = False
            continue
        ba = _extract_box_px(da)
        bb = _extract_box_px(db)
        other_min_d = 999.0
        other_touch = False
        near_a = 0
        near_b = 0
        for oid, det_o in dmap.items():
            if oid in (mid_a, mid_b):
                continue
            co_o = _extract_center_px(det_o)
            if co_o is None:
                continue
            d_ao = float(np.linalg.norm(ca - co_o)) * cm_per_px
            d_bo = float(np.linalg.norm(cb - co_o)) * cm_per_px
            other_min_d = min(other_min_d, d_ao, d_bo)
            if d_ao <= crowd_radius_cm:
                near_a += 1
            if d_bo <= crowd_radius_cm:
                near_b += 1
            bo = _extract_box_px(det_o)
            if bo is not None:
                if ba is not None and _box_gap_cm(ba, bo, cm_per_px) < min_other_box_gap_cm:
                    other_touch = True
                if bb is not None and _box_gap_cm(bb, bo, cm_per_px) < min_other_box_gap_cm:
                    other_touch = True
            if d_ao < single_neighbor_min_dist_cm or d_bo < single_neighbor_min_dist_cm:
                other_touch = True
        min_other[f] = other_min_d
        crowd_ok[f] = (not other_touch) and (near_a <= crowd_max_neighbors) and (near_b <= crowd_max_neighbors)
    return crowd_ok, min_other


def _actor_toward_cosine(
    pos_a: Mapping[int, np.ndarray | None],
    pos_b: Mapping[int, np.ndarray | None],
    vel_a: Mapping[int, np.ndarray],
    sm_a: Mapping[int, float],
    f: int,
) -> float:
    pa = pos_a.get(f)
    pb = pos_b.get(f)
    va = vel_a.get(f)
    sa = sm_a.get(f, 0.0)
    if pa is None or pb is None or va is None or sa <= 1e-4:
        return 0.0
    rel = pb - pa
    norm_rel = float(np.linalg.norm(rel))
    if norm_rel <= 1e-4:
        return 0.0
    return float(np.dot(va, rel) / (sa * norm_rel))


def _compute_approach_final_directionality(
    mid_actor: int,
    pos_actor: Mapping[int, np.ndarray | None],
    pos_target: Mapping[int, np.ndarray | None],
    vel_actor: Mapping[int, np.ndarray],
    vel_target: Mapping[int, np.ndarray] | None,
    sm_actor: Mapping[int, float],
    sm_target: Mapping[int, float],
    dist_cm: Mapping[int, float],
    co_obs: Mapping[int, bool],
    ts: int,
    te: int,
    fps: float,
    *,
    rows_by_frame: Mapping[int, Mapping[str, Any]] | None = None,
    cm_per_px: float | None = None,
) -> dict[str, float | None]:
    """Measure final-phase (last 0.40 s) velocity pointing, head pointing, and asymmetry."""
    tail_win = max(4, int(round(0.40 * fps)))
    tail_fs = [f for f in range(max(ts, te - tail_win), te) if co_obs.get(f, False)]
    if not tail_fs:
        return {
            "end_actor_proj_cm_s": 0.0,
            "end_target_proj_cm_s": 0.0,
            "end_actor_dominance_cm_s": 0.0,
            "end_toward_cos": 0.0,
            "end_closing_rate_cm_s": 0.0,
            "end_heading_cos": None,
            "end_body_axis_cos": None,
            "end_snout_tail_adv_cm": None,
        }

    actor_projs: list[float] = []
    target_projs: list[float] = []
    actor_coss: list[float] = []
    for f in tail_fs:
        ca = _actor_toward_cosine(pos_actor, pos_target, vel_actor, sm_actor, f)
        sa = sm_actor.get(f, 0.0)
        actor_projs.append(ca * sa)
        if sa >= 2.0:
            actor_coss.append(ca)
        if vel_target is not None:
            cb = _actor_toward_cosine(pos_target, pos_actor, vel_target, sm_target, f)
            target_projs.append(cb * sm_target.get(f, 0.0))

    end_actor_proj = float(np.mean(actor_projs)) if actor_projs else 0.0
    end_target_proj = float(np.mean(target_projs)) if target_projs else 0.0
    end_actor_dom = end_actor_proj - max(0.0, end_target_proj)
    end_toward_cos = float(np.mean(actor_coss)) if actor_coss else 0.0

    if len(tail_fs) >= 2 and tail_fs[-1] > tail_fs[0] and tail_fs[0] in dist_cm and tail_fs[-1] in dist_cm:
        end_closing = float((dist_cm[tail_fs[0]] - dist_cm[tail_fs[-1]]) / ((tail_fs[-1] - tail_fs[0]) / fps))
    else:
        end_closing = 0.0

    hcos_vals: list[float] = []
    baxis_vals: list[float] = []
    adv_cm_vals: list[float] = []
    if rows_by_frame is not None and cm_per_px is not None:
        for f in tail_fs:
            row = rows_by_frame.get(f)
            pt = pos_target.get(f)
            if row is None or pt is None:
                continue
            da = _detection_map(row).get(mid_actor)
            if da is None:
                continue
            kpa = _extract_keypoints_px(da)
            if kpa is None:
                continue
            nose = kpa[0] if np.isfinite(kpa[0]).all() else None
            neck = kpa[3] if np.isfinite(kpa[3]).all() else None
            tail = kpa[6] if np.isfinite(kpa[6]).all() else None
            if nose is not None and neck is not None:
                hv = nose - neck
                rel = pt - neck
                nh, nr = float(np.linalg.norm(hv)), float(np.linalg.norm(rel))
                if nh > 1e-3 and nr > 1e-3:
                    hcos_vals.append(float(np.dot(hv, rel) / (nh * nr)))
            if nose is not None and tail is not None:
                blen = float(np.linalg.norm(nose - tail))
                adv_px = float(np.linalg.norm(tail - pt)) - float(np.linalg.norm(nose - pt))
                adv_cm_vals.append(adv_px * float(cm_per_px))
                if blen > 1e-3:
                    baxis_vals.append(adv_px / blen)

    return {
        "end_actor_proj_cm_s": end_actor_proj,
        "end_target_proj_cm_s": end_target_proj,
        "end_actor_dominance_cm_s": end_actor_dom,
        "end_toward_cos": end_toward_cos,
        "end_closing_rate_cm_s": end_closing,
        "end_heading_cos": float(np.mean(hcos_vals)) if hcos_vals else None,
        "end_body_axis_cos": float(np.mean(baxis_vals)) if baxis_vals else None,
        "end_snout_tail_adv_cm": float(np.mean(adv_cm_vals)) if adv_cm_vals else None,
    }


def _validate_approach_span(
    event: Mapping[str, Any],
    cand_s: int,
    cand_e: int,
    store: Any,
    video_info: Mapping[str, Any],
    fps: float,
    params: HeuristicParameters,
    mid_actor: int,
    mid_target: int,
    pos_actor: Mapping[int, np.ndarray | None],
    pos_target: Mapping[int, np.ndarray | None],
    dist_cm: Mapping[int, float],
    vel_actor: Mapping[int, np.ndarray],
    sm_actor: Mapping[int, float],
    sm_target: Mapping[int, float],
    raw1f_a: Mapping[int, float],
    raw1f_b: Mapping[int, float],
    pair_box_gap_cm: Mapping[int, float],
    min_other_dist_cm: Mapping[int, float],
    co_obs: Mapping[int, bool],
    crowd_ok: Mapping[int, bool],
    *,
    vel_target: Mapping[int, np.ndarray] | None = None,
    rows_by_frame: Mapping[int, Mapping[str, Any]] | None = None,
    cm_per_px: float | None = None,
    min_duration_s: float = 1.0,
    require_box_touch: bool = True,
    max_touch_box_gap_cm: float = 0.15,
    min_start_dist_cm: float = 20.0,
    min_closing_cm: float = 10.0,
    min_actor_speed_cm_s: float = 8.0,
    min_toward_cos: float = 0.40,
    max_jump_speed_cm_s: float = 120.0,
    min_end_proj_toward_cm_s: float = 7.0,
    min_end_toward_cos: float = 0.82,
    min_end_closing_rate_cm_s: float = 7.0,
    min_end_actor_dominance_cm_s: float = 3.0,
    min_end_heading_cos: float = 0.82,
    min_end_body_axis_cos: float = 0.72,
    min_end_snout_tail_adv_cm: float = 3.0,
) -> tuple[dict[str, Any] | None, str]:
    """Validate >=1.0s non-contact outside-crowd Approach ending just when boxes touch."""
    if cand_e <= cand_s:
        return None, "rejected_below_min_duration"

    obs_frames = [f for f in range(cand_s, cand_e) if co_obs.get(f, False)]
    if not obs_frames:
        return None, "rejected_incomplete_endpoints"

    ts, te = obs_frames[0], obs_frames[-1] + 1
    span = te - ts
    obs_in_span = [f for f in range(ts, te) if co_obs.get(f, False)]
    if len(obs_in_span) / max(1, span) < 0.85:
        return None, "rejected_incomplete_endpoints"

    if any(not crowd_ok.get(f, False) for f in range(ts, te)):
        return None, "rejected_contact_or_crowd"

    pre_touch_fs = [f for f in range(ts, te - 1) if co_obs.get(f, False)]
    if not pre_touch_fs:
        return None, "rejected_below_min_duration"
    if any(pair_box_gap_cm.get(f, 0.0) <= 0.0 or dist_cm.get(f, 0.0) < 4.5 for f in pre_touch_fs):
        return None, "rejected_contact_or_crowd"

    end_bgap = pair_box_gap_cm.get(te - 1, 99.0)
    if require_box_touch and end_bgap > max_touch_box_gap_cm:
        return None, "rejected_contact_or_crowd"

    cropped = update_segment_bounds(event, ts, te, fps, suffix="approach_crop")
    if len(set(event_participant_ids(cropped))) < 2:
        return None, "rejected_insufficient_participants"
    if not endpoints_fully_observed(store, cropped, video_info):
        return None, "rejected_incomplete_endpoints"
    if not qualify_event_duration(cropped, fps, parameters=params)["ok"] or (span / fps) + 1e-9 < float(min_duration_s):
        return None, "rejected_below_min_duration"

    # Reject any 1-frame teleport jump inside the cropped bout (excluding initial boundary entry).
    if any(
        max(raw1f_a.get(f, 0.0), raw1f_b.get(f, 0.0)) > max_jump_speed_cm_s
        for f in range(ts + 1, te)
    ):
        return None, "rejected_tracking_jump"

    d_start = dist_cm.get(ts)
    d_end = dist_cm.get(te - 1)
    if (
        d_start is None
        or d_end is None
        or d_start < min_start_dist_cm
        or (d_start - d_end) < min_closing_cm
    ):
        return None, "rejected_insufficient_closing"

    actor_sp = [sm_actor.get(f, 0.0) for f in obs_in_span]
    target_sp = [sm_target.get(f, 0.0) for f in obs_in_span]
    actor_mean = float(np.mean(actor_sp)) if actor_sp else 0.0
    target_mean = float(np.mean(target_sp)) if target_sp else 0.0
    toward_cos_vals = [
        _actor_toward_cosine(pos_actor, pos_target, vel_actor, sm_actor, f)
        for f in obs_in_span
        if sm_actor.get(f, 0.0) >= 2.0
    ]
    mean_toward_cos = float(np.mean(toward_cos_vals)) if toward_cos_vals else 0.0
    if (
        actor_mean < min_actor_speed_cm_s
        or mean_toward_cos < min_toward_cos
        or (target_mean > 0.0 and actor_mean < 0.9 * target_mean)
    ):
        return None, "rejected_insufficient_closing"

    end_dir = _compute_approach_final_directionality(
        mid_actor,
        pos_actor,
        pos_target,
        vel_actor,
        vel_target,
        sm_actor,
        sm_target,
        dist_cm,
        co_obs,
        ts,
        te,
        fps,
        rows_by_frame=rows_by_frame,
        cm_per_px=cm_per_px,
    )
    end_hcos = end_dir["end_heading_cos"]
    end_baxis = end_dir["end_body_axis_cos"]
    end_adv = end_dir["end_snout_tail_adv_cm"]
    if (
        float(end_dir["end_actor_proj_cm_s"] or 0.0) < min_end_proj_toward_cm_s
        or float(end_dir["end_toward_cos"] or 0.0) < min_end_toward_cos
        or float(end_dir["end_closing_rate_cm_s"] or 0.0) < min_end_closing_rate_cm_s
        or float(end_dir["end_actor_dominance_cm_s"] or 0.0) < min_end_actor_dominance_cm_s
        or (end_hcos is not None and end_hcos < min_end_heading_cos)
        or (end_baxis is not None and end_baxis < min_end_body_axis_cos)
        or (end_adv is not None and end_adv < min_end_snout_tail_adv_cm)
    ):
        return None, "rejected_weak_final_directionality"

    obs_mask = np.array([co_obs.get(f, False) and crowd_ok.get(f, False) for f in range(ts, te)], dtype=bool)
    supp_intervals = _observed_support_intervals(obs_mask, ts)
    if not supp_intervals:
        return None, "rejected_incomplete_endpoints"

    pre_min_bgap = min(pair_box_gap_cm.get(f, 0.0) for f in pre_touch_fs)
    min_other = min(min_other_dist_cm.get(f, 999.0) for f in obs_in_span)
    cropped["canonical_support_frame_intervals"] = supp_intervals
    cropped["canonical_support_frame_count"] = int(sum(b - a for a, b in supp_intervals))
    cropped["approach_complete_refined"] = True
    cropped["approach_end_box_gap_cm"] = round(float(end_bgap), 3)
    cropped["approach_pre_touch_min_box_gap_cm"] = round(float(pre_min_bgap), 3)
    cropped["min_pair_box_gap_cm"] = round(float(pre_min_bgap), 3)
    cropped["min_other_mouse_dist_cm"] = round(float(min_other), 3)
    cropped["approach_start_dist_cm"] = round(float(d_start), 2)
    cropped["approach_end_dist_cm"] = round(float(d_end), 2)
    cropped["approach_closing_cm"] = round(float(d_start - d_end), 2)
    cropped["approach_actor_speed_cm_s"] = round(actor_mean, 2)
    cropped["approach_target_speed_cm_s"] = round(target_mean, 2)
    cropped["approach_mean_speed_cm_s"] = round(actor_mean, 2)
    cropped["approach_toward_cos"] = round(mean_toward_cos, 3)
    cropped["approach_end_proj_toward_cm_s"] = round(float(end_dir["end_actor_proj_cm_s"] or 0.0), 2)
    cropped["approach_end_target_proj_cm_s"] = round(float(end_dir["end_target_proj_cm_s"] or 0.0), 2)
    cropped["approach_end_actor_dominance_cm_s"] = round(float(end_dir["end_actor_dominance_cm_s"] or 0.0), 2)
    cropped["approach_end_toward_cos"] = round(float(end_dir["end_toward_cos"] or 0.0), 3)
    cropped["approach_end_closing_rate_cm_s"] = round(float(end_dir["end_closing_rate_cm_s"] or 0.0), 2)
    if end_hcos is not None:
        cropped["approach_end_heading_cos"] = round(float(end_hcos), 3)
    if end_baxis is not None:
        cropped["approach_end_body_axis_cos"] = round(float(end_baxis), 3)
    if end_adv is not None:
        cropped["approach_end_snout_tail_adv_cm"] = round(float(end_adv), 2)
    attach_social_roles(cropped, "approach", mid_actor, mid_target, supp_intervals)
    return cropped, "accepted"


def _refine_approach_kinematic_single(
    event: Mapping[str, Any],
    store: Any,
    video_info: Mapping[str, Any],
    binding: Mapping[str, Any] | None,
    scale: Mapping[str, Any] | None,
    params: HeuristicParameters,
    *,
    min_duration_s: float = 1.0,
    require_box_touch: bool = True,
) -> tuple[dict[str, Any] | None, str]:
    """Recover >=1.0s far-start to box-just-touch outside-crowd Approach with directed roles."""
    fps = _resolve_fps(binding)
    frame_count = video_info.get("frame_count")
    min_frames = max(2, int(np.ceil(float(min_duration_s) * fps - 1e-9)))

    ordered = _ordered_pair_ids(event)
    if len(ordered) < 2:
        return None, "rejected_insufficient_participants"
    start = int(event["start_frame"])
    end = int(event["end_frame"])
    if end <= start:
        return None, "rejected_below_min_duration"

    mid_a, mid_b = ordered[0], ordered[1]
    lookback = int(round(4.0 * fps))
    lookforward = int(round(1.5 * fps))
    scan_s = max(0, start - lookback)
    scan_e = min(int(frame_count), end + lookforward) if frame_count is not None else end + lookforward

    rows = list(store.rows(scan_s, scan_e))
    if not rows:
        return None, "rejected_no_observations"
    rows_by_frame = {int(r.get("frame", r.get("source_frame", 0))): r for r in rows}

    cm_per_px = _resolve_scale_cm_per_px(rows, ordered[:2], binding, scale=scale)
    (
        pos_a,
        pos_b,
        dist_cm,
        vel_a,
        vel_b,
        sm_a,
        sm_b,
        raw1f_a,
        raw1f_b,
        pair_box_gap_cm,
        min_other_dist_cm,
        co_obs,
        _clean_legacy,
        crowd_ok,
    ) = _analyze_approach_scene(rows, mid_a, mid_b, fps, cm_per_px, min_pair_box_gap_cm=0.01, min_pair_center_dist_cm=4.5)

    # Identify first box-just-touch transition near/after [start, end) that has a valid clean pre-touch prefix.
    max_jump_cm_s = 120.0
    touch_search_lo = max(scan_s + 1, start - max(0, int(round(0.3 * fps))))
    touch_search_hi = min(scan_e, end + lookforward)
    touch_candidates = [
        f
        for f in range(touch_search_lo, touch_search_hi)
        if co_obs.get(f, False)
        and crowd_ok.get(f, False)
        and pair_box_gap_cm.get(f, 99.0) <= 0.15
        and pair_box_gap_cm.get(f - 1, 0.0) > 0.0
        and max(raw1f_a.get(f, 0.0), raw1f_b.get(f, 0.0)) <= max_jump_cm_s
    ]

    def _orient_actor(
        fs_span: list[int],
    ) -> tuple[
        int,
        int,
        Mapping[int, np.ndarray | None],
        Mapping[int, np.ndarray | None],
        Mapping[int, np.ndarray],
        Mapping[int, np.ndarray],
        Mapping[int, float],
        Mapping[int, float],
    ]:
        ca_scores = [
            _actor_toward_cosine(pos_a, pos_b, vel_a, sm_a, f) * sm_a.get(f, 0.0)
            for f in fs_span
        ]
        cb_scores = [
            _actor_toward_cosine(pos_b, pos_a, vel_b, sm_b, f) * sm_b.get(f, 0.0)
            for f in fs_span
        ]
        if sum(ca_scores) >= sum(cb_scores):
            return mid_a, mid_b, pos_a, pos_b, vel_a, vel_b, sm_a, sm_b
        return mid_b, mid_a, pos_b, pos_a, vel_b, vel_a, sm_b, sm_a

    best_fail_reason = "rejected_contact_or_crowd"
    for f_touch in touch_candidates:
        pre: list[int] = []
        for f in range(f_touch - 1, scan_s - 1, -1):
            if (
                not co_obs.get(f, False)
                or not crowd_ok.get(f, False)
                or pair_box_gap_cm.get(f, 0.0) <= 0.0
                or dist_cm.get(f, 0.0) < 4.5
                or max(raw1f_a.get(f + 1, 0.0), raw1f_b.get(f + 1, 0.0)) > max_jump_cm_s
            ):
                break
            pre.append(f)
        pre.reverse()
        if len(pre) + 1 < min_frames:
            if (f_touch - scan_s + 1) >= 2:
                best_fail_reason = "rejected_below_min_duration"
            continue

        f_start = max(pre, key=lambda fn: dist_cm.get(fn, 0.0))
        sub_init = [f for f in range(f_start, f_touch + 1) if co_obs.get(f, False)]
        m_act, m_tgt, p_act, p_tgt, v_act, v_tgt, s_act, s_tgt = _orient_actor(sub_init)

        # Trim stationary leading wait frames without dipping below required 1.0s duration.
        while (f_touch - f_start + 1) > min_frames:
            sp = s_act.get(f_start, 0.0)
            tc = _actor_toward_cosine(p_act, p_tgt, v_act, s_act, f_start)
            d_curr = dist_cm.get(f_start)
            d_next = dist_cm.get(f_start + 1)
            not_closing = d_curr is None or d_next is None or d_next >= d_curr - 0.12
            if (sp < 4.0 and not_closing) or (tc < 0.05 and not_closing):
                f_start += 1
            else:
                break

        res, reason = _validate_approach_span(
            event,
            f_start,
            f_touch + 1,
            store,
            video_info,
            fps,
            params,
            m_act,
            m_tgt,
            p_act,
            p_tgt,
            dist_cm,
            v_act,
            s_act,
            s_tgt,
            raw1f_a,
            raw1f_b,
            pair_box_gap_cm,
            min_other_dist_cm,
            co_obs,
            crowd_ok,
            vel_target=v_tgt,
            rows_by_frame=rows_by_frame,
            cm_per_px=cm_per_px,
            min_duration_s=min_duration_s,
            require_box_touch=require_box_touch,
        )
        if res is not None:
            return res, "accepted"
        best_fail_reason = reason

    if require_box_touch:
        if any(co_obs.get(f, False) and not crowd_ok.get(f, False) for f in range(start, end)):
            return None, "rejected_contact_or_crowd"
        if not touch_candidates and (end - start) < min_frames:
            return None, "rejected_below_min_duration"
        return None, best_fail_reason

    # Optional non-touch fallback for unit tests with require_box_touch=False.
    runs_clean: list[tuple[int, int]] = []
    cur_s: int | None = None
    for f in range(scan_s, scan_e):
        ok_f = co_obs.get(f, False) and crowd_ok.get(f, False) and pair_box_gap_cm.get(f, 0.0) > 0.0
        if ok_f:
            jump_break = cur_s is not None and max(raw1f_a.get(f, 0.0), raw1f_b.get(f, 0.0)) > max_jump_cm_s
            if cur_s is None:
                cur_s = f
            elif jump_break:
                runs_clean.append((cur_s, f))
                cur_s = f
        elif cur_s is not None:
            runs_clean.append((cur_s, f))
            cur_s = None
    if cur_s is not None:
        runs_clean.append((cur_s, scan_e))
    ov = [r for r in runs_clean if r[1] > start and r[0] < end]
    if not ov:
        return None, "rejected_contact_or_crowd"
    r_s, r_e = max(ov, key=lambda r: (min(r[1], end) - max(r[0], start), r[1] - r[0]))
    run_fs = [f for f in range(r_s, r_e) if f in dist_cm]
    if not run_fs:
        return None, "rejected_below_min_duration"
    f_end = min([f for f in run_fs if f >= start] or run_fs, key=lambda fn: dist_cm[fn])
    pre_fs = [f for f in run_fs if f < f_end]
    f_start = max(pre_fs, key=lambda fn: dist_cm[fn]) if pre_fs else r_s
    sub = [f for f in range(f_start, f_end + 1) if co_obs.get(f, False)]
    m_act, m_tgt, p_act, p_tgt, v_act, v_tgt, s_act, s_tgt = _orient_actor(sub)
    return _validate_approach_span(
        event,
        f_start,
        f_end + 1,
        store,
        video_info,
        fps,
        params,
        m_act,
        m_tgt,
        p_act,
        p_tgt,
        dist_cm,
        v_act,
        s_act,
        s_tgt,
        raw1f_a,
        raw1f_b,
        pair_box_gap_cm,
        min_other_dist_cm,
        co_obs,
        crowd_ok,
        vel_target=v_tgt,
        rows_by_frame=rows_by_frame,
        cm_per_px=cm_per_px,
        min_duration_s=min_duration_s,
        require_box_touch=False,
    )


def refine_approach_segments(
    events: Sequence[Mapping[str, Any]],
    store: Any,
    binding: Mapping[str, Any] | None = None,
    scale: Mapping[str, Any] | None = None,
    parameters: HeuristicParameters | None = None,
    *,
    scene_guard: bool = True,
    requalify: bool = True,
    min_duration_s: float = 1.0,
    require_box_touch: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Refine approach candidates enforcing >=1.0s far-start to box-touch, outside crowd & directed roles."""
    from mouse_behavior.evaluation.approach_events import requalify_source_approach_event

    video_info = dict(_resolve_video_info(binding))
    fps = _resolve_fps(binding)
    params = _resolve_parameters(binding, parameters)
    adapted_store = store if hasattr(store, "window") else _StoreWindowAdapter(store)

    refined: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for event in events:
        beh_key = normalize_behavior_key(str(event.get("canonical_behavior") or event.get("behavior") or ""))
        if beh_key != "approach":
            refined.append(copy.deepcopy(dict(event)))
            continue

        if len(set(event_participant_ids(event))) < 2 or int(event["end_frame"]) <= int(event["start_frame"]):
            audit.append({
                "parent_event_id": event.get("event_id"),
                "accepted": [],
                "reason": "rejected_insufficient_participants",
            })
            continue

        if requalify and scale is not None:
            req_info = dict(video_info)
            req_info.setdefault("fps", fps)
            if "frame_count" not in req_info:
                probe_end = int(event["end_frame"]) + int(round(2.0 * fps))
                probe_rows = list(store.rows(max(0, int(event["start_frame"])), probe_end))
                req_info["frame_count"] = (
                    max((int(r.get("frame", r.get("source_frame", 0))) + 1) for r in probe_rows)
                    if probe_rows
                    else int(event["end_frame"])
                )
            try:
                children = requalify_source_approach_event(
                    adapted_store,
                    dict(event),
                    req_info,
                    dict(scale),
                    params,
                    scene_guard=scene_guard,
                )
            except (KeyError, ValueError):
                candidate, reason = _refine_approach_kinematic_single(
                    event,
                    store,
                    video_info,
                    binding,
                    scale,
                    params,
                    min_duration_s=min_duration_s,
                    require_box_touch=require_box_touch,
                )
                if candidate is not None:
                    refined.append(candidate)
                    audit.append({"parent_event_id": event.get("event_id"), "accepted": [candidate.get("event_id")]})
                else:
                    audit.append({"parent_event_id": event.get("event_id"), "accepted": [], "reason": reason})
                continue

            if not children:
                candidate, reason = _refine_approach_kinematic_single(
                    event,
                    store,
                    req_info,
                    binding,
                    scale,
                    params,
                    min_duration_s=min_duration_s,
                    require_box_touch=require_box_touch,
                )
                if candidate is not None:
                    refined.append(candidate)
                    audit.append({"parent_event_id": event.get("event_id"), "accepted": [candidate.get("event_id")]})
                else:
                    audit.append({
                        "parent_event_id": event.get("event_id"),
                        "accepted": [],
                        "reason": reason or "rejected_by_source_requalification",
                    })
                continue

            accepted_ids: list[Any] = []
            reject_reasons: list[str] = []
            for child in children:
                validated, reason = _refine_approach_kinematic_single(
                    child,
                    store,
                    req_info,
                    binding,
                    scale,
                    params,
                    min_duration_s=min_duration_s,
                    require_box_touch=require_box_touch,
                )
                if validated is not None:
                    refined.append(validated)
                    accepted_ids.append(validated.get("event_id"))
                else:
                    reject_reasons.append(reason)
            if accepted_ids:
                audit.append({"parent_event_id": event.get("event_id"), "accepted": accepted_ids})
            else:
                audit.append({
                    "parent_event_id": event.get("event_id"),
                    "accepted": [],
                    "reason": reject_reasons[0] if reject_reasons else "rejected_contact_or_crowd",
                })
        else:
            candidate, reason = _refine_approach_kinematic_single(
                event,
                store,
                video_info,
                binding,
                scale,
                params,
                min_duration_s=min_duration_s,
                require_box_touch=require_box_touch,
            )
            if candidate is not None:
                refined.append(candidate)
                audit.append({"parent_event_id": event.get("event_id"), "accepted": [candidate.get("event_id")]})
            else:
                audit.append({"parent_event_id": event.get("event_id"), "accepted": [], "reason": reason})

    return refined, audit
