"""Contact (head/tail) and Together segment clipping with complete bout bounds and duration gates."""
from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

import cv2
import numpy as np

from mouse_behavior.behavior.segment_clipping_common import (
    _clip_multi_track_series,
    _detection_map,
    _extract_box_px,
    _extract_keypoints_px,
    _observed_run_velocity_cm_s,
    _observed_support_intervals,
    _resolve_fps,
    _resolve_parameters,
    _resolve_scale_cm_per_px,
    _resolve_video_info,
    endpoints_fully_observed,
    event_participant_ids,
    normalize_behavior_key,
    qualify_clip_duration,
    update_segment_bounds,
)
from mouse_behavior.evaluation.exported_dataset_models import HeuristicParameters

__all__ = [
    "_box_gap_cm",
    "_box_iou",
    "_directed_head_distance_cm",
    "attach_social_roles",
    "build_social_roles",
    "refine_contact_and_together_segments",
]

_DIRECTED_SOCIAL_ROLE_LABELS: dict[str, tuple[str, str, str, str]] = {
    "approach": ("Approaching", "Approached", "主动接近者", "被接近者"),
    "attack": ("Attacker", "Attacked", "攻击发起者", "被攻击者"),
    "chase": ("Chasing", "Chased", "追逐者", "被追逐者"),
    "nose_head_contact": ("Snout_contacting", "Head_contacted", "主动鼻嗅者", "被嗅头部鼠"),
    "nose_tail_contact": ("Snout_contacting", "Rear_contacted", "主动尾嗅者", "被尾嗅者"),
}


def _box_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute intersection-over-union between two [x1, y1, x2, y2] boxes."""
    ix1 = max(float(box_a[0]), float(box_b[0]))
    iy1 = max(float(box_a[1]), float(box_b[1]))
    ix2 = min(float(box_a[2]), float(box_b[2]))
    iy2 = min(float(box_a[3]), float(box_b[3]))
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, float(box_a[2]) - float(box_a[0])) * max(0.0, float(box_a[3]) - float(box_a[1]))
    area_b = max(0.0, float(box_b[2]) - float(box_b[0])) * max(float(0.0), float(box_b[3]) - float(box_b[1]))
    return float(inter / max(1e-6, area_a + area_b - inter))


def build_social_roles(
    canonical_behavior: str,
    actor_id: int,
    target_id: int,
) -> dict[str, Any] | None:
    """Build bilingual initiator/receiver role metadata for directed social behaviors."""
    key = normalize_behavior_key(canonical_behavior)
    key = {
        "chasing": "chase",
        "head_contact": "nose_head_contact",
        "tail_contact": "nose_tail_contact",
    }.get(key, key)
    if key == "together":
        return None
    spec = _DIRECTED_SOCIAL_ROLE_LABELS.get(key)
    if spec is None or int(actor_id) < 0 or int(target_id) < 0 or int(actor_id) == int(target_id):
        return None
    init_en, tgt_en, init_zh, tgt_zh = spec
    return {
        "initiator_role": init_en,
        "target_role": tgt_en,
        "initiator_label_zh": init_zh,
        "target_label_zh": tgt_zh,
        "initiator_id": int(actor_id),
        "target_id": int(target_id),
        "summary_zh": f"V{int(actor_id)}({init_zh}) → V{int(target_id)}({tgt_zh})",
    }


def attach_social_roles(
    event: dict[str, Any],
    canonical_behavior: str,
    actor_id: int | None,
    target_id: int | None,
    support_intervals: Sequence[Sequence[int]] | None = None,
) -> None:
    """Attach standardized role fields to directed social events; mark Together undirected."""
    key = normalize_behavior_key(canonical_behavior)
    key = {
        "chasing": "chase",
        "head_contact": "nose_head_contact",
        "tail_contact": "nose_tail_contact",
    }.get(key, key)
    if (
        key == "together"
        or actor_id is None
        or target_id is None
        or int(actor_id) < 0
        or int(target_id) < 0
        or int(actor_id) == int(target_id)
    ):
        event["actor_id"] = -1
        event["target_id"] = -1
        event["initiator_id"] = None
        event["receiver_id"] = None
        event["role_status"] = "undirected_not_applicable"
        event["roles"] = None
        event["participants"] = []
        return

    aid, tid = int(actor_id), int(target_id)
    event["actor_id"] = aid
    event["target_id"] = tid
    event["initiator_id"] = aid
    event["receiver_id"] = tid
    event["role_status"] = "measured_directed_role"
    roles_obj = build_social_roles(key, aid, tid)
    event["roles"] = roles_obj
    if roles_obj is not None:
        init_role = str(roles_obj["initiator_role"])
        tgt_role = str(roles_obj["target_role"])
        event["participants"] = [
            {"role_name": init_role, "role_key": init_role.lower(), "track_ids": [aid]},
            {"role_name": tgt_role, "role_key": tgt_role.lower(), "track_ids": [tid]},
        ]
    if support_intervals:
        event["role_support_frame_intervals"] = [[int(a), int(b)] for a, b in support_intervals]
    elif "start_frame" in event and "end_frame" in event:
        event["role_support_frame_intervals"] = [[int(event["start_frame"]), int(event["end_frame"])]]


def _directed_head_distance_cm(
    actor_kp_cm: np.ndarray,
    target_kp_cm: np.ndarray,
) -> float:
    nose = actor_kp_cm[0]
    if not np.isfinite(nose).all():
        return float("nan")
    head_tri = target_kp_cm[:3]
    finite = np.isfinite(head_tri).all(axis=1)
    if not finite.any():
        return float("nan")
    if finite.all():
        hull = cv2.convexHull(head_tri.astype(np.float32))
        if len(hull) >= 3 and cv2.contourArea(hull) > 1e-9:
            return float(max(0.0, -cv2.pointPolygonTest(hull, (float(nose[0]), float(nose[1])), True)))
    dists = np.linalg.norm(head_tri[finite] - nose, axis=1)
    return float(dists.min())


def _box_gap_cm(box_a: np.ndarray, box_b: np.ndarray, cm_per_px: float) -> float:
    dx = max(0.0, max(float(box_a[0]) - float(box_b[2]), float(box_b[0]) - float(box_a[2])))
    dy = max(0.0, max(float(box_a[1]) - float(box_b[3]), float(box_b[1]) - float(box_a[3])))
    return float(np.hypot(dx, dy) * cm_per_px)


def refine_contact_and_together_segments(
    events: Sequence[Mapping[str, Any]],
    store: Any,
    binding: Mapping[str, Any] | None = None,
    *,
    scale: Mapping[str, Any] | None = None,
    parameters: HeuristicParameters | None = None,
) -> list[dict[str, Any]]:
    """Fine-crop contact and together segments over bounded local context for complete bout recovery."""
    from mouse_behavior.evaluation.screening_signals import runs
    from mouse_behavior.evaluation.social_heuristic_config import DEFAULT_SOCIAL_HEURISTIC_CONFIG

    fps = _resolve_fps(binding)
    info = _resolve_video_info(binding)
    frame_count = info.get("frame_count")
    params = _resolve_parameters(binding, parameters)
    social_cfg = DEFAULT_SOCIAL_HEURISTIC_CONFIG
    ref_body_cm = 7.918
    if isinstance(scale, Mapping) and scale.get("reference_body_cm"):
        ref_body_cm = float(scale["reference_body_cm"])

    refined: list[dict[str, Any]] = []
    for event in events:
        beh_key = normalize_behavior_key(str(event.get("canonical_behavior") or event.get("behavior") or ""))
        if beh_key not in {"head_contact", "tail_contact", "together"}:
            refined.append(copy.deepcopy(dict(event)))
            continue

        start = int(event["start_frame"])
        end = int(event["end_frame"])
        ordered_mids = list(dict.fromkeys(int(m) for m in event.get("mouse_ids", ()) if not isinstance(m, bool)))
        if len(ordered_mids) < 2 or len(set(event_participant_ids(event))) < 2 or end <= start:
            continue
        mid_a, mid_b = ordered_mids[0], ordered_mids[1]

        context_s = 3.0 if beh_key == "together" else 2.0
        context_frames = int(round(context_s * fps))
        scan_s = max(0, start - context_frames)
        scan_e = min(int(frame_count), end + context_frames) if frame_count is not None else end + context_frames

        rows = list(store.rows(scan_s, scan_e))
        if not rows:
            continue
        span = scan_e - scan_s
        if span < 2:
            continue

        cm_per_px = _resolve_scale_cm_per_px(rows, [mid_a, mid_b], binding, scale=scale)
        centers_cm, observed = _clip_multi_track_series(rows, scan_s, scan_e, [mid_a, mid_b], cm_per_px)
        pair_obs = observed[:, 0] & observed[:, 1]
        if int(pair_obs.sum()) < 2:
            continue
        pair_dist = np.where(
            pair_obs,
            np.linalg.norm(centers_cm[:, 0] - centers_cm[:, 1], axis=1),
            np.nan,
        )

        if beh_key == "together":
            _, _, speed = _observed_run_velocity_cm_s(centers_cm, observed, fps, motion_window_s=0.3)
            max_dist = float(social_cfg.dist_together_max_cm)
            max_spd = float(social_cfg.vel_together_max_cm_s)
            strict = (
                pair_obs
                & np.isfinite(pair_dist)
                & (pair_dist < max_dist)
                & np.isfinite(speed[:, 0])
                & (speed[:, 0] < max_spd)
                & np.isfinite(speed[:, 1])
                & (speed[:, 1] < max_spd)
            )
            unknown = ~(
                pair_obs
                & np.isfinite(pair_dist)
                & np.isfinite(speed[:, 0])
                & np.isfinite(speed[:, 1])
            )
            bridge_frames = max(0, int(round(float(social_cfg.together_bridge_max_gap_s) * fps)))
            bridged = strict.copy()
            for gap_lo, gap_hi in runs(~strict):
                if (
                    0 < gap_lo
                    and gap_hi < span
                    and (gap_hi - gap_lo) <= bridge_frames
                    and unknown[gap_lo:gap_hi].all()
                    and np.isfinite(pair_dist[gap_lo - 1])
                    and np.isfinite(pair_dist[gap_hi])
                    and pair_dist[gap_lo - 1] < max_dist
                    and pair_dist[gap_hi] < max_dist
                ):
                    bridged[gap_lo:gap_hi] = True
            suffix = "together_crop"
        else:
            kind = "head" if beh_key == "head_contact" else "tail"
            cap_cm = float(
                social_cfg.dist_nose_head_contact_max_cm
                if kind == "head"
                else social_cfg.dist_nose_tail_contact_max_cm
            )
            bridge_s = float(
                social_cfg.contact_bridge_max_gap_s
                if kind == "head"
                else social_cfg.contact_tail_bridge_max_gap_s
            )
            bridge_frames = max(0, int(round(bridge_s * fps)))
            support_rad = float(social_cfg.contact_bridge_support_max_body_lengths) * ref_body_cm
            ceiling_pad = float(social_cfg.contact_signal_max_body_lengths) * ref_body_cm

            kp_a = np.full((span, 7, 2), np.nan, dtype=float)
            kp_b = np.full((span, 7, 2), np.nan, dtype=float)
            box_a = np.full((span, 4), np.nan, dtype=float)
            box_b = np.full((span, 4), np.nan, dtype=float)
            for idx, row in enumerate(rows):
                raw_frame = row.get("frame", row.get("source_frame"))
                rel = int(raw_frame) - scan_s if raw_frame is not None else idx
                if not (0 <= rel < span):
                    continue
                dmap = _detection_map(row)
                da = dmap.get(mid_a)
                db = dmap.get(mid_b)
                if da is not None:
                    ka = _extract_keypoints_px(da)
                    if ka is not None:
                        kp_a[rel] = ka * cm_per_px
                    ba = _extract_box_px(da)
                    if ba is not None:
                        box_a[rel] = ba
                if db is not None:
                    kb = _extract_keypoints_px(db)
                    if kb is not None:
                        kp_b[rel] = kb * cm_per_px
                    bb = _extract_box_px(db)
                    if bb is not None:
                        box_b[rel] = bb

            d_min = np.full(span, np.nan, dtype=float)
            for rel in np.flatnonzero(pair_obs):
                if np.isfinite(kp_a[rel]).any() and np.isfinite(kp_b[rel]).any():
                    if kind == "head":
                        dab = _directed_head_distance_cm(kp_a[rel], kp_b[rel])
                        dba = _directed_head_distance_cm(kp_b[rel], kp_a[rel])
                    else:
                        dab = (
                            float(np.linalg.norm(kp_a[rel, 0] - kp_b[rel, 6]))
                            if np.isfinite(kp_a[rel, 0]).all() and np.isfinite(kp_b[rel, 6]).all()
                            else float("nan")
                        )
                        dba = (
                            float(np.linalg.norm(kp_b[rel, 0] - kp_a[rel, 6]))
                            if np.isfinite(kp_b[rel, 0]).all() and np.isfinite(kp_a[rel, 6]).all()
                            else float("nan")
                        )
                    vals = [v for v in (dab, dba) if np.isfinite(v)]
                    if vals:
                        d_min[rel] = min(vals)
                elif np.isfinite(box_a[rel]).all() and np.isfinite(box_b[rel]).all():
                    d_min[rel] = _box_gap_cm(box_a[rel], box_b[rel], cm_per_px)

            strict = pair_obs & np.isfinite(d_min) & (d_min <= cap_cm)
            ceiling = pair_dist + ceiling_pad
            unknown = (~pair_obs) | (~np.isfinite(d_min)) | (np.isfinite(ceiling) & (d_min > ceiling))
            bridged = strict.copy()
            for gap_lo, gap_hi in runs(~strict):
                if (
                    0 < gap_lo
                    and gap_hi < span
                    and (gap_hi - gap_lo) <= bridge_frames
                    and unknown[gap_lo:gap_hi].all()
                    and np.isfinite(pair_dist[gap_lo - 1])
                    and np.isfinite(pair_dist[gap_hi])
                    and pair_dist[gap_lo - 1] <= support_rad
                    and pair_dist[gap_hi] <= support_rad
                ):
                    bridged[gap_lo:gap_hi] = True
            suffix = f"{beh_key}_crop"

        if not bridged.any():
            continue

        for rel_s, rel_e in runs(bridged):
            rel_si, rel_ei = int(rel_s), int(rel_e)
            obs_sub = np.flatnonzero(strict[rel_si:rel_ei])
            if len(obs_sub) < 2:
                continue
            true_s = int(scan_s + rel_si + int(obs_sub[0]))
            true_e = int(scan_s + rel_si + int(obs_sub[-1]) + 1)
            if true_e <= start or true_s >= end:
                continue
            cropped = update_segment_bounds(event, true_s, true_e, fps, suffix=suffix)
            supp_slice = strict[true_s - scan_s : true_e - scan_s]
            supp_intervals = _observed_support_intervals(supp_slice, true_s)
            if not supp_intervals:
                continue
            cropped["canonical_support_frame_intervals"] = supp_intervals
            cropped["canonical_support_frame_count"] = int(sum(b - a for a, b in supp_intervals))
            if qualify_clip_duration(cropped, fps, parameters=params)["ok"] and endpoints_fully_observed(
                store, cropped, info
            ):
                if beh_key == "together":
                    attach_social_roles(cropped, "together", None, None)
                else:
                    # Attribute active sniffer -> target over [true_s, true_e), tie-breaking by pre-contact closing.
                    rel_lo = max(0, true_s - scan_s)
                    rel_hi = min(span, true_e - scan_s)
                    dab_vals: list[float] = []
                    dba_vals: list[float] = []
                    for rr in range(rel_lo, rel_hi):
                        if not pair_obs[rr]:
                            continue
                        if kind == "head":
                            va = _directed_head_distance_cm(kp_a[rr], kp_b[rr])
                            vb = _directed_head_distance_cm(kp_b[rr], kp_a[rr])
                        else:
                            va = (
                                float(np.linalg.norm(kp_a[rr, 0] - kp_b[rr, 6]))
                                if np.isfinite(kp_a[rr, 0]).all() and np.isfinite(kp_b[rr, 6]).all()
                                else float("nan")
                            )
                            vb = (
                                float(np.linalg.norm(kp_b[rr, 0] - kp_a[rr, 6]))
                                if np.isfinite(kp_b[rr, 0]).all() and np.isfinite(kp_a[rr, 6]).all()
                                else float("nan")
                            )
                        if np.isfinite(va):
                            dab_vals.append(float(va))
                        if np.isfinite(vb):
                            dba_vals.append(float(vb))
                    med_ab = float(np.median(dab_vals)) if dab_vals else 999.0
                    med_ba = float(np.median(dba_vals)) if dba_vals else 999.0
                    if med_ab + 0.15 < med_ba:
                        actor, target = mid_a, mid_b
                    elif med_ba + 0.15 < med_ab:
                        actor, target = mid_b, mid_a
                    else:
                        # Pre-contact approach displacement tie-breaker.
                        pre_lo = max(0, rel_lo - max(2, int(round(0.4 * fps))))
                        obs_pre = [k for k in range(pre_lo, min(span, rel_lo + 1)) if pair_obs[k]]
                        if len(obs_pre) >= 2:
                            k0, k1 = obs_pre[0], obs_pre[-1]
                            rel_vec = centers_cm[k0, 1] - centers_cm[k0, 0]
                            nr = float(np.linalg.norm(rel_vec))
                            if nr > 1e-4:
                                u = rel_vec / nr
                                close_a = float(np.dot(centers_cm[k1, 0] - centers_cm[k0, 0], u))
                                close_b = float(np.dot(centers_cm[k1, 1] - centers_cm[k0, 1], -u))
                                actor, target = (mid_a, mid_b) if close_a >= close_b else (mid_b, mid_a)
                            else:
                                actor, target = mid_a, mid_b
                        else:
                            ev_a = event.get("actor_id")
                            actor = int(ev_a) if ev_a is not None and int(ev_a) in {mid_a, mid_b} else mid_a
                            target = mid_b if actor == mid_a else mid_a
                    canon = "nose_head_contact" if kind == "head" else "nose_tail_contact"
                    attach_social_roles(cropped, canon, actor, target, supp_intervals)
                refined.append(cropped)

    return refined
