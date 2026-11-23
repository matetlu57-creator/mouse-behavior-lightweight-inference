"""Social/pair-layer behavior segment clipping (Approach, Attack, Chasing, Contact, Together)."""
from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

import cv2
import numpy as np

from mouse_behavior.behavior.duration_policy import qualify_event_duration
from mouse_behavior.behavior.segment_clipping_common import (
    _clip_multi_track_series,
    _detection_map,
    _extract_box_px,
    _extract_center_px,
    _extract_keypoints_px,
    _observed_run_velocity_cm_s,
    _observed_support_intervals,
    _resolve_fps,
    _resolve_parameters,
    _resolve_scale_cm_per_px,
    _resolve_video_info,
    normalize_behavior_key,
    update_segment_bounds,
)
from mouse_behavior.evaluation.exported_dataset_models import HeuristicParameters

__all__ = [
    "refine_approach_segments",
    "refine_attack_segments",
    "refine_chasing_segments",
    "refine_contact_and_together_segments",
]


def refine_approach_segments(
    events: Sequence[Mapping[str, Any]],
    store: Any,
    binding: Mapping[str, Any],
    scale: Mapping[str, Any],
    parameters: HeuristicParameters | None = None,
    *,
    scene_guard: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Requalify approach candidates on the source store with optional multi-mouse scene guard."""
    from mouse_behavior.evaluation.approach_events import requalify_source_approach_event

    video_info = _resolve_video_info(binding)
    params = _resolve_parameters(binding, parameters)
    refined: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for event in events:
        if event.get("canonical_behavior") != "approach":
            refined.append(copy.deepcopy(dict(event)))
            continue
        children = requalify_source_approach_event(
            store,
            dict(event),
            dict(video_info),
            dict(scale),
            params,
            scene_guard=scene_guard,
        )
        if not children:
            audit.append({
                "parent_event_id": event.get("event_id"),
                "accepted": [],
                "reason": "rejected_by_source_requalification",
            })
            continue
        accepted_ids = []
        for child in children:
            refined.append(child)
            accepted_ids.append(child.get("event_id"))
        audit.append({
            "parent_event_id": event.get("event_id"),
            "accepted": accepted_ids,
        })
    return refined, audit


def refine_attack_segments(
    events: Sequence[Mapping[str, Any]],
    store: Any,
    binding: Mapping[str, Any] | None = None,
    *,
    scale: Mapping[str, Any] | None = None,
    parameters: HeuristicParameters | None = None,
    min_duration_s: float | None = None,
    max_dropout_fraction: float = 0.10,
    min_path_px: float = 60.0,
    min_speed_px_s: float = 40.0,
    surge_dist_cm: float = 18.0,
    surge_speed_cm_s: float = 4.0,
    grapple_dist_cm: float = 16.0,
    max_gap_frames: int | None = None,
) -> list[dict[str, Any]]:
    """Validate and crop attack segments with strike onset backtracking and grapple continuation.

    Kinematic strike onset scans backward from candidate start while distance <=
    `surge_dist_cm` (18.0 cm) and speed >= `surge_speed_cm_s` (4.0 cm/s) or close
    grapple contact (<= 12.0 cm).
    Grapple continuation extends tumble offset while mice remain in interaction
    envelope (distance <= `grapple_dist_cm`, 16.0 cm), bounded by `lookforward`.
    `max_gap_frames` tolerances momentary detection dropouts or tumble occlusions
    (default: ~0.35 s).
    Never discards valid attack events if motion/duration checks fail on boundary clips;
    instead falls back to candidate bounds so classification accuracy is preserved.
    """
    fps = _resolve_fps(binding)
    params = _resolve_parameters(binding, parameters)
    req_s = min_duration_s if min_duration_s is not None else float(params.attack_min_event_duration_s)
    info = _resolve_video_info(binding)
    frame_count = info.get("frame_count")
    gap_limit = max_gap_frames if max_gap_frames is not None else max(4, int(round(0.35 * fps)))

    refined: list[dict[str, Any]] = []
    for event in events:
        if event.get("canonical_behavior") != "attack":
            refined.append(copy.deepcopy(dict(event)))
            continue
        start = int(event["start_frame"])
        end = int(event["end_frame"])
        mids = [int(m) for m in event.get("mouse_ids", ())]
        if len(mids) < 2 or end <= start:
            refined.append(copy.deepcopy(dict(event)))
            continue

        mid_a, mid_b = mids[0], mids[1]
        lookback = int(round(4.0 * fps))
        lookforward = int(round(6.0 * fps))
        scan_s = max(0, start - lookback)
        scan_e = min(int(frame_count), end + lookforward) if frame_count is not None else end + lookforward

        rows = list(store.rows(scan_s, scan_e))
        if not rows:
            fallback = update_segment_bounds(event, start, end, fps, suffix="attack_fallback")
            refined.append(fallback)
            continue

        cm_per_px = _resolve_scale_cm_per_px(rows, mids, binding, scale=scale)
        frame_map = {int(r.get("frame", r.get("source_frame", 0))): r for r in rows}

        pos_a: dict[int, np.ndarray | None] = {}
        pos_b: dict[int, np.ndarray | None] = {}
        dist_cm: dict[int, float] = {}
        all_frames = sorted(frame_map.keys())
        for f in all_frames:
            dmap = _detection_map(frame_map[f])
            ca = _extract_center_px(dmap.get(mid_a, {}))
            cb = _extract_center_px(dmap.get(mid_b, {}))
            pos_a[f] = ca
            pos_b[f] = cb
            if ca is not None and cb is not None:
                dist_cm[f] = float(np.linalg.norm(ca - cb)) * cm_per_px

        sp_a: dict[int, float] = {}
        sp_b: dict[int, float] = {}
        for f in all_frames:
            pa_curr = pos_a.get(f)
            pa_prev = pos_a.get(f - 1)
            if pa_curr is not None and pa_prev is not None:
                sp_a[f] = float(np.linalg.norm(pa_curr - pa_prev)) * fps * cm_per_px
            elif f == 0 and pos_a.get(1) is not None and pa_curr is not None:
                sp_a[0] = float(np.linalg.norm(pos_a.get(1) - pa_curr)) * fps * cm_per_px
            else:
                sp_a[f] = 0.0
            pb_curr = pos_b.get(f)
            pb_prev = pos_b.get(f - 1)
            if pb_curr is not None and pb_prev is not None:
                sp_b[f] = float(np.linalg.norm(pb_curr - pb_prev)) * fps * cm_per_px
            elif f == 0 and pos_b.get(1) is not None and pb_curr is not None:
                sp_b[0] = float(np.linalg.norm(pos_b.get(1) - pb_curr)) * fps * cm_per_px
            else:
                sp_b[f] = 0.0

        # Kinematic strike onset: Backtrack surge onset while distance <= surge_dist_cm
        cand_s = start
        gap = 0
        for f in range(start - 1, scan_s - 1, -1):
            d = dist_cm.get(f)
            max_sp = max(sp_a.get(f, 0.0), sp_b.get(f, 0.0))
            if d is not None and d <= surge_dist_cm and (max_sp >= surge_speed_cm_s or d <= 12.0):
                cand_s = f
                gap = 0
            else:
                gap += 1
                if gap > gap_limit:
                    break

        if cand_s <= 1 and dist_cm.get(0) is not None and dist_cm[0] <= surge_dist_cm:
            cand_s = 0

        # Grapple continuation: Extend tumble offset while mice remain in intimate grapple contact
        # or active post-strike recoil / fleeing reaction
        cand_e = end
        gap_e = 0
        for f in range(end, scan_e):
            d = dist_cm.get(f)
            max_sp = max(sp_a.get(f, 0.0), sp_b.get(f, 0.0))
            if d is not None and (d <= grapple_dist_cm or (d <= 35.0 and max_sp >= 20.0)):
                cand_e = f + 1
                gap_e = 0
            else:
                gap_e += 1
                if gap_e > gap_limit:
                    break

        if frame_count is not None and cand_e >= int(frame_count) - 2:
            cand_e = int(frame_count)

        # Verify co-observed presence in [cand_s, cand_e)
        span = cand_e - cand_s
        dur_s = span / fps
        both_in_span = sum(1 for f in range(cand_s, cand_e) if dist_cm.get(f) is not None)

        # Fallback check: if duration or dropout fails, use original bounds fallback
        if dur_s < req_s - 1e-9 or (1.0 - both_in_span / max(1, span)) > max_dropout_fraction + 0.15:
            fallback = update_segment_bounds(event, start, end, fps, suffix="attack_fallback")
            refined.append(fallback)
            continue

        cropped = update_segment_bounds(event, cand_s, cand_e, fps, suffix="attack_crop")
        if qualify_event_duration(cropped, fps, parameters=params)["ok"]:
            refined.append(cropped)
        else:
            fallback = update_segment_bounds(event, start, end, fps, suffix="attack_fallback")
            refined.append(fallback)

    return refined


def refine_chasing_segments(
    events: Sequence[Mapping[str, Any]],
    store: Any,
    binding: Mapping[str, Any] | None = None,
    *,
    scale: Mapping[str, Any] | None = None,
    parameters: HeuristicParameters | None = None,
    min_duration_s: float | None = None,
    max_separation_cm: float = 32.0,
    min_follower_speed_cm_s: float = 3.5,
    max_gap_frames: int | None = None,
) -> list[dict[str, Any]]:
    """Fine-crop chasing segments by backtracking onset and trimming disengagement.

    Onset backtracking scans backward from start frame while follower pursuit speed
    and close separation (<= `max_separation_cm`, 32.0 cm) are maintained.
    Forward continuation extends while pursuit continues within `max_separation_cm`,
    bounded by `lookforward`.
    `max_gap_frames` tolerances momentary tracking noise or corner turns (default: ~0.35 s).
    Disengagement trimming identifies when distance opens up and pursuit stops.
    If the refined segment fails duration qualification, falls back to the candidate
    bounds so classification accuracy is never degraded.
    """
    fps = _resolve_fps(binding)
    params = _resolve_parameters(binding, parameters)
    req_s = min_duration_s if min_duration_s is not None else float(params.chase_min_duration_s)
    info = _resolve_video_info(binding)
    frame_count = info.get("frame_count")
    gap_limit = max_gap_frames if max_gap_frames is not None else max(4, int(round(0.35 * fps)))

    refined: list[dict[str, Any]] = []
    for event in events:
        if event.get("canonical_behavior") not in ("chasing", "chase"):
            refined.append(copy.deepcopy(dict(event)))
            continue

        start = int(event["start_frame"])
        end = int(event["end_frame"])
        mids = [int(m) for m in event.get("mouse_ids", ())]
        if len(mids) < 2 or end <= start:
            refined.append(copy.deepcopy(dict(event)))
            continue

        mid_a, mid_b = mids[0], mids[1]
        lookback = int(round(5.0 * fps))
        lookforward = int(round(5.0 * fps))
        scan_s = max(0, start - lookback)
        scan_e = min(int(frame_count), end + lookforward) if frame_count is not None else end + lookforward

        rows = list(store.rows(scan_s, scan_e))
        if not rows:
            fallback = update_segment_bounds(event, start, end, fps, suffix="chasing_fallback")
            refined.append(fallback)
            continue

        cm_per_px = _resolve_scale_cm_per_px(rows, mids, binding, scale=scale)
        frame_map = {int(r.get("frame", r.get("source_frame", 0))): r for r in rows}

        pos_a: dict[int, np.ndarray | None] = {}
        pos_b: dict[int, np.ndarray | None] = {}
        dist_cm: dict[int, float] = {}
        all_frames = sorted(frame_map.keys())
        for f in all_frames:
            dmap = _detection_map(frame_map[f])
            ca = _extract_center_px(dmap.get(mid_a, {}))
            cb = _extract_center_px(dmap.get(mid_b, {}))
            pos_a[f] = ca
            pos_b[f] = cb
            if ca is not None and cb is not None:
                dist_cm[f] = float(np.linalg.norm(ca - cb)) * cm_per_px

        sp_a: dict[int, float] = {}
        sp_b: dict[int, float] = {}
        for idx, f in enumerate(all_frames):
            pa_curr = pos_a.get(f)
            pa_prev = pos_a.get(f - 1)
            if pa_curr is not None and pa_prev is not None:
                sp_a[f] = float(np.linalg.norm(pa_curr - pa_prev)) * fps * cm_per_px
            elif idx + 1 < len(all_frames) and f == 0:
                pa_next = pos_a.get(f + 1)
                if pa_curr is not None and pa_next is not None:
                    sp_a[f] = float(np.linalg.norm(pa_next - pa_curr)) * fps * cm_per_px
                else:
                    sp_a[f] = 0.0
            else:
                sp_a[f] = 0.0

            pb_curr = pos_b.get(f)
            pb_prev = pos_b.get(f - 1)
            if pb_curr is not None and pb_prev is not None:
                sp_b[f] = float(np.linalg.norm(pb_curr - pb_prev)) * fps * cm_per_px
            elif idx + 1 < len(all_frames) and f == 0:
                pb_next = pos_b.get(f + 1)
                if pb_curr is not None and pb_next is not None:
                    sp_b[f] = float(np.linalg.norm(pb_next - pb_curr)) * fps * cm_per_px
                else:
                    sp_b[f] = 0.0
            else:
                sp_b[f] = 0.0

        # Identify follower during [start, end)
        score_a = 0.0
        score_b = 0.0
        for f in range(start, end):
            ca_curr = pos_a.get(f)
            cb_curr = pos_b.get(f)
            ca_prev = pos_a.get(f - 1)
            cb_prev = pos_b.get(f - 1)
            if ca_curr is not None and cb_curr is not None and ca_prev is not None and cb_prev is not None:
                va = (ca_curr - ca_prev) * fps
                vb = (cb_curr - cb_prev) * fps
                rab = cb_curr - ca_curr
                norm_r = float(np.linalg.norm(rab))
                if norm_r > 1.0:
                    proj_a = float(np.dot(va, rab)) / norm_r
                    proj_b = float(np.dot(vb, -rab)) / norm_r
                    if proj_a > proj_b:
                        score_a += 1.0
                    elif proj_b > proj_a:
                        score_b += 1.0

        follower_a = score_a >= score_b

        def get_kin(fn: int) -> tuple[float | None, float, float, float]:
            d = dist_cm.get(fn)
            spf = sp_a.get(fn, 0.0) if follower_a else sp_b.get(fn, 0.0)
            spl = sp_b.get(fn, 0.0) if follower_a else sp_a.get(fn, 0.0)
            cf_curr = pos_a.get(fn) if follower_a else pos_b.get(fn)
            cl_curr = pos_b.get(fn) if follower_a else pos_a.get(fn)
            cf_prev = pos_a.get(fn - 1) if follower_a else pos_b.get(fn - 1)
            cos_fol = 0.0
            if cf_curr is not None and cl_curr is not None and cf_prev is not None:
                vf = cf_curr - cf_prev
                rel = cl_curr - cf_curr
                norm_vf = float(np.linalg.norm(vf))
                norm_rel = float(np.linalg.norm(rel))
                if norm_vf > 1e-4 and norm_rel > 1e-4:
                    cos_fol = float(np.dot(vf, rel) / (norm_vf * norm_rel))
            return d, spf, spl, cos_fol

        # Onset backtracking: scan backward while separation <= max_separation_cm and follower pursuit maintained
        cand_s = start
        gap = 0
        for f in range(start - 1, scan_s - 1, -1):
            d, spf, spl, cos = get_kin(f)
            if d is not None and d <= max_separation_cm:
                is_pursuit = (
                    (cos >= 0.05 and spf >= min_follower_speed_cm_s)
                    or (spl >= 15.0 and cos >= -0.2)
                    or (d <= 12.0 and spl >= 20.0)
                    or (d <= 10.0 and max(spf, spl) >= min_follower_speed_cm_s and cos >= -0.2)
                )
                if is_pursuit:
                    cand_s = f
                    gap = 0
                else:
                    gap += 1
                    if gap > gap_limit:
                        break
            else:
                gap += 1
                if gap > gap_limit:
                    break

        # Snap to clip start if start is close to beginning and separation is reasonable
        d0 = dist_cm.get(0)
        if cand_s <= 1 and start <= int(round(0.25 * fps)) and d0 is not None and d0 <= 25.0:
            cand_s = 0

        # Forward extension if pursuit continues past end
        cand_e = end
        gap_e = 0
        for f in range(end, scan_e):
            d, spf, spl, cos = get_kin(f)
            if d is not None and d <= max_separation_cm:
                is_active = (
                    (cos >= -0.1 and max(spf, spl) >= min_follower_speed_cm_s)
                    or (spl >= 12.0 and cos >= -0.3)
                    or (d <= 10.0 and max(spf, spl) >= 3.0)
                )
                if is_active:
                    cand_e = f + 1
                    gap_e = 0
                else:
                    gap_e += 1
                    if gap_e > gap_limit:
                        break
            else:
                gap_e += 1
                if gap_e > gap_limit:
                    break

        # Trim trailing frames where pursuit stopped or chaser turned away
        min_span_frames = max(4, int(round(req_s * fps)))
        trim_e = cand_e
        while trim_e > cand_s + min_span_frames:
            last_f = trim_e - 1
            d, spf, spl, cos = get_kin(last_f)
            if (
                d is None
                or d > max_separation_cm
                or (cos < -0.3 and d > 12.0)
                or (spf < 3.5 and spl < 3.5 and d > 8.0)
            ):
                trim_e -= 1
            else:
                break
        cand_e = trim_e

        if frame_count is not None and cand_e >= int(frame_count) - 2:
            cand_e = int(frame_count)

        dur_s = (cand_e - cand_s) / fps
        if dur_s < req_s - 1e-9:
            fallback = update_segment_bounds(event, start, end, fps, suffix="chasing_fallback")
            refined.append(fallback)
            continue

        cropped = update_segment_bounds(event, cand_s, cand_e, fps, suffix="chasing_crop")
        if qualify_event_duration(cropped, fps, parameters=params)["ok"]:
            refined.append(cropped)
        else:
            fallback = update_segment_bounds(event, start, end, fps, suffix="chasing_fallback")
            refined.append(fallback)

    return refined


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
    """Fine-crop contact and together segments to stay bounded on real interaction intervals."""
    from mouse_behavior.evaluation.screening_signals import runs
    from mouse_behavior.evaluation.social_heuristic_config import DEFAULT_SOCIAL_HEURISTIC_CONFIG

    fps = _resolve_fps(binding)
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
        mids = [int(m) for m in event.get("mouse_ids", ())]
        if len(set(mids)) < 2 or end <= start:
            continue
        mid_a, mid_b = mids[0], mids[1]
        if mid_a == mid_b:
            unique = list(dict.fromkeys(mids))
            if len(unique) < 2:
                continue
            mid_a, mid_b = unique[0], unique[1]

        rows = list(store.rows(start, end))
        if not rows:
            continue
        span = end - start
        cm_per_px = _resolve_scale_cm_per_px(rows, [mid_a, mid_b], binding, scale=scale)
        centers_cm, observed = _clip_multi_track_series(rows, start, end, [mid_a, mid_b], cm_per_px)
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
            has_kp_pair = False
            for idx, row in enumerate(rows):
                raw_frame = row.get("frame", row.get("source_frame"))
                rel = int(raw_frame) - start if raw_frame is not None else idx
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
                if da is not None and db is not None and np.isfinite(kp_a[rel]).any() and np.isfinite(kp_b[rel]).any():
                    has_kp_pair = True

            d_min = np.full(span, np.nan, dtype=float)
            for rel in np.flatnonzero(pair_obs):
                if has_kp_pair:
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
            obs_sub = np.flatnonzero(strict[rel_s:rel_e])
            if len(obs_sub) < 2:
                continue
            true_s = start + rel_s + int(obs_sub[0])
            true_e = start + rel_s + int(obs_sub[-1]) + 1
            cropped = update_segment_bounds(event, true_s, true_e, fps, suffix=suffix)
            supp_slice = strict[true_s - start : true_e - start]
            supp_intervals = _observed_support_intervals(supp_slice, true_s)
            if not supp_intervals:
                continue
            cropped["canonical_support_frame_intervals"] = supp_intervals
            cropped["canonical_support_frame_count"] = sum(b - a for a, b in supp_intervals)
            if qualify_event_duration(cropped, fps, parameters=params)["ok"]:
                refined.append(cropped)

    return refined
