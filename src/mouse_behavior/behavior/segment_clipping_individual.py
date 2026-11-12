"""Individual-layer behavior segment clipping (Static, Locomotion, Rearing)."""
from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from mouse_behavior.behavior.duration_policy import qualify_event_duration
from mouse_behavior.behavior.segment_clipping_common import (
    _clip_dense_track_series,
    _detection_map,
    _observed_support_intervals,
    _resolve_fps,
    _resolve_parameters,
    _resolve_scale_cm_per_px,
    update_segment_bounds,
)

__all__ = [
    "refine_locomotion_segments",
    "refine_rearing_segments",
    "refine_static_segments",
]


def refine_rearing_segments(
    events: Sequence[Mapping[str, Any]],
    store: Any,
    binding: Mapping[str, Any] | None = None,
    *,
    min_duration_s: float = 0.5,
    max_duration_s: float | None = None,
    max_mounted_fraction: float = 0.20,
    max_box_aspect_ratio: float = 1.65,
    max_ear_neck_px: float = 20.0,
) -> list[dict[str, Any]]:
    """Fine-crop and validate standalone upright rearing segments.

    Suppresses social mounting/climbing (bounding box overlap with another
    detected mouse) and curled dorsal grooming (`cos_angle > 0.15`), and trims
    unobserved endpoint frames. `max_duration_s` defaults to `None` so long
    rearing bouts are never arbitrarily discarded unless a caller specifies a cap.
    """
    fps = _resolve_fps(binding)
    refined: list[dict[str, Any]] = []
    for event in events:
        if event.get("canonical_behavior") != "rearing":
            refined.append(copy.deepcopy(dict(event)))
            continue
        start = int(event["start_frame"])
        end = int(event["end_frame"])
        mouse_ids = event.get("mouse_ids", ())
        if not mouse_ids or end <= start:
            continue
        mid = int(mouse_ids[0])
        rows = list(store.rows(start, end))
        if not rows:
            continue

        valid_posture = np.zeros(len(rows), dtype=bool)
        present_mask = np.zeros(len(rows), dtype=bool)
        mounted_frames = 0
        ear_neck_vals: list[float] = []
        aspect_vals: list[float] = []
        cos_vals: list[float] = []

        for idx, row in enumerate(rows):
            det_map = _detection_map(row)
            det = det_map.get(mid)
            if det is None:
                continue
            box = np.asarray(det.get("box", []), dtype=float)
            pts = np.asarray(det.get("keypoints", []), dtype=float)
            if box.shape != (4,) or not np.isfinite(box).all() or not (box[2:] > box[:2]).all():
                continue
            present_mask[idx] = True
            bw = max(1.0, float(box[2] - box[0]))
            bh = max(1.0, float(box[3] - box[1]))
            aspect = float(max(bw, bh) / min(bw, bh))
            aspect_vals.append(aspect)

            other_overlap = False
            for oid, odet in det_map.items():
                if oid == mid:
                    continue
                obox = np.asarray(odet.get("box", []), dtype=float)
                if obox.shape == (4,) and np.isfinite(obox).all():
                    ix1 = max(float(box[0]), float(obox[0]))
                    iy1 = max(float(box[1]), float(obox[1]))
                    ix2 = min(float(box[2]), float(obox[2]))
                    iy2 = min(float(box[3]), float(obox[3]))
                    if ix2 > ix1 and iy2 > iy1:
                        inter = (ix2 - ix1) * (iy2 - iy1)
                        if inter / (bw * bh) >= 0.12:
                            other_overlap = True
                            break
            if other_overlap:
                mounted_frames += 1

            if pts.shape == (7, 3) and np.all(pts[[0, 1, 2, 3, 6], 2] >= 0.25):
                ear_mid = 0.5 * (pts[1, :2] + pts[2, :2])
                ear_neck = float(np.linalg.norm(ear_mid - pts[3, :2]))
                nose_tail = float(np.linalg.norm(pts[0, :2] - pts[6, :2]))
                v_hn = pts[0, :2] - pts[3, :2]
                v_tn = pts[6, :2] - pts[3, :2]
                nhn = float(np.linalg.norm(v_hn))
                ntn = float(np.linalg.norm(v_tn))
                cos_ang = float(np.dot(v_hn, v_tn) / (nhn * ntn)) if (nhn > 2.0 and ntn > 2.0) else 0.0
                ear_neck_vals.append(ear_neck)
                cos_vals.append(cos_ang)
                if (
                    not other_overlap
                    and ear_neck <= max_ear_neck_px
                    and nose_tail < 72.0
                    and aspect <= max_box_aspect_ratio
                    and cos_ang <= 0.15
                ):
                    valid_posture[idx] = True

        if not present_mask.any():
            continue

        # Bridge short internal posture jitter gaps (<= 6 frames) when endpoints are unmounted
        bridged = valid_posture.copy()
        pos_idx = np.flatnonzero(bridged)
        if len(pos_idx) >= 2:
            for a, b in zip(pos_idx[:-1], pos_idx[1:]):
                if 1 < (b - a) <= 6 and present_mask[a:b].all():
                    bridged[a:b] = True

        # Extract candidate sub-bouts or fall back to full event if whole-event medians qualify
        candidates: list[tuple[int, int]] = []
        idx = 0
        while idx < len(bridged):
            if not bridged[idx]:
                idx += 1
                continue
            j = idx + 1
            while j < len(bridged) and bridged[j]:
                j += 1
            candidates.append((start + idx, start + j))
            idx = j

        if not candidates:
            med_ear = float(np.median(ear_neck_vals)) if ear_neck_vals else 999.0
            med_asp = float(np.median(aspect_vals)) if aspect_vals else 999.0
            med_cos = float(np.median(cos_vals)) if cos_vals else 1.0
            mounted_ratio = mounted_frames / max(1, int(present_mask.sum()))
            if (
                mounted_ratio <= max_mounted_fraction
                and med_ear <= max_ear_neck_px
                and med_asp <= max_box_aspect_ratio
                and med_cos <= 0.10
            ):
                p_idx = np.flatnonzero(present_mask)
                candidates.append((start + int(p_idx[0]), start + int(p_idx[-1]) + 1))

        for sub_start, sub_end in candidates:
            rel_s = sub_start - start
            rel_e = sub_end - start
            p_sub = np.flatnonzero(present_mask[rel_s:rel_e])
            if not len(p_sub):
                continue
            true_s = sub_start + int(p_sub[0])
            true_e = sub_start + int(p_sub[-1]) + 1
            dur_s = (true_e - true_s) / fps
            if dur_s + 1e-9 < min_duration_s:
                continue
            if max_duration_s is not None and dur_s - 1e-9 > max_duration_s:
                continue
            cropped = update_segment_bounds(event, true_s, true_e, fps, suffix="rearing_crop")
            if qualify_event_duration(cropped, fps, parameters=_resolve_parameters(binding))["ok"]:
                refined.append(cropped)
    return refined


def refine_static_segments(
    events: Sequence[Mapping[str, Any]],
    store: Any,
    binding: Mapping[str, Any] | None = None,
    *,
    min_duration_s: float = 1.0,
    max_radius_cm: float = 4.0,
    max_mean_speed_cm_s: float = 6.0,
    snap_head_tail_s: float = 2.0,
) -> list[dict[str, Any]]:
    """Fine-crop and validate static (stationary) sub-segments for resting mice.

    Uses physical-unit frame-level stationarity (``individual_rule_masks``) to
    trim leading/trailing locomotion transitions and extract qualified stationary
    bouts, snapping only unobserved leading/trailing detection dropouts within
    ``snap_head_tail_s``.
    """
    from mouse_behavior.behavior.individual_rules import (
        IndividualHeuristicConfig,
        _median,
        individual_rule_masks,
    )

    fps = _resolve_fps(binding)
    params = _resolve_parameters(binding)
    refined: list[dict[str, Any]] = []

    req_dur = (
        float(min_duration_s)
        if min_duration_s != 1.0
        else float(getattr(params, "stationary_min_duration_s", min_duration_s))
    )
    snap_frames = max(0, int(round(snap_head_tail_s * fps)))
    speed_gate = min(4.0, float(max_mean_speed_cm_s)) if max_mean_speed_cm_s < 4.0 else 4.0

    for event in events:
        c_beh = str(event.get("canonical_behavior") or event.get("behavior") or "").lower()
        if c_beh not in {"static", "stationary"}:
            refined.append(copy.deepcopy(dict(event)))
            continue
        start = int(event["start_frame"])
        end = int(event["end_frame"])
        mids = [int(m) for m in event.get("mouse_ids", ())]
        if len(mids) == 0 or end <= start:
            continue
        if len(mids) > 1 and "target_mouse_id" not in event:
            continue
        target_id = int(event.get("target_mouse_id", mids[0]))

        rows = list(store.rows(start, end))
        if not rows:
            continue

        scale_cm_per_px = _resolve_scale_cm_per_px(rows, mids, binding)
        centers_cm, observed = _clip_dense_track_series(rows, start, end, target_id, scale_cm_per_px)
        if int(observed.sum()) < 2:
            continue

        cfg = IndividualHeuristicConfig(
            motion_speed_threshold_cm_s=speed_gate,
            static_min_duration_s=req_dur,
            static_max_radius_cm=float(max_radius_cm),
            static_transient_max_displacement_cm=float(max_radius_cm),
            static_transient_max_s=1.0,
            static_detection_gap_max_s=1.0,
        )
        res = individual_rule_masks(centers_cm, observed, fps, config=cfg)
        static_mask = res.masks["stationary"][:, 0]
        if not static_mask.any():
            continue

        smoothed = _median(centers_cm, max(1, round(cfg.smooth_window_s * fps)))
        smoothed[~observed] = np.nan
        span = end - start
        spd_diag = res.diagnostics.get("body_speed_cm_s", np.full((span, 1), np.nan))[:, 0]

        changes = np.diff(np.r_[False, static_mask, False].astype(np.int8))
        bout_runs = list(zip(np.flatnonzero(changes == 1).tolist(), np.flatnonzero(changes == -1).tolist()))

        for run_idx, (rel_s, rel_e) in enumerate(bout_runs):
            sub_obs = np.flatnonzero(observed[rel_s:rel_e, 0])
            if len(sub_obs) < 2:
                continue
            pts_raw = centers_cm[rel_s:rel_e, 0][sub_obs]
            med_center = np.median(pts_raw, axis=0)

            # Include observed leading/trailing boundary frames whose speed was NaN
            # across an adjacent dropout when their position stays within max_radius_cm.
            if run_idx == 0:
                while (
                    rel_s > 0
                    and observed[rel_s - 1, 0]
                    and not np.isfinite(spd_diag[rel_s - 1])
                    and float(np.linalg.norm(centers_cm[rel_s - 1, 0] - med_center)) <= max_radius_cm
                ):
                    rel_s -= 1
            if run_idx == len(bout_runs) - 1:
                while (
                    rel_e < span
                    and observed[rel_e, 0]
                    and not np.isfinite(spd_diag[rel_e])
                    and float(np.linalg.norm(centers_cm[rel_e, 0] - med_center)) <= max_radius_cm
                ):
                    rel_e += 1

            sub_obs = np.flatnonzero(observed[rel_s:rel_e, 0])
            if len(sub_obs) < 2:
                continue
            pts_raw = centers_cm[rel_s:rel_e, 0][sub_obs]
            med_center = np.median(pts_raw, axis=0)
            max_rad_cm = float(np.linalg.norm(pts_raw - med_center, axis=1).max())
            if max_rad_cm > max_radius_cm:
                continue

            sm_pts = smoothed[rel_s:rel_e, 0]
            sm_valid = observed[rel_s:rel_e, 0] & np.isfinite(sm_pts).all(axis=1)
            pts_eval = sm_pts[sm_valid] if int(sm_valid.sum()) >= 2 else pts_raw
            path_cm = float(np.linalg.norm(np.diff(pts_eval, axis=0), axis=1).sum())
            obs_dur_s = (int(sub_obs[-1]) - int(sub_obs[0]) + 1) / fps
            if (path_cm / max(obs_dur_s, 1e-6)) > max_mean_speed_cm_s:
                continue

            # Snap leading/trailing margins ONLY if those margin frames are unobserved dropouts
            if run_idx == 0 and rel_s > 0 and rel_s <= snap_frames and not observed[:rel_s, 0].any():
                true_s = start
            else:
                true_s = start + rel_s

            if (
                run_idx == len(bout_runs) - 1
                and rel_e < span
                and (span - rel_e) <= snap_frames
                and not observed[rel_e:, 0].any()
            ):
                true_e = end
            else:
                true_e = start + rel_e

            dur_s = (true_e - true_s) / fps
            if dur_s < req_dur - 1e-9:
                continue

            cropped = update_segment_bounds(event, true_s, true_e, fps, suffix="static_crop")
            supp_intervals = _observed_support_intervals(observed[true_s - start : true_e - start, 0], true_s)
            cropped["canonical_support_frame_intervals"] = supp_intervals
            cropped["canonical_support_frame_count"] = sum(e - s for s, e in supp_intervals)

            if qualify_event_duration(cropped, fps, parameters=params)["ok"]:
                refined.append(cropped)

    return refined


def refine_locomotion_segments(
    events: Sequence[Mapping[str, Any]],
    store: Any,
    binding: Mapping[str, Any] | None = None,
    *,
    min_duration_s: float = 0.5,
    min_displacement_cm: float = 3.0,
    min_speed_cm_s: float = 3.0,
    snap_head_tail_s: float = 2.0,
) -> list[dict[str, Any]]:
    """Fine-crop and validate locomotion sub-segments for moving mice.

    Uses physical-unit frame-level locomotion detection (``individual_rule_masks``)
    and smoothed center kinematics to trim leading/trailing stationary segments
    and reject static jitter, snapping only unobserved leading/trailing detection
    dropouts within ``snap_head_tail_s``.
    """
    from mouse_behavior.behavior.individual_rules import (
        IndividualHeuristicConfig,
        _median,
        individual_rule_masks,
    )

    fps = _resolve_fps(binding)
    params = _resolve_parameters(binding)
    refined: list[dict[str, Any]] = []

    req_dur = (
        float(min_duration_s)
        if min_duration_s != 0.5
        else float(getattr(params, "locomotion_min_duration_s", min_duration_s))
    )
    static_min_dur = float(getattr(params, "stationary_min_duration_s", 1.0))
    snap_frames = max(0, int(round(snap_head_tail_s * fps)))

    for event in events:
        c_beh = str(event.get("canonical_behavior") or event.get("behavior") or "").lower()
        if c_beh != "locomotion":
            refined.append(copy.deepcopy(dict(event)))
            continue
        start = int(event["start_frame"])
        end = int(event["end_frame"])
        mids = [int(m) for m in event.get("mouse_ids", ())]
        if len(mids) == 0 or end <= start:
            continue
        if len(mids) > 1 and "target_mouse_id" not in event:
            continue
        target_id = int(event.get("target_mouse_id", mids[0]))

        rows = list(store.rows(start, end))
        if not rows:
            continue

        scale_cm_per_px = _resolve_scale_cm_per_px(rows, mids, binding)
        centers_cm, observed = _clip_dense_track_series(rows, start, end, target_id, scale_cm_per_px)
        if int(observed.sum()) < 2:
            continue

        cfg = IndividualHeuristicConfig(
            locomotion_min_duration_s=req_dur,
            static_min_duration_s=static_min_dur,
            static_max_radius_cm=2.5,
            static_transient_max_displacement_cm=2.5,
        )
        res = individual_rule_masks(centers_cm, observed, fps, config=cfg)
        loco_mask = res.masks["locomotion"][:, 0]
        if not loco_mask.any():
            continue

        smoothed = _median(centers_cm, max(1, round(cfg.smooth_window_s * fps)))
        smoothed[~observed] = np.nan
        span = end - start
        spd_diag = res.diagnostics.get("body_speed_cm_s", np.full((span, 1), np.nan))[:, 0]

        changes = np.diff(np.r_[False, loco_mask, False].astype(np.int8))
        bout_runs = list(zip(np.flatnonzero(changes == 1).tolist(), np.flatnonzero(changes == -1).tolist()))

        for run_idx, (rel_s, rel_e) in enumerate(bout_runs):
            if run_idx == 0:
                while (
                    rel_s > 0
                    and observed[rel_s - 1, 0]
                    and not np.isfinite(spd_diag[rel_s - 1])
                    and observed[rel_s, 0]
                    and float(np.linalg.norm(centers_cm[rel_s, 0] - centers_cm[rel_s - 1, 0])) * fps >= min_speed_cm_s
                ):
                    rel_s -= 1
            if run_idx == len(bout_runs) - 1:
                while (
                    rel_e < span
                    and observed[rel_e, 0]
                    and not np.isfinite(spd_diag[rel_e])
                    and observed[rel_e - 1, 0]
                    and float(np.linalg.norm(centers_cm[rel_e, 0] - centers_cm[rel_e - 1, 0])) * fps >= min_speed_cm_s
                ):
                    rel_e += 1

            sub_obs = np.flatnonzero(observed[rel_s:rel_e, 0])
            if len(sub_obs) < 2:
                continue

            sm_pts = smoothed[rel_s:rel_e, 0]
            sm_valid = observed[rel_s:rel_e, 0] & np.isfinite(sm_pts).all(axis=1)
            pts_eval = sm_pts[sm_valid] if int(sm_valid.sum()) >= 2 else centers_cm[rel_s:rel_e, 0][sub_obs]

            path_cm = float(np.linalg.norm(np.diff(pts_eval, axis=0), axis=1).sum())
            net_cm = float(np.linalg.norm(pts_eval[-1] - pts_eval[0]))
            span_cm = float(np.linalg.norm(pts_eval - np.median(pts_eval, axis=0), axis=1).max())

            obs_dur_s = (int(sub_obs[-1]) - int(sub_obs[0]) + 1) / fps
            mean_speed = path_cm / max(obs_dur_s, 1e-6)

            if (
                path_cm < min_displacement_cm
                or max(net_cm, span_cm) < min_displacement_cm
                or mean_speed < min_speed_cm_s
            ):
                continue

            # Snap leading/trailing margins ONLY if those margin frames are unobserved dropouts
            if run_idx == 0 and rel_s > 0 and rel_s <= snap_frames and not observed[:rel_s, 0].any():
                true_s = start
            else:
                true_s = start + rel_s

            if (
                run_idx == len(bout_runs) - 1
                and rel_e < span
                and (span - rel_e) <= snap_frames
                and not observed[rel_e:, 0].any()
            ):
                true_e = end
            else:
                true_e = start + rel_e

            dur_s = (true_e - true_s) / fps
            if dur_s < req_dur - 1e-9:
                continue

            cropped = update_segment_bounds(event, true_s, true_e, fps, suffix="locomotion_crop")
            supp_intervals = _observed_support_intervals(observed[true_s - start : true_e - start, 0], true_s)
            cropped["canonical_support_frame_intervals"] = supp_intervals
            cropped["canonical_support_frame_count"] = sum(e - s for s, e in supp_intervals)

            if qualify_event_duration(cropped, fps, parameters=params)["ok"]:
                refined.append(cropped)

    return refined
