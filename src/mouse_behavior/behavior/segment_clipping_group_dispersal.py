"""Dispersal behavior segment clipping and origin-member recovery."""
from __future__ import annotations

import copy
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from mouse_behavior.behavior.duration_policy import qualify_event_duration
from mouse_behavior.behavior.segment_clipping_common import (
    _detection_map,
    _extract_center_px,
    _resolve_fps,
    _resolve_parameters,
    _resolve_scale_cm_per_px,
    update_segment_bounds,
)
from mouse_behavior.evaluation.exported_dataset_models import HeuristicParameters

__all__ = [
    "refine_dispersal_segments",
]


def _recover_dispersal_origin_members(
    search_rows: Sequence[Mapping[str, Any]],
    in_rows: Sequence[Mapping[str, Any]],
    raw_mids: list[int],
    cluster_radius_px: float,
) -> list[int]:
    """Recover full >=3 compact origin group around onset when raw dispersal lists 2 departing IDs."""
    raw_set = set(raw_mids)
    in_counts: Counter[int] = Counter()
    for row in in_rows:
        for tid in _detection_map(row):
            in_counts[int(tid)] += 1

    best_origin_frame: int | None = None
    best_anchor: np.ndarray | None = None
    best_rad = float("inf")
    for row in search_rows:
        fn = int(row.get("frame", row.get("source_frame", 0)))
        dmap = _detection_map(row)
        pts = []
        for m in raw_mids:
            if m in dmap:
                c = _extract_center_px(dmap[m])
                if c is not None:
                    pts.append(c)
        if len(pts) == len(raw_mids):
            arr = np.asarray(pts, dtype=float)
            anchor = arr.mean(axis=0)
            rad = float(np.linalg.norm(arr - anchor, axis=1).mean())
            if rad <= cluster_radius_px and (best_origin_frame is None or rad < best_rad):
                best_origin_frame = fn
                best_anchor = anchor
                best_rad = rad

    if best_origin_frame is None or best_anchor is None:
        return raw_mids

    recovered = set(raw_set)
    first_seen_after: dict[int, tuple[int, np.ndarray]] = {}
    for row in search_rows:
        fn = int(row.get("frame", row.get("source_frame", 0)))
        dmap = _detection_map(row)
        if fn == best_origin_frame:
            for tid, det in dmap.items():
                if tid in raw_set or in_counts[tid] < 2:
                    continue
                c = _extract_center_px(det)
                if c is not None and float(np.linalg.norm(c - best_anchor)) <= cluster_radius_px:
                    recovered.add(int(tid))
        elif fn > best_origin_frame:
            for tid, det in dmap.items():
                if tid in raw_set or tid in first_seen_after or in_counts[tid] < 2:
                    continue
                c = _extract_center_px(det)
                if c is not None:
                    first_seen_after[int(tid)] = (fn, c)
    for tid, (_, pos) in first_seen_after.items():
        if float(np.linalg.norm(pos - best_anchor)) <= cluster_radius_px:
            recovered.add(tid)
    return sorted(recovered)


def refine_dispersal_segments(
    events: Sequence[Mapping[str, Any]],
    store: Any,
    binding: Mapping[str, Any] | None = None,
    *,
    scale: Mapping[str, Any] | None = None,
    parameters: HeuristicParameters | None = None,
    min_duration_s: float = 5.0,
    max_duration_s: float | None = 15.0,
    cluster_radius_px: float | None = None,
    min_radius_gain_px: float | None = None,
    min_net_gain_px: float | None = None,
) -> list[dict[str, Any]]:
    """Fine-crop dispersal segments from compact/motion onset through sustained separation."""
    fps = _resolve_fps(binding)
    params = _resolve_parameters(binding, parameters)
    eff_min_duration_s = (
        float(params.dispersal_min_duration_s) if min_duration_s == 5.0 else float(min_duration_s)
    )
    eff_max_duration_s = float(max_duration_s) if max_duration_s is not None else None
    max_frames = (
        max(1, int(math.floor(eff_max_duration_s * fps + 1e-9)))
        if eff_max_duration_s is not None
        else None
    )
    binding_scale = (
        binding.get("scale")
        if isinstance(binding, Mapping) and isinstance(binding.get("scale"), Mapping)
        else {}
    )
    binding_video = (
        binding.get("source_video")
        if isinstance(binding, Mapping) and isinstance(binding.get("source_video"), Mapping)
        else {}
    )
    has_explicit_scale = any(
        isinstance(m, Mapping)
        and ("cm_per_pixel" in m or "px_per_cm" in m or "scale_px_per_cm" in m)
        for m in (scale, binding_scale, binding_video, binding)
    )
    lookback_frames = int(round(4.0 * fps))
    early_cutoff_frames = int(round(1.5 * fps))
    snap_gap_frames = int(round(2.0 * fps))
    reconv_min_frames = max(3, int(round(0.5 * fps)))
    quiet_tail_limit_frames = int(round(6.0 * fps))
    settle_hold_frames = int(round(2.0 * fps))

    refined: list[dict[str, Any]] = []
    for event in events:
        if event.get("canonical_behavior") != "dispersal":
            refined.append(copy.deepcopy(dict(event)))
            continue
        start = int(event["start_frame"])
        end = int(event["end_frame"])
        raw_mids = list(dict.fromkeys(int(m) for m in event.get("mouse_ids", ())))
        if len(raw_mids) < 2 or end <= start:
            continue
        rows = list(store.rows(start, end))
        if not rows:
            continue

        cm_per_px = _resolve_scale_cm_per_px(rows, raw_mids, binding, scale=scale)
        scale_factor = (0.10 / max(cm_per_px, 1e-6)) if has_explicit_scale else 1.0
        eff_cluster_radius_px = (
            float(cluster_radius_px)
            if cluster_radius_px is not None
            else 120.0 * max(1.0, scale_factor)
        )
        eff_min_radius_gain_px = (
            float(min_radius_gain_px)
            if min_radius_gain_px is not None
            else 35.0 * scale_factor
        )
        eff_min_net_gain_px = (
            float(min_net_gain_px) if min_net_gain_px is not None else 25.0 * scale_factor
        )

        lb_start = max(0, start - lookback_frames)
        pre_rows = list(store.rows(lb_start, start)) if lb_start < start else []

        departing_subset: list[int] | None = None
        mids = list(raw_mids)
        if len(mids) < 3:
            search_rows = pre_rows + [
                r for r in rows if int(r.get("frame", r.get("source_frame", 0))) <= start + early_cutoff_frames
            ]
            if not search_rows:
                search_rows = pre_rows + rows
            recovered_mids = _recover_dispersal_origin_members(
                search_rows,
                rows,
                raw_mids,
                eff_cluster_radius_px,
            )
            if len(recovered_mids) < 3 and len(search_rows) < len(pre_rows) + len(rows):
                recovered_mids = _recover_dispersal_origin_members(
                    pre_rows + rows,
                    rows,
                    raw_mids,
                    eff_cluster_radius_px,
                )
            if len(recovered_mids) < 3:
                continue
            departing_subset = list(raw_mids)
            mids = recovered_mids

        obs_counts: dict[int, int] = {m: 0 for m in mids}
        in_counts: dict[int, int] = {m: 0 for m in mids}
        pre_rads: list[tuple[int, int, float]] = []
        for row in pre_rows:
            fn = int(row.get("frame", row.get("source_frame", 0)))
            det_map = _detection_map(row)
            centers = []
            for m in mids:
                if m in det_map:
                    c = _extract_center_px(det_map[m])
                    if c is not None:
                        obs_counts[m] += 1
                        centers.append(c)
            if len(centers) >= 2:
                pts = np.asarray(centers, dtype=float)
                rad = float(np.linalg.norm(pts - pts.mean(axis=0), axis=1).mean())
                pre_rads.append((fn, len(centers), rad))

        rads: list[tuple[int, int, float, np.ndarray | None]] = []
        dep_rads: list[float] = []
        for row in rows:
            fn = int(row.get("frame", row.get("source_frame", 0)))
            det_map = _detection_map(row)
            centers = []
            dep_centers = []
            for m in mids:
                if m in det_map:
                    c = _extract_center_px(det_map[m])
                    if c is not None:
                        obs_counts[m] += 1
                        in_counts[m] += 1
                        centers.append(c)
                        if departing_subset is not None and m in departing_subset:
                            dep_centers.append(c)
            if len(centers) >= 2:
                pts = np.asarray(centers, dtype=float)
                rad = float(np.linalg.norm(pts - pts.mean(axis=0), axis=1).mean())
                rads.append((fn, len(centers), rad, pts.mean(axis=0)))
            if departing_subset is not None and len(dep_centers) >= 2:
                dpts = np.asarray(dep_centers, dtype=float)
                dep_rads.append(float(np.linalg.norm(dpts - dpts.mean(axis=0), axis=1).mean()))

        if len(rads) < 10 or sum(1 for cnt in in_counts.values() if cnt >= 2) < 3:
            continue
        if any(obs_counts[m] < 1 for m in mids):
            continue

        frames_arr = np.array([r[0] for r in rads], dtype=int)
        nobs_arr = np.array([r[1] for r in rads], dtype=int)
        vals_arr = np.array([r[2] for r in rads], dtype=float)
        first_obs_frame = int(frames_arr[0])

        early_all_compact = [
            i
            for i in range(len(rads))
            if int(frames_arr[i]) <= start + early_cutoff_frames
            and int(nobs_arr[i]) == len(mids)
            and float(vals_arr[i]) <= eff_cluster_radius_px
        ]
        early_any_compact = [
            i
            for i in range(len(rads))
            if int(frames_arr[i]) <= start + early_cutoff_frames
            and float(vals_arr[i]) <= eff_cluster_radius_px
        ]
        if early_all_compact or early_any_compact:
            cand_compact = early_all_compact if early_all_compact else early_any_compact
            new_s = start if (first_obs_frame - start) <= snap_gap_frames else first_obs_frame
            if early_all_compact:
                all_compact = [
                    i
                    for i in range(len(rads))
                    if int(nobs_arr[i]) == len(mids)
                    and float(vals_arr[i]) <= eff_cluster_radius_px
                ]
                origin_rad = float(
                    min(vals_arr[i] for i in all_compact[: max(1, len(early_all_compact) + 30)])
                )
            else:
                origin_rad = float(min(vals_arr[i] for i in cand_compact))
        else:
            pre_all_compact = [
                r[2] for r in pre_rads if r[1] == len(mids) and r[2] <= eff_cluster_radius_px
            ]
            pre_any_compact = [r[2] for r in pre_rads if r[2] <= eff_cluster_radius_px]
            if pre_all_compact or pre_any_compact:
                origin_rad = float(min(pre_all_compact if pre_all_compact else pre_any_compact))
                new_s = start if (first_obs_frame - start) <= snap_gap_frames else first_obs_frame
            else:
                all_compact = [
                    i
                    for i in range(len(rads))
                    if int(nobs_arr[i]) == len(mids)
                    and float(vals_arr[i]) <= eff_cluster_radius_px
                ]
                any_compact = [
                    i for i in range(len(rads)) if float(vals_arr[i]) <= eff_cluster_radius_px
                ]
                chosen = all_compact if all_compact else any_compact
                if not chosen:
                    continue
                new_s = int(frames_arr[chosen[0]])
                origin_rad = float(vals_arr[chosen[0]])

        after_indices = [i for i in range(len(rads)) if int(frames_arr[i]) >= new_s]
        if not after_indices:
            continue
        peak_i = max(after_indices, key=lambda idx: vals_arr[idx])
        peak_rad = float(vals_arr[peak_i])
        dep_gain_ok = (
            departing_subset is not None
            and len(dep_rads) >= 2
            and (max(dep_rads) - min(dep_rads)) >= eff_min_radius_gain_px
            and (peak_rad - origin_rad) >= min(eff_min_radius_gain_px, 2.0 / max(cm_per_px, 1e-6))
        )
        if (peak_rad - origin_rad) < eff_min_radius_gain_px and not dep_gain_ok:
            continue

        tail_cut_i: int | None = None
        low_streak: list[int] = []
        for idx in range(peak_i + 1, len(rads)):
            if (float(vals_arr[idx]) - origin_rad) < eff_min_net_gain_px:
                low_streak.append(idx)
                if len(low_streak) >= reconv_min_frames:
                    tail_cut_i = low_streak[0]
                    break
            else:
                low_streak.clear()

        if tail_cut_i is not None:
            exp_before = [
                idx
                for idx in range(peak_i, tail_cut_i)
                if (float(vals_arr[idx]) - origin_rad) >= eff_min_net_gain_px
            ]
            if not exp_before:
                continue
            new_e = int(frames_arr[exp_before[-1]]) + 1
        else:
            exp_indices = [
                idx
                for idx in after_indices
                if (float(vals_arr[idx]) - origin_rad) >= eff_min_net_gain_px
            ]
            if not exp_indices:
                continue
            last_exp_f = int(frames_arr[exp_indices[-1]])
            new_e = (
                end
                if (end - (last_exp_f + 1)) <= snap_gap_frames
                else (last_exp_f + 1)
            )

        # Trim prolonged post-departure quiet tail when exceeding 6.0s after peak or exceeding max_duration_s
        post_peak_indices = [
            idx for idx in range(peak_i, len(rads)) if int(frames_arr[idx]) < new_e
        ]
        if len(post_peak_indices) > 2:
            post_tail_frames = int(frames_arr[post_peak_indices[-1]]) - int(frames_arr[peak_i])
            if post_tail_frames > quiet_tail_limit_frames or (
                max_frames is not None and (new_e - new_s) > max_frames
            ):
                rad_diffs = np.abs(np.diff(vals_arr[post_peak_indices]))
                quiet_thresh_px = 1.5 * scale_factor
                last_active_idx = peak_i
                for pos_k, d_rad in enumerate(rad_diffs):
                    if d_rad >= quiet_thresh_px:
                        last_active_idx = post_peak_indices[pos_k + 1]
                active_end_f = int(frames_arr[last_active_idx]) + settle_hold_frames + 1
                if (new_e - active_end_f) > int(round(fps)) and (active_end_f - new_s) / fps > eff_min_duration_s:
                    cands_e = [
                        idx
                        for idx in post_peak_indices
                        if int(frames_arr[idx]) < active_end_f
                        and (float(vals_arr[idx]) - origin_rad) >= eff_min_net_gain_px
                        and int(nobs_arr[idx]) == len(mids)
                    ]
                    if cands_e:
                        new_e = int(frames_arr[cands_e[-1]]) + 1

        # Trim excess compact pre-departure idle prefix & enforce max_duration_s (15.0s)
        if max_frames is not None and (new_e - new_s) > max_frames:
            rise_thresh = origin_rad + max(10.0 * scale_factor, 0.20 * eff_min_radius_gain_px)
            rise_candidates = [
                idx
                for idx in after_indices
                if idx <= peak_i and float(vals_arr[idx]) >= rise_thresh
            ]
            if rise_candidates:
                onset_idx = rise_candidates[0]
                prefix_keep = int(round(1.5 * fps))
                cand_s = max(new_s, int(frames_arr[onset_idx]) - prefix_keep)
                if (new_e - cand_s) / fps > eff_min_duration_s and cand_s > new_s:
                    obs_s = [
                        idx for idx in after_indices
                        if int(frames_arr[idx]) >= cand_s and int(nobs_arr[idx]) == len(mids)
                    ]
                    if obs_s:
                        new_s = int(frames_arr[obs_s[0]])
            if (new_e - new_s) > max_frames:
                cap_e = new_s + max_frames
                within_cap = [
                    idx
                    for idx in range(len(rads))
                    if new_s <= int(frames_arr[idx]) < cap_e
                    and int(nobs_arr[idx]) == len(mids)
                    and (float(vals_arr[idx]) - origin_rad) >= eff_min_net_gain_px
                ]
                if not within_cap:
                    within_cap = [
                        idx
                        for idx in range(len(rads))
                        if new_s <= int(frames_arr[idx]) < cap_e
                        and (float(vals_arr[idx]) - origin_rad) >= eff_min_net_gain_px
                    ]
                if not within_cap:
                    continue
                new_e = int(frames_arr[within_cap[-1]]) + 1
                sub_vals = [
                    float(vals_arr[idx])
                    for idx in range(len(rads))
                    if new_s <= int(frames_arr[idx]) < new_e
                ]
                if not sub_vals or (max(sub_vals) - origin_rad) < eff_min_radius_gain_px:
                    if not dep_gain_ok:
                        continue

        dur_s = (new_e - new_s) / fps
        if dur_s <= eff_min_duration_s:
            continue
        if eff_max_duration_s is not None and dur_s > eff_max_duration_s + 1e-9:
            continue

        cropped = update_segment_bounds(event, new_s, new_e, fps, suffix="dispersal_crop")
        if departing_subset is not None:
            cropped["departing_mouse_ids"] = departing_subset
            cropped["mouse_ids"] = list(mids)
        if qualify_event_duration(cropped, fps, parameters=params)["ok"]:
            refined.append(cropped)
    return refined
