"""Collective motion, huddling, and isolation behavior segment clipping."""
from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from itertools import combinations
from typing import Any

import numpy as np

from mouse_behavior.behavior.duration_policy import qualify_event_duration
from mouse_behavior.behavior.segment_clipping_common import (
    _clip_multi_track_series,
    _detection_map,
    _extract_center_px,
    _observed_run_velocity_cm_s,
    _observed_support_intervals,
    _resolve_fps,
    _resolve_parameters,
    _resolve_scale_cm_per_px,
    update_segment_bounds,
)
from mouse_behavior.evaluation.exported_dataset_models import HeuristicParameters
from mouse_behavior.evaluation.group_heuristic_config import DEFAULT_GROUP_HEURISTIC_CONFIG

__all__ = [
    "refine_group_locomotion_segments",
    "refine_huddle_segments",
    "refine_isolation_segments",
]


def refine_huddle_segments(
    events: Sequence[Mapping[str, Any]],
    store: Any,
    binding: Mapping[str, Any],
    scale: Mapping[str, Any],
    *,
    max_duration_s: float | None = 15.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Refine huddle candidates with sliding stable-state qualification and observed touching endpoints."""
    from mouse_behavior.visualization.behavior_packages import huddle_output_events

    return huddle_output_events(
        [dict(e) for e in events],
        store,
        dict(binding),
        dict(scale),
        max_duration_s=max_duration_s,
    )


def refine_group_locomotion_segments(
    events: Sequence[Mapping[str, Any]],
    store: Any,
    binding: Mapping[str, Any] | None = None,
    *,
    scale: Mapping[str, Any] | None = None,
    parameters: HeuristicParameters | None = None,
    min_duration_s: float = 3.0,
    min_support_frames: int | None = None,
    min_participation_ratio: float = 0.40,
    max_end_radius_px: float = 145.0,
    min_speed_px_s: float = 12.0,
    min_sync_ratio: float = 0.15,
) -> list[dict[str, Any]]:
    """Fine-crop group locomotion segments using synchronous-motion bout extraction."""
    from mouse_behavior.evaluation.screening_signals import runs

    fps = _resolve_fps(binding)
    params = _resolve_parameters(binding, parameters)
    cfg = DEFAULT_GROUP_HEURISTIC_CONFIG
    eff_min_duration_s = (
        float(params.group_locomotion_min_duration_s)
        if min_duration_s == 3.0
        else float(min_duration_s)
    )
    eff_min_support_frames = (
        int(min_support_frames)
        if min_support_frames is not None
        else max(1, int(round(0.75 * fps)))
    )
    max_pause_frames = max(1, int(math.floor(1.5 * fps + 1e-9)))
    snap_frames = int(round(2.0 * fps))

    refined: list[dict[str, Any]] = []
    for event in events:
        if event.get("canonical_behavior") != "group_locomotion":
            refined.append(copy.deepcopy(dict(event)))
            continue
        start = int(event["start_frame"])
        end = int(event["end_frame"])
        mids = list(dict.fromkeys(int(m) for m in event.get("mouse_ids", ())))
        k = len(mids)
        if k < 3 or end <= start:
            continue

        rows = list(store.rows(start, end))
        if not rows:
            continue

        scale_cm_per_px = _resolve_scale_cm_per_px(rows, mids, binding, scale=scale)
        centers_cm, observed = _clip_multi_track_series(rows, start, end, mids, scale_cm_per_px)
        span = end - start
        if span < 5 or int(observed.all(axis=1).sum()) < 2:
            continue

        _, velocity, speed = _observed_run_velocity_cm_s(centers_cm, observed, fps, motion_window_s=0.3)

        eff_max_end_radius_cm = (
            max_end_radius_px * scale_cm_per_px
            if max_end_radius_px != 145.0
            else max(145.0 * scale_cm_per_px, 14.5)
        )
        eff_min_speed_cm_s = (
            min_speed_px_s * scale_cm_per_px
            if min_speed_px_s != 12.0
            else max(12.0 * scale_cm_per_px, 1.2)
        )
        active_speed_cm_s = float(cfg.locomotion_active_speed_cm_s)
        max_pair_d = float(cfg.locomotion_max_pair_distance_cm)
        dir_thresh = min(
            float(params.group_locomotion_min_direction_similarity),
            float(cfg.locomotion_min_direction_similarity),
        )

        all_obs = observed.all(axis=1)
        moving = observed & np.isfinite(speed) & (speed >= active_speed_cm_s)
        moving_all = moving.all(axis=1)

        contradicted = np.zeros(span, dtype=bool)
        spatial_ok = all_obs.copy()
        dir_cos = np.full(span, np.nan, dtype=float)
        centroids_cm = np.full((span, 2), np.nan, dtype=float)
        rads_cm = np.full(span, np.nan, dtype=float)

        for idx in range(span):
            obs_cols = np.flatnonzero(observed[idx])
            if len(obs_cols) >= 2:
                pts = centers_cm[idx, obs_cols]
                c_mean = pts.mean(axis=0)
                centroids_cm[idx] = c_mean
                rads_cm[idx] = float(np.linalg.norm(pts - c_mean, axis=1).mean())
                pair_d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
                if float(pair_d.max()) >= max_pair_d:
                    contradicted[idx] = True
                    spatial_ok[idx] = False
            if all_obs[idx] and moving_all[idx]:
                sp_row = speed[idx]
                if (sp_row > 1e-6).all():
                    units = velocity[idx] / sp_row[:, None]
                    cos_vals = [
                        float(np.dot(units[a], units[b]))
                        for a, b in combinations(range(k), 2)
                    ]
                    if cos_vals:
                        dir_cos[idx] = float(np.mean(cos_vals))

        sync_mask = (
            all_obs
            & (moving.sum(axis=1) >= 3)
            & moving_all
            & spatial_ok
            & np.isfinite(dir_cos)
            & (dir_cos >= dir_thresh)
        )
        if not sync_mask.any():
            continue

        # Extract moving bouts bridged across non-contradicted brief pauses <= 1.5s
        bout_mask = moving.any(axis=1) & ~contradicted
        for gap_lo, gap_hi in runs(~bout_mask):
            if (
                0 < gap_lo
                and gap_hi < span
                and (gap_hi - gap_lo) <= max_pause_frames
                and bout_mask[gap_lo - 1]
                and bout_mask[gap_hi]
                and not contradicted[gap_lo:gap_hi].any()
            ):
                bout_mask[gap_lo:gap_hi] = True

        for rel_s, rel_e in runs(bout_mask):
            if not sync_mask[rel_s:rel_e].any():
                continue

            # Allow head/tail dropout snap ONLY when margin frames are partially unobserved
            # and non-contradictory (never snapping across stationary stretches).
            if (
                rel_s > 0
                and rel_s <= snap_frames
                and not all_obs[:rel_s].any()
                and not contradicted[:rel_s].any()
                and moving[:rel_s].any()
            ):
                true_s = start
            else:
                true_s = start + rel_s

            if (
                rel_e < span
                and (span - rel_e) <= snap_frames
                and not all_obs[rel_e:].any()
                and not contradicted[rel_e:].any()
                and moving[rel_e:].any()
            ):
                true_e = end
            else:
                true_e = start + rel_e

            b_s = true_s - start
            b_e = true_e - start
            dur_s = (true_e - true_s) / fps
            if dur_s <= eff_min_duration_s + 1e-9:
                continue

            bout_sync = sync_mask[b_s:b_e]
            sync_count = int(bout_sync.sum())
            if sync_count < eff_min_support_frames:
                continue
            if (sync_count / max(1, true_e - true_s)) + 1e-9 < min_sync_ratio:
                continue
            valid_dir = np.isfinite(dir_cos[b_s:b_e])
            if not valid_dir.any():
                continue
            if float(np.mean(dir_cos[b_s:b_e][valid_dir])) + 1e-9 < min(0.20, dir_thresh):
                continue

            bout_rads = rads_cm[b_s:b_e]
            bout_rads_valid = bout_rads[np.isfinite(bout_rads)]
            if not len(bout_rads_valid):
                continue
            end_rad_cm = float(np.mean(bout_rads_valid[-min(15, len(bout_rads_valid)) :]))
            if end_rad_cm > eff_max_end_radius_cm:
                continue

            bout_c = centroids_cm[b_s:b_e]
            c_valid = bout_c[np.isfinite(bout_c).all(axis=1)]
            if len(c_valid) < 2:
                continue
            c_path_cm = float(np.linalg.norm(np.diff(c_valid, axis=0), axis=1).sum())
            if (c_path_cm / max(dur_s, 1e-6)) < eff_min_speed_cm_s:
                continue

            mouse_paths: list[float] = []
            disps_cm: list[np.ndarray] = []
            for col in range(k):
                obs_idx = np.flatnonzero(observed[b_s:b_e, col])
                if len(obs_idx) < 2:
                    mouse_paths.append(0.0)
                    continue
                pts_m = centers_cm[b_s:b_e, col][obs_idx]
                mouse_paths.append(float(np.linalg.norm(np.diff(pts_m, axis=0), axis=1).sum()))
                disps_cm.append(pts_m[-1] - pts_m[0])
            if not mouse_paths or max(mouse_paths) <= 0.0:
                continue
            if (min(mouse_paths) / max(mouse_paths)) < min_participation_ratio:
                continue
            if len(disps_cm) != k:
                continue
            norms = [float(np.linalg.norm(d)) for d in disps_cm]
            if min(norms) <= 1e-6:
                continue
            min_net_req = float(cfg.locomotion_min_net_displacement_cm) * 0.7 * min(1.0, eff_min_duration_s / 3.0)
            if min(norms) + 1e-9 < min_net_req:
                continue
            units = [d / max(n, 1e-9) for d, n in zip(disps_cm, norms)]
            pair_cos = [
                float(np.dot(units[a], units[b]))
                for a, b in combinations(range(k), 2)
            ]
            if not pair_cos or float(np.mean(pair_cos)) < dir_thresh - 0.1:
                continue

            cropped = update_segment_bounds(event, true_s, true_e, fps, suffix="gloco_crop")
            supp_intervals = _observed_support_intervals(bout_sync, true_s)
            cropped["canonical_support_frame_intervals"] = supp_intervals
            cropped["canonical_support_frame_count"] = sync_count
            if qualify_event_duration(cropped, fps, parameters=params)["ok"]:
                refined.append(cropped)
    return refined


def refine_isolation_segments(
    events: Sequence[Mapping[str, Any]],
    store: Any,
    binding: Mapping[str, Any] | None = None,
    *,
    scale: Mapping[str, Any] | None = None,
    parameters: HeuristicParameters | None = None,
    min_duration_s: float = 10.0,
    isolation_distance_cm: float = 20.0,
    min_isolated_fraction: float = 0.85,
) -> list[dict[str, Any]]:
    """Fine-crop and validate isolation segments for single mice separated from conspecifics."""
    fps = _resolve_fps(binding)
    params = _resolve_parameters(binding, parameters)
    refined: list[dict[str, Any]] = []

    req_dur = getattr(params, "isolation_min_duration_s", min_duration_s)
    req_dist_cm = getattr(params, "isolation_distance_cm", isolation_distance_cm)

    for event in events:
        if event.get("canonical_behavior") != "isolation":
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

        scale_cm_per_px = _resolve_scale_cm_per_px(rows, mids, binding, scale=scale)
        iso_dist_px = req_dist_cm / scale_cm_per_px

        obs_isolated_indices = []
        obs_all_indices = []
        for idx, row in enumerate(rows):
            det_map = _detection_map(row)
            if target_id not in det_map:
                continue
            obs_all_indices.append(idx)
            c_target = _extract_center_px(det_map[target_id])
            if c_target is None:
                continue
            other_dists = []
            for oid, odet in det_map.items():
                if oid == target_id:
                    continue
                c_other = _extract_center_px(odet)
                if c_other is not None:
                    other_dists.append(float(np.linalg.norm(c_target - c_other)))
            if not other_dists or min(other_dists) >= iso_dist_px * 0.75:
                obs_isolated_indices.append(idx)

        if not obs_all_indices:
            continue

        if len(obs_isolated_indices) / max(len(obs_all_indices), 1) < min_isolated_fraction:
            continue

        true_s = start if int(obs_all_indices[0]) <= int(round(2.0 * fps)) else (start + int(obs_all_indices[0]))
        if end - (start + int(obs_all_indices[-1]) + 1) <= int(round(2.0 * fps)):
            true_e = end
        else:
            true_e = start + int(obs_all_indices[-1]) + 1

        dur_s = (true_e - true_s) / fps
        if dur_s < req_dur - 1e-9:
            continue

        cropped = update_segment_bounds(event, true_s, true_e, fps, suffix="isolation_crop")

        supp_intervals = []
        if obs_isolated_indices:
            curr_s = obs_isolated_indices[0]
            prev = curr_s
            for idx in obs_isolated_indices[1:]:
                if idx == prev + 1:
                    prev = idx
                else:
                    supp_intervals.append([start + curr_s, start + prev + 1])
                    curr_s = idx
                    prev = idx
            supp_intervals.append([start + curr_s, start + prev + 1])
        else:
            supp_intervals = [[true_s, true_e]]

        cropped["canonical_support_frame_intervals"] = supp_intervals
        cropped["canonical_support_frame_count"] = sum(e - s for s, e in supp_intervals)

        if qualify_event_duration(cropped, fps, parameters=params)["ok"]:
            refined.append(cropped)

    return refined
