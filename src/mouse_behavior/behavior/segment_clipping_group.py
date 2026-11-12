"""Group/collective-layer behavior segment clipping (Huddling, Dispersal, Group locomotion, Isolation)."""
from __future__ import annotations

import copy
import math
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
    "refine_group_locomotion_segments",
    "refine_huddle_segments",
    "refine_isolation_segments",
]


def refine_huddle_segments(
    events: Sequence[Mapping[str, Any]],
    store: Any,
    binding: Mapping[str, Any],
    scale: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Refine huddle candidates with sliding stable-state qualification and observed touching endpoints."""
    from mouse_behavior.visualization.behavior_packages import huddle_output_events

    return huddle_output_events([dict(e) for e in events], store, dict(binding), dict(scale))


def refine_dispersal_segments(
    events: Sequence[Mapping[str, Any]],
    store: Any,
    binding: Mapping[str, Any] | None = None,
    *,
    scale: Mapping[str, Any] | None = None,
    parameters: HeuristicParameters | None = None,
    min_duration_s: float = 5.0,
    cluster_radius_px: float | None = None,
    min_radius_gain_px: float | None = None,
    min_net_gain_px: float | None = None,
) -> list[dict[str, Any]]:
    """Fine-crop dispersal segments to start at a compact cluster and end after outward expansion."""
    fps = _resolve_fps(binding)
    params = _resolve_parameters(binding, parameters)
    has_explicit_scale = (
        isinstance(scale, Mapping)
        and ("cm_per_pixel" in scale or "px_per_cm" in scale or "scale_px_per_cm" in scale)
    )
    refined: list[dict[str, Any]] = []
    for event in events:
        if event.get("canonical_behavior") != "dispersal":
            refined.append(copy.deepcopy(dict(event)))
            continue
        start = int(event["start_frame"])
        end = int(event["end_frame"])
        mids = [int(m) for m in event.get("mouse_ids", ())]
        if len(mids) < 2 or end <= start:
            continue
        rows = list(store.rows(start, end))
        if not rows:
            continue

        cm_per_px = _resolve_scale_cm_per_px(rows, mids, binding, scale=scale)
        scale_factor = (0.10 / max(cm_per_px, 1e-6)) if has_explicit_scale else 1.0
        eff_cluster_radius_px = float(cluster_radius_px) if cluster_radius_px is not None else 120.0 * max(1.0, scale_factor)
        eff_min_radius_gain_px = float(min_radius_gain_px) if min_radius_gain_px is not None else 35.0 * scale_factor
        eff_min_net_gain_px = float(min_net_gain_px) if min_net_gain_px is not None else 25.0 * scale_factor

        rads: list[tuple[int, float]] = []
        all_obs_frames: set[int] = set()
        for row in rows:
            fn = int(row.get("frame", row.get("source_frame", 0)))
            det_map = _detection_map(row)
            centers = []
            for m in mids:
                if m in det_map:
                    c = _extract_center_px(det_map[m])
                    if c is not None:
                        centers.append(c)
            if len(centers) == len(mids):
                all_obs_frames.add(fn)
            if len(centers) >= 2:
                pts = np.asarray(centers, dtype=float)
                rad = float(np.linalg.norm(pts - pts.mean(axis=0), axis=1).mean())
                rads.append((fn, rad))

        if len(rads) < 10:
            continue
        frames_arr = np.array([r[0] for r in rads], dtype=int)
        vals_arr = np.array([r[1] for r in rads], dtype=float)

        compact_idx = [
            i for i, r in enumerate(vals_arr)
            if r <= eff_cluster_radius_px and int(frames_arr[i]) in all_obs_frames
        ]
        if not compact_idx:
            continue
        best_i = min(compact_idx, key=lambda idx: vals_arr[idx])
        new_s = int(frames_arr[best_i])
        start_rad = float(vals_arr[best_i])

        after_vals = vals_arr[best_i:]
        peak_offset = int(np.argmax(after_vals))
        peak_i = best_i + peak_offset
        peak_rad = float(vals_arr[peak_i])
        if peak_rad - start_rad < eff_min_radius_gain_px:
            continue

        min_end_frame = new_s + int(math.floor(min_duration_s * fps)) + 1
        target_end = max(int(frames_arr[peak_i]) + int(round(1.0 * fps)), min_end_frame)
        target_end = min(end, target_end)

        valid_end_indices = [
            i for i in range(best_i + 1, len(frames_arr))
            if int(frames_arr[i]) + 1 >= min_end_frame
            and int(frames_arr[i]) <= max(target_end, min_end_frame + int(round(2.0 * fps)))
            and int(frames_arr[i]) in all_obs_frames
            and (vals_arr[i] - start_rad) >= eff_min_net_gain_px
        ]
        if not valid_end_indices:
            valid_end_indices = [
                i for i in range(best_i + 1, len(frames_arr))
                if int(frames_arr[i]) + 1 >= min_end_frame
                and int(frames_arr[i]) in all_obs_frames
                and (vals_arr[i] - start_rad) >= eff_min_net_gain_px
            ]
        if not valid_end_indices:
            continue

        end_i = min(valid_end_indices, key=lambda idx: abs(int(frames_arr[idx]) + 1 - target_end))
        new_e = int(frames_arr[end_i]) + 1
        if (new_e - new_s) / fps <= min_duration_s:
            continue

        cropped = update_segment_bounds(event, new_s, new_e, fps, suffix="dispersal_crop")
        if qualify_event_duration(cropped, fps, parameters=params)["ok"]:
            refined.append(cropped)
    return refined


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
) -> list[dict[str, Any]]:
    """Fine-crop and validate group locomotion segments for coordinated multi-animal movement."""
    fps = _resolve_fps(binding)
    params = _resolve_parameters(binding, parameters)
    eff_min_support_frames = (
        int(min_support_frames)
        if min_support_frames is not None
        else max(1, int(math.ceil(0.93295 * fps - 1e-9)))
    )
    refined: list[dict[str, Any]] = []
    for event in events:
        if event.get("canonical_behavior") != "group_locomotion":
            refined.append(copy.deepcopy(dict(event)))
            continue
        start = int(event["start_frame"])
        end = int(event["end_frame"])
        mids = [int(m) for m in event.get("mouse_ids", ())]
        if len(mids) < 3 or end <= start:
            continue

        rows = list(store.rows(start, end))
        if not rows:
            continue

        all_obs_indices = []
        rads: list[float] = []
        centroids: list[np.ndarray] = []
        per_mouse_pos: dict[int, list[np.ndarray]] = {m: [] for m in mids}

        for idx, row in enumerate(rows):
            det_map = _detection_map(row)
            centers = {}
            for m in mids:
                if m in det_map:
                    c = _extract_center_px(det_map[m])
                    if c is not None:
                        centers[m] = c
                        per_mouse_pos[m].append(c)
            if len(centers) == len(mids):
                all_obs_indices.append(idx)
            if len(centers) >= 2:
                pts = np.asarray(list(centers.values()), dtype=float)
                c_mean = pts.mean(axis=0)
                centroids.append(c_mean)
                rads.append(float(np.linalg.norm(pts - c_mean, axis=1).mean()))

        if not all_obs_indices or len(centroids) < 5:
            continue

        scale_cm_per_px = _resolve_scale_cm_per_px(rows, mids, binding, scale=scale)
        eff_max_end_radius_px = (
            max_end_radius_px
            if max_end_radius_px != 145.0
            else max(max_end_radius_px, 14.5 / max(scale_cm_per_px, 1e-6))
        )
        eff_min_speed_px_s = (
            min_speed_px_s
            if min_speed_px_s != 12.0
            else max(min_speed_px_s, 1.2 / max(scale_cm_per_px, 1e-6))
        )

        true_s = start if int(all_obs_indices[0]) <= int(round(2.0 * fps)) else (start + int(all_obs_indices[0]))
        if end - (start + int(all_obs_indices[-1]) + 1) <= int(round(2.0 * fps)):
            true_e = end
        else:
            true_e = start + int(all_obs_indices[-1]) + 1

        dur_s = (true_e - true_s) / fps
        if dur_s < min_duration_s - 1e-9:
            continue

        cropped = update_segment_bounds(event, true_s, true_e, fps, suffix="gloco_crop")

        # Ensure canonical support intervals are populated
        if "canonical_support_frame_intervals" not in cropped or not cropped["canonical_support_frame_intervals"]:
            supp_intervals = []
            if all_obs_indices:
                curr_s = all_obs_indices[0]
                prev = curr_s
                for idx in all_obs_indices[1:]:
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

        supp_frames = int(cropped.get("canonical_support_frame_count", true_e - true_s))
        if supp_frames < eff_min_support_frames:
            continue

        end_rad = float(np.mean(rads[-min(15, len(rads)) :]))
        if end_rad > eff_max_end_radius_px:
            continue

        c_arr = np.asarray(centroids, dtype=float)
        c_path = float(np.linalg.norm(np.diff(c_arr, axis=0), axis=1).sum())
        if (c_path / max(dur_s, 1e-6)) < eff_min_speed_px_s:
            continue

        mouse_paths = []
        for m in mids:
            mp = np.asarray(per_mouse_pos[m], dtype=float)
            if len(mp) >= 2:
                mouse_paths.append(float(np.linalg.norm(np.diff(mp, axis=0), axis=1).sum()))
            else:
                mouse_paths.append(0.0)
        if not mouse_paths or max(mouse_paths) <= 0.0:
            continue
        if (min(mouse_paths) / max(mouse_paths)) < min_participation_ratio:
            continue

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
