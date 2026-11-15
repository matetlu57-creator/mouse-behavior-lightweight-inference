"""Group/collective-layer behavior segment clipping (Huddling, Social clustering, Dispersal, Group locomotion, Isolation)."""
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
from mouse_behavior.evaluation.group_heuristic_config import DEFAULT_GROUP_HEURISTIC_CONFIG

__all__ = [
    "refine_dispersal_segments",
    "refine_group_locomotion_segments",
    "refine_huddle_segments",
    "refine_isolation_segments",
    "refine_social_clustering_segments",
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
    """Fine-crop dispersal segments from compact/motion onset through sustained separation."""
    fps = _resolve_fps(binding)
    params = _resolve_parameters(binding, parameters)
    eff_min_duration_s = (
        float(params.dispersal_min_duration_s) if min_duration_s == 5.0 else float(min_duration_s)
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

        rads: list[tuple[int, int, float]] = []
        for row in rows:
            fn = int(row.get("frame", row.get("source_frame", 0)))
            det_map = _detection_map(row)
            centers = []
            for m in mids:
                if m in det_map:
                    c = _extract_center_px(det_map[m])
                    if c is not None:
                        obs_counts[m] += 1
                        in_counts[m] += 1
                        centers.append(c)
            if len(centers) >= 2:
                pts = np.asarray(centers, dtype=float)
                rad = float(np.linalg.norm(pts - pts.mean(axis=0), axis=1).mean())
                rads.append((fn, len(centers), rad))

        if len(rads) < 10 or sum(1 for cnt in in_counts.values() if cnt >= 2) < 2:
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
        if peak_rad - origin_rad < eff_min_radius_gain_px:
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

        if (new_e - new_s) / fps <= eff_min_duration_s:
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
    eff_min_duration_s = (
        float(params.group_locomotion_min_duration_s)
        if min_duration_s == 3.0
        else float(min_duration_s)
    )
    eff_min_support_frames = (
        int(min_support_frames)
        if min_support_frames is not None
        else max(
            1,
            int(math.ceil(DEFAULT_GROUP_HEURISTIC_CONFIG.locomotion_simultaneous_min_s * fps - 1e-9)),
        )
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
        if dur_s <= eff_min_duration_s + 1e-9:
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


def refine_social_clustering_segments(
    events: Sequence[Mapping[str, Any]],
    store: Any,
    binding: Mapping[str, Any] | None = None,
    *,
    scale: Mapping[str, Any] | None = None,
    parameters: HeuristicParameters | None = None,
    min_duration_s: float = DEFAULT_GROUP_HEURISTIC_CONFIG.clustering_final_stability_s,
    neighbor_distance_cm: float = DEFAULT_GROUP_HEURISTIC_CONFIG.neighbor_distance_cm,
    min_actor_path_cm: float = DEFAULT_GROUP_HEURISTIC_CONFIG.clustering_actor_min_path_cm,
    min_radius_drop_cm: float = DEFAULT_GROUP_HEURISTIC_CONFIG.clustering_distance_drop_cm,
    min_actor_speed_cm_s: float = DEFAULT_GROUP_HEURISTIC_CONFIG.clustering_actor_speed_cm_s,
    max_dropout_gap_s: float = 2.0,
) -> list[dict[str, Any]]:
    """Fine-crop and validate social clustering segments while retaining convergence + dense cluster."""
    fps = _resolve_fps(binding)
    params = _resolve_parameters(binding, parameters)
    eff_min_duration_s = float(min_duration_s)
    min_stable_frames = max(1, int(math.ceil(eff_min_duration_s * fps - 1e-9)))
    max_dropout_frames = max(1, int(round(max_dropout_gap_s * fps)))
    lookback_frames = int(round(DEFAULT_GROUP_HEURISTIC_CONFIG.clustering_motion_lookback_s * fps))
    snap_gap_frames = int(round(2.0 * fps))
    redisp_min_frames = max(3, int(round(0.5 * fps)))

    def _extract_seq(row_list: Sequence[Mapping[str, Any]], mids: list[int], cm_per_px: float):
        k = len(mids)
        f_nos: list[int] = []
        centers_list: list[np.ndarray] = []
        obs_list: list[np.ndarray] = []
        for row in row_list:
            fn = int(row.get("frame", row.get("source_frame", 0)))
            det_map = _detection_map(row)
            pts_row = np.full((k, 2), np.nan, dtype=float)
            obs_row = np.zeros(k, dtype=bool)
            for j, mid in enumerate(mids):
                if mid in det_map:
                    center = _extract_center_px(det_map[mid])
                    if center is not None:
                        pts_row[j] = center * cm_per_px
                        obs_row[j] = True
            f_nos.append(fn)
            centers_list.append(pts_row)
            obs_list.append(obs_row)
        return (
            np.asarray(f_nos, dtype=int),
            np.asarray(centers_list, dtype=float),
            np.asarray(obs_list, dtype=bool),
        )

    sc_groups: dict[tuple[int, ...], list[tuple[int, dict[str, Any]]]] = {}
    passthrough: list[tuple[int, dict[str, Any]]] = []
    for pos, event in enumerate(events):
        if event.get("canonical_behavior") != "social_clustering":
            passthrough.append((pos, copy.deepcopy(dict(event))))
        else:
            key = tuple(sorted(int(m) for m in event.get("mouse_ids", ())))
            sc_groups.setdefault(key, []).append((pos, copy.deepcopy(dict(event))))

    coalesced: list[tuple[int, dict[str, Any]]] = []
    for key, items in sc_groups.items():
        items.sort(key=lambda item: (int(item[1]["start_frame"]), int(item[1]["end_frame"]), item[0]))
        cur_pos, cur = items[0]
        mids_key = list(key)
        for nxt_pos, nxt in items[1:]:
            gap = int(nxt["start_frame"]) - int(cur["end_frame"])
            can_merge = False
            if gap <= 0:
                can_merge = True
            elif gap <= max_dropout_frames and len(key) >= 3:
                prev_tail_rows = list(store.rows(max(0, int(cur["end_frame"]) - min_stable_frames), int(cur["end_frame"])))
                g_rows = list(store.rows(max(0, int(cur["end_frame"])), int(nxt["start_frame"])))
                ref_rows = g_rows if g_rows else prev_tail_rows
                cm_per_px_g = _resolve_scale_cm_per_px(ref_rows, mids_key, binding, scale=scale)
                env_g = float(neighbor_distance_cm) * (1.0 + 0.20 * max(0, len(key) - 3))
                prev_dense_reached = False
                for r in prev_tail_rows:
                    dm = _detection_map(r)
                    pts = [_extract_center_px(dm[m]) for m in mids_key if m in dm]
                    pts = [p for p in pts if p is not None]
                    if len(pts) >= max(2, len(key) - 1):
                        arr = np.asarray(pts, dtype=float)
                        if float(np.max(np.linalg.norm(arr - arr.mean(axis=0), axis=1)) * cm_per_px_g) <= env_g:
                            prev_dense_reached = True
                            break
                dispersed_in_gap = False
                if prev_dense_reached:
                    for r in g_rows:
                        dm = _detection_map(r)
                        pts = [_extract_center_px(dm[m]) for m in mids_key if m in dm]
                        pts = [p for p in pts if p is not None]
                        if len(pts) == len(key):
                            arr = np.asarray(pts, dtype=float)
                            mrad = float(np.max(np.linalg.norm(arr - arr.mean(axis=0), axis=1)) * cm_per_px_g)
                            if mrad > env_g * 1.25:
                                dispersed_in_gap = True
                                break
                can_merge = not dispersed_in_gap
            if can_merge:
                cur["end_frame"] = max(int(cur["end_frame"]), int(nxt["end_frame"]))
                if (
                    "canonical_support_frame_intervals" in cur
                    or "canonical_support_frame_intervals" in nxt
                ):
                    raw_supp = [
                        [int(iv[0]), int(iv[1])]
                        for iv in cur.get("canonical_support_frame_intervals", ())
                    ] + [
                        [int(iv[0]), int(iv[1])]
                        for iv in nxt.get("canonical_support_frame_intervals", ())
                    ]
                    out_supp: list[list[int]] = []
                    for a, b in sorted(raw_supp):
                        if out_supp and a <= out_supp[-1][1]:
                            out_supp[-1][1] = max(out_supp[-1][1], b)
                        else:
                            out_supp.append([a, b])
                    cur["canonical_support_frame_intervals"] = out_supp
                    cur["canonical_support_frame_count"] = sum(b - a for a, b in out_supp)
            else:
                coalesced.append((cur_pos, cur))
                cur_pos, cur = nxt_pos, nxt
        coalesced.append((cur_pos, cur))

    all_items = sorted(passthrough + coalesced, key=lambda item: item[0])
    refined: list[dict[str, Any]] = []
    for _, event in all_items:
        if event.get("canonical_behavior") != "social_clustering":
            refined.append(event)
            continue
        start = int(event["start_frame"])
        end = int(event["end_frame"])
        mids = [int(m) for m in event.get("mouse_ids", ())]
        k = len(mids)
        if k < 3 or end <= start:
            continue
        rows = list(store.rows(start, end))
        if not rows:
            continue

        cm_per_px = _resolve_scale_cm_per_px(rows, mids, binding, scale=scale)
        env_cm = float(neighbor_distance_cm) * (1.0 + 0.20 * max(0, k - 3))
        lb_start = max(0, start - lookback_frames)
        pre_rows = list(store.rows(lb_start, start)) if lb_start < start else []

        f_in, c_in, o_in = _extract_seq(rows, mids, cm_per_px)
        n = len(f_in)
        if n < min_stable_frames or not (o_in.sum(axis=0) >= 2).all():
            continue

        obs_cnt = o_in.sum(axis=1)
        all_obs = obs_cnt == k
        vis_k = max(2, k - 1) if k >= 4 else k
        vis_obs = obs_cnt >= vis_k
        rad_all = np.full(n, np.nan, dtype=float)
        rad_vis = np.full(n, np.nan, dtype=float)
        for idx in range(n):
            if obs_cnt[idx] >= 2:
                pts = c_in[idx, o_in[idx]]
                rad = float(np.max(np.linalg.norm(pts - pts.mean(axis=0), axis=1)))
                rad_vis[idx] = rad
                if all_obs[idx]:
                    rad_all[idx] = rad

        dense_all = all_obs & (rad_all <= env_cm)
        dense_vis = vis_obs & (rad_vis <= env_cm)
        occ_max = max(2, int(round(0.5 * fps))) if k >= 4 else max(1, int(round(0.2 * fps)))

        dense_env = dense_all.copy()
        edges = np.diff(np.r_[False, ~dense_all, False].astype(np.int8))
        for low, high in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            if low > 0 and high < n and (high - low) <= occ_max:
                measured_wide = np.any(all_obs[low:high] & (rad_all[low:high] > env_cm))
                vis_ok = np.all(
                    np.where(obs_cnt[low:high] >= 2, rad_vis[low:high] <= env_cm * 1.15, True)
                )
                if not measured_wide and vis_ok:
                    dense_env[low:high] = True

        de0 = np.diff(np.r_[False, dense_env, False].astype(np.int8))
        has_all_run = any(
            (b - a) >= min_stable_frames
            for a, b in zip(np.flatnonzero(de0 == 1), np.flatnonzero(de0 == -1))
        )
        if not has_all_run and k >= 4:
            dense_env2 = dense_vis.copy()
            e2 = np.diff(np.r_[False, ~dense_vis, False].astype(np.int8))
            for low, high in zip(np.flatnonzero(e2 == 1), np.flatnonzero(e2 == -1)):
                if low > 0 and high < n and (high - low) <= occ_max:
                    if not np.any(all_obs[low:high] & (rad_all[low:high] > env_cm)):
                        dense_env2[low:high] = True
            dense_env = dense_env2

        de = np.diff(np.r_[False, dense_env, False].astype(np.int8))
        dense_runs = [
            (int(a), int(b))
            for a, b in zip(np.flatnonzero(de == 1), np.flatnonzero(de == -1))
            if (b - a) >= min_stable_frames
            and int(dense_all[a:b].sum()) >= max(4, min_stable_frames // 2)
        ]
        if not dense_runs:
            continue

        d_s, d_e = dense_runs[-1]
        new_e = end
        redisp_streak: list[int] = []
        for idx in range(d_e, n):
            if obs_cnt[idx] >= 2 and rad_vis[idx] > env_cm * 1.25:
                redisp_streak.append(idx)
                if len(redisp_streak) >= redisp_min_frames:
                    last_ok = max(d_s + min_stable_frames - 1, redisp_streak[0] - 1)
                    new_e = int(f_in[last_ok]) + 1
                    break
            else:
                redisp_streak.clear()
        else:
            last_obs_idx = int(np.flatnonzero(obs_cnt >= 2)[-1])
            if (end - (int(f_in[last_obs_idx]) + 1)) > snap_gap_frames:
                new_e = int(f_in[last_obs_idx]) + 1
            tail_obs = [
                idx
                for idx in range(max(0, n - min_stable_frames), n)
                if obs_cnt[idx] >= 2
            ]
            if tail_obs and float(np.mean([rad_vis[idx] for idx in tail_obs])) > env_cm * 1.35:
                new_e = int(f_in[d_e - 1]) + 1

        first_obs_idx = int(np.flatnonzero(obs_cnt >= 2)[0])
        new_s = (
            start
            if (int(f_in[first_obs_idx]) - start) <= snap_gap_frames
            else int(f_in[first_obs_idx])
        )
        if (new_e - new_s) / fps + 1e-9 < eff_min_duration_s:
            continue

        if pre_rows:
            f_pre, c_pre, o_pre = _extract_seq(pre_rows, mids, cm_per_px)
            c_full = np.concatenate([c_pre, c_in], axis=0)
            o_full = np.concatenate([o_pre, o_in], axis=0)
            f_full = np.concatenate([f_pre, f_in], axis=0)
            offset_pre = len(f_pre)
        else:
            c_full, o_full, f_full = c_in, o_in, f_in
            offset_pre = 0

        first_d_s = dense_runs[0][0] + offset_pre
        motion_origin = 0
        if offset_pre > 0:
            pre_dense = np.zeros(offset_pre, dtype=bool)
            for idx in range(offset_pre):
                if bool(o_full[idx].all()):
                    pts = c_full[idx]
                    if float(np.max(np.linalg.norm(pts - pts.mean(axis=0), axis=1))) <= env_cm:
                        pre_dense[idx] = True
            pde = np.diff(np.r_[False, pre_dense, False].astype(np.int8))
            pre_runs = [
                (int(a), int(b))
                for a, b in zip(np.flatnonzero(pde == 1), np.flatnonzero(pde == -1))
                if (b - a) >= min_stable_frames
            ]
            if pre_runs:
                last_pb = pre_runs[-1][1]
                has_intervening_dispersal = any(
                    (
                        int(o_full[idx].sum()) >= 2
                        and float(
                            np.max(
                                np.linalg.norm(
                                    c_full[idx, o_full[idx]]
                                    - c_full[idx, o_full[idx]].mean(axis=0),
                                    axis=1,
                                )
                            )
                        )
                        > env_cm
                    )
                    for idx in range(last_pb, first_d_s)
                )
                if has_intervening_dispersal:
                    motion_origin = last_pb

        pre_rads_cm = [
            float(
                np.max(
                    np.linalg.norm(
                        c_full[idx, o_full[idx]] - c_full[idx, o_full[idx]].mean(axis=0),
                        axis=1,
                    )
                )
            )
            for idx in range(motion_origin, first_d_s)
            if int(o_full[idx].sum()) >= 2
        ]
        dense_slice_rads = rad_all[dense_runs[0][0] : dense_runs[0][1]]
        dense_slice_rads = dense_slice_rads[np.isfinite(dense_slice_rads)]
        if not pre_rads_cm or not len(dense_slice_rads):
            continue
        if max(pre_rads_cm) <= env_cm:
            continue
        if (
            float(np.percentile(pre_rads_cm, 90)) - float(np.median(dense_slice_rads))
            < float(min_radius_drop_cm)
        ):
            continue

        conv_end_full = min(len(f_full), offset_pre + dense_runs[0][1])
        if conv_end_full - motion_origin < 2:
            continue
        adj = o_full[motion_origin + 1 : conv_end_full] & o_full[motion_origin : conv_end_full - 1]
        diffs = np.diff(c_full[motion_origin:conv_end_full], axis=0)
        step_lens = np.where(adj, np.linalg.norm(diffs, axis=2), 0.0)
        if float(step_lens.sum(axis=0).max()) <= float(min_actor_path_cm) + 1e-9:
            continue
        max_spd = float((step_lens * fps).max()) if len(step_lens) else 0.0
        if max_spd < float(min_actor_speed_cm_s) - 1e-9:
            continue

        # Reject synchronized parallel Group locomotion masquerading as formation
        dur_in_s = (new_e - new_s) / fps
        if dur_in_s > float(params.group_locomotion_min_duration_s):
            in_window_mask = (f_in >= new_s) & (f_in < new_e)
            disps = []
            for j in range(k):
                j_obs = np.flatnonzero(in_window_mask & o_in[:, j])
                if len(j_obs) >= 2:
                    disps.append(c_in[j_obs[-1], j] - c_in[j_obs[0], j])
            if len(disps) == k:
                norms = [float(np.linalg.norm(d)) for d in disps]
                if min(norms) >= DEFAULT_GROUP_HEURISTIC_CONFIG.locomotion_min_net_displacement_cm:
                    units = [d / max(nrm, 1e-9) for d, nrm in zip(disps, norms)]
                    pair_cos = [
                        float(np.dot(units[a], units[b]))
                        for a in range(k)
                        for b in range(a + 1, k)
                    ]
                    if (
                        pair_cos
                        and float(np.mean(pair_cos))
                        >= DEFAULT_GROUP_HEURISTIC_CONFIG.locomotion_min_direction_similarity
                    ):
                        continue

        cropped = update_segment_bounds(event, new_s, new_e, fps, suffix="social_clustering_crop")
        if qualify_event_duration(cropped, fps, parameters=params)["ok"]:
            refined.append(cropped)
    return refined
