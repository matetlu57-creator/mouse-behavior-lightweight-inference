"""Social clustering behavior segment clipping, occlusion verification, and scene-cluster veto."""
from __future__ import annotations

import copy
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from itertools import combinations
from typing import Any

import numpy as np

from mouse_behavior.behavior.duration_policy import qualify_event_duration
from mouse_behavior.behavior.segment_clipping_common import (
    MIN_CLIP_DURATION_S_BY_BEHAVIOR,
    _detection_map,
    _extract_center_px,
    _observed_run_velocity_cm_s,
    _observed_support_intervals,
    _resolve_fps,
    _resolve_parameters,
    _resolve_scale_cm_per_px,
    qualify_clip_duration,
    update_segment_bounds,
)
from mouse_behavior.evaluation.exported_dataset_models import HeuristicParameters
from mouse_behavior.evaluation.group_heuristic_config import DEFAULT_GROUP_HEURISTIC_CONFIG

__all__ = [
    "refine_social_clustering_segments",
]


def _sc_occluded_members_verified(
    c_full: np.ndarray,
    o_full: np.ndarray,
    occ_dense_in: np.ndarray,
    offset_pre: int,
    run_low: int,
    run_high: int,
    lookback_frames: int,
    env_cm: float,
    neighbor_cm: float,
) -> bool:
    """Verify partially occluded pile members entered near the pile without distant contradiction or teleport."""
    low = offset_pre + run_low
    high = offset_pre + run_high
    if o_full[low:high].all():
        return True
    obs_cnt = o_full.sum(axis=1)
    pile_rel = np.flatnonzero(occ_dense_in[run_low:run_high] & (obs_cnt[low:high] >= 2))
    if not len(pile_rel):
        return False
    pile_idx = low + pile_rel
    pile_centers = np.array(
        [c_full[idx, o_full[idx]].mean(axis=0) for idx in pile_idx],
        dtype=float,
    )
    search_lo = max(0, low - lookback_frames)
    edge_limit = env_cm + 5.0
    from mouse_behavior.evaluation.screening_signals import runs

    for col in range(o_full.shape[1]):
        if o_full[low:high, col].all():
            continue
        if not o_full[search_lo:high, col].any():
            return False
        for gap_lo, gap_hi in runs(~o_full[:, col]):
            if gap_hi <= low or gap_lo >= high:
                continue
            last_t = gap_lo - 1
            if last_t < search_lo:
                return False
            others = o_full[last_t].copy()
            others[col] = False
            if others.any():
                ref = c_full[last_t, others].mean(axis=0)
            else:
                nearest_pos = int(np.argmin(np.abs(pile_idx - last_t)))
                ref = pile_centers[nearest_pos]
            if float(np.linalg.norm(c_full[last_t, col] - ref)) > edge_limit:
                return False
            if gap_hi < len(o_full) and o_full[gap_hi, col]:
                if float(np.linalg.norm(c_full[gap_hi, col] - c_full[last_t, col])) > neighbor_cm:
                    return False
    return True


def _has_preexisting_scene_cluster_veto(
    store: Any,
    mids: list[int],
    onset_frame: int,
    fps: float,
    cm_per_px: float,
    neighbor_distance_cm: float,
    lookback_frames: int,
    min_radius_drop_cm: float,
    min_actor_path_cm: float,
    min_actor_speed_cm_s: float,
    min_stable_frames: int,
) -> bool:
    """Reject social clustering that steals >=2 already-clustered mice plus an outsider."""
    if onset_frame < max(10, int(round(0.5 * fps))):
        return False
    pre_lo = max(0, onset_frame - int(round(1.5 * fps)))
    post_hi = onset_frame + int(round(0.5 * fps))
    scan_lo = max(0, onset_frame - lookback_frames)
    rows = list(store.rows(scan_lo, post_hi))
    if not rows:
        return False

    from mouse_behavior.evaluation.group_event_rules import _components

    frame_comps: dict[int, tuple[tuple[int, ...], ...]] = {}
    frame_obs_count: dict[int, int] = {}
    frame_pts: dict[int, dict[int, np.ndarray]] = {}
    for row in rows:
        fn = int(row.get("frame", row.get("source_frame", 0)))
        dmap = _detection_map(row)
        pts_map: dict[int, np.ndarray] = {}
        for tid, det in dmap.items():
            c = _extract_center_px(det)
            if c is not None:
                pts_map[int(tid)] = c * cm_per_px
        frame_pts[fn] = pts_map
        tids = sorted(pts_map)
        frame_obs_count[fn] = len(tids)
        if len(tids) < 3 or fn < pre_lo:
            frame_comps[fn] = ()
            continue
        arr = np.asarray([pts_map[t] for t in tids], dtype=float)
        dists = np.linalg.norm(arr[:, None, :] - arr[None, :, :], axis=2)
        adj = dists <= float(neighbor_distance_cm)
        np.fill_diagonal(adj, False)
        comps: list[tuple[int, ...]] = []
        for comp in _components(adj, np.ones(len(tids), dtype=bool)):
            if len(comp) >= 3:
                c_pts = arr[list(comp)]
                rad = float(np.linalg.norm(c_pts - c_pts.mean(axis=0), axis=1).max())
                env = float(neighbor_distance_cm) * (1.0 + 0.20 * max(0, len(comp) - 3))
                if rad <= env:
                    comps.append(tuple(tids[i] for i in comp))
        frame_comps[fn] = tuple(comps)

    pre_frames = [fn for fn in range(pre_lo, onset_frame) if frame_obs_count.get(fn, 0) >= 3]
    if len(pre_frames) < 5:
        return False
    post_frames = [fn for fn in range(onset_frame, post_hi) if frame_obs_count.get(fn, 0) >= 3]

    pre_counts: Counter[tuple[int, ...]] = Counter()
    for fn in pre_frames:
        for comp in frame_comps.get(fn, ()):
            pre_counts[comp] += 1

    def _cluster_had_fresh_formation(cluster: tuple[int, ...]) -> bool:
        c_set = set(cluster)
        env_c = float(neighbor_distance_cm) * (1.0 + 0.20 * max(0, len(cluster) - 3))
        hist_fns = sorted(fn for fn in frame_pts if scan_lo <= fn < post_hi)
        dense_hist: list[tuple[int, float]] = []
        rads_pre: list[float] = []
        path_by_id: dict[int, float] = {tid: 0.0 for tid in cluster}
        max_spd_pre = 0.0
        prev_pts: dict[int, np.ndarray] = {}
        for fn in hist_fns:
            pm = frame_pts[fn]
            if not c_set.issubset(pm):
                prev_pts.clear()
                continue
            pts = np.asarray([pm[tid] for tid in cluster], dtype=float)
            rad = float(np.linalg.norm(pts - pts.mean(axis=0), axis=1).max())
            if fn < onset_frame:
                rads_pre.append(rad)
                for tid in cluster:
                    if tid in prev_pts:
                        step = float(np.linalg.norm(pm[tid] - prev_pts[tid]))
                        path_by_id[tid] += step
                        max_spd_pre = max(max_spd_pre, step * fps)
                    prev_pts[tid] = pm[tid]
            if rad <= env_c:
                dense_hist.append((fn, rad))
        if not rads_pre or not dense_hist:
            return False
        if max(rads_pre) <= env_c:
            return False
        overlaps_onset = any(
            abs(fn - onset_frame) <= int(round(0.5 * fps)) for fn, _ in dense_hist
        )
        if not overlaps_onset:
            return False
        drop_ok = (float(np.percentile(rads_pre, 90)) - min(r for _, r in dense_hist)) >= min_radius_drop_cm
        motion_ok = max(path_by_id.values()) > min_actor_path_cm and max_spd_pre >= min_actor_speed_cm_s
        return bool(drop_ok and motion_ok)

    member_set = set(mids)
    for pre_cluster, cnt in pre_counts.items():
        if cnt < 0.50 * len(pre_frames):
            continue
        overlap = member_set & set(pre_cluster)
        if len(overlap) < 2:
            continue
        if post_frames:
            persist = sum(
                1 for fn in post_frames
                if any(len(overlap & set(c)) >= 2 for c in frame_comps.get(fn, ()))
            )
            if persist < 0.50 * len(post_frames):
                continue
        if len(overlap) < len(pre_cluster):
            return True
        if not _cluster_had_fresh_formation(pre_cluster):
            return True

    if len(mids) <= 8:
        for sub3 in combinations(mids, 3):
            sub3_set = set(sub3)
            co_cnt = sum(
                1 for fn in pre_frames if any(sub3_set.issubset(c) for c in frame_comps.get(fn, ()))
            )
            if co_cnt < 0.50 * len(pre_frames):
                continue
            if post_frames:
                persist3 = sum(
                    1 for fn in post_frames
                    if any(len(sub3_set & set(c)) >= 2 for c in frame_comps.get(fn, ()))
                )
                if persist3 < 0.50 * len(post_frames):
                    continue
            if not _cluster_had_fresh_formation(sub3):
                return True
    return False


def refine_social_clustering_segments(
    events: Sequence[Mapping[str, Any]],
    store: Any,
    binding: Mapping[str, Any] | None = None,
    *,
    scale: Mapping[str, Any] | None = None,
    parameters: HeuristicParameters | None = None,
    min_duration_s: float = MIN_CLIP_DURATION_S_BY_BEHAVIOR["social_clustering"][0],
    min_stability_s: float = DEFAULT_GROUP_HEURISTIC_CONFIG.clustering_final_stability_s,
    neighbor_distance_cm: float = DEFAULT_GROUP_HEURISTIC_CONFIG.neighbor_distance_cm,
    min_actor_path_cm: float = DEFAULT_GROUP_HEURISTIC_CONFIG.clustering_actor_min_path_cm,
    min_radius_drop_cm: float = DEFAULT_GROUP_HEURISTIC_CONFIG.clustering_distance_drop_cm,
    min_actor_speed_cm_s: float = DEFAULT_GROUP_HEURISTIC_CONFIG.clustering_actor_speed_cm_s,
    max_dropout_gap_s: float = 2.0,
    max_post_arrival_hold_s: float | None = 3.0,
) -> list[dict[str, Any]]:
    """Fine-crop and validate social clustering segments while retaining convergence + dense cluster."""
    from mouse_behavior.evaluation.group_event_rules import _components

    fps = _resolve_fps(binding)
    params = _resolve_parameters(binding, parameters)
    eff_min_duration_s = float(min_duration_s)
    eff_min_stability_s = float(min_stability_s)
    min_stable_frames = max(1, int(math.ceil(eff_min_stability_s * fps - 1e-9)))
    max_dropout_frames = max(1, int(round(max_dropout_gap_s * fps)))
    lookback_frames = int(round(DEFAULT_GROUP_HEURISTIC_CONFIG.clustering_motion_lookback_s * fps))
    snap_gap_frames = int(round(2.0 * fps))
    redisp_min_frames = max(3, int(round(0.5 * fps)))
    unknown_gap_frames = max(1, int(math.floor(DEFAULT_GROUP_HEURISTIC_CONFIG.clustering_unknown_gap_s * fps + 1e-9)))

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
            key = tuple(sorted(int(m) for m in set(event.get("mouse_ids", ()))))
            if len(key) < 3:
                continue
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
        mids = list(dict.fromkeys(int(m) for m in event.get("mouse_ids", ())))
        k = len(mids)
        if k < 3 or end <= start:
            continue
        rows = list(store.rows(start, end))
        if not rows:
            continue

        cm_per_px = _resolve_scale_cm_per_px(rows, mids, binding, scale=scale)
        env_cm = float(neighbor_distance_cm) * (1.0 + 0.20 * max(0, k - 3))
        occ_edge_limit = max(float(neighbor_distance_cm) * 1.35, env_cm)
        lb_start = max(0, start - lookback_frames)
        pre_rows = list(store.rows(lb_start, start)) if lb_start < start else []

        f_in, c_in, o_in = _extract_seq(rows, mids, cm_per_px)
        n = len(f_in)
        if n < min_stable_frames or not (o_in.sum(axis=0) >= 2).all():
            continue

        obs_cnt = o_in.sum(axis=1)
        all_obs = obs_cnt == k
        rad_all = np.full(n, np.nan, dtype=float)
        rad_vis = np.full(n, np.nan, dtype=float)
        occ_connected = np.zeros(n, dtype=bool)
        for idx in range(n):
            if obs_cnt[idx] >= 2:
                obs_cols = np.flatnonzero(o_in[idx])
                pts = c_in[idx, obs_cols]
                rad = float(np.max(np.linalg.norm(pts - pts.mean(axis=0), axis=1)))
                rad_vis[idx] = rad
                if all_obs[idx]:
                    rad_all[idx] = rad
                if rad <= env_cm:
                    pmat = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
                    adj = pmat <= occ_edge_limit + 1e-9
                    np.fill_diagonal(adj, False)
                    occ_connected[idx] = len(_components(adj, np.ones(len(obs_cols), dtype=bool))) == 1

        dense_all = all_obs & (rad_all <= env_cm)
        occ_dense = (obs_cnt >= 2) & (rad_vis <= env_cm) & occ_connected
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

        de = np.diff(np.r_[False, dense_env, False].astype(np.int8))
        dense_runs = [
            (int(a), int(b))
            for a, b in zip(np.flatnonzero(de == 1), np.flatnonzero(de == -1))
            if (b - a) >= min_stable_frames
            and int(dense_all[a:b].sum()) >= max(4, min_stable_frames // 2)
        ]

        used_occ_channel = False
        if not dense_runs:
            # Bounded top-down pile occlusion tolerance: require a contiguous occ_dense run >= min_stable_frames
            # without bridging across multi-frame all-unknown gaps to manufacture stability.
            occ_env = occ_dense.copy()
            oe_edges = np.diff(np.r_[False, ~occ_dense, False].astype(np.int8))
            for low, high in zip(np.flatnonzero(oe_edges == 1), np.flatnonzero(oe_edges == -1)):
                if (
                    low > 0
                    and high < n
                    and (high - low) <= unknown_gap_frames
                    and not (obs_cnt[low:high] > 0).any()
                ):
                    occ_env[low:high] = True
            ode = np.diff(np.r_[False, occ_env, False].astype(np.int8))
            occ_cands = []
            for a, b in zip(np.flatnonzero(ode == 1), np.flatnonzero(ode == -1)):
                a_i, b_i = int(a), int(b)
                od_edges = np.diff(np.r_[False, occ_dense[a_i:b_i], False].astype(np.int8))
                has_contig = any(
                    (sb - sa) >= min_stable_frames
                    for sa, sb in zip(np.flatnonzero(od_edges == 1), np.flatnonzero(od_edges == -1))
                )
                if has_contig and float(occ_dense[a_i:b_i].mean()) >= DEFAULT_GROUP_HEURISTIC_CONFIG.clustering_min_observed_fraction:
                    occ_cands.append((a_i, b_i))
            if occ_cands:
                dense_env = occ_env
                dense_runs = occ_cands
                used_occ_channel = True

        if not dense_runs:
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

        if used_occ_channel:
            dense_runs = [
                (a, b)
                for a, b in dense_runs
                if _sc_occluded_members_verified(
                    c_full,
                    o_full,
                    occ_dense,
                    offset_pre,
                    a,
                    b,
                    lookback_frames,
                    env_cm,
                    float(neighbor_distance_cm),
                )
            ]
            if not dense_runs:
                continue

        d_s, d_e = dense_runs[-1]
        arr_s = dense_runs[0][0]
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

        # Trim pre-convergence idle prefix when mice sit motionless before convergence starts
        if arr_s > first_obs_idx + 2:
            _, _, spd_in = _observed_run_velocity_cm_s(
                c_in[: arr_s + 1],
                o_in[: arr_s + 1],
                fps,
                motion_window_s=0.3,
            )
            active_conv = np.zeros(arr_s, dtype=bool)
            for idx in range(arr_s):
                if obs_cnt[idx] >= 2:
                    row_spd = spd_in[idx, o_in[idx]]
                    if np.isfinite(row_spd).any() and float(np.nanmax(row_spd)) >= float(min_actor_speed_cm_s):
                        active_conv[idx] = True
            if active_conv.any():
                max_idle_gap = max(2, int(round(1.0 * fps)))
                conv_start_idx = arr_s
                idle_run = 0
                for idx in range(arr_s - 1, -1, -1):
                    if active_conv[idx]:
                        conv_start_idx = idx
                        idle_run = 0
                    else:
                        idle_run += 1
                        if idle_run > max_idle_gap:
                            break
                cand_s_frame = int(f_in[conv_start_idx])
                if (cand_s_frame - new_s) > int(round(0.5 * fps)):
                    # Snap to a nearby all-observed frame if within 2 frames
                    snap_cands = [
                        idx
                        for idx in range(max(0, conv_start_idx - 2), min(arr_s, conv_start_idx + 3))
                        if all_obs[idx] and (np.isfinite(rad_vis[idx]) and rad_vis[idx] > env_cm)
                    ]
                    if snap_cands:
                        cand_s_frame = int(f_in[snap_cands[0]])
                    if np.isfinite(rad_vis[conv_start_idx]) and rad_vis[conv_start_idx] > env_cm:
                        new_s = cand_s_frame

        # Cap post-arrival hold at max_post_arrival_hold_s (default 3.0s) while preserving >eff_min_duration_s
        if max_post_arrival_hold_s is not None:
            hold_frames = max(min_stable_frames, int(math.floor(float(max_post_arrival_hold_s) * fps + 1e-9)))
            min_total_frames = int(math.floor(eff_min_duration_s * fps + 1e-9)) + 1
            cap_end_frame = max(int(f_in[arr_s]) + hold_frames, new_s + min_total_frames)
            if new_e > cap_end_frame:
                cand_all = [
                    idx
                    for idx in range(arr_s + min_stable_frames - 1, n)
                    if int(f_in[idx]) < cap_end_frame and all_obs[idx] and dense_env[idx]
                ]
                cand_vis = [
                    idx
                    for idx in range(arr_s + min_stable_frames - 1, n)
                    if int(f_in[idx]) < cap_end_frame and obs_cnt[idx] >= 2 and dense_env[idx]
                ]
                chosen_e = cand_all if cand_all else cand_vis
                if chosen_e and (
                    (int(f_in[chosen_e[-1]]) + 1 - new_s) >= min_total_frames
                    or (new_e - new_s) < min_total_frames
                ):
                    new_e = int(f_in[chosen_e[-1]]) + 1
                else:
                    post_min = [
                        idx
                        for idx in range(arr_s + min_stable_frames - 1, n)
                        if (new_s + min_total_frames - 1) <= int(f_in[idx]) < new_e
                        and obs_cnt[idx] >= 2
                        and dense_env[idx]
                    ]
                    if post_min and (new_e - new_s) >= min_total_frames:
                        new_e = int(f_in[post_min[0]]) + 1
                    else:
                        new_e = min(new_e, cap_end_frame)

        if (new_e - new_s) / fps <= eff_min_duration_s + 1e-9:
            continue

        first_d_s = arr_s + offset_pre
        motion_origin = max(0, first_d_s - lookback_frames)
        if pre_rows and offset_pre > 0:
            # Respect pre-existing dense blocks separated by dispersal
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
                    motion_origin = max(motion_origin, last_pb)

        if pre_rows:
            # If new_s advanced inside rows, don't let an earlier idle pre-prefix outside new_s count if new_s > start
            if new_s > start:
                new_s_full_idx = int(np.searchsorted(f_full, new_s))
                motion_origin = max(motion_origin, max(0, new_s_full_idx - int(round(0.5 * fps))))

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
        if not len(dense_slice_rads) and used_occ_channel:
            dense_slice_rads = rad_vis[dense_runs[0][0] : dense_runs[0][1]]
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

        # Pre-existing >=3-mouse scene cluster veto (aligned with Stage 1)
        if _has_preexisting_scene_cluster_veto(
            store,
            mids,
            new_s,
            fps,
            cm_per_px,
            float(neighbor_distance_cm),
            lookback_frames,
            float(min_radius_drop_cm),
            float(min_actor_path_cm),
            float(min_actor_speed_cm_s),
            min_stable_frames,
        ):
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
        # Never credit all-unknown (obs_cnt == 0) frames in support intervals
        in_crop = (f_in >= new_s) & (f_in < new_e)
        supp_mask = np.zeros(new_e - new_s, dtype=bool)
        for idx in np.flatnonzero(in_crop):
            if obs_cnt[idx] >= 2:
                supp_mask[int(f_in[idx]) - new_s] = True
        if "canonical_support_frame_intervals" in cropped and cropped["canonical_support_frame_intervals"]:
            # Intersect existing support with actually observed >=2-mouse frames (preserving coalesced gaps)
            existing_mask = np.zeros(new_e - new_s, dtype=bool)
            for a, b in cropped["canonical_support_frame_intervals"]:
                ca, cb = max(0, int(a) - new_s), min(new_e - new_s, int(b) - new_s)
                if ca < cb:
                    existing_mask[ca:cb] = True
            supp_mask &= existing_mask
        supp_intervals = _observed_support_intervals(supp_mask, new_s)
        cropped["canonical_support_frame_intervals"] = supp_intervals
        cropped["canonical_support_frame_count"] = sum(b - a for a, b in supp_intervals)
        dur_ok = (
            qualify_clip_duration(cropped, fps, parameters=params)["ok"]
            if eff_min_duration_s >= MIN_CLIP_DURATION_S_BY_BEHAVIOR["social_clustering"][0] - 1e-9
            else qualify_event_duration(cropped, fps, parameters=params)["ok"]
        )
        if dur_ok:
            refined.append(cropped)
    return refined
