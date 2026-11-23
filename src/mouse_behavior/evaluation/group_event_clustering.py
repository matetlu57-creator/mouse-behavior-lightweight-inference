"""Huddling, Social clustering, and Group locomotion process rules."""
from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Any

import numpy as np

from .clustering_formation import formation_masks
from .event_frames import mask_intervals
from .exported_dataset_models import ExportedFeatures, HeuristicParameters
from .group_event_geometry import (
    MIN_MOTION_SUPPORT_SECONDS,
    _GroupProcess,
    _components,
    _geometry,
)
from .group_heuristic_config import GroupHeuristicConfig
from .screening_signals import runs


def _huddle_window(compact: np.ndarray, enough: np.ndarray, observed: np.ndarray,
                   fps: float, minimum_s: float, *,
                   max_interior_unknown_s: float | None = None) -> tuple[int, int, int]:
    """Best qualifying huddle window as ``(start, stop, support_frames)``.

    The reported evidence is the COMPACT SUPPORT inside the window, never the
    window span. A window may bridge frames where fewer than two members were
    observed, and bridged time is an absence of contradiction, not evidence of
    a pile: reporting the span let 0.17 s of real compact evidence inside a 6 s
    episode publish a 6.0 s huddle. The qualifying gate stays on the span, so
    the ethogram minimum still describes the process envelope.

    Each interior unknown gap is additionally capped, which stops two
    temporally disjoint piles from merging into one long episode whose
    inflated evidence outranks the correct smaller set. A visible disconnected
    subset still splits the episode, as before.
    """
    best = (0, 0, 0)
    for start, end in runs(~(enough & ~compact)):
        evidence = compact[start:end]
        support_frames = int(evidence.sum())
        if support_frames / fps < MIN_MOTION_SUPPORT_SECONDS:
            continue
        if not (observed[start:end] & evidence[:, None]).any(axis=0).all():
            continue
        if (end - start) / fps + 1e-12 < minimum_s:
            continue
        support_index = np.flatnonzero(evidence)
        first, last = int(support_index[0]), int(support_index[-1])
        if max_interior_unknown_s is not None:
            longest_unknown = max(
                (stop - begin for begin, stop in runs(~evidence[first:last + 1])),
                default=0,
            )
            if longest_unknown / fps > max_interior_unknown_s + 1e-9:
                continue
        if support_frames > best[2]:
            best = (start + first, start + last + 1, support_frames)
    return best


def _huddle_event(compact: np.ndarray, enough: np.ndarray,
                  observed: np.ndarray, fps: float, minimum_s: float, *,
                  max_interior_unknown_s: float | None = None) -> float:
    """Compact-support seconds of the best qualifying huddle window."""
    _, _, support_frames = _huddle_window(
        compact, enough, observed, fps, minimum_s,
        max_interior_unknown_s=max_interior_unknown_s,
    )
    return support_frames / fps if support_frames else 0.0


def _huddle_windows(compact, enough, observed, fps, minimum_s, maximum_gap_s):
    """All maximal supported processes, separated by observed contradictions.

    A short unknown interior can preserve a process but supplies no compact
    evidence. Long unknown interiors split episodes instead of hiding a valid
    bout on either side. Leading/trailing unknowns never extend its endpoints.
    """
    envelope = ~(enough & ~compact)
    for low, high in runs(~compact):
        if (high - low) / fps > maximum_gap_s + 1e-9:
            envelope[low:high] = False
    windows = []
    for low, high in runs(envelope):
        start, stop, support = _huddle_window(compact[low:high], enough[low:high],
            observed[low:high], fps, minimum_s, max_interior_unknown_s=maximum_gap_s)
        if support:
            windows.append((low + start, low + stop, support))
    return windows


def _is_parallel_group_locomotion(
    f: ExportedFeatures,
    members: tuple[int, ...],
    velocity: np.ndarray,
    speed: np.ndarray,
    config: GroupHeuristicConfig,
    observed: np.ndarray,
    centers: np.ndarray,
    env_cm: float,
) -> bool:
    """Detect parallel group locomotion without sharp radial collapse so it is not mislabeled as social clustering."""
    obs_pos = observed & np.isfinite(centers).all(axis=2)
    net_disps = []
    net_norms = []
    for col in range(len(members)):
        idx = np.flatnonzero(obs_pos[:, col])
        if len(idx) < 2:
            return False
        delta = centers[idx[-1], col] - centers[idx[0], col]
        net_disps.append(delta)
        net_norms.append(float(np.linalg.norm(delta)))
    if min(net_norms) + 1e-9 < config.locomotion_min_net_displacement_cm:
        return False
    net_cos = [
        float(np.dot(net_disps[a], net_disps[b]) / max(net_norms[a] * net_norms[b], 1e-9))
        for a, b in combinations(range(len(members)), 2)
    ]
    if float(np.mean(net_cos)) < 0.55:
        return False

    obs_valid = obs_pos & np.isfinite(speed[:, members])
    moving = obs_valid & (speed[:, members] >= config.locomotion_active_speed_cm_s)
    active_all = moving.all(axis=1)
    if int(active_all.sum()) < max(2, int(round(0.5 * f.fps))):
        return False

    unit = velocity[:, members] / np.maximum(speed[:, members, None], 1e-9)
    pair_cos = [
        np.sum(unit[:, a] * unit[:, b], axis=1)
        for a, b in combinations(range(len(members)), 2)
    ]
    direction = np.mean(pair_cos, axis=0)
    valid_dir = active_all & np.isfinite(direction)
    if not valid_dir.any():
        return False
    mean_v_corr = float(np.mean(direction[valid_dir]))
    if mean_v_corr < 0.50:
        return False

    obs_cnt = obs_pos.sum(axis=1)
    obs2_idx = np.flatnonzero(obs_cnt >= 2)
    sharp_collapse = False
    if len(obs2_idx) >= 2:
        safe_c = np.where(obs_pos[:, :, None], centers, 0.0)
        mean_pt = safe_c.sum(axis=1) / np.maximum(obs_cnt[:, None], 1)
        d_mean = np.where(obs_pos, np.linalg.norm(safe_c - mean_pt[:, None], axis=2), 0.0)
        obs_rad = d_mean.max(axis=1)
        k_win = min(10, len(obs2_idx))
        r_start = float(np.median(obs_rad[obs2_idx[:k_win]]))
        r_end = float(np.median(obs_rad[obs2_idx[-k_win:]]))
        sharp_collapse = (r_start > env_cm * 1.15) and ((r_start - r_end) >= max(4.0, 0.40 * r_start))
    return not sharp_collapse


def _clustering_process(f: ExportedFeatures, members: tuple[int, ...], near: np.ndarray,
                      velocity: np.ndarray, speed: np.ndarray,
                      config: GroupHeuristicConfig, *, neighbor_cm: float,
                      max_gap_s: float,
                      geometry: tuple[np.ndarray, np.ndarray, np.ndarray,
                                      np.ndarray, np.ndarray] | None = None) -> _GroupProcess:
    """Observed formation prefix and dense state, with zero credit for unknowns.

    Actor kinematics are recomputed only from adjacent actual center rows.
    The historical joining channel remains intact. A convergence channel also
    measures motion before the recipients become compact. Both require actual
    final density; huddle holds and cross-gap jumps supply no support.
    """
    if len(members) < 3:
        return _GroupProcess()
    compact, _, observed = _geometry(f, members, near)
    centers = f.centers_cm[:, members]
    # The neighbor distance is a radius, not the group's diameter. A shared
    # center envelope bounds long connected chains without halving that scale.
    env_cm = neighbor_cm * (1.0 + 0.20 * max(0, len(members) - 3))
    common_center = np.mean(centers, axis=1)
    radius = np.linalg.norm(centers - common_center[:, None], axis=2).max(axis=1)
    dense = compact & observed.all(axis=1) & (radius <= env_cm)
    minimum = max(1, int(np.ceil(config.clustering_final_stability_s * f.fps - 1e-9)))
    mask = np.zeros(len(f.valid), bool)
    motion_mask = np.zeros(len(f.valid), bool)
    if any(b - a >= minimum for a, b in runs(dense)):
        for actor_column, actor in enumerate(members):
            recipients = tuple(m for m in members if m != actor)
            seed_compact, _, seed_observed = _geometry(f, recipients, near)
            recipient_centers = f.centers_cm[:, recipients]
            seed_center = np.mean(recipient_centers, axis=1)
            seed_env_cm = neighbor_cm * (1.0 + 0.20 * max(0, len(recipients) - 3))
            seed_dense = (np.linalg.norm(recipient_centers - seed_center[:, None], axis=2)
                          <= seed_env_cm).all(axis=1)
            known = (observed.all(axis=1) & seed_observed.all(axis=1)
                     & seed_compact & seed_dense)
            actor_centers = centers[:, actor_column]
            offset = seed_center - actor_centers
            distance = np.linalg.norm(offset, axis=1)
            # V26 uses the common group-center envelope. A second radius around
            # recipients' centroid silently rejects valid nose-connected states.
            final = known & compact & dense
            if not any(b - a >= minimum for a, b in runs(final)):
                continue
            # Use actual displacements, not a possibly lagged/interpolated velocity
            # supplied by a caller. The next row never supplies a missing operand.
            steps = np.diff(actor_centers, axis=0)
            lengths = np.linalg.norm(steps, axis=1)
            step_speed = lengths * f.fps
            relative_steps = steps - np.diff(seed_center, axis=0)
            relative_lengths = np.linalg.norm(relative_steps, axis=1)
            denom = relative_lengths * np.linalg.norm(offset[:-1], axis=1)
            cosine = np.divide(np.sum(relative_steps * offset[:-1], axis=1), denom,
                               out=np.full(len(steps), np.nan), where=denom > 1e-9)
            toward = np.r_[False, known[1:] & known[:-1]
                           & (step_speed >= config.clustering_actor_speed_cm_s)
                           & (cosine >= config.clustering_toward_cosine)
                           & (np.diff(distance) < 0.)]
            for seed_low, seed_high in runs(known):
                for low, high in runs(final[seed_low:seed_high]):
                    low, high = low + seed_low, high + seed_low
                    if high - low < minimum:
                        continue
                    active = np.flatnonzero(toward[seed_low + 1:high - minimum + 1]) + seed_low + 1
                    valid_onsets = [
                        int(idx) - 1 for idx in active
                        if not dense[int(idx) - 1] and distance[int(idx) - 1] > env_cm
                    ]
                    if not valid_onsets:
                        continue
                    onset = valid_onsets[0]
                    for arrival in range(max(low, onset + 1), high - minimum + 1):
                        delta = actor_centers[arrival] - actor_centers[onset]
                        net = float(np.linalg.norm(delta))
                        drop = float(distance[onset] - distance[arrival])
                        relative_delta = delta - (seed_center[arrival] - seed_center[onset])
                        toward_denominator = np.linalg.norm(relative_delta) * np.linalg.norm(offset[onset])
                        net_cosine = (float(np.dot(relative_delta, offset[onset])) / toward_denominator
                                      if toward_denominator > 1e-9 else -1.)
                        elapsed = (arrival - onset) / f.fps
                        path = float(lengths[onset:arrival].sum())
                        support_s = float(toward[onset + 1:arrival + 1].sum()) / f.fps
                        if (net + 1e-9 < config.clustering_actor_min_displacement_cm
                                or path <= config.clustering_actor_min_path_cm + 1e-9
                                or drop + 1e-9 < config.clustering_distance_drop_cm
                                or net_cosine < config.clustering_toward_cosine
                                or path / elapsed + 1e-9 < config.clustering_actor_speed_cm_s
                                or support_s + 1e-9 < MIN_MOTION_SUPPORT_SECONDS):
                            continue
                        mask[low:high] = True
                        motion_mask[onset:arrival + 1] |= toward[onset:arrival + 1]
                        break
    mask, motion_mask = formation_masks(
        centers, observed, compact, f.fps, config, neighbor_cm, mask, motion_mask)
    if mask.any() and _is_parallel_group_locomotion(
        f, members, velocity, speed, config, observed, centers, env_cm
    ):
        return _GroupProcess()
    return _GroupProcess(float(mask.sum()) / f.fps, mask_intervals(mask),
                         mask_intervals(motion_mask))


def _clustering_event(*args, **kwargs) -> float:
    """Compatibility score for consumers that still request only seconds."""
    return _clustering_process(*args, **kwargs).evidence


def _synchronous_process(f: ExportedFeatures, members: tuple[int, ...], velocity: np.ndarray,
                       speed: np.ndarray, p: HeuristicParameters,
                       config: GroupHeuristicConfig) -> _GroupProcess:
    observed = f.valid[:, members] & np.isfinite(speed[:, members])
    moving = observed & (speed[:, members] >= p.collective_active_speed_cm_s)
    unit = velocity[:, members] / np.maximum(speed[:, members, None], 1e-9)
    pair_cosines = [np.sum(unit[:, left] * unit[:, right], axis=1)
                   for left, right in combinations(range(len(members)), 2)]
    direction = np.mean(pair_cosines, axis=0)
    internal = np.flatnonzero(np.isin(f.pair_i, members) & np.isin(f.pair_j, members))
    distances = f.pair_distance_cm[:, internal]
    max_pair_d = min(p.group_locomotion_max_distance_cm, config.locomotion_max_pair_distance_cm)
    spatial = (np.isfinite(distances).all(axis=1) & (distances < max_pair_d).all(axis=1))
    synchronous = moving.all(axis=1) & spatial & (direction >= p.group_locomotion_min_direction_similarity)
    if np.count_nonzero(synchronous) / f.fps + 1e-12 < config.locomotion_simultaneous_min_s:
        return _GroupProcess()
    if not (moving.sum(axis=0) / f.fps >= MIN_MOTION_SUPPORT_SECONDS).all():
        return _GroupProcess()

    pair_observed = f.valid[:, f.pair_i[internal]] & f.valid[:, f.pair_j[internal]]
    contradicted = (pair_observed & (~np.isfinite(distances) | (distances >= max_pair_d))).any(axis=1)

    # Extract contiguous moving bouts rather than bridging arbitrary long stationary pauses across a clip.
    active_bout = moving.any(axis=1) & ~contradicted
    max_pause_frames = int(np.floor(3.0 * f.fps + 1e-9))
    for gap_lo, gap_hi in runs(~active_bout):
        if (gap_lo > 0 and gap_hi < len(active_bout)
                and active_bout[gap_lo - 1] and active_bout[gap_hi]
                and not contradicted[gap_lo:gap_hi].any()
                and (gap_hi - gap_lo) <= max_pause_frames):
            active_bout[gap_lo:gap_hi] = True

    qualifying: list[tuple[int, int]] = []
    support_mask = np.zeros(len(f.valid), dtype=bool)
    for start, stop in runs(active_bout):
        duration = (stop - start) / f.fps
        if duration <= p.group_locomotion_min_duration_s:
            continue
        bout_sync = synchronous[start:stop]
        if np.count_nonzero(bout_sync) / f.fps + 1e-12 < config.locomotion_simultaneous_min_s:
            continue
        if np.count_nonzero(bout_sync) / max(stop - start, 1) < 0.15:
            continue
        if not (moving[start:stop].sum(axis=0) / f.fps >= MIN_MOTION_SUPPORT_SECONDS).all():
            continue
        move_all_bout = moving[start:stop].all(axis=1)
        valid_dir = move_all_bout & np.isfinite(direction[start:stop])
        if not valid_dir.any():
            continue
        if float(np.mean(direction[start:stop][valid_dir])) < min(0.20, p.group_locomotion_min_direction_similarity):
            continue
        displacements = []
        valid_disp = True
        for slot in members:
            indices = np.flatnonzero(f.valid[start:stop, slot]) + start
            if len(indices) < 2:
                valid_disp = False
                break
            displacement = f.centers_cm[indices[-1], slot] - f.centers_cm[indices[0], slot]
            norm = float(np.linalg.norm(displacement))
            if norm <= 1e-6:
                # Some callers provide measured velocities without position
                # changes; integrated observed motion is still real evidence.
                displacement = np.nansum(velocity[start:stop, slot], axis=0) / f.fps
                norm = float(np.linalg.norm(displacement))
            if norm <= 1e-6 or norm + 1e-9 < config.locomotion_min_net_displacement_cm:
                valid_disp = False
                break
            displacements.append(displacement / norm)
        if not valid_disp:
            continue
        coherence = np.mean([np.dot(displacements[left], displacements[right])
                             for left, right in combinations(range(len(members)), 2)])
        if coherence >= p.group_locomotion_min_direction_similarity:
            qualifying.append((int(start), int(stop)))
            support_mask[start:stop] |= bout_sync

    if not qualifying:
        return _GroupProcess()
    total_evidence = float(sum(stop - start for start, stop in qualifying)) / float(f.fps)
    support_frames = tuple((int(a), int(b)) for a, b in mask_intervals(support_mask))
    return _GroupProcess(total_evidence, tuple(qualifying), support_frames)


def _synchronous_event(*args, **kwargs) -> float:
    """Compatibility score; event frames are exported by the process path."""
    return _synchronous_process(*args, **kwargs).evidence


def _members_in_preexisting_scene_cluster(
    f: ExportedFeatures,
    members: tuple[int, ...],
    process: _GroupProcess,
    near: np.ndarray,
    velocity: np.ndarray,
    speed: np.ndarray,
    config: GroupHeuristicConfig,
    neighbor_cm: float,
    max_gap_s: float,
    scene_cache: dict[str, Any],
) -> bool:
    """Reject Social clustering that grabs >=2 already-clustered mice plus an outsider."""
    if not process.frames:
        return False
    onset = process.frames[0][0]
    if process.support_frames:
        onset = min(onset, process.support_frames[0][0])
    if onset < max(10, int(round(0.5 * f.fps))):
        return False

    scene_comps: list[tuple[tuple[int, ...], ...]] | None = scene_cache.get("scene_comps")
    if scene_comps is None:
        tracks = len(f.track_ids)
        built: list[tuple[tuple[int, ...], ...]] = []
        for frame in range(len(f.valid)):
            edges = np.flatnonzero(near[frame])
            if not edges.size or int(f.valid[frame].sum()) < 3:
                built.append(())
                continue
            adj = np.zeros((tracks, tracks), dtype=bool)
            adj[f.pair_i[edges], f.pair_j[edges]] = True
            adj |= adj.T
            comps_at_frame: list[tuple[int, ...]] = []
            for comp in _components(adj, f.valid[frame]):
                if len(comp) >= 3:
                    pts = f.centers_cm[frame, comp]
                    if np.isfinite(pts).all():
                        rad = float(np.linalg.norm(pts - pts.mean(axis=0), axis=1).max())
                        env = neighbor_cm * (1.0 + 0.20 * max(0, len(comp) - 3))
                        if rad <= env:
                            comps_at_frame.append(comp)
            built.append(tuple(comps_at_frame))
        scene_comps = built
        scene_cache["scene_comps"] = scene_comps

    pre_lo = max(0, onset - int(round(1.5 * f.fps)))
    post_hi = min(len(f.valid), onset + int(round(0.5 * f.fps)))
    pre_valid = sum(1 for fr in range(pre_lo, onset) if int(f.valid[fr].sum()) >= 3)
    if pre_valid < 5:
        return False

    pre_counts: Counter[tuple[int, ...]] = Counter()
    for fr in range(pre_lo, onset):
        for comp in scene_comps[fr]:
            pre_counts[comp] += 1

    post_valid = sum(1 for fr in range(onset, post_hi) if int(f.valid[fr].sum()) >= 3)
    member_set = set(members)
    fresh_cache: dict[tuple[int, ...], bool] = scene_cache.setdefault("fresh_cache", {})

    def _had_fresh_formation(cluster: tuple[int, ...]) -> bool:
        if cluster not in fresh_cache:
            pre_proc = _clustering_process(
                f, cluster, near, velocity, speed, config,
                neighbor_cm=neighbor_cm, max_gap_s=max_gap_s,
            )
            pre_onset = pre_proc.frames[0][0] if pre_proc.frames else onset
            if pre_proc.support_frames:
                pre_onset = min(pre_onset, pre_proc.support_frames[0][0])
            overlaps_onset = any(
                e > max(0, onset - int(round(0.5 * f.fps))) and s < post_hi
                for s, e in (pre_proc.frames or ())
            )
            fresh_cache[cluster] = bool(pre_proc.evidence > 0 and pre_onset < onset and overlaps_onset)
        return fresh_cache[cluster]

    for pre_cluster, cnt in pre_counts.items():
        if cnt < 0.50 * pre_valid:
            continue
        overlap = member_set & set(pre_cluster)
        if len(overlap) < 2:
            continue
        # Confirm the pre-existing cluster persists into [onset, onset + 0.5s).
        if post_valid > 0:
            persist_cnt = sum(
                1 for fr in range(onset, post_hi)
                if any(len(overlap & set(c)) >= 2 for c in scene_comps[fr])
            )
            if persist_cnt < 0.50 * post_valid:
                continue
        # Partial subset stealing (>=2 pre-clustered mice while dropping another) is always invalid.
        if len(overlap) < len(pre_cluster):
            return True
        if not _had_fresh_formation(pre_cluster):
            return True

    # Also veto if >=3 of the selected members already sat together in compact scene clusters
    # >=50% of [pre_lo, onset) (even when peripheral bystanders flickered in/out of the component).
    if len(members) <= 8:
        for sub3 in combinations(members, 3):
            sub3_set = set(sub3)
            co_cnt = sum(1 for fr in range(pre_lo, onset) if any(sub3_set.issubset(c) for c in scene_comps[fr]))
            if co_cnt < 0.50 * pre_valid:
                continue
            if post_valid > 0:
                persist3 = sum(
                    1 for fr in range(onset, post_hi)
                    if any(len(sub3_set & set(c)) >= 2 for c in scene_comps[fr])
                )
                if persist3 < 0.50 * post_valid:
                    continue
            if not _had_fresh_formation(sub3):
                return True
    return False
