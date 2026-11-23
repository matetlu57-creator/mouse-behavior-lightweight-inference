"""Observed group processes; missing visual IDs never imply a new identity.

Event durations measure a process envelope, not an uninterrupted velocity mask.
Huddling holds missing observations for a bounded five-second interval. Observed spatial
separation remains contradictory evidence and is never filled as a dropout.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Sequence

import numpy as np

from mouse_behavior.behavior.reviewed_comparisons import nearest_comparison_distances
from . import screening_signals
from .blind_top1 import Candidate
from .clustering_formation import formation_masks as formation_masks
from .event_frames import FrameIntervals as FrameIntervals, mask_intervals
from .exported_dataset_models import ExportedFeatures, HeuristicParameters
from .group_event_clustering import (
    _clustering_event as _clustering_event,
    _clustering_process as _clustering_process,
    _huddle_event as _huddle_event,
    _huddle_window as _huddle_window,
    _huddle_windows as _huddle_windows,
    _is_parallel_group_locomotion as _is_parallel_group_locomotion,
    _members_in_preexisting_scene_cluster as _members_in_preexisting_scene_cluster,
    _synchronous_event as _synchronous_event,
    _synchronous_process as _synchronous_process,
)
from .group_event_dispersal import (
    _compact_origin_episodes as _compact_origin_episodes,
    _departing_member_slots as _departing_member_slots,
    _departure_origins as _departure_origins,
    _departure_process_mask as _departure_process_mask,
    _dispersal_endpoint_separated as _dispersal_endpoint_separated,
    _dispersal_process as _dispersal_process,
    _initial_compact_components as _initial_compact_components,
    _observed_attempt_end as _observed_attempt_end,
    _observed_departure_velocity as _observed_departure_velocity,
    _prepare_departure_process as _prepare_departure_process,
    _select_departure_for_checkpoint as _select_departure_for_checkpoint,
    dispersal_phase as dispersal_phase,
)
from .group_event_geometry import (
    HUDDLE_MEMBER_SUPPORT_FRACTION as HUDDLE_MEMBER_SUPPORT_FRACTION,
    MAX_PROPOSALS as MAX_PROPOSALS,
    MINIMUM_REGION_KEYPOINTS as MINIMUM_REGION_KEYPOINTS,
    MIN_MOTION_SUPPORT_SECONDS as MIN_MOTION_SUPPORT_SECONDS,
    NEIGHBOR_BODY_LENGTHS as NEIGHBOR_BODY_LENGTHS,
    _GroupProcess as _GroupProcess,
    _components as _components,
    _geometry as _geometry,
    _huddle_geometry as _huddle_geometry,
    _inside_pile_envelope as _inside_pile_envelope,
    _proposal_sets as _proposal_sets,
    _quiet_at as _quiet_at,
    _region_gap as _region_gap,
    _window_positions as _window_positions,
    group_separation as group_separation,
)
from .group_heuristic_config import (
    DEFAULT_GROUP_HEURISTIC_CONFIG as DEFAULT_GROUP_HEURISTIC_CONFIG,
    GroupHeuristicConfig as GroupHeuristicConfig,
)
from .screening_signals import ScreenPolicy, runs as runs


# [ERA 2026-10-16 group-v16.2] Bumped from
# "confirmed_group_rules_v14_total_support_whole_component". v16 added
# _huddle_geometry, rewrote _initial_compact_components, extended the
# dispersal_phase signature and relaxed _clustering_event; v16.2 bounds the
# huddle ID union to its stable window, reports compact support instead of the
# episode span, and adds the single-degraded-recipient region-gap fallback.
# policy_receipt() publishes this string into validation receipts, so a group
# run is now version-separable from v14.
GROUP_RULE_REVISION = "2026-10-29-v34.1-member-verified-complete-huddle"


def _group_behavior_candidates(f: ExportedFeatures, p: HeuristicParameters,
                          proposals: Sequence[Candidate],
                          config: GroupHeuristicConfig, behavior: str,
                          geometry_cache: dict) -> tuple[Candidate, ...]:
    if f.centers_cm is None or len(f.track_ids) < 3:
        return ()
    neighbor = (config.huddle_neighbor_distance_cm if behavior == "huddle"
                else config.neighbor_distance_cm)
    if behavior == "group_locomotion":
        p = replace(p, collective_active_speed_cm_s=config.locomotion_active_speed_cm_s,
                    group_locomotion_min_direction_similarity=config.locomotion_min_direction_similarity,
                    group_locomotion_max_distance_cm=config.locomotion_max_pair_distance_cm)
    near = np.isfinite(f.pair_distance_cm) & (f.pair_distance_cm <= neighbor)
    if f.pair_nose_distance_cm is not None:
        near |= np.isfinite(f.pair_nose_distance_cm) & (f.pair_nose_distance_cm <= p.huddle_nose_distance_cm)
    near &= f.valid[:, f.pair_i] & f.valid[:, f.pair_j]
    _, velocity, speed = screening_signals.body_motion(f, ScreenPolicy())
    if behavior == "dispersal":
        velocity = _observed_departure_velocity(f)
    result = []
    member_proposals = list(_proposal_sets(f, near, velocity, speed, p, proposals)
                            if behavior != "dispersal" else ())
    scheduled_members = set(member_proposals)
    scene_cache: dict[str, Any] = {}
    for members in member_proposals:
        # One geometry per proposal. _clustering_event used to rebuild the
        # identical result, and the plain _geometry call here was dead: its
        # compact/enough were discarded and _huddle_geometry returns the same
        # `observed`.
        geometry = None
        if behavior == "social_clustering":
            process = _clustering_process(
                f, members, near, velocity, speed, config,
                neighbor_cm=neighbor, max_gap_s=p.huddle_max_gap_s)
            if process.evidence > 0 and not _members_in_preexisting_scene_cluster(
                f, members, process, near, velocity, speed, config,
                neighbor_cm=neighbor, max_gap_s=p.huddle_max_gap_s,
                scene_cache=scene_cache,
            ):
                result.append(Candidate(behavior, tuple(sorted(int(f.track_ids[m]) for m in members)),
                                        process.evidence, process.frames, process.support_frames))
            continue
        if behavior == "huddle":
            key = members, neighbor, p.huddle_max_gap_s, config.huddle_hold_max_speed_cm_s
            if key not in geometry_cache:
                geometry_cache[key] = _huddle_geometry(
                    f, members, near, neighbor_cm=neighbor, max_gap_s=p.huddle_max_gap_s,
                    speed=speed, quiet_speed_cm_s=config.huddle_hold_max_speed_cm_s,
                )
            geometry = geometry_cache[key]
        if geometry is None:
            observed = f.valid[:, members] & np.isfinite(f.centers_cm[:, members]).all(axis=2)
            process = _synchronous_process(f, members, velocity, speed, p, config)
            if process.evidence > 0:
                result.append(Candidate(behavior, tuple(sorted(int(f.track_ids[m]) for m in members)),
                                        process.evidence, process.frames, process.support_frames))
            continue
        huddle_compact, huddle_enough, observed, _, held = geometry
        windows = _huddle_windows(
            huddle_compact, huddle_enough, observed, f.fps, p.huddle_min_duration_s,
            p.huddle_max_gap_s,
        )
        for window in windows:
            huddle_positions = _window_positions(held, window)
            if len(huddle_positions) < 3:
                continue
            published_members = tuple(members[position] for position in huddle_positions)
            if published_members != members:
                # The omitted mouse may have been a bridge in the proposal's
                # connectivity graph. Reuse neither its compact mask nor its
                # endpoints as evidence for the published participant subset.
                # Derived sets strictly shrink, so this queue terminates; the
                # geometry cache and scheduled set avoid repeated evaluation.
                if published_members not in scheduled_members:
                    scheduled_members.add(published_members)
                    member_proposals.append(published_members)
                continue
            # Preserve the approved bounded ID-union/occlusion semantics.
            # Simultaneous tracks for every member on every frame would turn
            # an ordinary detector dropout into a false end of a stable pile.
            # A member still cannot be backfilled before its first support or
            # retained beyond its last observed/verified-held position.
            supported = [np.flatnonzero(held[window[0]:window[1], position]) + window[0]
                         for position in huddle_positions]
            first = max(int(indices[0]) for indices in supported)
            last = min(int(indices[-1]) + 1 for indices in supported)
            support_mask = np.zeros(len(f.valid), bool)
            support_mask[first:last] = huddle_compact[first:last]
            support_count = int(support_mask.sum())
            # Retain the existing minimum amount of compact evidence, without
            # demanding an uninterrupted support run or cutting a longer event
            # down to that minimum. Unknown rows are not credited or filled.
            if ((last - first) / f.fps + 1e-12 < p.huddle_min_duration_s
                    or support_count / f.fps + 1e-12 < p.huddle_min_duration_s):
                continue
            if not observed[first:last, list(huddle_positions)].any(axis=0).all():
                continue
            result.append(Candidate(
                "huddle",
                tuple(sorted(int(f.track_ids[members[position]]) for position in huddle_positions)),
                support_count / f.fps, ((first, last),), mask_intervals(support_mask)))
    for start, stop, members in (_compact_origin_episodes(
            f, near, neighbor, velocity_cm_s=velocity, config=config)
                    if behavior == "dispersal" else ()):
        key = "dispersal", members, neighbor
        if key not in geometry_cache:
            geometry_cache[key] = _geometry(f, members, near)
        compact, enough, observed = geometry_cache[key]
        internal = np.isin(f.pair_i, members) & np.isin(f.pair_j, members)
        pair_observed = f.valid[start:stop, f.pair_i[internal]] & f.valid[start:stop, f.pair_j[internal]]
        distances = np.where(pair_observed, f.pair_distance_cm[start:stop, internal], np.nan)
        confirmations = []
        _dispersal_process(
            distances, enough[start:stop], compact[start:stop], p, f.fps,
            centers_cm=f.centers_cm[start:stop, members],
            member_valid=observed[start:stop],
            initial_group_radius_cm=neighbor,
            extend_actual_motion=config.dispersal_extend_actual_motion,
            minimum_outward_speed_cm_s=config.dispersal_min_outward_speed_cm_s,
            pause_max_s=config.dispersal_pause_max_s,
            member_velocity_cm_s=velocity[start:stop, members],
            confirmations=confirmations, origin_start=0,
        )
        for phase, leaving, checkpoint in confirmations:
            endpoints = np.flatnonzero(phase)
            # Each distinct set must independently pass one real checkpoint.
            # Never union members from incompatible episodes or missing rows.
            duration = ((int(endpoints[-1]) - int(endpoints[0])) / f.fps
                        if len(endpoints) > 1 else 0.0)
            if duration > 0:
                intervals = tuple((int(low + start), int(high + start)) for low, high in mask_intervals(phase))
                leaving_subset = tuple(sorted(int(f.track_ids[members[slot]]) for slot in leaving))
                if not any(c.target_ids == leaving_subset and c.frame_intervals == intervals for c in result):
                    result.append(Candidate("dispersal", leaving_subset, duration, intervals))
                full_group = tuple(sorted(int(f.track_ids[m]) for m in members))
                if full_group != leaving_subset:
                    displacements = [
                        float(np.nanmax(np.linalg.norm(f.centers_cm[start:stop, m] - f.centers_cm[start, m], axis=1)))
                        for m in members if f.valid[start, m]
                    ]
                    ref_body = (
                        float(f.reference_body_cm)
                        if f.reference_body_cm is not None and f.reference_body_cm > 0
                        else 7.918
                    )
                    min_disp = 0.18 * ref_body
                    if len(displacements) == len(members) and all(d >= min_disp for d in displacements):
                        if not any(c.target_ids == full_group and c.frame_intervals == intervals for c in result):
                            result.append(Candidate("dispersal", full_group, duration, intervals))
    if behavior == "social_clustering" and len(result) > 1:
        cand_masks = []
        for cand in result:
            m = np.zeros(len(f.valid), bool)
            for s, e in (cand.frame_intervals or ()):
                m[s:e] = True
            cand_masks.append(m)
        n = len(result)
        adj = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if (cand_masks[i] & cand_masks[j]).any() and len(set(result[i].target_ids) & set(result[j].target_ids)) >= 2:
                    adj[i].append(j)
                    adj[j].append(i)
        visited = set()
        for i in range(n):
            if i not in visited:
                comp = []
                queue = [i]
                visited.add(i)
                while queue:
                    curr = queue.pop(0)
                    comp.append(curr)
                    for nbr in adj[curr]:
                        if nbr not in visited:
                            visited.add(nbr)
                            queue.append(nbr)
                if len(comp) > 1:
                    union_ids = set()
                    for idx in comp:
                        union_ids.update(result[idx].target_ids)
                    if len(union_ids) > max(len(result[idx].target_ids) for idx in comp):
                        lookup = {int(track): slot for slot, track in enumerate(f.track_ids)}
                        if all(tid in lookup for tid in union_ids):
                            union_members = tuple(sorted(lookup[tid] for tid in union_ids))
                            union_proc = _clustering_process(
                                f, union_members, near, velocity, speed, config,
                                neighbor_cm=neighbor, max_gap_s=p.huddle_max_gap_s,
                            )
                            if union_proc.evidence > 0 and not _members_in_preexisting_scene_cluster(
                                f, union_members, union_proc, near, velocity, speed, config,
                                neighbor_cm=neighbor, max_gap_s=p.huddle_max_gap_s,
                                scene_cache=scene_cache,
                            ):
                                cand = Candidate(
                                    "social_clustering",
                                    tuple(sorted(union_ids)),
                                    union_proc.evidence,
                                    union_proc.frames,
                                    union_proc.support_frames,
                                )
                                if cand not in result:
                                    result.append(cand)
    return tuple(result)


def group_rule_candidates(f: ExportedFeatures, p: HeuristicParameters,
                          proposals: Sequence[Candidate],
                          config: GroupHeuristicConfig = DEFAULT_GROUP_HEURISTIC_CONFIG,
                          *, behaviors: Sequence[str] | None = None) -> tuple[Candidate, ...]:
    """Generate each process with its own limits, without annotation inputs.

    ``behaviors`` selects output families only; it cannot supply members or
    change a process rule. The default still generates all four processes.
    """
    selected = ("huddle", "social_clustering", "group_locomotion", "dispersal")
    if behaviors is not None:
        unknown = set(behaviors).difference(selected)
        if unknown:
            raise ValueError(f"Unknown group behavior(s): {sorted(unknown)}")
        selected = tuple(name for name in selected if name in behaviors)
    geometry_cache: dict = {}
    by_behavior = {
        name: _group_behavior_candidates(f, p, proposals, config, name, geometry_cache)
        for name in selected
    }
    if "social_clustering" in by_behavior and "group_locomotion" in by_behavior:
        gl_by_ids = {
            c.target_ids: c.evidence for c in by_behavior["group_locomotion"]
        }
        by_behavior["social_clustering"] = tuple(
            c for c in by_behavior["social_clustering"]
            if gl_by_ids.get(c.target_ids, 0.0) < c.evidence
        )
    return tuple(candidate for name in selected for candidate in by_behavior[name])


def isolation_summaries(f: ExportedFeatures, excluded_pairs=()) -> dict[str, dict[str, float | int | None]]:
    result = {}
    nearest = nearest_comparison_distances(f, excluded_pairs)
    comparison = f.valid & (f.valid.sum(axis=1) >= 2)[:, None] & np.isfinite(nearest)
    for slot, track in enumerate(f.track_ids):
        values = nearest[comparison[:, slot], slot]
        result[str(int(track))] = {
            "observed_frames": len(values), "minimum_cm": float(values.min()) if len(values) else None,
            "q05_cm": float(np.quantile(values, 0.05)) if len(values) else None,
            "median_cm": float(np.median(values)) if len(values) else None,
            "maximum_cm": float(values.max()) if len(values) else None,
        }
    return result


def isolation_candidates(f: ExportedFeatures, p: HeuristicParameters,
                         config: GroupHeuristicConfig = DEFAULT_GROUP_HEURISTIC_CONFIG, *, excluded_pairs=()) -> tuple[Candidate, ...]:
    result = []
    nearest = nearest_comparison_distances(f, excluded_pairs)
    for slot, (track, row) in enumerate(isolation_summaries(f, excluded_pairs).items()):
        seconds = float(row["observed_frames"]) / f.fps
        if (row["minimum_cm"] is not None and row["minimum_cm"] > p.isolation_distance_cm
                and seconds > max(p.isolation_min_duration_s, config.isolation_min_duration_s)):
            mask = (f.valid[:, slot] & (f.valid.sum(axis=1) >= 2)
                    & np.isfinite(nearest[:, slot]))
            indices = np.flatnonzero(mask)
            # The rule has always required total valid comparison time, not
            # uninterrupted detections. Preserve one observed process envelope
            # and carry actual comparison rows separately. No pose, identity,
            # distance, or effective time is filled across the unknown rows.
            result.append(Candidate("isolation", (int(track),), seconds,
                ((int(indices[0]), int(indices[-1]) + 1),), mask_intervals(mask)))
    return tuple(result)
