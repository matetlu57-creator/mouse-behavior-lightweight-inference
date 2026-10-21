"""Observed group processes; missing visual IDs never imply a new identity.

Event durations measure a process envelope, not an uninterrupted velocity mask.
Huddling holds missing observations for a bounded five-second interval. Observed spatial
separation remains contradictory evidence and is never filled as a dropout.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from itertools import combinations
from typing import Sequence
import warnings

import cv2
import numpy as np

from .blind_top1 import Candidate
from .exported_dataset_models import ExportedFeatures, HeuristicParameters
from .group_heuristic_config import DEFAULT_GROUP_HEURISTIC_CONFIG, GroupHeuristicConfig
from .screening_signals import ScreenPolicy, runs
from . import screening_signals
from .event_frames import FrameIntervals, mask_intervals
from .clustering_formation import formation_masks


# [ERA 2026-10-16 group-v16.2] Bumped from
# "confirmed_group_rules_v14_total_support_whole_component". v16 added
# _huddle_geometry, rewrote _initial_compact_components, extended the
# dispersal_phase signature and relaxed _clustering_event; v16.2 bounds the
# huddle ID union to its stable window, reports compact support instead of the
# episode span, and adds the single-degraded-recipient region-gap fallback.
# policy_receipt() publishes this string into validation receipts, so a group
# run is now version-separable from v14.
GROUP_RULE_REVISION = "2026-10-20-group-v29-observed-formation-and-state-memory"
NEIGHBOR_BODY_LENGTHS = 1.25
MAX_PROPOSALS = 160
MIN_MOTION_SUPPORT_SECONDS = 0.15
# [ERA 2026-10-16 group-v16.2] Every threshold added by v16.2 is a
# dimensionless fraction on purpose. preprocessing/scale_migration.py
# rescales cm_per_pixel by 0.74 at load, so a new absolute cm literal taken
# from the plan document (all of which are written in the un-migrated 10.7 cm
# body era) would be 1.35x mis-scaled against the runtime quantity it is
# compared to; see DEFECT ERA-1. Ratios need no re-derivation.
#
# A member must be observed-or-verified-held for at least this fraction of the
# qualifying stable window to remain in the emitted huddle ID union. Mirrors
# the 0.6 connected-group support convention already used by
# calibrated_rule_ranker._aggregate_group_candidates.
HUDDLE_MEMBER_SUPPORT_FRACTION = 0.6
# A recipient body region needs at least this many finite keypoints to take
# part in the convex hull (was an inline literal 4).
MINIMUM_REGION_KEYPOINTS = 4


@dataclass(frozen=True)
class _GroupProcess:
    evidence: float = 0.0
    frames: FrameIntervals = ()
    support_frames: FrameIntervals | None = None


def group_separation(distances: np.ndarray, percentile: float = 75.0) -> np.ndarray:
    values = np.asarray(distances, float)
    if values.ndim != 2:
        raise ValueError("group distances must be a frame-by-pair matrix")
    if not values.shape[1]:
        return np.full(len(values), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanpercentile(np.where(np.isfinite(values), values, np.nan), percentile, axis=1)


def _components(adjacency: np.ndarray, active: np.ndarray) -> list[tuple[int, ...]]:
    unseen = set(int(value) for value in np.flatnonzero(active))
    result = []
    while unseen:
        root = min(unseen)
        reached, queue = {root}, [root]
        unseen.remove(root)
        while queue:
            current = queue.pop()
            neighbors = unseen.intersection(np.flatnonzero(adjacency[current]).tolist())
            unseen.difference_update(neighbors)
            reached.update(neighbors)
            queue.extend(neighbors)
        result.append(tuple(sorted(reached)))
    return result


def _geometry(f: ExportedFeatures, members: tuple[int, ...], near: np.ndarray):
    observed = f.valid[:, members] & np.isfinite(f.centers_cm[:, members]).all(axis=2)
    enough = observed.sum(axis=1) >= 2
    compact = np.zeros(len(observed), bool)
    internal = np.flatnonzero(np.isin(f.pair_i, members) & np.isin(f.pair_j, members))
    lookup = {slot: index for index, slot in enumerate(members)}
    for frame in np.flatnonzero(enough):
        adjacency = np.zeros((len(members), len(members)), bool)
        for edge in internal[near[frame, internal]]:
            left, right = lookup[int(f.pair_i[edge])], lookup[int(f.pair_j[edge])]
            adjacency[left, right] = adjacency[right, left] = True
        components = _components(adjacency, observed[frame])
        compact[frame] = len(components) == 1
    return compact, enough, observed


def _inside_pile_envelope(centers: np.ndarray, observed: np.ndarray, frame: int,
                          column: int, neighbor_cm: float) -> bool:
    """True when this member sits inside the envelope of the pile it claims.

    The envelope is the centroid of the members observed in that same frame,
    with the neighbour radius as tolerance. A dropout may only be bridged for a
    mouse that was inside the pile when it vanished AND is inside the pile when
    it reappears; the member's own displacement across the gap cannot show
    either, which is what let a mouse 25 cm outside the pile stay a member.
    """
    others = [slot for slot in range(centers.shape[1])
              if slot != column and observed[frame, slot]]
    if not others:
        return False
    position = centers[frame, column]
    if not np.isfinite(position).all():
        return False
    centroid = np.mean(centers[frame, others], axis=0)
    return bool(np.linalg.norm(position - centroid) <= neighbor_cm)


def _quiet_at(speed: np.ndarray | None, frame: int, column: int,
              quiet_speed_cm_s: float | None) -> bool:
    """Quiet-speed half of the hold precondition.

    ``quiet_speed_cm_s`` is the dedicated huddle_hold_max_speed_cm_s (2.0),
    separate from the clustering quiet speed (4.0 in v17). A caller with no speed, or a frame
    whose speed is not finite, cannot be gated and is let through: this is a
    hold verifier, not a huddle detector.
    """
    if speed is None or quiet_speed_cm_s is None:
        return True
    value = speed[frame, column]
    return bool(not np.isfinite(value) or value <= quiet_speed_cm_s)


def _huddle_geometry(
    f: ExportedFeatures, members: tuple[int, ...], near: np.ndarray,
    *, neighbor_cm: float, max_gap_s: float,
    speed: np.ndarray | None = None, quiet_speed_cm_s: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Forward-hold an observed pile member without requiring reappearance.

    The last observed position must be quiet and inside the pile. A hold is
    bounded by max_gap_s and ends at the next real observation. Visible movement
    of the remaining pile away from that position also ends the hold. Held rows
    stay distinguishable from raw observed rows; no new visual ID is invented.
    """
    centers = np.asarray(f.centers_cm[:, members], dtype=float)
    observed = f.valid[:, members] & np.isfinite(centers).all(axis=2)
    member_speed = None if speed is None else np.asarray(speed, dtype=float)[:, members]
    held_centers = centers.copy()
    held = observed.copy()
    max_gap = int(np.floor(max_gap_s * f.fps + 1e-9))
    for column in range(len(members)):
        for left in np.flatnonzero(observed[:, column]):
            left = int(left)
            if not (_inside_pile_envelope(centers, observed, left, column, neighbor_cm)
                    and _quiet_at(member_speed, left, column, quiet_speed_cm_s)):
                continue
            position = centers[left, column]
            for frame in range(left + 1, min(len(observed), left + max_gap + 1)):
                if observed[frame, column]:
                    break
                other = observed[frame].copy()
                other[column] = False
                if other.sum() >= 2:
                    anchor = np.median(centers[frame, other], axis=0)
                    if np.linalg.norm(position - anchor) > neighbor_cm:
                        break
                held_centers[frame, column] = position
                held[frame, column] = True

    enough = observed.sum(axis=1) >= 2
    compact = np.zeros(len(observed), dtype=bool)
    # Only intra-member pairs are ever looked up, so the table is built from
    # those edges instead of every pair in the clip on every call.
    internal_edges = np.flatnonzero(np.isin(f.pair_i, members) & np.isin(f.pair_j, members))
    edge_lookup = {
        (int(f.pair_i[edge]), int(f.pair_j[edge])): int(edge)
        for edge in internal_edges
    }
    for frame in np.flatnonzero(enough):
        adjacency = np.zeros((len(members), len(members)), dtype=bool)
        for left_index, left in enumerate(members):
            if not held[frame, left_index]:
                continue
            for right_index in range(left_index + 1, len(members)):
                if not held[frame, right_index]:
                    continue
                right = members[right_index]
                edge = edge_lookup.get((left, right))
                if edge is None:
                    continue
                if observed[frame, left_index] and observed[frame, right_index]:
                    connected = bool(near[frame, edge])
                else:
                    connected = (np.linalg.norm(held_centers[frame, left_index]
                                                - held_centers[frame, right_index]) <= neighbor_cm)
                if connected:
                    adjacency[left_index, right_index] = adjacency[right_index, left_index] = True
        compact[frame] = len(_components(adjacency, held[frame])) == 1
    return compact, enough, observed, held_centers, held


def _proposal_sets(f: ExportedFeatures, near: np.ndarray, velocity: np.ndarray,
                   speed: np.ndarray, p: HeuristicParameters,
                   proposals: Sequence[Candidate]) -> list[tuple[int, ...]]:
    """Frame components and their shared-core unions preserve late/recovered IDs."""
    count: Counter[tuple[int, ...]] = Counter()
    tracks = len(f.track_ids)
    aggregate = np.zeros((tracks, tracks), bool)
    for frame in range(len(f.valid)):
        adjacency = np.zeros_like(aggregate)
        edges = np.flatnonzero(near[frame])
        adjacency[f.pair_i[edges], f.pair_j[edges]] = True
        adjacency |= adjacency.T
        aggregate |= adjacency
        for members in _components(adjacency, f.valid[frame]):
            if len(members) >= 2:
                count[members] += 1
        active = f.valid[frame] & np.isfinite(speed[frame]) & (speed[frame] >= p.collective_active_speed_cm_s)
        slots = np.flatnonzero(active)
        if len(slots) >= 3:
            unit = velocity[frame, slots] / np.maximum(speed[frame, slots, None], 1e-9)
            direction_graph = np.zeros_like(aggregate)
            direction_graph[np.ix_(slots, slots)] = unit @ unit.T >= p.group_locomotion_min_direction_similarity
            for members in _components(direction_graph, active):
                if len(members) >= 3:
                    count[members] += 1
    selected = {members for members, _ in count.most_common(MAX_PROPOSALS) if len(members) >= 3}
    selected.update(members for members in _components(aggregate, f.valid.any(axis=0)) if len(members) >= 3)
    cores = [members for members, _ in count.most_common(64)]
    for core in cores:
        merged = set(core)
        for other in cores:
            if len(merged.intersection(other)) >= 2:
                merged.update(other)
        if len(merged) >= 3:
            selected.add(tuple(sorted(merged)))
    lookup = {int(track): slot for slot, track in enumerate(f.track_ids)}
    selected.update(tuple(sorted(lookup[track] for track in candidate.target_ids))
                    for candidate in proposals if len(candidate.target_ids) >= 3
                    and all(track in lookup for track in candidate.target_ids))
    # Triples permit an actor to join a pair even if an unrelated mouse shares
    # the larger component. Existing IDs alone define every proposal.
    for members, _ in count.most_common(32):
        if 3 <= len(members) <= 8:
            selected.update(combinations(members, 3))
    return sorted(selected, key=lambda members: (-count[members], -len(members), members))[:MAX_PROPOSALS]


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


def _window_positions(held: np.ndarray, window: tuple[int, int, int]) -> tuple[int, ...]:
    """Member POSITIONS the stable window supports, i.e. its bounded ID union.

    Positions index into the proposal tuple, matching the columns of ``held``.
    A mouse that passes within the neighbour radius for a fraction of a second,
    or that departs and is never re-detected, is not a participant even though
    the clip-wide transitive closure of the near-graph puts it in the proposal.
    """
    start, stop, support_frames = window
    if support_frames <= 0 or stop <= start:
        return ()
    fraction = held[start:stop].sum(axis=0) / float(stop - start)
    return tuple(position for position in range(held.shape[1])
                 if fraction[position] + 1e-9 >= HUDDLE_MEMBER_SUPPORT_FRACTION)


def _region_gap(f: ExportedFeatures, actor: int, recipients: tuple[int, ...]) -> np.ndarray:
    """Snout gap to the existing group's convex observed body region.

    A directed nearest-keypoint distance is a conservative observation-only
    fallback. Center distance never substitutes for missing snout geometry.
    """
    result = np.full(len(f.valid), np.nan)
    points = f.collective_keypoints_cm
    shared = f.valid[:, (actor,) + recipients].all(axis=1)
    if points is not None:
        for frame in np.flatnonzero(shared):
            nose = points[frame, actor, 0]
            if not np.isfinite(nose).all():
                continue
            # One degraded recipient must not void the frame for the whole
            # group. Dropping it leaves a smaller observed body region, which
            # is a conservative gap estimate; only two or more missing
            # recipients leave too little region to mean anything.
            region = []
            for slot in recipients:
                body = points[frame, slot]
                finite = np.isfinite(body).all(axis=1)
                if finite.sum() < MINIMUM_REGION_KEYPOINTS:
                    continue
                region.append(body[finite])
            if len(region) < max(1, len(recipients) - 1):
                continue
            hull = cv2.convexHull(np.concatenate(region).astype(np.float32))
            if len(hull) >= 3 and cv2.contourArea(hull) > 1e-9:
                result[frame] = max(0.0, -cv2.pointPolygonTest(hull, tuple(map(float, nose)), True))
        return result
    directed = []
    for target in recipients:
        indices = np.flatnonzero(((f.pair_i == actor) & (f.pair_j == target))
                                | ((f.pair_i == target) & (f.pair_j == actor)))
        if not len(indices):
            return result
        edge = int(indices[0])
        values = (f.pair_nose_nearest_left_to_right_cm if int(f.pair_i[edge]) == actor
                  else f.pair_nose_nearest_right_to_left_cm)
        if values is None:
            return result
        directed.append(values[:, edge])
    measured = np.stack(directed, axis=1)
    known = shared & np.isfinite(measured).all(axis=1)
    result[known] = measured[known].min(axis=1)
    return result


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
    common_center = np.mean(centers, axis=1)
    dense = (np.linalg.norm(centers - common_center[:, None], axis=2) <= neighbor_cm).all(axis=1)
    minimum = max(1, int(np.ceil(config.clustering_final_stability_s * f.fps - 1e-9)))
    mask = np.zeros(len(f.valid), bool)
    motion_mask = np.zeros(len(f.valid), bool)
    for actor_column, actor in enumerate(members):
        recipients = tuple(m for m in members if m != actor)
        seed_compact, _, seed_observed = _geometry(f, recipients, near)
        recipient_centers = f.centers_cm[:, recipients]
        seed_center = np.mean(recipient_centers, axis=1)
        seed_dense = (np.linalg.norm(recipient_centers - seed_center[:, None], axis=2)
                      <= neighbor_cm).all(axis=1)
        known = (observed.all(axis=1) & seed_observed.all(axis=1)
                 & seed_compact & seed_dense)
        actor_centers = centers[:, actor_column]
        offset = seed_center - actor_centers
        distance = np.linalg.norm(offset, axis=1)
        # V26 uses the common group-center envelope. A second radius around
        # recipients' centroid silently rejects valid nose-connected states.
        final = known & compact & dense
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
                if not len(active):
                    continue
                onset = int(active[0]) - 1
                # Restore V26's full observed dense-state window, with a new
                # measured formation gate. The state window cannot supply
                # missing actor travel or turn a static pile into formation.
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
    return _GroupProcess(float(mask.sum()) / f.fps, mask_intervals(mask),
                         mask_intervals(motion_mask))

def _clustering_event(*args, **kwargs) -> float:
    """Compatibility score for consumers that still request only seconds."""
    return _clustering_process(*args, **kwargs).evidence


def _departure_origins(centers: np.ndarray, observed: np.ndarray, start: int,
                       radius_cm: float) -> tuple[np.ndarray, np.ndarray]:
    """Record each existing visual ID's first observation in the fixed origin.

    IDs may become visible progressively. A first observation outside the
    origin is not backfilled, even if that track enters the origin later.
    No point or ID is interpolated across its missing prefix.
    """
    initial = centers[start, observed[start]]
    joins = np.full(centers.shape[1], len(centers), int)
    origins = np.full((centers.shape[1], 2), np.nan)
    if len(initial) < 2:
        return joins, origins
    anchor = np.median(initial, axis=0)
    for slot in range(centers.shape[1]):
        seen = np.flatnonzero(observed[start:, slot]) + start
        if not len(seen):
            continue
        first = int(seen[0])
        position = centers[first, slot]
        if np.isfinite(position).all() and np.linalg.norm(position - anchor) <= radius_cm:
            joins[slot], origins[slot] = first, position
    return joins, origins


def _dispersal_endpoint_separated(points: np.ndarray, distance_cm: float = 3.) -> bool:
    """Close (<3cm) relations form a matching: each mouse has degree <=1."""
    points = np.asarray(points, float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        return False
    if not np.isfinite(points).all():
        return False
    distances = np.linalg.norm(points[:, None] - points[None, :], axis=2)
    close = distances < distance_cm
    np.fill_diagonal(close, False)
    return bool((close.sum(axis=1) <= 1).all())


def _departing_member_slots(
    centers: np.ndarray, observed: np.ndarray, start: int, endpoint: int,
    *, minimum_outward_displacement_cm: float, minimum_visible_fraction: float,
    member_starts: np.ndarray | None = None, origin_points: np.ndarray | None = None,
) -> tuple[int, ...]:
    """Measured departure from each member's own first in-origin observation."""
    if member_starts is None:
        member_starts = np.where(observed[start], start, len(centers))
    if origin_points is None:
        origin_points = np.where(observed[start, :, None], centers[start], np.nan)
    admitted = member_starts <= endpoint
    points = origin_points[admitted & np.isfinite(origin_points).all(axis=1)]
    if len(points) < 3:
        return ()
    anchor = np.median(points, axis=0)
    hull = cv2.convexHull(points.astype(np.float32))
    leaving = []
    for slot in np.flatnonzero(admitted & observed[endpoint]):
        first = int(member_starts[slot])
        if observed[first:endpoint + 1, slot].mean() < minimum_visible_fraction:
            continue
        initial, final = origin_points[slot], centers[endpoint, slot]
        if not np.isfinite(final).all():
            continue
        outward = np.linalg.norm(final - anchor) - np.linalg.norm(initial - anchor)
        net = np.linalg.norm(final - initial)
        outside = cv2.pointPolygonTest(hull, tuple(map(float, final)), False) < 0
        if outside and outward >= minimum_outward_displacement_cm and net >= minimum_outward_displacement_cm:
            leaving.append(int(slot))
    return tuple(leaving)

def _dispersal_process(distances: np.ndarray, visible: np.ndarray, compact: np.ndarray,
                    p: HeuristicParameters, fps: float, *,
                    centers_cm: np.ndarray | None = None,
                    member_valid: np.ndarray | None = None,
                    initial_group_radius_cm: float = 10.0,
                    minimum_outward_members: int = 2,
                    minimum_outward_displacement_cm: float = 4.0,
                    minimum_visible_fraction: float = 0.6,
                    extend_actual_motion: bool = True,
                    minimum_outward_speed_cm_s: float = DEFAULT_GROUP_HEURISTIC_CONFIG.dispersal_min_outward_speed_cm_s,
                    pause_max_s: float = DEFAULT_GROUP_HEURISTIC_CONFIG.dispersal_pause_max_s,
                    member_velocity_cm_s: np.ndarray | None = None,
                    confirmations: list | None = None) -> tuple[np.ndarray, tuple[int, ...], int | None]:
    """Confirm a >5-second actual departure at a measured checkpoint after5s.

    A connected observed pair anchors the original pile. At least three IDs
    must have their own first observed positions inside its fixed radius by
    confirmation; they need not coexist at the first frame. Each departing ID
    needs >=60% observation since its own entry and >=4cm net/radial departure.
    At the checkpoint every departed mouse may have at most one neighbor <3cm.
    Unknown positions never establish membership or endpoint separation.
    With ``confirmations`` supplied, retain each independently qualified
    (member set, process interval) instead of stopping at the first subset.
    """
    if minimum_outward_members < 2 or not 0 < minimum_visible_fraction <= 1:
        raise ValueError("outward-member and visible-fraction thresholds are invalid")
    if not np.isfinite(minimum_outward_displacement_cm) or minimum_outward_displacement_cm <= 0:
        raise ValueError("minimum_outward_displacement_cm must be finite and positive")
    if not np.isfinite(initial_group_radius_cm) or initial_group_radius_cm <= 0:
        raise ValueError("initial_group_radius_cm must be finite and positive")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    values = np.asarray(distances)
    empty = np.zeros(len(visible), bool)
    if values.ndim != 2 or values.shape[0] != len(visible) or len(compact) != len(visible):
        raise ValueError("distances/visible/compact must align on frames")
    if centers_cm is None or member_valid is None:
        return empty, (), None
    centers = np.asarray(centers_cm, dtype=float)
    observed = np.asarray(member_valid, dtype=bool)
    if (centers.ndim != 3 or centers.shape[0] != len(visible)
            or centers.shape[-1] != 2 or observed.shape != centers.shape[:2]):
        raise ValueError("centers_cm/member_valid must align as (frames, members, 2)/(frames, members)")
    starts = np.flatnonzero(np.asarray(visible, bool) & np.asarray(compact, bool)
                           & (observed.sum(axis=1) >= 2))
    if not len(starts):
        return empty, (), None
    start = int(starts[0])
    joins, origins = _departure_origins(centers, observed, start, initial_group_radius_cm)
    # Historical explicit checkpoint mode is available for old receipts only.
    # The canonical path requires a measured checkpoint strictly AFTER5s and
    # a genuine motion process longer than5s, with no quiet-tail padding.
    first_check = start + (int(np.floor(5.0 * fps + 1e-9)) + 1
                           if extend_actual_motion else int(np.ceil(5.0 * fps - 1e-9)))
    checkpoints = range(first_check, len(visible)) if extend_actual_motion else (first_check,)
    seen_confirmations = set()
    first_confirmation = None
    for endpoint in checkpoints:
        if endpoint >= len(visible) or observed[endpoint].sum() < minimum_outward_members:
            continue
        leaving = _departing_member_slots(
            centers, observed, start, endpoint,
            minimum_outward_displacement_cm=minimum_outward_displacement_cm,
            minimum_visible_fraction=minimum_visible_fraction,
            member_starts=joins, origin_points=origins,
        )
        if len(leaving) < minimum_outward_members:
            continue
        final = centers[endpoint, list(leaving)]
        if not _dispersal_endpoint_separated(final):
            continue
        if not extend_actual_motion:
            phase = empty.copy()
            phase[start:endpoint + 1] = True
        else:
            phase = _departure_process_mask(
                centers, observed, start, endpoint, leaving, fps,
                minimum_outward_speed_cm_s=minimum_outward_speed_cm_s,
                pause_max_s=pause_max_s, velocity_cm_s=member_velocity_cm_s,
                minimum_duration_s=p.dispersal_min_duration_s,
                member_starts=joins, origin_points=origins[joins <= endpoint],
            )
        if phase.any():
            boundaries = np.flatnonzero(phase)
            # Both the confirmation delay and actual process span are measured
            # from observed motion onset, not an earlier quiet origin frame.
            if extend_actual_motion and (endpoint - int(boundaries[0])) / fps <= 5.0:
                continue
            if np.count_nonzero(joins <= boundaries[-1]) < 3:
                continue
            # The process may continue after its confirmation. Its actual end
            # must still have measured endpoint geometry for every emitted ID.
            last = int(boundaries[-1])
            if (not observed[last, list(leaving)].all()
                    or not _dispersal_endpoint_separated(centers[last, list(leaving)])):
                continue
            confirmation = phase, leaving, endpoint
            if confirmations is None:
                return confirmation  # Historical mask API keeps its first event.
            key = leaving, mask_intervals(phase)
            if key not in seen_confirmations:
                confirmations.append(confirmation)
                seen_confirmations.add(key)
                if first_confirmation is None:
                    first_confirmation = confirmation
    return first_confirmation if first_confirmation is not None else (empty, (), None)


def dispersal_phase(distances: np.ndarray, visible: np.ndarray, compact: np.ndarray,
                    p: HeuristicParameters, fps: float, *,
                    centers_cm: np.ndarray | None = None,
                    member_valid: np.ndarray | None = None,
                    initial_group_radius_cm: float = 10.0,
                    minimum_outward_members: int = 2,
                    minimum_outward_displacement_cm: float = 4.0,
                    minimum_visible_fraction: float = 0.6,
                    extend_actual_motion: bool = True,
                    minimum_outward_speed_cm_s: float = DEFAULT_GROUP_HEURISTIC_CONFIG.dispersal_min_outward_speed_cm_s,
                    pause_max_s: float = DEFAULT_GROUP_HEURISTIC_CONFIG.dispersal_pause_max_s,
                    member_velocity_cm_s: np.ndarray | None = None) -> np.ndarray:
    """Mask compatibility API; canonical candidates retain confirmation members."""
    return _dispersal_process(
        distances, visible, compact, p, fps, centers_cm=centers_cm,
        member_valid=member_valid, initial_group_radius_cm=initial_group_radius_cm,
        minimum_outward_members=minimum_outward_members,
        minimum_outward_displacement_cm=minimum_outward_displacement_cm,
        minimum_visible_fraction=minimum_visible_fraction,
        extend_actual_motion=extend_actual_motion,
        minimum_outward_speed_cm_s=minimum_outward_speed_cm_s,
        pause_max_s=pause_max_s, member_velocity_cm_s=member_velocity_cm_s,
    )[0]


def _departure_process_mask(centers: np.ndarray, observed: np.ndarray,
                            start: int, checkpoint: int, leaving: tuple[int, ...], fps: float,
                            *, minimum_outward_speed_cm_s: float, pause_max_s: float,
                            velocity_cm_s: np.ndarray | None = None,
                            minimum_duration_s: float = 0.0,
                            member_starts: np.ndarray | None = None,
                            origin_points: np.ndarray | None = None) -> np.ndarray:
    """A confirmed +5 s departure is bounded by observed outward movement.

    Select the departure bout with greatest outward displacement before the
    checkpoint. A brief later jitter bout cannot replace the actual departure.
    A member's observed return into the initial pile ends the process.
    """
    if (not np.isfinite(minimum_outward_speed_cm_s) or minimum_outward_speed_cm_s <= 0
            or not np.isfinite(pause_max_s) or pause_max_s <= 0):
        raise ValueError("departure motion thresholds must be positive")
    count = len(centers)
    result = np.zeros(count, bool)
    initial = (centers[start, observed[start]] if origin_points is None
               else np.asarray(origin_points, float))
    if len(initial) < 3 or not np.isfinite(initial).all():
        return result
    anchor = np.median(initial, axis=0)
    hull = cv2.convexHull(initial.astype(np.float32))
    if velocity_cm_s is None:
        velocity = np.full_like(centers, np.nan, dtype=float)
        measured = observed[1:] & observed[:-1]
        differences = np.diff(centers, axis=0) * fps
        velocity[1:] = np.where(measured[:, :, None], differences, np.nan)
    else:
        velocity = np.asarray(velocity_cm_s, float)
        if velocity.shape != centers.shape:
            raise ValueError("member velocities must align with centers")
    positions = centers[:, leaving]
    valid = observed[:, leaving] & np.isfinite(positions).all(axis=2)
    if member_starts is not None:
        valid &= np.arange(count)[:, None] >= np.asarray(member_starts)[list(leaving)]
    offset = positions - anchor
    radius = np.linalg.norm(offset, axis=2)
    radial = np.divide(np.sum(offset * velocity[:, leaving], axis=2), radius,
                       out=np.full_like(radius, np.nan), where=radius > 1e-9)
    measured = valid & np.isfinite(radial)
    active = (measured.sum(axis=1) >= 2) & np.any(
        measured & (radial >= minimum_outward_speed_cm_s), axis=1,
    )
    active[:start] = False
    # Confirmation and event end are different timestamps. A later return is
    # contradictory evidence, so no pause bridge may carry an event across it.
    returns = np.zeros(count, bool)
    outside_seen = np.zeros(len(leaving), bool)
    for frame in range(start, count):
        for slot in range(len(leaving)):
            if not valid[frame, slot]:
                continue
            outside = cv2.pointPolygonTest(
                hull, tuple(map(float, positions[frame, slot])), False) < 0
            if outside_seen[slot] and not outside:
                returns[frame] = True
            outside_seen[slot] = outside
    active[returns] = False
    bridged = active.copy()
    bounds = runs(active)
    limit = int(np.floor(pause_max_s * fps + 1e-9))
    for (_, left), (right, _) in zip(bounds, bounds[1:]):
        if right - left <= limit and not returns[left:right].any():
            bridged[left:right] = True
    choices = []
    for low, high in runs(bridged):
        if (high - low - 1) / fps <= minimum_duration_s + 1e-9:
            continue
        supports = np.flatnonzero(active[low:high]) + low
        before = supports[supports <= checkpoint]
        if not len(before) or len(supports) / fps < MIN_MOTION_SUPPORT_SECONDS:
            continue
        if returns[low:checkpoint + 1].any():
            continue  # A later departure cannot confirm the pre-return episode.
        first, last = int(before[0]), int(before[-1])
        good = valid[first] & valid[last]
        gain = float(np.maximum(radius[last, good] - radius[first, good], 0.).sum())
        choices.append((gain, len(before), low, high))
    if choices:
        _, _, low, high = max(choices)
        result[low:high] = True
    return result


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
    spatial = (np.isfinite(distances).all(axis=1)
               & (distances < min(p.group_locomotion_max_distance_cm,
                                  config.locomotion_max_pair_distance_cm)).all(axis=1))
    synchronous = moving.all(axis=1) & spatial & (direction >= p.group_locomotion_min_direction_similarity)
    if np.count_nonzero(synchronous) / f.fps + 1e-12 < config.locomotion_simultaneous_min_s:
        return _GroupProcess()
    # The simultaneous one-second gate and the longer process span are
    # separate: pauses may extend an established shared-motion process.
    if not (moving.sum(axis=0) / f.fps >= MIN_MOTION_SUPPORT_SECONDS).all():
        return _GroupProcess()
    activity = np.flatnonzero(moving.any(axis=1))
    if activity.size < 2:
        return _GroupProcess()
    start, stop = int(activity[0]), int(activity[-1] + 1)
    pair_observed = f.valid[:, f.pair_i[internal]] & f.valid[:, f.pair_j[internal]]
    # A shared-motion seed cannot legitimize observed separation elsewhere
    # inside the process envelope. Missing comparisons remain unknown.
    contradicted = pair_observed & (~np.isfinite(distances)
                                   | (distances >= min(p.group_locomotion_max_distance_cm,
                                                       config.locomotion_max_pair_distance_cm)))
    if contradicted[start:stop].any():
        return _GroupProcess()
    displacements = []
    for slot in members:
        indices = np.flatnonzero(f.valid[start:stop, slot]) + start
        if len(indices) < 2:
            return _GroupProcess()
        displacement = f.centers_cm[indices[-1], slot] - f.centers_cm[indices[0], slot]
        norm = float(np.linalg.norm(displacement))
        if norm <= 1e-6:
            # Some callers provide measured velocities without position
            # changes; integrated observed motion is still real evidence.
            displacement = np.nansum(velocity[start:stop, slot], axis=0) / f.fps
            norm = float(np.linalg.norm(displacement))
        if norm <= 1e-6:
            return _GroupProcess()
        if norm + 1e-9 < config.locomotion_min_net_displacement_cm:
            return _GroupProcess()
        displacements.append(displacement / norm)
    coherence = np.mean([np.dot(displacements[left], displacements[right])
                         for left, right in combinations(range(len(members)), 2)])
    duration = (stop - start) / f.fps
    if coherence >= p.group_locomotion_min_direction_similarity and duration > p.group_locomotion_min_duration_s:
        return _GroupProcess(duration, ((start, stop),), mask_intervals(synchronous))
    return _GroupProcess()


def _synchronous_event(*args, **kwargs) -> float:
    """Compatibility score; event frames are exported by the process path."""
    return _synchronous_process(*args, **kwargs).evidence


def _initial_compact_components(
    f: ExportedFeatures, near: np.ndarray, neighbor_cm: float,
) -> list[tuple[int, ...]]:
    """Recover first-visible IDs whose first position lies inside the initial pile."""
    visible = f.valid & np.isfinite(f.centers_cm).all(axis=2)
    frames = np.flatnonzero(visible.sum(axis=1) >= 2)
    if not len(frames):
        return []
    first = int(frames[0])
    tracks = len(f.track_ids)
    adjacency = np.zeros((tracks, tracks), bool)
    edges = np.flatnonzero(near[first])
    adjacency[f.pair_i[edges], f.pair_j[edges]] = True
    adjacency |= adjacency.T
    first_observed = np.full(tracks, len(visible), dtype=int)
    for slot in range(tracks):
        seen = np.flatnonzero(visible[:, slot])
        if len(seen):
            first_observed[slot] = int(seen[0])
    result = []
    for seed in _components(adjacency, visible[first]):
        if len(seed) < 2:
            continue
        members = set(seed)
        anchor = np.median(f.centers_cm[first, list(seed)], axis=0)
        for slot in range(tracks):
            if slot in members or first_observed[slot] >= len(visible):
                continue
            first_frame = int(first_observed[slot])
            position = f.centers_cm[first_frame, slot]
            if (np.isfinite(position).all()
                    and np.linalg.norm(position - anchor) <= neighbor_cm):
                members.add(slot)
        # The back-fill admits a late ID by a CENTROID envelope test, while
        # dispersal_phase's entry gate reads `compact[start]` from _geometry, a
        # PAIRWISE-adjacency test. A back-filled point can sit inside the
        # centroid envelope yet farther than neighbor_cm from every seed member,
        # which splits the component, makes compact[start] False and returns an
        # all-False phase. Emitting the bare seed alongside the expansion stops
        # the back-fill from deleting the very candidate it was written to
        # rescue; the ranker keeps whichever set carries evidence.
        if len(members) >= 3:
            result.append(tuple(sorted(members)))
        if len(seed) >= 3:
            result.append(tuple(sorted(seed)))
    return list(dict.fromkeys(result))


def _observed_departure_velocity(f: ExportedFeatures) -> np.ndarray:
    """Smoothed member motion, recomputed inside each observed track run."""
    velocity = np.full_like(f.centers_cm, np.nan)
    policy = ScreenPolicy()
    for slot in range(len(f.track_ids)):
        known = f.valid[:, slot] & np.isfinite(f.centers_cm[:, slot]).all(axis=1)
        for low, high in runs(known):
            _, measured, _ = screening_signals.observed_run_motion(
                f.centers_cm[low:high, slot], f.fps, policy)
            velocity[low:high, slot] = measured
    return velocity


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
    for members in (_proposal_sets(f, near, velocity, speed, p, proposals)
                    if behavior != "dispersal" else ()):
        # One geometry per proposal. _clustering_event used to rebuild the
        # identical result, and the plain _geometry call here was dead: its
        # compact/enough were discarded and _huddle_geometry returns the same
        # `observed`.
        geometry = None
        if behavior == "social_clustering":
            process = _clustering_process(
                f, members, near, velocity, speed, config,
                neighbor_cm=neighbor, max_gap_s=p.huddle_max_gap_s)
            if process.evidence > 0:
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
        window = _huddle_window(
            huddle_compact, huddle_enough, observed, f.fps, p.huddle_min_duration_s,
            max_interior_unknown_s=p.huddle_max_gap_s,
        ) if behavior == "huddle" else (0, 0, 0)
        huddle_positions = _window_positions(held, window)
        huddle_mask = np.zeros(len(f.valid), bool)
        huddle_mask[window[0]:window[1]] = huddle_compact[window[0]:window[1]]
        values = {
            # Compact support frames, not the unknown episode envelope.
            "huddle": _GroupProcess(window[2] / f.fps if window[2] else 0.0,
                                     mask_intervals(huddle_mask)),
            "social_clustering": _clustering_process(
                f, members, near, velocity, speed, config,
                neighbor_cm=neighbor, max_gap_s=p.huddle_max_gap_s,
                geometry=geometry,
            ) if behavior == "social_clustering" else _GroupProcess(),
        }
        for label, process in values.items():
            duration = process.evidence
            if duration <= 0 or label != behavior:
                continue
            # Huddling publishes the ID union bounded by its stable window;
            # the other two keep the whole proposal, whose own gates already
            # require every member to be simultaneously supported.
            positions = (huddle_positions if label == "huddle"
                         else tuple(range(len(members))))
            if len(positions) < 3 or not observed[:, list(positions)].any(axis=0).all():
                continue
            result.append(Candidate(
                label,
                tuple(sorted(int(f.track_ids[members[position]]) for position in positions)),
                duration, process.frames, process.support_frames))
    for members in (_initial_compact_components(f, near, neighbor)
                    if behavior == "dispersal" else ()):
        compact, enough, observed = _geometry(f, members, near)
        internal = np.isin(f.pair_i, members) & np.isin(f.pair_j, members)
        pair_observed = f.valid[:, f.pair_i[internal]] & f.valid[:, f.pair_j[internal]]
        distances = np.where(pair_observed, f.pair_distance_cm[:, internal], np.nan)
        confirmations = []
        _dispersal_process(
            distances, enough, compact, p, f.fps,
            centers_cm=f.centers_cm[:, members],
            member_valid=f.valid[:, members] & np.isfinite(f.centers_cm[:, members]).all(axis=2),
            initial_group_radius_cm=neighbor,
            extend_actual_motion=config.dispersal_extend_actual_motion,
            minimum_outward_speed_cm_s=config.dispersal_min_outward_speed_cm_s,
            pause_max_s=config.dispersal_pause_max_s,
            member_velocity_cm_s=velocity[:, members],
            confirmations=confirmations,
        )
        for phase, leaving, checkpoint in confirmations:
            endpoints = np.flatnonzero(phase)
            # Each distinct set must independently pass one real checkpoint.
            # Never union members from incompatible episodes or missing rows.
            duration = ((int(endpoints[-1]) - int(endpoints[0])) / f.fps
                        if len(endpoints) > 1 else 0.0)
            if duration > 0:
                candidate = Candidate("dispersal", tuple(sorted(int(f.track_ids[members[slot]])
                                                                for slot in leaving)), duration, mask_intervals(phase))
                if candidate not in result:
                    result.append(candidate)
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
    return tuple(candidate for name in selected
                 for candidate in _group_behavior_candidates(f, p, proposals, config, name, geometry_cache))


def isolation_summaries(f: ExportedFeatures) -> dict[str, dict[str, float | int | None]]:
    result = {}
    comparison = f.valid & (f.valid.sum(axis=1) >= 2)[:, None] & np.isfinite(f.nearest_distance_cm)
    for slot, track in enumerate(f.track_ids):
        values = f.nearest_distance_cm[comparison[:, slot], slot]
        result[str(int(track))] = {
            "observed_frames": len(values), "minimum_cm": float(values.min()) if len(values) else None,
            "q05_cm": float(np.quantile(values, 0.05)) if len(values) else None,
            "median_cm": float(np.median(values)) if len(values) else None,
            "maximum_cm": float(values.max()) if len(values) else None,
        }
    return result


def isolation_candidates(f: ExportedFeatures, p: HeuristicParameters,
                         config: GroupHeuristicConfig = DEFAULT_GROUP_HEURISTIC_CONFIG) -> tuple[Candidate, ...]:
    result = []
    for slot, (track, row) in enumerate(isolation_summaries(f).items()):
        seconds = float(row["observed_frames"]) / f.fps
        if (row["minimum_cm"] is not None and row["minimum_cm"] > p.isolation_distance_cm
                and seconds > max(p.isolation_min_duration_s, config.isolation_min_duration_s)):
            mask = (f.valid[:, slot] & (f.valid.sum(axis=1) >= 2)
                    & np.isfinite(f.nearest_distance_cm[:, slot]))
            result.append(Candidate("isolation", (int(track),), seconds, mask_intervals(mask)))
    return tuple(result)
