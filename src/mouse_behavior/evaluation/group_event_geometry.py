"""Geometry, connected-component, and proposal primitives for observed group events.

Event durations measure a process envelope, not an uninterrupted velocity mask.
Huddling holds missing observations for a bounded five-second interval. Observed spatial
separation remains contradictory evidence and is never filled as a dropout.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from typing import Sequence
import warnings

import cv2
import numpy as np

from .blind_top1 import Candidate
from .event_frames import FrameIntervals
from .exported_dataset_models import ExportedFeatures, HeuristicParameters


NEIGHBOR_BODY_LENGTHS = 1.25
MAX_PROPOSALS = 240
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
    participants: tuple[int, ...] | None = None


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
    obs_counts = observed.sum(axis=1)
    enough = obs_counts >= 2
    compact = np.zeros(len(observed), bool)
    member_mask = np.zeros(f.valid.shape[1], bool)
    member_mask[list(members)] = True
    internal = np.flatnonzero(member_mask[f.pair_i] & member_mask[f.pair_j])
    if len(internal) == 0:
        return compact, enough, observed
    if len(members) == 2 and len(internal) == 1:
        compact = enough & near[:, internal[0]]
        return compact, enough, observed
    lookup = {slot: index for index, slot in enumerate(members)}
    local_l = np.asarray([lookup[int(f.pair_i[edge])] for edge in internal], dtype=int)
    local_r = np.asarray([lookup[int(f.pair_j[edge])] for edge in internal], dtype=int)
    near_internal = near[:, internal]
    active_edge_counts = near_internal.sum(axis=1)
    # A graph with obs_counts[frame] nodes requires at least (obs_counts[frame] - 1) active edges to be connected.
    cand_frames = np.flatnonzero(enough & (active_edge_counts >= obs_counts - 1))
    n_members = len(members)
    for frame in cand_frames:
        adjacency = np.zeros((n_members, n_members), bool)
        mask_row = near_internal[frame]
        if mask_row.any():
            ls = local_l[mask_row]
            rs = local_r[mask_row]
            adjacency[ls, rs] = True
            adjacency[rs, ls] = True
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
    if max_gap >= 1 and len(observed) > 1:
        next_missing = ~observed[1:]
        for column in range(len(members)):
            gap_starts = np.flatnonzero(observed[:-1, column] & next_missing[:, column])
            for left in gap_starts:
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
    member_mask = np.zeros(f.valid.shape[1], bool)
    member_mask[list(members)] = True
    internal_edges = np.flatnonzero(member_mask[f.pair_i] & member_mask[f.pair_j])
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
    pair_near = np.zeros((tracks, tracks), dtype=int)
    near_counts = np.sum(near, axis=0)
    np.add.at(pair_near, (f.pair_i, f.pair_j), near_counts)
    np.add.at(pair_near, (f.pair_j, f.pair_i), near_counts)
    pair_dir = np.zeros((tracks, tracks), dtype=int)
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
        if len(slots) >= 2:
            unit = velocity[frame, slots] / np.maximum(speed[frame, slots, None], 1e-9)
            pair_cos = unit @ unit.T >= p.group_locomotion_min_direction_similarity
            np.fill_diagonal(pair_cos, False)
            pair_dir[np.ix_(slots, slots)] += pair_cos
            if len(slots) >= 3:
                direction_graph = np.zeros_like(aggregate)
                direction_graph[np.ix_(slots, slots)] = pair_cos
                for members in _components(direction_graph, active):
                    if len(members) >= 3:
                        count[members] += 1
    pair_cohesion = pair_near + pair_dir
    valid_i = f.valid.astype(np.int32)
    pair_obs = valid_i.T @ valid_i
    min_pair_obs = min(15, max(3, int(round(0.3 * f.fps))))

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
    # Pairwise core unions and 1-track additions capture k>=4 groups split by brief occlusion.
    for idx_a, core_a in enumerate(cores[:16]):
        if 3 <= len(core_a) <= 7:
            core_set = set(core_a)
            for v in range(tracks):
                if v not in core_set and int((pair_cohesion[list(core_a), v] > 0).sum()) >= 2:
                    selected.add(tuple(sorted(core_set | {v})))
        for core_b in cores[idx_a + 1:16]:
            if len(set(core_a).intersection(core_b)) >= 2:
                u = tuple(sorted(set(core_a) | set(core_b)))
                if 3 <= len(u) <= 8:
                    selected.add(u)
    lookup = {int(track): slot for slot, track in enumerate(f.track_ids)}
    selected.update(tuple(sorted(lookup[track] for track in candidate.target_ids))
                    for candidate in proposals if len(candidate.target_ids) >= 3
                    and all(track in lookup for track in candidate.target_ids))
    # Subsets permit an actor to join a cluster even if an unrelated mouse shares
    # the larger component. Existing IDs alone define every proposal.
    for members, _ in count.most_common(32):
        if 3 <= len(members) <= 8:
            for k in range(3, min(len(members) + 1, 8)):
                selected.update(combinations(members, k))

    def _proposal_rank(members: tuple[int, ...]) -> tuple[int, float, float, int, tuple[int, ...]]:
        c = count[members]
        if c > 0:
            return (0, float(-c), float(-len(members)), 0, members)
        pair_rates = []
        for u, v in combinations(members, 2):
            obs_uv = int(pair_obs[u, v])
            rate = (float(pair_cohesion[u, v]) / obs_uv) if obs_uv >= min_pair_obs else 0.0
            pair_rates.append(rate)
        min_rate = min(pair_rates) if pair_rates else 0.0
        mean_rate = (sum(pair_rates) / len(pair_rates)) if pair_rates else 0.0
        return (1, -round(min_rate, 4), -round(mean_rate, 4), -len(members), members)

    return sorted(selected, key=_proposal_rank)[:MAX_PROPOSALS]


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
