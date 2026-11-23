"""Observed dispersal origin, departure motion, and confirmation rules."""
from __future__ import annotations

import sys
from typing import Any

import cv2
import numpy as np

from . import screening_signals
from .event_frames import mask_intervals
from .exported_dataset_models import ExportedFeatures, HeuristicParameters
from .group_event_geometry import MIN_MOTION_SUPPORT_SECONDS, _components
from .group_heuristic_config import DEFAULT_GROUP_HEURISTIC_CONFIG, GroupHeuristicConfig
from .screening_signals import ScreenPolicy, runs


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
                    confirmations: list | None = None,
                    origin_start: int | None = None) -> tuple[np.ndarray, tuple[int, ...], int | None]:
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
    if origin_start is not None:
        if origin_start not in starts:
            return empty, (), None
        start = int(origin_start)
    else:
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
    rules_mod = sys.modules.get("mouse_behavior.evaluation.group_event_rules")
    process_fn = (
        getattr(rules_mod, "_dispersal_process", _dispersal_process)
        if rules_mod is not None
        else _dispersal_process
    )
    return process_fn(
        distances, visible, compact, p, fps, centers_cm=centers_cm,
        member_valid=member_valid, initial_group_radius_cm=initial_group_radius_cm,
        minimum_outward_members=minimum_outward_members,
        minimum_outward_displacement_cm=minimum_outward_displacement_cm,
        minimum_visible_fraction=minimum_visible_fraction,
        extend_actual_motion=extend_actual_motion,
        minimum_outward_speed_cm_s=minimum_outward_speed_cm_s,
        pause_max_s=pause_max_s, member_velocity_cm_s=member_velocity_cm_s,
    )[0]


def _prepare_departure_process(
    centers: np.ndarray,
    observed: np.ndarray,
    start: int,
    leaving: tuple[int, ...],
    fps: float,
    *,
    minimum_outward_speed_cm_s: float,
    pause_max_s: float,
    velocity_cm_s: np.ndarray | None = None,
    minimum_duration_s: float = 0.0,
    member_starts: np.ndarray | None = None,
    origin_points: np.ndarray | None = None,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int, np.ndarray]]] | None:
    """Precompute checkpoint-independent departure kinematics, return mask, and bridged runs."""
    if (not np.isfinite(minimum_outward_speed_cm_s) or minimum_outward_speed_cm_s <= 0
            or not np.isfinite(pause_max_s) or pause_max_s <= 0):
        raise ValueError("departure motion thresholds must be positive")
    count = len(centers)
    initial = (centers[start, observed[start]] if origin_points is None
               else np.asarray(origin_points, float))
    if len(initial) < 3 or not np.isfinite(initial).all():
        return None
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
    sub_valid = valid[start:]
    sub_pos = positions[start:]
    start_positions = np.full((len(leaving), 2), np.nan, dtype=float)
    for slot in range(len(leaving)):
        valid_slots = np.flatnonzero(sub_valid[:, slot])
        if len(valid_slots) > 0:
            start_positions[slot] = sub_pos[valid_slots[0], slot]

    start_ok = np.isfinite(start_positions).all(axis=1)
    diff = sub_pos - start_positions[None, :, :]
    ret_mat = np.where(start_ok[None, :], (diff[:, :, 0] ** 2 + diff[:, :, 1] ** 2) <= 0.0001, False)
    for slot in range(len(leaving)):
        outside_seen = False
        for offset_f in np.flatnonzero(sub_valid[:, slot]):
            px = float(sub_pos[offset_f, slot, 0])
            py = float(sub_pos[offset_f, slot, 1])
            dist = cv2.pointPolygonTest(hull, (px, py), True)
            inside = (dist > 0.25) or bool(ret_mat[offset_f, slot])
            if outside_seen and inside:
                returns[start + offset_f] = True
            outside_seen = dist <= 0.0

    active[returns] = False
    bridged = active.copy()
    bounds = runs(active)
    limit = int(np.floor(max(pause_max_s, 1.5) * fps + 1e-9))
    for (_, left), (right, _) in zip(bounds, bounds[1:]):
        if right - left <= limit and not returns[left:right].any():
            bridged[left:right] = True

    candidate_spans: list[tuple[int, int, np.ndarray]] = []
    for low, high in runs(bridged):
        if (high - low - 1) / fps <= minimum_duration_s + 1e-9:
            continue
        supports = np.flatnonzero(active[low:high]) + low
        if len(supports) / fps < MIN_MOTION_SUPPORT_SECONDS:
            continue
        candidate_spans.append((low, high, supports))
    return count, returns, valid, radius, candidate_spans


def _select_departure_for_checkpoint(
    prepared: tuple[int, np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int, np.ndarray]]] | None,
    checkpoint: int,
) -> np.ndarray:
    """Select the highest-displacement departure run before ``checkpoint``."""
    if prepared is None:
        return np.zeros(0, bool)
    count, returns, valid, radius, candidate_spans = prepared
    result = np.zeros(count, bool)
    if not candidate_spans:
        return result
    choices = []
    for low, high, supports in candidate_spans:
        before = supports[supports <= checkpoint]
        if not len(before):
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
    prepared = _prepare_departure_process(
        centers, observed, start, leaving, fps,
        minimum_outward_speed_cm_s=minimum_outward_speed_cm_s,
        pause_max_s=pause_max_s,
        velocity_cm_s=velocity_cm_s,
        minimum_duration_s=minimum_duration_s,
        member_starts=member_starts,
        origin_points=origin_points,
    )
    if prepared is None:
        return np.zeros(len(centers), bool)
    return _select_departure_for_checkpoint(prepared, checkpoint)


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


def _observed_attempt_end(
    centers: np.ndarray, observed: np.ndarray, velocity: np.ndarray,
    leaving: tuple[int, ...], end: int, origin_points: np.ndarray, fps: float,
    config: GroupHeuristicConfig,
) -> int | None:
    """Resolve an unknown bout end only with later measured contradiction.

    A gap is not cessation. After it, a return into the original hull or an
    entirely observed inactive run exceeding the existing pause allowance
    closes the attempt. The inactive support never includes unknown frames.
    """
    count = len(centers)
    if end == count or (observed[end, list(leaving)].all()
                       and np.isfinite(velocity[end, list(leaving)]).all()):
        return end
    positions = centers[end:, leaving]
    valid = observed[end:, leaving] & np.isfinite(positions).all(axis=2)
    anchor = np.median(origin_points, axis=0)
    offset = positions - anchor
    radius = np.linalg.norm(offset, axis=2)
    radial = np.divide(np.sum(offset * velocity[end:, leaving], axis=2), radius,
                       out=np.full_like(radius, np.nan), where=radius > 1e-9)
    measured = valid & np.isfinite(radial)
    inactive = measured.all(axis=1) & ~np.any(
        measured & (radial >= config.dispersal_min_outward_speed_cm_s), axis=1)
    limit = int(np.floor(config.dispersal_pause_max_s * fps + 1e-9))
    pauses = [low + end for low, high in runs(inactive) if high - low > limit]
    completion = pauses[0] if pauses else None
    hull = cv2.convexHull(origin_points.astype(np.float32))
    # Every leaving member was observed outside at the measured checkpoint.
    # Its later observed return is sufficient even if another ID stays hidden.
    for frame in range(end, count if completion is None else completion + 1):
        for slot in np.flatnonzero(valid[frame - end]):
            if cv2.pointPolygonTest(hull, tuple(map(float, positions[frame - end, slot])), False) >= 0:
                return frame
    return completion


def _compact_origin_episodes(
    f: ExportedFeatures, near: np.ndarray, neighbor_cm: float,
    *, velocity_cm_s: np.ndarray | None = None,
    config: GroupHeuristicConfig = DEFAULT_GROUP_HEURISTIC_CONFIG,
) -> list[tuple[int, int, tuple[int, ...]]]:
    """Find observed formations without resetting a continuing pile's ID clock.

    Each origin admits a late ID only at that ID's first observation after
    formation, using the original fixed envelope. A connected ancestor keeps
    that clock when more IDs appear; an ID first seen outside cannot be made
    eligible by repeatedly reseeding the same pile. A neighbor graph change
    does not end a departure: departing mice may reconnect a 10cm edge while
    still outside the origin. Only a later observed formation after the actual
    outward bout has ended may retire the old origin. The completed attempt
    need not pass the final >5s label gate, and the next pile may form elsewhere.

    Scan adjacency once and resolve each origin's first measured departure
    attempt with the existing motion/return policy. Successive origins then
    evaluate disjoint slices; neither arbitrary graph flicker nor missing rows
    can cut off the predecessor before its real confirmation.
    """
    visible = f.valid & np.isfinite(f.centers_cm).all(axis=2)
    count, tracks = visible.shape
    if not count:
        return []
    next_observed = np.minimum.accumulate(
        np.where(visible, np.arange(count)[:, None], count)[::-1], axis=0)[::-1]
    velocity = (_observed_departure_velocity(f) if velocity_cm_s is None else velocity_cm_s)
    origins: list[tuple[int, tuple[int, ...], tuple[int, ...]]] = []
    departures = {}
    attempt_ends: dict[int, int] = {}
    stops: dict[int, int] = {}
    active: set[int] = set()
    previous = np.zeros(near.shape[1], bool)
    dep_cache: dict[tuple[int, tuple[int, ...], bytes], Any] = {}
    for frame in range(count):
        edges = np.flatnonzero(near[frame])
        # A real measured departure supplies the attempt boundary. Reuse all
        # endpoint gates and motion rules; only the final duration gate is
        # deferred so a short first attempt cannot swallow a valid later one.
        for index in tuple(active):
            if index in attempt_ends:
                continue
            start, _, members = origins[index]
            centers, observed, joins, points = departures[index]
            leaving = _departing_member_slots(
                centers, observed, start, frame,
                minimum_outward_displacement_cm=4., minimum_visible_fraction=.6,
                member_starts=joins, origin_points=points)
            if len(leaving) < 2 or not _dispersal_endpoint_separated(centers[frame, list(leaving)]):
                continue
            join_mask = joins <= frame
            cache_key = (index, leaving, join_mask.tobytes())
            if cache_key not in dep_cache:
                dep_cache[cache_key] = _prepare_departure_process(
                    centers, observed, start, leaving, f.fps,
                    minimum_outward_speed_cm_s=config.dispersal_min_outward_speed_cm_s,
                    pause_max_s=config.dispersal_pause_max_s,
                    velocity_cm_s=velocity[:, members], minimum_duration_s=0.,
                    member_starts=joins, origin_points=points[join_mask])
            phase = _select_departure_for_checkpoint(dep_cache[cache_key], frame)
            if phase.any():
                end = int(np.flatnonzero(phase)[-1]) + 1
                completion = _observed_attempt_end(
                    centers, observed, velocity[:, members], leaving, end,
                    points[join_mask], f.fps, config)
                if completion is not None:
                    attempt_ends[index] = completion
        if edges.size:
            adjacency = np.zeros((tracks, tracks), bool)
            adjacency[f.pair_i[edges], f.pair_j[edges]] = True
            adjacency |= adjacency.T
            seed_candidates = _components(adjacency, visible[frame])
        else:
            seed_candidates = ()
        for seed in seed_candidates:
            if len(seed) < 2:
                continue
            seed_set = set(seed)
            ancestors = [index for index in active
                         if len(seed_set.intersection(origins[index][2])) >= 2]
            if ancestors and (len(seed) < 3 or any(
                    frame < attempt_ends.get(index, count) for index in ancestors)):
                continue  # Graph reconnection is not proof that a bout ended.
            seed_mask = np.zeros(tracks, bool)
            seed_mask[list(seed)] = True
            internal = seed_mask[f.pair_i] & seed_mask[f.pair_j]
            # A long measured pause may confirm cessation after the compact
            # reappearance edge; that group need not reconnect another edge.
            completed_here = any(frame == attempt_ends.get(index) for index in ancestors)
            if not np.any(internal & near[frame] & ~previous) and not completed_here:
                continue  # A split leaves an old subset, not a new formation.
            retired = [index for index in ancestors if set(origins[index][2]).issubset(seed_set)]
            # A regrouped subset can coexist with the remaining old members'
            # departure. It cannot cancel their independently measured process.
            for index in retired:
                stops[index] = frame
            active.difference_update(retired)
            anchor = np.median(f.centers_cm[frame, list(seed)], axis=0)
            first = next_observed[frame]
            members = set(seed)
            for slot in np.flatnonzero(first < count):
                position = f.centers_cm[first[slot], slot]
                if np.linalg.norm(position - anchor) <= neighbor_cm:
                    members.add(int(slot))
            members = tuple(sorted(members))
            centers, observed = f.centers_cm[:, members], visible[:, members]
            joins, points = _departure_origins(centers, observed, frame, neighbor_cm)
            index = len(origins)
            active.add(index)
            departures[index] = centers, observed, joins, points
            origins.append((frame, seed, members))
        previous = near[frame]

    result = []
    for index, (start, seed, members) in enumerate(origins):
        stop = stops.get(index, count)
        # A later formation cannot backfill the preceding episode's members.
        admitted = tuple(slot for slot in members if next_observed[start, slot] < stop)
        if len(admitted) >= 3:
            result.append((start, stop, admitted))
        if len(seed) >= 3 and seed != admitted:
            result.append((start, stop, seed))
    return sorted(set(result))


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
