"""Calibrate interpretable trajectory/rule evidence, never filenames or identities.

This is a supervised calibration of a weighted heuristic score, not an identity
tracker. All candidate sets and features are constructed before labels enter
the optimizer. Each behavior has a linear score with inspectable coefficients.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations
from typing import Any, Sequence

import numpy as np

from .blind_top1 import (
    Candidate, LabeledClip, Top1Prediction, candidates_from_evidence, evaluate_top1,
    hierarchical_candidate_indices,
)
from .exported_dataset import (
    CANONICAL_BEHAVIORS, INDIVIDUAL_BEHAVIORS, PAIR_BEHAVIORS,
    # [STEP6 2026-10-05] Huddling 速度门删除后不再需要 onset 静息检查
    ExportedFeatures, HeuristicParameters,
    _huddle_pair_distance_cm,
    _durations_by_column, predict_features,
)


@dataclass(frozen=True)
class CandidateTable:
    candidates: tuple[Candidate, ...]
    values: np.ndarray
    feature_names: tuple[str, ...]


def rule_profiles(seed: HeuristicParameters) -> tuple[HeuristicParameters, ...]:
    """Train-selectable, biologically bounded threshold bank.

    Required event durations remain fixed at the user's ethogram minima. The
    train-only ranker learns which speed/distance interpretation best matches
    the labeled training examples; no validation labels enter this bank.
    """
    variants = (
        seed,
        replace(seed, stationary_max_speed_cm_s=2.675, running_min_speed_cm_s=113.6875,
                together_max_distance_cm=10.7, huddle_width_multiplier=0.8,
                isolation_distance_cm=20.0625, clustering_min_nearest_neighbor_drop_cm=1.3375),
        replace(seed, stationary_max_speed_cm_s=5.35, running_min_speed_cm_s=100.3125,
                together_max_distance_cm=10.7, huddle_width_multiplier=1.0,
                isolation_distance_cm=26.75, clustering_min_nearest_neighbor_drop_cm=2.675),
        replace(seed, stationary_max_speed_cm_s=8.025, running_min_speed_cm_s=113.6875,
                together_max_distance_cm=10.7, huddle_width_multiplier=1.2,
                isolation_distance_cm=33.4375, clustering_min_nearest_neighbor_drop_cm=4.0125),
        replace(seed, stationary_max_speed_cm_s=10.7, running_min_speed_cm_s=133.75,
                together_max_distance_cm=10.7, huddle_width_multiplier=1.5,
                isolation_distance_cm=40.125, clustering_min_nearest_neighbor_drop_cm=5.35),
        replace(seed, stationary_max_speed_cm_s=4.0125, running_min_speed_cm_s=80.25,
                together_max_distance_cm=10.7, huddle_width_multiplier=1.0,
                isolation_distance_cm=46.8125, clustering_min_nearest_neighbor_drop_cm=1.3375),
        replace(seed, stationary_max_speed_cm_s=6.6875, running_min_speed_cm_s=160.5,
                together_max_distance_cm=10.7, huddle_width_multiplier=1.3,
                isolation_distance_cm=53.5, clustering_min_nearest_neighbor_drop_cm=6.6875),
        replace(seed, stationary_max_speed_cm_s=13.375, running_min_speed_cm_s=113.6875,
                together_max_distance_cm=10.7, huddle_width_multiplier=2.0,
                isolation_distance_cm=66.875, clustering_min_nearest_neighbor_drop_cm=8.025,
                clustering_min_mean_speed_cm_s=1.3375, huddle_max_mean_speed_cm_s=12),
        replace(seed, stationary_max_speed_cm_s=6.6875, running_min_speed_cm_s=113.6875,
                together_max_distance_cm=10.7, huddle_width_multiplier=3.0,
                huddle_distance_cm=13.375, clustering_max_distance_cm=16.05,
                isolation_distance_cm=33.4375, clustering_min_nearest_neighbor_drop_cm=2.675,
                clustering_min_mean_speed_cm_s=2.675, huddle_max_mean_speed_cm_s=8),
    )
    # Resting group speed cannot exceed the calibrated individual Static
    # ceiling, even in a profile that otherwise explores a looser huddle gate.
    return tuple(
        replace(
            profile,
            huddle_max_mean_speed_cm_s=min(
                profile.huddle_max_mean_speed_cm_s,
                profile.stationary_max_speed_cm_s,
            ),
        )
        for profile in dict.fromkeys(variants)
    )


def _components(edges: np.ndarray, left: np.ndarray, right: np.ndarray) -> list[tuple[int, ...]]:
    adjacency: dict[int, set[int]] = {}
    for a, b in zip(left[edges], right[edges]):
        adjacency.setdefault(int(a), set()).add(int(b))
        adjacency.setdefault(int(b), set()).add(int(a))
    result = []
    while adjacency:
        pending = [min(adjacency)]
        members: set[int] = set()
        while pending:
            node = pending.pop()
            if node in members:
                continue
            members.add(node)
            pending.extend(adjacency.pop(node, set()) - members)
        if len(members) >= 3:
            result.append(tuple(sorted(members)))
    return result


def _formation_frames(
    f: ExportedFeatures,
    members: tuple[int, ...],
    internal: np.ndarray,
    *,
    window_seconds: float = 2.0,
    minimum_drop_cm: float = 2.675,
) -> np.ndarray:
    """Return event starts with an observed group formation over two seconds.

    The returned index is the beginning of a measurable nearest-neighbour
    decrease, so the formation itself counts toward the required event
    duration. The distance gate is supplied by the frozen heuristic profile.
    """
    nearest = np.full((len(f.valid), len(members)), np.inf, dtype=np.float32)
    positions = {slot: index for index, slot in enumerate(members)}
    for edge in np.flatnonzero(internal):
        left = positions[int(f.pair_i[edge])]
        right = positions[int(f.pair_j[edge])]
        distance = f.pair_distance_cm[:, edge]
        nearest[:, left] = np.minimum(nearest[:, left], distance)
        nearest[:, right] = np.minimum(nearest[:, right], distance)
    lag = max(1, int(round(max(float(window_seconds), 0.0) * f.fps)))
    if len(nearest) <= lag:
        return np.empty(0, dtype=int)
    with np.errstate(invalid="ignore"):
        decrease = nearest[:-lag] - nearest[lag:]
    finite = np.isfinite(decrease)
    count = finite.sum(axis=1)
    mean_decrease = np.divide(
        np.where(finite, decrease, 0.0).sum(axis=1), count,
        out=np.zeros(len(count), dtype=float), where=count >= 3,
    )
    formed = np.flatnonzero(
        (count >= 3) & (mean_decrease >= max(float(minimum_drop_cm), 0.0))
    ) + lag
    return np.maximum(formed - lag, 0)


def _aggregate_group_candidates(
    f: ExportedFeatures,
    parameters: HeuristicParameters | None = None,
) -> tuple[Candidate, ...]:
    """Offer clip-level participant unions at several spatial support levels.

    These are alternative event participant sets, never inferred animal IDs.
    Components must have a real, simultaneous connected triple for the required
    duration. Sporadic isolated detections cannot create a group on their own.
    """
    parameters = parameters or HeuristicParameters()
    pair_valid = np.isfinite(f.pair_distance_cm)
    co_visible = pair_valid.sum(axis=0)
    result: dict[tuple[str, tuple[int, ...]], Candidate] = {}
    formation_cache: dict[tuple[int, ...], np.ndarray] = {}

    def remember(behavior: str, ids: tuple[int, ...], score: float) -> None:
        key = (behavior, ids)
        old = result.get(key)
        if old is None or score > old.evidence:
            result[key] = Candidate(behavior, ids, score)

    huddle_distance = _huddle_pair_distance_cm(f, parameters)
    # [STEP6 2026-10-05] Huddling 删除全部速度门（用户决策）
    # huddle_speed_limit = min(
    #     float(parameters.huddle_max_mean_speed_cm_s),
    #     float(parameters.stationary_max_speed_cm_s),
    # )
    distance_levels = tuple(sorted({
        float(parameters.clustering_max_distance_cm),
        float(parameters.group_locomotion_max_distance_cm),
        float(parameters.dispersal_start_distance_cm),
        huddle_distance,
    }))
    for distance in distance_levels:
        close = pair_valid & (f.pair_distance_cm <= distance)
        count = close.sum(axis=0)
        support = count / np.maximum(co_visible, 1)
        # One train-calibratable support level avoids multiplying every
        # proposal by several nearly duplicate edge-frequency thresholds.
        for fraction in (0.15,):
            edges = (support >= fraction) & (count >= max(2, int(0.1 * f.fps)))
            components = _components(edges, f.pair_i, f.pair_j)
            proposals = set(components)
            adjacency: dict[int, set[int]] = {}
            for edge in np.flatnonzero(edges):
                left, right = int(f.pair_i[edge]), int(f.pair_j[edge])
                adjacency.setdefault(left, set()).add(right)
                adjacency.setdefault(right, set()).add(left)
            # A single bridge animal must not force a dense local group to use
            # the entire connected component's ID union. Ego neighbourhoods
            # and common-neighbour cores are deterministic graph rules that
            # retain spatially supported subgroups without consulting labels.
            for center, neighbours in adjacency.items():
                ego = tuple(sorted({center, *neighbours}))
                if len(ego) >= 3:
                    proposals.add(ego)
            for edge in np.flatnonzero(edges):
                left, right = int(f.pair_i[edge]), int(f.pair_j[edge])
                core = tuple(sorted({left, right, *(adjacency[left] & adjacency[right])}))
                if len(core) >= 3:
                    proposals.add(core)
            one_member_prunable = tuple(proposals)
            # Tracklet unions often contain one weak bridge member. Iterative
            # weighted-degree peeling exposes each nested stable core while
            # preserving only groups supported by measured close-pair edges.
            edge_support = {
                tuple(sorted((int(f.pair_i[edge]), int(f.pair_j[edge])))): float(support[edge])
                for edge in np.flatnonzero(edges)
            }
            for component in components:
                remaining = set(component)
                while len(remaining) > 3:
                    strength = {
                        member: sum(
                            edge_support.get(tuple(sorted((member, other))), 0.0)
                            for other in remaining if other != member
                        )
                        for member in remaining
                    }
                    weakest = min(remaining, key=lambda member: (strength[member], member))
                    remaining.remove(weakest)
                    proposals.add(tuple(sorted(remaining)))
            # When two members have similar aggregate support, choosing only
            # the single weakest one is unstable. Offer each one-member-pruned
            # alternative for moderate groups; the later frame-level duration
            # check still rejects unsupported combinations.
            for proposal in one_member_prunable:
                if 4 <= len(proposal) <= 16:
                    for removed in proposal:
                        proposals.add(tuple(member for member in proposal if member != removed))
            # Preserve stable three-member cores instead of only the union of
            # a changing episode. Every triple still needs real frame evidence.
            if fraction == 0.15:
                for center in range(len(f.track_ids)):
                    neighbors = set(f.pair_j[edges & (f.pair_i == center)].tolist())
                    neighbors.update(f.pair_i[edges & (f.pair_j == center)].tolist())
                    proposals.update(tuple(sorted((center, a, b))) for a, b in combinations(sorted(neighbors), 2))
            if len(proposals) > 128:
                def proposal_strength(members: tuple[int, ...]) -> tuple[float, float, int]:
                    weights = [
                        edge_support.get(tuple(sorted((left, right))), 0.0)
                        for left, right in combinations(members, 2)
                    ]
                    positive = [value for value in weights if value > 0]
                    density = len(positive) / max(len(weights), 1)
                    return density, (sum(positive) / len(positive) if positive else 0.0), -len(members)

                # Preserve representation across group sizes before filling by
                # graph strength. This bounds crowded-scene runtime without
                # allowing only tiny triples to consume the entire budget.
                ranked = sorted(proposals, key=lambda item: (proposal_strength(item), item), reverse=True)
                selected: set[tuple[int, ...]] = set(components)
                per_size: dict[int, int] = {}
                for proposal in ranked:
                    if len(selected) >= 96:
                        break
                    size = len(proposal)
                    if per_size.get(size, 0) < 8:
                        selected.add(proposal)
                        per_size[size] = per_size.get(size, 0) + 1
                for proposal in ranked:
                    if len(selected) >= 128:
                        break
                    selected.add(proposal)
                proposals = selected
            for members in sorted(proposals):
                internal = np.isin(f.pair_i, members) & np.isin(f.pair_j, members)
                # Degree >= 2 demonstrates a connected triple, unlike two
                # disjoint pairs; it does not demand every fragmented ID coexist.
                degrees = np.zeros((len(f.valid), len(members)), dtype=np.int16)
                member_slots = {slot: index for index, slot in enumerate(members)}
                for edge in np.flatnonzero(internal):
                    degrees[:, member_slots[int(f.pair_i[edge])]] += close[:, edge]
                    degrees[:, member_slots[int(f.pair_j[edge])]] += close[:, edge]
                grouped = np.max(degrees, axis=1) >= 2
                ids = tuple(sorted(int(f.track_ids[slot]) for slot in members))
                # Bridge short detection gaps only between observed group
                # frames. The duration is an episode span, with at least 60%
                # of its frames carrying actual connected-triple evidence.
                bridged = grouped.copy()
                observed = np.flatnonzero(grouped)
                max_gap = max(
                    0,
                    int(np.floor(max(float(parameters.clustering_max_gap_s), 0.0) * f.fps)),
                )
                for before, after in zip(observed[:-1], observed[1:]):
                    if 0 < after - before - 1 <= max_gap:
                        bridged[before + 1:after] = True
                duration = float(_durations_by_column(bridged[:, None], f.fps)[0])
                if duration > 0:
                    formation = formation_cache.get(members)
                    if formation is None:
                        formation = _formation_frames(
                            f,
                            members,
                            internal,
                            window_seconds=parameters.clustering_window_s,
                            minimum_drop_cm=parameters.clustering_min_nearest_neighbor_drop_cm,
                        )
                        formation_cache[members] = formation
                    # The annotated event starts with mice moving toward the
                    # group. Requiring five *additional* seconds of connected
                    # group evidence after that onset systematically drops
                    # valid five-second formation episodes. Count duration
                    # from formation onset through the connected-group end,
                    # while still requiring a majority of frames to show an
                    # observed connected triple.
                    padded = np.r_[False, bridged, False].astype(np.int8)
                    changes = np.diff(padded)
                    episode_starts = np.flatnonzero(changes == 1)
                    episode_ends = np.flatnonzero(changes == -1) - 1
                    if episode_starts.size:
                        episode_index = int(np.argmax(episode_ends - episode_starts + 1))
                        group_start = int(episode_starts[episode_index])
                        group_end = int(episode_ends[episode_index])
                        lag = max(1, int(round(parameters.clustering_window_s * f.fps)))
                        for event_start in formation:
                            event_start = int(event_start)
                            if event_start > group_end:
                                continue
                            formation_end = min(event_start + lag, len(grouped) - 1)
                            if formation_end < group_start - max_gap:
                                continue
                            episode_start = min(event_start, group_start)
                            episode_duration = (group_end - episode_start + 1) / max(f.fps, 1e-6)
                            group_fraction = float(
                                np.mean(grouped[episode_start : group_end + 1])
                            )
                            if (
                                episode_duration + 1e-12 < parameters.clustering_min_duration_s
                                or group_fraction < 0.6
                            ):
                                continue
                            # Speed belongs to the formation phase, not only
                            # the already-assembled group. Pool visible member
                            # observations so temporary ID gaps do not make a
                            # moving approach look stationary.
                            formation_slice = slice(event_start, formation_end + 1)
                            formation_speed = f.speed_cm_s[formation_slice][:, members]
                            formation_valid = (
                                f.valid[formation_slice][:, members]
                                & np.isfinite(formation_speed)
                            )
                            mean_formation_speed = (
                                float(formation_speed[formation_valid].mean())
                                if formation_valid.any()
                                else 0.0
                            )
                            member_observations = formation_valid.sum(axis=0)
                            member_mean_speed = np.divide(
                                np.where(formation_valid, formation_speed, 0.0).sum(axis=0),
                                member_observations,
                                out=np.zeros(len(members), dtype=float),
                                where=member_observations > 0,
                            )
                            moving_fraction = float(np.mean(member_mean_speed > 1e-9))
                            if (
                                mean_formation_speed >= parameters.clustering_min_mean_speed_cm_s
                                and moving_fraction + 1e-12
                                >= parameters.clustering_min_moving_member_fraction
                            ):
                                remember("social_clustering", ids, episode_duration * 1.25)
                                break
                if distance <= huddle_distance + 1e-9:
                    # Huddling is a stable, low-motion state. Exclude brief
                    # extra IDs from the candidate participant set by keeping
                    # a stable core, but also offer an observed-tracklet union
                    # when members disappear for long detector gaps. A sparse
                    # member must be spatially connected while it *is* seen;
                    # low global visibility alone must not delete its ID.
                    observed_group_frames = np.flatnonzero(grouped)
                    stable_positions = [
                        position
                        for position in range(len(members))
                        if observed_group_frames.size
                        and np.mean(degrees[observed_group_frames, position] >= 1) >= 0.75
                    ]
                    stable_members = tuple(members[position] for position in stable_positions)
                    if len(stable_members) >= 3:
                        stable_degrees = degrees[:, stable_positions]
                        stable_grouped = np.max(stable_degrees, axis=1) >= 2
                        stable_duration = float(
                            _durations_by_column(stable_grouped[:, None], f.fps)[0]
                        )
                        # # [STEP6 2026-10-05] Huddling 删除全部速度门（用户决策）
                        if stable_duration >= parameters.huddle_min_duration_s:
                            stable_ids = tuple(sorted(int(f.track_ids[slot]) for slot in stable_members))
                            remember("huddle", stable_ids, stable_duration)
                    if (
                        len(members) > len(stable_members)
                        and duration >= parameters.huddle_min_duration_s
                    ):  # # [STEP6 2026-10-05] Huddling 删除全部速度门（用户决策）
                        seen = f.valid[observed_group_frames][:, members]
                        connected = degrees[observed_group_frames] >= 1
                        connection_rate = np.divide(
                            (connected & seen).sum(axis=0), seen.sum(axis=0),
                            out=np.zeros(len(members), dtype=float), where=seen.sum(axis=0) > 0,
                        )
                        if np.all(connection_rate >= 0.15):
                            remember("huddle", ids, duration * 0.75)
                if distance <= parameters.group_locomotion_max_distance_cm + 1e-9:
                    member_speed = f.speed_cm_s[:, members]
                    member_visible = f.valid[:, members]
                    # [STEP4 2026-10-04] 个体层合并：Walking+Running -> Locomotion（用户决策/文档）
                    moving_member = (
                        (member_speed > parameters.stationary_max_speed_cm_s)
                        & np.isfinite(member_speed)
                        & member_visible
                    )
                    moving_fraction = np.divide(
                        moving_member.sum(axis=1),
                        np.maximum(member_visible.sum(axis=1), 1),
                    )
                    internal_directions = f.pair_direction_similarity[:, internal]
                    internal_valid = np.isfinite(f.pair_distance_cm[:, internal])
                    direction_mean = np.divide(
                        np.where(internal_valid, internal_directions, 0.0).sum(axis=1),
                        np.maximum(internal_valid.sum(axis=1), 1),
                    )
                    locomotion = (
                        grouped
                        & (member_visible.sum(axis=1) >= 3)
                        & (moving_fraction >= parameters.group_locomotion_min_moving_fraction)
                        & (direction_mean >= parameters.group_locomotion_min_direction_similarity)
                    )
                    locomotion_duration = float(
                        _durations_by_column(locomotion[:, None], f.fps)[0]
                    )
                    if locomotion_duration >= parameters.group_locomotion_min_duration_s:
                        remember("group_locomotion", ids, locomotion_duration)
                if distance <= parameters.dispersal_start_distance_cm + 1e-9:
                    # A dispersal candidate must show a compact three-mouse
                    # state first, then a sustained increase in within-group
                    # nearest-neighbour distance. A clip already dispersed at
                    # its first observation is not enough evidence.
                    compact = pair_valid & (f.pair_distance_cm <= parameters.dispersal_start_distance_cm)
                    compact_degree = np.zeros((len(f.valid), len(members)), dtype=np.int16)
                    for edge in np.flatnonzero(internal):
                        compact_degree[:, member_slots[int(f.pair_i[edge])]] += compact[:, edge]
                        compact_degree[:, member_slots[int(f.pair_j[edge])]] += compact[:, edge]
                    compact_frames = np.max(compact_degree, axis=1) >= 2
                    compact_start = np.flatnonzero(compact_frames)
                    if compact_start.size:
                        start = int(compact_start[0])
                        group_nearest = np.full((len(f.valid), len(members)), np.inf, dtype=float)
                        for edge in np.flatnonzero(internal):
                            left = member_slots[int(f.pair_i[edge])]
                            right = member_slots[int(f.pair_j[edge])]
                            group_nearest[:, left] = np.minimum(group_nearest[:, left], f.pair_distance_cm[:, edge])
                            group_nearest[:, right] = np.minimum(group_nearest[:, right], f.pair_distance_cm[:, edge])
                        finite_nearest = np.isfinite(group_nearest)
                        group_distance = np.divide(
                            np.where(finite_nearest, group_nearest, 0.0).sum(axis=1),
                            finite_nearest.sum(axis=1),
                            out=np.full(len(group_nearest), np.inf, dtype=float),
                            where=finite_nearest.sum(axis=1) > 0,
                        )
                        initial_end = min(len(group_distance), start + max(1, int(round(f.fps))))
                        initial = group_distance[start:initial_end]
                        final_start = max(start, len(group_distance) - max(1, int(round(f.fps))))
                        final = group_distance[final_start:]
                        span = (len(group_distance) - start) / max(f.fps, 1e-6)
                        if (
                            initial.size and final.size
                            and np.isfinite(initial).any() and np.isfinite(final).any()
                            and span >= parameters.dispersal_min_duration_s
                            and float(np.nanmedian(final) - np.nanmedian(initial))
                            >= parameters.dispersal_min_distance_increase_cm
                        ):
                            remember("dispersal", ids, span)
    return tuple(result.values())


def _physical_features(f: ExportedFeatures, ids: tuple[int, ...]) -> dict[str, float]:
    slots = np.flatnonzero(np.isin(f.track_ids, ids))
    mask = f.valid[:, slots]
    enough = mask.sum(axis=1) >= min(len(slots), 3)
    frames = len(mask)
    values: dict[str, float] = {
        "bias": 1.0,
        "clip_seconds_log": float(np.log1p(frames / f.fps) / 4),
        "member_count_log": float(np.log1p(len(slots)) / 4),
        "simultaneous_fraction": float(enough.mean()),
        "member_visibility_mean": float(mask.mean()),
        "member_visibility_min": float(mask.mean(axis=0).min()),
        "member_visibility_max": float(mask.mean(axis=0).max()),
        "scene_count": float(np.mean(f.valid.sum(axis=1)) / 10),
    }

    def summarize(name: str, data: np.ndarray, valid: np.ndarray, scale: float) -> None:
        good = data[valid & np.isfinite(data)]
        quantiles = np.quantile(good, [0.1, 0.5, 0.9]) / scale if good.size else np.zeros(3)
        for suffix, number in zip(("q10", "q50", "q90"), quantiles):
            values[f"{name}_{suffix}"] = float(np.clip(number, -4, 4))

    speed = f.speed_cm_s[:, slots]
    summarize("speed", speed, mask, 30)
    summarize("nearest", f.nearest_distance_cm[:, slots], mask, 30)
    for threshold in (1, 2, 4, 8, 16, 32):
        values[f"speed_le_{threshold}"] = float(((speed <= threshold) & mask).sum() / max(mask.sum(), 1))
    scene_speed = f.speed_cm_s[f.valid]
    values["scene_speed"] = float(np.clip(np.mean(scene_speed) / 30, 0, 4)) if scene_speed.size else 0.0
    values["relative_speed"] = values["speed_q50"] - values["scene_speed"]
    displacements = []
    for slot in slots:
        points = f.centers_cm[f.valid[:, slot], slot]
        if len(points) > 1:
            displacements.append(float(np.linalg.norm(points[-1] - points[0])))
    values["net_displacement"] = float(np.clip(np.mean(displacements) / 50, 0, 4)) if displacements else 0.0

    internal = np.isin(f.pair_i, slots) & np.isin(f.pair_j, slots)
    # Single-mouse context uses distances to neighbors but no neighbor's ID.
    if len(slots) == 1:
        internal = (f.pair_i == slots[0]) | (f.pair_j == slots[0])
    distances = f.pair_distance_cm[:, internal]
    pair_valid = np.isfinite(distances)
    summarize("distance", distances, pair_valid, 30)

    # A clip-wide distance quantile loses event order: a sustained approach
    # and a brief late encounter can have similar pooled distances. Keep the
    # same trajectory-only boundary while exposing early/late state and the
    # longest uninterrupted close episode to the train-only ranker.
    segment = max(1, frames // 3)
    for name, window in (("early", distances[:segment]), ("late", distances[-segment:])):
        finite = window[np.isfinite(window)]
        values[f"distance_{name}_q50"] = float(np.median(finite) / 30) if finite.size else 0.0

    def longest_close_fraction(distance_cm: float) -> float:
        if not distances.size:
            return 0.0
        close = np.any(pair_valid & (distances <= distance_cm), axis=1)
        padded = np.r_[False, close, False].astype(np.int8)
        changes = np.diff(padded)
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1)
        return float(np.max(ends - starts) / frames) if starts.size else 0.0

    values["distance_close_5_longest_fraction"] = longest_close_fraction(5.0)
    values["distance_close_10_longest_fraction"] = longest_close_fraction(10.0)
    summarize("closing", f.pair_closing_speed_cm_s[:, internal], pair_valid, 30)
    for threshold in (3, 5, 8, 12, 20, 30):
        values[f"distance_le_{threshold}"] = float(((distances <= threshold) & pair_valid).sum() / max(pair_valid.sum(), 1))
    for name, array in (("head", f.pair_nose_head_cm), ("tail", f.pair_nose_tail_cm)):
        data = array[:, internal]
        summarize(name, data, pair_valid, 10)
        for threshold in (1, 2, 3, 5):
            values[f"{name}_le_{threshold}"] = float(((data <= threshold) & pair_valid).sum() / max(pair_valid.sum(), 1))
    direction = f.pair_direction_similarity[:, internal]
    values["direction_similarity"] = float(direction[pair_valid].mean()) if pair_valid.any() else 0.0
    width = max(1, frames // 4)
    early = distances[:width][pair_valid[:width]]
    late = distances[-width:][pair_valid[-width:]]
    values["distance_net_drop"] = float(np.clip((np.median(early) - np.median(late)) / 30, -4, 4)) if early.size and late.size else 0.0
    return values


def build_candidate_table(f: ExportedFeatures, profiles: Sequence[HeuristicParameters]) -> CandidateTable:
    """The prediction boundary: only trajectory features and frozen rules enter."""
    scores: dict[tuple[str, tuple[int, ...]], np.ndarray] = {}
    if not profiles:
        raise ValueError("at least one heuristic profile is required")
    # The temporal group graph is the expensive part on long clips with
    # fragmented tracks. Evaluate the conservative seed and the low-motion
    # edge of the train threshold bank; intermediate group states remain
    # available through the inexpensive trajectory-rule profiles below.
    group_profile_indices = tuple(sorted({0, len(profiles) - 1}))
    formed_social_sets: set[tuple[int, ...]] = set()
    for index, parameters in enumerate(profiles):
        for candidate in candidates_from_evidence(predict_features(f, parameters)):
            key = (candidate.behavior, candidate.target_ids)
            scores.setdefault(key, np.zeros(len(profiles) + 1))[index] = candidate.evidence
        if index in group_profile_indices:
            for candidate in _aggregate_group_candidates(f, parameters):
                key = (candidate.behavior, candidate.target_ids)
                vector = scores.setdefault(key, np.zeros(len(profiles) + 1))
                vector[index] = candidate.evidence
                vector[-1] = max(vector[-1], candidate.evidence)
                if candidate.behavior == "social_clustering":
                    formed_social_sets.add(candidate.target_ids)

    # A clustering candidate needs the same explicit group formation evidence
    # in at least one frozen profile; identity membership alone is not enough.
    for key in tuple(scores):
        if key[0] == "social_clustering" and key[1] not in formed_social_sets:
            scores.pop(key)

    # Thresholds express evidence strength, not whether an observed mouse or
    # pair is allowed to compete at all. Keeping every visible singleton and
    # co-visible pair prevents a slightly mistuned threshold from making the
    # correct Top-1 answer mathematically unreachable. This is exhaustive over
    # observed visual tracks only and never invents or repairs an identity.
    empty_evidence = np.zeros(len(profiles) + 1)
    for track_id in f.track_ids:
        ids = (int(track_id),)
        # # [STEP4 2026-10-04] 个体层合并：Walking+Running -> Locomotion（用户决策/文档）：理毛/站立攀爬在 Phase B 进入标注 schema 前不作为 ranker 候选。
        for behavior in INDIVIDUAL_BEHAVIORS:
            if behavior not in CANONICAL_BEHAVIORS:
                continue
            scores.setdefault((behavior, ids), empty_evidence.copy())
        scores.setdefault(("isolation", ids), empty_evidence.copy())
    for edge, (left, right) in enumerate(zip(f.pair_i, f.pair_j)):
        if not np.isfinite(f.pair_distance_cm[:, edge]).any():
            continue
        ids = tuple(sorted((int(f.track_ids[left]), int(f.track_ids[right]))))
        for behavior in PAIR_BEHAVIORS:
            scores.setdefault((behavior, ids), empty_evidence.copy())
    duration = len(f.valid) / f.fps
    maxima = {
        behavior: max(
            max((float(value.max()) for (name, _), value in scores.items() if name == behavior), default=0.0),
            1e-6,
        )
        for behavior in CANONICAL_BEHAVIORS
    }
    candidates, vectors = [], []
    physical: dict[tuple[int, ...], dict[str, float]] = {}
    names: tuple[str, ...] = ()
    for (behavior, ids), evidence in sorted(scores.items()):
        if ids not in physical:
            physical[ids] = _physical_features(f, ids)
        vector = dict(physical[ids])
        for index, value in enumerate(evidence):
            vector[f"rule_{index}_fraction"] = float(value / max(duration, 1e-6))
            vector[f"rule_{index}_relative"] = float(value / maxima[behavior])
        vector["evidence_seconds_log"] = float(np.log1p(evidence.max()) / 4)
        vector["evidence_fraction"] = float(evidence.max() / max(duration, 1e-6))
        vector["evidence_relative"] = float(evidence.max() / maxima[behavior])
        names = tuple(vector)
        candidates.append(Candidate(behavior, ids, float(evidence.max())))
        vectors.append(list(vector.values()))
    if not names:
        # Obtain the same schema without fabricating a positive candidate.
        if not len(f.track_ids):
            raise ValueError("trajectory contains no usable visual tracks")
        names = tuple(_physical_features(f, (int(f.track_ids[0]),))) + tuple(
            name for index in range(len(profiles) + 1)
            for name in (f"rule_{index}_fraction", f"rule_{index}_relative")
        ) + ("evidence_seconds_log", "evidence_fraction", "evidence_relative")
    return CandidateTable(tuple(candidates), np.asarray(vectors, dtype=np.float32).reshape((-1, len(names))), names)


def predict_table(table: CandidateTable, model: dict[str, Any]) -> Top1Prediction:
    if tuple(model["feature_names"]) != table.feature_names:
        raise ValueError("candidate feature schema differs from the frozen model")
    if not table.candidates:
        return Top1Prediction(None, (), 0.0)
    weights = np.asarray([model["coefficients"][c.behavior] for c in table.candidates])
    scores = np.einsum("ij,ij->i", table.values, weights)
    eligible = hierarchical_candidate_indices(table.candidates)
    index = max(eligible, key=lambda candidate_index: scores[candidate_index])
    candidate = table.candidates[index]
    return Top1Prediction(candidate.behavior, candidate.target_ids, float(scores[index]))


def average_ranker_models(models: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Average compatible train-only rankers for session-bagged inference."""

    if not models:
        raise ValueError("at least one ranker model is required")
    names = tuple(models[0].get("feature_names", ()))
    behaviors = tuple(CANONICAL_BEHAVIORS)
    if not names:
        raise ValueError("ranker models must declare feature_names")
    for model in models:
        if tuple(model.get("feature_names", ())) != names:
            raise ValueError("ranker models have incompatible feature schemas")
        if set(model.get("coefficients", {})) != set(behaviors):
            raise ValueError("ranker models have incompatible behavior coefficients")
    coefficients = {
        behavior: np.mean(
            np.asarray([model["coefficients"][behavior] for model in models], dtype=float),
            axis=0,
        ).tolist()
        for behavior in behaviors
    }
    result = dict(models[0])
    result["feature_names"] = list(names)
    result["coefficients"] = coefficients
    result["ensemble_size"] = len(models)
    result["ensemble_type"] = "session_mean"
    return result


def metric_rank(metrics: dict[str, Any]) -> tuple[float, ...]:
    per_class = [v["joint_top1_accuracy"] for v in metrics["per_behavior"].values()]
    return (min(per_class, default=0.0), metrics["macro_joint_top1_accuracy"], metrics["joint_top1_accuracy"])


def fit_ranker(tables: Sequence[CandidateTable], labels: Sequence[LabeledClip], *,
               steps: int = 500, regularization: float = 0.01, learning_rate: float = 0.03,
               seed: int = 42, objective: str = "hard_margin") -> tuple[dict[str, Any], dict[str, Any]]:
    """Class-balanced optimization of heuristic scoring coefficients.

    ``hard_margin`` focuses each update on the strongest incorrect candidate,
    which directly matches the exact Top-1 acceptance metric.  ``listwise`` is
    retained as an auditable compatibility objective.  Impossible exact labels
    are counted as errors, never inserted as candidates.  No validation data or
    file IO is accepted by this function.
    """
    if objective not in {"hard_margin", "listwise"}:
        raise ValueError("objective must be 'hard_margin' or 'listwise'")
    if (not tables or len(tables) != len(labels) or steps < 1 or regularization < 0
            or learning_rate <= 0):
        raise ValueError("invalid calibration inputs")
    names = tables[0].feature_names
    if any(table.feature_names != names for table in tables):
        raise ValueError("inconsistent training feature schema")
    behaviors = tuple(CANONICAL_BEHAVIORS)
    n, d = len(tables), len(names)
    class_index = {name: index for index, name in enumerate(behaviors)}
    sizes = np.asarray([len(table.candidates) for table in tables])
    offsets = np.cumsum(np.r_[0, sizes])
    x = np.concatenate([table.values for table in tables])
    row = np.repeat(np.arange(n), sizes)
    kind = np.asarray([class_index[c.behavior] for table in tables for c in table.candidates], dtype=int)
    gold = np.full(n, -1, dtype=int)
    eligible_by_clip: list[np.ndarray] = []
    allowed = np.zeros(len(x), dtype=bool)
    for index, (table, label) in enumerate(zip(tables, labels)):
        eligible = np.asarray(hierarchical_candidate_indices(table.candidates), dtype=int)
        eligible_by_clip.append(eligible)
        allowed[offsets[index] + eligible] = True
        for j in eligible:
            c = table.candidates[int(j)]
            if (c.behavior == label.behavior
                    and set(label.target_ids).issubset(set(c.target_ids))):
                gold[index] = offsets[index] + j
                break
    counts = {name: sum(label.behavior == name for label in labels) for name in behaviors}
    importance = np.asarray([1 / max(counts[label.behavior], 1) for label in labels])
    importance[gold < 0] = 0
    importance /= max(float(importance.sum()), 1e-12)
    indices = [np.flatnonzero(kind == k) for k in range(len(behaviors))]
    # A fixed seed makes reruns deterministic without any file-name dependence.
    weights = np.random.default_rng(seed).normal(0, 0.001, (len(behaviors), d))
    moment = np.zeros_like(weights)
    variance = np.zeros_like(weights)
    best_metrics: dict[str, Any] | None = None
    best_model: dict[str, Any] = {}
    for step in range(1, steps + 1):
        gradient = np.zeros_like(weights)
        scores = np.einsum("ij,ij->i", x, weights[kind])
        if objective == "listwise":
            scores = np.where(allowed, scores, -np.inf)
            max_scores = np.full(n, -np.inf)
            np.maximum.at(max_scores, row, scores)
            exponent = np.where(allowed, np.exp(np.clip(scores - max_scores[row], -60, 0)), 0.0)
            denominator = np.bincount(row, weights=exponent, minlength=n)
            residual = np.divide(exponent, denominator[row], out=np.zeros_like(exponent), where=denominator[row] > 0)
            residual[gold[gold >= 0]] -= 1
            residual *= importance[row]
            for k, positions in enumerate(indices):
                gradient[k] = x[positions].T @ residual[positions]
        else:
            # Structured hinge update: compare the exact gold candidate with
            # the strongest available non-gold candidate in each clip.  A
            # missing gold candidate contributes no fabricated positive signal.
            negative = np.full(n, -1, dtype=int)
            negative_scores = np.full(n, -np.inf)
            gold_scores = np.full(n, -np.inf)
            for clip_index in range(n):
                if gold[clip_index] < 0:
                    continue
                start = int(offsets[clip_index])
                local_indices = eligible_by_clip[clip_index]
                local_scores = scores[start + local_indices].copy()
                gold_local = int(gold[clip_index] - start)
                gold_position = int(np.flatnonzero(local_indices == gold_local)[0])
                gold_scores[clip_index] = local_scores[gold_position]
                local_scores[gold_position] = -np.inf
                if np.isfinite(local_scores).any():
                    local_negative = int(np.argmax(local_scores))
                    negative[clip_index] = start + int(local_indices[local_negative])
                    negative_scores[clip_index] = local_scores[local_negative]
            margin = np.zeros(n, dtype=float)
            finite_pairs = np.isfinite(negative_scores) & np.isfinite(gold_scores)
            margin[finite_pairs] = 1.0 + negative_scores[finite_pairs] - gold_scores[finite_pairs]
            active = (gold >= 0) & (negative >= 0) & finite_pairs & (margin > 0)
            for clip_index in np.flatnonzero(active):
                positive = int(gold[clip_index])
                hard_negative = int(negative[clip_index])
                gradient[kind[hard_negative]] += importance[clip_index] * x[hard_negative]
                gradient[kind[positive]] -= importance[clip_index] * x[positive]
        penalty = weights.copy()
        penalty[:, 0] = 0  # Class intercepts do not encode physical thresholds.
        gradient += regularization * penalty
        moment = 0.9 * moment + 0.1 * gradient
        variance = 0.999 * variance + 0.001 * gradient * gradient
        weights -= learning_rate * (moment / (1 - 0.9 ** step)) / (np.sqrt(variance / (1 - 0.999 ** step)) + 1e-8)
        if step == 1 or step % 25 == 0 or step == steps:
            model = {"feature_names": list(names),
                     "coefficients": {name: weights[k].tolist() for k, name in enumerate(behaviors)},
                     "regularization": regularization, "optimizer_step": step,
                     "learning_rate": learning_rate, "objective": objective, "seed": seed}
            metrics = evaluate_top1([(label, predict_table(table, model)) for label, table in zip(labels, tables)])
            if best_metrics is None or metric_rank(metrics) > metric_rank(best_metrics):
                best_metrics, best_model = metrics, model
    assert best_metrics is not None
    return best_model, best_metrics
