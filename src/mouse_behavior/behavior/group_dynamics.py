"""Frame-level signals for coordinated multi-mouse behavior.

This module consumes already identified trajectories.  It never creates or
repairs identities; group membership is expressed with the existing behavior
array slots and is translated to RFID labels by the output layer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def _components(
    ids: np.ndarray,
    distances: np.ndarray,
    threshold_cm: float,
) -> list[tuple[int, ...]]:
    """Return connected components using a nearest-neighbour distance gate."""

    if len(ids) == 0:
        return []
    adjacency = np.asarray(distances < float(threshold_cm), dtype=bool)
    np.fill_diagonal(adjacency, False)
    unseen = set(range(len(ids)))
    components: list[tuple[int, ...]] = []
    while unseen:
        seed = unseen.pop()
        stack = [seed]
        local: list[int] = []
        while stack:
            current = stack.pop()
            local.append(current)
            neighbours = [item for item in unseen if bool(adjacency[current, item])]
            for item in neighbours:
                unseen.remove(item)
                stack.append(item)
        components.append(tuple(sorted(int(ids[index]) for index in local)))
    return components


def _member_distances(
    centers: np.ndarray,
    frame: int,
    members: Sequence[int],
) -> np.ndarray:
    indices = np.asarray(tuple(members), dtype=int)
    if len(indices) < 2:
        return np.empty((len(indices), len(indices)), dtype=float)
    points = centers[frame, indices]
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    distances[~np.isfinite(distances)] = np.inf
    np.fill_diagonal(distances, np.inf)
    return distances


def _mean_nearest_distance(distances: np.ndarray) -> float:
    if distances.shape[0] < 2:
        return float("inf")
    nearest = np.min(distances, axis=1)
    finite = nearest[np.isfinite(nearest)]
    return float(np.mean(finite)) if finite.size else float("inf")


def _largest_component(
    ids: np.ndarray,
    distances: np.ndarray,
    threshold_cm: float,
    minimum_size: int,
) -> tuple[int, ...]:
    candidates = [
        component
        for component in _components(ids, distances, threshold_cm)
        if len(component) >= int(minimum_size)
    ]
    return max(
        candidates, key=lambda values: (len(values), tuple(-item for item in values)), default=()
    )


def _overlap_is_stable(left: Sequence[int], right: Sequence[int], minimum: int = 2) -> bool:
    left_set = set(int(value) for value in left)
    right_set = set(int(value) for value in right)
    required = min(max(int(minimum), 1), len(left_set), len(right_set))
    return bool(required) and len(left_set.intersection(right_set)) >= required


def group_dynamic_signals(
    kin: Mapping[str, Any],
    *,
    group_config: Mapping[str, Any],
    analysis_fps: float,
) -> dict[str, Any]:
    """Compute causal masks and membership traces for three group behaviors."""

    valid = np.asarray(kin["valid"], dtype=bool)
    centers = np.asarray(kin["centers_cm"], dtype=float)
    velocity = np.asarray(kin.get("velocity", np.zeros_like(centers)), dtype=float)
    behavior_speed = np.asarray(
        kin.get("behavior_speed", np.linalg.norm(velocity, axis=2)),
        dtype=float,
    )
    if valid.ndim != 2:
        raise ValueError("kin['valid'] must be a two-dimensional frame-by-mouse array")
    frames, mice = valid.shape
    expected_centers = (frames, mice, 2)
    if centers.shape != expected_centers or velocity.shape != expected_centers:
        raise ValueError(
            f"centers_cm and velocity must have shape {expected_centers}, "
            f"got {centers.shape} and {velocity.shape}"
        )
    if behavior_speed.shape != (frames, mice):
        raise ValueError(
            f"behavior_speed must have shape {(frames, mice)}, got {behavior_speed.shape}"
        )

    fps = max(float(analysis_fps), 1e-9)
    minimum_group_size = max(int(group_config.get("dynamic_min_group_size", 3)), 3)
    organized_distance = max(
        float(group_config.get("dynamic_max_neighbor_distance_cm", 30.0)),
        0.0,
    )

    locomotion_cfg = dict(group_config.get("group_locomotion", {}))
    locomotion_distance = max(
        float(locomotion_cfg.get("max_neighbor_distance_cm", organized_distance)),
        0.0,
    )
    locomotion_min_direction = float(
        np.clip(locomotion_cfg.get("min_direction_similarity", 0.70), 0.0, 1.0)
    )

    clustering_cfg = dict(group_config.get("social_clustering", {}))
    clustering_distance = max(
        float(clustering_cfg.get("max_neighbor_distance_cm", organized_distance)),
        0.0,
    )
    formation_window = max(
        int(round(float(clustering_cfg.get("formation_window_seconds", 2.0)) * fps)),
        1,
    )
    minimum_cluster_drop = max(
        float(clustering_cfg.get("min_nearest_neighbor_drop_cm", 2.0)),
        0.0,
    )
    minimum_clustering_speed = max(
        float(clustering_cfg.get("min_mean_speed_cm_s", 5.0)),
        0.0,
    )
    minimum_moving_member_fraction = float(
        np.clip(clustering_cfg.get("min_moving_member_fraction", 0.5), 0.0, 1.0)
    )
    dispersal_cfg = dict(group_config.get("dispersal", {}))
    prior_cluster_distance = max(
        float(dispersal_cfg.get("prior_cluster_neighbor_distance_cm", 15.0)),
        0.0,
    )
    prior_cluster_frames = max(
        int(round(float(dispersal_cfg.get("prior_cluster_min_duration_seconds", 1.0)) * fps)),
        1,
    )
    minimum_dispersal_increase = max(
        float(dispersal_cfg.get("min_nearest_neighbor_increase_cm", 5.0)),
        0.0,
    )

    names = ("group_locomotion", "social_clustering", "dispersal")
    masks = {name: np.zeros(frames, dtype=bool) for name in names}
    scores = {name: np.zeros(frames, dtype=float) for name in names}
    members = {name: [() for _ in range(frames)] for name in names}
    mean_nearest = np.full(frames, np.nan, dtype=float)

    organized_members: list[tuple[int, ...]] = [() for _ in range(frames)]
    organized_mean_nearest = np.full(frames, np.inf, dtype=float)
    frame_ids: list[np.ndarray] = []
    frame_distances: list[np.ndarray] = []

    for frame in range(frames):
        ids = np.flatnonzero(valid[frame] & np.all(np.isfinite(centers[frame]), axis=1))
        distances = _member_distances(centers, frame, ids)
        frame_ids.append(ids)
        frame_distances.append(distances)
        if len(ids) < minimum_group_size:
            continue

        organized = _largest_component(
            ids,
            distances,
            clustering_distance,
            minimum_group_size,
        )
        if organized:
            organized_members[frame] = organized
            organized_mean_nearest[frame] = _mean_nearest_distance(
                _member_distances(centers, frame, organized)
            )
            mean_nearest[frame] = organized_mean_nearest[frame]

        locomotion_components = [
            component
            for component in _components(ids, distances, locomotion_distance)
            if len(component) >= minimum_group_size
        ]
        best_locomotion: tuple[float, tuple[int, ...]] | None = None
        for component in locomotion_components:
            index = np.asarray(component, dtype=int)
            component_speeds = behavior_speed[frame, index]
            # Use movement direction only when measurable. No absolute cm/s
            # floor is imposed; zero-speed stop frames can be bridged by the
            # event-duration layer and do not invalidate the group episode.
            moving = np.isfinite(component_speeds) & (component_speeds > 1e-9)
            moving_fraction = float(np.mean(moving))
            if not np.any(moving):
                continue
            vectors = velocity[frame, index[moving]]
            norms = np.linalg.norm(vectors, axis=1)
            usable = np.isfinite(norms) & (norms > 1e-9)
            if not np.any(usable):
                continue
            directions = vectors[usable] / norms[usable, None]
            direction_similarity = float(np.linalg.norm(np.mean(directions, axis=0)))
            if direction_similarity < locomotion_min_direction:
                continue
            candidate_score = 0.5 * moving_fraction + 0.5 * direction_similarity
            candidate = (candidate_score, component)
            if best_locomotion is None or (len(component), candidate_score) > (
                len(best_locomotion[1]),
                best_locomotion[0],
            ):
                best_locomotion = candidate
        if best_locomotion is not None:
            scores["group_locomotion"][frame] = best_locomotion[0]
            masks["group_locomotion"][frame] = True
            members["group_locomotion"][frame] = best_locomotion[1]

    clustering_active = False
    clustering_members: tuple[int, ...] = ()
    clustering_seed_score = 0.0
    for frame in range(frames):
        current = organized_members[frame]
        current_mean = organized_mean_nearest[frame]
        if not current or not np.isfinite(current_mean):
            clustering_active = False
            clustering_members = ()
            clustering_seed_score = 0.0
            continue
        history = np.asarray(
            [
                organized_mean_nearest[past]
                for past in range(max(0, frame - formation_window), frame)
                if _overlap_is_stable(organized_members[past], current)
                and np.isfinite(organized_mean_nearest[past])
            ],
            dtype=float,
        )
        drop = float(np.max(history) - current_mean) if history.size else 0.0
        member_speeds = behavior_speed[frame, np.asarray(current, dtype=int)]
        member_speeds = member_speeds[np.isfinite(member_speeds)]
        moving_speeds = member_speeds[member_speeds > 1e-9]
        moving_fraction = len(moving_speeds) / max(len(member_speeds), 1)
        # Clustering is the moving formation phase, not every later frame in
        # which the same group remains together. Once the active group slows,
        # the process event ends and the resting Huddle gate can take over.
        active_motion = (
            moving_speeds.size > 0
            and moving_fraction + 1e-12 >= minimum_moving_member_fraction
            and float(np.mean(moving_speeds)) + 1e-12 >= minimum_clustering_speed
        )
        seed = drop >= minimum_cluster_drop and active_motion
        if clustering_active and active_motion and _overlap_is_stable(clustering_members, current):
            clustering_members = current
        elif seed:
            clustering_active = True
            clustering_members = current
            clustering_seed_score = float(np.clip(drop / max(minimum_cluster_drop, 1e-6), 0.0, 1.0))
        else:
            clustering_active = False
            clustering_members = ()
            clustering_seed_score = 0.0
        if clustering_active:
            masks["social_clustering"][frame] = True
            members["social_clustering"][frame] = clustering_members
            cohesion = 1.0 - np.clip(current_mean / max(clustering_distance, 1e-6), 0.0, 1.0)
            scores["social_clustering"][frame] = max(clustering_seed_score, float(cohesion))

    anchor_members: tuple[int, ...] = ()
    anchor_baseline = float("inf")
    anchor_run = 0
    dispersal_active = False
    for frame in range(frames):
        ids = frame_ids[frame]
        distances = frame_distances[frame]
        if dispersal_active:
            visible_anchor = tuple(
                member for member in anchor_members if member in set(ids.tolist())
            )
            if len(visible_anchor) < minimum_group_size:
                dispersal_active = False
                anchor_members = ()
                anchor_baseline = float("inf")
                anchor_run = 0
                continue
            anchor_distances = _member_distances(centers, frame, visible_anchor)
            current_mean = _mean_nearest_distance(anchor_distances)
            remaining_cluster = _largest_component(
                np.asarray(visible_anchor, dtype=int),
                anchor_distances,
                prior_cluster_distance,
                minimum_group_size,
            )
            increase = current_mean - anchor_baseline
            dispersal_active = not remaining_cluster and increase >= minimum_dispersal_increase
            if dispersal_active:
                masks["dispersal"][frame] = True
                members["dispersal"][frame] = visible_anchor
                scores["dispersal"][frame] = float(
                    np.clip(increase / max(minimum_dispersal_increase, 1e-6), 0.0, 1.0)
                )
                continue

        current_cluster = _largest_component(
            ids,
            distances,
            prior_cluster_distance,
            minimum_group_size,
        )
        if current_cluster:
            current_baseline = _mean_nearest_distance(
                _member_distances(centers, frame, current_cluster)
            )
            if anchor_members and _overlap_is_stable(anchor_members, current_cluster):
                anchor_run += 1
                anchor_members = current_cluster
                anchor_baseline = min(anchor_baseline, current_baseline)
            else:
                anchor_members = current_cluster
                anchor_baseline = current_baseline
                anchor_run = 1
            continue

        if anchor_run < prior_cluster_frames or len(anchor_members) < minimum_group_size:
            continue
        visible_anchor = tuple(member for member in anchor_members if member in set(ids.tolist()))
        if len(visible_anchor) < minimum_group_size:
            continue
        current_mean = _mean_nearest_distance(_member_distances(centers, frame, visible_anchor))
        increase = current_mean - anchor_baseline
        if increase < minimum_dispersal_increase:
            continue
        dispersal_active = True
        masks["dispersal"][frame] = True
        members["dispersal"][frame] = visible_anchor
        scores["dispersal"][frame] = float(
            np.clip(increase / max(minimum_dispersal_increase, 1e-6), 0.0, 1.0)
        )

    return {
        "masks": masks,
        "scores": scores,
        "members_by_frame": members,
        "mean_nearest_neighbor_cm": mean_nearest,
    }


__all__ = ["group_dynamic_signals"]
