"""Group-episode scoring and collective evidence helpers."""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

import numpy as np

from mouse_behavior.behavior.collective_state import (
    CollectiveRuleConfig,
    classify_collective_windows,
    collective_candidate_windows,
)
from mouse_behavior.evaluation.exported_dataset_models import (
    ExportedFeatures,
    HeuristicParameters,
    _cached_collective_windows,
)
from mouse_behavior.evaluation.exported_dataset_records import (
    collective_track_quality_failures,
    collective_track_quality_metrics,
)
from mouse_behavior.evaluation.exported_dataset_temporal import _bridge_short_false_gaps

def _neighbor_counts(
    pair_mask: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    track_count: int,
) -> np.ndarray:
    """Count qualifying neighbours per frame and identity."""

    counts = np.zeros((pair_mask.shape[0], int(track_count)), dtype=np.int16)
    for pair_index, (left, right) in enumerate(zip(pair_i, pair_j)):
        active = np.asarray(pair_mask[:, pair_index], dtype=np.int16)
        counts[:, int(left)] += active
        counts[:, int(right)] += active
    return counts


def _group_resting_at_onset(
    grouped: np.ndarray,
    speed_cm_s: np.ndarray,
    valid: np.ndarray,
    member_slots: Sequence[int],
    *,
    fps: float,
    minimum_seconds: float,
    speed_limit_cm_s: float,
    minimum_group_fraction: float = 0.2,
    minimum_low_motion_fraction: float = 0.8,
) -> bool:
    """Return whether the first sustained close-group window is resting."""
    onset = np.flatnonzero(grouped)
    required_frames = max(1, int(np.ceil(max(float(minimum_seconds), 0.0) * fps)))
    if onset.size == 0:
        return False
    start = int(onset[0])
    stop = start + required_frames
    if stop > len(grouped) or float(np.mean(grouped[start:stop])) < minimum_group_fraction:
        return False

    slots = tuple(int(slot) for slot in member_slots)
    observed = valid[start:stop][:, slots]
    speeds = speed_cm_s[start:stop][:, slots]
    usable = observed & np.isfinite(speeds)
    if not usable.any():
        return False
    onset_speeds = speeds[usable]
    return (
        float(np.mean(onset_speeds <= speed_limit_cm_s)) >= minimum_low_motion_fraction
        and float(np.mean(onset_speeds)) <= speed_limit_cm_s
    )


def _windowed_body_speed(features: ExportedFeatures, window_s: float) -> np.ndarray:
    """Estimate group locomotion from sustained body-position change.

    Entire windows must be observed under the same visual track ID. Missing
    frames stay unknown rather than becoming zero-speed or inferred positions.
    """

    if window_s <= 0.0:
        return features.speed_cm_s
    positions = np.asarray(features.centers_cm, dtype=float)
    frames, tracks = positions.shape[:2]
    if frames < 2:
        return np.full((frames, tracks), np.nan, dtype=np.float32)
    total = max(2, int(round(window_s * features.fps)))
    before = total // 2
    after = total - before
    indices = np.arange(frames)
    starts = np.maximum(indices - before, 0)
    ends = np.minimum(indices + after, frames - 1)
    observed = features.valid & np.isfinite(positions).all(axis=2)
    missing_prefix = np.concatenate(
        (np.zeros((1, tracks), dtype=np.int32), np.cumsum(~observed, axis=0)),
        axis=0,
    )
    uninterrupted = missing_prefix[ends + 1] == missing_prefix[starts]
    duration_frames = ends - starts
    displacement = np.linalg.norm(positions[ends] - positions[starts], axis=2)
    speed = np.divide(
        displacement * float(features.fps),
        duration_frames[:, None],
        out=np.full((frames, tracks), np.nan, dtype=float),
        where=(duration_frames[:, None] > 0) & uninterrupted,
    )
    return speed.astype(np.float32)


def _group_episode_scores(
    pair_mask: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    track_ids: Sequence[int],
    *,
    fps: float,
    minimum_seconds: float,
    nearest_drop_cm: np.ndarray | None = None,
    min_nearest_drop_cm: float = 0.0,
    max_gap_seconds: float = 0.0,
    score_multiplier: float = 1.0,
    min_member_support_fraction: float = 0.0,
    formation_evidence: np.ndarray | None = None,
    min_formation_member_fraction: float = 0.0,
    motion_speed_cm_s: np.ndarray | None = None,
    minimum_motion_speed_cm_s: float = 0.0,
    minimum_moving_member_fraction: float = 0.0,
    emit_core_candidates: bool = False,
    event_behavior: str | None = None,
    event_output: list[dict[str, Any]] | None = None,
) -> dict[str, float]:
    """Build temporal spatial groups without merging unrelated components.

    A component can change membership across frames. The optional core mode
    tests observed member sets across an episode; a brief visitor is not
    granted the whole episode's duration. Visual IDs are never remapped.
    """

    active: list[dict[str, Any]] = []
    scores: dict[str, float] = {}
    minimum_frames = max(1, int(np.ceil(max(minimum_seconds, 0.0) * fps)))
    max_gap_frames = max(0, int(np.floor(max(float(max_gap_seconds), 0.0) * fps)))

    def finish(episode: Mapping[str, Any]) -> None:
        if episode["frames"] < minimum_frames:
            return
        if nearest_drop_cm is not None and episode["max_drop"] < min_nearest_drop_cm:
            return
        observed_frames = max(int(episode.get("observed_frames", 0)), 1)
        support_fraction = float(np.clip(min_member_support_fraction, 0.0, 1.0))
        if emit_core_candidates:
            member_frames = dict(episode.get("member_frames", {}))
            thresholds = {support_fraction, 0.0, 0.1, 0.25, 0.5, 0.75}
            candidate_slots = {
                frozenset(
                    slot
                    for slot, count in member_frames.items()
                    if count / observed_frames + 1e-12 >= threshold
                )
                for threshold in thresholds
            }
            candidate_slots.update(
                component for _, component in episode.get("observations", ())
            )
            first_frame = int(episode["start_frame"])
            last_frame = int(episode["last_observed_frame"])
            for slots in candidate_slots:
                if len(slots) < 3:
                    continue
                # A real member either persists in the lineage or disappears
                # and is observed again under the *same* ID. A contiguous
                # visitor cannot borrow the duration of the stable core.
                member_support = []
                for slot in slots:
                    seen_at = [
                        frame_index for frame_index, component in episode["observations"]
                        if slot in component
                    ]
                    fraction = len(seen_at) / observed_frames
                    reacquired = (
                        len(seen_at) >= 2
                        and seen_at[-1] - seen_at[0] >= max(2, int(round(fps * 0.5)))
                    )
                    if fraction < 0.15 and not reacquired:
                        break
                    member_support.append(fraction)
                if len(member_support) != len(slots):
                    continue
                if formation_evidence is not None:
                    formation_members = set(episode.get("formation_members", set()))
                    if (
                        len(slots.intersection(formation_members)) / len(slots) + 1e-12
                        < float(min_formation_member_fraction)
                    ):
                        continue
                observed = np.zeros(last_frame - first_frame + 1, dtype=bool)
                minimum_visible = max(3, int(np.ceil(len(slots) * 0.6)))
                for frame_index, component in episode["observations"]:
                    if len(slots.intersection(component)) >= minimum_visible:
                        observed[frame_index - first_frame] = True
                if np.count_nonzero(observed) < max(2, int(np.ceil(minimum_frames * 0.2))):
                    continue
                confirmed = _bridge_short_false_gaps(observed, max_gap_frames)
                best_start = best_stop = run_start = -1
                for frame_index, active_frame in enumerate(confirmed):
                    if active_frame and run_start < 0:
                        run_start = frame_index
                    if run_start >= 0 and (
                        not active_frame or frame_index == len(confirmed) - 1
                    ):
                        run_stop = frame_index if active_frame else frame_index - 1
                        if run_stop - run_start > best_stop - best_start:
                            best_start, best_stop = run_start, run_stop
                        run_start = -1
                duration_frames = best_stop - best_start + 1
                if duration_frames < minimum_frames:
                    continue
                start = first_frame + best_start
                stop = first_frame + best_stop
                if nearest_drop_cm is not None:
                    candidate_drop = nearest_drop_cm[start : stop + 1, sorted(slots)]
                    finite_drop = candidate_drop[np.isfinite(candidate_drop)]
                    if not finite_drop.size or np.max(finite_drop) < min_nearest_drop_cm:
                        continue
                ids = tuple(sorted(int(track_ids[slot]) for slot in slots))
                key = ",".join(map(str, ids))
                observed_fraction = float(np.mean(observed[best_start : best_stop + 1]))
                score = (
                    duration_frames
                    / fps
                    * max(float(score_multiplier), 1e-6)
                    * (0.75 + 0.25 * observed_fraction)
                    * (1.0 + 0.05 * max(len(slots) - 3, 0) * min(member_support))
                )
                scores[key] = max(scores.get(key, 0.0), score)
                if event_output is not None and event_behavior is not None:
                    event_output.append(
                        {
                            "behavior": event_behavior,
                            "member_ids": ids,
                            "start_frame": start,
                            "end_frame": stop,
                            "score": score,
                            "observed_fraction": observed_fraction,
                        }
                    )
            return
        members = tuple(
            sorted(
                int(track_ids[slot])
                for slot, count in episode.get("member_frames", {}).items()
                if count / observed_frames >= support_fraction
            )
        )
        if len(members) < 3:
            return
        if formation_evidence is not None:
            member_slots = {
                int(slot)
                for slot, count in episode.get("member_frames", {}).items()
                if count / observed_frames >= support_fraction
            }
            formation_members = set(episode.get("formation_members", set()))
            formation_fraction = (
                len(member_slots.intersection(formation_members)) / len(member_slots)
                if member_slots
                else 0.0
            )
            if formation_fraction + 1e-12 < float(min_formation_member_fraction):
                return
        key = ",".join(str(member) for member in members)
        scores[key] = max(
            scores.get(key, 0.0),
            episode["frames"] / fps * max(float(score_multiplier), 1e-6),
        )

    for frame_index, edge_mask in enumerate(pair_mask):
        adjacency: dict[int, set[int]] = {}
        for edge_index in np.flatnonzero(edge_mask):
            left = int(pair_i[edge_index])
            right = int(pair_j[edge_index])
            adjacency.setdefault(left, set()).add(right)
            adjacency.setdefault(right, set()).add(left)
        components: list[set[int]] = []
        visited: set[int] = set()
        for start in sorted(adjacency):
            if start in visited:
                continue
            component = {start}
            pending = [start]
            visited.add(start)
            while pending:
                for neighbor in adjacency[pending.pop()]:
                    if neighbor not in visited:
                        component.add(neighbor)
                        visited.add(neighbor)
                        pending.append(neighbor)
            if len(component) < 3:
                continue
            components.append(component)
        next_active: list[dict[str, Any]] = []
        matched: set[int] = set()
        for component in components:
            if motion_speed_cm_s is not None:
                component_speeds = np.asarray(
                    motion_speed_cm_s[frame_index, sorted(component)], dtype=float
                )
                component_speeds = component_speeds[np.isfinite(component_speeds)]
                moving_speeds = component_speeds[component_speeds > 1e-9]
                moving_fraction = len(moving_speeds) / max(len(component_speeds), 1)
                if (
                    moving_speeds.size == 0
                    or moving_fraction + 1e-12 < float(minimum_moving_member_fraction)
                    or float(np.mean(moving_speeds)) + 1e-12 < float(minimum_motion_speed_cm_s)
                ):
                    continue
            matches = [
                (len(component & episode["last"]), index)
                for index, episode in enumerate(active)
                if index not in matched
            ]
            overlap, index = max(matches, default=(0, -1))
            required_overlap = 2
            if index >= 0:
                smaller_size = min(len(component), len(active[index]["last"]))
                required_overlap = (
                    max(3, int(np.ceil(smaller_size * 0.6)))
                    if emit_core_candidates
                    else max(2, smaller_size // 2)
                )
            if index >= 0 and overlap >= required_overlap:
                episode = active[index]
                matched.add(index)
                episode["members"].update(component)
                episode["last"] = component
                episode["frames"] += 1
                episode["gap_frames"] = 0
                if emit_core_candidates:
                    episode["last_observed_frame"] = frame_index
                    episode["observations"].append((frame_index, frozenset(component)))
            else:
                episode = {
                    "members": set(component),
                    "last": component,
                    "frames": 1,
                    "max_drop": 0.0,
                    "gap_frames": 0,
                    "observed_frames": 0,
                    "member_frames": {},
                    "formation_members": set(),
                }
                if emit_core_candidates:
                    episode["start_frame"] = frame_index
                    episode["last_observed_frame"] = frame_index
                    episode["observations"] = [(frame_index, frozenset(component))]
            episode["observed_frames"] += 1
            for slot in component:
                episode["member_frames"][slot] = episode["member_frames"].get(slot, 0) + 1
                if formation_evidence is not None and bool(formation_evidence[frame_index, slot]):
                    episode["formation_members"].add(slot)
            if nearest_drop_cm is not None:
                episode["max_drop"] = max(
                    episode["max_drop"],
                    float(np.max(nearest_drop_cm[frame_index, sorted(component)])),
                )
            next_active.append(episode)
        for index, episode in enumerate(active):
            if index not in matched:
                if int(episode.get("gap_frames", 0)) < max_gap_frames:
                    episode["gap_frames"] = int(episode.get("gap_frames", 0)) + 1
                    episode["frames"] += 1
                    next_active.append(episode)
                else:
                    finish(episode)
        active = next_active
    for episode in active:
        finish(episode)
    return scores


def _rolling_drop(values: np.ndarray, frames: int) -> np.ndarray:
    result = np.zeros_like(values, dtype=float)
    frames = max(int(frames), 1)
    if len(values) <= frames:
        return result
    with np.errstate(invalid="ignore"):
        delta = values[:-frames] - values[frames:]
    finite = np.isfinite(delta)
    result[frames:][finite] = delta[finite]
    return result


def _dispersal_scores(
    features: ExportedFeatures,
    parameters: HeuristicParameters,
    moving: np.ndarray,
) -> dict[str, float]:
    """Detect separation only after an observed close group was present."""

    duration = max(1, int(np.ceil(parameters.dispersal_min_duration_s * features.fps)))
    frames = len(features.valid)
    if frames < duration or len(features.track_ids) < 3:
        return {}
    stride = max(1, int(round(features.fps)))
    result: dict[str, float] = {}
    for start in range(0, frames - duration + 1, stride):
        adjacency: dict[int, set[int]] = {}
        close = (
            np.isfinite(features.pair_distance_cm[start])
            & (features.pair_distance_cm[start] <= parameters.dispersal_start_distance_cm)
        )
        for edge in np.flatnonzero(close):
            left, right = int(features.pair_i[edge]), int(features.pair_j[edge])
            adjacency.setdefault(left, set()).add(right)
            adjacency.setdefault(right, set()).add(left)
        seen: set[int] = set()
        for initial in adjacency:
            if initial in seen:
                continue
            members = {initial}
            pending = [initial]
            seen.add(initial)
            while pending:
                for neighbor in adjacency[pending.pop()]:
                    if neighbor not in seen:
                        seen.add(neighbor)
                        members.add(neighbor)
                        pending.append(neighbor)
            if len(members) < 3:
                continue
            slots = tuple(sorted(members))
            internal = np.isin(features.pair_i, slots) & np.isin(features.pair_j, slots)
            pair_values = features.pair_distance_cm[:, internal]
            if pair_values.shape[1] < 2:
                continue
            checkpoints = np.linspace(start, start + duration - 1, 5, dtype=int)
            nearest_values: list[float] = []
            for frame in checkpoints:
                nearest = np.full(len(slots), np.inf)
                position = {slot: index for index, slot in enumerate(slots)}
                for edge in np.flatnonzero(internal):
                    distance = features.pair_distance_cm[frame, edge]
                    if np.isfinite(distance):
                        left = position[int(features.pair_i[edge])]
                        right = position[int(features.pair_j[edge])]
                        nearest[left] = min(nearest[left], float(distance))
                        nearest[right] = min(nearest[right], float(distance))
                finite = nearest[np.isfinite(nearest)]
                if len(finite) < 3:
                    break
                nearest_values.append(float(np.mean(finite)))
            if len(nearest_values) != 5:
                continue
            changes = np.diff(nearest_values)
            increase = nearest_values[-1] - nearest_values[0]
            moving_fraction = float(np.mean(moving[start : start + duration, slots]))
            if (
                increase >= parameters.dispersal_min_distance_increase_cm
                and np.sum(changes >= -1.0) >= 3
                and moving_fraction >= 0.2
            ):
                key = ",".join(str(int(features.track_ids[slot])) for slot in slots)
                result[key] = max(result.get(key, 0.0), duration / features.fps)
    return result


def _recent_contact_mask(contact: np.ndarray, post_frames: int) -> np.ndarray:
    """Mark observed contact and a short interval after each contact frame."""

    recent = np.asarray(contact, dtype=bool).copy()
    for offset in range(1, min(max(post_frames, 0), len(recent) - 1) + 1):
        recent[offset:] |= contact[:-offset]
    return recent


def _finite_quantile(values: np.ndarray, fraction: float, *, minimum: int = 3) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.quantile(finite, fraction)) if len(finite) >= minimum else float("nan")


def _directional_walker_in_group(
    features: ExportedFeatures,
    member_ids: Sequence[int],
    parameters: HeuristicParameters,
    *,
    start_frame: int = 0,
    end_frame: int | None = None,
) -> bool:
    """Require one observed member to walk toward the eventual group location."""

    if parameters.clustering_directional_walk_min_seconds <= 0.0:
        return True
    positions = {int(track_id): slot for slot, track_id in enumerate(features.track_ids)}
    slots = [positions[track_id] for track_id in member_ids if track_id in positions]
    if len(slots) < 3:
        return False
    start_frame = max(0, int(start_frame))
    end_frame = min(
        len(features.valid) - 1,
        int(end_frame) if end_frame is not None else len(features.valid) - 1,
    )
    frame_count = end_frame - start_frame + 1
    if frame_count < 3:
        return False
    last_fraction = features.centers_cm[
        start_frame + max(0, int(frame_count * 0.8)) : end_frame + 1, slots
    ]
    endpoints = last_fraction.reshape(-1, 2)
    endpoints = endpoints[np.isfinite(endpoints).all(axis=1)]
    if len(endpoints) < 3:
        return False
    destination = np.median(endpoints, axis=0)
    all_motion_valid = features.motion_valid if features.motion_valid is not None else features.valid
    motion_valid = all_motion_valid[start_frame : end_frame + 1]
    for slot in slots:
        points = features.centers_cm[start_frame : end_frame + 1, slot]
        to_group = destination - points
        distance = np.linalg.norm(to_group, axis=1)
        velocity = features.velocity_cm_s[start_frame : end_frame + 1, slot]
        speed = np.linalg.norm(velocity, axis=1)
        denominator = speed * distance
        direction = np.divide(
            np.sum(velocity * to_group, axis=1),
            denominator,
            out=np.zeros(frame_count, dtype=float),
            where=denominator > 1e-6,
        )
        # [STEP4 2026-10-04] 个体层合并：Walking+Running -> Locomotion（用户决策/文档）：方向性接近门改用合并后的移动掩码。
        approaching_move = (
            motion_valid[:, slot]
            & (speed > parameters.stationary_max_speed_cm_s)
            & (direction >= parameters.clustering_directional_min_cosine)
        )
        if np.count_nonzero(approaching_move) / features.fps < (
            parameters.clustering_directional_walk_min_seconds
        ):
            continue
        observed_distances = distance[np.isfinite(points).all(axis=1)]
        if len(observed_distances) < 3:
            continue
        boundary = max(1, len(observed_distances) // 5)
        toward_drop = float(
            np.median(observed_distances[:boundary])
            - np.median(observed_distances[-boundary:])
        )
        if toward_drop >= parameters.clustering_directional_min_distance_drop_cm:
            return True
    return False


def _group_formation_geometry_support(
    features: ExportedFeatures,
    member_ids: Sequence[int],
    *,
    start_frame: int,
    end_frame: int,
) -> float:
    """Combine body translation, spatial spread and overlap without head jitter."""

    slots_by_id = {int(track_id): index for index, track_id in enumerate(features.track_ids)}
    slots = [slots_by_id[int(track_id)] for track_id in member_ids if int(track_id) in slots_by_id]
    if len(slots) < 3:
        return 0.0
    window = slice(max(start_frame, 0), min(end_frame + 1, len(features.valid)))
    centers = np.asarray(features.centers_cm[window][:, slots], dtype=float)
    visible = np.isfinite(centers).all(axis=2) & features.valid[window][:, slots]
    displacements = []
    for slot in range(len(slots)):
        points = centers[visible[:, slot], slot]
        if len(points) >= 2:
            displacements.append(float(np.linalg.norm(points[-1] - points[0])))
    if not displacements:
        return 0.0
    radii = []
    for points, observed in zip(centers, visible):
        if np.count_nonzero(observed) >= 3:
            group_points = points[observed]
            center = np.mean(group_points, axis=0)
            radii.append(float(np.mean(np.linalg.norm(group_points - center, axis=1))))
    if not radii:
        return 0.0
    radius = float(np.median(radii))
    overlap_fraction = 0.0
    if features.pair_box_iou is not None:
        internal_edges = np.isin(features.pair_i, slots) & np.isin(features.pair_j, slots)
        overlap = np.asarray(features.pair_box_iou[window][:, internal_edges], dtype=float)
        finite = overlap[np.isfinite(overlap)]
        if finite.size:
            overlap_fraction = float(np.mean(finite > 0.0))
    body_length_cm = float(features.reference_body_cm)
    motion = min(max(displacements) / (1.5 * body_length_cm), 1.0)
    spread = min(radius / body_length_cm, 1.0)
    return float(0.6 * motion + 0.2 * spread + 0.2 * (1.0 - overlap_fraction))


def _collective_rule_config(parameters: HeuristicParameters) -> CollectiveRuleConfig:
    """Map the frozen two-group parameters to the shared CLU/HUD feature rule."""

    return CollectiveRuleConfig(
        proximity_distance_bw=parameters.collective_proximity_distance_bw,
        contact_score_threshold=parameters.collective_contact_score_threshold,
        active_speed_cm_s=parameters.collective_active_speed_cm_s,
        low_motion_speed_cm_s=parameters.collective_low_motion_speed_cm_s,
        social_min_duration_s=max(parameters.clustering_min_duration_s, 5.0),
        social_nnd_drop_bw_min=parameters.social_nnd_drop_bw_min,
        social_nnd_slope_bw_s_max=parameters.social_nnd_slope_bw_s_max,
        social_radius_drop_bw_min=parameters.social_radius_drop_bw_min,
        social_radius_slope_bw_s_max=parameters.social_radius_slope_bw_s_max,
        social_inward_fraction_min=parameters.social_inward_fraction_min,
        social_active_fraction_min=parameters.social_active_fraction_min,
        social_active_inward_member_fraction_min=(
            parameters.social_active_inward_member_fraction_min
        ),
        social_edge_gain_rate_min=parameters.social_edge_gain_rate_min,
        social_density_slope_min=parameters.social_density_slope_min,
        social_soft_score_min=parameters.social_soft_score_min,
        huddle_min_duration_s=max(parameters.huddle_min_duration_s, 1.0),
        huddle_min_group_visible_fraction=(
            parameters.collective_huddle_min_group_visible_fraction
        ),
        huddle_min_member_support_fraction=(
            parameters.collective_huddle_min_member_support_fraction
        ),
        huddle_recovery_near_group_max_bw=(
            parameters.collective_huddle_recovery_near_group_max_bw
        ),
        huddle_nnd_bw_max=parameters.huddle_nnd_bw_max,
        huddle_radius_bw_max=parameters.huddle_radius_bw_max,
        huddle_bbox_iom_min=parameters.huddle_bbox_iom_min,
        huddle_nnd_slope_abs_max=parameters.huddle_nnd_slope_abs_max,
        huddle_radius_slope_abs_max=parameters.huddle_radius_slope_abs_max,
        huddle_low_motion_fraction_min=parameters.huddle_low_motion_fraction_min,
        huddle_contact_density_min=parameters.huddle_contact_density_min,
        huddle_proximity_density_min=parameters.huddle_proximity_density_min,
        huddle_graph_stability_min=parameters.huddle_graph_stability_min,
        huddle_soft_score_min=parameters.huddle_soft_score_min,
        huddle_attack_speed_p90_max_cm_s=parameters.huddle_attack_speed_p90_max_cm_s,
        huddle_attack_relative_motion_max_cm_s=parameters.huddle_attack_relative_motion_max_cm_s,
    )


def _replace_collective_group_scores_with_features(
    features: ExportedFeatures,
    parameters: HeuristicParameters,
    *,
    huddle_groups: dict[str, float],
    cluster_groups: dict[str, float],
    group_events: list[dict[str, Any]],
) -> tuple[dict[str, float], dict[str, float], list[dict[str, Any]], list[dict[str, Any]]]:
    """Score exact, observed group candidates with the shared CLU/HUD rules."""

    centers = features.collective_centers_cm
    valid = features.collective_valid
    velocity = features.collective_velocity_cm_s
    speed = features.collective_speed_cm_s
    if centers is None or valid is None or velocity is None or speed is None:
        centers = features.centers_cm
        valid = features.valid
        velocity = features.velocity_cm_s
        speed = features.speed_cm_s
    centers = np.asarray(centers, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    velocity = np.asarray(velocity, dtype=float)
    speed = np.asarray(speed, dtype=float)
    tracks = len(features.track_ids)
    lengths = (
        np.asarray(features.body_length_ref_cm, dtype=float)
        if features.body_length_ref_cm is not None
        else np.full(tracks, float(features.reference_body_cm), dtype=float)
    )
    widths = (
        np.asarray(features.body_width_ref_cm, dtype=float)
        if features.body_width_ref_cm is not None
        else np.full(tracks, max(features.mouse_width_cm, 1.0), dtype=float)
    )
    config = _collective_rule_config(parameters)
    track_quality = collective_track_quality_metrics(features)
    quality_by_slot = {
        slot: {
            **quality,
            "failed_conditions": collective_track_quality_failures(quality, parameters),
        }
        for slot, quality in enumerate(track_quality)
    }
    eligible_slots = {
        slot
        for slot, quality in quality_by_slot.items()
        if not quality["failed_conditions"]
    }

    pair_distance = np.full_like(features.pair_distance_cm, np.inf, dtype=float)
    for pair_index, (left, right) in enumerate(zip(features.pair_i, features.pair_j)):
        if int(left) not in eligible_slots or int(right) not in eligible_slots:
            continue
        points_left = centers[:, int(left)]
        points_right = centers[:, int(right)]
        usable = valid[:, int(left)] & valid[:, int(right)]
        distance = np.linalg.norm(points_left - points_right, axis=1)
        pair_distance[usable, pair_index] = distance[usable]
    broad_group_pairs = np.isfinite(pair_distance) & (
        pair_distance < max(parameters.clustering_max_distance_cm, 1.0)
    )
    broad_candidates = _group_episode_scores(
        broad_group_pairs,
        features.pair_i,
        features.pair_j,
        features.track_ids.tolist(),
        fps=features.fps,
        minimum_seconds=max(min(parameters.huddle_min_duration_s, parameters.clustering_min_duration_s), 1.0),
        max_gap_seconds=max(parameters.huddle_max_gap_s, parameters.clustering_max_gap_s),
        min_member_support_fraction=max(
            min(parameters.huddle_min_member_support_fraction, 0.25), 0.15
        ),
        emit_core_candidates=True,
    )
    candidate_keys = set(huddle_groups) | set(cluster_groups) | set(broad_candidates)
    new_huddle: dict[str, float] = {}
    new_cluster: dict[str, float] = {}
    debug_rows: list[dict[str, Any]] = []
    track_slot = {int(track_id): slot for slot, track_id in enumerate(features.track_ids)}
    pair_contact_cache: dict[tuple[int, int, int], Any] = {}
    track_quality_rows = [
        {
            **quality,
            "eligible_for_collective_graph": not quality["failed_conditions"],
        }
        for quality in quality_by_slot.values()
    ]
    for key in sorted(candidate_keys):
        try:
            member_ids = tuple(int(value) for value in key.split(",") if value)
            member_slots = tuple(track_slot[track_id] for track_id in member_ids)
        except (KeyError, ValueError):
            continue
        if len(set(member_slots)) < config.min_members:
            continue
        if not set(member_slots).issubset(eligible_slots):
            continue
        cache_key = (
            id(features),
            member_ids,
            config.proximity_distance_bw,
            config.contact_score_threshold,
            config.active_speed_cm_s,
            config.low_motion_speed_cm_s,
            config.social_min_duration_s,
            config.huddle_min_duration_s,
        )
        windows = _cached_collective_windows(
            cache_key,
            features,
            lambda: collective_candidate_windows(
                centers_cm=centers,
                valid=valid,
                speed_cm_s=speed,
                velocity_cm_s=velocity,
                member_slots=member_slots,
                fps=features.fps,
                body_length_ref_cm=lengths,
                body_width_ref_cm=widths,
                config=config,
                keypoints_cm=features.collective_keypoints_cm,
                boxes_cm=features.collective_boxes_cm,
                pair_contact_cache=pair_contact_cache,
            ),
        )
        decision = classify_collective_windows(windows, config)
        if decision.social_clustering_score > 0.0:
            new_cluster[key] = decision.social_clustering_score
        if decision.huddling_score > 0.0:
            new_huddle[key] = decision.huddling_score
        for state, score, window, checks in (
            (
                "SOCIAL_CLUSTERING",
                decision.social_clustering_score,
                decision.social_window,
                decision.social_checks,
            ),
            ("HUDDLING", decision.huddling_score, decision.huddle_window, decision.huddle_checks),
        ):
            if window is None:
                continue
            values = window.debug_values()
            values.update(
                {
                    "group_id": key,
                    "member_ids": key,
                    "frame": window.end_frame,
                    "time_s": window.end_frame / max(features.fps, 1e-9),
                    "social_clustering_score": decision.social_clustering_score,
                    "huddling_score": decision.huddling_score,
                    "state": state,
                    "primary_behavior": (
                        "huddle"
                        if decision.huddling_score >= decision.social_clustering_score
                        and decision.huddling_score > 0.0
                        else "social_clustering"
                        if decision.social_clustering_score > 0.0
                        else ""
                    ),
                    "checks": json.dumps(checks, ensure_ascii=False, sort_keys=True),
                }
            )
            debug_rows.append(values)
        if decision.huddling_score > 0.0 and decision.huddle_window is not None:
            window = decision.huddle_window
            group_events.append(
                {
                    "behavior": "huddle",
                    "member_ids": member_ids,
                    "start_frame": window.start_frame,
                    "end_frame": window.end_frame,
                    "score": decision.huddling_score,
                    "collective_state": "HUDDLING",
                }
            )
        if decision.social_clustering_score > 0.0 and decision.social_window is not None:
            window = decision.social_window
            group_events.append(
                {
                    "behavior": "social_clustering",
                    "member_ids": member_ids,
                    "start_frame": window.start_frame,
                    "end_frame": window.end_frame,
                    "score": decision.social_clustering_score,
                    "collective_state": "SOCIAL_CLUSTERING",
                }
            )
    return new_huddle, new_cluster, debug_rows, track_quality_rows
