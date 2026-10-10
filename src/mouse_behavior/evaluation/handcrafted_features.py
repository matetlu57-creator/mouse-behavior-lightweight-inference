"""Auditable, behavior-specific summaries of exported visual trajectories.

These are measurements, not ground-truth labels or stable animal identities.
The centimetre values use the exported adapter's resolved reference body length
(arena calibration by default);
they are estimates until the camera is physically calibrated.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .exported_dataset import ExportedFeatures


def _median(values: np.ndarray) -> float | None:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.median(finite)) if finite.size else None


def _mean(values: np.ndarray) -> float | None:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.mean(finite)) if finite.size else None


def _quantile(values: np.ndarray, fraction: float) -> float | None:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.quantile(finite, fraction)) if finite.size else None


def _end_change(values: np.ndarray) -> float | None:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size < 2:
        return None
    width = max(1, int(np.ceil(finite.size * 0.2)))
    return float(np.median(finite[-width:]) - np.median(finite[:width]))


def _row_mean(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    count = finite.sum(axis=1)
    return np.divide(
        np.where(finite, values, 0.0).sum(axis=1),
        count,
        out=np.full(len(values), np.nan, dtype=float),
        where=count > 0,
    )


def _displacement(points: np.ndarray) -> tuple[float | None, float | None]:
    usable = np.asarray(points, dtype=float)
    usable = usable[np.isfinite(usable).all(axis=1)]
    if len(usable) < 2:
        return None, None
    steps = np.linalg.norm(np.diff(usable, axis=0), axis=1)
    net = float(np.linalg.norm(usable[-1] - usable[0]))
    return net, float(net / max(np.sum(steps), 1e-6))


def _longest_missing_seconds(observed: np.ndarray, fps: float) -> float:
    best = current = 0
    for present in observed:
        current = 0 if present else current + 1
        best = max(best, current)
    return best / fps


def _individual(f: ExportedFeatures, behavior: str, slot: int) -> dict[str, float | None]:
    visible = f.valid[:, slot]
    motion = visible if f.motion_valid is None else f.motion_valid[:, slot]
    speeds = np.where(motion, f.speed_cm_s[:, slot], np.nan)
    nearest = np.where(visible, f.nearest_distance_cm[:, slot], np.nan)
    displacement, straightness = _displacement(f.centers_cm[visible, slot])
    presence = float(np.mean(visible))
    confidence = (
        _median(f.detection_confidence[:, slot])
        if f.detection_confidence is not None
        else None
    )
    common = {
        "track_presence_fraction": presence,
        "median_detection_confidence": confidence,
        "longest_missing_s": _longest_missing_seconds(visible, f.fps),
    }
    # [STEP4 2026-10-04] 个体层合并：Walking+Running -> Locomotion（用户决策/文档）：原 running/walking 特征合并为 locomotion 联合特征。
    if behavior in {"locomotion", "grooming", "rearing"}:
        return {
            **common,
            "speed_p25_cm_s": _quantile(speeds, 0.25),
            "speed_p90_cm_s": _quantile(speeds, 0.90),
            "active_speed_median_cm_s": _median(speeds[speeds > 0]),
            "stop_fraction": float(np.mean(speeds[motion] == 0)) if motion.any() else None,
            "net_displacement_cm": displacement,
            "straightness": straightness,
        }
    if behavior == "stationary":
        points = f.centers_cm[visible, slot]
        center = np.median(points, axis=0) if len(points) else np.asarray([np.nan, np.nan])
        radius = np.linalg.norm(points - center, axis=1)
        return {
            **common,
            "center_speed_p90_cm_s": _quantile(speeds, 0.90),
            "center_radius_p90_cm": _quantile(radius, 0.90),
            "net_displacement_cm": displacement,
        }
    if behavior == "isolation":
        return {
            **common,
            "nearest_p10_cm": _quantile(nearest, 0.10),
            "nearest_median_cm": _median(nearest),
            "nearest_end_change_cm": _end_change(nearest),
            "net_displacement_cm": displacement,
        }
    raise ValueError(f"unsupported individual behavior: {behavior}")


def _pair(f: ExportedFeatures, behavior: str, left: int, right: int) -> dict[str, float | None]:
    indices = np.flatnonzero(
        ((f.pair_i == left) & (f.pair_j == right))
        | ((f.pair_i == right) & (f.pair_j == left))
    )
    if len(indices) != 1:
        raise ValueError("candidate pair does not have exactly one trajectory relation")
    pair = int(indices[0])
    observed = np.isfinite(f.pair_distance_cm[:, pair])
    distance = np.where(observed, f.pair_distance_cm[:, pair], np.nan)
    closing = np.where(observed, f.pair_closing_speed_cm_s[:, pair], np.nan)
    head = np.where(observed, f.pair_nose_head_cm[:, pair], np.nan)
    rear = np.where(observed, f.pair_nose_tail_cm[:, pair], np.nan)
    iou = f.pair_box_iou[:, pair] if f.pair_box_iou is not None else np.asarray([])
    gap = f.pair_box_gap_cm[:, pair] if f.pair_box_gap_cm is not None else np.asarray([])
    left_speed = np.where(f.valid[:, left], f.speed_cm_s[:, left], np.nan)
    right_speed = np.where(f.valid[:, right], f.speed_cm_s[:, right], np.nan)
    delta = f.centers_cm[:, right] - f.centers_cm[:, left]
    length = np.linalg.norm(delta, axis=1)
    toward = np.divide(
        delta,
        length[:, None],
        out=np.full_like(delta, np.nan, dtype=float),
        where=length[:, None] > 1e-6,
    )
    left_velocity = np.sum(f.velocity_cm_s[:, left] * toward, axis=1)
    right_velocity = np.sum(f.velocity_cm_s[:, right] * -toward, axis=1)
    heading_left = heading_right = None
    if f.heading_unit is not None:
        heading_left = _mean(np.sum(f.heading_unit[:, left] * toward, axis=1))
        heading_right = _mean(np.sum(f.heading_unit[:, right] * -toward, axis=1))
    facing = [value for value in (heading_left, heading_right) if value is not None]
    actor_facing = max(facing) if facing else None
    actor_toward = max(
        value for value in (_mean(left_velocity), _mean(right_velocity)) if value is not None
    ) if (np.isfinite(left_velocity).any() or np.isfinite(right_velocity).any()) else None
    common = {
        "co_visible_fraction": float(np.mean(observed)),
        "distance_median_cm": _median(distance),
        "distance_end_change_cm": _end_change(distance),
        "distance_median_body_lengths": (
            _median(distance) / max(float(f.reference_body_cm), 1e-6)
            if _median(distance) is not None
            else None
        ),
    }
    if behavior == "approach":
        return {
            **common,
            "closing_speed_p90_cm_s": _quantile(closing, 0.90),
            "final_distance_cm": _quantile(distance[-max(1, len(distance) // 5) :], 0.50),
            "actor_speed_max_cm_s": _quantile(np.maximum(left_speed, right_speed), 0.90),
            "actor_facing_target_best": actor_facing,
            "actor_motion_toward_target_best_cm_s": actor_toward,
            "box_iou_max": _quantile(iou, 1.0),
        }
    if behavior == "together":
        finite = distance[np.isfinite(distance)]
        return {
            **common,
            "distance_std_cm": float(np.std(finite)) if finite.size else None,
            "combined_speed_median_cm_s": _median(left_speed + right_speed),
            "box_gap_median_cm": _median(gap),
            "head_contact_s": float(np.sum(head < 2.0) / f.fps),
            "rear_contact_s": float(np.sum(rear < 2.0) / f.fps),
        }
    if behavior in {"chase"}:  # following suspended step3
        return {
            **common,
            "direction_similarity_median": _median(f.pair_direction_similarity[observed, pair]),
            "left_speed_median_cm_s": _median(left_speed),
            "right_speed_median_cm_s": _median(right_speed),
            "box_overlap_fraction": (
                float(np.mean(iou[np.isfinite(iou)] > 0)) if np.isfinite(iou).any() else None
            ),
            "follower_toward_leader_best_cm_s": actor_toward,
        }
    # [SUSPENDED-STEP3 2026-10-04] Following/Avoiding 暂停（用户决策），保留待恢复
    # if behavior == "avoidance":
    #     finite = distance[np.isfinite(distance)]
    #     middle = len(finite) // 2
    #     return {
    #         **common,
    #         "early_distance_change_cm": _end_change(finite[:middle]),
    #         "late_distance_change_cm": _end_change(finite[middle:]),
    #         "opening_speed_p90_cm_s": _quantile(-closing, 0.90),
    #     }
    if behavior == "attack":
        contact = observed & ((head < 2.0) | (rear < 2.0))
        recent_contact = contact.copy()
        for offset in range(1, max(1, int(np.ceil(0.2 * f.fps))) + 1):
            recent_contact[offset:] |= contact[:-offset]
        point_speed = point_acceleration = None
        if f.nose_speed_cm_s is not None and f.nose_acceleration_cm_s2 is not None:
            point_speed = np.fmax(
                f.nose_speed_cm_s[:, left], f.nose_speed_cm_s[:, right]
            )
            point_acceleration = np.fmax(
                f.nose_acceleration_cm_s2[:, left],
                f.nose_acceleration_cm_s2[:, right],
            )
        return {
            **common,
            "box_iou_max": _quantile(iou, 1.0),
            "box_overlap_fraction": (
                float(np.mean(iou[np.isfinite(iou)] > 0)) if np.isfinite(iou).any() else None
            ),
            "left_longest_missing_s": _longest_missing_seconds(f.valid[:, left], f.fps),
            "right_longest_missing_s": _longest_missing_seconds(f.valid[:, right], f.fps),
            "peak_speed_cm_s": _quantile(np.maximum(left_speed, right_speed), 1.0),
            "contact_nose_speed_p90_cm_s": (
                _quantile(point_speed[recent_contact], 0.90)
                if point_speed is not None
                else None
            ),
            "contact_nose_acceleration_p90_cm_s2": (
                _quantile(point_acceleration[recent_contact], 0.90)
                if point_acceleration is not None
                else None
            ),
            "contact_body_speed_p90_cm_s": _quantile(
                np.maximum(left_speed, right_speed)[recent_contact], 0.90
            ),
        }
    if behavior == "nose_head_contact":
        left_head = (
            f.pair_nose_head_left_to_right_cm[:, pair]
            if f.pair_nose_head_left_to_right_cm is not None
            else np.asarray([])
        )
        right_head = (
            f.pair_nose_head_right_to_left_cm[:, pair]
            if f.pair_nose_head_right_to_left_cm is not None
            else np.asarray([])
        )
        return {
            **common,
            "minimum_nose_head_cm": _quantile(head, 0.0),
            "nose_head_under_2cm_s": float(np.sum(head < 2.0) / f.fps),
            "left_to_right_contact_s": float(np.sum(left_head < 2.0) / f.fps),
            "right_to_left_contact_s": float(np.sum(right_head < 2.0) / f.fps),
            "actor_facing_target_best": actor_facing,
            "relative_speed_median_cm_s": _median(np.abs(left_speed - right_speed)),
        }
    if behavior == "nose_tail_contact":
        left_rear = (
            f.pair_nose_tail_left_to_right_cm[:, pair]
            if f.pair_nose_tail_left_to_right_cm is not None
            else np.asarray([])
        )
        right_rear = (
            f.pair_nose_tail_right_to_left_cm[:, pair]
            if f.pair_nose_tail_right_to_left_cm is not None
            else np.asarray([])
        )
        return {
            **common,
            "minimum_nose_rear_cm": _quantile(rear, 0.0),
            "nose_rear_under_2cm_s": float(np.sum(rear < 2.0) / f.fps),
            "left_to_right_contact_s": float(np.sum(left_rear < 2.0) / f.fps),
            "right_to_left_contact_s": float(np.sum(right_rear < 2.0) / f.fps),
            "actor_facing_target_best": actor_facing,
            "relative_speed_median_cm_s": _median(np.abs(left_speed - right_speed)),
        }
    raise ValueError(f"unsupported pair behavior: {behavior}")


def _group(f: ExportedFeatures, behavior: str, slots: tuple[int, ...]) -> dict[str, float | None]:
    internal = np.isin(f.pair_i, slots) & np.isin(f.pair_j, slots)
    distances = f.pair_distance_cm[:, internal]
    distances = np.where(np.isfinite(distances), distances, np.nan)
    observed = f.valid[:, slots]
    points = f.centers_cm[:, slots]
    centroid = np.full((len(points), 2), np.nan)
    radius = np.full(len(points), np.nan)
    for frame in range(len(points)):
        if np.sum(observed[frame]) < 3:
            continue
        centroid[frame] = np.mean(points[frame, observed[frame]], axis=0)
        radius[frame] = np.mean(
            np.linalg.norm(points[frame, observed[frame]] - centroid[frame], axis=1)
        )
    net, _ = _displacement(centroid)
    speed = np.where(observed, f.speed_cm_s[:, slots], np.nan)
    common = {
        "member_count": float(len(slots)),
        "simultaneous_fraction": float(np.mean(observed.sum(axis=1) >= 3)),
        "pair_distance_median_cm": _median(distances),
        "radius_end_change_cm": _end_change(radius),
        "member_presence_min": float(np.min(observed.mean(axis=0))),
    }
    if behavior == "huddle":
        return {
            **common,
            "pair_distance_body_widths": (
                _median(distances) / f.mouse_width_cm
                if _median(distances) is not None and f.mouse_width_cm > 0
                else None
            ),
            "speed_p90_cm_s": _quantile(speed, 0.90),
            "longest_member_missing_s": max(
                _longest_missing_seconds(observed[:, slot], f.fps)
                for slot in range(len(slots))
            ),
        }
    if behavior == "social_clustering":
        return {
            **common,
            "pair_distance_end_change_cm": _end_change(_row_mean(distances)),
            "group_centroid_displacement_cm": net,
            "speed_median_cm_s": _median(speed),
        }
    if behavior == "dispersal":
        return {
            **common,
            "pair_distance_end_change_cm": _end_change(_row_mean(distances)),
            "initial_radius_cm": _quantile(radius[: max(1, len(radius) // 5)], 0.5),
            "group_centroid_displacement_cm": net,
        }
    if behavior == "group_locomotion":
        return {
            **common,
            "group_centroid_displacement_cm": net,
            "radius_std_cm": (
                float(np.std(radius[np.isfinite(radius)]))
                if np.isfinite(radius).any()
                else None
            ),
            "speed_median_cm_s": _median(speed),
        }
    raise ValueError(f"unsupported group behavior: {behavior}")


def behavior_feature_vector(
    features: ExportedFeatures, behavior: str, target_ids: tuple[int, ...]
) -> dict[str, float | None]:
    """Return one behavior-specific vector for observed visual IDs only.

    The caller may use labels for a *training audit*, but this function never
    reads an annotation. Inference calls it for label-free candidate ID sets.
    """

    from .exported_dataset import GROUP_BEHAVIORS, INDIVIDUAL_BEHAVIORS, PAIR_BEHAVIORS

    slot_by_id = {int(track_id): slot for slot, track_id in enumerate(features.track_ids)}
    ids = tuple(sorted(set(int(value) for value in target_ids)))
    if len(ids) != len(target_ids) or not set(ids).issubset(slot_by_id):
        raise ValueError("candidate IDs must be unique and observed in tracks")
    slots = tuple(slot_by_id[value] for value in ids)
    if behavior in INDIVIDUAL_BEHAVIORS or behavior == "isolation":
        if len(slots) != 1:
            raise ValueError(f"{behavior} requires one visual ID")
        return _individual(features, behavior, slots[0])
    if behavior in PAIR_BEHAVIORS:
        if len(slots) != 2:
            raise ValueError(f"{behavior} requires two visual IDs")
        return _pair(features, behavior, *slots)
    if behavior in GROUP_BEHAVIORS:
        if len(slots) < 3:
            raise ValueError(f"{behavior} requires at least three visual IDs")
        return _group(features, behavior, slots)
    raise ValueError(f"unsupported behavior: {behavior}")
