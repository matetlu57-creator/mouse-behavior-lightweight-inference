"""Shared handcrafted features for group-formation and huddling states.

The module is deliberately independent from annotation labels and identity
repair.  Callers provide already identified trajectories and candidate member
slots.  It then measures two different phenomena:

* Social clustering is a *formation process* (contraction, inward movement,
  active members, and a growing proximity graph).
* Huddling is a *compact steady state* (small body-width-normalized geometry,
  low motion, contact evidence, and a stable graph).

The previous distance-only rules remain available in their adapters as a
fallback.  This module contains no final behavior ranking and therefore cannot
change individual, pair-social, isolation, dispersal, or group-locomotion
rules.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Any, Mapping, Sequence

import numpy as np


_BODY_POINT_INDICES = (3, 4, 5, 6)  # neck, left hip, right hip, tail base
_ACTIVE_INWARD_MEMBER_FRACTION_MIN = 0.20


@dataclass(frozen=True)
class CollectiveRuleConfig:
    """Centralized thresholds for the two related collective states."""

    min_members: int = 3
    minimum_group_visible_fraction: float = 0.60
    minimum_member_support_fraction: float = 0.25
    proximity_distance_bw: float = 3.0
    contact_score_threshold: float = 0.30
    contact_iou_weight: float = 0.15
    contact_iom_weight: float = 0.30
    contact_keypoint_weight: float = 0.30
    contact_gap_weight: float = 0.25
    active_speed_cm_s: float = 4.0125
    low_motion_speed_cm_s: float = 13.375

    social_min_duration_s: float = 5.0
    social_nnd_drop_bw_min: float = 0.20
    social_nnd_slope_bw_s_max: float = -0.02
    social_radius_drop_bw_min: float = 0.20
    social_radius_slope_bw_s_max: float = -0.02
    social_inward_fraction_min: float = 0.45
    social_active_fraction_min: float = 0.35
    social_active_inward_member_fraction_min: float = 0.20
    social_edge_gain_rate_min: float = 0.02
    social_density_slope_min: float = 0.005
    social_soft_score_min: float = 0.60

    huddle_min_duration_s: float = 1.0
    huddle_min_group_visible_fraction: float = 0.30
    huddle_min_member_support_fraction: float = 0.40
    huddle_recovery_near_group_max_bw: float = 3.0
    huddle_nnd_bw_max: float = 2.50
    huddle_radius_bw_max: float = 3.00
    huddle_nnd_slope_abs_max: float = 0.15
    huddle_radius_slope_abs_max: float = 0.15
    huddle_bbox_iom_min: float = 0.14
    huddle_low_motion_fraction_min: float = 0.60
    huddle_contact_density_min: float = 0.25
    huddle_proximity_density_min: float = 0.60
    huddle_graph_stability_min: float = 0.50
    huddle_soft_score_min: float = 0.60
    huddle_attack_speed_p90_max_cm_s: float = 60.1875
    huddle_attack_relative_motion_max_cm_s: float = 40.125

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "CollectiveRuleConfig":
        """Create a config while rejecting unknown, scattered thresholds."""

        if not values:
            return cls()
        known = {field.name for field in cls.__dataclass_fields__.values()}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"unknown collective-state parameters: {sorted(unknown)}")
        return cls(**{name: values[name] for name in values})


@dataclass(frozen=True)
class PairContactFeatures:
    """Per-frame pair geometry used as contact evidence, not ground truth."""

    mouse_i: int
    mouse_j: int
    center_distance_cm: float
    center_distance_bl: float
    center_distance_bw: float
    bbox_iou: float
    bbox_iom: float
    bbox_overlap_fraction_i: float
    bbox_overlap_fraction_j: float
    bbox_gap_cm: float
    bbox_gap_bw: float
    min_keypoint_distance_bw: float
    contact_score: float


@dataclass(frozen=True)
class GroupGraphFeatures:
    """Window-level summaries of the dynamic proximity/contact graphs."""

    proximity_density_mean: float
    contact_density_mean: float
    weighted_degree_mean: float
    minimum_degree_mean: float
    edge_gain_rate: float
    edge_loss_rate: float
    component_merge_count: int
    component_split_count: int
    component_growth: float
    graph_jaccard: float
    membership_stability: float


@dataclass(frozen=True)
class SocialClusteringFeatures:
    """Features dedicated to the dynamic group-formation process."""

    nnd_start_bw: float
    nnd_end_bw: float
    nnd_drop_bw: float
    nnd_drop_ratio: float
    nnd_slope_bw_s: float
    nnd_negative_fraction: float
    pair_distance_start_bw: float
    pair_distance_end_bw: float
    pair_distance_slope_bw_s: float
    radius_start_bw: float
    radius_end_bw: float
    radius_drop_bw: float
    radius_slope_bw_s: float
    radius_negative_fraction: float
    inward_velocity_mean_cm_s: float
    inward_velocity_median_cm_s: float
    inward_motion_fraction: float
    active_inward_member_fraction: float
    moving_fraction: float
    static_fraction: float
    mean_member_speed_cm_s: float
    active_member_fraction: float
    edge_count_start: float
    edge_count_end: float
    edge_gain: float
    density_start: float
    density_end: float
    density_slope_s: float
    component_count_start: float
    component_count_end: float
    largest_component_growth: float


@dataclass(frozen=True)
class HuddlingFeatures:
    """Features dedicated to an already compact, stable group state."""

    nnd_mean_bw: float
    nnd_median_bw: float
    nnd_p75_bw: float
    nnd_max_bw: float
    group_radius_bw: float
    nnd_std_bw: float
    nnd_slope_abs_bw_s: float
    radius_slope_abs_bw_s: float
    pair_distance_variance_bw: float
    relative_configuration_stability: float
    mean_member_speed_cm_s: float
    speed_p75_cm_s: float
    speed_p90_cm_s: float
    low_motion_fraction: float
    static_fraction: float
    moving_fraction: float
    group_internal_motion_cm_s: float
    bbox_iou_mean: float
    bbox_iou_max: float
    bbox_iom_mean: float
    bbox_overlap_fraction_mean: float
    bbox_gap_mean_cm: float
    bbox_gap_mean_bw: float
    contact_score_mean: float
    contact_score_max: float
    contact_neighbor_count_mean: float
    contact_degree_mean: float
    contact_graph_density: float
    proximity_graph_density: float
    graph_jaccard: float
    dropout_fraction: float
    mean_dropout_duration_s: float
    max_dropout_duration_s: float
    recovery_near_group_fraction: float
    reacquisition_score: float


@dataclass(frozen=True)
class GroupWindowFeatures:
    """All evidence for one candidate member set and one temporal window."""

    member_slots: tuple[int, ...]
    start_frame: int
    end_frame: int
    duration_s: float
    group_visible_fraction: float
    minimum_member_support_fraction: float
    huddle_group_visible_fraction: float
    huddle_minimum_member_support_fraction: float
    graph: GroupGraphFeatures
    social: SocialClusteringFeatures
    huddle: HuddlingFeatures

    def debug_values(self) -> dict[str, Any]:
        values: dict[str, Any] = {
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "duration_s": self.duration_s,
            "member_slots": ",".join(map(str, self.member_slots)),
            "group_visible_fraction": self.group_visible_fraction,
            "minimum_member_support_fraction": self.minimum_member_support_fraction,
            "huddle_group_visible_fraction": self.huddle_group_visible_fraction,
            "huddle_minimum_member_support_fraction": self.huddle_minimum_member_support_fraction,
        }
        values.update({f"GRAPH_{key}": value for key, value in asdict(self.graph).items()})
        values.update({f"CLU_{key}": value for key, value in asdict(self.social).items()})
        values.update({f"HUD_{key}": value for key, value in asdict(self.huddle).items()})
        return values


@dataclass(frozen=True)
class CollectiveDecision:
    """Best label-free evidence windows for one exact member set."""

    social_clustering_score: float
    huddling_score: float
    social_window: GroupWindowFeatures | None
    huddle_window: GroupWindowFeatures | None
    social_checks: Mapping[str, bool]
    huddle_checks: Mapping[str, bool]


def _finite(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=float).reshape(-1)
    return result[np.isfinite(result)]


def _safe_stat(values: np.ndarray, function: Any, default: float = float("nan")) -> float:
    usable = _finite(values)
    return float(function(usable)) if usable.size else float(default)


def _safe_slope(values: np.ndarray, fps: float) -> float:
    array = np.asarray(values, dtype=float).reshape(-1)
    finite = np.isfinite(array)
    if np.count_nonzero(finite) < 2:
        return float("nan")
    time_s = np.arange(len(array), dtype=float)[finite] / max(float(fps), 1e-9)
    return float(np.polyfit(time_s, array[finite], 1)[0])


def _end_state(values: np.ndarray) -> tuple[float, float]:
    array = np.asarray(values, dtype=float).reshape(-1)
    finite_indices = np.flatnonzero(np.isfinite(array))
    if not finite_indices.size:
        return float("nan"), float("nan")
    width = max(1, int(np.ceil(finite_indices.size * 0.20)))
    return (
        float(np.median(array[finite_indices[:width]])),
        float(np.median(array[finite_indices[-width:]])),
    )


def _change_fraction(values: np.ndarray, predicate: Any) -> float:
    array = np.asarray(values, dtype=float).reshape(-1)
    left, right = array[:-1], array[1:]
    finite = np.isfinite(left) & np.isfinite(right)
    return float(np.mean(predicate(right[finite] - left[finite]))) if np.any(finite) else 0.0


def robust_body_centers(keypoints_cm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return NaN-aware body centers using neck, hips, and tail base.

    All four body points use their mean.  A partial body uses its median, and
    fewer than two body points falls back to the median of all valid points.
    Missing coordinates remain NaN and are never replaced by ``(0, 0)``.
    """

    points = np.asarray(keypoints_cm, dtype=float)
    if points.ndim != 4 or points.shape[2] < 7 or points.shape[3] != 2:
        raise ValueError("keypoints_cm must have shape (frames, mice, >=7, 2)")
    frames, mice = points.shape[:2]
    centers = np.full((frames, mice, 2), np.nan, dtype=float)
    for frame in range(frames):
        for mouse in range(mice):
            body = points[frame, mouse, _BODY_POINT_INDICES]
            body = body[np.isfinite(body).all(axis=1)]
            if len(body) == len(_BODY_POINT_INDICES):
                centers[frame, mouse] = np.mean(body, axis=0)
            elif len(body) >= 2:
                centers[frame, mouse] = np.median(body, axis=0)
            else:
                available = points[frame, mouse]
                available = available[np.isfinite(available).all(axis=1)]
                if len(available):
                    centers[frame, mouse] = np.median(available, axis=0)
    return centers, np.isfinite(centers).all(axis=2)


def _frame_overlap_mask(boxes: np.ndarray | None) -> np.ndarray | None:
    if boxes is None:
        return None
    array = np.asarray(boxes, dtype=float)
    frames, mice = array.shape[:2]
    overlapping = np.zeros((frames, mice), dtype=bool)
    good = np.isfinite(array).all(axis=2)
    width = np.maximum(array[:, :, 2] - array[:, :, 0], 0.0)
    height = np.maximum(array[:, :, 3] - array[:, :, 1], 0.0)
    area = width * height
    for left, right in combinations(range(mice), 2):
        valid = good[:, left] & good[:, right]
        intersection_width = np.maximum(
            np.minimum(array[:, left, 2], array[:, right, 2])
            - np.maximum(array[:, left, 0], array[:, right, 0]),
            0.0,
        )
        intersection_height = np.maximum(
            np.minimum(array[:, left, 3], array[:, right, 3])
            - np.maximum(array[:, left, 1], array[:, right, 1]),
            0.0,
        )
        intersection = intersection_width * intersection_height
        union = area[:, left] + area[:, right] - intersection
        iou = np.divide(
            intersection,
            union,
            out=np.zeros(frames, dtype=float),
            where=valid & (union > 1e-9),
        )
        overlap = iou > 0.05
        overlapping[:, left] |= overlap
        overlapping[:, right] |= overlap
    return overlapping


def estimate_morphology(
    keypoints_cm: np.ndarray,
    valid: np.ndarray,
    *,
    pose_quality: np.ndarray | None = None,
    boxes: np.ndarray | None = None,
    reference_body_cm: float | None = None,
    reference_body_width_cm: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate stable per-ID length, width, and bbox area references."""

    points = np.asarray(keypoints_cm, dtype=float)
    observed = np.asarray(valid, dtype=bool)
    if points.shape[:2] != observed.shape:
        raise ValueError("keypoints and valid shapes disagree")
    quality = (
        np.asarray(pose_quality, dtype=float)
        if pose_quality is not None
        else np.ones(observed.shape, dtype=float)
    )
    if quality.shape != observed.shape:
        raise ValueError("pose_quality shape disagrees with valid")
    overlap = _frame_overlap_mask(boxes)
    high_quality = observed & np.isfinite(quality) & (quality >= 0.50)
    if overlap is not None:
        high_quality &= ~overlap
    length = np.linalg.norm(points[:, :, 3] - points[:, :, 6], axis=2)
    width = np.linalg.norm(points[:, :, 4] - points[:, :, 5], axis=2)
    lengths = np.full(observed.shape[1], np.nan, dtype=float)
    widths = np.full(observed.shape[1], np.nan, dtype=float)
    areas = np.full(observed.shape[1], np.nan, dtype=float)
    box_array = np.asarray(boxes, dtype=float) if boxes is not None else None
    for mouse in range(observed.shape[1]):
        for source, target, minimum in ((length, lengths, 0.5), (width, widths, 0.2)):
            values = source[:, mouse]
            usable = values[high_quality[:, mouse] & np.isfinite(values) & (values >= minimum)]
            if not usable.size:
                usable = values[observed[:, mouse] & np.isfinite(values) & (values >= minimum)]
            if usable.size:
                target[mouse] = float(np.median(usable))
        if box_array is not None:
            box = box_array[:, mouse]
            area = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
            usable = area[high_quality[:, mouse] & np.isfinite(area) & (area > 0)]
            if usable.size:
                areas[mouse] = float(np.median(usable))
    global_length = _safe_stat(lengths, np.median, float("nan"))
    if not np.isfinite(global_length):
        if reference_body_cm is None:
            raise ValueError(
                "estimate_morphology found no usable body-length samples; pass "
                "reference_body_cm from the resolved scene scale"
            )
        global_length = float(reference_body_cm)
    if reference_body_width_cm is not None:
        if not np.isfinite(reference_body_width_cm) or reference_body_width_cm <= 0:
            raise ValueError("reference_body_width_cm must be finite and positive")
        width_fallback = float(reference_body_width_cm)
    else:
        width_fallback = max(global_length * 0.30, 1.0)
    global_width = _safe_stat(widths, np.median, width_fallback)
    global_area = _safe_stat(areas, np.median, global_length * global_width)
    lengths[~np.isfinite(lengths)] = global_length
    widths[~np.isfinite(widths)] = global_width
    areas[~np.isfinite(areas)] = global_area
    return lengths, widths, areas


def bbox_contact_features(
    left_box: np.ndarray,
    right_box: np.ndarray,
    mean_body_width: float,
) -> tuple[float, float, float, float, float, float]:
    """Return IoU, IoM, per-box overlaps, raw gap, and BW-normalized gap."""

    left = np.asarray(left_box, dtype=float)
    right = np.asarray(right_box, dtype=float)
    if (
        left.shape != (4,)
        or right.shape != (4,)
        or not np.isfinite(left).all()
        or not np.isfinite(right).all()
        or left[2] <= left[0]
        or left[3] <= left[1]
        or right[2] <= right[0]
        or right[3] <= right[1]
    ):
        return (float("nan"),) * 6
    overlap_width = max(min(left[2], right[2]) - max(left[0], right[0]), 0.0)
    overlap_height = max(min(left[3], right[3]) - max(left[1], right[1]), 0.0)
    intersection = overlap_width * overlap_height
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    union = left_area + right_area - intersection
    iou = intersection / max(union, 1e-9)
    iom = intersection / max(min(left_area, right_area), 1e-9)
    overlap_left = intersection / max(left_area, 1e-9)
    overlap_right = intersection / max(right_area, 1e-9)
    dx = max(left[0] - right[2], right[0] - left[2], 0.0)
    dy = max(left[1] - right[3], right[1] - left[3], 0.0)
    gap = float(np.hypot(dx, dy))
    return iou, iom, overlap_left, overlap_right, gap, gap / max(mean_body_width, 1e-9)


def pair_contact_features(
    *,
    frame: int,
    mouse_i: int,
    mouse_j: int,
    centers_cm: np.ndarray,
    body_length_ref_cm: np.ndarray,
    body_width_ref_cm: np.ndarray,
    keypoints_cm: np.ndarray | None,
    boxes_cm: np.ndarray | None,
    config: CollectiveRuleConfig,
) -> PairContactFeatures:
    """Construct the document-defined contact evidence for one pair/frame."""

    center_left = centers_cm[frame, mouse_i]
    center_right = centers_cm[frame, mouse_j]
    distance = (
        float(np.linalg.norm(center_left - center_right))
        if np.isfinite(center_left).all() and np.isfinite(center_right).all()
        else float("nan")
    )
    mean_length = max(
        float((body_length_ref_cm[mouse_i] + body_length_ref_cm[mouse_j]) / 2.0), 1e-9
    )
    mean_width = max(
        float((body_width_ref_cm[mouse_i] + body_width_ref_cm[mouse_j]) / 2.0), 1e-9
    )
    bbox_values = (
        bbox_contact_features(boxes_cm[frame, mouse_i], boxes_cm[frame, mouse_j], mean_width)
        if boxes_cm is not None
        else (float("nan"),) * 6
    )
    minimum_keypoint_bw = float("nan")
    if keypoints_cm is not None:
        left_points = keypoints_cm[frame, mouse_i]
        right_points = keypoints_cm[frame, mouse_j]
        left_points = left_points[np.isfinite(left_points).all(axis=1)]
        right_points = right_points[np.isfinite(right_points).all(axis=1)]
        if len(left_points) and len(right_points):
            minimum_keypoint_bw = float(
                np.min(np.linalg.norm(left_points[:, None, :] - right_points[None, :, :], axis=2))
                / mean_width
            )
    iou, iom, overlap_i, overlap_j, gap_cm, gap_bw = bbox_values
    evidence = np.asarray(
        (
            iou,
            iom,
            np.exp(-minimum_keypoint_bw) if np.isfinite(minimum_keypoint_bw) else np.nan,
            np.exp(-gap_bw) if np.isfinite(gap_bw) else np.nan,
        ),
        dtype=float,
    )
    weights = np.asarray(
        (
            config.contact_iou_weight,
            config.contact_iom_weight,
            config.contact_keypoint_weight,
            config.contact_gap_weight,
        ),
        dtype=float,
    )
    usable = np.isfinite(evidence) & np.isfinite(weights) & (weights > 0)
    contact_score = (
        float(np.average(evidence[usable], weights=weights[usable])) if np.any(usable) else 0.0
    )
    return PairContactFeatures(
        mouse_i=mouse_i,
        mouse_j=mouse_j,
        center_distance_cm=distance,
        center_distance_bl=distance / mean_length if np.isfinite(distance) else float("nan"),
        center_distance_bw=distance / mean_width if np.isfinite(distance) else float("nan"),
        bbox_iou=iou,
        bbox_iom=iom,
        bbox_overlap_fraction_i=overlap_i,
        bbox_overlap_fraction_j=overlap_j,
        bbox_gap_cm=gap_cm,
        bbox_gap_bw=gap_bw,
        min_keypoint_distance_bw=minimum_keypoint_bw,
        contact_score=float(np.clip(contact_score, 0.0, 1.0)),
    )


def graph_density(adjacency: np.ndarray) -> float:
    matrix = np.asarray(adjacency, dtype=bool)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("adjacency must be square")
    nodes = matrix.shape[0]
    return float(np.count_nonzero(np.triu(matrix, 1)) / max(nodes * (nodes - 1) / 2, 1))


def connected_components(adjacency: np.ndarray) -> list[tuple[int, ...]]:
    matrix = np.asarray(adjacency, dtype=bool)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("adjacency must be square")
    unseen = set(range(matrix.shape[0]))
    result: list[tuple[int, ...]] = []
    while unseen:
        seed = unseen.pop()
        pending = [seed]
        component = [seed]
        while pending:
            current = pending.pop()
            neighbours = [index for index in tuple(unseen) if matrix[current, index]]
            for index in neighbours:
                unseen.remove(index)
                component.append(index)
                pending.append(index)
        result.append(tuple(sorted(component)))
    return result


def graph_jaccard(left: set[tuple[int, int]], right: set[tuple[int, int]]) -> float:
    union = left | right
    return float(len(left & right) / len(union)) if union else 1.0


def _dropout_evidence(
    observed: np.ndarray,
    centers_cm: np.ndarray,
    body_width_ref_cm: np.ndarray,
    fps: float,
    near_group_max_bw: float,
) -> tuple[float, float, float, np.ndarray]:
    """Measure same-ID gaps and accept only reacquisition beside its group.

    Missing positions are never interpolated. A gap can contribute support to
    Huddling only when the same track slot was observed on both sides and its
    reacquired center is near the median center of at least two other visible
    members. The returned mask marks only those bracketed, near-group gaps.
    """

    durations: list[float] = []
    near_group_recoveries = 0
    dropout_frames = np.zeros_like(observed, dtype=bool)
    for mouse in range(observed.shape[1]):
        values = observed[:, mouse]
        padded = np.r_[True, values, True].astype(np.int8)
        changes = np.diff(padded)
        starts = np.flatnonzero(changes == -1)
        ends = np.flatnonzero(changes == 1)
        for start, end in zip(starts, ends):
            if start == 0 or end >= len(values):
                continue
            durations.append((end - start) / max(float(fps), 1e-9))
            if not (values[start - 1] and values[end]):
                continue
            peers = np.flatnonzero(observed[end])
            peers = peers[peers != mouse]
            if len(peers) < 2:
                continue
            reacquired_center = centers_cm[end, mouse]
            peer_centers = centers_cm[end, peers]
            if not (
                np.isfinite(reacquired_center).all()
                and np.isfinite(peer_centers).all()
            ):
                continue
            group_center = np.median(peer_centers, axis=0)
            reference_width = float(
                np.median(np.asarray(body_width_ref_cm, dtype=float)[[mouse, *peers]])
            )
            distance_bw = float(
                np.linalg.norm(reacquired_center - group_center)
                / max(reference_width, 1e-9)
            )
            if distance_bw <= near_group_max_bw:
                near_group_recoveries += 1
                dropout_frames[start:end, mouse] = True
    mean_duration = float(np.mean(durations)) if durations else 0.0
    max_duration = float(np.max(durations)) if durations else 0.0
    recovery_fraction = near_group_recoveries / max(len(durations), 1)
    return mean_duration, max_duration, float(recovery_fraction), dropout_frames


def extract_group_window_features(
    *,
    centers_cm: np.ndarray,
    valid: np.ndarray,
    speed_cm_s: np.ndarray,
    velocity_cm_s: np.ndarray,
    member_slots: Sequence[int],
    fps: float,
    body_length_ref_cm: np.ndarray,
    body_width_ref_cm: np.ndarray,
    start_frame: int,
    end_frame: int,
    config: CollectiveRuleConfig,
    keypoints_cm: np.ndarray | None = None,
    boxes_cm: np.ndarray | None = None,
    pair_contact_cache: dict[tuple[int, int, int], PairContactFeatures] | None = None,
) -> GroupWindowFeatures:
    """Extract all CLU/HUD features for one exact member set and window."""

    centers = np.asarray(centers_cm, dtype=float)
    observed_all = np.asarray(valid, dtype=bool)
    speed = np.asarray(speed_cm_s, dtype=float)
    velocity = np.asarray(velocity_cm_s, dtype=float)
    members = tuple(sorted({int(value) for value in member_slots}))
    if len(members) < config.min_members:
        raise ValueError("collective candidate must contain at least three unique members")
    if start_frame < 0 or end_frame < start_frame or end_frame >= len(centers):
        raise ValueError("invalid collective feature window")
    window = slice(start_frame, end_frame + 1)
    observed = observed_all[window][:, members] & np.isfinite(centers[window][:, members]).all(axis=2)
    frames = observed.shape[0]
    member_support = observed.mean(axis=0) if frames else np.zeros(len(members))
    visible_count = observed.sum(axis=1)
    group_visible = visible_count >= config.min_members
    local_widths = np.asarray(body_width_ref_cm, dtype=float)[list(members)]
    mean_dropout, max_dropout, reacquisition, recovered_dropout = _dropout_evidence(
        observed,
        centers[window][:, members],
        local_widths,
        fps,
        config.huddle_recovery_near_group_max_bw,
    )
    huddle_effective_observed = observed | recovered_dropout
    huddle_member_support = (
        huddle_effective_observed.mean(axis=0) if frames else np.zeros(len(members))
    )
    huddle_group_visible = group_visible | (
        (visible_count >= 2) & np.any(recovered_dropout, axis=1)
    )

    nnd_bw = np.full(frames, np.nan, dtype=float)
    pair_mean_bw = np.full(frames, np.nan, dtype=float)
    radius_bw = np.full(frames, np.nan, dtype=float)
    inward_mean = np.full(frames, np.nan, dtype=float)
    inward_fraction = np.full(frames, np.nan, dtype=float)
    active_inward_fraction = np.full(frames, np.nan, dtype=float)
    active_fraction = np.full(frames, np.nan, dtype=float)
    moving_fraction = np.full(frames, np.nan, dtype=float)
    static_fraction = np.full(frames, np.nan, dtype=float)
    member_speed_mean = np.full(frames, np.nan, dtype=float)
    internal_motion = np.full(frames, np.nan, dtype=float)
    proximity_density = np.full(frames, np.nan, dtype=float)
    contact_density = np.full(frames, np.nan, dtype=float)
    edge_count = np.full(frames, np.nan, dtype=float)
    component_count = np.full(frames, np.nan, dtype=float)
    largest_component = np.full(frames, np.nan, dtype=float)
    weighted_degree = np.full(frames, np.nan, dtype=float)
    minimum_degree = np.full(frames, np.nan, dtype=float)
    contact_score_mean = np.full(frames, np.nan, dtype=float)
    contact_score_max = np.full(frames, np.nan, dtype=float)
    contact_neighbor_mean = np.full(frames, np.nan, dtype=float)
    contact_degree_mean = np.full(frames, np.nan, dtype=float)
    bbox_iou_mean = np.full(frames, np.nan, dtype=float)
    bbox_iou_max = np.full(frames, np.nan, dtype=float)
    bbox_iom_mean = np.full(frames, np.nan, dtype=float)
    bbox_overlap_mean = np.full(frames, np.nan, dtype=float)
    bbox_gap_mean = np.full(frames, np.nan, dtype=float)
    bbox_gap_bw_mean = np.full(frames, np.nan, dtype=float)
    edge_sets: list[set[tuple[int, int]]] = [set() for _ in range(frames)]
    contact_edge_sets: list[set[tuple[int, int]]] = [set() for _ in range(frames)]
    active_inward_frames_by_member = np.zeros(len(members), dtype=int)
    inward_motion_frames_by_member = np.zeros(len(members), dtype=int)

    for local_frame in range(frames):
        absolute_frame = start_frame + local_frame
        visible_local = np.flatnonzero(observed[local_frame])
        # Two observed members can still provide pair geometry during a
        # confirmed same-ID dropout; this does not invent the missing mouse's
        # location, and Social clustering retains its all-members visibility
        # gate below.
        if len(visible_local) < 2:
            continue
        slots = np.asarray([members[index] for index in visible_local], dtype=int)
        points = centers[absolute_frame, slots]
        widths = body_width_ref_cm[slots]
        center = np.mean(points, axis=0)
        radius_bw[local_frame] = float(
            np.mean(np.linalg.norm(points - center, axis=1)) / max(np.median(widths), 1e-9)
        )
        local_velocity = velocity[absolute_frame, slots]
        toward = center - points
        toward_norm = np.linalg.norm(toward, axis=1, keepdims=True)
        toward_unit = np.divide(
            toward,
            toward_norm,
            out=np.zeros_like(toward),
            where=toward_norm > 1e-9,
        )
        inward = np.sum(local_velocity * toward_unit, axis=1)
        usable_inward = np.isfinite(inward)
        if np.any(usable_inward):
            inward_mean[local_frame] = float(np.mean(inward[usable_inward]))
            inward_fraction[local_frame] = float(np.mean(inward[usable_inward] > 0.0))
        local_speed = speed[absolute_frame, slots]
        usable_speed = np.isfinite(local_speed)
        if np.any(usable_speed):
            speeds = local_speed[usable_speed]
            member_speed_mean[local_frame] = float(np.mean(speeds))
            active = speeds >= config.active_speed_cm_s
            active_fraction[local_frame] = float(np.mean(active))
            static_fraction[local_frame] = float(np.mean(speeds <= config.low_motion_speed_cm_s))
            moving_fraction[local_frame] = float(
                np.mean(
                    (speeds >= config.active_speed_cm_s)
                    & (speeds <= config.huddle_attack_speed_p90_max_cm_s)
                )
            )
            active_inward_fraction[local_frame] = float(
                np.mean((inward[usable_speed] > 0.0) & active)
            )
        for visible_position, member_position in enumerate(visible_local):
            if not usable_speed[visible_position] or not usable_inward[visible_position]:
                continue
            inward_motion_frames_by_member[member_position] += 1
            if (
                local_speed[visible_position] >= config.active_speed_cm_s
                and inward[visible_position] > 0.0
            ):
                active_inward_frames_by_member[member_position] += 1

        count = len(slots)
        proximity = np.zeros((count, count), dtype=bool)
        contact = np.zeros((count, count), dtype=bool)
        contact_weights = np.zeros((count, count), dtype=float)
        distance_matrix = np.full((count, count), np.nan, dtype=float)
        pair_contacts: list[PairContactFeatures] = []
        for left, right in combinations(range(count), 2):
            left_slot = int(slots[left])
            right_slot = int(slots[right])
            cache_key = (absolute_frame, left_slot, right_slot)
            pair = pair_contact_cache.get(cache_key) if pair_contact_cache is not None else None
            if pair is None:
                pair = pair_contact_features(
                    frame=absolute_frame,
                    mouse_i=left_slot,
                    mouse_j=right_slot,
                    centers_cm=centers,
                    body_length_ref_cm=body_length_ref_cm,
                    body_width_ref_cm=body_width_ref_cm,
                    keypoints_cm=keypoints_cm,
                    boxes_cm=boxes_cm,
                    config=config,
                )
                if pair_contact_cache is not None:
                    pair_contact_cache[cache_key] = pair
            pair_contacts.append(pair)
            distance_matrix[left, right] = distance_matrix[right, left] = pair.center_distance_bw
            if np.isfinite(pair.center_distance_bw) and (
                pair.center_distance_bw <= config.proximity_distance_bw
            ):
                proximity[left, right] = proximity[right, left] = True
                edge_sets[local_frame].add(tuple(sorted((pair.mouse_i, pair.mouse_j))))
            if pair.contact_score >= config.contact_score_threshold:
                contact[left, right] = contact[right, left] = True
                contact_edge_sets[local_frame].add(tuple(sorted((pair.mouse_i, pair.mouse_j))))
            contact_weights[left, right] = contact_weights[right, left] = pair.contact_score
        finite_distances = distance_matrix[np.triu_indices(count, 1)]
        finite_distances = finite_distances[np.isfinite(finite_distances)]
        if finite_distances.size:
            pair_mean_bw[local_frame] = float(np.mean(finite_distances))
            nearest = np.nanmin(
                np.where(np.eye(count, dtype=bool), np.nan, distance_matrix), axis=1
            )
            nnd_bw[local_frame] = float(np.median(nearest[np.isfinite(nearest)]))
            if local_frame > 0 and np.isfinite(pair_mean_bw[local_frame - 1]):
                internal_motion[local_frame] = float(
                    abs(pair_mean_bw[local_frame] - pair_mean_bw[local_frame - 1])
                    * np.median(widths)
                    * fps
                )
        proximity_density[local_frame] = graph_density(proximity)
        contact_density[local_frame] = graph_density(contact)
        edge_count[local_frame] = len(edge_sets[local_frame])
        components = connected_components(proximity)
        component_count[local_frame] = len(components)
        largest_component[local_frame] = max((len(item) for item in components), default=0)
        degrees = proximity.sum(axis=1)
        minimum_degree[local_frame] = float(np.min(degrees)) if len(degrees) else 0.0
        weighted_degree[local_frame] = float(np.mean(contact_weights.sum(axis=1)))
        contact_neighbor_mean[local_frame] = float(np.mean(contact.sum(axis=1)))
        contact_degree_mean[local_frame] = float(np.mean(contact_weights.sum(axis=1)))
        if pair_contacts:
            contact_score_mean[local_frame] = _safe_stat(
                np.asarray([item.contact_score for item in pair_contacts]), np.mean, 0.0
            )
            contact_score_max[local_frame] = _safe_stat(
                np.asarray([item.contact_score for item in pair_contacts]), np.max, 0.0
            )
            bbox_iou_mean[local_frame] = _safe_stat(
                np.asarray([item.bbox_iou for item in pair_contacts]), np.mean
            )
            bbox_iou_max[local_frame] = _safe_stat(
                np.asarray([item.bbox_iou for item in pair_contacts]), np.max
            )
            bbox_iom_mean[local_frame] = _safe_stat(
                np.asarray([item.bbox_iom for item in pair_contacts]), np.mean
            )
            bbox_overlap_mean[local_frame] = _safe_stat(
                np.asarray(
                    [
                        (item.bbox_overlap_fraction_i + item.bbox_overlap_fraction_j) / 2.0
                        for item in pair_contacts
                    ]
                ),
                np.mean,
            )
            bbox_gap_mean[local_frame] = _safe_stat(
                np.asarray([item.bbox_gap_cm for item in pair_contacts]), np.mean
            )
            bbox_gap_bw_mean[local_frame] = _safe_stat(
                np.asarray([item.bbox_gap_bw for item in pair_contacts]), np.mean
            )

    jaccard_values = np.asarray(
        [graph_jaccard(left, right) for left, right in zip(edge_sets[:-1], edge_sets[1:])],
        dtype=float,
    )
    gains = np.asarray(
        [len(right - left) for left, right in zip(edge_sets[:-1], edge_sets[1:])], dtype=float
    )
    losses = np.asarray(
        [len(left - right) for left, right in zip(edge_sets[:-1], edge_sets[1:])], dtype=float
    )
    merges = int(
        np.count_nonzero(
            np.diff(component_count[np.isfinite(component_count)]) < 0
        )
    )
    splits = int(
        np.count_nonzero(
            np.diff(component_count[np.isfinite(component_count)]) > 0
        )
    )
    edge_start, edge_end = _end_state(edge_count)
    density_start, density_end = _end_state(proximity_density)
    component_start, component_end = _end_state(component_count)
    largest_start, largest_end = _end_state(largest_component)
    nnd_start, nnd_end = _end_state(nnd_bw)
    pair_start, pair_end = _end_state(pair_mean_bw)
    radius_start, radius_end = _end_state(radius_bw)
    nnd_drop = nnd_start - nnd_end
    radius_drop = radius_start - radius_end
    duration_s = frames / max(float(fps), 1e-9)
    active_inward_member_fractions = np.divide(
        active_inward_frames_by_member,
        inward_motion_frames_by_member,
        out=np.zeros(len(members), dtype=float),
        where=inward_motion_frames_by_member > 0,
    )
    active_inward_member_fraction = float(
        np.mean(
            active_inward_member_fractions
            >= _ACTIVE_INWARD_MEMBER_FRACTION_MIN
        )
    )
    social = SocialClusteringFeatures(
        nnd_start_bw=nnd_start,
        nnd_end_bw=nnd_end,
        nnd_drop_bw=nnd_drop,
        nnd_drop_ratio=nnd_drop / max(abs(nnd_start), 1e-9) if np.isfinite(nnd_drop) else float("nan"),
        nnd_slope_bw_s=_safe_slope(nnd_bw, fps),
        nnd_negative_fraction=_change_fraction(nnd_bw, lambda delta: delta < 0.0),
        pair_distance_start_bw=pair_start,
        pair_distance_end_bw=pair_end,
        pair_distance_slope_bw_s=_safe_slope(pair_mean_bw, fps),
        radius_start_bw=radius_start,
        radius_end_bw=radius_end,
        radius_drop_bw=radius_drop,
        radius_slope_bw_s=_safe_slope(radius_bw, fps),
        radius_negative_fraction=_change_fraction(radius_bw, lambda delta: delta < 0.0),
        inward_velocity_mean_cm_s=_safe_stat(inward_mean, np.mean, 0.0),
        inward_velocity_median_cm_s=_safe_stat(inward_mean, np.median, 0.0),
        inward_motion_fraction=_safe_stat(inward_fraction, np.mean, 0.0),
        active_inward_member_fraction=active_inward_member_fraction,
        moving_fraction=_safe_stat(moving_fraction, np.mean, 0.0),
        static_fraction=_safe_stat(static_fraction, np.mean, 0.0),
        mean_member_speed_cm_s=_safe_stat(member_speed_mean, np.mean, 0.0),
        active_member_fraction=_safe_stat(active_fraction, np.mean, 0.0),
        edge_count_start=edge_start,
        edge_count_end=edge_end,
        edge_gain=edge_end - edge_start,
        density_start=density_start,
        density_end=density_end,
        density_slope_s=_safe_slope(proximity_density, fps),
        component_count_start=component_start,
        component_count_end=component_end,
        largest_component_growth=largest_end - largest_start,
    )
    usable_speed = speed[window][:, members][observed]
    nnd_values = _finite(nnd_bw)
    pair_values = _finite(pair_mean_bw)
    internal_values = _finite(internal_motion)
    huddle = HuddlingFeatures(
        nnd_mean_bw=_safe_stat(nnd_values, np.mean),
        nnd_median_bw=_safe_stat(nnd_values, np.median),
        nnd_p75_bw=_safe_stat(nnd_values, lambda value: np.quantile(value, 0.75)),
        nnd_max_bw=_safe_stat(nnd_values, np.max),
        group_radius_bw=_safe_stat(radius_bw, np.median),
        nnd_std_bw=_safe_stat(nnd_values, np.std),
        nnd_slope_abs_bw_s=abs(_safe_slope(nnd_bw, fps)),
        radius_slope_abs_bw_s=abs(_safe_slope(radius_bw, fps)),
        pair_distance_variance_bw=_safe_stat(pair_values, np.var),
        relative_configuration_stability=float(
            1.0 / (1.0 + _safe_stat(pair_values, np.std, float("inf")))
        ),
        mean_member_speed_cm_s=_safe_stat(usable_speed, np.mean, 0.0),
        speed_p75_cm_s=_safe_stat(usable_speed, lambda value: np.quantile(value, 0.75), 0.0),
        speed_p90_cm_s=_safe_stat(usable_speed, lambda value: np.quantile(value, 0.90), 0.0),
        low_motion_fraction=float(
            np.mean(usable_speed <= config.low_motion_speed_cm_s) if usable_speed.size else 0.0
        ),
        static_fraction=_safe_stat(static_fraction, np.mean, 0.0),
        moving_fraction=_safe_stat(moving_fraction, np.mean, 0.0),
        group_internal_motion_cm_s=_safe_stat(internal_values, np.mean, 0.0),
        bbox_iou_mean=_safe_stat(bbox_iou_mean, np.mean, 0.0),
        bbox_iou_max=_safe_stat(bbox_iou_max, np.max, 0.0),
        bbox_iom_mean=_safe_stat(bbox_iom_mean, np.mean, 0.0),
        bbox_overlap_fraction_mean=_safe_stat(bbox_overlap_mean, np.mean, 0.0),
        bbox_gap_mean_cm=_safe_stat(bbox_gap_mean, np.mean, float("inf")),
        bbox_gap_mean_bw=_safe_stat(bbox_gap_bw_mean, np.mean, float("inf")),
        contact_score_mean=_safe_stat(contact_score_mean, np.mean, 0.0),
        contact_score_max=_safe_stat(contact_score_max, np.max, 0.0),
        contact_neighbor_count_mean=_safe_stat(contact_neighbor_mean, np.mean, 0.0),
        contact_degree_mean=_safe_stat(contact_degree_mean, np.mean, 0.0),
        contact_graph_density=_safe_stat(contact_density, np.mean, 0.0),
        proximity_graph_density=_safe_stat(proximity_density, np.mean, 0.0),
        graph_jaccard=_safe_stat(jaccard_values, np.mean, 1.0),
        dropout_fraction=float(1.0 - np.mean(observed)),
        mean_dropout_duration_s=mean_dropout,
        max_dropout_duration_s=max_dropout,
        recovery_near_group_fraction=reacquisition,
        reacquisition_score=reacquisition,
    )
    graph = GroupGraphFeatures(
        proximity_density_mean=huddle.proximity_graph_density,
        contact_density_mean=huddle.contact_graph_density,
        weighted_degree_mean=_safe_stat(weighted_degree, np.mean, 0.0),
        minimum_degree_mean=_safe_stat(minimum_degree, np.mean, 0.0),
        edge_gain_rate=_safe_stat(gains, np.mean, 0.0) * fps,
        edge_loss_rate=_safe_stat(losses, np.mean, 0.0) * fps,
        component_merge_count=merges,
        component_split_count=splits,
        component_growth=largest_end - largest_start,
        graph_jaccard=huddle.graph_jaccard,
        membership_stability=float(np.mean(observed)),
    )
    return GroupWindowFeatures(
        member_slots=members,
        start_frame=start_frame,
        end_frame=end_frame,
        duration_s=duration_s,
        group_visible_fraction=float(np.mean(group_visible)),
        minimum_member_support_fraction=float(np.min(member_support)),
        huddle_group_visible_fraction=float(np.mean(huddle_group_visible)),
        huddle_minimum_member_support_fraction=float(np.min(huddle_member_support)),
        graph=graph,
        social=social,
        huddle=huddle,
    )


def score_social_clustering(
    features: GroupWindowFeatures,
    config: CollectiveRuleConfig,
) -> tuple[float, dict[str, bool]]:
    """Apply hard duration/member gates plus independent formation evidence."""

    social = features.social
    checks = {
        "member_count": len(features.member_slots) >= config.min_members,
        "duration": features.duration_s + 1e-12 >= config.social_min_duration_s,
        "group_visibility": (
            features.group_visible_fraction + 1e-12
            >= config.minimum_group_visible_fraction
        ),
        "member_support": (
            features.minimum_member_support_fraction + 1e-12
            >= config.minimum_member_support_fraction
        ),
        "nnd_contraction": bool(
            social.nnd_drop_bw >= config.social_nnd_drop_bw_min
            or social.nnd_slope_bw_s <= config.social_nnd_slope_bw_s_max
        ),
        "radius_contraction": bool(
            social.radius_drop_bw >= config.social_radius_drop_bw_min
            or social.radius_slope_bw_s <= config.social_radius_slope_bw_s_max
        ),
        "inward_motion": social.inward_motion_fraction >= config.social_inward_fraction_min,
        "active_members": social.active_member_fraction >= config.social_active_fraction_min,
        "active_inward_member": social.active_inward_member_fraction + 1e-12 >= min(
            config.social_active_inward_member_fraction_min,
            1.0 / max(len(features.member_slots), 1),
        ),
        "graph_growth": bool(
            features.graph.edge_gain_rate >= config.social_edge_gain_rate_min
            or social.density_slope_s >= config.social_density_slope_min
            or features.graph.component_merge_count > 0
            or social.largest_component_growth > 0
        ),
    }
    hard = all(checks[name] for name in ("member_count", "duration", "group_visibility", "member_support"))
    # Both spatial contraction axes are required.  This prevents a translating
    # but shape-stable group (group locomotion) from becoming clustering.
    contraction = checks["nnd_contraction"] and checks["radius_contraction"]
    soft_names = (
        "nnd_contraction",
        "radius_contraction",
        "active_members",
        "active_inward_member",
        "graph_growth",
    )
    soft_score = float(np.mean([checks[name] for name in soft_names]))
    accepted = (
        hard
        and contraction
        and checks["inward_motion"]
        and checks["active_inward_member"]
        and soft_score + 1e-12 >= config.social_soft_score_min
    )
    # Duration confirms an event but must not make a weak, long formation
    # window outrank a short, clearly observed formation phase.
    score = 1.0 + soft_score if accepted else 0.0
    return float(score), checks


def score_huddling(
    features: GroupWindowFeatures,
    config: CollectiveRuleConfig,
) -> tuple[float, dict[str, bool]]:
    """Apply compact-steady-state evidence and reject violent close contact."""

    huddle = features.huddle
    checks = {
        "member_count": len(features.member_slots) >= config.min_members,
        "duration": features.duration_s + 1e-12 >= config.huddle_min_duration_s,
        "group_visibility": (
            features.huddle_group_visible_fraction + 1e-12
            >= config.huddle_min_group_visible_fraction
        ),
        "member_support": (
            features.huddle_minimum_member_support_fraction + 1e-12
            >= config.huddle_min_member_support_fraction
        ),
        "compact_nnd": huddle.nnd_median_bw <= config.huddle_nnd_bw_max,
        "compact_radius": huddle.group_radius_bw <= config.huddle_radius_bw_max,
        "stable_nnd": huddle.nnd_slope_abs_bw_s <= config.huddle_nnd_slope_abs_max,
        "stable_radius": huddle.radius_slope_abs_bw_s <= config.huddle_radius_slope_abs_max,
        "bbox_overlap": huddle.bbox_iom_mean >= config.huddle_bbox_iom_min,
        "low_motion": huddle.low_motion_fraction >= config.huddle_low_motion_fraction_min,
        "contact_density": huddle.contact_graph_density >= config.huddle_contact_density_min,
        "proximity_density": (
            huddle.proximity_graph_density >= config.huddle_proximity_density_min
        ),
        "graph_stability": huddle.graph_jaccard >= config.huddle_graph_stability_min,
        "not_attack": bool(
            huddle.speed_p90_cm_s <= config.huddle_attack_speed_p90_max_cm_s
            and huddle.group_internal_motion_cm_s
            <= config.huddle_attack_relative_motion_max_cm_s
        ),
    }
    hard = all(
        checks[name]
        for name in ("member_count", "duration", "group_visibility", "member_support", "not_attack")
    )
    compact = checks["compact_nnd"] and checks["compact_radius"]
    soft_names = (
        "compact_nnd",
        "compact_radius",
        "stable_nnd",
        "stable_radius",
        "bbox_overlap",
        "low_motion",
        "contact_density",
        "proximity_density",
        "graph_stability",
    )
    soft_score = float(np.mean([checks[name] for name in soft_names]))
    accepted = hard and compact and soft_score + 1e-12 >= config.huddle_soft_score_min
    # Duration is a minimum gate, not a ranking bonus: a later long stable
    # window should not dominate a preceding formation state.
    score = 1.0 + soft_score if accepted else 0.0
    return float(score), checks


def _window_bounds(
    frame_count: int,
    minimum_frames: int,
    fps: float,
    *,
    step_seconds: float = 1.0,
) -> list[tuple[int, int]]:
    if frame_count < minimum_frames:
        return []
    stride = max(1, int(round(max(float(fps), 1.0) * max(float(step_seconds), 0.05))))
    bounds = [
        (start, start + minimum_frames - 1)
        for start in range(0, frame_count - minimum_frames + 1, stride)
    ]
    final = (frame_count - minimum_frames, frame_count - 1)
    if final not in bounds:
        bounds.append(final)
    if frame_count > minimum_frames:
        bounds.append((0, frame_count - 1))
    return sorted(set(bounds))


def collective_candidate_windows(
    *,
    centers_cm: np.ndarray,
    valid: np.ndarray,
    speed_cm_s: np.ndarray,
    velocity_cm_s: np.ndarray,
    member_slots: Sequence[int],
    fps: float,
    body_length_ref_cm: np.ndarray,
    body_width_ref_cm: np.ndarray,
    config: CollectiveRuleConfig,
    keypoints_cm: np.ndarray | None = None,
    boxes_cm: np.ndarray | None = None,
    pair_contact_cache: dict[tuple[int, int, int], PairContactFeatures] | None = None,
) -> tuple[GroupWindowFeatures, ...]:
    """Compute label-free rolling feature windows once for one candidate set."""

    frame_count = len(valid)
    cache: dict[tuple[int, int], GroupWindowFeatures] = {}

    def features_for(bounds: tuple[int, int]) -> GroupWindowFeatures:
        if bounds not in cache:
            cache[bounds] = extract_group_window_features(
                centers_cm=centers_cm,
                valid=valid,
                speed_cm_s=speed_cm_s,
                velocity_cm_s=velocity_cm_s,
                member_slots=member_slots,
                fps=fps,
                body_length_ref_cm=body_length_ref_cm,
                body_width_ref_cm=body_width_ref_cm,
                start_frame=bounds[0],
                end_frame=bounds[1],
                config=config,
                keypoints_cm=keypoints_cm,
                boxes_cm=boxes_cm,
                pair_contact_cache=pair_contact_cache,
            )
        return cache[bounds]

    bounds_to_compute: set[tuple[int, int]] = set()
    social_frames = max(1, int(np.ceil(config.social_min_duration_s * fps)))
    bounds_to_compute.update(
        _window_bounds(frame_count, social_frames, fps, step_seconds=1.0)
    )

    huddle_frames = max(1, int(np.ceil(config.huddle_min_duration_s * fps)))
    huddle_bounds = _window_bounds(frame_count, huddle_frames, fps, step_seconds=0.5)
    analysis_frames = max(huddle_frames, int(np.ceil(min(5.0, frame_count / max(fps, 1e-9)) * fps)))
    if analysis_frames <= frame_count:
        huddle_bounds.extend(
            _window_bounds(frame_count, analysis_frames, fps, step_seconds=1.0)
        )
    bounds_to_compute.update(huddle_bounds)
    return tuple(features_for(bounds) for bounds in sorted(bounds_to_compute))


def classify_collective_windows(
    windows: Sequence[GroupWindowFeatures],
    config: CollectiveRuleConfig,
) -> CollectiveDecision:
    """Resolve group formation and huddling as consecutive states.

    A long, later stable window must not erase an earlier formation phase just
    because its duration produces a larger score. Huddling is an established
    compact state when it is present at clip onset; otherwise, an accepted
    Social-clustering phase remains the clip's primary group behavior. This is
    label-free and operates only on the supplied group windows.
    """

    best_social: tuple[float, GroupWindowFeatures | None, dict[str, bool]] = (0.0, None, {})
    best_huddle: tuple[float, GroupWindowFeatures | None, dict[str, bool]] = (0.0, None, {})
    best_social_rank = -1.0
    best_huddle_rank = -1.0
    huddle_at_onset: tuple[float, GroupWindowFeatures | None, dict[str, bool]] = (
        0.0,
        None,
        {},
    )
    onset_window_max_seconds = config.huddle_min_duration_s + max(
        0.5, config.huddle_min_duration_s * 0.5
    )
    for window_features in windows:
        if window_features.duration_s + 1e-12 >= config.social_min_duration_s:
            score, checks = score_social_clustering(window_features, config)
            rank = score if score > 0.0 else sum(checks.values()) / max(len(checks), 1) * 1e-3
            if rank > best_social_rank:
                best_social_rank = rank
                best_social = score, window_features, checks
        if window_features.duration_s + 1e-12 >= config.huddle_min_duration_s:
            score, checks = score_huddling(window_features, config)
            rank = score if score > 0.0 else sum(checks.values()) / max(len(checks), 1) * 1e-3
            if rank > best_huddle_rank:
                best_huddle_rank = rank
                best_huddle = score, window_features, checks
            if (
                window_features.start_frame == 0
                and window_features.duration_s <= onset_window_max_seconds + 1e-12
                and score > huddle_at_onset[0]
            ):
                huddle_at_onset = score, window_features, checks

    # A cluster that contracts and moves inward is the formation phase even if
    # it later settles into a compact state. Conversely, a group already
    # tightly organized at the start is Huddling, not a new formation event.
    social_score, social_window, social_checks = best_social
    huddle_score, huddle_window, huddle_checks = best_huddle
    if social_score > 0.0:
        if huddle_at_onset[0] > 0.0:
            huddle_score, huddle_window, huddle_checks = huddle_at_onset
            # The phase at onset is the primary state; duration is only a
            # minimum gate and must not let the later formation score win.
            huddle_score = max(huddle_score, social_score + 1e-9)
            huddle_checks = {**huddle_checks, "huddling_at_clip_onset": True}
        else:
            # Keep the evidence window for diagnostics, but do not let a later
            # stable period overwrite the earlier formation label.
            huddle_score = 0.0
            huddle_checks = {**huddle_checks, "huddling_at_clip_onset": False}
    else:
        huddle_checks = {
            **huddle_checks,
            "huddling_at_clip_onset": huddle_at_onset[0] > 0.0,
        }

    return CollectiveDecision(
        social_clustering_score=float(social_score),
        huddling_score=float(huddle_score),
        social_window=social_window,
        huddle_window=huddle_window,
        social_checks=social_checks,
        huddle_checks=huddle_checks,
    )


def classify_collective_candidate(
    **arguments: Any,
) -> CollectiveDecision:
    """Convenience wrapper that extracts and scores a candidate in one call."""

    config = arguments["config"]
    windows = collective_candidate_windows(**arguments)
    return classify_collective_windows(windows, config)


__all__ = [
    "CollectiveDecision",
    "CollectiveRuleConfig",
    "GroupGraphFeatures",
    "GroupWindowFeatures",
    "HuddlingFeatures",
    "PairContactFeatures",
    "SocialClusteringFeatures",
    "bbox_contact_features",
    "classify_collective_candidate",
    "classify_collective_windows",
    "connected_components",
    "estimate_morphology",
    "extract_group_window_features",
    "collective_candidate_windows",
    "graph_density",
    "graph_jaccard",
    "pair_contact_features",
    "robust_body_centers",
    "score_huddling",
    "score_social_clustering",
]
