"""Pose-cache validation, detection normalization and lightweight tracking."""

from __future__ import annotations

import gzip
import json
import logging
import pickle
from pathlib import Path
from typing import Any, Iterator, Mapping

import cv2
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover
    linear_sum_assignment = None

from ..preprocessing.constants import (
    KEYPOINTS,
    KP_LEFT_EAR,
    KP_LEFT_HIP,
    KP_NECK,
    KP_NOSE,
    KP_RIGHT_EAR,
    KP_RIGHT_HIP,
    KP_TAIL,
)
from ..preprocessing.geometry import _box_iou, _finite_point, _weighted_mean

LOGGER = logging.getLogger("mouse_behavior.lightweight_behavior_inference")


def _payload_detection(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    points = np.asarray(payload.get("keypoints_px", []), dtype=float)
    confidence = np.asarray(payload.get("keypoint_conf", []), dtype=float).reshape(-1)
    bbox = np.asarray(payload.get("bbox_xyxy", []), dtype=float).reshape(-1)
    if points.shape != (KEYPOINTS, 2) or confidence.shape != (KEYPOINTS,) or bbox.shape != (4,):
        return None
    bbox_valid = (
        np.all(np.isfinite(bbox))
        and float(bbox[2]) > float(bbox[0])
        and float(bbox[3]) > float(bbox[1])
    )
    if not bbox_valid:
        return None
    point_valid = (
        np.all(np.isfinite(points), axis=1) & np.isfinite(confidence) & (confidence >= 0.10)
    )
    # Keep the box even when an occlusion makes the pose incomplete.  Invalid
    # points are masked before they reach the kinematics/pair geometry code so
    # a low-confidence hallucinated landmark cannot create a contact event.
    clean_points = points.copy()
    clean_points[~point_valid] = np.nan
    center = _weighted_mean(clean_points, confidence, (KP_NECK, KP_LEFT_HIP, KP_RIGHT_HIP))
    if not _finite_point(center):
        center = _weighted_mean(clean_points, confidence, range(KEYPOINTS))
    if not _finite_point(center):
        center = np.asarray(
            [(float(bbox[0]) + float(bbox[2])) * 0.5, (float(bbox[1]) + float(bbox[3])) * 0.5],
            dtype=float,
        )
    head = _weighted_mean(clean_points, confidence, (KP_NOSE, KP_LEFT_EAR, KP_RIGHT_EAR))
    if np.all(np.isfinite(clean_points[[KP_NOSE, KP_TAIL]])):
        body_length = float(np.linalg.norm(clean_points[KP_NOSE] - clean_points[KP_TAIL]))
    else:
        body_length = float(
            max(
                min(float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1])),
                8.0,
            )
        )
    if not np.isfinite(body_length) or body_length < 8.0:
        body_length = float(
            max(
                min(float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1])),
                8.0,
            )
        )
    observed_quality = float(np.mean(confidence[point_valid])) if bool(np.any(point_valid)) else 0.0
    payload_quality = float(payload.get("pose_quality", observed_quality))
    if not np.isfinite(payload_quality):
        payload_quality = observed_quality
    # A box-only detection is useful for tracking and attack evidence, but it
    # must not be reported as a high-quality pose observation for pose-based
    # individual behavior or nose geometry.
    pose_quality = min(payload_quality, observed_quality) if observed_quality else 0.0
    if int(point_valid.sum()) < 4:
        pose_quality *= float(point_valid.sum()) / 4.0
    box_conf = float(payload.get("box_conf", 0.0))
    return {
        "points": clean_points,
        "confidence": confidence,
        "bbox": bbox,
        "center": center,
        "head": head,
        "body_length": body_length,
        "pose_quality": pose_quality if np.isfinite(pose_quality) else 0.0,
        "box_conf": box_conf if np.isfinite(box_conf) else 0.0,
        "score": (box_conf if np.isfinite(box_conf) else 0.0)
        + 0.10 * (pose_quality if np.isfinite(pose_quality) else 0.0),
    }


def _deduplicate(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove only near-identical pose boxes; preserve close distinct mice."""
    ordered = sorted(detections, key=lambda item: float(item["score"]), reverse=True)
    kept: list[dict[str, Any]] = []
    for candidate in ordered:
        duplicate = False
        for previous in kept:
            distance = float(np.linalg.norm(candidate["center"] - previous["center"]))
            scale = max(float(candidate["body_length"]), float(previous["body_length"]), 8.0)
            if distance <= 0.24 * scale and _box_iou(candidate["bbox"], previous["bbox"]) >= 0.18:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return kept


def _solve_track_assignments(
    cost: np.ndarray,
    center_cost: np.ndarray,
    gates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve gated assignments while allowing every track to stay unmatched.

    A rectangular Hungarian solve still forces every row to consume a real
    detection whenever the number of detections is at least the number of
    active tracks. In a crowded frame, one missing mouse plus one false
    detection can therefore start an identity cascade: a good detection is
    stolen by the wrong track and the impossible match is rejected only after
    the solve. Interchangeable dummy columns, one unit of capacity per active
    track, make ``unmatched`` part of the optimization itself. Impossible real
    matches are gated before solving, so they cannot displace a valid match.
    """

    pair_cost = np.asarray(cost, dtype=float)
    distance_cost = np.asarray(center_cost, dtype=float)
    row_gates = np.asarray(gates, dtype=float).reshape(-1)
    if pair_cost.ndim != 2 or distance_cost.shape != pair_cost.shape:
        raise ValueError("track assignment cost matrices must have the same 2-D shape")
    row_count, detection_count = pair_cost.shape
    if row_gates.shape != (row_count,):
        raise ValueError("track assignment gates must contain one value per row")
    if row_count == 0 or detection_count == 0:
        return np.asarray([], dtype=int), np.asarray([], dtype=int)

    forbidden_cost = 1e6
    allowed = (
        np.isfinite(pair_cost)
        & np.isfinite(distance_cost)
        & (pair_cost <= row_gates[:, None])
        & (distance_cost <= row_gates[:, None])
    )
    augmented = np.full(
        (row_count, detection_count + row_count),
        forbidden_cost,
        dtype=float,
    )
    augmented[:, :detection_count] = np.where(allowed, pair_cost, forbidden_cost)
    # ``row_gates`` define which real matches are admissible; they must not
    # also become row-specific unmatched penalties.  Doing so makes a track
    # with the wider reacquisition gate (5.0 after several missed frames)
    # artificially more expensive to leave unmatched than a continuously
    # observed track (3.2).  In a one-box occlusion the global solver then
    # hands the box back and forth between those IDs even when the continuous
    # track has the clearly lower real-match cost.  A common dummy cost keeps
    # the optimization neutral between rows while still preferring every
    # allowed real match over its dummy assignment.
    unmatched_cost = float(np.max(row_gates)) + 1e-6
    augmented[:, detection_count:] = unmatched_cost

    if linear_sum_assignment is None:  # pragma: no cover - SciPy is a declared dependency
        raise RuntimeError(
            "scipy is required for globally optimal gated track assignment; "
            "a greedy fallback can cause identity swaps"
        )
    rows, columns = linear_sum_assignment(augmented)

    real_pairs = [
        (int(row), int(column))
        for row, column in zip(rows, columns)
        if int(column) < detection_count and bool(allowed[int(row), int(column)])
    ]
    if not real_pairs:
        return np.asarray([], dtype=int), np.asarray([], dtype=int)
    real_rows, real_columns = zip(*real_pairs)
    return np.asarray(real_rows, dtype=int), np.asarray(real_columns, dtype=int)


def _iter_cache_records(cache_dir: Path) -> Iterator[Mapping[str, Any]]:
    parts = sorted(cache_dir.glob("yolo_results.*.*.pkl.gz"))
    if not parts:
        raise FileNotFoundError(f"没有找到 YOLO 缓存分块: {cache_dir}")
    for part in parts:
        with gzip.open(part, "rb") as handle:
            records = pickle.load(handle)
        if not isinstance(records, list):
            raise ValueError(f"缓存分块格式错误: {part}")
        for record in records:
            if isinstance(record, Mapping):
                yield record


def _cache_total_frames(cache_dir: Path) -> int:
    status_path = cache_dir / "yolo_results_status.json"
    with status_path.open("r", encoding="utf-8") as handle:
        status = json.load(handle)
    if str(status.get("status")) != "complete":
        raise RuntimeError(f"YOLO 缓存尚未完成: {status_path}")
    total = int(status.get("total_frames", 0))
    if total <= 0 or int(status.get("next_frame", -1)) != total:
        raise RuntimeError(f"YOLO 缓存帧数不完整: {status_path}")
    return total


def _inside_arena(center: Any, polygon: np.ndarray | None, tolerance_px: float = 2.0) -> bool:
    if polygon is None:
        return True
    point = np.asarray(center, dtype=float).reshape(-1)
    if point.size < 2 or not np.all(np.isfinite(point[:2])):
        return False
    contour = np.asarray(polygon, dtype=np.float32).reshape(-1, 1, 2)
    if len(contour) < 3:
        return True
    return float(cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), True)) >= -float(
        max(tolerance_px, 0.0)
    )


def _assign_tracks(
    detections: list[dict[str, Any]],
    last_center: np.ndarray,
    last_points: np.ndarray,
    last_body: np.ndarray,
    last_velocity: np.ndarray,
    missed: np.ndarray,
    initialized: bool,
    expected_mice: int,
    initial_min_detection_score: float = 0.0,
    recent_assignment_gate: float = 3.2,
    reacquisition_assignment_gate: float = 5.0,
    reacquisition_after_missed_frames: int = 3,
) -> tuple[dict[int, dict[str, Any]], bool]:
    detections = _deduplicate(detections)
    assignments: dict[int, dict[str, Any]] = {}
    initialization_columns = [
        column
        for column, detection in enumerate(detections)
        if float(detection.get("score", 0.0)) >= float(initial_min_detection_score)
    ]

    def initialize_track(logical_id: int, detection: dict[str, Any]) -> None:
        assignments[logical_id] = detection
        last_center[logical_id] = detection["center"]
        last_points[logical_id] = detection["points"]
        last_body[logical_id] = detection["body_length"]
        last_velocity[logical_id] = 0.0
        missed[logical_id] = 0

    if not detections:
        missed[:] = missed + 1
        last_velocity[:] = last_velocity * 0.90
        return assignments, initialized

    if not initialized:
        selected = sorted(
            (detections[column] for column in initialization_columns),
            key=lambda item: (float(item["center"][1]), float(item["center"][0])),
        )[:expected_mice]
        for logical_id, detection in enumerate(selected):
            initialize_track(logical_id, detection)
        missed[:] = np.where(np.arange(expected_mice) < len(selected), missed, 1)
        return assignments, bool(selected)

    detection_count = len(detections)
    active_ids = np.flatnonzero(np.all(np.isfinite(last_center), axis=1))
    if len(active_ids) == 0:
        selected = sorted(
            (detections[column] for column in initialization_columns),
            key=lambda item: (float(item["center"][1]), float(item["center"][0])),
        )[:expected_mice]
        for logical_id, detection in enumerate(selected):
            initialize_track(logical_id, detection)
        return assignments, bool(selected)

    cost = np.full((len(active_ids), detection_count), 1e6, dtype=float)
    center_cost = np.full_like(cost, 1e6)
    for row, logical_id in enumerate(active_ids):
        prediction = last_center[logical_id] + last_velocity[logical_id] * max(
            int(missed[logical_id]) + 1, 1
        )
        for column, detection in enumerate(detections):
            scale = max(float(last_body[logical_id]), float(detection["body_length"]), 20.0)
            normalized_center_cost = float(np.linalg.norm(prediction - detection["center"])) / scale
            center_cost[row, column] = normalized_center_cost
            previous_points = last_points[logical_id]
            current_points = detection["points"]
            valid = np.all(np.isfinite(previous_points), axis=1) & np.all(
                np.isfinite(current_points), axis=1
            )
            if int(valid.sum()) >= 4:
                pose_cost = (
                    float(
                        np.mean(
                            np.linalg.norm(previous_points[valid] - current_points[valid], axis=1)
                        )
                    )
                    / scale
                )
            else:
                pose_cost = normalized_center_cost
            cost[row, column] = 0.58 * normalized_center_cost + 0.42 * pose_cost

    try:
        recent_gate = max(float(recent_assignment_gate), 0.1)
    except (TypeError, ValueError, OverflowError):
        recent_gate = 3.2
    try:
        reacquisition_gate = max(float(reacquisition_assignment_gate), 0.1)
    except (TypeError, ValueError, OverflowError):
        reacquisition_gate = 5.0
    try:
        reacquisition_after = max(int(reacquisition_after_missed_frames), 0)
    except (TypeError, ValueError, OverflowError):
        reacquisition_after = 3
    gates = np.asarray(
        [
            recent_gate if int(missed[logical_id]) <= reacquisition_after else reacquisition_gate
            for logical_id in active_ids
        ],
        dtype=float,
    )
    rows, columns = _solve_track_assignments(cost, center_cost, gates)

    matched_ids: set[int] = set()
    matched_columns: set[int] = set()
    for row, column in zip(rows, columns):
        logical_id = int(active_ids[int(row)])
        scale = max(
            float(last_body[logical_id]), float(detections[int(column)]["body_length"]), 20.0
        )
        center_distance = (
            float(
                np.linalg.norm(
                    last_center[logical_id]
                    + last_velocity[logical_id] * max(int(missed[logical_id]) + 1, 1)
                    - detections[int(column)]["center"]
                )
            )
            / scale
        )
        gate = float(gates[int(row)])
        if float(cost[int(row), int(column)]) > gate or center_distance > gate:
            continue
        detection = detections[int(column)]
        assignments[logical_id] = detection
        new_center = np.asarray(detection["center"], dtype=float)
        if np.all(np.isfinite(last_center[logical_id])):
            dt = max(int(missed[logical_id]) + 1, 1)
            last_velocity[logical_id] = (new_center - last_center[logical_id]) / dt
        last_center[logical_id] = new_center
        last_points[logical_id] = detection["points"]
        last_body[logical_id] = detection["body_length"]
        missed[logical_id] = 0
        matched_ids.add(logical_id)
        matched_columns.add(int(column))

    # A short or occluded first frame may initialize fewer logical IDs than
    # the configured mouse count.  The historical implementation only
    # considered IDs with a finite ``last_center`` and therefore left those
    # empty slots unusable for the rest of the video.  Attach unmatched,
    # deduplicated detections to never-initialized slots as soon as they
    # become visible.  Existing IDs retain priority in the Hungarian match.
    free_ids = [
        int(logical_id)
        for logical_id in range(expected_mice)
        if not np.all(np.isfinite(last_center[logical_id]))
    ]
    unmatched_columns = [
        column for column in initialization_columns if column not in matched_columns
    ]
    unmatched_columns.sort(
        key=lambda column: (
            float(detections[column]["center"][1]),
            float(detections[column]["center"][0]),
            -float(detections[column]["score"]),
        )
    )
    for logical_id, column in zip(free_ids, unmatched_columns):
        initialize_track(logical_id, detections[column])
        matched_ids.add(logical_id)
        matched_columns.add(column)

    for logical_id in range(expected_mice):
        if logical_id not in matched_ids:
            missed[logical_id] += 1
            last_velocity[logical_id] *= 0.90
    return assignments, initialized


def _track_cache(
    cache_dir: Path,
    total_frames: int,
    expected_mice: int,
    arena_polygon: np.ndarray | None = None,
    arena_tolerance_px: float = 2.0,
    bbox_occlusion_max_gap_frames: int = 0,
    initial_min_detection_score: float = 0.0,
    recent_assignment_gate: float = 3.2,
    reacquisition_assignment_gate: float = 5.0,
    reacquisition_after_missed_frames: int = 3,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Track cached detections and optionally hold boxes through short gaps.

    The optional box hold is deliberately separate from the normal track
    validity state.  A predicted box can support the occlusion-tolerant
    attack/contact FSM, but it must not be promoted to a real pose observation
    for individual behavior, nose geometry, rendering, or group counts.
    """

    bbox_occlusion_max_gap_frames = max(int(bbox_occlusion_max_gap_frames), 0)
    try:
        initial_min_detection_score = max(float(initial_min_detection_score), 0.0)
    except (TypeError, ValueError, OverflowError):
        initial_min_detection_score = 0.0
    if not np.isfinite(initial_min_detection_score):
        initial_min_detection_score = 0.0
    centers = np.full((total_frames, expected_mice, 2), np.nan, dtype=np.float32)
    keypoints = np.full((total_frames, expected_mice, KEYPOINTS, 2), np.nan, dtype=np.float32)
    confidences = np.zeros((total_frames, expected_mice, KEYPOINTS), dtype=np.float32)
    bboxes = np.full((total_frames, expected_mice, 4), np.nan, dtype=np.float32)
    bbox_observed = np.zeros((total_frames, expected_mice), dtype=bool)
    bbox_imputed = np.zeros((total_frames, expected_mice), dtype=bool)
    pose_quality = np.zeros((total_frames, expected_mice), dtype=np.float32)
    body_lengths = np.full((total_frames, expected_mice), np.nan, dtype=np.float32)
    last_center = np.full((expected_mice, 2), np.nan, dtype=float)
    last_points = np.full((expected_mice, KEYPOINTS, 2), np.nan, dtype=float)
    last_body = np.full(expected_mice, np.nan, dtype=float)
    last_velocity = np.zeros((expected_mice, 2), dtype=float)
    last_bbox = np.full((expected_mice, 4), np.nan, dtype=float)
    last_bbox_center = np.full((expected_mice, 2), np.nan, dtype=float)
    last_bbox_size = np.full((expected_mice, 2), np.nan, dtype=float)
    last_bbox_velocity = np.zeros((expected_mice, 2), dtype=float)
    last_bbox_frame = np.full(expected_mice, -1, dtype=int)
    missed = np.full(expected_mice, 999, dtype=np.int32)
    initialized = False
    raw_counts: list[int] = []
    frame_seen = 0
    for record in _iter_cache_records(cache_dir):
        frame = int(record.get("frame", -1))
        if frame < 0 or frame >= total_frames:
            continue
        payloads = record.get("pose_detections", [])
        detections = [
            detection
            for payload in payloads
            if isinstance(payload, Mapping)
            for detection in [_payload_detection(payload)]
            if detection is not None
            and _inside_arena(detection["center"], arena_polygon, arena_tolerance_px)
        ]
        raw_counts.append(len(detections))
        assignments, initialized = _assign_tracks(
            detections,
            last_center,
            last_points,
            last_body,
            last_velocity,
            missed,
            initialized,
            expected_mice,
            initial_min_detection_score,
            recent_assignment_gate,
            reacquisition_assignment_gate,
            reacquisition_after_missed_frames,
        )
        assigned_ids = set(assignments)
        for logical_id, detection in assignments.items():
            centers[frame, logical_id] = detection["center"]
            keypoints[frame, logical_id] = detection["points"]
            confidences[frame, logical_id] = detection["confidence"]
            bboxes[frame, logical_id] = detection["bbox"]
            bbox_observed[frame, logical_id] = True
            pose_quality[frame, logical_id] = detection["pose_quality"]
            body_lengths[frame, logical_id] = detection["body_length"]

            current_bbox = np.asarray(detection["bbox"], dtype=float).reshape(4)
            current_center = np.asarray(
                [
                    (current_bbox[0] + current_bbox[2]) * 0.5,
                    (current_bbox[1] + current_bbox[3]) * 0.5,
                ],
                dtype=float,
            )
            current_size = np.asarray(
                [current_bbox[2] - current_bbox[0], current_bbox[3] - current_bbox[1]],
                dtype=float,
            )
            previous_frame = int(last_bbox_frame[logical_id])
            if (
                previous_frame >= 0
                and np.all(np.isfinite(last_bbox_center[logical_id]))
                and frame > previous_frame
            ):
                elapsed = float(frame - previous_frame)
                last_bbox_velocity[logical_id] = (
                    current_center - last_bbox_center[logical_id]
                ) / elapsed
            else:
                last_bbox_velocity[logical_id] = 0.0
            last_bbox[logical_id] = current_bbox
            last_bbox_center[logical_id] = current_center
            last_bbox_size[logical_id] = current_size
            last_bbox_frame[logical_id] = frame

        # A short box-only hold is used exclusively for a bounded occlusion
        # interval.  Keep the last observed box size and extrapolate only the
        # center with a velocity learned from the same logical ID.  The
        # displacement is clipped to 1.5 body/box scales per missing frame so
        # an ID hand-off cannot create a far-away synthetic trajectory.
        if bbox_occlusion_max_gap_frames > 0 and initialized:
            for logical_id in range(expected_mice):
                if logical_id in assigned_ids:
                    continue
                previous_frame = int(last_bbox_frame[logical_id])
                gap = frame - previous_frame if previous_frame >= 0 else 0
                if gap <= 0 or gap > bbox_occlusion_max_gap_frames:
                    continue
                if not (
                    np.all(np.isfinite(last_bbox_center[logical_id]))
                    and np.all(np.isfinite(last_bbox_size[logical_id]))
                    and np.all(np.isfinite(last_bbox[logical_id]))
                ):
                    continue
                box_scale = max(
                    float(np.min(last_bbox_size[logical_id])),
                    float(last_body[logical_id]) if np.isfinite(last_body[logical_id]) else 0.0,
                    8.0,
                )
                displacement = last_bbox_velocity[logical_id] * float(gap)
                max_displacement = 1.5 * box_scale * float(gap)
                displacement_norm = float(np.linalg.norm(displacement))
                if displacement_norm > max_displacement and displacement_norm > 1e-9:
                    displacement *= max_displacement / displacement_norm
                predicted_center = last_bbox_center[logical_id] + displacement
                half_size = last_bbox_size[logical_id] * 0.5
                predicted_bbox = np.asarray(
                    [
                        predicted_center[0] - half_size[0],
                        predicted_center[1] - half_size[1],
                        predicted_center[0] + half_size[0],
                        predicted_center[1] + half_size[1],
                    ],
                    dtype=float,
                )
                bboxes[frame, logical_id] = predicted_bbox
                bbox_imputed[frame, logical_id] = True
        frame_seen += 1
        if frame_seen == 1 or frame_seen % 1000 == 0 or frame_seen == total_frames:
            LOGGER.info("[cache tracking] %d/%d frames", frame_seen, total_frames)

    valid = np.all(np.isfinite(centers), axis=2)
    stats = {
        "cache_frames_seen": int(frame_seen),
        "raw_detection_min": int(min(raw_counts)) if raw_counts else 0,
        "raw_detection_max": int(max(raw_counts)) if raw_counts else 0,
        "raw_detection_mean": float(np.mean(raw_counts)) if raw_counts else 0.0,
        "track_valid_rate": float(np.mean(valid)),
        "track_valid_frames_mean": float(np.mean(valid.sum(axis=1))),
        "track_valid_frames_min": int(np.min(valid.sum(axis=1))) if len(valid) else 0,
        "track_valid_frames_max": int(np.max(valid.sum(axis=1))) if len(valid) else 0,
        "bbox_occlusion_max_gap_frames": int(bbox_occlusion_max_gap_frames),
        "initial_min_detection_score": float(initial_min_detection_score),
        "recent_assignment_gate": float(recent_assignment_gate),
        "reacquisition_assignment_gate": float(reacquisition_assignment_gate),
        "reacquisition_after_missed_frames": int(reacquisition_after_missed_frames),
        "bbox_observed_count": int(bbox_observed.sum()),
        "bbox_imputed_count": int(bbox_imputed.sum()),
    }
    return {
        "centers_px": centers,
        "keypoints_px": keypoints,
        "confidences": confidences,
        "bboxes": bboxes,
        "bbox_observed": bbox_observed,
        "bbox_imputed": bbox_imputed,
        "pose_quality": pose_quality,
        "body_lengths_px": body_lengths,
        "valid": valid,
    }, stats
