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
    valid = np.all(np.isfinite(points), axis=1) & np.isfinite(confidence) & (confidence >= 0.10)
    if int(valid.sum()) < 4:
        return None
    center = _weighted_mean(points, confidence, (KP_NECK, KP_LEFT_HIP, KP_RIGHT_HIP))
    if not _finite_point(center):
        center = _weighted_mean(points, confidence, range(KEYPOINTS))
    head = _weighted_mean(points, confidence, (KP_NOSE, KP_LEFT_EAR, KP_RIGHT_EAR))
    body_length = float(np.linalg.norm(points[KP_NOSE] - points[KP_TAIL]))
    if not np.isfinite(body_length) or body_length < 8.0:
        return None
    pose_quality = float(payload.get("pose_quality", np.mean(confidence[valid])))
    box_conf = float(payload.get("box_conf", 0.0))
    return {
        "points": points,
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
) -> tuple[dict[int, dict[str, Any]], bool]:
    detections = _deduplicate(detections)
    assignments: dict[int, dict[str, Any]] = {}
    if not detections:
        missed[:] = missed + 1
        last_velocity[:] = last_velocity * 0.90
        return assignments, initialized

    if not initialized:
        selected = sorted(
            detections, key=lambda item: (float(item["center"][1]), float(item["center"][0]))
        )[:expected_mice]
        for logical_id, detection in enumerate(selected):
            assignments[logical_id] = detection
            last_center[logical_id] = detection["center"]
            last_points[logical_id] = detection["points"]
            last_body[logical_id] = detection["body_length"]
            last_velocity[logical_id] = 0.0
            missed[logical_id] = 0
        missed[:] = np.where(np.arange(expected_mice) < len(selected), missed, 1)
        return assignments, True

    detection_count = len(detections)
    active_ids = np.flatnonzero(np.all(np.isfinite(last_center), axis=1))
    if len(active_ids) == 0:
        missed[:] = missed + 1
        return assignments, initialized

    cost = np.full((len(active_ids), detection_count), 1e6, dtype=float)
    for row, logical_id in enumerate(active_ids):
        prediction = last_center[logical_id] + last_velocity[logical_id] * max(
            int(missed[logical_id]) + 1, 1
        )
        for column, detection in enumerate(detections):
            scale = max(float(last_body[logical_id]), float(detection["body_length"]), 20.0)
            center_cost = float(np.linalg.norm(prediction - detection["center"])) / scale
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
                pose_cost = center_cost
            cost[row, column] = 0.58 * center_cost + 0.42 * pose_cost

    if linear_sum_assignment is not None:
        rows, columns = linear_sum_assignment(cost)
    else:  # pragma: no cover
        rows, columns = [], []
        used: set[int] = set()
        for row in range(cost.shape[0]):
            column = int(
                np.argmin(np.where(np.isin(np.arange(detection_count), list(used)), 1e6, cost[row]))
            )
            if column not in used:
                rows.append(row)
                columns.append(column)
                used.add(column)

    matched_ids: set[int] = set()
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
        gate = 3.2 if int(missed[logical_id]) <= 3 else 5.0
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
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    centers = np.full((total_frames, expected_mice, 2), np.nan, dtype=np.float32)
    keypoints = np.full((total_frames, expected_mice, KEYPOINTS, 2), np.nan, dtype=np.float32)
    confidences = np.zeros((total_frames, expected_mice, KEYPOINTS), dtype=np.float32)
    bboxes = np.full((total_frames, expected_mice, 4), np.nan, dtype=np.float32)
    pose_quality = np.zeros((total_frames, expected_mice), dtype=np.float32)
    body_lengths = np.full((total_frames, expected_mice), np.nan, dtype=np.float32)
    last_center = np.full((expected_mice, 2), np.nan, dtype=float)
    last_points = np.full((expected_mice, KEYPOINTS, 2), np.nan, dtype=float)
    last_body = np.full(expected_mice, np.nan, dtype=float)
    last_velocity = np.zeros((expected_mice, 2), dtype=float)
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
        )
        for logical_id, detection in assignments.items():
            centers[frame, logical_id] = detection["center"]
            keypoints[frame, logical_id] = detection["points"]
            confidences[frame, logical_id] = detection["confidence"]
            bboxes[frame, logical_id] = detection["bbox"]
            pose_quality[frame, logical_id] = detection["pose_quality"]
            body_lengths[frame, logical_id] = detection["body_length"]
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
    }
    return {
        "centers_px": centers,
        "keypoints_px": keypoints,
        "confidences": confidences,
        "bboxes": bboxes,
        "pose_quality": pose_quality,
        "body_lengths_px": body_lengths,
        "valid": valid,
    }, stats
