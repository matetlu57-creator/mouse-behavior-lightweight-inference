#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Mouse Behavior Lightweight Inference entry point.

Analyze one completed YOLO cache without rerunning the heavy tracker.

This is intentionally a bounded, single-video fallback for long Windows
videos.  It reads only ``yolo_precompute`` records for the requested video,
keeps at most ``expected_mice`` tracks with position+keypoint matching, builds
the pair-wise kinematics required by the v1.43 standard behavior engine, and
then runs the same standard chase/attack FSM and thresholds offline.

It does not claim to replace the full occlusion/ReID pipeline.  The output
metadata explicitly records that limitation so a detected event can be
reviewed separately from a full-pipeline result.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import math
import pickle
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import cv2
import numpy as np
import pandas as pd
import yaml

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover - a deterministic greedy fallback is below
    linear_sum_assignment = None

from . import adaptive_arena_boundary as arena_boundary
from .parallel_behavior_fsm import ParallelBehaviorFSM
from . import standard_behavior_engine as behavior_engine
from .annotation_website_export import export_complete_video_package
from .config import load_config
from .logging_config import configure_logging
from .utils.rolling import rolling_corr as _rolling_corr
from .utils.rolling import rolling_sum as _rolling_sum
from .utils.timer import Timer


LOGGER = logging.getLogger(__name__)


PROJECT_NAME = "mouse-behavior-lightweight-inference"

# The lightweight ethogram is intentionally separate from the legacy four-class
# chase/attack compatibility output.  These labels are the behavior names used
# by the Beiyi example set and are inferred from the same per-video tracks.
SOCIAL_BEHAVIORS = (
    "together",
    "approach",
    "chase",
    "avoidance",
    "attack",
    "nose_head_contact",
    "nose_tail_contact",
)
GROUP_BEHAVIORS = ("huddle", "isolation")
INDIVIDUAL_BEHAVIORS = ("running", "walking", "stationary")
EXTENDED_BEHAVIORS = SOCIAL_BEHAVIORS + GROUP_BEHAVIORS + INDIVIDUAL_BEHAVIORS
BEHAVIOR_NAMES_ZH = {
    "together": "一起",
    "approach": "接近",
    "chase": "追逐",
    "avoidance": "回避",
    "attack": "攻击",
    "nose_head_contact": "鼻头接触",
    "nose_tail_contact": "鼻尾接触",
    "huddle": "扎堆",
    "isolation": "孤立",
    "running": "奔跑",
    "walking": "行走",
    "stationary": "静止",
}


KP_NOSE = 0
KP_LEFT_EAR = 1
KP_RIGHT_EAR = 2
KP_NECK = 3
KP_LEFT_HIP = 4
KP_RIGHT_HIP = 5
KP_TAIL = 6
KEYPOINTS = 7

# This is the exact eight-edge graph used by the user's reference image:
# nose -> ears, ears -> neck, neck -> hips, hips -> tail.
SKELETON_EDGES = (
    (KP_NOSE, KP_LEFT_EAR),
    (KP_NOSE, KP_RIGHT_EAR),
    (KP_LEFT_EAR, KP_NECK),
    (KP_RIGHT_EAR, KP_NECK),
    (KP_NECK, KP_LEFT_HIP),
    (KP_NECK, KP_RIGHT_HIP),
    (KP_LEFT_HIP, KP_TAIL),
    (KP_RIGHT_HIP, KP_TAIL),
)


def _finite_point(value: Any) -> bool:
    arr = np.asarray(value, dtype=float).reshape(-1)
    return bool(arr.size >= 2 and np.all(np.isfinite(arr[:2])))


def _unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-9 and np.isfinite(norm) else np.full(2, np.nan)


def _cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity for [..., 2] arrays, with zero for invalid vectors."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    dot = np.sum(a * b, axis=-1)
    denom = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    out = np.zeros_like(dot, dtype=float)
    valid = np.isfinite(dot) & np.isfinite(denom) & (denom > 1e-9)
    out[valid] = np.clip(dot[valid] / denom[valid], -1.0, 1.0)
    return out


def _angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    cosine = _cosine(a, b)
    valid = np.all(np.isfinite(a), axis=-1) & np.all(np.isfinite(b), axis=-1)
    out = np.zeros(cosine.shape, dtype=float)
    out[valid] = np.degrees(np.arccos(np.clip(cosine[valid], -1.0, 1.0)))
    return out


def _weighted_mean(points: np.ndarray, confidence: np.ndarray, indices: Sequence[int]) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    confidence = np.asarray(confidence, dtype=float).reshape(-1)
    valid_indices = [
        int(index)
        for index in indices
        if int(index) < len(points)
        and int(index) < len(confidence)
        and np.all(np.isfinite(points[int(index)]))
        and np.isfinite(confidence[int(index)])
        and float(confidence[int(index)]) >= 0.10
    ]
    if not valid_indices:
        return np.full(2, np.nan, dtype=float)
    values = points[valid_indices]
    weights = np.maximum(confidence[valid_indices], 0.01)
    return np.average(values, axis=0, weights=weights).astype(float)


def _box_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    if a.size < 4 or b.size < 4 or not np.all(np.isfinite(a[:4])) or not np.all(np.isfinite(b[:4])):
        return 0.0
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - inter
    return float(inter / union) if union > 1e-9 else 0.0


def _payload_detection(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    points = np.asarray(payload.get("keypoints_px", []), dtype=float)
    confidence = np.asarray(payload.get("keypoint_conf", []), dtype=float).reshape(-1)
    bbox = np.asarray(payload.get("bbox_xyxy", []), dtype=float).reshape(-1)
    if points.shape != (KEYPOINTS, 2) or confidence.shape != (KEYPOINTS,) or bbox.shape != (4,):
        return None
    valid = (
        np.all(np.isfinite(points), axis=1)
        & np.isfinite(confidence)
        & (confidence >= 0.10)
    )
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
        "score": (box_conf if np.isfinite(box_conf) else 0.0) + 0.10 * (pose_quality if np.isfinite(pose_quality) else 0.0),
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


def _resolve_boundary_reuse_path(value: Any, output_dir: Path, config_path: Path) -> Path:
    raw = str(value or "").strip()
    path = Path(raw)
    if path.is_absolute():
        return path
    output_candidate = output_dir / path
    if output_candidate.exists():
        return output_candidate
    return config_path.parent / path


def _prepare_video_arena_boundary(
    video_path: Path,
    cache_dir: Path,
    output_dir: Path,
    config_path: Path,
    config: Mapping[str, Any],
    width: int,
    height: int,
    max_frames: int | None = None,
) -> tuple[arena_boundary.ArenaBoundaryResult | None, np.ndarray | None]:
    """Learn/reuse this video's boundary and persist auditable artifacts.

    The default is always per-video learning.  A JSON can only be reused when
    the caller explicitly sets ``reuse_boundary_json``; the boundary module
    then checks the saved source path, resolution, and file fingerprint.
    """

    arena_cfg = dict(config.get("adaptive_arena", {}))
    if not bool(arena_cfg.get("enabled", True)):
        return None, None
    configured_polygon = (
        dict(config.get("detector_first", {}))
        .get("arena_mask", {})
        .get("polygon", [])
    )
    reuse_value = str(arena_cfg.get("reuse_boundary_json", "") or "").strip()
    if reuse_value:
        reuse_path = _resolve_boundary_reuse_path(reuse_value, output_dir, config_path)
        result = arena_boundary.load_boundary_json(
            reuse_path,
            width=width,
            height=height,
            source_video=video_path,
            require_video_match=bool(arena_cfg.get("reuse_require_video_match", True)),
        )
        heatmap = np.zeros(
            (
                max(int(math.ceil(height / max(result.heatmap_cell_px, 1))), 1),
                max(int(math.ceil(width / max(result.heatmap_cell_px, 1))), 1),
            ),
            dtype=np.float32,
        )
    else:
        records: Iterable[Mapping[str, Any]] = _iter_cache_records(cache_dir)
        if max_frames is not None:
            frame_limit = max(int(max_frames), 1)
            source_records = records

            def limited_records() -> Iterator[Mapping[str, Any]]:
                for record in source_records:
                    frame = int(record.get("frame", -1))
                    if frame >= frame_limit:
                        break
                    yield record

            records = limited_records()
        result, heatmap = arena_boundary.learn_from_yolo_records(
            records,
            width=width,
            height=height,
            config=arena_cfg,
            configured_polygon=configured_polygon,
            source_video=video_path,
        )

    json_path = output_dir / "阶段一_自适应笼界.json"
    png_path = output_dir / "阶段一_运动热力图与笼界.png"
    comparison_path = output_dir / "阶段一_原视频帧叠加笼界.png"
    arena_boundary.save_boundary_artifacts(
        result,
        heatmap,
        json_path,
        png_path,
        comparison_path,
    )
    return result, heatmap


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
        selected = sorted(detections, key=lambda item: (float(item["center"][1]), float(item["center"][0])))[:expected_mice]
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
        prediction = last_center[logical_id] + last_velocity[logical_id] * max(int(missed[logical_id]) + 1, 1)
        for column, detection in enumerate(detections):
            scale = max(float(last_body[logical_id]), float(detection["body_length"]), 20.0)
            center_cost = float(np.linalg.norm(prediction - detection["center"])) / scale
            previous_points = last_points[logical_id]
            current_points = detection["points"]
            valid = np.all(np.isfinite(previous_points), axis=1) & np.all(np.isfinite(current_points), axis=1)
            if int(valid.sum()) >= 4:
                pose_cost = float(np.mean(np.linalg.norm(previous_points[valid] - current_points[valid], axis=1))) / scale
            else:
                pose_cost = center_cost
            cost[row, column] = 0.58 * center_cost + 0.42 * pose_cost

    if linear_sum_assignment is not None:
        rows, columns = linear_sum_assignment(cost)
    else:  # pragma: no cover
        rows, columns = [], []
        used: set[int] = set()
        for row in range(cost.shape[0]):
            column = int(np.argmin(np.where(np.isin(np.arange(detection_count), list(used)), 1e6, cost[row])))
            if column not in used:
                rows.append(row)
                columns.append(column)
                used.add(column)

    matched_ids: set[int] = set()
    for row, column in zip(rows, columns):
        logical_id = int(active_ids[int(row)])
        scale = max(float(last_body[logical_id]), float(detections[int(column)]["body_length"]), 20.0)
        center_distance = float(np.linalg.norm(
            last_center[logical_id] + last_velocity[logical_id] * max(int(missed[logical_id]) + 1, 1)
            - detections[int(column)]["center"]
        )) / scale
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


def _ema_smooth(values: np.ndarray, valid: np.ndarray, alpha: float = 0.70) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    out = np.full_like(values, np.nan, dtype=float)
    tracks = values.shape[1]
    previous = np.full(values.shape[1:], np.nan, dtype=float)
    for frame in range(values.shape[0]):
        for track in range(tracks):
            if not bool(valid[frame, track]):
                continue
            current = values[frame, track]
            if not np.all(np.isfinite(current)):
                continue
            if np.all(np.isfinite(previous[track])):
                previous[track] = alpha * current + (1.0 - alpha) * previous[track]
            else:
                previous[track] = current
            out[frame, track] = previous[track]
    return out


def _ema_smooth_keypoints(values: np.ndarray, valid: np.ndarray, alpha: float = 0.70) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    out = np.full_like(values, np.nan, dtype=float)
    tracks = values.shape[1]
    previous = np.full(values.shape[1:], np.nan, dtype=float)
    for frame in range(values.shape[0]):
        for track in range(tracks):
            if not bool(valid[frame, track]):
                continue
            for point in range(values.shape[2]):
                current = values[frame, track, point]
                if not np.all(np.isfinite(current)):
                    continue
                if np.all(np.isfinite(previous[track, point])):
                    previous[track, point] = alpha * current + (1.0 - alpha) * previous[track, point]
                else:
                    previous[track, point] = current
                out[frame, track, point] = previous[track, point]
    return out


def _pose_deformation_energy(
    keypoints_cm: np.ndarray,
    centers_cm: np.ndarray,
    heading: np.ndarray,
    body_cm: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """Measure frame-to-frame shape change in each mouse's body frame.

    Translation, global rotation and apparent scale are removed before the
    comparison.  Ordinary nose contact should therefore remain close to zero
    when the two mice merely hold a stable posture, while a bite/lunge or
    wrestling motion can contribute a real internal pose change.  The
    lightweight path previously exported zeros here, which disabled the
    standard engine's anti-contact grapple evidence entirely.
    """
    keypoints_cm = np.asarray(keypoints_cm, dtype=float)
    centers_cm = np.asarray(centers_cm, dtype=float)
    heading = np.asarray(heading, dtype=float)
    body_cm = np.asarray(body_cm, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    frames, mice = valid.shape
    body_pose = np.full((frames, mice, KEYPOINTS, 2), np.nan, dtype=float)
    pose_valid = np.zeros((frames, mice, KEYPOINTS), dtype=bool)

    for frame in range(frames):
        for mouse in range(mice):
            if not valid[frame, mouse]:
                continue
            center = centers_cm[frame, mouse]
            forward = heading[frame, mouse]
            scale = body_cm[frame, mouse]
            points = keypoints_cm[frame, mouse]
            if (
                not np.all(np.isfinite(center))
                or not np.all(np.isfinite(forward))
                or not np.isfinite(scale)
                or scale < 1e-3
            ):
                continue
            lateral = np.asarray([-forward[1], forward[0]], dtype=float)
            relative = (points - center) / scale
            good = np.all(np.isfinite(relative), axis=1)
            if not np.any(good):
                continue
            body_pose[frame, mouse, good, 0] = relative[good] @ forward
            body_pose[frame, mouse, good, 1] = relative[good] @ lateral
            pose_valid[frame, mouse, good] = True

    deformation = np.zeros((frames, mice), dtype=float)
    for frame in range(1, frames):
        common = pose_valid[frame] & pose_valid[frame - 1]
        for mouse in np.flatnonzero(np.sum(common, axis=1) >= 3):
            delta = body_pose[frame, mouse, common[mouse]] - body_pose[frame - 1, mouse, common[mouse]]
            deformation[frame, mouse] = float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))
    deformation[~valid] = 0.0
    return deformation


def _kinematics(tracks: Mapping[str, np.ndarray], fps: float, body_length_cm: float = 8.0) -> dict[str, np.ndarray | float]:
    valid = np.asarray(tracks["valid"], dtype=bool)
    pose_quality = np.asarray(tracks["pose_quality"], dtype=float)
    raw_kp = np.asarray(tracks["keypoints_px"], dtype=float)
    raw_centers = np.asarray(tracks["centers_px"], dtype=float)
    body_values = np.asarray(tracks["body_lengths_px"], dtype=float)
    body_values = body_values[np.isfinite(body_values) & (body_values >= 10.0) & (body_values <= 300.0)]
    reference_body_px = float(np.median(body_values)) if body_values.size else 60.0
    cm_per_pixel = float(body_length_cm / max(reference_body_px, 1e-6))
    smooth_kp = _ema_smooth_keypoints(raw_kp, valid, alpha=0.70)
    smooth_centers = _ema_smooth(raw_centers, valid, alpha=0.70)
    frames, mice = valid.shape
    keypoints_cm = smooth_kp * cm_per_pixel
    centers_cm = smooth_centers * cm_per_pixel
    head_cm = np.full((frames, mice, 2), np.nan, dtype=float)
    heading = np.full((frames, mice, 2), np.nan, dtype=float)
    body_cm = np.full((frames, mice), np.nan, dtype=float)
    for frame in range(frames):
        for mouse in range(mice):
            if not valid[frame, mouse]:
                continue
            points = keypoints_cm[frame, mouse]
            head_cm[frame, mouse] = np.nanmean(points[[KP_NOSE, KP_LEFT_EAR, KP_RIGHT_EAR]], axis=0)
            center = np.nanmean(points[[KP_NECK, KP_LEFT_HIP, KP_RIGHT_HIP]], axis=0)
            if np.all(np.isfinite(center)):
                centers_cm[frame, mouse] = center
            axis = points[KP_NOSE] - points[KP_NECK]
            if np.all(np.isfinite(axis)):
                heading[frame, mouse] = _unit(axis)
            if np.all(np.isfinite(points[[KP_NOSE, KP_TAIL]])):
                body_cm[frame, mouse] = float(np.linalg.norm(points[KP_NOSE] - points[KP_TAIL]))

    velocity = np.zeros((frames, mice, 2), dtype=float)
    nose_velocity = np.zeros((frames, mice, 2), dtype=float)
    acceleration = np.zeros((frames, mice), dtype=float)
    angular_speed = np.zeros((frames, mice), dtype=float)
    last_valid = np.full(mice, -1, dtype=int)
    for frame in range(frames):
        for mouse in range(mice):
            if not valid[frame, mouse]:
                continue
            previous = int(last_valid[mouse])
            if previous >= 0 and frame - previous <= 3 and np.all(np.isfinite(centers_cm[[previous, frame], mouse])):
                dt = max(frame - previous, 1) / fps
                velocity[frame, mouse] = (centers_cm[frame, mouse] - centers_cm[previous, mouse]) / dt
                if np.all(np.isfinite(keypoints_cm[[previous, frame], mouse, KP_NOSE])):
                    nose_velocity[frame, mouse] = (keypoints_cm[frame, mouse, KP_NOSE] - keypoints_cm[previous, mouse, KP_NOSE]) / dt
                if previous > 0:
                    acceleration[frame, mouse] = abs(
                        np.linalg.norm(velocity[frame, mouse]) - np.linalg.norm(velocity[previous, mouse])
                    ) / dt
                if np.all(np.isfinite(heading[[previous, frame], mouse])):
                    angular_speed[frame, mouse] = float(_angle_deg(heading[frame, mouse], heading[previous, mouse])) / dt
            last_valid[mouse] = frame
    speed = np.linalg.norm(velocity, axis=2)
    nose_speed = np.linalg.norm(nose_velocity, axis=2)
    # The original speed is intentionally kept for chase/attack compatibility.
    # Individual labels use a longer-baseline displacement speed, followed by a
    # rolling median, so one identity-assignment jump cannot turn stationary
    # into running and a one-frame detector jitter cannot turn approach into
    # contact.
    behavior_speed_raw = np.full((frames, mice), np.nan, dtype=float)
    behavior_lookback = max(int(round(fps * 0.15)), 1)
    if behavior_lookback < frames:
        for mouse in range(mice):
            current = centers_cm[behavior_lookback:, mouse]
            previous = centers_cm[:-behavior_lookback, mouse]
            valid_motion = (
                valid[behavior_lookback:, mouse]
                & valid[:-behavior_lookback, mouse]
                & np.all(np.isfinite(current), axis=1)
                & np.all(np.isfinite(previous), axis=1)
            )
            if np.any(valid_motion):
                target_values = behavior_speed_raw[behavior_lookback:, mouse]
                target_values[valid_motion] = (
                    np.linalg.norm(current[valid_motion] - previous[valid_motion], axis=1)
                    / (behavior_lookback / max(fps, 1e-9))
                )
    behavior_speed = np.zeros((frames, mice), dtype=float)
    behavior_window = max(int(round(fps * 0.30)), 3)
    for mouse in range(mice):
        robust = _rolling_quantile(behavior_speed_raw[:, mouse], behavior_window, 0.50)
        behavior_speed[:, mouse] = np.nan_to_num(robust, nan=0.0, posinf=0.0, neginf=0.0)
    pose_deformation = _pose_deformation_energy(
        keypoints_cm,
        centers_cm,
        heading,
        body_cm,
        valid,
    )
    velocity[~valid] = 0.0
    speed[~valid] = 0.0
    acceleration[~valid] = 0.0
    angular_speed[~valid] = 0.0
    nose_speed[~valid] = 0.0
    behavior_speed[~valid] = 0.0
    return {
        "valid": valid,
        "pose_quality": pose_quality,
        "keypoints_cm": keypoints_cm,
        "centers_cm": centers_cm,
        "head_cm": head_cm,
        "heading": heading,
        "body_cm": body_cm,
        "velocity": velocity,
        "speed": speed,
        "behavior_speed": behavior_speed,
        "nose_speed": nose_speed,
        "acceleration": acceleration,
        "angular_speed": angular_speed,
        "pose_deformation": pose_deformation,
        "cm_per_pixel": cm_per_pixel,
        "reference_body_px": reference_body_px,
        "reference_body_cm": float(body_length_cm),
    }


def _pair_dataframe(
    metrics: Mapping[str, Any],
    pair_index: int,
    mouse_a: int,
    mouse_b: int,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    fps: float,
    cm_per_pixel: float,
) -> pd.DataFrame:
    valid = np.asarray(metrics["valid_pair"][:, pair_index], dtype=bool)
    distance = np.asarray(metrics["distance"][:, pair_index], dtype=float)
    speed = np.asarray(metrics["speed"], dtype=float)
    nose_speed = np.asarray(metrics["nose_speed"], dtype=float)
    acceleration = np.asarray(metrics["acceleration"], dtype=float)
    angular = np.asarray(metrics["angular_speed"], dtype=float)
    pose_deformation = np.asarray(metrics["pose_deformation"], dtype=float)
    i, j = int(mouse_a), int(mouse_b)
    p = len(valid)
    direction = np.asarray(metrics["direction"][:, pair_index], dtype=float)
    pursuit_ab = np.asarray(metrics["pursuit_ab"][:, pair_index], dtype=float)
    pursuit_ba = np.asarray(metrics["pursuit_ba"][:, pair_index], dtype=float)
    escape_ab = np.asarray(metrics["escape_ab"][:, pair_index], dtype=float)
    escape_ba = np.asarray(metrics["escape_ba"][:, pair_index], dtype=float)
    behind_ab = np.asarray(metrics["behind_ab"][:, pair_index], dtype=bool)
    behind_ba = np.asarray(metrics["behind_ba"][:, pair_index], dtype=bool)
    turn = np.asarray(metrics["turn"], dtype=float)
    corr = np.asarray(metrics["trajectory_corr"][:, pair_index], dtype=float)
    drop = np.asarray(metrics["distance_drop"][:, pair_index], dtype=float)
    nose_body_ab = np.asarray(metrics["nose_body_ab"][:, pair_index], dtype=float)
    nose_body_ba = np.asarray(metrics["nose_body_ba"][:, pair_index], dtype=float)
    nose_tail_ab = np.asarray(metrics["nose_tail_ab"][:, pair_index], dtype=float)
    nose_tail_ba = np.asarray(metrics["nose_tail_ba"][:, pair_index], dtype=float)
    nose_head_ab = np.asarray(metrics["nose_head_ab"][:, pair_index], dtype=float)
    nose_head_ba = np.asarray(metrics["nose_head_ba"][:, pair_index], dtype=float)
    repeated = np.asarray(metrics["repeated_contact"][:, pair_index], dtype=int)
    behavior_speed = np.asarray(metrics["behavior_speed"], dtype=float)
    distance_body_lengths = distance / 8.0
    score_ab = pursuit_ab + escape_ab + 0.15 * speed[:, i]
    score_ba = pursuit_ba + escape_ba + 0.15 * speed[:, j]
    tie = np.abs(score_ab - score_ba) <= 0.05
    selected_ab = score_ab >= score_ba
    selected_actor = np.where(tie, -1, np.where(selected_ab, i, j))
    selected_target = np.where(tie, -1, np.where(selected_ab, j, i))
    selected_nose_body = np.where(selected_ab, nose_body_ab, nose_body_ba)
    selected_turn = np.where(selected_ab, turn[:, j], turn[:, i])
    closing_speed = drop / max(0.30, 1.0 / fps)
    frame = np.arange(p, dtype=int)
    data: dict[str, Any] = {
        "frame": frame,
        "time_s": frame / fps,
        "pair_key": f"{i}_{j}",
        "mouse_a_id": i,
        "mouse_b_id": j,
        "mouse_a_raw_track_id": i,
        "mouse_b_raw_track_id": j,
        "mouse_a_track_state": "tracked",
        "mouse_b_track_state": "tracked",
        "valid_pair": valid,
        "center_distance_cm": distance,
        "center_distance_body_lengths": distance_body_lengths,
        "head_distance_cm": np.asarray(metrics["head_distance"][:, pair_index], dtype=float),
        "mouse_a_speed_cm_s": speed[:, i],
        "mouse_b_speed_cm_s": speed[:, j],
        "mouse_a_behavior_speed_cm_s": behavior_speed[:, i],
        "mouse_b_behavior_speed_cm_s": behavior_speed[:, j],
        "pose_pair_quality": np.asarray(metrics["pose_pair_quality"][:, pair_index], dtype=float),
        "identity_pair_quality": valid.astype(float),
        "pair_wall_jump_excluded": np.zeros(p, dtype=bool),
        "cluster_attack_hint": np.zeros(p, dtype=bool),
        "cluster_overlap_iou": np.zeros(p, dtype=float),
        "cluster_motion_bl_per_frame": np.zeros(p, dtype=float),
        "cluster_active_frames": np.zeros(p, dtype=int),
        "selected_actor_id": selected_actor,
        "selected_target_id": selected_target,
        "selected_nose_body_distance_cm": selected_nose_body,
        "selected_target_turn_angle_deg": selected_turn,
        "selected_distance_drop_cm": drop,
        "selected_closing_speed_cm_s": closing_speed,
        "selected_actor_speed_cm_s": np.where(selected_ab, speed[:, i], speed[:, j]),
        "selected_target_speed_cm_s": np.where(selected_ab, speed[:, j], speed[:, i]),
        "selected_actor_behavior_speed_cm_s": np.where(
            selected_ab,
            behavior_speed[:, i],
            behavior_speed[:, j],
        ),
        "selected_target_behavior_speed_cm_s": np.where(
            selected_ab,
            behavior_speed[:, j],
            behavior_speed[:, i],
        ),
        "selected_target_escape_alignment": np.where(selected_ab, escape_ab, escape_ba),
        "selected_actor_pursuit_alignment": np.where(selected_ab, pursuit_ab, pursuit_ba),
        "selected_nose_tail_distance_cm": np.where(selected_ab, nose_tail_ab, nose_tail_ba),
        "selected_nose_head_distance_cm": np.where(selected_ab, nose_head_ab, nose_head_ba),
        "selected_weak_chase_score": np.zeros(p, dtype=float),
        "selected_strong_chase_score": np.zeros(p, dtype=float),
        "selected_weak_attack_evidence": np.zeros(p, dtype=float),
        "selected_strong_attack_evidence": np.zeros(p, dtype=float),
        "weak_contact": (np.minimum(nose_body_ab, nose_body_ba) <= 4.0),
        "strong_contact": (np.minimum(nose_body_ab, nose_body_ba) <= 3.0),
        "weak_potential_attack": (np.minimum(nose_body_ab, nose_body_ba) <= 4.0),
        "strong_potential_attack": (np.minimum(nose_body_ab, nose_body_ba) <= 3.0),
        "weak_raw_chase": np.zeros(p, dtype=bool),
        "strong_raw_chase": np.zeros(p, dtype=bool),
        "weak_raw_attack": np.zeros(p, dtype=bool),
        "strong_raw_attack": np.zeros(p, dtype=bool),
        "scale_mode": "body_length",
        "cm_per_pixel": cm_per_pixel,
    }
    for level in ("weak", "strong"):
        for provider in ("strict_chase", "window_chase", "near_recovery_chase", "close_follow_chase", "strict_attack", "impulse_attack", "grapple_attack", "occlusion_overlap_attack"):
            data[f"{level}_{provider}"] = np.zeros(p, dtype=bool)

    direction_columns = {
        "a_to_b": (i, j, pursuit_ab, escape_ab, behind_ab, turn[:, j], nose_body_ab, nose_tail_ab, nose_head_ab),
        "b_to_a": (j, i, pursuit_ba, escape_ba, behind_ba, turn[:, i], nose_body_ba, nose_tail_ba, nose_head_ba),
    }
    for prefix, (actor, target, pursuit, escape, behind, target_turn, nose_body, nose_tail, nose_head) in direction_columns.items():
        actor_speed = speed[:, actor]
        target_speed = speed[:, target]
        actor_nose_speed = nose_speed[:, actor]
        target_nose_speed = nose_speed[:, target]
        actor_acceleration = acceleration[:, actor]
        target_acceleration = acceleration[:, target]
        actor_angular = angular[:, actor]
        target_angular = angular[:, target]
        data.update({
            f"{prefix}_actor_speed_cm_s": actor_speed,
            f"{prefix}_target_speed_cm_s": target_speed,
            f"{prefix}_actor_behavior_speed_cm_s": behavior_speed[:, actor],
            f"{prefix}_target_behavior_speed_cm_s": behavior_speed[:, target],
            f"{prefix}_actor_acceleration_cm_s2": actor_acceleration,
            f"{prefix}_target_acceleration_cm_s2": target_acceleration,
            f"{prefix}_actor_nose_speed_cm_s": actor_nose_speed,
            f"{prefix}_target_nose_speed_cm_s": target_nose_speed,
            f"{prefix}_actor_head_relative_speed_cm_s": np.maximum(actor_nose_speed - actor_speed, 0.0),
            f"{prefix}_actor_angular_speed_deg_s": actor_angular,
            f"{prefix}_target_angular_speed_deg_s": target_angular,
            f"{prefix}_direction_similarity": direction,
            f"{prefix}_pursuit_alignment": pursuit,
            f"{prefix}_target_escape_alignment": escape,
            f"{prefix}_trajectory_correlation": corr,
            f"{prefix}_closing_speed_cm_s": closing_speed,
            f"{prefix}_center_distance_body_lengths": distance_body_lengths,
            f"{prefix}_actor_behind_target": behind,
            f"{prefix}_behind_score": behind.astype(float),
            f"{prefix}_target_turn_angle_deg": target_turn,
            f"{prefix}_nose_body_distance_cm": nose_body,
            f"{prefix}_nose_tail_distance_cm": nose_tail,
            f"{prefix}_nose_head_distance_cm": nose_head,
            f"{prefix}_actor_pose_deformation_energy": pose_deformation[:, actor],
            f"{prefix}_target_pose_deformation_energy": pose_deformation[:, target],
        })
    return pd.DataFrame(data)


def _pair_metrics(
    kin: Mapping[str, Any],
    fps: float,
    pair_indices: np.ndarray | Sequence[int] | None = None,
    frame_mask: np.ndarray | None = None,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Build pair-wise features for the requested logical pair columns.

    ``pair_indices`` refers to the stable ordering returned by
    ``np.triu_indices(mice, k=1)``.  Keeping this optional preserves the old
    all-pair API for callers that need it, while the lightweight analyzer can
    now run the expensive nose/trajectory features only for spatially useful
    pairs selected by its prefilter.  ``frame_mask`` optionally limits those
    expensive pair features to padded interaction windows while retaining the
    selected pair's full output timeline for the temporal behavior engine.
    """
    centers = np.asarray(kin["centers_cm"], dtype=float)
    heads = np.asarray(kin["head_cm"], dtype=float)
    kp = np.asarray(kin["keypoints_cm"], dtype=float)
    heading = np.asarray(kin["heading"], dtype=float)
    velocity = np.asarray(kin["velocity"], dtype=float)
    valid = np.asarray(kin["valid"], dtype=bool)
    speed = np.asarray(kin["speed"], dtype=float)
    nose_speed = np.asarray(kin["nose_speed"], dtype=float)
    acceleration = np.asarray(kin["acceleration"], dtype=float)
    angular_speed = np.asarray(kin["angular_speed"], dtype=float)
    frames, mice = valid.shape
    all_pair_i, all_pair_j = np.triu_indices(mice, k=1)
    if pair_indices is None:
        selected_pair_indices = np.arange(len(all_pair_i), dtype=int)
    else:
        selected_pair_indices = np.asarray(pair_indices, dtype=int).reshape(-1)
        if np.any(selected_pair_indices < 0) or np.any(selected_pair_indices >= len(all_pair_i)):
            raise IndexError("pair_indices contains an invalid all-pair column index")
    pair_i = all_pair_i[selected_pair_indices]
    pair_j = all_pair_j[selected_pair_indices]
    pairs = len(pair_i)
    raw_valid_pair = valid[:, pair_i] & valid[:, pair_j]
    if frame_mask is None:
        frame_mask = np.ones((frames, pairs), dtype=bool)
    else:
        frame_mask = np.asarray(frame_mask, dtype=bool)
        if frame_mask.shape != (frames, pairs):
            raise ValueError(
                "frame_mask must have shape "
                f"({frames}, {pairs}), got {frame_mask.shape}"
            )
    valid_pair = raw_valid_pair & frame_mask
    delta = centers[:, pair_j] - centers[:, pair_i]
    distance = np.linalg.norm(delta, axis=2)
    head_distance = np.linalg.norm(heads[:, pair_j] - heads[:, pair_i], axis=2)
    direction = _cosine(velocity[:, pair_i], velocity[:, pair_j])
    pursuit_ab = _cosine(velocity[:, pair_i], delta)
    pursuit_ba = _cosine(velocity[:, pair_j], -delta)
    escape_ab = _cosine(velocity[:, pair_j], delta)
    escape_ba = _cosine(velocity[:, pair_i], -delta)
    behind_ab = np.sum((centers[:, pair_i] - centers[:, pair_j]) * heading[:, pair_j], axis=2) < 0.0
    behind_ba = np.sum((centers[:, pair_j] - centers[:, pair_i]) * heading[:, pair_i], axis=2) < 0.0
    distance_for_lookback = distance.copy()
    distance_for_lookback[~raw_valid_pair] = np.nan
    for array in (distance, head_distance, direction, pursuit_ab, pursuit_ba, escape_ab, escape_ba):
        array[~valid_pair] = np.nan
    behind_ab[~valid_pair] = False
    behind_ba[~valid_pair] = False

    nose_body_ab = np.full((frames, pairs), np.nan, dtype=np.float32)
    nose_body_ba = np.full((frames, pairs), np.nan, dtype=np.float32)
    nose_tail_ab = np.full((frames, pairs), np.nan, dtype=np.float32)
    nose_tail_ba = np.full((frames, pairs), np.nan, dtype=np.float32)
    nose_head_ab = np.full((frames, pairs), np.nan, dtype=np.float32)
    nose_head_ba = np.full((frames, pairs), np.nan, dtype=np.float32)
    for start in range(0, frames, 500):
        end = min(start + 500, frames)
        active_rows, active_pair_columns = np.nonzero(valid_pair[start:end])
        if not len(active_rows):
            continue
        frame_indices = start + active_rows
        pair_columns = active_pair_columns.astype(int, copy=False)
        body_b = kp[frame_indices, pair_j[pair_columns]]
        body_a = kp[frame_indices, pair_i[pair_columns]]
        nose_a = kp[frame_indices, pair_i[pair_columns], KP_NOSE]
        nose_b = kp[frame_indices, pair_j[pair_columns], KP_NOSE]
        distances_ab = np.linalg.norm(body_b - nose_a[:, None, :], axis=2)
        distances_ba = np.linalg.norm(body_a - nose_b[:, None, :], axis=2)
        nose_body_ab[frame_indices, pair_columns] = np.nanmin(distances_ab, axis=1)
        nose_body_ba[frame_indices, pair_columns] = np.nanmin(distances_ba, axis=1)
        nose_tail_ab[frame_indices, pair_columns] = np.linalg.norm(
            nose_a - kp[frame_indices, pair_j[pair_columns], KP_TAIL],
            axis=1,
        )
        nose_tail_ba[frame_indices, pair_columns] = np.linalg.norm(
            nose_b - kp[frame_indices, pair_i[pair_columns], KP_TAIL],
            axis=1,
        )
        nose_head_ab[frame_indices, pair_columns] = np.linalg.norm(
            nose_a - heads[frame_indices, pair_j[pair_columns]],
            axis=1,
        )
        nose_head_ba[frame_indices, pair_columns] = np.linalg.norm(
            nose_b - heads[frame_indices, pair_i[pair_columns]],
            axis=1,
        )
    nose_body_ab[~valid_pair] = np.nan
    nose_body_ba[~valid_pair] = np.nan
    nose_tail_ab[~valid_pair] = np.nan
    nose_tail_ba[~valid_pair] = np.nan
    nose_head_ab[~valid_pair] = np.nan
    nose_head_ba[~valid_pair] = np.nan

    lookback = max(int(round(fps * 0.30)), 1)
    distance_drop = np.zeros((frames, pairs), dtype=np.float32)
    if lookback < frames:
        distance_drop[lookback:] = (
            distance_for_lookback[:-lookback]
            - distance_for_lookback[lookback:]
        )
    distance_drop[~valid_pair] = 0.0
    distance_drop_seconds = float(lookback / max(fps, 1e-9))
    turn = np.zeros((frames, mice), dtype=np.float32)
    if lookback < frames:
        turn[lookback:] = _angle_deg(heading[lookback:], heading[:-lookback])
    turn[~valid] = 0.0
    velocity_a = velocity[:, pair_i]
    velocity_b = velocity[:, pair_j]
    trajectory_valid = valid_pair & np.all(np.isfinite(velocity_a), axis=2) & np.all(np.isfinite(velocity_b), axis=2)
    trajectory_corr = _rolling_corr(
        velocity_a,
        velocity_b,
        trajectory_valid,
        max(int(round(fps * 2.5)), 4),
        active_mask=frame_mask,
    )
    path_a = _rolling_sum(
        speed[:, pair_i],
        max(int(round(fps * 2.5)), 4),
        active_mask=frame_mask,
    ) / fps
    path_b = _rolling_sum(
        speed[:, pair_j],
        max(int(round(fps * 2.5)), 4),
        active_mask=frame_mask,
    ) / fps
    contact = np.minimum(nose_body_ab, nose_body_ba) <= 4.0
    contact &= valid_pair
    repeated_contact = _rolling_sum(
        contact.astype(float),
        max(int(round(fps * 2.0)), 1),
        active_mask=frame_mask,
    ).astype(np.int16)
    pose_pair_quality = np.sqrt(
        np.asarray(kin["pose_quality"][:, pair_i], dtype=float)
        * np.asarray(kin["pose_quality"][:, pair_j], dtype=float)
    )
    pose_pair_quality[~valid_pair] = 0.0
    metrics = {
        "valid_pair": valid_pair,
        "distance": distance,
        "head_distance": head_distance,
        "direction": direction,
        "pursuit_ab": pursuit_ab,
        "pursuit_ba": pursuit_ba,
        "escape_ab": escape_ab,
        "escape_ba": escape_ba,
        "behind_ab": behind_ab,
        "behind_ba": behind_ba,
        "nose_body_ab": nose_body_ab,
        "nose_body_ba": nose_body_ba,
        "nose_tail_ab": nose_tail_ab,
        "nose_tail_ba": nose_tail_ba,
        "nose_head_ab": nose_head_ab,
        "nose_head_ba": nose_head_ba,
        "distance_drop": distance_drop,
        "distance_drop_seconds": distance_drop_seconds,
        "turn": turn,
        "trajectory_corr": trajectory_corr,
        "path_a": path_a,
        "path_b": path_b,
        "repeated_contact": repeated_contact,
        "pose_pair_quality": pose_pair_quality,
        "speed": np.asarray(kin["speed"], dtype=float),
        "behavior_speed": np.asarray(kin["behavior_speed"], dtype=float),
        "nose_speed": np.asarray(kin["nose_speed"], dtype=float),
        "acceleration": np.asarray(kin["acceleration"], dtype=float),
        "angular_speed": np.asarray(kin["angular_speed"], dtype=float),
        "pose_deformation": np.asarray(kin["pose_deformation"], dtype=float),
    }
    return metrics, pair_i, pair_j


def _rolling_quantile(values: np.ndarray, window: int, quantile: float) -> np.ndarray:
    """Return a causal rolling quantile without introducing a new dependency."""
    values = np.asarray(values, dtype=float)
    window = max(int(window), 1)
    quantile = float(np.clip(quantile, 0.0, 1.0))
    result = np.full(values.shape, np.nan, dtype=float)
    for index in range(len(values)):
        start = max(0, index - window + 1)
        sample = values[start : index + 1]
        sample = sample[np.isfinite(sample)]
        if sample.size:
            result[index] = float(np.quantile(sample, quantile))
    return result


def _boolean_runs_with_gap(values: np.ndarray, max_gap: int = 0) -> list[tuple[int, int]]:
    """Group true runs while filling short false gaps."""
    values = np.asarray(values, dtype=bool).copy()
    max_gap = max(int(max_gap), 0)
    if max_gap and values.size:
        false_runs = _boolean_runs(~values)
        for start, end in false_runs:
            if start == 0 or end == len(values) - 1:
                continue
            if end - start + 1 <= max_gap:
                values[start : end + 1] = True
    return _boolean_runs(values)


def _event_rows_from_mask(
    mask: np.ndarray,
    *,
    behavior: str,
    level: str,
    fps: float,
    source_video: Path,
    sample_stride: int,
    score: np.ndarray | None = None,
    actor_id: np.ndarray | None = None,
    target_id: np.ndarray | None = None,
    pair_key: str = "",
    min_duration_seconds: float = 0.15,
    fill_gap_seconds: float = 0.10,
    event_scope: str = "pair",
    fsm_coordinator: ParallelBehaviorFSM | None = None,
) -> list[dict[str, Any]]:
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    min_frames = max(int(round(float(min_duration_seconds) * fps)), 1)
    gap_frames = max(int(round(float(fill_gap_seconds) * fps)), 0)
    score_values = np.asarray(score if score is not None else mask.astype(float), dtype=float)
    actor_values = np.asarray(actor_id if actor_id is not None else np.full(mask.shape, -1), dtype=int)
    target_values = np.asarray(target_id if target_id is not None else np.full(mask.shape, -1), dtype=int)
    events: list[dict[str, Any]] = []
    coordinator = fsm_coordinator or ParallelBehaviorFSM()
    fsm_result = coordinator.run_boolean_region(
        scope=str(event_scope),
        region_id=str(pair_key or event_scope),
        behavior=str(behavior),
        mask=mask,
        min_duration_frames=min_frames,
        max_gap_frames=gap_frames,
    )
    for span in fsm_result.spans:
        start, end = int(span.start), int(span.end)
        segment = score_values[start : end + 1]
        finite_segment = segment[np.isfinite(segment)]
        peak_offset = int(np.nanargmax(segment)) if finite_segment.size else 0
        peak = start + peak_offset
        actor = int(actor_values[peak]) if actor_values.size > peak else -1
        target = int(target_values[peak]) if target_values.size > peak else -1
        source_start = int(start * sample_stride)
        source_peak = int(peak * sample_stride)
        source_end = int(end * sample_stride)
        events.append(
            {
                "behavior": str(behavior),
                "behavior_name_zh": BEHAVIOR_NAMES_ZH.get(str(behavior), str(behavior)),
                "candidate_level": str(level),
                "behavior_engine": "lightweight_extended_ethogram",
                "event_scope": str(event_scope),
                "pair_key": str(pair_key),
                "actor_id": actor,
                "target_id": target,
                "role_ambiguous": bool(actor < 0 or target < 0),
                "analysis_start_frame": int(start),
                "analysis_peak_frame": int(peak),
                "analysis_end_frame": int(end),
                "start_frame": source_start,
                "peak_frame": source_peak,
                "end_frame": source_end,
                "start_time_s": source_start / max(float(fps * sample_stride), 1e-9),
                "end_time_s": source_end / max(float(fps * sample_stride), 1e-9),
                "duration_s": (source_end - source_start + 1) / max(float(fps * sample_stride), 1e-9),
                "mean_score": float(np.nanmean(segment)) if finite_segment.size else 0.0,
                "peak_score": float(np.nanmax(segment)) if finite_segment.size else 0.0,
                "source_video": str(source_video),
                "analysis_mode": "lightweight_cache_tracking",
            }
        )
    return events


def _extended_behavior_config(config: Mapping[str, Any]) -> dict[str, Any]:
    configured = dict(config.get("extended_behavior", {}))
    defaults = {
        "enabled": True,
        "individual": {
            "stationary_max_speed_cm_s": 4.0,
            "walking_max_speed_cm_s": 18.0,
            "running_min_speed_cm_s": 18.0,
            "confirm_seconds": 0.30,
            "fill_gap_seconds": 0.20,
            "min_pose_quality": 0.20,
        },
        "social": {
            "pair_max_distance_cm": 17.0,
            "together_max_distance_cm": 8.0,
            "together_max_combined_speed_cm_s": 28.0,
            "approach_min_distance_drop_cm": 1.5,
            "approach_min_closing_speed_cm_s": 2.0,
            "approach_max_actor_speed_cm_s": 22.0,
            "approach_max_target_speed_cm_s": 22.0,
            "approach_min_speed_gap_cm_s": 1.5,
            "approach_min_actor_speed_cm_s": 3.0,
            "approach_allow_together_transition": True,
            "approach_min_duration_seconds": 0.10,
            "chase_fallback": {
                "enabled": True,
                "max_distance_cm": 12.0,
                "min_score": 0.70,
                "min_actor_speed_cm_s": 3.0,
                "min_target_speed_cm_s": 2.5,
                "min_pursuit_alignment": 0.40,
                "min_target_escape_alignment": 0.30,
                "min_direction_similarity": 0.55,
                "min_role_confidence": 0.04,
                "min_duration_seconds": 0.25,
                "fill_gap_seconds": 0.15,
            },
            "attack_fallback": {
                "enabled": True,
                "max_distance_cm": 7.6,
                "min_dynamic_score": 0.78,
                "min_context_frames": 1,
                "min_attack_evidence": 3,
                "min_raw_actor_speed_cm_s": 8.0,
                "min_impact_pursuit_alignment": 0.55,
                "max_impact_target_escape_alignment": 0.10,
                "min_impact_behavior_speed_gap_cm_s": 1.5,
                "min_impact_actor_acceleration_cm_s2": 120.0,
                "min_impact_initiation_score": 0.90,
                "min_rebound_target_escape_alignment": 0.55,
                "max_rebound_pursuit_alignment": 0.20,
                "min_rebound_post_distance_increase_cm": 0.50,
                "min_rebound_immediate_distance_increase_cm": 0.25,
                "rebound_distance_window_seconds": 0.10,
                "min_rebound_reaction_score": 0.70,
                "min_rebound_contact_pursuit_alignment": 0.55,
                "min_target_turn_angle_deg": 25.0,
                "min_target_nose_speed_cm_s": 12.0,
                "min_role_confidence": 0.04,
                "min_duration_seconds": 0.033,
                "fill_gap_seconds": 0.10,
            },
            "avoidance_min_target_escape_alignment": 0.45,
            "avoidance_min_target_speed_cm_s": 1.5,
            "avoidance_min_distance_increase_cm": 1.0,
            "avoidance_min_actor_speed_cm_s": 1.5,
            "avoidance_max_distance_cm": 17.0,
            "avoidance_require_pursuit_context": True,
            "avoidance_context_seconds": 1.00,
            "avoidance_close_context_distance_cm": 10.0,
            "avoidance_min_pursuit_alignment": 0.35,
            "avoidance_min_prior_distance_drop_cm": 0.50,
            "avoidance_min_duration_seconds": 0.15,
            "pair_fill_gap_seconds": 0.15,
        },
        "group": {
            "huddle_distance_cm": 9.0,
            "huddle_fraction": 0.55,
            "huddle_min_cluster_size": 3,
            "huddle_min_cluster_fraction": 0.30,
            "huddle_min_cluster_density": 0.50,
            "isolation_distance_cm": 15.0,
            "isolation_neighbor_fraction": 0.15,
            "confirm_seconds": 0.30,
            "fill_gap_seconds": 0.20,
        },
    }
    for section, values in defaults.items():
        if isinstance(values, dict):
            merged = dict(values)
            if isinstance(configured.get(section), Mapping):
                merged.update(dict(configured[section]))
            configured[section] = merged
    return configured


def _extended_short_clip_pair_events(
    pair_df: pd.DataFrame,
    enriched: pd.DataFrame,
    *,
    source_video: Path,
    source_fps: float,
    sample_stride: int,
    config: Mapping[str, Any],
    fsm_coordinator: ParallelBehaviorFSM | None = None,
) -> list[dict[str, Any]]:
    """Recover short high-evidence chase/attack bouts missed by the legacy FSM.

    The standard engine intentionally requires a quality gate and a longer
    causal confirmation window.  Some Beiyi examples are shorter than that
    window, so this fallback is narrow and emits ``candidate_level=extended``:
    chase still needs directional motion; attack is restricted to an impact or
    post-contact rebound pattern with direction-consistent evidence.  A
    nose-head/nose-tail contact alone cannot satisfy these gates.
    """
    if pair_df.empty or enriched.empty:
        return []
    cfg = _extended_behavior_config(config)
    social = dict(cfg["social"])
    chase_cfg = dict(social.get("chase_fallback", {}))
    attack_cfg = dict(social.get("attack_fallback", {}))
    valid = pair_df.get("valid_pair", pd.Series(True, index=pair_df.index)).fillna(False).astype(bool).to_numpy()
    n = len(pair_df)
    fsm_coordinator = fsm_coordinator or ParallelBehaviorFSM(
        dict(config.get("parallel_behavior_fsm", {}))
    )

    def pair_values(column: str, default: float = 0.0) -> np.ndarray:
        values = pair_df[column] if column in pair_df else pd.Series(default, index=pair_df.index)
        return pd.to_numeric(values, errors="coerce").fillna(default).to_numpy(float)

    def engine_values(column: str, default: float = 0.0) -> np.ndarray:
        values = enriched[column] if column in enriched else pd.Series(default, index=enriched.index)
        return pd.to_numeric(values, errors="coerce").fillna(default).to_numpy(float)

    distance = pair_values("center_distance_cm", np.inf)
    actor_speed = pair_values("selected_actor_behavior_speed_cm_s")
    target_speed = pair_values("selected_target_behavior_speed_cm_s")
    actor_raw_speed = pair_values("selected_actor_speed_cm_s")
    target_raw_speed = pair_values("selected_target_speed_cm_s")
    pursuit = pair_values("selected_actor_pursuit_alignment")
    escape = pair_values("selected_target_escape_alignment")
    selected_actor = pd.to_numeric(pair_df.get("selected_actor_id", -1), errors="coerce").fillna(-1).to_numpy(int)
    selected_target = pd.to_numeric(pair_df.get("selected_target_id", -1), errors="coerce").fillna(-1).to_numpy(int)
    mouse_a = pd.to_numeric(pair_df.get("mouse_a_id", -2), errors="coerce").fillna(-2).to_numpy(int)
    mouse_b = pd.to_numeric(pair_df.get("mouse_b_id", -2), errors="coerce").fillna(-2).to_numpy(int)
    selected_ab = selected_actor == mouse_a
    selected_ba = selected_actor == mouse_b
    direction_known = selected_ab | selected_ba
    distance_drop = pair_values("selected_distance_drop_cm")
    speed_gap = actor_speed - target_speed
    target_turn = np.where(
        selected_ab,
        pair_values("a_to_b_target_turn_angle_deg"),
        pair_values("b_to_a_target_turn_angle_deg"),
    )
    target_nose_speed = np.where(
        selected_ab,
        pair_values("a_to_b_target_nose_speed_cm_s"),
        pair_values("b_to_a_target_nose_speed_cm_s"),
    )
    actor_acceleration = np.where(
        selected_ab,
        pair_values("a_to_b_actor_acceleration_cm_s2"),
        pair_values("b_to_a_actor_acceleration_cm_s2"),
    )
    direction_similarity = np.where(
        selected_ab,
        pair_values("a_to_b_direction_similarity"),
        pair_values("b_to_a_direction_similarity"),
    )
    nose_head = np.minimum(
        pair_values("a_to_b_nose_head_distance_cm", np.inf),
        pair_values("b_to_a_nose_head_distance_cm", np.inf),
    )
    nose_tail = np.minimum(
        pair_values("a_to_b_nose_tail_distance_cm", np.inf),
        pair_values("b_to_a_nose_tail_distance_cm", np.inf),
    )
    nose_body = np.minimum(
        pair_values("a_to_b_nose_body_distance_cm", np.inf),
        pair_values("b_to_a_nose_body_distance_cm", np.inf),
    )
    contact_cfg = dict(config.get("contact_detection", {}))
    contact = (
        (nose_head <= float(contact_cfg.get("nose_head_distance_cm", contact_cfg.get("distance_cm", 3.0))))
        | (nose_tail <= float(contact_cfg.get("nose_tail_distance_cm", contact_cfg.get("distance_cm", 3.0))))
        | (nose_body <= float(contact_cfg.get("distance_cm", 3.0)))
    )
    dynamic_score = np.maximum(
        engine_values("weak_standard_dynamic_attack_score"),
        engine_values("strong_standard_dynamic_attack_score"),
    )
    attack_score = np.maximum(
        engine_values("weak_standard_attack_score"),
        engine_values("strong_standard_attack_score"),
    )
    evidence_count = np.maximum(
        engine_values("weak_standard_attack_evidence_count"),
        engine_values("strong_standard_attack_evidence_count"),
    )
    role_confidence = np.maximum(
        engine_values("weak_standard_attack_role_confidence"),
        engine_values("strong_standard_attack_role_confidence"),
    )
    initiation = np.maximum(
        engine_values("weak_standard_initiation_score"),
        engine_values("strong_standard_initiation_score"),
    )
    reaction = np.maximum(
        engine_values("weak_standard_reaction_score"),
        engine_values("strong_standard_reaction_score"),
    )
    context_gate = (
        (enriched.get("weak_standard_attack_context_gate", pd.Series(False, index=enriched.index)).fillna(False).astype(bool).to_numpy())
        | (enriched.get("strong_standard_attack_context_gate", pd.Series(False, index=enriched.index)).fillna(False).astype(bool).to_numpy())
    )
    analysis_fps = source_fps / max(sample_stride, 1)
    contact_pursuit = pair_values("selected_actor_pursuit_alignment")
    contact_mask = (
        np.minimum(
            pair_values("a_to_b_nose_head_distance_cm", np.inf),
            pair_values("b_to_a_nose_head_distance_cm", np.inf),
        ) <= float(contact_cfg.get("nose_head_distance_cm", contact_cfg.get("distance_cm", 3.0)))
    ) | (
        np.minimum(
            pair_values("a_to_b_nose_tail_distance_cm", np.inf),
            pair_values("b_to_a_nose_tail_distance_cm", np.inf),
        ) <= float(contact_cfg.get("nose_tail_distance_cm", contact_cfg.get("distance_cm", 3.0)))
    ) | (
        np.minimum(
            pair_values("a_to_b_nose_body_distance_cm", np.inf),
            pair_values("b_to_a_nose_body_distance_cm", np.inf),
        ) <= float(contact_cfg.get("distance_cm", 3.0))
    )
    contact_direction_pursuit = contact_pursuit.copy()
    for index in range(n):
        if not contact_mask[index]:
            continue
        start = max(0, index - max(int(round(0.10 * analysis_fps)), 1))
        finite = contact_pursuit[start : index + 1]
        finite = finite[np.isfinite(finite)]
        contact_direction_pursuit[index] = float(np.max(finite)) if finite.size else 0.0
    events: list[dict[str, Any]] = []
    pair_key = str(pair_df["pair_key"].iloc[0])

    if bool(chase_cfg.get("enabled", True)):
        chase_score = np.maximum(
            engine_values("weak_standard_chase_score"),
            engine_values("strong_standard_chase_score"),
        )
        chase_mask = (
            valid
            & direction_known
            & np.isfinite(distance)
            & (distance <= float(chase_cfg.get("max_distance_cm", 12.0)))
            & (chase_score >= float(chase_cfg.get("min_score", 0.70)))
            & (actor_speed >= float(chase_cfg.get("min_actor_speed_cm_s", 1.5)))
            & (target_speed >= float(chase_cfg.get("min_target_speed_cm_s", 1.5)))
            & (pursuit >= float(chase_cfg.get("min_pursuit_alignment", 0.40)))
            & (escape >= float(chase_cfg.get("min_target_escape_alignment", 0.30)))
            & (direction_similarity >= float(chase_cfg.get("min_direction_similarity", 0.55)))
        )
        events.extend(
            _event_rows_from_mask(
                chase_mask,
                behavior="chase",
                level="extended",
                fps=analysis_fps,
                source_video=source_video,
                sample_stride=sample_stride,
                score=chase_score,
                actor_id=selected_actor,
                target_id=selected_target,
                pair_key=pair_key,
                min_duration_seconds=float(chase_cfg.get("min_duration_seconds", 0.25)),
                fill_gap_seconds=float(chase_cfg.get("fill_gap_seconds", 0.15)),
                fsm_coordinator=fsm_coordinator,
            )
        )

    if bool(attack_cfg.get("enabled", True)):
        attack_common = (
            valid
            & direction_known
            & np.isfinite(distance)
            & (distance <= float(attack_cfg.get("max_distance_cm", 7.6)))
            & contact
            & (dynamic_score >= float(attack_cfg.get("min_dynamic_score", 0.78)))
            & (evidence_count >= float(attack_cfg.get("min_attack_evidence", 3)))
            & (role_confidence >= float(attack_cfg.get("min_role_confidence", 0.04)))
        )
        # A rebound is allowed to occur a few frames after the collision.  It
        # is deliberately computed from pair distance rather than from the
        # selected direction, because an impact can reverse the actor/target
        # velocity ordering in the next frame.
        post_window = max(
            int(round(float(attack_cfg.get("rebound_distance_window_seconds", 0.10)) * analysis_fps)),
            1,
        )
        post_distance_increase = np.zeros(n, dtype=float)
        immediate_post_distance_increase = np.full(n, -np.inf, dtype=float)
        for offset in range(1, post_window + 1):
            if offset >= n:
                break
            future = np.full(n, np.nan, dtype=float)
            future[:-offset] = distance[offset:]
            with np.errstate(invalid="ignore"):
                candidate = future - distance
            candidate[~np.isfinite(candidate)] = -np.inf
            post_distance_increase = np.maximum(post_distance_increase, candidate)
            if offset == 1:
                immediate_post_distance_increase = candidate

        # Impact-type attack: the attacker has a short, high-energy approach
        # while the target is not escaping in the same smooth direction.  This
        # rejects ordinary nose-tail contact, which has positive pursuit and
        # positive target escape over a sustained approach.
        impact_attack = (
            attack_common
            & (actor_raw_speed >= float(attack_cfg.get("min_raw_actor_speed_cm_s", 8.0)))
            & (pursuit >= float(attack_cfg.get("min_impact_pursuit_alignment", 0.55)))
            & (escape <= float(attack_cfg.get("max_impact_target_escape_alignment", 0.10)))
            & (speed_gap >= float(attack_cfg.get("min_impact_behavior_speed_gap_cm_s", 1.5)))
            & (
                (actor_acceleration >= float(attack_cfg.get("min_impact_actor_acceleration_cm_s2", 120.0)))
                | (initiation >= float(attack_cfg.get("min_impact_initiation_score", 0.90)))
            )
        )

        # Rebound-type attack: the pair separates immediately after contact,
        # and the selected direction reverses or the target shows a reaction.
        # This is the short terminal pattern in the second attack example.
        rebound_attack = (
            attack_common
            & (actor_raw_speed >= float(attack_cfg.get("min_raw_actor_speed_cm_s", 8.0)))
            & (escape >= float(attack_cfg.get("min_rebound_target_escape_alignment", 0.55)))
            & (pursuit <= float(attack_cfg.get("max_rebound_pursuit_alignment", 0.20)))
            & (contact_direction_pursuit >= float(attack_cfg.get("min_rebound_contact_pursuit_alignment", 0.55)))
            & (post_distance_increase >= float(attack_cfg.get("min_rebound_post_distance_increase_cm", 0.50)))
            & (immediate_post_distance_increase >= float(attack_cfg.get("min_rebound_immediate_distance_increase_cm", 0.25)))
            & (
                (reaction >= float(attack_cfg.get("min_rebound_reaction_score", 0.70)))
                | (target_turn >= float(attack_cfg.get("min_target_turn_angle_deg", 25.0)))
                | (target_nose_speed >= float(attack_cfg.get("min_target_nose_speed_cm_s", 12.0)))
            )
        )
        attack_mask = impact_attack | rebound_attack
        events.extend(
            _event_rows_from_mask(
                attack_mask,
                behavior="attack",
                level="extended",
                fps=analysis_fps,
                source_video=source_video,
                sample_stride=sample_stride,
                score=attack_score,
                actor_id=selected_actor,
                target_id=selected_target,
                pair_key=pair_key,
                min_duration_seconds=float(attack_cfg.get("min_duration_seconds", 0.08)),
                fill_gap_seconds=float(attack_cfg.get("fill_gap_seconds", 0.10)),
                fsm_coordinator=fsm_coordinator,
            )
        )
    return events


def _extended_pair_events(
    pair_df: pd.DataFrame,
    *,
    metrics: Mapping[str, Any],
    pair_index: int,
    enriched: pd.DataFrame | None = None,
    source_video: Path,
    source_fps: float,
    sample_stride: int,
    config: Mapping[str, Any],
    fsm_coordinator: ParallelBehaviorFSM | None = None,
) -> list[dict[str, Any]]:
    """Infer social labels that are not part of the legacy chase/attack FSM.

    Approach is intentionally slower and weaker than chase: it requires a
    measurable distance decrease but rejects the sustained two-mouse speed and
    pursuit evidence used by chase.  Avoidance is the opposite directional
    pattern: a moving target escapes and the pair distance increases after a
    close interaction.  Contact events remain independent and are added by the
    caller.
    """
    if pair_df.empty:
        return []
    cfg = _extended_behavior_config(config)
    social = dict(cfg["social"])
    n = len(pair_df)
    fsm_coordinator = fsm_coordinator or ParallelBehaviorFSM(
        dict(config.get("parallel_behavior_fsm", {}))
    )
    distance = pd.to_numeric(pair_df["center_distance_cm"], errors="coerce").to_numpy(float)
    valid = pair_df.get("valid_pair", pd.Series(True, index=pair_df.index)).fillna(False).astype(bool).to_numpy()
    actor_speed = pd.to_numeric(pair_df.get("selected_actor_behavior_speed_cm_s", 0.0), errors="coerce").fillna(0).to_numpy(float)
    target_speed = pd.to_numeric(pair_df.get("selected_target_behavior_speed_cm_s", 0.0), errors="coerce").fillna(0).to_numpy(float)
    combined_speed = actor_speed + target_speed
    drop = pd.to_numeric(pair_df.get("selected_distance_drop_cm", 0.0), errors="coerce").fillna(0).to_numpy(float)
    closing = pd.to_numeric(pair_df.get("selected_closing_speed_cm_s", 0.0), errors="coerce").fillna(0).to_numpy(float)
    # The selected direction is the one used by the lightweight pair row.
    selected_ab = pair_df.get("selected_actor_id", pd.Series(-1, index=pair_df.index)).to_numpy() == pair_df.get("mouse_a_id", pd.Series(-2, index=pair_df.index)).to_numpy()
    selected_escape = np.where(
        selected_ab,
        pd.to_numeric(pair_df.get("a_to_b_target_escape_alignment", 0.0), errors="coerce").fillna(0).to_numpy(float),
        pd.to_numeric(pair_df.get("b_to_a_target_escape_alignment", 0.0), errors="coerce").fillna(0).to_numpy(float),
    )
    selected_actor_speed = actor_speed
    selected_target_speed = target_speed
    selected_actor = pd.to_numeric(pair_df.get("selected_actor_id", -1), errors="coerce").fillna(-1).to_numpy(int)
    selected_target = pd.to_numeric(pair_df.get("selected_target_id", -1), errors="coerce").fillna(-1).to_numpy(int)

    def pair_numeric(column: str, default: float) -> np.ndarray:
        values = pair_df[column] if column in pair_df else pd.Series(default, index=pair_df.index)
        return pd.to_numeric(values, errors="coerce").fillna(default).to_numpy(float)

    # Nose contact and approach are close-range states, but they are not the
    # same label.  Remove contact-geometry samples from approach so a nose
    # touching a head/tail is emitted only by the independent contact stream.
    contact_config = dict(config.get("contact_detection", {}))
    nose_head_threshold = max(
        float(contact_config.get("nose_head_distance_cm", contact_config.get("distance_cm", 3.0))),
        0.0,
    )
    nose_tail_threshold = max(
        float(contact_config.get("nose_tail_distance_cm", contact_config.get("distance_cm", 3.0))),
        0.0,
    )
    contact_geometry = (
        np.minimum(
            pair_numeric("a_to_b_nose_head_distance_cm", np.inf),
            pair_numeric("b_to_a_nose_head_distance_cm", np.inf),
        ) <= nose_head_threshold
    ) | (
        np.minimum(
            pair_numeric("a_to_b_nose_tail_distance_cm", np.inf),
            pair_numeric("b_to_a_nose_tail_distance_cm", np.inf),
        ) <= nose_tail_threshold
    )

    # ``together`` is a low-motion, close pair state. It does not require a
    # contact threshold, so it can represent the labeled together examples.
    together = (
        valid
        & np.isfinite(distance)
        & (distance <= float(social["together_max_distance_cm"]))
        & (combined_speed <= float(social["together_max_combined_speed_cm_s"]))
    )
    approach = (
        valid
        & np.isfinite(distance)
        & (distance <= float(social["pair_max_distance_cm"]))
        & (drop >= float(social["approach_min_distance_drop_cm"]))
        & (closing >= float(social["approach_min_closing_speed_cm_s"]))
        & (selected_actor_speed >= float(social.get("approach_min_actor_speed_cm_s", 0.0)))
        & (selected_actor_speed <= float(social["approach_max_actor_speed_cm_s"]))
        & (selected_target_speed <= float(social["approach_max_target_speed_cm_s"]))
        & ((selected_actor_speed - selected_target_speed) >= float(social["approach_min_speed_gap_cm_s"]))
    )
    # Keep the approach-to-together transition: the last approach samples can
    # be close and low-motion after the speed difference collapses.  Contact
    # geometry remains excluded, so this does not turn nose contact into
    # approach.
    if not bool(social.get("approach_allow_together_transition", True)):
        approach &= ~together
    distance_increase = np.zeros(n, dtype=float)
    lookback = max(int(round(source_fps / max(sample_stride, 1) * 0.25)), 1)
    if lookback < n:
        distance_increase[lookback:] = distance[lookback:] - distance[:-lookback]
    avoidance = (
        valid
        & np.isfinite(distance)
        & (distance <= float(social["avoidance_max_distance_cm"]))
        & (selected_target_speed >= float(social["avoidance_min_target_speed_cm_s"]))
        & (selected_actor_speed >= float(social["avoidance_min_actor_speed_cm_s"]))
        & (selected_escape >= float(social["avoidance_min_target_escape_alignment"]))
        & (distance_increase >= float(social["avoidance_min_distance_increase_cm"]))
    )

    # Avoidance is a reaction after an interaction, not arbitrary separation
    # of two moving mice.  Require prior pursuit/approach and a prior close
    # state within a causal window.  Legacy chase/attack masks are context
    # only; the actual avoidance frame still has to satisfy the directional
    # escape conditions above.
    context_frames = max(
        int(round(float(social["avoidance_context_seconds"]) * source_fps / max(sample_stride, 1))),
        1,
    )
    prior_close = np.zeros(n, dtype=bool)
    close_context = np.isfinite(distance) & (
        distance <= float(social["avoidance_close_context_distance_cm"])
    )
    if n > 1:
        prior_close[1:] = _rolling_sum(close_context[:-1].astype(float), context_frames) > 0

    pursuit_context = approach.copy()
    if enriched is not None and not enriched.empty:
        for column in (
            "weak_standard_final_chase",
            "strong_standard_final_chase",
            "weak_standard_final_attack",
            "strong_standard_final_attack",
        ):
            if column in enriched:
                pursuit_context |= enriched[column].fillna(False).astype(bool).to_numpy()
    prior_pursuit = np.zeros(n, dtype=bool)
    if n > 1:
        prior_pursuit[1:] = _rolling_sum(pursuit_context[:-1].astype(float), context_frames) > 0
    prior_drop = np.zeros(n, dtype=float)
    if n > 1:
        prior_drop[1:] = np.maximum(drop[:-1], 0.0)
    pursuit_alignment_context = np.maximum(
        pair_numeric("selected_actor_pursuit_alignment", 0.0),
        pair_numeric("selected_closing_speed_cm_s", 0.0)
        / max(float(social.get("approach_min_closing_speed_cm_s", 2.0)), 1e-6),
    )
    if bool(social.get("avoidance_require_pursuit_context", True)):
        avoidance &= prior_close & (
            prior_pursuit
            | (pursuit_alignment_context >= float(social.get("avoidance_min_pursuit_alignment", 0.35)))
            | (prior_drop >= float(social.get("avoidance_min_prior_distance_drop_cm", 0.50)))
        )

    # A labeled approach is a low-speed social transition.  Contact samples
    # were removed above, so nose-head/nose-tail events remain independent and
    # cannot masquerade as approach or attack.  The final approach-to-together
    # transition is retained when explicitly enabled because it is the part
    # most often lost in short Beiyi approach clips.
    if not bool(social.get("approach_allow_together_transition", True)):
        approach &= ~together
    events: list[dict[str, Any]] = []
    events.extend(
        _event_rows_from_mask(
            together,
            behavior="together",
            level="extended",
            fps=source_fps / max(sample_stride, 1),
            source_video=source_video,
            sample_stride=sample_stride,
            score=np.where(together, 1.0 - np.clip(distance / max(float(social["together_max_distance_cm"]), 1e-6), 0, 1), 0.0),
            actor_id=np.full(n, -1),
            target_id=np.full(n, -1),
            pair_key=str(pair_df["pair_key"].iloc[0]),
            min_duration_seconds=0.30,
            fill_gap_seconds=float(social["pair_fill_gap_seconds"]),
            fsm_coordinator=fsm_coordinator,
        )
    )
    events.extend(
        _event_rows_from_mask(
            approach,
            behavior="approach",
            level="extended",
            fps=source_fps / max(sample_stride, 1),
            source_video=source_video,
            sample_stride=sample_stride,
            score=np.where(approach, np.clip(drop / max(float(social["approach_min_distance_drop_cm"]), 1e-6), 0, 1), 0.0),
            actor_id=selected_actor,
            target_id=selected_target,
            pair_key=str(pair_df["pair_key"].iloc[0]),
            min_duration_seconds=float(social["approach_min_duration_seconds"]),
            fill_gap_seconds=float(social["pair_fill_gap_seconds"]),
            fsm_coordinator=fsm_coordinator,
        )
    )
    events.extend(
        _event_rows_from_mask(
            avoidance,
            behavior="avoidance",
            level="extended",
            fps=source_fps / max(sample_stride, 1),
            source_video=source_video,
            sample_stride=sample_stride,
            score=np.where(avoidance, np.clip(distance_increase / max(float(social["avoidance_min_distance_increase_cm"]), 1e-6), 0, 1), 0.0),
            actor_id=selected_actor,
            target_id=selected_target,
            pair_key=str(pair_df["pair_key"].iloc[0]),
            min_duration_seconds=float(social["avoidance_min_duration_seconds"]),
            fill_gap_seconds=float(social["pair_fill_gap_seconds"]),
            fsm_coordinator=fsm_coordinator,
        )
    )
    return events


def _extended_individual_and_group_events(
    kin: Mapping[str, Any],
    *,
    pair_metrics: Mapping[str, Any],
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    source_video: Path,
    source_fps: float,
    sample_stride: int,
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Infer individual and group labels from all tracked mice per frame."""
    cfg = _extended_behavior_config(config)
    individual_cfg = dict(cfg["individual"])
    group_cfg = dict(cfg["group"])
    valid = np.asarray(kin["valid"], dtype=bool)
    speed = np.asarray(kin["behavior_speed"], dtype=float)
    pose_quality = np.asarray(kin["pose_quality"], dtype=float)
    centers = np.asarray(kin["centers_cm"], dtype=float)
    frames, mice = valid.shape
    analysis_fps = source_fps / max(sample_stride, 1)
    events: list[dict[str, Any]] = []
    fsm_coordinator = ParallelBehaviorFSM(
        dict(config.get("parallel_behavior_fsm", {}))
    )

    stationary = valid & (speed <= float(individual_cfg["stationary_max_speed_cm_s"])) & (
        pose_quality >= float(individual_cfg["min_pose_quality"])
    )
    walking = valid & (speed > float(individual_cfg["stationary_max_speed_cm_s"])) & (
        speed < float(individual_cfg["running_min_speed_cm_s"])
    )
    running = valid & (speed >= float(individual_cfg["running_min_speed_cm_s"]))
    for mouse in range(mice):
        for behavior, mask, score in (
            ("stationary", stationary[:, mouse], np.maximum(0.0, 1.0 - speed[:, mouse] / max(float(individual_cfg["stationary_max_speed_cm_s"]), 1e-6))),
            ("walking", walking[:, mouse], np.clip(speed[:, mouse] / max(float(individual_cfg["walking_max_speed_cm_s"]), 1e-6), 0.0, 1.0)),
            ("running", running[:, mouse], np.clip(speed[:, mouse] / max(float(individual_cfg["running_min_speed_cm_s"]), 1e-6), 0.0, 1.0)),
        ):
            events.extend(
                _event_rows_from_mask(
                    mask,
                    behavior=behavior,
                    level="extended",
                    fps=analysis_fps,
                    source_video=source_video,
                    sample_stride=sample_stride,
                    score=score,
                    actor_id=np.full(frames, mouse),
                    target_id=np.full(frames, -1),
                    pair_key=f"mouse_{mouse}",
                    min_duration_seconds=float(individual_cfg["confirm_seconds"]),
                    fill_gap_seconds=float(individual_cfg["fill_gap_seconds"]),
                    event_scope="individual",
                    fsm_coordinator=fsm_coordinator,
                )
            )

    # Construct a per-frame nearest-neighbour graph.  Distances are already in
    # cm and therefore adapt to each video's learned scale; no video-specific
    # pixel constant is used here.
    close_fraction = np.zeros(frames, dtype=float)
    isolated_fraction = np.zeros(frames, dtype=float)
    group_size = np.zeros(frames, dtype=int)
    largest_cluster_size = np.zeros(frames, dtype=int)
    largest_cluster_fraction = np.zeros(frames, dtype=float)
    largest_cluster_density = np.zeros(frames, dtype=float)
    for frame in range(frames):
        ids = np.flatnonzero(valid[frame])
        group_size[frame] = len(ids)
        if len(ids) < 2:
            continue
        points = centers[frame, ids]
        distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
        distances[~np.isfinite(distances)] = np.inf
        np.fill_diagonal(distances, np.inf)
        nearest = np.min(distances, axis=1)
        close_fraction[frame] = float(np.mean(nearest <= float(group_cfg["huddle_distance_cm"])))
        isolated_fraction[frame] = float(np.mean(nearest >= float(group_cfg["isolation_distance_cm"])))

        # A multi-mouse cage can contain a local huddle while other visible
        # mice remain spread out.  Use connected components of the complete
        # visible-mouse graph instead of requiring every visible mouse to be
        # close to a neighbour.  This is still a group statistic: no
        # single-mouse or two-mouse clip is fabricated from the video.
        close_threshold = float(group_cfg["huddle_distance_cm"])
        adjacency = distances <= close_threshold
        unseen = set(range(len(ids)))
        components: list[list[int]] = []
        while unseen:
            seed = unseen.pop()
            stack = [seed]
            component: list[int] = []
            while stack:
                current = stack.pop()
                component.append(current)
                neighbours = [item for item in unseen if adjacency[current, item]]
                for item in neighbours:
                    unseen.remove(item)
                    stack.append(item)
            components.append(component)
        largest = max(components, key=len, default=[])
        largest_cluster_size[frame] = len(largest)
        largest_cluster_fraction[frame] = len(largest) / max(len(ids), 1)
        if len(largest) >= 2:
            component_indices = np.asarray(largest, dtype=int)
            component_distances = distances[np.ix_(component_indices, component_indices)]
            edge_count = int(np.sum(component_distances <= close_threshold) // 2)
            possible_edges = len(largest) * (len(largest) - 1) // 2
            largest_cluster_density[frame] = edge_count / max(possible_edges, 1)

    huddle = (
        (group_size >= 2)
        & (
            (close_fraction >= float(group_cfg["huddle_fraction"]))
            | (
                (largest_cluster_size >= int(group_cfg.get("huddle_min_cluster_size", 3)))
                & (largest_cluster_fraction >= float(group_cfg.get("huddle_min_cluster_fraction", 0.30)))
                & (largest_cluster_density >= float(group_cfg.get("huddle_min_cluster_density", 0.50)))
            )
        )
    )
    # Isolation is a group-level state only if a substantial fraction of the
    # visible mice have no close neighbour; it is not emitted for an empty or
    # one-mouse frame.
    isolation = (
        (group_size >= 3)
        & (isolated_fraction >= float(group_cfg["isolation_neighbor_fraction"]))
    )
    for behavior, mask, score in (
        (
            "huddle",
            huddle,
            np.maximum(close_fraction, largest_cluster_fraction * largest_cluster_density),
        ),
        ("isolation", isolation, isolated_fraction),
    ):
        events.extend(
            _event_rows_from_mask(
                mask,
                behavior=behavior,
                level="extended",
                fps=analysis_fps,
                source_video=source_video,
                sample_stride=sample_stride,
                score=score,
                actor_id=np.full(frames, -1),
                target_id=np.full(frames, -1),
                pair_key="group",
                min_duration_seconds=float(group_cfg["confirm_seconds"]),
                fill_gap_seconds=float(group_cfg["fill_gap_seconds"]),
                event_scope="group",
                fsm_coordinator=fsm_coordinator,
            )
        )
    return events


def _write_csv(
    path: Path,
    rows: list[Mapping[str, Any]],
    columns: Sequence[str] | None = None,
) -> None:
    """Write a UTF-8 CSV while preserving the schema for empty results."""
    if not rows:
        pd.DataFrame(columns=list(columns or [])).to_csv(
            path,
            index=False,
            encoding="utf-8-sig",
        )
        return
    frame = pd.DataFrame(rows)
    if columns:
        for column in columns:
            if column not in frame.columns:
                frame[column] = pd.NA
        frame = frame.loc[:, list(columns)]
    frame.to_csv(path, index=False, encoding="utf-8-sig")


CONTACT_EVENT_COLUMNS = (
    "contact_event_id",
    "contact_detector",
    "pair_key",
    "contact_type",
    "contact_type_components",
    "contact_direction",
    "contact_actor_id",
    "contact_target_id",
    "role_ambiguous",
    "analysis_start_frame",
    "analysis_peak_frame",
    "analysis_end_frame",
    "start_frame",
    "peak_frame",
    "end_frame",
    "start_time_s",
    "end_time_s",
    "duration_s",
    "sample_count",
    "min_contact_distance_cm",
    "mean_contact_distance_cm",
    "min_nose_head_distance_cm",
    "min_nose_tail_distance_cm",
    "source_video",
    "analysis_mode",
)


def _contact_distance(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("inf")
    return number if np.isfinite(number) else float("inf")


def _contact_components(
    nose_head_distance: float,
    nose_tail_distance: float,
    nose_head_threshold: float,
    nose_tail_threshold: float,
) -> tuple[str, ...]:
    components: list[str] = []
    if nose_head_distance <= nose_head_threshold:
        components.append("nose_head")
    if nose_tail_distance <= nose_tail_threshold:
        components.append("nose_tail")
    return tuple(components)


def _extract_contact_events(
    pair_df: pd.DataFrame,
    *,
    pair_key: str,
    source_video: Path,
    source_fps: float,
    sample_stride: int,
    contact_config: Mapping[str, Any],
    fsm_coordinator: ParallelBehaviorFSM | None = None,
) -> list[dict[str, Any]]:
    """Extract nose-head and nose-tail contacts independently of behavior.

    Contact is deliberately not a behavior class.  It is a geometric event
    stream that can coexist with a chase or an attack, while ordinary contact
    alone never opens either behavior FSM.  The event state is evaluated at
    every analyzed sample and consecutive samples with the same contact
    geometry are grouped into one event.
    """
    if pair_df.empty or not bool(contact_config.get("enabled", True)):
        return []

    source_fps = max(float(source_fps), 1e-9)
    sample_stride = max(int(sample_stride), 1)
    nose_head_threshold = max(
        float(contact_config.get("nose_head_distance_cm", contact_config.get("distance_cm", 3.0))),
        0.0,
    )
    nose_tail_threshold = max(
        float(contact_config.get("nose_tail_distance_cm", contact_config.get("distance_cm", 3.0))),
        0.0,
    )

    frame_values = pd.to_numeric(
        pair_df.get("frame", pd.Series(range(len(pair_df)))),
        errors="coerce",
    ).fillna(-1).astype(int).to_numpy()
    row_count = len(pair_df)

    def numeric_column(column: str, default: float) -> np.ndarray:
        values = (
            pair_df[column]
            if column in pair_df
            else pd.Series(default, index=pair_df.index)
        )
        result = pd.to_numeric(values, errors="coerce").to_numpy(
            dtype=float,
            copy=True,
        )
        result[~np.isfinite(result)] = default
        return result

    def id_column(column: str) -> tuple[np.ndarray, np.ndarray]:
        values = (
            pair_df[column]
            if column in pair_df
            else pd.Series(-1, index=pair_df.index)
        )
        numeric = pd.to_numeric(values, errors="coerce").to_numpy(
            dtype=float,
            copy=True,
        )
        convertible = np.isfinite(numeric)
        numeric[~convertible] = -1
        return numeric.astype(int), convertible

    valid_values = (
        pair_df["valid_pair"]
        if "valid_pair" in pair_df
        else pd.Series(True, index=pair_df.index)
    )
    valid_pair = valid_values.fillna(False).astype(bool).to_numpy()
    mouse_a_ids, mouse_a_id_valid = id_column("mouse_a_id")
    mouse_b_ids, mouse_b_id_valid = id_column("mouse_b_id")
    head_ab = numeric_column("a_to_b_nose_head_distance_cm", float("inf"))
    tail_ab = numeric_column("a_to_b_nose_tail_distance_cm", float("inf"))
    head_ba = numeric_column("b_to_a_nose_head_distance_cm", float("inf"))
    tail_ba = numeric_column("b_to_a_nose_tail_distance_cm", float("inf"))
    contact_ab = (head_ab <= nose_head_threshold) | (tail_ab <= nose_tail_threshold)
    contact_ba = (head_ba <= nose_head_threshold) | (tail_ba <= nose_tail_threshold)

    states: list[dict[str, Any] | None] = [None] * row_count
    contact_indices = np.flatnonzero(valid_pair & (contact_ab | contact_ba))
    for index in contact_indices:
        if mouse_a_id_valid[index] and mouse_b_id_valid[index]:
            mouse_a_id = int(mouse_a_ids[index])
            mouse_b_id = int(mouse_b_ids[index])
        else:
            # Preserve the historical record-loop contract: conversion of
            # either endpoint failing makes both contact roles unknown.
            mouse_a_id = -1
            mouse_b_id = -1
        direction_hits: list[dict[str, Any]] = []
        direction_specs = (
            (
                "a_to_b",
                mouse_a_id,
                mouse_b_id,
                float(head_ab[index]),
                float(tail_ab[index]),
            ),
            (
                "b_to_a",
                mouse_b_id,
                mouse_a_id,
                float(head_ba[index]),
                float(tail_ba[index]),
            ),
        )
        for direction, actor_id, target_id, head_distance, tail_distance in direction_specs:
            components = _contact_components(
                head_distance,
                tail_distance,
                nose_head_threshold,
                nose_tail_threshold,
            )
            if not components:
                continue
            direction_hits.append(
                {
                    "direction": direction,
                    "actor_id": actor_id,
                    "target_id": target_id,
                    "components": components,
                    "head_distance": head_distance,
                    "tail_distance": tail_distance,
                }
            )

        if not direction_hits:
            states.append(None)
            continue

        components = tuple(
            name
            for name in ("nose_head", "nose_tail")
            if any(name in hit["components"] for hit in direction_hits)
        )
        contact_type = (
            "nose_head_and_nose_tail"
            if len(components) == 2
            else components[0]
        )
        directions = tuple(hit["direction"] for hit in direction_hits)
        if len(directions) == 1:
            contact_direction = directions[0]
            contact_actor_id = int(direction_hits[0]["actor_id"])
            contact_target_id = int(direction_hits[0]["target_id"])
        else:
            contact_direction = "both"
            contact_actor_id = -1
            contact_target_id = -1

        head_distances = [hit["head_distance"] for hit in direction_hits]
        tail_distances = [hit["tail_distance"] for hit in direction_hits]
        contact_distances = [min(head_distances), min(tail_distances)]
        states[int(index)] = {
            "contact_type": contact_type,
            "contact_type_components": ";".join(components),
            "contact_direction": contact_direction,
            "contact_actor_id": contact_actor_id,
            "contact_target_id": contact_target_id,
            "role_ambiguous": bool(contact_actor_id < 0 or contact_target_id < 0),
            "contact_distance_cm": min(contact_distances),
            "nose_head_distance_cm": min(head_distances),
            "nose_tail_distance_cm": min(tail_distances),
        }

    def state_key(state: Mapping[str, Any] | None) -> tuple[Any, ...] | None:
        if state is None:
            return None
        return (
            state["contact_type"],
            state["contact_type_components"],
            state["contact_direction"],
            state["contact_actor_id"],
            state["contact_target_id"],
        )

    coordinator = fsm_coordinator or ParallelBehaviorFSM()
    contact_fsm = coordinator.run_categorical_region(
        scope="pair",
        region_id=str(pair_key),
        states=states,
        state_key=state_key,
    )
    events: list[dict[str, Any]] = []
    for span in contact_fsm.spans:
        start, end = int(span.start), int(span.end)
        segment = [state for state in states[start : end + 1] if state is not None]
        assert segment
        distances = np.asarray([state["contact_distance_cm"] for state in segment], dtype=float)
        peak_offset = int(np.argmin(distances)) if len(distances) else 0
        analysis_start = int(frame_values[start])
        analysis_peak = int(frame_values[start + peak_offset])
        analysis_end = int(frame_values[end])
        source_start = analysis_start * sample_stride
        source_peak = analysis_peak * sample_stride
        source_end = analysis_end * sample_stride
        events.append(
            {
                "contact_detector": "nose_head_nose_tail_geometry",
                "pair_key": str(pair_key),
                "contact_type": segment[0]["contact_type"],
                "contact_type_components": segment[0]["contact_type_components"],
                "contact_direction": segment[0]["contact_direction"],
                "contact_actor_id": int(segment[0]["contact_actor_id"]),
                "contact_target_id": int(segment[0]["contact_target_id"]),
                "role_ambiguous": bool(segment[0]["role_ambiguous"]),
                "analysis_start_frame": analysis_start,
                "analysis_peak_frame": analysis_peak,
                "analysis_end_frame": analysis_end,
                "start_frame": source_start,
                "peak_frame": source_peak,
                "end_frame": source_end,
                "start_time_s": source_start / source_fps,
                "end_time_s": source_end / source_fps,
                "duration_s": (source_end - source_start + 1) / source_fps,
                "sample_count": len(segment),
                "min_contact_distance_cm": float(np.min(distances)),
                "mean_contact_distance_cm": float(np.mean(distances)),
                "min_nose_head_distance_cm": float(
                    min(state["nose_head_distance_cm"] for state in segment)
                ),
                "min_nose_tail_distance_cm": float(
                    min(state["nose_tail_distance_cm"] for state in segment)
                ),
                "source_video": str(source_video),
                "analysis_mode": "lightweight_cache_tracking",
            }
        )
    return events


def render_behavior_video(
    video_path: Path,
    cache_dir: Path,
    events_path: Path,
    output_path: Path,
    expected_mice: int = 20,
    max_frames: int | None = None,
) -> Path:
    """Render exactly one annotated MP4 for one source video.

    The renderer deliberately consumes the lightweight track IDs and event
    CSV produced by this module.  It does not create event clips or any other
    video outputs.
    """
    total_cache_frames = _cache_total_frames(cache_dir)
    if max_frames is not None:
        total_cache_frames = min(total_cache_frames, max(int(max_frames), 1))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开源视频: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"无法读取源视频尺寸: {video_path}")
    boundary_json = events_path.parent / "阶段一_自适应笼界.json"
    arena_polygon: np.ndarray | None = None
    if boundary_json.exists():
        boundary = arena_boundary.load_boundary_json(
            boundary_json,
            width=width,
            height=height,
            source_video=video_path,
            require_video_match=True,
        )
        arena_polygon = np.asarray(boundary.polygon, dtype=np.float64)
    tracks, tracking_stats = _track_cache(
        cache_dir,
        total_cache_frames,
        expected_mice,
        arena_polygon=arena_polygon,
    )

    event_frame_map: list[list[dict[str, Any]]] = [
        [] for _ in range(total_cache_frames)
    ]
    if not events_path.exists():
        raise FileNotFoundError(f"行为事件 CSV 不存在: {events_path}")
    events_df = pd.read_csv(events_path)
    required = {"start_frame", "end_frame", "candidate_level", "behavior"}
    missing = sorted(required.difference(events_df.columns))
    if missing:
        raise ValueError(f"行为事件 CSV 缺少字段: {missing}")
    for event in events_df.to_dict("records"):
        try:
            start = max(int(event.get("start_frame", 0)), 0)
            end = min(int(event.get("end_frame", -1)), total_cache_frames - 1)
        except (TypeError, ValueError):
            continue
        if end < start:
            continue
        for frame_index in range(start, end + 1):
            event_frame_map[frame_index].append(event)

    if not np.isfinite(fps) or fps <= 0:
        fps = 29.329
    frame_limit = total_cache_frames
    if video_frame_count > 0:
        frame_limit = min(frame_limit, video_frame_count)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"无法创建渲染视频: {output_path}")

    skeleton_edges = SKELETON_EDGES
    default_color = (80, 220, 80)
    role_colors = {
        ("strong", "chase", "actor"): (0, 165, 255),
        ("strong", "chase", "target"): (255, 140, 0),
        ("strong", "attack", "actor"): (0, 0, 255),
        ("strong", "attack", "target"): (255, 0, 255),
        ("weak", "chase", "actor"): (0, 215, 255),
        ("weak", "chase", "target"): (255, 200, 80),
        ("weak", "attack", "actor"): (80, 80, 255),
        ("weak", "attack", "target"): (255, 100, 200),
    }

    frame_index = 0
    try:
        while frame_index < frame_limit:
            ok, frame = cap.read()
            if not ok:
                break
            active_events = event_frame_map[frame_index]
            # Keep one best row per pair/level/behavior so overlapping FSM
            # rows do not cover the whole top panel.
            best_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
            for event in active_events:
                key = (
                    str(event.get("candidate_level", "weak")),
                    str(event.get("behavior", "")),
                    str(event.get("pair_key", "")),
                )
                previous = best_by_key.get(key)
                if previous is None or float(event.get("peak_score", 0.0)) > float(previous.get("peak_score", 0.0)):
                    best_by_key[key] = event
            display_events = sorted(
                best_by_key.values(),
                key=lambda item: (
                    0 if str(item.get("candidate_level")) == "strong" else 1,
                    -float(item.get("peak_score", 0.0)),
                ),
            )

            role_map: dict[int, tuple[int, tuple[int, int, int]]] = {}
            active_actor_ids: set[int] = set()
            active_target_ids: set[int] = set()
            for event in display_events:
                level = str(event.get("candidate_level", "weak"))
                behavior = str(event.get("behavior", ""))
                priority = 2 if level == "strong" else 1
                for role_name, column in (("actor", "actor_id"), ("target", "target_id")):
                    try:
                        logical_id = int(event.get(column, -1))
                    except (TypeError, ValueError):
                        logical_id = -1
                    if logical_id < 0 or logical_id >= expected_mice:
                        continue
                    if role_name == "actor":
                        active_actor_ids.add(logical_id)
                    else:
                        active_target_ids.add(logical_id)
                    old = role_map.get(logical_id)
                    if old is None or priority > old[0]:
                        role_map[logical_id] = (
                            priority,
                            role_colors.get((level, behavior, role_name), default_color),
                        )

            for logical_id in range(expected_mice):
                if not bool(tracks["valid"][frame_index, logical_id]):
                    continue
                bbox = np.asarray(tracks["bboxes"][frame_index, logical_id], dtype=float)
                points = np.asarray(tracks["keypoints_px"][frame_index, logical_id], dtype=float)
                confidence = np.asarray(tracks["confidences"][frame_index, logical_id], dtype=float)
                if bbox.shape != (4,) or not np.all(np.isfinite(bbox)):
                    continue
                x1 = max(0, min(width - 1, int(round(float(bbox[0])))))
                y1 = max(0, min(height - 1, int(round(float(bbox[1])))))
                x2 = max(0, min(width - 1, int(round(float(bbox[2])))))
                y2 = max(0, min(height - 1, int(round(float(bbox[3])))))
                if x2 <= x1 or y2 <= y1:
                    continue
                role_info = role_map.get(logical_id)
                color = role_info[1] if role_info is not None else default_color
                thickness = 3 if role_info is not None else 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                role_suffix = ""
                if role_info is not None:
                    if logical_id in active_actor_ids and logical_id in active_target_ids:
                        role_suffix = " A/T"
                    elif logical_id in active_actor_ids:
                        role_suffix = " A"
                    else:
                        role_suffix = " T"
                label = f"ID{logical_id}{role_suffix}"
                (text_w, text_h), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
                )
                label_y = max(text_h + baseline + 2, y1)
                cv2.rectangle(
                    frame,
                    (x1, label_y - text_h - baseline - 2),
                    (x1 + text_w + 4, label_y + 2),
                    (0, 0, 0),
                    -1,
                )
                cv2.putText(
                    frame,
                    label,
                    (x1 + 2, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                    cv2.LINE_AA,
                )
                for first, second in skeleton_edges:
                    if (
                        first < len(points)
                        and second < len(points)
                        and first < len(confidence)
                        and second < len(confidence)
                        and float(confidence[first]) >= 0.10
                        and float(confidence[second]) >= 0.10
                        and np.all(np.isfinite(points[first]))
                        and np.all(np.isfinite(points[second]))
                    ):
                        p1 = tuple(np.rint(points[first]).astype(int).tolist())
                        p2 = tuple(np.rint(points[second]).astype(int).tolist())
                        cv2.line(frame, p1, p2, color, 1, cv2.LINE_AA)
                for point, point_conf in zip(points, confidence):
                    if float(point_conf) >= 0.10 and np.all(np.isfinite(point)):
                        cv2.circle(
                            frame,
                            tuple(np.rint(point).astype(int).tolist()),
                            2,
                            color,
                            -1,
                            cv2.LINE_AA,
                        )

            panel_lines = [
                f"frame {frame_index}/{frame_limit - 1}  time {frame_index / fps:.2f}s",
                f"active events: {len(active_events)}  unique displayed: {len(display_events)}",
            ]
            for event in display_events[:8]:
                level = str(event.get("candidate_level", "weak")).upper()
                behavior = str(event.get("behavior", "")).upper()
                pair = str(event.get("pair_key", ""))
                score = float(event.get("peak_score", 0.0))
                panel_lines.append(f"{level} {behavior} {pair}  score={score:.3f}")
            panel_height = 12 + 22 * len(panel_lines)
            cv2.rectangle(frame, (8, 8), (min(width - 8, 720), panel_height), (0, 0, 0), -1)
            for line_index, line in enumerate(panel_lines):
                cv2.putText(
                    frame,
                    line,
                    (18, 30 + line_index * 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
            cv2.putText(
                frame,
                "green=tracked  orange/blue=chase  red/magenta=attack  A=actor T=target",
                (12, height - 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            writer.write(frame)
            frame_index += 1
            if frame_index == 1 or frame_index % 500 == 0 or frame_index == frame_limit:
                LOGGER.info(
                    "[render] %d/%d frames (%.1f%%)",
                    frame_index,
                    frame_limit,
                    frame_index / max(frame_limit, 1) * 100,
                )
    finally:
        cap.release()
        writer.release()
    if frame_index <= 0 or not output_path.exists() or output_path.stat().st_size <= 0:
        raise RuntimeError(f"渲染没有产生有效视频: {output_path}")
    LOGGER.info(
        "[render] completed: %s (%.2f GB)",
        output_path,
        output_path.stat().st_size / (1024 ** 3),
    )
    return output_path


FOUR_CLASS_NAMES = {
    0: "00_非追逐非攻击",
    1: "01_非攻击性追逐",
    2: "02_非追逐攻击",
    3: "03_攻击性追逐",
}


def _boolean_runs(values: np.ndarray) -> list[tuple[int, int]]:
    values = np.asarray(values, dtype=bool)
    if values.size == 0:
        return []
    starts = np.flatnonzero(values & np.r_[True, ~values[:-1]])
    ends = np.flatnonzero(values & np.r_[~values[1:], True])
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def _clip_intervals_from_state(
    state: np.ndarray,
    label_id: int,
    fps: float,
    clip_seconds: float,
    min_start_interval_seconds: float,
    max_clips: int,
) -> list[tuple[int, int]]:
    total_frames = int(len(state))
    clip_frames = max(int(round(float(clip_seconds) * fps)), 1)
    min_interval = max(int(round(float(min_start_interval_seconds) * fps)), 1)
    if total_frames < clip_frames or max_clips <= 0:
        return []

    intervals: list[tuple[int, int]] = []
    if label_id == 0:
        # Class 0 is sampled only from windows with no strong chase/attack
        # state anywhere in the clip.  These are raw source frames, not
        # rendered or annotated negatives.
        for start in range(0, total_frames - clip_frames + 1, min_interval):
            end = start + clip_frames - 1
            if bool(np.all(state[start : end + 1] == 0)):
                intervals.append((int(start), int(end)))
                if len(intervals) >= max_clips:
                    break
        return intervals

    for run_start, run_end in _boolean_runs(state == label_id):
        center = (run_start + run_end) // 2
        start = max(0, min(center - clip_frames // 2, total_frames - clip_frames))
        end = start + clip_frames - 1
        if intervals and start - intervals[-1][0] < min_interval:
            continue
        intervals.append((int(start), int(end)))
        if len(intervals) >= max_clips:
            break
    return intervals


def extract_four_class_clips(
    video_path: Path,
    events_path: Path,
    output_dir: Path,
    expected_level: str = "strong",
    clip_seconds: float = 5.0,
    min_start_interval_seconds: float = 5.0,
    max_clips_per_class: int = 200,
) -> Path:
    """Extract raw source clips into the four mutually exclusive classes.

    Class IDs follow the established extractor contract:
    ``chase + 2 * attack`` -> 0/1/2/3.  This function never draws boxes,
    skeletons, IDs, labels, or overlays; it writes only source-video crops.
    """
    import cv2

    expected_level = str(expected_level).strip().lower()
    if expected_level not in {"weak", "strong"}:
        raise ValueError("expected_level 必须是 weak 或 strong")
    if not events_path.exists():
        raise FileNotFoundError(f"行为事件 CSV 不存在: {events_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开源视频: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if not np.isfinite(fps) or fps <= 0 or total_frames <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(
            f"源视频元数据无效: fps={fps}, frames={total_frames}, size={width}x{height}"
        )

    events_df = pd.read_csv(events_path)
    required = {"start_frame", "end_frame", "candidate_level", "behavior"}
    missing = sorted(required.difference(events_df.columns))
    if missing:
        raise ValueError(f"行为事件 CSV 缺少字段: {missing}")

    chase = np.zeros(total_frames, dtype=bool)
    attack = np.zeros(total_frames, dtype=bool)
    used_event_rows = 0
    for event in events_df.to_dict("records"):
        if str(event.get("candidate_level", "")).strip().lower() != expected_level:
            continue
        behavior = str(event.get("behavior", "")).strip().lower()
        if behavior not in {"chase", "attack"}:
            continue
        try:
            start = max(int(event.get("start_frame", 0)), 0)
            end = min(int(event.get("end_frame", -1)), total_frames - 1)
        except (TypeError, ValueError):
            continue
        if end < start:
            continue
        if behavior == "chase":
            chase[start : end + 1] = True
        else:
            attack[start : end + 1] = True
        used_event_rows += 1

    state = chase.astype(np.int8) + 2 * attack.astype(np.int8)
    output_dir.mkdir(parents=True, exist_ok=True)
    for class_name in FOUR_CLASS_NAMES.values():
        (output_dir / class_name).mkdir(parents=True, exist_ok=True)

    intervals: list[dict[str, Any]] = []
    per_class_counts: dict[int, int] = {}
    for label_id, class_name in FOUR_CLASS_NAMES.items():
        selected = _clip_intervals_from_state(
            state,
            label_id,
            fps,
            clip_seconds,
            min_start_interval_seconds,
            max_clips_per_class,
        )
        per_class_counts[label_id] = len(selected)
        for clip_index, (start, end) in enumerate(selected, start=1):
            path = output_dir / class_name / (
                f"{class_name}_{clip_index:04d}_{start / fps:.2f}s_{end / fps:.2f}s.mp4"
            )
            intervals.append(
                {
                    "clip_index": clip_index,
                    "label_id": label_id,
                    "label_name": class_name,
                    "start_frame": start,
                    "end_frame": end,
                    "start_time_s": start / fps,
                    "end_time_s": end / fps,
                    "duration_s": (end - start + 1) / fps,
                    "path": str(path),
                    "source_video": str(video_path),
                    "source_events": str(events_path),
                    "event_level": expected_level,
                }
            )

    # One sequential source read serves every clip; no annotations are drawn.
    start_map: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for interval in intervals:
        start_map[int(interval["start_frame"])].append(interval)
    active: dict[str, tuple[Any, dict[str, Any]]] = {}
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法重新打开源视频: {video_path}")
    frame_index = 0
    try:
        while frame_index < total_frames:
            ok, frame = cap.read()
            if not ok:
                break
            for interval in start_map.get(frame_index, []):
                writer = cv2.VideoWriter(
                    interval["path"],
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (width, height),
                )
                if not writer.isOpened():
                    writer.release()
                    raise RuntimeError(f"无法创建分类片段: {interval['path']}")
                active[interval["path"]] = (writer, interval)
            finished: list[str] = []
            for path, (writer, interval) in active.items():
                writer.write(frame)
                if frame_index >= int(interval["end_frame"]):
                    writer.release()
                    finished.append(path)
            for path in finished:
                active.pop(path, None)
            frame_index += 1
            if frame_index == 1 or frame_index % 1000 == 0 or frame_index == total_frames:
                LOGGER.info(
                    "[four-class clips] source %d/%d frames",
                    frame_index,
                    total_frames,
                )
    finally:
        cap.release()
        for writer, _interval in active.values():
            writer.release()

    manifest_path = output_dir / "four_class_clip_manifest.csv"
    _write_csv(manifest_path, intervals)
    summary = {
        "source_video": str(video_path),
        "events_csv": str(events_path),
        "event_level": expected_level,
        "fps": fps,
        "source_frames": total_frames,
        "clip_seconds": float(clip_seconds),
        "min_start_interval_seconds": float(min_start_interval_seconds),
        "max_clips_per_class": int(max_clips_per_class),
        "used_event_rows": int(used_event_rows),
        "state_frame_counts": {
            FOUR_CLASS_NAMES[label_id]: int((state == label_id).sum())
            for label_id in FOUR_CLASS_NAMES
        },
        "clip_counts": {
            FOUR_CLASS_NAMES[label_id]: int(per_class_counts.get(label_id, 0))
            for label_id in FOUR_CLASS_NAMES
        },
        "rendered_video": False,
    }
    with (output_dir / "four_class_clip_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    LOGGER.info(
        "[four-class clips] completed: %s",
        ", ".join(
            f"{FOUR_CLASS_NAMES[label_id]}={per_class_counts.get(label_id, 0)}"
            for label_id in FOUR_CLASS_NAMES
        ),
    )
    return output_dir


def extract_behavior_clips(
    video_path: Path,
    events_path: Path,
    output_dir: Path,
    behavior_names: Sequence[str] | None = None,
    event_level: str = "all",
    clip_seconds: float = 5.0,
    min_start_interval_seconds: float = 5.0,
    max_clips_per_behavior: int = 200,
) -> Path:
    """Extract fixed-length raw clips separately for each behavior label.

    This is the default clip contract for the lightweight ethogram.  It reads
    the already generated behavior-event CSV, builds one frame state per
    behavior, and writes raw source-video clips under one directory per label.
    It does not create four mutually exclusive classes and does not draw
    overlays.  ``event_level='all'`` keeps both the extended ethogram rows and
    the weak/strong legacy chase/attack rows.
    """
    import cv2

    event_level = str(event_level).strip().lower()
    if event_level not in {"all", "weak", "strong"}:
        raise ValueError("event_level 必须是 all、weak 或 strong")
    if not events_path.exists():
        raise FileNotFoundError(f"行为事件 CSV 不存在: {events_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开源视频: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if not np.isfinite(fps) or fps <= 0 or total_frames <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(
            f"源视频元数据无效: fps={fps}, frames={total_frames}, size={width}x{height}"
        )

    events_df = pd.read_csv(events_path)
    required = {"start_frame", "end_frame", "candidate_level", "behavior"}
    missing = sorted(required.difference(events_df.columns))
    if missing:
        raise ValueError(f"行为事件 CSV 缺少字段: {missing}")

    available = {
        str(value).strip().lower()
        for value in events_df["behavior"].dropna().tolist()
        if str(value).strip()
    }
    if behavior_names is None:
        selected_behaviors = [
            behavior
            for behavior in EXTENDED_BEHAVIORS
            if behavior in available
        ]
        selected_behaviors.extend(sorted(available.difference(selected_behaviors)))
    else:
        selected_behaviors = []
        for value in behavior_names:
            behavior = str(value).strip().lower()
            if behavior and behavior not in selected_behaviors:
                selected_behaviors.append(behavior)
    selected_behaviors = [behavior for behavior in selected_behaviors if behavior in available]
    if not selected_behaviors:
        raise ValueError("行为事件 CSV 中没有可切片的指定行为")

    event_rows_by_behavior: dict[str, list[dict[str, Any]]] = {
        behavior: [] for behavior in selected_behaviors
    }
    source_event_rows = {behavior: 0 for behavior in selected_behaviors}
    level_counts: dict[str, dict[str, int]] = {
        behavior: defaultdict(int) for behavior in selected_behaviors
    }
    for event in events_df.to_dict("records"):
        behavior = str(event.get("behavior", "")).strip().lower()
        if behavior not in event_rows_by_behavior:
            continue
        level = str(event.get("candidate_level", "")).strip().lower()
        if event_level != "all" and level != event_level:
            continue
        try:
            start = max(int(event.get("start_frame", 0)), 0)
            end = min(int(event.get("end_frame", -1)), total_frames - 1)
        except (TypeError, ValueError):
            continue
        if end < start:
            continue
        try:
            peak = int(event.get("peak_frame", (start + end) // 2))
        except (TypeError, ValueError):
            peak = (start + end) // 2
        event_rows_by_behavior[behavior].append(
            {
                "start_frame": start,
                "end_frame": end,
                "peak_frame": min(max(peak, start), end),
            }
        )
        source_event_rows[behavior] += 1
        level_counts[behavior][level or "extended"] += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    intervals: list[dict[str, Any]] = []
    clip_counts: dict[str, int] = {}
    clip_frames = max(int(round(float(clip_seconds) * fps)), 1)
    for behavior in selected_behaviors:
        behavior_dir = output_dir / behavior
        behavior_dir.mkdir(parents=True, exist_ok=True)
        selected: list[tuple[int, int]] = []
        min_interval = max(int(round(float(min_start_interval_seconds) * fps)), 1)
        for event in sorted(
            event_rows_by_behavior[behavior],
            key=lambda row: (int(row["peak_frame"]), int(row["start_frame"])),
        ):
            center = int(event["peak_frame"])
            start = max(0, min(center - clip_frames // 2, total_frames - clip_frames))
            end = start + clip_frames - 1
            if selected and start - selected[-1][0] < min_interval:
                continue
            selected.append((int(start), int(end)))
            if len(selected) >= max_clips_per_behavior:
                break
        clip_counts[behavior] = len(selected)
        for clip_index, (start, end) in enumerate(selected, start=1):
            path = behavior_dir / (
                f"{behavior}_{clip_index:04d}_{start / fps:.2f}s_{end / fps:.2f}s.mp4"
            )
            intervals.append(
                {
                    "clip_index": clip_index,
                    "behavior": behavior,
                    "behavior_name_zh": BEHAVIOR_NAMES_ZH.get(behavior, behavior),
                    "source_event_rows": int(source_event_rows[behavior]),
                    "event_level": event_level,
                    "start_frame": start,
                    "end_frame": end,
                    "start_time_s": start / fps,
                    "end_time_s": end / fps,
                    "duration_s": (end - start + 1) / fps,
                    "path": str(path),
                    "source_video": str(video_path),
                    "source_events": str(events_path),
                }
            )

    # One sequential source read serves all behavior clips; no annotations are drawn.
    start_map: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for interval in intervals:
        start_map[int(interval["start_frame"])].append(interval)
    active: dict[str, tuple[Any, dict[str, Any]]] = {}
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法重新打开源视频: {video_path}")
    frame_index = 0
    try:
        while frame_index < total_frames:
            ok, frame = cap.read()
            if not ok:
                break
            for interval in start_map.get(frame_index, []):
                writer = cv2.VideoWriter(
                    interval["path"],
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (width, height),
                )
                if not writer.isOpened():
                    writer.release()
                    raise RuntimeError(f"无法创建行为片段: {interval['path']}")
                active[interval["path"]] = (writer, interval)
            finished: list[str] = []
            for path, (writer, interval) in active.items():
                writer.write(frame)
                if frame_index >= int(interval["end_frame"]):
                    writer.release()
                    finished.append(path)
            for path in finished:
                active.pop(path, None)
            frame_index += 1
            if frame_index == 1 or frame_index % 1000 == 0 or frame_index == total_frames:
                LOGGER.info(
                    "[behavior clips] source %d/%d frames",
                    frame_index,
                    total_frames,
                )
    finally:
        cap.release()
        for writer, _interval in active.values():
            writer.release()

    manifest_path = output_dir / "behavior_clip_manifest.csv"
    _write_csv(manifest_path, intervals)
    summary = {
        "source_video": str(video_path),
        "events_csv": str(events_path),
        "event_level": event_level,
        "behaviors": selected_behaviors,
        "fps": fps,
        "source_frames": total_frames,
        "clip_seconds": float(clip_seconds),
        "clip_frames": clip_frames,
        "min_start_interval_seconds": float(min_start_interval_seconds),
        "max_clips_per_behavior": int(max_clips_per_behavior),
        "source_event_rows": source_event_rows,
        "source_event_level_counts": {
            behavior: dict(level_counts[behavior])
            for behavior in selected_behaviors
        },
        "clip_counts": clip_counts,
        "total_clips": int(len(intervals)),
        "rendered_video": False,
    }
    with (output_dir / "behavior_clip_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    LOGGER.info(
        "[behavior clips] completed: %s",
        ", ".join(f"{behavior}={clip_counts.get(behavior, 0)}" for behavior in selected_behaviors),
    )
    return output_dir


def _interaction_radius(config: Mapping[str, Any]) -> float:
    engine_cfg = dict(config.get("standard_behavior_engine", {}))
    interaction_cfg = dict(engine_cfg.get("interaction_graph", {}))
    if "radius_cm" in interaction_cfg:
        return float(interaction_cfg["radius_cm"])
    chase_cfg = config.get("chase", {})
    attack_cfg = config.get("attack", {})
    weak_chase = dict(chase_cfg.get("weak", {}))
    weak_attack = dict(attack_cfg.get("weak", {}))
    return max(
        float(weak_chase.get("max_distance_cm", 12.0)),
        float(weak_attack.get("body_center_contact_distance_cm", 6.0)),
    ) + float(interaction_cfg.get("buffer_cm", 5.0))


def _pair_prefilter(
    kin: Mapping[str, Any],
    config: Mapping[str, Any],
    interaction_radius: float,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Find spatially and directionally meaningful pair columns cheaply.

    The full pair feature builder contains the expensive nose-to-body
    geometry and rolling trajectory calculations.  This pass intentionally
    uses only center distance and body heading, so it can decide which stable
    logical pair columns deserve those features.

    A close-distance fallback is important: heading estimates can be noisy
    during contact or occlusion, while a very close pair is still relevant to
    nose/head/tail contact and attack detection.  For wider interactions at
    least one mouse must face the other within the configured cosine gate.
    The returned ``valuable_frame`` is frame-level diagnostic information; the
    analyzer promotes a pair if it is valuable at any analyzed frame and then
    keeps that pair's full time series for temporal FSM context.
    """
    lightweight_cfg = dict(config.get("lightweight_behavior_inference", {}))
    configured = dict(lightweight_cfg.get("pair_prefilter", {}))
    enabled = bool(configured.get("enabled", True))
    radius = max(float(interaction_radius), 0.0)
    close_distance = min(
        max(float(configured.get("close_distance_cm", 10.0)), 0.0),
        radius,
    )
    min_heading_cosine = float(
        np.clip(float(configured.get("min_heading_cosine", 0.0)), -1.0, 1.0)
    )

    centers = np.asarray(kin["centers_cm"], dtype=float)
    heading = np.asarray(kin["heading"], dtype=float)
    valid = np.asarray(kin["valid"], dtype=bool)
    frames, mice = valid.shape
    pair_i, pair_j = np.triu_indices(mice, k=1)
    valid_pair = valid[:, pair_i] & valid[:, pair_j]
    delta = centers[:, pair_j] - centers[:, pair_i]
    distance = np.linalg.norm(delta, axis=2)
    within_radius = valid_pair & np.isfinite(distance) & (distance <= radius)

    heading_to_partner_a = _cosine(heading[:, pair_i], delta)
    heading_to_partner_b = _cosine(heading[:, pair_j], -delta)
    heading_values_a = heading[:, pair_i]
    heading_values_b = heading[:, pair_j]
    heading_valid_a = (
        np.all(np.isfinite(heading_values_a), axis=2)
        & (np.linalg.norm(heading_values_a, axis=2) > 1e-9)
    )
    heading_valid_b = (
        np.all(np.isfinite(heading_values_b), axis=2)
        & (np.linalg.norm(heading_values_b, axis=2) > 1e-9)
    )
    heading_relevant = (
        (heading_valid_a & (heading_to_partner_a >= min_heading_cosine))
        | (heading_valid_b & (heading_to_partner_b >= min_heading_cosine))
    )
    close_fallback = distance <= close_distance
    valuable_frame = within_radius & (close_fallback | heading_relevant)

    if enabled:
        candidate_pair_mask = np.any(valuable_frame, axis=0)
    else:
        # Compatibility switch: preserve the previous distance-only candidate
        # rule when the caller explicitly disables the orientation prefilter.
        valuable_frame = within_radius
        candidate_pair_mask = np.any(within_radius, axis=0)

    return {
        "valid_pair": valid_pair,
        "distance": distance,
        "heading_to_partner_a": heading_to_partner_a,
        "heading_to_partner_b": heading_to_partner_b,
        "valuable_frame": valuable_frame,
        "candidate_pair_mask": candidate_pair_mask,
        "enabled": enabled,
        "close_distance_cm": close_distance,
        "min_heading_cosine": min_heading_cosine,
        "frames": int(frames),
    }, pair_i, pair_j


def _pair_window_mask(
    valuable_frame: np.ndarray,
    fps: float,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Expand valuable pair frames into padded windows for expensive metrics.

    The window is expanded on both sides so causal rolling features and FSM
    context are still available when a pair first becomes relevant.  Short
    gaps inside a valuable run are filled before expansion, which prevents a
    noisy heading estimate from fragmenting one interaction into many small
    windows.
    """
    valuable_frame = np.asarray(valuable_frame, dtype=bool)
    if valuable_frame.ndim != 2:
        raise ValueError(
            f"valuable_frame must be a 2D array, got shape={valuable_frame.shape}"
        )
    lightweight_cfg = dict(config.get("lightweight_behavior_inference", {}))
    prefilter_cfg = dict(lightweight_cfg.get("pair_prefilter", {}))
    window_cfg = dict(prefilter_cfg.get("window", {}))
    enabled = bool(window_cfg.get("enabled", True))
    padding_seconds = max(float(window_cfg.get("padding_seconds", 2.5)), 0.0)
    fill_gap_seconds = max(float(window_cfg.get("fill_gap_seconds", 0.15)), 0.0)
    padding_frames = max(int(math.ceil(padding_seconds * max(float(fps), 0.0))), 0)
    fill_gap_frames = max(int(math.ceil(fill_gap_seconds * max(float(fps), 0.0))), 0)

    if not enabled:
        return np.ones_like(valuable_frame, dtype=bool), {
            "enabled": False,
            "padding_seconds": padding_seconds,
            "fill_gap_seconds": fill_gap_seconds,
            "padding_frames": padding_frames,
            "fill_gap_frames": fill_gap_frames,
            "active_frame_count": int(valuable_frame.size),
            "active_frame_fraction": 1.0 if valuable_frame.size else 0.0,
        }

    frames, pairs = valuable_frame.shape
    window_mask = np.zeros_like(valuable_frame, dtype=bool)
    for pair_index in range(pairs):
        runs = _boolean_runs_with_gap(valuable_frame[:, pair_index], fill_gap_frames)
        for start, end in runs:
            window_start = max(start - padding_frames, 0)
            window_end = min(end + padding_frames, frames - 1)
            if window_start <= window_end:
                window_mask[window_start : window_end + 1, pair_index] = True

    return window_mask, {
        "enabled": True,
        "padding_seconds": padding_seconds,
        "fill_gap_seconds": fill_gap_seconds,
        "padding_frames": padding_frames,
        "fill_gap_frames": fill_gap_frames,
        "active_frame_count": int(window_mask.sum()),
        "active_frame_fraction": float(window_mask.mean()) if window_mask.size else 0.0,
    }


@dataclass(frozen=True)
class _PairWorkset:
    """All-pair prefilter state and candidate-only metric arrays for one run."""

    interaction_radius: float
    prefilter: Mapping[str, Any]
    all_pair_i: np.ndarray
    all_pair_j: np.ndarray
    candidate_pair_indices: tuple[int, ...]
    candidate_metric_index: Mapping[int, int]
    candidate_frame_mask: np.ndarray
    pair_window_stats: Mapping[str, Any]
    metrics: Mapping[str, Any]
    pair_i: np.ndarray
    pair_j: np.ndarray


def _prepare_pair_workset(
    kin: Mapping[str, Any],
    fps: float,
    config: Mapping[str, Any],
    *,
    stage_timings: dict[str, float] | None = None,
) -> _PairWorkset:
    """Prepare vectorized all-pair gates and candidate-only heavy metrics."""

    pair_filter_timer = Timer(
        "pair_filter_and_windows",
        logger=LOGGER,
        sink=stage_timings,
    ).start()
    interaction_radius = _interaction_radius(config)
    prefilter, all_pair_i, all_pair_j = _pair_prefilter(
        kin,
        config,
        interaction_radius,
    )
    candidate_pair_indices = tuple(
        int(index)
        for index in np.flatnonzero(
            np.asarray(prefilter["candidate_pair_mask"], dtype=bool)
        )
    )
    candidate_pair_indices_array = np.asarray(candidate_pair_indices, dtype=int)
    pair_window_mask, pair_window_stats = _pair_window_mask(
        np.asarray(prefilter["valuable_frame"], dtype=bool),
        fps,
        config,
    )
    candidate_frame_mask = pair_window_mask[:, candidate_pair_indices_array]
    pair_filter_timer.stop()

    pair_metrics_timer = Timer(
        "pair_metrics",
        logger=LOGGER,
        sink=stage_timings,
    ).start()
    metrics, pair_i, pair_j = _pair_metrics(
        kin,
        fps,
        pair_indices=candidate_pair_indices_array,
        frame_mask=candidate_frame_mask,
    )
    pair_metrics_timer.stop()
    candidate_metric_index = {
        int(original_index): int(metric_index)
        for metric_index, original_index in enumerate(candidate_pair_indices)
    }
    LOGGER.info(
        "[pair filter] %d/%d pairs retained (distance <= %.2f cm, close fallback <= %.2f cm, heading cosine >= %.2f)",
        len(candidate_pair_indices),
        len(all_pair_i),
        interaction_radius,
        float(prefilter["close_distance_cm"]),
        float(prefilter["min_heading_cosine"]),
    )
    LOGGER.info(
        "[pair windows] active %.1f%% of pair-frame slots (padding %.2fs, fill gap %.2fs)",
        100.0 * float(pair_window_stats["active_frame_fraction"]),
        float(pair_window_stats["padding_seconds"]),
        float(pair_window_stats["fill_gap_seconds"]),
    )
    return _PairWorkset(
        interaction_radius=float(interaction_radius),
        prefilter=prefilter,
        all_pair_i=all_pair_i,
        all_pair_j=all_pair_j,
        candidate_pair_indices=candidate_pair_indices,
        candidate_metric_index=candidate_metric_index,
        candidate_frame_mask=candidate_frame_mask,
        pair_window_stats=pair_window_stats,
        metrics=metrics,
        pair_i=pair_i,
        pair_j=pair_j,
    )


@dataclass
class _PairAnalysisResult:
    """In-memory outputs produced by the candidate-pair analysis stage."""

    events: list[dict[str, Any]]
    contact_events: list[dict[str, Any]]
    extended_events: list[dict[str, Any]]
    pair_summaries: list[dict[str, Any]]
    top_evidence: list[dict[str, Any]]
    fsm_coordinator: ParallelBehaviorFSM


def _analyze_candidate_pairs(
    workset: _PairWorkset,
    kin: Mapping[str, Any],
    *,
    fps: float,
    source_fps: float,
    sample_stride: int,
    video_path: Path,
    config: Mapping[str, Any],
    stage_timings: dict[str, float] | None = None,
) -> _PairAnalysisResult:
    """Analyze retained pairs while preserving the complete pair timeline."""

    events: list[dict[str, Any]] = []
    contact_events: list[dict[str, Any]] = []
    extended_events: list[dict[str, Any]] = []
    pair_summaries: list[dict[str, Any]] = []
    top_evidence: list[dict[str, Any]] = []
    pair_analysis_timer = Timer(
        "candidate_pair_analysis",
        logger=LOGGER,
        sink=stage_timings,
    ).start()
    pair_fsm_coordinator = ParallelBehaviorFSM(
        dict(config.get("parallel_behavior_fsm", {}))
    )
    candidate_ordinal = 0
    for pair_index, (mouse_a, mouse_b) in enumerate(
        zip(workset.all_pair_i, workset.all_pair_j)
    ):
        metric_index = workset.candidate_metric_index.get(pair_index)
        base_summary: dict[str, Any] = {
            "pair_key": f"{int(mouse_a)}_{int(mouse_b)}",
            "mouse_a_id": int(mouse_a),
            "mouse_b_id": int(mouse_b),
            "valid_frames": int(
                np.asarray(
                    workset.prefilter["valid_pair"][:, pair_index],
                    dtype=bool,
                ).sum()
            ),
            "min_distance_cm": float(
                np.nanmin(workset.prefilter["distance"][:, pair_index])
            )
            if np.isfinite(workset.prefilter["distance"][:, pair_index]).any()
            else float("nan"),
            "max_speed_cm_s": float(
                max(
                    float(
                        np.asarray(
                            workset.metrics["speed"][:, int(mouse_a)]
                        ).max(initial=0.0)
                    ),
                    float(
                        np.asarray(
                            workset.metrics["speed"][:, int(mouse_b)]
                        ).max(initial=0.0)
                    ),
                )
            ),
            "engine_evaluated": metric_index is not None,
            "fsm_evaluated_frames": 0,
            "fsm_skipped_frames": 0,
            "fsm_evaluated_fraction": 0.0,
            "nose_head_contact_event_count": 0,
            "nose_tail_contact_event_count": 0,
            "combined_nose_head_nose_tail_event_count": 0,
            "contact_sample_count": 0,
        }
        if metric_index is None:
            for level in ("weak", "strong"):
                for behavior in ("chase", "attack"):
                    base_summary[f"{level}_{behavior}_frames"] = 0
                    base_summary[f"{level}_{behavior}_max_score"] = 0.0
                    base_summary[f"{level}_{behavior}_role_known_rate"] = None
            pair_summaries.append(base_summary)
            continue

        candidate_ordinal += 1
        if (
            pair_index == 0
            or pair_index % 20 == 0
            or pair_index == len(workset.all_pair_i) - 1
        ):
            LOGGER.info(
                "[pair analysis] pair %d/%d (%d/%d candidates)",
                pair_index + 1,
                len(workset.all_pair_i),
                candidate_ordinal,
                len(workset.candidate_pair_indices),
            )
        pair_df = _pair_dataframe(
            workset.metrics,
            metric_index,
            int(mouse_a),
            int(mouse_b),
            workset.pair_i,
            workset.pair_j,
            fps,
            float(kin["cm_per_pixel"]),
        )
        pair_contact_events = _extract_contact_events(
            pair_df,
            pair_key=f"{int(mouse_a)}_{int(mouse_b)}",
            source_video=video_path,
            source_fps=source_fps,
            sample_stride=sample_stride,
            contact_config=dict(config.get("contact_detection", {})),
            fsm_coordinator=pair_fsm_coordinator,
        )
        contact_events.extend(pair_contact_events)
        enriched = behavior_engine.apply_standard_behavior_engine(
            pair_df,
            fps,
            config,
            copy_input=False,
        )
        extended_events.extend(
            _extended_pair_events(
                pair_df,
                metrics=workset.metrics,
                pair_index=metric_index,
                enriched=enriched,
                source_video=video_path,
                source_fps=source_fps,
                sample_stride=sample_stride,
                config=config,
                fsm_coordinator=pair_fsm_coordinator,
            )
        )
        extended_events.extend(
            _extended_short_clip_pair_events(
                pair_df,
                enriched,
                source_video=video_path,
                source_fps=source_fps,
                sample_stride=sample_stride,
                config=config,
                fsm_coordinator=pair_fsm_coordinator,
            )
        )
        summary = base_summary
        fsm_compute = enriched.get(
            "standard_behavior_compute_row",
            pair_df.get("valid_pair", pd.Series(True, index=pair_df.index)),
        ).fillna(False).astype(bool)
        summary["fsm_evaluated_frames"] = int(fsm_compute.sum())
        summary["fsm_skipped_frames"] = int(len(fsm_compute) - fsm_compute.sum())
        summary["fsm_evaluated_fraction"] = (
            float(fsm_compute.mean()) if len(fsm_compute) else 0.0
        )
        summary["nose_head_contact_event_count"] = int(
            sum(
                "nose_head"
                in str(event.get("contact_type_components", "")).split(";")
                for event in pair_contact_events
            )
        )
        summary["nose_tail_contact_event_count"] = int(
            sum(
                "nose_tail"
                in str(event.get("contact_type_components", "")).split(";")
                for event in pair_contact_events
            )
        )
        summary["combined_nose_head_nose_tail_event_count"] = int(
            sum(
                event.get("contact_type") == "nose_head_and_nose_tail"
                for event in pair_contact_events
            )
        )
        summary["contact_sample_count"] = int(
            sum(int(event.get("sample_count", 0)) for event in pair_contact_events)
        )
        for level in ("weak", "strong"):
            for behavior in ("chase", "attack"):
                mask_col = f"{level}_standard_final_{behavior}"
                score_col = f"{level}_standard_{behavior}_score"
                actor_col = f"{level}_standard_{behavior}_actor_id"
                target_col = f"{level}_standard_{behavior}_target_id"
                active = (
                    enriched[mask_col].fillna(False).astype(bool)
                    if mask_col in enriched
                    else pd.Series(False, index=enriched.index)
                )
                summary[f"{level}_{behavior}_frames"] = int(active.sum())
                summary[f"{level}_{behavior}_max_score"] = (
                    float(enriched[score_col].max())
                    if score_col in enriched
                    else 0.0
                )
                if active.any() and actor_col in enriched and target_col in enriched:
                    known = (
                        pd.to_numeric(
                            enriched.loc[active, actor_col], errors="coerce"
                        )
                        >= 0
                    ) & (
                        pd.to_numeric(
                            enriched.loc[active, target_col], errors="coerce"
                        )
                        >= 0
                    )
                    summary[f"{level}_{behavior}_role_known_rate"] = float(
                        known.mean()
                    )
                else:
                    summary[f"{level}_{behavior}_role_known_rate"] = None
            for event in behavior_engine.extract_standard_behavior_events(
                enriched,
                fps,
                level,
                pair_key=f"{int(mouse_a)}_{int(mouse_b)}",
            ):
                event = dict(event)
                event["source_video"] = str(video_path)
                event["analysis_mode"] = "lightweight_cache_tracking"
                event["analysis_start_frame"] = int(event.get("start_frame", 0))
                event["analysis_peak_frame"] = int(event.get("peak_frame", 0))
                event["analysis_end_frame"] = int(event.get("end_frame", 0))
                event["start_frame"] = (
                    int(event.get("start_frame", 0)) * sample_stride
                )
                event["peak_frame"] = (
                    int(event.get("peak_frame", 0)) * sample_stride
                )
                event["end_frame"] = int(event.get("end_frame", 0)) * sample_stride
                events.append(event)
        for behavior in ("chase", "attack"):
            score_column = f"strong_standard_{behavior}_score"
            if score_column not in enriched:
                continue
            candidate = enriched[
                ["frame", "time_s", "pair_key", "center_distance_cm", score_column]
            ].copy()
            candidate["behavior"] = behavior
            candidate["level"] = "strong"
            candidate = candidate.rename(columns={score_column: "score"})
            evidence_rows = candidate.nlargest(5, "score").to_dict("records")
            for row in evidence_rows:
                row["analysis_frame"] = int(row["frame"])
                row["source_frame"] = int(row["frame"]) * sample_stride
                row["source_time_s"] = (
                    float(row["frame"]) * sample_stride / source_fps
                )
            top_evidence.extend(evidence_rows)
        pair_summaries.append(summary)
    pair_analysis_timer.stop()
    return _PairAnalysisResult(
        events=events,
        contact_events=contact_events,
        extended_events=extended_events,
        pair_summaries=pair_summaries,
        top_evidence=top_evidence,
        fsm_coordinator=pair_fsm_coordinator,
    )


def _finalize_event_records_in_place(
    events: list[dict[str, Any]],
    contact_events: list[dict[str, Any]],
    source_fps: float,
) -> None:
    """Assign stable IDs and source-time fields without changing event order."""

    for index, event in enumerate(
        sorted(
            events,
            key=lambda item: (
                int(item.get("start_frame", 0)),
                str(item.get("pair_key", "")),
                str(item.get("level", "")),
            ),
        ),
        start=1,
    ):
        event["light_event_id"] = f"LWE{index:05d}"
        event["start_time_s"] = float(event.get("start_frame", 0)) / source_fps
        event["end_time_s"] = float(event.get("end_frame", 0)) / source_fps
        event["duration_s"] = (
            float(event.get("end_frame", 0) - event.get("start_frame", 0) + 1)
            / source_fps
        )

    for index, event in enumerate(
        sorted(
            contact_events,
            key=lambda item: (
                int(item.get("start_frame", 0)),
                str(item.get("pair_key", "")),
                str(item.get("contact_type", "")),
            ),
        ),
        start=1,
    ):
        event["contact_event_id"] = f"LCE{index:05d}"


def analyze(
    video_path: Path,
    cache_dir: Path,
    config_path: Path,
    output_dir: Path,
    expected_mice: int = 20,
    max_frames: int | None = None,
    sample_stride: int = 1,
    fps_override: float | None = None,
) -> Path:
    started = time.perf_counter()
    stage_timings: dict[str, float] = {}
    setup_timer = Timer(
        "setup_and_video_probe",
        logger=LOGGER,
        sink=stage_timings,
    ).start()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)
    configured_fps = config.pop("_fps_override", 29.329)
    source_fps = float(configured_fps if fps_override is None else fps_override)
    if not np.isfinite(source_fps) or source_fps <= 0.0:
        raise ValueError(f"视频 FPS 必须是正数，实际为：{source_fps}")
    sample_stride = max(int(sample_stride), 1)
    total_frames = _cache_total_frames(cache_dir)
    if max_frames is not None:
        total_frames = min(total_frames, max(int(max_frames), 1))
    video_cap = cv2.VideoCapture(str(video_path))
    if not video_cap.isOpened():
        raise RuntimeError(f"无法打开源视频: {video_path}")
    width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_frame_count = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = float(video_cap.get(cv2.CAP_PROP_FPS))
    video_cap.release()
    if width <= 0 or height <= 0:
        raise RuntimeError(f"无法读取源视频尺寸: {video_path}")
    setup_timer.stop()

    arena_timer = Timer(
        "arena_boundary",
        logger=LOGGER,
        sink=stage_timings,
    ).start()
    arena_result, _arena_heatmap = _prepare_video_arena_boundary(
        video_path,
        cache_dir,
        output_dir,
        config_path,
        config,
        width,
        height,
        max_frames=total_frames,
    )
    arena_polygon = (
        np.asarray(arena_result.polygon, dtype=np.float64)
        if arena_result is not None
        else None
    )
    arena_tolerance = float(
        dict(config.get("adaptive_arena", {})).get("hard_gate_tolerance_px", 2.0)
    )
    arena_timer.stop()

    track_timer = Timer(
        "track_cache",
        logger=LOGGER,
        sink=stage_timings,
    ).start()
    tracks, tracking_stats = _track_cache(
        cache_dir,
        total_frames,
        expected_mice,
        arena_polygon=arena_polygon,
        arena_tolerance_px=arena_tolerance,
    )
    # Preserve full-resolution source-frame tracks for the annotation website.
    # Behavior analysis may sample this array below; the export contract may not.
    source_tracks = tracks
    track_timer.stop()

    kinematics_timer = Timer(
        "kinematics",
        logger=LOGGER,
        sink=stage_timings,
    ).start()
    if sample_stride > 1:
        tracks = {
            key: (value[::sample_stride] if isinstance(value, np.ndarray) and value.ndim > 0 else value)
            for key, value in tracks.items()
        }
    analysis_frames = int(tracks["valid"].shape[0])
    fps = source_fps / sample_stride
    kin = _kinematics(tracks, fps=fps)
    kinematics_timer.stop()
    pair_workset = _prepare_pair_workset(
        kin,
        fps,
        config,
        stage_timings=stage_timings,
    )
    pair_analysis = _analyze_candidate_pairs(
        pair_workset,
        kin,
        fps=fps,
        source_fps=source_fps,
        sample_stride=sample_stride,
        video_path=video_path,
        config=config,
        stage_timings=stage_timings,
    )
    events = pair_analysis.events
    contact_events = pair_analysis.contact_events
    extended_events = pair_analysis.extended_events
    pair_summaries = pair_analysis.pair_summaries
    top_evidence = pair_analysis.top_evidence
    pair_fsm_coordinator = pair_analysis.fsm_coordinator

    interaction_radius = pair_workset.interaction_radius
    prefilter = pair_workset.prefilter
    all_pair_i = pair_workset.all_pair_i
    all_pair_j = pair_workset.all_pair_j
    candidate_pair_indices = pair_workset.candidate_pair_indices
    candidate_frame_mask = pair_workset.candidate_frame_mask
    pair_window_stats = pair_workset.pair_window_stats
    metrics = pair_workset.metrics
    pair_i = pair_workset.pair_i
    pair_j = pair_workset.pair_j

    global_events_timer = Timer(
        "global_events_and_finalization",
        logger=LOGGER,
        sink=stage_timings,
    ).start()
    if bool(_extended_behavior_config(config).get("enabled", True)):
        extended_events.extend(
            _extended_individual_and_group_events(
                kin,
                pair_metrics=metrics,
                pair_i=pair_i,
                pair_j=pair_j,
                source_video=video_path,
                source_fps=source_fps,
                sample_stride=sample_stride,
                config=config,
            )
        )
    events.extend(extended_events)
    _finalize_event_records_in_place(events, contact_events, source_fps)
    global_events_timer.stop()

    website_frame_count = (
        int(video_frame_count) if int(video_frame_count) > 0 else max(int(total_frames), 1)
    )
    website_fps = (
        float(video_fps)
        if np.isfinite(video_fps) and float(video_fps) > 0.0
        else float(source_fps)
    )
    extended_cfg = _extended_behavior_config(config)
    website_timer = Timer(
        "website_export",
        logger=LOGGER,
        sink=stage_timings,
    ).start()
    website_export = export_complete_video_package(
        source_video=video_path,
        output_dir=output_dir,
        behavior_events=events,
        contact_events=contact_events,
        tracks=source_tracks,
        fps=website_fps,
        frame_count=website_frame_count,
        width=width,
        height=height,
        skeleton_edges=SKELETON_EDGES,
        cm_per_pixel=float(kin["cm_per_pixel"]),
        huddle_distance_cm=float(
            dict(extended_cfg.get("group", {})).get("huddle_distance_cm", 9.0)
        ),
        tracker_params={
            "expected_mice": int(expected_mice),
            "sample_stride": int(sample_stride),
            "behavior_analysis_fps": float(source_fps),
            "video_container_fps": float(website_fps),
            "analysis_mode": "lightweight_cache_tracking",
            "full_pipeline_not_run": True,
        },
    )
    website_timer.stop()

    csv_timer = Timer(
        "csv_output",
        logger=LOGGER,
        sink=stage_timings,
    ).start()
    _write_csv(output_dir / "lightweight_behavior_events.csv", events)
    _write_csv(
        output_dir / "lightweight_contact_events.csv",
        contact_events,
        columns=CONTACT_EVENT_COLUMNS,
    )
    _write_csv(output_dir / "lightweight_pair_summary.csv", pair_summaries)
    _write_csv(output_dir / "lightweight_top_evidence.csv", top_evidence)
    fsm_evaluated_pair_frames = int(
        sum(int(summary.get("fsm_evaluated_frames", 0)) for summary in pair_summaries)
    )
    fsm_candidate_timeline_frames = int(analysis_frames * len(candidate_pair_indices))
    fsm_skipped_pair_frames = max(
        fsm_candidate_timeline_frames - fsm_evaluated_pair_frames,
        0,
    )
    csv_timer.stop()
    metadata = {
        "source_video": str(video_path),
        "yolo_cache": str(cache_dir),
        "config": str(config_path),
        "analysis_mode": "lightweight_cache_tracking",
        "full_pipeline_not_run": True,
        "tracker": "position_plus_keypoint_hungarian",
        "expected_mice": int(expected_mice),
        "source_frames": int(total_frames),
        "analysis_frames": int(analysis_frames),
        "source_fps": float(source_fps),
        "analysis_fps": float(fps),
        "sample_stride": int(sample_stride),
        "duration_s": float(total_frames / source_fps),
        "cm_per_pixel": float(kin["cm_per_pixel"]),
        "reference_body_px": float(kin["reference_body_px"]),
        "tracking": tracking_stats,
        "arena_boundary": (
            asdict(arena_result)
            if arena_result is not None
            else None
        ),
        "event_counts": {
            f"{level}_{behavior}": int(
                sum(
                    1
                    for event in events
                    if event.get("candidate_level") == level
                    and event.get("behavior") == behavior
                )
            )
            for level in ("weak", "strong")
            for behavior in ("chase", "attack")
        },
        "extended_behavior_counts": {
            behavior: int(sum(event.get("behavior") == behavior for event in events))
            for behavior in EXTENDED_BEHAVIORS
        },
        "extended_behavior_scopes": {
            scope: int(sum(event.get("event_scope") == scope for event in events))
            for scope in ("pair", "individual", "group")
        },
        "contact_event_counts": {
            contact_type: int(
                sum(event.get("contact_type") == contact_type for event in contact_events)
            )
            for contact_type in (
                "nose_head",
                "nose_tail",
                "nose_head_and_nose_tail",
            )
        },
        "contact_event_csv": str(output_dir / "lightweight_contact_events.csv"),
        "annotation_website_export": website_export,
        "notes": [
            "仅读取指定视频的完整 YOLO 预推理缓存，没有读取其他行为目录。",
            "该结果用于当前长视频的快速行为筛查；它不包含完整流水线的遮挡簇 ReID、ROI Pose 恢复和伪掩码身份保护。",
            "行为事件 CSV 同时包含 legacy chase/attack 与扩展 ethogram 标签；鼻头/鼻尾接触写入独立 lightweight_contact_events.csv，接触本身不会单独升级为 attack。",
            "候选鼠对的昂贵鼻体几何、滚动轨迹特征和标准行为连续证据只在距离/朝向窗口及其上下文 padding 内计算；窗口外帧保留在输出时间轴中并作为FSM硬否决/状态重置行。",
            "新视频没有人工行为标签，因此不能据此计算 Precision、Recall、F1 或 actor/target accuracy；事件中的角色 ID 和 role confidence 仅是模型诊断。",
        ],
        "interaction_radius_cm": float(interaction_radius),
        "candidate_pair_count": int(len(candidate_pair_indices)),
        "total_pair_count": int(len(all_pair_i)),
        "standard_behavior_engine": {
            "skip_inactive_rows": bool(
                dict(config.get("standard_behavior_engine", {})).get(
                    "skip_inactive_rows", True
                )
            ),
            "evaluated_pair_frame_count": fsm_evaluated_pair_frames,
            "skipped_pair_frame_count": fsm_skipped_pair_frames,
            "candidate_timeline_pair_frame_count": fsm_candidate_timeline_frames,
            "evaluated_pair_frame_fraction": float(
                fsm_evaluated_pair_frames / fsm_candidate_timeline_frames
            )
            if fsm_candidate_timeline_frames
            else 0.0,
        },
        "parallel_behavior_fsm": {
            "enabled": bool(pair_fsm_coordinator.enabled),
            "mode": str(pair_fsm_coordinator.mode),
            "version": ParallelBehaviorFSM.VERSION,
            "collect_diagnostics": bool(pair_fsm_coordinator.collect_diagnostics),
            "execution_semantics": (
                "active_temporal_regions"
                if pair_fsm_coordinator.enabled
                else "disabled_no_parallel_events"
            ),
            "regions": {
                "individual": "stationary|walking|running per mouse",
                "pair": "together|approach|chase|avoidance|attack per pair",
                "contact": "nose_head|nose_tail|combined per pair",
                "group": "huddle|isolation per video",
            },
        },
        "pair_prefilter": {
            "enabled": bool(prefilter["enabled"]),
            "close_distance_cm": float(prefilter["close_distance_cm"]),
            "min_heading_cosine": float(prefilter["min_heading_cosine"]),
            "valuable_frame_count": int(np.asarray(prefilter["valuable_frame"], dtype=bool).sum()),
            "valuable_frame_fraction": float(
                np.asarray(prefilter["valuable_frame"], dtype=bool).mean()
            )
            if np.asarray(prefilter["valuable_frame"]).size
            else 0.0,
        },
        "pair_window": {
            **pair_window_stats,
            "candidate_active_frame_count": int(candidate_frame_mask.sum()),
            "candidate_active_frame_fraction": float(candidate_frame_mask.mean())
            if candidate_frame_mask.size
            else 0.0,
        },
        "stage_timings_s": stage_timings,
        "elapsed_s": float(time.perf_counter() - started),
    }
    with (output_dir / "lightweight_analysis_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--yolo-cache", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=False)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, required=False)
    parser.add_argument("--expected-mice", type=int, default=20)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--sample-stride",
        type=int,
        default=3,
        help="Analyze every Nth cached frame; FPS is reduced by the same factor.",
    )
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="只读取已有事件 CSV，输出一个带框/骨架/行为标签的 MP4，不生成事件片段。",
    )
    parser.add_argument(
        "--events",
        type=Path,
        default=None,
        help="已有 lightweight_behavior_events.csv；render-only 时默认从 output-dir 读取。",
    )
    parser.add_argument(
        "--render-output",
        type=Path,
        default=None,
        help="render-only 的唯一 MP4 输出路径。",
    )
    parser.add_argument(
        "--extract-four-class-clips",
        action="store_true",
        help="兼容旧接口：只从已有事件 CSV 裁剪四类原始视频，不生成渲染视频。",
    )
    parser.add_argument(
        "--extract-behavior-clips",
        action="store_true",
        help="按 lightweight_behavior_events.csv 中的行为名称分别裁剪原始视频。",
    )
    parser.add_argument(
        "--behaviors",
        nargs="+",
        default=None,
        help="行为切片名称；默认输出事件 CSV 中出现的全部行为。",
    )
    parser.add_argument(
        "--behavior-level",
        choices=("all", "weak", "strong"),
        default="all",
        help="行为切片保留的 candidate_level；默认 all，也包含扩展 ethogram 行。",
    )
    parser.add_argument(
        "--behavior-clips-output",
        type=Path,
        default=None,
        help="按行为切片的输出目录；默认 output-dir/behavior_clips。",
    )
    parser.add_argument(
        "--behavior-clip-seconds",
        type=float,
        default=5.0,
        help="每个行为切片的长度，默认 5 秒。",
    )
    parser.add_argument(
        "--max-clips-per-behavior",
        type=int,
        default=200,
        help="每种行为最多输出的片段数，默认 200。",
    )
    parser.add_argument(
        "--clip-level",
        choices=("weak", "strong"),
        default="strong",
        help="四类裁剪使用 weak 或 strong 事件层，默认 strong。",
    )
    parser.add_argument(
        "--clips-output",
        type=Path,
        default=None,
        help="四类视频输出目录；默认 output-dir/四类视频。",
    )
    parser.add_argument(
        "--clip-seconds",
        type=float,
        default=5.0,
        help="每个原始视频片段的长度，默认 5 秒。",
    )
    parser.add_argument(
        "--max-clips-per-class",
        type=int,
        default=200,
        help="每类最多输出的片段数，默认 200。",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        default="INFO",
        help="日志级别，默认 INFO。",
    )
    args = parser.parse_args()
    configure_logging(args.log_level)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.render_only:
        events_path = args.events or (args.output_dir / "lightweight_behavior_events.csv")
        render_output = args.render_output or (args.output_dir / "part001_追逐攻击渲染.mp4")
        render_behavior_video(
            args.video,
            args.yolo_cache,
            events_path,
            render_output,
            expected_mice=max(int(args.expected_mice), 2),
            max_frames=args.max_frames,
        )
        LOGGER.info("render_output=%s", render_output)
        return 0

    if args.extract_behavior_clips:
        events_path = args.events or (args.output_dir / "lightweight_behavior_events.csv")
        clips_output = args.behavior_clips_output or (args.output_dir / "behavior_clips")
        extract_behavior_clips(
            args.video,
            events_path,
            clips_output,
            behavior_names=args.behaviors,
            event_level=args.behavior_level,
            clip_seconds=max(float(args.behavior_clip_seconds), 0.1),
            max_clips_per_behavior=max(int(args.max_clips_per_behavior), 1),
        )
        LOGGER.info("behavior_clips_output=%s", clips_output)
        return 0

    if args.extract_four_class_clips:
        events_path = args.events or (args.output_dir / "lightweight_behavior_events.csv")
        clips_output = args.clips_output or (args.output_dir / "四类视频")
        extract_four_class_clips(
            args.video,
            events_path,
            clips_output,
            expected_level=args.clip_level,
            clip_seconds=max(float(args.clip_seconds), 0.1),
            max_clips_per_class=max(int(args.max_clips_per_class), 1),
        )
        LOGGER.info("clips_output=%s", clips_output)
        return 0

    if args.config is None or args.fps is None:
        parser.error("普通分析模式需要同时提供 --config 和 --fps；渲染已有结果请使用 --render-only。")
    # Keep the function signature self-contained while passing the video FPS.
    config = load_config(args.config)
    config["_fps_override"] = float(args.fps)
    temp_config = args.output_dir / ".lightweight_runtime_config.yaml"
    with temp_config.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
    try:
        result_dir = analyze(
            args.video,
            args.yolo_cache,
            temp_config,
            args.output_dir,
            expected_mice=max(int(args.expected_mice), 2),
            max_frames=args.max_frames,
            sample_stride=max(int(args.sample_stride), 1),
        )
        metadata_path = result_dir / "lightweight_analysis_metadata.json"
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        metadata["config"] = str(args.config)
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)
    finally:
        temp_config.unlink(missing_ok=True)
    LOGGER.info("output_dir=%s", args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
