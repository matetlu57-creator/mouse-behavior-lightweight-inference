"""Load Ultralytics tracklets into behavior-analysis trajectory arrays.

Tracking is intentionally external to this module.  Cache records must already
contain ``visual_track_id`` values emitted by ``YOLO.track``.  When an mTrack
identity map is supplied, unresolved visual detections stay unresolved; this
module never guesses an animal identity from position, pose, color, or motion.
"""

from __future__ import annotations

import gzip
import json
import logging
import pickle
from collections import Counter
from pathlib import Path
from typing import Any, Iterator, Mapping

import cv2
import numpy as np

from ..identity.mtrack import IdentityMap
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
from ..preprocessing.geometry import _finite_point, _weighted_mean


LOGGER = logging.getLogger(__name__)


class TrajectoryCacheError(ValueError):
    """Raised when a cache cannot satisfy the tracked-trajectory contract."""


def _visual_track_id(payload: Mapping[str, Any]) -> int | None:
    raw = payload.get("visual_track_id")
    if raw is None:
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if value >= 0 else None


def _payload_detection(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    """Normalize one tracked YOLO Pose payload without assigning an identity."""

    points = np.asarray(payload.get("keypoints_px", []), dtype=float)
    confidence = np.asarray(payload.get("keypoint_conf", []), dtype=float).reshape(-1)
    bbox = np.asarray(payload.get("bbox_xyxy", []), dtype=float).reshape(-1)
    if points.shape != (KEYPOINTS, 2) or confidence.shape != (KEYPOINTS,) or bbox.shape != (4,):
        return None
    if not (
        np.all(np.isfinite(bbox))
        and float(bbox[2]) > float(bbox[0])
        and float(bbox[3]) > float(bbox[1])
    ):
        return None

    point_valid = (
        np.all(np.isfinite(points), axis=1)
        & np.isfinite(confidence)
        & (confidence >= 0.10)
    )
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
        body_length = float("nan")
    if not np.isfinite(body_length) or body_length < 8.0:
        body_length = float(
            max(min(float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1])), 8.0)
        )

    observed_quality = (
        float(np.mean(confidence[point_valid])) if bool(np.any(point_valid)) else 0.0
    )
    try:
        payload_quality = float(payload.get("pose_quality", observed_quality))
    except (TypeError, ValueError, OverflowError):
        payload_quality = observed_quality
    if not np.isfinite(payload_quality):
        payload_quality = observed_quality
    pose_quality = min(payload_quality, observed_quality) if observed_quality else 0.0
    if int(point_valid.sum()) < 4:
        pose_quality *= float(point_valid.sum()) / 4.0
    try:
        box_conf = float(payload.get("box_conf", 0.0))
    except (TypeError, ValueError, OverflowError):
        box_conf = 0.0
    if not np.isfinite(box_conf):
        box_conf = 0.0

    return {
        "visual_track_id": _visual_track_id(payload),
        "points": clean_points,
        "confidence": confidence,
        "bbox": bbox,
        "center": center,
        "head": head,
        "body_length": body_length,
        "pose_quality": pose_quality if np.isfinite(pose_quality) else 0.0,
        "box_conf": box_conf,
        "score": box_conf + 0.10 * (pose_quality if np.isfinite(pose_quality) else 0.0),
    }


def _iter_cache_records(cache_dir: Path) -> Iterator[Mapping[str, Any]]:
    parts = sorted(Path(cache_dir).glob("yolo_results.*.*.pkl.gz"))
    if not parts:
        raise FileNotFoundError(f"没有找到 YOLO 轨迹缓存分块: {cache_dir}")
    for part in parts:
        with gzip.open(part, "rb") as handle:
            records = pickle.load(handle)
        if not isinstance(records, list):
            raise TrajectoryCacheError(f"缓存分块格式错误: {part}")
        for record in records:
            if isinstance(record, Mapping):
                yield record


def _cache_total_frames(cache_dir: Path) -> int:
    status_path = Path(cache_dir) / "yolo_results_status.json"
    with status_path.open("r", encoding="utf-8") as handle:
        status = json.load(handle)
    if str(status.get("status")) != "complete":
        raise RuntimeError(f"YOLO 轨迹缓存尚未完成: {status_path}")
    total = int(status.get("total_frames", 0))
    if total <= 0 or int(status.get("next_frame", -1)) != total:
        raise RuntimeError(f"YOLO 轨迹缓存帧数不完整: {status_path}")
    schema = int(status.get("trajectory_schema_version", 0))
    if schema < 1:
        raise TrajectoryCacheError(
            "缓存没有 visual_track_id 轨迹契约；请用新的 build_lightweight_pose_cache.py 重建"
        )
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


def _record_detections(
    record: Mapping[str, Any],
    arena_polygon: np.ndarray | None,
    arena_tolerance_px: float,
) -> tuple[list[dict[str, Any]], int]:
    payloads = record.get("pose_detections", [])
    if not isinstance(payloads, list):
        return [], 0
    detections: list[dict[str, Any]] = []
    missing_track_ids = 0
    for payload in payloads:
        if not isinstance(payload, Mapping):
            continue
        detection = _payload_detection(payload)
        if detection is None:
            continue
        if not _inside_arena(detection["center"], arena_polygon, arena_tolerance_px):
            continue
        if detection["visual_track_id"] is None:
            missing_track_ids += 1
            continue
        detections.append(detection)
    return detections, missing_track_ids


def _visual_slots(
    observations: Counter[int],
    first_seen: Mapping[int, int],
    expected_mice: int,
) -> tuple[dict[int, int], tuple[str, ...], tuple[int, ...]]:
    ranked = sorted(
        observations,
        key=lambda visual_id: (-observations[visual_id], first_seen[visual_id], visual_id),
    )[:expected_mice]
    selected = sorted(ranked, key=lambda visual_id: (first_seen[visual_id], visual_id))
    slot_by_visual = {visual_id: slot for slot, visual_id in enumerate(selected)}
    labels = [f"visual:{visual_id}" for visual_id in selected]
    visual_ids = list(selected)
    while len(labels) < expected_mice:
        labels.append(f"unassigned:{len(labels)}")
        visual_ids.append(-1)
    return slot_by_visual, tuple(labels), tuple(visual_ids)


def _empty_arrays(total_frames: int, expected_mice: int) -> dict[str, np.ndarray]:
    return {
        "centers_px": np.full((total_frames, expected_mice, 2), np.nan, dtype=np.float32),
        "keypoints_px": np.full(
            (total_frames, expected_mice, KEYPOINTS, 2), np.nan, dtype=np.float32
        ),
        "confidences": np.zeros((total_frames, expected_mice, KEYPOINTS), dtype=np.float32),
        "bboxes": np.full((total_frames, expected_mice, 4), np.nan, dtype=np.float32),
        "bbox_observed": np.zeros((total_frames, expected_mice), dtype=bool),
        "bbox_imputed": np.zeros((total_frames, expected_mice), dtype=bool),
        "pose_quality": np.zeros((total_frames, expected_mice), dtype=np.float32),
        "body_lengths_px": np.full((total_frames, expected_mice), np.nan, dtype=np.float32),
        "identity_confidence": np.zeros((total_frames, expected_mice), dtype=np.float32),
        "visual_track_ids": np.full((total_frames, expected_mice), -1, dtype=np.int64),
    }


def load_trajectory_cache(
    cache_dir: Path,
    total_frames: int,
    expected_mice: int,
    *,
    identity_map: IdentityMap | None = None,
    arena_polygon: np.ndarray | None = None,
    arena_tolerance_px: float = 2.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load tracked pose records and optionally resolve them to RFID identities."""

    total_frames = max(int(total_frames), 0)
    expected_mice = max(int(expected_mice), 1)
    observations: Counter[int] = Counter()
    first_seen: dict[int, int] = {}
    missing_track_id_count = 0
    raw_detection_count = 0
    cache_frames_seen: set[int] = set()

    for record in _iter_cache_records(Path(cache_dir)):
        frame = int(record.get("frame", -1))
        if frame < 0 or frame >= total_frames:
            continue
        cache_frames_seen.add(frame)
        detections, missing = _record_detections(record, arena_polygon, arena_tolerance_px)
        missing_track_id_count += missing
        raw_detection_count += len(detections) + missing
        for detection in detections:
            visual_id = int(detection["visual_track_id"])
            observations[visual_id] += 1
            first_seen.setdefault(visual_id, frame)

    if missing_track_id_count:
        raise TrajectoryCacheError(
            f"缓存中有 {missing_track_id_count} 个检测缺少 visual_track_id；"
            "行为分析不再自行分配 ID，请用 Ultralytics model.track 重建缓存"
        )
    if not observations:
        raise TrajectoryCacheError("轨迹缓存中没有可用的 Ultralytics visual_track_id")

    if identity_map is None:
        slot_by_visual, identity_labels, visual_track_ids_by_slot = _visual_slots(
            observations, first_seen, expected_mice
        )
        slot_by_animal: dict[str, int] = {}
        identity_source = "visual_only"
    else:
        if len(identity_map.animal_ids) > expected_mice:
            raise TrajectoryCacheError(
                f"mTrack identity map contains {len(identity_map.animal_ids)} animals, "
                f"but expected_mice={expected_mice}"
            )
        labels = list(identity_map.animal_ids)
        while len(labels) < expected_mice:
            labels.append(f"unassigned:{len(labels)}")
        identity_labels = tuple(labels)
        slot_by_animal = {
            animal_id: slot for slot, animal_id in enumerate(identity_map.animal_ids)
        }
        slot_by_visual = {}
        visual_track_ids_by_slot = tuple(() for _ in range(expected_mice))
        identity_source = "rfid_mtrack"

    arrays = _empty_arrays(total_frames, expected_mice)
    selected_scores = np.full((total_frames, expected_mice), -np.inf, dtype=float)
    unresolved_detection_count = 0
    capacity_dropped_count = 0
    same_identity_conflict_count = 0
    resolved_count = 0
    used_visual_ids_by_slot: list[set[int]] = [set() for _ in range(expected_mice)]

    for record in _iter_cache_records(Path(cache_dir)):
        frame = int(record.get("frame", -1))
        if frame < 0 or frame >= total_frames:
            continue
        detections, _missing = _record_detections(record, arena_polygon, arena_tolerance_px)
        for detection in detections:
            visual_id = int(detection["visual_track_id"])
            if identity_map is None:
                slot = slot_by_visual.get(visual_id)
                if slot is None:
                    capacity_dropped_count += 1
                    continue
                identity_confidence = 1.0
            else:
                assignment = identity_map.resolve(frame, visual_id)
                if assignment is None:
                    unresolved_detection_count += 1
                    continue
                slot = slot_by_animal[assignment.animal_id]
                identity_confidence = float(assignment.confidence)

            score = float(detection["score"]) * max(identity_confidence, 1e-6)
            if np.isfinite(selected_scores[frame, slot]):
                same_identity_conflict_count += 1
                if score <= selected_scores[frame, slot]:
                    continue
            selected_scores[frame, slot] = score
            arrays["centers_px"][frame, slot] = detection["center"]
            arrays["keypoints_px"][frame, slot] = detection["points"]
            arrays["confidences"][frame, slot] = detection["confidence"]
            arrays["bboxes"][frame, slot] = detection["bbox"]
            arrays["bbox_observed"][frame, slot] = True
            arrays["pose_quality"][frame, slot] = detection["pose_quality"]
            arrays["body_lengths_px"][frame, slot] = detection["body_length"]
            arrays["identity_confidence"][frame, slot] = identity_confidence
            arrays["visual_track_ids"][frame, slot] = visual_id
            used_visual_ids_by_slot[slot].add(visual_id)
            resolved_count += 1

    arrays["valid"] = np.all(np.isfinite(arrays["centers_px"]), axis=2)
    tracks: dict[str, Any] = {
        **arrays,
        "identity_labels": identity_labels,
        "identity_source": identity_source,
        "identity_map": str(identity_map.source_path) if identity_map and identity_map.source_path else None,
        "identity_map_sha256": identity_map.source_sha256 if identity_map else None,
        "visual_track_ids_by_slot": (
            visual_track_ids_by_slot
            if identity_map is None
            else tuple(tuple(sorted(values)) for values in used_visual_ids_by_slot)
        ),
    }
    valid = arrays["valid"]
    stats = {
        "trajectory_schema_version": 1,
        "identity_source": identity_source,
        "identity_map": str(identity_map.source_path) if identity_map and identity_map.source_path else None,
        "identity_labels": list(identity_labels),
        "cache_frames_seen": len(cache_frames_seen),
        "raw_detection_count": int(raw_detection_count),
        "resolved_detection_count": int(resolved_count),
        "unresolved_detection_count": int(unresolved_detection_count),
        "capacity_dropped_detection_count": int(capacity_dropped_count),
        "same_identity_conflict_count": int(same_identity_conflict_count),
        "visual_track_count": len(observations),
        "track_valid_rate": float(np.mean(valid)) if valid.size else 0.0,
        "track_valid_frames_mean": float(np.mean(valid.sum(axis=1))) if len(valid) else 0.0,
        "bbox_observed_count": int(arrays["bbox_observed"].sum()),
        "bbox_imputed_count": 0,
        "identity_semantics": (
            "RFID-resolved animal identity"
            if identity_map is not None
            else "provisional Ultralytics visual track ID"
        ),
    }
    return tracks, stats


def _track_cache(
    cache_dir: Path,
    total_frames: int,
    expected_mice: int,
    *,
    identity_map: IdentityMap | None = None,
    arena_polygon: np.ndarray | None = None,
    arena_tolerance_px: float = 2.0,
    **deprecated_tracking_options: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compatibility alias for callers migrating to ``load_trajectory_cache``.

    Legacy assignment and occlusion-recovery options are rejected so a stale
    caller cannot silently re-enable behavior-layer tracking.
    """

    active = {
        name: value
        for name, value in deprecated_tracking_options.items()
        if value not in (None, 0, 0.0, False, "")
    }
    if active:
        raise TypeError(
            "custom behavior-layer tracking options were removed; "
            f"unsupported options: {', '.join(sorted(active))}"
        )
    return load_trajectory_cache(
        cache_dir,
        total_frames,
        expected_mice,
        identity_map=identity_map,
        arena_polygon=arena_polygon,
        arena_tolerance_px=arena_tolerance_px,
    )


__all__ = [
    "TrajectoryCacheError",
    "load_trajectory_cache",
]
