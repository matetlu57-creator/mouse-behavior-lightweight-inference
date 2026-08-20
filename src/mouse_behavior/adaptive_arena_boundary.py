"""Learn a per-video arena polygon from mouse motion in YOLO cache records.

The full behavior pipeline calls this module before formal identity assignment.
The input is intentionally the small, model-independent record written by
``YOLOPrecomputeCache`` rather than a live Ultralytics result.  A learned
boundary is therefore tied to the video that produced the cache and can be
saved/reused safely only for that same video.

The estimator is deliberately conservative:

* only sufficiently confident pose detections contribute;
* frame-to-frame centers are matched with a bounded nearest-neighbour motion
  track so static false detections do not become a wall;
* a smoothed occupancy heatmap is thresholded and morphologically connected;
* the largest connected component is converted to an axis-aligned rectangle
  and resized around the rectangle center by the configured margin;
* weak/insufficient evidence falls back to a configured polygon or the full
  frame instead of producing an overconfident tiny cage.

This module has no dependency on the identity tracker.  It can be imported by
the full pipeline before identities exist, and it is also easy to test with
synthetic cache records.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np


SCHEMA_VERSION = 1
DEFAULT_KEYPOINT_CONFIDENCE = 0.08
CORE_KEYPOINTS = (3, 4, 5, 6)  # neck, left hip, right hip, base of tail


@dataclass
class ArenaBoundaryResult:
    """Serializable result consumed by ``mouse_chase_attack_high_recall``."""

    polygon: list[list[float]]
    source: str
    motion_sample_count: int
    sample_count: int
    occupied_area_ratio: float
    expansion_ratio: float
    width: int
    height: int
    heatmap_cell_px: int = 20
    source_video: str = ""
    source_video_fingerprint: dict[str, Any] = field(default_factory=dict)
    accepted: bool = True
    rejection_reason: str = ""


@dataclass
class _DetectionPoint:
    center: np.ndarray
    body_length: float


@dataclass
class _MotionTrack:
    last_center: np.ndarray
    velocity: np.ndarray
    last_frame: int
    hits: int = 1
    body_length: float = 1.0


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if np.isfinite(result) else float(default)


def _finite_point(value: Any) -> Optional[np.ndarray]:
    point = np.asarray(value, dtype=np.float64).reshape(-1)
    if point.size < 2 or not np.all(np.isfinite(point[:2])):
        return None
    return point[:2].copy()


def _video_fingerprint(video_path: Optional[str | os.PathLike[str]]) -> dict[str, Any]:
    if not video_path:
        return {}
    path = Path(video_path).expanduser()
    try:
        resolved = path.resolve()
        stat = resolved.stat()
    except OSError:
        return {"path": os.path.normcase(str(path))}
    return {
        "path": os.path.normcase(str(resolved)),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _same_path(left: Any, right: Any) -> bool:
    try:
        return os.path.normcase(str(Path(str(left)).expanduser().resolve())) == os.path.normcase(
            str(Path(str(right)).expanduser().resolve())
        )
    except (OSError, ValueError):
        return os.path.normcase(str(left)) == os.path.normcase(str(right))


def _coerce_polygon(points: Any, width: int, height: int) -> Optional[np.ndarray]:
    try:
        polygon = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    except (TypeError, ValueError):
        return None
    if len(polygon) < 3 or not np.all(np.isfinite(polygon)):
        return None
    polygon[:, 0] = np.clip(polygon[:, 0], 0.0, max(float(width - 1), 0.0))
    polygon[:, 1] = np.clip(polygon[:, 1], 0.0, max(float(height - 1), 0.0))
    rectangle = _axis_aligned_rectangle(polygon, width, height)
    if rectangle is None or cv2.contourArea(rectangle.astype(np.float32)) <= 1.0:
        return None
    return rectangle


def _axis_aligned_rectangle(
    points: Any,
    width: int,
    height: int,
) -> Optional[np.ndarray]:
    """Return the four-corner image-axis-aligned rectangle around points."""

    try:
        values = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    except (TypeError, ValueError):
        return None
    if len(values) < 3 or not np.all(np.isfinite(values)):
        return None
    min_x = float(np.clip(np.min(values[:, 0]), 0.0, max(float(width - 1), 0.0)))
    max_x = float(np.clip(np.max(values[:, 0]), 0.0, max(float(width - 1), 0.0)))
    min_y = float(np.clip(np.min(values[:, 1]), 0.0, max(float(height - 1), 0.0)))
    max_y = float(np.clip(np.max(values[:, 1]), 0.0, max(float(height - 1), 0.0)))
    if max_x - min_x <= 1.0 or max_y - min_y <= 1.0:
        return None
    return np.array(
        [
            [min_x, min_y],
            [max_x, min_y],
            [max_x, max_y],
            [min_x, max_y],
        ],
        dtype=np.float64,
    )


def _polygon_lists(polygon: np.ndarray) -> list[list[float]]:
    return [[round(float(x), 3), round(float(y), 3)] for x, y in polygon]


def _polygon_area(polygon: np.ndarray) -> float:
    if polygon is None or len(polygon) < 3:
        return 0.0
    return abs(float(cv2.contourArea(np.asarray(polygon, dtype=np.float32))))


def _fallback_polygon(
    configured_polygon: Any,
    width: int,
    height: int,
) -> tuple[np.ndarray, str]:
    configured = _coerce_polygon(configured_polygon, width, height)
    if configured is not None:
        return configured, "configured_polygon"
    frame = np.array(
        [
            [0.0, 0.0],
            [float(max(width - 1, 0)), 0.0],
            [float(max(width - 1, 0)), float(max(height - 1, 0))],
            [0.0, float(max(height - 1, 0))],
        ],
        dtype=np.float64,
    )
    return frame, "frame_fallback"


def _detection_center(payload: Mapping[str, Any]) -> Optional[_DetectionPoint]:
    """Use the same body-core center policy as the identity tracker."""

    points = np.asarray(payload.get("keypoints_px", []), dtype=np.float64)
    confidence = np.asarray(payload.get("keypoint_conf", []), dtype=np.float64).reshape(-1)
    bbox = np.asarray(payload.get("bbox_xyxy", []), dtype=np.float64).reshape(-1)
    if points.ndim != 2 or points.shape[1] < 2:
        points = np.empty((0, 2), dtype=np.float64)
    else:
        points = points[:, :2]
    valid = np.all(np.isfinite(points), axis=1) if len(points) else np.zeros(0, dtype=bool)
    if len(confidence) < len(points):
        confidence = np.pad(confidence, (0, len(points) - len(confidence)), constant_values=0.0)
    confidence = confidence[: len(points)]
    valid &= confidence >= DEFAULT_KEYPOINT_CONFIDENCE

    center: Optional[np.ndarray] = None
    core = np.zeros(len(points), dtype=bool)
    for index in CORE_KEYPOINTS:
        if index < len(core):
            core[index] = True
    use = valid & core
    if int(use.sum()) >= 2:
        weights = np.clip(confidence[use], 0.05, 1.0)
        center = np.average(points[use], axis=0, weights=weights).astype(np.float64)
    elif int(valid.sum()) >= 2:
        weights = np.clip(confidence[valid], 0.05, 1.0)
        center = np.average(points[valid], axis=0, weights=weights).astype(np.float64)
    elif bbox.size >= 4 and np.all(np.isfinite(bbox[:4])):
        center = np.array(
            [(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0],
            dtype=np.float64,
        )
    if center is None or not np.all(np.isfinite(center)):
        return None

    body_length = float("nan")
    if len(points) > 6 and len(confidence) > 6 and valid[0] and valid[6]:
        body_length = float(np.linalg.norm(points[0] - points[6]))
    if not np.isfinite(body_length) or body_length <= 3.0:
        if bbox.size >= 4 and np.all(np.isfinite(bbox[:4])):
            body_length = max(abs(float(bbox[2] - bbox[0])), abs(float(bbox[3] - bbox[1]))) * 0.80
    if not np.isfinite(body_length) or body_length <= 1.0:
        body_length = 1.0
    return _DetectionPoint(center=center, body_length=float(body_length))


def _frame_points(entry: Mapping[str, Any], cfg: Mapping[str, Any]) -> list[_DetectionPoint]:
    minimum_box = max(_finite_float(cfg.get("min_box_confidence", 0.12), 0.12), 0.0)
    minimum_pose = max(_finite_float(cfg.get("min_pose_quality", 0.08), 0.08), 0.0)
    output: list[_DetectionPoint] = []
    raw_detections = entry.get("pose_detections", [])
    if not isinstance(raw_detections, Sequence) or isinstance(raw_detections, (str, bytes)):
        return output
    for payload in raw_detections:
        if not isinstance(payload, Mapping):
            continue
        if _finite_float(payload.get("box_conf", 0.0)) < minimum_box:
            continue
        if _finite_float(payload.get("pose_quality", 0.0)) < minimum_pose:
            continue
        point = _detection_center(payload)
        if point is not None:
            output.append(point)
    return output


def _match_motion_tracks(
    tracks: list[_MotionTrack],
    detections: list[_DetectionPoint],
    frame: int,
    max_gap_frames: int,
    max_motion_px_per_frame: float,
) -> list[tuple[_MotionTrack, _DetectionPoint, float]]:
    active = [
        track
        for track in tracks
        if 0 <= int(frame) - int(track.last_frame) <= max(int(max_gap_frames), 0)
    ]
    candidates: list[tuple[float, int, int, float]] = []
    for track_index, track in enumerate(active):
        gap = max(int(frame) - int(track.last_frame), 1)
        predicted = track.last_center + track.velocity * float(gap)
        gate = max(
            float(max_motion_px_per_frame) * float(gap) + 0.50 * max(track.body_length, 1.0),
            12.0,
        )
        for detection_index, detection in enumerate(detections):
            distance = float(np.linalg.norm(detection.center - predicted))
            if np.isfinite(distance) and distance <= gate:
                candidates.append((distance, track_index, detection_index, float(gap)))
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    used_tracks: set[int] = set()
    used_detections: set[int] = set()
    matches: list[tuple[_MotionTrack, _DetectionPoint, float]] = []
    for distance, track_index, detection_index, gap in candidates:
        if track_index in used_tracks or detection_index in used_detections:
            continue
        used_tracks.add(track_index)
        used_detections.add(detection_index)
        matches.append((active[track_index], detections[detection_index], gap))
    return matches


def _expanded_rectangle(
    rectangle: np.ndarray,
    expansion_ratio: float,
    width: int,
    height: int,
) -> np.ndarray:
    polygon = np.asarray(rectangle, dtype=np.float64).reshape(-1, 2)
    min_x = float(np.min(polygon[:, 0]))
    max_x = float(np.max(polygon[:, 0]))
    min_y = float(np.min(polygon[:, 1]))
    max_y = float(np.max(polygon[:, 1]))
    center_x = 0.5 * (min_x + max_x)
    center_y = 0.5 * (min_y + max_y)
    half_width = 0.5 * (max_x - min_x) * float(expansion_ratio)
    half_height = 0.5 * (max_y - min_y) * float(expansion_ratio)
    left = np.clip(center_x - half_width, 0.0, max(float(width - 1), 0.0))
    right = np.clip(center_x + half_width, 0.0, max(float(width - 1), 0.0))
    top = np.clip(center_y - half_height, 0.0, max(float(height - 1), 0.0))
    bottom = np.clip(center_y + half_height, 0.0, max(float(height - 1), 0.0))
    return np.array(
        [[left, top], [right, top], [right, bottom], [left, bottom]],
        dtype=np.float64,
    )


def _boundary_scale(cfg: Mapping[str, Any]) -> float:
    """Read the rectangle scale; 0.97 means shrinking the box by 3%."""

    value = _finite_float(cfg.get("boundary_expansion_ratio", 0.97), 0.97)
    return float(np.clip(value, 0.10, 3.00))


def _empty_heatmap(width: int, height: int, cell_px: int) -> np.ndarray:
    return np.zeros(
        (
            max(int(math.ceil(height / max(cell_px, 1))), 1),
            max(int(math.ceil(width / max(cell_px, 1))), 1),
        ),
        dtype=np.float32,
    )


def _learn_polygon_from_heatmap(
    heatmap: np.ndarray,
    width: int,
    height: int,
    cfg: Mapping[str, Any],
) -> tuple[Optional[np.ndarray], float, str]:
    if heatmap.size == 0 or not np.any(heatmap > 0.0):
        return None, 0.0, "no_positive_heatmap"
    sigma = max(_finite_float(cfg.get("heatmap_blur_sigma_cells", 1.25), 1.25), 0.0)
    if sigma > 0.0:
        blurred = cv2.GaussianBlur(heatmap.astype(np.float32), (0, 0), sigmaX=sigma, sigmaY=sigma)
    else:
        blurred = heatmap.astype(np.float32, copy=True)
    peak = float(np.max(blurred))
    if not np.isfinite(peak) or peak <= 0.0:
        return None, 0.0, "empty_blurred_heatmap"
    normalized = blurred / peak
    positive = normalized[normalized > 0.0]
    quantile = float(
        np.clip(_finite_float(cfg.get("heatmap_positive_quantile", 0.20), 0.20), 0.0, 1.0)
    )
    minimum_density = float(
        np.clip(_finite_float(cfg.get("heatmap_min_density", 0.08), 0.08), 0.0, 1.0)
    )
    threshold = max(float(np.quantile(positive, quantile)), minimum_density)
    binary = (normalized >= threshold).astype(np.uint8)

    close_radius = max(int(cfg.get("heatmap_close_radius_cells", 3)), 0)
    if close_radius > 0:
        size = 2 * close_radius + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    dilate_radius = max(int(cfg.get("heatmap_dilate_radius_cells", 2)), 0)
    if dilate_radius > 0:
        size = 2 * dilate_radius + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        binary = cv2.dilate(binary, kernel)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return None, 0.0, "no_connected_motion_region"
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    component = np.where(labels == largest, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, 0.0, "no_motion_contour"
    contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(contour).reshape(-1, 2).astype(np.float64)
    if len(hull) < 3:
        return None, 0.0, "degenerate_motion_hull"
    # Heatmap coordinates are cell coordinates; convert to pixel coordinates
    # before area checks and rectangle fitting.
    cell_px = max(int(cfg.get("heatmap_cell_px", 20)), 1)
    hull = (hull + 0.5) * float(cell_px)
    hull[:, 0] = np.clip(hull[:, 0], 0.0, max(float(width - 1), 0.0))
    hull[:, 1] = np.clip(hull[:, 1], 0.0, max(float(height - 1), 0.0))
    rectangle = _axis_aligned_rectangle(hull, width, height)
    if rectangle is None:
        return None, 0.0, "degenerate_motion_rectangle"
    area_ratio = _polygon_area(rectangle) / max(float(width * height), 1.0)
    minimum_area = max(_finite_float(cfg.get("min_boundary_area_ratio", 0.20), 0.20), 0.0)
    maximum_area = min(_finite_float(cfg.get("max_boundary_area_ratio", 0.95), 0.95), 1.0)
    if area_ratio < minimum_area or area_ratio > maximum_area:
        return None, float(area_ratio), f"area_ratio_out_of_range:{area_ratio:.6f}"
    expansion = _boundary_scale(cfg)
    return _expanded_rectangle(rectangle, expansion, width, height), float(area_ratio), "accepted"


def learn_from_yolo_records(
    records: Iterable[Mapping[str, Any]],
    *,
    width: int,
    height: int,
    config: Optional[Mapping[str, Any]] = None,
    configured_polygon: Any = None,
    source_video: Optional[str | os.PathLike[str]] = None,
) -> tuple[ArenaBoundaryResult, np.ndarray]:
    """Learn one arena boundary from one video's YOLO cache records.

    ``records`` is consumed once.  The caller should pass the cache for the
    current video only; no global or cross-video state is retained here.
    """

    cfg = dict(config or {})
    width = max(int(width), 1)
    height = max(int(height), 1)
    cell_px = max(int(cfg.get("heatmap_cell_px", 20)), 1)
    heatmap = _empty_heatmap(width, height, cell_px)
    sample_count = 0
    motion_sample_count = 0
    tracks: list[_MotionTrack] = []
    minimum_motion = max(_finite_float(cfg.get("min_motion_px_per_frame", 1.5), 1.5), 0.0)
    maximum_motion = max(
        _finite_float(cfg.get("max_motion_px_per_frame", 120.0), 120.0),
        minimum_motion,
    )
    minimum_track_frames = max(int(cfg.get("long_track_min_frames", 30)), 1)
    max_gap_frames = max(int(cfg.get("max_track_gap_frames", 2)), 0)

    for entry in records:
        if not isinstance(entry, Mapping):
            continue
        frame = int(_finite_float(entry.get("frame", -1), -1.0))
        if frame < 0:
            continue
        detections = _frame_points(entry, cfg)
        sample_count += len(detections)
        matches = _match_motion_tracks(tracks, detections, frame, max_gap_frames, maximum_motion)
        matched_detection_ids: set[int] = set()
        for track, detection, gap_value in matches:
            gap = max(float(gap_value), 1.0)
            delta = detection.center - track.last_center
            speed = float(np.linalg.norm(delta)) / gap
            track.velocity = 0.65 * track.velocity + 0.35 * (delta / gap)
            track.last_center = detection.center.copy()
            track.last_frame = int(frame)
            track.hits += 1
            track.body_length = 0.70 * track.body_length + 0.30 * detection.body_length
            matched_detection_ids.add(id(detection))
            if track.hits >= minimum_track_frames and minimum_motion <= speed <= maximum_motion:
                for point in (track.last_center, track.last_center - delta):
                    x = int(np.clip(math.floor(float(point[0]) / cell_px), 0, heatmap.shape[1] - 1))
                    y = int(np.clip(math.floor(float(point[1]) / cell_px), 0, heatmap.shape[0] - 1))
                    heatmap[y, x] += 1.0
                motion_sample_count += 1

        for detection in detections:
            if id(detection) not in matched_detection_ids:
                tracks.append(
                    _MotionTrack(
                        last_center=detection.center.copy(),
                        velocity=np.zeros(2, dtype=np.float64),
                        last_frame=int(frame),
                        hits=1,
                        body_length=float(detection.body_length),
                    )
                )
        tracks = [track for track in tracks if int(frame) - int(track.last_frame) <= max_gap_frames]

    minimum_motion_samples = max(int(cfg.get("min_motion_samples", 300)), 1)
    if motion_sample_count < minimum_motion_samples:
        polygon, area_ratio, reason = (
            None,
            0.0,
            (f"insufficient_motion_samples:{motion_sample_count}<{minimum_motion_samples}"),
        )
    else:
        polygon, area_ratio, reason = _learn_polygon_from_heatmap(heatmap, width, height, cfg)
    expansion = _boundary_scale(cfg)
    accepted = polygon is not None and reason == "accepted"
    if not accepted:
        polygon, fallback_source = _fallback_polygon(configured_polygon, width, height)
        source = fallback_source
        expansion = 1.0
        if fallback_source == "configured_polygon":
            area_ratio = _polygon_area(polygon) / max(float(width * height), 1.0)
        else:
            area_ratio = 1.0
    else:
        source = "learned_motion_heatmap"

    result = ArenaBoundaryResult(
        polygon=_polygon_lists(polygon),
        source=source,
        motion_sample_count=int(motion_sample_count),
        sample_count=int(sample_count),
        occupied_area_ratio=float(area_ratio),
        expansion_ratio=float(expansion),
        width=int(width),
        height=int(height),
        heatmap_cell_px=int(cell_px),
        source_video=str(Path(source_video).resolve()) if source_video else "",
        source_video_fingerprint=_video_fingerprint(str(source_video)) if source_video else {},
        accepted=bool(accepted),
        rejection_reason="" if accepted else str(reason),
    )
    return result, heatmap


def load_boundary_json(
    path: Path,
    *,
    width: int,
    height: int,
    source_video: Optional[str | os.PathLike[str]] = None,
    require_video_match: bool = True,
) -> ArenaBoundaryResult:
    """Load a boundary and reject cross-video reuse when metadata is present."""

    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"笼界 JSON 顶层必须是对象：{path}")
    raw = payload.get("result", payload)
    if not isinstance(raw, Mapping):
        raise ValueError(f"笼界 JSON 缺少 result 对象：{path}")
    stored_width = int(raw.get("width", width))
    stored_height = int(raw.get("height", height))
    if stored_width != int(width) or stored_height != int(height):
        raise ValueError(
            f"笼界 JSON 分辨率不匹配：文件={stored_width}x{stored_height}，当前={width}x{height}"
        )
    current_video = str(Path(source_video).resolve()) if source_video else ""
    stored_video = str(raw.get("source_video", "") or "")
    if require_video_match and current_video:
        if not stored_video:
            raise ValueError("笼界 JSON 没有 source_video，拒绝将未绑定边界用于当前视频")
        if not _same_path(stored_video, current_video):
            raise ValueError(f"笼界 JSON 属于其他视频：文件={stored_video}，当前={current_video}")
        stored_fingerprint = raw.get("source_video_fingerprint", {})
        current_fingerprint = _video_fingerprint(current_video)
        if isinstance(stored_fingerprint, Mapping) and stored_fingerprint and current_fingerprint:
            for key in ("size", "mtime_ns"):
                if key in stored_fingerprint and key in current_fingerprint:
                    if int(stored_fingerprint[key]) != int(current_fingerprint[key]):
                        raise ValueError(f"笼界 JSON 的视频文件指纹不匹配：{key}")

    polygon = _coerce_polygon(raw.get("polygon", []), int(width), int(height))
    if polygon is None:
        raise ValueError(f"笼界 JSON polygon 无效：{path}")
    return ArenaBoundaryResult(
        polygon=_polygon_lists(polygon),
        source=str(raw.get("source", "reused_json")),
        motion_sample_count=int(raw.get("motion_sample_count", 0)),
        sample_count=int(raw.get("sample_count", 0)),
        occupied_area_ratio=_finite_float(raw.get("occupied_area_ratio", 0.0)),
        expansion_ratio=_finite_float(raw.get("expansion_ratio", 1.0), 1.0),
        width=int(width),
        height=int(height),
        heatmap_cell_px=max(int(raw.get("heatmap_cell_px", 20)), 1),
        source_video=stored_video or current_video,
        source_video_fingerprint=dict(raw.get("source_video_fingerprint", {}) or {}),
        accepted=bool(raw.get("accepted", True)),
        rejection_reason=str(raw.get("rejection_reason", "") or ""),
    )


def _atomic_json_write(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_png(image: np.ndarray, path: Path) -> None:
    """Write PNG bytes through pathlib so Unicode Windows paths are safe."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded_ok, encoded = cv2.imencode(".png", np.asarray(image))
    if not encoded_ok:
        raise OSError(f"无法编码 PNG：{path}")
    temporary = path.with_suffix(path.suffix + ".tmp.png")
    temporary.write_bytes(encoded.tobytes())
    os.replace(temporary, path)


def save_boundary_overlay_frame(
    video_path: str | os.PathLike[str],
    result: ArenaBoundaryResult,
    output_path: Path,
    frame_index: int = 0,
) -> None:
    """Draw this video's learned polygon on an actual source-video frame."""

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开原视频以生成笼界对比图：{video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target = max(int(frame_index), 0)
    if total_frames > 0:
        target = min(target, total_frames - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(target))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"无法读取原视频第 {target} 帧：{video_path}")

    polygon = np.asarray(result.polygon, dtype=np.int32).reshape(-1, 1, 2)
    if len(polygon) >= 3:
        overlay = frame.copy()
        cv2.fillPoly(overlay, [polygon], (0, 190, 0))
        frame = cv2.addWeighted(overlay, 0.16, frame, 0.84, 0.0)
        cv2.polylines(frame, [polygon], True, (0, 255, 0), 5, cv2.LINE_AA)
    adjustment = 100.0 * (float(result.expansion_ratio) - 1.0)
    label = (
        f"arena={result.source} | source_frame={target} | "
        f"area={result.occupied_area_ratio:.3f} | boundary={adjustment:+.1f}%"
    )
    cv2.rectangle(frame, (10, 8), (min(frame.shape[1] - 1, 760), 48), (0, 0, 0), -1)
    cv2.putText(
        frame,
        label[:110],
        (20, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    _write_png(frame, Path(output_path))


def save_boundary_artifacts(
    result: ArenaBoundaryResult,
    heatmap: np.ndarray,
    json_path: Path,
    png_path: Path,
    comparison_path: Optional[Path] = None,
) -> None:
    """Write the per-video boundary JSON and an auditable heatmap overlay."""

    json_path = Path(json_path)
    png_path = Path(png_path)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "result": asdict(result),
    }
    _atomic_json_write(payload, json_path)

    heatmap = np.asarray(heatmap, dtype=np.float32)
    if heatmap.size == 0:
        heatmap = np.zeros((2, 2), dtype=np.float32)
    maximum = float(np.max(heatmap)) if np.any(np.isfinite(heatmap)) else 0.0
    normalized = np.zeros_like(heatmap, dtype=np.uint8)
    if maximum > 0.0 and np.isfinite(maximum):
        normalized = np.clip(255.0 * np.nan_to_num(heatmap, nan=0.0) / maximum, 0.0, 255.0).astype(
            np.uint8
        )
    color = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    canvas = cv2.resize(
        color, (int(result.width), int(result.height)), interpolation=cv2.INTER_NEAREST
    )
    canvas = cv2.addWeighted(canvas, 0.72, np.full_like(canvas, 35), 0.28, 0.0)
    polygon = np.asarray(result.polygon, dtype=np.float32).reshape(-1, 1, 2)
    if len(polygon) >= 3:
        cv2.polylines(canvas, [polygon.astype(np.int32)], True, (0, 255, 0), 3, cv2.LINE_AA)
    adjustment = 100.0 * (float(result.expansion_ratio) - 1.0)
    label = (
        f"{result.source} | samples={result.motion_sample_count}/{result.sample_count} | "
        f"area={result.occupied_area_ratio:.3f} | boundary={adjustment:+.1f}%"
    )
    cv2.putText(
        canvas,
        label[:180],
        (16, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    _write_png(canvas, png_path)
    if comparison_path is not None and result.source_video:
        save_boundary_overlay_frame(result.source_video, result, comparison_path, frame_index=0)
