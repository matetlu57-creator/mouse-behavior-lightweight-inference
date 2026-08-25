#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export lightweight inference results for the annotation website.

This module is deliberately an output adapter.  It does not participate in
tracking, feature calculation, thresholding, or behavior classification.  The
adapter follows ``已标记行为数据导入格式说明.md`` schema version 1.0 and the
website's existing three-file detection-import contract.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "1.0"
KEYPOINT_NAMES = (
    "nose",
    "left_ear",
    "right_ear",
    "neck",
    "left_hip",
    "right_hip",
    "tail",
)

# These strings must exactly match the annotation website's default category
# names.  Internal algorithm names remain unchanged.
WEBSITE_BEHAVIOR_NAMES = {
    "together": "一起",
    "approach": "接近",
    "chase": "追逐",
    "avoidance": "回避",
    "attack": "攻击行为",
    "nose_head_contact": "鼻头接触",
    "nose_tail_contact": "鼻尾接触",
    "huddle": "扎堆行为",
    "isolation": "孤立行为",
    "running": "奔跑",
    "walking": "行走",
    "stationary": "静止",
}

PAIR_BEHAVIORS = {
    "together",
    "approach",
    "chase",
    "avoidance",
    "attack",
    "nose_head_contact",
    "nose_tail_contact",
}
INDIVIDUAL_BEHAVIORS = {"running", "walking", "stationary", "isolation"}


def _json_number(value: Any, *, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def _json_int(value: Any, *, default: int = -1) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return int(default)
    return number


def _safe_directory_name(value: str) -> str:
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", str(value)).strip(" .")
    return value[:120] or "video"


def _pair_ids(pair_key: Any) -> list[int]:
    text = str(pair_key or "")
    match = re.fullmatch(r"(\d+)_(\d+)", text)
    if match is None:
        return []
    return sorted({int(match.group(1)), int(match.group(2))})


def _member_ids(value: Any) -> list[int]:
    """Parse event-level group members from native or CSV round-trip values."""

    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if parsed is not None and parsed != value:
            return _member_ids(parsed)
        values: Any = re.findall(r"-?\d+", text)
    elif isinstance(value, (list, tuple, set)):
        values = value
    elif hasattr(value, "tolist"):
        values = value.tolist()
        if not isinstance(values, (list, tuple, set)):
            values = [values]
    else:
        values = [value]
    result: set[int] = set()
    for item in values:
        try:
            member_id = int(item)
        except (TypeError, ValueError, OverflowError):
            continue
        if member_id >= 0:
            result.add(member_id)
    return sorted(result)


def _valid_track_ids(
    tracks: Mapping[str, np.ndarray], start_frame: int, end_frame: int
) -> set[int]:
    valid = np.asarray(tracks["valid"], dtype=bool)
    if valid.ndim != 2 or len(valid) == 0:
        return set()
    start = max(0, min(int(start_frame), len(valid) - 1))
    end = max(start, min(int(end_frame), len(valid) - 1))
    return {int(item) for item in np.flatnonzero(valid[start : end + 1].any(axis=0))}


def _largest_component_ids(points: np.ndarray, ids: np.ndarray, distance_cm: float) -> list[int]:
    if len(ids) < 2:
        return [int(item) for item in ids]
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    distances[~np.isfinite(distances)] = np.inf
    adjacency = distances <= float(distance_cm)
    np.fill_diagonal(adjacency, False)
    unseen = set(range(len(ids)))
    components: list[list[int]] = []
    while unseen:
        seed = unseen.pop()
        stack = [seed]
        component: list[int] = []
        while stack:
            current = stack.pop()
            component.append(current)
            neighbours = [candidate for candidate in unseen if adjacency[current, candidate]]
            for candidate in neighbours:
                unseen.remove(candidate)
                stack.append(candidate)
        components.append(component)
    largest = max(components, key=lambda item: (len(item), [-int(ids[index]) for index in item]))
    return sorted(int(ids[index]) for index in largest)


def _huddle_ids(
    tracks: Mapping[str, np.ndarray],
    start_frame: int,
    end_frame: int,
    distance_cm: float,
    cm_per_pixel: float,
) -> list[int]:
    valid = np.asarray(tracks["valid"], dtype=bool)
    centers_px = np.asarray(tracks["centers_px"], dtype=float)
    if valid.ndim != 2 or centers_px.ndim != 3 or len(valid) == 0:
        return []
    start = max(0, min(int(start_frame), len(valid) - 1))
    end = max(start, min(int(end_frame), len(valid) - 1))
    membership = np.zeros(valid.shape[1], dtype=int)
    maximum_size = 0
    for frame in range(start, end + 1):
        ids = np.flatnonzero(valid[frame])
        if len(ids) < 2:
            continue
        points = centers_px[frame, ids] * float(cm_per_pixel)
        component = _largest_component_ids(points, ids, distance_cm)
        maximum_size = max(maximum_size, len(component))
        for track_id in component:
            membership[track_id] += 1
    if maximum_size < 2 or membership.max(initial=0) <= 0:
        return []
    ranked = sorted(
        (int(track_id) for track_id in np.flatnonzero(membership > 0)),
        key=lambda track_id: (-int(membership[track_id]), track_id),
    )
    selected = sorted(ranked[:maximum_size])
    return selected if len(selected) >= 2 else []


def _isolation_id(
    tracks: Mapping[str, np.ndarray],
    start_frame: int,
    end_frame: int,
    cm_per_pixel: float,
) -> list[int]:
    valid = np.asarray(tracks["valid"], dtype=bool)
    centers_px = np.asarray(tracks["centers_px"], dtype=float)
    if valid.ndim != 2 or centers_px.ndim != 3 or len(valid) == 0:
        return []
    start = max(0, min(int(start_frame), len(valid) - 1))
    end = max(start, min(int(end_frame), len(valid) - 1))
    nearest_by_id: dict[int, list[float]] = {}
    for frame in range(start, end + 1):
        ids = np.flatnonzero(valid[frame])
        if len(ids) < 2:
            continue
        points = centers_px[frame, ids] * float(cm_per_pixel)
        distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
        distances[~np.isfinite(distances)] = np.inf
        np.fill_diagonal(distances, np.inf)
        nearest = np.min(distances, axis=1)
        for index, track_id in enumerate(ids):
            if math.isfinite(float(nearest[index])):
                nearest_by_id.setdefault(int(track_id), []).append(float(nearest[index]))
    if not nearest_by_id:
        return []
    selected = max(
        nearest_by_id,
        key=lambda track_id: (
            float(np.median(nearest_by_id[track_id])),
            len(nearest_by_id[track_id]),
            -track_id,
        ),
    )
    return [int(selected)]


def _event_mouse_ids(
    event: Mapping[str, Any],
    behavior: str,
    tracks: Mapping[str, np.ndarray],
    *,
    huddle_distance_cm: float,
    cm_per_pixel: float,
) -> list[int]:
    start = _json_int(event.get("start_frame"), default=0)
    end = _json_int(event.get("end_frame"), default=start)
    visible = _valid_track_ids(tracks, start, end)
    if behavior == "huddle":
        explicit_members = sorted(
            set(_member_ids(event.get("member_ids")))
            | set(_member_ids(event.get("member_ids_at_peak")))
        )
        if len(explicit_members) >= 2:
            return [track_id for track_id in explicit_members if track_id in visible]
        return [
            track_id
            for track_id in _huddle_ids(tracks, start, end, huddle_distance_cm, cm_per_pixel)
            if track_id in visible
        ]
    if behavior == "isolation":
        peak_members = _member_ids(event.get("member_ids_at_peak"))
        if len(peak_members) == 1 and peak_members[0] in visible:
            return peak_members
        return [
            track_id
            for track_id in _isolation_id(tracks, start, end, cm_per_pixel)
            if track_id in visible
        ]
    if behavior in {"running", "walking", "stationary"}:
        actor = _json_int(event.get("actor_id"), default=-1)
        return [actor] if actor >= 0 and actor in visible else []

    ids = set(_pair_ids(event.get("pair_key")))
    actor = _json_int(event.get("actor_id"), default=-1)
    target = _json_int(event.get("target_id"), default=-1)
    if actor >= 0:
        ids.add(actor)
    if target >= 0:
        ids.add(target)
    return sorted(track_id for track_id in ids if track_id in visible)


def _confidence(event: Mapping[str, Any]) -> str:
    ambiguous = str(event.get("role_ambiguous", "")).strip().lower() in {"1", "true", "yes"}
    level = str(event.get("candidate_level", event.get("level", ""))).strip().lower()
    peak_score = _json_number(event.get("peak_score"), default=0.0)
    if ambiguous or level == "weak" or (level == "extended" and peak_score < 0.5):
        return "uncertain"
    return "certain"


def _annotation_record(
    event: Mapping[str, Any],
    behavior: str,
    mouse_ids: Sequence[int],
    *,
    fps: float,
    frame_count: int,
) -> dict[str, Any]:
    start = max(0, min(_json_int(event.get("start_frame"), default=0), frame_count - 1))
    end = max(start, min(_json_int(event.get("end_frame"), default=start), frame_count - 1))
    return {
        "behavior": WEBSITE_BEHAVIOR_NAMES[behavior],
        "start_frame": int(start),
        "end_frame": int(end),
        "start_time": float(start / fps),
        "end_time": float((end + 1) / fps),
        "mouse_ids": sorted({int(item) for item in mouse_ids}),
        "confidence": _confidence(event),
        "crop_region": None,
    }


def _contact_behavior_rows(contact: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    contact_type = str(contact.get("contact_type", "")).strip()
    if contact_type == "nose_head":
        behaviors = ["nose_head_contact"]
    elif contact_type == "nose_tail":
        behaviors = ["nose_tail_contact"]
    elif contact_type == "nose_head_and_nose_tail":
        behaviors = ["nose_head_contact", "nose_tail_contact"]
    else:
        return []
    event = dict(contact)
    event["actor_id"] = contact.get("contact_actor_id", -1)
    event["target_id"] = contact.get("contact_target_id", -1)
    event["candidate_level"] = "extended"
    event["peak_score"] = 1.0
    return [(behavior, event) for behavior in behaviors]


def build_annotations(
    behavior_events: Sequence[Mapping[str, Any]],
    contact_events: Sequence[Mapping[str, Any]],
    tracks: Mapping[str, np.ndarray],
    *,
    fps: float,
    frame_count: int,
    huddle_distance_cm: float,
    cm_per_pixel: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Convert algorithm events without changing the source event objects."""
    candidates: list[tuple[str, Mapping[str, Any]]] = []
    for event in behavior_events:
        behavior = str(event.get("behavior", "")).strip()
        if behavior in WEBSITE_BEHAVIOR_NAMES and behavior not in {
            "nose_head_contact",
            "nose_tail_contact",
        }:
            candidates.append((behavior, event))
    for contact in contact_events:
        candidates.extend(_contact_behavior_rows(contact))

    annotations: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for behavior, event in candidates:
        mouse_ids = _event_mouse_ids(
            event,
            behavior,
            tracks,
            huddle_distance_cm=huddle_distance_cm,
            cm_per_pixel=cm_per_pixel,
        )
        expected_ok = (
            len(mouse_ids) == 1
            if behavior in INDIVIDUAL_BEHAVIORS
            else len(mouse_ids) >= 2
            if behavior == "huddle"
            else len(mouse_ids) == 2
        )
        if not expected_ok:
            skipped.append(
                {
                    "source_event_id": str(
                        event.get("light_event_id") or event.get("contact_event_id") or ""
                    ),
                    "behavior": WEBSITE_BEHAVIOR_NAMES[behavior],
                    "reason": "participant_count_mismatch",
                    "resolved_mouse_ids": mouse_ids,
                }
            )
            continue
        record = _annotation_record(
            event,
            behavior,
            mouse_ids,
            fps=fps,
            frame_count=frame_count,
        )
        dedupe_key = (
            record["behavior"],
            record["start_frame"],
            record["end_frame"],
            tuple(record["mouse_ids"]),
        )
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        annotations.append(record)
    annotations.sort(
        key=lambda item: (
            int(item["start_frame"]),
            int(item["end_frame"]),
            str(item["behavior"]),
            tuple(item["mouse_ids"]),
        )
    )
    return annotations, skipped


def _write_tracks_jsonl(
    path: Path,
    tracks: Mapping[str, np.ndarray],
    *,
    fps: float,
    frame_count: int,
    video_id: str,
    width: int,
    height: int,
) -> int:
    valid = np.asarray(tracks["valid"], dtype=bool)
    boxes = np.asarray(tracks["bboxes"], dtype=float)
    keypoints = np.asarray(tracks["keypoints_px"], dtype=float)
    confidences = np.asarray(tracks["confidences"], dtype=float)
    pose_quality = np.asarray(tracks["pose_quality"], dtype=float)
    available_frames = min(len(valid), len(boxes), len(keypoints), len(confidences))
    detection_count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for frame_index in range(frame_count):
            detections: list[dict[str, Any]] = []
            if frame_index < available_frames:
                for track_id in np.flatnonzero(valid[frame_index]):
                    box = boxes[frame_index, track_id]
                    points = keypoints[frame_index, track_id]
                    point_conf = confidences[frame_index, track_id]
                    if box.shape != (4,) or not np.all(np.isfinite(box)):
                        continue
                    if points.shape != (len(KEYPOINT_NAMES), 2):
                        continue
                    x1 = float(np.clip(box[0], 0.0, max(width - 1.0, 0.0)))
                    y1 = float(np.clip(box[1], 0.0, max(height - 1.0, 0.0)))
                    x2 = float(np.clip(box[2], 0.0, float(width)))
                    y2 = float(np.clip(box[3], 0.0, float(height)))
                    if x1 >= x2 or y1 >= y2:
                        continue
                    website_keypoints = []
                    for point_index in range(len(KEYPOINT_NAMES)):
                        x = _json_number(points[point_index, 0], default=0.0)
                        y = _json_number(points[point_index, 1], default=0.0)
                        confidence = float(
                            np.clip(
                                _json_number(point_conf[point_index], default=0.0),
                                0.0,
                                1.0,
                            )
                        )
                        website_keypoints.append(
                            {
                                "x_px": float(np.clip(x, 0.0, float(width))),
                                "y_px": float(np.clip(y, 0.0, float(height))),
                                "confidence": confidence,
                            }
                        )
                    detections.append(
                        {
                            "track_id": int(track_id),
                            "box_xyxy_px": [x1, y1, x2, y2],
                            "detection_confidence": float(
                                np.clip(
                                    _json_number(pose_quality[frame_index, track_id], default=0.0),
                                    0.0,
                                    1.0,
                                )
                            ),
                            "class_id": 0,
                            "class_name": "mouse",
                            "keypoints": website_keypoints,
                        }
                    )
            detections.sort(key=lambda item: int(item["track_id"]))
            detection_count += len(detections)
            frame = {
                "schema_version": SCHEMA_VERSION,
                "video_id": video_id,
                "frame_index": int(frame_index),
                "timestamp_sec": float(frame_index / fps),
                "detection_count": len(detections),
                "detections": detections,
            }
            handle.write(json.dumps(frame, ensure_ascii=False, separators=(",", ":")) + "\n")
    return int(detection_count)


def _materialize_video(source: Path, destination: Path) -> str:
    """Use an atomic hard link when possible and a copy on another volume."""
    source = source.resolve()
    if destination.exists():
        try:
            if os.path.samefile(source, destination):
                return "existing"
        except OSError:
            pass
    temporary = destination.with_name(destination.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    mode = "hardlink"
    try:
        os.link(source, temporary)
    except OSError:
        mode = "copy"
        shutil.copy2(source, temporary)
    os.replace(temporary, destination)
    return mode


def export_complete_video_package(
    *,
    source_video: Path,
    output_dir: Path,
    behavior_events: Sequence[Mapping[str, Any]],
    contact_events: Sequence[Mapping[str, Any]],
    tracks: Mapping[str, np.ndarray],
    fps: float,
    frame_count: int,
    width: int,
    height: int,
    skeleton_edges: Sequence[Sequence[int]],
    cm_per_pixel: float,
    huddle_distance_cm: float,
    tracker_params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write one full-video directory that is ready to place in an upload ZIP."""
    if not source_video.is_file():
        raise FileNotFoundError(f"源视频不存在: {source_video}")
    if not math.isfinite(float(fps)) or float(fps) <= 0:
        raise ValueError(f"FPS 必须为正数: {fps}")
    if int(frame_count) <= 0 or int(width) <= 0 or int(height) <= 0:
        raise ValueError("frame_count、width 和 height 必须为正整数")

    package_root = output_dir / "annotation_website_import"
    video_id = _safe_directory_name(source_video.stem)
    video_dir = package_root / video_id
    video_dir.mkdir(parents=True, exist_ok=True)
    suffix = source_video.suffix.lower() if source_video.suffix else ".mp4"
    video_filename = f"video{suffix}"
    video_mode = _materialize_video(source_video, video_dir / video_filename)

    annotations, skipped = build_annotations(
        behavior_events,
        contact_events,
        tracks,
        fps=float(fps),
        frame_count=int(frame_count),
        huddle_distance_cm=float(huddle_distance_cm),
        cm_per_pixel=float(cm_per_pixel),
    )
    annotation_payload = {
        "schema_version": SCHEMA_VERSION,
        "video_file": video_filename,
        "annotations": annotations,
    }
    (video_dir / "annotations.json").write_text(
        json.dumps(annotation_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    detection_count = _write_tracks_jsonl(
        video_dir / "tracks.jsonl",
        tracks,
        fps=float(fps),
        frame_count=int(frame_count),
        video_id=video_id,
        width=int(width),
        height=int(height),
    )
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "video_id": video_id,
        "source_relative": video_filename,
        "width": int(width),
        "height": int(height),
        "fps": float(fps),
        "frame_count": int(frame_count),
        "coordinate_system": "pixel_xy_origin_top_left",
        "box_format": "xyxy",
        "keypoint_format": "object_x_px_y_px_confidence",
        "keypoint_names": list(KEYPOINT_NAMES),
        "skeleton_edges": [[int(a), int(b)] for a, b in skeleton_edges],
        "model_name": "mouse-pose-v8 best.pt",
        "tracker_name": "position_plus_keypoint_hungarian",
        "tracker_params": dict(tracker_params or {}),
        "class_names": ["mouse"],
    }
    (video_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    report = {
        "schema_version": SCHEMA_VERSION,
        "package_root": str(package_root),
        "video_directory": str(video_dir),
        "video_materialization": video_mode,
        "video_file": video_filename,
        "frame_count": int(frame_count),
        "detection_count": int(detection_count),
        "annotation_count": len(annotations),
        "skipped_event_count": len(skipped),
        "skipped_events": skipped,
        "upload_note": "将 annotation_website_import 目录内的视频子目录整体打成 ZIP。",
    }
    (output_dir / "annotation_website_export_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report
