"""Rendering and raw behavior-clip extraction from persisted results."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import pandas as pd

from .. import adaptive_arena_boundary as arena_boundary
from ..preprocessing.constants import (
    BEHAVIOR_NAMES_ZH,
    EXTENDED_BEHAVIORS,
    FOUR_CLASS_NAMES,
    SKELETON_EDGES,
)
from ..preprocessing.pair_features import _boolean_runs
from ..tracking.cache import _cache_total_frames, _track_cache
from ..io.csv import _write_csv

LOGGER = logging.getLogger("mouse_behavior.lightweight_behavior_inference")


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

    event_frame_map: list[list[dict[str, Any]]] = [[] for _ in range(total_cache_frames)]
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
                if previous is None or float(event.get("peak_score", 0.0)) > float(
                    previous.get("peak_score", 0.0)
                ):
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
        output_path.stat().st_size / (1024**3),
    )
    return output_path


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
            path = (
                output_dir
                / class_name
                / (f"{class_name}_{clip_index:04d}_{start / fps:.2f}s_{end / fps:.2f}s.mp4")
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
        selected_behaviors = [behavior for behavior in EXTENDED_BEHAVIORS if behavior in available]
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
            behavior: dict(level_counts[behavior]) for behavior in selected_behaviors
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
