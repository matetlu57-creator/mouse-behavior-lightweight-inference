"""Rendering and raw behavior-clip extraction from persisted results."""

from __future__ import annotations

import json
import hashlib
import logging
import math
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import pandas as pd

from .. import adaptive_arena_boundary as arena_boundary
from ..identity.labels import identity_label
from ..identity.mtrack import load_identity_map
from ..preprocessing.constants import (
    BEHAVIOR_NAMES_ZH,
    EXTENDED_BEHAVIORS,
    FOUR_CLASS_NAMES,
    SKELETON_EDGES,
)
from ..preprocessing.pair_features import _boolean_runs
from ..tracking.cache import _cache_total_frames, load_trajectory_cache
from ..io.csv import _write_csv
from ..evaluation.segment_export import _sha256, _verify_video
from ..pose_cache import verified_cache_source_binding
from .overlay import (
    BoxLabel,
    MouseOverlay,
    build_mouse_overlays,
    draw_behavior_sidebar,
    draw_text_overlay,
    font_size_for_frame,
    format_mouse_id,
    load_font,
    normalize_contact_events,
    canonical_behavior,
    normalize_focus_behavior,
    resolve_event_for_frame,
    sidebar_width_for_frame,
    select_display_events,
)

LOGGER = logging.getLogger("mouse_behavior.lightweight_behavior_inference")


def _verify_render_source(video: Path, cache: Path, metadata: dict, fps: float, count: int) -> dict:
    """Refuse missing/mismatched source timelines instead of truncating them."""
    status_path = cache / "yolo_results_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if not status.get("video") or not metadata.get("source_video") or not metadata.get("yolo_cache"):
        raise ValueError("render source binding is missing; rerun cache generation/analysis")
    if (Path(status["video"]).resolve() != video.resolve()
            or Path(metadata["source_video"]).resolve() != video.resolve()
            or Path(metadata["yolo_cache"]).resolve() != cache.resolve()):
        raise ValueError("render source/cache path differs from saved cache or analysis")
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("source FPS must be finite and positive")
    for name, expected in (("cache", status.get("fps")), ("analysis", metadata.get("source_fps"))):
        if not isinstance(expected, (int, float)) or not math.isclose(fps, expected, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(f"source FPS differs from {name} binding")
    if count <= 0 or count != status.get("total_frames"):
        raise ValueError("source frame count differs from complete cache binding")
    analyzed = metadata.get("source_frames")
    if type(analyzed) is not int or not 0 < analyzed <= count:
        raise ValueError("analysis source frame count is missing or invalid")
    binding = verified_cache_source_binding(video, cache, fps=fps, frame_count=count)
    if metadata.get("source_binding") != binding:
        raise ValueError("analysis source/cache binding missing or changed; rerun analysis")
    return dict(**binding, source_binding_level="sha256_and_path_timebase",
                analyzed_source_frame_count=analyzed)


def _verified_render_identity(metadata_path: Path, explicit_map: Path | None):
    """Bind event slot identities to the mapping bytes used during analysis."""
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    if not isinstance(metadata, dict):
        raise ValueError("analysis identity metadata must be an object")
    map_path = explicit_map or metadata.get("identity_map")
    if map_path is None:
        if metadata.get("identity_source") == "rfid_mtrack":
            raise ValueError("RFID analysis is missing its identity map")
        return metadata, None
    mapping = load_identity_map(map_path)
    if (metadata.get("identity_source") != "rfid_mtrack"
            or not metadata.get("identity_map_sha256")
            or not metadata.get("identity_labels")):
        raise ValueError("RFID render requires sealed analysis identity metadata; rerun analysis")
    if mapping.source_sha256 != metadata["identity_map_sha256"]:
        raise ValueError("identity map hash changed since analysis")
    return metadata, mapping


def render_behavior_video(
    video_path: Path,
    cache_dir: Path,
    events_path: Path,
    output_path: Path,
    expected_mice: int = 20,
    max_frames: int | None = None,
    contact_events_path: Path | None = None,
    font_path: Path | None = None,
    focus_behavior: str | None = None,
    identity_map_path: Path | None = None,
) -> Path:
    """Render exactly one annotated MP4 for one source video.

    The renderer deliberately consumes the lightweight track IDs and event
    CSV produced by this module.  It does not create event clips or any other
    video outputs.
    """
    total_cache_frames = _cache_total_frames(cache_dir)
    if max_frames is not None and (type(max_frames) is not int or max_frames <= 0):
        raise ValueError("max_frames must be a positive integer")

    metadata, identity_map = _verified_render_identity(
        events_path.parent / "lightweight_analysis_metadata.json", identity_map_path,
    )
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开源视频: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"无法读取源视频尺寸: {video_path}")
    source_binding = _verify_render_source(video_path, cache_dir, metadata, fps, video_frame_count)
    event_hashes = metadata.get("event_files_sha256", {})
    if not events_path.is_file() or event_hashes.get(events_path.name) != _sha256(events_path):
        raise ValueError("analysis event CSV binding missing or changed; rerun analysis")
    if max_frames is None and source_binding["analyzed_source_frame_count"] != total_cache_frames:
        raise ValueError("partial analysis requires an explicit max_frames preview")
    if max_frames is not None:
        total_cache_frames = min(total_cache_frames, max_frames)
    if total_cache_frames > source_binding["analyzed_source_frame_count"]:
        raise ValueError("render frames exceed saved analysis source frame count")
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
    tracks, _tracking_stats = load_trajectory_cache(
        cache_dir,
        total_cache_frames,
        expected_mice,
        identity_map=identity_map,
        arena_polygon=arena_polygon,
    )
    identity_labels = tuple(str(value) for value in tracks["identity_labels"])
    if (("identity_labels" in metadata and tuple(metadata["identity_labels"]) != identity_labels)
            or ("identity_source" in metadata
                and metadata["identity_source"] != tracks["identity_source"])):
        cap.release()
        raise ValueError("render identity labels/source differ from the saved analysis")

    event_frame_map: list[list[dict[str, Any]]] = [[] for _ in range(total_cache_frames)]
    if not events_path.exists():
        raise FileNotFoundError(f"行为事件 CSV 不存在: {events_path}")
    events_df = pd.read_csv(events_path)
    required = {"start_frame", "end_frame", "candidate_level", "behavior"}
    missing = sorted(required.difference(events_df.columns))
    if missing:
        raise ValueError(f"行为事件 CSV 缺少字段: {missing}")
    event_rows: list[dict[str, Any]] = events_df.to_dict("records")
    normalized_focus = normalize_focus_behavior(focus_behavior)
    contact_path = contact_events_path or (events_path.parent / "lightweight_contact_events.csv")
    if contact_path.exists():
        if event_hashes.get(contact_path.name) != _sha256(contact_path):
            raise ValueError("analysis contact event CSV binding missing or changed; rerun analysis")
        contact_df = pd.read_csv(contact_path)
        event_rows.extend(normalize_contact_events(contact_df.to_dict("records")))
        LOGGER.info("[render] contact events loaded: %s", contact_path)
    for event in event_rows:
        try:
            start = max(int(event.get("start_frame", 0)), 0)
            end = min(int(event.get("end_frame", -1)), total_cache_frames - 1)
        except (TypeError, ValueError):
            continue
        if end < start:
            continue
        for frame_index in range(start, end + 1):
            resolved_event = resolve_event_for_frame(event, frame_index)
            if resolved_event is not None:
                event_frame_map[frame_index].append(resolved_event)

    frame_limit = total_cache_frames

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() or output_path.with_suffix(".identity.json").exists():
        raise FileExistsError(f"refusing to overwrite render: {output_path}")
    temporary_output = output_path.with_name(f".{output_path.stem}.{uuid.uuid4().hex}.mp4")
    cap = cv2.VideoCapture(str(video_path))
    sidebar_width = sidebar_width_for_frame(width, height)
    writer = cv2.VideoWriter(
        str(temporary_output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width + sidebar_width, height),
    )
    if not writer.isOpened():
        cap.release()
        writer.release()
        temporary_output.unlink(missing_ok=True)
        raise RuntimeError(f"无法创建渲染视频: {output_path}")

    skeleton_edges = SKELETON_EDGES
    overlay_size = font_size_for_frame(width, height)
    overlay_font = load_font(font_path, overlay_size)
    sidebar_header_font = load_font(font_path, max(overlay_size + 4, 18))
    sidebar_small_font = load_font(font_path, max(overlay_size - 1, 12))

    frame_index = 0
    focus_evidence_frames = 0
    try:
        while frame_index < frame_limit:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"source decode ended at {frame_index}; expected {frame_limit} frames")
            active_events = event_frame_map[frame_index]
            focus_active = bool(
                normalized_focus
                and any(
                    canonical_behavior(event.get("behavior")) == normalized_focus
                    for event in active_events
                )
            )
            if focus_active:
                focus_evidence_frames += 1
            display_events, display_layer = select_display_events(
                active_events,
                focus_behavior=normalized_focus,
            )
            valid_ids = [
                logical_id
                for logical_id in range(expected_mice)
                if bool(tracks["valid"][frame_index, logical_id])
                or bool(tracks["bbox_imputed"][frame_index, logical_id])
            ]
            mouse_overlays = build_mouse_overlays(
                display_events,
                display_layer,
                valid_ids,
                focus_behavior=normalized_focus,
            )
            for logical_id, overlay in tuple(mouse_overlays.items()):
                display_identity = identity_label(logical_id, identity_labels)
                if display_identity is None:
                    continue
                mouse_overlays[logical_id] = MouseOverlay(
                    text=overlay.text.replace(
                        format_mouse_id(logical_id), display_identity, 1
                    ),
                    color_bgr=overlay.color_bgr,
                    priority=overlay.priority,
                )
            box_labels: list[BoxLabel] = []

            for logical_id in range(expected_mice):
                pose_valid = bool(tracks["valid"][frame_index, logical_id])
                bbox_held = bool(tracks["bbox_imputed"][frame_index, logical_id])
                if not pose_valid and not bbox_held:
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
                overlay = mouse_overlays.get(
                    logical_id,
                    MouseOverlay(
                        text=identity_label(logical_id, identity_labels)
                        or format_mouse_id(logical_id),
                        color_bgr=(170, 170, 170),
                        priority=(0, 0.0),
                    ),
                )
                if bbox_held:
                    overlay = MouseOverlay(
                        text=f"{overlay.text}｜遮挡保持",
                        color_bgr=overlay.color_bgr,
                        priority=overlay.priority,
                    )
                color = overlay.color_bgr
                thickness = 3 if overlay.priority[0] > 0 else 1
                line_type = cv2.LINE_AA
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, line_type)
                box_labels.append(BoxLabel((x1, y1, x2, y2), overlay.text, color))
                if bbox_held:
                    # A dashed inner line makes a predicted box visibly
                    # different from a real detector observation without
                    # hiding the behavior/ID label.
                    dash = max(min((x2 - x1) // 8, (y2 - y1) // 8), 4)
                    for left in range(x1, x2, dash * 2):
                        cv2.line(
                            frame,
                            (left, y1),
                            (min(left + dash, x2), y1),
                            color,
                            1,
                            line_type,
                        )
                        cv2.line(
                            frame,
                            (left, y2),
                            (min(left + dash, x2), y2),
                            color,
                            1,
                            line_type,
                        )
                    for top in range(y1, y2, dash * 2):
                        cv2.line(
                            frame,
                            (x1, top),
                            (x1, min(top + dash, y2)),
                            color,
                            1,
                            line_type,
                        )
                        cv2.line(
                            frame,
                            (x2, top),
                            (x2, min(top + dash, y2)),
                            color,
                            1,
                            line_type,
                        )
                if not pose_valid:
                    continue
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

            frame = draw_text_overlay(frame, box_labels, [], overlay_font)
            frame = draw_behavior_sidebar(
                frame,
                frame_index=frame_index,
                total_frames=frame_limit,
                fps=fps,
                active_event_count=len(active_events),
                display_events=display_events,
                display_layer=display_layer,
                mouse_overlays=mouse_overlays,
                valid_ids=valid_ids,
                font=overlay_font,
                header_font=sidebar_header_font,
                small_font=sidebar_small_font,
                panel_width=sidebar_width,
                focus_behavior=normalized_focus,
                focus_active=focus_active,
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
        if frame_limit == video_frame_count and cap.read()[0]:
            raise RuntimeError("source decodes more frames than its bound frame count")
    except BaseException:
        cap.release()
        writer.release()
        temporary_output.unlink(missing_ok=True)
        raise
    finally:
        cap.release()
        writer.release()
    try:
        verification = _verify_video(temporary_output, frame_limit, fps)
        if _verify_render_source(video_path, cache_dir, metadata, fps, video_frame_count) != source_binding:
            raise ValueError("source/cache binding changed during rendering")
        for path in (events_path, contact_path):
            if path.exists() and event_hashes.get(path.name) != _sha256(path):
                raise ValueError("event CSV binding changed during rendering")
        temporary_output.replace(output_path)
    except BaseException:
        temporary_output.unlink(missing_ok=True)
        raise
    LOGGER.info(
        "[render] completed: %s (%.2f GB, %dx%d including sidebar)",
        output_path,
        output_path.stat().st_size / (1024**3),
        width + sidebar_width,
        height,
    )
    if normalized_focus:
        LOGGER.info(
            "[render] review_heading_focus=%s heading_persistent=true evidence_coverage=%.3f",
            normalized_focus,
            focus_evidence_frames / max(frame_index, 1),
        )
    output_path.with_suffix(".identity.json").write_text(
        json.dumps({
            "source_video": str(video_path), "events_path": str(events_path),
            "events_sha256": hashlib.sha256(events_path.read_bytes()).hexdigest(),
            "identity_source": tracks["identity_source"], "identity_labels": identity_labels,
            "identity_map": tracks.get("identity_map"),
            "identity_map_sha256": tracks.get("identity_map_sha256"),
            **source_binding,
            "render_contract": "explicit_preview" if max_frames is not None else "complete_analysis",
            "requested_max_frames": max_frames, "source_frame_count": video_frame_count,
            "rendered_frame_count": frame_index, "source_fps": fps,
            "source_frame_range_half_open": [0, frame_index], "verification": verification,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
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


def _write_showcase_videos(video_path: Path, intervals: list[dict], total_frames: int,
                           fps: float, width: int, height: int) -> None:
    """Decode and verify the whole batch before publishing any final clip."""
    if not intervals:
        return
    paths = [Path(interval["path"]) for interval in intervals]
    if any(path.exists() for path in paths):
        raise FileExistsError("refusing to overwrite an existing showcase clip")
    temporary = {str(path): path.with_name(f".{path.stem}.{uuid.uuid4().hex}.mp4") for path in paths}
    start_map: dict[int, list[dict]] = defaultdict(list)
    for interval in intervals:
        start_map[interval["start_frame"]].append(interval)
    active = {}
    published = []
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"cannot open showcase source: {video_path}")
        for frame_index in range(total_frames):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"source decode ended at {frame_index}; expected {total_frames} frames")
            for interval in start_map.get(frame_index, []):
                writer = cv2.VideoWriter(str(temporary[interval["path"]]), cv2.VideoWriter_fourcc(*"mp4v"),
                                         fps, (width, height))
                if not writer.isOpened():
                    writer.release()
                    raise RuntimeError(f"cannot create showcase clip: {interval['path']}")
                active[interval["path"]] = (writer, interval)
            for path, (writer, interval) in list(active.items()):
                writer.write(frame)
                if frame_index == interval["end_frame"]:
                    writer.release()
                    del active[path]
        if cap.read()[0]:
            raise RuntimeError("showcase source decodes more frames than its bound frame count")
        cap.release()
        for interval in intervals:
            verified = _verify_video(temporary[interval["path"]], interval["frame_count"], fps)
            interval.update(encoded_fps=verified["fps"], duration_s=verified["duration_s"],
                            decoded_frame_count=verified["frame_count"])
        for path in paths:
            temporary[str(path)].rename(path)
            published.append(path)
    except BaseException:
        for writer, _ in active.values():
            writer.release()
        for path in [*temporary.values(), *published]:
            path.unlink(missing_ok=True)
        raise
    finally:
        cap.release()
        for writer, _ in active.values():
            writer.release()


def extract_four_class_clips(
    video_path: Path,
    events_path: Path,
    output_dir: Path,
    expected_level: str = "strong",
    clip_seconds: float = 5.0,
    min_start_interval_seconds: float = 5.0,
    max_clips_per_class: int = 200,
) -> Path:
    """Extract fixed-window showcases into the four mutually exclusive classes.

    Class IDs follow the established extractor contract:
    ``chase + 2 * attack`` -> 0/1/2/3.  This function never draws boxes,
    skeletons, IDs, labels, or overlays; it writes only source-video crops.
    A class records event presence within a sample, not an independently
    qualified behavior process. Use canonical behavior packages for that contract.
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
                    "frame_count": end - start + 1,
                    "start_frame": start,
                    "end_frame": end,
                    "start_time_s": start / fps,
                    "end_time_s": end / fps,
                    "duration_s": (end - start + 1) / fps,
                    "path": str(path),
                    "source_video": str(video_path),
                    "source_events": str(events_path),
                    "event_level": expected_level,
                    "clip_contract": "fixed_window_showcase",
                    "standalone_behavior_qualification": False,
                }
            )

    _write_showcase_videos(video_path, intervals, total_frames, fps, width, height)
    manifest_path = output_dir / "four_class_clip_manifest.csv"
    _write_csv(manifest_path, intervals, columns=list(intervals[0]) if intervals else [
        "clip_index", "label_name", "start_frame", "end_frame", "frame_count", "duration_s", "path",
    ])
    summary = {
        "source_video": str(video_path),
        "events_csv": str(events_path),
        "event_level": expected_level,
        "fps": fps,
        "source_frames": total_frames,
        "clip_seconds": float(clip_seconds),
        "min_start_interval_seconds": float(min_start_interval_seconds),
        "max_clips_per_class": int(max_clips_per_class),
        "insufficient_source_policy": "skip_incomplete_fixed_window",
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
        "clip_contract": "fixed_window_showcase",
        "standalone_behavior_qualification": False,
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
    """Extract fixed-window raw showcases separately for each behavior label.

    This is the default clip contract for the lightweight ethogram.  It reads
    the already generated behavior-event CSV, builds one frame state per
    behavior, and writes raw source-video clips under one directory per label.
    It does not create four mutually exclusive classes and does not draw
    overlays.  ``event_level='all'`` keeps both the extended ethogram rows and
    the weak/strong legacy chase/attack rows.
    Presence of a label in the sampled window does not qualify its standalone
    process duration. Canonical behavior packages enforce that semantic contract.
    Sources shorter than the requested fixed window yield zero clips. Every
    published clip is decoded to verify its complete requested frame range.
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
            if total_frames < clip_frames or max_clips_per_behavior <= 0:
                break
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
                    "frame_count": end - start + 1,
                    "path": str(path),
                    "source_video": str(video_path),
                    "source_events": str(events_path),
                    "clip_contract": "fixed_window_showcase",
                    "standalone_behavior_qualification": False,
                }
            )

    _write_showcase_videos(video_path, intervals, total_frames, fps, width, height)
    manifest_path = output_dir / "behavior_clip_manifest.csv"
    _write_csv(manifest_path, intervals, columns=list(intervals[0]) if intervals else [
        "clip_index", "behavior", "start_frame", "end_frame", "frame_count", "duration_s", "path",
    ])
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
        "insufficient_source_policy": "skip_incomplete_fixed_window",
        "rendered_video": False,
        "clip_contract": "fixed_window_showcase",
        "standalone_behavior_qualification": False,
    }
    with (output_dir / "behavior_clip_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    LOGGER.info(
        "[behavior clips] completed: %s",
        ", ".join(f"{behavior}={clip_counts.get(behavior, 0)}" for behavior in selected_behaviors),
    )
    return output_dir
