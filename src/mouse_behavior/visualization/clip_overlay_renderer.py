"""Shared browser-ready H.264 video cropping and overlay rendering for behavior comparisons."""
from __future__ import annotations

import json
import logging
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import imageio_ffmpeg
import numpy as np

LOGGER = logging.getLogger(__name__)

ROLE_COLORS: dict[str, tuple[int, int, int]] = {
    "Attacker": (30, 30, 235),       # Bright Red
    "Attacked": (240, 175, 40),      # Cyan-Blue
    "Chasing": (20, 140, 255),       # Bright Orange
    "Chaser": (20, 140, 255),        # Bright Orange
    "Chased": (50, 220, 50),         # Bright Green
    "Static": (250, 180, 50),        # Sky Blue
    "Locomotion": (60, 210, 100),    # Emerald
    "Isolation": (255, 180, 20),     # Cyan-Amber
    "Isolated": (255, 180, 20),      # Cyan-Amber
    "Group locomotion": (20, 140, 255),  # Bright Orange
    "Group Locomotion": (20, 140, 255),  # Bright Orange
}
NON_TARGET_COLOR: tuple[int, int, int] = (150, 150, 150)

__all__ = [
    "NON_TARGET_COLOR",
    "ROLE_COLORS",
    "normalize_roles_map",
    "render_annotated_original_video",
    "render_cropped_comparison_video",
    "transcode_to_h264",
    "video_frame_count",
]


def normalize_roles_map(raw_roles: Any) -> dict[str, list[int]]:
    """Normalize role mapping JSON (list/tuple/set or scalar ID) into ``role -> [int]``."""
    parsed: dict[str, list[int]] = {}
    if not isinstance(raw_roles, dict):
        return parsed
    for role_name, val in raw_roles.items():
        if isinstance(val, (list, tuple, set)):
            parsed[str(role_name)] = [int(v) for v in val]
        elif isinstance(val, (int, float)) and not isinstance(val, bool):
            parsed[str(role_name)] = [int(val)]
    return parsed


def video_frame_count(video_path: Path) -> int:
    """Return frame count of a video file, or -1 if unreadable."""
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            return -1
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()


def transcode_to_h264(
    raw_video: Path,
    final_output: Path,
    *,
    ffmpeg_exe: str | None = None,
    remove_raw: bool = True,
) -> bool:
    """Transcode a raw MP4 to browser-compatible H.264 (yuv420p + faststart)."""
    exe = ffmpeg_exe or imageio_ffmpeg.get_ffmpeg_exe()
    tmp_h264 = final_output.with_suffix(".tmp_h264.mp4")
    conv = subprocess.run(
        [
            exe,
            "-y",
            "-i",
            str(raw_video),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(tmp_h264),
        ],
        capture_output=True,
    )
    if remove_raw and raw_video.is_file():
        raw_video.unlink(missing_ok=True)
    if conv.returncode == 0 and tmp_h264.is_file() and tmp_h264.stat().st_size > 0:
        tmp_h264.replace(final_output)
        return True
    tmp_h264.unlink(missing_ok=True)
    return False


def _index_tracks(tracks: Sequence[Mapping[str, Any]] | None) -> dict[int, list[Mapping[str, Any]]]:
    tracks_by_frame: dict[int, list[Mapping[str, Any]]] = {}
    if tracks:
        for item in tracks:
            f_idx = int(item.get("frame", -1))
            if f_idx >= 0:
                tracks_by_frame[f_idx] = list(item.get("detections", []))
    return tracks_by_frame


def _draw_detection_boxes(
    frame: np.ndarray,
    detections: Sequence[Mapping[str, Any]],
    *,
    target_ids: set[int],
    roles_map: Mapping[str, Sequence[int]],
    behavior: str | None,
    scale: float,
    thick: int,
    min_tag_y: int,
    default_target_role: str | None = None,
    non_target_role: str | None = None,
) -> None:
    beh_role = default_target_role
    if not beh_role and behavior:
        if behavior == "Isolation":
            beh_role = "Isolated"
        elif behavior == "Group locomotion":
            beh_role = "Group Locomotion"
        else:
            beh_role = behavior

    for det in detections:
        tid = int(det.get("track_id", -1))
        box = det.get("box")
        if not box or len(box) != 4 or not np.isfinite(box).all():
            continue
        x1, y1, x2, y2 = [int(round(float(v))) for v in box]
        is_target = tid in target_ids

        role_name = ""
        for r_name, r_ids in roles_map.items():
            if tid in r_ids:
                role_name = r_name
                break

        if is_target:
            effective_role = role_name or (beh_role or "")
            color = ROLE_COLORS.get(effective_role, ROLE_COLORS.get(behavior or "", (0, 230, 255)))
            box_th = max(2, int(round(3 * scale)))
            tag_role = effective_role
        else:
            color = NON_TARGET_COLOR
            box_th = max(1, int(round(1 * scale)))
            tag_role = non_target_role or ""

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_th)

        tag = f"ID: {tid}" + (f" [{tag_role}]" if tag_role else "")
        font = cv2.FONT_HERSHEY_SIMPLEX
        tag_sc = 0.55 * scale
        tag_th = max(1, thick - 1)
        (tw, th_box), _ = cv2.getTextSize(tag, font, tag_sc, tag_th)
        tag_y = max(y1 - 6, th_box + min_tag_y)
        cv2.rectangle(frame, (x1, tag_y - th_box - 4), (x1 + tw + 6, tag_y + 2), color, -1)
        text_col = (0, 0, 0) if (color[0] + color[1] + color[2]) > 380 else (255, 255, 255)
        cv2.putText(frame, tag, (x1 + 3, tag_y - 2), font, tag_sc, text_col, tag_th, cv2.LINE_AA)

        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(frame, (cx, cy), max(2, int(round(4 * scale))), color, -1)


def render_cropped_comparison_video(
    source_video: Path,
    output_video: Path,
    start_frame: int,
    end_frame: int,
    fps: float,
    *,
    tracks: Sequence[Mapping[str, Any]] | None = None,
    target_mouse_ids: Sequence[int] | None = None,
    semantic_roles: Mapping[str, Sequence[int]] | None = None,
    pred_behavior: str | None = None,
    gt_behavior: str | None = None,
    iou: float = 0.0,
    visualize: bool = True,
    ffmpeg_exe: str | None = None,
) -> bool:
    """Crop video segment ``[start_frame, end_frame)`` and encode as browser-ready H.264 MP4."""
    output_video = Path(output_video)
    output_video.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(source_video))
    try:
        if not cap.isOpened():
            return False
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if w <= 0 or h <= 0 or end_frame <= start_frame:
            return False

        tracks_by_frame = _index_tracks(tracks)
        target_ids_set = set(target_mouse_ids or [])
        roles_map = normalize_roles_map(dict(semantic_roles) if semantic_roles else {})
        scale = max(0.6, min(w, h) / 1080.0)
        thick = max(1, int(round(2 * scale)))
        banner_h = int(round(75 * scale))

        raw_tmp = output_video.with_suffix(".tmp_raw.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(raw_tmp), fourcc, fps, (w, h))

        try:
            cur = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if start_frame <= cur < end_frame:
                    if visualize:
                        hud = frame.copy()
                        cv2.rectangle(hud, (0, 0), (w, banner_h), (25, 22, 22), -1)
                        cv2.addWeighted(hud, 0.75, frame, 0.25, 0, frame)

                        is_match = bool(
                            pred_behavior and gt_behavior and pred_behavior.lower() == gt_behavior.lower()
                        )
                        pred_col = (50, 225, 50) if is_match else (40, 100, 240)
                        match_tag = "MATCH (PASS)" if is_match else "MISMATCH"

                        p_name = pred_behavior.upper() if pred_behavior else "NONE"
                        g_name = gt_behavior.upper() if gt_behavior else "UNKNOWN"
                        hud_l1 = (
                            f"PREDICTED: {p_name} [{match_tag}]  |  GT: {g_name}"
                            f"  |  Temporal IoU: {iou * 100:.1f}%"
                        )
                        cv2.putText(
                            frame,
                            hud_l1,
                            (int(20 * scale), int(32 * scale)),
                            cv2.FONT_HERSHEY_DUPLEX,
                            0.75 * scale,
                            pred_col,
                            thick,
                            cv2.LINE_AA,
                        )

                        t_sec = cur / max(fps, 1e-6)
                        targets_str = str(sorted(target_ids_set))
                        hud_l2 = (
                            f"Frame: {cur}/{total_source_frames} ({t_sec:.2f}s)"
                            f"  |  Cropped Segment: [{start_frame} : {end_frame})"
                            f"  |  Targets: {targets_str}"
                        )
                        cv2.putText(
                            frame,
                            hud_l2,
                            (int(20 * scale), int(62 * scale)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55 * scale,
                            (230, 230, 230),
                            max(1, thick - 1),
                            cv2.LINE_AA,
                        )

                        _draw_detection_boxes(
                            frame,
                            tracks_by_frame.get(cur, []),
                            target_ids=target_ids_set,
                            roles_map=roles_map,
                            behavior=gt_behavior or pred_behavior,
                            scale=scale,
                            thick=thick,
                            min_tag_y=banner_h + 4,
                            non_target_role="Conspecific" if gt_behavior in {"Isolation", "Group locomotion", "Static", "Locomotion"} else None,
                        )

                    writer.write(frame)
                cur += 1
                if cur >= end_frame:
                    break
        finally:
            writer.release()

        return transcode_to_h264(raw_tmp, output_video, ffmpeg_exe=ffmpeg_exe)
    except Exception:
        LOGGER.exception("Failed to render cropped video: %s", output_video)
        return False
    finally:
        cap.release()


def render_annotated_original_video(
    source_video: Path,
    output_video: Path,
    fps: float,
    *,
    tracks: Sequence[Mapping[str, Any]] | None = None,
    target_mouse_ids: Sequence[int] | None = None,
    semantic_roles: Mapping[str, Sequence[int]] | None = None,
    behavior: str | None = None,
    show_banner: bool = False,
    ffmpeg_exe: str | None = None,
) -> bool:
    """Render full-length original video with detection bounding boxes and role labels."""
    output_video = Path(output_video)
    output_video.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(source_video))
    try:
        if not cap.isOpened():
            return False
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if w <= 0 or h <= 0 or total_frames <= 0:
            return False

        tracks_by_frame = _index_tracks(tracks)
        target_ids_set = set(target_mouse_ids or [])
        roles_map = normalize_roles_map(dict(semantic_roles) if semantic_roles else {})
        scale = max(0.6, min(w, h) / 1080.0)
        thick = max(1, int(round(2 * scale)))
        banner_h = int(round(75 * scale)) if show_banner else 0

        raw_tmp = output_video.with_suffix(".tmp_raw_orig.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(raw_tmp), fourcc, fps, (w, h))

        try:
            cur = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                if show_banner:
                    hud = frame.copy()
                    cv2.rectangle(hud, (0, 0), (w, banner_h), (25, 22, 22), -1)
                    cv2.addWeighted(hud, 0.75, frame, 0.25, 0, frame)

                    g_name = behavior.upper() if behavior else "UNKNOWN"
                    t_ids_str = str(sorted(target_ids_set))
                    l1_text = f"ORIGINAL FULL-LENGTH MONITORING  |  Ground Truth: {g_name}  |  Targets: {t_ids_str}"
                    cv2.putText(
                        frame,
                        l1_text,
                        (int(20 * scale), int(32 * scale)),
                        cv2.FONT_HERSHEY_DUPLEX,
                        0.72 * scale,
                        (56, 189, 248),
                        thick,
                        cv2.LINE_AA,
                    )

                    t_sec = cur / max(fps, 1e-6)
                    l2_text = (
                        f"Frame: {cur}/{total_frames} ({t_sec:.2f}s)"
                        "  |  Full Arena Detection BBoxes & Target Mouse Roles"
                    )
                    cv2.putText(
                        frame,
                        l2_text,
                        (int(20 * scale), int(62 * scale)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.53 * scale,
                        (230, 230, 230),
                        max(1, thick - 1),
                        cv2.LINE_AA,
                    )

                _draw_detection_boxes(
                    frame,
                    tracks_by_frame.get(cur, []),
                    target_ids=target_ids_set,
                    roles_map=roles_map,
                    behavior=behavior,
                    scale=scale,
                    thick=thick,
                    min_tag_y=(banner_h + 4) if show_banner else 4,
                    non_target_role=None if show_banner else "Conspecific",
                )

                writer.write(frame)
                cur += 1
        finally:
            writer.release()

        return transcode_to_h264(raw_tmp, output_video, ffmpeg_exe=ffmpeg_exe)
    except Exception:
        LOGGER.exception("Failed to render annotated original video: %s", output_video)
        return False
    finally:
        cap.release()


def render_annotated_clip_from_dir(
    clip_dir: Path,
    output_dir: Path,
    ffmpeg_exe: str,
    target_ids: set[int] | None = None,
    roles_map: dict[str, list[int]] | None = None,
    gt_behavior: str | None = None,
) -> bool:
    """Load annotation/tracks from ``clip_dir`` and render ``clip_annotated.mp4``."""
    source_video = clip_dir / "clip.mp4"
    tracks_file = clip_dir / "tracks.json"
    annot_file = clip_dir / "annotation.json"
    if not source_video.is_file():
        return False

    if target_ids is None or roles_map is None or gt_behavior is None:
        if annot_file.is_file():
            try:
                annot_data = json.loads(annot_file.read_text(encoding="utf-8"))
                raw_targets = annot_data.get("target_mouse_ids")
                if raw_targets is None:
                    raw_targets = annot_data.get("mouse_ids") or []
                target_ids = {int(x) for x in raw_targets}
                raw_roles = annot_data.get("roles")
                if raw_roles is None:
                    raw_roles = annot_data.get("participants") or {}
                roles_map = normalize_roles_map(raw_roles)
                gt_behavior = str(annot_data.get("behavior", ""))
            except Exception:
                target_ids = set()
                roles_map = {}
                gt_behavior = ""
        else:
            target_ids = set()
            roles_map = {}
            gt_behavior = ""
    else:
        roles_map = normalize_roles_map(roles_map)

    raw_tracks: list[dict[str, Any]] = []
    if tracks_file.is_file():
        try:
            raw_tracks = json.loads(tracks_file.read_text(encoding="utf-8"))
        except Exception:
            raw_tracks = []

    cap = cv2.VideoCapture(str(source_video))
    if not cap.isOpened():
        return False
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    cap.release()

    final_output = Path(output_dir) / clip_dir.name / "clip_annotated.mp4"
    return render_annotated_original_video(
        source_video,
        final_output,
        fps,
        tracks=raw_tracks,
        target_mouse_ids=sorted(target_ids),
        semantic_roles=roles_map,
        behavior=gt_behavior,
        show_banner=True,
        ffmpeg_exe=ffmpeg_exe,
    )
