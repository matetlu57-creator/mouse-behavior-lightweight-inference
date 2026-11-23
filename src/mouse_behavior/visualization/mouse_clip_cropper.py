"""Frame-accurate H.264 clip cropping and manual behavior package export.

All exports read ``source_dir`` strictly read-only and write verified H.264
(``libx264``, ``yuv420p``, ``+faststart``) clip packages into
``<output_dir>/collected_clips/<safe_behavior>/<export_id>/`` via staging
directories and atomic moves.
"""
from __future__ import annotations

import copy
import logging
import math
import os
import shutil
import subprocess
import tempfile
import threading
import uuid
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import cv2
import numpy as np

from mouse_behavior.evaluation.review_transaction import safe_path, write_json
from mouse_behavior.visualization.clip_overlay_renderer import (
    _draw_detection_boxes,
    normalize_roles_map,
)
from mouse_behavior.visualization.mouse_clip_review_store import (
    MouseClipReviewStore,
    _is_strict_int,
    sanitize_dir_segment,
)

LOGGER = logging.getLogger(__name__)

_DEFAULT_KEYPOINT_NAMES = [
    "nose",
    "right_ear",
    "left_ear",
    "base_of_neck",
    "left_hip",
    "right_hip",
    "base_of_tail",
]
_DEFAULT_SKELETON_EDGES = [
    [0, 1],
    [0, 2],
    [1, 3],
    [2, 3],
    [3, 4],
    [3, 5],
    [4, 6],
    [5, 6],
]
_MAX_PADDING_FRAMES = 600
_EXPORT_DIR_LOCK = threading.Lock()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_ffmpeg_executable() -> str | None:
    """Resolve FFmpeg binary, preferring ``imageio_ffmpeg`` then system PATH."""
    try:
        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and Path(exe).is_file():
            return exe
    except Exception:
        pass
    return shutil.which("ffmpeg")


def is_ffmpeg_available() -> bool:
    return get_ffmpeg_executable() is not None


def _open_capture(video_path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(str(video_path))
    return cap


def _decode_fourcc(cap: cv2.VideoCapture) -> str:
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
    chars = []
    for i in range(4):
        ch = (fourcc_int >> (8 * i)) & 0xFF
        if 32 <= ch < 127:
            chars.append(chr(ch))
    return "".join(chars)


def _verify_exported_video(
    video_path: Path,
    expected_frames: int,
    expected_fps: float,
) -> dict[str, Any]:
    """Verify exported H.264 clip decodes exact expected frame count and dimensions."""
    if not video_path.is_file():
        raise RuntimeError(f"exported video missing: {video_path}")
    size = video_path.stat().st_size
    if size <= 0:
        raise RuntimeError(f"exported video is empty: {video_path}")

    cap = _open_capture(video_path)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"exported video cannot be opened: {video_path}")
        actual_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if not math.isfinite(actual_fps) or actual_fps <= 0:
            raise RuntimeError(f"exported video has invalid FPS: {actual_fps}")
        if not math.isclose(actual_fps, expected_fps, rel_tol=1e-2, abs_tol=0.05):
            raise RuntimeError(
                f"exported FPS {actual_fps:.4f} differs from source FPS {expected_fps:.4f}"
            )
        codec = _decode_fourcc(cap)
        decoded = 0
        width = 0
        height = 0
        while cap.grab():
            cur_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            cur_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            if cur_w <= 0 or cur_h <= 0:
                raise RuntimeError("decoded frame has invalid dimensions")
            if decoded == 0:
                width, height = cur_w, cur_h
            elif (cur_w, cur_h) != (width, height):
                raise RuntimeError("decoded frame dimensions changed mid-clip")
            decoded += 1
            if decoded == 1 or decoded == expected_frames:
                ok, frame = cap.retrieve()
                if not ok or frame is None or frame.shape != (height, width, 3):
                    raise RuntimeError("failed to retrieve boundary frame from exported clip")
        if decoded != expected_frames:
            raise RuntimeError(
                f"exported video frame count mismatch: got {decoded}, expected {expected_frames}"
            )
        return {
            "frame_count": decoded,
            "fps": round(actual_fps, 6),
            "duration_s": round(decoded / actual_fps, 4),
            "width": width,
            "height": height,
            "bytes": size,
            "codec": codec or "h264",
        }
    finally:
        cap.release()


def _normalize_participants(raw_participants: Any) -> tuple[list[dict[str, Any]], dict[str, list[int]]]:
    """Normalize participants to canonical ``F:\\mouse`` list and ``role_name -> [ids]`` map."""
    participants: list[dict[str, Any]] = []
    roles_map: dict[str, list[int]] = {}
    if not raw_participants:
        return participants, roles_map

    if isinstance(raw_participants, Mapping):
        raw_roles = normalize_roles_map(dict(raw_participants))
        for role_name, tids in raw_roles.items():
            if not tids:
                continue
            role_key = role_name.lower().replace(" ", "_")
            participants.append(
                {
                    "role_key": role_key,
                    "role_name": role_name,
                    "track_ids": tids,
                }
            )
            roles_map[role_name] = list(tids)
        return participants, roles_map

    if isinstance(raw_participants, Sequence) and not isinstance(raw_participants, (str, bytes)):
        for item in raw_participants:
            if not isinstance(item, Mapping):
                continue
            role_name = str(item.get("role_name") or item.get("role_key") or "").strip()
            role_key = str(item.get("role_key") or role_name.lower().replace(" ", "_")).strip()
            raw_tids = item.get("track_ids")
            if isinstance(raw_tids, Sequence) and not isinstance(raw_tids, (str, bytes)):
                tids = [int(t) for t in raw_tids if _is_strict_int(t)]
            elif _is_strict_int(raw_tids):
                tids = [int(raw_tids)]
            else:
                tids = []
            if role_name and tids:
                participants.append(
                    {
                        "role_key": role_key or role_name.lower(),
                        "role_name": role_name,
                        "track_ids": tids,
                    }
                )
                roles_map[role_name] = list(tids)
        return participants, roles_map

    raise ValueError("participants must be a list of role objects or a role mapping")


def _draw_collection_overlay(
    frame: np.ndarray,
    *,
    source_frame: int,
    local_frame: int,
    total_local_frames: int,
    in_active_behavior: bool,
    fps: float,
    behavior: str,
    detections: Sequence[Mapping[str, Any]],
    target_ids: set[int],
    roles_map: Mapping[str, Sequence[int]],
) -> np.ndarray:
    """Draw measured detection boxes, target highlights, and concise HUD banner."""
    h, w = frame.shape[:2]
    scale = max(0.6, min(w, h) / 1080.0)
    thick = max(1, int(round(2 * scale)))
    banner_h = int(round(68 * scale))

    hud = frame.copy()
    cv2.rectangle(hud, (0, 0), (w, banner_h), (18, 24, 38), -1)
    cv2.addWeighted(hud, 0.78, frame, 0.22, 0, frame)

    status_tag = "ACTIVE" if in_active_behavior else "CONTEXT PADDING"
    status_color = (80, 235, 130) if in_active_behavior else (80, 195, 255)
    targets_str = ",".join(str(t) for t in sorted(target_ids)) if target_ids else "-"
    l1 = f"MANUAL COLLECTION: {behavior} [{status_tag}]  |  Targets: [{targets_str}]"
    cv2.putText(
        frame,
        l1,
        (int(18 * scale), int(28 * scale)),
        cv2.FONT_HERSHEY_DUPLEX,
        0.68 * scale,
        status_color,
        thick,
        cv2.LINE_AA,
    )

    visible_target_ids = {
        int(d.get("track_id", -1))
        for d in detections
        if _is_strict_int(d.get("track_id")) and int(d["track_id"]) in target_ids
    }
    missing = sorted(target_ids - visible_target_ids)
    missing_txt = f"  |  Missing in frame: {missing}" if missing else ""
    l2 = (
        f"Clip Frame: {local_frame}/{max(0, total_local_frames - 1)}"
        f" ({local_frame / max(fps, 1e-6):.2f}s)  |  Source Frame: {source_frame}{missing_txt}"
    )
    cv2.putText(
        frame,
        l2,
        (int(18 * scale), int(56 * scale)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52 * scale,
        (230, 236, 244),
        max(1, thick - 1),
        cv2.LINE_AA,
    )

    # Highlight measured detections only; never fabricate boxes for missing IDs.
    _draw_detection_boxes(
        frame,
        detections,
        target_ids=target_ids,
        roles_map=roles_map,
        behavior=behavior,
        scale=scale,
        thick=thick,
        min_tag_y=banner_h + 4,
    )
    return frame


def _encode_segment_h264(
    source_video: Path,
    output_video: Path,
    clip_start: int,
    clip_end: int,
    fps: float,
    *,
    ffmpeg_exe: str,
    overlay_spec: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Sequentially decode ``[clip_start, clip_end)`` and encode browser-ready H.264 MP4."""
    expected_frames = clip_end - clip_start
    if expected_frames < 1:
        raise ValueError("clip_end must be greater than clip_start")

    cap = _open_capture(source_video)
    process: subprocess.Popen[bytes] | None = None
    with tempfile.TemporaryFile() as stderr_file:
        try:
            if not cap.isOpened():
                raise RuntimeError(f"cannot open source video: {source_video}")
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            if width <= 0 or height <= 0:
                raise RuntimeError(f"source video has invalid dimensions {width}x{height}")

            command = [
                ffmpeg_exe,
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s",
                f"{width}x{height}",
                "-r",
                f"{fps:.6f}",
                "-i",
                "pipe:0",
            ]
            if width % 2 != 0 or height % 2 != 0:
                command += ["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"]
            command += [
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "21",
                "-threads",
                "1",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(output_video),
            ]
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=stderr_file,
            )
            assert process.stdin is not None

            tracks_by_frame: dict[int, list[Mapping[str, Any]]] = (
                overlay_spec["tracks_by_frame"] if overlay_spec is not None else {}
            )
            target_ids: set[int] = overlay_spec["target_ids"] if overlay_spec is not None else set()
            roles_map: Mapping[str, Sequence[int]] = (
                overlay_spec["roles_map"] if overlay_spec is not None else {}
            )
            behavior: str = overlay_spec["behavior"] if overlay_spec is not None else ""
            active_start: int = overlay_spec["active_start"] if overlay_spec is not None else clip_start
            active_end: int = overlay_spec["active_end"] if overlay_spec is not None else clip_end

            cur = 0
            written = 0
            while cur < clip_start:
                if not cap.grab():
                    raise RuntimeError(
                        f"source video ended at frame {cur} before clip_start={clip_start}"
                    )
                cur += 1

            while cur < clip_end:
                ok, frame = cap.read()
                if not ok or frame is None:
                    raise RuntimeError(
                        f"source video ended at frame {cur} before clip_end={clip_end}"
                    )
                if frame.shape[:2] != (height, width):
                    raise RuntimeError("source frame dimensions changed during cropping")
                if overlay_spec is not None:
                    frame = _draw_collection_overlay(
                        frame,
                        source_frame=cur,
                        local_frame=written,
                        total_local_frames=expected_frames,
                        in_active_behavior=(active_start <= cur < active_end),
                        fps=fps,
                        behavior=behavior,
                        detections=tracks_by_frame.get(cur, ()),
                        target_ids=target_ids,
                        roles_map=roles_map,
                    )
                process.stdin.write(frame.tobytes())
                written += 1
                cur += 1

            process.stdin.close()
            returncode = process.wait(timeout=300)
            if returncode != 0:
                stderr_file.seek(0)
                err_txt = stderr_file.read().decode(errors="replace")[-3000:]
                raise RuntimeError(f"ffmpeg libx264 encoding failed ({returncode}): {err_txt}")
        except BaseException as exc:
            if process is not None:
                if process.stdin and not process.stdin.closed:
                    try:
                        process.stdin.close()
                    except OSError:
                        pass
                if process.poll() is None:
                    process.kill()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    pass
            if isinstance(exc, BrokenPipeError):
                stderr_file.seek(0)
                err_txt = stderr_file.read().decode(errors="replace")[-3000:]
                raise RuntimeError(f"ffmpeg rejected frames: {err_txt}") from exc
            raise
        finally:
            cap.release()

    return _verify_exported_video(output_video, expected_frames, fps)


def _build_rebased_tracks(
    source_frames: Sequence[Mapping[str, Any]],
    clip_start: int,
    clip_end: int,
    fps: float,
) -> list[dict[str, Any]]:
    by_frame = {int(row["frame"]): row for row in source_frames if _is_strict_int(row.get("frame"))}
    rebased: list[dict[str, Any]] = []
    for idx, src_frame in enumerate(range(clip_start, clip_end)):
        row = by_frame.get(src_frame)
        dets = copy.deepcopy(list(row.get("detections", []))) if row is not None else []
        rebased.append(
            {
                "frame": idx,
                "time": round(idx / fps, 6),
                "source_frame": src_frame,
                "detections": dets,
            }
        )
    return rebased


def _build_collection_metadata(
    source_metadata: Mapping[str, Any] | None,
    *,
    rec: Mapping[str, Any],
    export_id: str,
    clip_start: int,
    clip_end: int,
    start_frame: int,
    end_frame: int,
    padding_frames: int,
    include_overlay: bool,
    verification: Mapping[str, Any],
    created_at: str,
) -> dict[str, Any]:
    fps = float(rec["fps"])
    out_frame_count = clip_end - clip_start

    src_meta = copy.deepcopy(dict(source_metadata)) if isinstance(source_metadata, Mapping) else {}
    src_tracks = src_meta.get("tracks")
    tracks_block = (
        dict(src_tracks)
        if isinstance(src_tracks, Mapping)
        else {
            "frame_origin": 0,
            "time_origin": 0.0,
            "coordinate_system": "clip_pixels",
        }
    )
    tracks_block["frame_origin"] = 0
    tracks_block["time_origin"] = 0.0

    src_pose = src_meta.get("pose")
    pose_block = (
        copy.deepcopy(dict(src_pose))
        if isinstance(src_pose, Mapping)
        else {
            "keypoint_names": list(_DEFAULT_KEYPOINT_NAMES),
            "skeleton_edges": copy.deepcopy(_DEFAULT_SKELETON_EDGES),
        }
    )

    old_source = src_meta.get("source") if isinstance(src_meta.get("source"), Mapping) else {}
    old_clip_src = old_source.get("clip") if isinstance(old_source.get("clip"), Mapping) else {}
    base_origin = (
        int(old_clip_src["start_frame"])
        if _is_strict_int(old_clip_src.get("start_frame"))
        else 0
    )
    orig_clip_start = base_origin + clip_start
    orig_clip_end = base_origin + clip_end - 1
    orig_ann_start = base_origin + start_frame
    orig_ann_end = base_origin + end_frame - 1

    crop_region = (
        copy.deepcopy(dict(old_source["crop_region"]))
        if isinstance(old_source.get("crop_region"), Mapping)
        else {
            "x": 0,
            "y": 0,
            "w": int(verification["width"]),
            "h": int(verification["height"]),
        }
    )

    source_block: dict[str, Any] = {
        "video_filename": str(old_source.get("video_filename") or rec.get("source_video") or ""),
        "clip": {
            "start_frame": orig_clip_start,
            "end_frame": orig_clip_end,
            "start_time": round(orig_clip_start / fps, 6),
            "end_time": round((orig_clip_end + 1) / fps, 6),
        },
        "annotation": {
            "start_frame": orig_ann_start,
            "end_frame": orig_ann_end,
            "start_time": round(orig_ann_start / fps, 6),
            "end_time": round((orig_ann_end + 1) / fps, 6),
        },
        "crop_region": crop_region,
    }

    metadata: dict[str, Any] = {
        "schema_version": "2.0",
        "clip": {
            "filename": "clip.mp4",
            "fps": fps,
            "width": int(verification["width"]),
            "height": int(verification["height"]),
            "frame_count": out_frame_count,
            "decoded_duration_s": verification["duration_s"],
            "encoded_fps": verification["fps"],
        },
        "tracks": tracks_block,
        "pose": pose_block,
        "source": source_block,
    }
    if isinstance(src_meta.get("distance_scale"), Mapping):
        metadata["distance_scale"] = copy.deepcopy(dict(src_meta["distance_scale"]))

    metadata["manual_collection"] = {
        "export_id": export_id,
        "source_clip_uid": rec["clip_uid"],
        "source_behavior": rec["behavior"],
        "source_clip_id": rec["clip_id"],
        "selected_start_frame": start_frame,
        "selected_end_frame": end_frame,
        "clip_start_frame": clip_start,
        "clip_end_frame": clip_end,
        "padding_frames": padding_frames,
        "include_overlay": include_overlay,
        "video_encoding": "H.264/yuv420p/+faststart",
        "created_at": created_at,
    }
    return metadata


def export_collected_clip(
    store: MouseClipReviewStore,
    request: Mapping[str, Any],
    *,
    ffmpeg_exe: str | None = None,
) -> dict[str, Any]:
    """Crop a behavior segment from read-only ``source_dir`` into ``output_dir/collected_clips``."""
    if not isinstance(request, Mapping):
        raise ValueError("collection request must be a JSON object")

    clip_uid = request.get("clip_uid")
    if not isinstance(clip_uid, str) or not clip_uid.strip():
        raise ValueError("clip_uid is required")
    clip_uid = clip_uid.strip()

    rec = store.get_record(clip_uid)
    source_video = store.get_video_path(clip_uid)
    source_frame_count = int(rec["frame_count"])
    fps = float(rec["fps"])
    if source_frame_count <= 0 or fps <= 0:
        raise ValueError("source clip has invalid frame_count or fps")

    start_frame = request.get("start_frame")
    end_frame = request.get("end_frame")
    if not (_is_strict_int(start_frame) and _is_strict_int(end_frame)):
        raise ValueError("start_frame and end_frame must be integers")
    if not (0 <= start_frame < end_frame <= source_frame_count):
        raise ValueError(
            f"invalid half-open range [{start_frame}, {end_frame}) for frame_count={source_frame_count}"
        )

    raw_behavior = request.get("behavior")
    if raw_behavior is None or not str(raw_behavior).strip():
        behavior = str(rec["behavior"])
    else:
        behavior = str(raw_behavior).strip()
    safe_behavior = sanitize_dir_segment(behavior)

    raw_mouse_ids = request.get("mouse_ids")
    if raw_mouse_ids is None:
        mouse_ids = list(rec["target_mouse_ids"])
    else:
        if not isinstance(raw_mouse_ids, list) or not all(_is_strict_int(m) for m in raw_mouse_ids):
            raise ValueError("mouse_ids must be a list of integers")
        mouse_ids = []
        for mid in raw_mouse_ids:
            mid_i = int(mid)
            if mid_i not in mouse_ids:
                mouse_ids.append(mid_i)

    participants, roles_map = _normalize_participants(request.get("participants"))

    padding_frames = request.get("padding_frames", 0)
    if not _is_strict_int(padding_frames) or not (0 <= padding_frames <= _MAX_PADDING_FRAMES):
        raise ValueError(f"padding_frames must be an integer in [0, {_MAX_PADDING_FRAMES}]")

    include_overlay = bool(request.get("include_overlay", True))
    overwrite = bool(request.get("overwrite", False))

    confidence = request.get("confidence", "certain")
    if not isinstance(confidence, str) or not confidence.strip():
        confidence = "certain"
    else:
        confidence = confidence.strip()[:64]

    raw_note = request.get("note", "")
    if raw_note is None:
        raw_note = ""
    if not isinstance(raw_note, str):
        raise ValueError("note must be a string")
    note = raw_note[:4000]

    exe = ffmpeg_exe or get_ffmpeg_executable()
    if not exe:
        raise RuntimeError("FFmpeg executable not available (install imageio-ffmpeg or ffmpeg)")

    # Fail fast if collections_manifest.json is corrupt before spending CPU on encoding.
    store.ensure_collections_writable()

    clip_start = max(0, start_frame - padding_frames)
    clip_end = min(source_frame_count, end_frame + padding_frames)
    out_frame_count = clip_end - clip_start
    ann_rel_start = start_frame - clip_start
    ann_rel_end = end_frame - 1 - clip_start

    safe_clip_id = sanitize_dir_segment(rec["clip_id"])
    pad_tag = f"_p{padding_frames}" if padding_frames > 0 else ""
    base_export_id = f"{safe_behavior}__{safe_clip_id}__f{start_frame:06d}-{end_frame:06d}{pad_tag}"
    behavior_out_dir = store.collected_clips_dir / safe_behavior
    behavior_out_dir.mkdir(parents=True, exist_ok=True)
    safe_path(behavior_out_dir, [store.collected_clips_dir])

    with _EXPORT_DIR_LOCK:
        export_id = base_export_id
        dest_dir = behavior_out_dir / export_id
        safe_path(dest_dir, [store.collected_clips_dir])
        if dest_dir.exists() and not overwrite:
            raise FileExistsError(
                f"export package already exists: {export_id} (set overwrite=True to replace)"
            )

    _, source_metadata = store.read_clip_json_files(clip_uid)
    source_frames, _ = store.load_tracks_for_clip(clip_uid)
    tracks_by_frame = {
        int(row["frame"]): list(row.get("detections", []))
        for row in source_frames
        if _is_strict_int(row.get("frame"))
    }

    stage_dir = behavior_out_dir / f".stage-{uuid.uuid4().hex}"
    stage_dir.mkdir(parents=True, exist_ok=False)
    created_at = _utc_now_iso()

    try:
        raw_mp4_path = stage_dir / "clip.mp4"
        verification = _encode_segment_h264(
            source_video,
            raw_mp4_path,
            clip_start,
            clip_end,
            fps,
            ffmpeg_exe=exe,
            overlay_spec=None,
        )

        files_written = ["clip.mp4", "annotation.json", "metadata.json", "tracks.json"]

        overlay_verification: dict[str, Any] | None = None
        if include_overlay:
            overlay_mp4_path = stage_dir / "clip_overlay.mp4"
            overlay_verification = _encode_segment_h264(
                source_video,
                overlay_mp4_path,
                clip_start,
                clip_end,
                fps,
                ffmpeg_exe=exe,
                overlay_spec={
                    "tracks_by_frame": tracks_by_frame,
                    "target_ids": set(mouse_ids),
                    "roles_map": roles_map,
                    "behavior": behavior,
                    "active_start": start_frame,
                    "active_end": end_frame,
                },
            )
            files_written.append("clip_overlay.mp4")

        rebased_tracks = _build_rebased_tracks(source_frames, clip_start, clip_end, fps)
        write_json(stage_dir / "tracks.json", rebased_tracks)

        annotation_payload: dict[str, Any] = {
            "behavior": behavior,
            "mouse_ids": mouse_ids,
            "participants": participants,
            "confidence": confidence,
            "frame_range": {
                "start": ann_rel_start,
                "end": ann_rel_end,
            },
            "time_range": {
                "start": round(ann_rel_start / fps, 6),
                "end": round((ann_rel_end + 1) / fps, 6),
            },
            "collection_provenance": {
                "export_id": export_id,
                "source_clip_uid": clip_uid,
                "source_behavior": rec["behavior"],
                "source_clip_id": rec["clip_id"],
                "source_clip_dir": str(rec["clip_dir"]),
                "selected_half_open": [start_frame, end_frame],
                "exported_clip_half_open": [clip_start, clip_end],
                "padding_frames": padding_frames,
                "note": note,
                "created_at": created_at,
            },
        }
        write_json(stage_dir / "annotation.json", annotation_payload)

        metadata_payload = _build_collection_metadata(
            source_metadata,
            rec=rec,
            export_id=export_id,
            clip_start=clip_start,
            clip_end=clip_end,
            start_frame=start_frame,
            end_frame=end_frame,
            padding_frames=padding_frames,
            include_overlay=include_overlay,
            verification=verification,
            created_at=created_at,
        )
        write_json(stage_dir / "metadata.json", metadata_payload)

        with _EXPORT_DIR_LOCK:
            backup_dir: Path | None = None
            if dest_dir.exists():
                if not overwrite:
                    raise FileExistsError(
                        f"export package already exists: {export_id} (set overwrite=True to replace)"
                    )
                backup_dir = behavior_out_dir / f".old-{uuid.uuid4().hex}"
                os.replace(dest_dir, backup_dir)
            try:
                os.replace(stage_dir, dest_dir)
            except BaseException:
                if backup_dir is not None and backup_dir.exists() and not dest_dir.exists():
                    os.replace(backup_dir, dest_dir)
                raise
            if backup_dir is not None and backup_dir.exists():
                shutil.rmtree(backup_dir, ignore_errors=True)
    except BaseException:
        if stage_dir.exists():
            shutil.rmtree(stage_dir, ignore_errors=True)
        raise

    quoted_id = quote(export_id, safe="")
    collection_record: dict[str, Any] = {
        "export_id": export_id,
        "clip_uid": clip_uid,
        "source_behavior": rec["behavior"],
        "source_clip_id": rec["clip_id"],
        "behavior": behavior,
        "mouse_ids": mouse_ids,
        "participants": participants,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "clip_start_frame": clip_start,
        "clip_end_frame": clip_end,
        "padding_frames": padding_frames,
        "frame_count": out_frame_count,
        "duration_s": round(out_frame_count / fps, 4),
        "fps": fps,
        "output_dir": str(dest_dir.resolve()),
        "files": files_written,
        "verification": {
            "raw": verification,
            **({"overlay": overlay_verification} if overlay_verification is not None else {}),
            "frame_count": verification["frame_count"],
            "fps": verification["fps"],
            "width": verification["width"],
            "height": verification["height"],
            "codec": verification["codec"],
        },
        "raw_video_url": f"/api/exported-video/{quoted_id}?variant=raw",
        "overlay_video_url": (
            f"/api/exported-video/{quoted_id}?variant=overlay" if include_overlay else None
        ),
        "note": note,
        "created_at": created_at,
    }
    store.record_collection(collection_record)
    return collection_record


class BatchCollectionManager:
    """Bounded thread-pool runner for batch clip collection jobs."""

    def __init__(self, store: MouseClipReviewStore, *, max_workers: int = 2) -> None:
        self._store = store
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, int(max_workers)),
            thread_name_prefix="clip-collect",
        )
        self._lock = threading.Lock()
        self._jobs: dict[str, dict[str, Any]] = {}

    def submit(self, items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not isinstance(items, Sequence) or isinstance(items, (str, bytes)) or len(items) == 0:
            raise ValueError("items must be a non-empty list of collection requests")
        if len(items) > 200:
            raise ValueError("batch size exceeds maximum of 200 items")
        normalized_items = []
        for idx, item in enumerate(items):
            if not isinstance(item, Mapping):
                raise ValueError(f"items[{idx}] must be a JSON object")
            normalized_items.append(dict(item))

        self._store.ensure_collections_writable()
        job_id = f"batch_{uuid.uuid4().hex[:12]}"
        job_state: dict[str, Any] = {
            "job_id": job_id,
            "status": "queued",
            "total": len(normalized_items),
            "completed": 0,
            "failed": 0,
            "results": [],
            "errors": [],
            "created_at": _utc_now_iso(),
            "updated_at": _utc_now_iso(),
        }
        with self._lock:
            self._jobs[job_id] = job_state

        self._executor.submit(self._run_job, job_id, normalized_items)
        return {"status": "accepted", "job_id": job_id, "total": len(normalized_items)}

    def _run_job(self, job_id: str, items: list[dict[str, Any]]) -> None:
        with self._lock:
            self._jobs[job_id]["status"] = "running"
            self._jobs[job_id]["updated_at"] = _utc_now_iso()

        for idx, item in enumerate(items):
            try:
                result = export_collected_clip(self._store, item)
                with self._lock:
                    job = self._jobs[job_id]
                    job["results"].append(result)
                    job["completed"] += 1
                    job["updated_at"] = _utc_now_iso()
            except Exception as exc:
                LOGGER.warning("Batch collection item %d failed: %s", idx, exc)
                with self._lock:
                    job = self._jobs[job_id]
                    job["errors"].append(
                        {
                            "index": idx,
                            "clip_uid": item.get("clip_uid"),
                            "error": str(exc),
                        }
                    )
                    job["failed"] += 1
                    job["updated_at"] = _utc_now_iso()

        with self._lock:
            job = self._jobs[job_id]
            if job["failed"] == 0:
                job["status"] = "completed"
            elif job["completed"] == 0:
                job["status"] = "failed"
            else:
                job["status"] = "completed_with_errors"
            job["updated_at"] = _utc_now_iso()

    def get_job(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(f"unknown batch job_id: {job_id}")
            return copy.deepcopy(self._jobs[job_id])

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=False)
