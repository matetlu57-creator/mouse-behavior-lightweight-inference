#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build an Ultralytics-tracked Pose cache for behavior inference.

This helper intentionally delegates visual tracking to ``YOLO.track``.  It
stores the tracker-produced ID beside each seven-keypoint pose and never
implements an identity matcher of its own.  Cache records are chunked so a
long video is not accumulated as one Python list in memory.
"""

from __future__ import annotations

import gzip
import json
import logging
import pickle
from pathlib import Path
from typing import Any

import cv2
import numpy as np


LOGGER = logging.getLogger(__name__)


KEYPOINTS = 7
VIDEO_EXTENSIONS = {".mov", ".mp4", ".avi", ".mkv", ".wmv", ".m4v"}


def _payloads(result: Any) -> list[dict[str, Any]]:
    if result.keypoints is None or result.boxes is None:
        return []
    points = result.keypoints.xy.detach().cpu().numpy()
    confidence = result.keypoints.conf.detach().cpu().numpy()
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    box_conf = result.boxes.conf.detach().cpu().numpy()
    track_ids_tensor = getattr(result.boxes, "id", None)
    if len(points) and track_ids_tensor is None:
        # YOLO.track() may leave raw detections in a frame when ByteTrack has
        # no confirmed track for them. They cannot enter a trajectory cache
        # without a tracker-owned ID, so omit them instead of inventing one.
        return []
    track_ids = (
        track_ids_tensor.detach().cpu().numpy().reshape(-1)
        if track_ids_tensor is not None
        else np.asarray([], dtype=np.int64)
    )
    if not (len(points) == len(boxes) == len(box_conf) == len(track_ids)):
        raise RuntimeError("Ultralytics track result has inconsistent pose/box/ID lengths")
    output: list[dict[str, Any]] = []
    for index in range(len(points)):
        if points[index].shape != (KEYPOINTS, 2):
            continue
        output.append(
            {
                "keypoints_px": points[index].astype(np.float32),
                "keypoint_conf": confidence[index].astype(np.float32),
                "bbox_xyxy": boxes[index].astype(np.float32),
                "box_conf": float(box_conf[index]),
                "pose_quality": float(np.nanmean(confidence[index])),
                "visual_track_id": int(track_ids[index]),
            }
        )
    return output


def _write_chunk(cache_dir: Path, chunk_index: int, records: list[dict[str, Any]]) -> None:
    path = cache_dir / f"yolo_results.000000.{chunk_index:06d}.pkl.gz"
    with gzip.open(path, "wb") as handle:
        pickle.dump(records, handle, protocol=pickle.HIGHEST_PROTOCOL)


def build_cache(
    video: Path,
    output: Path,
    model_path: Path,
    *,
    batch_size: int = 4,
    chunk_frames: int = 300,
    imgsz: int = 768,
    conf: float = 0.15,
    iou: float = 0.50,
    max_det: int = 100,
    device: str | int = 0,
    tracker: str = "bytetrack.yaml",
    model: Any | None = None,
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise RuntimeError(f"视频帧数无效: {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        cap.release()
        raise RuntimeError(f"视频 FPS 无效: {video} ({fps})")

    if model is None:
        from ultralytics import YOLO

        model = YOLO(str(model_path))
    # ``batch_size`` remains accepted for CLI compatibility. Trackers are
    # stateful and frames are deliberately submitted in temporal order, one
    # at a time; detector batching must not split or reset tracker state.
    batch_size = max(int(batch_size), 1)
    chunk_frames = max(int(chunk_frames), 1)
    records: list[dict[str, Any]] = []
    frame_index = 0
    chunk_index = 0
    untracked_detection_count = 0
    untracked_detection_frame_count = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            results = model.track(
                frame,
                persist=True,
                tracker=str(tracker),
                imgsz=int(imgsz),
                conf=float(conf),
                iou=float(iou),
                max_det=int(max_det),
                device=device,
                verbose=False,
            )
            if not isinstance(results, (list, tuple)) or len(results) != 1:
                raise RuntimeError("Ultralytics model.track() must return one result per frame")
            result = results[0]
            if (
                result.keypoints is not None
                and result.boxes is not None
                and getattr(result.boxes, "id", None) is None
            ):
                untracked_count = len(result.keypoints.xy.detach().cpu().numpy())
                if untracked_count:
                    untracked_detection_count += untracked_count
                    untracked_detection_frame_count += 1
            records.append(
                {
                    "frame": int(frame_index),
                    "timestamp_seconds": float(frame_index / fps),
                    "pose_detections": _payloads(result),
                }
            )
            frame_index += 1
            if len(records) >= chunk_frames:
                _write_chunk(output, chunk_index, records)
                chunk_index += 1
                records.clear()
                LOGGER.info("[pose cache] %d/%d frames", frame_index, total_frames)
        if records:
            _write_chunk(output, chunk_index, records)
            chunk_index += 1
    finally:
        cap.release()

    status = {
        "status": "complete",
        "video": str(video.resolve()),
        "model": str(model_path.resolve()),
        "total_frames": int(total_frames),
        "fps": float(fps),
        "next_frame": int(frame_index),
        "chunk_count": int(chunk_index),
        "pose_only": True,
        "obb_used": False,
        "trajectory_schema_version": 1,
        "tracking_backend": "ultralytics",
        "tracker": str(tracker),
        "identity_semantics": "provisional_visual_track_id",
        "sequential_tracking": True,
        "requested_batch_size": int(batch_size),
        "untracked_detection_count": int(untracked_detection_count),
        "untracked_detection_frame_count": int(untracked_detection_frame_count),
    }
    (output / "yolo_results_status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    LOGGER.info("[pose cache] completed: %s (%d frames)", output, frame_index)
    if untracked_detection_count:
        LOGGER.warning(
            "[pose cache] omitted %d untracked detections from %d frames; "
            "these detections had no Ultralytics visual_track_id",
            untracked_detection_count,
            untracked_detection_frame_count,
        )
    return output
