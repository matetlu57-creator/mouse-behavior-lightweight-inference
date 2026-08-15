#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reusable writer for the cache consumed by lightweight inference.

This helper intentionally loads only the seven-keypoint YOLO Pose model.  It
does not load or require an OBB model.  Cache records are chunked so a long
video is not accumulated as one Python list in memory.
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

    if model is None:
        from ultralytics import YOLO

        model = YOLO(str(model_path))
    batch_size = max(int(batch_size), 1)
    chunk_frames = max(int(chunk_frames), 1)
    records: list[dict[str, Any]] = []
    frames: list[np.ndarray] = []
    frame_indices: list[int] = []
    frame_index = 0
    chunk_index = 0

    def flush_batch() -> None:
        if not frames:
            return
        results = model.predict(
            frames,
            imgsz=int(imgsz),
            conf=float(conf),
            iou=float(iou),
            max_det=int(max_det),
            device=device,
            verbose=False,
        )
        for index, result in zip(frame_indices, results):
            records.append({"frame": int(index), "pose_detections": _payloads(result)})
        frames.clear()
        frame_indices.clear()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
            frame_indices.append(frame_index)
            frame_index += 1
            if len(frames) >= batch_size:
                flush_batch()
            if len(records) >= chunk_frames:
                _write_chunk(output, chunk_index, records)
                chunk_index += 1
                records.clear()
                LOGGER.info("[pose cache] %d/%d frames", frame_index, total_frames)
        flush_batch()
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
        "next_frame": int(frame_index),
        "chunk_count": int(chunk_index),
        "pose_only": True,
        "obb_used": False,
    }
    (output / "yolo_results_status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    LOGGER.info("[pose cache] completed: %s (%d frames)", output, frame_index)
    return output
