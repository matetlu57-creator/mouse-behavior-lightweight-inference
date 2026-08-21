"""Persistence and visual audit artifacts for learned arena boundaries."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Optional

import cv2
import numpy as np

from ..preprocessing.arena_learning import (
    SCHEMA_VERSION,
    ArenaBoundaryResult,
    _coerce_polygon,
    _finite_float,
    _polygon_lists,
    _same_path,
    _video_fingerprint,
)


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


__all__ = [
    "load_boundary_json",
    "_atomic_json_write",
    "_write_png",
    "save_boundary_overlay_frame",
    "save_boundary_artifacts",
]
