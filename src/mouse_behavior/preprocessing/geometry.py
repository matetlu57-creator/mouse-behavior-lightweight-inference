"""Numerical geometry helpers shared by tracking and pair analysis."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def _finite_point(value: Any) -> bool:
    arr = np.asarray(value, dtype=float).reshape(-1)
    return bool(arr.size >= 2 and np.all(np.isfinite(arr[:2])))


def _unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-9 and np.isfinite(norm) else np.full(2, np.nan)


def _cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity for [..., 2] arrays, with zero for invalid vectors."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    dot = np.sum(a * b, axis=-1)
    denom = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    out = np.zeros_like(dot, dtype=float)
    valid = np.isfinite(dot) & np.isfinite(denom) & (denom > 1e-9)
    out[valid] = np.clip(dot[valid] / denom[valid], -1.0, 1.0)
    return out


def _angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    cosine = _cosine(a, b)
    valid = np.all(np.isfinite(a), axis=-1) & np.all(np.isfinite(b), axis=-1)
    out = np.zeros(cosine.shape, dtype=float)
    out[valid] = np.degrees(np.arccos(np.clip(cosine[valid], -1.0, 1.0)))
    return out


def _weighted_mean(
    points: np.ndarray, confidence: np.ndarray, indices: Sequence[int]
) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    confidence = np.asarray(confidence, dtype=float).reshape(-1)
    valid_indices = [
        int(index)
        for index in indices
        if int(index) < len(points)
        and int(index) < len(confidence)
        and np.all(np.isfinite(points[int(index)]))
        and np.isfinite(confidence[int(index)])
        and float(confidence[int(index)]) >= 0.10
    ]
    if not valid_indices:
        return np.full(2, np.nan, dtype=float)
    values = points[valid_indices]
    weights = np.maximum(confidence[valid_indices], 0.01)
    return np.average(values, axis=0, weights=weights).astype(float)


def _box_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    if a.size < 4 or b.size < 4 or not np.all(np.isfinite(a[:4])) or not np.all(np.isfinite(b[:4])):
        return 0.0
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - inter
    return float(inter / union) if union > 1e-9 else 0.0
