"""Shared spatial and temporal bounds for Static bouts."""

from __future__ import annotations

import math

import numpy as np


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    changes = np.diff(np.r_[False, mask, False].astype(np.int8))
    return list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))


def bridge_gaps_outside_static(
    moving: np.ndarray,
    stationary: np.ndarray,
    *,
    max_gap_frames: int,
) -> np.ndarray:
    """Recover short movement gaps without overlapping a Static observation."""

    result = np.asarray(moving, dtype=bool).copy()
    static = np.asarray(stationary, dtype=bool)
    if result.ndim != 2 or static.shape != result.shape:
        raise ValueError("moving and stationary must have matching (frames, mice) shapes")
    gap_limit = max(int(max_gap_frames), 0)
    for mouse in range(result.shape[1]):
        segments = _runs(result[:, mouse])
        for (_, left_end), (right_start, _) in zip(segments, segments[1:]):
            if (0 < right_start - left_end <= gap_limit
                    and not static[left_end:right_start, mouse].any()):
                result[left_end:right_start, mouse] = True
    return result


def bounded_stationary_mask(
    stationary: np.ndarray,
    centers_cm: np.ndarray,
    valid: np.ndarray,
    *,
    fps: float,
    max_gap_s: float,
    max_transient_displacement_cm: float,
    max_center_radius_cm: float,
) -> np.ndarray:
    """Bridge brief body-center excursions only inside a bounded Static bout.

    The input mask is the adapter's low-speed evidence. A gap can be included
    only when it is fully observed, no longer than ``max_gap_s``, and the
    complete candidate bout stays within both the transient-displacement and
    center-radius bounds. Unstable bouts are left out of Static so that the
    caller can retain them as locomotion or Unknown evidence.
    """

    base = np.asarray(stationary, dtype=bool).copy()
    centers = np.asarray(centers_cm, dtype=float)
    observed = np.asarray(valid, dtype=bool)
    if base.ndim != 2:
        raise ValueError("stationary must have shape (frames, mice)")
    if centers.shape != (*base.shape, 2):
        raise ValueError("centers_cm must have shape (frames, mice, 2)")
    if observed.shape != base.shape:
        raise ValueError("valid must have the same shape as stationary")
    rate = float(fps)
    gap_limit = float(max_gap_s)
    excursion_limit = float(max_transient_displacement_cm)
    radius_limit = float(max_center_radius_cm)
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("fps must be finite and positive")
    for name, value in (
        ("max_gap_s", gap_limit),
        ("max_transient_displacement_cm", excursion_limit),
        ("max_center_radius_cm", radius_limit),
    ):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")

    usable = observed & np.isfinite(centers).all(axis=2)
    base &= usable
    max_gap_frames = int(math.floor(gap_limit * rate + 1e-9))
    result = np.zeros_like(base)

    for mouse in range(base.shape[1]):
        segments = _runs(base[:, mouse])
        if not segments:
            continue
        bout_start, bout_end = segments[0]
        bout_gaps: list[tuple[int, int]] = []

        def accept_bout(start: int, end: int, gaps: list[tuple[int, int]]) -> None:
            points = centers[start:end, mouse]
            if not len(points) or not np.isfinite(points).all():
                return
            anchor = points[0]
            if np.linalg.norm(points - anchor, axis=1).max(initial=0.0) > excursion_limit:
                return
            center = np.median(points, axis=0)
            if np.linalg.norm(points - center, axis=1).max(initial=0.0) > radius_limit:
                return
            result[start:end, mouse] = base[start:end, mouse]
            for gap_start, gap_end in gaps:
                result[gap_start:gap_end, mouse] = True

        for next_start, next_end in segments[1:]:
            gap_start, gap_end = bout_end, next_start
            gap_length = gap_end - gap_start
            candidate_end = next_end
            candidate_points = centers[bout_start:candidate_end, mouse]
            candidate_valid = usable[bout_start:candidate_end, mouse]
            can_bridge = (
                0 < gap_length <= max_gap_frames
                and candidate_valid.all()
                and np.linalg.norm(
                    candidate_points - candidate_points[0], axis=1
                ).max(initial=0.0)
                <= excursion_limit
            )
            if can_bridge:
                bout_end = next_end
                bout_gaps.append((gap_start, gap_end))
            else:
                accept_bout(bout_start, bout_end, bout_gaps)
                bout_start, bout_end = next_start, next_end
                bout_gaps = []
        accept_bout(bout_start, bout_end, bout_gaps)

    return result


__all__ = ["bounded_stationary_mask", "bridge_gaps_outside_static"]
