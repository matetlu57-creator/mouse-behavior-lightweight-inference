"""Additive per-keypoint and posture primitives for behavior rules.

The gates in :mod:`mouse_behavior.behavior` consume window statistics that the
trajectory adapters already provide.  Grooming, rearing/climbing and the
acceleration-based split between Attack and plain nose contact need evidence
that no adapter computed before, so those derivations live here as pure
functions instead of being inlined into one adapter.  Nothing in this module
changes a label by itself: a rule has to opt in explicitly, which keeps every
historical metric reproducible until the corresponding gate is switched over.

Conventions
-----------
* Coordinates are centimeters unless the name ends in ``_px``.
* ``valid`` is the track-level ``(frames, mice)`` observation mask used
  everywhere else in the pipeline.
* Undecidable results are ``nan``, never ``0.0``.  A dropout must not degrade
  into a calm-looking sample: near-stationary hind legs are positive grooming
  evidence, so a missing keypoint has to stay undecidable.
* Apparent body length is measured in pixels.  A rearing mouse foreshortens
  against the camera, and converting that shortened span with the shared
  cm-per-pixel scale would feed the posture back into the scene calibration.
"""

from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np

from ..utils.rolling import rolling_quantile
from .constants import (
    KEYPOINTS,
    KP_LEFT_EAR,
    KP_LEFT_HIP,
    KP_NECK,
    KP_NOSE,
    KP_RIGHT_EAR,
    KP_RIGHT_HIP,
    KP_TAIL,
)

# Mirrors the reacquisition window used by ``preprocessing.kinematics``: a
# derivative may bridge at most this many frames of a missing track before the
# sample becomes undecidable instead of artificially slow.
MAX_GAP_FRAMES = 3

HEAD_KEYPOINT_INDICES: tuple[int, ...] = (KP_NOSE, KP_LEFT_EAR, KP_RIGHT_EAR)
HIND_KEYPOINT_INDICES: tuple[int, ...] = (KP_LEFT_HIP, KP_RIGHT_HIP, KP_TAIL)
# A top-down view cannot separate dorsal from ventral surface, so the torso
# polygon built from the hip/neck/tail keypoints is the observable stand-in for
# both the back and the belly region that grooming descriptions refer to.
TORSO_KEYPOINT_INDICES: tuple[int, ...] = (KP_NECK, KP_LEFT_HIP, KP_RIGHT_HIP, KP_TAIL)

_MAX_MATRIX_ELEMENTS = 4_000_000


def _as_float_array(name: str, value: object, *, ndim: int) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got shape={array.shape}")
    return array


def _as_track_mask(name: str, value: object, shape: tuple[int, ...]) -> np.ndarray:
    mask = np.asarray(value, dtype=bool)
    if mask.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {mask.shape}")
    return mask


def _check_fps(fps: float) -> float:
    rate = float(fps)
    if not np.isfinite(rate) or rate <= 0.0:
        raise ValueError(f"fps must be finite and > 0, got {fps}")
    return rate


def _check_max_gap(max_gap_frames: int) -> int:
    gap = int(max_gap_frames)
    if gap < 1:
        raise ValueError(f"max_gap_frames must be >= 1, got {max_gap_frames}")
    return gap


def _previous_valid_index(valid: np.ndarray, max_gap_frames: int) -> np.ndarray:
    """Return the most recent earlier observed frame per sample, else ``-1``."""

    frames, mice = valid.shape
    observed = np.where(valid, np.arange(frames, dtype=np.int64)[:, None], -1)
    filled = np.maximum.accumulate(observed, axis=0)
    previous = np.concatenate(
        [np.full((1, mice), -1, dtype=np.int64), filled[:-1]],
        axis=0,
    )
    gap = np.arange(frames, dtype=np.int64)[:, None] - previous
    return np.where((previous >= 0) & (gap <= max_gap_frames), previous, -1)


def keypoint_speed_cm_s(
    keypoints_cm: np.ndarray,
    valid: np.ndarray,
    fps: float,
    max_gap_frames: int = MAX_GAP_FRAMES,
) -> np.ndarray:
    """Per-keypoint displacement speed in cm/s with shape ``(frames, mice, K)``."""

    points = _as_float_array("keypoints_cm", keypoints_cm, ndim=4)
    if points.shape[-1] != 2:
        raise ValueError(f"keypoints_cm last axis must be 2, got shape={points.shape}")
    frames, mice, count = points.shape[:3]
    mask = _as_track_mask("valid", valid, (frames, mice))
    rate = _check_fps(fps)
    gap_limit = _check_max_gap(max_gap_frames)

    previous = _previous_valid_index(mask, gap_limit)
    usable = previous >= 0
    gather = np.where(usable, previous, 0)
    earlier = np.take_along_axis(points, gather[:, :, None, None], axis=0)
    step = np.linalg.norm(points - earlier, axis=-1)
    finite = np.isfinite(points).all(axis=-1) & np.isfinite(earlier).all(axis=-1)
    good = usable[:, :, None] & finite & mask[:, :, None]
    elapsed = (np.arange(frames, dtype=np.int64)[:, None] - gather) / rate

    speeds = np.full((frames, mice, count), np.nan, dtype=float)
    broadcast = np.broadcast_to(elapsed[:, :, None], (frames, mice, count))
    speeds[good] = step[good] / broadcast[good]
    return speeds


def keypoint_acceleration_cm_s2(
    speeds_cm_s: np.ndarray,
    valid: np.ndarray,
    fps: float,
    max_gap_frames: int = MAX_GAP_FRAMES,
) -> np.ndarray:
    """Per-keypoint ``|d speed|/dt`` in cm/s^2 with shape ``(frames, mice, K)``.

    This mirrors the scalar center acceleration already exported by
    ``preprocessing.kinematics``: the signed direction is dropped because a
    lunge and an equally hard braking maneuver are both attack evidence.
    """

    speeds = _as_float_array("speeds_cm_s", speeds_cm_s, ndim=3)
    frames, mice, count = speeds.shape
    mask = _as_track_mask("valid", valid, (frames, mice))
    rate = _check_fps(fps)
    gap_limit = _check_max_gap(max_gap_frames)

    previous = _previous_valid_index(mask, gap_limit)
    usable = previous >= 0
    gather = np.where(usable, previous, 0)
    earlier = np.take_along_axis(speeds, gather[:, :, None], axis=0)
    delta = np.abs(speeds - earlier)
    good = (
        usable[:, :, None]
        & np.isfinite(delta)
        & mask[:, :, None]
    )
    elapsed = (np.arange(frames, dtype=np.int64)[:, None] - gather) / rate

    acceleration = np.full((frames, mice, count), np.nan, dtype=float)
    broadcast = np.broadcast_to(elapsed[:, :, None], (frames, mice, count))
    acceleration[good] = delta[good] / broadcast[good]
    return acceleration


def group_speed_cm_s(
    speeds_cm_s: np.ndarray,
    indices: Sequence[int],
    min_valid: int = 1,
) -> np.ndarray:
    """Mean speed over the selected keypoints, ``nan`` when too few are known."""

    speeds = _as_float_array("speeds_cm_s", speeds_cm_s, ndim=3)
    selected_indices = tuple(int(index) for index in indices)
    if not selected_indices:
        raise ValueError("indices must not be empty")
    available = speeds.shape[2]
    for index in selected_indices:
        if not 0 <= index < available:
            raise ValueError(f"keypoint index {index} is outside [0, {available})")
    required = max(int(min_valid), 1)
    if required > len(selected_indices):
        raise ValueError("min_valid cannot exceed the number of selected keypoints")

    selected = speeds[:, :, list(selected_indices)]
    finite = np.isfinite(selected)
    observed = finite.sum(axis=-1)
    total = np.where(finite, selected, 0.0).sum(axis=-1)
    out = np.full(observed.shape, np.nan, dtype=float)
    good = observed >= required
    out[good] = total[good] / observed[good]
    return out


def head_speed_cm_s(speeds_cm_s: np.ndarray, min_valid: int = 2) -> np.ndarray:
    """Mean nose/ear speed; the head leads both grooming and rearing."""

    return group_speed_cm_s(speeds_cm_s, HEAD_KEYPOINT_INDICES, min_valid)


def hind_speed_cm_s(speeds_cm_s: np.ndarray, min_valid: int = 2) -> np.ndarray:
    """Mean hip/tail speed; grooming and rearing both anchor the rear."""

    return group_speed_cm_s(speeds_cm_s, HIND_KEYPOINT_INDICES, min_valid)


def ratio_or_nan(
    numerator: np.ndarray,
    denominator: np.ndarray,
    min_denominator: float = 1e-6,
) -> np.ndarray:
    """Element-wise ratio that stays ``nan`` below a floor instead of exploding."""

    top = np.asarray(numerator, dtype=float)
    bottom = np.asarray(denominator, dtype=float)
    if top.shape != bottom.shape:
        raise ValueError(f"shapes must match, got {top.shape} and {bottom.shape}")
    floor = float(min_denominator)
    if not np.isfinite(floor) or floor < 0.0:
        raise ValueError(f"min_denominator must be finite and >= 0, got {min_denominator}")
    good = np.isfinite(top) & np.isfinite(bottom) & (np.abs(bottom) >= floor)
    out = np.full(top.shape, np.nan, dtype=float)
    out[good] = top[good] / bottom[good]
    return out


def body_length_px(keypoints_px: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Apparent nose-to-tail span in pixels, ``nan`` when either end is missing."""

    points = _as_float_array("keypoints_px", keypoints_px, ndim=4)
    if points.shape[-1] != 2 or points.shape[2] < KEYPOINTS:
        raise ValueError(f"keypoints_px must have shape (F, M, >= {KEYPOINTS}, 2), got {points.shape}")
    frames, mice = points.shape[:2]
    mask = _as_track_mask("valid", valid, (frames, mice))
    nose = points[:, :, KP_NOSE, :]
    tail = points[:, :, KP_TAIL, :]
    length = np.linalg.norm(nose - tail, axis=-1)
    length[~(mask & np.isfinite(nose).all(axis=-1) & np.isfinite(tail).all(axis=-1))] = np.nan
    return length


def track_quantile(values: np.ndarray, valid: np.ndarray, quantile: float) -> np.ndarray:
    """Per-track quantile over observed samples, shape ``(mice,)``."""

    array = _as_float_array("values", values, ndim=2)
    frames, mice = array.shape
    mask = _as_track_mask("valid", valid, (frames, mice))
    level = float(quantile)
    if not np.isfinite(level) or not 0.0 <= level <= 1.0:
        raise ValueError(f"quantile must be finite and in [0, 1], got {quantile}")
    out = np.full(mice, np.nan, dtype=float)
    for mouse in range(mice):
        sample = array[:, mouse][mask[:, mouse] & np.isfinite(array[:, mouse])]
        if sample.size:
            out[mouse] = float(np.quantile(sample, level))
    return out


def foreshorten_ratio(
    body_px: np.ndarray,
    valid: np.ndarray,
    *,
    window: int,
    window_quantile: float = 0.85,
    track_quantile_level: float = 0.85,
) -> np.ndarray:
    """Apparent body length divided by a robust per-track full-extension base.

    A prone, stretched mouse sits near ``1.0``; standing or climbing folds the
    nose toward the tail and pushes the ratio well below it, so rearing gates
    read ``ratio <= 1 - margin``.  The base is the larger of a causal window
    quantile and a whole-track quantile, which keeps a mouse that rears for
    most of the clip from redefining its own baseline.
    """

    values = _as_float_array("body_px", body_px, ndim=2)
    frames, mice = values.shape
    mask = _as_track_mask("valid", valid, (frames, mice))
    size = max(int(window), 1)

    local = rolling_quantile(values, size, float(window_quantile), active_mask=mask)
    global_base = track_quantile(values, mask, track_quantile_level)[None, :]
    base = np.fmax(local, global_base)
    good = mask & np.isfinite(values) & np.isfinite(base) & (base > 0.0)
    out = np.full((frames, mice), np.nan, dtype=float)
    out[good] = values[good] / base[good]
    return out


def rolling_iqr(
    values: np.ndarray,
    window: int,
    active_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Causal interquartile range, used to separate jitter from a real spread."""

    size = max(int(window), 1)
    upper = rolling_quantile(values, size, 0.75, active_mask=active_mask)
    lower = rolling_quantile(values, size, 0.25, active_mask=active_mask)
    return upper - lower


def deformation_burst_ratio(
    deformation: np.ndarray,
    valid: np.ndarray,
    *,
    window: int,
    high_quantile: float = 0.95,
    min_median: float = 1e-9,
) -> np.ndarray:
    """Window high-quantile over window median of pose deformation energy.

    Grooming is a sustained, non-explosive reshape: the median stays above zero
    while this ratio stays small.  A bite or a wrestling burst drives the tail
    of the window far above its median, so the same statistic separates the two.
    """

    values = _as_float_array("deformation", deformation, ndim=2)
    frames, mice = values.shape
    mask = _as_track_mask("valid", valid, (frames, mice))
    size = max(int(window), 1)
    median = rolling_quantile(values, size, 0.50, active_mask=mask)
    high = rolling_quantile(values, size, float(high_quantile), active_mask=mask)
    return ratio_or_nan(high, median, min_denominator=min_median)


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _convex_hull(points: np.ndarray) -> np.ndarray:
    """Andrew monotone chain hull; the torso never has more than four points."""

    unique = np.unique(np.round(np.asarray(points, dtype=float), 9), axis=0)
    if len(unique) < 3:
        return unique
    ordered = unique[np.lexsort((unique[:, 1], unique[:, 0]))]

    def half(sequence: np.ndarray) -> list[np.ndarray]:
        hull: list[np.ndarray] = []
        for point in sequence:
            while len(hull) >= 2 and _cross(hull[-1] - hull[-2], point - hull[-1]) <= 0.0:
                hull.pop()
            hull.append(point)
        return hull

    lower = half(ordered)
    upper = half(ordered[::-1])
    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def _inflate_polygon(polygon: np.ndarray, inflate_cm: float) -> np.ndarray:
    """Push each hull vertex away from the centroid to approximate a dilation."""

    if inflate_cm <= 0.0:
        return polygon
    centroid = polygon.mean(axis=0)
    offset = polygon - centroid
    norm = np.linalg.norm(offset, axis=1, keepdims=True)
    unit = np.where(norm > 1e-12, offset / np.maximum(norm, 1e-12), 0.0)
    return polygon + unit * float(inflate_cm)


def _point_in_convex(polygon: np.ndarray, point: np.ndarray) -> bool:
    if len(polygon) < 3:
        return False
    relative = polygon - point
    nxt = np.roll(polygon, -1, axis=0) - point
    signs = np.sign(_cross(relative, nxt))
    return not (bool(np.any(signs > 0.0)) and bool(np.any(signs < 0.0)))


def nose_in_torso(
    keypoints_cm: np.ndarray,
    valid: np.ndarray,
    *,
    inflate_cm: float = 0.0,
) -> np.ndarray:
    """True when the nose sits inside the inflated neck/hip/tail hull.

    Undecidable frames (missing nose, fewer than three usable torso keypoints,
    or an unobserved track) are reported as ``False``; callers must therefore
    combine this with a keypoint-coverage requirement instead of treating it as
    a standalone negative.
    """

    points = _as_float_array("keypoints_cm", keypoints_cm, ndim=4)
    if points.shape[-1] != 2 or points.shape[2] < KEYPOINTS:
        raise ValueError(f"keypoints_cm must have shape (F, M, >= {KEYPOINTS}, 2), got {points.shape}")
    frames, mice = points.shape[:2]
    mask = _as_track_mask("valid", valid, (frames, mice))
    if not np.isfinite(inflate_cm) or inflate_cm < 0.0:
        raise ValueError(f"inflate_cm must be finite and >= 0, got {inflate_cm}")

    torso = points[:, :, list(TORSO_KEYPOINT_INDICES), :]
    torso_finite = np.isfinite(torso).all(axis=-1)
    out = np.zeros((frames, mice), dtype=bool)
    for frame, mouse in zip(*np.nonzero(mask)):
        usable = torso_finite[frame, mouse]
        if int(usable.sum()) < 3:
            continue
        nose = points[frame, mouse, KP_NOSE]
        if not np.isfinite(nose).all():
            continue
        hull = _convex_hull(torso[frame, mouse][usable])
        if len(hull) < 3:
            continue
        out[frame, mouse] = _point_in_convex(_inflate_polygon(hull, inflate_cm), nose)
    return out


def _apply_exclusions(
    keep: np.ndarray,
    exclude_indices: Sequence[int],
    keypoint_count: int,
) -> np.ndarray:
    for raw in exclude_indices:
        index = int(raw)
        if not 0 <= index < keypoint_count:
            raise ValueError(f"exclude index {index} is outside [0, {keypoint_count})")
        keep[..., index] = False
    return keep


def _nan_min(values: np.ndarray, axis: int) -> np.ndarray:
    with warnings.catch_warnings():
        # Rows without a single usable target keypoint are legitimately
        # undecidable; nanmin signals that with NaN rather than an exception.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmin(values, axis=axis)


def nose_to_nearest_keypoint_cm(
    actor_nose_cm: np.ndarray,
    target_keypoints_cm: np.ndarray,
    target_keypoint_valid: np.ndarray | None = None,
    *,
    exclude_indices: Sequence[int] = (),
) -> np.ndarray:
    """Distance from one actor nose to the nearest observed target keypoint.

    This is the Approach metric: the actor closes on whichever part of the
    target is actually nearest, and the bout ends on that same distance, so the
    start/end thresholds describe one continuous quantity.
    """

    actor = _as_float_array("actor_nose_cm", actor_nose_cm, ndim=2)
    if actor.shape[-1] != 2:
        raise ValueError(f"actor_nose_cm must have shape (frames, 2), got {actor.shape}")
    target = _as_float_array("target_keypoints_cm", target_keypoints_cm, ndim=3)
    if target.shape[-1] != 2:
        raise ValueError(f"target_keypoints_cm last axis must be 2, got {target.shape}")
    frames = actor.shape[0]
    if target.shape[0] != frames:
        raise ValueError(
            f"actor and target must cover the same frames, got {frames} and {target.shape[0]}"
        )
    count = target.shape[1]

    distance = np.linalg.norm(target - actor[:, None, :], axis=-1)
    keep = np.isfinite(distance)
    if target_keypoint_valid is not None:
        keep &= _as_track_mask("target_keypoint_valid", target_keypoint_valid, (frames, count))
    keep = _apply_exclusions(keep, exclude_indices, count)
    return _nan_min(np.where(keep, distance, np.nan), axis=1)


def nose_to_nearest_keypoint_matrix(
    actor_nose_cm: np.ndarray,
    target_keypoints_cm: np.ndarray,
    target_keypoint_valid: np.ndarray | None = None,
    *,
    exclude_indices: Sequence[int] = (),
) -> np.ndarray:
    """All ordered-pair Approach distances as ``(frames, actor, target)``.

    The diagonal is ``nan``: a mouse is never its own target.  Frames are
    processed in bounded chunks so a long recording cannot allocate a
    frames x mice x mice x keypoints temporary.
    """

    actor = _as_float_array("actor_nose_cm", actor_nose_cm, ndim=3)
    if actor.shape[-1] != 2:
        raise ValueError(f"actor_nose_cm must have shape (F, M, 2), got {actor.shape}")
    target = _as_float_array("target_keypoints_cm", target_keypoints_cm, ndim=4)
    if target.shape[-1] != 2:
        raise ValueError(f"target_keypoints_cm last axis must be 2, got {target.shape}")
    frames, mice = actor.shape[:2]
    if target.shape[:2] != (frames, mice):
        raise ValueError(
            f"target_keypoints_cm must cover {(frames, mice)}, got {target.shape[:2]}"
        )
    count = target.shape[2]

    if target_keypoint_valid is None:
        observed = np.isfinite(target).all(axis=-1)
    else:
        observed = _as_track_mask(
            "target_keypoint_valid", target_keypoint_valid, (frames, mice, count)
        ) & np.isfinite(target).all(axis=-1)
    observed = _apply_exclusions(observed, exclude_indices, count)

    chunk = max(int(_MAX_MATRIX_ELEMENTS // max(mice * mice * count, 1)), 1)
    out = np.full((frames, mice, mice), np.nan, dtype=float)
    for start in range(0, frames, chunk):
        stop = min(start + chunk, frames)
        delta = (
            target[start:stop, None, :, :, :] - actor[start:stop, :, None, None, :]
        )
        distance = np.linalg.norm(delta, axis=-1)
        keep = np.broadcast_to(observed[start:stop, None, :, :], distance.shape)
        out[start:stop] = _nan_min(np.where(keep, distance, np.nan), axis=-1)
    for index in range(mice):
        out[:, index, index] = np.nan
    return out


__all__ = [
    "MAX_GAP_FRAMES",
    "HEAD_KEYPOINT_INDICES",
    "HIND_KEYPOINT_INDICES",
    "TORSO_KEYPOINT_INDICES",
    "keypoint_speed_cm_s",
    "keypoint_acceleration_cm_s2",
    "group_speed_cm_s",
    "head_speed_cm_s",
    "hind_speed_cm_s",
    "ratio_or_nan",
    "body_length_px",
    "track_quantile",
    "foreshorten_ratio",
    "rolling_iqr",
    "deformation_burst_ratio",
    "nose_in_torso",
    "nose_to_nearest_keypoint_cm",
    "nose_to_nearest_keypoint_matrix",
]