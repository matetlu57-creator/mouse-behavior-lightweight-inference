"""Experimental posture motion descriptors; never imported by active detectors.

All distances use an image-plane projected body reference L, not height.
The existing V3 measurements, thresholds and candidate streams are untouched.
No annotation, class label, or stable animal identity is an inference input.
"""

from __future__ import annotations

from dataclasses import replace
import numpy as np

from mouse_behavior.behavior.posture_measurements import (
    MeasurementConfig,
    extract_posture_windows,
    nanmean,
    q,
    smooth3,
)

REVISION = "2026-10-26-motion-posture-experimental-v4"


def _speed(points, observed, fps, scale):
    lag = max(1, round(0.067 * fps))
    result = np.full(len(points), np.nan)
    for t in range(lag, len(points)):
        if observed[t - lag : t + 1].all() and np.isfinite(points[t - lag : t + 1]).all():
            result[t] = np.linalg.norm(points[t] - points[t - lag]) * fps / lag / scale
    return result


def motion_windows(clip, *, window_s=1.2, reader=MeasurementConfig(), body_length_px=None):
    """Add fast-before/settled-after motion and head-triangle geometry to V3 windows.

    All derivatives reset at missing observations. Triangle geometry requires
    all three head points and a nondegenerate ear baseline. The clip q90 area
    reference is label-free, offline context; it is not a normal-pose annotation.
    """
    windows, scale = extract_posture_windows(
        clip, replace(reader, window_s=window_s), body_length_px
    )
    if not windows:
        return [], scale
    raw = np.array(clip["points"], dtype=float, copy=True)
    raw[~np.asarray(clip["observed"], bool)] = np.nan
    p = smooth3(raw)
    fps = clip["fps"]
    rear = nanmean(p[:, 4:7], axis=1)
    rear[np.isfinite(p[:, 4:7]).all(axis=2).sum(axis=1) < 2] = np.nan
    ears = nanmean(p[:, 1:3], axis=1)
    head = (p[:, 0] + ears) / 2
    neck = p[:, 3]
    observed = np.asarray(clip["observed"], bool)
    hs = _speed(head, observed, fps, scale)
    ns = _speed(neck, observed, fps, scale)
    rs = _speed(rear, observed, fps, scale)
    relative_speed = _speed(head - neck, observed, fps, scale)
    a, b = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
    area = np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]) / 2 / scale**2
    ear_width = np.linalg.norm(p[:, 1] - p[:, 2], axis=1) / scale
    area[~np.isfinite(p[:, :3]).all(axis=(1, 2)) | (ear_width < 0.01)] = np.nan
    area_reference = q(area, 0.9)
    neck_radius = np.linalg.norm(neck - rear, axis=1) / scale
    result = []
    for w in windows:
        start, end, center = w["start"], w["end"], w["center"]
        sl = slice(start, end)
        before = slice(max(0, center - round(fps)), center + 1)
        hold_frames = max(1, round(0.4 * fps))
        after = slice(center, min(len(p), center + hold_frames))
        radius_hi, radius_lo = q(neck_radius[before], 0.9), q(neck_radius[before], 0.1)
        mean_area = q(area[sl])
        before_speed = q(np.minimum(hs[before], ns[before]), 0.75)
        settling = q(np.maximum(hs[after], ns[after]))
        rear_speed = q(rs[sl], 0.75)
        result.append(
            dict(
                w,
                head_speed=q(hs[sl], 0.75),
                neck_speed=q(ns[sl], 0.75),
                rear_speed=rear_speed,
                relative_head_speed=q(relative_speed[sl], 0.75),
                before_forebody_speed=before_speed,
                after_forebody_speed=settling,
                after_coverage=float(np.mean(np.isfinite(hs[after]) & np.isfinite(ns[after]))),
                after_window_complete=after.stop - after.start == hold_frames,
                forebody_to_rear_speed=(
                    before_speed / max(rear_speed, 0.01)
                    if before_speed is not None and rear_speed is not None
                    else None
                ),
                before_neck_radius_change=(
                    radius_hi - radius_lo if radius_hi is not None else None
                ),
                neck_rear_radius=q(neck_radius[sl]),
                head_triangle_area=mean_area,
                head_triangle_ratio=(
                    mean_area / area_reference
                    if mean_area is not None
                    and area_reference is not None
                    and area_reference > 1e-9
                    else None
                ),
                triangle_coverage=float(np.isfinite(area[sl]).mean()),
            )
        )
    return result, scale


def grooming_window_mask(w, cfg):
    """Three explicit motion branches; no consultation of Rearing scores."""

    def ge(k, v):
        return w.get(k) is not None and w[k] >= v

    def le(k, v):
        return w.get(k) is not None and w[k] <= v

    observed = ge("coverage", cfg["coverage"])
    stable = le("rear_speed", cfg["max_rear_speed"])
    fast = ge("relative_head_speed", cfg["min_head_speed"])
    measured_motion = ge("body_relative_amplitude", cfg["min_amplitude"])
    small = le("body_relative_amplitude", cfg["max_amplitude"])
    reversals = max(w.get("reversals_s") or 0, w.get("lateral_reversals_s") or 0)
    repeated = reversals >= cfg["min_reversals_s"]
    back = ge("back_fraction", cfg["min_back_fraction"])
    abdomen = (
        ge("triangle_coverage", cfg["coverage"])
        and le("head_triangle_area", cfg["max_triangle_area"])
        and le("head_triangle_ratio", cfg["max_triangle_ratio"])
    )
    common = observed and stable and fast and measured_motion and repeated
    # Head oscillation, back-turn grooming, and abdominal head contraction.
    # The micro-oscillation ceiling belongs to frontal grooming only. A turn
    # towards the back or abdomen can have a larger projected displacement.
    front = cfg["front_enabled"] and small and ge("coherence", cfg["min_coherence"])
    return bool(common and (front or back or abdomen))


def rearing_window_mask(w, cfg):
    """Rapid head+neck rise followed by quiet hold, or already-held compact pose."""

    def ge(k, v):
        return w.get(k) is not None and w[k] >= v

    def le(k, v):
        return w.get(k) is not None and w[k] <= v

    stable = ge("coverage", cfg["coverage"]) and le("rear_speed", cfg["max_rear_speed"])
    rise = (
        w.get("after_window_complete", False)
        and ge("before_forebody_speed", cfg["min_rise_speed"])
        and ge("forebody_to_rear_speed", cfg["min_forebody_ratio"])
        and ge("before_neck_radius_change", cfg["min_neck_change"])
        and ge("after_coverage", cfg["coverage"])
        and le("after_forebody_speed", cfg["max_hold_speed"])
        and w["after_forebody_speed"] <= w["before_forebody_speed"] * cfg["max_hold_to_rise_ratio"]
    )
    held = (
        cfg["held_enabled"]
        and le("neck_rear_radius", cfg["max_hold_radius"])
        and ge("spine_cosine", cfg["min_hold_alignment"])
        and le("head_speed", cfg["max_hold_speed"])
        and le("neck_speed", cfg["max_hold_speed"])
    )
    return bool(stable and (rise or held))


def window_candidate_mask(
    clip, windows, window_results, *, min_duration_s=0.3, support_fraction=0.15
):
    """Observed-only weighted vote and independent continuous qualification."""
    from mouse_behavior.behavior.posture_measurements import quality_frames
    from mouse_behavior.behavior.posture_rules import runs

    n = len(clip["points"])
    votes = np.zeros(n)
    den = np.zeros(n)
    counts = np.zeros(n, int)
    for w, accepted in zip(windows, window_results):
        if w["coverage"] < 0.7:
            continue
        a, b, c = w["start"], w["end"], w["center"]
        weight = 1 - 0.75 * np.abs(np.arange(a, b) - c) / max(c - a, b - 1 - c, 1)
        den[a:b] += weight
        if accepted:
            votes[a:b] += weight
            counts[a:b] += 1
    mask = (
        quality_frames(clip)
        & (np.divide(votes, den, out=np.zeros(n), where=den > 0) >= support_fraction)
        & (counts >= 2)
    )
    for a, b in runs(mask):
        if mask[a] and (b - a) / clip["fps"] < min_duration_s:
            mask[a:b] = False
    return mask
