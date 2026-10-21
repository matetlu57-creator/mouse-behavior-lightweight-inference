"""Label-free, offline measurements of seven image-plane pose points.

The fixed point order is nose, right ear, left ear, base of neck, left hip,
right hip, and base of tail. Coordinates and the optional external projected
body reference are pixels. These measurements establish neither true height nor
stable animal identity and contain no behavior classifiers.

The window and descriptor formulas preserve the reviewed delivery's v1/v2
measurement definitions. Missing observations remain missing; neither smoothing
nor temporal derivatives interpolate across a dropout. Inputs are never mutated.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import math
from numbers import Real
from typing import Any
import warnings

import numpy as np

NAMES = (
    "nose",
    "right_ear",
    "left_ear",
    "base_of_neck",
    "left_hip",
    "right_hip",
    "base_of_tail",
)


@dataclass(frozen=True)
class MeasurementConfig:
    """Reader confidence metadata and offline window measurement settings.

    The adapter applies the confidence thresholds before constructing ``clip``;
    this module receives already masked coordinates and ``observed`` flags.
    ``min_coverage`` is retained for consumers; extraction reports every window's
    coverage without silently removing low-coverage evidence.
    """

    keypoint_confidence: float = 0.8
    detection_confidence: float = 0.4
    window_s: float = 0.8
    stride_s: float = 0.1
    min_window_s: float = 0.4
    min_coverage: float = 0.7
    step_deadzone_body: float = 0.006

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name}: expected a finite nonnegative number")
        for name in ("window_s", "stride_s", "min_window_s"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("keypoint_confidence", "detection_confidence", "min_coverage"):
            if getattr(self, name) > 1:
                raise ValueError(f"{name} must be <= 1")


def _positive_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite and positive")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _validated_clip(clip: Mapping[str, Any]) -> dict[str, Any]:
    """Copy coordinates before masking so stale unobserved points cannot leak."""
    if not isinstance(clip, Mapping):
        raise ValueError("clip must be a mapping")
    missing = {"points", "boxes", "observed", "fps"} - clip.keys()
    if missing:
        raise ValueError(f"clip is missing required fields: {', '.join(sorted(missing))}")
    fps = _positive_number(clip["fps"], "fps")
    try:
        points = np.asarray(clip["points"], dtype=float)
        boxes = np.asarray(clip["boxes"], dtype=float)
        observed = np.asarray(clip["observed"])
    except (TypeError, ValueError) as exc:
        raise ValueError("clip arrays must be numeric") from exc
    if points.ndim != 3 or points.shape[1:] != (7, 2):
        raise ValueError("points must have shape (T, 7, 2) in the documented point order")
    n = len(points)
    if boxes.shape != (n, 4):
        raise ValueError("boxes must have shape (T, 4)")
    if observed.shape != (n,):
        raise ValueError("observed must have shape (T,)")
    if observed.dtype.kind != "b":
        raise ValueError("observed must contain boolean flags")
    points = points.copy()
    points[~np.isfinite(points)] = np.nan
    points[~observed] = np.nan
    return {"points": points, "boxes": boxes, "observed": observed, "fps": fps}


def extract_posture_windows(
    clip: Mapping[str, Any],
    config: MeasurementConfig = MeasurementConfig(),
    body_length_px: float | None = None,
) -> tuple[list[dict[str, Any]], float | None]:
    """Return base and enriched descriptors plus a label-free projected scale.

    Prefer a calibrated external projected body reference. The fallback is the
    target clip's q90(nose-neck + neck-tail) path length, which is a proxy.
    An unavailable fallback scale yields ``([], None)``. A supplied invalid
    scale is rejected. Low-coverage windows are preserved as in the source
    measurement definitions; ``min_window_s`` controls window eligibility.
    """
    if not isinstance(config, MeasurementConfig):
        raise TypeError("config must be MeasurementConfig")
    prepared = _validated_clip(clip)
    if body_length_px is not None:
        body_length_px = _positive_number(body_length_px, "body_length_px")
    windows, scale = _extract_base_windows(prepared, config, body_length_px)
    return _enrich(prepared, windows, scale), scale


def quality_frames(clip: Mapping[str, Any]) -> np.ndarray:
    """Require observed nose/neck, at least one ear, and two rear points."""
    prepared = _validated_clip(clip)
    p = prepared["points"]
    return (
        prepared["observed"]
        & np.isfinite(p[:, 0]).all(axis=1)
        & np.isfinite(p[:, 3]).all(axis=1)
        & np.isfinite(p[:, 1:3]).all(axis=2).any(axis=1)
        & (np.isfinite(p[:, 4:7]).all(axis=2).sum(axis=1) >= 2)
    )


def q(values, quantile=0.5):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.quantile(a, quantile)) if a.size else None


def nanmean(a, axis):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a, axis=axis)


def smooth3(p):
    """Do not fill any missing coordinate or bridge dropouts."""
    out = p.copy()
    for i in range(1, len(p) - 1):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out[i] = np.nanmedian(p[i - 1 : i + 2], axis=0)
    out[~np.isfinite(p)] = np.nan
    return out


def excursion(v, scale):
    a = v[np.isfinite(v).all(axis=-1)]
    return (
        float(np.linalg.norm(np.quantile(a, 0.9, axis=0) - np.quantile(a, 0.1, axis=0)) / scale)
        if len(a) > 1
        else None
    )


def reversal_rate(x, fps, dead):
    """Changes of nonzero step sign; reset state at every missing step."""
    prev, count = 0, 0
    for d in np.diff(x):
        if not np.isfinite(d):
            prev = 0
            continue
        s = int(np.sign(d)) if abs(d) >= dead else 0
        if s and prev and s != prev:
            count += 1
        if s:
            prev = s
    return count / max((len(x) - 1) / fps, 1 / fps)


def _extract_base_windows(clip, cfg, body_length_px=None):
    """Returns feature windows and label-free per-target projected scale.

    Prefer calibrated body_length_px from a separate, extended-pose reference.
    Fallback q90(nose-neck + neck-tail) of this target clip is only a proxy.
    """
    raw, fps = clip["points"], clip["fps"]
    path = np.linalg.norm(raw[:, 0] - raw[:, 3], axis=1) + np.linalg.norm(
        raw[:, 3] - raw[:, 6], axis=1
    )
    if body_length_px is not None and (
        not math.isfinite(float(body_length_px)) or float(body_length_px) <= 0
    ):
        raise ValueError("body_length_px must be finite and positive")
    scale = float(body_length_px) if body_length_px is not None else q(path, 0.9)
    if scale is None or not math.isfinite(scale) or scale <= 1e-6:
        return [], None
    p = smooth3(raw)
    rear_count = np.isfinite(p[:, 4:7]).all(axis=2).sum(axis=1)
    hind = nanmean(p[:, 4:7], axis=1)
    hind[rear_count < 2] = np.nan
    ears = nanmean(p[:, 1:3], axis=1)
    nose, neck, tail = p[:, 0], p[:, 3], p[:, 6]
    head = (nose + ears) / 2
    axis = neck - hind
    axis_norm = np.linalg.norm(axis, axis=1)
    unit = np.divide(
        axis,
        axis_norm[:, None],
        out=np.full_like(axis, np.nan),
        where=axis_norm[:, None] > 0.05 * scale,
    )
    relative = head - neck
    axial = np.sum(relative * unit, axis=1) / scale
    lateral = (relative[:, 0] * unit[:, 1] - relative[:, 1] * unit[:, 0]) / scale
    nose_axial = np.sum((nose - neck) * unit, axis=1) / scale
    projection = np.linalg.norm(nose - tail, axis=1) / scale
    good = (
        clip["observed"]
        & np.isfinite(head).all(axis=1)
        & np.isfinite(neck).all(axis=1)
        & np.isfinite(hind).all(axis=1)
        & np.isfinite(unit).all(axis=1)
    )
    windows = []
    half = max(1, round(cfg.window_s * fps / 2))
    stride = max(1, round(cfg.stride_s * fps))
    for center in range(0, len(p), stride):
        a, b = max(0, center - half), min(len(p), center + half + 1)
        if (b - a) / fps < cfg.min_window_s:
            continue
        sl = slice(a, b)
        valid = good[sl]
        # Quantiles do not span missing time for derivatives; absent endpoints stay NaN.
        nr = (nose - neck)[sl] / scale
        er = (ears - neck)[sl] / scale
        dvn, dve = np.diff(nr, axis=0), np.diff(er, axis=0)
        paired = valid[1:] & valid[:-1]
        active = (np.linalg.norm(dvn, axis=1) >= cfg.step_deadzone_body) & paired
        cos = np.sum(dvn * dve, axis=1)
        coherence = float(np.mean(cos[active] > 0)) if active.any() else 0.0
        speed = np.linalg.norm(np.diff(relative[sl] / scale, axis=0), axis=1) * fps
        speed[~paired] = np.nan
        speed[speed < cfg.step_deadzone_body * fps] = 0
        x, y = axial[sl].copy(), lateral[sl].copy()
        x[~valid], y[~valid] = np.nan, np.nan

        def amp(z):
            hi, lo = q(z, 0.9), q(z, 0.1)
            return hi - lo if hi is not None else None

        context = slice(max(0, center - round(2 * fps)), min(len(p), center + round(2 * fps) + 1))
        front, back = head[sl] - neck[sl], neck[sl] - hind[sl]
        den = np.linalg.norm(front, axis=1) * np.linalg.norm(back, axis=1)
        spine = np.divide(
            np.sum(front * back, axis=1), den, out=np.full(len(front), np.nan), where=den > 1e-8
        )
        dh, dn = np.diff((head - hind)[sl], axis=0), np.diff((neck - hind)[sl], axis=0)
        dnrm = np.linalg.norm(dh, axis=1) * np.linalg.norm(dn, axis=1)
        c = np.divide(
            np.sum(dh * dn, axis=1), dnrm, out=np.full(len(dh), np.nan), where=dnrm > 1e-8
        )
        c[~paired] = np.nan
        prhi, prlo = q(projection[context], 0.9), q(projection[context], 0.1)
        npdist = np.linalg.norm(nose[sl] - hind[sl], axis=1) / scale
        backmask = (nose_axial[sl] < 0) & (npdist < 0.5) & valid
        windows.append(
            dict(
                start=a,
                end=b,
                center=center,
                coverage=float(valid.mean()),
                activity=q(speed),
                amplitude=amp(x),
                lateral_amplitude=amp(y),
                reversals_s=reversal_rate(x, fps, cfg.step_deadzone_body),
                lateral_reversals_s=reversal_rate(y, fps, cfg.step_deadzone_body),
                coherence=coherence,
                posterior_excursion=excursion(hind[sl], scale),
                projection_ratio=q(projection[sl]),
                projection_change=prhi - prlo if prhi is not None else None,
                spine_cosine=q(spine),
                comotion=q(c),
                back_fraction=float(backmask.mean()),
                head_excursion=excursion((head - hind)[sl], scale),
                neck_excursion=excursion((neck - hind)[sl], scale),
            )
        )
    return windows, scale


def _enrich(clip, windows, scale):
    if scale is None:
        return []
    p = smooth3(clip["points"])
    fps = clip["fps"]
    rear = nanmean(p[:, 4:7], axis=1)
    rear[np.isfinite(p[:, 4:7]).all(axis=2).sum(axis=1) < 2] = np.nan
    ears = nanmean(p[:, 1:3], axis=1)
    head = (p[:, 0] + ears) / 2
    neck = p[:, 3]
    axis = neck - rear
    length = np.linalg.norm(axis, axis=1)
    u = np.divide(
        axis, length[:, None], out=np.full_like(axis, np.nan), where=length[:, None] > 0.05 * scale
    )
    rel = (head - neck) / scale
    x = np.sum(rel * u, axis=1)
    y = rel[:, 0] * u[:, 1] - rel[:, 1] * u[:, 0]
    bodyrel = np.stack([x, y], axis=1)
    h = (head - rear) / scale
    n = (neck - rear) / scale
    lag = max(1, round(0.067 * fps))
    delta = np.full_like(bodyrel, np.nan)
    delta[lag:] = (bodyrel[lag:] - bodyrel[:-lag]) * fps / lag
    dh = np.full_like(h, np.nan)
    dh[lag:] = (h[lag:] - h[:-lag]) * fps / lag
    dn = np.full_like(n, np.nan)
    dn[lag:] = (n[lag:] - n[:-lag]) * fps / lag
    # A lag endpoint pair is usable only if EVERY intervening frame is observed.
    observed = (
        np.isfinite(head).all(axis=1)
        & np.isfinite(neck).all(axis=1)
        & np.isfinite(rear).all(axis=1)
        & clip["observed"]
    )
    for t in range(len(p)):
        if t < lag or not observed[t - lag : t + 1].all():
            delta[t] = np.nan
            dh[t] = np.nan
            dn[t] = np.nan
    out = []
    for w in windows:
        sl = slice(w["start"], w["end"])

        def span(a):
            good = a[np.isfinite(a).all(axis=1)]
            return (
                float(
                    np.linalg.norm(np.quantile(good, 0.9, axis=0) - np.quantile(good, 0.1, axis=0))
                )
                if len(good) > 1
                else None
            )

        amp = span(bodyrel[sl])
        he = span(h[sl])
        ne = span(n[sl])
        rs = q(np.linalg.norm(delta[sl], axis=1), 0.75)
        hs = q(np.linalg.norm(dh[sl], axis=1), 0.75)
        ns = q(np.linalg.norm(dn[sl], axis=1), 0.75)
        relfrac = rs / hs if rs is not None and hs is not None and hs > 1e-6 else None
        dominance = ne / max(amp, 0.006) if ne is not None and amp is not None else None
        radius = np.linalg.norm(n[sl], axis=1)
        upper, lower = q(radius, 0.9), q(radius, 0.1)
        out.append(
            dict(
                w,
                body_relative_amplitude=amp,
                body_relative_speed=rs,
                head_rear_speed=hs,
                neck_rear_speed=ns,
                relative_motion_fraction=relfrac,
                neck_to_relative_excursion=dominance,
                head_rear_excursion=he,
                neck_rear_excursion=ne,
                neck_length_change=upper - lower if upper is not None else None,
            )
        )
    return out
