"""Conservative trial screen of frozen V3 posture evidence, not a new V3 profile.

Thresholds are explicit review filters, not fitted or scientifically validated.
Image projection does not establish height. Static reconciliation belongs to
the caller. Cached windows cannot establish actual short-projection frames;
without those measurements this API deliberately emits no Rearing frames.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np

from mouse_behavior.behavior.posture_measurements import (
    MeasurementConfig,
    extract_posture_windows,
    quality_frames,
)
from mouse_behavior.behavior.posture_rules import (
    grooming_evidence,
    rearing_evidence,
    runs,
    ge,
    le,
)

REVISION = "2026-10-26-v32-conservative-posture-review-trial"


@dataclass(frozen=True)
class PostureScreenConfig:
    """Fixed trial filters; modifying these is not parameter validation."""

    min_amplitude: float = 0.07
    min_body_relative_speed: float = 7.5 / 10.7
    min_reversals_s: float = 2.0
    min_coherence: float = 0.55
    max_groom_posterior_excursion: float = 0.075267
    max_groom_projection_ratio: float = 0.92
    max_rear_projection_ratio: float = 0.55
    min_projection_change: float = 0.15
    max_rear_posterior_excursion: float = 0.15
    support_fraction: float = 0.15
    minimum_positive_windows: int = 2
    min_duration_s: float = 0.3

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"invalid {name}")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid {name}")
        if not 0 <= self.support_fraction <= 1 or not 0 <= self.min_coherence <= 1:
            raise ValueError("fractions must be <=1")
        if self.min_duration_s <= 0 or self.minimum_positive_windows < 1:
            raise ValueError("positive duration/window count required")
        if int(self.minimum_positive_windows) != self.minimum_positive_windows:
            raise ValueError("window count must be an integer")


DEFAULT_CONFIG = PostureScreenConfig()


def screen_window_evidence(window, config=DEFAULT_CONFIG):
    """Return independent V3 and trial gates, without editing cached features."""
    w = dict(window)
    neck, posterior = w.get("neck_rear_excursion"), w.get("posterior_excursion")
    w["shape_to_translation_ratio"] = (
        neck / max(posterior, 0.01) if neck is not None and posterior is not None else None
    )
    base_g = grooming_evidence(w)["candidate"]
    base_r = rearing_evidence(w)["candidate"]
    repeated = max(w.get("reversals_s") or 0, w.get("lateral_reversals_s") or 0)
    motion = (
        ge(w, "body_relative_amplitude", config.min_amplitude)
        and ge(w, "body_relative_speed", config.min_body_relative_speed)
        and repeated >= config.min_reversals_s
        and ge(w, "coherence", config.min_coherence)
    )
    shortening = (
        le(w, "projection_ratio", config.max_rear_projection_ratio)
        and ge(w, "projection_change", config.min_projection_change)
        and le(w, "posterior_excursion", config.max_rear_posterior_excursion)
    )
    observed = ge(w, "coverage", 0.7)
    return dict(
        base_Grooming=bool(base_g),
        base_Rearing=bool(base_r),
        Grooming=bool(
            base_g
            and motion
            and le(w, "posterior_excursion", config.max_groom_posterior_excursion)
            and le(w, "projection_ratio", config.max_groom_projection_ratio)
        ),
        Rearing=bool(base_r and shortening),
        posture_activity=bool(observed and (motion or shortening)),
    )


def _frame_array(values, n, name, *, boolean=False):
    result = np.asarray(values)
    if result.shape != (n,) or (boolean and result.dtype.kind != "b"):
        raise ValueError(f"{name} must be a {'boolean ' if boolean else ''}frame vector")
    return result.copy() if boolean else result.astype(float, copy=True)


def _qualify(mask, fps, config):
    result = mask.copy()
    for a, b in runs(result):
        if result[a] and (b - a) / fps < config.min_duration_s:
            result[a:b] = False
    return result


def screen_cached_windows(
    windows,
    quality,
    fps,
    *,
    frame_projection_ratio=None,
    social_probe_mask=None,
    config=DEFAULT_CONFIG,
):
    """Screen immutable cached features using measured frame quality.

    ``frame_projection_ratio`` is raw measured nose-tail distance / L, never
    a window median broadcast across frames. When absent, Rearing is withheld
    and its window eligibility is still reported. Social probe flags must be
    supplied from measured nose-in-other-body-box geometry by the caller.
    """
    if not isinstance(config, PostureScreenConfig):
        raise TypeError("config must be PostureScreenConfig")
    if isinstance(fps, bool) or not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    quality = np.asarray(quality)
    if quality.ndim != 1 or quality.dtype.kind != "b":
        raise ValueError("quality must be a boolean frame vector")
    n = len(quality)
    social = (
        np.zeros(n, bool)
        if social_probe_mask is None
        else _frame_array(social_probe_mask, n, "social_probe_mask", boolean=True)
    )
    projection = (
        np.full(n, np.nan)
        if frame_projection_ratio is None
        else _frame_array(frame_projection_ratio, n, "frame_projection_ratio")
    )
    den = np.zeros(n)
    keys = ("base_Grooming", "base_Rearing", "Grooming", "Rearing", "posture_activity")
    votes = {key: np.zeros(n) for key in keys}
    counts = {key: np.zeros(n, int) for key in keys}
    window_counts = dict.fromkeys(keys, 0)
    for w in windows:
        a, b, c = w["start"], w["end"], w["center"]
        if any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in (a, b, c)):
            raise ValueError("window indices must be integers")
        if not 0 <= a <= c < b <= n:
            raise ValueError("window outside frame bounds")
        evidence = screen_window_evidence(w, config)
        for key in keys:
            window_counts[key] += int(evidence[key])
        if not ge(w, "coverage", 0.7):
            continue
        weight = 1 - 0.75 * np.abs(np.arange(a, b) - c) / max(c - a, b - 1 - c, 1)
        den[a:b] += weight
        for key in keys:
            if evidence[key]:
                votes[key][a:b] += weight
                counts[key][a:b] += 1
    masks = {}
    for key in keys:
        support = np.divide(votes[key], den, out=np.zeros(n), where=den > 0)
        mask = quality & (support >= config.support_fraction)
        mask &= counts[key] >= config.minimum_positive_windows
        if key == "Rearing":
            mask &= np.isfinite(projection) & (projection <= config.max_rear_projection_ratio)
        masks[key] = _qualify(mask, fps, config)
    g, r = masks["Grooming"], masks["Rearing"]
    conflict = g & r
    masks["Grooming"] = _qualify(g & ~r & ~social, fps, config)
    masks["Rearing"] = _qualify(r & ~g & ~social, fps, config)
    review = (masks["posture_activity"] | conflict) & ~masks["Grooming"] & ~masks["Rearing"]
    masks["PostureReview"] = _qualify(review, fps, config)
    return dict(
        revision=REVISION,
        requires_review=True,
        masks={key: masks[key] for key in ("Grooming", "Rearing", "PostureReview")},
        base_v3_masks={key: masks[f"base_{key}"] for key in ("Grooming", "Rearing")},
        diagnostics=dict(
            window_counts=window_counts,
            frame_counts={
                key: int(masks[key].sum()) for key in ("Grooming", "Rearing", "PostureReview")
            },
            conflict_frames=int(conflict.sum()),
            social_probe_frames=int((social & quality).sum()),
            measured_projection_available=frame_projection_ratio is not None,
            settings=asdict(config),
            validation_status="trial_filters_not_validated",
        ),
    )


def screen_posture_clip(clip, body_length_px=None, social_probe_mask=None, config=DEFAULT_CONFIG):
    """Measure 1.2 s V3 windows then apply the explicitly versioned trial screen.

    Confidence .8/.4 is a reader contract: the adapter must mask coordinates
    before calling this API, as for the frozen V3 reader. No interpolation is
    performed and unobserved coordinates cannot contribute positive frames.
    """
    reader = MeasurementConfig(window_s=1.2, keypoint_confidence=0.8, detection_confidence=0.4)
    windows, scale = extract_posture_windows(clip, reader, body_length_px)
    quality = quality_frames(clip)
    p = np.asarray(clip["points"], float)
    projection = np.full(len(p), np.nan)
    if scale is not None:
        projection[quality] = np.linalg.norm(p[quality, 0] - p[quality, 6], axis=1) / scale
    result = screen_cached_windows(
        windows,
        quality,
        clip["fps"],
        frame_projection_ratio=projection,
        social_probe_mask=social_probe_mask,
        config=config,
    )
    result["diagnostics"]["body_scale_px"] = scale
    return result
