"""Independent Grooming/Rearing review candidates from the frozen V3.1 profile.

Ported from the user-provided 2026-10-20 V3 delivery. Evidence, thresholds,
windows and event qualification are independent; conflict resolution cannot
rewrite either raw candidate stream. These seven-point image projections do
not confirm height or wall support and require human review.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict, replace

import numpy as np

from mouse_behavior.behavior.posture_measurements import (
    MeasurementConfig,
    extract_posture_windows,
    quality_frames,
)

POSTURE_REVISION = "2026-10-21-v3-decoupled-negative-aware-review"
POSTURE_SOURCE_ZIP_SHA256 = "72f6892901a9ebe3ec9dc7d3689cf0f3e584276fb60c10ee64e98e957cad5eb7"


def runs(labels):
    if len(labels) == 0:
        return []
    changes = np.flatnonzero(np.asarray(labels)[1:] != np.asarray(labels)[:-1]) + 1
    return list(zip(np.r_[0, changes].tolist(), np.r_[changes, len(labels)].tolist()))


def validate_numbers(config):
    for k, v in asdict(config).items():
        if v is None:
            continue
        if isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) or v < 0:
            raise ValueError(f"invalid {k}")


@dataclass(frozen=True)
class GroomingConfig:
    min_relative_amplitude: float = 0.026531
    max_relative_amplitude: float | None = None
    min_reversals_s: float = 2.0
    min_coherence: float = 0.481215
    max_posterior_excursion: float = 0.075267
    min_relative_motion_fraction: float = 0.711642
    bent_spine_max_cosine: float = 0.9
    min_coverage: float = 0.7

    def __post_init__(self):
        validate_numbers(self)
        if self.min_coherence > 1 or self.min_coverage > 1 or self.bent_spine_max_cosine > 1:
            raise ValueError("fractions/cosines must be <=1")
        if (
            self.max_relative_amplitude is not None
            and self.max_relative_amplitude < self.min_relative_amplitude
        ):
            raise ValueError("max amplitude below min")


@dataclass(frozen=True)
class RearingConfig:
    min_shape_to_translation_ratio: float = 2.914367
    max_relative_motion_fraction: float = 0.594252
    min_neck_excursion: float = 0.064025
    min_spine_cosine: float = 0.967879
    min_neck_radius_change: float = 0.033314
    min_comotion: float = 0.7
    min_coverage: float = 0.7

    def __post_init__(self):
        validate_numbers(self)
        if self.min_spine_cosine > 1 or self.min_comotion > 1 or self.min_coverage > 1:
            raise ValueError("fractions/cosines must be <=1")


@dataclass(frozen=True)
class TemporalConfig:
    window_s: float = 1.2
    support_fraction: float = 0.15
    minimum_positive_windows: int = 2
    min_duration_s: float = 0.3
    min_clip_fraction: float = 0.15

    def __post_init__(self):
        validate_numbers(self)
        if self.window_s <= 0 or self.min_duration_s <= 0:
            raise ValueError("positive time required")
        if self.support_fraction > 1 or self.min_clip_fraction > 1:
            raise ValueError("fraction must be <=1")
        if (
            self.minimum_positive_windows < 1
            or int(self.minimum_positive_windows) != self.minimum_positive_windows
        ):
            raise ValueError("positive integer window count required")


def ge(w, k, x):
    return w.get(k) is not None and w[k] >= x


def le(w, k, x):
    return w.get(k) is not None and w[k] <= x


def grooming_evidence(w, cfg=GroomingConfig()):
    # Does not accept or inspect a RearingConfig or another behavior's result.
    gates = dict(
        observation=ge(w, "coverage", cfg.min_coverage),
        amplitude=ge(w, "body_relative_amplitude", cfg.min_relative_amplitude),
        amplitude_not_extreme=(
            cfg.max_relative_amplitude is None
            or le(w, "body_relative_amplitude", cfg.max_relative_amplitude)
        ),
        posterior_stable=le(w, "posterior_excursion", cfg.max_posterior_excursion),
        repeated=ge(w, "reversals_s", cfg.min_reversals_s),
        nose_ear_coherence=ge(w, "coherence", cfg.min_coherence),
        relative_or_bent=(
            ge(w, "relative_motion_fraction", cfg.min_relative_motion_fraction)
            or le(w, "spine_cosine", cfg.bent_spine_max_cosine)
        ),
    )
    back = (
        ge(w, "back_fraction", 0.2)
        and max(w.get("reversals_s") or 0, w.get("lateral_reversals_s") or 0) >= cfg.min_reversals_s
    )
    required = ["observation", "amplitude", "amplitude_not_extreme", "posterior_stable"]
    front = all(gates[k] for k in ["repeated", "nose_ear_coherence", "relative_or_bent"])
    candidate = all(gates[k] for k in required) and (front or back)
    # Completeness of necessary gates, not a calibrated probability.
    gates["cyclic_relative_motion"] = True  # Frozen V3.1 profile disables this optional gate.
    score = sum(gates.values()) / len(gates) if gates["observation"] else 0.0
    return dict(
        candidate=bool(candidate),
        score=float(score),
        gates=gates,
        reason="back_repeated"
        if candidate and back
        else "relative_head_repeated"
        if candidate
        else "insufficient_grooming_evidence",
        failed_gates=[k for k, v in gates.items() if not v],
    )


def rearing_evidence(w, cfg=RearingConfig()):
    # Positive geometry/motion evidence, never "not Grooming".
    gates = dict(
        observation=ge(w, "coverage", cfg.min_coverage),
        aligned=ge(w, "spine_cosine", cfg.min_spine_cosine),
        forebody_dominant=le(w, "relative_motion_fraction", cfg.max_relative_motion_fraction),
        neck_motion=ge(w, "neck_rear_excursion", cfg.min_neck_excursion),
        projection_shape_change=ge(w, "neck_length_change", cfg.min_neck_radius_change),
        head_neck_comotion=ge(w, "comotion", cfg.min_comotion),
        shape_exceeds_translation=(
            cfg.min_shape_to_translation_ratio == 0
            or ge(w, "shape_to_translation_ratio", cfg.min_shape_to_translation_ratio)
        ),
    )
    candidate = all(gates.values())
    score = sum(gates.values()) / len(gates) if gates["observation"] else 0.0
    return dict(
        candidate=bool(candidate),
        score=float(score),
        gates=gates,
        reason="forebody_projection_pattern_requires_review"
        if candidate
        else "insufficient_rearing_evidence",
        failed_gates=[k for k, v in gates.items() if not v],
    )


def measure(clip, temporal, reader=MeasurementConfig(), body_length_px=None):
    # Changing one class's window changes only that class's measured windows.
    windows, scale = extract_posture_windows(
        clip, replace(reader, window_s=temporal.window_s), body_length_px
    )
    return [
        dict(
            w,
            shape_to_translation_ratio=w["neck_rear_excursion"]
            / max(w["posterior_excursion"], 0.01)
            if w["neck_rear_excursion"] is not None and w["posterior_excursion"] is not None
            else None,
        )
        for w in windows
    ], scale


def behavior_candidates(clip, windows, rule, temporal, behavior, diagnostics=True):
    if behavior == "Grooming":
        evidence = grooming_evidence
    elif behavior == "Rearing":
        evidence = rearing_evidence
    else:
        raise ValueError("unknown behavior")
    n = len(clip["points"])
    fps = clip["fps"]
    den = np.zeros(n)
    votes = np.zeros(n)
    counts = np.zeros(n, int)
    details = []
    for w in windows:
        e = evidence(w, rule)
        if diagnostics:
            details.append(dict(w, evidence=e))
        if not e["gates"]["observation"]:
            continue
        a, b = w["start"], w["end"]
        distance = np.abs(np.arange(a, b) - w["center"])
        radius = max(w["center"] - a, b - 1 - w["center"], 1)
        weight = 1 - 0.75 * distance / radius
        den[a:b] += weight
        if e["candidate"]:
            votes[a:b] += weight
            counts[a:b] += 1
    support = np.divide(votes, den, out=np.zeros(n), where=den > 0)
    mask = (
        quality_frames(clip)
        & (support >= temporal.support_fraction)
        & (counts >= temporal.minimum_positive_windows)
    )
    # Qualify this behavior BEFORE any other behavior or resolver is consulted.
    labels = mask.astype(np.int8)
    events = []
    for a, b in runs(labels):
        if not labels[a]:
            continue
        if (b - a) / fps < temporal.min_duration_s:
            mask[a:b] = False
        else:
            events.append(
                dict(
                    behavior=behavior,
                    track_id=clip["track_id"],
                    start_frame=int(a),
                    end_frame_exclusive=int(b),
                    start_s=float(a / fps),
                    end_s=float(b / fps),
                    requires_review=True,
                )
            )
    fraction = float(mask.mean()) if n else 0.0
    return dict(
        behavior=behavior,
        frame_candidate=mask.tolist(),
        frame_support=support.tolist(),
        candidate_frame_count=int(mask.sum()),
        candidate_fraction=fraction,
        clip_candidate=bool(fraction >= temporal.min_clip_fraction and mask.any()),
        events=events,
        windows=details,
        requires_review=True,
    )


def resolve_candidates(groom, rear, mode="unknown_on_conflict"):
    if mode not in ["unknown_on_conflict", "disabled", "prefer_grooming", "prefer_rearing"]:
        raise ValueError("unknown resolver mode")
    g = np.asarray(groom["frame_candidate"], bool)
    r = np.asarray(rear["frame_candidate"], bool)
    if g.shape != r.shape:
        raise ValueError("candidate frame lengths differ")
    conflict = g & r
    none = ~g & ~r
    state = np.where(conflict, "Both", np.where(g, "Grooming", np.where(r, "Rearing", "None")))
    result = dict(
        mode=mode,
        conflict_frames=int(conflict.sum()),
        no_evidence_frames=int(none.sum()),
        frame_candidate_state=state.tolist(),
        clip_candidate_conflict=bool(groom["clip_candidate"] and rear["clip_candidate"]),
    )
    if mode == "disabled":
        return dict(result, label=None, frame_labels=None, reason="independent_candidates_only")
    labels = np.full(len(g), "Unknown", dtype="<U8")
    labels[g & ~r] = "Grooming"
    labels[r & ~g] = "Rearing"
    if mode == "prefer_grooming":
        labels[conflict] = "Grooming"
    if mode == "prefer_rearing":
        labels[conflict] = "Rearing"
    gc, rc = groom["clip_candidate"], rear["clip_candidate"]
    if gc and rc:
        label = (
            "Grooming"
            if mode == "prefer_grooming"
            else "Rearing"
            if mode == "prefer_rearing"
            else "Unknown"
        )
        reason = "both_clip_candidates_requires_review"
    elif gc:
        label, reason = "Grooming", "only_grooming_clip_evidence"
    elif rc:
        label, reason = "Rearing", "only_rearing_clip_evidence"
    else:
        label, reason = "Unknown", "neither_clip_candidate"
    return dict(result, label=label, frame_labels=labels.tolist(), reason=reason)


def predict(
    clip,
    groom_windows,
    rear_windows,
    groom_rule=GroomingConfig(),
    rear_rule=RearingConfig(),
    groom_temporal=TemporalConfig(),
    rear_temporal=TemporalConfig(),
    resolver_mode="unknown_on_conflict",
    diagnostics=True,
):
    groom = behavior_candidates(
        clip, groom_windows, groom_rule, groom_temporal, "Grooming", diagnostics
    )
    rear = behavior_candidates(clip, rear_windows, rear_rule, rear_temporal, "Rearing", diagnostics)
    resolved = resolve_candidates(groom, rear, resolver_mode)
    return dict(
        version="v3_decoupled",
        track_id=clip["track_id"],
        requires_review=True,
        candidates=dict(Grooming=groom, Rearing=rear),
        resolution=resolved,
        mode="offline_independent_projection_candidates_not_height_or_wall_confirmation",
    )


DEFAULT_GROOMING_CONFIG = GroomingConfig()
DEFAULT_REARING_CONFIG = RearingConfig()
DEFAULT_GROOMING_TEMPORAL = TemporalConfig()
DEFAULT_REARING_TEMPORAL = TemporalConfig()


def infer_posture(
    clip,
    *,
    grooming=DEFAULT_GROOMING_CONFIG,
    rearing=DEFAULT_REARING_CONFIG,
    groom_temporal=DEFAULT_GROOMING_TEMPORAL,
    rear_temporal=DEFAULT_REARING_TEMPORAL,
    reader=MeasurementConfig(),
    body_length_px=None,
    resolver_mode="unknown_on_conflict",
    diagnostics=True,
):
    """Measure each class with its own immutable temporal settings; never fit."""
    gw, gs = measure(clip, groom_temporal, reader, body_length_px)
    rw, rs = measure(clip, rear_temporal, reader, body_length_px)
    result = predict(
        clip, gw, rw, grooming, rearing, groom_temporal, rear_temporal, resolver_mode, diagnostics
    )
    result.update(
        revision=POSTURE_REVISION,
        profile="v3.1_class_balanced_frozen",
        body_scale_px={"Grooming": gs, "Rearing": rs},
        settings={
            "reader": asdict(reader),
            "grooming": asdict(grooming),
            "rearing": asdict(rearing),
            "groom_temporal": asdict(groom_temporal),
            "rear_temporal": asdict(rear_temporal),
        },
        source_zip_sha256=POSTURE_SOURCE_ZIP_SHA256,
    )
    # Conflict removal can shorten an event; requalify fragments without
    # changing independent masks/events, adding padding or filling missing data.
    resolved_events = []
    labels = result["resolution"]["frame_labels"]
    if labels is not None:
        for a, b in runs(labels):
            label = labels[a]
            if label not in {"Grooming", "Rearing"}:
                continue
            temporal = groom_temporal if label == "Grooming" else rear_temporal
            if (b - a) / clip["fps"] >= temporal.min_duration_s:
                resolved_events.append(
                    dict(
                        behavior=label,
                        track_id=clip["track_id"],
                        start_frame=a,
                        end_frame_exclusive=b,
                        requires_review=True,
                    )
                )
    result["resolution"]["events"] = resolved_events
    return result
