"""Duration gates for standalone, half-open canonical behavior events.

These gates check the exported process interval. Measured support remains a
separate ledger; parent events, context frames and disjoint evidence totals
cannot qualify a shortened child clip. Passing is a duration check only, not a
new claim that the upstream geometry or biological label has been validated.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from numbers import Integral, Real
from typing import Mapping, TYPE_CHECKING

if TYPE_CHECKING:
    from mouse_behavior.evaluation.exported_dataset_models import HeuristicParameters


@dataclass(frozen=True)
class DurationPolicy:
    behavior: str
    required_s: float
    comparator: str
    source: str
    support_required_s: float | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def resolve_duration_policy(behavior: str, *, parameters: HeuristicParameters | None = None) -> DurationPolicy:
    """Resolve current defaults or explicitly supplied effective rule parameters.

    ``parameters`` is a trusted pipeline/configuration object, never a threshold
    dictionary taken from the event. Only fields actually consumed by canonical
    upstream duration rules are used; social mirrors remain non-authoritative.
    """
    from mouse_behavior.behavior.individual_rules import DEFAULT_INDIVIDUAL_CONFIG
    from mouse_behavior.evaluation.exported_dataset_models import HeuristicParameters, LABEL_TO_CANONICAL
    from mouse_behavior.evaluation.group_heuristic_config import DEFAULT_GROUP_HEURISTIC_CONFIG
    from mouse_behavior.evaluation.social_heuristic_config import DEFAULT_SOCIAL_HEURISTIC_CONFIG

    if not isinstance(behavior, str) or not behavior.strip():
        raise ValueError("behavior must be a nonempty canonical or display label")
    canonical = LABEL_TO_CANONICAL.get(behavior, behavior.strip().lower().replace(" ", "_"))
    canonical = {"static": "stationary", "chasing": "chase", "huddling": "huddle",
                 "snout-head_contact": "nose_head_contact", "snout-rear_contact": "nose_tail_contact",
                 "爬墙": "rearing"}.get(canonical, canonical)
    individual, social, group = DEFAULT_INDIVIDUAL_CONFIG, DEFAULT_SOCIAL_HEURISTIC_CONFIG, DEFAULT_GROUP_HEURISTIC_CONFIG
    defaults = HeuristicParameters()
    policies = {
        "stationary": (individual.static_min_duration_s, ">=", "individual.static_min_duration_s"),
        "locomotion": (individual.locomotion_min_duration_s, ">=", "individual.locomotion_min_duration_s"),
        "grooming": (individual.grooming_min_duration_s, ">=", "individual.grooming_min_duration_s"),
        "rearing": (individual.rearing_min_duration_s, ">=", "individual.rearing_min_duration_s"),
        "approach": (social.approach_min_duration_s, ">=", "social.approach_min_duration_s"),
        "together": (social.together_min_duration_s, ">=", "social.together_min_duration_s"),
        "attack": (social.attack_min_duration_s, ">=", "social.attack_min_duration_s"),
        "chase": (social.chasing_min_duration_s, ">=", "social.chasing_min_duration_s"),
        "nose_head_contact": (social.contact_head_extension_min_duration_s, ">=", "social.contact_head_extension_min_duration_s"),
        "nose_tail_contact": (social.contact_tail_extension_min_duration_s, ">=", "social.contact_tail_extension_min_duration_s"),
        "social_clustering": (group.clustering_final_stability_s, ">=", "group.clustering_final_stability_s"),
        "huddle": (defaults.huddle_min_duration_s, ">=", "parameters.huddle_min_duration_s"),
        "isolation": (max(defaults.isolation_min_duration_s, group.isolation_min_duration_s), ">", "parameters/group.isolation_min_duration_s"),
        "dispersal": (defaults.dispersal_min_duration_s, ">", "parameters.dispersal_min_duration_s"),
        "group_locomotion": (defaults.group_locomotion_min_duration_s, ">", "parameters.group_locomotion_min_duration_s"),
    }
    if canonical not in policies:
        raise ValueError(f"no standalone duration policy for behavior: {behavior!r}")
    minimum, comparator, source = policies[canonical]
    if parameters is not None:
        if not isinstance(parameters, HeuristicParameters):
            raise ValueError("parameters must be a trusted HeuristicParameters instance")
        fields = {
            "stationary": "stationary_min_duration_s", "locomotion": "locomotion_min_duration_s",
            "huddle": "huddle_min_duration_s", "isolation": "isolation_min_duration_s",
            "dispersal": "dispersal_min_duration_s", "group_locomotion": "group_locomotion_min_duration_s",
        }
        for field in fields.values():
            value = getattr(parameters, field)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"parameters.{field} must be finite and positive")
        if canonical in fields:
            field = fields[canonical]
            minimum = float(getattr(parameters, field))
            source = f"parameters.{field}"
            if canonical == "isolation":
                minimum = max(minimum, group.isolation_min_duration_s)
                source = "max(parameters.isolation_min_duration_s,group.isolation_min_duration_s)"
    return DurationPolicy(canonical, float(minimum), comparator, source,
                          float(group.locomotion_simultaneous_min_s) if canonical == "group_locomotion" else None)


def _support_count(intervals, start: int, end: int) -> int:
    if not isinstance(intervals, (list, tuple)):
        raise ValueError("canonical_support_frame_intervals must be a list of half-open ranges")
    merged: list[list[int]] = []
    ranges = []
    for interval in intervals:
        if not isinstance(interval, (list, tuple)) or len(interval) != 2:
            raise ValueError("support ranges must contain two integer frame indices")
        left, right = interval
        if (isinstance(left, bool) or isinstance(right, bool)
                or not isinstance(left, Integral) or not isinstance(right, Integral)
                or not start <= left < right <= end):
            raise ValueError("support ranges must be half-open integer ranges inside the event")
        ranges.append((int(left), int(right)))
    for left, right in sorted(ranges):
        if merged and left <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], right)
        else:
            merged.append([left, right])
    return sum(right - left for left, right in merged)


def qualify_event_duration(event: Mapping, fps: float, *, parameters: HeuristicParameters | None = None) -> dict:
    """Return a JSON-safe duration diagnostic; invalid inputs never qualify.

    Optional contact ``duration_channel``/``contact_channel`` values ``baseline``
    and ``broad_baseline`` enforce the broad tail minimum. Canonical contact
    output combines prequalified baseline and short-extension channels, so its
    default minimum is the valid short-extension minimum.
    Posture events explicitly tagged ``posture_engine='v3_decoupled'`` use the
    authoritative decoupled temporal defaults. Untagged events retain their
    legacy minimum; caller-supplied numeric threshold fields are never read.
    """
    result = dict(ok=False, required_s=None, comparator=None, actual_s=None, reason="invalid_event",
                  duration_basis="half_open_process_frames_over_fps", standalone=True)
    if not isinstance(event, Mapping):
        return result
    try:
        policy = resolve_duration_policy(event.get("canonical_behavior", event.get("behavior")), parameters=parameters)
        if policy.behavior in {"grooming", "rearing"} and "posture_engine" in event:
            if event["posture_engine"] != "v3_decoupled":
                raise ValueError(f"unknown posture_engine: {event['posture_engine']!r}")
            # The optional v3 path must not create a module-load dependency or
            # silently lower the legacy posture gates for untagged ledgers.
            from mouse_behavior.behavior.posture_rules import DEFAULT_GROOMING_TEMPORAL, DEFAULT_REARING_TEMPORAL
            temporal = DEFAULT_GROOMING_TEMPORAL if policy.behavior == "grooming" else DEFAULT_REARING_TEMPORAL
            temporal_name = "DEFAULT_GROOMING_TEMPORAL" if policy.behavior == "grooming" else "DEFAULT_REARING_TEMPORAL"
            policy = replace(policy, required_s=float(temporal.min_duration_s),
                             source=f"posture_rules.{temporal_name}.min_duration_s")
            result["posture_engine"] = "v3_decoupled"
        channel = event.get("duration_channel", event.get("contact_channel"))
        if policy.behavior in {"nose_head_contact", "nose_tail_contact"} and channel is not None:
            if channel not in {"baseline", "broad_baseline", "short_extension", "extension", "combined"}:
                raise ValueError(f"unknown contact duration channel: {channel!r}")
            if channel in {"baseline", "broad_baseline"}:
                from mouse_behavior.evaluation.social_heuristic_config import DEFAULT_SOCIAL_HEURISTIC_CONFIG
                minimum = (DEFAULT_SOCIAL_HEURISTIC_CONFIG.nose_tail_min_duration_s if policy.behavior == "nose_tail_contact"
                           else DEFAULT_SOCIAL_HEURISTIC_CONFIG.nose_contact_min_duration_s)
                policy = replace(policy, required_s=float(minimum), source="social.baseline_contact_min_duration_s")
        result.update(policy.as_dict())
        if isinstance(fps, bool) or not isinstance(fps, Real) or not math.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be finite and positive")
        start, end = event.get("start_frame"), event.get("end_frame")
        if (isinstance(start, bool) or isinstance(end, bool)
                or not isinstance(start, Integral) or not isinstance(end, Integral)
                or not 0 <= start < end):
            raise ValueError("event bounds must be nonnegative half-open integer frame indices")
        result["actual_s"] = (int(end) - int(start)) / float(fps)
        result["frame_count"] = int(end) - int(start)
        if policy.support_required_s is not None:
            if "canonical_support_frame_intervals" not in event:
                raise ValueError("group locomotion requires measured canonical support intervals")
            support = _support_count(event["canonical_support_frame_intervals"], int(start), int(end))
            result["support_frame_count"] = support
            result["support_actual_s"] = support / float(fps)
        passes = result["actual_s"] > policy.required_s if policy.comparator == ">" else result["actual_s"] >= policy.required_s
        if not passes:
            result["reason"] = "process_duration_below_requirement"
        elif (policy.support_required_s is not None
              and result["support_actual_s"] < policy.support_required_s):
            result["reason"] = "measured_sync_support_below_requirement"
        else:
            result.update(ok=True, reason="qualified")
    except (TypeError, ValueError, OverflowError) as error:
        result["reason"] = str(error)
    return result


def require_event_duration(event: Mapping, fps: float, *, parameters: HeuristicParameters | None = None) -> dict:
    """Return passing diagnostics or reject invalid/too-short standalone events."""
    result = qualify_event_duration(event, fps, parameters=parameters)
    if not result["ok"]:
        raise ValueError(f"standalone event duration rejected: {result['reason']} "
                         f"(actual={result['actual_s']}, required {result['comparator']} {result['required_s']} s)")
    return result
