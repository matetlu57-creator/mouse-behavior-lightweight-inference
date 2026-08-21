"""Continuous evidence and shared data structures for standard behavior.

The module is intentionally free of orchestration.  It converts measured pair
features into auditable evidence consumed by the standard FSM layer.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("mouse_behavior.standard_behavior_engine")

_PROVIDER_SUFFIXES = (
    "strict_chase",
    "window_chase",
    "near_recovery_chase",
    "close_follow_chase",
    "strict_attack",
    "impulse_attack",
    "grapple_attack",
    "occlusion_overlap_attack",
)

_EVIDENCE_BASE_COLUMNS = {
    "pose_pair_quality",
    "identity_pair_quality",
    "mouse_a_track_state",
    "mouse_b_track_state",
    "center_distance_cm",
    "selected_actor_speed_cm_s",
    "selected_target_speed_cm_s",
    "direction_similarity",
    "pursuit_alignment",
    "target_escape_alignment",
    "trajectory_correlation",
    "actor_behind_target",
    "selected_distance_drop_cm",
    "selected_nose_body_distance_cm",
    "selected_target_turn_angle_deg",
    "repeated_contact_count",
    "cluster_attack_hint",
    "cluster_detection_deficit",
    "cluster_merged_like",
    "cluster_overlap_iou",
    "cluster_motion_bl_per_frame",
    "cluster_active_frames",
    "cluster_expected_count",
    "cluster_observed_count",
}

_EVIDENCE_DIRECTION_SUFFIXES = {
    "center_distance_body_lengths",
    "actor_speed_cm_s",
    "target_speed_cm_s",
    "direction_similarity",
    "pursuit_alignment",
    "target_escape_alignment",
    "trajectory_correlation",
    "behind_score",
    "actor_behind_target",
    "closing_speed_cm_s",
    "nose_body_distance_cm",
    "actor_acceleration_cm_s2",
    "target_acceleration_cm_s2",
    "actor_nose_speed_cm_s",
    "actor_head_relative_speed_cm_s",
    "target_turn_angle_deg",
    "actor_angular_speed_deg_s",
    "target_angular_speed_deg_s",
    "actor_pose_deformation_energy",
    "target_pose_deformation_energy",
}


def _evidence_record_columns(columns: Sequence[str]) -> list[str]:
    """Return only columns consumed by row-wise evidence calculations."""

    required = set(_EVIDENCE_BASE_COLUMNS)
    for prefix in ("a_to_b", "b_to_a"):
        required.update(f"{prefix}_{suffix}" for suffix in _EVIDENCE_DIRECTION_SUFFIXES)
    for level in ("weak", "strong"):
        required.update(f"{level}_{suffix}" for suffix in _PROVIDER_SUFFIXES)
    return [column for column in columns if column in required]


def _clip01(value: Any) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(x):
        return 0.0
    return float(np.clip(x, 0.0, 1.0))


def _num(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(key, default))
    except (TypeError, ValueError):
        return float(default)
    return value if np.isfinite(value) else float(default)


def _slice_max(values: np.ndarray, start: int, end: int) -> float:
    """Return max(values[start:end]) without requiring NumPy initial=."""
    window = np.asarray(values[start:end], dtype=float)
    return float(np.max(window)) if window.size else 0.0


def _bool(row: Mapping[str, Any], key: str, default: bool = False) -> bool:
    value = row.get(key, default)
    if value is None:
        return bool(default)
    try:
        if pd.isna(value):
            return bool(default)
    except Exception:
        pass
    return bool(value)


def _ramp_high(value: float, zero_at: float, full_at: float) -> float:
    """Fuzzy membership for x being high."""
    if not np.isfinite(value):
        return 0.0
    zero_at = float(zero_at)
    full_at = float(full_at)
    if full_at <= zero_at:
        return float(value >= full_at)
    return _clip01((float(value) - zero_at) / (full_at - zero_at))


def _ramp_low(value: float, full_at: float, zero_at: float) -> float:
    """Fuzzy membership for x being low."""
    if not np.isfinite(value):
        return 0.0
    full_at = float(full_at)
    zero_at = float(zero_at)
    if zero_at <= full_at:
        return float(value <= full_at)
    return _clip01((zero_at - float(value)) / (zero_at - full_at))


def _threshold_membership(value: float, threshold: float, softness: float = 0.20) -> float:
    threshold = float(threshold)
    span = max(abs(threshold) * float(softness), 0.05)
    return _ramp_high(float(value), threshold - span, threshold + span)


def _distance_membership(distance: float, max_distance: float, full_ratio: float = 0.65) -> float:
    maximum = max(float(max_distance), 1e-6)
    return _ramp_low(float(distance), maximum * float(full_ratio), maximum)


def _speed_membership(speed: float, minimum: float, softness: float = 0.30) -> float:
    minimum = max(float(minimum), 1e-6)
    return _ramp_high(float(speed), minimum * (1.0 - softness), minimum * (1.0 + softness))


def _identity_quality_from_state(state: Any, mapping: Mapping[str, Any]) -> float:
    key = str(state if state is not None else "tracked")
    if key in mapping:
        return _clip01(mapping[key])
    return _clip01(mapping.get("default", 0.50))


def _pair_quality(row: Mapping[str, Any], cfg: Mapping[str, Any]) -> Tuple[float, float, float]:
    pose = _clip01(row.get("pose_pair_quality", 0.0))
    if "identity_pair_quality" in row:
        identity = _clip01(row.get("identity_pair_quality", 0.0))
    else:
        quality_cfg = cfg.get("quality", {})
        mapping = quality_cfg.get(
            "track_state_quality",
            {"tracked": 1.0, "tentative": 0.65, "suspicious": 0.35, "lost": 0.0, "default": 0.50},
        )
        qa = _identity_quality_from_state(row.get("mouse_a_track_state", "tracked"), mapping)
        qb = _identity_quality_from_state(row.get("mouse_b_track_state", "tracked"), mapping)
        identity = float(math.sqrt(max(qa, 0.0) * max(qb, 0.0)))
    combined = float(math.sqrt(max(pose, 0.0) * max(identity, 0.0)))
    return pose, identity, _clip01(combined)


def _prefix_value(
    row: Mapping[str, Any], prefix: str, name: str, fallback: Optional[str] = None
) -> float:
    key = f"{prefix}_{name}"
    if key in row:
        return _num(row, key)
    if fallback is not None:
        return _num(row, fallback)
    return 0.0


def _prefix_bool(
    row: Mapping[str, Any], prefix: str, name: str, fallback: Optional[str] = None
) -> bool:
    key = f"{prefix}_{name}"
    if key in row:
        return _bool(row, key)
    if fallback is not None:
        return _bool(row, fallback)
    return False


@dataclass(frozen=True)
class DirectionEvidence:
    chase: float
    approach: float
    contact: float
    initiation: float
    reaction: float
    dynamic_attack: float
    grapple: float
    pose_deformation: float
    closing: float
    behind: float
    potential_contact: bool
    actor_initiation_gate: bool
    target_reaction_gate: bool
    dynamic_attack_context_gate: bool
    dynamic_attack_direction_gate: bool
    dynamic_attack_gate: bool
    impulse_attack_gate: bool
    stationary_fight_gate: bool
    attack_evidence_count: int


@dataclass
class ChaseFSMResult:
    mask: np.ndarray
    state: np.ndarray


@dataclass
class AttackFSMResult:
    mask: np.ndarray
    state: np.ndarray
    subtype: np.ndarray


def _chase_evidence(
    row: Mapping[str, Any],
    prefix: str,
    chase_cfg: Mapping[str, Any],
    attack_cfg: Mapping[str, Any],
    engine_cfg: Mapping[str, Any],
) -> DirectionEvidence:
    center_distance = _num(row, "center_distance_cm", float("nan"))
    center_distance_bl = _prefix_value(row, prefix, "center_distance_body_lengths", None)
    actor_speed = _prefix_value(row, prefix, "actor_speed_cm_s", "selected_actor_speed_cm_s")
    target_speed = _prefix_value(row, prefix, "target_speed_cm_s", "selected_target_speed_cm_s")
    direction = _prefix_value(row, prefix, "direction_similarity", "direction_similarity")
    pursuit = _prefix_value(row, prefix, "pursuit_alignment", "pursuit_alignment")
    escape = _prefix_value(row, prefix, "target_escape_alignment", "target_escape_alignment")
    trajectory = _prefix_value(row, prefix, "trajectory_correlation", "trajectory_correlation")
    behind_score = _prefix_value(row, prefix, "behind_score", None)
    if f"{prefix}_behind_score" not in row:
        behind_score = (
            1.0 if _prefix_bool(row, prefix, "actor_behind_target", "actor_behind_target") else 0.0
        )
    closing_speed = _prefix_value(row, prefix, "closing_speed_cm_s", None)
    if f"{prefix}_closing_speed_cm_s" not in row:
        lookback = max(float(engine_cfg.get("response_lookback_seconds", 0.30)), 1e-6)
        closing_speed = _num(row, "selected_distance_drop_cm", 0.0) / lookback

    distance_cm_m = _distance_membership(center_distance, float(chase_cfg["max_distance_cm"]))
    reference_body_length = max(float(engine_cfg.get("reference_body_length_cm", 8.0)), 1e-6)
    max_distance_bl = float(chase_cfg["max_distance_cm"]) / reference_body_length
    if np.isfinite(center_distance_bl) and center_distance_bl > 0.0:
        distance_bl_m = _distance_membership(center_distance_bl, max_distance_bl)
        # Physical centimetres remain primary; BL normalization stabilizes
        # videos whose scale estimate or apparent mouse size drifts slightly.
        distance_m = 0.65 * distance_cm_m + 0.35 * distance_bl_m
    else:
        distance_m = distance_cm_m
    actor_speed_m = _speed_membership(actor_speed, float(chase_cfg["actor_min_speed_cm_s"]))
    target_speed_m = _speed_membership(target_speed, float(chase_cfg["target_min_speed_cm_s"]))
    direction_m = _threshold_membership(
        direction, float(chase_cfg["direction_similarity_min"]), 0.18
    )
    pursuit_m = _threshold_membership(pursuit, float(chase_cfg["pursuit_alignment_min"]), 0.22)
    escape_m = _threshold_membership(
        escape, float(chase_cfg.get("target_escape_alignment_min", 0.35)), 0.25
    )
    trajectory_m = _threshold_membership(
        trajectory, float(chase_cfg["trajectory_correlation_min"]), 0.18
    )
    behind_m = _ramp_high(
        behind_score,
        float(engine_cfg.get("behind_zero", -0.10)),
        float(engine_cfg.get("behind_full", 0.55)),
    )

    weights = engine_cfg.get(
        "chase_weights",
        {
            "distance": 0.15,
            "actor_speed": 0.10,
            "target_speed": 0.10,
            "direction": 0.10,
            "pursuit": 0.20,
            "escape": 0.15,
            "behind": 0.10,
            "trajectory": 0.10,
        },
    )
    parts = {
        "distance": distance_m,
        "actor_speed": actor_speed_m,
        "target_speed": target_speed_m,
        "direction": direction_m,
        "pursuit": pursuit_m,
        "escape": escape_m,
        "behind": behind_m,
        "trajectory": trajectory_m,
    }
    total_weight = sum(max(float(weights.get(k, 0.0)), 0.0) for k in parts) or 1.0
    chase = sum(max(float(weights.get(k, 0.0)), 0.0) * v for k, v in parts.items()) / total_weight

    rapid_distance = float(attack_cfg.get("rapid_closing_distance_cm", 2.0))
    response_seconds = max(float(engine_cfg.get("response_lookback_seconds", 0.30)), 1e-6)
    closing_reference = rapid_distance / response_seconds
    closing_m = _ramp_high(closing_speed, 0.0, max(closing_reference, 1.0))
    approach = 0.35 * distance_m + 0.35 * pursuit_m + 0.20 * closing_m + 0.10 * actor_speed_m

    nose_body = _prefix_value(
        row, prefix, "nose_body_distance_cm", "selected_nose_body_distance_cm"
    )
    contact = _ramp_low(
        nose_body,
        float(attack_cfg.get("contact_distance_cm", 3.0)) * 0.65,
        float(attack_cfg.get("contact_distance_cm", 3.0)) * 1.20,
    )

    actor_accel = _prefix_value(row, prefix, "actor_acceleration_cm_s2", None)
    target_accel = _prefix_value(row, prefix, "target_acceleration_cm_s2", None)
    actor_nose_speed = _prefix_value(row, prefix, "actor_nose_speed_cm_s", None)
    actor_head_relative = _prefix_value(row, prefix, "actor_head_relative_speed_cm_s", None)
    if f"{prefix}_actor_head_relative_speed_cm_s" not in row:
        actor_head_relative = max(actor_nose_speed - actor_speed, 0.0)
    target_turn = _prefix_value(
        row, prefix, "target_turn_angle_deg", "selected_target_turn_angle_deg"
    )
    actor_angular = _prefix_value(row, prefix, "actor_angular_speed_deg_s", None)
    target_angular = _prefix_value(row, prefix, "target_angular_speed_deg_s", None)
    actor_deformation = _prefix_value(row, prefix, "actor_pose_deformation_energy", None)
    target_deformation = _prefix_value(row, prefix, "target_pose_deformation_energy", None)

    lunge_m = _speed_membership(
        actor_speed, float(attack_cfg.get("actor_lunge_speed_cm_s", 8.0)), 0.25
    )
    attack_pursuit_m = _threshold_membership(
        pursuit, float(attack_cfg.get("attack_pursuit_alignment_min", 0.50)), 0.22
    )
    head_speed_threshold = float(attack_cfg.get("head_motion_speed_cm_s", 12.0))
    head_motion_m = max(
        _speed_membership(actor_nose_speed, head_speed_threshold, 0.25),
        _ramp_high(actor_head_relative, head_speed_threshold * 0.15, head_speed_threshold * 0.55),
    )
    accel_reference = max(
        float(attack_cfg.get("actor_lunge_speed_cm_s", 8.0)) / response_seconds, 10.0
    )
    accel_m = _ramp_high(abs(actor_accel), accel_reference * 0.35, accel_reference * 1.25)
    initiation = (
        0.30 * attack_pursuit_m
        + 0.25 * max(lunge_m, closing_m)
        + 0.20 * closing_m
        + 0.15 * head_motion_m
        + 0.10 * accel_m
    )

    escape_speed_m = _speed_membership(
        target_speed, float(attack_cfg.get("target_escape_speed_cm_s", 7.0)), 0.25
    )
    escape_align_m = _threshold_membership(
        escape, float(attack_cfg.get("target_escape_alignment_min", 0.30)), 0.25
    )
    turn_m = _threshold_membership(
        target_turn, float(attack_cfg.get("target_turn_angle_deg", 40.0)), 0.20
    )
    target_accel_m = _ramp_high(abs(target_accel), accel_reference * 0.25, accel_reference * 1.10)
    reaction = 0.35 * escape_speed_m + 0.35 * escape_align_m + 0.20 * turn_m + 0.10 * target_accel_m

    # A continuous frame score is useful for confidence, but causality is
    # enforced later by the Attack FSM: initiation -> contact -> reaction.
    dynamic_attack = 0.34 * initiation + 0.32 * contact + 0.34 * reaction

    repeated = _num(row, "repeated_contact_count", 0.0)
    # The continuous scores above are useful for ranking, but they are not
    # sufficient to distinguish an attack from ordinary nose-head/nose-tail
    # contact.  Build the explicit causal evidence used by the calibrated
    # legacy gate: physical contact + actor initiation + target reaction.
    contact_distance = float(attack_cfg.get("contact_distance_cm", 3.0))
    potential_contact = bool(np.isfinite(nose_body) and nose_body < contact_distance)
    distance_drop = _num(row, "selected_distance_drop_cm", 0.0)
    rapid_closing = distance_drop >= float(attack_cfg.get("rapid_closing_distance_cm", 2.0))
    lunge_gate = actor_speed >= float(attack_cfg.get("actor_lunge_speed_cm_s", 8.0))
    target_escape_gate = bool(
        target_speed >= float(attack_cfg.get("target_escape_speed_cm_s", 7.0))
        and escape >= float(attack_cfg.get("target_escape_alignment_min", 0.30))
    )
    target_turn_gate = target_turn >= float(attack_cfg.get("target_turn_angle_deg", 40.0))
    repeated_gate = repeated >= int(attack_cfg.get("repeated_contact_count", 2))
    head_motion_gate = bool(
        actor_nose_speed >= float(attack_cfg.get("head_motion_speed_cm_s", 12.0))
        and actor_nose_speed
        >= float(attack_cfg.get("head_to_center_speed_ratio", 1.35)) * max(actor_speed, 1.0)
    )
    actor_toward_gate = pursuit >= float(attack_cfg.get("attack_pursuit_alignment_min", 0.50))
    actor_initiation_gate = bool(
        actor_toward_gate and (lunge_gate or rapid_closing or (head_motion_gate and rapid_closing))
    )
    target_reaction_gate = target_escape_gate
    attack_evidence_count = int(
        sum(
            [
                lunge_gate,
                rapid_closing,
                target_escape_gate,
                target_turn_gate,
                repeated_gate,
                head_motion_gate,
            ]
        )
    )
    dynamic_attack_context_gate = bool(
        potential_contact
        and attack_evidence_count >= int(attack_cfg.get("min_dynamic_evidence", 2))
        and actor_initiation_gate
        and target_reaction_gate
    )
    repeated_m = _ramp_high(
        repeated,
        max(float(attack_cfg.get("repeated_contact_count", 2)) - 1.0, 0.0),
        max(float(attack_cfg.get("repeated_contact_count", 2)) + 1.0, 1.0),
    )
    angular_threshold = float(attack_cfg.get("stationary_fight_min_angular_speed_deg_s", 110.0))
    angular_m = _ramp_high(
        min(abs(actor_angular), abs(target_angular)), angular_threshold * 0.55, angular_threshold
    )
    center_speed = max(actor_speed, target_speed)
    slow_center_m = _ramp_low(
        center_speed,
        float(attack_cfg.get("stationary_fight_max_center_speed_cm_s", 5.0)),
        max(float(attack_cfg.get("stationary_fight_max_center_speed_cm_s", 5.0)) * 2.5, 8.0),
    )
    local_motion_m = max(head_motion_m, angular_m)
    deformation_pair = math.sqrt(max(actor_deformation, 0.0) * max(target_deformation, 0.0))
    deformation_m = _ramp_high(
        deformation_pair,
        float(engine_cfg.get("pose_deformation_zero", 0.015)),
        float(engine_cfg.get("pose_deformation_full", 0.10)),
    )
    direction_similarity_max = float(attack_cfg.get("attack_direction_similarity_max", 0.90))
    direction_pose_min = float(attack_cfg.get("attack_direction_min_pose_deformation", 0.05))
    direction_escape_max = float(
        attack_cfg.get("attack_direction_max_target_escape_alignment", 0.98)
    )
    direction_turn_min = float(attack_cfg.get("attack_direction_min_target_turn_deg", 30.0))
    direction_drop_min = float(attack_cfg.get("attack_direction_min_distance_drop_cm", 6.0))
    # The actor and target do not have to deform symmetrically during a real
    # attack.  Use the stronger single-mouse deformation, while rejecting the
    # nearly perfectly same-direction motion common in ordinary contact.
    deformation_anchor = max(actor_deformation, target_deformation)
    pose_evidence_available = (
        f"{prefix}_actor_pose_deformation_energy" in row
        or f"{prefix}_target_pose_deformation_energy" in row
    )
    dynamic_attack_direction_gate = bool(
        not pose_evidence_available
        or (
            direction <= direction_similarity_max
            and deformation_anchor >= direction_pose_min
            and escape <= direction_escape_max
            and (target_turn >= direction_turn_min or distance_drop >= direction_drop_min)
        )
    )
    dynamic_attack_gate = bool(dynamic_attack_context_gate and dynamic_attack_direction_gate)
    impulse_attack_gate = bool(
        dynamic_attack_gate
        and actor_speed + target_speed
        >= float(attack_cfg.get("impulse_min_combined_speed_cm_s", 70.0))
        and distance_drop
        >= float(
            attack_cfg.get(
                "impulse_min_distance_drop_cm",
                max(float(attack_cfg.get("attack_direction_min_distance_drop_cm", 6.0)), 8.0),
            )
        )
    )
    grapple = contact * (
        0.24 * repeated_m
        + 0.22 * angular_m
        + 0.24 * deformation_m
        + 0.20 * local_motion_m
        + 0.10 * slow_center_m
    )
    stationary_fight_gate = bool(
        potential_contact
        and np.isfinite(center_distance)
        and center_distance < float(attack_cfg.get("stationary_fight_distance_cm", 5.0))
        and max(actor_speed, target_speed)
        < float(attack_cfg.get("stationary_fight_max_center_speed_cm_s", 5.0))
        and min(abs(actor_angular), abs(target_angular))
        >= float(attack_cfg.get("stationary_fight_min_angular_speed_deg_s", 110.0))
        and repeated_gate
    )

    return DirectionEvidence(
        chase=_clip01(chase),
        approach=_clip01(approach),
        contact=_clip01(contact),
        initiation=_clip01(initiation),
        reaction=_clip01(reaction),
        dynamic_attack=_clip01(dynamic_attack),
        grapple=_clip01(grapple),
        pose_deformation=_clip01(deformation_m),
        closing=_clip01(closing_m),
        behind=_clip01(behind_m),
        potential_contact=potential_contact,
        actor_initiation_gate=actor_initiation_gate,
        target_reaction_gate=target_reaction_gate,
        dynamic_attack_context_gate=dynamic_attack_context_gate,
        dynamic_attack_direction_gate=dynamic_attack_direction_gate,
        dynamic_attack_gate=dynamic_attack_gate,
        impulse_attack_gate=impulse_attack_gate,
        stationary_fight_gate=stationary_fight_gate,
        attack_evidence_count=attack_evidence_count,
    )


def _occlusion_score(row: Mapping[str, Any], attack_cfg: Mapping[str, Any]) -> float:
    if not _bool(row, "cluster_attack_hint", False):
        return 0.0
    deficit = 1.0 if _bool(row, "cluster_detection_deficit", False) else 0.0
    merged = 1.0 if _bool(row, "cluster_merged_like", False) else 0.0
    overlap = _ramp_high(
        _num(row, "cluster_overlap_iou", 0.0),
        float(attack_cfg.get("occlusion_min_overlap_iou", 0.12)) * 0.60,
        float(attack_cfg.get("occlusion_min_overlap_iou", 0.12)) * 1.40,
    )
    motion = _ramp_high(
        _num(row, "cluster_motion_bl_per_frame", 0.0),
        float(attack_cfg.get("occlusion_min_motion_body_lengths_per_frame", 0.08)) * 0.60,
        float(attack_cfg.get("occlusion_min_motion_body_lengths_per_frame", 0.08)) * 1.40,
    )
    active = _ramp_high(
        _num(row, "cluster_active_frames", 0.0),
        max(float(attack_cfg.get("occlusion_min_active_frames", 2)) - 1.0, 0.0),
        float(attack_cfg.get("occlusion_min_active_frames", 2)) + 1.0,
    )
    exact_pair = 1.0
    if bool(attack_cfg.get("occlusion_require_exact_pair_size", True)):
        exact_pair = float(
            int(_num(row, "cluster_expected_count", 0)) == 2
            and int(_num(row, "cluster_observed_count", 0)) < 2
        )
    return _clip01(
        exact_pair
        * (0.25 * deficit + 0.20 * merged + 0.20 * overlap + 0.20 * motion + 0.15 * active)
    )


def _provider_floor(row: Mapping[str, Any], columns: Sequence[str], floor: float) -> float:
    return float(floor) if any(_bool(row, c, False) for c in columns) else 0.0


def _direction_ids(row: Mapping[str, Any], prefix: str) -> Tuple[int, int]:
    a = int(_num(row, "mouse_a_id", -1))
    b = int(_num(row, "mouse_b_id", -1))
    return (a, b) if prefix == "a_to_b" else (b, a)


__all__ = [
    "_PROVIDER_SUFFIXES",
    "_EVIDENCE_BASE_COLUMNS",
    "_EVIDENCE_DIRECTION_SUFFIXES",
    "_evidence_record_columns",
    "_clip01",
    "_num",
    "_slice_max",
    "_bool",
    "_ramp_high",
    "_ramp_low",
    "_threshold_membership",
    "_distance_membership",
    "_speed_membership",
    "_identity_quality_from_state",
    "_pair_quality",
    "_prefix_value",
    "_prefix_bool",
    "DirectionEvidence",
    "ChaseFSMResult",
    "AttackFSMResult",
    "_chase_evidence",
    "_occlusion_score",
    "_provider_floor",
    "_direction_ids",
]
