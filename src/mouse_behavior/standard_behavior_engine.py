#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v1.43 Standard Behavior Engine.

This module turns frame-wise pair geometry/kinematics into auditable temporal
behavior states.  It deliberately separates four layers:

    measured features -> continuous evidence -> role inference -> FSM/events

The legacy chase/attack gates are accepted only as *evidence providers*.  They
never directly OR a frame into the final label when decision_mode=standard.
This keeps the existing domain knowledge while removing the growing collection
of mutually independent final-decision branches.

The engine is pure post-processing: it does not modify YOLO detections,
identity assignment, keypoint smoothing or observation history.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


ENGINE_VERSION = "1.43.0-standard-behavior-engine"
LOGGER = logging.getLogger(__name__)


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


def _prefix_value(row: Mapping[str, Any], prefix: str, name: str, fallback: Optional[str] = None) -> float:
    key = f"{prefix}_{name}"
    if key in row:
        return _num(row, key)
    if fallback is not None:
        return _num(row, fallback)
    return 0.0


def _prefix_bool(row: Mapping[str, Any], prefix: str, name: str, fallback: Optional[str] = None) -> bool:
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
        behind_score = 1.0 if _prefix_bool(row, prefix, "actor_behind_target", "actor_behind_target") else 0.0
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
    direction_m = _threshold_membership(direction, float(chase_cfg["direction_similarity_min"]), 0.18)
    pursuit_m = _threshold_membership(pursuit, float(chase_cfg["pursuit_alignment_min"]), 0.22)
    escape_m = _threshold_membership(
        escape, float(chase_cfg.get("target_escape_alignment_min", 0.35)), 0.25
    )
    trajectory_m = _threshold_membership(
        trajectory, float(chase_cfg["trajectory_correlation_min"]), 0.18
    )
    behind_m = _ramp_high(behind_score, float(engine_cfg.get("behind_zero", -0.10)), float(engine_cfg.get("behind_full", 0.55)))

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
    approach = (
        0.35 * distance_m
        + 0.35 * pursuit_m
        + 0.20 * closing_m
        + 0.10 * actor_speed_m
    )

    nose_body = _prefix_value(row, prefix, "nose_body_distance_cm", "selected_nose_body_distance_cm")
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
    target_turn = _prefix_value(row, prefix, "target_turn_angle_deg", "selected_target_turn_angle_deg")
    actor_angular = _prefix_value(row, prefix, "actor_angular_speed_deg_s", None)
    target_angular = _prefix_value(row, prefix, "target_angular_speed_deg_s", None)
    actor_deformation = _prefix_value(row, prefix, "actor_pose_deformation_energy", None)
    target_deformation = _prefix_value(row, prefix, "target_pose_deformation_energy", None)

    lunge_m = _speed_membership(actor_speed, float(attack_cfg.get("actor_lunge_speed_cm_s", 8.0)), 0.25)
    attack_pursuit_m = _threshold_membership(
        pursuit, float(attack_cfg.get("attack_pursuit_alignment_min", 0.50)), 0.22
    )
    head_speed_threshold = float(attack_cfg.get("head_motion_speed_cm_s", 12.0))
    head_motion_m = max(
        _speed_membership(actor_nose_speed, head_speed_threshold, 0.25),
        _ramp_high(actor_head_relative, head_speed_threshold * 0.15, head_speed_threshold * 0.55),
    )
    accel_reference = max(float(attack_cfg.get("actor_lunge_speed_cm_s", 8.0)) / response_seconds, 10.0)
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
        >= float(attack_cfg.get("head_to_center_speed_ratio", 1.35))
        * max(actor_speed, 1.0)
    )
    actor_toward_gate = pursuit >= float(attack_cfg.get("attack_pursuit_alignment_min", 0.50))
    actor_initiation_gate = bool(
        actor_toward_gate
        and (lunge_gate or rapid_closing or (head_motion_gate and rapid_closing))
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
    angular_m = _ramp_high(min(abs(actor_angular), abs(target_angular)), angular_threshold * 0.55, angular_threshold)
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
    direction_similarity_max = float(
        attack_cfg.get("attack_direction_similarity_max", 0.90)
    )
    direction_pose_min = float(
        attack_cfg.get("attack_direction_min_pose_deformation", 0.05)
    )
    direction_escape_max = float(
        attack_cfg.get("attack_direction_max_target_escape_alignment", 0.98)
    )
    direction_turn_min = float(
        attack_cfg.get("attack_direction_min_target_turn_deg", 30.0)
    )
    direction_drop_min = float(
        attack_cfg.get("attack_direction_min_distance_drop_cm", 6.0)
    )
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
    dynamic_attack_gate = bool(
        dynamic_attack_context_gate and dynamic_attack_direction_gate
    )
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
    return _clip01(exact_pair * (0.25 * deficit + 0.20 * merged + 0.20 * overlap + 0.20 * motion + 0.15 * active))


def _provider_floor(row: Mapping[str, Any], columns: Sequence[str], floor: float) -> float:
    return float(floor) if any(_bool(row, c, False) for c in columns) else 0.0


def _direction_ids(row: Mapping[str, Any], prefix: str) -> Tuple[int, int]:
    a = int(_num(row, "mouse_a_id", -1))
    b = int(_num(row, "mouse_b_id", -1))
    return (a, b) if prefix == "a_to_b" else (b, a)


def _run_chase_fsm(
    chase_score: np.ndarray,
    approach_score: np.ndarray,
    quality: np.ndarray,
    role_confidence: np.ndarray,
    hard_veto: np.ndarray,
    fps: float,
    cfg: Mapping[str, Any],
) -> ChaseFSMResult:
    n = len(chase_score)
    mask = np.zeros(n, dtype=bool)
    states = np.full(n, "IDLE", dtype=object)
    enter = float(cfg.get("enter_score", 0.66))
    exit_score = float(cfg.get("exit_score", 0.44))
    approach_enter = float(cfg.get("approach_enter_score", 0.45))
    role_min = float(cfg.get("min_role_confidence", 0.08))
    min_quality_open = float(cfg.get("min_quality_to_open", 0.42))
    min_quality_hold = float(cfg.get("min_quality_to_hold", 0.20))
    confirm_frames = max(int(math.ceil(float(cfg.get("confirm_seconds", 0.30)) * fps)), 1)
    exit_hold_frames = max(int(round(float(cfg.get("exit_hold_seconds", 0.20)) * fps)), 0)

    active = False
    candidate_start: Optional[int] = None
    candidate_streak = 0
    low_streak = 0
    for i in range(n):
        if hard_veto[i]:
            active = False
            candidate_start = None
            candidate_streak = 0
            low_streak = 0
            states[i] = "IDLE"
            continue

        if active:
            maintain = bool(chase_score[i] >= exit_score and quality[i] >= min_quality_hold)
            if maintain:
                low_streak = 0
                mask[i] = True
                states[i] = "CHASE"
            else:
                low_streak += 1
                if low_streak <= exit_hold_frames:
                    mask[i] = True
                    states[i] = "RECOVERY"
                else:
                    active = False
                    candidate_start = None
                    candidate_streak = 0
                    low_streak = 0
                    states[i] = "IDLE"
            continue

        qualifies = bool(
            chase_score[i] >= enter
            and role_confidence[i] >= role_min
            and quality[i] >= min_quality_open
        )
        if qualifies:
            if candidate_start is None:
                candidate_start = i
                candidate_streak = 1
            else:
                candidate_streak += 1
            states[i] = "APPROACH"
            if candidate_streak >= confirm_frames:
                active = True
                start = int(candidate_start)
                mask[start : i + 1] = True
                states[start : i + 1] = "CHASE"
                low_streak = 0
        else:
            candidate_start = None
            candidate_streak = 0
            states[i] = "APPROACH" if approach_score[i] >= approach_enter else "IDLE"
    return ChaseFSMResult(mask=mask, state=states)


def _run_attack_fsm(
    initiation: np.ndarray,
    contact: np.ndarray,
    reaction: np.ndarray,
    dynamic: np.ndarray,
    grapple: np.ndarray,
    occlusion: np.ndarray,
    quality: np.ndarray,
    role_confidence: np.ndarray,
    hard_veto: np.ndarray,
    dynamic_gate: np.ndarray,
    stationary_gate: np.ndarray,
    dynamic_context_gate: np.ndarray,
    impulse_gate: np.ndarray,
    fps: float,
    cfg: Mapping[str, Any],
) -> AttackFSMResult:
    n = len(initiation)
    mask = np.zeros(n, dtype=bool)
    states = np.full(n, "NONE", dtype=object)
    subtypes = np.full(n, "", dtype=object)

    prepare_enter = float(cfg.get("prepare_enter_score", 0.55))
    contact_enter = float(cfg.get("contact_enter_score", 0.62))
    reaction_enter = float(cfg.get("reaction_enter_score", 0.52))
    dynamic_confirm = float(cfg.get("dynamic_confirm_score", 0.62))
    grapple_confirm = float(cfg.get("grapple_confirm_score", 0.66))
    occlusion_confirm = float(cfg.get("occlusion_confirm_score", 0.70))
    exit_score = float(cfg.get("exit_score", 0.34))
    min_quality_open = float(cfg.get("min_quality_to_open", 0.40))
    min_quality_hold = float(cfg.get("min_quality_to_hold", 0.18))
    min_role_confidence = float(cfg.get("min_role_confidence", 0.06))
    pre_frames = max(int(round(float(cfg.get("pre_contact_seconds", 0.50)) * fps)), 1)
    reaction_frames = max(int(round(float(cfg.get("reaction_window_seconds", 0.65)) * fps)), 1)
    grapple_frames = max(int(math.ceil(float(cfg.get("grapple_confirm_seconds", 0.35)) * fps)), 1)
    exit_hold_frames = max(int(round(float(cfg.get("exit_hold_seconds", 0.25)) * fps)), 0)
    specificity_hold_frames = max(
        int(round(float(cfg.get("specificity_hold_seconds", 0.35)) * fps)),
        1,
    )
    require_causal_gate = bool(cfg.get("require_causal_attack_gate", True))
    min_dynamic_gate_frames = max(int(cfg.get("min_dynamic_gate_frames", 2)), 1)
    min_stationary_gate_frames = max(int(cfg.get("min_stationary_gate_frames", 2)), 1)
    dynamic_gate_count_frames = max(
        int(
            round(
                float(
                    cfg.get(
                        "dynamic_gate_count_window_seconds",
                        cfg.get("pre_contact_seconds", 0.50),
                    )
                )
                * fps
            )
        ),
        pre_frames,
    )

    state = "NONE"
    candidate_start: Optional[int] = None
    prepare_age = 0
    contact_age = 0
    grapple_streak = 0
    exit_streak = 0
    active_subtype = ""

    for i in range(n):
        if hard_veto[i] and occlusion[i] < occlusion_confirm:
            state = "NONE"
            candidate_start = None
            prepare_age = contact_age = grapple_streak = exit_streak = 0
            active_subtype = ""
            states[i] = "NONE"
            continue

        open_ok = quality[i] >= min_quality_open or occlusion[i] >= occlusion_confirm
        if state == "ATTACK":
            evidence_now = max(dynamic[i], grapple[i], occlusion[i], contact[i], reaction[i])
            recent_dynamic_gate = bool(
                _slice_max(
                    dynamic_gate,
                    max(0, i - specificity_hold_frames + 1),
                    i + 1,
                )
            )
            recent_stationary_gate = bool(
                _slice_max(
                    stationary_gate,
                    max(0, i - specificity_hold_frames + 1),
                    i + 1,
                )
            )
            specificity_ok = (
                not require_causal_gate
                or active_subtype == "occlusion_fight"
                or recent_dynamic_gate
                or recent_stationary_gate
            )
            if (
                evidence_now >= exit_score
                and specificity_ok
                and (quality[i] >= min_quality_hold or occlusion[i] >= occlusion_confirm)
            ):
                exit_streak = 0
                mask[i] = True
                states[i] = "ATTACK"
                subtypes[i] = active_subtype
            else:
                exit_streak += 1
                if exit_streak <= exit_hold_frames:
                    mask[i] = True
                    states[i] = "RECOVERY"
                    subtypes[i] = active_subtype
                else:
                    state = "NONE"
                    candidate_start = None
                    active_subtype = ""
                    exit_streak = 0
                    states[i] = "NONE"
            continue

        recent_start = max(0, i - dynamic_gate_count_frames)
        dynamic_gate_count = int(
            np.count_nonzero(dynamic_context_gate[recent_start : i + 1])
        )
        stationary_gate_count = int(np.count_nonzero(stationary_gate[recent_start : i + 1]))
        dynamic_open_ok = bool(
            not require_causal_gate
            or (
                bool(dynamic_gate[i])
                and (
                    dynamic_gate_count >= min_dynamic_gate_frames
                    or bool(impulse_gate[i])
                )
            )
        )
        stationary_open_ok = bool(
            not require_causal_gate
            or (
                bool(stationary_gate[i])
                and stationary_gate_count >= min_stationary_gate_frames
            )
        )

        # High-confidence occlusion is only accepted with recent physical or
        # initiation context; a cluster hint alone cannot create an attack.
        recent_start = max(0, i - pre_frames)
        recent_context = bool(
            _slice_max(contact, recent_start, i + 1) >= contact_enter
            or _slice_max(initiation, recent_start, i + 1) >= prepare_enter
        )
        if open_ok and occlusion[i] >= occlusion_confirm and recent_context:
            start = candidate_start if candidate_start is not None else recent_start
            state = "ATTACK"
            active_subtype = "occlusion_fight"
            mask[int(start) : i + 1] = True
            states[int(start) : i + 1] = "ATTACK"
            subtypes[int(start) : i + 1] = active_subtype
            exit_streak = 0
            continue

        if (
            contact[i] >= contact_enter
            and grapple[i] >= grapple_confirm
            and stationary_open_ok
        ):
            grapple_streak += 1
        else:
            grapple_streak = max(grapple_streak - 1, 0)
        if open_ok and grapple_streak >= grapple_frames:
            start = candidate_start if candidate_start is not None else max(0, i - grapple_streak + 1)
            state = "ATTACK"
            active_subtype = "grapple_fight"
            mask[int(start) : i + 1] = True
            states[int(start) : i + 1] = "ATTACK"
            subtypes[int(start) : i + 1] = active_subtype
            exit_streak = 0
            continue

        if state == "NONE":
            if open_ok and initiation[i] >= prepare_enter:
                state = "PREPARE"
                candidate_start = i
                prepare_age = 0
                states[i] = "ATTACK_PREPARE"
            elif open_ok and contact[i] >= contact_enter:
                # Contact without prior initiation can still become a grapple,
                # but cannot become a lunge until causal evidence appears.
                state = "CONTACT"
                candidate_start = i
                contact_age = 0
                states[i] = "CONTACT"
            else:
                states[i] = "NONE"
            continue

        if state == "PREPARE":
            prepare_age += 1
            states[i] = "ATTACK_PREPARE"
            if contact[i] >= contact_enter:
                state = "CONTACT"
                contact_age = 0
                states[i] = "CONTACT"
                if (
                    role_confidence[i] >= min_role_confidence
                    and reaction[i] >= reaction_enter
                    and dynamic[i] >= dynamic_confirm
                    and dynamic_open_ok
                ):
                    state = "ATTACK"
                    active_subtype = "lunge_attack"
                    start = candidate_start if candidate_start is not None else i
                    mask[int(start) : i + 1] = True
                    states[int(start) : i + 1] = "ATTACK"
                    subtypes[int(start) : i + 1] = active_subtype
            elif prepare_age > pre_frames:
                state = "NONE"
                candidate_start = None
                prepare_age = 0
            continue

        if state == "CONTACT":
            contact_age += 1
            states[i] = "CONTACT"
            causal_initiation = bool(
                _slice_max(initiation, max(0, i - pre_frames), i + 1) >= prepare_enter
            )
            if (
                open_ok
                and causal_initiation
                and role_confidence[i] >= min_role_confidence
                and reaction[i] >= reaction_enter
                and dynamic[i] >= dynamic_confirm
                and dynamic_open_ok
            ):
                state = "ATTACK"
                active_subtype = "lunge_attack"
                start = candidate_start if candidate_start is not None else max(0, i - pre_frames)
                mask[int(start) : i + 1] = True
                states[int(start) : i + 1] = "ATTACK"
                subtypes[int(start) : i + 1] = active_subtype
                exit_streak = 0
            elif contact_age > reaction_frames and grapple_streak == 0:
                state = "NONE"
                candidate_start = None
                contact_age = 0
            continue

    return AttackFSMResult(mask=mask, state=states, subtype=subtypes)


def apply_standard_behavior_engine(
    df: pd.DataFrame,
    fps: float,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Add v1.43 evidence/state columns and optionally replace final labels.

    ``decision_mode``:
      - ``standard``: standard FSM owns weak/strong final_chase/final_attack.
      - ``shadow``: compute every standard column but retain legacy final labels.
      - ``legacy``: no replacement; useful for emergency rollback.
    """
    if df.empty:
        return df.copy()
    LOGGER.debug("Applying %s to %d pair rows at %.3f FPS", ENGINE_VERSION, len(df), fps)
    engine_cfg = dict(config.get("standard_behavior_engine", {}))
    if not bool(engine_cfg.get("enabled", True)):
        return df.copy()

    output = df.copy()
    decision_mode = str(engine_cfg.get("decision_mode", "standard")).strip().lower()
    if decision_mode not in {"standard", "shadow", "legacy"}:
        raise ValueError("standard_behavior_engine.decision_mode must be standard/shadow/legacy")

    response_seconds = float(config.get("features", {}).get("response_lookback_seconds", 0.30))
    common_engine = dict(engine_cfg)
    common_engine["response_lookback_seconds"] = response_seconds
    common_engine["reference_body_length_cm"] = float(
        config.get("scale", {}).get("assumed_mouse_body_length_cm", 8.0)
    )

    row_count = len(output)
    pose_quality = np.zeros(row_count, dtype=float)
    identity_quality = np.zeros(row_count, dtype=float)
    behavior_quality = np.zeros(row_count, dtype=float)
    interaction_cfg = engine_cfg.get("interaction_graph", {})
    interaction_radius = float(
        interaction_cfg.get(
            "radius_cm",
            max(
                float(config["chase"]["weak"]["max_distance_cm"]),
                float(config["attack"]["weak"].get("body_center_contact_distance_cm", 6.0)),
            )
            + float(interaction_cfg.get("buffer_cm", 5.0)),
        )
    )

    # The lightweight analyzer keeps the pair's complete timeline for audit,
    # but marks frames outside its padded distance/heading windows invalid.
    # Turning every one of those rows into a Python dictionary and evaluating
    # four directional evidence objects was the dominant long-video cost.  A
    # hard-vetoed row cannot open or hold an FSM state, so it is safe to skip
    # its expensive evidence calculation.  Explicit provider/occlusion hints
    # remain computable for full-pipeline callers even when valid_pair is false.
    valid_pair_input = output.get(
        "valid_pair", pd.Series(True, index=output.index)
    ).fillna(False).astype(bool).to_numpy()
    provider_hint = np.zeros(row_count, dtype=bool)
    provider_suffixes = (
        "strict_chase",
        "window_chase",
        "near_recovery_chase",
        "close_follow_chase",
        "strict_attack",
        "impulse_attack",
        "grapple_attack",
        "occlusion_overlap_attack",
    )
    for level in ("weak", "strong"):
        for suffix in provider_suffixes:
            column = f"{level}_{suffix}"
            if column in output:
                provider_hint |= output[column].fillna(False).astype(bool).to_numpy()
    if "cluster_attack_hint" in output:
        provider_hint |= output["cluster_attack_hint"].fillna(False).astype(bool).to_numpy()

    compute_candidate = valid_pair_input | provider_hint
    skip_inactive_rows = bool(engine_cfg.get("skip_inactive_rows", True))
    compute_row_mask = (
        compute_candidate.copy()
        if skip_inactive_rows
        else np.ones(row_count, dtype=bool)
    )
    compute_indices = np.flatnonzero(compute_row_mask)
    compute_rows = output.iloc[compute_indices].to_dict("records")

    distance_all = pd.to_numeric(
        output.get("center_distance_cm", pd.Series(np.nan, index=output.index)),
        errors="coerce",
    ).to_numpy(dtype=float)
    interaction_candidate = np.isfinite(distance_all) & (
        distance_all <= interaction_radius
    )
    for i, row in zip(compute_indices, compute_rows):
        pq, iq, bq = _pair_quality(row, engine_cfg)
        pose_quality[i], identity_quality[i], behavior_quality[i] = pq, iq, bq
    output["standard_pose_quality"] = pose_quality
    output["standard_identity_quality"] = identity_quality
    output["standard_behavior_quality"] = behavior_quality
    output["standard_interaction_candidate"] = interaction_candidate
    output["standard_behavior_compute_candidate"] = compute_candidate
    output["standard_behavior_compute_row"] = compute_row_mask

    for level in ("weak", "strong"):
        chase_cfg = config["chase"][level]
        attack_cfg = config["attack"][level]
        level_cfg = dict(engine_cfg.get(level, {}))
        chase_fsm_cfg = dict(level_cfg.get("chase_fsm", {}))
        attack_fsm_cfg = dict(level_cfg.get("attack_fsm", {}))
        provider_cfg = dict(level_cfg.get("provider_floors", {}))

        ab_list = []
        ba_list = []
        occ = np.zeros(row_count, dtype=float)
        for i, row in zip(compute_indices, compute_rows):
            ab = _chase_evidence(row, "a_to_b", chase_cfg, attack_cfg, common_engine)
            ba = _chase_evidence(row, "b_to_a", chase_cfg, attack_cfg, common_engine)
            ab_list.append(ab)
            ba_list.append(ba)
            occ[i] = _occlusion_score(row, attack_cfg)

        def scatter_evidence(
            values: Sequence[DirectionEvidence],
            attribute: str,
            dtype: Any = float,
        ) -> np.ndarray:
            scattered = np.zeros(row_count, dtype=dtype)
            if len(compute_indices):
                scattered[compute_indices] = np.asarray(
                    [getattr(value, attribute) for value in values], dtype=dtype
                )
            return scattered

        ab_chase = scatter_evidence(ab_list, "chase")
        ba_chase = scatter_evidence(ba_list, "chase")
        ab_approach = scatter_evidence(ab_list, "approach")
        ba_approach = scatter_evidence(ba_list, "approach")
        ab_contact = scatter_evidence(ab_list, "contact")
        ba_contact = scatter_evidence(ba_list, "contact")
        ab_initiation = scatter_evidence(ab_list, "initiation")
        ba_initiation = scatter_evidence(ba_list, "initiation")
        ab_reaction = scatter_evidence(ab_list, "reaction")
        ba_reaction = scatter_evidence(ba_list, "reaction")
        ab_dynamic = scatter_evidence(ab_list, "dynamic_attack")
        ba_dynamic = scatter_evidence(ba_list, "dynamic_attack")
        ab_grapple = scatter_evidence(ab_list, "grapple")
        ba_grapple = scatter_evidence(ba_list, "grapple")
        ab_dynamic_gate = scatter_evidence(ab_list, "dynamic_attack_gate", bool)
        ba_dynamic_gate = scatter_evidence(ba_list, "dynamic_attack_gate", bool)
        ab_impulse_gate = scatter_evidence(ab_list, "impulse_attack_gate", bool)
        ba_impulse_gate = scatter_evidence(ba_list, "impulse_attack_gate", bool)
        ab_dynamic_context_gate = scatter_evidence(
            ab_list, "dynamic_attack_context_gate", bool
        )
        ba_dynamic_context_gate = scatter_evidence(
            ba_list, "dynamic_attack_context_gate", bool
        )
        ab_stationary_gate = scatter_evidence(ab_list, "stationary_fight_gate", bool)
        ba_stationary_gate = scatter_evidence(ba_list, "stationary_fight_gate", bool)
        ab_potential_contact = scatter_evidence(ab_list, "potential_contact", bool)
        ba_potential_contact = scatter_evidence(ba_list, "potential_contact", bool)
        ab_evidence_count = scatter_evidence(ab_list, "attack_evidence_count", int)
        ba_evidence_count = scatter_evidence(ba_list, "attack_evidence_count", int)
        pair_deformation = np.maximum(
            scatter_evidence(ab_list, "pose_deformation"),
            scatter_evidence(ba_list, "pose_deformation"),
        )

        # Existing specialized gates become lower bounds on evidence only.
        def scatter_provider(columns: Sequence[str], floor: float) -> np.ndarray:
            scattered = np.zeros(row_count, dtype=float)
            if len(compute_indices):
                scattered[compute_indices] = np.asarray(
                    [_provider_floor(row, columns, floor) for row in compute_rows],
                    dtype=float,
                )
            return scattered

        chase_provider = scatter_provider(
            [
                f"{level}_strict_chase",
                f"{level}_window_chase",
                f"{level}_near_recovery_chase",
                f"{level}_close_follow_chase",
            ],
            float(provider_cfg.get("chase", 0.70 if level == "weak" else 0.78)),
        )
        dynamic_provider = scatter_provider(
            [f"{level}_strict_attack", f"{level}_impulse_attack"],
            float(provider_cfg.get("dynamic_attack", 0.70 if level == "weak" else 0.80)),
        )
        grapple_provider = scatter_provider(
            [f"{level}_grapple_attack"],
            float(provider_cfg.get("grapple", 0.72 if level == "weak" else 0.82)),
        )
        occlusion_provider = scatter_provider(
            [f"{level}_occlusion_overlap_attack"],
            float(provider_cfg.get("occlusion", 0.78 if level == "weak" else 0.86)),
        )
        occ = np.maximum(occ, occlusion_provider)

        # Role inference is behavior-specific.  Chase and attack can disagree
        # during close contact, so never force both through one role score.
        a_ids = pd.to_numeric(output["mouse_a_id"], errors="coerce").fillna(-1).astype(int).to_numpy()
        b_ids = pd.to_numeric(output["mouse_b_id"], errors="coerce").fillna(-1).astype(int).to_numpy()
        chase_role_conf = np.abs(ab_chase - ba_chase)
        chase_ab_is_actor = ab_chase >= ba_chase
        selected_actor = pd.to_numeric(
            output.get("selected_actor_id", pd.Series(-1, index=output.index)),
            errors="coerce",
        ).fillna(-1).astype(int).to_numpy()
        selected_target = pd.to_numeric(
            output.get("selected_target_id", pd.Series(-1, index=output.index)),
            errors="coerce",
        ).fillna(-1).astype(int).to_numpy()
        selected_role_cfg = dict(engine_cfg.get("selected_role_fallback", {}))
        selected_role_fallback = np.zeros(row_count, dtype=bool)
        if bool(selected_role_cfg.get("enabled", False)):
            selected_valid = (
                ((selected_actor == a_ids) | (selected_actor == b_ids))
                & ((selected_target == a_ids) | (selected_target == b_ids))
                & (selected_actor != selected_target)
            )
            # This is deliberately limited to exact directional ties.  A
            # real directional score wins whenever the pair features provide
            # one, so the fallback cannot overwrite a resolved role.
            selected_role_fallback = selected_valid & (chase_role_conf <= 1e-9)
            fallback_conf = float(selected_role_cfg.get("confidence", 0.20))
            chase_role_conf = np.where(
                selected_role_fallback,
                np.maximum(chase_role_conf, fallback_conf),
                chase_role_conf,
            )
            chase_ab_is_actor = np.where(
                selected_role_fallback,
                selected_actor == a_ids,
                chase_ab_is_actor,
            )
        ab_attack_role_score = 0.58 * ab_initiation + 0.42 * ab_reaction
        ba_attack_role_score = 0.58 * ba_initiation + 0.42 * ba_reaction
        attack_role_conf = np.abs(ab_attack_role_score - ba_attack_role_score)
        attack_ab_is_actor = ab_attack_role_score >= ba_attack_role_score

        chase_score = np.maximum(np.maximum(ab_chase, ba_chase), chase_provider)
        approach_score = np.maximum(ab_approach, ba_approach)
        contact_score = np.maximum(ab_contact, ba_contact)
        initiation_score = np.where(attack_ab_is_actor, ab_initiation, ba_initiation)
        reaction_score = np.where(attack_ab_is_actor, ab_reaction, ba_reaction)
        dynamic_gate = np.where(attack_ab_is_actor, ab_dynamic_gate, ba_dynamic_gate)
        impulse_gate = np.where(attack_ab_is_actor, ab_impulse_gate, ba_impulse_gate)
        dynamic_context_gate = np.where(
            attack_ab_is_actor,
            ab_dynamic_context_gate,
            ba_dynamic_context_gate,
        )
        stationary_gate = np.where(attack_ab_is_actor, ab_stationary_gate, ba_stationary_gate)
        potential_contact = np.where(attack_ab_is_actor, ab_potential_contact, ba_potential_contact)
        attack_evidence_count = np.where(attack_ab_is_actor, ab_evidence_count, ba_evidence_count)
        causal_distance = pd.to_numeric(
            output["center_distance_cm"], errors="coerce"
        ).to_numpy(dtype=float)
        causal_distance_max = float(
            attack_fsm_cfg.get("causal_max_center_distance_cm", float("inf"))
        )
        if np.isfinite(causal_distance_max):
            causal_near = np.isfinite(causal_distance) & (
                causal_distance <= causal_distance_max
            )
            dynamic_gate &= causal_near
            dynamic_context_gate &= causal_near
            impulse_gate &= causal_near
            stationary_gate &= causal_near
        dynamic_score = np.maximum(np.where(attack_ab_is_actor, ab_dynamic, ba_dynamic), dynamic_provider)
        grapple_score = np.maximum(np.maximum(ab_grapple, ba_grapple), grapple_provider)
        attack_score = np.maximum.reduce([dynamic_score, grapple_score, occ])

        valid_pair = output["valid_pair"].fillna(False).astype(bool).to_numpy()
        wall_veto = output.get(
            "pair_wall_jump_excluded", pd.Series(False, index=output.index)
        ).fillna(False).astype(bool).to_numpy()
        max_distance = float(chase_cfg["max_distance_cm"])
        distance = pd.to_numeric(output["center_distance_cm"], errors="coerce").to_numpy(float)
        physics_veto = (~valid_pair) | wall_veto | (~np.isfinite(distance))
        # The interaction graph radius is a compute/QA boundary, not the chase
        # threshold itself.  It prevents far-away pairs from opening states.
        physics_veto |= distance > interaction_radius

        chase_fsm = _run_chase_fsm(
            chase_score,
            approach_score,
            behavior_quality,
            chase_role_conf,
            physics_veto,
            fps,
            chase_fsm_cfg,
        )
        attack_fsm = _run_attack_fsm(
            initiation_score,
            contact_score,
            reaction_score,
            dynamic_score,
            grapple_score,
            occ,
            behavior_quality,
            attack_role_conf,
            physics_veto,
            dynamic_gate,
            stationary_gate,
            dynamic_context_gate,
            impulse_gate,
            fps,
            attack_fsm_cfg,
        )

        chase_role_min = float(chase_fsm_cfg.get("min_role_confidence", 0.08))
        attack_role_min = float(attack_fsm_cfg.get("min_role_confidence", 0.06))
        chase_role_known = chase_role_conf >= chase_role_min
        attack_role_known = attack_role_conf >= attack_role_min
        chase_actor_ids = np.where(chase_ab_is_actor, a_ids, b_ids)
        chase_target_ids = np.where(chase_ab_is_actor, b_ids, a_ids)
        attack_actor_ids = np.where(attack_ab_is_actor, a_ids, b_ids)
        attack_target_ids = np.where(attack_ab_is_actor, b_ids, a_ids)
        chase_actor_ids = np.where(chase_role_known, chase_actor_ids, -1)
        chase_target_ids = np.where(chase_role_known, chase_target_ids, -1)
        attack_actor_ids = np.where(attack_role_known, attack_actor_ids, -1)
        attack_target_ids = np.where(attack_role_known, attack_target_ids, -1)
        attack_dominates = attack_score > chase_score
        actor_ids = np.where(attack_dominates, attack_actor_ids, chase_actor_ids)
        target_ids = np.where(attack_dominates, attack_target_ids, chase_target_ids)
        role_conf = np.where(attack_dominates, attack_role_conf, chase_role_conf)

        contact_types = np.full(row_count, "", dtype=object)
        contact_threshold = float(attack_cfg.get("contact_distance_cm", 3.0))
        for i, row in zip(compute_indices, compute_rows):
            head_values = [
                _num(row, "a_to_b_nose_head_distance_cm", float("inf")),
                _num(row, "b_to_a_nose_head_distance_cm", float("inf")),
            ]
            tail_values = [
                _num(row, "a_to_b_nose_tail_distance_cm", float("inf")),
                _num(row, "b_to_a_nose_tail_distance_cm", float("inf")),
            ]
            body_values = [
                _num(row, "a_to_b_nose_body_distance_cm", float("inf")),
                _num(row, "b_to_a_nose_body_distance_cm", float("inf")),
            ]
            head_min = min(head_values)
            tail_min = min(tail_values)
            body_min = min(body_values)
            head_contact = head_min <= contact_threshold
            tail_contact = tail_min <= contact_threshold
            if head_contact and tail_contact:
                contact_types[i] = "nose_head_and_nose_tail"
            elif head_contact:
                contact_types[i] = "nose_head"
            elif tail_contact:
                contact_types[i] = "nose_tail"
            elif body_min <= contact_threshold:
                contact_types[i] = "nose_body"

        output[f"{level}_standard_a_to_b_chase_score"] = ab_chase
        output[f"{level}_standard_b_to_a_chase_score"] = ba_chase
        output[f"{level}_standard_chase_score"] = chase_score
        output[f"{level}_standard_approach_score"] = approach_score
        output[f"{level}_standard_contact_score"] = contact_score
        output[f"{level}_standard_initiation_score"] = initiation_score
        output[f"{level}_standard_reaction_score"] = reaction_score
        output[f"{level}_standard_dynamic_attack_score"] = dynamic_score
        output[f"{level}_standard_attack_dynamic_gate"] = dynamic_gate
        output[f"{level}_standard_attack_context_gate"] = dynamic_context_gate
        output[f"{level}_standard_attack_impulse_gate"] = impulse_gate
        output[f"{level}_standard_attack_stationary_gate"] = stationary_gate
        output[f"{level}_standard_attack_potential_contact"] = potential_contact
        output[f"{level}_standard_attack_evidence_count"] = attack_evidence_count
        output[f"{level}_standard_grapple_score"] = grapple_score
        output[f"{level}_standard_pose_deformation_score"] = pair_deformation
        output[f"{level}_standard_contact_type"] = contact_types
        output[f"{level}_standard_occlusion_score"] = occ
        output[f"{level}_standard_attack_score"] = attack_score
        output[f"{level}_standard_chase_role_confidence"] = chase_role_conf
        output[f"{level}_standard_chase_role_fallback"] = selected_role_fallback
        output[f"{level}_standard_attack_role_confidence"] = attack_role_conf
        output[f"{level}_standard_role_confidence"] = role_conf
        output[f"{level}_standard_chase_actor_id"] = chase_actor_ids
        output[f"{level}_standard_chase_target_id"] = chase_target_ids
        output[f"{level}_standard_attack_actor_id"] = attack_actor_ids
        output[f"{level}_standard_attack_target_id"] = attack_target_ids
        output[f"{level}_standard_actor_id"] = actor_ids
        output[f"{level}_standard_target_id"] = target_ids
        output[f"{level}_standard_chase_state"] = chase_fsm.state
        output[f"{level}_standard_attack_state"] = attack_fsm.state
        output[f"{level}_standard_attack_subtype"] = attack_fsm.subtype
        output[f"{level}_standard_final_chase"] = chase_fsm.mask & valid_pair
        output[f"{level}_standard_final_attack"] = attack_fsm.mask & valid_pair
        output[f"{level}_standard_behavior_confidence"] = np.where(
            chase_fsm.mask | attack_fsm.mask,
            np.maximum(chase_score, attack_score) * behavior_quality,
            0.0,
        )

        if decision_mode == "standard":
            output[f"{level}_final_chase"] = output[f"{level}_standard_final_chase"].astype(bool)
            output[f"{level}_final_attack"] = output[f"{level}_standard_final_attack"].astype(bool)

    output["standard_behavior_engine_version"] = ENGINE_VERSION
    output["standard_behavior_decision_mode"] = decision_mode
    return output


def extract_standard_behavior_events(
    df: pd.DataFrame,
    fps: float,
    level: str,
    pair_key: str = "",
) -> list[dict[str, Any]]:
    """Extract independent chase and attack ethogram events from FSM masks.

    Unlike the legacy four-class segmenter, chase and attack are aggregated
    independently, so an attack beginning during an ongoing chase does not
    split the underlying chase bout.  The four-class API can therefore remain
    backward compatible while this CSV provides the scientific ethogram.
    """
    if df.empty:
        return []
    fps = max(float(fps), 1e-9)
    events: list[dict[str, Any]] = []
    frame_values = pd.to_numeric(df.get("frame", pd.Series(range(len(df)))), errors="coerce").fillna(-1).astype(int).to_numpy()
    for behavior in ("chase", "attack"):
        mask_col = f"{level}_standard_final_{behavior}"
        if mask_col not in df.columns:
            continue
        mask = df[mask_col].fillna(False).astype(bool).to_numpy()
        if behavior == "chase":
            actor_col = f"{level}_standard_chase_actor_id"
            target_col = f"{level}_standard_chase_target_id"
            score_col = f"{level}_standard_chase_score"
            role_col = f"{level}_standard_chase_role_confidence"
            subtype_col = None
        else:
            actor_col = f"{level}_standard_attack_actor_id"
            target_col = f"{level}_standard_attack_target_id"
            score_col = f"{level}_standard_attack_score"
            role_col = f"{level}_standard_attack_role_confidence"
            subtype_col = f"{level}_standard_attack_subtype"
        quality_col = f"{level}_standard_behavior_confidence"
        i = 0
        while i < len(mask):
            if not mask[i]:
                i += 1
                continue
            start = i
            while i + 1 < len(mask) and mask[i + 1]:
                i += 1
            end = i
            segment = df.iloc[start : end + 1]
            actors = pd.to_numeric(
                segment.get(actor_col, pd.Series(-1, index=segment.index)), errors="coerce"
            ).fillna(-1).astype(int).to_numpy()
            targets = pd.to_numeric(
                segment.get(target_col, pd.Series(-1, index=segment.index)), errors="coerce"
            ).fillna(-1).astype(int).to_numpy()
            role_pairs = [
                (int(actor), int(target))
                for actor, target in zip(actors, targets)
                if int(actor) >= 0 and int(target) >= 0
            ]
            if role_pairs:
                counts: Dict[Tuple[int, int], int] = {}
                for pair in role_pairs:
                    counts[pair] = counts.get(pair, 0) + 1
                actor_id, target_id = max(
                    counts, key=lambda pair: (counts[pair], -pair[0], -pair[1])
                )
            else:
                actor_id, target_id = -1, -1
            scores = pd.to_numeric(segment.get(score_col, pd.Series(0.0, index=segment.index)), errors="coerce").fillna(0.0)
            confidence = pd.to_numeric(segment.get(quality_col, pd.Series(0.0, index=segment.index)), errors="coerce").fillna(0.0)
            roles = pd.to_numeric(segment.get(role_col, pd.Series(0.0, index=segment.index)), errors="coerce").fillna(0.0)
            peak_offset = int(np.argmax(scores.to_numpy(dtype=float))) if len(scores) else 0
            subtype = ""
            if subtype_col and subtype_col in segment.columns:
                values = [str(v) for v in segment[subtype_col].tolist() if str(v) not in {"", "none", "nan"}]
                if values:
                    subtype = max(set(values), key=values.count)
            start_frame = int(frame_values[start])
            end_frame = int(frame_values[end])
            events.append({
                "behavior_engine": ENGINE_VERSION,
                "candidate_level": str(level),
                "behavior": behavior,
                "subtype": subtype,
                "pair_key": str(pair_key),
                "actor_id": actor_id,
                "target_id": target_id,
                "role_ambiguous": bool(actor_id < 0 or target_id < 0),
                "start_frame": start_frame,
                "peak_frame": int(frame_values[start + peak_offset]),
                "end_frame": end_frame,
                "start_time_s": start_frame / fps,
                "end_time_s": end_frame / fps,
                "duration_s": (end_frame - start_frame + 1) / fps,
                "mean_score": float(scores.mean()) if len(scores) else 0.0,
                "peak_score": float(scores.max()) if len(scores) else 0.0,
                "mean_behavior_confidence": float(confidence.mean()) if len(confidence) else 0.0,
                "mean_role_confidence": float(roles.mean()) if len(roles) else 0.0,
            })
            i += 1
    events.sort(key=lambda item: (int(item["start_frame"]), str(item["behavior"])))
    for index, event in enumerate(events, start=1):
        event["standard_event_id"] = f"{level[:1].upper()}SB{index:06d}"
    return events
