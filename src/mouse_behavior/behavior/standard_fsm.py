"""Finite-state transitions for standard chase and attack evidence."""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional

import numpy as np

from .standard_evidence import AttackFSMResult, ChaseFSMResult, _slice_max


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
        dynamic_gate_count = int(np.count_nonzero(dynamic_context_gate[recent_start : i + 1]))
        stationary_gate_count = int(np.count_nonzero(stationary_gate[recent_start : i + 1]))
        dynamic_open_ok = bool(
            not require_causal_gate
            or (
                bool(dynamic_gate[i])
                and (dynamic_gate_count >= min_dynamic_gate_frames or bool(impulse_gate[i]))
            )
        )
        stationary_open_ok = bool(
            not require_causal_gate
            or (bool(stationary_gate[i]) and stationary_gate_count >= min_stationary_gate_frames)
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

        if contact[i] >= contact_enter and grapple[i] >= grapple_confirm and stationary_open_ok:
            grapple_streak += 1
        else:
            grapple_streak = max(grapple_streak - 1, 0)
        if open_ok and grapple_streak >= grapple_frames:
            start = (
                candidate_start if candidate_start is not None else max(0, i - grapple_streak + 1)
            )
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


__all__ = [
    "ChaseFSMResult",
    "AttackFSMResult",
    "_run_chase_fsm",
    "_run_attack_fsm",
]
