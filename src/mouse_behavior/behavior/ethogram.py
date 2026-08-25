"""Extended ethogram and contact-event post-processing."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from ..parallel_behavior_fsm import ParallelBehaviorFSM
from ..preprocessing.constants import BEHAVIOR_NAMES_ZH
from ..utils.rolling import rolling_sum as _rolling_sum

LOGGER = logging.getLogger("mouse_behavior.lightweight_behavior_inference")


def _frame_member_ids(
    member_ids_by_frame: Sequence[Iterable[int]] | Mapping[int, Iterable[int]] | None,
    frame: int,
) -> tuple[int, ...]:
    """Read stable group participants from a frame-indexed membership source."""

    if member_ids_by_frame is None:
        return ()
    if isinstance(member_ids_by_frame, Mapping):
        values = member_ids_by_frame.get(int(frame), ())
    elif 0 <= int(frame) < len(member_ids_by_frame):
        values = member_ids_by_frame[int(frame)]
    else:
        values = ()
    result: set[int] = set()
    iterable = () if values is None else values
    for value in iterable:
        try:
            number = int(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if number >= 0:
            result.add(number)
    return tuple(sorted(result))


def _confirmed_mask(
    mask: np.ndarray,
    *,
    scope: str,
    region_id: str,
    behavior: str,
    min_duration_frames: int,
    max_gap_frames: int,
    fsm_coordinator: ParallelBehaviorFSM,
) -> np.ndarray:
    """Return the frames that the regular FSM would have emitted.

    Short-recovery channels need to distinguish a one-frame candidate from a
    normal confirmed event.  Reusing the same coordinator keeps the recovery
    path subject to the global FSM enable switch and the exact gap policy.
    """

    result = fsm_coordinator.run_boolean_region(
        scope=scope,
        region_id=f"{region_id}:confirmed",
        behavior=f"{behavior}:confirmed",
        mask=mask,
        min_duration_frames=min_duration_frames,
        max_gap_frames=max_gap_frames,
    )
    confirmed = np.zeros(np.asarray(mask, dtype=bool).shape, dtype=bool)
    for span in result.spans:
        confirmed[int(span.start) : int(span.end) + 1] = True
    return confirmed


def _require_temporal_support(
    candidate: np.ndarray,
    support: np.ndarray,
    *,
    min_support_frames: int,
    support_window_frames: int,
) -> np.ndarray:
    """Keep candidate frames only when nearby evidence supports the bout.

    A single high score can be caused by a pose glitch or a contact geometry
    spike.  This helper is intentionally local to one candidate pair and does
    not invent positive frames: it only rejects isolated candidates.  The
    caller may still use the normal FSM gap policy to join two supported
    samples into one public event.
    """

    candidate_mask = np.asarray(candidate, dtype=bool)
    support_mask = np.asarray(support, dtype=bool)
    if candidate_mask.shape != support_mask.shape:
        raise ValueError(
            "candidate and support masks must have the same shape, "
            f"got {candidate_mask.shape} and {support_mask.shape}"
        )
    minimum = max(int(min_support_frames), 1)
    if minimum <= 1 or not candidate_mask.any():
        return candidate_mask.copy()
    radius = max(int(support_window_frames), 0)
    cumulative = np.concatenate(([0], np.cumsum(support_mask.astype(np.int16))))
    positions = np.arange(len(support_mask))
    left = np.maximum(positions - radius, 0)
    right = np.minimum(positions + radius + 1, len(support_mask))
    counts = cumulative[right] - cumulative[left]
    return candidate_mask & (counts >= minimum)


def _expand_with_temporal_support(
    candidate: np.ndarray,
    support: np.ndarray,
    *,
    min_support_frames: int,
    support_window_frames: int,
) -> np.ndarray:
    """Return a short supported bout without promoting isolated evidence.

    ``candidate`` carries the causal signature.  ``support`` may contain
    adjacent high-dynamic frames whose attack score is slightly below the
    causal threshold.  Once a candidate has at least the required number of
    support frames, those nearby support frames become the event's compact
    core.  This recovers a two-frame attack bout while still rejecting a lone
    spike and never reaching outside the same pair timeline.
    """

    candidate_mask = np.asarray(candidate, dtype=bool)
    support_mask = np.asarray(support, dtype=bool)
    reliable = _require_temporal_support(
        candidate_mask,
        support_mask,
        min_support_frames=min_support_frames,
        support_window_frames=support_window_frames,
    )
    expanded = np.zeros(candidate_mask.shape, dtype=bool)
    radius = max(int(support_window_frames), 0)
    for index in np.flatnonzero(reliable):
        left = max(int(index) - radius, 0)
        right = min(int(index) + radius + 1, len(expanded))
        expanded[left:right] |= support_mask[left:right]
    return expanded


def _event_rows_from_mask(
    mask: np.ndarray,
    *,
    behavior: str,
    level: str,
    fps: float,
    source_video: Path,
    sample_stride: int,
    score: np.ndarray | None = None,
    actor_id: np.ndarray | None = None,
    target_id: np.ndarray | None = None,
    pair_key: str = "",
    min_duration_seconds: float = 0.15,
    fill_gap_seconds: float = 0.10,
    event_scope: str = "pair",
    fsm_coordinator: ParallelBehaviorFSM | None = None,
    member_ids_by_frame: Sequence[Iterable[int]] | Mapping[int, Iterable[int]] | None = None,
    short_event_padding_seconds: float = 0.0,
    short_event_max_duration_seconds: float = 0.0,
    event_recovery: str | None = None,
) -> list[dict[str, Any]]:
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    min_frames = max(int(round(float(min_duration_seconds) * fps)), 1)
    gap_frames = max(int(round(float(fill_gap_seconds) * fps)), 0)
    score_values = np.asarray(score if score is not None else mask.astype(float), dtype=float)
    actor_values = np.asarray(
        actor_id if actor_id is not None else np.full(mask.shape, -1), dtype=int
    )
    target_values = np.asarray(
        target_id if target_id is not None else np.full(mask.shape, -1), dtype=int
    )
    events: list[dict[str, Any]] = []
    coordinator = fsm_coordinator or ParallelBehaviorFSM()
    fsm_result = coordinator.run_boolean_region(
        scope=str(event_scope),
        region_id=str(pair_key or event_scope),
        behavior=str(behavior),
        mask=mask,
        min_duration_frames=min_frames,
        max_gap_frames=gap_frames,
    )
    for span in fsm_result.spans:
        start, end = int(span.start), int(span.end)
        segment = score_values[start : end + 1]
        finite_segment = segment[np.isfinite(segment)]
        peak_offset = int(np.nanargmax(segment)) if finite_segment.size else 0
        peak = start + peak_offset
        actor = int(actor_values[peak]) if actor_values.size > peak else -1
        target = int(target_values[peak]) if target_values.size > peak else -1
        core_source_start = int(start * sample_stride)
        source_peak = int(peak * sample_stride)
        core_source_end = int(end * sample_stride)
        padding_frames = max(
            int(np.ceil(max(float(short_event_padding_seconds), 0.0) * float(fps))),
            0,
        )
        max_short_frames = max(
            int(round(max(float(short_event_max_duration_seconds), 0.0) * float(fps))),
            0,
        )
        core_frames = end - start + 1
        public_start = start
        public_end = end
        if padding_frames and max_short_frames and core_frames <= max_short_frames:
            public_start = max(start - padding_frames, 0)
            public_end = min(end + padding_frames, len(mask) - 1)
        source_start = int(public_start * sample_stride)
        source_end = int(public_end * sample_stride)
        member_ids: set[int] = set()
        member_ids_at_peak = _frame_member_ids(member_ids_by_frame, peak)
        for member_frame in range(start, end + 1):
            member_ids.update(_frame_member_ids(member_ids_by_frame, member_frame))
        temporal_padding_frames = (public_end - public_start + 1) - core_frames
        row: dict[str, Any] = {
            "behavior": str(behavior),
            "behavior_name_zh": BEHAVIOR_NAMES_ZH.get(str(behavior), str(behavior)),
            "candidate_level": str(level),
            "behavior_engine": "lightweight_extended_ethogram",
            "event_scope": str(event_scope),
            "pair_key": str(pair_key),
            "actor_id": actor,
            "target_id": target,
            "role_ambiguous": bool(actor < 0 or target < 0),
            # analysis_* are the evidence span.  start/end are the public
            # display/export span and may include bounded temporal context.
            "analysis_start_frame": int(start),
            "analysis_peak_frame": int(peak),
            "analysis_end_frame": int(end),
            "core_start_frame": core_source_start,
            "core_end_frame": core_source_end,
            "temporal_padding_frames": int(temporal_padding_frames),
            "event_recovery": str(event_recovery or "none"),
            "start_frame": source_start,
            "peak_frame": source_peak,
            "end_frame": source_end,
            "start_time_s": source_start / max(float(fps * sample_stride), 1e-9),
            "end_time_s": source_end / max(float(fps * sample_stride), 1e-9),
            "duration_s": (source_end - source_start + 1) / max(float(fps * sample_stride), 1e-9),
            "core_duration_s": (core_source_end - core_source_start + 1)
            / max(float(fps * sample_stride), 1e-9),
            "mean_score": float(np.nanmean(segment)) if finite_segment.size else 0.0,
            "peak_score": float(np.nanmax(segment)) if finite_segment.size else 0.0,
            "source_video": str(source_video),
            "analysis_mode": "lightweight_cache_tracking",
        }
        if member_ids:
            row["member_ids"] = sorted(member_ids)
        if member_ids_at_peak:
            row["member_ids_at_peak"] = list(member_ids_at_peak)
        events.append(row)
    return events


def _extended_behavior_config(config: Mapping[str, Any]) -> dict[str, Any]:
    configured = dict(config.get("extended_behavior", {}))
    defaults = {
        "enabled": True,
        "individual": {
            "stationary_max_speed_cm_s": 4.0,
            "walking_max_speed_cm_s": 18.0,
            "running_min_speed_cm_s": 18.0,
            "confirm_seconds": 0.30,
            "fill_gap_seconds": 0.20,
            "min_pose_quality": 0.20,
        },
        "social": {
            "pair_max_distance_cm": 17.0,
            "together_max_distance_cm": 8.0,
            "together_max_combined_speed_cm_s": 28.0,
            "approach_min_distance_drop_cm": 1.5,
            "approach_min_closing_speed_cm_s": 2.0,
            "approach_max_actor_speed_cm_s": 22.0,
            "approach_max_target_speed_cm_s": 22.0,
            "approach_min_speed_gap_cm_s": 1.5,
            "approach_min_actor_speed_cm_s": 3.0,
            "approach_allow_together_transition": True,
            "approach_min_duration_seconds": 0.10,
            "approach_short_event_padding_seconds": 0.10,
            "approach_short_event_max_duration_seconds": 0.35,
            "chase_fallback": {
                "enabled": True,
                "max_distance_cm": 12.0,
                "min_score": 0.70,
                "min_actor_speed_cm_s": 3.0,
                "min_target_speed_cm_s": 2.5,
                "min_pursuit_alignment": 0.40,
                "min_target_escape_alignment": 0.30,
                "min_direction_similarity": 0.55,
                "min_role_confidence": 0.04,
                "min_duration_seconds": 0.25,
                "fill_gap_seconds": 0.15,
                "short_event_padding_seconds": 0.10,
                "short_event_max_duration_seconds": 0.60,
            },
            "attack_fallback": {
                "enabled": True,
                "max_distance_cm": 7.6,
                "min_dynamic_score": 0.78,
                "min_context_frames": 1,
                "min_attack_evidence": 3,
                "min_raw_actor_speed_cm_s": 8.0,
                "min_impact_pursuit_alignment": 0.55,
                "max_impact_target_escape_alignment": 0.10,
                "min_impact_behavior_speed_gap_cm_s": 1.5,
                "min_impact_actor_acceleration_cm_s2": 120.0,
                "min_impact_initiation_score": 0.90,
                "min_rebound_target_escape_alignment": 0.55,
                "max_rebound_pursuit_alignment": 0.20,
                "min_rebound_post_distance_increase_cm": 0.50,
                "min_rebound_immediate_distance_increase_cm": 0.25,
                "rebound_distance_window_seconds": 0.10,
                "min_rebound_reaction_score": 0.70,
                "min_rebound_contact_pursuit_alignment": 0.55,
                "min_target_turn_angle_deg": 25.0,
                "min_target_nose_speed_cm_s": 12.0,
                "min_role_confidence": 0.04,
                # A confirmed attack must have at least two analyzed samples.
                # A one-sample spike is retained only as diagnostic evidence,
                # never as an attack event.
                "min_confirmed_frames": 2,
                "support_window_seconds": 0.20,
                "min_duration_seconds": 0.033,
                "fill_gap_seconds": 0.10,
                "short_event_padding_seconds": 0.15,
                "short_event_max_duration_seconds": 0.40,
                "short_recovery": {
                    "enabled": True,
                    "min_attack_score": 0.80,
                    "max_distance_cm": 9.0,
                    "min_raw_actor_speed_cm_s": 8.0,
                    "min_attack_evidence": 2,
                    "min_role_confidence": 0.04,
                    "min_support_frames": 2,
                    "support_window_seconds": 0.20,
                    "min_support_attack_score": 0.70,
                    "min_support_dynamic_score": 0.65,
                    "min_support_raw_actor_speed_cm_s": 6.0,
                    "min_escape_alignment": 0.65,
                    "min_pursuit_alignment": 0.60,
                    "min_initiation_score": 0.85,
                    "max_rebound_pursuit_alignment": 0.10,
                    "min_reaction_score": 0.70,
                    "min_duration_seconds": 0.033,
                },
            },
            "avoidance_min_target_escape_alignment": 0.45,
            "avoidance_min_target_speed_cm_s": 1.5,
            "avoidance_min_distance_increase_cm": 1.0,
            "avoidance_min_actor_speed_cm_s": 1.5,
            "avoidance_max_distance_cm": 17.0,
            "avoidance_require_pursuit_context": True,
            "avoidance_context_seconds": 1.00,
            "avoidance_close_context_distance_cm": 10.0,
            "avoidance_min_pursuit_alignment": 0.35,
            "avoidance_min_prior_distance_drop_cm": 0.50,
            "avoidance_min_duration_seconds": 0.15,
            "avoidance_short_event_padding_seconds": 0.10,
            "avoidance_short_event_max_duration_seconds": 0.35,
            "avoidance_short_recovery": {
                "enabled": True,
                "max_distance_cm": 10.0,
                "min_target_escape_alignment": 0.70,
                "min_target_speed_cm_s": 2.0,
                "min_actor_speed_cm_s": 2.0,
                "min_distance_increase_cm": 1.0,
                "min_duration_seconds": 0.033,
            },
            "pair_fill_gap_seconds": 0.15,
        },
        "group": {
            "huddle_distance_cm": 9.0,
            "huddle_fraction": 0.55,
            "huddle_min_cluster_size": 3,
            "huddle_min_cluster_fraction": 0.30,
            "huddle_min_cluster_density": 0.50,
            "isolation_distance_cm": 15.0,
            "isolation_neighbor_fraction": 0.15,
            "confirm_seconds": 0.30,
            "fill_gap_seconds": 0.20,
        },
    }
    for section, values in defaults.items():
        if isinstance(values, dict):
            merged = dict(values)
            if isinstance(configured.get(section), Mapping):
                merged.update(dict(configured[section]))
            configured[section] = merged
    return configured


def _extended_short_clip_pair_events(
    pair_df: pd.DataFrame,
    enriched: pd.DataFrame,
    *,
    source_video: Path,
    source_fps: float,
    sample_stride: int,
    config: Mapping[str, Any],
    fsm_coordinator: ParallelBehaviorFSM | None = None,
) -> list[dict[str, Any]]:
    """Recover short high-evidence chase/attack bouts missed by the legacy FSM.

    The standard engine intentionally requires a quality gate and a longer
    causal confirmation window.  Some Beiyi examples are shorter than that
    window, so this fallback is narrow and emits ``candidate_level=extended``:
    chase still needs directional motion; attack is restricted to an impact or
    post-contact rebound pattern with direction-consistent evidence.  A
    nose-head/nose-tail contact alone cannot satisfy these gates.
    """
    if pair_df.empty or enriched.empty:
        return []
    cfg = _extended_behavior_config(config)
    social = dict(cfg["social"])
    chase_cfg = dict(social.get("chase_fallback", {}))
    attack_cfg = dict(social.get("attack_fallback", {}))
    valid = (
        pair_df.get("valid_pair", pd.Series(True, index=pair_df.index))
        .fillna(False)
        .astype(bool)
        .to_numpy()
    )
    n = len(pair_df)
    fsm_coordinator = fsm_coordinator or ParallelBehaviorFSM(
        dict(config.get("parallel_behavior_fsm", {}))
    )

    def pair_values(column: str, default: float = 0.0) -> np.ndarray:
        values = pair_df[column] if column in pair_df else pd.Series(default, index=pair_df.index)
        return pd.to_numeric(values, errors="coerce").fillna(default).to_numpy(float)

    def engine_values(column: str, default: float = 0.0) -> np.ndarray:
        values = (
            enriched[column] if column in enriched else pd.Series(default, index=enriched.index)
        )
        return pd.to_numeric(values, errors="coerce").fillna(default).to_numpy(float)

    distance = pair_values("center_distance_cm", np.inf)
    actor_speed = pair_values("selected_actor_behavior_speed_cm_s")
    target_speed = pair_values("selected_target_behavior_speed_cm_s")
    actor_raw_speed = pair_values("selected_actor_speed_cm_s")
    pursuit = pair_values("selected_actor_pursuit_alignment")
    escape = pair_values("selected_target_escape_alignment")
    selected_actor = (
        pd.to_numeric(pair_df.get("selected_actor_id", -1), errors="coerce")
        .fillna(-1)
        .to_numpy(int)
    )
    selected_target = (
        pd.to_numeric(pair_df.get("selected_target_id", -1), errors="coerce")
        .fillna(-1)
        .to_numpy(int)
    )
    mouse_a = pd.to_numeric(pair_df.get("mouse_a_id", -2), errors="coerce").fillna(-2).to_numpy(int)
    mouse_b = pd.to_numeric(pair_df.get("mouse_b_id", -2), errors="coerce").fillna(-2).to_numpy(int)
    selected_ab = selected_actor == mouse_a
    selected_ba = selected_actor == mouse_b
    direction_known = selected_ab | selected_ba
    speed_gap = actor_speed - target_speed
    target_turn = np.where(
        selected_ab,
        pair_values("a_to_b_target_turn_angle_deg"),
        pair_values("b_to_a_target_turn_angle_deg"),
    )
    target_nose_speed = np.where(
        selected_ab,
        pair_values("a_to_b_target_nose_speed_cm_s"),
        pair_values("b_to_a_target_nose_speed_cm_s"),
    )
    actor_acceleration = np.where(
        selected_ab,
        pair_values("a_to_b_actor_acceleration_cm_s2"),
        pair_values("b_to_a_actor_acceleration_cm_s2"),
    )
    direction_similarity = np.where(
        selected_ab,
        pair_values("a_to_b_direction_similarity"),
        pair_values("b_to_a_direction_similarity"),
    )
    nose_head = np.minimum(
        pair_values("a_to_b_nose_head_distance_cm", np.inf),
        pair_values("b_to_a_nose_head_distance_cm", np.inf),
    )
    nose_tail = np.minimum(
        pair_values("a_to_b_nose_tail_distance_cm", np.inf),
        pair_values("b_to_a_nose_tail_distance_cm", np.inf),
    )
    nose_body = np.minimum(
        pair_values("a_to_b_nose_body_distance_cm", np.inf),
        pair_values("b_to_a_nose_body_distance_cm", np.inf),
    )
    contact_cfg = dict(config.get("contact_detection", {}))
    contact = (
        (
            nose_head
            <= float(contact_cfg.get("nose_head_distance_cm", contact_cfg.get("distance_cm", 3.0)))
        )
        | (
            nose_tail
            <= float(contact_cfg.get("nose_tail_distance_cm", contact_cfg.get("distance_cm", 3.0)))
        )
        | (nose_body <= float(contact_cfg.get("distance_cm", 3.0)))
    )
    dynamic_score = np.maximum(
        engine_values("weak_standard_dynamic_attack_score"),
        engine_values("strong_standard_dynamic_attack_score"),
    )
    attack_score = np.maximum(
        engine_values("weak_standard_attack_score"),
        engine_values("strong_standard_attack_score"),
    )
    evidence_count = np.maximum(
        engine_values("weak_standard_attack_evidence_count"),
        engine_values("strong_standard_attack_evidence_count"),
    )
    role_confidence = np.maximum(
        engine_values("weak_standard_attack_role_confidence"),
        engine_values("strong_standard_attack_role_confidence"),
    )
    initiation = np.maximum(
        engine_values("weak_standard_initiation_score"),
        engine_values("strong_standard_initiation_score"),
    )
    reaction = np.maximum(
        engine_values("weak_standard_reaction_score"),
        engine_values("strong_standard_reaction_score"),
    )
    analysis_fps = source_fps / max(sample_stride, 1)
    contact_pursuit = pair_values("selected_actor_pursuit_alignment")
    contact_mask = (
        (
            np.minimum(
                pair_values("a_to_b_nose_head_distance_cm", np.inf),
                pair_values("b_to_a_nose_head_distance_cm", np.inf),
            )
            <= float(contact_cfg.get("nose_head_distance_cm", contact_cfg.get("distance_cm", 3.0)))
        )
        | (
            np.minimum(
                pair_values("a_to_b_nose_tail_distance_cm", np.inf),
                pair_values("b_to_a_nose_tail_distance_cm", np.inf),
            )
            <= float(contact_cfg.get("nose_tail_distance_cm", contact_cfg.get("distance_cm", 3.0)))
        )
        | (
            np.minimum(
                pair_values("a_to_b_nose_body_distance_cm", np.inf),
                pair_values("b_to_a_nose_body_distance_cm", np.inf),
            )
            <= float(contact_cfg.get("distance_cm", 3.0))
        )
    )
    contact_direction_pursuit = contact_pursuit.copy()
    for index in range(n):
        if not contact_mask[index]:
            continue
        start = max(0, index - max(int(round(0.10 * analysis_fps)), 1))
        finite = contact_pursuit[start : index + 1]
        finite = finite[np.isfinite(finite)]
        contact_direction_pursuit[index] = float(np.max(finite)) if finite.size else 0.0
    events: list[dict[str, Any]] = []
    pair_key = str(pair_df["pair_key"].iloc[0])

    if bool(chase_cfg.get("enabled", True)):
        chase_score = np.maximum(
            engine_values("weak_standard_chase_score"),
            engine_values("strong_standard_chase_score"),
        )
        chase_mask = (
            valid
            & direction_known
            & np.isfinite(distance)
            & (distance <= float(chase_cfg.get("max_distance_cm", 12.0)))
            & (chase_score >= float(chase_cfg.get("min_score", 0.70)))
            & (actor_speed >= float(chase_cfg.get("min_actor_speed_cm_s", 1.5)))
            & (target_speed >= float(chase_cfg.get("min_target_speed_cm_s", 1.5)))
            & (pursuit >= float(chase_cfg.get("min_pursuit_alignment", 0.40)))
            & (escape >= float(chase_cfg.get("min_target_escape_alignment", 0.30)))
            & (direction_similarity >= float(chase_cfg.get("min_direction_similarity", 0.55)))
        )
        events.extend(
            _event_rows_from_mask(
                chase_mask,
                behavior="chase",
                level="extended",
                fps=analysis_fps,
                source_video=source_video,
                sample_stride=sample_stride,
                score=chase_score,
                actor_id=selected_actor,
                target_id=selected_target,
                pair_key=pair_key,
                min_duration_seconds=float(chase_cfg.get("min_duration_seconds", 0.25)),
                fill_gap_seconds=float(chase_cfg.get("fill_gap_seconds", 0.15)),
                short_event_padding_seconds=float(
                    chase_cfg.get("short_event_padding_seconds", 0.10)
                ),
                short_event_max_duration_seconds=float(
                    chase_cfg.get("short_event_max_duration_seconds", 0.60)
                ),
                fsm_coordinator=fsm_coordinator,
            )
        )

    if bool(attack_cfg.get("enabled", True)):
        attack_common = (
            valid
            & direction_known
            & np.isfinite(distance)
            & (distance <= float(attack_cfg.get("max_distance_cm", 7.6)))
            & contact
            & (dynamic_score >= float(attack_cfg.get("min_dynamic_score", 0.78)))
            & (evidence_count >= float(attack_cfg.get("min_attack_evidence", 3)))
            & (role_confidence >= float(attack_cfg.get("min_role_confidence", 0.04)))
        )
        # A rebound is allowed to occur a few frames after the collision.  It
        # is deliberately computed from pair distance rather than from the
        # selected direction, because an impact can reverse the actor/target
        # velocity ordering in the next frame.
        post_window = max(
            int(
                round(float(attack_cfg.get("rebound_distance_window_seconds", 0.10)) * analysis_fps)
            ),
            1,
        )
        post_distance_increase = np.zeros(n, dtype=float)
        immediate_post_distance_increase = np.full(n, -np.inf, dtype=float)
        for offset in range(1, post_window + 1):
            if offset >= n:
                break
            future = np.full(n, np.nan, dtype=float)
            future[:-offset] = distance[offset:]
            with np.errstate(invalid="ignore"):
                candidate = future - distance
            candidate[~np.isfinite(candidate)] = -np.inf
            post_distance_increase = np.maximum(post_distance_increase, candidate)
            if offset == 1:
                immediate_post_distance_increase = candidate

        # Impact-type attack: the attacker has a short, high-energy approach
        # while the target is not escaping in the same smooth direction.  This
        # rejects ordinary nose-tail contact, which has positive pursuit and
        # positive target escape over a sustained approach.
        impact_attack = (
            attack_common
            & (actor_raw_speed >= float(attack_cfg.get("min_raw_actor_speed_cm_s", 8.0)))
            & (pursuit >= float(attack_cfg.get("min_impact_pursuit_alignment", 0.55)))
            & (escape <= float(attack_cfg.get("max_impact_target_escape_alignment", 0.10)))
            & (speed_gap >= float(attack_cfg.get("min_impact_behavior_speed_gap_cm_s", 1.5)))
            & (
                (
                    actor_acceleration
                    >= float(attack_cfg.get("min_impact_actor_acceleration_cm_s2", 120.0))
                )
                | (initiation >= float(attack_cfg.get("min_impact_initiation_score", 0.90)))
            )
        )

        # Rebound-type attack: the pair separates immediately after contact,
        # and the selected direction reverses or the target shows a reaction.
        # This is the short terminal pattern in the second attack example.
        rebound_attack = (
            attack_common
            & (actor_raw_speed >= float(attack_cfg.get("min_raw_actor_speed_cm_s", 8.0)))
            & (escape >= float(attack_cfg.get("min_rebound_target_escape_alignment", 0.55)))
            & (pursuit <= float(attack_cfg.get("max_rebound_pursuit_alignment", 0.20)))
            & (
                contact_direction_pursuit
                >= float(attack_cfg.get("min_rebound_contact_pursuit_alignment", 0.55))
            )
            & (
                post_distance_increase
                >= float(attack_cfg.get("min_rebound_post_distance_increase_cm", 0.50))
            )
            & (
                immediate_post_distance_increase
                >= float(attack_cfg.get("min_rebound_immediate_distance_increase_cm", 0.25))
            )
            & (
                (reaction >= float(attack_cfg.get("min_rebound_reaction_score", 0.70)))
                | (target_turn >= float(attack_cfg.get("min_target_turn_angle_deg", 25.0)))
                | (target_nose_speed >= float(attack_cfg.get("min_target_nose_speed_cm_s", 12.0)))
            )
        )
        attack_mask = impact_attack | rebound_attack
        attack_min_confirmed_frames = max(
            int(attack_cfg.get("min_confirmed_frames", 2)),
            2,
        )
        attack_support_window_frames = max(
            int(round(float(attack_cfg.get("support_window_seconds", 0.20)) * analysis_fps / 2.0)),
            1,
        )
        attack_mask = _require_temporal_support(
            attack_mask,
            attack_mask,
            min_support_frames=attack_min_confirmed_frames,
            support_window_frames=attack_support_window_frames,
        )
        attack_min_duration_seconds = max(
            float(attack_cfg.get("min_duration_seconds", 0.08)),
            attack_min_confirmed_frames / max(analysis_fps, 1e-9),
        )
        events.extend(
            _event_rows_from_mask(
                attack_mask,
                behavior="attack",
                level="extended",
                fps=analysis_fps,
                source_video=source_video,
                sample_stride=sample_stride,
                score=attack_score,
                actor_id=selected_actor,
                target_id=selected_target,
                pair_key=pair_key,
                min_duration_seconds=attack_min_duration_seconds,
                fill_gap_seconds=float(attack_cfg.get("fill_gap_seconds", 0.10)),
                short_event_padding_seconds=float(
                    attack_cfg.get("short_event_padding_seconds", 0.15)
                ),
                short_event_max_duration_seconds=float(
                    attack_cfg.get("short_event_max_duration_seconds", 0.40)
                ),
                fsm_coordinator=fsm_coordinator,
            )
        )

        # Very short Beiyi attack examples can contain a complete contact and
        # a high attack score in one or two frames, but not enough frames for
        # the causal impact/rebound FSM to reach ACTIVE.  This recovery gate
        # is deliberately narrower than the normal fallback: it still needs
        # contact geometry, dynamic/evidence/role support, high raw actor
        # speed, and one of two interpretable signatures (initiated impact or
        # target reaction after a rebound).  It never uses the video name.
        short_cfg = dict(attack_cfg.get("short_recovery", {}))
        if bool(short_cfg.get("enabled", True)):
            short_attack_common = (
                valid
                & direction_known
                & np.isfinite(distance)
                & (distance <= float(short_cfg.get("max_distance_cm", 9.0)))
                & contact
                & (attack_score >= float(short_cfg.get("min_attack_score", 0.80)))
                & (dynamic_score >= float(attack_cfg.get("min_dynamic_score", 0.78)))
                & (evidence_count >= float(short_cfg.get("min_attack_evidence", 3)))
                & (role_confidence >= float(short_cfg.get("min_role_confidence", 0.04)))
                & (actor_raw_speed >= float(short_cfg.get("min_raw_actor_speed_cm_s", 8.0)))
            )
            # A neighboring frame may have the same contact and dynamic
            # evidence while its attack score is just below the causal gate.
            # It can support a short bout, but it cannot open the bout by
            # itself.  This is deliberately narrower than ordinary contact.
            short_support_attack_score = float(short_cfg.get("min_support_attack_score", 0.70))
            short_support_dynamic_score = float(
                short_cfg.get(
                    "min_support_dynamic_score",
                    attack_cfg.get("min_dynamic_score", 0.78),
                )
            )
            short_support_raw_actor_speed = float(
                short_cfg.get(
                    "min_support_raw_actor_speed_cm_s",
                    max(float(short_cfg.get("min_raw_actor_speed_cm_s", 8.0)) - 2.0, 0.0),
                )
            )
            short_attack_support = (
                valid
                & direction_known
                & np.isfinite(distance)
                & (distance <= float(short_cfg.get("max_distance_cm", 9.0)))
                & contact
                & (attack_score >= short_support_attack_score)
                & (dynamic_score >= short_support_dynamic_score)
                & (evidence_count >= float(short_cfg.get("min_attack_evidence", 3)))
                & (role_confidence >= float(short_cfg.get("min_role_confidence", 0.04)))
                & (actor_raw_speed >= short_support_raw_actor_speed)
            )
            short_impact = (
                (pursuit >= float(short_cfg.get("min_pursuit_alignment", 0.60)))
                & (escape >= float(short_cfg.get("min_escape_alignment", 0.65)))
                & (initiation >= float(short_cfg.get("min_initiation_score", 0.85)))
            )
            short_rebound = (
                (pursuit <= float(short_cfg.get("max_rebound_pursuit_alignment", 0.10)))
                & (escape >= float(short_cfg.get("min_escape_alignment", 0.65)))
                & (reaction >= float(short_cfg.get("min_reaction_score", 0.70)))
            )
            short_attack = short_attack_common & (short_impact | short_rebound)
            short_min_support_frames = max(
                int(
                    short_cfg.get(
                        "min_support_frames",
                        attack_cfg.get("min_confirmed_frames", 2),
                    )
                ),
                2,
            )
            short_support_window_frames = max(
                int(
                    round(
                        float(
                            short_cfg.get(
                                "support_window_seconds",
                                attack_cfg.get("support_window_seconds", 0.20),
                            )
                        )
                        * analysis_fps
                        / 2.0
                    )
                ),
                1,
            )
            short_attack = _expand_with_temporal_support(
                short_attack,
                short_attack_support,
                min_support_frames=short_min_support_frames,
                support_window_frames=short_support_window_frames,
            )
            confirmed_attack = _confirmed_mask(
                attack_mask,
                scope="pair",
                region_id=pair_key,
                behavior="attack",
                min_duration_frames=max(int(round(attack_min_duration_seconds * analysis_fps)), 1),
                max_gap_frames=max(
                    int(round(float(attack_cfg.get("fill_gap_seconds", 0.10)) * analysis_fps)),
                    0,
                ),
                fsm_coordinator=fsm_coordinator,
            )
            short_attack &= ~confirmed_attack
            events.extend(
                _event_rows_from_mask(
                    short_attack,
                    behavior="attack",
                    level="extended",
                    fps=analysis_fps,
                    source_video=source_video,
                    sample_stride=sample_stride,
                    score=attack_score,
                    actor_id=selected_actor,
                    target_id=selected_target,
                    pair_key=pair_key,
                    min_duration_seconds=max(
                        float(short_cfg.get("min_duration_seconds", 0.033)),
                        short_min_support_frames / max(analysis_fps, 1e-9),
                    ),
                    fill_gap_seconds=0.0,
                    short_event_padding_seconds=float(
                        attack_cfg.get("short_event_padding_seconds", 0.15)
                    ),
                    short_event_max_duration_seconds=float(
                        attack_cfg.get("short_event_max_duration_seconds", 0.40)
                    ),
                    event_recovery="short_high_evidence",
                    fsm_coordinator=fsm_coordinator,
                )
            )
    return events


def _extended_pair_events(
    pair_df: pd.DataFrame,
    *,
    metrics: Mapping[str, Any],
    pair_index: int,
    enriched: pd.DataFrame | None = None,
    source_video: Path,
    source_fps: float,
    sample_stride: int,
    config: Mapping[str, Any],
    fsm_coordinator: ParallelBehaviorFSM | None = None,
) -> list[dict[str, Any]]:
    """Infer social labels that are not part of the legacy chase/attack FSM.

    Approach is intentionally slower and weaker than chase: it requires a
    measurable distance decrease but rejects the sustained two-mouse speed and
    pursuit evidence used by chase.  Avoidance is the opposite directional
    pattern: a moving target escapes and the pair distance increases after a
    close interaction.  Contact events remain independent and are added by the
    caller.
    """
    if pair_df.empty:
        return []
    cfg = _extended_behavior_config(config)
    social = dict(cfg["social"])
    n = len(pair_df)
    analysis_fps = source_fps / max(sample_stride, 1)
    fsm_coordinator = fsm_coordinator or ParallelBehaviorFSM(
        dict(config.get("parallel_behavior_fsm", {}))
    )
    distance = pd.to_numeric(pair_df["center_distance_cm"], errors="coerce").to_numpy(float)
    valid = (
        pair_df.get("valid_pair", pd.Series(True, index=pair_df.index))
        .fillna(False)
        .astype(bool)
        .to_numpy()
    )
    actor_speed = (
        pd.to_numeric(pair_df.get("selected_actor_behavior_speed_cm_s", 0.0), errors="coerce")
        .fillna(0)
        .to_numpy(float)
    )
    target_speed = (
        pd.to_numeric(pair_df.get("selected_target_behavior_speed_cm_s", 0.0), errors="coerce")
        .fillna(0)
        .to_numpy(float)
    )
    combined_speed = actor_speed + target_speed
    drop = (
        pd.to_numeric(pair_df.get("selected_distance_drop_cm", 0.0), errors="coerce")
        .fillna(0)
        .to_numpy(float)
    )
    closing = (
        pd.to_numeric(pair_df.get("selected_closing_speed_cm_s", 0.0), errors="coerce")
        .fillna(0)
        .to_numpy(float)
    )
    # The selected direction is the one used by the lightweight pair row.
    selected_ab = (
        pair_df.get("selected_actor_id", pd.Series(-1, index=pair_df.index)).to_numpy()
        == pair_df.get("mouse_a_id", pd.Series(-2, index=pair_df.index)).to_numpy()
    )
    selected_escape = np.where(
        selected_ab,
        pd.to_numeric(pair_df.get("a_to_b_target_escape_alignment", 0.0), errors="coerce")
        .fillna(0)
        .to_numpy(float),
        pd.to_numeric(pair_df.get("b_to_a_target_escape_alignment", 0.0), errors="coerce")
        .fillna(0)
        .to_numpy(float),
    )
    selected_actor_speed = actor_speed
    selected_target_speed = target_speed
    selected_actor = (
        pd.to_numeric(pair_df.get("selected_actor_id", -1), errors="coerce")
        .fillna(-1)
        .to_numpy(int)
    )
    selected_target = (
        pd.to_numeric(pair_df.get("selected_target_id", -1), errors="coerce")
        .fillna(-1)
        .to_numpy(int)
    )

    def pair_numeric(column: str, default: float) -> np.ndarray:
        values = pair_df[column] if column in pair_df else pd.Series(default, index=pair_df.index)
        return pd.to_numeric(values, errors="coerce").fillna(default).to_numpy(float)

    # Nose contact and approach remain independent event streams. Approach is
    # not masked by contact geometry because doing so would change the existing
    # scientific output contract without labeled-data acceptance evidence.

    # ``together`` is a low-motion, close pair state. It does not require a
    # contact threshold, so it can represent the labeled together examples.
    together = (
        valid
        & np.isfinite(distance)
        & (distance <= float(social["together_max_distance_cm"]))
        & (combined_speed <= float(social["together_max_combined_speed_cm_s"]))
    )
    approach = (
        valid
        & np.isfinite(distance)
        & (distance <= float(social["pair_max_distance_cm"]))
        & (drop >= float(social["approach_min_distance_drop_cm"]))
        & (closing >= float(social["approach_min_closing_speed_cm_s"]))
        & (selected_actor_speed >= float(social.get("approach_min_actor_speed_cm_s", 0.0)))
        & (selected_actor_speed <= float(social["approach_max_actor_speed_cm_s"]))
        & (selected_target_speed <= float(social["approach_max_target_speed_cm_s"]))
        & (
            (selected_actor_speed - selected_target_speed)
            >= float(social["approach_min_speed_gap_cm_s"])
        )
    )
    # Keep the approach-to-together transition: the last approach samples can
    # be close and low-motion after the speed difference collapses.  Contact
    # geometry remains excluded, so this does not turn nose contact into
    # approach.
    if not bool(social.get("approach_allow_together_transition", True)):
        approach &= ~together
    distance_increase = np.zeros(n, dtype=float)
    lookback = max(int(round(source_fps / max(sample_stride, 1) * 0.25)), 1)
    if lookback < n:
        distance_increase[lookback:] = distance[lookback:] - distance[:-lookback]
    avoidance = (
        valid
        & np.isfinite(distance)
        & (distance <= float(social["avoidance_max_distance_cm"]))
        & (selected_target_speed >= float(social["avoidance_min_target_speed_cm_s"]))
        & (selected_actor_speed >= float(social["avoidance_min_actor_speed_cm_s"]))
        & (selected_escape >= float(social["avoidance_min_target_escape_alignment"]))
        & (distance_increase >= float(social["avoidance_min_distance_increase_cm"]))
    )

    # Avoidance is a reaction after an interaction, not arbitrary separation
    # of two moving mice.  Require prior pursuit/approach and a prior close
    # state within a causal window.  Legacy chase/attack masks are context
    # only; the actual avoidance frame still has to satisfy the directional
    # escape conditions above.
    context_frames = max(
        int(round(float(social["avoidance_context_seconds"]) * source_fps / max(sample_stride, 1))),
        1,
    )
    prior_close = np.zeros(n, dtype=bool)
    close_context = np.isfinite(distance) & (
        distance <= float(social["avoidance_close_context_distance_cm"])
    )
    if n > 1:
        prior_close[1:] = _rolling_sum(close_context[:-1].astype(float), context_frames) > 0

    pursuit_context = approach.copy()
    if enriched is not None and not enriched.empty:
        for column in (
            "weak_standard_final_chase",
            "strong_standard_final_chase",
            "weak_standard_final_attack",
            "strong_standard_final_attack",
        ):
            if column in enriched:
                pursuit_context |= enriched[column].fillna(False).astype(bool).to_numpy()
    prior_pursuit = np.zeros(n, dtype=bool)
    if n > 1:
        prior_pursuit[1:] = _rolling_sum(pursuit_context[:-1].astype(float), context_frames) > 0
    prior_drop = np.zeros(n, dtype=float)
    if n > 1:
        prior_drop[1:] = np.maximum(drop[:-1], 0.0)
    pursuit_alignment_context = np.maximum(
        pair_numeric("selected_actor_pursuit_alignment", 0.0),
        pair_numeric("selected_closing_speed_cm_s", 0.0)
        / max(float(social.get("approach_min_closing_speed_cm_s", 2.0)), 1e-6),
    )
    if bool(social.get("avoidance_require_pursuit_context", True)):
        avoidance &= prior_close & (
            prior_pursuit
            | (
                pursuit_alignment_context
                >= float(social.get("avoidance_min_pursuit_alignment", 0.35))
            )
            | (prior_drop >= float(social.get("avoidance_min_prior_distance_drop_cm", 0.50)))
        )

    # A labeled approach is a low-speed social transition.  Contact samples
    # were removed above, so nose-head/nose-tail events remain independent and
    # cannot masquerade as approach or attack.  The final approach-to-together
    # transition is retained when explicitly enabled because it is the part
    # most often lost in short Beiyi approach clips.
    if not bool(social.get("approach_allow_together_transition", True)):
        approach &= ~together
    events: list[dict[str, Any]] = []
    events.extend(
        _event_rows_from_mask(
            together,
            behavior="together",
            level="extended",
            fps=source_fps / max(sample_stride, 1),
            source_video=source_video,
            sample_stride=sample_stride,
            score=np.where(
                together,
                1.0
                - np.clip(distance / max(float(social["together_max_distance_cm"]), 1e-6), 0, 1),
                0.0,
            ),
            actor_id=np.full(n, -1),
            target_id=np.full(n, -1),
            pair_key=str(pair_df["pair_key"].iloc[0]),
            min_duration_seconds=0.30,
            fill_gap_seconds=float(social["pair_fill_gap_seconds"]),
            fsm_coordinator=fsm_coordinator,
        )
    )
    events.extend(
        _event_rows_from_mask(
            approach,
            behavior="approach",
            level="extended",
            fps=source_fps / max(sample_stride, 1),
            source_video=source_video,
            sample_stride=sample_stride,
            score=np.where(
                approach,
                np.clip(drop / max(float(social["approach_min_distance_drop_cm"]), 1e-6), 0, 1),
                0.0,
            ),
            actor_id=selected_actor,
            target_id=selected_target,
            pair_key=str(pair_df["pair_key"].iloc[0]),
            min_duration_seconds=float(social["approach_min_duration_seconds"]),
            fill_gap_seconds=float(social["pair_fill_gap_seconds"]),
            short_event_padding_seconds=float(
                social.get("approach_short_event_padding_seconds", 0.10)
            ),
            short_event_max_duration_seconds=float(
                social.get("approach_short_event_max_duration_seconds", 0.35)
            ),
            fsm_coordinator=fsm_coordinator,
        )
    )
    avoidance_min_duration_seconds = float(social["avoidance_min_duration_seconds"])
    events.extend(
        _event_rows_from_mask(
            avoidance,
            behavior="avoidance",
            level="extended",
            fps=source_fps / max(sample_stride, 1),
            source_video=source_video,
            sample_stride=sample_stride,
            score=np.where(
                avoidance,
                np.clip(
                    distance_increase
                    / max(float(social["avoidance_min_distance_increase_cm"]), 1e-6),
                    0,
                    1,
                ),
                0.0,
            ),
            actor_id=selected_actor,
            target_id=selected_target,
            pair_key=str(pair_df["pair_key"].iloc[0]),
            min_duration_seconds=avoidance_min_duration_seconds,
            fill_gap_seconds=float(social["pair_fill_gap_seconds"]),
            short_event_padding_seconds=float(
                social.get("avoidance_short_event_padding_seconds", 0.10)
            ),
            short_event_max_duration_seconds=float(
                social.get("avoidance_short_event_max_duration_seconds", 0.35)
            ),
            fsm_coordinator=fsm_coordinator,
        )
    )

    # A short avoidance clip can contain only the first visible escape frame.
    # Recover that frame only when it is both close to the interaction and
    # directionally unambiguous; ordinary separation remains rejected.
    avoidance_recovery = dict(social.get("avoidance_short_recovery", {}))
    if bool(avoidance_recovery.get("enabled", True)):
        short_avoidance = (
            valid
            & np.isfinite(distance)
            & (distance <= float(avoidance_recovery.get("max_distance_cm", 10.0)))
            & (selected_target_speed >= float(avoidance_recovery.get("min_target_speed_cm_s", 2.0)))
            & (selected_actor_speed >= float(avoidance_recovery.get("min_actor_speed_cm_s", 2.0)))
            & (
                selected_escape
                >= float(avoidance_recovery.get("min_target_escape_alignment", 0.70))
            )
            & (distance_increase >= float(avoidance_recovery.get("min_distance_increase_cm", 1.0)))
            & prior_close
            & (
                prior_pursuit
                | (
                    pursuit_alignment_context
                    >= float(social.get("avoidance_min_pursuit_alignment", 0.35))
                )
                | (prior_drop >= float(social.get("avoidance_min_prior_distance_drop_cm", 0.50)))
            )
        )
        confirmed_avoidance = _confirmed_mask(
            avoidance,
            scope="pair",
            region_id=str(pair_df["pair_key"].iloc[0]),
            behavior="avoidance",
            min_duration_frames=max(int(round(avoidance_min_duration_seconds * analysis_fps)), 1),
            max_gap_frames=max(
                int(round(float(social["pair_fill_gap_seconds"]) * analysis_fps)),
                0,
            ),
            fsm_coordinator=fsm_coordinator,
        )
        short_avoidance &= ~confirmed_avoidance
        events.extend(
            _event_rows_from_mask(
                short_avoidance,
                behavior="avoidance",
                level="extended",
                fps=source_fps / max(sample_stride, 1),
                source_video=source_video,
                sample_stride=sample_stride,
                score=np.where(
                    short_avoidance,
                    np.clip(
                        distance_increase
                        / max(float(social["avoidance_min_distance_increase_cm"]), 1e-6),
                        0,
                        1,
                    ),
                    0.0,
                ),
                actor_id=selected_actor,
                target_id=selected_target,
                pair_key=str(pair_df["pair_key"].iloc[0]),
                min_duration_seconds=float(avoidance_recovery.get("min_duration_seconds", 0.033)),
                fill_gap_seconds=0.0,
                short_event_padding_seconds=float(
                    social.get("avoidance_short_event_padding_seconds", 0.10)
                ),
                short_event_max_duration_seconds=float(
                    social.get("avoidance_short_event_max_duration_seconds", 0.35)
                ),
                event_recovery="short_escape_evidence",
                fsm_coordinator=fsm_coordinator,
            )
        )
    return events


def _extended_individual_and_group_events(
    kin: Mapping[str, Any],
    *,
    pair_metrics: Mapping[str, Any],
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    source_video: Path,
    source_fps: float,
    sample_stride: int,
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Infer individual and group labels from all tracked mice per frame."""
    cfg = _extended_behavior_config(config)
    individual_cfg = dict(cfg["individual"])
    group_cfg = dict(cfg["group"])
    valid = np.asarray(kin["valid"], dtype=bool)
    speed = np.asarray(kin["behavior_speed"], dtype=float)
    pose_quality = np.asarray(kin["pose_quality"], dtype=float)
    centers = np.asarray(kin["centers_cm"], dtype=float)
    frames, mice = valid.shape
    analysis_fps = source_fps / max(sample_stride, 1)
    events: list[dict[str, Any]] = []
    fsm_coordinator = ParallelBehaviorFSM(dict(config.get("parallel_behavior_fsm", {})))

    stationary = (
        valid
        & (speed <= float(individual_cfg["stationary_max_speed_cm_s"]))
        & (pose_quality >= float(individual_cfg["min_pose_quality"]))
    )
    walking = (
        valid
        & (speed > float(individual_cfg["stationary_max_speed_cm_s"]))
        & (speed < float(individual_cfg["running_min_speed_cm_s"]))
    )
    running = valid & (speed >= float(individual_cfg["running_min_speed_cm_s"]))
    for mouse in range(mice):
        for behavior, mask, score in (
            (
                "stationary",
                stationary[:, mouse],
                np.maximum(
                    0.0,
                    1.0
                    - speed[:, mouse]
                    / max(float(individual_cfg["stationary_max_speed_cm_s"]), 1e-6),
                ),
            ),
            (
                "walking",
                walking[:, mouse],
                np.clip(
                    speed[:, mouse] / max(float(individual_cfg["walking_max_speed_cm_s"]), 1e-6),
                    0.0,
                    1.0,
                ),
            ),
            (
                "running",
                running[:, mouse],
                np.clip(
                    speed[:, mouse] / max(float(individual_cfg["running_min_speed_cm_s"]), 1e-6),
                    0.0,
                    1.0,
                ),
            ),
        ):
            events.extend(
                _event_rows_from_mask(
                    mask,
                    behavior=behavior,
                    level="extended",
                    fps=analysis_fps,
                    source_video=source_video,
                    sample_stride=sample_stride,
                    score=score,
                    actor_id=np.full(frames, mouse),
                    target_id=np.full(frames, -1),
                    pair_key=f"mouse_{mouse}",
                    min_duration_seconds=float(individual_cfg["confirm_seconds"]),
                    fill_gap_seconds=float(individual_cfg["fill_gap_seconds"]),
                    event_scope="individual",
                    fsm_coordinator=fsm_coordinator,
                )
            )

    # Construct a per-frame nearest-neighbour graph.  Distances are already in
    # cm and therefore adapt to each video's learned scale; no video-specific
    # pixel constant is used here.
    close_fraction = np.zeros(frames, dtype=float)
    isolated_fraction = np.zeros(frames, dtype=float)
    group_size = np.zeros(frames, dtype=int)
    largest_cluster_size = np.zeros(frames, dtype=int)
    largest_cluster_fraction = np.zeros(frames, dtype=float)
    largest_cluster_density = np.zeros(frames, dtype=float)
    huddle_members_by_frame: list[tuple[int, ...]] = [() for _ in range(frames)]
    isolation_members_by_frame: list[tuple[int, ...]] = [() for _ in range(frames)]
    for frame in range(frames):
        ids = np.flatnonzero(valid[frame])
        group_size[frame] = len(ids)
        if len(ids) < 2:
            continue
        points = centers[frame, ids]
        distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
        distances[~np.isfinite(distances)] = np.inf
        np.fill_diagonal(distances, np.inf)
        nearest = np.min(distances, axis=1)
        close_fraction[frame] = float(np.mean(nearest <= float(group_cfg["huddle_distance_cm"])))
        isolated_fraction[frame] = float(
            np.mean(nearest >= float(group_cfg["isolation_distance_cm"]))
        )

        # A multi-mouse cage can contain a local huddle while other visible
        # mice remain spread out.  Use connected components of the complete
        # visible-mouse graph instead of requiring every visible mouse to be
        # close to a neighbour.  This is still a group statistic: no
        # single-mouse or two-mouse clip is fabricated from the video.
        close_threshold = float(group_cfg["huddle_distance_cm"])
        adjacency = distances <= close_threshold
        unseen = set(range(len(ids)))
        components: list[list[int]] = []
        while unseen:
            seed = unseen.pop()
            stack = [seed]
            component: list[int] = []
            while stack:
                current = stack.pop()
                component.append(current)
                neighbours = [item for item in unseen if adjacency[current, item]]
                for item in neighbours:
                    unseen.remove(item)
                    stack.append(item)
            components.append(component)
        largest = max(components, key=len, default=[])
        largest_cluster_size[frame] = len(largest)
        largest_cluster_fraction[frame] = len(largest) / max(len(ids), 1)
        if len(largest) >= 2:
            component_indices = np.asarray(largest, dtype=int)
            huddle_members_by_frame[frame] = tuple(sorted(int(ids[index]) for index in largest))
            component_distances = distances[np.ix_(component_indices, component_indices)]
            edge_count = int(np.sum(component_distances <= close_threshold) // 2)
            possible_edges = len(largest) * (len(largest) - 1) // 2
            largest_cluster_density[frame] = edge_count / max(possible_edges, 1)
        close_ids = ids[nearest <= close_threshold]
        if len(close_ids) >= 2 and not huddle_members_by_frame[frame]:
            huddle_members_by_frame[frame] = tuple(sorted(int(item) for item in close_ids))
        isolated_ids = ids[nearest >= float(group_cfg["isolation_distance_cm"])]
        isolation_members_by_frame[frame] = tuple(sorted(int(item) for item in isolated_ids))

    huddle = (group_size >= 2) & (
        (close_fraction >= float(group_cfg["huddle_fraction"]))
        | (
            (largest_cluster_size >= int(group_cfg.get("huddle_min_cluster_size", 3)))
            & (
                largest_cluster_fraction
                >= float(group_cfg.get("huddle_min_cluster_fraction", 0.30))
            )
            & (largest_cluster_density >= float(group_cfg.get("huddle_min_cluster_density", 0.50)))
        )
    )
    # Isolation is a group-level state only if a substantial fraction of the
    # visible mice have no close neighbour; it is not emitted for an empty or
    # one-mouse frame.
    isolation = (group_size >= 3) & (
        isolated_fraction >= float(group_cfg["isolation_neighbor_fraction"])
    )
    for behavior, mask, score in (
        (
            "huddle",
            huddle,
            np.maximum(close_fraction, largest_cluster_fraction * largest_cluster_density),
        ),
        ("isolation", isolation, isolated_fraction),
    ):
        member_ids_by_frame = (
            huddle_members_by_frame if behavior == "huddle" else isolation_members_by_frame
        )
        events.extend(
            _event_rows_from_mask(
                mask,
                behavior=behavior,
                level="extended",
                fps=analysis_fps,
                source_video=source_video,
                sample_stride=sample_stride,
                score=score,
                actor_id=np.full(frames, -1),
                target_id=np.full(frames, -1),
                pair_key="group",
                min_duration_seconds=float(group_cfg["confirm_seconds"]),
                fill_gap_seconds=float(group_cfg["fill_gap_seconds"]),
                event_scope="group",
                member_ids_by_frame=member_ids_by_frame,
                fsm_coordinator=fsm_coordinator,
            )
        )
    return events


CONTACT_EVENT_COLUMNS = (
    "contact_event_id",
    "contact_detector",
    "pair_key",
    "contact_type",
    "contact_type_components",
    "contact_direction",
    "contact_actor_id",
    "contact_target_id",
    "role_ambiguous",
    "analysis_start_frame",
    "analysis_peak_frame",
    "analysis_end_frame",
    "start_frame",
    "peak_frame",
    "end_frame",
    "start_time_s",
    "end_time_s",
    "duration_s",
    "sample_count",
    "min_contact_distance_cm",
    "mean_contact_distance_cm",
    "min_nose_head_distance_cm",
    "min_nose_tail_distance_cm",
    "source_video",
    "analysis_mode",
)


def _contact_distance(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("inf")
    return number if np.isfinite(number) else float("inf")


def _contact_components(
    nose_head_distance: float,
    nose_tail_distance: float,
    nose_head_threshold: float,
    nose_tail_threshold: float,
) -> tuple[str, ...]:
    components: list[str] = []
    if nose_head_distance <= nose_head_threshold:
        components.append("nose_head")
    if nose_tail_distance <= nose_tail_threshold:
        components.append("nose_tail")
    return tuple(components)


def _extract_contact_events(
    pair_df: pd.DataFrame,
    *,
    pair_key: str,
    source_video: Path,
    source_fps: float,
    sample_stride: int,
    contact_config: Mapping[str, Any],
    fsm_coordinator: ParallelBehaviorFSM | None = None,
) -> list[dict[str, Any]]:
    """Extract nose-head and nose-tail contacts independently of behavior.

    Contact is deliberately not a behavior class.  It is a geometric event
    stream that can coexist with a chase or an attack, while ordinary contact
    alone never opens either behavior FSM.  The event state is evaluated at
    every analyzed sample and consecutive samples with the same contact
    geometry are grouped into one event.
    """
    if pair_df.empty or not bool(contact_config.get("enabled", True)):
        return []

    source_fps = max(float(source_fps), 1e-9)
    sample_stride = max(int(sample_stride), 1)
    nose_head_threshold = max(
        float(contact_config.get("nose_head_distance_cm", contact_config.get("distance_cm", 3.0))),
        0.0,
    )
    nose_tail_threshold = max(
        float(contact_config.get("nose_tail_distance_cm", contact_config.get("distance_cm", 3.0))),
        0.0,
    )

    frame_values = (
        pd.to_numeric(
            pair_df.get("frame", pd.Series(range(len(pair_df)))),
            errors="coerce",
        )
        .fillna(-1)
        .astype(int)
        .to_numpy()
    )
    row_count = len(pair_df)

    def numeric_column(column: str, default: float) -> np.ndarray:
        values = pair_df[column] if column in pair_df else pd.Series(default, index=pair_df.index)
        result = pd.to_numeric(values, errors="coerce").to_numpy(
            dtype=float,
            copy=True,
        )
        result[~np.isfinite(result)] = default
        return result

    def id_column(column: str) -> tuple[np.ndarray, np.ndarray]:
        values = pair_df[column] if column in pair_df else pd.Series(-1, index=pair_df.index)
        numeric = pd.to_numeric(values, errors="coerce").to_numpy(
            dtype=float,
            copy=True,
        )
        convertible = np.isfinite(numeric)
        numeric[~convertible] = -1
        return numeric.astype(int), convertible

    valid_values = (
        pair_df["valid_pair"] if "valid_pair" in pair_df else pd.Series(True, index=pair_df.index)
    )
    valid_pair = valid_values.fillna(False).astype(bool).to_numpy()
    mouse_a_ids, mouse_a_id_valid = id_column("mouse_a_id")
    mouse_b_ids, mouse_b_id_valid = id_column("mouse_b_id")
    head_ab = numeric_column("a_to_b_nose_head_distance_cm", float("inf"))
    tail_ab = numeric_column("a_to_b_nose_tail_distance_cm", float("inf"))
    head_ba = numeric_column("b_to_a_nose_head_distance_cm", float("inf"))
    tail_ba = numeric_column("b_to_a_nose_tail_distance_cm", float("inf"))
    contact_ab = (head_ab <= nose_head_threshold) | (tail_ab <= nose_tail_threshold)
    contact_ba = (head_ba <= nose_head_threshold) | (tail_ba <= nose_tail_threshold)

    states: list[dict[str, Any] | None] = [None] * row_count
    contact_indices = np.flatnonzero(valid_pair & (contact_ab | contact_ba))
    for index in contact_indices:
        if mouse_a_id_valid[index] and mouse_b_id_valid[index]:
            mouse_a_id = int(mouse_a_ids[index])
            mouse_b_id = int(mouse_b_ids[index])
        else:
            # Preserve the historical record-loop contract: conversion of
            # either endpoint failing makes both contact roles unknown.
            mouse_a_id = -1
            mouse_b_id = -1
        direction_hits: list[dict[str, Any]] = []
        direction_specs = (
            (
                "a_to_b",
                mouse_a_id,
                mouse_b_id,
                float(head_ab[index]),
                float(tail_ab[index]),
            ),
            (
                "b_to_a",
                mouse_b_id,
                mouse_a_id,
                float(head_ba[index]),
                float(tail_ba[index]),
            ),
        )
        for direction, actor_id, target_id, head_distance, tail_distance in direction_specs:
            components = _contact_components(
                head_distance,
                tail_distance,
                nose_head_threshold,
                nose_tail_threshold,
            )
            if not components:
                continue
            direction_hits.append(
                {
                    "direction": direction,
                    "actor_id": actor_id,
                    "target_id": target_id,
                    "components": components,
                    "head_distance": head_distance,
                    "tail_distance": tail_distance,
                }
            )

        if not direction_hits:
            states.append(None)
            continue

        components = tuple(
            name
            for name in ("nose_head", "nose_tail")
            if any(name in hit["components"] for hit in direction_hits)
        )
        contact_type = "nose_head_and_nose_tail" if len(components) == 2 else components[0]
        directions = tuple(hit["direction"] for hit in direction_hits)
        if len(directions) == 1:
            contact_direction = directions[0]
            contact_actor_id = int(direction_hits[0]["actor_id"])
            contact_target_id = int(direction_hits[0]["target_id"])
        else:
            contact_direction = "both"
            contact_actor_id = -1
            contact_target_id = -1

        head_distances = [hit["head_distance"] for hit in direction_hits]
        tail_distances = [hit["tail_distance"] for hit in direction_hits]
        contact_distances = [min(head_distances), min(tail_distances)]
        states[int(index)] = {
            "contact_type": contact_type,
            "contact_type_components": ";".join(components),
            "contact_direction": contact_direction,
            "contact_actor_id": contact_actor_id,
            "contact_target_id": contact_target_id,
            "role_ambiguous": bool(contact_actor_id < 0 or contact_target_id < 0),
            "contact_distance_cm": min(contact_distances),
            "nose_head_distance_cm": min(head_distances),
            "nose_tail_distance_cm": min(tail_distances),
        }

    def state_key(state: Mapping[str, Any] | None) -> tuple[Any, ...] | None:
        if state is None:
            return None
        return (
            state["contact_type"],
            state["contact_type_components"],
            state["contact_direction"],
            state["contact_actor_id"],
            state["contact_target_id"],
        )

    coordinator = fsm_coordinator or ParallelBehaviorFSM()
    contact_fsm = coordinator.run_categorical_region(
        scope="pair",
        region_id=str(pair_key),
        states=states,
        state_key=state_key,
    )
    events: list[dict[str, Any]] = []
    for span in contact_fsm.spans:
        start, end = int(span.start), int(span.end)
        segment = [state for state in states[start : end + 1] if state is not None]
        assert segment
        distances = np.asarray([state["contact_distance_cm"] for state in segment], dtype=float)
        peak_offset = int(np.argmin(distances)) if len(distances) else 0
        analysis_start = int(frame_values[start])
        analysis_peak = int(frame_values[start + peak_offset])
        analysis_end = int(frame_values[end])
        source_start = analysis_start * sample_stride
        source_peak = analysis_peak * sample_stride
        source_end = analysis_end * sample_stride
        events.append(
            {
                "contact_detector": "nose_head_nose_tail_geometry",
                "pair_key": str(pair_key),
                "contact_type": segment[0]["contact_type"],
                "contact_type_components": segment[0]["contact_type_components"],
                "contact_direction": segment[0]["contact_direction"],
                "contact_actor_id": int(segment[0]["contact_actor_id"]),
                "contact_target_id": int(segment[0]["contact_target_id"]),
                "role_ambiguous": bool(segment[0]["role_ambiguous"]),
                "analysis_start_frame": analysis_start,
                "analysis_peak_frame": analysis_peak,
                "analysis_end_frame": analysis_end,
                "start_frame": source_start,
                "peak_frame": source_peak,
                "end_frame": source_end,
                "start_time_s": source_start / source_fps,
                "end_time_s": source_end / source_fps,
                "duration_s": (source_end - source_start + 1) / source_fps,
                "sample_count": len(segment),
                "min_contact_distance_cm": float(np.min(distances)),
                "mean_contact_distance_cm": float(np.mean(distances)),
                "min_nose_head_distance_cm": float(
                    min(state["nose_head_distance_cm"] for state in segment)
                ),
                "min_nose_tail_distance_cm": float(
                    min(state["nose_tail_distance_cm"] for state in segment)
                ),
                "source_video": str(source_video),
                "analysis_mode": "lightweight_cache_tracking",
            }
        )
    return events
