"""Candidate-pair orchestration and event finalization."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .. import standard_behavior_engine as behavior_engine
from ..parallel_behavior_fsm import ParallelBehaviorFSM
from ..preprocessing.pair_features import _PairWorkset, _pair_dataframe
from .ethogram import (
    _extended_pair_events,
    _extended_short_clip_pair_events,
    _extract_contact_events,
)
from .social_fsm import event_pair_ids
from ..utils.timer import Timer

LOGGER = logging.getLogger("mouse_behavior.lightweight_behavior_inference")


@dataclass
class _PairAnalysisResult:
    """In-memory outputs produced by the candidate-pair analysis stage."""

    events: list[dict[str, Any]]
    contact_events: list[dict[str, Any]]
    extended_events: list[dict[str, Any]]
    pair_summaries: list[dict[str, Any]]
    top_evidence: list[dict[str, Any]]
    fsm_coordinator: ParallelBehaviorFSM


def _event_interval(event: Mapping[str, Any], *, core: bool = False) -> tuple[int, int]:
    """Return a normalized source-frame interval for an event record."""

    start_key = "core_start_frame" if core else "start_frame"
    end_key = "core_end_frame" if core else "end_frame"
    try:
        start = int(event.get(start_key, event.get("start_frame", 0)) or 0)
    except (TypeError, ValueError, OverflowError):
        start = 0
    try:
        end = int(event.get(end_key, event.get("end_frame", start)) or start)
    except (TypeError, ValueError, OverflowError):
        end = start
    return (min(start, end), max(start, end))


def _semantic_bridge_settings(config: Mapping[str, Any]) -> tuple[float, int, int]:
    extended = config.get("extended_behavior", {})
    social = extended.get("social", {}) if isinstance(extended, Mapping) else {}
    semantic = social.get("semantic_fsm", {}) if isinstance(social, Mapping) else {}
    if not isinstance(semantic, Mapping):
        semantic = {}
    try:
        bridge_seconds = max(float(semantic.get("identity_bridge_seconds", 0.50)), 0.0)
    except (TypeError, ValueError):
        bridge_seconds = 0.50
    try:
        max_participants = max(
            int(semantic.get("max_identity_bridge_participants", 6)),
            2,
        )
    except (TypeError, ValueError):
        max_participants = 6
    try:
        max_segments = max(
            int(semantic.get("max_identity_bridge_segments", 2)),
            1,
        )
    except (TypeError, ValueError):
        max_segments = 2
    return bridge_seconds, max_participants, max_segments


def _merge_identity_bridged_events(
    component: list[Mapping[str, Any]],
    *,
    source_fps: float,
) -> dict[str, Any]:
    """Merge one identity-continuous semantic event component.

    The representative actor/target is taken from the highest-scoring segment
    so the existing renderer remains compatible.  ``role_trace`` and
    ``participant_ids`` retain the changing IDs for downstream auditing.
    """

    ordered = sorted(
        (dict(event) for event in component),
        key=lambda event: _event_interval(event, core=True),
    )
    representative = max(
        ordered,
        key=lambda event: float(event.get("peak_score", event.get("mean_score", 0.0)) or 0.0),
    )
    merged = dict(representative)
    public_intervals = [_event_interval(event) for event in ordered]
    core_intervals = [_event_interval(event, core=True) for event in ordered]
    analysis_intervals = [
        _event_interval(
            {
                "start_frame": event.get("analysis_start_frame", event.get("start_frame", 0)),
                "end_frame": event.get("analysis_end_frame", event.get("end_frame", 0)),
            }
        )
        for event in ordered
    ]
    public_start = min(start for start, _ in public_intervals)
    public_end = max(end for _, end in public_intervals)
    core_start = min(start for start, _ in core_intervals)
    core_end = max(end for _, end in core_intervals)
    analysis_start = min(start for start, _ in analysis_intervals)
    analysis_end = max(end for _, end in analysis_intervals)
    participants = sorted(
        {
            participant
            for event in ordered
            for participant in event_pair_ids(event)
            if participant >= 0
        }
    )
    pair_keys = sorted(
        {str(event.get("pair_key", "")) for event in ordered if str(event.get("pair_key", ""))}
    )

    def safe_id(value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError, OverflowError):
            return -1

    role_trace = [
        {
            "pair_key": str(event.get("pair_key", "")),
            "actor_id": safe_id(event.get("actor_id", -1)),
            "target_id": safe_id(event.get("target_id", -1)),
            "start_frame": _event_interval(event, core=True)[0],
            "end_frame": _event_interval(event, core=True)[1],
        }
        for event in ordered
    ]
    merged.update(
        {
            "pair_key": "|".join(pair_keys),
            "participant_ids": participants,
            "member_ids": participants,
            "identity_bridge": True,
            "identity_bridge_pairs": pair_keys,
            "role_trace": role_trace,
            "event_recovery": "identity_bridge",
            "analysis_start_frame": analysis_start,
            "analysis_end_frame": analysis_end,
            "core_start_frame": core_start,
            "core_end_frame": core_end,
            "start_frame": public_start,
            "end_frame": public_end,
            "core_duration_s": (core_end - core_start + 1) / max(source_fps, 1e-9),
            "duration_s": (public_end - public_start + 1) / max(source_fps, 1e-9),
            "mean_score": float(
                np.mean([float(event.get("mean_score", 0.0) or 0.0) for event in ordered])
            ),
            "peak_score": max(
                float(event.get("peak_score", event.get("mean_score", 0.0)) or 0.0)
                for event in ordered
            ),
            "role_ambiguous": bool(
                any(bool(event.get("role_ambiguous", False)) for event in ordered)
            ),
        }
    )
    return merged


def _stitch_identity_bridged_events(
    events: list[dict[str, Any]],
    *,
    source_fps: float,
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Join semantic pair events across short detector/ID-switch gaps.

    This is deliberately limited to semantic chase, avoidance, and attack
    candidates.  It never joins different behaviors, unrelated source videos,
    or pairs with no shared identity evidence.  Approach is left unstitched
    because its actor/target transition is itself the event being measured.
    """

    if not events:
        return []
    bridge_seconds, max_participants, max_segments = _semantic_bridge_settings(config)
    bridge_frames = int(round(bridge_seconds * max(float(source_fps), 1e-9)))
    bridge_behaviors = {"chase", "attack"}
    # Avoidance is opt-in because a shared ID in a crowded scene can otherwise
    # merge unrelated evasion candidates.  The Beiyi profile enables it after
    # validating the short rider/occlusion examples; generic profiles keep the
    # conservative historical default.
    semantic = config.get("extended_behavior", {})
    social = semantic.get("social", {}) if isinstance(semantic, Mapping) else {}
    semantic_fsm = social.get("semantic_fsm", {}) if isinstance(social, Mapping) else {}
    if isinstance(semantic_fsm, Mapping) and bool(semantic_fsm.get("bridge_avoidance", False)):
        bridge_behaviors.add("avoidance")
    candidates: list[int] = []
    for index, event in enumerate(events):
        recovery = str(event.get("event_recovery", ""))
        behavior = str(event.get("behavior", "")).strip().lower()
        if (
            str(event.get("event_scope", "pair")) == "pair"
            and behavior in bridge_behaviors
            and recovery.startswith("semantic_")
            and len(event_pair_ids(event)) >= 2
        ):
            candidates.append(index)
    if not candidates:
        return list(events)

    used: set[int] = set()
    replacements: dict[int, dict[str, Any]] = {}
    for seed_index in candidates:
        if seed_index in used:
            continue
        component = [seed_index]
        used.add(seed_index)
        changed = True
        while changed and len(component) < max_segments:
            changed = False
            component_events = [events[index] for index in component]
            behavior = str(component_events[0].get("behavior", "")).strip().lower()
            source_videos = {
                str(event.get("source_video", ""))
                for event in component_events
                if str(event.get("source_video", ""))
            }
            component_ids = set().union(*(event_pair_ids(event) for event in component_events))
            component_end = max(_event_interval(event, core=True)[1] for event in component_events)
            for index in candidates:
                if index in used:
                    continue
                candidate = events[index]
                if str(candidate.get("behavior", "")).strip().lower() != behavior:
                    continue
                candidate_source = str(candidate.get("source_video", ""))
                if source_videos and candidate_source and candidate_source not in source_videos:
                    continue
                candidate_ids = event_pair_ids(candidate)
                if not component_ids.intersection(candidate_ids):
                    continue
                candidate_start, candidate_end = _event_interval(candidate, core=True)
                # Only bridge a later segment.  Overlapping events are not an
                # identity switch; allowing them caused unrelated pair events
                # from the same noisy frame to collapse into one long event.
                if candidate_start <= component_end:
                    continue
                gap = candidate_start - component_end - 1
                if gap > bridge_frames:
                    continue
                if len(component_ids | candidate_ids) > max_participants:
                    continue
                component.append(index)
                used.add(index)
                changed = True
                component_ids.update(candidate_ids)
                component_end = max(component_end, candidate_end)
                if len(component) >= max_segments:
                    break
        if len(component) > 1:
            merged = _merge_identity_bridged_events(
                [events[index] for index in component],
                source_fps=source_fps,
            )
            replacements[seed_index] = merged
            LOGGER.info(
                "[identity bridge] %s %s segments: %s",
                merged.get("behavior", ""),
                merged.get("source_video", ""),
                " -> ".join(merged.get("identity_bridge_pairs", [])),
            )
        else:
            replacements[seed_index] = events[seed_index]

    result: list[dict[str, Any]] = []
    for index, event in enumerate(events):
        if index in replacements:
            result.append(replacements[index])
        elif index not in used:
            result.append(event)
    return result


def _analyze_candidate_pairs(
    workset: _PairWorkset,
    kin: Mapping[str, Any],
    *,
    fps: float,
    source_fps: float,
    sample_stride: int,
    video_path: Path,
    config: Mapping[str, Any],
    stage_timings: dict[str, float] | None = None,
) -> _PairAnalysisResult:
    """Analyze retained pairs while preserving the complete pair timeline."""

    events: list[dict[str, Any]] = []
    contact_events: list[dict[str, Any]] = []
    extended_events: list[dict[str, Any]] = []
    pair_summaries: list[dict[str, Any]] = []
    top_evidence: list[dict[str, Any]] = []
    pair_analysis_timer = Timer(
        "candidate_pair_analysis",
        logger=LOGGER,
        sink=stage_timings,
    ).start()
    pair_fsm_coordinator = ParallelBehaviorFSM(dict(config.get("parallel_behavior_fsm", {})))
    candidate_ordinal = 0
    for pair_index, (mouse_a, mouse_b) in enumerate(zip(workset.all_pair_i, workset.all_pair_j)):
        metric_index = workset.candidate_metric_index.get(pair_index)
        base_summary: dict[str, Any] = {
            "pair_key": f"{int(mouse_a)}_{int(mouse_b)}",
            "mouse_a_id": int(mouse_a),
            "mouse_b_id": int(mouse_b),
            "valid_frames": int(
                np.asarray(
                    workset.prefilter["valid_pair"][:, pair_index],
                    dtype=bool,
                ).sum()
            ),
            "min_distance_cm": float(np.nanmin(workset.prefilter["distance"][:, pair_index]))
            if np.isfinite(workset.prefilter["distance"][:, pair_index]).any()
            else float("nan"),
            "max_speed_cm_s": float(
                max(
                    float(np.asarray(workset.metrics["speed"][:, int(mouse_a)]).max(initial=0.0)),
                    float(np.asarray(workset.metrics["speed"][:, int(mouse_b)]).max(initial=0.0)),
                )
            ),
            "engine_evaluated": metric_index is not None,
            "fsm_evaluated_frames": 0,
            "fsm_skipped_frames": 0,
            "fsm_evaluated_fraction": 0.0,
            "nose_head_contact_event_count": 0,
            "nose_tail_contact_event_count": 0,
            "combined_nose_head_nose_tail_event_count": 0,
            "contact_sample_count": 0,
        }
        if metric_index is None:
            for level in ("weak", "strong"):
                for behavior in ("chase", "attack"):
                    base_summary[f"{level}_{behavior}_frames"] = 0
                    base_summary[f"{level}_{behavior}_max_score"] = 0.0
                    base_summary[f"{level}_{behavior}_role_known_rate"] = None
            pair_summaries.append(base_summary)
            continue

        candidate_ordinal += 1
        if pair_index == 0 or pair_index % 20 == 0 or pair_index == len(workset.all_pair_i) - 1:
            LOGGER.info(
                "[pair analysis] pair %d/%d (%d/%d candidates)",
                pair_index + 1,
                len(workset.all_pair_i),
                candidate_ordinal,
                len(workset.candidate_pair_indices),
            )
        pair_df = _pair_dataframe(
            workset.metrics,
            metric_index,
            int(mouse_a),
            int(mouse_b),
            workset.pair_i,
            workset.pair_j,
            fps,
            float(kin["cm_per_pixel"]),
        )
        pair_contact_events = _extract_contact_events(
            pair_df,
            pair_key=f"{int(mouse_a)}_{int(mouse_b)}",
            source_video=video_path,
            source_fps=source_fps,
            sample_stride=sample_stride,
            contact_config=dict(config.get("contact_detection", {})),
            fsm_coordinator=pair_fsm_coordinator,
        )
        contact_events.extend(pair_contact_events)
        enriched = behavior_engine.apply_standard_behavior_engine(
            pair_df,
            fps,
            config,
            copy_input=False,
        )
        extended_events.extend(
            _extended_pair_events(
                pair_df,
                metrics=workset.metrics,
                pair_index=metric_index,
                enriched=enriched,
                source_video=video_path,
                source_fps=source_fps,
                sample_stride=sample_stride,
                config=config,
                fsm_coordinator=pair_fsm_coordinator,
            )
        )
        extended_events.extend(
            _extended_short_clip_pair_events(
                pair_df,
                enriched,
                source_video=video_path,
                source_fps=source_fps,
                sample_stride=sample_stride,
                config=config,
                fsm_coordinator=pair_fsm_coordinator,
            )
        )
        summary = base_summary
        fsm_compute = (
            enriched.get(
                "standard_behavior_compute_row",
                pair_df.get("valid_pair", pd.Series(True, index=pair_df.index)),
            )
            .fillna(False)
            .astype(bool)
        )
        summary["fsm_evaluated_frames"] = int(fsm_compute.sum())
        summary["fsm_skipped_frames"] = int(len(fsm_compute) - fsm_compute.sum())
        summary["fsm_evaluated_fraction"] = float(fsm_compute.mean()) if len(fsm_compute) else 0.0
        summary["nose_head_contact_event_count"] = int(
            sum(
                "nose_head" in str(event.get("contact_type_components", "")).split(";")
                for event in pair_contact_events
            )
        )
        summary["nose_tail_contact_event_count"] = int(
            sum(
                "nose_tail" in str(event.get("contact_type_components", "")).split(";")
                for event in pair_contact_events
            )
        )
        summary["combined_nose_head_nose_tail_event_count"] = int(
            sum(
                event.get("contact_type") == "nose_head_and_nose_tail"
                for event in pair_contact_events
            )
        )
        summary["contact_sample_count"] = int(
            sum(int(event.get("sample_count", 0)) for event in pair_contact_events)
        )
        for level in ("weak", "strong"):
            for behavior in ("chase", "attack"):
                mask_col = f"{level}_standard_final_{behavior}"
                score_col = f"{level}_standard_{behavior}_score"
                actor_col = f"{level}_standard_{behavior}_actor_id"
                target_col = f"{level}_standard_{behavior}_target_id"
                active = (
                    enriched[mask_col].fillna(False).astype(bool)
                    if mask_col in enriched
                    else pd.Series(False, index=enriched.index)
                )
                summary[f"{level}_{behavior}_frames"] = int(active.sum())
                summary[f"{level}_{behavior}_max_score"] = (
                    float(enriched[score_col].max()) if score_col in enriched else 0.0
                )
                if active.any() and actor_col in enriched and target_col in enriched:
                    known = (
                        pd.to_numeric(enriched.loc[active, actor_col], errors="coerce") >= 0
                    ) & (pd.to_numeric(enriched.loc[active, target_col], errors="coerce") >= 0)
                    summary[f"{level}_{behavior}_role_known_rate"] = float(known.mean())
                else:
                    summary[f"{level}_{behavior}_role_known_rate"] = None
            for event in behavior_engine.extract_standard_behavior_events(
                enriched,
                fps,
                level,
                pair_key=f"{int(mouse_a)}_{int(mouse_b)}",
            ):
                event = dict(event)
                event["source_video"] = str(video_path)
                event["analysis_mode"] = "lightweight_cache_tracking"
                event["analysis_start_frame"] = int(event.get("start_frame", 0))
                event["analysis_peak_frame"] = int(event.get("peak_frame", 0))
                event["analysis_end_frame"] = int(event.get("end_frame", 0))
                event["start_frame"] = int(event.get("start_frame", 0)) * sample_stride
                event["peak_frame"] = int(event.get("peak_frame", 0)) * sample_stride
                event["end_frame"] = int(event.get("end_frame", 0)) * sample_stride
                events.append(event)
        for behavior in ("chase", "attack"):
            score_column = f"strong_standard_{behavior}_score"
            if score_column not in enriched:
                continue
            candidate = enriched[
                ["frame", "time_s", "pair_key", "center_distance_cm", score_column]
            ].copy()
            candidate["behavior"] = behavior
            candidate["level"] = "strong"
            candidate = candidate.rename(columns={score_column: "score"})
            evidence_rows = candidate.nlargest(5, "score").to_dict("records")
            for row in evidence_rows:
                row["analysis_frame"] = int(row["frame"])
                row["source_frame"] = int(row["frame"]) * sample_stride
                row["source_time_s"] = float(row["frame"]) * sample_stride / source_fps
            top_evidence.extend(evidence_rows)
        pair_summaries.append(summary)
    extended_events = _stitch_identity_bridged_events(
        extended_events,
        source_fps=source_fps,
        config=config,
    )
    pair_analysis_timer.stop()
    return _PairAnalysisResult(
        events=events,
        contact_events=contact_events,
        extended_events=extended_events,
        pair_summaries=pair_summaries,
        top_evidence=top_evidence,
        fsm_coordinator=pair_fsm_coordinator,
    )


def _finalize_event_records_in_place(
    events: list[dict[str, Any]],
    contact_events: list[dict[str, Any]],
    source_fps: float,
    minimum_durations: Mapping[str, float] | None = None,
) -> None:
    """Finalize event records and apply the configured temporal reliability gates.

    A one-frame attack can still be useful while debugging the feature matrix,
    but it is not a reliable behavior event for a rendered video or an export.
    The attack-specific temporal gate normally removes it earlier.  This final
    safety net also covers legacy standard-FSM rows, which are produced by a
    separate extractor and therefore cannot share the extended ethogram gate.
    ``minimum_durations`` is intentionally optional so callers using the
    historical API retain their previous behavior; production profiles can
    provide a behavior-level minimum for labels whose document definition does
    not specify a fixed duration.
    """

    configured_minimums: dict[str, float] = {}
    for behavior, value in (minimum_durations or {}).items():
        try:
            configured_minimums[str(behavior).strip().lower()] = max(float(value), 0.0)
        except (TypeError, ValueError):
            continue

    reliable_events: list[dict[str, Any]] = []
    suppressed_single_frame_attacks = 0
    suppressed_huddle_contained_attacks = 0
    suppressed_short_duration: dict[str, int] = {}
    for event in events:
        behavior = str(event.get("behavior", "")).strip().lower()
        if behavior == "attack":
            if event.get("huddle_conflict_status") == "contained_by_stable_huddle":
                suppressed_huddle_contained_attacks += 1
                continue
            start = int(event.get("analysis_start_frame", event.get("start_frame", 0)) or 0)
            end = int(event.get("analysis_end_frame", event.get("end_frame", start)) or start)
            if end <= start:
                suppressed_single_frame_attacks += 1
                continue
        required = configured_minimums.get(behavior)
        if required is not None:
            raw_duration = event.get("core_duration_s", event.get("duration_s"))
            try:
                core_duration = float(raw_duration)
            except (TypeError, ValueError):
                core_duration = float("nan")
            if not np.isfinite(core_duration):
                # Some legacy standard-FSM rows predate ``core_duration_s``
                # and carry a null placeholder after CSV round-trip.  Fall
                # back to the public duration before deciding whether the
                # attack is reliable; otherwise a one-frame/short attack can
                # escape the configured temporal gate as NaN.
                try:
                    core_duration = float(event.get("duration_s"))
                except (TypeError, ValueError):
                    core_duration = float("nan")
            if np.isfinite(core_duration) and core_duration < required:
                suppressed_short_duration[behavior] = suppressed_short_duration.get(behavior, 0) + 1
                continue
        reliable_events.append(event)
    events[:] = reliable_events
    if suppressed_single_frame_attacks:
        LOGGER.info(
            "[behavior] suppressed %d unsupported one-frame attack event(s)",
            suppressed_single_frame_attacks,
        )
    if suppressed_huddle_contained_attacks:
        LOGGER.info(
            "[behavior] suppressed %d attack event(s) contained by a stable huddle",
            suppressed_huddle_contained_attacks,
        )
    if suppressed_short_duration:
        LOGGER.info(
            "[behavior] suppressed events below configured minimum duration: %s",
            ", ".join(
                f"{behavior}={count}"
                for behavior, count in sorted(suppressed_short_duration.items())
            ),
        )

    for index, event in enumerate(
        sorted(
            events,
            key=lambda item: (
                int(item.get("start_frame", 0)),
                str(item.get("pair_key", "")),
                str(item.get("level", "")),
            ),
        ),
        start=1,
    ):
        event["light_event_id"] = f"LWE{index:05d}"
        event["start_time_s"] = float(event.get("start_frame", 0)) / source_fps
        event["end_time_s"] = float(event.get("end_frame", 0)) / source_fps
        event["duration_s"] = (
            float(event.get("end_frame", 0) - event.get("start_frame", 0) + 1) / source_fps
        )

    for index, event in enumerate(
        sorted(
            contact_events,
            key=lambda item: (
                int(item.get("start_frame", 0)),
                str(item.get("pair_key", "")),
                str(item.get("contact_type", "")),
            ),
        ),
        start=1,
    ):
        event["contact_event_id"] = f"LCE{index:05d}"
