"""Modular behavior segment clipping, boundary verification, fine-cropping, and selection.

Unifies and modularizes:
- Shared trajectory/scale/boundary utilities (`segment_clipping_common`)
- Individual-layer fine-cropping (`segment_clipping_individual`: Rearing, Grooming, Static, Locomotion)
- Social/pair-layer fine-cropping (`segment_clipping_social`: Approach, Attack, Chasing, Contact, Together)
- Group/collective-layer fine-cropping (`segment_clipping_group`: Huddling, Social clustering, Dispersal, Group locomotion, Isolation)
- Non-overlapping behavior segment selection (`select_behavior_segments`) with
  NO default output count cap (`max_per_behavior=None` by default).

Visual IDs remain strictly tracker-owned (`track_id` / `visual_track_id`); missing
identities or unobserved endpoints are never fabricated.
"""
from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from typing import Any

from mouse_behavior.behavior.duration_policy import qualify_event_duration
from mouse_behavior.behavior.segment_clipping_common import (
    InMemoryTrackStore,
    MIN_PARTICIPANTS_BY_BEHAVIOR,
    _resolve_fps,
    _resolve_parameters,
    _resolve_video_info,
    endpoint_observations,
    endpoints_fully_observed,
    event_participant_ids,
    is_process_boundary_closed,
    min_participants_for_behavior,
    normalize_behavior_key,
    update_segment_bounds,
)
from mouse_behavior.behavior.segment_clipping_group import (
    refine_dispersal_segments,
    refine_group_locomotion_segments,
    refine_huddle_segments,
    refine_isolation_segments,
    refine_social_clustering_segments,
)
from mouse_behavior.behavior.segment_clipping_individual import (
    refine_grooming_segments,
    refine_locomotion_segments,
    refine_rearing_segments,
    refine_static_segments,
)
from mouse_behavior.behavior.segment_clipping_social import (
    refine_approach_segments,
    refine_attack_segments,
    refine_chasing_segments,
    refine_contact_and_together_segments,
)
from mouse_behavior.evaluation.exported_dataset_models import HeuristicParameters

__all__ = [
    "InMemoryTrackStore",
    "MIN_PARTICIPANTS_BY_BEHAVIOR",
    "endpoint_observations",
    "endpoints_fully_observed",
    "is_process_boundary_closed",
    "min_participants_for_behavior",
    "refine_approach_segments",
    "refine_attack_segments",
    "refine_behavior_segments",
    "refine_chasing_segments",
    "refine_contact_and_together_segments",
    "refine_dispersal_segments",
    "refine_grooming_segments",
    "refine_group_locomotion_segments",
    "refine_huddle_segments",
    "refine_isolation_segments",
    "refine_locomotion_segments",
    "refine_rearing_segments",
    "refine_social_clustering_segments",
    "refine_static_segments",
    "select_behavior_segments",
    "update_segment_bounds",
]


def refine_behavior_segments(
    events: Sequence[Mapping[str, Any]],
    store: Any,
    binding: Mapping[str, Any],
    scale: Mapping[str, Any] | None = None,
    parameters: HeuristicParameters | None = None,
    *,
    enable_huddle_sliding: bool = True,
    enable_approach_requalification: bool = False,
    enable_fine_cropping: bool = False,
    scene_guard: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Unified segment refinement pipeline combining Codex and calibrated fine-cropping."""
    current = [copy.deepcopy(dict(e)) for e in events]
    audit: dict[str, Any] = {
        "input_count": len(current),
        "huddle_audit": [],
        "approach_audit": [],
        "rejected_insufficient_participants": 0,
    }
    if enable_huddle_sliding and scale is not None and any(e.get("canonical_behavior") == "huddle" for e in current):
        current, huddle_audit = refine_huddle_segments(current, store, binding, scale)
        audit["huddle_audit"] = huddle_audit

    if (
        enable_approach_requalification
        and scale is not None
        and any(e.get("canonical_behavior") == "approach" for e in current)
    ):
        current, approach_audit = refine_approach_segments(
            current,
            store,
            binding,
            scale,
            parameters=parameters,
            scene_guard=scene_guard,
        )
        audit["approach_audit"] = approach_audit

    if enable_fine_cropping:
        current = refine_rearing_segments(current, store, binding, scale=scale, parameters=parameters)
        current = refine_grooming_segments(current, store, binding, scale=scale, parameters=parameters)
        current = refine_attack_segments(current, store, binding, scale=scale, parameters=parameters)
        current = refine_chasing_segments(current, store, binding, scale=scale, parameters=parameters)
        current = refine_contact_and_together_segments(
            current, store, binding, scale=scale, parameters=parameters
        )
        current = refine_social_clustering_segments(current, store, binding, scale=scale, parameters=parameters)
        current = refine_dispersal_segments(current, store, binding, scale=scale, parameters=parameters)
        current = refine_group_locomotion_segments(current, store, binding, scale=scale, parameters=parameters)
        current = refine_isolation_segments(current, store, binding, scale=scale, parameters=parameters)
        current = refine_static_segments(current, store, binding, scale=scale, parameters=parameters)
        current = refine_locomotion_segments(current, store, binding, scale=scale, parameters=parameters)

    filtered: list[dict[str, Any]] = []
    for ev in current:
        beh = str(ev.get("canonical_behavior") or ev.get("behavior") or "")
        if len(event_participant_ids(ev)) < min_participants_for_behavior(beh):
            audit["rejected_insufficient_participants"] += 1
            continue
        filtered.append(ev)

    audit["output_count"] = len(filtered)
    return filtered, audit


def _segment_priority_key(
    event: Mapping[str, Any],
    fps: float,
    *,
    params: HeuristicParameters | None = None,
) -> tuple[int, int, int, int, float, float, int, float, int]:
    """Higher tuple values indicate stronger measured support for deduplicating overlapping bouts."""
    start = int(event["start_frame"])
    end = int(event["end_frame"])
    span = max(1, end - start)
    beh = str(event.get("canonical_behavior") or event.get("behavior") or "")
    unique_ids = event_participant_ids(event)
    valid_participants = 1 if len(unique_ids) >= min_participants_for_behavior(beh) else 0
    duration_ok = 1 if qualify_event_duration(event, fps, parameters=params)["ok"] else 0
    audit_blk = event.get("audit") if isinstance(event.get("audit"), Mapping) else {}
    kin_raw = event.get(
        "kinematic_ok",
        event.get(
            "full_crop_precision_ok",
            audit_blk.get("full_crop_precision_ok", audit_blk.get("kinematic_ok", True)),
        ),
    )
    kinematic_ok = 1 if bool(kin_raw) else 0
    boundary_closed = 1 if is_process_boundary_closed(event) else 0
    observed = int(event.get("observed_all_participants_frame_count", span))
    support = int(event.get("canonical_support_frame_count", observed))
    obs_ratio = min(1.0, observed / span)
    supp_ratio = min(1.0, support / span)
    dur_s = span / fps
    return (
        valid_participants,
        duration_ok,
        kinematic_ok,
        boundary_closed,
        round(supp_ratio, 4),
        round(obs_ratio, 4),
        len(unique_ids),
        round(dur_s, 3),
        -start,
    )


def select_behavior_segments(
    events: Sequence[Mapping[str, Any]],
    store: Any = None,
    binding: Mapping[str, Any] | None = None,
    *,
    behaviors: Sequence[str] | None = None,
    max_per_behavior: int | None = None,
    require_closed_boundaries: bool = True,
    require_endpoint_visibility: bool = True,
    strict_non_overlapping_same_ids: bool = True,
    strict_non_overlapping_shared_mouse: bool = True,
    time_bucket_seconds: float | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select and deduplicate behavior segments without any default output count limit.

    Parameters
    ----------
    events:
        Candidate behavior segment dicts.
    store:
        Optional `TrackStore` used to verify endpoint visibility when
        `require_endpoint_visibility=True`.
    binding:
        Optional run binding or `source_video` dict providing `fps`, `frame_count`,
        and `effective_parameters`.
    behaviors:
        Optional subset of canonical or display behavior names to retain. When
        `None` (default), all behaviors present in `events` are retained.
    max_per_behavior:
        Optional maximum number of segments per behavior. Defaults to `None`
        (unlimited: every qualifying segment is returned).
    require_closed_boundaries:
        Reject segments whose `process_boundary_status` marks window clipping or
        unclosed context boundaries.
    require_endpoint_visibility:
        When `store` is provided, require all participants in `mouse_ids` to be
        observed at both `start_frame` and `end_frame - 1`.
    strict_non_overlapping_same_ids:
        Prevent temporally overlapping segments for the exact same
        `(canonical_behavior, tuple(mouse_ids))`.
    strict_non_overlapping_shared_mouse:
        Prevent any two retained segments of the same behavior from sharing any
        participant mouse and overlapping temporally (`[start_frame, end_frame)`).
    time_bucket_seconds:
        Optional coarse time-bucket spacing in seconds. Defaults to `None`
        (disabled so no valid segments are dropped by time bucketing).
    """
    if max_per_behavior is not None and (type(max_per_behavior) is not int or max_per_behavior < 1):
        raise ValueError("max_per_behavior must be None or a positive integer")
    if time_bucket_seconds is not None and (not math.isfinite(time_bucket_seconds) or time_bucket_seconds <= 0):
        raise ValueError("time_bucket_seconds must be None or a positive finite number")

    fps = _resolve_fps(binding)
    video_info = _resolve_video_info(binding)
    params = _resolve_parameters(binding)
    allowed = set(behaviors) if behaviors is not None else None

    grouped: dict[str, list[dict[str, Any]]] = {}
    rejected_counts: dict[str, int] = {
        "filtered_behavior": 0,
        "insufficient_participants": 0,
        "duration_policy": 0,
        "unclosed_boundary": 0,
        "unobserved_endpoints": 0,
        "duplicate_or_overlapping": 0,
    }

    seen_exact: set[tuple[str, tuple[int, ...], int, int]] = set()
    for raw in events:
        event = copy.deepcopy(dict(raw))
        canonical = str(event.get("canonical_behavior") or event.get("behavior") or "")
        display = str(event.get("behavior") or canonical)
        norm_beh = normalize_behavior_key(canonical) or canonical
        if allowed is not None and canonical not in allowed and display not in allowed and norm_beh not in allowed:
            rejected_counts["filtered_behavior"] += 1
            continue
        if len(event_participant_ids(event)) < min_participants_for_behavior(canonical):
            rejected_counts["insufficient_participants"] += 1
            continue
        if not qualify_event_duration(event, fps, parameters=params)["ok"]:
            rejected_counts["duration_policy"] += 1
            continue
        if require_closed_boundaries and not is_process_boundary_closed(event):
            rejected_counts["unclosed_boundary"] += 1
            continue
        if require_endpoint_visibility and store is not None:
            if not endpoints_fully_observed(store, event, video_info):
                rejected_counts["unobserved_endpoints"] += 1
                continue
        key = (
            norm_beh,
            tuple(int(m) for m in event.get("mouse_ids", ())),
            int(event["start_frame"]),
            int(event["end_frame"]),
        )
        if key in seen_exact:
            rejected_counts["duplicate_or_overlapping"] += 1
            continue
        seen_exact.add(key)
        grouped.setdefault(norm_beh, []).append(event)

    selected_all: list[dict[str, Any]] = []
    counts_by_behavior: dict[str, int] = {}

    for norm_beh, candidates in grouped.items():
        ranked = sorted(
            candidates,
            key=lambda ev: _segment_priority_key(ev, fps, params=params),
            reverse=True,
        )
        chosen: list[dict[str, Any]] = []
        used_buckets: set[tuple[tuple[int, ...], int]] = set()

        for cand in ranked:
            mids = tuple(int(m) for m in cand.get("mouse_ids", ()))
            cand_mouse_set = set(event_participant_ids(cand))
            s = int(cand["start_frame"])
            e = int(cand["end_frame"])
            is_directed = norm_beh in {"approach", "chasing"}
            cand_ids = tuple(mids) if is_directed else tuple(sorted(mids))
            if strict_non_overlapping_shared_mouse:
                overlaps = any(
                    bool(cand_mouse_set & set(event_participant_ids(existing)))
                    and not (e <= int(existing["start_frame"]) or s >= int(existing["end_frame"]))
                    for existing in chosen
                )
                if overlaps:
                    rejected_counts["duplicate_or_overlapping"] += 1
                    continue
            elif strict_non_overlapping_same_ids:
                overlaps = any(
                    (
                        tuple(int(m) for m in existing.get("mouse_ids", ()))
                        if is_directed
                        else tuple(sorted(int(m) for m in existing.get("mouse_ids", ())))
                    ) == cand_ids
                    and not (e <= int(existing["start_frame"]) or s >= int(existing["end_frame"]))
                    for existing in chosen
                )
                if overlaps:
                    rejected_counts["duplicate_or_overlapping"] += 1
                    continue
            if time_bucket_seconds is not None:
                bucket_frames = max(1, int(round(time_bucket_seconds * fps)))
                bkey = (cand_ids, s // bucket_frames)
                if bkey in used_buckets:
                    rejected_counts["duplicate_or_overlapping"] += 1
                    continue
                used_buckets.add(bkey)
            chosen.append(cand)
            if max_per_behavior is not None and len(chosen) >= max_per_behavior:
                break

        chosen.sort(key=lambda ev: (int(ev["start_frame"]), int(ev["end_frame"]), str(ev.get("event_id", ""))))
        for ev in chosen:
            c_key = str(ev.get("canonical_behavior") or ev.get("behavior") or norm_beh)
            counts_by_behavior[c_key] = counts_by_behavior.get(c_key, 0) + 1
        selected_all.extend(chosen)

    selected_all.sort(
        key=lambda ev: (
            int(ev["start_frame"]),
            int(ev["end_frame"]),
            str(ev.get("canonical_behavior", "")),
            str(ev.get("event_id", "")),
        )
    )
    summary = {
        "total_input_events": len(events),
        "total_selected_events": len(selected_all),
        "max_per_behavior": max_per_behavior,
        "counts_by_behavior": counts_by_behavior,
        "rejected_counts": rejected_counts,
    }
    return selected_all, summary
