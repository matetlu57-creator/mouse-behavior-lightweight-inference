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
from .social_fsm import build_semantic_pair_signals

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


def _member_trace_segments(
    member_ids_by_frame: Sequence[Iterable[int]] | Mapping[int, Iterable[int]] | None,
    *,
    start: int,
    end: int,
    sample_stride: int,
) -> list[dict[str, Any]]:
    """Compress frame-specific group membership into auditable source spans."""

    if member_ids_by_frame is None or end < start:
        return []
    segments: list[dict[str, Any]] = []
    active_members: tuple[int, ...] = ()
    active_start = -1

    def append_segment(segment_start: int, segment_end: int) -> None:
        if not active_members or segment_end < segment_start:
            return
        segments.append(
            {
                "member_ids": list(active_members),
                "start_frame": int(segment_start * sample_stride),
                "end_frame": int(segment_end * sample_stride),
            }
        )

    for frame in range(start, end + 1):
        members = _frame_member_ids(member_ids_by_frame, frame)
        if members == active_members:
            continue
        append_segment(active_start, frame - 1)
        active_members = members
        active_start = frame
    append_segment(active_start, end)
    return segments


def _sustained_member_ids_by_frame(
    members_by_frame: Sequence[Iterable[int]],
    *,
    frames: int,
    mice: int,
    min_duration_frames: int,
    max_gap_frames: int,
) -> list[tuple[int, ...]]:
    """Keep group members whose own state lasts long enough.

    Group masks are evaluated at the video level, but membership can flicker
    when a detector briefly swaps or drops one mouse. Unioning every member
    seen during the whole event therefore labels a transient outlier as
    isolated. This helper applies the isolation duration rule per logical ID,
    bridges only the configured short gaps, and returns stable members for
    downstream rendering/export.
    """

    membership = np.zeros((max(int(frames), 0), max(int(mice), 0)), dtype=bool)
    for frame, values in enumerate(members_by_frame):
        if frame >= len(membership):
            break
        for value in values:
            try:
                mouse_id = int(value)
            except (TypeError, ValueError, OverflowError):
                continue
            if 0 <= mouse_id < membership.shape[1]:
                membership[frame, mouse_id] = True

    sustained = np.zeros_like(membership)
    minimum = max(int(min_duration_frames), 1)
    gap_limit = max(int(max_gap_frames), 0)
    for mouse_id in range(membership.shape[1]):
        state = membership[:, mouse_id].copy()
        positions = np.flatnonzero(state)
        if positions.size > 1 and gap_limit:
            for left, right in zip(positions[:-1], positions[1:]):
                if int(right - left - 1) <= gap_limit:
                    state[int(left) : int(right) + 1] = True
        starts = np.flatnonzero(state & np.r_[True, ~state[:-1]])
        ends = np.flatnonzero(state & np.r_[~state[1:], True])
        for start, end in zip(starts, ends):
            if int(end - start + 1) >= minimum:
                sustained[int(start) : int(end) + 1, mouse_id] = True

    return [
        tuple(int(mouse_id) for mouse_id in np.flatnonzero(sustained[frame]))
        for frame in range(len(sustained))
    ]


def _huddle_core_indices(
    adjacency: np.ndarray,
    component: Iterable[int],
    *,
    min_member_neighbors: int,
) -> list[int]:
    """Return a local dense core instead of an end-to-end graph chain.

    A connected component is only a candidate: a sequence of mice can be
    connected through pairwise links while its endpoints are not part of the
    same aggregation. Iterative k-core pruning removes members that do not
    have enough neighbours inside the same local group. For a huddle of
    three or more mice the default degree is two, so a three-mouse chain can
    never pass merely because its middle mouse connects both ends.
    """

    active = sorted({int(index) for index in component if int(index) >= 0})
    minimum = max(int(min_member_neighbors), 1)
    while len(active) >= 2:
        local = np.asarray(active, dtype=int)
        degrees = np.sum(adjacency[np.ix_(local, local)], axis=1).astype(int)
        remove = [int(index) for index, degree in zip(active, degrees) if int(degree) < minimum]
        if not remove:
            break
        remove_set = set(remove)
        active = [index for index in active if index not in remove_set]
    return active


def _huddle_core_density(
    adjacency: np.ndarray,
    core: Iterable[int],
    *,
    mode: str,
    local_neighbor_cap: int,
) -> float:
    """Measure cohesion without requiring a large huddle to be a clique.

    ``global`` preserves the historical complete-graph density measure.  The
    ``local`` mode measures each member's support against a bounded local
    neighbourhood instead.  This is important for a real huddle containing
    many mice: opposite members can be farther apart than the pair threshold,
    while every mouse can still have enough nearby group members.  The k-core
    gate remains responsible for rejecting a three-mouse end-to-end chain.
    """
    active = sorted({int(index) for index in core if int(index) >= 0})
    if len(active) < 2:
        return 0.0
    local = np.asarray(adjacency[np.ix_(active, active)], dtype=bool)
    np.fill_diagonal(local, False)
    degrees = np.sum(local, axis=1).astype(float)
    normalized_mode = str(mode).strip().lower()
    if normalized_mode == "local":
        cap = max(int(local_neighbor_cap), 1)
        possible_local_neighbors = max(min(len(active) - 1, cap), 1)
        support = np.clip(degrees / float(possible_local_neighbors), 0.0, 1.0)
        return float(np.mean(support))

    edge_count = int(np.count_nonzero(np.triu(local, k=1)))
    possible_edges = len(active) * (len(active) - 1) // 2
    return float(edge_count / max(possible_edges, 1))


def _sustained_huddle_members_by_frame(
    members_by_frame: Sequence[Iterable[int]],
    *,
    min_duration_frames: int,
    max_gap_frames: int,
) -> list[tuple[int, ...]]:
    """Keep huddle members whose own local-core membership is sustained.

    Requiring the *entire* member tuple to remain byte-for-byte identical is
    too strict in a crowded huddle: one occluded mouse can disappear or receive
    a replacement track ID while the other animals remain in the same dense
    local core.  Confirm each logical member independently, bridge only short
    gaps, and still require at least three confirmed members downstream.  A
    moving pair with a rapidly changing third mouse therefore remains rejected.
    """

    normalized = [
        tuple(sorted({int(value) for value in members if int(value) >= 0}))
        for members in members_by_frame
    ]
    largest_id = max((value for members in normalized for value in members), default=-1)
    if largest_id < 0:
        return [() for _ in normalized]
    return _sustained_member_ids_by_frame(
        normalized,
        frames=len(normalized),
        mice=largest_id + 1,
        min_duration_frames=max(int(min_duration_frames), 1),
        max_gap_frames=max(int(max_gap_frames), 0),
    )


def _identity_continuous_huddle_segments(
    mask: np.ndarray,
    member_ids_by_frame: Sequence[Iterable[int]],
    *,
    max_gap_frames: int,
    min_shared_members: int = 2,
    min_overlap_fraction: float = 0.50,
    min_member_duration_frames: int = 1,
) -> list[tuple[np.ndarray, list[tuple[int, ...]]]]:
    """Split a global huddle channel into identity-continuous group events.

    A cage can contain different local huddles at different positions.  A
    boolean video-level mask alone cannot distinguish those components: the
    generic FSM may fill a short false gap and incorrectly merge one group
    into a later, unrelated group.  Keep a huddle lineage only when adjacent
    observations share enough tracked members.  One detector hand-off is
    tolerated because a three-mouse group still shares its other two members.

    The returned membership timeline is restricted to IDs observed in that
    lineage.  This prevents IDs from a neighbouring group or a recovery gap
    leaking into the event-level union used by rendering and export.
    """

    values = np.asarray(mask, dtype=bool).reshape(-1)
    if not values.size:
        return []
    gap_limit = max(int(max_gap_frames), 0)
    shared_floor = max(int(min_shared_members), 1)
    overlap_fraction = float(np.clip(min_overlap_fraction, 0.0, 1.0))
    positive_frames = np.flatnonzero(values)
    if not positive_frames.size:
        return []

    segments: list[list[int]] = []
    active_frames: list[int] = []
    previous_frame = -1
    previous_members: set[int] = set()
    anchor_members: set[int] = set()

    def required_overlap(left: set[int], right: set[int]) -> int:
        smaller_group = min(len(left), len(right))
        required = max(
            shared_floor,
            int(np.ceil(smaller_group * overlap_fraction)),
        )
        return min(required, smaller_group)

    for frame_value in positive_frames:
        frame = int(frame_value)
        current_members = set(_frame_member_ids(member_ids_by_frame, frame))
        continue_lineage = False
        if active_frames and current_members and previous_members:
            gap = frame - previous_frame - 1
            continue_lineage = (
                gap <= gap_limit
                and len(previous_members.intersection(current_members))
                >= required_overlap(previous_members, current_members)
                and len(anchor_members.intersection(current_members))
                >= required_overlap(anchor_members, current_members)
            )
        if not continue_lineage and active_frames:
            segments.append(active_frames)
            active_frames = []
            anchor_members = set()
        if not active_frames:
            anchor_members = current_members.copy()
        active_frames.append(frame)
        previous_frame = frame
        previous_members = current_members
    if active_frames:
        segments.append(active_frames)

    result: list[tuple[np.ndarray, list[tuple[int, ...]]]] = []
    for segment_frames in segments:
        member_counts: dict[int, int] = {}
        for frame in segment_frames:
            for member_id in _frame_member_ids(member_ids_by_frame, frame):
                member_counts[member_id] = member_counts.get(member_id, 0) + 1
        member_minimum = max(int(min_member_duration_frames) - gap_limit, 1)
        allowed_members = {
            member_id for member_id, count in member_counts.items() if count >= member_minimum
        }
        start, end = segment_frames[0], segment_frames[-1]
        lineage_members: list[tuple[int, ...]] = [() for _ in range(len(values))]
        segment_mask = np.zeros(values.shape, dtype=bool)
        for frame in range(start, end + 1):
            lineage_members[frame] = tuple(
                member_id
                for member_id in _frame_member_ids(member_ids_by_frame, frame)
                if member_id in allowed_members
            )
            segment_mask[frame] = values[frame] and len(lineage_members[frame]) >= 3
        if not segment_mask.any():
            continue
        result.append((segment_mask, lineage_members))
    return result


def _independent_pair_behavior_members_by_frame(
    events: Sequence[Mapping[str, Any]] | None,
    *,
    behavior: str,
    frames: int,
    sample_stride: int,
    min_duration_frames: int,
    group_mask: np.ndarray,
    group_members_by_frame: Sequence[Iterable[int]],
    min_independent_frames: int,
) -> list[set[int]]:
    """Return pair-event members with evidence outside a stable group state.

    Dense huddles can create box overlap and detector jitter that resembles an
    attack.  The group layer therefore keeps priority when a putative attack
    exists only inside the same stable huddle.  Conversely, an attack that
    continues independently before or after that group state is allowed to
    remove its two participants from the huddle candidate.  This consumes
    only in-memory FSM events; video names and label folders are never read.
    """

    result = [set() for _ in range(max(int(frames), 0))]
    if not result:
        return result
    stride = max(int(sample_stride), 1)
    minimum = max(int(min_duration_frames), 1)
    independent_minimum = max(int(min_independent_frames), 1)
    reference_mask = np.asarray(group_mask, dtype=bool).reshape(-1)
    if len(reference_mask) != len(result):
        raise ValueError("group_mask must match the requested analysis-frame count")

    def integer(value: Any, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError, OverflowError):
            return int(default)

    def analysis_bounds(event: Mapping[str, Any]) -> tuple[int, int]:
        if "analysis_start_frame" in event or "analysis_end_frame" in event:
            start = integer(event.get("analysis_start_frame"), 0)
            end = integer(event.get("analysis_end_frame"), start)
            return start, end
        start = integer(event.get("core_start_frame", event.get("start_frame", 0)), 0)
        end = integer(event.get("core_end_frame", event.get("end_frame", start)), start)
        return start // stride, end // stride

    def record_members(record: Mapping[str, Any]) -> set[int]:
        members: set[int] = set()
        for key in ("actor_id", "target_id"):
            value = integer(record.get(key), -1)
            if value >= 0:
                members.add(value)
        return members

    for event in events or ():
        if str(event.get("behavior", "")).strip().lower() != str(behavior).strip().lower():
            continue
        level = str(event.get("candidate_level", "extended")).strip().lower()
        if level not in {"extended", "strong"}:
            continue
        event_start, event_end = analysis_bounds(event)
        if event_end - event_start + 1 < minimum:
            continue

        event_members = [set() for _ in range(len(result))]
        traces = event.get("role_trace")
        trace_records = (
            [trace for trace in traces if isinstance(trace, Mapping)]
            if isinstance(traces, Sequence) and not isinstance(traces, (str, bytes))
            else []
        )
        if trace_records:
            for trace in trace_records:
                members = record_members(trace)
                if not members:
                    continue
                # role_trace is exported in source-frame coordinates.
                start = integer(trace.get("start_frame"), event_start * stride) // stride
                end = integer(trace.get("end_frame"), event_end * stride) // stride
                start = max(start, 0)
                end = min(end, len(result) - 1)
                for frame in range(start, end + 1):
                    event_members[frame].update(members)
        else:
            members = record_members(event)
            if not members:
                continue
            start = max(event_start, 0)
            end = min(event_end, len(result) - 1)
            for frame in range(start, end + 1):
                event_members[frame].update(members)

        overlap_frames = 0
        independent_frames = 0
        for frame, members in enumerate(event_members):
            if len(members) < 2:
                continue
            group_members = set(_frame_member_ids(group_members_by_frame, frame))
            if reference_mask[frame] and members.issubset(group_members):
                overlap_frames += 1
            else:
                independent_frames += 1
        if not overlap_frames:
            continue
        if independent_frames < independent_minimum:
            if isinstance(event, dict):
                event["huddle_conflict_status"] = "contained_by_stable_huddle"
                event["huddle_independent_frames"] = int(independent_frames)
            continue
        if isinstance(event, dict):
            event["huddle_conflict_status"] = "independent_pair_behavior"
            event["huddle_independent_frames"] = int(independent_frames)
        for frame, members in enumerate(event_members):
            result[frame].update(members)
    return result


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
    fsm_region_id: str | None = None,
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
        region_id=str(fsm_region_id or pair_key or event_scope),
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

        # Role is an event-level causal relationship, not a property of the
        # single frame that happens to have the highest score. Occlusion or a
        # crossing can reverse the peak-frame geometry even when the majority
        # of the confirmed FSM span has one stable actor/target direction.
        if actor_values.size > end and target_values.size > end:
            role_pairs = np.column_stack(
                (actor_values[start : end + 1], target_values[start : end + 1])
            )
            known_pairs = role_pairs[(role_pairs[:, 0] >= 0) & (role_pairs[:, 1] >= 0)]
            if known_pairs.size:
                unique_pairs, pair_counts = np.unique(known_pairs, axis=0, return_counts=True)
                modal_pair = unique_pairs[int(np.argmax(pair_counts))]
                actor, target = int(modal_pair[0]), int(modal_pair[1])

        def majority_known(values: np.ndarray) -> int:
            known = values[start : end + 1]
            known = known[np.isfinite(known) & (known >= 0)]
            if not known.size:
                return -1
            unique, counts = np.unique(known.astype(int), return_counts=True)
            return int(unique[int(np.argmax(counts))])

        # A direction can be tied exactly at the score peak even when the
        # surrounding FSM span has a stable role. Prefer the modal known role
        # within that span instead of exporting ``-1`` merely because one
        # frame was ambiguous. This is especially important for short clips
        # that begin in an avoidance reaction.
        if actor < 0:
            actor = majority_known(actor_values)
        if target < 0:
            target = majority_known(target_values)
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
        member_trace = _member_trace_segments(
            member_ids_by_frame,
            start=start,
            end=end,
            sample_stride=sample_stride,
        )
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
        if member_trace:
            row["member_trace"] = member_trace
        if str(event_scope) == "pair" and actor >= 0 and target >= 0:
            row["role_trace"] = [
                {
                    "pair_key": str(pair_key),
                    "actor_id": actor,
                    "target_id": target,
                    "start_frame": core_source_start,
                    "end_frame": core_source_end,
                }
            ]
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
            # The generic profile keeps the historical absolute-threshold
            # path.  Short labelled clips can opt into the relative,
            # hysteretic path explicitly without changing other users.
            "semantic_fsm": {
                "enabled": False,
                "semantic_approach": {
                    "min_relative_distance_drop": 0.03,
                    "min_pursuit_alignment": 0.25,
                    "min_speed_gap_ratio": 0.10,
                    "hold_seconds": 0.75,
                },
                "semantic_chase": {
                    "min_pursuit_alignment": 0.35,
                    "min_target_escape_alignment": 0.20,
                    "min_direction_similarity": 0.20,
                    "min_combined_speed_cm_s": 3.0,
                    "hold_seconds": 0.35,
                    "min_duration_seconds": 1.0,
                    "fill_gap_seconds": 0.35,
                },
                "semantic_avoidance": {
                    "near_distance_quantile": 0.35,
                    "near_distance_multiplier": 1.50,
                    "context_seconds": 3.0,
                    "boundary_context_seconds": 0.50,
                    "allow_clip_start_context": True,
                    "min_target_escape_alignment": 0.35,
                    "min_target_turn_angle_deg": 25.0,
                    "turn_window_seconds": 0.75,
                    "min_evader_speed_cm_s": 1.0,
                    "min_relative_distance_increase": 0.03,
                    "hold_seconds": 0.75,
                    "min_duration_seconds": 1.0,
                    "fill_gap_seconds": 0.35,
                },
                "semantic_attack": {
                    "min_attack_score": 0.65,
                    "min_dynamic_score": 0.55,
                    "min_raw_speed_cm_s": 8.0,
                    "min_target_turn_angle_deg": 35.0,
                    "min_pose_deformation": 0.12,
                    "min_candidate_duration_seconds": 0.10,
                    "min_contact_frames": 2,
                    "contact_support_seconds": 0.20,
                    "contact_hold_min_fraction": 0.25,
                    "min_normalized_distance_increase": 0.50,
                    "separation_context_seconds": 0.75,
                    "fill_gap_seconds": 0.50,
                },
                "identity_bridge_seconds": 0.50,
                "max_identity_bridge_participants": 6,
                "max_identity_bridge_segments": 2,
            },
            "pair_fill_gap_seconds": 0.15,
        },
        "group": {
            # The Beiyi definition uses a strict five-centimetre spatial
            # condition. A profile can override this for a calibrated setup.
            "huddle_distance_cm": 5.0,
            "huddle_fraction": 0.55,
            "huddle_min_cluster_size": 3,
            "huddle_min_cluster_fraction": 0.0,
            "huddle_min_cluster_density": 0.5,
            # ``False`` selects the local k-core rule. ``True`` remains a
            # backwards-compatible full-clique mode for older profiles.
            "huddle_require_clique": False,
            "huddle_min_member_neighbors": 2,
            # Local density is the default because a large huddle need not be
            # a complete graph: far diagonal members can be valid group
            # members when each mouse has enough nearby neighbours. Profiles
            # that need the historical all-pairs density can opt into global.
            "huddle_density_mode": "local",
            "huddle_local_neighbor_cap": 4,
            # A huddle event may tolerate one changing detector ID, but a
            # later component with no shared physical members is a new event.
            "huddle_event_min_shared_members": 2,
            "huddle_event_min_overlap_fraction": 0.50,
            # A fight must continue outside the same stable huddle before it
            # may displace huddle membership. This rejects box-jitter attacks
            # that exist only while a dense group is visible.
            "huddle_resolve_attack_conflicts": True,
            "huddle_attack_independent_seconds": 0.75,
            # The body-length cap is only a scale-drift guard. A calibrated
            # profile can disable it so mixed body sizes cannot shrink a fixed
            # spatial threshold.
            "huddle_body_length_cap_enabled": True,
            "huddle_max_pair_distance_body_lengths": 1.25,
            "isolation_distance_cm": 15.0,
            "isolation_neighbor_fraction": 0.15,
            "isolation_min_cluster_size": 3,
            "isolation_min_main_cluster_fraction": 0.60,
            "isolation_max_member_fraction": 0.25,
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


def _minimum_unspecified_duration_seconds(extended_config: Mapping[str, Any]) -> float:
    """Return the profile-wide floor for behaviors without a fixed duration.

    The generic configuration keeps its historical permissive defaults.  A
    profile can opt into a stricter document policy with
    ``extended_behavior.unspecified_min_duration_seconds`` without changing
    other profiles or the underlying distance/velocity thresholds.
    """

    try:
        value = float(extended_config.get("unspecified_min_duration_seconds", 0.0))
    except (TypeError, ValueError):
        return 0.0
    return max(value, 0.0)


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
    # The Beiyi profile uses the relative semantic FSM below.  Do not append
    # the historical short-clip fallback on top of it: that would duplicate
    # events and could reintroduce labels that do not satisfy the causal rules
    # (for example, an avoidance without a preceding approach).
    if bool(dict(social.get("semantic_fsm", {})).get("enabled", False)):
        return []
    chase_cfg = dict(social.get("chase_fallback", {}))
    attack_cfg = dict(social.get("attack_fallback", {}))
    unspecified_min_duration_seconds = _minimum_unspecified_duration_seconds(cfg)
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
            unspecified_min_duration_seconds,
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
                        unspecified_min_duration_seconds,
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


def _semantic_extended_pair_events(
    pair_df: pd.DataFrame,
    *,
    enriched: pd.DataFrame,
    source_video: Path,
    source_fps: float,
    sample_stride: int,
    social: Mapping[str, Any],
    config: Mapping[str, Any],
    fsm_coordinator: ParallelBehaviorFSM,
) -> list[dict[str, Any]]:
    """Emit relative social FSM channels for a short labelled clip.

    This channel is deliberately opt-in.  It is used by the Beiyi profile,
    where the examples are too short to justify hard absolute speed/distance
    gates.  The final event-duration policy is still applied by the common
    finalizer after all candidate pairs have been identity-stitched.
    """

    semantic = dict(social.get("semantic_fsm", {}))
    if not bool(semantic.get("enabled", False)):
        return []
    signals = build_semantic_pair_signals(
        pair_df,
        enriched,
        fps=source_fps / max(sample_stride, 1),
        social_config=semantic,
        contact_config=dict(config.get("contact_detection", {})),
    )
    analysis_fps = source_fps / max(sample_stride, 1)
    unspecified_min_duration_seconds = _minimum_unspecified_duration_seconds(
        _extended_behavior_config(config)
    )
    pair_key = str(pair_df["pair_key"].iloc[0])
    events: list[dict[str, Any]] = []

    def emit(
        behavior: str,
        mask_key: str,
        score_key: str,
        actor_key: str,
        target_key: str,
        *,
        minimum_seconds: float,
        fill_gap_seconds: float,
        recovery: str,
        padding_seconds: float = 0.0,
        maximum_short_seconds: float = 0.0,
    ) -> None:
        events.extend(
            _event_rows_from_mask(
                signals[mask_key],
                behavior=behavior,
                level="extended",
                fps=analysis_fps,
                source_video=source_video,
                sample_stride=sample_stride,
                score=signals[score_key],
                actor_id=signals[actor_key],
                target_id=signals[target_key],
                pair_key=pair_key,
                min_duration_seconds=max(float(minimum_seconds), 1.0 / analysis_fps),
                fill_gap_seconds=max(float(fill_gap_seconds), 0.0),
                short_event_padding_seconds=max(float(padding_seconds), 0.0),
                short_event_max_duration_seconds=max(float(maximum_short_seconds), 0.0),
                event_recovery=recovery,
                fsm_coordinator=fsm_coordinator,
            )
        )

    social_chase = dict(semantic.get("semantic_chase", {}))
    social_avoidance = dict(semantic.get("semantic_avoidance", {}))
    social_attack = dict(semantic.get("semantic_attack", {}))
    try:
        duration_floor = max(float(unspecified_min_duration_seconds), 0.0)
    except (TypeError, ValueError):
        duration_floor = 0.0

    # Together is retained as a separate social state.  Approach/chase/
    # avoidance are independent transition channels and may overlap it in the
    # evidence timeline; the renderer resolves that overlap with approach as
    # the highest social display state, without deleting the underlying event
    # rows.  The persisted evidence remains parallel and auditable.
    valid = signals["contact_mask"] | np.asarray(
        pair_df.get("valid_pair", pd.Series(False, index=pair_df.index)), dtype=bool
    )
    distance = pd.to_numeric(pair_df["center_distance_cm"], errors="coerce").to_numpy(float)
    selected_actor_speed = (
        pd.to_numeric(pair_df.get("selected_actor_behavior_speed_cm_s", 0.0), errors="coerce")
        .fillna(0.0)
        .to_numpy(float)
    )
    selected_target_speed = (
        pd.to_numeric(pair_df.get("selected_target_behavior_speed_cm_s", 0.0), errors="coerce")
        .fillna(0.0)
        .to_numpy(float)
    )
    together = (
        valid
        & np.isfinite(distance)
        & (distance <= float(social.get("together_max_distance_cm", 8.0)))
        & (
            selected_actor_speed + selected_target_speed
            <= float(social.get("together_max_combined_speed_cm_s", 28.0))
        )
    )
    events.extend(
        _event_rows_from_mask(
            together,
            behavior="together",
            level="extended",
            fps=analysis_fps,
            source_video=source_video,
            sample_stride=sample_stride,
            score=np.where(
                together,
                1.0
                - np.clip(
                    distance / max(float(social.get("together_max_distance_cm", 8.0)), 1e-6),
                    0.0,
                    1.0,
                ),
                0.0,
            ),
            actor_id=np.full(len(pair_df), -1),
            target_id=np.full(len(pair_df), -1),
            pair_key=pair_key,
            min_duration_seconds=float(social.get("together_min_duration_seconds", 0.30)),
            fill_gap_seconds=float(social.get("pair_fill_gap_seconds", 0.15)),
            fsm_coordinator=fsm_coordinator,
        )
    )
    emit(
        "approach",
        "approach_mask",
        "approach_score",
        "approach_actor",
        "approach_target",
        minimum_seconds=max(
            float(social.get("approach_min_duration_seconds", 0.10)),
            duration_floor,
        ),
        fill_gap_seconds=float(social.get("pair_fill_gap_seconds", 0.15)),
        recovery="semantic_approach",
        padding_seconds=float(social.get("approach_short_event_padding_seconds", 0.10)),
        maximum_short_seconds=float(social.get("approach_short_event_max_duration_seconds", 0.35)),
    )
    emit(
        "chase",
        "chase_mask",
        "chase_score",
        "chase_actor",
        "chase_target",
        minimum_seconds=max(
            float(social_chase.get("min_duration_seconds", 1.0)),
            duration_floor,
        ),
        fill_gap_seconds=float(social_chase.get("fill_gap_seconds", 0.35)),
        recovery="semantic_chase",
        padding_seconds=float(social.get("approach_short_event_padding_seconds", 0.10)),
        maximum_short_seconds=float(social.get("approach_short_event_max_duration_seconds", 0.60)),
    )
    emit(
        "avoidance",
        "avoidance_mask",
        "avoidance_score",
        "avoidance_actor",
        "avoidance_target",
        minimum_seconds=max(
            float(social_avoidance.get("min_duration_seconds", 1.0)),
            duration_floor,
        ),
        fill_gap_seconds=float(social_avoidance.get("fill_gap_seconds", 0.35)),
        recovery="semantic_avoidance",
        padding_seconds=float(social.get("avoidance_short_event_padding_seconds", 0.10)),
        maximum_short_seconds=float(social.get("avoidance_short_event_max_duration_seconds", 0.35)),
    )

    # Attack candidates intentionally use a one-frame local candidate floor so
    # that the identity bridge can join several supported frames.  The common
    # finalizer still requires the configured one-second core duration, which
    # prevents a lone detector spike from becoming a rendered attack event.
    emit(
        "attack",
        "attack_mask",
        "attack_score",
        "attack_actor",
        "attack_target",
        minimum_seconds=float(social_attack.get("min_candidate_duration_seconds", 0.10)),
        fill_gap_seconds=float(social_attack.get("fill_gap_seconds", 0.50)),
        recovery="semantic_attack_candidate",
        padding_seconds=float(social.get("attack_short_event_padding_seconds", 0.15)),
        maximum_short_seconds=float(social.get("attack_short_event_max_duration_seconds", 0.40)),
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
    unspecified_min_duration_seconds = _minimum_unspecified_duration_seconds(cfg)
    n = len(pair_df)
    analysis_fps = source_fps / max(sample_stride, 1)
    fsm_coordinator = fsm_coordinator or ParallelBehaviorFSM(
        dict(config.get("parallel_behavior_fsm", {}))
    )
    semantic_cfg = dict(social.get("semantic_fsm", {}))
    if bool(semantic_cfg.get("enabled", False)):
        return _semantic_extended_pair_events(
            pair_df,
            enriched=(enriched if enriched is not None else pd.DataFrame(index=pair_df.index)),
            source_video=source_video,
            source_fps=source_fps,
            sample_stride=sample_stride,
            social=social,
            config=config,
            fsm_coordinator=fsm_coordinator,
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
            min_duration_seconds=float(social.get("together_min_duration_seconds", 0.30)),
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
            min_duration_seconds=max(
                float(social["approach_min_duration_seconds"]),
                unspecified_min_duration_seconds,
            ),
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
    avoidance_min_duration_seconds = max(
        float(social["avoidance_min_duration_seconds"]),
        unspecified_min_duration_seconds,
    )
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
    pair_behavior_events: Sequence[Mapping[str, Any]] | None = None,
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
                    min_duration_seconds=float(
                        individual_cfg.get(
                            f"{behavior}_min_duration_seconds",
                            individual_cfg["confirm_seconds"],
                        )
                    ),
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
    huddle_core_size = np.zeros(frames, dtype=int)
    huddle_core_fraction = np.zeros(frames, dtype=float)
    huddle_core_density = np.zeros(frames, dtype=float)
    huddle_min_cluster_size = max(int(group_cfg.get("huddle_min_cluster_size", 3)), 3)
    huddle_min_cluster_fraction = max(
        float(group_cfg.get("huddle_min_cluster_fraction", 0.0)),
        0.0,
    )
    huddle_min_cluster_density = float(
        np.clip(group_cfg.get("huddle_min_cluster_density", 0.5), 0.0, 1.0)
    )
    huddle_require_clique = bool(group_cfg.get("huddle_require_clique", False))
    huddle_min_member_neighbors = max(
        int(group_cfg.get("huddle_min_member_neighbors", 2)),
        2,
    )
    huddle_density_mode = str(group_cfg.get("huddle_density_mode", "local")).strip().lower()
    if huddle_density_mode not in {"global", "local"}:
        huddle_density_mode = "local"
    try:
        huddle_local_neighbor_cap = max(
            int(group_cfg.get("huddle_local_neighbor_cap", 4)),
            huddle_min_member_neighbors,
        )
    except (TypeError, ValueError):
        huddle_local_neighbor_cap = max(huddle_min_member_neighbors, 4)
    huddle_body_length_cap_enabled = bool(group_cfg.get("huddle_body_length_cap_enabled", True))
    try:
        huddle_max_pair_distance_body_lengths = float(
            group_cfg.get("huddle_max_pair_distance_body_lengths", float("inf"))
        )
    except (TypeError, ValueError):
        huddle_max_pair_distance_body_lengths = float("inf")
    isolation_min_main_cluster_fraction = float(
        np.clip(group_cfg.get("isolation_min_main_cluster_fraction", 0.60), 0.0, 1.0)
    )
    body_cm = np.asarray(
        kin.get("body_cm", np.full((frames, mice), np.nan)),
        dtype=float,
    )
    if body_cm.shape != (frames, mice):
        body_cm = np.full((frames, mice), np.nan, dtype=float)
    try:
        reference_body_cm = max(float(kin.get("reference_body_cm", 8.0)), 1e-6)
    except (TypeError, ValueError):
        reference_body_cm = 8.0
    huddle_members_by_frame: list[tuple[int, ...]] = [() for _ in range(frames)]
    raw_isolation_members_by_frame: list[tuple[int, ...]] = [() for _ in range(frames)]
    for frame in range(frames):
        ids = np.flatnonzero(valid[frame])
        group_size[frame] = len(ids)
        if len(ids) < 2:
            continue
        points = centers[frame, ids]
        distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
        distances[~np.isfinite(distances)] = np.inf
        np.fill_diagonal(distances, np.inf)
        frame_body = body_cm[frame, ids]
        finite_body = frame_body[np.isfinite(frame_body) & (frame_body >= 1.0)]
        frame_body_scale = float(np.median(finite_body)) if finite_body.size else reference_body_cm
        close_threshold = float(group_cfg["huddle_distance_cm"])
        if huddle_body_length_cap_enabled and np.isfinite(huddle_max_pair_distance_body_lengths):
            close_threshold = min(
                close_threshold,
                frame_body_scale * max(huddle_max_pair_distance_body_lengths, 0.0),
            )
        nearest = np.min(distances, axis=1)
        close_fraction[frame] = float(np.mean(nearest <= close_threshold))

        # A multi-mouse cage can contain a local huddle while other visible
        # mice remain spread out. First find local connected components, then
        # apply the strict local-core gate below. Thus a valid local
        # three-mouse huddle is retained, while a three-mouse end-to-end chain
        # is not promoted to a group event.
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
        largest_component = max(components, key=len, default=[])
        largest_cluster_size[frame] = len(largest_component)
        largest_cluster_fraction[frame] = len(largest_component) / max(len(ids), 1)
        if len(largest_component) >= 2:
            component_indices = np.asarray(largest_component, dtype=int)
            component_distances = distances[np.ix_(component_indices, component_indices)]
            edge_count = int(np.sum(component_distances <= close_threshold) // 2)
            possible_edges = len(largest_component) * (len(largest_component) - 1) // 2
            largest_cluster_density[frame] = edge_count / max(possible_edges, 1)

        # A huddle must contain a local core of at least three mice. The
        # legacy full-clique option is retained for old profiles, while the
        # default k-core path tolerates a larger scene around the huddle and
        # still rejects sparse connected chains.
        huddle_candidates: list[tuple[int, float, list[int]]] = []
        for component in components:
            if huddle_require_clique:
                core = sorted(int(index) for index in component)
                required_core_density = 1.0
                density_mode = "global"
            else:
                core = _huddle_core_indices(
                    adjacency,
                    component,
                    min_member_neighbors=huddle_min_member_neighbors,
                )
                required_core_density = huddle_min_cluster_density
                density_mode = huddle_density_mode
            if len(core) < huddle_min_cluster_size:
                continue
            core_density = _huddle_core_density(
                adjacency,
                core,
                mode=density_mode,
                local_neighbor_cap=huddle_local_neighbor_cap,
            )
            if core_density >= required_core_density:
                huddle_candidates.append((len(core), core_density, core))
        if huddle_candidates:
            _, core_density, huddle_core = max(
                huddle_candidates,
                key=lambda item: (item[0], item[1]),
            )
            huddle_core_size[frame] = len(huddle_core)
            huddle_core_fraction[frame] = len(huddle_core) / max(len(ids), 1)
            huddle_core_density[frame] = core_density
            huddle_members_by_frame[frame] = tuple(sorted(int(ids[index]) for index in huddle_core))
        # Isolation is a relationship to the main group, not a requirement
        # that isolated mice make up a fixed fraction of all visible mice.
        # This is important for a 10-mouse scene with one genuinely isolated
        # mouse: the old fraction gate silently discarded that event.
        isolation_min_cluster_size = max(
            int(group_cfg.get("isolation_min_cluster_size", 3)),
            3,
        )
        main_cluster_fraction = len(largest_component) / max(len(ids), 1)
        main_cluster_ids = (
            set(int(ids[index]) for index in largest_component)
            if (
                len(largest_component) >= isolation_min_cluster_size
                and main_cluster_fraction >= isolation_min_main_cluster_fraction
            )
            else set()
        )
        if main_cluster_ids:
            isolation_candidates = ids[
                (nearest >= float(group_cfg["isolation_distance_cm"]))
                & ~np.isin(ids, tuple(sorted(main_cluster_ids)))
            ]
            if len(isolation_candidates):
                candidate_positions = np.flatnonzero(np.isin(ids, isolation_candidates))
                # ``isolation`` is a single-mouse group relation. A crowded
                # cage can contain several mice outside the main component,
                # especially when tracking duplicates a partially occluded
                # animal. Keep the strongest spatial outlier and let the
                # per-ID duration gate below reject unstable candidates. The
                # previous ratio gate discarded every candidate whenever more
                # than 20-25% of the scene was far from the main component.
                strongest_position = candidate_positions[
                    int(np.argmax(nearest[candidate_positions]))
                ]
                isolated_ids = np.asarray([ids[strongest_position]], dtype=ids.dtype)
            else:
                isolated_ids = np.asarray([], dtype=ids.dtype)
        else:
            isolated_ids = np.asarray([], dtype=ids.dtype)
        isolated_fraction[frame] = len(isolated_ids) / max(len(ids), 1)
        raw_isolation_members_by_frame[frame] = tuple(sorted(int(item) for item in isolated_ids))

    raw_huddle = huddle_core_size >= huddle_min_cluster_size
    raw_huddle &= huddle_core_density >= huddle_min_cluster_density
    if huddle_min_cluster_fraction > 0.0:
        raw_huddle &= largest_cluster_fraction >= huddle_min_cluster_fraction

    huddle_min_duration_seconds = max(
        float(
            group_cfg.get(
                "huddle_min_duration_seconds",
                group_cfg.get("confirm_seconds", 0.30),
            )
        ),
        0.0,
    )
    huddle_min_duration_frames = max(
        int(round(huddle_min_duration_seconds * analysis_fps)),
        1,
    )
    huddle_fill_gap_frames = max(
        int(round(float(group_cfg.get("fill_gap_seconds", 0.20)) * analysis_fps)),
        0,
    )
    # Conflict resolution must consume the same temporally confirmed huddle
    # state that can become a public event.  Using ``raw_huddle`` here lets a
    # transient three-mouse geometry (shorter than the huddle duration gate)
    # suppress a valid attack even though no huddle is ever emitted.
    stable_huddle_members_for_conflict = _sustained_huddle_members_by_frame(
        huddle_members_by_frame,
        min_duration_frames=huddle_min_duration_frames,
        max_gap_frames=huddle_fill_gap_frames,
    )
    stable_huddle_for_conflict = np.asarray(
        [len(members) >= huddle_min_cluster_size for members in stable_huddle_members_for_conflict],
        dtype=bool,
    )

    # Resolve the only cross-layer semantic conflict before temporal huddle
    # confirmation.  Group priority wins when an apparent attack exists only
    # inside the same dense aggregation (a common detector-jitter pattern).
    # A fight can displace its two members only when the pair FSM also has a
    # bounded amount of evidence outside that group state.
    if bool(group_cfg.get("huddle_resolve_attack_conflicts", True)):
        social_cfg = dict(cfg.get("social", {}))
        attack_minimum_seconds = max(
            float(cfg.get("unspecified_min_duration_seconds", 0.0)),
            float(social_cfg.get("attack_min_duration_seconds", 0.0)),
        )
        independent_attack_members = _independent_pair_behavior_members_by_frame(
            pair_behavior_events,
            behavior="attack",
            frames=frames,
            sample_stride=sample_stride,
            min_duration_frames=max(
                int(round(attack_minimum_seconds * analysis_fps)),
                1,
            ),
            group_mask=stable_huddle_for_conflict,
            group_members_by_frame=stable_huddle_members_for_conflict,
            min_independent_frames=max(
                int(
                    round(
                        float(group_cfg.get("huddle_attack_independent_seconds", 0.75))
                        * analysis_fps
                    )
                ),
                1,
            ),
        )
    else:
        independent_attack_members = [set() for _ in range(frames)]

    for frame, blocked_members in enumerate(independent_attack_members):
        if not blocked_members or not huddle_members_by_frame[frame]:
            continue
        remaining_ids = [
            member_id
            for member_id in huddle_members_by_frame[frame]
            if member_id not in blocked_members
        ]
        if len(remaining_ids) < huddle_min_cluster_size:
            huddle_members_by_frame[frame] = ()
            huddle_core_size[frame] = 0
            huddle_core_fraction[frame] = 0.0
            huddle_core_density[frame] = 0.0
            continue

        local_points = centers[frame, remaining_ids]
        local_distances = np.linalg.norm(
            local_points[:, None, :] - local_points[None, :, :],
            axis=2,
        )
        local_distances[~np.isfinite(local_distances)] = np.inf
        np.fill_diagonal(local_distances, np.inf)
        local_body = body_cm[frame, remaining_ids]
        finite_local_body = local_body[np.isfinite(local_body) & (local_body >= 1.0)]
        local_body_scale = (
            float(np.median(finite_local_body)) if finite_local_body.size else reference_body_cm
        )
        close_threshold = float(group_cfg["huddle_distance_cm"])
        if huddle_body_length_cap_enabled and np.isfinite(huddle_max_pair_distance_body_lengths):
            close_threshold = min(
                close_threshold,
                local_body_scale * max(huddle_max_pair_distance_body_lengths, 0.0),
            )
        local_adjacency = local_distances <= close_threshold
        if huddle_require_clique:
            local_core = list(range(len(remaining_ids)))
            required_density = 1.0
            density_mode = "global"
        else:
            local_core = _huddle_core_indices(
                local_adjacency,
                range(len(remaining_ids)),
                min_member_neighbors=huddle_min_member_neighbors,
            )
            required_density = huddle_min_cluster_density
            density_mode = huddle_density_mode
        local_density = _huddle_core_density(
            local_adjacency,
            local_core,
            mode=density_mode,
            local_neighbor_cap=huddle_local_neighbor_cap,
        )
        if len(local_core) < huddle_min_cluster_size or local_density < required_density:
            huddle_members_by_frame[frame] = ()
            huddle_core_size[frame] = 0
            huddle_core_fraction[frame] = 0.0
            huddle_core_density[frame] = 0.0
            continue
        resolved_members = tuple(sorted(int(remaining_ids[index]) for index in local_core))
        huddle_members_by_frame[frame] = resolved_members
        huddle_core_size[frame] = len(resolved_members)
        huddle_core_fraction[frame] = len(resolved_members) / max(group_size[frame], 1)
        huddle_core_density[frame] = local_density

    huddle_members_by_frame = _sustained_huddle_members_by_frame(
        huddle_members_by_frame,
        min_duration_frames=huddle_min_duration_frames,
        max_gap_frames=huddle_fill_gap_frames,
    )

    # Apply the isolation duration per mouse instead of collecting every ID
    # that was ever far from the main cluster. The video-level mask below
    # still keeps the full observed group event, while the member list used by
    # rendering/export contains only stable isolated individuals. This
    # prevents a short ID swap from changing ID 00 isolation into a false
    # ID 09 isolation label.
    isolation_min_duration_seconds = max(
        float(
            group_cfg.get(
                "isolation_min_duration_seconds",
                group_cfg.get("confirm_seconds", 0.30),
            )
        ),
        0.0,
    )
    isolation_members_by_frame = _sustained_member_ids_by_frame(
        raw_isolation_members_by_frame,
        frames=frames,
        mice=mice,
        min_duration_frames=max(
            int(round(isolation_min_duration_seconds * analysis_fps)),
            1,
        ),
        max_gap_frames=max(
            int(round(float(group_cfg.get("fill_gap_seconds", 0.20)) * analysis_fps)),
            0,
        ),
    )
    isolation_actor_by_frame = np.full(frames, -1, dtype=int)
    for frame, members in enumerate(isolation_members_by_frame):
        if len(members) == 1:
            isolation_actor_by_frame[frame] = int(members[0])

    # A huddle is a group behavior, so two close mice remain a social pair.
    # Require a local connected cluster of at least three visible mice before
    # emitting ``huddle``; this also keeps pair events eligible for display.
    stable_huddle = np.asarray(
        [len(members) >= huddle_min_cluster_size for members in huddle_members_by_frame],
        dtype=bool,
    )
    huddle = huddle_core_size >= huddle_min_cluster_size
    huddle &= huddle_core_density >= huddle_min_cluster_density
    huddle &= stable_huddle
    if huddle_min_cluster_fraction > 0.0:
        huddle &= largest_cluster_fraction >= huddle_min_cluster_fraction
    # A single mouse can be isolated from a large group.  Require a visible
    # main cluster of at least three mice and at least one isolated member;
    # do not require the isolated members to be a fixed fraction of the scene.
    isolation = (
        (group_size >= 3)
        & (largest_cluster_size >= max(int(group_cfg.get("isolation_min_cluster_size", 3)), 3))
        & (largest_cluster_fraction >= isolation_min_main_cluster_fraction)
        & (
            np.asarray(
                [len(item) for item in isolation_members_by_frame],
                dtype=int,
            )
            > 0
        )
    )
    for behavior, mask, score in (
        (
            "huddle",
            huddle,
            np.maximum(close_fraction, huddle_core_fraction * huddle_core_density),
        ),
        ("isolation", isolation, isolated_fraction),
    ):
        member_ids_by_frame = (
            huddle_members_by_frame if behavior == "huddle" else isolation_members_by_frame
        )
        if behavior == "huddle":
            group_segments = _identity_continuous_huddle_segments(
                mask,
                member_ids_by_frame,
                max_gap_frames=max(
                    int(round(float(group_cfg["fill_gap_seconds"]) * analysis_fps)),
                    0,
                ),
                min_shared_members=max(
                    int(group_cfg.get("huddle_event_min_shared_members", 2)),
                    1,
                ),
                min_overlap_fraction=float(
                    group_cfg.get("huddle_event_min_overlap_fraction", 0.50)
                ),
                min_member_duration_frames=max(
                    int(round(huddle_min_duration_seconds * analysis_fps)),
                    1,
                ),
            )
        else:
            group_segments = [(mask, list(member_ids_by_frame))]

        for segment_index, (segment_mask, segment_members) in enumerate(group_segments):
            events.extend(
                _event_rows_from_mask(
                    segment_mask,
                    behavior=behavior,
                    level="extended",
                    fps=analysis_fps,
                    source_video=source_video,
                    sample_stride=sample_stride,
                    score=score,
                    actor_id=(
                        isolation_actor_by_frame if behavior == "isolation" else np.full(frames, -1)
                    ),
                    target_id=np.full(frames, -1),
                    pair_key="group",
                    min_duration_seconds=(
                        huddle_min_duration_seconds
                        if behavior == "huddle"
                        else float(
                            group_cfg.get(
                                f"{behavior}_min_duration_seconds",
                                group_cfg["confirm_seconds"],
                            )
                        )
                    ),
                    fill_gap_seconds=float(group_cfg["fill_gap_seconds"]),
                    event_scope="group",
                    member_ids_by_frame=segment_members,
                    fsm_coordinator=fsm_coordinator,
                    fsm_region_id=(
                        f"huddle_lineage_{segment_index}" if behavior == "huddle" else None
                    ),
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


def _fill_contact_state_gaps(
    states: list[dict[str, Any] | None],
    max_gap_frames: int,
) -> None:
    """Bridge short missing contact observations with matching geometry.

    Pose confidence can briefly drop while two noses remain in contact.  The
    gap is eligible only when the same contact component is present on both
    sides.  This keeps a head contact from being joined to a tail contact and
    marks changed roles as ambiguous instead of inventing a direction.
    """

    gap_limit = max(int(max_gap_frames), 0)
    if gap_limit == 0 or not states:
        return
    index = 0
    while index < len(states):
        if states[index] is not None:
            index += 1
            continue
        start = index
        while index < len(states) and states[index] is None:
            index += 1
        end = index - 1
        if start == 0 or index >= len(states) or end - start + 1 > gap_limit:
            continue
        left = states[start - 1]
        right = states[index]
        if left is None or right is None:
            continue
        left_components = str(left.get("contact_type_components", ""))
        right_components = str(right.get("contact_type_components", ""))
        if left_components != right_components:
            continue
        filled = dict(left)
        left_role = (
            int(left.get("contact_actor_id", -1)),
            int(left.get("contact_target_id", -1)),
        )
        right_role = (
            int(right.get("contact_actor_id", -1)),
            int(right.get("contact_target_id", -1)),
        )
        if left_role != right_role:
            filled.update(
                {
                    "contact_direction": "both",
                    "contact_actor_id": -1,
                    "contact_target_id": -1,
                    "role_ambiguous": True,
                }
            )
        for field in (
            "contact_distance_cm",
            "nose_head_distance_cm",
            "nose_tail_distance_cm",
        ):
            try:
                left_value = float(left.get(field, float("inf")))
                right_value = float(right.get(field, float("inf")))
            except (TypeError, ValueError):
                continue
            if np.isfinite(left_value) and np.isfinite(right_value):
                filled[field] = (left_value + right_value) / 2.0
        for missing in range(start, end + 1):
            states[missing] = dict(filled)


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
    try:
        nose_head_multiplier = max(
            float(contact_config.get("nose_head_distance_multiplier", 1.0)),
            1.0,
        )
    except (TypeError, ValueError):
        nose_head_multiplier = 1.0
    nose_head_threshold *= nose_head_multiplier
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
            # ``states`` is pre-sized to the input timeline.  Assign by row
            # index instead of appending, otherwise one invalid endpoint can
            # shift all later contact states and corrupt the FSM spans.
            states[int(index)] = None
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

    try:
        fill_gap_seconds = max(float(contact_config.get("fill_gap_seconds", 0.0)), 0.0)
    except (TypeError, ValueError):
        fill_gap_seconds = 0.0
    fill_gap_frames = int(round(fill_gap_seconds * source_fps / sample_stride))
    _fill_contact_state_gaps(states, fill_gap_frames)

    def state_key(state: Mapping[str, Any] | None) -> tuple[Any, ...] | None:
        if state is None:
            return None
        # Direction and role are evidence attributes, not contact geometry.
        # Tracking jitter can flip them for one frame while the same nose
        # geometry remains continuously present.  Keeping them in the state
        # key used to fragment a single nose event into many short events.
        return (state["contact_type_components"],)

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
        representative = segment[peak_offset] if segment else segment[0]
        components = {
            component
            for state in segment
            for component in str(state["contact_type_components"]).split(";")
            if component
        }
        ordered_components = tuple(
            name for name in ("nose_head", "nose_tail") if name in components
        )
        directions = {
            str(state.get("contact_direction", ""))
            for state in segment
            if str(state.get("contact_direction", ""))
        }
        role_pairs = {
            (
                int(state.get("contact_actor_id", -1)),
                int(state.get("contact_target_id", -1)),
            )
            for state in segment
        }
        roles_known = all(actor >= 0 and target >= 0 for actor, target in role_pairs)
        if len(directions) == 1 and len(role_pairs) == 1 and roles_known:
            contact_direction = next(iter(directions))
            contact_actor_id, contact_target_id = next(iter(role_pairs))
            role_ambiguous = False
        else:
            contact_direction = (
                "both"
                if len(directions) > 1
                else str(representative.get("contact_direction", "both"))
            )
            contact_actor_id = -1
            contact_target_id = -1
            role_ambiguous = True
        contact_type = (
            "nose_head_and_nose_tail"
            if len(ordered_components) == 2
            else (
                ordered_components[0] if ordered_components else str(representative["contact_type"])
            )
        )
        contact_type_components = ";".join(ordered_components)
        component_minimums = []
        if "nose_head" in ordered_components:
            component_minimums.append(
                float(contact_config.get("nose_head_min_duration_seconds", 0.0))
            )
        if "nose_tail" in ordered_components:
            component_minimums.append(
                float(contact_config.get("nose_tail_min_duration_seconds", 0.0))
            )
        min_duration_seconds = max(max(component_minimums, default=0.0), 0.0)
        duration_seconds = (source_end - source_start + 1) / source_fps
        if duration_seconds < min_duration_seconds:
            continue
        events.append(
            {
                "contact_detector": "nose_head_nose_tail_geometry",
                "pair_key": str(pair_key),
                "contact_type": contact_type,
                "contact_type_components": contact_type_components,
                "contact_direction": contact_direction,
                "contact_actor_id": int(contact_actor_id),
                "contact_target_id": int(contact_target_id),
                "role_ambiguous": bool(role_ambiguous),
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
