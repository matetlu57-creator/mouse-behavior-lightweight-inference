"""Defer contradictory Static/posture intervals without changing source observations."""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import math
from numbers import Integral, Real

from .duration_policy import qualify_event_duration
from .event_boundaries import EventBoundary


def _union(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    return merged


def _intersection(first, second):
    return _union((max(a, c), min(b, d)) for a, b in first for c, d in second
                  if max(a, c) < min(b, d))


def _subtract(start, end, removed):
    result = []
    cursor = start
    for left, right in removed:
        if right <= cursor or left >= end:
            continue
        if cursor < left:
            result.append([cursor, min(left, end)])
        cursor = max(cursor, right)
    if cursor < end:
        result.append([cursor, end])
    return result


def _stable_id(event):
    content = {key: value for key, value in event.items() if key != "event_id"}
    digest = hashlib.sha256(json.dumps(content, sort_keys=True, allow_nan=False).encode()).hexdigest()[:10]
    return f"{event['canonical_behavior']}_{event['start_frame']:08d}_{event['end_frame']:08d}_{digest}"


def _fragment(event, start, end, fps):
    result = deepcopy(event)
    result.update(start_frame=start, end_frame=end,
                  parent_candidate_event_id=event.get("event_id"))
    result["boundary"] = EventBoundary(start, end, start, end, fps).as_dict()
    for key, value in list(result.items()):
        if key.endswith("_frame_intervals"):
            intervals = _intersection(value, [[start, end]])
            result[key] = intervals
            result[key.replace("_intervals", "_count")] = sum(b - a for a, b in intervals)
    if "observed_all_participants_frame_count" in result:
        result["inferred_or_partially_unobserved_frame_count"] = (
            end - start - result["observed_all_participants_frame_count"])
    result["event_id"] = _stable_id(result)
    return result


def defer_static_posture_conflicts(events, fps, parameters=None):
    """Return formal events, review-only conflicts, and an explicit audit.

    Union Static and Grooming/Rearing process bounds for each visual ID, then
    remove their intersection from both labels. Remaining independent pieces
    requalify using the existing duration policy, including tagged V3 minima.
    All positive-length conflicts remain in review even below posture minima.
    Source event dictionaries and nested support ledgers are never mutated.
    """
    if isinstance(fps, bool) or not isinstance(fps, Real) or not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    inputs = list(events)
    by_id = defaultdict(lambda: dict(stationary=[], posture=[], sources=[]))
    recognized = {}
    for index, event in enumerate(inputs):
        label = str(event.get("canonical_behavior", event.get("behavior", ""))).lower()
        label = {"static": "stationary"}.get(label, label)
        ids = event.get("mouse_ids", [])
        if label not in {"stationary", "grooming", "rearing"} or len(ids) != 1:
            continue
        if label in {"grooming", "rearing"} and event.get("posture_engine") != "v3_decoupled":
            continue
        start, end = event["start_frame"], event["end_frame"]
        if (isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, Integral)
                or not isinstance(end, Integral) or not 0 <= start < end):
            raise ValueError("individual event bounds must be positive half-open integer intervals")
        key = ids[0]
        recognized[index] = key
        by_id[key]["stationary" if label == "stationary" else "posture"].append([start, end])
        by_id[key]["sources"].append((event, label))
    conflicts = {key: _intersection(_union(group["stationary"]), _union(group["posture"]))
                 for key, group in by_id.items()}
    formal, review, rejected, changes = [], [], [], []
    for index, event in enumerate(inputs):
        key = recognized.get(index)
        if key is None or not conflicts[key]:
            formal.append(deepcopy(event))
            continue
        start, end = event["start_frame"], event["end_frame"]
        removed = _intersection([[start, end]], conflicts[key])
        if not removed:
            formal.append(deepcopy(event))
            continue
        retained = []
        for left, right in _subtract(start, end, removed):
            fragment = _fragment(event, left, right, fps)
            diagnostic = qualify_event_duration(fragment, fps, parameters=parameters)
            if diagnostic["ok"]:
                formal.append(fragment)
                retained.append(fragment["event_id"])
            else:
                rejected.append(dict(parent_candidate_event_id=event.get("event_id"),
                                     start_frame=left, end_frame=right, duration_qualification=diagnostic))
        changes.append(dict(parent_candidate_event_id=event.get("event_id"), mouse_ids=[key],
                            removed_frame_intervals=removed, retained_fragment_event_ids=retained))
    for key, intervals in conflicts.items():
        for start, end in intervals:
            sources = [(event, label) for event, label in by_id[key]["sources"]
                       if event["start_frame"] < end and event["end_frame"] > start]
            row = dict(behavior="PostureReview", canonical_behavior="posture_review", mouse_ids=[key],
                       start_frame=start, end_frame=end, review_queue_only=True,
                       requires_review=True, biological_label_confirmed=False,
                       review_reason="static_posture_process_overlap", source_static=True,
                       source_grooming=any(label == "grooming" for _, label in sources),
                       source_rearing=any(label == "rearing" for _, label in sources),
                       source_event_ids=sorted({event["event_id"] for event, _ in sources if "event_id" in event}),
                       source_canonical_behaviors=sorted({label for _, label in sources}))
            # Preserve each source ledger separately because overlapping support
            # from different candidate labels must never be summed as new evidence.
            row["source_support"] = [dict(source_event_id=event.get("event_id"),
                canonical_behavior=label, support={field: _intersection(value, [[start, end]])
                for field, value in event.items() if field.endswith("_frame_intervals")})
                for event, label in sources]
            for source in row["source_support"]:
                source["support_counts"] = {field.replace("_intervals", "_count"): sum(b - a for a, b in ranges)
                                            for field, ranges in source["support"].items()}
            row["event_id"] = _stable_id(row)
            review.append(row)
    def sort_key(row):
        return (row["start_frame"], row["end_frame"], row.get("canonical_behavior", ""), str(row.get("mouse_ids", [])))
    formal.sort(key=sort_key)
    review.sort(key=sort_key)
    audit = dict(revision="2026-10-26-v32-static-posture-conflicts-review-only",
                 input_event_count=len(inputs), formal_event_count=len(formal),
                 review_event_count=len(review), conflict_frame_count=sum(
                     end - start for intervals in conflicts.values() for start, end in intervals),
                 changed_events=changes, duration_rejected_fragments=rejected,
                 source_data_mutated=False, review_is_biological_label=False)
    return formal, review, audit
