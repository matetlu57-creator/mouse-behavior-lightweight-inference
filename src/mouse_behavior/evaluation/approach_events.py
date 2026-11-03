"""Requalify directional Approach children with the original shared rule.

Role splitting and window union change the candidate's endpoints. Duration and
path alone cannot prove that the new final frame still meets Approach geometry.
"""
from copy import deepcopy
import math
from pathlib import Path
import tempfile

import numpy as np

from mouse_behavior.behavior.event_boundaries import EventBoundary
from .screening_signals import runs
from .social_event_geometry import observed_path_and_net
from .social_heuristic_config import DEFAULT_SOCIAL_HEURISTIC_CONFIG


def requalify_approach_event(features, edge, event, parameters, *, retained_mask=None):
    from .confirmed_rules import _open_approach_mask
    slots = {int(track): slot for slot, track in enumerate(features.track_ids)}
    actor, target = slots.get(event.get("actor_id")), slots.get(event.get("target_id"))
    if actor is None or target is None or actor == target:
        return []
    if {actor, target} != {int(features.pair_i[edge]), int(features.pair_j[edge])}:
        raise ValueError("Approach actor and target must match the measured pair")
    start, end = event["start_frame"], event["end_frame"]
    EventBoundary(start, end, start, end, features.fps)
    if end > len(features.valid):
        raise ValueError("Approach bounds exceed the feature window")
    retained = np.zeros(len(features.valid), bool)
    retained[start:end] = True
    if retained_mask is not None:
        retained &= retained_mask
    evidence = {}
    _open_approach_mask(features, edge, parameters, direction_evidence=evidence, retained_mask=retained)
    direction = 0 if actor == int(features.pair_i[edge]) else 1
    mask = evidence["candidates"][direction] & retained
    observed = features.valid[:, actor] & features.valid[:, target]
    output = []
    for low, high in runs(mask):
        child = deepcopy(event)
        child.update(start_frame=int(low), end_frame=int(high))
        for key, value in event.items():
            if key.endswith("_frame_intervals"):
                spans = [[max(a, int(low)), min(b, int(high))] for a, b in value
                         if max(a, low) < min(b, high)]
                child[key] = spans
                child[key.replace("_intervals", "_count")] = sum(b-a for a, b in spans)
        if "observed_all_participants_frame_count" in child:
            child["inferred_or_partially_unobserved_frame_count"] = int(high-low)-child["observed_all_participants_frame_count"]
        path, _ = observed_path_and_net(features.centers_cm[low:high, actor], observed[low:high])
        child["actor_observed_path_lower_bound_cm"] = float(path)
        child["boundary"] = EventBoundary(int(low), int(high), int(low), int(high), features.fps).as_dict()
        child["approach_endpoints_requalified_after_split"] = True
        output.append(child)
    return output


def requalify_source_approach_event(store, event, video_info, scale, parameters, *, scene_guard=False):
    """A source-window union must independently qualify its own endpoints."""
    from .exported_dataset_features import load_exported_features
    from .segment_export import _write_json
    fps = video_info["fps"]
    config = DEFAULT_SOCIAL_HEURISTIC_CONFIG
    context = math.ceil(max(1., config.approach_closing_window_s, config.approach_window_max_gap_s) * fps)
    offset = max(0, event["start_frame"]-context)
    stop = min(video_info["frame_count"], event["end_frame"]+context)
    rows = store.window(offset, stop)
    if len(rows) != stop-offset or [r["frame"] for r in rows] != list(range(stop-offset)):
        raise ValueError("source Approach requalification requires all original frame rows")
    if not scene_guard:
        for row in rows:
            row["detections"] = [d for d in row["detections"] if d["track_id"] in event["mouse_ids"]]
    track_count = len({d["track_id"] for row in rows for d in row["detections"]})
    with tempfile.TemporaryDirectory(prefix="approach_source_qualification_") as temporary:
        path = Path(temporary) / "tracks.json"
        _write_json(path, rows)
        features = load_exported_features(path, fps=fps, max_tracks=max(2, track_count),
            scale_config=dict(mode="fixed", cm_per_pixel=scale["cm_per_pixel"],
                              assumed_mouse_body_length_cm=scale["reference_body_cm"]))
    if not set(event["mouse_ids"]).issubset(set(map(int, features.track_ids))):
        return []
    local = deepcopy(event)
    local.update(start_frame=event["start_frame"]-offset, end_frame=event["end_frame"]-offset)
    for field, value in event.items():
        if field.endswith("_frame_intervals"):
            local[field] = [[a-offset, b-offset] for a, b in value]
    slots = {int(track): slot for slot, track in enumerate(features.track_ids)}
    actor, target = slots[event["actor_id"]], slots[event["target_id"]]
    edge = next(i for i, (a, b) in enumerate(zip(features.pair_i, features.pair_j))
                if {int(a), int(b)} == {actor, target})
    retained, scene_audit = approach_scene_support(features, actor, target) if scene_guard else (None, None)
    output = requalify_approach_event(features, edge, local, parameters, retained_mask=retained)
    for child in output:
        child.update(start_frame=child["start_frame"]+offset, end_frame=child["end_frame"]+offset,
                     parent_candidate_event_id=event.get("event_id"), approach_union_endpoints_requalified=True)
        child.pop("event_id", None)
        if scene_guard:
            child["process_boundary_status"] = dict(output_revision="approach-clear-target-v38",
                complete_scene_observations_used=True, actor_third_contact_excluded=True,
                target_third_contact_excluded=True, target_crowd_excluded=True,
                crowd_neighbor_cm=10., crowd_minimum_other_members=2,
                endpoint_duration_path_requalified=True, context_seconds=0.)
            child["approach_scene_audit"] = scene_audit
        for field, value in list(child.items()):
            if field.endswith("_frame_intervals"):
                child[field] = [[a+offset, b+offset] for a, b in value]
        child["boundary"] = EventBoundary(child["start_frame"], child["end_frame"],
            child["start_frame"], child["end_frame"], fps).as_dict()
    return output


def approach_scene_support(features, actor: int, target: int):
    """Measured scene veto for unambiguous pair-only Approach review outputs.

    The detector itself is unchanged. Contact uses the existing projected body
    separation cue; unavailable close-body geometry is not certified as clear.
    A target with two other observed neighbors within 10 cm is a crowded target.
    The actor is excluded from crowd membership, so arrival cannot veto itself.
    """
    from .social_event_geometry import observed_body_separation

    count = len(features.valid)
    actor_contact = np.zeros(count, bool)
    target_contact = np.zeros(count, bool)
    config = DEFAULT_SOCIAL_HEURISTIC_CONFIG
    for edge, (left, right) in enumerate(zip(features.pair_i, features.pair_j)):
        left, right = int(left), int(right)
        pair = {left, right}
        if actor in pair and target not in pair:
            observed = features.valid[:, left] & features.valid[:, right]
            actor_contact |= observed & ~observed_body_separation(
                features, edge, max_body_lengths=config.approach_max_footprint_body_lengths)
        elif target in pair and actor not in pair:
            observed = features.valid[:, left] & features.valid[:, right]
            target_contact |= observed & ~observed_body_separation(
                features, edge, max_body_lengths=config.approach_max_footprint_body_lengths)
    distances = np.linalg.norm(features.centers_cm-features.centers_cm[:, target, None], axis=2)
    nearby = features.valid & np.isfinite(distances) & (distances <= 10.)
    nearby[:, [actor, target]] = False
    crowd = features.valid[:, target] & (nearby.sum(axis=1) >= 2)
    retained = ~(actor_contact | target_contact | crowd)
    return retained, dict(actor_third_contact_or_unknown_frames=int(actor_contact.sum()),
        target_third_contact_or_unknown_frames=int(target_contact.sum()),
        target_crowd_frames=int(crowd.sum()), blocked_frames=int((~retained).sum()))
