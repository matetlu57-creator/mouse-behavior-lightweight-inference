"""Conservative clip-boundary screening; candidates are not new ground truth."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from mouse_behavior.evaluation.exported_dataset import (
    ExportedFeatures, HeuristicParameters, SampleRecord, predict_features,
)
from mouse_behavior.evaluation.screening_signals import (
    ScreenPolicy, body_motion, bridge, pair_phases, runs,
)


@dataclass(frozen=True)
class SegmentCandidate:
    behavior: str
    mouse_ids: tuple[int, ...]
    start_frame: int
    end_frame: int  # Half-open in the ORIGINAL clip; never padded.
    reason: str
    status: str = "rule_candidate_needs_review"

    def as_row(self) -> dict[str, Any]:
        return asdict(self)


def contiguous_windows(mask: np.ndarray, fps: float, minimum_s: float) -> list[tuple[int, int]]:
    values = np.asarray(mask, dtype=bool)
    if values.ndim != 1 or not np.isfinite(fps) or fps <= 0:
        raise ValueError("a one-dimensional mask and positive finite FPS are required")
    minimum = max(1, int(np.ceil(minimum_s * fps - 1e-9)))
    changes = np.diff(np.r_[False, values, False].astype(np.int8))
    return [(int(a), int(b)) for a, b in zip(np.flatnonzero(changes == 1),
                                            np.flatnonzero(changes == -1)) if b - a >= minimum]


def _group_geometry(features: ExportedFeatures, slots: list[int], params: HeuristicParameters):
    """Observed target members only; a missing ID is never replaced by another."""
    count = len(features.valid)
    visible = features.valid[:, slots]
    minimum = max(3, int(np.ceil(len(slots) * .6)))
    sufficient = visible.sum(axis=1) >= minimum
    connected = np.zeros(count, dtype=bool)
    nose = features.pair_nose_distance_cm
    edges = (np.isfinite(features.pair_distance_cm)
             & (features.pair_distance_cm <= params.huddle_distance_cm))
    if nose is not None:
        edges |= np.isfinite(nose) & (nose <= params.huddle_nose_distance_cm)
    internal = [k for k, (i, j) in enumerate(zip(features.pair_i, features.pair_j))
                if int(i) in slots and int(j) in slots]
    for frame in np.flatnonzero(sufficient):
        active = {slot for slot in slots if features.valid[frame, slot]}
        reached = {min(active)}
        while True:
            previous = len(reached)
            for edge in internal:
                if not edges[frame, edge]:
                    continue
                i, j = int(features.pair_i[edge]), int(features.pair_j[edge])
                if i in active and j in active and (i in reached or j in reached):
                    reached.update((i, j))
            if len(reached) == previous:
                break
        connected[frame] = active <= reached
    return connected, sufficient


def group_phase_masks(features: ExportedFeatures, slots: list[int], params: HeuristicParameters,
                      policy: ScreenPolicy | None = None, *,
                      measured_motion: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None):
    policy = policy or ScreenPolicy()
    compact, visible = _group_geometry(features, slots, params)
    formation = np.zeros(len(features.valid), dtype=bool)
    continuing = np.zeros(len(features.valid), dtype=bool)
    centers, velocity, speed = (body_motion(features, policy)
                                if measured_motion is None else measured_motion)
    still = features.valid & (speed <= policy.quiet_speed_cm_s)
    # Formation is an actual moving actor directed toward at least two other
    # stationary targets. Merely sharing a loose component is insufficient.
    for actor in slots:
        toward_count = np.zeros(len(formation), dtype=int)
        continuing_count = np.zeros(len(formation), dtype=int)
        for other in slots:
            if actor == other:
                continue
            vector = centers[:, other] - centers[:, actor]
            distance = np.linalg.norm(vector, axis=1)
            radial = np.sum(velocity[:, actor] * vector, axis=1) / np.maximum(distance, 1e-9)
            toward_count += (still[:, other] & np.isfinite(radial)
                             & (radial >= params.approach_min_closing_speed_cm_s))
            continuing_count += still[:, other] & np.isfinite(radial) & (radial > .5)
        formation |= (features.valid[:, actor]
                      & (speed[:, actor] >= params.approach_min_actor_speed_cm_s)
                      & (toward_count >= 2))
        continuing |= features.valid[:, actor] & (speed[:, actor] > policy.quiet_speed_cm_s) & (continuing_count >= 2)
    formation &= visible
    gap = max(1, round(policy.evidence_gap_s * features.fps))
    # Median displacement and bounded evidence dips retain the formation
    # process, instead of exporting isolated velocity spikes.
    formation = bridge(formation, visible & continuing, gap)
    # Observed disconnection is a real boundary, even when very brief.
    # Visibility alone cannot turn separated animals into a compact group.
    formation &= ~compact
    huddles = contiguous_windows(compact, features.fps, policy.export_event_min_s)
    if huddles:
        # The onset belongs to Huddling, never to its prelude.
        formation[huddles[0][0]:] = False
    return compact, formation


def _posture_candidates(features: ExportedFeatures, evidence: Mapping[str, Any],
                        episodes: Sequence[Mapping[str, Any]], audit: Sequence[Mapping[str, Any]]):
    result = []
    ids = list(evidence["track_ids"])
    stationary = evidence["individual_masks"]["stationary"]
    for episode in episodes:
        track = int(episode["track_id"])
        if track not in ids:
            continue
        behavior = str(episode["candidate"])
        slot = ids.index(track)
        mask = np.zeros(len(stationary), dtype=bool)
        a, b = int(episode["start_frame"]), int(episode["end_frame"])
        mask[a:b] = (features.valid[a:b, slot] if episode.get("body_static_verified")
                     else stationary[a:b, slot])
        for reviewed in audit:
            if int(reviewed["track_id"]) != track:
                continue
            # Every reviewed window supersedes the unsupervised probe there;
            # positives are emitted separately under observed_behavior.
            if reviewed.get("decision") in {"confirmed", "rejected", "uncertain"}:
                mask[int(reviewed["start_frame"]):int(reviewed["end_frame"])] = False
        for start, end in contiguous_windows(mask, features.fps, .5):
            result.append(SegmentCandidate(behavior.title(), (track,), start, end,
                                           "posture_pattern_" + str(episode.get("variant", "unspecified")) + "_inside_original_Static",
                                           "posture_candidate_needs_review"))
    for reviewed in audit:
        behavior = reviewed.get("observed_behavior")
        track = int(reviewed["track_id"])
        if reviewed.get("decision") != "confirmed" or behavior not in {"grooming", "rearing"} or track not in ids:
            continue
        a, b = int(reviewed["start_frame"]), int(reviewed["end_frame"])
        if 0 <= a < b <= len(stationary):
            result.append(SegmentCandidate(str(behavior).title(), (track,), a, b,
                                           "human_audit_observed_behavior_" + str(reviewed["review_id"]),
                                           "human_confirmed"))
    return result


def screen_record(record: SampleRecord, features: ExportedFeatures, params: HeuristicParameters,
                  *, posture_episodes: Sequence[Mapping[str, Any]] = (),
                  audit_rows: Sequence[Mapping[str, Any]] = (),
                  policy: ScreenPolicy | None = None) -> tuple[list[SegmentCandidate], str]:
    policy = policy or ScreenPolicy()
    ids = tuple(sorted(record.mouse_ids))
    lookup = {int(track): slot for slot, track in enumerate(features.track_ids)}
    if not ids or any(track not in lookup for track in ids):
        return [], "annotated_target_missing_from_selected_tracks"
    if features.track_selection_truncated:
        return [], "track_selection_truncated_requires_review"
    if len(features.valid) != record.frame_count:
        return [], "track_video_frame_count_mismatch"
    slots = [lookup[track] for track in ids]
    if record.behavior in {"Following", "Avoiding"}:
        return [], "behavior_suspended_by_current_rules"
    unchanged = {"Walking", "Running", "Group locomotion", "Dispersal", "Isolation", "Huddling"}
    if record.behavior in unchanged:
        label = "Locomotion" if record.behavior in {"Walking", "Running"} else record.behavior
        return [SegmentCandidate(label, ids, 0, record.frame_count,
                                 "unchanged_semantics_full_clip", "original_label_preserved")], "preserved"
    evidence = predict_features(features, params, temporal_only=True)
    individual = evidence["individual_masks"]
    minimum = params.pair_min_duration_s
    extras = []
    if record.behavior == "Static":
        if len(ids) != 1:
            return [], "individual_target_cardinality_invalid"
        mask = individual["stationary"][:, slots[0]].copy()
        for reviewed in audit_rows:
            if (int(reviewed["track_id"]) == ids[0] and reviewed.get("decision") == "confirmed"
                    and reviewed.get("observed_behavior") in {"grooming", "rearing", "locomotion"}):
                mask[int(reviewed["start_frame"]):int(reviewed["end_frame"])] = False
        minimum = params.stationary_min_duration_s
        extras = _posture_candidates(features, evidence,
            posture_episodes if policy.posture_export_mode == "review_candidates" else (), audit_rows)
        reason = "body_static_head_turn_allowed"
    elif record.behavior in {"Huddling", "Social clustering"}:
        if len(ids) < 3:
            return [], "group_target_cardinality_invalid"
        compact, formation = group_phase_masks(features, slots, params, policy)
        mask = compact if record.behavior == "Huddling" else formation
        minimum = params.huddle_min_duration_s if record.behavior == "Huddling" else policy.export_event_min_s
        reason = "compact_connected_targets" if record.behavior == "Huddling" else "moving_actor_to_two_static_targets_before_huddle"
    else:
        if len(ids) != 2 or ids not in evidence["pair_ids"]:
            return [], "pair_target_cardinality_invalid"
        pair = evidence["pair_ids"].index(ids)
        masks = {k: v[:, pair] for k, v in evidence["pair_masks"].items()}
        phases = pair_phases(features, pair, params, policy)
        if record.behavior == "Together":
            mask = phases["together"]
            minimum = params.together_min_duration_s
            reason = "separated_quiet_pair_or_static_touch_without_head_body_interaction"
        elif record.behavior == "Approach":
            mask = phases["approach"] & ~phases["guarded_contact"]
            reason = "approach_until_physical_surface_contact_and_phase_end"
            # Approach sources also contain the downstream event. Mine it
            # only after this exact pair's approach has actually ended.
            ends = [end for _, end in runs(mask)]
            if ends:
                after = np.arange(len(mask)) >= ends[0]
                for label, downstream, duration in (
                    ("Attack", phases["strict_attack"], policy.export_event_min_s),
                    ("Chasing", phases["chasing"], params.chase_min_duration_s),
                    ("Snout-head_contact", masks["nose_head_contact"] & ~phases["strict_attack"], params.contact_min_cumulative_seconds),
                    ("Snout-rear_contact", masks["nose_tail_contact"] & ~phases["strict_attack"], params.contact_min_cumulative_seconds),
                ):
                    for start, end in contiguous_windows(downstream & after & ~mask, features.fps, duration):
                        extras.append(SegmentCandidate(label, ids, start, end,
                            "downstream_" + label + "_after_approach_end"))
        elif record.behavior == "Attack":
            mask = phases["attack"] & ~phases["approach"]
            minimum = policy.export_event_min_s
            reason = "ongoing_attack_episode_from_action_seed_after_approach_no_quiet_lead_in"
        elif record.behavior == "Chasing":
            mask = phases["chasing"] & ~phases["approach"]
            minimum = params.chase_min_duration_s
            reason = "coherent_pursuit_tail_contact_allowed"
        elif record.behavior in {"Snout-head_contact", "Snout-rear_contact"}:
            kind = "nose_head_contact" if record.behavior == "Snout-head_contact" else "nose_tail_contact"
            # Source contact labels are not silently converted into Attack
            # because of noisy second derivatives of a nose keypoint.
            mask = masks[kind] & ~phases["approach"] & ~phases["chasing"] & ~phases["strict_attack"]
            if mask.sum() / features.fps + 1e-9 < params.contact_min_cumulative_seconds:
                return [], "pure_contact_cumulative_duration_too_short"
            # Exported cores stand alone: a short touch cannot borrow contact
            # duration from another bout that is outside the cropped video.
            minimum = params.contact_min_cumulative_seconds
            reason = "actual_contact_bout_without_approach_or_attack"
        else:
            return [], "unsupported_original_behavior"
    result = []
    for start, end in contiguous_windows(mask, features.fps, minimum):
        # A window cannot claim an annotated member never observed inside it.
        if not np.all(np.any(features.valid[start:end, slots], axis=0)):
            continue
        result.append(SegmentCandidate(record.behavior, ids, start, end, reason))
    return result + extras, "candidate_windows_found" if result or extras else "no_qualifying_core_window_needs_review"
