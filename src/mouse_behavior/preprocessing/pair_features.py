"""Candidate-pair filtering and expensive pair feature construction."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .constants import KP_NOSE, KP_TAIL
from .geometry import _angle_deg, _cosine
from ..utils.rolling import rolling_corr as _rolling_corr
from ..utils.rolling import rolling_sum as _rolling_sum
from ..utils.timer import Timer

LOGGER = logging.getLogger("mouse_behavior.lightweight_behavior_inference")


def _pair_dataframe(
    metrics: Mapping[str, Any],
    pair_index: int,
    mouse_a: int,
    mouse_b: int,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    fps: float,
    cm_per_pixel: float,
) -> pd.DataFrame:
    valid = np.asarray(metrics["valid_pair"][:, pair_index], dtype=bool)
    distance = np.asarray(metrics["distance"][:, pair_index], dtype=float)
    speed = np.asarray(metrics["speed"], dtype=float)
    nose_speed = np.asarray(metrics["nose_speed"], dtype=float)
    acceleration = np.asarray(metrics["acceleration"], dtype=float)
    angular = np.asarray(metrics["angular_speed"], dtype=float)
    pose_deformation = np.asarray(metrics["pose_deformation"], dtype=float)
    i, j = int(mouse_a), int(mouse_b)
    p = len(valid)
    direction = np.asarray(metrics["direction"][:, pair_index], dtype=float)
    pursuit_ab = np.asarray(metrics["pursuit_ab"][:, pair_index], dtype=float)
    pursuit_ba = np.asarray(metrics["pursuit_ba"][:, pair_index], dtype=float)
    escape_ab = np.asarray(metrics["escape_ab"][:, pair_index], dtype=float)
    escape_ba = np.asarray(metrics["escape_ba"][:, pair_index], dtype=float)
    behind_ab = np.asarray(metrics["behind_ab"][:, pair_index], dtype=bool)
    behind_ba = np.asarray(metrics["behind_ba"][:, pair_index], dtype=bool)
    turn = np.asarray(metrics["turn"], dtype=float)
    corr = np.asarray(metrics["trajectory_corr"][:, pair_index], dtype=float)
    drop = np.asarray(metrics["distance_drop"][:, pair_index], dtype=float)
    nose_body_ab = np.asarray(metrics["nose_body_ab"][:, pair_index], dtype=float)
    nose_body_ba = np.asarray(metrics["nose_body_ba"][:, pair_index], dtype=float)
    nose_tail_ab = np.asarray(metrics["nose_tail_ab"][:, pair_index], dtype=float)
    nose_tail_ba = np.asarray(metrics["nose_tail_ba"][:, pair_index], dtype=float)
    nose_head_ab = np.asarray(metrics["nose_head_ab"][:, pair_index], dtype=float)
    nose_head_ba = np.asarray(metrics["nose_head_ba"][:, pair_index], dtype=float)
    behavior_speed = np.asarray(metrics["behavior_speed"], dtype=float)
    distance_body_lengths = distance / 8.0
    score_ab = pursuit_ab + escape_ab + 0.15 * speed[:, i]
    score_ba = pursuit_ba + escape_ba + 0.15 * speed[:, j]
    tie = np.abs(score_ab - score_ba) <= 0.05
    selected_ab = score_ab >= score_ba
    selected_actor = np.where(tie, -1, np.where(selected_ab, i, j))
    selected_target = np.where(tie, -1, np.where(selected_ab, j, i))
    selected_nose_body = np.where(selected_ab, nose_body_ab, nose_body_ba)
    selected_turn = np.where(selected_ab, turn[:, j], turn[:, i])
    closing_speed = drop / max(0.30, 1.0 / fps)
    frame = np.arange(p, dtype=int)
    data: dict[str, Any] = {
        "frame": frame,
        "time_s": frame / fps,
        "pair_key": f"{i}_{j}",
        "mouse_a_id": i,
        "mouse_b_id": j,
        "mouse_a_raw_track_id": i,
        "mouse_b_raw_track_id": j,
        "mouse_a_track_state": "tracked",
        "mouse_b_track_state": "tracked",
        "valid_pair": valid,
        "center_distance_cm": distance,
        "center_distance_body_lengths": distance_body_lengths,
        "head_distance_cm": np.asarray(metrics["head_distance"][:, pair_index], dtype=float),
        "mouse_a_speed_cm_s": speed[:, i],
        "mouse_b_speed_cm_s": speed[:, j],
        "mouse_a_behavior_speed_cm_s": behavior_speed[:, i],
        "mouse_b_behavior_speed_cm_s": behavior_speed[:, j],
        "pose_pair_quality": np.asarray(metrics["pose_pair_quality"][:, pair_index], dtype=float),
        "identity_pair_quality": valid.astype(float),
        "pair_wall_jump_excluded": np.zeros(p, dtype=bool),
        "cluster_attack_hint": np.zeros(p, dtype=bool),
        "cluster_overlap_iou": np.zeros(p, dtype=float),
        "cluster_motion_bl_per_frame": np.zeros(p, dtype=float),
        "cluster_active_frames": np.zeros(p, dtype=int),
        "selected_actor_id": selected_actor,
        "selected_target_id": selected_target,
        "selected_nose_body_distance_cm": selected_nose_body,
        "selected_target_turn_angle_deg": selected_turn,
        "selected_distance_drop_cm": drop,
        "selected_closing_speed_cm_s": closing_speed,
        "selected_actor_speed_cm_s": np.where(selected_ab, speed[:, i], speed[:, j]),
        "selected_target_speed_cm_s": np.where(selected_ab, speed[:, j], speed[:, i]),
        "selected_actor_behavior_speed_cm_s": np.where(
            selected_ab,
            behavior_speed[:, i],
            behavior_speed[:, j],
        ),
        "selected_target_behavior_speed_cm_s": np.where(
            selected_ab,
            behavior_speed[:, j],
            behavior_speed[:, i],
        ),
        "selected_target_escape_alignment": np.where(selected_ab, escape_ab, escape_ba),
        "selected_actor_pursuit_alignment": np.where(selected_ab, pursuit_ab, pursuit_ba),
        "selected_nose_tail_distance_cm": np.where(selected_ab, nose_tail_ab, nose_tail_ba),
        "selected_nose_head_distance_cm": np.where(selected_ab, nose_head_ab, nose_head_ba),
        "selected_weak_chase_score": np.zeros(p, dtype=float),
        "selected_strong_chase_score": np.zeros(p, dtype=float),
        "selected_weak_attack_evidence": np.zeros(p, dtype=float),
        "selected_strong_attack_evidence": np.zeros(p, dtype=float),
        "weak_contact": (np.minimum(nose_body_ab, nose_body_ba) <= 4.0),
        "strong_contact": (np.minimum(nose_body_ab, nose_body_ba) <= 3.0),
        "weak_potential_attack": (np.minimum(nose_body_ab, nose_body_ba) <= 4.0),
        "strong_potential_attack": (np.minimum(nose_body_ab, nose_body_ba) <= 3.0),
        "weak_raw_chase": np.zeros(p, dtype=bool),
        "strong_raw_chase": np.zeros(p, dtype=bool),
        "weak_raw_attack": np.zeros(p, dtype=bool),
        "strong_raw_attack": np.zeros(p, dtype=bool),
        "scale_mode": "body_length",
        "cm_per_pixel": cm_per_pixel,
    }
    for level in ("weak", "strong"):
        for provider in (
            "strict_chase",
            "window_chase",
            "near_recovery_chase",
            "close_follow_chase",
            "strict_attack",
            "impulse_attack",
            "grapple_attack",
            "occlusion_overlap_attack",
        ):
            data[f"{level}_{provider}"] = np.zeros(p, dtype=bool)

    direction_columns = {
        "a_to_b": (
            i,
            j,
            pursuit_ab,
            escape_ab,
            behind_ab,
            turn[:, j],
            nose_body_ab,
            nose_tail_ab,
            nose_head_ab,
        ),
        "b_to_a": (
            j,
            i,
            pursuit_ba,
            escape_ba,
            behind_ba,
            turn[:, i],
            nose_body_ba,
            nose_tail_ba,
            nose_head_ba,
        ),
    }
    for prefix, (
        actor,
        target,
        pursuit,
        escape,
        behind,
        target_turn,
        nose_body,
        nose_tail,
        nose_head,
    ) in direction_columns.items():
        actor_speed = speed[:, actor]
        target_speed = speed[:, target]
        actor_nose_speed = nose_speed[:, actor]
        target_nose_speed = nose_speed[:, target]
        actor_acceleration = acceleration[:, actor]
        target_acceleration = acceleration[:, target]
        actor_angular = angular[:, actor]
        target_angular = angular[:, target]
        data.update(
            {
                f"{prefix}_actor_speed_cm_s": actor_speed,
                f"{prefix}_target_speed_cm_s": target_speed,
                f"{prefix}_actor_behavior_speed_cm_s": behavior_speed[:, actor],
                f"{prefix}_target_behavior_speed_cm_s": behavior_speed[:, target],
                f"{prefix}_actor_acceleration_cm_s2": actor_acceleration,
                f"{prefix}_target_acceleration_cm_s2": target_acceleration,
                f"{prefix}_actor_nose_speed_cm_s": actor_nose_speed,
                f"{prefix}_target_nose_speed_cm_s": target_nose_speed,
                f"{prefix}_actor_head_relative_speed_cm_s": np.maximum(
                    actor_nose_speed - actor_speed, 0.0
                ),
                f"{prefix}_actor_angular_speed_deg_s": actor_angular,
                f"{prefix}_target_angular_speed_deg_s": target_angular,
                f"{prefix}_direction_similarity": direction,
                f"{prefix}_pursuit_alignment": pursuit,
                f"{prefix}_target_escape_alignment": escape,
                f"{prefix}_trajectory_correlation": corr,
                f"{prefix}_closing_speed_cm_s": closing_speed,
                f"{prefix}_center_distance_body_lengths": distance_body_lengths,
                f"{prefix}_actor_behind_target": behind,
                f"{prefix}_behind_score": behind.astype(float),
                f"{prefix}_target_turn_angle_deg": target_turn,
                f"{prefix}_nose_body_distance_cm": nose_body,
                f"{prefix}_nose_tail_distance_cm": nose_tail,
                f"{prefix}_nose_head_distance_cm": nose_head,
                f"{prefix}_actor_pose_deformation_energy": pose_deformation[:, actor],
                f"{prefix}_target_pose_deformation_energy": pose_deformation[:, target],
            }
        )
    return pd.DataFrame(data)


def _pair_metrics(
    kin: Mapping[str, Any],
    fps: float,
    pair_indices: np.ndarray | Sequence[int] | None = None,
    frame_mask: np.ndarray | None = None,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Build pair-wise features for the requested logical pair columns.

    ``pair_indices`` refers to the stable ordering returned by
    ``np.triu_indices(mice, k=1)``.  Keeping this optional preserves the old
    all-pair API for callers that need it, while the lightweight analyzer can
    now run the expensive nose/trajectory features only for spatially useful
    pairs selected by its prefilter.  ``frame_mask`` optionally limits those
    expensive pair features to padded interaction windows while retaining the
    selected pair's full output timeline for the temporal behavior engine.
    """
    centers = np.asarray(kin["centers_cm"], dtype=float)
    heads = np.asarray(kin["head_cm"], dtype=float)
    kp = np.asarray(kin["keypoints_cm"], dtype=float)
    heading = np.asarray(kin["heading"], dtype=float)
    velocity = np.asarray(kin["velocity"], dtype=float)
    valid = np.asarray(kin["valid"], dtype=bool)
    speed = np.asarray(kin["speed"], dtype=float)
    frames, mice = valid.shape
    all_pair_i, all_pair_j = np.triu_indices(mice, k=1)
    if pair_indices is None:
        selected_pair_indices = np.arange(len(all_pair_i), dtype=int)
    else:
        selected_pair_indices = np.asarray(pair_indices, dtype=int).reshape(-1)
        if np.any(selected_pair_indices < 0) or np.any(selected_pair_indices >= len(all_pair_i)):
            raise IndexError("pair_indices contains an invalid all-pair column index")
    pair_i = all_pair_i[selected_pair_indices]
    pair_j = all_pair_j[selected_pair_indices]
    pairs = len(pair_i)
    raw_valid_pair = valid[:, pair_i] & valid[:, pair_j]
    if frame_mask is None:
        frame_mask = np.ones((frames, pairs), dtype=bool)
    else:
        frame_mask = np.asarray(frame_mask, dtype=bool)
        if frame_mask.shape != (frames, pairs):
            raise ValueError(
                f"frame_mask must have shape ({frames}, {pairs}), got {frame_mask.shape}"
            )
    valid_pair = raw_valid_pair & frame_mask
    delta = centers[:, pair_j] - centers[:, pair_i]
    distance = np.linalg.norm(delta, axis=2)
    head_distance = np.linalg.norm(heads[:, pair_j] - heads[:, pair_i], axis=2)
    direction = _cosine(velocity[:, pair_i], velocity[:, pair_j])
    pursuit_ab = _cosine(velocity[:, pair_i], delta)
    pursuit_ba = _cosine(velocity[:, pair_j], -delta)
    escape_ab = _cosine(velocity[:, pair_j], delta)
    escape_ba = _cosine(velocity[:, pair_i], -delta)
    behind_ab = np.sum((centers[:, pair_i] - centers[:, pair_j]) * heading[:, pair_j], axis=2) < 0.0
    behind_ba = np.sum((centers[:, pair_j] - centers[:, pair_i]) * heading[:, pair_i], axis=2) < 0.0
    distance_for_lookback = distance.copy()
    distance_for_lookback[~raw_valid_pair] = np.nan
    for array in (distance, head_distance, direction, pursuit_ab, pursuit_ba, escape_ab, escape_ba):
        array[~valid_pair] = np.nan
    behind_ab[~valid_pair] = False
    behind_ba[~valid_pair] = False

    nose_body_ab = np.full((frames, pairs), np.nan, dtype=np.float32)
    nose_body_ba = np.full((frames, pairs), np.nan, dtype=np.float32)
    nose_tail_ab = np.full((frames, pairs), np.nan, dtype=np.float32)
    nose_tail_ba = np.full((frames, pairs), np.nan, dtype=np.float32)
    nose_head_ab = np.full((frames, pairs), np.nan, dtype=np.float32)
    nose_head_ba = np.full((frames, pairs), np.nan, dtype=np.float32)
    for start in range(0, frames, 500):
        end = min(start + 500, frames)
        active_rows, active_pair_columns = np.nonzero(valid_pair[start:end])
        if not len(active_rows):
            continue
        frame_indices = start + active_rows
        pair_columns = active_pair_columns.astype(int, copy=False)
        body_b = kp[frame_indices, pair_j[pair_columns]]
        body_a = kp[frame_indices, pair_i[pair_columns]]
        nose_a = kp[frame_indices, pair_i[pair_columns], KP_NOSE]
        nose_b = kp[frame_indices, pair_j[pair_columns], KP_NOSE]
        distances_ab = np.linalg.norm(body_b - nose_a[:, None, :], axis=2)
        distances_ba = np.linalg.norm(body_a - nose_b[:, None, :], axis=2)
        nose_body_ab[frame_indices, pair_columns] = np.nanmin(distances_ab, axis=1)
        nose_body_ba[frame_indices, pair_columns] = np.nanmin(distances_ba, axis=1)
        nose_tail_ab[frame_indices, pair_columns] = np.linalg.norm(
            nose_a - kp[frame_indices, pair_j[pair_columns], KP_TAIL],
            axis=1,
        )
        nose_tail_ba[frame_indices, pair_columns] = np.linalg.norm(
            nose_b - kp[frame_indices, pair_i[pair_columns], KP_TAIL],
            axis=1,
        )
        nose_head_ab[frame_indices, pair_columns] = np.linalg.norm(
            nose_a - heads[frame_indices, pair_j[pair_columns]],
            axis=1,
        )
        nose_head_ba[frame_indices, pair_columns] = np.linalg.norm(
            nose_b - heads[frame_indices, pair_i[pair_columns]],
            axis=1,
        )
    nose_body_ab[~valid_pair] = np.nan
    nose_body_ba[~valid_pair] = np.nan
    nose_tail_ab[~valid_pair] = np.nan
    nose_tail_ba[~valid_pair] = np.nan
    nose_head_ab[~valid_pair] = np.nan
    nose_head_ba[~valid_pair] = np.nan

    lookback = max(int(round(fps * 0.30)), 1)
    distance_drop = np.zeros((frames, pairs), dtype=np.float32)
    if lookback < frames:
        distance_drop[lookback:] = (
            distance_for_lookback[:-lookback] - distance_for_lookback[lookback:]
        )
    distance_drop[~valid_pair] = 0.0
    distance_drop_seconds = float(lookback / max(fps, 1e-9))
    turn = np.zeros((frames, mice), dtype=np.float32)
    if lookback < frames:
        turn[lookback:] = _angle_deg(heading[lookback:], heading[:-lookback])
    turn[~valid] = 0.0
    velocity_a = velocity[:, pair_i]
    velocity_b = velocity[:, pair_j]
    trajectory_valid = (
        valid_pair
        & np.all(np.isfinite(velocity_a), axis=2)
        & np.all(np.isfinite(velocity_b), axis=2)
    )
    trajectory_corr = _rolling_corr(
        velocity_a,
        velocity_b,
        trajectory_valid,
        max(int(round(fps * 2.5)), 4),
        active_mask=frame_mask,
    )
    path_a = (
        _rolling_sum(
            speed[:, pair_i],
            max(int(round(fps * 2.5)), 4),
            active_mask=frame_mask,
        )
        / fps
    )
    path_b = (
        _rolling_sum(
            speed[:, pair_j],
            max(int(round(fps * 2.5)), 4),
            active_mask=frame_mask,
        )
        / fps
    )
    contact = np.minimum(nose_body_ab, nose_body_ba) <= 4.0
    contact &= valid_pair
    repeated_contact = _rolling_sum(
        contact.astype(float),
        max(int(round(fps * 2.0)), 1),
        active_mask=frame_mask,
    ).astype(np.int16)
    pose_pair_quality = np.sqrt(
        np.asarray(kin["pose_quality"][:, pair_i], dtype=float)
        * np.asarray(kin["pose_quality"][:, pair_j], dtype=float)
    )
    pose_pair_quality[~valid_pair] = 0.0
    metrics = {
        "valid_pair": valid_pair,
        "distance": distance,
        "head_distance": head_distance,
        "direction": direction,
        "pursuit_ab": pursuit_ab,
        "pursuit_ba": pursuit_ba,
        "escape_ab": escape_ab,
        "escape_ba": escape_ba,
        "behind_ab": behind_ab,
        "behind_ba": behind_ba,
        "nose_body_ab": nose_body_ab,
        "nose_body_ba": nose_body_ba,
        "nose_tail_ab": nose_tail_ab,
        "nose_tail_ba": nose_tail_ba,
        "nose_head_ab": nose_head_ab,
        "nose_head_ba": nose_head_ba,
        "distance_drop": distance_drop,
        "distance_drop_seconds": distance_drop_seconds,
        "turn": turn,
        "trajectory_corr": trajectory_corr,
        "path_a": path_a,
        "path_b": path_b,
        "repeated_contact": repeated_contact,
        "pose_pair_quality": pose_pair_quality,
        "speed": np.asarray(kin["speed"], dtype=float),
        "behavior_speed": np.asarray(kin["behavior_speed"], dtype=float),
        "nose_speed": np.asarray(kin["nose_speed"], dtype=float),
        "acceleration": np.asarray(kin["acceleration"], dtype=float),
        "angular_speed": np.asarray(kin["angular_speed"], dtype=float),
        "pose_deformation": np.asarray(kin["pose_deformation"], dtype=float),
    }
    return metrics, pair_i, pair_j


def _boolean_runs_with_gap(values: np.ndarray, max_gap: int = 0) -> list[tuple[int, int]]:
    """Group true runs while filling short false gaps."""
    values = np.asarray(values, dtype=bool).copy()
    max_gap = max(int(max_gap), 0)
    if max_gap and values.size:
        false_runs = _boolean_runs(~values)
        for start, end in false_runs:
            if start == 0 or end == len(values) - 1:
                continue
            if end - start + 1 <= max_gap:
                values[start : end + 1] = True
    return _boolean_runs(values)


def _boolean_runs(values: np.ndarray) -> list[tuple[int, int]]:
    values = np.asarray(values, dtype=bool)
    if values.size == 0:
        return []
    starts = np.flatnonzero(values & np.r_[True, ~values[:-1]])
    ends = np.flatnonzero(values & np.r_[~values[1:], True])
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def _interaction_radius(config: Mapping[str, Any]) -> float:
    engine_cfg = dict(config.get("standard_behavior_engine", {}))
    interaction_cfg = dict(engine_cfg.get("interaction_graph", {}))
    if "radius_cm" in interaction_cfg:
        return float(interaction_cfg["radius_cm"])
    chase_cfg = config.get("chase", {})
    attack_cfg = config.get("attack", {})
    weak_chase = dict(chase_cfg.get("weak", {}))
    weak_attack = dict(attack_cfg.get("weak", {}))
    return max(
        float(weak_chase.get("max_distance_cm", 12.0)),
        float(weak_attack.get("body_center_contact_distance_cm", 6.0)),
    ) + float(interaction_cfg.get("buffer_cm", 5.0))


def _pair_prefilter(
    kin: Mapping[str, Any],
    config: Mapping[str, Any],
    interaction_radius: float,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Find spatially and directionally meaningful pair columns cheaply.

    The full pair feature builder contains the expensive nose-to-body
    geometry and rolling trajectory calculations.  This pass intentionally
    uses only center distance and body heading, so it can decide which stable
    logical pair columns deserve those features.

    A close-distance fallback is important: heading estimates can be noisy
    during contact or occlusion, while a very close pair is still relevant to
    nose/head/tail contact and attack detection.  For wider interactions at
    least one mouse must face the other within the configured cosine gate.
    The returned ``valuable_frame`` is frame-level diagnostic information; the
    analyzer promotes a pair if it is valuable at any analyzed frame and then
    keeps that pair's full time series for temporal FSM context.
    """
    lightweight_cfg = dict(config.get("lightweight_behavior_inference", {}))
    configured = dict(lightweight_cfg.get("pair_prefilter", {}))
    enabled = bool(configured.get("enabled", True))
    radius = max(float(interaction_radius), 0.0)
    close_distance = min(
        max(float(configured.get("close_distance_cm", 10.0)), 0.0),
        radius,
    )
    min_heading_cosine = float(np.clip(float(configured.get("min_heading_cosine", 0.0)), -1.0, 1.0))

    centers = np.asarray(kin["centers_cm"], dtype=float)
    heading = np.asarray(kin["heading"], dtype=float)
    valid = np.asarray(kin["valid"], dtype=bool)
    frames, mice = valid.shape
    pair_i, pair_j = np.triu_indices(mice, k=1)
    valid_pair = valid[:, pair_i] & valid[:, pair_j]
    delta = centers[:, pair_j] - centers[:, pair_i]
    distance = np.linalg.norm(delta, axis=2)
    within_radius = valid_pair & np.isfinite(distance) & (distance <= radius)

    heading_to_partner_a = _cosine(heading[:, pair_i], delta)
    heading_to_partner_b = _cosine(heading[:, pair_j], -delta)
    heading_values_a = heading[:, pair_i]
    heading_values_b = heading[:, pair_j]
    heading_valid_a = np.all(np.isfinite(heading_values_a), axis=2) & (
        np.linalg.norm(heading_values_a, axis=2) > 1e-9
    )
    heading_valid_b = np.all(np.isfinite(heading_values_b), axis=2) & (
        np.linalg.norm(heading_values_b, axis=2) > 1e-9
    )
    heading_relevant = (heading_valid_a & (heading_to_partner_a >= min_heading_cosine)) | (
        heading_valid_b & (heading_to_partner_b >= min_heading_cosine)
    )
    close_fallback = distance <= close_distance
    valuable_frame = within_radius & (close_fallback | heading_relevant)

    if enabled:
        candidate_pair_mask = np.any(valuable_frame, axis=0)
    else:
        # Compatibility switch: preserve the previous distance-only candidate
        # rule when the caller explicitly disables the orientation prefilter.
        valuable_frame = within_radius
        candidate_pair_mask = np.any(within_radius, axis=0)

    return (
        {
            "valid_pair": valid_pair,
            "distance": distance,
            "heading_to_partner_a": heading_to_partner_a,
            "heading_to_partner_b": heading_to_partner_b,
            "valuable_frame": valuable_frame,
            "candidate_pair_mask": candidate_pair_mask,
            "enabled": enabled,
            "close_distance_cm": close_distance,
            "min_heading_cosine": min_heading_cosine,
            "frames": int(frames),
        },
        pair_i,
        pair_j,
    )


def _pair_window_mask(
    valuable_frame: np.ndarray,
    fps: float,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Expand valuable pair frames into padded windows for expensive metrics.

    The window is expanded on both sides so causal rolling features and FSM
    context are still available when a pair first becomes relevant.  Short
    gaps inside a valuable run are filled before expansion, which prevents a
    noisy heading estimate from fragmenting one interaction into many small
    windows.
    """
    valuable_frame = np.asarray(valuable_frame, dtype=bool)
    if valuable_frame.ndim != 2:
        raise ValueError(f"valuable_frame must be a 2D array, got shape={valuable_frame.shape}")
    lightweight_cfg = dict(config.get("lightweight_behavior_inference", {}))
    prefilter_cfg = dict(lightweight_cfg.get("pair_prefilter", {}))
    window_cfg = dict(prefilter_cfg.get("window", {}))
    enabled = bool(window_cfg.get("enabled", True))
    padding_seconds = max(float(window_cfg.get("padding_seconds", 2.5)), 0.0)
    fill_gap_seconds = max(float(window_cfg.get("fill_gap_seconds", 0.15)), 0.0)
    padding_frames = max(int(math.ceil(padding_seconds * max(float(fps), 0.0))), 0)
    fill_gap_frames = max(int(math.ceil(fill_gap_seconds * max(float(fps), 0.0))), 0)

    if not enabled:
        return np.ones_like(valuable_frame, dtype=bool), {
            "enabled": False,
            "padding_seconds": padding_seconds,
            "fill_gap_seconds": fill_gap_seconds,
            "padding_frames": padding_frames,
            "fill_gap_frames": fill_gap_frames,
            "active_frame_count": int(valuable_frame.size),
            "active_frame_fraction": 1.0 if valuable_frame.size else 0.0,
        }

    frames, pairs = valuable_frame.shape
    window_mask = np.zeros_like(valuable_frame, dtype=bool)
    for pair_index in range(pairs):
        runs = _boolean_runs_with_gap(valuable_frame[:, pair_index], fill_gap_frames)
        for start, end in runs:
            window_start = max(start - padding_frames, 0)
            window_end = min(end + padding_frames, frames - 1)
            if window_start <= window_end:
                window_mask[window_start : window_end + 1, pair_index] = True

    return window_mask, {
        "enabled": True,
        "padding_seconds": padding_seconds,
        "fill_gap_seconds": fill_gap_seconds,
        "padding_frames": padding_frames,
        "fill_gap_frames": fill_gap_frames,
        "active_frame_count": int(window_mask.sum()),
        "active_frame_fraction": float(window_mask.mean()) if window_mask.size else 0.0,
    }


@dataclass(frozen=True)
class _PairWorkset:
    """All-pair prefilter state and candidate-only metric arrays for one run."""

    interaction_radius: float
    prefilter: Mapping[str, Any]
    all_pair_i: np.ndarray
    all_pair_j: np.ndarray
    candidate_pair_indices: tuple[int, ...]
    candidate_metric_index: Mapping[int, int]
    candidate_frame_mask: np.ndarray
    pair_window_stats: Mapping[str, Any]
    metrics: Mapping[str, Any]
    pair_i: np.ndarray
    pair_j: np.ndarray


def _prepare_pair_workset(
    kin: Mapping[str, Any],
    fps: float,
    config: Mapping[str, Any],
    *,
    stage_timings: dict[str, float] | None = None,
) -> _PairWorkset:
    """Prepare vectorized all-pair gates and candidate-only heavy metrics."""

    pair_filter_timer = Timer(
        "pair_filter_and_windows",
        logger=LOGGER,
        sink=stage_timings,
    ).start()
    interaction_radius = _interaction_radius(config)
    prefilter, all_pair_i, all_pair_j = _pair_prefilter(
        kin,
        config,
        interaction_radius,
    )
    candidate_pair_indices = tuple(
        int(index)
        for index in np.flatnonzero(np.asarray(prefilter["candidate_pair_mask"], dtype=bool))
    )
    candidate_pair_indices_array = np.asarray(candidate_pair_indices, dtype=int)
    pair_window_mask, pair_window_stats = _pair_window_mask(
        np.asarray(prefilter["valuable_frame"], dtype=bool),
        fps,
        config,
    )
    candidate_frame_mask = pair_window_mask[:, candidate_pair_indices_array]
    pair_filter_timer.stop()

    pair_metrics_timer = Timer(
        "pair_metrics",
        logger=LOGGER,
        sink=stage_timings,
    ).start()
    metrics, pair_i, pair_j = _pair_metrics(
        kin,
        fps,
        pair_indices=candidate_pair_indices_array,
        frame_mask=candidate_frame_mask,
    )
    pair_metrics_timer.stop()
    candidate_metric_index = {
        int(original_index): int(metric_index)
        for metric_index, original_index in enumerate(candidate_pair_indices)
    }
    LOGGER.info(
        "[pair filter] %d/%d pairs retained (distance <= %.2f cm, close fallback <= %.2f cm, heading cosine >= %.2f)",
        len(candidate_pair_indices),
        len(all_pair_i),
        interaction_radius,
        float(prefilter["close_distance_cm"]),
        float(prefilter["min_heading_cosine"]),
    )
    LOGGER.info(
        "[pair windows] active %.1f%% of pair-frame slots (padding %.2fs, fill gap %.2fs)",
        100.0 * float(pair_window_stats["active_frame_fraction"]),
        float(pair_window_stats["padding_seconds"]),
        float(pair_window_stats["fill_gap_seconds"]),
    )
    return _PairWorkset(
        interaction_radius=float(interaction_radius),
        prefilter=prefilter,
        all_pair_i=all_pair_i,
        all_pair_j=all_pair_j,
        candidate_pair_indices=candidate_pair_indices,
        candidate_metric_index=candidate_metric_index,
        candidate_frame_mask=candidate_frame_mask,
        pair_window_stats=pair_window_stats,
        metrics=metrics,
        pair_i=pair_i,
        pair_j=pair_j,
    )
