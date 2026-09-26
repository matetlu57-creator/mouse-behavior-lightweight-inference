"""Relative social-behavior evidence and temporal hysteresis.

The Beiyi examples are short clips and their physical pixel-to-centimetre
scale is not an acceptance target for this iteration.  This module therefore
keeps the social rules relational where possible:

* approach uses a relative fall in pair distance and a directional speed gap;
* chase uses pursuit, escape, and common-direction evidence;
* avoidance uses the opposite role (the evader is the event actor), a prior
  interaction context, distance expansion, and repeated target turns;
* attack uses contact geometry plus independent dynamic evidence and is never
  accepted from a single isolated frame; the Beiyi profile can use raw
  bounding-box motion as the primary dynamic/contact channel when pose points
  are occluded by riding or wrestling.

The functions are pure with respect to their inputs.  They do not inspect
video names or label-folder names; those remain validation-only metadata.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ..utils.rolling import rolling_sum


def _numeric_column(frame: pd.DataFrame, column: str, default: float = 0.0) -> np.ndarray:
    values = frame[column] if column in frame else pd.Series(default, index=frame.index)
    return pd.to_numeric(values, errors="coerce").fillna(default).to_numpy(float)


def _rolling_any(mask: np.ndarray, window: int) -> np.ndarray:
    """Return a causal hold mask without importing pandas rolling state."""

    values = np.asarray(mask, dtype=bool).reshape(-1)
    if not values.size:
        return values.copy()
    return rolling_sum(values.astype(float), max(int(window), 1)) > 0.0


def _rolling_future_any(mask: np.ndarray, window: int) -> np.ndarray:
    """Return whether evidence occurs in the current/future causal window."""

    values = np.asarray(mask, dtype=bool).reshape(-1)
    if not values.size:
        return values.copy()
    return _rolling_any(values[::-1], window)[::-1]


def _rolling_fraction(mask: np.ndarray, window: int) -> np.ndarray:
    """Return causal positive support fraction for a boolean signal."""

    values = np.asarray(mask, dtype=bool).reshape(-1)
    if not values.size:
        return np.zeros(0, dtype=float)
    radius = max(int(window), 1)
    totals = rolling_sum(values.astype(float), radius)
    counts = rolling_sum(np.ones(values.shape, dtype=float), radius)
    return np.divide(totals, counts, out=np.zeros_like(totals), where=counts > 0.0)


def _rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    """Smooth detector jitter without filling missing samples with motion."""

    array = np.asarray(values, dtype=float).reshape(-1)
    if not array.size or int(window) <= 1:
        return array.copy()
    return (
        pd.Series(array)
        .rolling(max(int(window), 1), center=True, min_periods=1)
        .median()
        .to_numpy(float)
    )


def _symmetric_hold(
    mask: np.ndarray,
    window: int,
    *,
    min_fraction: float = 0.5,
) -> np.ndarray:
    """Hold a state only when a centered window has enough real evidence.

    The old implementation used an ``any`` dilation.  A single noisy frame
    could therefore expand into an entire short clip.  Social states now use
    a support fraction so the temporal hysteresis repairs detector gaps while
    still rejecting isolated direction spikes.
    """

    values = np.asarray(mask, dtype=bool).reshape(-1)
    if not values.size:
        return values.copy()
    radius = max(int(window), 1)
    backward = rolling_sum(values.astype(float), radius)
    forward = rolling_sum(values[::-1].astype(float), radius)[::-1]
    totals = backward + forward - values.astype(float)
    index = np.arange(len(values))
    left_count = np.minimum(index + 1, radius)
    right_count = np.minimum(len(values) - index, radius)
    counts = left_count + right_count - 1
    fraction = np.divide(
        totals,
        counts,
        out=np.zeros_like(totals),
        where=counts > 0,
    )
    threshold = float(np.clip(min_fraction, 0.0, 1.0))
    return fraction >= threshold


def _fill_supported_internal_gaps(
    mask: np.ndarray,
    support: np.ndarray,
    max_gap_frames: int,
    *,
    min_support_fraction: float,
) -> np.ndarray:
    """Fill bounded internal gaps only when an independent bridge supports them."""

    result = np.asarray(mask, dtype=bool).reshape(-1).copy()
    bridge = np.asarray(support, dtype=bool).reshape(-1)
    if not result.size or result.shape != bridge.shape or max_gap_frames <= 0:
        return result
    false_starts, false_ends = _boolean_spans(~result)
    required = float(np.clip(min_support_fraction, 0.0, 1.0))
    for start, end in zip(false_starts, false_ends):
        left, right = int(start), int(end)
        if left == 0 or right == len(result) - 1:
            continue
        if right - left + 1 > int(max_gap_frames):
            continue
        if float(np.mean(bridge[left : right + 1])) < required:
            continue
        result[left : right + 1] = True
    return result


def _retain_bouts_with_minimum_support(
    mask: np.ndarray,
    support: np.ndarray,
    *,
    minimum_fraction: float,
) -> np.ndarray:
    """Reject a temporally expanded bout that contains too little real evidence.

    Hysteresis and short box prediction may bridge a genuine detector gap, but
    they must not turn a few visible frames into a long semantic event.  The
    check is applied independently to every final bout so a well-observed
    event elsewhere in the same pair cannot rescue a mostly imputed one.
    """

    result = np.asarray(mask, dtype=bool).reshape(-1).copy()
    evidence = np.asarray(support, dtype=bool).reshape(-1)
    if result.shape != evidence.shape:
        raise ValueError("mask and support must have identical one-dimensional shapes")
    required = float(np.clip(minimum_fraction, 0.0, 1.0))
    if not result.size or required <= 0.0:
        return result
    starts, ends = _boolean_spans(result)
    for start, end in zip(starts, ends):
        left, right = int(start), int(end) + 1
        if float(np.mean(evidence[left:right])) < required:
            result[left:right] = False
    return result


def _extend_bouts_through_supported_trailing_gap(
    mask: np.ndarray,
    support: np.ndarray,
    *,
    max_gap_frames: int,
    min_seed_frames: int,
) -> np.ndarray:
    """Extend a confirmed bout through one bounded trailing occlusion.

    Internal gaps are handled separately by ``_fill_supported_internal_gaps``.
    This helper covers the common short-clip case where the target disappears
    after a real pursuit seed and no second pose-valid sample exists to close
    the gap.  Extension stops at the first unsupported frame and cannot start
    from a one-frame/noisy seed.
    """

    result = np.asarray(mask, dtype=bool).reshape(-1).copy()
    bridge = np.asarray(support, dtype=bool).reshape(-1)
    if result.shape != bridge.shape:
        raise ValueError("mask and support must have identical one-dimensional shapes")
    maximum = max(int(max_gap_frames), 0)
    minimum_seed = max(int(min_seed_frames), 1)
    if not result.size or maximum <= 0:
        return result
    starts, ends = _boolean_spans(result)
    # Iterate backwards so extending an earlier bout cannot become the seed
    # for a later, originally independent bout.
    for start, end in reversed(list(zip(starts, ends))):
        left, right = int(start), int(end)
        if right - left + 1 < minimum_seed:
            continue
        extension_end = min(right + maximum, len(result) - 1)
        cursor = right + 1
        while cursor <= extension_end and bool(bridge[cursor]):
            result[cursor] = True
            cursor += 1
    return result


def _recover_avoidance_after_initiator_occlusion(
    pair_df: pd.DataFrame,
    valid: np.ndarray,
    first_direction: Mapping[str, np.ndarray],
    second_direction: Mapping[str, np.ndarray],
    *,
    fps: float,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Recover evasion when the approaching mouse becomes occluded.

    The recovery is deliberately causal and asymmetric: a jointly observed,
    close, directional approach must precede the dropout; the still-visible
    target must make an abrupt turn and move substantially farther from the
    initiator's last observed position.  A missing box by itself is never
    avoidance evidence.
    """

    n = len(pair_df)
    empty = np.zeros(n, dtype=bool)
    if n == 0 or not bool(config.get("occlusion_recovery_enabled", False)):
        return empty.copy(), empty.copy()

    analysis_fps = max(float(fps), 1e-9)
    lookback_frames = max(
        int(round(float(config.get("occlusion_recovery_lookback_seconds", 0.75)) * analysis_fps)),
        1,
    )
    confirmation_frames = max(
        int(
            round(float(config.get("occlusion_recovery_confirmation_seconds", 0.75)) * analysis_fps)
        ),
        1,
    )
    state_frames = max(
        int(round(float(config.get("occlusion_recovery_state_seconds", 3.0)) * analysis_fps)),
        1,
    )
    pre_context_frames = max(
        int(
            round(float(config.get("occlusion_recovery_pre_context_seconds", 0.50)) * analysis_fps)
        ),
        0,
    )
    min_joint_frames = max(int(config.get("occlusion_recovery_min_joint_frames", 4)), 1)
    min_pursuit = float(config.get("occlusion_recovery_min_pursuit_alignment", 0.35))
    min_pursuit_fraction = float(
        np.clip(config.get("occlusion_recovery_min_pursuit_fraction", 0.30), 0.0, 1.0)
    )
    min_turn = float(config.get("occlusion_recovery_min_turn_angle_deg", 45.0))
    max_start_distance = float(
        config.get("occlusion_recovery_max_start_distance_body_lengths", 1.25)
    )
    min_distance_growth = float(
        config.get("occlusion_recovery_min_distance_growth_body_lengths", 0.45)
    )
    min_path_displacement = float(
        config.get("occlusion_recovery_min_path_displacement_body_lengths", 0.45)
    )

    a_observed = _numeric_column(pair_df, "mouse_a_bbox_observed").astype(bool)
    b_observed = _numeric_column(pair_df, "mouse_b_bbox_observed").astype(bool)
    a_x = _numeric_column(pair_df, "mouse_a_bbox_center_x_px", np.nan)
    a_y = _numeric_column(pair_df, "mouse_a_bbox_center_y_px", np.nan)
    b_x = _numeric_column(pair_df, "mouse_b_bbox_center_x_px", np.nan)
    b_y = _numeric_column(pair_df, "mouse_b_bbox_center_y_px", np.nan)
    a_scale = _numeric_column(pair_df, "mouse_a_bbox_scale_px", np.nan)
    b_scale = _numeric_column(pair_df, "mouse_b_bbox_scale_px", np.nan)
    start_distance = _numeric_column(
        pair_df,
        "bbox_center_distance_body_lengths",
        np.inf,
    )
    jointly_observed = np.asarray(valid, dtype=bool) & a_observed & b_observed

    def recover_one_direction(
        *,
        initiator_observed: np.ndarray,
        evader_observed: np.ndarray,
        initiator_x: np.ndarray,
        initiator_y: np.ndarray,
        evader_x: np.ndarray,
        evader_y: np.ndarray,
        initiator_scale: np.ndarray,
        evader_scale: np.ndarray,
        direction: Mapping[str, np.ndarray],
    ) -> np.ndarray:
        recovered = np.zeros(n, dtype=bool)
        dropout = ~initiator_observed & evader_observed
        starts, ends = _boolean_spans(dropout)
        for raw_start, raw_end in zip(starts, ends):
            dropout_start, dropout_end = int(raw_start), int(raw_end)
            context_start = max(0, dropout_start - lookback_frames)
            recent_joint = np.flatnonzero(jointly_observed[context_start:dropout_start])
            if recent_joint.size < min_joint_frames:
                continue
            recent_joint = recent_joint + context_start
            last_joint = int(recent_joint[-1])
            if not np.isfinite(start_distance[last_joint]) or (
                start_distance[last_joint] > max_start_distance
            ):
                continue
            pursuit_values = np.asarray(direction["pursuit"], dtype=float)[recent_joint]
            pursuit_support = np.isfinite(pursuit_values) & (pursuit_values >= min_pursuit)
            if float(np.mean(pursuit_support)) < min_pursuit_fraction:
                continue

            reaction_end = min(n - 1, dropout_start + confirmation_frames)
            reaction_turn = np.asarray(direction["turn"], dtype=float)[
                max(context_start, last_joint - 1) : reaction_end + 1
            ]
            if not np.any(np.isfinite(reaction_turn) & (reaction_turn >= min_turn)):
                continue

            anchor = np.asarray(
                [initiator_x[last_joint], initiator_y[last_joint]],
                dtype=float,
            )
            initial_evader = np.asarray(
                [evader_x[last_joint], evader_y[last_joint]],
                dtype=float,
            )
            scale = max(
                float(initiator_scale[last_joint])
                if np.isfinite(initiator_scale[last_joint])
                else 0.0,
                float(evader_scale[last_joint]) if np.isfinite(evader_scale[last_joint]) else 0.0,
                8.0,
            )
            if not np.all(np.isfinite(anchor)) or not np.all(np.isfinite(initial_evader)):
                continue
            initial_radius = float(np.linalg.norm(initial_evader - anchor)) / scale
            confirmation = None
            for frame in range(dropout_start, min(dropout_end, reaction_end) + 1):
                position = np.asarray([evader_x[frame], evader_y[frame]], dtype=float)
                if not evader_observed[frame] or not np.all(np.isfinite(position)):
                    break
                radial_growth = float(np.linalg.norm(position - anchor)) / scale - initial_radius
                path_displacement = float(np.linalg.norm(position - initial_evader)) / scale
                if (
                    radial_growth >= min_distance_growth
                    and path_displacement >= min_path_displacement
                ):
                    confirmation = frame
                    break
            if confirmation is None:
                continue

            event_start = max(0, last_joint - pre_context_frames)
            event_end = min(n - 1, int(confirmation) + state_frames)
            # The evader identity must remain directly observed.  A later
            # reappearance of the initiator does not invalidate the already
            # confirmed short avoidance state, but an evader dropout does.
            missing_evader = np.flatnonzero(~evader_observed[int(confirmation) : event_end + 1])
            if missing_evader.size:
                event_end = int(confirmation) + int(missing_evader[0]) - 1
            if event_end >= event_start:
                recovered[event_start : event_end + 1] = True
        return recovered

    # A->B approach followed by A disappearing means B is the evader; the
    # second return value is the symmetric B->A case.
    a_to_b = recover_one_direction(
        initiator_observed=a_observed,
        evader_observed=b_observed,
        initiator_x=a_x,
        initiator_y=a_y,
        evader_x=b_x,
        evader_y=b_y,
        initiator_scale=a_scale,
        evader_scale=b_scale,
        direction=first_direction,
    )
    b_to_a = recover_one_direction(
        initiator_observed=b_observed,
        evader_observed=a_observed,
        initiator_x=b_x,
        initiator_y=b_y,
        evader_x=a_x,
        evader_y=a_y,
        initiator_scale=b_scale,
        evader_scale=a_scale,
        direction=second_direction,
    )
    return a_to_b, b_to_a


def _relative_change(current: np.ndarray, previous: np.ndarray) -> np.ndarray:
    """Return a finite relative change, defaulting to zero at clip boundaries."""

    result = np.zeros_like(current, dtype=float)
    finite = np.isfinite(current) & np.isfinite(previous)
    with np.errstate(divide="ignore", invalid="ignore"):
        result[finite] = (current[finite] - previous[finite]) / np.maximum(
            np.abs(previous[finite]), 1e-6
        )
    result[~np.isfinite(result)] = 0.0
    return result


def _boolean_spans(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return inclusive spans for a one-dimensional boolean signal."""

    values = np.asarray(mask, dtype=bool).reshape(-1)
    if not values.size:
        empty = np.asarray([], dtype=int)
        return empty, empty
    starts = np.flatnonzero(values & np.r_[True, ~values[:-1]])
    ends = np.flatnonzero(values & np.r_[~values[1:], True])
    return starts, ends


def _bbox_coherent_translation_bouts(
    pair_df: pd.DataFrame,
    attack_mask: np.ndarray,
    *,
    fps: float,
    attack_config: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Find contact bouts that travel like pursuit rather than a local grapple.

    Attack and chase can both contain overlapping boxes and rapid motion.  The
    separating cue is the *pair centroid*: a chase translates coherently across
    the arena, while a fight normally spends most of its path oscillating in a
    small area.  This helper uses only box geometry, so it remains available
    when wrestling or riding hides one animal's keypoints.

    The returned directional masks describe which mouse begins behind the
    other along the bout's travel direction.  They are only role evidence; the
    caller still applies the normal temporal FSM and event-duration rules.
    """

    mask = np.asarray(attack_mask, dtype=bool).reshape(-1)
    n = len(mask)
    empty_bool = np.zeros(n, dtype=bool)
    empty_float = np.zeros(n, dtype=float)
    if not bool(attack_config.get("reclassify_coherent_translation_as_chase", False)):
        return empty_bool, empty_bool.copy(), empty_bool.copy(), empty_float

    required_columns = {
        "bbox_pair_valid",
        "mouse_a_bbox_center_x_px",
        "mouse_a_bbox_center_y_px",
        "mouse_b_bbox_center_x_px",
        "mouse_b_bbox_center_y_px",
        "mouse_a_bbox_scale_px",
        "mouse_b_bbox_scale_px",
    }
    if not required_columns.issubset(pair_df.columns):
        return empty_bool, empty_bool.copy(), empty_bool.copy(), empty_float

    a_center = np.column_stack(
        (
            _numeric_column(pair_df, "mouse_a_bbox_center_x_px", np.nan),
            _numeric_column(pair_df, "mouse_a_bbox_center_y_px", np.nan),
        )
    )
    b_center = np.column_stack(
        (
            _numeric_column(pair_df, "mouse_b_bbox_center_x_px", np.nan),
            _numeric_column(pair_df, "mouse_b_bbox_center_y_px", np.nan),
        )
    )
    a_scale = _numeric_column(pair_df, "mouse_a_bbox_scale_px", np.nan)
    b_scale = _numeric_column(pair_df, "mouse_b_bbox_scale_px", np.nan)
    pair_valid = _numeric_column(pair_df, "bbox_pair_valid").astype(bool)
    pair_scale = 0.5 * (a_scale + b_scale)
    finite = (
        pair_valid
        & np.isfinite(a_center).all(axis=1)
        & np.isfinite(b_center).all(axis=1)
        & np.isfinite(pair_scale)
        & (pair_scale >= 8.0)
    )

    bridged = mask.copy()
    gap_limit = max(
        int(
            round(
                max(
                    float(
                        attack_config.get(
                            "coherent_translation_fill_gap_seconds",
                            attack_config.get("fill_gap_seconds", 0.0),
                        )
                    ),
                    0.0,
                )
                * max(float(fps), 1e-9)
            )
        ),
        0,
    )
    active = np.flatnonzero(bridged)
    if gap_limit and active.size > 1:
        for left, right in zip(active[:-1], active[1:]):
            if int(right - left - 1) <= gap_limit:
                bridged[int(left) : int(right) + 1] = True

    minimum_frames = max(
        int(
            round(
                max(
                    float(attack_config.get("coherent_translation_min_duration_seconds", 0.75)),
                    0.0,
                )
                * max(float(fps), 1e-9)
            )
        ),
        2,
    )
    minimum_valid_fraction = float(
        np.clip(
            attack_config.get("coherent_translation_min_valid_fraction", 0.50),
            0.0,
            1.0,
        )
    )
    minimum_net = max(
        float(attack_config.get("coherent_translation_min_net_body_lengths", 2.0)),
        0.0,
    )
    minimum_efficiency = float(
        np.clip(
            attack_config.get("coherent_translation_min_path_efficiency", 0.50),
            0.0,
            1.0,
        )
    )
    maximum_step = max(
        float(attack_config.get("coherent_translation_max_step_body_lengths", 1.50)),
        0.0,
    )
    maximum_sample_gap = max(
        int(attack_config.get("coherent_translation_max_sample_gap_frames", 3)),
        1,
    )
    role_fraction = float(
        np.clip(
            attack_config.get("coherent_translation_role_min_fraction", 0.55),
            0.50,
            1.0,
        )
    )
    role_context_frames = max(
        int(
            round(
                max(
                    float(attack_config.get("coherent_translation_role_context_seconds", 1.0)),
                    0.0,
                )
                * max(float(fps), 1e-9)
            )
        ),
        1,
    )

    translating = np.zeros(n, dtype=bool)
    a_to_b = np.zeros(n, dtype=bool)
    b_to_a = np.zeros(n, dtype=bool)
    score = np.zeros(n, dtype=float)
    pair_centroid = 0.5 * (a_center + b_center)
    starts, ends = _boolean_spans(bridged)
    for raw_start, raw_end in zip(starts, ends):
        start, end = int(raw_start), int(raw_end)
        if end - start + 1 < minimum_frames:
            continue
        valid_indices = np.flatnonzero(finite[start : end + 1]) + start
        if valid_indices.size < 3:
            continue
        if valid_indices.size / float(end - start + 1) < minimum_valid_fraction:
            continue

        left_indices = valid_indices[:-1]
        right_indices = valid_indices[1:]
        sample_gaps = right_indices - left_indices
        scale = 0.5 * (pair_scale[left_indices] + pair_scale[right_indices])
        steps_px = pair_centroid[right_indices] - pair_centroid[left_indices]
        steps = steps_px / np.maximum(scale[:, None], 8.0)
        step_length = np.linalg.norm(steps, axis=1)
        accepted = sample_gaps <= maximum_sample_gap
        if maximum_step > 0.0:
            accepted &= step_length <= maximum_step
        if int(np.count_nonzero(accepted)) < 2:
            continue
        accepted_steps = steps[accepted]
        accepted_steps_px = steps_px[accepted]
        path_length = float(np.sum(np.linalg.norm(accepted_steps, axis=1)))
        net_vector = np.sum(accepted_steps, axis=0)
        net_distance = float(np.linalg.norm(net_vector))
        efficiency = net_distance / max(path_length, 1e-9)
        if net_distance < minimum_net or efficiency < minimum_efficiency:
            continue

        translating[start : end + 1] = finite[start : end + 1]
        normalized_score = min(
            net_distance / max(minimum_net, 1e-9),
            efficiency / max(minimum_efficiency, 1e-9),
        )
        score[start : end + 1] = float(np.clip(normalized_score, 0.0, 2.0))

        travel_px = np.sum(accepted_steps_px, axis=0)
        travel_norm = float(np.linalg.norm(travel_px))
        if travel_norm <= 1e-9:
            continue
        travel_unit = travel_px / travel_norm
        role_end = min(end + 1, start + role_context_frames)
        role_indices = np.flatnonzero(finite[start:role_end]) + start
        if role_indices.size < 2:
            role_indices = valid_indices
        projections = (a_center[role_indices] - b_center[role_indices]) @ travel_unit
        projections = projections[np.isfinite(projections) & (np.abs(projections) > 1e-6)]
        if not projections.size:
            continue
        a_behind_fraction = float(np.mean(projections < 0.0))
        b_behind_fraction = float(np.mean(projections > 0.0))
        if a_behind_fraction >= role_fraction:
            a_to_b[start : end + 1] = translating[start : end + 1]
        elif b_behind_fraction >= role_fraction:
            b_to_a[start : end + 1] = translating[start : end + 1]
        elif float(np.median(projections)) <= 0.0:
            a_to_b[start : end + 1] = translating[start : end + 1]
        else:
            b_to_a[start : end + 1] = translating[start : end + 1]

    return translating, a_to_b, b_to_a, score


def _direction_arrays(
    pair_df: pd.DataFrame,
    prefix: str,
    *,
    smoothing_window: int = 1,
) -> dict[str, np.ndarray]:
    actor_speed = _numeric_column(pair_df, f"{prefix}_actor_behavior_speed_cm_s")
    target_speed = _numeric_column(pair_df, f"{prefix}_target_behavior_speed_cm_s")
    actor_raw_speed = _numeric_column(pair_df, f"{prefix}_actor_speed_cm_s")
    target_raw_speed = _numeric_column(pair_df, f"{prefix}_target_speed_cm_s")
    actor_acceleration = _numeric_column(pair_df, f"{prefix}_actor_acceleration_cm_s2")
    target_acceleration = _numeric_column(pair_df, f"{prefix}_target_acceleration_cm_s2")
    actor_nose_speed = _numeric_column(pair_df, f"{prefix}_actor_nose_speed_cm_s")
    target_nose_speed = _numeric_column(pair_df, f"{prefix}_target_nose_speed_cm_s")
    pursuit = _numeric_column(pair_df, f"{prefix}_pursuit_alignment")
    escape = _numeric_column(pair_df, f"{prefix}_target_escape_alignment")
    similarity = _numeric_column(pair_df, f"{prefix}_direction_similarity")
    behind = _numeric_column(pair_df, f"{prefix}_actor_behind_target")
    turn = _numeric_column(pair_df, f"{prefix}_target_turn_angle_deg")
    actor_deformation = _numeric_column(pair_df, f"{prefix}_actor_pose_deformation_energy")
    target_deformation = _numeric_column(pair_df, f"{prefix}_target_pose_deformation_energy")
    actor_bbox_speed = _numeric_column(
        pair_df,
        f"{prefix}_actor_bbox_speed_body_lengths_per_frame",
    )
    target_bbox_speed = _numeric_column(
        pair_df,
        f"{prefix}_target_bbox_speed_body_lengths_per_frame",
    )
    actor_bbox_acceleration = _numeric_column(
        pair_df,
        f"{prefix}_actor_bbox_acceleration_body_lengths_per_frame2",
    )
    target_bbox_acceleration = _numeric_column(
        pair_df,
        f"{prefix}_target_bbox_acceleration_body_lengths_per_frame2",
    )
    actor_bbox_area_change = _numeric_column(
        pair_df,
        f"{prefix}_actor_bbox_area_change_ratio",
    )
    target_bbox_area_change = _numeric_column(
        pair_df,
        f"{prefix}_target_bbox_area_change_ratio",
    )
    actor_bbox_jump = _numeric_column(pair_df, f"{prefix}_actor_bbox_jump_score")
    target_bbox_jump = _numeric_column(pair_df, f"{prefix}_target_bbox_jump_score")
    actor_bbox_observed = _numeric_column(
        pair_df,
        f"{prefix}_actor_bbox_observed",
        1.0,
    ).astype(bool)
    actor_bbox_imputed = _numeric_column(
        pair_df,
        f"{prefix}_actor_bbox_imputed",
        0.0,
    ).astype(bool)
    target_bbox_observed = _numeric_column(
        pair_df,
        f"{prefix}_target_bbox_observed",
        1.0,
    ).astype(bool)
    target_bbox_imputed = _numeric_column(
        pair_df,
        f"{prefix}_target_bbox_imputed",
        0.0,
    ).astype(bool)

    def smooth(values: np.ndarray) -> np.ndarray:
        return _rolling_median(values, smoothing_window)

    actor_speed = smooth(actor_speed)
    target_speed = smooth(target_speed)
    pursuit = smooth(pursuit)
    escape = smooth(escape)
    similarity = smooth(similarity)
    behind = np.clip(smooth(behind), 0.0, 1.0)
    turn = smooth(turn)
    actor_deformation = smooth(actor_deformation)
    target_deformation = smooth(target_deformation)
    speed_sum = actor_speed + target_speed
    speed_gap_ratio = np.zeros_like(speed_sum, dtype=float)
    moving = speed_sum > 1e-6
    speed_gap_ratio[moving] = (actor_speed[moving] - target_speed[moving]) / speed_sum[moving]
    return {
        "actor_speed": actor_speed,
        "target_speed": target_speed,
        "speed_gap_ratio": speed_gap_ratio,
        "pursuit": pursuit,
        "escape": escape,
        "similarity": similarity,
        "behind": behind,
        "turn": turn,
        # Keep the unsmoothed speed for attack's high-energy gate.  The social
        # direction channels above use the robust median-smoothed values.
        "actor_raw_speed": actor_raw_speed,
        "target_raw_speed": target_raw_speed,
        # These are intentionally unsmoothed. An impact is a short transient
        # and a median filter would erase the signature that separates it from
        # ordinary sustained chase motion.
        "actor_acceleration": actor_acceleration,
        "target_acceleration": target_acceleration,
        "actor_nose_speed": actor_nose_speed,
        "target_nose_speed": target_nose_speed,
        "actor_deformation": actor_deformation,
        "target_deformation": target_deformation,
        "actor_bbox_speed": actor_bbox_speed,
        "target_bbox_speed": target_bbox_speed,
        "actor_bbox_acceleration": actor_bbox_acceleration,
        "target_bbox_acceleration": target_bbox_acceleration,
        "actor_bbox_area_change": actor_bbox_area_change,
        "target_bbox_area_change": target_bbox_area_change,
        "actor_bbox_jump": actor_bbox_jump,
        "target_bbox_jump": target_bbox_jump,
        "actor_bbox_observed": actor_bbox_observed,
        "actor_bbox_imputed": actor_bbox_imputed,
        "target_bbox_observed": target_bbox_observed,
        "target_bbox_imputed": target_bbox_imputed,
    }


def _direction_ids(pair_df: pd.DataFrame, prefix: str) -> tuple[int, int]:
    if prefix == "a_to_b":
        actor_column, target_column = "mouse_a_id", "mouse_b_id"
    else:
        actor_column, target_column = "mouse_b_id", "mouse_a_id"
    if not len(pair_df):
        return -1, -1
    actor_values = _numeric_column(pair_df, actor_column, -1)
    target_values = _numeric_column(pair_df, target_column, -1)
    valid = (actor_values >= 0) & (target_values >= 0)
    if not valid.any():
        return -1, -1
    first_valid = int(np.flatnonzero(valid)[0])
    return int(actor_values[first_valid]), int(target_values[first_valid])


def _stabilize_roles(
    mask: np.ndarray,
    first_direction: np.ndarray,
    second_direction: np.ndarray,
    first_strength: np.ndarray,
    second_strength: np.ndarray,
    first_actor: int,
    first_target: int,
    second_actor: int,
    second_target: int,
    *,
    evidence_context_frames: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose one direction for each event span instead of each frame.

    Pair geometry can make both directions look plausible for a few frames,
    especially during occlusion.  Selecting the role independently at every
    frame creates actor/target flips and makes a single event appear to change
    meaning.  The temporal FSM already defines an event span, so role choice
    is made from the aggregate directional evidence in that same span.
    """

    values = np.asarray(mask, dtype=bool).reshape(-1)
    first_values = np.asarray(first_direction, dtype=bool).reshape(-1)
    second_values = np.asarray(second_direction, dtype=bool).reshape(-1)
    first_scores = np.asarray(first_strength, dtype=float).reshape(-1)
    second_scores = np.asarray(second_strength, dtype=float).reshape(-1)
    actor = np.full(values.shape, -1, dtype=int)
    target = np.full(values.shape, -1, dtype=int)
    starts, ends = _boolean_spans(values)
    for start, end in zip(starts, ends):
        left, right = int(start), int(end) + 1
        context_right = right
        if evidence_context_frames is not None and int(evidence_context_frames) > 0:
            context_right = min(right, left + int(evidence_context_frames))
        first_support = first_values[left:context_right]
        second_support = second_values[left:context_right]
        # The first configured context can be an FSM recovery-only prefix.
        # Fall back to the complete event only when it contains no directional
        # evidence at all.
        if not first_support.any() and not second_support.any() and context_right < right:
            context_right = right
            first_support = first_values[left:right]
            second_support = second_values[left:right]
        if first_support.any() and not second_support.any():
            use_first = True
        elif second_support.any() and not first_support.any():
            use_first = False
        else:
            first_segment = first_scores[left:context_right][first_support]
            second_segment = second_scores[left:context_right][second_support]
            first_total = (
                float(np.nanmean(first_segment))
                if first_segment.size
                else float(np.nanmean(first_scores[left:context_right]))
            )
            second_total = (
                float(np.nanmean(second_segment))
                if second_segment.size
                else float(np.nanmean(second_scores[left:context_right]))
            )
            if not np.isfinite(first_total):
                first_total = float("-inf")
            if not np.isfinite(second_total):
                second_total = float("-inf")
            use_first = first_total >= second_total
        if use_first:
            actor[left:right] = int(first_actor)
            target[left:right] = int(first_target)
        else:
            actor[left:right] = int(second_actor)
            target[left:right] = int(second_target)
    return actor, target


def _contact_mask(
    pair_df: pd.DataFrame,
    *,
    contact_config: Mapping[str, Any],
) -> np.ndarray:
    valid = _numeric_column(pair_df, "valid_pair").astype(bool)
    head = np.minimum(
        _numeric_column(pair_df, "a_to_b_nose_head_distance_cm", np.inf),
        _numeric_column(pair_df, "b_to_a_nose_head_distance_cm", np.inf),
    )
    tail = np.minimum(
        _numeric_column(pair_df, "a_to_b_nose_tail_distance_cm", np.inf),
        _numeric_column(pair_df, "b_to_a_nose_tail_distance_cm", np.inf),
    )
    body = np.minimum(
        _numeric_column(pair_df, "a_to_b_nose_body_distance_cm", np.inf),
        _numeric_column(pair_df, "b_to_a_nose_body_distance_cm", np.inf),
    )
    generic_threshold = float(contact_config.get("distance_cm", 3.0))
    head_threshold = float(contact_config.get("nose_head_distance_cm", generic_threshold))
    try:
        head_tolerance = max(
            float(contact_config.get("nose_head_distance_multiplier", 1.0)),
            1.0,
        )
    except (TypeError, ValueError):
        head_tolerance = 1.0
    head_threshold *= head_tolerance
    tail_threshold = float(contact_config.get("nose_tail_distance_cm", generic_threshold))
    return valid & (
        (head <= head_threshold) | (tail <= tail_threshold) | (body <= generic_threshold)
    )


def _nose_head_contact_mask(
    pair_df: pd.DataFrame,
    *,
    contact_config: Mapping[str, Any],
) -> np.ndarray:
    """Return the tolerant direct nose-head geometry channel."""

    valid = _numeric_column(pair_df, "valid_pair").astype(bool)
    head = np.minimum(
        _numeric_column(pair_df, "a_to_b_nose_head_distance_cm", np.inf),
        _numeric_column(pair_df, "b_to_a_nose_head_distance_cm", np.inf),
    )
    generic_threshold = float(contact_config.get("distance_cm", 3.0))
    head_threshold = float(contact_config.get("nose_head_distance_cm", generic_threshold))
    try:
        multiplier = max(
            float(contact_config.get("nose_head_distance_multiplier", 1.0)),
            1.0,
        )
    except (TypeError, ValueError):
        multiplier = 1.0
    return valid & (head <= max(head_threshold, 0.0) * multiplier)


def _bbox_contact_mask(
    pair_df: pd.DataFrame,
    *,
    attack_config: Mapping[str, Any],
) -> np.ndarray:
    """Return a conservative box-overlap/proximity contact channel.

    This channel deliberately does not use a fixed pixel distance.  The
    preprocessing layer exports overlap and center distance normalized by the
    detected body/box scale, which remains usable when one animal is partly
    hidden.  A box is considered contact-like only when both boxes are
    available and either they overlap or their normalized centers are very
    close.
    """

    if "bbox_pair_valid" in pair_df:
        bbox_valid = _numeric_column(pair_df, "bbox_pair_valid").astype(bool)
        # A predicted box is usable only as an occlusion bridge when at least
        # one member of the pair is still a real detector observation.  This
        # prevents two stale boxes from creating an attack while both mice are
        # absent.  Older callers without this diagnostic column retain the
        # historical all-valid fallback.
        bbox_observed = _numeric_column(pair_df, "bbox_pair_observed", 1.0).astype(bool)
    else:
        overlap_values = _numeric_column(pair_df, "bbox_overlap_iou", np.nan)
        distance_values = _numeric_column(
            pair_df,
            "bbox_center_distance_body_lengths",
            np.nan,
        )
        bbox_valid = np.isfinite(overlap_values) | np.isfinite(distance_values)
        bbox_observed = np.ones_like(bbox_valid, dtype=bool)
    overlap = _numeric_column(pair_df, "bbox_overlap_iou", 0.0)
    center_distance = _numeric_column(
        pair_df,
        "bbox_center_distance_body_lengths",
        np.inf,
    )
    min_overlap = max(float(attack_config.get("min_bbox_overlap_iou", 0.05)), 0.0)
    max_distance = max(
        float(attack_config.get("max_bbox_center_distance_body_lengths", 1.35)),
        0.0,
    )
    if bool(attack_config.get("bbox_require_overlap", False)):
        contact_geometry = overlap >= min_overlap
    else:
        contact_geometry = (overlap >= min_overlap) | (center_distance <= max_distance)
    return bbox_valid & bbox_observed & contact_geometry


def build_semantic_pair_signals(
    pair_df: pd.DataFrame,
    enriched: pd.DataFrame,
    *,
    fps: float,
    social_config: Mapping[str, Any],
    contact_config: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """Build relative social evidence for one pair timeline.

    The returned masks are intentionally evidence masks, not finalized events.
    The caller still runs them through the shared parallel FSM so minimum
    durations, gap filling, and event IDs remain auditable in one place.
    """

    n = len(pair_df)
    if n == 0:
        empty_bool = np.zeros(0, dtype=bool)
        empty_float = np.zeros(0, dtype=float)
        empty_int = np.zeros(0, dtype=int)
        return {
            "approach_mask": empty_bool,
            "approach_score": empty_float,
            "approach_actor": empty_int,
            "approach_target": empty_int,
            "chase_mask": empty_bool,
            "chase_score": empty_float,
            "chase_actor": empty_int,
            "chase_target": empty_int,
            "chase_occlusion_bridge_mask": empty_bool,
            "chase_bbox_translation_mask": empty_bool,
            "avoidance_mask": empty_bool,
            "avoidance_score": empty_float,
            "avoidance_actor": empty_int,
            "avoidance_target": empty_int,
            "attack_mask": empty_bool,
            "attack_score": empty_float,
            "attack_actor": empty_int,
            "attack_target": empty_int,
            "attack_state_bridge_mask": empty_bool,
            "contact_mask": empty_bool,
            "bbox_contact_mask": empty_bool,
        }

    analysis_fps = max(float(fps), 1e-9)
    valid = _numeric_column(pair_df, "valid_pair").astype(bool)
    distance = _rolling_median(
        _numeric_column(pair_df, "center_distance_cm", np.nan),
        max(
            int(round(analysis_fps * float(social_config.get("feature_smoothing_seconds", 0.25)))),
            1,
        ),
    )
    approach_cfg = dict(social_config.get("semantic_approach", {}))
    trend_lookback = max(
        int(round(analysis_fps * float(approach_cfg.get("trend_lookback_seconds", 0.75)))),
        1,
    )
    lookback = trend_lookback
    previous_distance = np.full(n, np.nan, dtype=float)
    if lookback < n:
        previous_distance[lookback:] = distance[:-lookback]
    relative_drop = np.maximum(
        _relative_change(distance, previous_distance) * -1.0,
        0.0,
    )
    current_distance_gate = np.ones(n, dtype=bool)
    if "max_current_distance_cm" in approach_cfg:
        current_distance_gate &= distance <= float(approach_cfg["max_current_distance_cm"])

    # A body-length-derived centimetre scale is useful for reporting, but it
    # is not accurate enough to make a 10.0 cm versus 10.3 cm distinction in
    # short mixed-size Beiyi clips. Profiles can therefore define arrival as
    # the low-distance portion of this pair's own timeline. Explicit absolute
    # thresholds remain supported for calibrated experiments and tests.
    if "arrival_distance_quantile" in approach_cfg:
        finite_arrival_distances = distance[valid & np.isfinite(distance)]
        if finite_arrival_distances.size:
            arrival_distance = float(
                np.quantile(
                    finite_arrival_distances,
                    np.clip(float(approach_cfg["arrival_distance_quantile"]), 0.0, 1.0),
                )
            ) * max(float(approach_cfg.get("arrival_distance_multiplier", 1.0)), 0.0)
        else:
            arrival_distance = float("-inf")
    else:
        arrival_distance = float(approach_cfg.get("max_final_distance_cm", float("inf")))
    arrival_context_frames = max(
        int(round(analysis_fps * float(approach_cfg.get("arrival_context_seconds", 3.0)))),
        1,
    )
    near_arrival = valid & np.isfinite(distance) & (distance <= arrival_distance)
    future_near_arrival = _rolling_future_any(near_arrival, arrival_context_frames)
    max_target_speed_fraction = max(
        float(approach_cfg.get("max_target_speed_fraction_of_actor", 1.0)),
        0.0,
    )

    smoothing_window = max(
        int(round(analysis_fps * float(social_config.get("feature_smoothing_seconds", 0.25)))),
        1,
    )
    ab = _direction_arrays(
        pair_df,
        "a_to_b",
        smoothing_window=smoothing_window,
    )
    ba = _direction_arrays(
        pair_df,
        "b_to_a",
        smoothing_window=smoothing_window,
    )
    actor_a, target_a = _direction_ids(pair_df, "a_to_b")
    actor_b, target_b = _direction_ids(pair_df, "b_to_a")

    approach_direction = {}
    for name, values in (("a_to_b", ab), ("b_to_a", ba)):
        approach_direction[name] = (
            valid
            & np.isfinite(distance)
            & current_distance_gate
            & future_near_arrival
            & (values["pursuit"] >= float(approach_cfg.get("min_pursuit_alignment", 0.25)))
            & (values["speed_gap_ratio"] >= float(approach_cfg.get("min_speed_gap_ratio", 0.10)))
            & (values["target_speed"] <= values["actor_speed"] * max_target_speed_fraction)
            & (relative_drop >= float(approach_cfg.get("min_relative_distance_drop", 0.03)))
        )
    approach_hold_frames = max(
        int(round(float(approach_cfg.get("hold_seconds", 0.75)) * analysis_fps)),
        1,
    )
    approach_hold_fraction = float(approach_cfg.get("hold_min_fraction", 0.20))
    approach_hold_ab = _symmetric_hold(
        approach_direction["a_to_b"],
        approach_hold_frames,
        min_fraction=approach_hold_fraction,
    )
    approach_hold_ba = _symmetric_hold(
        approach_direction["b_to_a"],
        approach_hold_frames,
        min_fraction=approach_hold_fraction,
    )
    approach_hold = (approach_hold_ab | approach_hold_ba) & valid
    approach_strength_ab = np.maximum(ab["pursuit"], 0.0) + np.maximum(ab["speed_gap_ratio"], 0.0)
    approach_strength_ba = np.maximum(ba["pursuit"], 0.0) + np.maximum(ba["speed_gap_ratio"], 0.0)
    approach_actor, approach_target = _stabilize_roles(
        approach_hold,
        approach_direction["a_to_b"],
        approach_direction["b_to_a"],
        approach_strength_ab,
        approach_strength_ba,
        actor_a,
        target_a,
        actor_b,
        target_b,
    )
    approach_score = (
        np.maximum(
            approach_strength_ab,
            approach_strength_ba,
        )
        + relative_drop
    )

    chase_cfg = dict(social_config.get("semantic_chase", {}))
    try:
        chase_role_context_seconds = float(chase_cfg.get("role_initial_context_seconds", 0.0))
    except (TypeError, ValueError, OverflowError):
        chase_role_context_seconds = 0.0
    if not np.isfinite(chase_role_context_seconds):
        chase_role_context_seconds = 0.0
    chase_role_context_frames = max(
        int(round(chase_role_context_seconds * analysis_fps)),
        0,
    )
    min_pursuit = float(chase_cfg.get("min_pursuit_alignment", 0.35))
    min_escape = float(chase_cfg.get("min_target_escape_alignment", 0.20))
    min_similarity = float(chase_cfg.get("min_direction_similarity", 0.45))
    min_directional_score = float(chase_cfg.get("min_directional_score", 0.35))
    min_speed_gap = float(chase_cfg.get("min_actor_speed_gap_ratio", -0.15))
    min_combined_speed = float(chase_cfg.get("min_combined_speed_cm_s", 3.0))
    min_follow_actor_speed = float(chase_cfg.get("min_follow_actor_speed_cm_s", 3.0))
    min_follow_target_speed = float(chase_cfg.get("min_follow_target_speed_cm_s", 1.0))
    min_chase_actor_speed = float(chase_cfg.get("min_actor_speed_cm_s", 3.0))
    min_chase_target_speed = float(chase_cfg.get("min_target_speed_cm_s", 1.0))
    min_chase_target_speed_fraction = max(
        float(chase_cfg.get("min_target_speed_fraction_of_actor", 0.0)),
        0.0,
    )
    occlusion_observed_fraction = float(
        np.clip(
            chase_cfg.get("target_occlusion_observed_fraction", 0.0),
            0.0,
            1.0,
        )
    )
    follow_distance_quantile = float(chase_cfg.get("follow_distance_quantile", 0.85))
    center_distance_body_lengths = _numeric_column(
        pair_df,
        "center_distance_body_lengths",
        np.inf,
    )
    finite_follow_distances = distance[np.isfinite(distance)]
    follow_distance = (
        float(
            np.quantile(
                finite_follow_distances,
                np.clip(follow_distance_quantile, 0.0, 1.0),
            )
        )
        if finite_follow_distances.size
        else float("inf")
    )
    follow_context_frames = max(
        int(round(float(chase_cfg.get("follow_context_seconds", 1.0)) * analysis_fps)),
        1,
    )
    closing_context = _rolling_any(
        relative_drop >= float(chase_cfg.get("follow_min_relative_distance_drop", 0.08)),
        follow_context_frames,
    )

    def chase_direction(values: Mapping[str, np.ndarray]) -> np.ndarray:
        # A chase requires common travel direction and a faster pursuing mouse.
        # The second directional branch tolerates one noisy pursuit/escape
        # channel, but still requires both channels to be directionally positive
        # and their combined evidence to be substantial.
        directional = ((values["pursuit"] >= min_pursuit) & (values["escape"] >= min_escape)) | (
            (values["pursuit"] >= min_pursuit * 0.40)
            & (values["escape"] >= min_escape * 0.40)
            & ((values["pursuit"] + values["escape"]) >= min_directional_score)
        )
        close_gate = np.ones(n, dtype=bool)
        if "max_distance_cm" in chase_cfg:
            close_gate &= distance <= float(chase_cfg["max_distance_cm"])
        if "max_distance_body_lengths" in chase_cfg:
            close_gate &= np.isfinite(center_distance_body_lengths) & (
                center_distance_body_lengths <= float(chase_cfg["max_distance_body_lengths"])
            )
        if "max_distance_quantile" in chase_cfg:
            finite = distance[np.isfinite(distance)]
            if finite.size:
                close_gate &= distance <= float(
                    np.quantile(
                        finite,
                        np.clip(float(chase_cfg["max_distance_quantile"]), 0.0, 1.0),
                    )
                )
        following = (
            (values["similarity"] >= min_similarity)
            & (
                values["speed_gap_ratio"]
                >= float(chase_cfg.get("min_follow_speed_gap_ratio", 0.10))
            )
            & (values["actor_speed"] >= min_follow_actor_speed)
            & (values["target_speed"] >= min_follow_target_speed)
            & ((distance <= follow_distance) | closing_context)
        )
        target_observed_fraction = float(np.mean(values["target_bbox_observed"]))
        target_occlusion_bridge = bool(
            values["target_bbox_imputed"].any()
            and target_observed_fraction < occlusion_observed_fraction
        )
        target_relative_motion = (
            values["target_speed"] >= values["actor_speed"] * min_chase_target_speed_fraction
        ) | target_occlusion_bridge
        return (
            valid
            & np.isfinite(distance)
            & close_gate
            & (values["similarity"] >= min_similarity)
            & (values["speed_gap_ratio"] >= min_speed_gap)
            & (values["actor_speed"] >= min_chase_actor_speed)
            & (values["target_speed"] >= min_chase_target_speed)
            & target_relative_motion
            & ((values["actor_speed"] + values["target_speed"]) >= min_combined_speed)
            & (directional | following)
        )

    chase_ab = chase_direction(ab)
    chase_ba = chase_direction(ba)
    occlusion_fill_gap_frames = max(
        int(round(float(chase_cfg.get("occlusion_fill_gap_seconds", 0.0)) * analysis_fps)),
        0,
    )
    occlusion_gap_min_fraction = float(chase_cfg.get("occlusion_gap_min_imputed_fraction", 0.50))
    raw_chase_ab = chase_ab.copy()
    raw_chase_ba = chase_ba.copy()
    chase_ab = _fill_supported_internal_gaps(
        chase_ab,
        ab["target_bbox_imputed"],
        occlusion_fill_gap_frames,
        min_support_fraction=occlusion_gap_min_fraction,
    )
    chase_ba = _fill_supported_internal_gaps(
        chase_ba,
        ba["target_bbox_imputed"],
        occlusion_fill_gap_frames,
        min_support_fraction=occlusion_gap_min_fraction,
    )
    bbox_pair_valid = _numeric_column(pair_df, "bbox_pair_valid").astype(bool)
    bbox_pair_observed = _numeric_column(pair_df, "bbox_pair_observed").astype(bool)
    chase_occlusion_bridge = (
        ((chase_ab & ~raw_chase_ab) | (chase_ba & ~raw_chase_ba))
        & bbox_pair_valid
        & bbox_pair_observed
    )
    directional_chase = chase_ab | chase_ba
    combined_chase = _fill_supported_internal_gaps(
        directional_chase,
        ab["target_bbox_imputed"] | ba["target_bbox_imputed"],
        occlusion_fill_gap_frames,
        min_support_fraction=occlusion_gap_min_fraction,
    )
    chase_occlusion_bridge |= (
        (combined_chase & ~directional_chase) & bbox_pair_valid & bbox_pair_observed
    )
    chase_strength_ab = (
        np.maximum(ab["pursuit"], 0.0)
        + np.maximum(ab["escape"], 0.0)
        + np.maximum(ab["similarity"], 0.0)
        + np.maximum(ab["speed_gap_ratio"], 0.0)
        + float(chase_cfg.get("role_behind_weight", 1.0)) * ab["behind"]
    )
    chase_strength_ba = (
        np.maximum(ba["pursuit"], 0.0)
        + np.maximum(ba["escape"], 0.0)
        + np.maximum(ba["similarity"], 0.0)
        + np.maximum(ba["speed_gap_ratio"], 0.0)
        + float(chase_cfg.get("role_behind_weight", 1.0)) * ba["behind"]
    )
    chase_ab_or_ba = combined_chase
    # The direction is stabilized after the boolean hold below.  Keeping the
    # role choice at event-span granularity prevents a noisy peak frame from
    # reversing the chase actor.
    chase_score = np.maximum(chase_strength_ab, chase_strength_ba)

    avoidance_cfg = dict(social_config.get("semantic_avoidance", {}))
    near_quantile = float(avoidance_cfg.get("near_distance_quantile", 0.35))
    finite_distances = distance[np.isfinite(distance)]
    near_reference = (
        float(np.quantile(finite_distances, np.clip(near_quantile, 0.0, 1.0)))
        if finite_distances.size
        else float("inf")
    )
    near_distance = float(avoidance_cfg.get("near_distance_multiplier", 1.50)) * near_reference
    near_context = valid & np.isfinite(distance) & (distance <= near_distance)
    context_frames = max(
        int(round(float(avoidance_cfg.get("context_seconds", 3.0)) * analysis_fps)),
        1,
    )
    # Avoidance is a causal transition: the pair must have shown approach
    # evidence before the escape phase.  A near frame alone is not enough,
    # because ordinary separation of two unrelated mice otherwise looks like
    # avoidance.
    prior_approach_ab = np.zeros(n, dtype=bool)
    prior_approach_ba = np.zeros(n, dtype=bool)
    approach_hold_frames = max(
        int(round(float(approach_cfg.get("hold_seconds", 0.75)) * analysis_fps)),
        1,
    )
    approach_hold_fraction = float(approach_cfg.get("hold_min_fraction", 0.20))
    # Avoidance must follow a real directional approach state, not an
    # isolated approach-looking frame somewhere in the previous context
    # window.  Keeping the two directions separate also preserves the causal
    # actor/evader relationship.
    approach_state_ab = _symmetric_hold(
        approach_direction["a_to_b"],
        approach_hold_frames,
        min_fraction=approach_hold_fraction,
    )
    approach_state_ba = _symmetric_hold(
        approach_direction["b_to_a"],
        approach_hold_frames,
        min_fraction=approach_hold_fraction,
    )
    if n > 1:
        prior_approach_ab[1:] = _rolling_any(approach_state_ab[:-1], context_frames)
        prior_approach_ba[1:] = _rolling_any(approach_state_ba[:-1], context_frames)

    # A short clip may begin in the reaction phase.  Treat the first context
    # window as a bounded continuation only when a close, directional escape
    # signature is visible; this does not create an event in a distant pair.
    boundary_context_ab = np.zeros(n, dtype=bool)
    boundary_context_ba = np.zeros(n, dtype=bool)
    boundary_window = min(
        n,
        max(
            int(round(float(avoidance_cfg.get("boundary_context_seconds", 0.50)) * analysis_fps)), 1
        ),
    )
    boundary_signature_ab = (
        (ab["escape"] >= float(avoidance_cfg.get("min_target_escape_alignment", 0.35)))
        | (ab["turn"] >= float(avoidance_cfg.get("min_target_turn_angle_deg", 25.0)))
        | approach_direction["a_to_b"]
    )
    boundary_signature_ba = (
        (ba["escape"] >= float(avoidance_cfg.get("min_target_escape_alignment", 0.35)))
        | (ba["turn"] >= float(avoidance_cfg.get("min_target_turn_angle_deg", 25.0)))
        | approach_direction["b_to_a"]
    )
    if n > 1:
        # A clip can start immediately after the approach trigger.  A bounded
        # one-step closing signal lets the causal context survive a brief
        # partner occlusion without turning every near pair into avoidance.
        boundary_closing = np.zeros(n, dtype=bool)
        boundary_closing[1:] = distance[1:] < distance[:-1]
        boundary_signature_ab |= boundary_closing
        boundary_signature_ba |= boundary_closing
    max_boundary_clip_frames = max(
        int(round(float(avoidance_cfg.get("boundary_max_clip_seconds", 3.0)) * analysis_fps)),
        boundary_window,
    )
    allow_boundary_context = (
        bool(avoidance_cfg.get("allow_clip_start_context", True)) and n <= max_boundary_clip_frames
    )
    if allow_boundary_context:
        boundary_context_ab[:boundary_window] = (
            near_context[:boundary_window] & boundary_signature_ab[:boundary_window]
        )
        boundary_context_ba[:boundary_window] = (
            near_context[:boundary_window] & boundary_signature_ba[:boundary_window]
        )
    # A clip-start exception is deliberately limited to the configured
    # minimum event duration. Expanding it through the full multi-second
    # interaction context would make any later separation in an unrelated
    # pair look like avoidance, even though no approach state was observed in
    # the clip. One second is enough to recover a short clip that starts in
    # the reaction phase while keeping the exception local to its boundary.
    boundary_min_frames = max(
        int(round(float(avoidance_cfg.get("min_duration_seconds", 1.0)) * analysis_fps)),
        boundary_window,
        1,
    )
    boundary_interaction_ab = _rolling_any(
        boundary_context_ab,
        boundary_min_frames,
    )
    boundary_interaction_ba = _rolling_any(
        boundary_context_ba,
        boundary_min_frames,
    )

    turn_window = max(
        int(round(float(avoidance_cfg.get("turn_window_seconds", 0.75)) * analysis_fps)),
        1,
    )
    turn_threshold = float(avoidance_cfg.get("min_target_turn_angle_deg", 25.0))
    turn_fraction_threshold = float(avoidance_cfg.get("min_turn_fraction", 0.25))
    turn_support_ab = _rolling_fraction(ab["turn"] >= turn_threshold, turn_window)
    turn_support_ba = _rolling_fraction(ba["turn"] >= turn_threshold, turn_window)
    turns_ab = turn_support_ab >= turn_fraction_threshold
    turns_ba = turn_support_ba >= turn_fraction_threshold
    reaction_lookback = max(
        int(round(analysis_fps * float(avoidance_cfg.get("distance_trend_seconds", 0.30)))),
        1,
    )
    previous_reaction_distance = np.full(n, np.nan, dtype=float)
    if reaction_lookback < n:
        previous_reaction_distance[reaction_lookback:] = distance[:-reaction_lookback]
    relative_increase = np.maximum(
        _relative_change(distance, previous_reaction_distance),
        0.0,
    )
    increase_required = float(avoidance_cfg.get("min_relative_distance_increase", 0.03))
    # ``boundary_context`` is the bounded exception for a clip that starts in
    # the reaction phase and therefore contains no pre-approach frames.  It is
    # only granted when the first window is both near and directionally
    # reactive; normal clips still require prior approach.
    prior_interaction_ab = prior_approach_ab | boundary_interaction_ab
    prior_interaction_ba = prior_approach_ba | boundary_interaction_ba
    avoidance_ab = (
        valid
        & prior_interaction_ab
        & (ab["escape"] >= float(avoidance_cfg.get("min_target_escape_alignment", 0.35)))
        & (
            (ab["target_speed"] >= float(avoidance_cfg.get("min_evader_speed_cm_s", 1.0)))
            | turns_ab
        )
        & (relative_increase >= increase_required)
        & turns_ab
    )
    avoidance_ba = (
        valid
        & prior_interaction_ba
        & (ba["escape"] >= float(avoidance_cfg.get("min_target_escape_alignment", 0.35)))
        & (
            (ba["target_speed"] >= float(avoidance_cfg.get("min_evader_speed_cm_s", 1.0)))
            | turns_ba
        )
        & (relative_increase >= increase_required)
        & turns_ba
    )
    occlusion_avoidance_ab, occlusion_avoidance_ba = _recover_avoidance_after_initiator_occlusion(
        pair_df,
        valid,
        ab,
        ba,
        fps=analysis_fps,
        config=avoidance_cfg,
    )
    avoidance_ab |= occlusion_avoidance_ab
    avoidance_ba |= occlusion_avoidance_ba
    avoidance_ab_or_ba = avoidance_ab | avoidance_ba
    avoidance_strength_ab = np.maximum(ab["escape"], 0.0) + turn_support_ab
    avoidance_strength_ba = np.maximum(ba["escape"], 0.0) + turn_support_ba
    avoidance_score = np.maximum(
        avoidance_strength_ab,
        avoidance_strength_ba,
    )
    avoidance_score = np.where(
        occlusion_avoidance_ab | occlusion_avoidance_ba,
        np.maximum(avoidance_score, 1.0),
        avoidance_score,
    )

    # Escape with repeated turns is the disambiguating signal for avoidance.
    # Suppress chase only inside that reaction state, leaving an earlier chase
    # phase available when the two states genuinely occur in sequence.
    chase_mask = chase_ab_or_ba & ~avoidance_ab_or_ba
    avoidance_mask = avoidance_ab_or_ba
    chase_hold = max(int(round(float(chase_cfg.get("hold_seconds", 0.35)) * analysis_fps)), 1)
    chase_mask = _symmetric_hold(
        chase_mask,
        chase_hold,
        min_fraction=float(chase_cfg.get("hold_min_fraction", 0.25)),
    ) & (valid | chase_occlusion_bridge)
    trailing_occlusion_frames = max(
        int(round(float(chase_cfg.get("occlusion_trailing_hold_seconds", 0.0)) * analysis_fps)),
        0,
    )
    trailing_seed_frames = max(
        int(
            round(float(chase_cfg.get("occlusion_trailing_min_seed_seconds", 0.35)) * analysis_fps)
        ),
        1,
    )
    if trailing_occlusion_frames:
        one_target_occluded = (
            bbox_pair_valid
            & bbox_pair_observed
            & (
                (ab["actor_bbox_observed"] & ab["target_bbox_imputed"])
                | (ba["actor_bbox_observed"] & ba["target_bbox_imputed"])
            )
        )
        # A detector can briefly emit both boxes at the start of an
        # occlusion, while pose direction is already unreliable.  Allow only
        # a tiny, near-pair observed bridge when a genuine one-target
        # occlusion follows immediately.  Requiring the future occlusion keeps
        # this from extending an ordinary fully observed non-chase interval.
        observed_bridge_limit = max(
            int(chase_cfg.get("occlusion_trailing_observed_bridge_frames", 0)),
            0,
        )
        observed_bridge = np.zeros(n, dtype=bool)
        if observed_bridge_limit:
            bbox_center_distance = _numeric_column(
                pair_df,
                "bbox_center_distance_body_lengths",
                np.inf,
            )
            observed_pair_near = (
                bbox_pair_valid
                & bbox_pair_observed
                & np.isfinite(bbox_center_distance)
                & (
                    bbox_center_distance
                    <= float(
                        chase_cfg.get(
                            "occlusion_trailing_max_distance_body_lengths",
                            chase_cfg.get("max_distance_body_lengths", 2.5),
                        )
                    )
                )
            )
            occlusion_ahead = np.zeros(n, dtype=bool)
            for offset in range(1, observed_bridge_limit + 1):
                if offset >= n:
                    break
                occlusion_ahead[:-offset] |= one_target_occluded[offset:]
            observed_bridge = observed_pair_near & occlusion_ahead
        chase_before_trailing_bridge = chase_mask.copy()
        chase_mask = _extend_bouts_through_supported_trailing_gap(
            chase_mask,
            # A one-frame fully observed reacquisition can be inside the held
            # chase state without independently satisfying every directional
            # gate.  Treat that already accepted state as continuity so a
            # following identity hand-off/occlusion does not split the bout.
            one_target_occluded | observed_bridge | chase_mask,
            max_gap_frames=trailing_occlusion_frames,
            min_seed_frames=trailing_seed_frames,
        )
        chase_occlusion_bridge |= (
            chase_mask & ~chase_before_trailing_bridge & (~valid | observed_bridge)
        )
    avoidance_hold = max(
        int(round(float(avoidance_cfg.get("hold_seconds", 0.75)) * analysis_fps)), 1
    )
    avoidance_mask = _symmetric_hold(
        avoidance_mask,
        avoidance_hold,
        min_fraction=float(avoidance_cfg.get("hold_min_fraction", 0.30)),
    ) & (valid | occlusion_avoidance_ab | occlusion_avoidance_ba)
    avoidance_mask |= occlusion_avoidance_ab | occlusion_avoidance_ba
    boundary_context = boundary_context_ab | boundary_context_ba
    if boundary_context.any():
        # If the clip begins in the reaction phase, the visible trigger may
        # cover fewer than one second.  Extend only that already directional
        # boundary state inside valid observations; the shared event finalizer
        # still applies the one-second reliability gate.
        boundary_min_frames = max(
            int(round(float(avoidance_cfg.get("min_duration_seconds", 1.0)) * analysis_fps)),
            1,
        )
        boundary_recovery = _symmetric_hold(
            avoidance_ab_or_ba,
            boundary_min_frames,
            min_fraction=float(avoidance_cfg.get("boundary_hold_min_fraction", 0.02)),
        )
        avoidance_mask |= boundary_recovery & valid

    chase_actor, chase_target = _stabilize_roles(
        chase_mask,
        chase_ab,
        chase_ba,
        chase_strength_ab,
        chase_strength_ba,
        actor_a,
        target_a,
        actor_b,
        target_b,
        evidence_context_frames=chase_role_context_frames,
    )
    # The causal direction says which approach/interaction opened the
    # avoidance state, but it does not always identify the evader after an ID
    # dropout. Choose the displayed evader from the reaction itself: moving
    # away is primary, with relative speed and repeated turning as supporting
    # evidence. Both role candidates are compared across the full event span.
    evader_speed_total = np.maximum(ab["target_speed"], 0.0) + np.maximum(ba["target_speed"], 0.0)
    evader_speed_share_ab = np.divide(
        np.maximum(ab["target_speed"], 0.0),
        np.maximum(evader_speed_total, 1e-6),
    )
    evader_speed_share_ba = np.divide(
        np.maximum(ba["target_speed"], 0.0),
        np.maximum(evader_speed_total, 1e-6),
    )
    avoidance_role_strength_ab = (
        2.0 * np.maximum(ab["escape"], 0.0) + 0.5 * evader_speed_share_ab + 0.5 * turn_support_ab
    )
    avoidance_role_strength_ba = (
        2.0 * np.maximum(ba["escape"], 0.0) + 0.5 * evader_speed_share_ba + 0.5 * turn_support_ba
    )
    # For A->B pursuit, B is the evader and therefore the avoidance actor.
    avoidance_actor, avoidance_target = _stabilize_roles(
        avoidance_mask,
        avoidance_ab,
        avoidance_ba,
        avoidance_role_strength_ab,
        avoidance_role_strength_ba,
        target_a,
        actor_a,
        target_b,
        actor_b,
    )

    attack_cfg = dict(social_config.get("semantic_attack", {}))
    contact = _contact_mask(pair_df, contact_config=contact_config)
    use_bbox_motion = bool(attack_cfg.get("use_bbox_motion", False)) and all(
        column in pair_df.columns
        for column in (
            "bbox_pair_valid",
            "bbox_overlap_iou",
            "bbox_center_distance_body_lengths",
        )
    )
    bbox_contact = _bbox_contact_mask(
        pair_df,
        attack_config=attack_cfg,
    )
    if use_bbox_motion:
        # Box contact is the primary attack anchor for profiles that opt in.
        # Keypoint contact may be retained only as an explicit fallback; it is
        # disabled for Beiyi because occluded nose points are not reliable.
        attack_contact = bbox_contact.copy()
        if bool(attack_cfg.get("bbox_allow_keypoint_contact_fallback", False)):
            attack_contact |= contact
    else:
        attack_contact = contact
    # A short occlusion bridge may have a valid box pair while the normal
    # pose/center pair is invalid.  Keep that bridge local to attack logic;
    # chase, avoidance, contact geometry, and individual behavior continue to
    # require two real pose tracks.
    # ``attack_valid`` deliberately accepts a wider bbox-observed state than
    # the pose-valid pair used by chase/avoidance.  Keep it as an independent
    # array: in-place ``|=`` on an alias would silently promote predicted-box
    # frames to pose evidence and let a short visible pursuit expand into a
    # long false chase bout.
    attack_valid = valid.copy()
    if use_bbox_motion:
        bbox_pair_valid = _numeric_column(pair_df, "bbox_pair_valid").astype(bool)
        bbox_pair_observed = _numeric_column(pair_df, "bbox_pair_observed", 1.0).astype(bool)
        attack_valid |= bbox_pair_valid & bbox_pair_observed
    if not use_bbox_motion and bool(attack_cfg.get("require_nose_head_contact", False)):
        attack_contact = _nose_head_contact_mask(
            pair_df,
            contact_config=contact_config,
        )
    attack_score = np.maximum(
        _numeric_column(enriched, "weak_standard_attack_score"),
        _numeric_column(enriched, "strong_standard_attack_score"),
    )
    dynamic_score = np.maximum(
        _numeric_column(enriched, "weak_standard_dynamic_attack_score"),
        _numeric_column(enriched, "strong_standard_dynamic_attack_score"),
    )
    raw_speed = np.maximum(ab["actor_raw_speed"], ba["actor_raw_speed"])
    target_turn = np.maximum(ab["turn"], ba["turn"])
    deformation = np.maximum.reduce(
        [
            ab["actor_deformation"],
            ab["target_deformation"],
            ba["actor_deformation"],
            ba["target_deformation"],
        ]
    )
    min_raw_speed = float(attack_cfg.get("min_raw_speed_cm_s", 8.0))
    min_target_turn = float(attack_cfg.get("min_target_turn_angle_deg", 35.0))
    min_pose_deformation = float(attack_cfg.get("min_pose_deformation", 0.12))
    min_impact_pursuit = float(attack_cfg.get("min_impact_pursuit_alignment", 0.35))
    min_reaction_escape = float(attack_cfg.get("min_reaction_escape_alignment", 0.35))
    bbox_speed_ab = np.maximum(ab["actor_bbox_speed"], ab["target_bbox_speed"])
    bbox_speed_ba = np.maximum(ba["actor_bbox_speed"], ba["target_bbox_speed"])
    bbox_acceleration_ab = np.maximum(
        ab["actor_bbox_acceleration"],
        ab["target_bbox_acceleration"],
    )
    bbox_acceleration_ba = np.maximum(
        ba["actor_bbox_acceleration"],
        ba["target_bbox_acceleration"],
    )
    bbox_area_change_ab = np.maximum(
        ab["actor_bbox_area_change"],
        ab["target_bbox_area_change"],
    )
    bbox_area_change_ba = np.maximum(
        ba["actor_bbox_area_change"],
        ba["target_bbox_area_change"],
    )
    bbox_jump_ab = np.maximum(ab["actor_bbox_jump"], ab["target_bbox_jump"])
    bbox_jump_ba = np.maximum(ba["actor_bbox_jump"], ba["target_bbox_jump"])
    bbox_dynamic_ab = np.zeros(n, dtype=bool)
    bbox_dynamic_ba = np.zeros(n, dtype=bool)
    if use_bbox_motion:
        min_bbox_speed = max(
            float(attack_cfg.get("min_bbox_speed_body_lengths_per_frame", 0.10)),
            0.0,
        )
        min_bbox_acceleration = max(
            float(attack_cfg.get("min_bbox_acceleration_body_lengths_per_frame2", 0.08)),
            0.0,
        )
        min_bbox_area_change = max(
            float(attack_cfg.get("min_bbox_area_change_ratio", 0.10)),
            0.0,
        )
        min_bbox_jump = max(
            float(attack_cfg.get("min_bbox_jump_score", 0.20)),
            0.0,
        )
        bbox_dynamic_ab = (bbox_speed_ab >= min_bbox_speed) & (
            (bbox_acceleration_ab >= min_bbox_acceleration)
            | (bbox_area_change_ab >= min_bbox_area_change)
            | (bbox_jump_ab >= min_bbox_jump)
        )
        bbox_dynamic_ba = (bbox_speed_ba >= min_bbox_speed) & (
            (bbox_acceleration_ba >= min_bbox_acceleration)
            | (bbox_area_change_ba >= min_bbox_area_change)
            | (bbox_jump_ba >= min_bbox_jump)
        )
    bbox_dynamic = bbox_dynamic_ab | bbox_dynamic_ba
    bbox_contact_dynamic = bbox_contact & bbox_dynamic
    impact_ab = (ab["actor_raw_speed"] >= min_raw_speed) & (ab["pursuit"] >= min_impact_pursuit)
    impact_ba = (ba["actor_raw_speed"] >= min_raw_speed) & (ba["pursuit"] >= min_impact_pursuit)
    reaction_ab = ab["turn"] >= min_target_turn
    reaction_ab &= ab["escape"] >= min_reaction_escape
    reaction_ba = ba["turn"] >= min_target_turn
    reaction_ba &= ba["escape"] >= min_reaction_escape
    deformation_ab = (ab["actor_deformation"] >= min_pose_deformation) & (
        (ab["pursuit"] >= min_impact_pursuit) | (ab["escape"] >= min_reaction_escape)
    )
    deformation_ba = (ba["actor_deformation"] >= min_pose_deformation) & (
        (ba["pursuit"] >= min_impact_pursuit) | (ba["escape"] >= min_reaction_escape)
    )
    # A high speed alone is common in ordinary chase clips.  Beiyi's attack
    # profile can request a stronger impact signature when acceleration and
    # pose channels are available, while old callers without those columns
    # keep the historical speed-based fallback.
    has_action_kinematics = all(
        column in pair_df.columns
        for column in (
            "a_to_b_actor_acceleration_cm_s2",
            "b_to_a_actor_acceleration_cm_s2",
            "a_to_b_actor_nose_speed_cm_s",
            "b_to_a_actor_nose_speed_cm_s",
        )
    )
    has_bbox_action_kinematics = use_bbox_motion and all(
        column in pair_df.columns
        for column in (
            "a_to_b_actor_bbox_speed_body_lengths_per_frame",
            "b_to_a_actor_bbox_speed_body_lengths_per_frame",
            "a_to_b_actor_bbox_acceleration_body_lengths_per_frame2",
            "b_to_a_actor_bbox_acceleration_body_lengths_per_frame2",
        )
    )
    require_action_signature = bool(attack_cfg.get("require_action_signature", False))
    if use_bbox_motion and has_bbox_action_kinematics:
        # During a grapple, the box can move/reshape abruptly even though the
        # nose, ears, or neck are hidden.  Treat repeated box motion at box
        # contact as the primary impact signature.  Keypoint reaction evidence
        # remains additive, never a hard requirement.
        reaction_fraction_window = max(
            int(round(analysis_fps * float(attack_cfg.get("reaction_turn_window_seconds", 0.75)))),
            1,
        )
        reaction_fraction_min = float(attack_cfg.get("min_reaction_turn_fraction", 0.15))
        reaction_signature_ab = reaction_ab & (
            _rolling_fraction(
                reaction_ab,
                reaction_fraction_window,
            )
            >= reaction_fraction_min
        )
        reaction_signature_ba = reaction_ba & (
            _rolling_fraction(
                reaction_ba,
                reaction_fraction_window,
            )
            >= reaction_fraction_min
        )
        impact_evidence = bbox_contact_dynamic | reaction_signature_ab | reaction_signature_ba
    elif require_action_signature and has_action_kinematics:
        min_impact_acceleration = float(attack_cfg.get("min_impact_acceleration_cm_s2", 1500.0))
        lunge_ab = impact_ab & (np.abs(ab["actor_acceleration"]) >= min_impact_acceleration)
        lunge_ba = impact_ba & (np.abs(ba["actor_acceleration"]) >= min_impact_acceleration)
        reaction_fraction_window = max(
            int(round(analysis_fps * float(attack_cfg.get("reaction_turn_window_seconds", 0.75)))),
            1,
        )
        reaction_fraction_min = float(attack_cfg.get("min_reaction_turn_fraction", 0.15))
        reaction_signature_ab = reaction_ab & (
            _rolling_fraction(
                reaction_ab,
                reaction_fraction_window,
            )
            >= reaction_fraction_min
        )
        reaction_signature_ba = reaction_ba & (
            _rolling_fraction(
                reaction_ba,
                reaction_fraction_window,
            )
            >= reaction_fraction_min
        )
        impact_evidence = (
            lunge_ab
            | lunge_ba
            | deformation_ab
            | deformation_ba
            | reaction_signature_ab
            | reaction_signature_ba
        )
    else:
        impact_evidence = impact_ab | impact_ba | deformation_ab | deformation_ba
    reaction_context_frames = max(
        int(round(analysis_fps * float(attack_cfg.get("reaction_context_seconds", 0.75)))),
        1,
    )
    prior_impact = np.zeros(n, dtype=bool)
    if n > 1:
        prior_impact[1:] = _rolling_any(impact_evidence[:-1], reaction_context_frames)
    # A turn or escape is an attack reaction only after a directional impact
    # has been observed.  This prevents an ordinary chase turn from opening
    # an attack event by itself.
    reaction_ab_supported = reaction_ab & (
        prior_impact | impact_ab | deformation_ab | bbox_dynamic_ab
    )
    reaction_ba_supported = reaction_ba & (
        prior_impact | impact_ba | deformation_ba | bbox_dynamic_ba
    )
    directional_dynamic = impact_evidence | reaction_ab_supported | reaction_ba_supported
    # Keep the combined signal for diagnostics and for compatibility with
    # profiles that only supply aggregate pose features.
    dynamic_motion = directional_dynamic | (
        (raw_speed >= min_raw_speed)
        & (target_turn >= min_target_turn)
        & (deformation >= min_pose_deformation)
    )
    if use_bbox_motion:
        dynamic_motion |= bbox_dynamic
    contact_support_window = max(
        int(round(analysis_fps * float(attack_cfg.get("contact_support_seconds", 0.20)))),
        2,
    )
    min_contact_frames = max(int(attack_cfg.get("min_contact_frames", 2)), 1)
    contact_support = (
        rolling_sum(attack_contact.astype(float), contact_support_window) >= min_contact_frames
    )
    contact_support = (
        _symmetric_hold(
            contact_support,
            max(int(round(analysis_fps * 0.10)), 1),
            min_fraction=float(attack_cfg.get("contact_hold_min_fraction", 0.25)),
        )
        & attack_valid
    )
    # Attack is a contact-plus-action state.  Ordinary nose/body contact and
    # detector jitter can also produce high provider scores, so require a
    # short, relative post-contact separation or escape signature.  The
    # look-ahead is only used as bounded context for the lunge frame; it does
    # not change the public event time span.
    separation_threshold = float(attack_cfg.get("min_normalized_distance_increase", 0.50))
    separation_window = max(
        int(round(analysis_fps * float(attack_cfg.get("separation_context_seconds", 0.75)))),
        1,
    )
    finite_valid_distance = distance[valid & np.isfinite(distance)]
    separation_scale = (
        float(np.quantile(finite_valid_distance, 0.75)) if finite_valid_distance.size else 1.0
    )
    normalized_separation = np.maximum(
        (distance - previous_reaction_distance) / max(separation_scale, 1e-6),
        0.0,
    )
    separation_evidence = normalized_separation >= separation_threshold
    separation_support_window = max(
        int(round(analysis_fps * float(attack_cfg.get("separation_support_seconds", 0.30)))),
        1,
    )
    separation_support = rolling_sum(
        separation_evidence.astype(float), separation_support_window
    ) >= max(int(attack_cfg.get("min_separation_frames", 2)), 1)
    prior_contact = np.zeros(n, dtype=bool)
    if n > 1:
        prior_contact[1:] = _rolling_any(attack_contact[:-1], separation_window)
    post_contact_separation = separation_support & prior_contact
    attack_separation_context = _rolling_future_any(
        post_contact_separation,
        separation_window,
    ) | _rolling_any(post_contact_separation, separation_window)
    bbox_trigger_window = max(
        int(round(analysis_fps * float(attack_cfg.get("bbox_motion_support_seconds", 0.30)))),
        1,
    )
    bbox_motion_frames = max(
        int(attack_cfg.get("min_bbox_motion_frames", 2)),
        1,
    )
    bbox_motion_supported = (
        rolling_sum(
            bbox_contact_dynamic.astype(float),
            bbox_trigger_window,
        )
        >= bbox_motion_frames
    )
    score_gate = attack_score >= float(attack_cfg.get("min_attack_score", 0.65))
    dynamic_score_gate = dynamic_score >= float(attack_cfg.get("min_dynamic_score", 0.55))
    if use_bbox_motion:
        # The legacy standard engine scores are pose-derived.  They may be
        # zero exactly when the target is occluded, so repeated box contact
        # motion is an explicit, bounded replacement for those two gates.
        score_gate |= bbox_motion_supported
        dynamic_score_gate |= bbox_motion_supported
    attack_common = contact_support & score_gate & dynamic_score_gate & dynamic_motion
    trigger_window = max(
        int(round(analysis_fps * float(attack_cfg.get("trigger_support_seconds", 0.40)))),
        1,
    )
    min_trigger_frames = max(int(attack_cfg.get("min_trigger_frames", 2)), 1)
    # An impact frame can open an attack bout before the animals visibly
    # separate.  This is needed for short attack clips where the lunge and
    # nose-head contact occupy the first part of the event.  It remains
    # directional and still requires repeated support in the trigger window.
    # Do not let a rolling contact-support frame open an attack bout by
    # itself. The lunge/impact must coincide with direct contact. For the
    # bbox-enabled path this means box overlap/near-contact plus repeated box
    # motion; the keypoint nose channel is intentionally not required during
    # occlusion.
    attack_action_trigger = attack_common & attack_contact & impact_evidence
    if use_bbox_motion:
        # A single overlap/motion coincidence is still compatible with
        # detector jitter or an ID hand-off. Require repeated box evidence in
        # the short support window before opening an attack bout.
        attack_action_trigger &= bbox_motion_supported
    action_count = rolling_sum(
        attack_action_trigger.astype(float),
        trigger_window,
    )
    action_supported = attack_action_trigger & (action_count >= min_trigger_frames)
    action_context = _rolling_future_any(
        action_supported,
        max(
            int(round(analysis_fps * float(attack_cfg.get("bout_context_seconds", 0.75)))),
            1,
        ),
    ) | _rolling_any(
        action_supported,
        max(
            int(round(analysis_fps * float(attack_cfg.get("bout_context_seconds", 0.75)))),
            1,
        ),
    )
    # Separation remains a second, causal route.  It is useful when the
    # action is visible primarily in the evader's reaction, but it must still
    # follow contact rather than creating an attack from a distant chase.
    attack_trigger = (
        attack_common
        & attack_contact
        & (action_context | (attack_separation_context & directional_dynamic))
    )
    trigger_count = rolling_sum(attack_trigger.astype(float), trigger_window)
    trigger_supported = attack_trigger & (trigger_count >= min_trigger_frames)
    bout_context = max(
        int(round(analysis_fps * float(attack_cfg.get("bout_context_seconds", 0.75)))),
        1,
    )
    attack_context = _rolling_future_any(
        trigger_supported,
        bout_context,
    ) | _rolling_any(trigger_supported, bout_context)
    # A supported trigger opens a bounded *direct-contact* bout.  The rolling
    # ``contact_support`` signal is intentionally used above to tolerate a
    # missed detector frame while deciding whether a trigger exists, but it is
    # too broad to be the final attack mask: during an ordinary chase it can
    # turn several nearby nose-head samples into a long false attack.  Keep
    # the public event anchored to the direct contact channel and use only the
    # bounded temporal context to recover a short, real attack bout.
    attack_contact_for_mask = attack_contact
    if use_bbox_motion:
        contact_gap_frames = max(
            int(round(analysis_fps * float(attack_cfg.get("bbox_contact_gap_seconds", 0.20)))),
            0,
        )
        if contact_gap_frames:
            attack_contact_for_mask = _symmetric_hold(
                attack_contact,
                contact_gap_frames,
                min_fraction=float(attack_cfg.get("bbox_contact_gap_min_fraction", 0.20)),
            )
    attack_mask = attack_contact_for_mask & attack_context
    (
        bbox_translation_chase,
        bbox_translation_ab,
        bbox_translation_ba,
        bbox_translation_score,
    ) = _bbox_coherent_translation_bouts(
        pair_df,
        attack_mask,
        fps=analysis_fps,
        attack_config=attack_cfg,
    )
    if bbox_translation_chase.any():
        # Prefer directional pursuit immediately preceding contact. This is
        # causal role evidence when the two boxes later cross during an
        # occlusion; a late crossing must not rewrite who initiated pursuit.
        try:
            role_lookback_seconds = float(
                attack_cfg.get(
                    "coherent_translation_prior_chase_seconds",
                    chase_role_context_seconds or 1.0,
                )
            )
        except (TypeError, ValueError, OverflowError):
            role_lookback_seconds = chase_role_context_seconds or 1.0
        if not np.isfinite(role_lookback_seconds):
            role_lookback_seconds = chase_role_context_seconds or 1.0
        role_lookback_frames = max(
            int(round(max(role_lookback_seconds, 0.0) * analysis_fps)),
            1,
        )
        translation_starts, translation_ends = _boolean_spans(bbox_translation_chase)
        for translation_start, translation_end in zip(translation_starts, translation_ends):
            start, end = int(translation_start), int(translation_end)
            context_start = max(0, start - role_lookback_frames)
            context = directional_chase[context_start:start]
            if not context.any():
                continue
            ab_context = chase_strength_ab[context_start:start][context]
            ba_context = chase_strength_ba[context_start:start][context]
            ab_total = float(np.nanmean(ab_context)) if ab_context.size else float("-inf")
            ba_total = float(np.nanmean(ba_context)) if ba_context.size else float("-inf")
            bbox_translation_ab[start : end + 1] = False
            bbox_translation_ba[start : end + 1] = False
            if ab_total >= ba_total:
                bbox_translation_ab[start : end + 1] = bbox_translation_chase[start : end + 1]
            else:
                bbox_translation_ba[start : end + 1] = bbox_translation_chase[start : end + 1]

        # A coherent whole-pair translation is pursuit evidence, not a local
        # grapple. It enters the ordinary chase FSM and is removed only from
        # the overlapping attack candidate; all duration gates remain active.
        attack_mask &= ~bbox_translation_chase
        chase_mask |= bbox_translation_chase
        chase_ab |= bbox_translation_ab
        chase_ba |= bbox_translation_ba
        chase_strength_ab = np.maximum(
            chase_strength_ab,
            bbox_translation_score * bbox_translation_ab.astype(float),
        )
        chase_strength_ba = np.maximum(
            chase_strength_ba,
            bbox_translation_score * bbox_translation_ba.astype(float),
        )
        chase_score = np.maximum(chase_score, bbox_translation_score)
        chase_occlusion_bridge |= bbox_translation_chase & ~valid

    # A confirmed grapple is an episode, not a collection of unrelated
    # overlap frames. During riding/wrestling the two boxes can separate for
    # a short reaction, merge into one detector box, or briefly lose the
    # direct-IoU contact used by the conservative trigger. Extend only from a
    # real attack seed, only while the same tracked pair remains locally
    # close, and only for a bounded interval. This cannot create an attack
    # from ordinary proximity because an empty seed stays empty.
    attack_state_bridge = np.zeros(n, dtype=bool)
    attack_state_hold_frames = max(
        int(round(float(attack_cfg.get("state_hold_seconds", 0.0)) * analysis_fps)),
        0,
    )
    attack_state_pre_hold_frames = max(
        int(round(float(attack_cfg.get("state_pre_hold_seconds", 0.0)) * analysis_fps)),
        0,
    )
    if use_bbox_motion and (attack_state_hold_frames or attack_state_pre_hold_frames):
        attack_seed_frames = max(
            int(round(float(attack_cfg.get("state_min_seed_seconds", 0.50)) * analysis_fps)),
            1,
        )
        attack_state_distance = max(
            float(attack_cfg.get("state_max_distance_body_lengths", 1.50)),
            0.0,
        )
        bbox_pair_valid = _numeric_column(pair_df, "bbox_pair_valid").astype(bool)
        bbox_pair_observed = _numeric_column(
            pair_df,
            "bbox_pair_observed",
            1.0,
        ).astype(bool)
        bbox_pair_distance = _numeric_column(
            pair_df,
            "bbox_center_distance_body_lengths",
            np.inf,
        )
        episode_support = (
            bbox_pair_valid
            & bbox_pair_observed
            & np.isfinite(bbox_pair_distance)
            & (bbox_pair_distance <= attack_state_distance)
        )
        attack_seed = attack_mask.copy()
        episode_support |= attack_seed
        recovered = attack_seed.copy()
        if attack_state_pre_hold_frames:
            recovered |= _extend_bouts_through_supported_trailing_gap(
                attack_seed[::-1],
                episode_support[::-1],
                max_gap_frames=attack_state_pre_hold_frames,
                min_seed_frames=attack_seed_frames,
            )[::-1]
        if attack_state_hold_frames:
            recovered |= _extend_bouts_through_supported_trailing_gap(
                attack_seed,
                episode_support,
                max_gap_frames=attack_state_hold_frames,
                min_seed_frames=attack_seed_frames,
            )
        attack_mask = recovered
        attack_state_bridge = attack_mask & ~attack_seed

    # Keep the parallel FSM evidence channels independent here.  ``attack_mask``
    # is still provisional: the temporal reliability gate or the group layer
    # can later reject it (for example, detector jitter inside a stable
    # huddle).  Destructively removing chase/avoidance at this point loses the
    # valid fallback behavior if that provisional attack is rejected.  The
    # final event renderer already resolves simultaneous confirmed social
    # events by semantic priority, so a surviving attack remains the displayed
    # label without erasing the underlying chase/avoidance evidence.
    attack_direction_ab = impact_ab | reaction_ab_supported | deformation_ab | bbox_dynamic_ab
    attack_direction_ba = impact_ba | reaction_ba_supported | deformation_ba | bbox_dynamic_ba
    attack_strength_ab = (
        np.maximum(ab["pursuit"], 0.0)
        + np.maximum(ab["escape"], 0.0)
        + np.maximum(ab["actor_deformation"], 0.0)
    )
    attack_strength_ba = (
        np.maximum(ba["pursuit"], 0.0)
        + np.maximum(ba["escape"], 0.0)
        + np.maximum(ba["actor_deformation"], 0.0)
    )
    attack_actor, attack_target = _stabilize_roles(
        attack_mask,
        attack_direction_ab,
        attack_direction_ba,
        attack_strength_ab,
        attack_strength_ba,
        actor_a,
        target_a,
        actor_b,
        target_b,
    )

    # Approach is the directed transition into an interaction. It must end
    # when a causal reaction/attack state is confirmed, but ordinary contact
    # does not erase the already observed approach. A mostly stationary target
    # can produce a few noisy >1 cm/s samples; when that creates a simultaneous
    # chase candidate, the directed approach state wins. Genuine chase remains
    # available because its target-motion gate excludes mostly stationary
    # targets.
    approach_mask = approach_hold & ~(avoidance_mask | attack_mask)
    approach_contact_hold_frames = max(
        int(
            round(float(approach_cfg.get("post_arrival_contact_hold_seconds", 0.0)) * analysis_fps)
        ),
        0,
    )
    if approach_contact_hold_frames:
        approach_contact_seed_frames = max(
            int(
                round(float(approach_cfg.get("post_arrival_min_seed_seconds", 0.50)) * analysis_fps)
            ),
            1,
        )
        # Once a directed approach has been confirmed, retain that transition
        # while the same pair remains in uninterrupted arrival contact.  This
        # does not create approach from contact alone, and a reaction/attack
        # state always terminates the continuation.
        approach_mask = _extend_bouts_through_supported_trailing_gap(
            approach_mask,
            approach_mask | contact,
            max_gap_frames=approach_contact_hold_frames,
            min_seed_frames=approach_contact_seed_frames,
        )
        approach_mask &= ~(avoidance_mask | attack_mask)
    if bool(approach_cfg.get("suppress_during_contact", True)):
        approach_mask &= ~(contact | bbox_contact)
    chase_mask &= ~approach_mask
    chase_mask = _retain_bouts_with_minimum_support(
        chase_mask,
        directional_chase | bbox_translation_chase,
        minimum_fraction=float(chase_cfg.get("min_directional_evidence_fraction_per_bout", 0.0)),
    )
    chase_mask = _retain_bouts_with_minimum_support(
        chase_mask,
        valid,
        minimum_fraction=float(chase_cfg.get("min_pose_valid_fraction_per_bout", 0.0)),
    )
    chase_ab &= chase_mask
    chase_ba &= chase_mask
    chase_occlusion_bridge &= chase_mask
    bbox_translation_chase &= chase_mask
    chase_actor, chase_target = _stabilize_roles(
        chase_mask,
        chase_ab,
        chase_ba,
        chase_strength_ab,
        chase_strength_ba,
        actor_a,
        target_a,
        actor_b,
        target_b,
        evidence_context_frames=chase_role_context_frames,
    )
    approach_actor, approach_target = _stabilize_roles(
        approach_mask,
        approach_direction["a_to_b"],
        approach_direction["b_to_a"],
        approach_strength_ab,
        approach_strength_ba,
        actor_a,
        target_a,
        actor_b,
        target_b,
    )

    return {
        "approach_mask": approach_mask,
        "approach_score": approach_score,
        "approach_actor": approach_actor,
        "approach_target": approach_target,
        "chase_mask": chase_mask,
        "chase_score": chase_score,
        "chase_actor": chase_actor,
        "chase_target": chase_target,
        "chase_occlusion_bridge_mask": chase_occlusion_bridge,
        "chase_bbox_translation_mask": bbox_translation_chase,
        "avoidance_mask": avoidance_mask,
        "avoidance_score": avoidance_score,
        "avoidance_actor": avoidance_actor,
        "avoidance_target": avoidance_target,
        "attack_mask": attack_mask,
        "attack_score": attack_score,
        "attack_actor": attack_actor,
        "attack_target": attack_target,
        "attack_state_bridge_mask": attack_state_bridge,
        "contact_mask": contact,
        "bbox_contact_mask": bbox_contact,
    }


def event_pair_ids(event: Mapping[str, Any]) -> set[int]:
    """Extract all IDs from a normal or identity-bridged pair key."""

    def numeric_ids(value: Any) -> set[int]:
        if isinstance(value, (list, tuple, set)):
            result: set[int] = set()
            for item in value:
                try:
                    number = int(item)
                except (TypeError, ValueError, OverflowError):
                    continue
                if number >= 0:
                    result.add(number)
            return result
        result = set()
        for raw in re.findall(r"-?\d+", str(value)):
            try:
                number = int(raw)
            except (TypeError, ValueError, OverflowError):
                continue
            if number >= 0:
                result.add(number)
        return result

    values: set[int] = set()
    values.update(numeric_ids(event.get("pair_key", "")))
    for field in ("actor_id", "target_id", "participant_ids", "member_ids"):
        raw_value = event.get(field)
        if raw_value is not None:
            values.update(numeric_ids(raw_value))
    return values


__all__ = ["build_semantic_pair_signals", "event_pair_ids"]
