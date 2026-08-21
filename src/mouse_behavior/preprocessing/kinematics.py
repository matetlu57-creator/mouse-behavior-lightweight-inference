"""Per-track smoothing and motion/pose kinematics."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from .constants import (
    KEYPOINTS,
    KP_LEFT_EAR,
    KP_LEFT_HIP,
    KP_NECK,
    KP_NOSE,
    KP_RIGHT_EAR,
    KP_RIGHT_HIP,
    KP_TAIL,
)
from .geometry import _angle_deg, _unit


def _ema_smooth(values: np.ndarray, valid: np.ndarray, alpha: float = 0.70) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    out = np.full_like(values, np.nan, dtype=float)
    tracks = values.shape[1]
    previous = np.full(values.shape[1:], np.nan, dtype=float)
    for frame in range(values.shape[0]):
        for track in range(tracks):
            if not bool(valid[frame, track]):
                continue
            current = values[frame, track]
            if not np.all(np.isfinite(current)):
                continue
            if np.all(np.isfinite(previous[track])):
                previous[track] = alpha * current + (1.0 - alpha) * previous[track]
            else:
                previous[track] = current
            out[frame, track] = previous[track]
    return out


def _ema_smooth_keypoints(values: np.ndarray, valid: np.ndarray, alpha: float = 0.70) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    out = np.full_like(values, np.nan, dtype=float)
    tracks = values.shape[1]
    previous = np.full(values.shape[1:], np.nan, dtype=float)
    for frame in range(values.shape[0]):
        for track in range(tracks):
            if not bool(valid[frame, track]):
                continue
            for point in range(values.shape[2]):
                current = values[frame, track, point]
                if not np.all(np.isfinite(current)):
                    continue
                if np.all(np.isfinite(previous[track, point])):
                    previous[track, point] = (
                        alpha * current + (1.0 - alpha) * previous[track, point]
                    )
                else:
                    previous[track, point] = current
                out[frame, track, point] = previous[track, point]
    return out


def _pose_deformation_energy(
    keypoints_cm: np.ndarray,
    centers_cm: np.ndarray,
    heading: np.ndarray,
    body_cm: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """Measure frame-to-frame shape change in each mouse's body frame.

    Translation, global rotation and apparent scale are removed before the
    comparison.  Ordinary nose contact should therefore remain close to zero
    when the two mice merely hold a stable posture, while a bite/lunge or
    wrestling motion can contribute a real internal pose change.  The
    lightweight path previously exported zeros here, which disabled the
    standard engine's anti-contact grapple evidence entirely.
    """
    keypoints_cm = np.asarray(keypoints_cm, dtype=float)
    centers_cm = np.asarray(centers_cm, dtype=float)
    heading = np.asarray(heading, dtype=float)
    body_cm = np.asarray(body_cm, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    frames, mice = valid.shape
    body_pose = np.full((frames, mice, KEYPOINTS, 2), np.nan, dtype=float)
    pose_valid = np.zeros((frames, mice, KEYPOINTS), dtype=bool)

    for frame in range(frames):
        for mouse in range(mice):
            if not valid[frame, mouse]:
                continue
            center = centers_cm[frame, mouse]
            forward = heading[frame, mouse]
            scale = body_cm[frame, mouse]
            points = keypoints_cm[frame, mouse]
            if (
                not np.all(np.isfinite(center))
                or not np.all(np.isfinite(forward))
                or not np.isfinite(scale)
                or scale < 1e-3
            ):
                continue
            lateral = np.asarray([-forward[1], forward[0]], dtype=float)
            relative = (points - center) / scale
            good = np.all(np.isfinite(relative), axis=1)
            if not np.any(good):
                continue
            body_pose[frame, mouse, good, 0] = relative[good] @ forward
            body_pose[frame, mouse, good, 1] = relative[good] @ lateral
            pose_valid[frame, mouse, good] = True

    deformation = np.zeros((frames, mice), dtype=float)
    for frame in range(1, frames):
        common = pose_valid[frame] & pose_valid[frame - 1]
        for mouse in np.flatnonzero(np.sum(common, axis=1) >= 3):
            delta = (
                body_pose[frame, mouse, common[mouse]] - body_pose[frame - 1, mouse, common[mouse]]
            )
            deformation[frame, mouse] = float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))
    deformation[~valid] = 0.0
    return deformation


def _kinematics(
    tracks: Mapping[str, np.ndarray], fps: float, body_length_cm: float = 8.0
) -> dict[str, np.ndarray | float]:
    valid = np.asarray(tracks["valid"], dtype=bool)
    pose_quality = np.asarray(tracks["pose_quality"], dtype=float)
    raw_kp = np.asarray(tracks["keypoints_px"], dtype=float)
    raw_centers = np.asarray(tracks["centers_px"], dtype=float)
    body_values = np.asarray(tracks["body_lengths_px"], dtype=float)
    body_values = body_values[
        np.isfinite(body_values) & (body_values >= 10.0) & (body_values <= 300.0)
    ]
    reference_body_px = float(np.median(body_values)) if body_values.size else 60.0
    cm_per_pixel = float(body_length_cm / max(reference_body_px, 1e-6))
    smooth_kp = _ema_smooth_keypoints(raw_kp, valid, alpha=0.70)
    smooth_centers = _ema_smooth(raw_centers, valid, alpha=0.70)
    frames, mice = valid.shape
    keypoints_cm = smooth_kp * cm_per_pixel
    centers_cm = smooth_centers * cm_per_pixel
    head_cm = np.full((frames, mice, 2), np.nan, dtype=float)
    heading = np.full((frames, mice, 2), np.nan, dtype=float)
    body_cm = np.full((frames, mice), np.nan, dtype=float)
    for frame in range(frames):
        for mouse in range(mice):
            if not valid[frame, mouse]:
                continue
            points = keypoints_cm[frame, mouse]
            head_cm[frame, mouse] = np.nanmean(points[[KP_NOSE, KP_LEFT_EAR, KP_RIGHT_EAR]], axis=0)
            center = np.nanmean(points[[KP_NECK, KP_LEFT_HIP, KP_RIGHT_HIP]], axis=0)
            if np.all(np.isfinite(center)):
                centers_cm[frame, mouse] = center
            axis = points[KP_NOSE] - points[KP_NECK]
            if np.all(np.isfinite(axis)):
                heading[frame, mouse] = _unit(axis)
            if np.all(np.isfinite(points[[KP_NOSE, KP_TAIL]])):
                body_cm[frame, mouse] = float(np.linalg.norm(points[KP_NOSE] - points[KP_TAIL]))

    velocity = np.zeros((frames, mice, 2), dtype=float)
    nose_velocity = np.zeros((frames, mice, 2), dtype=float)
    acceleration = np.zeros((frames, mice), dtype=float)
    angular_speed = np.zeros((frames, mice), dtype=float)
    last_valid = np.full(mice, -1, dtype=int)
    for frame in range(frames):
        for mouse in range(mice):
            if not valid[frame, mouse]:
                continue
            previous = int(last_valid[mouse])
            if (
                previous >= 0
                and frame - previous <= 3
                and np.all(np.isfinite(centers_cm[[previous, frame], mouse]))
            ):
                dt = max(frame - previous, 1) / fps
                velocity[frame, mouse] = (
                    centers_cm[frame, mouse] - centers_cm[previous, mouse]
                ) / dt
                if np.all(np.isfinite(keypoints_cm[[previous, frame], mouse, KP_NOSE])):
                    nose_velocity[frame, mouse] = (
                        keypoints_cm[frame, mouse, KP_NOSE] - keypoints_cm[previous, mouse, KP_NOSE]
                    ) / dt
                if previous > 0:
                    acceleration[frame, mouse] = (
                        abs(
                            np.linalg.norm(velocity[frame, mouse])
                            - np.linalg.norm(velocity[previous, mouse])
                        )
                        / dt
                    )
                if np.all(np.isfinite(heading[[previous, frame], mouse])):
                    angular_speed[frame, mouse] = (
                        float(_angle_deg(heading[frame, mouse], heading[previous, mouse])) / dt
                    )
            last_valid[mouse] = frame
    speed = np.linalg.norm(velocity, axis=2)
    nose_speed = np.linalg.norm(nose_velocity, axis=2)
    # The original speed is intentionally kept for chase/attack compatibility.
    # Individual labels use a longer-baseline displacement speed, followed by a
    # rolling median, so one identity-assignment jump cannot turn stationary
    # into running and a one-frame detector jitter cannot turn approach into
    # contact.
    behavior_speed_raw = np.full((frames, mice), np.nan, dtype=float)
    behavior_lookback = max(int(round(fps * 0.15)), 1)
    if behavior_lookback < frames:
        for mouse in range(mice):
            current = centers_cm[behavior_lookback:, mouse]
            previous = centers_cm[:-behavior_lookback, mouse]
            valid_motion = (
                valid[behavior_lookback:, mouse]
                & valid[:-behavior_lookback, mouse]
                & np.all(np.isfinite(current), axis=1)
                & np.all(np.isfinite(previous), axis=1)
            )
            if np.any(valid_motion):
                target_values = behavior_speed_raw[behavior_lookback:, mouse]
                target_values[valid_motion] = np.linalg.norm(
                    current[valid_motion] - previous[valid_motion], axis=1
                ) / (behavior_lookback / max(fps, 1e-9))
    behavior_speed = np.zeros((frames, mice), dtype=float)
    behavior_window = max(int(round(fps * 0.30)), 3)
    for mouse in range(mice):
        robust = _rolling_quantile(behavior_speed_raw[:, mouse], behavior_window, 0.50)
        behavior_speed[:, mouse] = np.nan_to_num(robust, nan=0.0, posinf=0.0, neginf=0.0)
    pose_deformation = _pose_deformation_energy(
        keypoints_cm,
        centers_cm,
        heading,
        body_cm,
        valid,
    )
    velocity[~valid] = 0.0
    speed[~valid] = 0.0
    acceleration[~valid] = 0.0
    angular_speed[~valid] = 0.0
    nose_speed[~valid] = 0.0
    behavior_speed[~valid] = 0.0
    return {
        "valid": valid,
        "pose_quality": pose_quality,
        "keypoints_cm": keypoints_cm,
        "centers_cm": centers_cm,
        "head_cm": head_cm,
        "heading": heading,
        "body_cm": body_cm,
        "velocity": velocity,
        "speed": speed,
        "behavior_speed": behavior_speed,
        "nose_speed": nose_speed,
        "acceleration": acceleration,
        "angular_speed": angular_speed,
        "pose_deformation": pose_deformation,
        "cm_per_pixel": cm_per_pixel,
        "reference_body_px": reference_body_px,
        "reference_body_cm": float(body_length_cm),
    }


def _rolling_quantile(values: np.ndarray, window: int, quantile: float) -> np.ndarray:
    """Return a causal rolling quantile without introducing a new dependency."""
    values = np.asarray(values, dtype=float)
    window = max(int(window), 1)
    quantile = float(np.clip(quantile, 0.0, 1.0))
    result = np.full(values.shape, np.nan, dtype=float)
    for index in range(len(values)):
        start = max(0, index - window + 1)
        sample = values[start : index + 1]
        sample = sample[np.isfinite(sample)]
        if sample.size:
            result[index] = float(np.quantile(sample, quantile))
    return result
