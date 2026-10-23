"""Measured pose directions and conservative visual actor/target role buffers.

Legacy ``heading_unit`` can contain a velocity fallback. This module deliberately
never reads it. Vectors use the actual finite neck/nose/tail-root observations;
neither missing points nor identities are reconstructed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .exported_dataset_models import ExportedFeatures
from .screening_signals import runs
from .social_heuristic_config import SocialHeuristicConfig


def unit(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, float)
    norm = np.linalg.norm(values, axis=-1, keepdims=True)
    return np.divide(values, norm, out=np.full_like(values, np.nan), where=norm > 1e-9)


@dataclass(frozen=True)
class ObservedOrientations:
    head_unit: np.ndarray
    body_unit: np.ndarray


def observed_orientations(features: ExportedFeatures) -> ObservedOrientations:
    """Head = neck3 -> nose0; body = tailroot6 -> neck3; no fallback."""
    shape = (*features.valid.shape, 2)
    head, body = np.full(shape, np.nan), np.full(shape, np.nan)
    points = features.collective_keypoints_cm
    if points is not None:
        points = np.asarray(points, float)
        if points.shape != (*features.valid.shape, 7, 2):
            raise ValueError("directed roles require observed keypoints shaped (frames,tracks,7,2)")
        head = unit(points[:, :, 0] - points[:, :, 3])
        body = unit(points[:, :, 3] - points[:, :, 6])
        head[~features.valid] = np.nan
        body[~features.valid] = np.nan
    return ObservedOrientations(head, body)


def pose_direction_support(
    features: ExportedFeatures, actor: int, target: int,
    config: SocialHeuristicConfig, behavior: str,
    *, orientation: ObservedOrientations | None = None,
    turn_path: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (known, positive) for actual head and optional measured body.

    Missing head geometry is unknown. An observed body contradiction cannot be
    hidden by missing head geometry. Turn support compares the measured pose
    against the follower's measured path direction, never substitutes velocity
    for a missing pose vector.
    """
    orientation = orientation or observed_orientations(features)
    head, body = orientation.head_unit[:, actor], orientation.body_unit[:, actor]
    head_known, body_known = np.isfinite(head).all(axis=1), np.isfinite(body).all(axis=1)
    offset = unit(features.centers_cm[:, target] - features.centers_cm[:, actor])
    prefix = "approach" if behavior == "approach" else "chasing"
    head_cap = getattr(config, prefix + "_head_facing_max_deg")
    body_cap = getattr(config, prefix + "_body_facing_max_deg")
    head_cos = np.sum(head * offset, axis=1)
    body_cos = np.sum(body * offset, axis=1)
    if turn_path is not None:
        path = unit(features.velocity_cm_s[:, actor])
        head_cos = np.where(turn_path, np.fmax(head_cos, np.sum(head * path, axis=1)), head_cos)
        body_cos = np.where(turn_path, np.fmax(body_cos, np.sum(body * path, axis=1)), body_cos)
    head_ok = head_known & (head_cos >= np.cos(np.deg2rad(head_cap)))
    body_ok = ~body_known | (body_cos >= np.cos(np.deg2rad(body_cap)))
    # A body-only observation can contradict the actor hypothesis, but cannot
    # establish a measured head direction or a directed role.
    positive = head_ok & body_ok
    known = head_known | (body_known & ~body_ok)
    return known, positive


def bridge_pose_turns(candidate: np.ndarray, motion_support: np.ndarray, fps: float,
                      maximum_s: float) -> np.ndarray:
    """Keep bounded observed pose changes inside an established moving event."""
    result = np.asarray(candidate, bool).copy()
    maximum = int(np.floor(fps * maximum_s + 1e-9))
    bounds = runs(result)
    for (_, end), (start, _) in zip(bounds, bounds[1:]):
        if start - end <= maximum and motion_support[end:start].all():
            result[end:start] = True
    return result


def new_role_buffers(shape: tuple[int, int]) -> dict[str, np.ndarray]:
    return {behavior + "_" + role: np.full(shape, -1, dtype=np.int64)
            for behavior in ("approach", "chasing") for role in ("actor", "target")}


def assign_event_roles(
    selected: np.ndarray, directed: np.ndarray, measured_facing: np.ndarray,
    track_ids: np.ndarray, left: int, right: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Publish unique measured support from independently qualified directions.

    Candidate masks already enforce the behavior's sustained motion/pose rules.
    Preserve each qualified span so canonical events can split at a real role
    change. Ambiguous or missing rows stay unknown, and a short head turn cannot
    exchange the role of an entire sustained event.
    """
    actor_ids, target_ids = np.full(len(selected), -1, np.int64), np.full(len(selected), -1, np.int64)
    support = directed & measured_facing & np.asarray(selected, bool)[None, :]
    unique = support.sum(axis=0) == 1
    for direction, (actor, target) in enumerate(((left, right), (right, left))):
        rows = unique & support[direction]
        actor_ids[rows], target_ids[rows] = int(track_ids[actor]), int(track_ids[target])
    return actor_ids, target_ids
