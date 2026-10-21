"""Shared spatial and temporal bounds for Static bouts."""

from __future__ import annotations

import math

import numpy as np


def static_posture_eligible_mask(
    keypoints_cm: np.ndarray,
    reference_body_cm: float | np.ndarray,
    *,
    min_projected_body_length_ratio: float = 0.55,
    allow_shortened_when_stationary: bool = True,
    stationary_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Return projection-length evidence for callers that explicitly need it.

    The primary individual classifier does not use this mask to veto Static or
    Locomotion. A short projection is not sufficient evidence to label a mouse
    as rearing, and must not erase bounded body-center stationarity. It stays out
    of that veto path here too; this function serves posture-class callers and
    artifact suppression.

    A curled mouse shortens its 2D nose-to-tail projection well below
    ``min_projected_body_length_ratio`` without standing, and a paw over the nose
    or a raised head can drop the head/tail keypoints entirely. Both used to make
    the frame ineligible even though the body centre never left the static bound.
    When ``allow_shortened_when_stationary`` is set and ``stationary_mask`` is
    supplied, any frame already established as inside that bound stays eligible:
    bounded body-center stationarity is stronger evidence about the frame than a
    shortened or partially missing projection is about posture.
    """

    points = np.asarray(keypoints_cm, dtype=float)
    if points.ndim != 4 or points.shape[-1] != 2 or points.shape[2] < 7:
        raise ValueError("keypoints_cm must have shape (frames, mice, >=7, 2)")
    ratio_limit = float(min_projected_body_length_ratio)
    if not math.isfinite(ratio_limit) or not 0.0 <= ratio_limit <= 1.0:
        raise ValueError("min_projected_body_length_ratio must be finite and in [0, 1]")
    reference = np.asarray(reference_body_cm, dtype=float)
    if reference.ndim == 1 and reference.shape == (points.shape[1],):
        reference = reference[None, :]
    try:
        reference = np.broadcast_to(reference, points.shape[:2])
    except ValueError as exc:
        raise ValueError("reference_body_cm must be scalar or broadcastable to (frames, mice)") from exc
    if np.any(np.isfinite(reference) & (reference <= 0.0)):
        raise ValueError("finite reference_body_cm values must be positive")

    projected_length = np.linalg.norm(points[:, :, 0] - points[:, :, 6], axis=2)
    eligible = (
        np.isfinite(projected_length)
        & np.isfinite(reference)
        & (reference > 0.0)
        & (projected_length > ratio_limit * reference)
    )
    if stationary_mask is not None:
        stationary = np.asarray(stationary_mask, dtype=bool)
        if stationary.shape != points.shape[:2]:
            raise ValueError(
                "stationary_mask must have shape (frames, mice) matching keypoints_cm"
            )
        if allow_shortened_when_stationary:
            eligible = eligible | stationary
    return eligible


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    changes = np.diff(np.r_[False, mask, False].astype(np.int8))
    return list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))


def bridge_gaps_outside_static(
    moving: np.ndarray,
    stationary: np.ndarray,
    *,
    max_gap_frames: int,
) -> np.ndarray:
    """Recover short movement gaps without overlapping a Static observation."""

    result = np.asarray(moving, dtype=bool).copy()
    static = np.asarray(stationary, dtype=bool)
    if result.ndim != 2 or static.shape != result.shape:
        raise ValueError("moving and stationary must have matching (frames, mice) shapes")
    gap_limit = max(int(max_gap_frames), 0)
    for mouse in range(result.shape[1]):
        segments = _runs(result[:, mouse])
        for (_, left_end), (right_start, _) in zip(segments, segments[1:]):
            if (0 < right_start - left_end <= gap_limit
                    and not static[left_end:right_start, mouse].any()):
                result[left_end:right_start, mouse] = True
    return result


def bounded_stationary_mask(
    stationary: np.ndarray,
    centers_cm: np.ndarray,
    valid: np.ndarray,
    *,
    fps: float,
    max_gap_s: float,
    max_missing_gap_s: float | None = None,
    max_transient_displacement_cm: float,
    max_center_radius_cm: float,
    pose_eligible: np.ndarray | None = None,
    bridged_out: list[tuple[int, int, int]] | None = None,
    max_bridged_missing_fraction: float | None = None,
) -> np.ndarray:
    """Bridge brief body-center excursions only inside a bounded Static bout.

    The input mask is the adapter's low-speed evidence. A gap can be included
    Observed transient gaps use ``max_gap_s``. Detection gaps may use the
    separate ``max_missing_gap_s`` limit; only observed anchor positions are
    checked against the displacement and radius bounds. Unstable bouts are
    left out of Static so callers can retain them as locomotion or Unknown.

    ``max_bridged_missing_fraction`` bounds the CUMULATIVE unobserved time inside
    one accepted bout as a fraction of that bout's length. A per-gap limit alone
    still lets many short dropouts chain into a bout that is mostly fabricated, so
    ``None`` (the default, which keeps every existing caller unchanged) means no
    cumulative bound and a value in [0, 1] requires the bout to keep at least that
    much genuinely detected evidence.

    ``bridged_out`` optionally collects ``(mouse, gap_start, gap_end)`` for every
    gap this function actually filled inside an accepted bout. Callers use it as
    the authoritative allow-list of frames that may be reported Static while the
    detector did not report the animal, so the emitted mask can be checked
    against the gaps that were genuinely bridged under the bounds above.
    """

    base = np.asarray(stationary, dtype=bool).copy()
    centers = np.asarray(centers_cm, dtype=float)
    observed = np.asarray(valid, dtype=bool)
    if base.ndim != 2:
        raise ValueError("stationary must have shape (frames, mice)")
    if centers.shape != (*base.shape, 2):
        raise ValueError("centers_cm must have shape (frames, mice, 2)")
    if observed.shape != base.shape:
        raise ValueError("valid must have the same shape as stationary")
    if pose_eligible is not None:
        pose = np.asarray(pose_eligible, dtype=bool)
        if pose.shape != base.shape:
            raise ValueError("pose_eligible must have the same shape as stationary")
        observed = observed & pose
    rate = float(fps)
    gap_limit = float(max_gap_s)
    missing_gap_limit = gap_limit if max_missing_gap_s is None else float(max_missing_gap_s)
    excursion_limit = float(max_transient_displacement_cm)
    radius_limit = float(max_center_radius_cm)
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("fps must be finite and positive")
    for name, value in (
        ("max_gap_s", gap_limit),
        ("max_missing_gap_s", missing_gap_limit),
        ("max_transient_displacement_cm", excursion_limit),
        ("max_center_radius_cm", radius_limit),
    ):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    missing_fraction = max_bridged_missing_fraction
    if missing_fraction is not None:
        missing_fraction = float(missing_fraction)
        if not math.isfinite(missing_fraction) or not 0.0 <= missing_fraction <= 1.0:
            raise ValueError("max_bridged_missing_fraction must be finite and within [0, 1]")

    usable = observed & np.isfinite(centers).all(axis=2)
    base &= usable
    result = np.zeros_like(base)

    for mouse in range(base.shape[1]):
        segments = _runs(base[:, mouse])
        if not segments:
            continue
        bout_start, bout_end = segments[0]
        bout_gaps: list[tuple[int, int]] = []
        # `base` was already intersected with `usable`, so a fresh bout starts
        # with no unobserved frame; only bridges can add them.
        bout_missing = 0

        def accept_bout(start: int, end: int, gaps: list[tuple[int, int]]) -> None:
            def bounded(stop: int) -> bool:
                visible_points = usable[start:stop, mouse]
                points = centers[start:stop, mouse][visible_points]
                if not len(points) or not np.isfinite(points).all():
                    return False
                if (missing_fraction is not None
                        and np.count_nonzero(~visible_points) > missing_fraction * (stop - start)):
                    return False
                anchor = points[0]
                center = np.median(points, axis=0)
                return bool(
                    np.linalg.norm(points - anchor, axis=1).max(initial=0.0) <= excursion_limit
                    and np.linalg.norm(points - center, axis=1).max(initial=0.0) <= radius_limit)

            accepted_end = end
            if not bounded(end):
                # A brief observed movement at the end must not erase an
                # already bounded long standstill. Search only within the
                # existing transient-time allowance, never tile a long drift
                # into artificial Static chunks or extrapolate a missing tail.
                earliest = max(start + 1, end - int(math.floor(gap_limit * rate + 1e-9)))
                accepted_end = next((stop for stop in range(end - 1, earliest - 1, -1)
                                     if usable[stop:end, mouse].all() and bounded(stop)), start)
            if accepted_end <= start:
                return
            result[start:accepted_end, mouse] = base[start:accepted_end, mouse]
            for gap_start, gap_end in gaps:
                if gap_end >= accepted_end:
                    continue  # A bridge needs a retained right observed anchor.
                result[gap_start:gap_end, mouse] = True
                if bridged_out is not None:
                    bridged_out.append((mouse, gap_start, gap_end))

        for next_start, next_end in segments[1:]:
            gap_start, gap_end = bout_end, next_start
            gap_length = gap_end - gap_start
            candidate_end = next_end
            candidate_observed = usable[bout_start:candidate_end, mouse]
            candidate_points = centers[bout_start:candidate_end, mouse][candidate_observed]
            missing_gap = not usable[gap_start:gap_end, mouse].all()
            allowed_gap = missing_gap_limit if missing_gap else gap_limit
            candidate_bounded = bool(len(candidate_points)) and (
                np.linalg.norm(candidate_points - candidate_points[0], axis=1).max(initial=0.0)
                <= excursion_limit
            )
            can_bridge = (
                0 < gap_length <= int(math.floor(allowed_gap * rate + 1e-9))
                and candidate_bounded
            )
            if can_bridge and missing_fraction is not None:
                gap_missing = int(np.count_nonzero(~usable[gap_start:gap_end, mouse]))
                candidate_missing = bout_missing + gap_missing
                if candidate_missing > missing_fraction * (candidate_end - bout_start):
                    can_bridge = False
            else:
                gap_missing = 0
            if can_bridge:
                bout_end = next_end
                bout_gaps.append((gap_start, gap_end))
                bout_missing += gap_missing
            else:
                accept_bout(bout_start, bout_end, bout_gaps)
                bout_start, bout_end = next_start, next_end
                bout_gaps = []
                bout_missing = 0
        accept_bout(bout_start, bout_end, bout_gaps)

    return result


__all__ = [
    "bounded_stationary_mask",
    "bridge_gaps_outside_static",
    "static_posture_eligible_mask",
]
