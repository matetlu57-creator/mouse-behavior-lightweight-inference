"""Shared temporal evidence helpers for exported behavior analysis."""

from __future__ import annotations

import numpy as np

def _longest_run(mask: np.ndarray) -> int:
    values = np.asarray(mask, dtype=bool).reshape(-1)
    best = current = 0
    for value in values:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def _bridge_short_false_gaps(mask: np.ndarray, max_gap_frames: int) -> np.ndarray:
    """Bridge only interior false runs, preserving clip-boundary gaps."""

    result = np.asarray(mask, dtype=bool).copy()
    if result.ndim != 1 or max_gap_frames <= 0:
        return result
    true_indices = np.flatnonzero(result)
    for before, after in zip(true_indices[:-1], true_indices[1:]):
        gap = int(after - before - 1)
        if 0 < gap <= int(max_gap_frames):
            result[before + 1 : after] = True
    return result


def _duration_score(mask: np.ndarray, fps: float, minimum_seconds: float) -> tuple[bool, float]:
    duration = _longest_run(mask) / max(float(fps), 1e-6)
    minimum = max(float(minimum_seconds), 1.0 / max(float(fps), 1e-6))
    return duration >= minimum, duration


def _max_track_duration(mask: np.ndarray, fps: float) -> float:
    if mask.ndim != 2 or not mask.size:
        return 0.0
    return max((_longest_run(mask[:, index]) for index in range(mask.shape[1])), default=0) / max(
        fps, 1e-6
    )


def _pair_any_duration(mask: np.ndarray, fps: float) -> float:
    if mask.ndim != 2 or not mask.size:
        return 0.0
    return _longest_run(np.any(mask, axis=1)) / max(fps, 1e-6)


def _durations_by_column(mask: np.ndarray, fps: float) -> np.ndarray:
    """Return the longest continuous run in seconds for every identity/pair column."""

    values = np.asarray(mask, dtype=bool)
    if values.ndim != 2:
        raise ValueError(f"duration mask must be two-dimensional, got shape={values.shape}")
    longest = np.zeros(values.shape[1], dtype=np.int64)
    for column in range(values.shape[1]):
        padded = np.r_[False, values[:, column], False].astype(np.int8)
        changes = np.diff(padded)
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1)
        if starts.size:
            longest[column] = int(np.max(ends - starts))
    return longest.astype(float) / max(float(fps), 1e-6)


def _isolation_cumulative_durations(
    evidence_mask: np.ndarray,
    unknown_mask: np.ndarray,
    fps: float,
    max_gap_s: float,
) -> np.ndarray:
    """Count observed far-away frames per episode across brief ID dropouts."""

    evidence = np.asarray(evidence_mask, dtype=bool)
    unknown = np.asarray(unknown_mask, dtype=bool)
    if evidence.ndim != 2 or unknown.shape != evidence.shape:
        raise ValueError("isolation evidence and unknown masks must have matching 2D shapes")
    gap_limit = max(int(round(max_gap_s * fps)), 0)
    best = np.zeros(evidence.shape[1], dtype=np.int64)
    for column in range(evidence.shape[1]):
        observed = evidence[:, column]
        connected = observed.copy()
        if gap_limit:
            positions = np.flatnonzero(observed)
            for before, after in zip(positions[:-1], positions[1:]):
                start, end = int(before) + 1, int(after)
                if 0 < end - start <= gap_limit and np.all(unknown[start:end, column]):
                    connected[start:end] = True
        padded = np.r_[False, connected, False].astype(np.int8)
        changes = np.diff(padded)
        for start, end in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)):
            best[column] = max(best[column], np.count_nonzero(observed[start:end]))
    return best.astype(float) / max(float(fps), 1e-6)


def _qualified_bout_mask(mask: np.ndarray, minimum_frames: int) -> np.ndarray:
    """Retain only complete qualifying bouts; do not join disjoint episodes."""

    result = np.zeros_like(mask, dtype=bool)
    for column in range(mask.shape[1]):
        padded = np.r_[False, mask[:, column], False].astype(np.int8)
        changes = np.diff(padded)
        for start, end in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)):
            if end - start >= minimum_frames:
                result[start:end, column] = True
    return result


def _qualifying_score(duration: float, minimum_seconds: float, fps: float) -> float:
    minimum = max(float(minimum_seconds), 1.0 / max(float(fps), 1e-6))
    return float(duration) if float(duration) + 1e-12 >= minimum else 0.0


def _pair_key(left_id: int, right_id: int) -> str:
    left, right = sorted((int(left_id), int(right_id)))
    return f"{left},{right}"


def _prioritize_attack_over_approach(scores: dict[str, float], multiplier: float) -> None:
    """Break an Attack/Approach tie without displacing contact labels."""
    if (
        scores.get("approach", 0.0) > 0.0
        and scores.get("attack", 0.0) > 0.0
        and scores["approach"] >= max(scores.values())
    ):
        scores["attack"] *= max(float(multiplier), 1.0)
