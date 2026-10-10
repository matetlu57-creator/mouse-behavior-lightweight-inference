"""Windowed numeric features used by behavior-analysis pipelines.

The helpers in this module treat every contiguous ``active_mask`` run as an
independent time series.  This is important for pair-analysis windows: samples
outside a candidate window must reset rolling history instead of leaking old
evidence into a later interaction.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator

import numpy as np


def _true_runs(values: np.ndarray) -> Iterator[tuple[int, int]]:
    """Yield half-open ``[start, end)`` bounds for contiguous true runs."""

    mask = np.asarray(values, dtype=bool).reshape(-1)
    if not mask.size or not mask.any():
        return
    padded = np.concatenate(([False], mask, [False]))
    boundaries = np.flatnonzero(padded[1:] != padded[:-1])
    for start, end in boundaries.reshape(-1, 2):
        yield int(start), int(end)


def _window_totals(values: np.ndarray, window: int) -> np.ndarray:
    """Return causal rolling totals for one contiguous active segment."""

    length = len(values)
    cumulative = np.empty(length + 1, dtype=float)
    cumulative[0] = 0.0
    np.cumsum(values, dtype=float, out=cumulative[1:])
    ends = np.arange(1, length + 1)
    starts = np.maximum(ends - window, 0)
    return cumulative[ends] - cumulative[starts]


def rolling_sum(
    values: np.ndarray,
    window: int,
    active_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Return a causal rolling sum that resets at inactive samples.

    The dense all-active path is vectorized over all columns.  Sparse pair
    timelines are processed one contiguous run at a time, avoiding a Python
    loop over every video frame while preserving the previous reset semantics.
    """

    values_array = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0)
    window = max(int(window), 1)
    was_1d = values_array.ndim == 1
    if was_1d:
        values_array = values_array[:, None]
    if values_array.ndim != 2:
        raise ValueError(f"rolling_sum expects a 1D or 2D array, got shape={values_array.shape}")

    if active_mask is None:
        active = np.ones_like(values_array, dtype=bool)
    else:
        active = np.asarray(active_mask, dtype=bool)
        if active.ndim == 1 and was_1d:
            active = active[:, None]
        if active.shape != values_array.shape:
            raise ValueError(
                f"active_mask must have shape {values_array.shape}, got {active.shape}"
            )

    if bool(np.all(active)):
        cumulative = np.concatenate(
            [np.zeros((1, values_array.shape[1])), np.cumsum(values_array, axis=0)],
            axis=0,
        )
        ends = np.arange(1, values_array.shape[0] + 1)
        starts = np.maximum(ends - window, 0)
        result = cumulative[ends] - cumulative[starts]
        return result[:, 0] if was_1d else result

    result = np.zeros_like(values_array, dtype=float)
    for column in range(values_array.shape[1]):
        for start, end in _true_runs(active[:, column]):
            result[start:end, column] = _window_totals(
                values_array[start:end, column],
                window,
            )
    return result[:, 0] if was_1d else result


def rolling_corr(
    velocity_a: np.ndarray,
    velocity_b: np.ndarray,
    valid: np.ndarray,
    window: int,
    active_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Rolling Pearson correlation over flattened 2-D velocity vectors.

    Each valid frame contributes its x and y components to the Pearson
    statistics, matching the historical behavior engine definition.  Inactive
    samples split the timeline into independent runs and always emit zero.
    """

    values_a = np.asarray(velocity_a, dtype=float)
    values_b = np.asarray(velocity_b, dtype=float)
    if values_a.shape != values_b.shape or values_a.ndim != 3 or values_a.shape[2] != 2:
        raise ValueError("velocity_a and velocity_b must have matching shape (frames, pairs, 2)")
    frames, pairs, _ = values_a.shape
    valid_array = np.asarray(valid, dtype=bool)
    if valid_array.shape != (frames, pairs):
        raise ValueError(f"valid must have shape ({frames}, {pairs}), got {valid_array.shape}")
    window = max(int(window), 4)
    if active_mask is None:
        active = np.ones((frames, pairs), dtype=bool)
    else:
        active = np.asarray(active_mask, dtype=bool)
        if active.shape != (frames, pairs):
            raise ValueError(f"active_mask must have shape ({frames}, {pairs}), got {active.shape}")

    result = np.zeros((frames, pairs), dtype=np.float32)
    for pair in range(pairs):
        for start, end in _true_runs(active[:, pair]):
            segment_valid = valid_array[start:end, pair]
            segment_a = np.nan_to_num(values_a[start:end, pair], nan=0.0)
            segment_b = np.nan_to_num(values_b[start:end, pair], nan=0.0)

            sum_a = _window_totals(
                np.where(segment_valid, segment_a.sum(axis=1), 0.0),
                window,
            )
            sum_b = _window_totals(
                np.where(segment_valid, segment_b.sum(axis=1), 0.0),
                window,
            )
            sum_a2 = _window_totals(
                np.where(segment_valid, np.square(segment_a).sum(axis=1), 0.0),
                window,
            )
            sum_b2 = _window_totals(
                np.where(segment_valid, np.square(segment_b).sum(axis=1), 0.0),
                window,
            )
            sum_ab = _window_totals(
                np.where(segment_valid, (segment_a * segment_b).sum(axis=1), 0.0),
                window,
            )
            count = _window_totals(segment_valid.astype(float), window)
            sample_count = 2.0 * count
            numerator = sample_count * sum_ab - sum_a * sum_b
            variance_a = np.maximum(sample_count * sum_a2 - np.square(sum_a), 0.0)
            variance_b = np.maximum(sample_count * sum_b2 - np.square(sum_b), 0.0)
            denominator = np.sqrt(variance_a * variance_b)
            good = (count >= 4.0) & (denominator > 1e-9)
            segment_result = np.zeros(end - start, dtype=np.float32)
            segment_result[good] = np.clip(
                numerator[good] / denominator[good],
                -1.0,
                1.0,
            )
            result[start:end, pair] = segment_result
    return result


# Vectorized sliding windows stay cheap only while the temporary buffer is
# small; large windows fall back to a per-sample loop instead of allocating a
# frames x window matrix.
_MAX_VECTORIZED_QUANTILE_SAMPLES = 4_000_000


def _segment_quantiles(segment: np.ndarray, window: int, quantile: float) -> np.ndarray:
    """Causal rolling quantile for one contiguous run, skipping non-finite values."""

    length = len(segment)
    if length == 0:
        return np.empty(0, dtype=float)
    padded = np.concatenate([np.full(window - 1, np.nan, dtype=float), segment])
    if length * window <= _MAX_VECTORIZED_QUANTILE_SAMPLES:
        windows = np.lib.stride_tricks.sliding_window_view(padded, window)
        with warnings.catch_warnings():
            # All-NaN windows are legitimate (a track that has not been seen
            # yet); nanquantile reports them as NaN rather than as an error.
            warnings.simplefilter("ignore", category=RuntimeWarning)
            result = np.nanquantile(windows, quantile, axis=1)
        return np.asarray(result, dtype=float)
    result = np.full(length, np.nan, dtype=float)
    for index in range(length):
        start = max(0, index - window + 1)
        sample = segment[start : index + 1]
        sample = sample[np.isfinite(sample)]
        if sample.size:
            result[index] = float(np.quantile(sample, quantile))
    return result


def rolling_quantile(
    values: np.ndarray,
    window: int,
    quantile: float,
    active_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Return a causal rolling quantile that ignores non-finite samples.

    Windows reset at inactive samples exactly like :func:`rolling_sum`, so a
    candidate window never inherits evidence from an earlier interaction.  The
    output is ``nan`` wherever a window holds no finite sample, which keeps an
    unobserved track undecidable instead of silently reporting a low value.
    """

    values_array = np.asarray(values, dtype=float)
    quantile_value = float(quantile)
    if not np.isfinite(quantile_value) or not 0.0 <= quantile_value <= 1.0:
        raise ValueError(f"quantile must be finite and in [0, 1], got {quantile}")
    window = max(int(window), 1)
    was_1d = values_array.ndim == 1
    if was_1d:
        values_array = values_array[:, None]
    if values_array.ndim != 2:
        raise ValueError(
            f"rolling_quantile expects a 1D or 2D array, got shape={values_array.shape}"
        )

    if active_mask is None:
        active = np.ones_like(values_array, dtype=bool)
    else:
        active = np.asarray(active_mask, dtype=bool)
        if active.ndim == 1 and was_1d:
            active = active[:, None]
        if active.shape != values_array.shape:
            raise ValueError(
                f"active_mask must have shape {values_array.shape}, got {active.shape}"
            )

    result = np.full(values_array.shape, np.nan, dtype=float)
    for column in range(values_array.shape[1]):
        for start, end in _true_runs(active[:, column]):
            result[start:end, column] = _segment_quantiles(
                values_array[start:end, column],
                window,
                quantile_value,
            )
    return result[:, 0] if was_1d else result


__all__ = ["rolling_corr", "rolling_quantile", "rolling_sum"]
