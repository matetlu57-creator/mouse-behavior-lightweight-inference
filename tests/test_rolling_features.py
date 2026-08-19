from __future__ import annotations

import numpy as np
import pytest

from mouse_behavior.utils.rolling import rolling_corr, rolling_sum


def _reference_rolling_sum(
    values: np.ndarray,
    window: int,
    active_mask: np.ndarray | None = None,
) -> np.ndarray:
    values_array = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0)
    was_1d = values_array.ndim == 1
    if was_1d:
        values_array = values_array[:, None]
    active = (
        np.ones_like(values_array, dtype=bool)
        if active_mask is None
        else np.asarray(active_mask, dtype=bool).reshape(values_array.shape)
    )
    result = np.zeros_like(values_array, dtype=float)
    window = max(int(window), 1)
    for column in range(values_array.shape[1]):
        run_start = 0
        for frame in range(values_array.shape[0]):
            if not active[frame, column]:
                run_start = frame + 1
                continue
            start = max(run_start, frame - window + 1)
            result[frame, column] = values_array[start : frame + 1, column].sum()
    return result[:, 0] if was_1d else result


def _reference_rolling_corr(
    velocity_a: np.ndarray,
    velocity_b: np.ndarray,
    valid: np.ndarray,
    window: int,
    active_mask: np.ndarray | None = None,
) -> np.ndarray:
    values_a = np.asarray(velocity_a, dtype=float)
    values_b = np.asarray(velocity_b, dtype=float)
    frames, pairs, _ = values_a.shape
    active = (
        np.ones((frames, pairs), dtype=bool)
        if active_mask is None
        else np.asarray(active_mask, dtype=bool)
    )
    valid_array = np.asarray(valid, dtype=bool)
    result = np.zeros((frames, pairs), dtype=np.float32)
    window = max(int(window), 4)
    for pair in range(pairs):
        run_start = 0
        for frame in range(frames):
            if not active[frame, pair]:
                run_start = frame + 1
                continue
            start = max(run_start, frame - window + 1)
            frame_valid = valid_array[start : frame + 1, pair]
            if int(frame_valid.sum()) < 4:
                continue
            sample_a = np.nan_to_num(
                values_a[start : frame + 1, pair][frame_valid],
                nan=0.0,
            ).reshape(-1)
            sample_b = np.nan_to_num(
                values_b[start : frame + 1, pair][frame_valid],
                nan=0.0,
            ).reshape(-1)
            sample_count = float(len(sample_a))
            numerator = (
                sample_count * float(np.dot(sample_a, sample_b))
                - float(sample_a.sum()) * float(sample_b.sum())
            )
            variance_a = max(
                sample_count * float(np.dot(sample_a, sample_a))
                - float(sample_a.sum()) ** 2,
                0.0,
            )
            variance_b = max(
                sample_count * float(np.dot(sample_b, sample_b))
                - float(sample_b.sum()) ** 2,
                0.0,
            )
            denominator = float(np.sqrt(variance_a * variance_b))
            if denominator > 1e-9:
                result[frame, pair] = np.clip(
                    numerator / denominator,
                    -1.0,
                    1.0,
                )
    return result


@pytest.mark.parametrize("window", [1, 3, 20])
def test_rolling_sum_matches_reference_for_dense_and_sparse_windows(window: int):
    rng = np.random.default_rng(20260819)
    values = rng.normal(size=(47, 5))
    values[7, 2] = np.nan
    active = rng.random((47, 5)) > 0.35
    active[10:18, 0] = True
    active[23:28, 0] = False

    np.testing.assert_allclose(
        rolling_sum(values, window, active),
        _reference_rolling_sum(values, window, active),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        rolling_sum(values[:, 0], window),
        _reference_rolling_sum(values[:, 0], window),
        rtol=1e-12,
        atol=1e-12,
    )


def test_rolling_corr_matches_historical_definition_across_active_gaps():
    rng = np.random.default_rng(143)
    velocity_a = rng.normal(size=(83, 6, 2))
    velocity_b = 0.65 * velocity_a + rng.normal(scale=0.4, size=(83, 6, 2))
    valid = rng.random((83, 6)) > 0.18
    active = rng.random((83, 6)) > 0.30
    active[8:31, 0] = True
    active[16, 0] = False
    active[:, 5] = False
    velocity_a[12, 1, 0] = np.nan
    velocity_b[39, 3, 1] = np.nan

    expected = _reference_rolling_corr(
        velocity_a,
        velocity_b,
        valid,
        window=15,
        active_mask=active,
    )
    actual = rolling_corr(
        velocity_a,
        velocity_b,
        valid,
        window=15,
        active_mask=active,
    )

    assert actual.dtype == np.float32
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    assert np.all(actual[~active] == 0.0)


def test_rolling_helpers_reject_incompatible_shapes():
    with pytest.raises(ValueError, match="active_mask"):
        rolling_sum(np.zeros((3, 2)), 2, np.ones(3, dtype=bool))
    with pytest.raises(ValueError, match="matching shape"):
        rolling_corr(
            np.zeros((3, 2, 2)),
            np.zeros((3, 3, 2)),
            np.ones((3, 2), dtype=bool),
            4,
        )
