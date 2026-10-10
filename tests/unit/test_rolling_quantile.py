"""Unit tests for the causal rolling quantile helper."""

from __future__ import annotations

import numpy as np
import pytest

from mouse_behavior.utils import rolling


def test_rolling_quantile_is_causal():
    values = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0])

    result = rolling.rolling_quantile(values, window=3, quantile=0.5)

    np.testing.assert_allclose(result, [1.0, 1.5, 2.0, 3.0, 4.0])


def test_rolling_quantile_skips_non_finite_samples():
    values = np.asarray([np.nan, np.nan, 4.0, 8.0])

    result = rolling.rolling_quantile(values, window=4, quantile=0.5)

    assert np.isnan(result[0])
    assert np.isnan(result[1])
    np.testing.assert_allclose(result[2], 4.0)
    np.testing.assert_allclose(result[3], 6.0)


def test_rolling_quantile_resets_between_inactive_runs():
    values = np.asarray([100.0, 100.0, 1.0, 2.0])
    active = np.asarray([True, True, False, False])

    result = rolling.rolling_quantile(values, window=4, quantile=0.5, active_mask=active)

    np.testing.assert_allclose(result[0], 100.0)
    np.testing.assert_allclose(result[1], 100.0)
    assert np.all(np.isnan(result[2:]))


def test_rolling_quantile_handles_two_dimensional_input():
    values = np.asarray([[1.0, 10.0], [3.0, 30.0]])
    valid = np.ones((2, 2), dtype=bool)

    result = rolling.rolling_quantile(values, window=2, quantile=1.0, active_mask=valid)

    assert result.shape == (2, 2)
    np.testing.assert_allclose(result[1], [3.0, 30.0])


def test_vectorized_and_loop_paths_agree(monkeypatch):
    rng = np.random.default_rng(7)
    values = rng.normal(size=(40, 3)) * 10.0
    values[rng.random(values.shape) < 0.2] = np.nan
    active = rng.random(values.shape) > 0.15

    vectorized = rolling.rolling_quantile(values, window=6, quantile=0.85, active_mask=active)
    monkeypatch.setattr(rolling, "_MAX_VECTORIZED_QUANTILE_SAMPLES", 0)
    looped = rolling.rolling_quantile(values, window=6, quantile=0.85, active_mask=active)

    np.testing.assert_allclose(looped, vectorized, equal_nan=True)


@pytest.mark.parametrize(
    "call",
    [
        lambda: rolling.rolling_quantile(np.zeros(4), 2, 1.5),
        lambda: rolling.rolling_quantile(np.zeros(4), 2, np.nan),
        lambda: rolling.rolling_quantile(np.zeros((2, 2, 2)), 2, 0.5),
        lambda: rolling.rolling_quantile(np.zeros(4), 2, 0.5, np.ones((3,))),
    ],
)
def test_rolling_quantile_rejects_malformed_inputs(call):
    with pytest.raises(ValueError):
        call()