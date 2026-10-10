"""Regression tests for train-derived physical ranges and layered Top-1."""

from __future__ import annotations

import numpy as np
import pytest

from mouse_behavior.evaluation.blind_top1 import Candidate
from mouse_behavior.evaluation.calibrated_rule_ranker import CandidateTable
from mouse_behavior.evaluation.empirical_range_tuner import fit_ranges, predict_range_top1


def _rows():
    samples = []
    for behavior, nearest, speed, distance, visibility, displacement in (
        ("isolation", 2.4, .15, 2.6, 1.0, .1),
        ("stationary", 1.2, .12, 1.8, 1.0, .01),
        # [STEP4 2026-10-04] 个体层合并：Walking+Running -> Locomotion
        ("locomotion", 1.0, .7, 1.6, 1.0, .7),
        ("huddle", .2, .15, .22, .3, .05),
        ("social_clustering", .25, .17, .35, .9, .3),
    ):
        for offset in (-.02, 0.0, .02):
            samples.append({"behavior": behavior, "features": {
                "nearest_q50": nearest + offset, "speed_q50": speed + offset,
                "distance_q50": distance + offset,
                "member_visibility_min": visibility + offset,
                "net_displacement": displacement + offset,
            }})
    return samples


def test_train_ranges_separate_distance_motion_and_formation() -> None:
    rules = fit_ranges(_rows())
    by_feature = {(rule["behavior"], rule["feature"]): rule for rule in rules}
    assert 1.2 < by_feature["isolation", "nearest_q50"]["threshold"] < 2.4
    assert .12 < by_feature["stationary", "speed_q50"]["threshold"] < .7
    assert .3 < by_feature["social_clustering", "member_visibility_min"]["threshold"] < .9
    assert all(rule["positive_n"] == 3 for rule in rules)


def test_group_does_not_automatically_override_social_candidate() -> None:
    table = CandidateTable(
        (Candidate("together", (1, 2), 2), Candidate("social_clustering", (1, 2, 3), 5)),
        np.asarray([[1, .2], [1, 1.0]], dtype=np.float32),
        ("bias", "member_visibility_min"),
    )
    ranker = {"feature_names": list(table.feature_names), "coefficients": {
        "together": [3.0, 0.0], "social_clustering": [2.0, 0.0],
    }}
    ranges = [{"behavior": "social_clustering", "feature": "member_visibility_min",
               "direction": "min", "threshold": .8, "scale": .1,
               "balanced_accuracy": 1.0}]
    result = predict_range_top1(table, ranker, ranges, range_strength=1, layer_bonus=.25)
    assert result.behavior == "together"
    assert result.target_ids == (1, 2)


def test_range_penalty_rejects_unphysical_isolation_and_preserves_ids() -> None:
    table = CandidateTable(
        (Candidate("isolation", (9,), 10), Candidate("stationary", (4,), 1)),
        np.asarray([[1, .2], [1, 0.0]], dtype=np.float32),
        ("bias", "nearest_q50"),
    )
    ranker = {"feature_names": list(table.feature_names), "coefficients": {
        "isolation": [3.0, 0.0], "stationary": [2.0, 0.0],
    }}
    ranges = [{"behavior": "isolation", "feature": "nearest_q50",
               "direction": "min", "threshold": 2.0, "scale": .5,
               "balanced_accuracy": 1.0}]
    result = predict_range_top1(table, ranker, ranges, range_strength=2, layer_bonus=0)
    assert (result.behavior, result.target_ids) == ("stationary", (4,))
    with pytest.raises(ValueError, match="schema"):
        predict_range_top1(table, {**ranker, "feature_names": ["wrong"]}, ranges,
                           range_strength=1, layer_bonus=0)


def test_group_size_bonus_requires_positive_spatial_evidence() -> None:
    table = CandidateTable(
        (Candidate("huddle", (1, 2, 3), 2),
         Candidate("huddle", (1, 2, 3, 4), 2),
         Candidate("huddle", (1, 2, 3, 4, 5), 0)),
        np.asarray([[1, .35], [1, .4], [1, .6]], dtype=np.float32),
        ("bias", "member_count_log"),
    )
    ranker = {"feature_names": list(table.feature_names),
              "coefficients": {"huddle": [1.0, -1.0]}}
    baseline = predict_range_top1(table, ranker, (), range_strength=0, layer_bonus=0)
    calibrated = predict_range_top1(table, ranker, (), range_strength=0, layer_bonus=0,
                                    group_size_bonus=8)
    assert baseline.target_ids == (1, 2, 3)
    assert calibrated.target_ids == (1, 2, 3, 4)
