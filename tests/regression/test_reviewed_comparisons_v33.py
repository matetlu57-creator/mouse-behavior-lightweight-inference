"""Reviewed exclusions are exact source-local edges, never inferred ID merges."""
from copy import deepcopy

import numpy as np
import pytest

from mouse_behavior.behavior.reviewed_comparisons import ReviewedComparisonScope, bound_comparison_review
from mouse_behavior.evaluation.group_event_rules import isolation_candidates, isolation_summaries
from tests.regression import test_heuristics_v16_audit as fixtures


def scope():
    return ReviewedComparisonScope("sample", "b"*64, "a"*64, "c"*64, ((100, 101),))


def test_only_explicit_reviewed_edge_is_excluded_without_mutating_features():
    f = fixtures.features(360, 4)
    fixtures.set_centers(f, [[0., 0.], [.1, 0.], [30., 0.], [30.1, 0.]])
    fixtures.sync(f)
    original = deepcopy(f.pair_distance_cm)
    before = isolation_summaries(f)
    after = isolation_summaries(f, scope().pairs)
    assert before["100"]["minimum_cm"] == pytest.approx(.1)
    assert after["100"]["minimum_cm"] == pytest.approx(30.)
    assert after["102"]["minimum_cm"] == pytest.approx(.1), "an unrelated overlapping pair is never automatically excluded"
    assert np.array_equal(f.pair_distance_cm, original)
    assert any(c.target_ids == (100,) for c in isolation_candidates(f, fixtures.confirmed_params(), excluded_pairs=scope().pairs))


def test_only_duplicate_visible_does_not_supply_nearest_real_neighbor_time():
    f = fixtures.features(360, 2)
    fixtures.set_centers(f, [[0., 0.], [.1, 0.]])
    fixtures.sync(f)
    assert isolation_summaries(f, scope().pairs)["100"]["observed_frames"] == 0
    assert not isolation_candidates(f, fixtures.confirmed_params(), excluded_pairs=scope().pairs)


@pytest.mark.parametrize("field", ["comparison_sample_id", "source_tracks_sha256", "video_sha256"])
def test_review_rejects_cross_sample_or_changed_asset_binding(field):
    binding = dict(source_video=dict(sha256="a"*64), comparison_sample_id="sample",
                   source_tracks_sha256="b"*64, isolation_comparison_review=scope().as_dict())
    assert bound_comparison_review(binding) == scope()
    if field == "video_sha256":
        binding["source_video"]["sha256"] = "d"*64
    else:
        binding[field] = "other"
    with pytest.raises(ValueError, match="scope does not match"):
        bound_comparison_review(binding)
