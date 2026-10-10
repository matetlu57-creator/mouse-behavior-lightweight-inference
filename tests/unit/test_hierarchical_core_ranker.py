import numpy as np
from pathlib import Path
import sys

from mouse_behavior.evaluation.blind_top1 import Candidate
from mouse_behavior.evaluation.calibrated_rule_ranker import CandidateTable
from mouse_behavior.evaluation.hierarchical_core_ranker import predict_hierarchical_top1

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from validate_hierarchical_core import _read_unlabeled_inputs  # noqa: E402


def _model(scores):
    return {
        "feature_names": ["bias"],
        "coefficients": {behavior: [score] for behavior, score in scores.items()},
    }


def test_higher_layer_huddle_suppresses_overlapping_social_and_individual_events():
    candidates = (
        Candidate("stationary", (1,), 2.0),
        Candidate("approach", (1, 2), 1.0),
        Candidate("huddle", (1, 2, 3), 1.0),
        Candidate("social_clustering", (1, 2, 3), 1.0),
    )
    table = CandidateTable(candidates, np.ones((4, 1), dtype=np.float32), ("bias",))
    model = _model({"stationary": 4.0, "approach": 10.0, "huddle": 1.0,
                    "social_clustering": 50.0})

    prediction = predict_hierarchical_top1(table, model)

    # The core's group priority selects Huddling over clustering for these
    # mice, and that active group event suppresses their lower-layer labels.
    assert prediction.behavior == "huddle"
    assert prediction.target_ids == (1, 2, 3)


def test_ranker_does_not_create_behavior_when_no_trajectory_rule_fired():
    table = CandidateTable(
        (Candidate("locomotion", (7,), 0.0),),
        np.ones((1, 1), dtype=np.float32),
        ("bias",),
    )

    prediction = predict_hierarchical_top1(table, _model({"locomotion": 100.0}))

    assert prediction.behavior is None
    assert prediction.target_ids == ()


def test_validation_projection_ignores_label_columns(tmp_path):
    import csv

    manifest = tmp_path / "val.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("sample_id", "tracks_path", "fps", "split", "canonical_behavior", "mouse_ids"))
        writer.writerow(("leaky_behavior_sample_name", "tracks.json", "25", "val", "not-a-real-label", "not-an-id"))

    inputs = _read_unlabeled_inputs(manifest)

    assert inputs == [("v-000001", "tracks.json", 25.0)]
