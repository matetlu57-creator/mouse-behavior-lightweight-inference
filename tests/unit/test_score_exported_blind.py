from __future__ import annotations

import sys
import csv
import argparse
import hashlib
import json
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from mouse_behavior.evaluation.blind_top1 import LabeledClip, Top1Prediction, evaluate_top1
from score_exported_blind import (
    _add_precision_recall,
    _write_scored_predictions,
    load_blind_labels_from_split,
)
import score_exported_blind as scorer


def test_precision_recall_uses_exact_target_ids_for_joint_metrics(tmp_path: Path) -> None:
    labels = {
        "v-1": LabeledClip("sample-1", "", 30.0, "", "huddle", (1, 2, 3)),
        "v-2": LabeledClip("sample-2", "", 30.0, "", "huddle", (1, 2, 3)),
    }
    predictions = {
        "v-1": Top1Prediction("huddle", (1, 2, 3), 4.0),
        "v-2": Top1Prediction("huddle", (1, 2, 4), 3.0),
    }
    metrics = evaluate_top1(
        [(labels[key], predictions[key]) for key in sorted(labels)]
    )

    _add_precision_recall(metrics, labels, predictions)

    huddle = metrics["per_behavior"]["huddle"]
    assert huddle["behavior_precision"] == 1.0
    assert huddle["behavior_recall"] == 1.0
    assert huddle["joint_id_precision"] == 0.5
    assert huddle["joint_id_recall"] == 0.5
    assert metrics["strict_exact_top1_accuracy"] == 0.5

    output = tmp_path / "scored.csv"
    _write_scored_predictions(output, labels, predictions)
    contents = output.read_text(encoding="utf-8-sig")
    assert "sample-1" in contents
    assert "sample-2" in contents


def test_labels_join_to_anonymous_copies_by_content_hash(tmp_path: Path) -> None:
    opaque_tracks = tmp_path / "opaque" / "v-000001.json"
    source_tracks = tmp_path / "labeled" / "annotated-track.json"
    opaque_tracks.parent.mkdir()
    source_tracks.parent.mkdir()
    payload = b'[{"frame":0,"detections":[{"track_id":7}]}]'
    opaque_tracks.write_bytes(payload)
    source_tracks.write_bytes(payload)

    manifest = tmp_path / "inference_manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("sample_id", "tracks_path", "fps"))
        writer.writeheader()
        writer.writerow({"sample_id": "v-000001", "tracks_path": opaque_tracks, "fps": 30})
    split = tmp_path / "val.csv"
    with split.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("sample_id", "tracks_path", "canonical_behavior", "mouse_ids", "split"),
        )
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": "annotated-1",
                "tracks_path": source_tracks,
                "canonical_behavior": "stationary",
                "mouse_ids": "7",
                "split": "val",
            }
        )

    labels = load_blind_labels_from_split(split, manifest)

    assert list(labels) == ["v-000001"]
    assert labels["v-000001"].behavior == "stationary"
    assert labels["v-000001"].target_ids == (7,)


def test_isolation_label_requires_one_target_mouse(tmp_path: Path) -> None:
    tracks = tmp_path / "tracks.json"
    tracks.write_text("[]", encoding="utf-8")
    manifest = tmp_path / "inference_manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("sample_id", "tracks_path", "fps"))
        writer.writeheader()
        writer.writerow({"sample_id": "v-isolation", "tracks_path": tracks, "fps": 30})
    split = tmp_path / "val.csv"
    with split.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("sample_id", "tracks_path", "canonical_behavior", "mouse_ids", "split"),
        )
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": "annotated-isolation",
                "tracks_path": tracks,
                "canonical_behavior": "isolation",
                "mouse_ids": "7",
                "split": "val",
            }
        )

    labels = load_blind_labels_from_split(split, manifest)

    assert labels["v-isolation"].behavior == "isolation"
    assert labels["v-isolation"].target_ids == (7,)


@pytest.mark.parametrize("substitute_manifest", [False, True])
def test_split_scoring_verifies_manifest_before_label_join(tmp_path, monkeypatch, substitute_manifest):
    manifest = tmp_path / "blind.csv"
    manifest.write_text("sample_id,tracks_path,fps\nopaque-1,first.json,30\n", encoding="utf-8")
    predicted_dir = tmp_path / "predictions"
    predicted_dir.mkdir()
    predictions = predicted_dir / "predictions.csv"
    predictions.write_text("sample_id,predicted_behavior,predicted_target_ids,score\n"
                           "opaque-1,stationary,7,1\n", encoding="utf-8")
    (predicted_dir / "prediction_receipt.json").write_text(json.dumps(dict(
        predictions_sha256=hashlib.sha256(predictions.read_bytes()).hexdigest(),
        prediction_count=1,
        blind_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
    )), encoding="utf-8")
    if substitute_manifest:
        manifest.write_text(manifest.read_text(encoding="utf-8").replace(
            "first.json", "different.json"
        ), encoding="utf-8")
    output = tmp_path / "score" / "metrics.json"
    monkeypatch.setattr(scorer, "parse_args", lambda: argparse.Namespace(
        predictions_dir=predicted_dir, labels=None, dataset_split=tmp_path / "labels.csv",
        blind_manifest=manifest, output=output,
    ))
    joined = []

    def label_join(*args):
        joined.append(True)
        return {"opaque-1": LabeledClip("annotated", "", 30.0, "", "stationary", (7,))}

    monkeypatch.setattr(scorer, "load_blind_labels_from_split", label_join)
    if substitute_manifest:
        with pytest.raises(ValueError, match="blind manifest.*receipt"):
            scorer.main()
        assert not joined
        assert not output.parent.exists()
    else:
        assert scorer.main() == 0
        assert joined
        assert json.loads(output.read_text(encoding="utf-8"))["strict_exact_top1_accuracy"] == 1.0
