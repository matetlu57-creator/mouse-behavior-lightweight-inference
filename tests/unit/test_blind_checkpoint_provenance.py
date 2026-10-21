from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import sys
from pathlib import Path

import pytest

from mouse_behavior.evaluation.exported_dataset import HeuristicParameters

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

predictor = importlib.import_module("predict_exported_blind")
resumer = importlib.import_module("resume_blind_prediction")


@pytest.fixture
def interrupted_run(tmp_path, monkeypatch):
    tracks = tmp_path / "tracks.json"
    tracks.write_text("[]", encoding="utf-8")
    manifest = tmp_path / "blind.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("sample_id", "tracks_path", "fps"))
        writer.writeheader()
        for sample in ("opaque-1", "opaque-2"):
            writer.writerow(dict(sample_id=sample, tracks_path=str(tracks), fps=30))
    model = tmp_path / "model"
    model.mkdir()
    frozen = model / "frozen_model.json"
    frozen.write_text(json.dumps(dict(
        heuristic_parameters=HeuristicParameters().as_dict(),
        ranking_weights={"stationary": 1.0},
    )), encoding="utf-8")
    (model / "run_provenance.json").write_text(json.dumps(dict(
        ready_for_blind=True, model_file=frozen.name,
        model_sha256=hashlib.sha256(frozen.read_bytes()).hexdigest(),
        source_sha256={}, max_tracks=64,
    )), encoding="utf-8")
    partial = tmp_path / "partial"
    output = tmp_path / "resumed"
    args = argparse.Namespace(blind_manifest=manifest, model_dir=model,
                              output_dir=partial, historical_model=True)
    monkeypatch.setattr(predictor, "parse_args", lambda: args)
    monkeypatch.setattr(predictor, "load_exported_features", lambda *a, **k: None)
    evidence = dict(identity_scores={"7": {"stationary": 1.0}}, pair_scores={},
                    group_scores={}, collective_feature_debug=[], selected_track_count=1,
                    track_selection_truncated=False)
    calls = iter((evidence, None))

    def interrupt_after_one(*args):
        result = next(calls)
        if result is None:
            raise RuntimeError("simulated interruption")
        return result

    monkeypatch.setattr(predictor, "predict_features", interrupt_after_one)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        predictor.main()
    monkeypatch.setattr(resumer, "parse_args", lambda: argparse.Namespace(
        blind_manifest=manifest, model_dir=model, partial_dir=partial,
        output_dir=output, historical_model=True,
    ))
    monkeypatch.setattr(resumer, "load_exported_features", lambda *a, **k: None)
    monkeypatch.setattr(resumer, "predict_features", lambda *a: evidence)
    return manifest, model, tracks, partial, output


def test_matching_checkpoint_resumes_interrupted_run(interrupted_run):
    _, model, _, partial, output = interrupted_run
    assert (partial / "inference_checkpoint.json").is_file()
    assert resumer.main() == 0
    receipt = json.loads((output / "prediction_receipt.json").read_text(encoding="utf-8"))
    assert receipt["resumed_evidence_count"] == 1
    assert receipt["prediction_count"] == 2
    assert receipt["parameters_sha256"] == hashlib.sha256(
        (model / "frozen_model.json").read_bytes()
    ).hexdigest()
    assert receipt["policy_scope"] == "historical_frozen_model_not_current_canonical_v27"


@pytest.mark.parametrize("drift", ["model", "manifest", "trajectory", "source", "evidence", "unsealed"])
def test_resume_rejects_drift_before_creating_any_output(interrupted_run, monkeypatch, drift):
    manifest, model, tracks, partial, output = interrupted_run
    if drift == "model":
        frozen = model / "frozen_model.json"
        frozen.write_text(frozen.read_text(encoding="utf-8") + " ", encoding="utf-8")
        provenance_path = model / "run_provenance.json"
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        provenance["model_sha256"] = hashlib.sha256(frozen.read_bytes()).hexdigest()
        provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    elif drift == "manifest":
        manifest.write_text(manifest.read_text(encoding="utf-8").replace(
            ",30", ",31"
        ), encoding="utf-8")
    elif drift == "trajectory":
        tracks.write_text("[{}]", encoding="utf-8")
    elif drift == "source":
        import blind_checkpoint

        hashes = blind_checkpoint.implementation_source_hashes()
        hashes["scripts/resume_blind_prediction.py"] = "0" * 64
        monkeypatch.setattr(blind_checkpoint, "implementation_source_hashes", lambda: hashes)
    elif drift == "evidence":
        evidence = partial / "trajectory_evidence.partial.jsonl"
        evidence.write_text(evidence.read_text(encoding="utf-8").replace(
            '"stationary": 1.0', '"stationary": 9.0'
        ), encoding="utf-8")
    else:
        (partial / "inference_checkpoint.json").unlink(missing_ok=True)
    with pytest.raises(ValueError, match="checkpoint"):
        resumer.main()
    assert not output.exists()


def test_resume_requires_explicit_historical_model(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["resume", "--blind-manifest", "blind.csv",
        "--model-dir", "model", "--partial-dir", "partial", "--output-dir", "out"])
    with pytest.raises(SystemExit) as exc:
        resumer.parse_args()
    assert exc.value.code == 2
