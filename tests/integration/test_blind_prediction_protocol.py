from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from mouse_behavior.evaluation.exported_dataset import HeuristicParameters


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_frozen_inference_precedes_separate_label_scoring(tmp_path: Path) -> None:
    points = [
        [10, 10, 1], [9, 10, 1], [11, 10, 1], [10, 12, 1],
        [10, 14, 1], [10, 16, 1], [10, 20, 1],
    ]
    tracks = tmp_path / "tracks.json"
    tracks.write_text(
        json.dumps(
            [
                {"frame": frame, "detections": [
                    {"track_id": 7, "keypoints": points, "box": [5, 5, 15, 25]}
                ]}
                for frame in range(12)
            ]
        ),
        encoding="utf-8",
    )
    unlabeled = tmp_path / "unlabeled.csv"
    with unlabeled.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("sample_id", "tracks_path", "fps"))
        writer.writeheader()
        writer.writerow({"sample_id": "opaque-1", "tracks_path": str(tracks), "fps": 10})
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    model_file = model_dir / "frozen_model.json"
    model_file.write_text(
        json.dumps({"heuristic_parameters": HeuristicParameters().as_dict(),
                    "ranking_weights": {"stationary": 1.0}}),
        encoding="utf-8",
    )
    (model_dir / "tuned_parameters.json").write_text(
        json.dumps(HeuristicParameters().as_dict()),
        encoding="utf-8",
    )
    source = REPO_ROOT / "src/mouse_behavior/evaluation/blind_top1.py"
    (model_dir / "run_provenance.json").write_text(
        json.dumps(
            {
                "ready_for_blind": True,
                "model_file": "frozen_model.json",
                "model_sha256": hashlib.sha256(model_file.read_bytes()).hexdigest(),
                "source_sha256": {
                    str(source.relative_to(REPO_ROOT)).replace("\\", "/"):
                    hashlib.sha256(source.read_bytes()).hexdigest()
                },
                "max_tracks": 64,
            }
        ),
        encoding="utf-8",
    )
    predictions_dir = tmp_path / "predictions"
    inference = subprocess.run(
        [
            sys.executable, str(REPO_ROOT / "scripts/predict_exported_blind.py"),
            "--blind-manifest", str(unlabeled), "--model-dir", str(model_dir),
            "--output-dir", str(predictions_dir),
        ],
        cwd=REPO_ROOT, capture_output=True, text=True, check=False,
    )
    assert inference.returncode == 0, inference.stderr
    with (predictions_dir / "predictions.csv").open("r", encoding="utf-8-sig") as handle:
        prediction = next(csv.DictReader(handle))
    assert prediction["predicted_behavior"] == "stationary"
    assert prediction["predicted_target_ids"] == "7"
    assert "canonical_behavior" not in prediction
    labels = tmp_path / "labels-revealed-after-prediction.csv"
    with labels.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("sample_id", "canonical_behavior", "mouse_ids"))
        writer.writeheader()
        writer.writerow({"sample_id": "opaque-1", "canonical_behavior": "stationary", "mouse_ids": "7"})
    score_path = tmp_path / "score.json"
    scoring = subprocess.run(
        [
            sys.executable, str(REPO_ROOT / "scripts/score_exported_blind.py"),
            "--predictions-dir", str(predictions_dir), "--labels", str(labels),
            "--output", str(score_path),
        ],
        cwd=REPO_ROOT, capture_output=True, text=True, check=False,
    )
    assert scoring.returncode == 0, scoring.stderr
    assert json.loads(score_path.read_text(encoding="utf-8"))["joint_top1_accuracy"] == 1.0

    model_file.write_text(model_file.read_text(encoding="utf-8") + " ", encoding="utf-8")
    blocked_dir = tmp_path / "blocked"
    tampered = subprocess.run(
        [
            sys.executable, str(REPO_ROOT / "scripts/predict_exported_blind.py"),
            "--blind-manifest", str(unlabeled), "--model-dir", str(model_dir),
            "--output-dir", str(blocked_dir),
        ],
        cwd=REPO_ROOT, capture_output=True, text=True, check=False,
    )
    assert tampered.returncode != 0
    assert "frozen model hash has changed" in tampered.stderr
    assert not blocked_dir.exists()
