"""The learned threshold tree uses annotations only at train/score boundaries."""

import csv
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/train_induced_rule_tree.py"


def _run(*args):
    env = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], cwd=ROOT,
                          capture_output=True, text=True, encoding="utf-8", env=env, check=False)


def _manifest(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_rule_tree_freezes_before_validation_label_join(tmp_path):
    rows = []
    for index in range(4):
        path = tmp_path / f"trajectory-{index}.json"
        points = [[10 + index + dx, 20 + dy, 0.99] for dx, dy in (
            (0, 0), (1, 1), (1, -1), (3, 0), (5, 1), (5, -1), (10, 0)
        )]
        path.write_text(json.dumps([{"frame": frame, "detections": [
            {"track_id": 7, "keypoints": points, "box": [10 + index, 19, 20 + index, 21]}
        ]} for frame in range(10)]), encoding="utf-8")
        rows.append({"sample_id": f"opaque-{index}", "tracks_path": str(path), "fps": 10,
                     "recording_session": f"session-{index}", "canonical_behavior": "stationary",
                     "mouse_ids": "7", "split": "train" if index < 3 else "val"})
    train_path = tmp_path / "train.csv"
    val_path = tmp_path / "val.csv"
    model_dir = tmp_path / "model"
    _manifest(train_path, rows[:3])
    result = _run("train", "--train-manifest", train_path, "--output-dir", model_dir,
                  "--depth", 3)
    assert result.returncode == 0, result.stderr
    assert not val_path.exists()
    assert json.loads((model_dir / "run_provenance.json").read_text())["ready_for_existing_validation"]
    rows[3]["sample_id"] = "stationary/opaque-3"
    _manifest(val_path, rows[3:])
    first = tmp_path / "validation-1"
    result = _run("validate", "--val-manifest", val_path, "--model-dir", model_dir,
                  "--output-dir", first)
    assert result.returncode == 0, result.stderr
    assert "stationary/opaque-3" not in (first / "predictions.csv").read_text(encoding="utf-8-sig")
    assert json.loads((first / "validation_metrics.json").read_text())["joint_top1_accuracy"] == 1.0
    rows[3]["mouse_ids"] = "99"
    _manifest(val_path, rows[3:])
    second = tmp_path / "validation-2"
    result = _run("validate", "--val-manifest", val_path, "--model-dir", model_dir,
                  "--output-dir", second)
    assert result.returncode == 0, result.stderr
    assert (first / "predictions.csv").read_bytes() == (second / "predictions.csv").read_bytes()
    assert json.loads((second / "validation_metrics.json").read_text())["joint_top1_accuracy"] == 0.0
