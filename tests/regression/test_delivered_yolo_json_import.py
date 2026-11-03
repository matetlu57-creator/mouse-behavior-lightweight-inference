"""Delivered JSON imports preserve raw observations and create fresh bindings."""
import gzip
import importlib
import json
import pickle
from pathlib import Path

import cv2
import numpy as np
import pytest

from mouse_behavior.canonical_video_pipeline import bind_cache, TrackStore
from mouse_behavior.evaluation.segment_export import _sha256


@pytest.fixture
def importer(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    return importlib.import_module("infer_from_delivered_yolo_json")


@pytest.fixture
def delivery(tmp_path):
    folder = tmp_path / "delivery"
    folder.mkdir()
    video = folder / "original.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10., (64, 48))
    assert writer.isOpened()
    for index in range(12):
        writer.write(np.full((48, 64, 3), index * 10, np.uint8))
    writer.release()
    model = tmp_path / "model.pt"
    model.write_bytes(b"bound model; import never loads it")
    detection = dict(visual_track_id=42, bbox_xyxy=[10, 10, 50, 30], box_conf=.9,
                     keypoints_px=[[40, 20], [36, 18], [36, 22], [30, 20], [20, 18], [20, 22], [10, 20]],
                     keypoint_conf=[1.] * 7, pose_quality=.95)
    rows = [dict(frame=index, timestamp_seconds=index / 10,
                 pose_detections=[detection] if index != 5 else []) for index in range(12)]
    (folder / "tracks.json").write_text(json.dumps(rows))
    (folder / "metadata.json").write_text(json.dumps(dict(
        source=str(tmp_path / "old_location" / video.name), fps=10., frame_count=12, width=64, height=48,
        model=str(model), model_sha256=_sha256(model),
        keypoint_names=["nose", "left_ear", "right_ear", "neck", "left_hip", "right_hip", "tail_root"],
        binding=dict(source_video_sha256=_sha256(video), cache_status_sha256="historic_only"))))
    # Its malformed contents would fail immediately if the importer used labels.
    (folder / "annotation.json").write_text("THIS IS NOT JSON")
    return folder, rows


def test_import_preserves_raw_records_and_binds_new_chunks(importer, delivery, tmp_path):
    folder, rows = delivery
    receipt = importer.import_delivered(folder, tmp_path / "run", chunk_frames=5)
    cache = Path(receipt["cache_directory"])
    recovered = []
    for path in sorted(cache.glob("*.pkl.gz")):
        with gzip.open(path, "rb") as handle:
            recovered.extend(pickle.load(handle))
        assert receipt["derived_cache_chunk_sha256"][path.name] == _sha256(path)
    assert recovered == rows
    assert receipt["derived_from_delivered_json"] is True
    assert receipt["new_model_inference"] is False
    assert receipt["historic_delivery_binding"]["cache_status_sha256"] == "historic_only"
    assert receipt["derived_cache_status_sha256"] != "historic_only"
    assert not (tmp_path / "run").exists()
    database = tmp_path / "tracks.sqlite"
    binding, _ = bind_cache(Path(receipt["video_path"]), cache, database,
                            source_receipt=Path(receipt["source_receipt"]))
    assert binding["visual_track_ids"] == [42]
    assert TrackStore(database).window(5, 6)[0]["detections"] == []
    importer.verify_inputs(receipt["original_input_sha256"])
    with pytest.raises(FileExistsError):
        importer.import_delivered(folder, tmp_path / "run")


@pytest.mark.parametrize("problem", ["bool_id", "duplicate", "missing_id", "bad_shape", "gap", "timestamp", "model_hash", "source_hash"])
def test_invalid_delivery_rejected_before_any_output(importer, delivery, tmp_path, problem):
    folder, rows = delivery
    if problem in {"bool_id", "missing_id", "bad_shape", "duplicate"}:
        detection = rows[0]["pose_detections"][0]
        if problem == "bool_id":
            detection["visual_track_id"] = True
        elif problem == "missing_id":
            detection.pop("visual_track_id")
        elif problem == "bad_shape":
            detection["keypoints_px"] = [[0, 0]]
        else:
            rows[0]["pose_detections"].append(dict(detection))
    elif problem == "gap":
        rows[1]["frame"] = 2
    elif problem == "timestamp":
        rows[1]["timestamp_seconds"] = .25
    else:
        path = folder / "metadata.json"
        metadata = json.loads(path.read_text())
        if problem == "model_hash":
            metadata["model_sha256"] = "0" * 64
        else:
            metadata["binding"]["source_video_sha256"] = "0" * 64
        path.write_text(json.dumps(metadata))
    (folder / "tracks.json").write_text(json.dumps(rows))
    with pytest.raises(ValueError):
        importer.import_delivered(folder, tmp_path / "run")
    assert not (tmp_path / "run").exists()
    assert not (tmp_path / "run_import").exists()


def test_verify_detects_changed_original_input(importer, delivery, tmp_path):
    folder, _ = delivery
    receipt = importer.import_delivered(folder, tmp_path / "run")
    (folder / "tracks.json").write_text("[]")
    with pytest.raises(ValueError, match="original input changed"):
        importer.verify_inputs(receipt["original_input_sha256"])
