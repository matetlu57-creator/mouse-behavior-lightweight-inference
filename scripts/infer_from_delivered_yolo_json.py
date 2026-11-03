#!/usr/bin/env python3
"""Validate delivered raw YOLO JSON, derive a cache, and run current rules only."""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import math
import pickle
from pathlib import Path

from _bootstrap import ensure_importable

ensure_importable()

from mouse_behavior.canonical_video_pipeline import _export_detection, run_pipeline, video_info  # noqa: E402
from mouse_behavior.evaluation.segment_export import _sha256, _write_json  # noqa: E402


def _digest(value, label: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value.lower())):
        raise ValueError(f"{label} must be a SHA-256 hex digest")
    return value.lower()


def verify_inputs(ledger: dict[str, str]) -> None:
    """Check original inputs again after import and after rule execution."""
    for filename, expected in ledger.items():
        if _sha256(Path(filename)) != expected:
            raise ValueError(f"original input changed: {filename}")


def import_delivered(input_dir: Path, output: Path, *, chunk_frames: int = 300) -> dict:
    """Create a new sibling cache without reading annotations or running YOLO.

    The delivery's historic cache binding is retained as evidence, never reused
    as the hash ledger for the newly serialized cache chunks.
    """
    input_dir, output = input_dir.resolve(), output.resolve()
    imported = output.with_name(output.name + "_import")
    for destination in (output, imported):
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"refusing to overwrite: {destination}")
    if type(chunk_frames) is not int or chunk_frames < 1:
        raise ValueError("chunk_frames must be a positive integer")
    metadata_path, tracks_path = input_dir / "metadata.json", input_dir / "tracks.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
    if not isinstance(metadata.get("source"), str) or not metadata["source"]:
        raise ValueError("metadata.source must identify the original video")
    video = input_dir / Path(metadata["source"]).name
    if not video.is_file():
        raise ValueError("delivered directory lacks the source video basename")
    model = Path(metadata.get("model", ""))
    if not model.is_file():
        raise ValueError("metadata.model must identify an existing model file")
    source_hash = _digest(metadata.get("binding", {}).get("source_video_sha256"), "source binding")
    model_hash = _digest(metadata.get("model_sha256"), "model binding")
    ledger = {str(path): _sha256(path) for path in (metadata_path, tracks_path, video, model)}
    if ledger[str(video)] != source_hash:
        raise ValueError("source video SHA-256 differs from delivered binding")
    if ledger[str(model)] != model_hash:
        raise ValueError("model SHA-256 differs from delivered binding")
    info = video_info(video)
    fps, count = metadata.get("fps"), metadata.get("frame_count")
    if (isinstance(fps, bool) or not isinstance(fps, (int, float))
            or not math.isfinite(fps) or fps <= 0 or type(count) is not int or count < 1):
        raise ValueError("metadata needs positive finite FPS and integer frame_count")
    if count != info["frame_count"] or not math.isclose(fps, info["fps"], rel_tol=1e-6):
        raise ValueError("delivered FPS/frame_count differs from actual source video")
    for dimension in ("width", "height"):
        if metadata.get(dimension) != info[dimension]:
            raise ValueError(f"delivered {dimension} differs from source video")
    if metadata.get("keypoint_names") != ["nose", "left_ear", "right_ear", "neck", "left_hip", "right_hip", "tail_root"]:
        raise ValueError("delivered keypoint order is not the supported raw YOLO order")
    records = json.loads(tracks_path.read_text(encoding="utf-8-sig"))
    if not isinstance(records, list) or len(records) != count:
        raise ValueError("raw tracks must be a list covering every source frame")
    for index, record in enumerate(records):
        if not isinstance(record, dict) or type(record.get("frame")) is not int or record["frame"] != index:
            raise ValueError("raw tracks must be contiguous integer frames from zero")
        timestamp = record.get("timestamp_seconds")
        if (isinstance(timestamp, bool) or not isinstance(timestamp, (int, float))
                or not math.isfinite(timestamp) or not math.isclose(timestamp, index / fps, abs_tol=1e-6, rel_tol=1e-9)):
            raise ValueError(f"raw timestamp disagrees with frame/FPS at frame {index}")
        detections = record.get("pose_detections")
        if not isinstance(detections, list):
            raise ValueError("raw frame needs a pose_detections list")
        seen = set()
        for detection in detections:
            if not isinstance(detection, dict) or type(detection.get("visual_track_id")) is not int:
                raise ValueError("visual_track_id must be an actual integer, never assigned during import")
            validated = _export_detection(detection)
            track = validated["track_id"]
            if track in seen:
                raise ValueError(f"duplicate visual_track_id in frame {index}")
            seen.add(track)
    # Complete validation precedes any output mutation. Preserve raw records;
    # validation's exported coordinate/ear ordering is not written into cache.
    verify_inputs(ledger)
    imported.mkdir(parents=True)
    chunk_hashes = {}
    for part, start in enumerate(range(0, count, chunk_frames)):
        path = imported / f"yolo_results.000000.{part:06d}.pkl.gz"
        with gzip.open(path, "xb") as handle:
            pickle.dump(records[start:start + chunk_frames], handle, protocol=pickle.HIGHEST_PROTOCOL)
        chunk_hashes[path.name] = _sha256(path)
    provenance = dict(derived_from_delivered_json=True, new_model_inference=False,
                      identity_semantics="provisional_visual_track_id", ids_assigned=False,
                      keypoints_filled=False, annotation_labels_used=False)
    status = dict(status="complete", video=str(video), source_video_sha256=source_hash,
                  total_frames=count, next_frame=count, fps=fps, chunk_count=len(chunk_hashes),
                  model=str(model.resolve()), model_sha256=model_hash,
                  tracking_backend="ultralytics", trajectory_schema_version=1, **provenance)
    _write_json(imported / "yolo_results_status.json", status)
    source_receipt = imported / "source_receipt.json"
    _write_json(source_receipt, dict(source_video=dict(info, sha256=source_hash),
                                    model=dict(path=str(model.resolve()), sha256=model_hash), **provenance))
    verify_inputs(ledger)
    receipt = dict(status="complete", input_dir=str(input_dir), cache_directory=str(imported),
                   source_receipt=str(source_receipt), video_path=str(video), original_input_sha256=ledger,
                   historic_delivery_binding=metadata.get("binding"),
                   historic_binding_role="historical evidence only; not hashes of this derived cache",
                   derived_cache_chunk_sha256=chunk_hashes,
                   derived_cache_status_sha256=_sha256(imported / "yolo_results_status.json"),
                   source_receipt_sha256=_sha256(source_receipt), original_inputs_preserved=True, **provenance)
    _write_json(imported / "import_receipt.json", receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--posture-mode", choices=("disabled", "v3_review"), default="disabled")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    imported = import_delivered(args.input_dir, args.output)
    try:
        receipt = run_pipeline(Path(imported["video_path"]), Path(imported["cache_directory"]), args.output,
                               Path(__file__).resolve().parents[1] / "configs/default.yaml", mode="infer",
                               source_receipt=Path(imported["source_receipt"]), workers=args.workers,
                               posture_mode=args.posture_mode)
    finally:
        verify_inputs(imported["original_input_sha256"])
    _write_json(args.output / "delivered_import_receipt.json", imported)
    logging.info("Rule inference complete: %s events=%d; no new model inference", args.output, receipt["event_count"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
