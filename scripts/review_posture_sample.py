#!/usr/bin/env python3
"""Render independent V3 posture candidates from an existing pixel-pose sample."""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
from pathlib import Path

from _bootstrap import ensure_importable

ensure_importable()

from mouse_behavior.behavior.posture_adapter import KEYPOINT_NAMES, infer_posture_rows  # noqa: E402
from mouse_behavior.behavior.posture_rules import POSTURE_REVISION  # noqa: E402
from mouse_behavior.canonical_video_pipeline import TrackStore, merge_events, video_info  # noqa: E402
from mouse_behavior.evaluation.segment_export import _sha256, _write_json  # noqa: E402
from mouse_behavior.visualization.behavior_packages import (  # noqa: E402
    export_packages, render_full_package, write_manifests,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, required=True, help="Directory containing clip.mp4, metadata.json and tracks.json")
    parser.add_argument("--track-id", type=int, action="append", help="Exact visual ID; repeat for multiple IDs; omission considers all observed IDs")
    parser.add_argument("--output", type=Path, required=True, help="New directory outside the source sample")
    args = parser.parse_args()
    sample, output = args.sample.resolve(), args.output.resolve()
    if output.exists() or output.is_relative_to(sample):
        raise ValueError("output must be new and outside source sample")
    metadata = json.loads((sample / "metadata.json").read_text(encoding="utf-8-sig"))
    if metadata["tracks"]["coordinate_system"] != "clip_pixels":
        raise ValueError("sample must use explicit clip_pixels coordinates")
    names = metadata["pose"]["keypoint_names"]
    if len(names) != 7 or set(names) != set(KEYPOINT_NAMES):
        raise ValueError("unsupported seven-point pose schema")
    order = [names.index(name) for name in KEYPOINT_NAMES]
    video = (sample / metadata["clip"]["filename"]).resolve()
    if not video.is_relative_to(sample):
        raise ValueError("clip filename must remain inside sample directory")
    info = video_info(video)
    if (info["frame_count"] != metadata["clip"]["frame_count"]
            or not math.isclose(info["fps"], metadata["clip"]["fps"], rel_tol=1e-6)):
        raise ValueError("source MP4 and pose metadata frame count/FPS differ")
    input_hashes = {str(path): _sha256(path) for path in (video, sample / "metadata.json", sample / "tracks.json")}
    rows = json.loads((sample / "tracks.json").read_text(encoding="utf-8-sig"))
    for row in rows:
        for detection in row["detections"]:
            detection["keypoints"] = [detection["keypoints"][index] for index in order]
    events, audit = infer_posture_rows(rows, info["fps"], frame_count=info["frame_count"], track_ids=args.track_id)
    events = merge_events(events)
    info["sha256"] = input_hashes[str(video)]
    binding = dict(source_video=info, cache_status_sha256=input_hashes[str(sample / "metadata.json")],
                   chunk_sha256={"tracks.json": input_hashes[str(sample / "tracks.json")]},
                   visual_track_ids=sorted({d["track_id"] for row in rows for d in row["detections"]}),
                   posture_mode="v3_review", rule_policy=dict(posture_revision=POSTURE_REVISION),
                   rule_source_sha256={})
    from mouse_behavior.behavior import posture_adapter, posture_measurements, posture_rules
    binding["rule_source_sha256"] = {str(Path(module.__file__).resolve()): _sha256(Path(module.__file__))
                                     for module in (posture_adapter, posture_measurements, posture_rules)}
    output.mkdir(parents=True)
    receipt = dict(status="running", requires_review=True, source_input_sha256=input_hashes,
                   posture_mode="v3_review", revision=POSTURE_REVISION)
    _write_json(output / "run_receipt.json", receipt)
    try:
        database = output / "tracks.sqlite"
        with sqlite3.connect(database) as connection:
            connection.execute("CREATE TABLE frames(frame INTEGER PRIMARY KEY,payload TEXT)")
            by_frame = {row["frame"]: row for row in rows}
            for frame in range(info["frame_count"]):
                row = dict(by_frame.get(frame, dict(frame=frame, time=frame/info["fps"], detections=[])))
                row.update(source_frame=frame, source_fps=info["fps"])
                connection.execute("INSERT INTO frames VALUES(?,?)", (frame, json.dumps(row, allow_nan=False)))
        store = TrackStore(database)
        _write_json(output / "events.json", events)
        _write_json(output / "posture_audit.json", audit)
        _write_json(output / "input_binding.json", binding)
        full = render_full_package(video, output / "full", store, events, binding, {})
        clips = export_packages(output / "clips", Path(full["video_path"]), store, events, binding, {})
        write_manifests(output, clips)
        if any(_sha256(Path(path)) != digest for path, digest in input_hashes.items()):
            raise ValueError("source changed during review render")
        receipt.update(status="complete", full=full, event_count=len(events), clip_count=len(clips),
                       protected_inputs_unchanged=True, payload_sha256={
                           name: _sha256(output / name) for name in ("events.json", "posture_audit.json", "input_binding.json", "tracks.sqlite", "manifest.json", "manifest.csv")})
        _write_json(output / "run_receipt.json", receipt)
    except Exception as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
        _write_json(output / "run_receipt.json", receipt)
        raise
    print(json.dumps(dict(output=str(output), event_count=len(events), clip_count=len(clips), requires_review=True)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
