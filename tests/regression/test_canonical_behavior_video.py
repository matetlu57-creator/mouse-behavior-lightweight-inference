"""Frame boundaries, provenance and real media checks for long-video exports."""
from __future__ import annotations

import gzip
import json
import pickle
from pathlib import Path

import cv2
import numpy as np
import pytest

from mouse_behavior.canonical_video_pipeline import (
    TrackStore, bind_cache, merge_events, infer_window_events, _directed_events, run_pipeline, infer_video_events,
)
from mouse_behavior.visualization.behavior_packages import (
    export_packages, render_full_package, draw_overlay,
)
from mouse_behavior.evaluation.segment_export import _sha256


@pytest.fixture
def source(tmp_path):
    video = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 96))
    assert writer.isOpened()
    for frame in range(12):
        writer.write(np.full((96, 160, 3), frame * 15, np.uint8))
    writer.release()
    cache = tmp_path / "cache"
    cache.mkdir()
    model = tmp_path / "model.pt"
    model.write_bytes(b"synthetic model binding fixture")
    rows = []
    for frame in range(12):
        detections = []
        for track in (1, 99, 1000):
            points = np.array([[70, 40], [65, 37], [65, 43], [60, 40],
                               [48, 37], [48, 43], [40, 40]], dtype=float)
            detections.append(dict(visual_track_id=track, keypoints_px=points,
                                   keypoint_conf=np.ones(7), bbox_xyxy=[35, 30, 75, 50],
                                   box_conf=.9))
        rows.append(dict(frame=frame, timestamp_seconds=frame / 10,
                         pose_detections=detections))
    with gzip.open(cache / "yolo_results.000000.000000.pkl.gz", "wb") as handle:
        pickle.dump(rows, handle)
    (cache / "yolo_results_status.json").write_text(json.dumps(dict(
        status="complete", video=str(video), total_frames=12, next_frame=12,
        chunk_count=1, fps=10, trajectory_schema_version=1, tracking_backend="ultralytics",
        model=str(model),
    )))
    (tmp_path / "source_receipt.json").write_text(json.dumps(dict(
        source_video=dict(path=str(video), sha256=_sha256(video), fps=10, frame_count=12),
        model=dict(path=str(model), sha256=_sha256(model)), cache_directory=str(cache),
    )))
    return video, cache, rows


def test_cache_requires_actual_source_hash_and_preserves_every_id(source, tmp_path):
    video, cache, _ = source
    with pytest.raises(ValueError, match="SHA-256"):
        bind_cache(video, cache, tmp_path / "bad.sqlite", expected_source_sha256="0" * 64,
                   source_receipt=cache.parent / "source_receipt.json")
    binding, scale = bind_cache(video, cache, tmp_path / "tracks.sqlite",
                                expected_source_sha256=_sha256(video), source_receipt=cache.parent / "source_receipt.json")
    assert binding["track_truncation"] is False
    assert binding["visual_track_ids"] == [1, 99, 1000]
    assert scale["reference_body_px"] == 30
    assert scale["cm_per_pixel"] == pytest.approx(7.918 / 30)
    assert len(TrackStore(tmp_path / "tracks.sqlite").window(0, 12)[0]["detections"]) == 3


def test_cache_rejects_missing_or_duplicate_frames(source, tmp_path):
    video, cache, rows = source
    rows[4]["frame"] = 3
    with gzip.open(cache / "yolo_results.000000.000000.pkl.gz", "wb") as handle:
        pickle.dump(rows, handle)
    with pytest.raises(ValueError, match="continuous"):
        bind_cache(video, cache, tmp_path / "tracks.sqlite",
                   expected_source_sha256=_sha256(video), source_receipt=cache.parent / "source_receipt.json")


def event(start, end, actor=99, target=1):
    return dict(behavior="Approach", canonical_behavior="approach", mouse_ids=[1, 99],
                actor_id=actor, target_id=target, start_frame=start, end_frame=end,
                participants=[dict(role_name="Approaching", track_ids=[actor]),
                              dict(role_name="Approached", track_ids=[target])])


def test_merge_only_continuous_same_roles_and_ids():
    merged = merge_events([event(0, 3), event(3, 5), event(6, 8), event(8, 10, 1, 99)])
    assert [(r["start_frame"], r["end_frame"]) for r in merged] == [(0, 5), (6, 8), (8, 10)]
    assert merged[-1]["actor_id"] == 1


def test_four_files_exact_boundaries_roles_all_ids_and_decode(source, tmp_path):
    video, cache, _ = source
    original = _sha256(video)
    binding, scale = bind_cache(video, cache, tmp_path / "tracks.sqlite",
                                expected_source_sha256=original, source_receipt=cache.parent / "source_receipt.json")
    store = TrackStore(tmp_path / "tracks.sqlite")
    events = merge_events([event(2, 7)])
    full = render_full_package(video, tmp_path / "full", store, events, binding, scale)
    rows = export_packages(tmp_path / "clips", Path(full["video_path"]), store,
                           events, binding, scale)
    package = Path(rows[0]["path"])
    assert set(p.name for p in package.iterdir()) == {
        "clip.mp4", "annotation.json", "metadata.json", "tracks.json"}
    annotation = json.loads((package / "annotation.json").read_text())
    metadata = json.loads((package / "metadata.json").read_text())
    tracks = json.loads((package / "tracks.json").read_text())
    assert annotation["frame_range"] == dict(start=0, end=4)
    assert annotation["confidence"] == "heuristic_candidate"
    assert annotation["requires_review"] is True
    assert annotation["participants"][0]["track_ids"] == [99]
    assert metadata["source"]["start_frame"] == 2
    assert metadata["source"]["end_frame"] == 6
    assert [r["source_frame"] for r in tracks] == list(range(2, 7))
    assert [r["frame"] for r in tracks] == list(range(5))
    assert {d["track_id"] for d in tracks[0]["detections"]} == {1, 99, 1000}
    assert rows[0]["verification"]["frame_count"] == 5
    assert _sha256(video) == original
    with pytest.raises(FileExistsError):
        render_full_package(video, tmp_path / "full", store, events, binding, scale)


def test_actual_canonical_window_keeps_all_ids_and_disabled_postures(source, tmp_path):
    from mouse_behavior.config import load_config
    from mouse_behavior.evaluation.exported_dataset_config import config_seed_parameters
    video, cache, _ = source
    binding, scale = bind_cache(video, cache, tmp_path / "tracks.sqlite",
                                expected_source_sha256=_sha256(video), source_receipt=cache.parent / "source_receipt.json")
    params = config_seed_parameters(load_config(Path(__file__).resolve().parents[2] / "configs/default.yaml"))
    result, audit = infer_window_events(TrackStore(tmp_path / "tracks.sqlite").window(0, 12),
                                       binding["source_video"]["fps"], scale, params)
    assert audit["track_ids"] == [1, 99, 1000]
    assert audit["track_truncation"] is False
    assert not {"Grooming", "Rearing"}.intersection(r["behavior"] for r in result)


def test_optin_posture_review_runs_real_pipeline_with_source_ledgers(source, tmp_path):
    video, cache, _ = source
    config = Path(__file__).resolve().parents[2] / "configs/default.yaml"
    output = tmp_path / "posture_review"
    receipt = run_pipeline(video, cache, output, config, mode="all", posture_mode="v3_review",
                           source_receipt=cache.parent / "source_receipt.json")
    assert receipt["status"] == "complete" and receipt["posture_mode"] == "v3_review"
    assert receipt["posture_review_behaviors"] == ["grooming", "rearing"]
    audit = receipt["inference_audit"]
    assert set(audit["windows"][0]["posture_review"]) == {"1", "99", "1000"}
    assert set(audit["global_posture_resolution"]) == {"1", "99", "1000"}
    binding = json.loads((output / "input_binding.json").read_text())
    sources = binding["rule_source_sha256"]
    for filename in ("posture_rules.py", "posture_adapter.py", "posture_measurements.py"):
        assert any(Path(path).name == filename and digest == _sha256(Path(path)) for path, digest in sources.items())
    # Opt-in never changes these unreviewed candidates into confirmed labels.
    assert receipt["disabled_behaviors"] == ["grooming", "rearing"]


def test_export_honors_effective_stricter_run_duration_before_creating_output(source, tmp_path):
    from dataclasses import asdict, replace
    from mouse_behavior.evaluation.exported_dataset_models import HeuristicParameters
    video, cache, _ = source
    binding, scale = bind_cache(video, cache, tmp_path / "tracks.sqlite", source_receipt=cache.parent / "source_receipt.json")
    binding["effective_parameters"] = asdict(replace(HeuristicParameters(), stationary_min_duration_s=2.))
    events = merge_events([dict(behavior="Static", canonical_behavior="stationary", start_frame=0, end_frame=12,
                                mouse_ids=[1], actor_id=-1, target_id=-1, required_s=.01)])
    output = tmp_path / "rejected"
    with pytest.raises(ValueError, match="duration rejected"):
        export_packages(output, video, TrackStore(tmp_path / "tracks.sqlite"), events, binding, scale)
    assert not output.exists()


def test_overlay_highlights_only_present_active_target_boxes(monkeypatch):
    import mouse_behavior.visualization.behavior_packages as packages
    calls = []
    original = packages.cv2.rectangle
    def spy(image, start, end, color, thickness, *args, **kwargs):
        calls.append((start, end, color, thickness))
        return original(image, start, end, color, thickness, *args, **kwargs)
    monkeypatch.setattr(packages.cv2, "rectangle", spy)
    detections = [dict(track_id=track, box=[20+track*20, 100, 35+track*20, 125],
                       keypoints=[[20, 100, 1.]]*7) for track in (1, 2)]
    active = dict(behavior="Rearing", canonical_behavior="rearing", mouse_ids=[1, 99],
                  start_frame=2, end_frame=6, event_id="candidate")
    draw_overlay(np.zeros((240, 320, 3), np.uint8), dict(detections=detections), [active], 3)
    green = [call for call in calls if call[2] == (0, 255, 0) and call[3] == 3]
    assert green == [((40, 100), (55, 125), (0, 255, 0), 3)]


def test_unknown_role_rows_do_not_split_process_and_reversal_does():
    mask = np.ones(8, bool)
    actor = np.array([99, 99, -1, -1, 99, 1, -1, 1])
    target = np.array([1, 1, -1, -1, 1, 99, -1, 99])
    rows = _directed_events(mask, actor, target, [1, 99], "approach", np.ones(8, bool), fps=5.)
    assert [(r["start_frame"], r["end_frame"], r["actor_id"]) for r in rows] == [(0, 5, 99), (5, 8, 1)]
    assert rows[0]["role_support_frame_intervals"] == [(0, 2), (4, 5)]
    assert rows[0]["role_unknown_frame_intervals"] == [(2, 4)]
    unknown = _directed_events(mask, actor * 0 - 1, target * 0 - 1, [1, 99], "chase", mask, fps=5.)
    assert len(unknown) == 1 and unknown[0]["participants"] == []
    assert unknown[0]["role_status"] == "unknown"


def test_event_inferred_phase_preserved_but_measured_support_separate():
    observed = np.array([True, True, False, False, True, True])
    rows = _directed_events(np.ones(6, bool), np.array([99, 99, -1, -1, 99, 99]),
                            np.array([1, 1, -1, -1, 1, 1]), [1, 99], "approach", observed, fps=10.)
    assert rows[0]["start_frame"] == 0 and rows[0]["end_frame"] == 6
    assert rows[0]["observed_all_participants_frame_count"] == 4
    assert rows[0]["inferred_or_partially_unobserved_frame_count"] == 2


@pytest.mark.parametrize("fps", [10., 29.967205])
def test_accurate_seek_keeps_numbered_source_first_and_last_frames(tmp_path, fps):
    """Colour-coded frames catch off-by-one crops that a frame-count check misses."""
    video = tmp_path / "numbered.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (160, 96))
    assert writer.isOpened()
    for frame in range(48):
        image = np.full((96, 160, 3), (frame * 4, 20, 230 - frame * 4), np.uint8)
        writer.write(image)
    writer.release()
    from mouse_behavior.canonical_video_pipeline import video_info
    info = video_info(video)
    info["sha256"] = _sha256(video)
    import sqlite3
    database = tmp_path / "tracks.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE frames(frame INTEGER PRIMARY KEY,payload TEXT)")
        for frame in range(48):
            connection.execute("INSERT INTO frames VALUES(?,?)", (frame, json.dumps(dict(
                frame=frame, time=frame / info["fps"], source_frame=frame,
                source_fps=info["fps"], detections=[]))))
    store = TrackStore(database)
    binding = dict(source_video=info, cache_status_sha256="synthetic", chunk_sha256={})
    events = merge_events([event(0, 10), event(17, 27), event(38, 48)])
    full = render_full_package(video, tmp_path / "full", store, events, binding, {})
    manifest = export_packages(tmp_path / "clips", Path(full["video_path"]), store, events, binding, {})
    source_capture = cv2.VideoCapture(full["video_path"])
    reference = []
    while True:
        ok, image = source_capture.read()
        if not ok:
            break
        reference.append(image[80:, 100:].mean(axis=(0, 1)))
    source_capture.release()
    for row in manifest:
        capture = cv2.VideoCapture(row["video_path"])
        matched = []
        while True:
            ok, image = capture.read()
            if not ok:
                break
            color = image[80:, 100:].mean(axis=(0, 1))
            matched.append(int(np.argmin(np.linalg.norm(np.array(reference) - color, axis=1))))
        capture.release()
        assert matched == list(range(row["start_frame"], row["end_frame"]))


def test_pipeline_refuses_overwrite_and_persists_failed_status(source, tmp_path):
    video, cache, _ = source
    root = tmp_path / "run"
    config = Path(__file__).resolve().parents[2] / "configs/default.yaml"
    with pytest.raises(ValueError, match="SHA-256"):
        run_pipeline(video, cache, root, config, mode="infer", expected_source_sha256="0" * 64,
                     source_receipt=cache.parent / "source_receipt.json")
    assert json.loads((root / "run_receipt.json").read_text())["status"] == "failed"
    before = (root / "run_receipt.json").read_bytes()
    with pytest.raises(FileExistsError):
        run_pipeline(video, cache, root, config, mode="infer", expected_source_sha256=_sha256(video),
                     source_receipt=cache.parent / "source_receipt.json")
    assert (root / "run_receipt.json").read_bytes() == before


def test_render_only_validates_persisted_events_without_inference(source, tmp_path, monkeypatch):
    video, cache, _ = source
    config = Path(__file__).resolve().parents[2] / "configs/default.yaml"
    initial = tmp_path / "infer"
    run_pipeline(video, cache, initial, config, mode="infer", expected_source_sha256=_sha256(video),
                 source_receipt=cache.parent / "source_receipt.json")
    import mouse_behavior.canonical_video_pipeline as pipeline
    def forbidden(*args, **kwargs):
        raise AssertionError("render-only must not run canonical inference or YOLO")
    monkeypatch.setattr(pipeline, "infer_video_events", forbidden)
    rendered = run_pipeline(video, None, tmp_path / "render", config, mode="full", events_from=initial)
    assert rendered["status"] == "complete" and rendered["full"]["verification"]["frame_count"] == 12
    (initial / "events.json").write_text("[]")
    with pytest.raises(ValueError, match="input changed"):
        run_pipeline(video, None, tmp_path / "tampered", config, mode="full", events_from=initial)


def test_overlay_shows_directed_social_before_many_static_and_never_fabricates_target(monkeypatch):
    import mouse_behavior.visualization.behavior_packages as packages
    texts, arrows = [], []
    real_text, real_arrow = packages.cv2.putText, packages.cv2.arrowedLine
    def text_spy(image, text, *args, **kwargs):
        texts.append(text)
        return real_text(image, text, *args, **kwargs)
    def arrow_spy(image, start, end, *args, **kwargs):
        arrows.append((start, end))
        return real_arrow(image, start, end, *args, **kwargs)
    monkeypatch.setattr(packages.cv2, "putText", text_spy)
    monkeypatch.setattr(packages.cv2, "arrowedLine", arrow_spy)
    events = [dict(canonical_behavior="stationary", behavior="Static", mouse_ids=[i],
                   start_frame=0, end_frame=20, event_id=str(i)) for i in range(9)]
    directed = {**event(0, 20), "event_id": "directed", "role_support_frame_intervals": [[0, 20]]}
    events.append(directed)
    detections = [dict(track_id=track, box=[x - 10, 35, x + 10, 65],
                       keypoints=[[x + 8, 50, 1], [x + 5, 47, 1], [x + 5, 53, 1],
                                  [x, 50, 1], [x - 5, 47, 1], [x - 5, 53, 1], [x - 10, 50, 1]])
                  for track, x in ((99, 40), (1, 110))]
    draw_overlay(np.zeros((240, 320, 3), np.uint8), dict(detections=detections), events, 3)
    assert any(text.startswith("Approach [1,99]") for text in texts)
    assert "Approaching V99" in texts and "Approached V1" in texts
    assert ((40, 50), (110, 50)) in arrows
    texts.clear()
    arrows.clear()
    draw_overlay(np.zeros((240, 320, 3), np.uint8), dict(detections=detections[:1]), events, 3)
    assert "Approached V1" not in texts
    assert ((40, 50), (110, 50)) not in arrows


def test_source_hash_alone_cannot_bypass_model_binding(source, tmp_path):
    video, cache, _ = source
    with pytest.raises(ValueError, match="model binding"):
        bind_cache(video, cache, tmp_path / "tracks.sqlite", expected_source_sha256=_sha256(video))


@pytest.mark.parametrize("model_binding", [None, {}, {"path": "model.pt"}, {"sha256": "0" * 64}])
def test_cache_rejects_incomplete_model_receipt(source, tmp_path, model_binding):
    video, cache, _ = source
    path = cache.parent / "source_receipt.json"
    receipt = json.loads(path.read_text())
    receipt["model"] = model_binding
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="model binding"):
        bind_cache(video, cache, tmp_path / "tracks.sqlite", source_receipt=path)


def test_cache_rejects_changed_model_hash(source, tmp_path):
    video, cache, _ = source
    path = cache.parent / "source_receipt.json"
    receipt = json.loads(path.read_text())
    Path(receipt["model"]["path"]).write_bytes(b"a different model")
    with pytest.raises(ValueError, match="model SHA-256"):
        bind_cache(video, cache, tmp_path / "tracks.sqlite", source_receipt=path)


def test_parallel_context_cores_are_bitexact_with_real_positive_chase_roles_and_unknown_rows(tmp_path):
    import sqlite3
    from mouse_behavior.evaluation.exported_dataset_config import config_seed_parameters
    database = tmp_path / "positive.sqlite"
    fps, count = 30., 150
    shape = np.array([[3.959, 0], [2.5, .8], [2.5, -.8], [1, 0],
                      [-2, 1], [-2, -1], [-3.959, 0]])
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE frames(frame INTEGER PRIMARY KEY,payload TEXT)")
        for frame in range(count):
            detections = []
            if not 90 <= frame < 94:
                for track, delay in ((100, .3), (101, 0.)):
                    time = frame / fps - delay
                    center = np.array([50 + 60 * min(time, 2), 50 + 60 * max(time - 2, 0)])
                    direction = np.array([1., 0.]) if time < 2 else np.array([0., 1.])
                    side = np.array([-direction[1], direction[0]])
                    points = center + shape[:, :1] * direction + shape[:, 1:] * side
                    if track == 100 and 40 <= frame < 45:
                        points[0] = 2 * points[3] - points[0]
                    points = points * 10
                    box = [*points.min(axis=0), *points.max(axis=0)]
                    detections.append(dict(track_id=track, box=box,
                        keypoints=[[*points[index], 1.] for index in (0, 2, 1, 3, 4, 5, 6)],
                        detection_confidence=1., class_id=0))
            # Fragmented visual IDs are retained even when another ID never
            # shares a frame; they are not merged into biological identities.
            if frame < 10 or frame > 135:
                track = 1000 if frame < 10 else 2000
                points = (shape + [200, 200]) * 10
                detections.append(dict(track_id=track, box=[1900, 1900, 2100, 2100],
                    keypoints=[[*points[index], 1.] for index in (0, 2, 1, 3, 4, 5, 6)],
                    detection_confidence=1., class_id=0))
            row = dict(frame=frame, time=frame / fps, source_frame=frame,
                       source_fps=fps, detections=detections)
            connection.execute("INSERT INTO frames VALUES(?,?)", (frame, json.dumps(row)))
    binding = dict(source_video=dict(fps=fps, frame_count=count))
    scale = dict(reference_body_cm=7.918, cm_per_pixel=.1)
    params = config_seed_parameters({})
    serial, audit = infer_video_events(TrackStore(database), binding, scale, params,
                                      core_seconds=1.5, context_seconds=10, workers=1)
    parallel, parallel_audit = infer_video_events(TrackStore(database), binding, scale, params,
                                                core_seconds=1.5, context_seconds=10, workers=2)
    chasing = [event for event in serial if event["canonical_behavior"] == "chase"]
    assert chasing and any(event["actor_id"] == 100 and event["target_id"] == 101 for event in chasing)
    assert any(event["role_unknown_frame_intervals"] for event in chasing)
    assert all(not (event["start_frame"] < 94 and event["end_frame"] > 90) for event in chasing)
    assert serial == parallel
    assert audit["windows"] == parallel_audit["windows"]
    assert all(window["track_ids"] == [100, 101, 1000, 2000] for window in audit["windows"])


@pytest.mark.parametrize("clip_preset", ["veryfast", "ultrafast"])
def test_clip_parallel_all_four_payload_hashes_equal_serial_for_multiple_positive_events(source, tmp_path, clip_preset):
    from time import perf_counter
    video, cache, _ = source
    binding, scale = bind_cache(video, cache, tmp_path / "tracks.sqlite",
                                source_receipt=cache.parent / "source_receipt.json")
    store = TrackStore(tmp_path / "tracks.sqlite")
    events = merge_events([event(0, 3), event(4, 8), event(9, 12, 1, 99)])
    for row in events:
        row["role_status"] = "event_direction_with_measured_support"
        row["role_support_frame_intervals"] = [[row["start_frame"], row["end_frame"]]]
    full = render_full_package(video, tmp_path / "full", store, events, binding, scale)
    started = perf_counter()
    serial = export_packages(tmp_path / "serial", Path(full["video_path"]), store, events, binding, scale, workers=1, clip_preset=clip_preset)
    serial_seconds = perf_counter() - started
    started = perf_counter()
    parallel = export_packages(tmp_path / "parallel", Path(full["video_path"]), store, events, binding, scale, workers=3, clip_preset=clip_preset)
    print(f"synthetic export {clip_preset}: {len(events)} events, serial={serial_seconds:.4f}s, parallel3={perf_counter() - started:.4f}s")
    assert [r["event_id"] for r in parallel] == [e["event_id"] for e in events]
    for left, right in zip(serial, parallel):
        assert left["sha256"] == right["sha256"]
        assert left["verification"]["frame_count"] == right["verification"]["frame_count"]
        assert right["verification"]["decoder_threads"] == 1
        metadata = json.loads((Path(right["path"]) / "metadata.json").read_text())
        assert metadata["provenance"]["encoding_settings"]["preset"] == clip_preset
        assert metadata["provenance"]["encoding_settings"]["crf"] == 21
        for name in ("annotation.json", "metadata.json", "tracks.json"):
            assert json.loads((Path(left["path"]) / name).read_text()) == json.loads((Path(right["path"]) / name).read_text())
        assert len(list(Path(right["path"]).iterdir())) == 4


@pytest.mark.parametrize("workers", [0, -1, True, 1.5])
def test_clip_export_rejects_bad_workers_before_creating_output(tmp_path, workers):
    target = tmp_path / "clips"
    with pytest.raises(ValueError, match="workers"):
        export_packages(target, tmp_path / "absent.mp4", None, [],
                        {"source_video": {"frame_count": 1}}, {}, workers=workers)
    assert not target.exists()


def test_clip_export_rejects_duplicate_event_ids_before_concurrent_writes(tmp_path):
    target = tmp_path / "clips"
    rows = merge_events([event(0, 2), event(3, 5)])
    rows[1]["event_id"] = rows[0]["event_id"]
    with pytest.raises(ValueError, match="duplicate event_id"):
        export_packages(target, tmp_path / "absent.mp4", None, rows,
                        {"source_video": {"frame_count": 5}}, {}, workers=2)
    assert not target.exists()


@pytest.fixture
def completed_full_run(source, tmp_path):
    video, cache, _ = source
    config = Path(__file__).resolve().parents[2] / "configs/default.yaml"
    root = tmp_path / "completed_full"
    receipt = run_pipeline(video, cache, root, config, mode="full",
                           source_receipt=cache.parent / "source_receipt.json")
    return video, root, config, receipt


def test_clips_reuses_verified_full_four_files_without_rendering_again(completed_full_run, tmp_path, monkeypatch):
    video, prior, config, receipt = completed_full_run
    import mouse_behavior.visualization.behavior_packages as packages
    def forbidden(*args, **kwargs):
        raise AssertionError("verified full package must be reused without another render")
    monkeypatch.setattr(packages, "_render_video", forbidden)
    output = tmp_path / "result"
    result = run_pipeline(video, None, output, config, mode="clips", events_from=prior, clip_workers=2)
    assert result["status"] == "complete"
    assert result["full_reused_from"] == str((prior / "full").resolve())
    assert result["full"]["sha256"] == receipt["full"]["sha256"]
    assert (output / "events.json").read_bytes() == (prior / "events.json").read_bytes()
    assert result["clip_count"] == receipt["event_count"] > 0
    assert len(list((output / "full").iterdir())) == 4
    rows = json.loads((output / "manifest.json").read_text())["clips"]
    metadata = json.loads((Path(rows[0]["path"]) / "metadata.json").read_text())
    qualification = metadata["provenance"]["event_qualification"]
    assert qualification["export_unit"] == "standalone_candidate_process_interval"
    assert qualification["standalone_duration_qualification"]["ok"] is True
    assert qualification["encoded_duration_qualification"]["ok"] is True
    assert qualification["core_seconds"] == 20 and qualification["context_seconds"] == 10
    assert qualification["source_window_bounds"]


@pytest.mark.parametrize("payload", ["clip.mp4", "metadata.json"])
def test_full_reuse_rejects_tampered_payload(completed_full_run, tmp_path, payload):
    video, prior, config, _ = completed_full_run
    with (prior / "full" / payload).open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="full.*changed|full.*hash"):
        run_pipeline(video, None, tmp_path / "result", config, mode="clips", events_from=prior)
    assert json.loads((tmp_path / "result/run_receipt.json").read_text())["status"] == "failed"


def test_full_reuse_checks_source_binding_even_if_metadata_hash_is_updated(completed_full_run, tmp_path):
    video, prior, config, _ = completed_full_run
    metadata_path = prior / "full/metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["source"]["sha256"] = "0" * 64
    metadata_path.write_text(json.dumps(metadata))
    receipt_path = prior / "run_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["full"]["sha256"]["metadata.json"] = _sha256(metadata_path)
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="full metadata|source binding"):
        run_pipeline(video, None, tmp_path / "result", config, mode="clips", events_from=prior)


@pytest.mark.parametrize("field,value", [("mode", "unknown"), ("status", "running")])
def test_full_reuse_rejects_unknown_mode_or_unfinished_run(completed_full_run, tmp_path, field, value):
    video, prior, config, _ = completed_full_run
    path = prior / "run_receipt.json"
    receipt = json.loads(path.read_text())
    receipt[field] = value
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="mode|not complete"):
        run_pipeline(video, None, tmp_path / "result", config, mode="clips", events_from=prior)


@pytest.mark.parametrize("clip_preset", ["fast", "", None])
def test_invalid_clip_preset_rejected_before_mutation(tmp_path, clip_preset):
    with pytest.raises(ValueError, match="preset"):
        export_packages(tmp_path / "clips", tmp_path / "absent.mp4", None, [], {}, {}, clip_preset=clip_preset)
    with pytest.raises(ValueError, match="preset"):
        run_pipeline(tmp_path / "absent.mp4", None, tmp_path / "run", tmp_path / "config.yaml", clip_preset=clip_preset)
    assert not (tmp_path / "clips").exists() and not (tmp_path / "run").exists()


@pytest.mark.parametrize("expected_count", [0, 11, 13])
def test_complete_grab_verifier_rejects_zero_short_or_excess_counts(source, expected_count):
    from mouse_behavior.visualization.behavior_packages import _verify_bounded_video
    video, _, _ = source
    with pytest.raises(ValueError):
        _verify_bounded_video(video, expected_count, 10)


def test_complete_grab_verifier_checks_fps_and_decodes_source_first_last(source):
    from mouse_behavior.visualization.behavior_packages import _verify_bounded_video
    video, _, _ = source
    result = _verify_bounded_video(video, 12, 10)
    assert result["frame_count"] == 12
    assert result["decode_method"] == "grab_every_frame_retrieve_first_last"
    assert result["retrieved_frame_indices"] == [0, 11]
    with pytest.raises(ValueError, match="FPS"):
        _verify_bounded_video(video, 12, 11)


@pytest.mark.parametrize("failure", ["dimensions", "last_retrieve"])
def test_complete_grab_verifier_rejects_changed_dimensions_or_unretrievable_last_frame(tmp_path, monkeypatch, failure):
    import mouse_behavior.visualization.behavior_packages as packages
    path = tmp_path / "decoder_fixture.mp4"
    path.write_bytes(b"external decoder boundary fixture")
    class Capture:
        index = -1
        released = False
        def isOpened(self):
            return True
        def get(self, prop):
            if prop == cv2.CAP_PROP_N_THREADS:
                return 1
            if prop == cv2.CAP_PROP_FPS:
                return 10
            if prop == cv2.CAP_PROP_FRAME_HEIGHT:
                return 2
            if prop == cv2.CAP_PROP_FRAME_WIDTH:
                return 5 if failure == "dimensions" and self.index == 1 else 4
            raise AssertionError("unexpected decoder query")
        def grab(self):
            self.index += 1
            return self.index < 3
        def retrieve(self):
            if failure == "last_retrieve" and self.index == 2:
                return False, None
            return True, np.zeros((2, 4, 3), np.uint8)
        def release(self):
            self.released = True
    capture = Capture()
    monkeypatch.setattr(packages.cv2, "VideoCapture", lambda *args: capture)
    with pytest.raises(ValueError, match="dimensions changed|boundary frame"):
        packages._verify_bounded_video(path, 3, 10)
    assert capture.released
