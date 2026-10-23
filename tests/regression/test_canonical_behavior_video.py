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


@pytest.mark.parametrize("relabel", [False, True])
def test_canonical_package_recrop_refreshes_semantics_and_media(source, tmp_path, relabel):
    from mouse_behavior.evaluation.exported_dataset import discover_samples
    from mouse_behavior.evaluation.segment_export import export_segment
    video, cache, _ = source
    binding, scale = bind_cache(video, cache, tmp_path / "tracks.sqlite",
                                source_receipt=cache.parent / "source_receipt.json")
    store = TrackStore(tmp_path / "tracks.sqlite")
    events = merge_events([event(2, 9)])
    events[0].update(role_support_frame_intervals=[[3, 8]], role_support_frame_count=5,
                     role_status="event_direction_with_measured_support")
    full = render_full_package(video, tmp_path / "full", store, events, binding, scale)
    export_packages(tmp_path / "packages", Path(full["video_path"]), store, events, binding, scale)
    record = discover_samples(tmp_path / "packages")[0]
    annotation_path = Path(record.annotation_path)
    annotation = json.loads(annotation_path.read_text())
    annotation["future_semantic_field"] = {"unrecognized": True}
    annotation_path.write_text(json.dumps(annotation))
    result = export_segment(record, tmp_path / "recrop", start_frame=2, end_frame=5,
                            behavior="Static" if relabel else "Approach",
                            mouse_ids=(1,) if relabel else (1, 99), provenance={})
    annotation = json.loads((tmp_path / "recrop/annotation.json").read_text())
    metadata = json.loads((tmp_path / "recrop/metadata.json").read_text())
    assert metadata["clip"]["decoded_duration_s"] == pytest.approx(.3)
    assert metadata["clip"]["encoded_fps"] == result["verification"]["fps"]
    assert "future_semantic_field" not in annotation
    assert "event_id" not in annotation
    assert metadata["provenance"]["original_annotation"]["event_id"] == events[0]["event_id"]
    if relabel:
        assert not {"canonical_behavior", "actor_id", "target_id", "role_status",
                    "role_support_frame_intervals", "role_support_frame_count"}.intersection(annotation)
    else:
        assert annotation["canonical_behavior"] == "approach"
        assert annotation["actor_id"] == 99 and annotation["target_id"] == 1
        assert annotation["role_support_frame_intervals"] == [[0, 3]]
        assert annotation["role_support_frame_count"] == 3


def test_showcase_skips_source_shorter_than_requested_window(source, tmp_path):
    from mouse_behavior.visualization.rendering import extract_behavior_clips
    video, _, _ = source
    events = tmp_path / "events.csv"
    events.write_text("start_frame,end_frame,candidate_level,behavior\n0,11,individual,stationary\n")
    output = tmp_path / "showcase"
    extract_behavior_clips(video, events, output, clip_seconds=5)
    summary = json.loads((output / "behavior_clip_summary.json").read_text())
    assert summary["total_clips"] == 0
    assert summary["insufficient_source_policy"] == "skip_incomplete_fixed_window"
    assert not list(output.rglob("*.mp4"))


@pytest.mark.parametrize("four_classes", [False, True])
def test_showcase_early_eof_leaves_no_partial_clips(source, tmp_path, monkeypatch, four_classes):
    from mouse_behavior.visualization import rendering
    video, _, _ = source
    events = tmp_path / "events.csv"
    events.write_text("start_frame,end_frame,candidate_level,behavior\n0,11,strong,attack\n")
    original = cv2.VideoCapture

    class EarlyEOF:
        def __init__(self, *args):
            self.capture = original(*args)
            self.reads = 0

        def __getattr__(self, name):
            return getattr(self.capture, name)

        def read(self):
            self.reads += 1
            return self.capture.read() if self.reads <= 5 else (False, None)

    monkeypatch.setattr(rendering.cv2, "VideoCapture", EarlyEOF)
    output = tmp_path / "showcase"
    extractor = rendering.extract_four_class_clips if four_classes else rendering.extract_behavior_clips
    with pytest.raises((ValueError, RuntimeError), match="frames|decode"):
        extractor(video, events, output, clip_seconds=.5)
    assert not list(output.rglob("*.mp4"))
    assert not list(output.glob("*manifest.csv"))


@pytest.mark.parametrize("provider,field", [
    ("social_heuristic_config", "approach_min_duration_s"),
    ("group_heuristic_config", "clustering_final_stability_s"),
])
def test_rerender_rejects_runtime_duration_policy_drift(source, tmp_path, monkeypatch, provider, field):
    from dataclasses import replace
    import importlib
    module = importlib.import_module("mouse_behavior.evaluation." + provider)
    name = "DEFAULT_SOCIAL_HEURISTIC_CONFIG" if provider.startswith("social") else "DEFAULT_GROUP_HEURISTIC_CONFIG"
    video, cache, _ = source
    config = Path(__file__).resolve().parents[2] / "configs/default.yaml"
    run_pipeline(video, cache, tmp_path / "original", config, mode="infer",
                 source_receipt=cache.parent / "source_receipt.json")
    monkeypatch.setattr(module, name, replace(getattr(module, name), **{field: .6}))
    with pytest.raises(ValueError, match="policy|snapshot"):
        run_pipeline(video, None, tmp_path / "drift", config, mode="full",
                     events_from=tmp_path / "original")
    assert not (tmp_path / "drift/full").exists()


@pytest.mark.parametrize("change", ["legacy", "source_hash", "policy"])
def test_rerender_rejects_incompatible_sealed_rule_binding(source, tmp_path, change):
    video, cache, _ = source
    config = Path(__file__).resolve().parents[2] / "configs/default.yaml"
    original = tmp_path / "original"
    run_pipeline(video, cache, original, config, mode="infer", source_receipt=cache.parent / "source_receipt.json")
    path = original / "input_binding.json"
    binding = json.loads(path.read_text())
    if change == "legacy":
        binding.pop("rule_runtime_snapshot")
    elif change == "source_hash":
        binding["rule_source_sha256"]["mouse_behavior.evaluation.pair_roles"] = "0" * 64
    else:
        binding["rule_policy"] = {"old_policy": True}
    path.write_text(json.dumps(binding))
    receipt_path = original / "run_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["payload_sha256"]["input_binding.json"] = _sha256(path)
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="snapshot"):
        run_pipeline(video, None, tmp_path / "rejected", config, mode="full", events_from=original)
    assert not (tmp_path / "rejected/full").exists()


def test_policy_snapshot_uses_portable_closure_and_full_temporal_defaults():
    from mouse_behavior.canonical_video_pipeline import rule_execution_binding
    from mouse_behavior.evaluation.exported_dataset_models import HeuristicParameters
    from mouse_behavior.behavior.posture_rules import DEFAULT_GROOMING_TEMPORAL
    from dataclasses import asdict
    binding = rule_execution_binding(HeuristicParameters(), "v3_review")
    hashes = binding["rule_source_sha256"]
    assert all(key.startswith("mouse_behavior.") and ":" not in key and "/" not in key for key in hashes)
    assert "mouse_behavior.evaluation.pair_roles" in hashes
    defaults = binding["rule_runtime_snapshot"]["module_defaults"]
    assert defaults["mouse_behavior.behavior.posture_rules"]["DEFAULT_GROOMING_TEMPORAL"] == asdict(DEFAULT_GROOMING_TEMPORAL)


def test_core_keeps_dispersal_endpoint_confirmation_from_context(tmp_path):
    import sqlite3
    from mouse_behavior.evaluation.exported_dataset_config import config_seed_parameters
    database = tmp_path / "dispersal.sqlite"
    fps, count = 10., 234
    shape = np.array([[3.959, 0], [2.5, .8], [2.5, -.8], [1, 0],
                      [-2, 1], [-2, -1], [-3.959, 0]])
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE frames(frame INTEGER PRIMARY KEY,payload TEXT)")
        for frame in range(count):
            time = frame / fps
            x = 2 * min(time, 5) + .12 * max(time - 5, 0)
            spread = .5 + max(time - 20, 0)
            detections = []
            for track, center in ((100, (x, -spread)), (101, (x, 0)), (102, (x, spread)), (103, (0, 2))):
                points = (shape + center) * 4
                detections.append(dict(track_id=track, box=[*points.min(axis=0), *points.max(axis=0)],
                                       keypoints=[[*points[index], 1.] for index in (0, 2, 1, 3, 4, 5, 6)],
                                       detection_confidence=1., class_id=0))
            row = dict(frame=frame, time=time, source_frame=frame, source_fps=fps, detections=detections)
            connection.execute("INSERT INTO frames VALUES(?,?)", (frame, json.dumps(row)))
    params = config_seed_parameters({})
    scale = dict(reference_body_cm=7.918, cm_per_pixel=.25)
    store = TrackStore(database)
    parent, _ = infer_window_events(store.window(0, count), fps, scale, params)
    dispersal = [event for event in parent if event["canonical_behavior"] == "dispersal"]
    assert dispersal and any(event["end_frame"] == count for event in dispersal)
    result, _ = infer_video_events(store, {"source_video": {"fps": fps, "frame_count": count}}, scale, params)
    children = [event for event in result if event["canonical_behavior"] == "dispersal"]
    assert children and all(event["end_frame"] == count for event in children)
    assert any(event.get("retained_complete_parent_for_semantics") for event in children)


@pytest.mark.parametrize("mice", [1, 3])
def test_lightweight_analysis_produces_source_cache_and_event_seal(source, tmp_path, mice):
    import yaml
    from mouse_behavior.config import load_config
    from mouse_behavior.lightweight_behavior_inference import analyze
    from mouse_behavior.pose_cache import verified_cache_source_binding
    from mouse_behavior.visualization.rendering import render_behavior_video
    video, cache, rows = source
    for row in rows:
        row["pose_detections"] = row["pose_detections"][:mice]
    with gzip.open(cache / "yolo_results.000000.000000.pkl.gz", "wb") as stream:
        pickle.dump(rows, stream)
    status_path = cache / "yolo_results_status.json"
    status = json.loads(status_path.read_text())
    status.update(source_video_sha256=_sha256(video),
                  chunk_sha256={path.name: _sha256(path) for path in cache.glob("*.pkl.gz")})
    status_path.write_text(json.dumps(status))
    config = load_config(Path(__file__).resolve().parents[2] / "configs/default.yaml")
    config["adaptive_arena"]["enabled"] = False
    config_path = tmp_path / "analysis_config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    output = analyze(video, cache, config_path, tmp_path / "analysis", expected_mice=mice)
    metadata = json.loads((output / "lightweight_analysis_metadata.json").read_text(encoding="utf-8"))
    assert metadata["source_fps"] == 10
    assert metadata["source_binding"] == verified_cache_source_binding(video, cache, fps=10, frame_count=12)
    for name, digest in metadata["event_files_sha256"].items():
        assert digest == _sha256(output / name)
    render_behavior_video(video, cache, output / "lightweight_behavior_events.csv", tmp_path / "sealed.mp4", expected_mice=mice)
    receipt = json.loads((tmp_path / "sealed.identity.json").read_text())
    assert receipt["source_video_sha256"] == _sha256(video)
    assert receipt["source_binding_level"] == "sha256_and_path_timebase"


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
        module = "mouse_behavior.behavior." + Path(filename).stem
        path = Path(__file__).resolve().parents[2] / "src/mouse_behavior/behavior" / filename
        assert sources[module] == _sha256(path)
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
