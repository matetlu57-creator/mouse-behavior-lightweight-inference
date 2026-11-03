"""Event-only overlays retain complete process bounds and observed visual IDs."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import cv2
import numpy as np
import pytest

from mouse_behavior.canonical_video_pipeline import TrackStore
from mouse_behavior.visualization import behavior_packages as packages


IDS = (1, 7, 99, 1000)


def detections():
    result = []
    for track, left in zip(IDS, (35, 185, 335, 485)):
        points = [[left + 70, 220, 1.], [left + 60, 210, 1.],
                  [left + 60, 230, 1.], [left + 50, 220, 1.],
                  [left + 25, 210, 1.], [left + 25, 230, 1.], [left + 10, 220, 1.]]
        result.append(dict(track_id=track, box=[left, 185, left + 90, 255],
                           keypoints=points, box_confidence=.91))
    return result


def event(start=10, end=130, *, behavior="Huddle", canonical="huddle", ids=IDS[:3]):
    return dict(event_id=f"{canonical}_{start}_{end}", behavior=behavior,
                canonical_behavior=canonical, mouse_ids=list(ids),
                start_frame=start, end_frame=end)


@pytest.mark.parametrize("source_frame,active", [(9, False), (10, True), (129, True), (130, False)])
def test_social_highlights_three_measured_participants_only_in_half_open_event(monkeypatch, source_frame, active):
    green = []
    rectangle = packages.cv2.rectangle

    def spy(frame, start, end, color, thickness, *args, **kwargs):
        if color == (0, 255, 0) and thickness == 3:
            green.append((start, end))
        return rectangle(frame, start, end, color, thickness, *args, **kwargs)

    monkeypatch.setattr(packages.cv2, "rectangle", spy)
    packages.draw_overlay(np.zeros((360, 640, 3), np.uint8), dict(detections=detections()),
                          [event(behavior="Social", canonical="social_clustering")], source_frame)
    expected = [((row["box"][0], 185), (row["box"][2], 255)) for row in detections()[:3]]
    assert green == (expected if active else [])


def test_missing_participant_is_reported_without_fabricated_box(monkeypatch):
    text, green = [], []
    put_text, rectangle = packages.cv2.putText, packages.cv2.rectangle

    def text_spy(frame, value, *args, **kwargs):
        text.append(value)
        return put_text(frame, value, *args, **kwargs)

    def box_spy(frame, start, end, color, thickness, *args, **kwargs):
        if color == (0, 255, 0) and thickness == 3:
            green.append((start, end))
        return rectangle(frame, start, end, color, thickness, *args, **kwargs)

    monkeypatch.setattr(packages.cv2, "putText", text_spy)
    monkeypatch.setattr(packages.cv2, "rectangle", box_spy)
    packages.draw_overlay(np.zeros((360, 640, 3), np.uint8), dict(detections=detections()),
                          [event(ids=(1, 777))], 10)
    assert any("V777 not detected" in value for value in text)
    assert green == [((35, 185), (125, 255))]


@pytest.fixture
def media(tmp_path):
    path = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10., (640, 360))
    assert writer.isOpened()
    try:
        for index in range(150):
            writer.write(np.full((360, 640, 3), (30 + index, 20, 210 - index), np.uint8))
    finally:
        writer.release()
    rows = [dict(frame=index, source_frame=index, time=index / 10., source_fps=10.,
                 detections=detections()) for index in range(150)]
    database = tmp_path / "tracks.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE frames(frame INTEGER PRIMARY KEY,payload TEXT)")
        connection.executemany("INSERT INTO frames VALUES(?,?)",
                               [(index, json.dumps(row)) for index, row in enumerate(rows)])
    # Deliberately retain the legacy fixture binding without a rule snapshot.
    binding = dict(source_video=dict(path=str(path), sha256=packages._sha256(path),
                                    fps=10., frame_count=150, width=640, height=360),
                   cache_status_sha256="synthetic", chunk_sha256={})
    return path, TrackStore(database), binding, rows


def boundary_frames(path):
    capture = cv2.VideoCapture(str(path))
    first = last = None
    count = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if first is None:
                first = frame.copy()
            last = frame
            count += 1
    finally:
        capture.release()
    assert first is not None and last is not None
    return count, first, last


def has_green_box(frame, box):
    left, top, right, _ = box
    pixels = frame[top - 2:top + 3, left + 5:right - 5].astype(int)
    return np.count_nonzero((pixels[:, :, 1] > 150)
                            & (pixels[:, :, 1] > pixels[:, :, 0] + 70)
                            & (pixels[:, :, 1] > pixels[:, :, 2] + 70)) > 20


def test_complete_12s_huddle_has_boundary_highlights_and_distinct_four_file_variants(media, tmp_path, monkeypatch):
    source, store, binding, source_rows = media
    huddle = event()
    other = event(20, 30, behavior="Locomotion", canonical="locomotion", ids=(1000,))
    rendered_frames, commands = [], []
    draw, popen = packages.draw_overlay, packages.subprocess.Popen

    def draw_spy(frame, row, events, source_frame):
        rendered_frames.append((source_frame, [item["event_id"] for item in events]))
        return draw(frame, row, events, source_frame)

    def popen_spy(command, *args, **kwargs):
        commands.append(command)
        return popen(command, *args, **kwargs)

    monkeypatch.setattr(packages, "draw_overlay", draw_spy)
    monkeypatch.setattr(packages.subprocess, "Popen", popen_spy)
    outputs = packages.export_packages(tmp_path / "overlay", tmp_path / "nonexistent_full_overlay.mp4",
                                       store, [huddle, other], binding, {}, target_overlay=True,
                                       clip_context_seconds=0., clip_preset="ultrafast")
    assert rendered_frames == ([(index, [huddle["event_id"]]) for index in range(10, 130)]
                               + [(index, [other["event_id"]]) for index in range(20, 30)])
    encoding_commands = [command for command in commands if "-c:v" in command]
    assert len(encoding_commands) == 2
    for command in encoding_commands:
        assert command[command.index("-c:v") + 1] == "libx264"
        assert command[command.index("-preset") + 1] == "ultrafast"
    for output, current, bounds in zip(outputs, (huddle, other), ((10, 130), (20, 30))):
        package = Path(output["path"])
        assert {path.name for path in package.iterdir()} == set(packages.PAYLOADS)
        count, first, last = boundary_frames(package / "clip.mp4")
        assert count == bounds[1] - bounds[0]
        for frame in (first, last):
            assert [has_green_box(frame, row["box"]) for row in detections()] == [
                track in current["mouse_ids"] for track in IDS]
        tracks = json.loads((package / "tracks.json").read_text())
        assert len(tracks) == count
        assert all(set(row) == {"frame", "time", "detections"} for row in tracks)
        assert [row["detections"] for row in tracks] == [
            row["detections"] for row in source_rows[bounds[0]:bounds[1]]]
        assert tracks[0]["frame"] == 0 and tracks[-1]["frame"] == count - 1
    package = Path(outputs[0]["path"])
    annotation = json.loads((package / "annotation.json").read_text())
    metadata = json.loads((package / "metadata.json").read_text())
    provenance = metadata["provenance"]
    assert annotation["frame_range"] == dict(start=0, end=119)
    assert annotation["boundary"]["event_duration_s"] == 12.
    assert outputs[0]["context_before_frames"] == outputs[0]["context_after_frames"] == 0
    assert metadata["clip"]["decoded_duration_s"] == 12.
    assert provenance["video_variant"] == "frame_exact_event_target_overlay"
    qualification = provenance["event_qualification"]["encoded_duration_qualification"]
    assert qualification["required_s"] == 5. and qualification["actual_s"] == 12.
    assert provenance["target_overlay"]["target_visual_track_ids"] == list(IDS[:3])
    assert "only actual cached detections" in provenance["target_overlay"]["highlight_basis"]
    assert provenance["encoding_settings"] == dict(codec="libx264", preset="ultrafast", crf=21,
                                                  pixel_format="yuv420p", encoder_threads=1)
    raw = packages.export_packages(tmp_path / "raw", source, store, [huddle], binding, {},
                                   original_video=True)[0]
    raw_package = Path(raw["path"])
    assert {path.name for path in raw_package.iterdir()} == set(packages.PAYLOADS)
    raw_metadata = json.loads((raw_package / "metadata.json").read_text())
    assert raw_metadata["provenance"]["video_variant"] == "frame_exact_cut_of_original_video"
    assert "target_overlay" not in raw_metadata["provenance"]
    assert (raw_package / "tracks.json").read_bytes() == (package / "tracks.json").read_bytes()
    count, raw_first, raw_last = boundary_frames(raw_package / "clip.mp4")
    assert count == 120
    _, overlay_first, overlay_last = boundary_frames(package / "clip.mp4")
    for raw_frame, overlay_frame in zip((raw_first, raw_last), (overlay_first, overlay_last)):
        assert not any(has_green_box(raw_frame, row["box"]) for row in detections())
        np.testing.assert_allclose(raw_frame[300:, 580:].mean(axis=(0, 1)),
                                   overlay_frame[300:, 580:].mean(axis=(0, 1)), atol=4)


@pytest.mark.parametrize("options,match", [
    (dict(target_overlay=True, original_video=True), "mutually exclusive"),
    (dict(target_overlay=1), "target_overlay must be a boolean"),
    (dict(target_overlay="true"), "target_overlay must be a boolean"),
    (dict(target_overlay=True, original_video=1), "original_video must be a boolean"),
    (dict(target_overlay=True, clip_context_seconds=True), "context"),
    (dict(target_overlay=True, clip_codec="invalid"), "clip codec"),
    (dict(target_overlay=True, clip_preset="invalid"), "clip preset"),
])
def test_invalid_overlay_options_are_rejected_before_output(tmp_path, options, match):
    output = tmp_path / "clips"
    with pytest.raises(ValueError, match=match):
        packages.export_packages(output, tmp_path / "unused.mp4", None, [], {}, {}, **options)
    assert list(tmp_path.iterdir()) == []


def test_overlay_rejects_changed_bound_original_before_output(media, tmp_path):
    source, store, binding, _ = media
    binding["source_video"]["sha256"] = "0" * 64
    output = tmp_path / "clips"
    with pytest.raises(ValueError, match="SHA-256"):
        packages.export_packages(output, source, store, [event()], binding, {}, target_overlay=True)
    assert not output.exists()
