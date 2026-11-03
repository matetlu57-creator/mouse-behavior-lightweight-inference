"""Real CFR crops keep playback context separate from qualified event bounds."""
import json
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import pytest

from mouse_behavior.visualization import behavior_packages as packages


class Store:
    def rows(self, start, end):
        return (dict(frame=i, source_frame=i, time=i / 20, detections=[])
                for i in range(start, end))

    def write_tracks(self, path, start, end):
        packages._write_core_tracks(path, self, start, end, 20.)


@pytest.fixture
def media(tmp_path):
    path = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 20., (160, 96))
    assert writer.isOpened()
    for index in range(60):
        writer.write(np.full((96, 160, 3), (index * 4, 20, 240-index * 4), np.uint8))
    writer.release()
    return path


def binding(path):
    return dict(source_video=dict(path=str(path), sha256=packages._sha256(path), fps=20.,
                                 frame_count=60, width=160, height=96),
                cache_status_sha256="synthetic", chunk_sha256={})


def event(start, end):
    return dict(event_id=f"event_{start}", behavior="Locomotion", canonical_behavior="locomotion",
                mouse_ids=[1], start_frame=start, end_frame=end,
                interaction_status="interacting", interaction_semantics="synthetic review evidence",
                measured_frame_intervals=[[start+1, end-1]])


def colors(path):
    capture = cv2.VideoCapture(str(path))
    result = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        result.append(frame.mean(axis=(0, 1)))
    capture.release()
    return np.asarray(result)


@pytest.mark.parametrize("start,end,left,right", [(20, 32, 15, 37), (0, 12, 0, 17), (48, 60, 43, 60)])
@pytest.mark.parametrize("raw", [False, True])
def test_context_crop_exact_frames_offsets_and_edges(media, tmp_path, start, end, left, right, raw):
    row = packages.export_packages(tmp_path / "clips", media, Store(), [event(start, end)],
                                   binding(media), {}, clip_context_seconds=.25, original_video=raw)[0]
    package = Path(row["path"])
    assert {p.name for p in package.iterdir()} == set(packages.PAYLOADS)
    reference, actual = colors(media), colors(package / "clip.mp4")
    assert [int(np.argmin(np.linalg.norm(reference-color, axis=1))) for color in actual] == list(range(left, right))
    annotation = json.loads((package / "annotation.json").read_text())
    assert annotation["frame_range"] == dict(start=start-left, end=end-left-1)
    assert annotation["time_range"] == dict(start=(start-left)/20, end=(end-left)/20)
    assert annotation["measured_frame_intervals"] == [[start+1-left, end-1-left]]
    assert annotation["interaction_status"] == "interacting"
    assert annotation["interaction_semantics"] == "synthetic review evidence"
    assert row["interaction_status"] == "interacting"
    assert row["interaction_semantics"] == "synthetic review evidence"
    metadata = json.loads((package / "metadata.json").read_text())
    assert metadata["source"]["start_frame"] == left
    assert metadata["source"]["end_frame"] == right-1
    provenance = metadata["provenance"]
    assert provenance["crop_half_open"] == dict(start=left, end=right)
    assert provenance["event_qualification"]["source_event_half_open"] == dict(start=start, end=end)
    qualification = provenance["event_qualification"]["encoded_duration_qualification"]
    assert qualification["actual_s"] == pytest.approx((end-start)/20)
    assert qualification["event_frame_count"] == end-start
    assert row["frame_count"] == right-left
    tracks = json.loads((package / "tracks.json").read_text())
    assert len(tracks) == right-left
    assert all(set(track) == {"frame", "time", "detections"} for track in tracks)
    assert tracks[0]["frame"] == 0 and tracks[0]["time"] == 0
    assert tracks[-1]["frame"] == right-left-1
    assert tracks[-1]["time"] == pytest.approx((right-left-1)/20)


def test_context_cannot_qualify_short_event(media, tmp_path):
    with pytest.raises(ValueError, match="duration rejected"):
        packages.export_packages(tmp_path / "clips", media, Store(), [event(20, 29)],
                                 binding(media), {}, clip_context_seconds=10, original_video=True)
    assert not (tmp_path / "clips").exists()


def test_encoded_fps_rechecks_true_event_not_long_crop(media, tmp_path, monkeypatch):
    monkeypatch.setattr(packages.subprocess, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(packages, "_verify_bounded_video", lambda *args: dict(
        frame_count=20, fps=20.0001, width=160, height=96, duration_s=20/20.0001))
    with pytest.raises(ValueError, match="duration rejected"):
        packages.export_packages(tmp_path / "clips", media, Store(), [event(20, 30)],
                                 binding(media), {}, clip_context_seconds=.25)
    assert not list((tmp_path / "clips").rglob("annotation.json"))


def test_raw_audio_is_aac_and_starts_at_crop_time(media, tmp_path):
    source = tmp_path / "audio.mp4"
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    # Ramp frequency distinguishes crop start from the original start and from
    # the half-frame input seek margin. Verify the decoded waveform directly.
    expression = "aevalsrc=sin(2*PI*(300*t+100*t*t)):s=48000:d=3"
    subprocess.run([ffmpeg, "-v", "error", "-i", str(media), "-f", "lavfi", "-i", expression,
                    "-c:v", "copy", "-c:a", "aac", "-shortest", str(source)], check=True)
    row = packages.export_packages(tmp_path / "clips", source, Store(), [event(20, 32)],
                                   binding(source), {}, clip_context_seconds=.25, original_video=True)[0]
    clip = row["video_path"]
    probe = subprocess.run([ffmpeg, "-hide_banner", "-i", clip], capture_output=True)
    assert "Audio: aac" in probe.stderr.decode(errors="replace")
    def audio(path, seek=None):
        command = [ffmpeg, "-v", "error", "-i", str(path)]
        if seek is not None:
            command += ["-ss", str(seek)]
        return np.frombuffer(subprocess.run(command + ["-map", "0:a:0", "-t", "0.6", "-f", "f32le", "-ac", "1", "-ar", "48000", "pipe:1"],
                                            check=True, capture_output=True).stdout, dtype=np.float32)
    expected = audio(source, .75)
    actual = audio(clip)
    assert len(actual) == len(expected)
    assert np.corrcoef(actual[2048:], expected[2048:])[0, 1] > .97
    all_audio = subprocess.run([ffmpeg, "-v", "error", "-i", clip, "-map", "0:a:0",
                                "-f", "f32le", "-ac", "1", "-ar", "48000", "pipe:1"],
                               check=True, capture_output=True).stdout
    assert abs(len(all_audio) / 4 / 48000 - 1.1) < .05


@pytest.mark.parametrize("context", [-1, float("nan"), float("inf")])
def test_invalid_context_rejected_before_output(media, tmp_path, context):
    with pytest.raises(ValueError, match="context"):
        packages.export_packages(tmp_path / "clips", media, None, [], binding(media), {}, clip_context_seconds=context)
    assert not (tmp_path / "clips").exists()



@pytest.mark.parametrize("raw", [False, True])
def test_approach_has_precontext_and_zero_trailing_context(media, tmp_path, raw):
    approach = dict(event(20, 42), behavior="Approach", canonical_behavior="approach")
    row = packages.export_packages(tmp_path / "approach", media, Store(), [approach],
        binding(media), {}, clip_context_seconds=.25, original_video=raw)[0]
    package = Path(row["path"])
    reference, actual = colors(media), colors(package / "clip.mp4")
    assert [int(np.argmin(np.linalg.norm(reference-color, axis=1)))
            for color in actual] == list(range(15, 42))
    annotation = json.loads((package / "annotation.json").read_text())
    assert annotation["frame_range"] == dict(start=5, end=26)
    assert annotation["time_range"] == dict(start=.25, end=1.35)
    provenance = json.loads((package / "metadata.json").read_text())["provenance"]
    assert provenance["crop_half_open"] == dict(start=15, end=42)
    assert provenance["clip_context_seconds_requested"] == .25
    assert provenance["clip_context_before_seconds_requested"] == .25
    assert provenance["clip_context_before_seconds_actual"] == .25
    assert provenance["clip_context_after_seconds_requested"] == 0.
    assert provenance["clip_context_after_seconds_actual"] == 0.
