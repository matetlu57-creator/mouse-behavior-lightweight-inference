"""Checks for auditable selection and clip-local video alignment."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from prepare_grooming_rearing_review import (  # noqa: E402
    render_candidate, review_batch_hash, select_batch, write_review_page,
)


def _row(behavior="grooming", clip="sample", track=1, start=4, end=10, hind=1.0):
    return {"candidate": behavior, "clip": clip, "clip_path": clip, "source_folder": "Static",
            "track_id": str(track), "start_frame": str(start), "end_frame": str(end),
            "duration_s": str((end - start) / 10), "mean_head_speed_cm_s": "3.0",
            "mean_hind_speed_cm_s": str(hind), "mean_foreshorten": "0.7",
            "mean_burst": "1.0", "nose_in_torso_fraction": "0.0"}


def test_batch_includes_controls_and_prefers_distinct_tracks():
    rows = [_row(clip="long", end=30), _row(clip="long", start=5, end=29),
            _row(clip="distinct", end=20), _row(clip="control", end=11, hind=9),
            _row("rearing", "rear")]
    selected = select_batch(rows + [dict(rows[0])], per_behavior=3)
    groom = [r for r in selected if r["candidate"] == "grooming"]
    assert [r["clip"] for r in groom] == ["control", "long", "distinct"]
    assert groom[0]["selection_reason"] == "high_hind_speed_control"
    assert all(r["decision"] == "pending" and not r["observed_behavior"] for r in selected)
    assert select_batch(list(reversed(rows)), 3) == selected
    assert "decision" not in rows[0]  # Selection must not mutate source evidence.


@pytest.mark.parametrize("changes", [{"start_frame": "-1"}, {"end_frame": "4"},
                                      {"mean_foreshorten": "nan"}, {"candidate": "attack"}])
def test_invalid_candidates_rejected(changes):
    row = _row()
    row.update(changes)
    with pytest.raises(ValueError):
        select_batch([row])


def test_controls_cannot_be_silently_truncated():
    with pytest.raises(ValueError, match="all high-hind-speed controls"):
        select_batch([_row(clip="a", hind=8), _row(clip="b", hind=7)], 1)


def test_browser_decisions_are_scoped_to_exact_batch():
    rows = select_batch([_row("rearing"), _row("grooming")])
    first = review_batch_hash(rows, "same-csv", 1.0)
    assert first == review_batch_hash(rows, "same-csv", 1.0)
    assert first != review_batch_hash(rows[1:], "same-csv", 1.0)
    assert first != review_batch_hash(list(reversed(rows)), "same-csv", 1.0)
    assert first != review_batch_hash(rows, "same-csv", 0.5)


def test_render_aligns_window_preserves_missing_id_and_refuses_overwrite(tmp_path):
    source = tmp_path / "clip"
    source.mkdir()
    raw = source / "clip.mp4"
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), 10, (96, 96))
    assert writer.isOpened()
    try:
        for i in range(20):
            writer.write(np.full((96, 96, 3), i * 10, dtype=np.uint8))
    finally:
        writer.release()
    points = [[20, 20, 1], [15, 25, 1], [25, 25, 1], [20, 35, 1],
              [15, 50, 1], [25, 50, 1], [20, 60, 1]]
    (source / "tracks.json").write_text(json.dumps([
        {"frame": i, "detections": [] if i == 6 else
         [{"track_id": 1, "box": [10, 10, 40, 70], "keypoints": points}]}
        for i in range(20)]), encoding="utf-8")
    (source / "metadata.json").write_text(json.dumps({"clip": {"fps": 10, "frame_count": 20}}))
    row = select_batch([_row(clip=str(source))])[0]
    output = tmp_path / "output"
    output.mkdir()
    result = render_candidate(row, output, context_s=0.2)
    assert (result["context_start_frame"], result["context_end_frame"]) == (2, 12)
    assert result["candidate_start_s_in_review"] == 0.2
    assert result["candidate_end_s_in_review"] == 0.8
    assert result["rendered_frames"] == 10
    assert result["target_missing_frames"] == 1
    assert not list(output.glob("*.avi"))
    check = cv2.VideoCapture(str(output / result["video_file"]))
    try:
        assert check.get(cv2.CAP_PROP_FRAME_COUNT) == 10
        ok, first = check.read()
        assert ok
        # Raw crop must come from source frame 2 (intensity 20), not frame 0.
        assert 15 < np.median(first[150:400, 800:1170]) < 25
    finally:
        check.release()
    row["clip"] = '</script><script>alert("x")</script>'
    write_review_page([{**result, "clip": row["clip"]}], output, "test-hash")
    page = (output / "index.html").read_text(encoding="utf-8")
    assert '</script><script>alert("x")</script>' not in page
    assert "\\u003c/script>" in page
    assert "localStorage" in page and "review_decisions.csv" in page
    with pytest.raises(FileExistsError):
        render_candidate(row, output)
