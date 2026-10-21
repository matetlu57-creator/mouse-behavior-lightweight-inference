from __future__ import annotations

from mouse_behavior.identity.mtrack import load_identity_map


def test_load_identity_map_supports_frame_ranges(tmp_path):
    path = tmp_path / "identity_map.csv"
    path.write_text(
        "start_frame,end_frame,visual_track_id,animal_id,identity_confidence\n"
        "0,4,10,mouse-A,0.98\n"
        "5,9,10,mouse-B,0.91\n",
        encoding="utf-8",
    )

    identity_map = load_identity_map(path)

    first = identity_map.resolve(frame=2, visual_track_id=10)
    second = identity_map.resolve(frame=7, visual_track_id=10)
    assert first is not None
    assert first.animal_id == "mouse-A"
    assert first.confidence == 0.98
    assert second is not None
    assert second.animal_id == "mouse-B"
    assert identity_map.resolve(frame=10, visual_track_id=10) is None


def test_load_identity_map_rejects_overlapping_conflicts(tmp_path):
    path = tmp_path / "identity_map.csv"
    path.write_text(
        "start_frame,end_frame,visual_track_id,animal_id\n"
        "0,5,10,mouse-A\n"
        "5,8,10,mouse-B\n",
        encoding="utf-8",
    )

    try:
        load_identity_map(path)
    except ValueError as exc:
        assert "overlap" in str(exc).lower() or "冲突" in str(exc)
    else:  # pragma: no cover - makes the regression assertion explicit
        raise AssertionError("conflicting identity intervals must be rejected")


def test_same_animal_nested_intervals_remain_resolvable(tmp_path):
    path = tmp_path / "identity_map.csv"
    path.write_text(
        "start_frame,end_frame,visual_track_id,animal_id,identity_confidence\n"
        "0,20,10,mouse-A,0.80\n"
        "5,7,10,mouse-A,0.99\n",
        encoding="utf-8",
    )

    identity_map = load_identity_map(path)

    assert identity_map.resolve(frame=6, visual_track_id=10).confidence == 0.99
    assert identity_map.resolve(frame=15, visual_track_id=10).animal_id == "mouse-A"
