from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load_lightweight():
    path = ROOT / "lightweight_behavior_inference.py"
    spec = importlib.util.spec_from_file_location("lightweight_contact_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def contact_frame(frame: int, head: float, tail: float) -> dict[str, object]:
    return {
        "frame": frame,
        "valid_pair": True,
        "mouse_a_id": 1,
        "mouse_b_id": 2,
        "a_to_b_nose_head_distance_cm": head,
        "a_to_b_nose_tail_distance_cm": tail,
        "b_to_a_nose_head_distance_cm": float("inf"),
        "b_to_a_nose_tail_distance_cm": float("inf"),
    }


def test_nose_head_and_nose_tail_contacts_are_separate_events():
    lightweight = load_lightweight()
    pair_df = pd.DataFrame(
        [
            contact_frame(0, 2.0, 5.0),
            contact_frame(1, 2.2, 5.1),
            contact_frame(2, 5.0, 2.0),
            contact_frame(3, 5.1, 2.1),
            contact_frame(4, 8.0, 8.0),
        ]
    )
    events = lightweight._extract_contact_events(
        pair_df,
        pair_key="1_2",
        source_video=Path("contact.mp4"),
        source_fps=30.0,
        sample_stride=1,
        contact_config={
            "enabled": True,
            "nose_head_distance_cm": 3.0,
            "nose_tail_distance_cm": 3.0,
        },
    )

    assert [event["contact_type"] for event in events] == ["nose_head", "nose_tail"]
    assert [(event["start_frame"], event["end_frame"]) for event in events] == [
        (0, 1),
        (2, 3),
    ]
    assert all(event["contact_actor_id"] == 1 for event in events)
    assert all(event["contact_target_id"] == 2 for event in events)


def test_simultaneous_head_and_tail_contact_keeps_both_components():
    lightweight = load_lightweight()
    pair_df = pd.DataFrame([contact_frame(4, 2.0, 2.0)])
    events = lightweight._extract_contact_events(
        pair_df,
        pair_key="1_2",
        source_video=Path("contact.mp4"),
        source_fps=30.0,
        sample_stride=1,
        contact_config={"enabled": True},
    )

    assert len(events) == 1
    assert events[0]["contact_type"] == "nose_head_and_nose_tail"
    assert events[0]["contact_type_components"] == "nose_head;nose_tail"
