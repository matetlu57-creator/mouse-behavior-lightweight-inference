from __future__ import annotations

import numpy as np

from mouse_behavior.visualization.overlay import (
    build_mouse_overlays,
    build_panel_lines,
    draw_behavior_sidebar,
    focus_behavior_name_zh,
    normalize_focus_behavior,
    normalize_contact_events,
    sidebar_width_for_frame,
    select_display_events,
)


def _event(behavior: str, *, actor: int = -1, target: int = -1, pair: str = "") -> dict:
    return {
        "behavior": behavior,
        "actor_id": actor,
        "target_id": target,
        "pair_key": pair,
        "start_frame": 0,
        "end_frame": 5,
        "peak_score": 1.0,
    }


def test_social_event_is_scoped_to_participants_and_keeps_individual_behavior() -> None:
    selected, layer = select_display_events(
        [
            _event("running", actor=0, pair="mouse_0"),
            _event("approach", actor=1, target=2, pair="1_2"),
        ]
    )

    assert layer == "mixed"
    assert {event["behavior"] for event in selected} == {"approach", "running"}
    overlays = build_mouse_overlays(selected, layer, [0, 1, 2])
    assert overlays[0].text == "ID 00｜奔跑"
    assert overlays[1].text == "ID 01｜主动接近"
    assert overlays[2].text == "ID 02｜被接近"
    assert "按参与ID分别显示" in build_panel_lines(selected, layer)[0]


def test_group_event_shows_specific_behavior_for_known_members() -> None:
    selected, layer = select_display_events(
        [
            _event("stationary", actor=0, pair="mouse_0"),
            {
                **_event("isolation", pair="group"),
                "member_ids": [1, 2],
            },
        ]
    )

    assert layer == "mixed"
    assert {event["behavior"] for event in selected} == {"isolation", "stationary"}
    overlays = build_mouse_overlays(selected, layer, [0, 1, 2])
    assert overlays[0].text == "ID 00｜静止"
    assert overlays[1].text == "ID 01｜群体：孤立"
    assert overlays[2].text == "ID 02｜群体：孤立"
    assert "孤立" in build_panel_lines(selected, layer)[1]


def test_attack_semantic_priority_beats_contact_score() -> None:
    contact = normalize_contact_events(
        [
            {
                "contact_type": "nose_head",
                "contact_actor_id": 1,
                "contact_target_id": 2,
                "pair_key": "1_2",
            }
        ]
    )
    selected, layer = select_display_events(
        [
            _event(
                "attack",
                actor=1,
                target=2,
                pair="1_2",
            ),
            *contact,
        ]
    )

    overlays = build_mouse_overlays(selected, layer, [1, 2])

    assert layer == "social"
    assert overlays[1].text == "ID 01｜攻击"
    assert overlays[2].text == "ID 02｜被攻击"


def test_individual_layer_labels_every_tracked_id_even_when_no_event_is_active() -> None:
    selected, layer = select_display_events([])

    overlays = build_mouse_overlays(selected, layer, [0, 3])

    assert layer == "none"
    assert overlays[0].text == "ID 00｜仅追踪"
    assert overlays[3].text == "ID 03｜仅追踪"
    assert "仅追踪" in build_panel_lines(selected, layer)[1]


def test_contact_events_are_adapted_to_directional_social_labels() -> None:
    events = normalize_contact_events(
        [
            {
                "contact_type": "nose_head",
                "contact_actor_id": 4,
                "contact_target_id": 5,
                "pair_key": "4_5",
                "start_frame": 1,
                "end_frame": 2,
            }
        ]
    )

    selected, layer = select_display_events(events)
    overlays = build_mouse_overlays(selected, layer, [4, 5])

    assert layer == "social"
    assert len(selected) == 1
    assert overlays[4].text == "ID 04｜鼻头接触"
    assert overlays[5].text == "ID 05｜鼻头接触"


def test_individual_behavior_is_visible_when_no_social_or_group_event_is_active() -> None:
    selected, layer = select_display_events([_event("running", actor=0, pair="mouse_0")])

    overlays = build_mouse_overlays(selected, layer, [0, 1])

    assert layer == "individual"
    assert overlays[0].text == "ID 00｜奔跑"
    assert overlays[1].text == "ID 01｜仅追踪"


def test_sidebar_is_appended_without_changing_source_frame_dimensions() -> None:
    frame = np.zeros((1080, 2044, 3), dtype=np.uint8)
    selected, layer = select_display_events([_event("approach", actor=1, target=2, pair="1_2")])
    overlays = build_mouse_overlays(selected, layer, [0, 1, 2])

    rendered = draw_behavior_sidebar(
        frame,
        frame_index=184,
        total_frames=199,
        fps=30.0,
        active_event_count=1,
        display_events=selected,
        display_layer=layer,
        mouse_overlays=overlays,
        valid_ids=[0, 1, 2],
    )

    assert rendered.shape == (1080, 2044 + sidebar_width_for_frame(2044, 1080), 3)


def test_render_focus_is_external_display_context_and_persistent_heading() -> None:
    assert normalize_focus_behavior("追逐-被追逐") == "chase"
    assert normalize_focus_behavior("追逐/被追逐") == "chase"
    assert normalize_focus_behavior("attack") == "attack"
    assert focus_behavior_name_zh("avoidance") == "回避/被回避"

    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    selected, layer = select_display_events([])
    overlays = build_mouse_overlays(selected, layer, [0])
    rendered = draw_behavior_sidebar(
        frame,
        frame_index=0,
        total_frames=10,
        fps=30.0,
        active_event_count=0,
        display_events=selected,
        display_layer=layer,
        mouse_overlays=overlays,
        valid_ids=[0],
        focus_behavior="attack",
        focus_active=False,
    )

    assert rendered.shape[0] == 120
    assert rendered.shape[1] > frame.shape[1]
