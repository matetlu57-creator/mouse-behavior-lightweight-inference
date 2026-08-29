from __future__ import annotations

import numpy as np

from mouse_behavior.visualization.overlay import (
    build_mouse_overlays,
    build_panel_lines,
    draw_behavior_sidebar,
    focus_behavior_name_zh,
    normalize_focus_behavior,
    normalize_contact_events,
    resolve_event_for_frame,
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

    assert layer == "group"
    assert {event["behavior"] for event in selected} == {"isolation", "stationary"}
    overlays = build_mouse_overlays(selected, layer, [0, 1, 2])
    assert overlays[0].text == "ID 00｜静止"
    assert overlays[1].text == "ID 01｜群体：孤立"
    assert overlays[2].text == "ID 02｜群体：孤立"
    assert "孤立" in build_panel_lines(selected, layer)[1]


def test_group_behavior_overrides_social_and_individual_for_huddle_members() -> None:
    selected, layer = select_display_events(
        [
            _event("running", actor=0, pair="mouse_0"),
            _event("attack", actor=0, target=1, pair="0_1"),
            {
                **_event("huddle", pair="group"),
                "member_ids": [0, 1, 2],
            },
            _event("walking", actor=3, pair="mouse_3"),
        ]
    )

    overlays = build_mouse_overlays(selected, layer, [0, 1, 2, 3])

    assert layer == "group"
    assert overlays[0].text == "ID 00｜群体：扎堆"
    assert overlays[1].text == "ID 01｜群体：扎堆"
    assert overlays[2].text == "ID 02｜群体：扎堆"
    assert overlays[3].text == "ID 03｜行走"
    assert not any(event["behavior"] == "attack" for event in selected)


def test_isolation_overrides_individual_only_for_isolated_mouse() -> None:
    selected, layer = select_display_events(
        [
            _event("stationary", actor=1, pair="mouse_1"),
            {
                **_event("isolation", pair="group"),
                "member_ids": [1],
            },
            _event("approach", actor=1, target=2, pair="1_2"),
        ]
    )

    overlays = build_mouse_overlays(selected, layer, [1, 2])

    assert layer == "group"
    assert overlays[1].text == "ID 01｜群体：孤立"
    assert overlays[2].text == "ID 02｜被接近"
    assert not any(event["behavior"] == "stationary" for event in selected)


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


def test_approach_is_highest_social_display_priority() -> None:
    events = [
        _event("attack", actor=1, target=2, pair="1_2"),
        _event("chase", actor=1, target=2, pair="1_2"),
        _event("avoidance", actor=2, target=1, pair="1_2"),
        _event("together", pair="1_2"),
        _event("nose_head_contact", actor=1, target=2, pair="1_2"),
        _event("approach", actor=1, target=2, pair="1_2"),
    ]

    selected, layer = select_display_events(events)
    overlays = build_mouse_overlays(selected, layer, [1, 2])

    assert layer == "social"
    assert selected[0]["behavior"] == "approach"
    assert overlays[1].text == "ID 01｜主动接近"
    assert overlays[2].text == "ID 02｜被接近"


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


def test_sidebar_preserves_source_pixels_while_rendering_in_appended_panel() -> None:
    source = np.full((120, 160, 3), 127, dtype=np.uint8)
    original = source.copy()
    selected, layer = select_display_events([_event("attack", actor=1, target=2, pair="1_2")])
    overlays = build_mouse_overlays(selected, layer, [1, 2])

    rendered = draw_behavior_sidebar(
        source,
        frame_index=10,
        total_frames=100,
        fps=30.0,
        active_event_count=1,
        display_events=selected,
        display_layer=layer,
        mouse_overlays=overlays,
        valid_ids=[1, 2],
        panel_width=320,
        focus_behavior="attack",
        focus_active=True,
    )

    assert rendered.shape == (120, 480, 3)
    assert np.array_equal(rendered[:, :160], original)
    assert np.any(rendered[:, 160:] != 127)


def test_render_focus_is_external_display_context_and_persistent_heading() -> None:
    assert normalize_focus_behavior("追逐-被追逐") == "chase"
    assert normalize_focus_behavior("追逐/被追逐") == "chase"
    assert normalize_focus_behavior("attack") == "attack"
    assert focus_behavior_name_zh("avoidance") == "回避/被回避"
    assert normalize_focus_behavior("接近-被接近") == "approach"
    assert normalize_focus_behavior("鼻头接触") == "nose_head_contact"
    assert normalize_focus_behavior("孤立行为") == "isolation"

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


def test_focus_is_heading_only_and_does_not_reorder_same_layer_events() -> None:
    selected, layer = select_display_events(
        [
            _event("chase", actor=1, target=2, pair="1_2"),
            _event("nose_head_contact", actor=1, target=2, pair="1_2"),
        ],
        focus_behavior="nose_head",
    )
    overlays = build_mouse_overlays(
        selected,
        layer,
        [1, 2],
        focus_behavior="nose_head",
    )

    assert selected[0]["behavior"] == "chase"
    assert overlays[1].text == "ID 01｜追逐"
    assert overlays[2].text == "ID 02｜被追逐"

    selected, layer = select_display_events(
        [
            _event("attack", actor=1, target=2, pair="1_2"),
            {**_event("huddle", pair="group"), "member_ids": [1, 2, 3]},
        ],
        focus_behavior="attack",
    )
    overlays = build_mouse_overlays(
        selected,
        layer,
        [1, 2, 3],
        focus_behavior="attack",
    )
    assert overlays[1].text == "ID 01｜群体：扎堆"
    assert overlays[2].text == "ID 02｜群体：扎堆"


def test_identity_bridge_roles_are_resolved_per_frame_without_ghost_ids() -> None:
    event = {
        **_event("attack", actor=2, target=9, pair="2_6|2_9"),
        "identity_bridge": True,
        "participant_ids": [2, 6, 9],
        "member_ids": [2, 6, 9],
        "role_trace": [
            {
                "pair_key": "2_6",
                "actor_id": 2,
                "target_id": 6,
                "start_frame": 0,
                "end_frame": 5,
            },
            {
                "pair_key": "2_9",
                "actor_id": 2,
                "target_id": 9,
                "start_frame": 8,
                "end_frame": 12,
            },
        ],
    }

    first = resolve_event_for_frame(event, 3)
    assert first is not None
    selected, layer = select_display_events([first])
    overlays = build_mouse_overlays(selected, layer, [2, 6, 9])
    assert overlays[2].text == "ID 02｜攻击"
    assert overlays[6].text == "ID 06｜被攻击"
    assert overlays[9].text == "ID 09｜仅追踪"

    assert resolve_event_for_frame(event, 6) is None

    second = resolve_event_for_frame(event, 10)
    assert second is not None
    selected, layer = select_display_events([second])
    overlays = build_mouse_overlays(selected, layer, [2, 6, 9])
    assert overlays[2].text == "ID 02｜攻击"
    assert overlays[6].text == "ID 06｜仅追踪"
    assert overlays[9].text == "ID 09｜被攻击"


def test_group_members_are_resolved_per_frame_without_union_leakage() -> None:
    event = {
        **_event("huddle", pair="group"),
        "member_ids": [2, 3, 4, 5],
        "member_ids_at_peak": [3, 4, 5],
        "member_trace": [
            {"member_ids": [2, 3, 4], "start_frame": 0, "end_frame": 4},
            {"member_ids": [3, 4, 5], "start_frame": 6, "end_frame": 10},
        ],
    }

    first = resolve_event_for_frame(event, 2)
    assert first is not None
    selected, layer = select_display_events([first])
    overlays = build_mouse_overlays(selected, layer, [2, 3, 4, 5])
    assert overlays[2].text == "ID 02｜群体：扎堆"
    assert overlays[5].text == "ID 05｜仅追踪"

    assert resolve_event_for_frame(event, 5) is None

    second = resolve_event_for_frame(event, 8)
    assert second is not None
    selected, layer = select_display_events([second])
    overlays = build_mouse_overlays(selected, layer, [2, 3, 4, 5])
    assert overlays[2].text == "ID 02｜仅追踪"
    assert overlays[5].text == "ID 05｜群体：扎堆"
