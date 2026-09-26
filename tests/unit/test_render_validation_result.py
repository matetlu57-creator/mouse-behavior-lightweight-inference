from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from render_validation_result import (  # noqa: E402
    _error_reason,
    _auxiliary_events_from_evidence,
    _mark_unrecognized_targets,
    _prediction_event,
)
from mouse_behavior.visualization.overlay import (  # noqa: E402
    build_mouse_overlays,
    build_panel_lines,
    select_display_events,
)


def test_isolation_prediction_is_rendered_for_single_group_member() -> None:
    event = _prediction_event({"predicted_behavior": "isolation"}, (5,), 30)

    assert event is not None
    assert event["event_scope"] == "group"
    assert event["member_ids"] == "5"

    overlays = build_mouse_overlays([event], "group", [5, 6])
    assert "孤立" in overlays[5].text
    assert "仅追踪" in overlays[6].text


def test_missing_prediction_is_explicitly_marked_on_target_only() -> None:
    overlays = build_mouse_overlays([], "none", [5, 6])

    _mark_unrecognized_targets(overlays, (5,), "")

    assert "未识别" in overlays[5].text
    assert "仅追踪" in overlays[6].text
    assert _error_reason({}, "接近", "未识别") == "未识别到接近"


def test_strict_top1_error_reason_distinguishes_target_hit() -> None:
    assert (
        _error_reason({"target_id_hit": "True"}, "接近", "攻击/被攻击")
        == "目标 ID 已命中，但行为识别为攻击/被攻击（标签为接近）"
    )
    assert (
        _error_reason({"target_id_hit": "False"}, "接近", "接近")
        == "行为识别正确，但目标 ID 未命中（标签：接近）"
    )
    assert build_panel_lines([], "none", "当前帧无行为预测，目标框显示“未识别”")[1] == (
        "当前帧无行为预测，目标框显示“未识别”"
    )


def test_auxiliary_events_cover_non_target_tracks_without_labels() -> None:
    events, summaries = _auxiliary_events_from_evidence(
        {
            "identity_scores": {
                "5": {"stationary": 0.4, "walking": 0.0, "isolation": 0.0},
                "6": {"stationary": 0.0, "walking": 1.2, "isolation": 0.0},
            },
            "pair_scores": {"6,7": {"together": 0.8}},
            "group_scores": {"6,7,8": {"huddle": 2.0, "social_clustering": 0.0}},
        },
        (5,),
        30,
    )

    assert any(event.get("actor_id") == 6 and event["behavior"] == "walking" for event in events)
    assert any(
        event.get("pair_key") == "6,7" and event["behavior"] == "together" for event in events
    )
    assert any(
        event.get("member_ids") == "6,7,8" and event["behavior"] == "huddle" for event in events
    )
    assert any(item["ids"] == [6] and item["behavior"] == "walking" for item in summaries)


def test_auxiliary_isolation_does_not_replace_target_locomotion_behavior() -> None:
    events, _ = _auxiliary_events_from_evidence(
        {
            "identity_scores": {
                "5": {"stationary": 0.4, "walking": 0.1, "isolation": 1.2},
            },
            "pair_scores": {},
            "group_scores": {},
        },
        (5,),
        30,
    )

    assert any(event["behavior"] == "stationary" and event.get("actor_id") == 5 for event in events)
    assert any(
        event["behavior"] == "isolation" and event.get("member_ids") == "5" for event in events
    )


def test_social_event_remains_visible_with_group_context_for_shared_members() -> None:
    social = {
        "behavior": "nose_head_contact",
        "event_scope": "pair",
        "pair_key": "5,6",
        "start_frame": 0,
        "end_frame": 20,
        "peak_score": 0.6,
    }
    group = {
        "behavior": "social_clustering",
        "event_scope": "group",
        "member_ids": "5,6,7",
        "start_frame": 0,
        "end_frame": 20,
        "peak_score": 0.9,
    }

    selected, layer = select_display_events([group, social])
    overlays = build_mouse_overlays(selected, layer, [5, 6, 7])

    assert layer == "social_group"
    assert "群体：社会聚集" in overlays[5].text
    assert "群体：社会聚集" in overlays[6].text
    assert "社交：鼻头接触" in overlays[5].text
    assert "社交：鼻头接触" in overlays[6].text
    assert "社会聚集" in overlays[7].text


def test_group_social_and_individual_behavior_are_all_shown_for_same_mouse() -> None:
    events = [
        {
            "behavior": "huddle",
            "event_scope": "group",
            "member_ids": "5,6,7",
            "start_frame": 0,
            "end_frame": 20,
            "peak_score": 0.9,
        },
        {
            "behavior": "approach",
            "event_scope": "pair",
            "actor_id": 5,
            "target_id": 6,
            "pair_key": "5,6",
            "start_frame": 0,
            "end_frame": 20,
            "peak_score": 0.7,
        },
        {
            "behavior": "stationary",
            "event_scope": "individual",
            "actor_id": 5,
            "start_frame": 0,
            "end_frame": 20,
            "peak_score": 0.5,
        },
    ]

    selected, layer = select_display_events(events)
    overlays = build_mouse_overlays(selected, layer, [5, 6, 7])

    assert layer == "mixed"
    assert "群体：扎堆" in overlays[5].text
    assert "社交：主动接近" in overlays[5].text
    assert "个体：静止" in overlays[5].text


def test_group_overlay_uses_documented_huddling_first_priority() -> None:
    events = [
        {"behavior": "huddle", "event_scope": "group", "member_ids": "5,6,7", "peak_score": 99.0},
        {
            "behavior": "social_clustering",
            "event_scope": "group",
            "member_ids": "5,6,7",
            "peak_score": 50.0,
        },
        {"behavior": "isolation", "event_scope": "group", "member_ids": "5", "peak_score": 1.0},
    ]

    selected, layer = select_display_events(events)
    overlays = build_mouse_overlays(selected, layer, [5, 6, 7])

    assert "群体：扎堆" in overlays[5].text
    assert "群体：社会聚集" not in overlays[5].text
    assert "群体：孤立" not in overlays[5].text


def test_target_top1_event_survives_auxiliary_deduplication() -> None:
    target = _prediction_event({"predicted_behavior": "together"}, (5, 6), 30)
    assert target is not None
    auxiliary = {
        "behavior": "together",
        "event_scope": "pair",
        "pair_key": "5,6",
        "start_frame": 0,
        "end_frame": 20,
        "peak_score": 99.0,
    }

    selected, _ = select_display_events([auxiliary, target])

    assert len(selected) == 1
    assert selected[0].get("event_source") == "target_top1"
