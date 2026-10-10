from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from render_validation_result import (  # noqa: E402
    _error_reason,
    _auxiliary_events_from_evidence,
    _auxiliary_events_for_render,
    _events_at_frame,
    _mark_unselected_label_targets,
    _mark_unrecognized_targets,
    _prediction_event,
    _prediction_event_for_row,
    _add_compatible_pair_overlay,
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


def test_prediction_overlay_uses_predicted_ids_and_keeps_label_ids_separate() -> None:
    row = {"mouse_ids": "3,6", "canonical_behavior": "approach"}
    prediction = {
        "predicted_behavior": "huddle",
        "predicted_target_ids": "1,4,7",
    }

    event = _prediction_event_for_row(row, prediction, 30)

    assert event is not None
    assert event["member_ids"] == "1,4,7"
    assert event["event_source"] == "target_top1"


def test_missed_ground_truth_targets_are_marked_without_changing_prediction_overlay() -> None:
    overlays = build_mouse_overlays(
        [_prediction_event({"predicted_behavior": "huddle"}, (1, 4, 7), 30)],
        "group",
        [1, 3, 4, 6, 7],
    )

    _mark_unselected_label_targets(overlays, (3, 6), (1, 4, 7))

    assert "扎堆" in overlays[1].text
    assert "标签目标" in overlays[3].text
    assert "算法未选中" in overlays[3].text


def test_missing_prediction_is_explicitly_marked_on_target_only() -> None:
    overlays = build_mouse_overlays([], "none", [5, 6])

    _mark_unrecognized_targets(overlays, (5,), "")

    assert "未识别" in overlays[5].text
    assert "仅追踪" in overlays[6].text
    assert _error_reason({}, "接近", "未识别") == "未识别到接近"


def test_compatible_social_label_is_visible_but_does_not_replace_primary() -> None:
    target = _prediction_event({"predicted_behavior": "together"}, (5, 6), 30)
    assert target is not None
    overlays = build_mouse_overlays([target], "social", [5, 6, 7])

    _add_compatible_pair_overlay(
        overlays,
        (5, 6),
        {"predicted_behavior": "together", "compatible_labels": "together,nose_head_contact"},
    )

    assert "一起" in overlays[5].text
    assert "鼻头接触" in overlays[5].text
    assert "鼻头接触" in overlays[6].text
    assert "仅追踪" in overlays[7].text


def test_approach_contact_compatible_label_is_visible_for_both_targets() -> None:
    target = _prediction_event({"predicted_behavior": "nose_head_contact"}, (5, 6), 30)
    assert target is not None
    overlays = build_mouse_overlays([target], "social", [5, 6])

    _add_compatible_pair_overlay(
        overlays,
        (5, 6),
        {
            "predicted_behavior": "nose_head_contact",
            "compatible_labels": "approach,nose_head_contact",
        },
    )

    assert "鼻头接触" in overlays[5].text
    assert "接近" in overlays[5].text
    assert "接近" in overlays[6].text


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
                "5": {"stationary": 0.4, "locomotion": 0.0, "isolation": 0.0},
                "6": {"stationary": 0.0, "locomotion": 1.2, "isolation": 0.0},
            },
            "pair_scores": {"6,7": {"together": 0.8}},
            "group_scores": {"6,7,8": {"huddle": 2.0, "social_clustering": 0.0}},
        },
        (5,),
        30,
    )

    assert any(
        event.get("actor_id") == 6 and event["behavior"] == "locomotion" for event in events
    )
    assert any(
        event.get("pair_key") == "6,7" and event["behavior"] == "together" for event in events
    )
    assert any(
        event.get("member_ids") == "6,7,8" and event["behavior"] == "huddle" for event in events
    )
    assert any(item["ids"] == [6] and item["behavior"] == "locomotion" for item in summaries)


def test_frozen_trajectory_evidence_renders_auxiliary_events_without_parameter_file() -> None:
    events, summaries = _auxiliary_events_for_render(
        {
            "_trajectory_evidence": {
                "identity_scores": {"6": {"locomotion": 1.2}},
                "pair_scores": {},
                "group_scores": {},
            }
        },
        Path("unused-tracks.json"),
        30.0,
        (5,),
        30,
        None,
        True,
    )

    assert any(
        event.get("actor_id") == 6 and event["behavior"] == "locomotion" for event in events
    )
    assert summaries[0]["ids"] == [6]


def test_auxiliary_individual_events_use_the_four_merged_identity_score_keys() -> None:
    # [STEP4 2026-10-04] 个体层合并：Walking+Running -> Locomotion
    # The renderer reads every individual identity-score key, so the merged
    # locomotion slot replaces walking/running and the disabled grooming/rearing
    # slots stay renderable the moment their detectors are switched on.
    events, summaries = _auxiliary_events_from_evidence(
        {
            "identity_scores": {
                "6": {"locomotion": 1.2, "stationary": 0.0,
                      "grooming": 0.0, "rearing": 0.0, "isolation": 0.0},
                "7": {"locomotion": 0.0, "stationary": 0.0,
                      "grooming": 0.9, "rearing": 0.0, "isolation": 0.0},
                "8": {"locomotion": 0.0, "stationary": 0.0,
                      "grooming": 0.0, "rearing": 1.1, "isolation": 0.0},
            },
            "pair_scores": {},
            "group_scores": {},
        },
        (6,),
        30,
    )

    individual = {(event.get("actor_id"), event["behavior"]) for event in events}
    assert individual == {(6, "locomotion"), (7, "grooming"), (8, "rearing")}
    assert {item["behavior"] for item in summaries} == {"locomotion", "grooming", "rearing"}
    # Zero-scored slots never become events, which keeps the disabled detectors silent.
    assert not any(event["behavior"] == "stationary" for event in events)


def test_legacy_walking_and_running_score_keys_are_not_rendered_as_individuals() -> None:
    # [STEP4 2026-10-04] 个体层合并：Walking+Running -> Locomotion
    # Regression: the deleted score keys must not be picked up as individual
    # events any more; only the merged identity-score keys are consumed.
    events, summaries = _auxiliary_events_from_evidence(
        {
            "identity_scores": {"6": {"walking": 1.2, "running": 0.8, "stationary": 0.0}},
            "pair_scores": {},
            "group_scores": {},
        },
        (5,),
        30,
    )

    assert events == []
    assert summaries == []


def test_auxiliary_group_events_keep_distinct_temporal_phases() -> None:
    events, _ = _auxiliary_events_from_evidence(
        {
            "identity_scores": {},
            "pair_scores": {},
            "group_scores": {
                "1,2,3": {"social_clustering": 5.0, "huddle": 1.5}
            },
            "group_events": [
                {"behavior": "social_clustering", "member_ids": (1, 2, 3),
                 "start_frame": 0, "end_frame": 49, "score": 5.0},
                {"behavior": "huddle", "member_ids": (1, 2, 3),
                 "start_frame": 50, "end_frame": 64, "score": 1.5},
            ],
        },
        (1, 2, 3),
        70,
    )

    group = [event for event in events if event["event_scope"] == "group"]
    assert [(event["behavior"], event["start_frame"], event["end_frame"]) for event in group] == [
        ("social_clustering", 0, 49),
        ("huddle", 50, 64),
    ]
    assert [event["behavior"] for event in _events_at_frame(group, 25, 70)] == [
        "social_clustering"
    ]
    assert [event["behavior"] for event in _events_at_frame(group, 55, 70)] == [
        "huddle"
    ]


def test_auxiliary_isolation_does_not_replace_target_locomotion_behavior() -> None:
    events, _ = _auxiliary_events_from_evidence(
        {
            "identity_scores": {
                "5": {"stationary": 0.4, "locomotion": 0.1, "isolation": 1.2},
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
