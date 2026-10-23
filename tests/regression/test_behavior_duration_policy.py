"""Standalone duration boundaries and actual exported-video qualification."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from mouse_behavior.behavior.duration_policy import (
    qualify_event_duration, require_event_duration, resolve_duration_policy,
)
from mouse_behavior.visualization import behavior_packages as packages

PARAMETERIZED_DURATIONS = [
    ("stationary", "stationary_min_duration_s"),
    ("locomotion", "locomotion_min_duration_s"),
    ("huddle", "huddle_min_duration_s"),
    ("isolation", "isolation_min_duration_s"),
    ("dispersal", "dispersal_min_duration_s"),
    ("group_locomotion", "group_locomotion_min_duration_s"),
]


def event(behavior, count, start=0, **extra):
    return dict(canonical_behavior=behavior, behavior=behavior, event_id=f"{behavior}_{start}_{count}",
                mouse_ids=[7], start_frame=start, end_frame=start + count, **extra)


@pytest.mark.parametrize("behavior,minimum,comparator", [
    ("stationary", 1., ">="), ("locomotion", .5, ">="), ("approach", .3, ">="),
    ("together", 1., ">="), ("attack", .15, ">="), ("chase", 1., ">="),
    ("nose_head_contact", .1, ">="), ("nose_tail_contact", .15, ">="),
    ("huddle", 5., ">="), ("social_clustering", .5, ">="),
    ("isolation", 10., ">"), ("dispersal", 5., ">"), ("group_locomotion", 3., ">"),
])
@pytest.mark.parametrize("offset", [-1, 0, 1])
def test_n_minus_one_n_n_plus_one(behavior, minimum, comparator, offset):
    fps = 100
    count = int(round(minimum * fps)) + offset
    row = event(behavior, count, canonical_support_frame_intervals=[[0, count]])
    result = qualify_event_duration(row, fps)
    assert result["required_s"] == minimum
    assert result["comparator"] == comparator
    assert result["ok"] is (offset > 0 if comparator == ">" else offset >= 0)
    assert result["actual_s"] == pytest.approx(count / fps)


@pytest.mark.parametrize("behavior,field", PARAMETERIZED_DURATIONS)
def test_stronger_trusted_parameters_reject_child_that_default_would_allow(behavior, field):
    from dataclasses import replace
    from mouse_behavior.evaluation.exported_dataset_models import HeuristicParameters

    default = resolve_duration_policy(behavior)
    count = int(round(default.required_s * 100)) + 1
    row = event(behavior, count, canonical_support_frame_intervals=[[0, count]],
                required_s=.01, effective_parameters={field: .01}, parameters={field: .01})
    parameters = replace(HeuristicParameters(), **{field: default.required_s * 2})
    assert require_event_duration(row, 100)["ok"]
    assert resolve_duration_policy(behavior, parameters=parameters).required_s == default.required_s * 2
    result = qualify_event_duration(row, 100, parameters=parameters)
    assert not result["ok"]
    assert result["required_s"] == default.required_s * 2
    assert field in result["source"]
    with pytest.raises(ValueError, match="process_duration_below_requirement"):
        require_event_duration(row, 100, parameters=parameters)


@pytest.mark.parametrize("behavior,field", PARAMETERIZED_DURATIONS)
@pytest.mark.parametrize("offset", [-1, 0, 1])
def test_trusted_parameter_boundaries_preserve_comparator(behavior, field, offset):
    from dataclasses import replace
    from mouse_behavior.evaluation.exported_dataset_models import HeuristicParameters

    baseline = resolve_duration_policy(behavior)
    required = baseline.required_s * 2
    parameters = replace(HeuristicParameters(), **{field: required})
    count = int(round(required * 100)) + offset
    row = event(behavior, count, canonical_support_frame_intervals=[[0, count]])
    result = qualify_event_duration(row, 100, parameters=parameters)
    assert result["comparator"] == baseline.comparator
    assert result["ok"] is (offset > 0 if baseline.comparator == ">" else offset >= 0)


def test_explicit_default_parameters_preserve_all_default_duration_results():
    from mouse_behavior.evaluation.exported_dataset_models import HeuristicParameters, CANONICAL_BEHAVIORS

    parameters = HeuristicParameters()
    for behavior in CANONICAL_BEHAVIORS:
        default = resolve_duration_policy(behavior)
        explicit = resolve_duration_policy(behavior, parameters=parameters)
        assert (explicit.required_s, explicit.comparator, explicit.support_required_s) == (
            default.required_s, default.comparator, default.support_required_s)
        for offset in (-1, 0, 1):
            count = int(round(default.required_s * 100)) + offset
            row = event(behavior, count, canonical_support_frame_intervals=[[0, count]])
            old = qualify_event_duration(row, 100)
            supplied = qualify_event_duration(row, 100, parameters=parameters)
            # An explicit configuration has more precise source provenance for
            # individual and Isolation thresholds, but identical decisions.
            old.pop("source")
            supplied.pop("source")
            assert old == supplied


@pytest.mark.parametrize("behavior,field", PARAMETERIZED_DURATIONS)
@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True, "2", None])
def test_invalid_trusted_duration_threshold_is_rejected(behavior, field, value):
    from dataclasses import replace
    from mouse_behavior.evaluation.exported_dataset_models import HeuristicParameters

    parameters = replace(HeuristicParameters(), **{field: value})
    row = event(behavior, 10000, canonical_support_frame_intervals=[[0, 10000]])
    with pytest.raises(ValueError, match=field + " must be finite and positive"):
        resolve_duration_policy(behavior, parameters=parameters)
    assert not qualify_event_duration(row, 100, parameters=parameters)["ok"]
    with pytest.raises(ValueError, match=field + " must be finite and positive"):
        require_event_duration(row, 100, parameters=parameters)


@pytest.mark.parametrize("parameters", [{"huddle_min_duration_s": .01}, "defaults", object()])
def test_trusted_parameter_api_rejects_threshold_dicts_and_non_parameter_objects(parameters):
    with pytest.raises(ValueError, match="HeuristicParameters instance"):
        require_event_duration(event("huddle", 10000), 100, parameters=parameters)


def test_isolation_preserves_upstream_group_minimum_and_v3_posture_ignores_social_mirrors():
    from dataclasses import replace
    from mouse_behavior.evaluation.exported_dataset_models import HeuristicParameters

    parameters = replace(HeuristicParameters(), isolation_min_duration_s=1.,
                         chase_min_duration_s=20., chase_min_directed_support_s=20.,
                         together_min_duration_s=20., attack_min_event_duration_s=20., pair_min_duration_s=20.)
    assert resolve_duration_policy("isolation", parameters=parameters).required_s == 10
    assert require_event_duration(event("chase", 100), 100, parameters=parameters)["required_s"] == 1
    assert require_event_duration(event("approach", 30), 100, parameters=parameters)["required_s"] == .3
    assert require_event_duration(event("attack", 15), 100, parameters=parameters)["required_s"] == .15
    assert require_event_duration(event("together", 100), 100, parameters=parameters)["required_s"] == 1
    for behavior in ("grooming", "rearing"):
        result = require_event_duration(event(behavior, 30, posture_engine="v3_decoupled"), 100, parameters=parameters)
        assert result["required_s"] == .3
        assert result["source"].startswith("posture_rules.DEFAULT_")


@pytest.mark.parametrize("fps", [0, -1, float("nan"), float("inf"), True, None, "30"])
def test_invalid_fps_is_diagnostic_and_require_rejects(fps):
    row = event("approach", 20)
    assert not qualify_event_duration(row, fps)["ok"]
    with pytest.raises(ValueError, match="fps"):
        require_event_duration(row, fps)


@pytest.mark.parametrize("start,end", [(-1, 10), (0, 0), (10, 9), (False, 10), (0, 1.5), (None, 10)])
def test_invalid_half_open_ranges(start, end):
    row = dict(canonical_behavior="approach", start_frame=start, end_frame=end)
    with pytest.raises(ValueError, match="half-open"):
        require_event_duration(row, 10)


def test_parent_total_support_and_declared_duration_cannot_qualify_short_child():
    row = event("huddle", 20, duration_s=50., parent_candidate_frame_intervals=[[0, 1000]],
                canonical_support_frame_intervals=[[0, 10], [10, 20]], canonical_support_frame_count=1000)
    assert qualify_event_duration(row, 10)["reason"] == "process_duration_below_requirement"


def test_group_sync_support_is_separate_clipped_union_not_duplicate_sum():
    policy = resolve_duration_policy("group_locomotion")
    row = event("group_locomotion", 40, canonical_support_frame_intervals=[[0, 4], [0, 4], [7, 10]])
    result = qualify_event_duration(row, 10)
    assert result["actual_s"] == 4
    assert result["support_frame_count"] == 7
    assert result["reason"] == "measured_sync_support_below_requirement"
    assert result["support_required_s"] == policy.support_required_s
    row["canonical_support_frame_intervals"] = [[0, 5], [20, 25]]
    assert require_event_duration(row, 10)["support_frame_count"] == 10
    row["end_frame"] = 25
    assert not qualify_event_duration(row, 10)["ok"]


@pytest.mark.parametrize("support", [None, [[-1, 10]], [[0, 41]], [[4, 4]], [[0, 1.5]], [[0]], "0-30"])
def test_invalid_group_support_cannot_be_replaced_by_counter(support):
    row = event("group_locomotion", 40, canonical_support_frame_count=40)
    if support is not None:
        row["canonical_support_frame_intervals"] = support
    assert not qualify_event_duration(row, 10)["ok"]


def test_tail_baseline_and_valid_short_extension_keep_distinct_thresholds():
    row = event("nose_tail_contact", 15)
    assert require_event_duration(row, 100)["required_s"] == .15
    row["duration_channel"] = "baseline"
    result = qualify_event_duration(row, 100)
    assert not result["ok"] and result["required_s"] == .5
    row["end_frame"] = 50
    assert require_event_duration(row, 100)["ok"]


@pytest.mark.parametrize("behavior", ["grooming", "rearing"])
@pytest.mark.parametrize("count,passes", [(29, False), (30, True), (31, True)])
def test_explicit_v3_posture_engine_uses_authoritative_temporal_minimum(behavior, count, passes):
    from mouse_behavior.behavior.posture_rules import DEFAULT_GROOMING_TEMPORAL, DEFAULT_REARING_TEMPORAL

    temporal = DEFAULT_GROOMING_TEMPORAL if behavior == "grooming" else DEFAULT_REARING_TEMPORAL
    row = event(behavior, count, posture_engine="v3_decoupled", min_duration_s=.01, required_s=.01)
    result = qualify_event_duration(row, 100)
    assert result["required_s"] == temporal.min_duration_s == .3
    assert result["ok"] is passes
    assert result["posture_engine"] == "v3_decoupled"
    config_name = "DEFAULT_GROOMING_TEMPORAL" if behavior == "grooming" else "DEFAULT_REARING_TEMPORAL"
    assert result["source"] == f"posture_rules.{config_name}.min_duration_s"


def test_untagged_legacy_rearing_keeps_half_second_gate_and_ignores_numeric_override():
    row = event("rearing", 30, min_duration_s=.01, required_s=.01, duration_policy={"required_s": .01})
    result = qualify_event_duration(row, 100)
    assert result["required_s"] == .5
    assert result["source"] == "individual.rearing_min_duration_s"
    assert not result["ok"]
    row["end_frame"] = 50
    assert require_event_duration(row, 100)["ok"]
    assert resolve_duration_policy("rearing").required_s == .5


@pytest.mark.parametrize("behavior", ["grooming", "rearing"])
@pytest.mark.parametrize("engine", ["v3", "legacy", "", None, False, {"required_s": .01}])
def test_unknown_explicit_posture_engine_is_rejected(behavior, engine):
    row = event(behavior, 1000, posture_engine=engine)
    assert not qualify_event_duration(row, 100)["ok"]
    with pytest.raises(ValueError, match="unknown posture_engine"):
        require_event_duration(row, 100)


@pytest.mark.parametrize("label,canonical", [("Static", "stationary"), ("Group locomotion", "group_locomotion"),
                                            ("Snout-rear_contact", "nose_tail_contact"), ("爬墙", "rearing")])
def test_display_labels_resolve_same_policy(label, canonical):
    assert resolve_duration_policy(label) == resolve_duration_policy(canonical)


def test_unsupported_behavior_is_rejected():
    with pytest.raises(ValueError, match="no standalone duration policy"):
        require_event_duration(event("following", 100), 10)


@pytest.mark.parametrize("workers", [1, 3])
def test_export_batch_rejects_short_event_before_creating_output_or_launching_encoder(tmp_path, monkeypatch, workers):
    def forbidden(*args, **kwargs):
        raise AssertionError("rejected batch must not open media or launch ffmpeg")
    monkeypatch.setattr(packages.cv2, "VideoCapture", forbidden)
    monkeypatch.setattr(packages.subprocess, "run", forbidden)
    output = tmp_path / "clips"
    with pytest.raises(ValueError, match="duration rejected"):
        packages.export_packages(output, tmp_path / "absent.mp4", None,
                                 [event("isolation", 101), event("isolation", 100)],
                                 {"source_video": dict(frame_count=110, fps=10)}, {}, workers=workers)
    assert not output.exists()


class Store:
    def rows(self, start, end):
        return iter(dict(frame=frame - start, source_frame=frame, detections=[]) for frame in range(start, end))

    def write_tracks(self, path, start, end):
        path.write_text(json.dumps(list(self.rows(start, end))), encoding="utf-8")


def test_real_export_duration_payloads_and_numbered_frames(tmp_path):
    source = tmp_path / "source.mp4"
    fps, count = 20., 40
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), fps, (160, 96))
    assert writer.isOpened()
    for frame in range(count):
        writer.write(np.full((96, 160, 3), (frame * 5, 20, 230 - frame * 5), np.uint8))
    writer.release()
    info = dict(path=str(source), sha256=packages._sha256(source), fps=fps,
                frame_count=count, width=160, height=96)
    binding = dict(source_video=info, cache_status_sha256="synthetic", chunk_sha256={})
    row = event("approach", 6, start=17)
    full = packages.render_full_package(source, tmp_path / "full", Store(), [row], binding, {})
    rows = packages.export_packages(tmp_path / "clips", Path(full["video_path"]), Store(), [row], binding, {})
    package = Path(rows[0]["path"])
    assert {item.name for item in package.iterdir()} == set(packages.PAYLOADS)
    verified = rows[0]["verification"]
    assert verified["frame_count"] == 6 and verified["duration_s"] == pytest.approx(.3)
    metadata = json.loads((package / "metadata.json").read_text())
    qualification = metadata["provenance"]["event_qualification"]
    assert qualification["standalone_duration_qualification"]["ok"]
    assert qualification["encoded_duration_qualification"]["ok"]
    assert qualification["encoded_duration_qualification"]["duration_basis"] == "decoded_frame_count_over_encoded_fps"
    assert "duration_is_not_standalone_rule_requalification" not in qualification
    assert "short_clip_warning" not in qualification
    reference = []
    capture = cv2.VideoCapture(full["video_path"])
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        reference.append(frame[80:, 100:].mean(axis=(0, 1)))
    capture.release()
    capture = cv2.VideoCapture(rows[0]["video_path"])
    matched = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        color = frame[80:, 100:].mean(axis=(0, 1))
        matched.append(int(np.argmin(np.linalg.norm(np.asarray(reference) - color, axis=1))))
    capture.release()
    assert matched == list(range(17, 23))


def test_post_encode_rechecks_actual_fps_before_publishing_annotation(tmp_path, monkeypatch):
    monkeypatch.setattr(packages.subprocess, "run", lambda *args, **kwargs: None)
    # Encoder output can have a small FPS change within the verifier's media
    # tolerance; at an exact >= boundary that still shortens actual playback.
    monkeypatch.setattr(packages, "_verify_bounded_video", lambda *args: dict(
        frame_count=3, fps=10.0001, width=160, height=96, duration_s=3 / 10.0001))
    row = event("approach", 3)
    with pytest.raises(ValueError, match="process_duration_below_requirement"):
        packages._export_one_package(tmp_path, tmp_path / "rendered.mp4", Store(), row,
                                     {"source_video": dict(fps=10, width=160, height=96)}, {}, 10, None, "veryfast")
    assert not (tmp_path / row["behavior"] / row["event_id"] / "annotation.json").exists()


@pytest.mark.parametrize("four_classes", [False, True])
def test_legacy_fixed_windows_explicitly_remain_showcases(tmp_path, four_classes):
    from mouse_behavior.visualization.rendering import extract_behavior_clips, extract_four_class_clips

    source = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10, (32, 24))
    assert writer.isOpened()
    for index in range(60):
        writer.write(np.full((24, 32, 3), index * 3, np.uint8))
    writer.release()
    events = tmp_path / "events.csv"
    # A five-second Isolation sample cannot satisfy the strict >10s canonical
    # contract, but remains useful as a clearly declared presentation window.
    behavior = "attack" if four_classes else "isolation"
    events.write_text(f"start_frame,end_frame,peak_frame,candidate_level,behavior\n0,59,30,strong,{behavior}\n")
    output = tmp_path / "showcases"
    if four_classes:
        extract_four_class_clips(source, events, output)
        summary_path = output / "four_class_clip_summary.json"
        manifest_path = output / "four_class_clip_manifest.csv"
    else:
        extract_behavior_clips(source, events, output)
        summary_path = output / "behavior_clip_summary.json"
        manifest_path = output / "behavior_clip_manifest.csv"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["clip_contract"] == "fixed_window_showcase"
    assert summary["standalone_behavior_qualification"] is False
    import csv
    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    assert manifest and all(row["clip_contract"] == "fixed_window_showcase" for row in manifest)
    assert all(row["standalone_behavior_qualification"] == "False" for row in manifest)
    from mouse_behavior.evaluation.segment_export import _verify_video
    for row in manifest:
        verified = _verify_video(Path(row["path"]), int(row["frame_count"]), float(row["encoded_fps"]))
        assert int(row["end_frame"]) - int(row["start_frame"]) + 1 == verified["frame_count"]
        assert int(row["decoded_frame_count"]) == verified["frame_count"]
        assert float(row["duration_s"]) == verified["duration_s"]
