"""Primary-failure semantics and recoverable local review routes."""
from __future__ import annotations

import hashlib
import json
import threading
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import quote

import pytest

from mouse_behavior.visualization.behavior_error_site import (
    ReviewStore, create_error_review_server, primary_reason,
)


def clip_row(tmp_path: Path, **updates):
    directory = tmp_path / "clip"
    directory.mkdir(exist_ok=True)
    tracks = [dict(frame=0, detections=[dict(track_id=1, box=[10, 20, 30, 40]),
                                        dict(track_id=99, box=[10, 20, 30, 40])]),
              dict(frame=2, detections=[dict(track_id=2, box=[40, 20, 60, 40])])]
    raw = json.dumps(tracks).encode()
    (directory / "tracks.json").write_bytes(raw)
    (directory / "metadata.json").write_text(json.dumps({"clip": {"width": 100, "height": 80, "fps": 10, "frame_count": 100}}))
    (directory / "clip.mp4").write_bytes(bytes(range(64)))
    row = dict(record_key="B:Huddling:test", dataset="B", behavior="Huddling", clip_id="test",
        clip_dir=str(directory), target_mouse_ids=[1, 2, 3], primary_correct=False,
        primary_predicted_behavior=None, frame_count=100, fps=10,
        tracks_sha256=hashlib.sha256(raw).hexdigest(), in_union_b_priority=True,
        true_target_coverage=0, benchmark_status="ok", failure_category="missed",
        pf_joint_predicted_behavior="Locomotion", pf_joint_predicted_target_ids=[99],
        huddle_diagnostics={"a1_any_positive": True, "a2_any_positive": False,
                            "best_huddle_target_ids": [1, 2, 99], "huddle_candidate_count": 1})
    row.update(updates)
    return row


def write_report(tmp_path, rows):
    path = tmp_path / "report.json"
    path.write_text(json.dumps(dict(schema_version="reconciled_behavior_evaluation_v1", clips=rows)), encoding="utf-8")
    return path


def test_huddle_other_ids_is_not_other_behavior_or_absent(tmp_path):
    row = clip_row(tmp_path)
    reason = primary_reason(row)
    assert reason["reason_code"] == "other_ids"
    assert "Locomotion" not in reason["summary"]
    assert "没有候选匹配完整" in reason["summary"]


@pytest.mark.parametrize("diagnostic,coverage,expected", [
    ({"a1_any_positive": False, "a2_any_positive": False}, 0, "not_detected"),
    ({"a1_any_positive": True, "a2_any_positive": True}, .2148, "short_coverage"),
])
def test_huddle_absent_and_short_coverage_have_distinct_reasons(tmp_path, diagnostic, coverage, expected):
    assert primary_reason(clip_row(tmp_path, huddle_diagnostics=diagnostic, true_target_coverage=coverage))["reason_code"] == expected


def test_primary_misclassification_uses_primary_output_not_global_winner(tmp_path):
    row = clip_row(tmp_path, behavior="Snout-head_contact", primary_predicted_behavior="Together", failure_category="misclassified")
    reason = primary_reason(row)
    assert reason["reason_code"] == "other_behavior"
    assert "共处" in reason["summary"]
    assert "Locomotion" not in reason["summary"]


def test_overlay_only_marks_raw_target_ids_and_keeps_absence(tmp_path):
    row = clip_row(tmp_path)
    store = ReviewStore(write_report(tmp_path, [row]))
    result = store.detail(row["record_key"])
    first = result["frames"][0]["detections"]
    assert first[0]["target"] is True
    assert first[1]["target"] is False  # Identical position is not identity evidence.
    assert first[1]["candidate"] is True
    assert [frame["frame"] for frame in result["frames"]] == [0, 2]  # No filled frame 1.
    absent = next(r for r in result["observations"] if r["id"] == 3)
    assert absent["observed_frames"] == 0 and absent["first_frame"] is None


def test_changed_tracks_are_rejected_instead_of_silent_stale_overlay(tmp_path):
    row = clip_row(tmp_path)
    path = write_report(tmp_path, [row])
    (Path(row["clip_dir"]) / "tracks.json").write_text("[]")
    with pytest.raises(ValueError, match="轨迹文件已变化"):
        ReviewStore(path).detail(row["record_key"])


def test_catalogue_counts_primary_once_and_keeps_all15_classes(tmp_path):
    row = clip_row(tmp_path)
    second = dict(row, record_key="B:Static:second", clip_id="second", behavior="Static", primary_correct=True)
    a = dict(row, record_key="A:Huddling:test", dataset="A", in_union_b_priority=False)
    catalogue = ReviewStore(write_report(tmp_path, [row, second, a])).load()
    assert catalogue["cohorts"]["B"]["total"] == 2
    assert catalogue["cohorts"]["B"]["correct"] == 1
    assert catalogue["cohorts"]["union"]["total"] == 2
    assert len(catalogue["cohorts"]["B"]["behaviors"]) == 15
    assert catalogue["cohorts"]["B"]["behaviors"][-1]["error_counts"] == {"other_ids": 1}


@pytest.fixture
def running_server(tmp_path):
    row = clip_row(tmp_path)
    report = write_report(tmp_path, [row])
    server = create_error_review_server(report, port=0, diagnostics_path=None)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", report, row
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


@pytest.mark.parametrize("broken", [None, "{broken", '{"schema_version":"wrong","clips":[]}', '{"schema_version":"reconciled_behavior_evaluation_v1","clips":[]}'])
def test_missing_malformed_empty_reports_preserve_shell_and_recover(running_server, broken):
    url, report, _ = running_server
    original = report.read_bytes()
    if broken is None:
        report.unlink()
    else:
        report.write_text(broken)
    with urllib.request.urlopen(url + "/") as response:
        assert response.status == 200
        assert b"<html" in response.read().lower()
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(url + "/api/catalogue")
    assert error.value.code == 503
    assert "application/json" in error.value.headers["Content-Type"]
    assert json.loads(error.value.read())["error"]
    report.write_bytes(original)
    with urllib.request.urlopen(url + "/api/catalogue") as response:
        assert json.load(response)["cohorts"]["B"]["total"] == 1


def test_video_range_unknown_api_and_unregistered_path(running_server):
    url, _, row = running_server
    request = urllib.request.Request(url + "/api/video/" + quote(row["record_key"], safe=""), headers={"Range": "bytes=8-15"})
    with urllib.request.urlopen(request) as response:
        assert response.status == 206
        assert response.headers["Content-Range"] == "bytes 8-15/64"
        assert response.read() == bytes(range(8, 16))
    for suffix in ("/api/no-such-route", "/api/video/..%2F..%2Fpyproject.toml"):
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(url + suffix)
        assert error.value.code == 404
        assert "application/json" in error.value.headers["Content-Type"]


def test_diagnostic_from_another_report_is_not_attached(tmp_path):
    row = clip_row(tmp_path)
    report = write_report(tmp_path, [row])
    diagnostic = tmp_path / "gates.json"
    diagnostic.write_text(json.dumps({"source_report_sha256": "wrong", "records": [{"clip_id": "test", "common_endpoint_span_s": 0}]}))
    assert ReviewStore(report, diagnostic).detail(row["record_key"])["huddle_gate"] is None


def test_bc_union_cohort_full_stats_and_pf_target_note(tmp_path):
    c_row = clip_row(
        tmp_path,
        record_key="C:Rearing:c1",
        dataset="C",
        behavior="Rearing",
        clip_id="c1",
        primary_correct=True,
        primary_predicted_behavior="Rearing",
        pf_target_correct=False,
        pf_target_predicted_behavior="Grooming",
        failure_category="ok",
    )
    b_row = clip_row(
        tmp_path,
        record_key="B:Grooming:b1",
        dataset="B",
        behavior="Grooming",
        clip_id="b1",
        primary_correct=False,
        primary_predicted_behavior="Rearing",
        failure_category="misclassified",
    )
    report = tmp_path / "showcase.json"
    report.write_text(
        json.dumps(
            {
                "schema_version": "reconciled_behavior_evaluation_v1",
                "full_cohort_stats": {
                    "union": {
                        "summary": {
                            "total_clips": 1369,
                            "primary_accuracy": 0.8627,
                            "pf_target_micro_precision": 0.88,
                            "pf_target_micro_recall": 0.86,
                        }
                    }
                },
                "clips": [c_row, b_row],
            }
        ),
        encoding="utf-8",
    )
    catalogue = ReviewStore(report).load()
    assert catalogue["default_cohort"] == "union"
    assert catalogue["default_status"] == "all"
    assert set(catalogue["cohorts"]) == {"union", "B", "C"}
    assert catalogue["cohorts"]["union"]["full_total"] == 1369
    reason = primary_reason(c_row)
    assert reason["reason_code"] == "passed"
    assert "15类竞争模型预测" in reason["summary"]
    assert "理毛" in reason["summary"]


def test_build_bc_error_showcase_caps_per_behavior_and_diversifies(monkeypatch):
    import importlib.util

    script_path = Path(__file__).resolve().parents[2] / "scripts" / "build_bc_error_showcase.py"
    monkeypatch.syspath_prepend(str(script_path.parent))
    spec = importlib.util.spec_from_file_location("build_bc_error_showcase", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    pool = []
    for idx, pred in enumerate(["Grooming", "Static", "Locomotion", "Grooming", "Static"]):
        pool.append(
            {
                "record_key": f"C:Rearing:mis_{idx}",
                "dataset": "C" if idx % 2 == 0 else "B",
                "behavior": "Rearing",
                "clip_id": f"mis_{idx}",
                "primary_correct": False,
                "primary_predicted_behavior": pred,
                "pf_target_correct": False,
                "pf_target_predicted_behavior": pred,
                "failure_category": "misclassified",
            }
        )
    for idx in range(5):
        pool.append(
            {
                "record_key": f"B:Rearing:miss_{idx}",
                "dataset": "B" if idx % 2 == 0 else "C",
                "behavior": "Rearing",
                "clip_id": f"miss_{idx}",
                "primary_correct": False,
                "primary_predicted_behavior": None,
                "pf_target_correct": False,
                "pf_target_predicted_behavior": None,
                "failure_category": "missed",
            }
        )
    for idx in range(5):
        pool.append(
            {
                "record_key": f"C:Rearing:ok_{idx}",
                "dataset": "C" if idx % 2 == 0 else "B",
                "behavior": "Rearing",
                "clip_id": f"ok_{idx}",
                "primary_correct": True,
                "primary_predicted_behavior": "Rearing",
                "pf_target_correct": True,
                "pf_target_predicted_behavior": "Rearing",
                "failure_category": "ok",
                "true_target_coverage": 0.9 - idx * 0.05,
            }
        )

    picked = module._select_curated_clips(pool, max_per_behavior=10)
    assert len(picked) == 10
    assert any(c["failure_category"] == "misclassified" for c in picked)
    assert any(c["failure_category"] == "missed" for c in picked)
    assert any(c["failure_category"] == "ok" for c in picked)
    preds = {c["primary_predicted_behavior"] for c in picked if c["failure_category"] == "misclassified"}
    assert {"Grooming", "Static", "Locomotion"}.issubset(preds)
