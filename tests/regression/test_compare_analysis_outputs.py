from __future__ import annotations

import json
from pathlib import Path

from tools.compare_analysis_outputs import (
    REQUIRED_FILES,
    compare_output_directories,
)


def _write_output_tree(root: Path, *, event_value: str = "chase") -> None:
    root.mkdir(parents=True)
    for required in REQUIRED_FILES:
        path = root / required
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.name == "lightweight_analysis_metadata.json":
            payload = {
                "config": str(root / "mouse_chase_attack_config.yaml"),
                "contact_event_csv": str(root / "lightweight_contact_events.csv"),
                "event_counts": {"weak_chase": 1},
                "parallel_behavior_fsm": {"enabled": True, "mode": "active"},
                "stage_timings_s": {"candidate_pair_analysis": 9.0},
                "elapsed_s": 12.0,
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
        elif path.name == "annotation_website_export_report.json":
            payload = {
                "package_root": str(root / "annotation_website_import"),
                "annotation_count": 1,
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
        else:
            path.write_text(f"behavior\n{event_value}\n", encoding="utf-8")

    website = root / "annotation_website_import" / "sample"
    website.mkdir(parents=True)
    (website / "annotations.json").write_text(
        json.dumps({"annotations": [{"behavior": event_value}]}),
        encoding="utf-8",
    )
    (website / "tracks.jsonl").write_text('{"frame":0}\n', encoding="utf-8")


def test_complete_output_comparison_ignores_only_runtime_metadata(tmp_path: Path):
    baseline = tmp_path / "baseline"
    optimized = tmp_path / "optimized"
    _write_output_tree(baseline)
    _write_output_tree(optimized)

    optimized_metadata = optimized / "lightweight_analysis_metadata.json"
    payload = json.loads(optimized_metadata.read_text(encoding="utf-8"))
    payload["stage_timings_s"] = {"candidate_pair_analysis": 1.0}
    payload["elapsed_s"] = 2.0
    payload["parallel_behavior_fsm"]["execution_semantics"] = "active_temporal_regions"
    optimized_metadata.write_text(json.dumps(payload), encoding="utf-8")

    report = compare_output_directories(baseline, optimized)

    assert report.equivalent
    assert report.differences == ()


def test_complete_output_comparison_detects_behavior_content_change(tmp_path: Path):
    baseline = tmp_path / "baseline"
    optimized = tmp_path / "optimized"
    _write_output_tree(baseline)
    _write_output_tree(optimized)
    (optimized / "lightweight_behavior_events.csv").write_text(
        "behavior\nattack\n",
        encoding="utf-8",
    )

    report = compare_output_directories(baseline, optimized)

    assert not report.equivalent
    assert any(
        difference.path == "lightweight_behavior_events.csv" and difference.kind == "content"
        for difference in report.differences
    )


def test_complete_output_comparison_detects_missing_required_file(tmp_path: Path):
    baseline = tmp_path / "baseline"
    optimized = tmp_path / "optimized"
    _write_output_tree(baseline)
    _write_output_tree(optimized)
    (optimized / "lightweight_contact_events.csv").unlink()

    report = compare_output_directories(baseline, optimized)

    assert not report.equivalent
    assert any(
        difference.path == "lightweight_contact_events.csv" and difference.kind == "missing"
        for difference in report.differences
    )
