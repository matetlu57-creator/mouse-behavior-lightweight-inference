"""Unit and regression tests for standalone offline failure report generation."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest


from scripts.generate_offline_failure_report import (
    BEHAVIOR_CHINESE,
    build_report_data,
    generate_offline_report,
)


def test_build_report_data_structure():
    eval_json = Path("outputs/reconciled_behavior_evaluation.json")
    comp_json = Path("outputs/dataset_comparison_table.json")

    assert eval_json.is_file(), "Evaluation JSON must exist"
    assert comp_json.is_file(), "Comparison JSON must exist"

    data = build_report_data(eval_json, comp_json)

    # 1. Structure checks
    assert "metadata" in data
    assert "reconciliation_summary" in data
    assert "overall_summary" in data
    assert "behaviors" in data
    assert "failures" in data

    # 2. Check 15 behaviors present with Precision & Recall fields
    assert len(data["behaviors"]) == 15
    for b in data["behaviors"]:
        assert b["behavior"] in BEHAVIOR_CHINESE
        assert "dataset_b" in b
        db = b["dataset_b"]
        assert "primary_precision" in db and "primary_recall" in db
        assert "pf_target_precision" in db and "pf_target_recall" in db
        assert "pf_joint_precision" in db and "pf_joint_recall" in db
        if db["primary_predicted_count"] > 0:
            assert db["primary_precision"] == pytest.approx(
                db["primary_correct"] / db["primary_predicted_count"], abs=1e-3
            )
        assert db["primary_recall"] == pytest.approx(db["primary_correct"] / db["support"], abs=1e-3)

    # 3. Check failures
    failures = data["failures"]
    assert len(failures) == 133, f"Expected 133 failure instances, got {len(failures)}"

    # Check failure categorization
    categories = {f["failure_category"] for f in failures}
    assert categories == {"misclassified", "missed"}

    # Check union_b_priority counts
    union_b_fails = [f for f in failures if f["in_union_b_priority"]]
    assert len(union_b_fails) == 107, f"Expected 107 failures in Union B, got {len(union_b_fails)}"

    missed_count = sum(1 for f in union_b_fails if f["failure_category"] == "missed")
    misclassified_count = sum(1 for f in union_b_fails if f["failure_category"] == "misclassified")
    assert missed_count == 52, f"Expected 52 missed, got {missed_count}"
    assert misclassified_count == 55, f"Expected 55 misclassified, got {misclassified_count}"


def test_generate_offline_report_html(tmp_path: Path):
    eval_json = Path("outputs/reconciled_behavior_evaluation.json")
    comp_json = Path("outputs/dataset_comparison_table.json")
    out_html = tmp_path / "test_report.html"

    result_path = generate_offline_report(eval_json, comp_json, out_html)
    assert result_path.is_file()
    assert result_path.stat().st_size > 100_000, "Report HTML must be non-empty and rich"

    content = result_path.read_text(encoding="utf-8")

    # 1. Zero external network dependencies
    ext_links = re.findall(r'(?:href|src)=["\']https?://[^"\']+["\']', content)
    assert len(ext_links) == 0, f"Found external network dependencies: {ext_links}"

    # 2. Inlined REPORT_DATA script
    assert "const REPORT_DATA = {" in content
    assert "// Global App State" in content

    # 3. Essential UI sections
    assert "全 15 类行为多数据集核对与准确率对比总表" in content
    assert "table-unified-pr" in content
    assert "精确率 (Precision) / 召回率 (Recall) / F1" in content
    assert "冲突样本深度排查与归因分析" in content
    assert "52512294eac95441035cb075" in content
    assert "未通过视频片段交互审查器" in content
    assert "双轨时序交互时间轴" in content

    # 4. Failure Categorization
    assert "【识别行为出错】" in content or "识别出错" in content
    assert "【行为未识别到】" in content or "未识别到" in content

    # 5. Local video references
    assert "./videos/" in content


def test_rendered_videos_exist_for_all_failures():
    videos_dir = Path("outputs/failure_report_offline/videos")
    assert videos_dir.is_dir(), "Videos directory must exist"

    eval_json = Path("outputs/reconciled_behavior_evaluation.json")
    data = json.loads(eval_json.read_text(encoding="utf-8"))
    failures = [c for c in data["clips"] if not c.get("primary_correct")]

    missing_ds_videos = []
    missing_generic_videos = []

    for f in failures:
        ds = f["dataset"]
        cid = f["clip_id"]
        ds_file = videos_dir / f"{ds}_{cid}.mp4"
        generic_file = videos_dir / f"{cid}.mp4"

        if not ds_file.is_file() or ds_file.stat().st_size == 0:
            missing_ds_videos.append(str(ds_file))
        if not generic_file.is_file() or generic_file.stat().st_size == 0:
            missing_generic_videos.append(str(generic_file))

    assert len(missing_ds_videos) == 0, f"Missing dataset-prefixed videos: {missing_ds_videos[:5]}"
    assert len(missing_generic_videos) == 0, f"Missing generic videos: {missing_generic_videos[:5]}"
