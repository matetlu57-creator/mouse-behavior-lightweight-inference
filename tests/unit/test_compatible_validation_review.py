from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from build_compatible_validation_review import (  # noqa: E402
    _browser_compatible_video_path,
    _metrics,
    _score_row,
)


def _row(
    truth: str,
    predicted: str,
    label_ids: str = "1,2",
    predicted_ids: str = "1,2",
) -> dict[str, str]:
    return {
        "label_behavior": truth,
        "label_target_ids": label_ids,
        "predicted_behavior": predicted,
        "predicted_target_ids": predicted_ids,
    }


def test_approach_contact_dual_label_credit_is_removed() -> None:
    # [SUSPENDED-STEP3 2026-10-04] dual-label credit removed: Together now allows contact without interaction
    # Approach ↔ 鼻头接触无论累计接触时长多长都不再获得 compatible 信用；只保留严格 Top-1。
    evidence = {"pair_scores": {"1,2": {"nose_head_contact": 5.0}}}
    scored = _score_row(_row("Approach", "nose_head_contact"), evidence)

    assert scored["strict_top1_correct"] == "false"
    assert scored["compatible_top1_correct"] == "false"
    assert scored["compatible_labels"] == ""
    assert "compatible_dual_label" not in scored
    assert "contact_seconds" not in scored


def test_together_contact_prediction_earns_no_compatible_credit() -> None:
    # [SUSPENDED-STEP3 2026-10-04] dual-label credit removed: Together now allows contact without interaction
    # 同一对 Together 标签被预测为 Snout-head contact 时，compatible 仍必须为 False。
    scored = _score_row(
        _row("Together", "nose_head_contact"),
        {"pair_scores": {"1,2": {"nose_head_contact": 0.0}}},
    )
    assert scored["strict_top1_correct"] == "false"
    assert scored["compatible_top1_correct"] == "false"

    wrong_pair = _score_row(
        _row("Together", "nose_head_contact", predicted_ids="1,3"),
        {"pair_scores": {"1,2": {"nose_head_contact": 2.0}}},
    )
    assert wrong_pair["compatible_top1_correct"] == "false"


def test_same_behavior_prediction_may_include_extra_target_ids() -> None:
    scored = _score_row(
        _row("Social clustering", "social_clustering", predicted_ids="1,2,3"),
        {},
    )
    assert scored["strict_top1_correct"] == "false"
    assert scored["compatible_top1_correct"] == "true"
    assert scored["target_id_hit"] == "true"
    assert scored["exact_target_id_hit"] == "false"


def test_compatible_metrics_preserve_strict_and_compatible_counts_separately() -> None:
    strict_row = _score_row(_row("Approach", "approach"), {})
    # [SUSPENDED-STEP3 2026-10-04] dual-label credit removed: Together now allows contact without interaction
    # 兼容命中现在只来自同行为目标 ID 包含路径；原双标签行已替换为包含命中行。
    containment_row = _score_row(
        _row("Approach", "approach", predicted_ids="1,2,3"),
        {},
    )
    metrics = _metrics([strict_row, containment_row])
    approach = next(row for row in metrics["perBehavior"] if row["key"] == "approach")

    assert metrics["strictCorrect"] == 1
    assert metrics["compatibleCorrect"] == 2
    assert approach["strictRecallCorrect"] == 1
    assert approach["compatibleRecallCorrect"] == 2


def test_viewer_prefers_h264_copy_and_preserves_rendered_source(tmp_path: Path) -> None:
    render_dir = tmp_path / "rendered"
    source_dir = render_dir / "videos"
    source_dir.mkdir(parents=True)
    source = source_dir / "clip.mp4"
    source.write_bytes(b"mp4v")

    assert _browser_compatible_video_path(source) == source.resolve()

    browser_copy = render_dir / "video_viewer_media" / "videos" / "clip.mp4"
    browser_copy.parent.mkdir(parents=True)
    browser_copy.write_bytes(b"h264")

    assert _browser_compatible_video_path(source) == browser_copy.resolve()
    assert source.read_bytes() == b"mp4v"
