from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from build_validation_review_data import (  # noqa: E402
    _error_diagnostic,
    _feature_failures,
    _strict_and_compatible,
    _video_url,
)
from build_video_review_viewer_data import _relative_video_url  # noqa: E402
from mouse_behavior.evaluation.exported_dataset import HeuristicParameters  # noqa: E402


def test_dual_label_credit_is_removed_and_compatible_equals_strict() -> None:
    """[SUSPENDED-STEP3 2026-10-04] dual-label credit removed: Together now allows contact without interaction."""
    ids = (3, 8)
    # 同一对目标 ID 的 Together↔鼻头/鼻尾接触同层双标签不再获得 compatible 信用。
    assert _strict_and_compatible("together", "nose_head_contact", ids, ids) == (False, False)
    assert _strict_and_compatible("together", "nose_tail_contact", ids, ids) == (False, False)
    assert _strict_and_compatible("nose_head_contact", "together", ids, ids) == (False, False)
    # 两个接触类别互不等价，双标签信用删除后依旧不兼容。
    assert _strict_and_compatible("nose_head_contact", "nose_tail_contact", ids, ids) == (False, False)
    # 不同目标 ID 对从来就不兼容，现在依旧不兼容。
    assert _strict_and_compatible("together", "nose_head_contact", ids, (3, 9)) == (False, False)
    # 严格 Top-1 计分不受影响：行为相同且目标 ID 集合完全相同才命中。
    assert _strict_and_compatible("together", "together", ids, ids) == (True, True)
    assert _strict_and_compatible("together", "together", ids, (3, 9)) == (False, False)


def test_failure_report_uses_behavior_thresholds_and_omits_unavailable_measurements() -> None:
    parameters = HeuristicParameters(isolation_distance_cm=8.0)
    failures = _feature_failures(
        "isolation",
        {"nearest_p10_cm": 6.2},
        parameters,
        mouse_width_cm=2.4,
    )
    assert failures == ["最近邻距离 P10=6.20cm，未满足 >=8cm"]

    unavailable = _feature_failures(
        "isolation",
        {"nearest_p10_cm": None},
        parameters,
        mouse_width_cm=2.4,
    )
    assert "轨迹摘要不可用" in unavailable[0]


def test_collective_diagnostic_names_failed_handcrafted_gates() -> None:
    diagnostic = _error_diagnostic(
        {
            "canonical_behavior": "huddle",
            "mouse_ids": "2,4,7",
            "tracks_path": "unused.json",
            "fps": "30",
        },
        {"predicted_behavior": "social_clustering", "predicted_target_ids": "2,4,7"},
        {},
        HeuristicParameters(),
        [
            {
                "state": "HUDDLING",
                "member_ids": "2,4,7",
                "checks": '{"duration": true, "low_motion": false, "contact_density": false}',
            }
        ],
    )

    assert diagnostic["failedConditions"] == [
        "冻结启发式的目标群体候选未通过：低运动成员比例不足；接触图密度不足"
    ]


def test_video_urls_keep_the_outputs_subdirectory_for_static_file_serving() -> None:
    output_root = REPO_ROOT / "outputs" / "collective_state_features_20260928_v7"
    rendered_video = (
        "outputs/collective_state_features_20260928_v7/"
        "rendered_validation_v9/sample.mp4"
    )

    expected = (
        "/video-files/collective_state_features_20260928_v7/"
        "rendered_validation_v9/sample.mp4"
    )
    assert _video_url(rendered_video, output_root) == expected
    assert _relative_video_url(rendered_video, output_root) == expected
