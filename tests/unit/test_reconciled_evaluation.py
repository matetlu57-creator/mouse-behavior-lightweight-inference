"""Unit and regression tests for reconciled Dataset A + Dataset B evaluation."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from mouse_behavior.evaluation.clip_benchmark import (
    evaluate_six_behavior_clip,
    evaluate_static_loco_clip,
)
from mouse_behavior.evaluation.exported_dataset_models import (
    CANONICAL_BEHAVIORS,
    LABEL_TO_CANONICAL,
)
from mouse_behavior.evaluation.reconciled_evaluation import (
    BEHAVIOR_DISPLAY_ORDER,
    DISPLAY_BEHAVIOR_ALIASES,
    FAILURE_SUBCATEGORIES,
    aggregate_cohort_metrics,
    evaluate_and_reconcile_datasets,
    evaluate_single_dataset,
    index_dataset,
    normalize_display_behavior,
    reconcile_datasets,
    write_reconciled_outputs,
)
from mouse_behavior.evaluation.social_heuristic_config import (
    DEFAULT_SOCIAL_HEURISTIC_CONFIG,
    OPTIMIZED_SOCIAL_HEURISTIC_CONFIG,
)


def _mouse_detection(
    track_id: int,
    center_cm: tuple[float, float],
    direction: tuple[float, float] = (1.0, 0.0),
    *,
    px_per_cm: float = 10.0,
    body_cm: float = 7.918,
    half_width_cm: float = 1.0,
) -> dict:
    dx, dy = float(direction[0]), float(direction[1])
    norm = float(np.hypot(dx, dy)) or 1.0
    fwd = np.array([dx / norm, dy / norm], dtype=float)
    side = np.array([-fwd[1], fwd[0]], dtype=float)
    c = np.array([float(center_cm[0]), float(center_cm[1])], dtype=float)
    half_len = 0.5 * body_cm
    # Export order: nose, right_ear, left_ear, neck, left_hip, right_hip, tail
    rel_points = (
        (half_len, 0.0),
        (0.32 * body_cm, -0.8 * half_width_cm),
        (0.32 * body_cm, 0.8 * half_width_cm),
        (0.12 * body_cm, 0.0),
        (-0.25 * body_cm, 0.9 * half_width_cm),
        (-0.25 * body_cm, -0.9 * half_width_cm),
        (-half_len, 0.0),
    )
    kps_cm = [c + rx * fwd + ry * side for rx, ry in rel_points]
    kps_px = [[float(p[0] * px_per_cm), float(p[1] * px_per_cm), 0.99] for p in kps_cm]
    xs = [p[0] for p in kps_px]
    ys = [p[1] for p in kps_px]
    box = [min(xs) - 8.0, min(ys) - 8.0, max(xs) + 8.0, max(ys) + 8.0]
    return {
        "track_id": int(track_id),
        "box": box,
        "keypoints": kps_px,
        "detection_confidence": 0.95,
        "class_id": 0,
    }


def _write_clip(
    dataset_root: Path,
    behavior: str,
    clip_id: str,
    *,
    mouse_ids: list[int],
    tracks: list[dict],
    fps: float = 30.0,
    participants: list[dict] | None = None,
) -> Path:
    clip_dir = dataset_root / behavior / clip_id
    clip_dir.mkdir(parents=True, exist_ok=True)
    frame_count = len(tracks)
    (clip_dir / "annotation.json").write_text(
        json.dumps(
            {
                "behavior": behavior,
                "mouse_ids": mouse_ids,
                "participants": participants or [],
                "confidence": "human_confirmed",
                "frame_range": {"start": 0, "end": frame_count - 1},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (clip_dir / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "clip": {
                    "filename": "clip.mp4",
                    "fps": fps,
                    "width": 1920,
                    "height": 1080,
                    "frame_count": frame_count,
                },
                "source": {"video_filename": "test_session_part001.mp4", "start_frame": 0, "end_frame": frame_count - 1},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (clip_dir / "tracks.json").write_text(json.dumps(tracks), encoding="utf-8")
    return clip_dir


def test_reconcile_evaluation_contracts_and_failure_diagnosis(tmp_path: Path) -> None:
    ds_a = tmp_path / "dataset_a"
    ds_b = tmp_path / "dataset_b"
    fps = 30.0

    # 1. Shared clip ID with label conflict (Snout-head_contact in A vs Snout-rear_contact in B)
    #    and different background mouse track count.
    conflict_frames_a = [
        {
            "frame": f,
            "detections": [
                _mouse_detection(10, (20.0, 20.0), (1.0, 0.0)),
                _mouse_detection(20, (28.4, 20.0), (-1.0, 0.0)),  # noses ~0.48cm apart
                _mouse_detection(99, (80.0 + 0.5 * f, 80.0), (1.0, 0.0)),  # fast/moving distractor
            ],
        }
        for f in range(45)
    ]
    conflict_frames_b = [
        {
            "frame": f,
            "detections": [
                _mouse_detection(10, (20.0, 20.0), (1.0, 0.0)),
                _mouse_detection(20, (28.4, 20.0), (-1.0, 0.0)),
            ],
        }
        for f in range(45)
    ]
    _write_clip(ds_a, "Snout-head_contact", "52512294eac95441035cb075", mouse_ids=[10, 20], tracks=conflict_frames_a, fps=fps)
    _write_clip(ds_b, "Snout-rear_contact", "52512294eac95441035cb075", mouse_ids=[10, 20], tracks=conflict_frames_b, fps=fps)

    # 2. Wide Together clip in A only (calm side-by-side at d=7.2 cm with body_gap >= 1.8 cm + moving background mouse)
    together_frames = [
        {
            "frame": f,
            "detections": [
                _mouse_detection(1, (30.0, 30.0), (1.0, 0.0)),
                _mouse_detection(2, (30.0, 37.2), (1.0, 0.0)),
                _mouse_detection(3, (80.0 + 0.6 * f, 80.0), (1.0, 0.0)),
            ],
        }
        for f in range(45)
    ]
    _write_clip(ds_a, "Together", "clip_together_only_a", mouse_ids=[1, 2], tracks=together_frames, fps=fps)

    # 3. Missing target ID clip in A (target ID 999 absent from tracks -> must NEVER infer ID!)
    missing_frames = [
        {
            "frame": f,
            "detections": [
                _mouse_detection(1, (20.0, 20.0), (1.0, 0.0)),
                _mouse_detection(2, (30.0, 20.0), (-1.0, 0.0)),
            ],
        }
        for f in range(30)
    ]
    _write_clip(ds_a, "Approach", "clip_missing_id", mouse_ids=[1, 999], tracks=missing_frames, fps=fps)

    # 4. Grooming & Rearing in B (explicitly evaluated under canonical default enable_posture_screen=False)
    static_mouse_frames = [
        {"frame": f, "detections": [_mouse_detection(7, (40.0, 40.0), (1.0, 0.0))]}
        for f in range(40)
    ]
    _write_clip(ds_b, "Grooming", "clip_grooming", mouse_ids=[7], tracks=static_mouse_frames, fps=fps)
    _write_clip(ds_b, "Rearing", "clip_rearing", mouse_ids=[7], tracks=static_mouse_frames, fps=fps)

    # 5. Static clip in B (should succeed in evaluate_six_behavior_clip & pf_joint)
    _write_clip(ds_b, "Static", "clip_static", mouse_ids=[7], tracks=static_mouse_frames, fps=fps)

    # 6. Huddling clip in B (3 mice close together for 6s = 180 frames)
    huddle_frames = [
        {
            "frame": f,
            "detections": [
                _mouse_detection(1, (40.0, 40.0), (1.0, 0.0)),
                _mouse_detection(2, (43.5, 40.0), (-1.0, 0.0)),
                _mouse_detection(3, (40.0, 43.5), (0.0, -1.0)),
            ],
        }
        for f in range(180)
    ]
    _write_clip(ds_b, "Huddling", "clip_huddle", mouse_ids=[1, 2, 3], tracks=huddle_frames, fps=fps)

    # 7. Huddling clip in B where all 4 target IDs appear in tracks.json, yet ID 4 only appears
    #    after ID 1 drops out (coobserved_frames == 0) while subset [1, 2, 3] triggers Huddling.
    disjoint_huddle_frames = [
        {
            "frame": f,
            "detections": [
                _mouse_detection(1 if f < 170 else 4, (40.0, 40.0), (1.0, 0.0)),
                _mouse_detection(2, (43.5, 40.0), (-1.0, 0.0)),
                _mouse_detection(3, (40.0, 43.5), (0.0, -1.0)),
            ],
        }
        for f in range(180)
    ]
    _write_clip(ds_b, "Huddling", "clip_huddle_disjoint", mouse_ids=[1, 2, 3, 4], tracks=disjoint_huddle_frames, fps=fps)

    # Verify DEFAULT_SOCIAL_HEURISTIC_CONFIG remains untouched vs OPTIMIZED_SOCIAL_HEURISTIC_CONFIG
    assert DEFAULT_SOCIAL_HEURISTIC_CONFIG != OPTIMIZED_SOCIAL_HEURISTIC_CONFIG

    entries_a = index_dataset(ds_a, "A")
    entries_b = index_dataset(ds_b, "B")
    rec = reconcile_datasets(entries_a, entries_b)
    assert rec["dataset_a_total"] == 3
    assert rec["dataset_b_total"] == 6
    assert rec["overlap_count"] == 1
    assert rec["union_count"] == 8
    assert len(rec["label_conflicts"]) == 1
    assert rec["label_conflicts"][0]["clip_id"] == "52512294eac95441035cb075"
    assert rec["overlap_file_differences"]["tracks_detections_differ"] == 1

    report_seq = evaluate_and_reconcile_datasets(
        dataset_a_root=ds_a,
        dataset_b_root=ds_b,
        workers=1,
    )
    report_par = evaluate_and_reconcile_datasets(
        dataset_a_root=ds_a,
        dataset_b_root=ds_b,
        workers=2,
    )
    assert report_seq["cohorts"]["union_b_priority"]["total_clips"] == 8
    assert report_par["cohorts"]["union_b_priority"]["total_clips"] == 8
    assert report_seq["inputs"]["dual_label_credit_enabled"] is False
    assert report_seq["inputs"]["enable_posture_screen"] is False

    by_key = {c["record_key"]: c for c in report_seq["clips"]}
    by_key_par = {c["record_key"]: c for c in report_par["clips"]}
    assert by_key.keys() == by_key_par.keys()
    for key in by_key:
        assert by_key[key]["primary_correct"] == by_key_par[key]["primary_correct"]
        assert by_key[key]["pf_joint_correct"] == by_key_par[key]["pf_joint_correct"]

    # Conflict clip in A (Snout-head_contact): pf_pair passes, but distractor mouse #99 steals pf_joint!
    rec_conf_a = by_key["A:Snout-head_contact:52512294eac95441035cb075"]
    assert rec_conf_a["primary_correct"] is True
    assert rec_conf_a["pf_joint_correct"] is False
    assert rec_conf_a["coverage_a2_correct"] is True
    assert rec_conf_a["true_target_coverage"] >= 0.5
    assert rec_conf_a["joint_failure_subcategory"] == "other_pair_won_joint"

    # Conflict clip in B (Snout-rear_contact): head_contact triggers -> misclassified wrong_behavior_triggered!
    rec_conf_b = by_key["B:Snout-rear_contact:52512294eac95441035cb075"]
    assert rec_conf_b["primary_correct"] is False
    assert rec_conf_b["failure_category"] == "misclassified"
    assert rec_conf_b["failure_subcategory"] == "wrong_behavior_triggered"
    assert rec_conf_b["primary_predicted_behavior"] == "Snout-head_contact"
    assert rec_conf_b["failure_reason_zh"]

    # Together clip in A: wide branch passes pf_pair, fails pf_joint due to distractor mouse #3
    rec_tog = by_key["A:Together:clip_together_only_a"]
    assert rec_tog["primary_correct"] is True
    assert rec_tog["hit_mask"] is True
    assert rec_tog["pf_joint_correct"] is False

    # Missing ID clip: never inferred, classified missed/target_id_missing_in_tracks
    rec_miss = by_key["A:Approach:clip_missing_id"]
    assert rec_miss["primary_correct"] is False
    assert rec_miss["coverage_a2_correct"] is False
    assert rec_miss["true_target_coverage"] == 0
    assert rec_miss["failure_category"] == "missed"
    assert rec_miss["failure_subcategory"] == "target_id_missing_in_tracks"

    # Grooming & Rearing: explicitly disabled under canonical default
    for beh, cid in (("Grooming", "clip_grooming"), ("Rearing", "clip_rearing")):
        r = by_key[f"B:{beh}:{cid}"]
        assert r["primary_correct"] is False
        assert r["canonical_rule_status"] == "disabled_suspended"
        assert r["failure_category"] == "missed"
        assert r["failure_subcategory"] == "disabled_rule"

    # Static & Huddling
    rec_static = by_key["B:Static:clip_static"]
    assert rec_static["primary_correct"] is True
    assert rec_static["pf_joint_correct"] is True

    rec_hud = by_key["B:Huddling:clip_huddle"]
    assert rec_hud["huddle_diagnostics"]["a1_hit"] is True
    assert rec_hud["huddle_diagnostics"]["a2_hit"] is True
    assert rec_hud["timeline_series"]["target_behavior_intervals"]

    rec_hud_disjoint = by_key["B:Huddling:clip_huddle_disjoint"]
    assert rec_hud_disjoint["missing_target_ids"] == []
    assert rec_hud_disjoint["huddle_diagnostics"]["a1_hit"] is True
    assert rec_hud_disjoint["failure_category"] == "misclassified"
    assert rec_hud_disjoint["failure_subcategory"] == "other_pair_won_joint"

    # Every failure has valid category/subcategory/Chinese explanation
    for clip in report_seq["clips"]:
        if not clip["primary_correct"]:
            assert clip["failure_category"] in {"missed", "misclassified"}
            assert clip["failure_subcategory"] in FAILURE_SUBCATEGORIES
            assert isinstance(clip["failure_reason_zh"], str) and len(clip["failure_reason_zh"]) > 10

    # Write outputs and verify strict RFC JSON & companion files
    out_json = tmp_path / "out" / "reconciled_behavior_evaluation.json"
    written = write_reconciled_outputs(report_seq, output_json=out_json)
    for path in written.values():
        assert path.is_file() and path.stat().st_size > 0
    parsed = json.loads(out_json.read_text(encoding="utf-8"))
    assert parsed["reconciliation"]["union_count"] == 8
    assert "union_b_priority" in parsed["cohorts"]
    assert "union_a_priority" in parsed["cohorts"]
    assert pytest.approx(parsed["cohorts"]["dataset_b"]["per_behavior"]["Grooming"]["primary_accuracy"]) == 0.0


@pytest.mark.parametrize("behavior,canonical", [("Grooming", "grooming"), ("Rearing", "rearing")])
def test_all_behaviors_enables_posture_and_separates_exact_coverage_from_top1(
    tmp_path: Path, monkeypatch, behavior: str, canonical: str,
) -> None:
    from mouse_behavior.evaluation import exported_dataset_prediction as prediction
    from mouse_behavior.evaluation import reconciled_evaluation as evaluation

    ds_a, ds_b = tmp_path / "a", tmp_path / "b"
    ds_a.mkdir()
    frames = [{"frame": f, "detections": [
        _mouse_detection(7, (40.0, 40.0)), _mouse_detection(99, (80.0, 80.0)),
    ]} for f in range(60)]
    _write_clip(ds_b, behavior, "posture", mouse_ids=[7], tracks=frames)
    # A non-posture truth label must receive exactly the same blind candidates.
    _write_clip(ds_b, "Static", "same_tracks", mouse_ids=[7], tracks=frames)
    original = prediction.individual_rule_masks
    enabled_calls = []

    def posture_evidence(*args, **kwargs):
        result = original(*args, **kwargs)
        enabled = kwargs["config"].enable_posture_screen
        enabled_calls.append(enabled)
        masks = {name: np.zeros_like(mask) for name, mask in result.masks.items()}
        masks["locomotion"][:, 1] = True  # Background ID wins blind Top-1.
        if enabled:
            assert kwargs["boxes_cm"] is not None
            masks[canonical][:36, 0] = True
        return replace(result, masks=masks)

    monkeypatch.setattr(prediction, "individual_rule_masks", posture_evidence)
    monkeypatch.setattr(evaluation, "individual_rule_masks", posture_evidence)
    report = evaluate_and_reconcile_datasets(
        dataset_a_root=ds_a, dataset_b_root=ds_b, workers=1, all_behaviors=True,
    )
    rec = next(r for r in report["clips"] if r["behavior"] == behavior)
    assert enabled_calls and all(enabled_calls)
    assert rec["enable_posture_screen"] is True
    assert rec["canonical_rule_status"] == "enabled"
    assert rec["failure_subcategory"] is None
    assert rec["joint_failure_subcategory"] != "disabled_rule"
    assert rec["true_target_coverage"] == pytest.approx(0.6)
    assert rec["coverage_a2_correct"] is True
    assert rec["primary_correct"] is True
    assert rec["pf_joint_correct"] is False
    assert any(c["canonical_behavior"] == canonical and c["target_ids"] == [7]
               for c in rec["top_competing_candidates"])
    assert report["clips"][0]["top_competing_candidates"] == report["clips"][1]["top_competing_candidates"]
    assert report["inputs"]["all_behaviors"] is True
    assert report["inputs"]["suspended_posture_behaviors"] == []
    assert report["provenance"]["git_head"]
    assert len(report["provenance"]["config_sha256"]) == 64
    hashes = report["provenance"]["source_sha256"]
    assert "src/mouse_behavior/evaluation/exported_dataset_prediction.py" in hashes
    assert "scripts/generate_reconciled_failure_report.py" in hashes
    assert report["cohorts"]["dataset_b"]["overall_coverage_a2_correct"] >= 1
    assert report["cohorts"]["dataset_b"]["per_behavior"][behavior]["rule_enabled"] is True
    markdown = evaluation.build_summary_markdown(report)
    assert "分项检测通过率（不同专项方法，不等于统一Top-1）" in markdown
    assert "统一 exact-ID 覆盖" in markdown
    assert "Dataset A" in markdown and "Dataset B" in markdown and "Union" in markdown

    # The unified total must retain all 15 classes, including disabled posture rows.
    all_rows = [{**rec, "behavior": b, "coverage_a2_correct": b == behavior,
                 "primary_correct": b == behavior} for b in BEHAVIOR_DISPLAY_ORDER]
    cohort = aggregate_cohort_metrics(all_rows, "all15")
    assert cohort["total_clips"] == 15
    assert len(cohort["per_behavior"]) == 15
    assert cohort["overall_coverage_a2_correct"] == 1
    assert cohort["overall_coverage_a2_accuracy"] == pytest.approx(1 / 15, abs=0.0001)


def test_exact_coverage_uses_single_id_and_isolation_event_frames(tmp_path: Path, monkeypatch) -> None:
    from mouse_behavior.evaluation import reconciled_evaluation as evaluation
    from mouse_behavior.evaluation.blind_top1 import Candidate

    ds_a, ds_b = tmp_path / "a", tmp_path / "b"
    ds_a.mkdir()
    frames = [{"frame": f, "detections": [
        _mouse_detection(7, (40.0, 40.0)), _mouse_detection(8, (80.0, 80.0)),
    ]} for f in range(60)]
    _write_clip(ds_b, "Static", "single_id", mouse_ids=[7], tracks=frames)
    _write_clip(ds_b, "Isolation", "isolation", mouse_ids=[7], tracks=frames)
    monkeypatch.setattr(evaluation, "isolation_candidates", lambda *args: (
        Candidate("isolation", (7,), 0.8, ((0, 24),)),
        Candidate("isolation", (8,), 1.2, ((24, 60),)),
    ))
    original = evaluation.individual_rule_masks

    def distinct_individual_intervals(*args, **kwargs):
        result = original(*args, **kwargs)
        masks = {name: np.zeros_like(mask) for name, mask in result.masks.items()}
        masks["stationary"][:24, 0] = True
        masks["stationary"][24:, 1] = True
        return replace(result, masks=masks)

    monkeypatch.setattr(evaluation, "individual_rule_masks", distinct_individual_intervals)
    report = evaluate_and_reconcile_datasets(
        dataset_a_root=ds_a, dataset_b_root=ds_b, workers=1, all_behaviors=True,
    )
    by_behavior = {r["behavior"]: r for r in report["clips"]}
    assert by_behavior["Static"]["true_target_coverage"] == pytest.approx(0.4)
    assert by_behavior["Static"]["coverage_a2_correct"] is False
    assert by_behavior["Isolation"]["true_target_coverage"] == pytest.approx(0.4)
    assert by_behavior["Isolation"]["coverage_a2_correct"] is False


def test_all_behavior_huddling_primary_uses_exact_a2_not_background_top1(tmp_path: Path) -> None:
    ds_a, ds_b = tmp_path / "a", tmp_path / "b"
    ds_a.mkdir()
    frames = [{"frame": f, "detections": [
        _mouse_detection(1, (40.0, 40.0)),
        _mouse_detection(2, (43.5, 40.0), (-1.0, 0.0)),
        _mouse_detection(3, (40.0, 43.5) if f < 165 else (200.0, 200.0), (0.0, -1.0)),
        _mouse_detection(99, (80.0 + f * 0.5, 80.0)),
    ]} for f in range(180)]
    _write_clip(ds_b, "Huddling", "huddling", mouse_ids=[1, 2, 3], tracks=frames)
    report = evaluate_and_reconcile_datasets(
        dataset_a_root=ds_a, dataset_b_root=ds_b, workers=1, all_behaviors=True,
    )
    rec = report["clips"][0]
    assert rec["huddle_diagnostics"]["a2_hit"] is True
    assert rec["coverage_a2_correct"] is True
    assert rec["primary_correct"] is True
    assert rec["pf_joint_correct"] is False


@pytest.mark.parametrize("behavior,canonical", [
    ("Huddling", "huddle"), ("Social clustering", "social_clustering"),
    ("Group locomotion", "group_locomotion"), ("Dispersal", "dispersal"),
])
@pytest.mark.parametrize("intervals,other_ids,coverage", [
    (((0, 18), (30, 48)), (1, 2, 3), 0.6),  # Same IDs, disjoint episodes add.
    (((0, 24), (12, 24)), (1, 2, 3), 0.4),  # Overlap counts once.
    (((0, 24), (24, 60)), (1, 2, 4), 0.4),  # Other IDs cannot fill coverage.
])
def test_group_coverage_unions_only_identical_ids(
    tmp_path, monkeypatch, behavior, canonical, intervals, other_ids, coverage,
):
    from mouse_behavior.evaluation import reconciled_evaluation as evaluation

    ds_a, ds_b = tmp_path / "a", tmp_path / "b"
    ds_a.mkdir()
    frames = [{"frame": f, "detections": [
        _mouse_detection(i, (20.0 * i, 40.0)) for i in range(1, 5)
    ]} for f in range(60)]
    _write_clip(ds_b, behavior, "episodes", mouse_ids=[1, 2, 3], tracks=frames)
    original = evaluation.predict_features

    def episodes(*args, **kwargs):
        pred = original(*args, **kwargs)
        pred["group_events"] = [
            {"behavior": canonical, "member_ids": ids, "frame_intervals": [iv],
             "evidence_seconds": (iv[1] - iv[0]) / 30.0}
            for ids, iv in zip(((1, 2, 3), other_ids), intervals)
        ]
        return pred

    monkeypatch.setattr(evaluation, "predict_features", episodes)
    report = evaluate_and_reconcile_datasets(
        dataset_a_root=ds_a, dataset_b_root=ds_b, workers=1, all_behaviors=True,
    )
    rec = report["clips"][0]
    assert rec["true_target_coverage"] == pytest.approx(coverage)
    assert rec["coverage_a2_correct"] is (coverage >= 0.5)
    if behavior == "Huddling":
        assert rec["huddle_diagnostics"]["a2_hit"] is (coverage >= 0.5)
        assert rec["primary_correct"] is (coverage >= 0.5)
        assert rec["huddle_diagnostics"]["a2_max_single_event_hit"] is False


def test_source_scales_apply_only_to_b_and_require_complete_mapping(tmp_path: Path) -> None:
    ds_a, ds_b = tmp_path / "a", tmp_path / "b"
    frames = [{"frame": f, "detections": [_mouse_detection(7, (40.0, 40.0))]}
              for f in range(40)]
    for root in (ds_a, ds_b):
        _write_clip(root, "Static", "shared", mouse_ids=[7], tracks=frames)
    scales = tmp_path / "scales.json"
    scales.write_text(json.dumps([{
        "sample_id": "shared", "scale_source": "frozen_test_receipt",
        "scale_config": {"mode": "fixed", "cm_per_pixel": 0.2,
                         "assumed_mouse_body_length_cm": 7.918},
    }]), encoding="utf-8")
    report = evaluate_and_reconcile_datasets(
        dataset_a_root=ds_a, dataset_b_root=ds_b, workers=1,
        all_behaviors=True, dataset_b_source_scales=scales,
    )
    a, b = report["clips"]
    assert a["cm_per_pixel"] == pytest.approx(0.1)
    assert b["cm_per_pixel"] == pytest.approx(0.2)
    assert b["scale_source"] == "frozen_test_receipt"
    assert a["scale_source"] != b["scale_source"]
    assert len(report["provenance"]["dataset_b_source_scales_sha256"]) == 64
    scales.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="missing.*shared"):
        evaluate_and_reconcile_datasets(
            dataset_a_root=ds_a, dataset_b_root=ds_b, workers=1,
            all_behaviors=True, dataset_b_source_scales=scales,
        )


@pytest.mark.parametrize("behavior,ids", [
    ("Grooming", [7, 8]), ("Isolation", [7, 8]), ("Together", [7]),
])
def test_invalid_target_cardinality_fails_at_existing_boundary(tmp_path, behavior, ids):
    ds_a, ds_b = tmp_path / "a", tmp_path / "b"
    ds_a.mkdir()
    frames = [{"frame": f, "detections": [
        _mouse_detection(7, (40.0, 40.0)), _mouse_detection(8, (80.0, 80.0)),
    ]} for f in range(30)]
    _write_clip(ds_b, behavior, "bad_ids", mouse_ids=ids, tracks=frames)
    with pytest.raises(ValueError, match="requires"):
        evaluate_and_reconcile_datasets(
            dataset_a_root=ds_a, dataset_b_root=ds_b, workers=1, all_behaviors=True,
        )


def test_moving_alias_normalizes_to_locomotion_preserving_15_canonical_classes(tmp_path: Path) -> None:
    assert len(LABEL_TO_CANONICAL) == 15
    assert len(CANONICAL_BEHAVIORS) == 15
    assert "Moving" not in LABEL_TO_CANONICAL
    assert "moving" not in CANONICAL_BEHAVIORS
    assert LABEL_TO_CANONICAL["Locomotion"] == "locomotion"
    assert DISPLAY_BEHAVIOR_ALIASES == {"Moving": "Locomotion"}
    assert normalize_display_behavior("Moving") == "Locomotion"
    assert normalize_display_behavior("moving") == "Locomotion"
    assert normalize_display_behavior("MOVING") == "Locomotion"

    ds_c = tmp_path / "dataset_c"
    fps = 30.0
    moving_frames = [
        {
            "frame": f,
            "detections": [_mouse_detection(12, (15.0 + 0.45 * f, 30.0), (1.0, 0.0))],
        }
        for f in range(60)
    ]
    moving_dir = _write_clip(ds_c, "Moving", "clip_moving_01", mouse_ids=[12], tracks=moving_frames, fps=fps)

    # Second clip with calibrated metadata.distance_scale
    static_frames = [
        {"frame": f, "detections": [_mouse_detection(5, (40.0, 40.0), (1.0, 0.0))]}
        for f in range(45)
    ]
    static_dir = _write_clip(ds_c, "Static", "clip_static_scaled", mouse_ids=[5], tracks=static_frames, fps=fps)
    meta_path = static_dir / "metadata.json"
    meta_obj = json.loads(meta_path.read_text(encoding="utf-8"))
    meta_obj["distance_scale"] = {"cm_per_pixel": 0.09900655529473237, "unit": "cm"}
    meta_path.write_text(json.dumps(meta_obj), encoding="utf-8")

    entries = index_dataset(ds_c, "C")
    by_id = {entry.clip_id: entry for entry in entries}
    moving_entry = by_id["clip_moving_01"]
    assert moving_entry.behavior == "Locomotion"
    assert moving_entry.canonical_behavior == "locomotion"
    assert moving_entry.annotation_behavior == "Moving"
    assert moving_entry.metadata_cm_per_pixel is None

    scaled_entry = by_id["clip_static_scaled"]
    assert scaled_entry.metadata_cm_per_pixel == pytest.approx(0.09900655529473237)
    assert scaled_entry.metadata_scale_source == "metadata_distance_scale"

    # Benchmark evaluators normalize Moving (case-insensitive) to Locomotion
    bench_direct = evaluate_static_loco_clip(moving_dir, expected_behavior="moving", render_videos=False)
    assert bench_direct["status"] == "ok"
    assert bench_direct["ground_truth_behavior"] == "Locomotion"
    assert bench_direct["predicted_behavior"] == "Locomotion"
    assert bench_direct["is_correct"] is True
    assert bench_direct["temporal_iou"] > 0.5

    bench_six = evaluate_six_behavior_clip(moving_dir, render_videos=False)
    assert bench_six["status"] == "ok"
    assert bench_six["ground_truth_behavior"] == "Locomotion"
    assert bench_six["predicted_behavior"] == "Locomotion"
    assert bench_six["is_correct"] is True

    # Full single-dataset evaluation under all_behaviors=True
    report = evaluate_single_dataset(ds_c, "C", workers=1, all_behaviors=True)
    assert report["inputs"]["all_behaviors"] is True
    assert report["raw_behavior_counts"] == {"Moving": 1, "Static": 1}
    assert report["normalized_behavior_counts"] == {"Locomotion": 1, "Static": 1}

    clips_by_id = {c["clip_id"]: c for c in report["clips"]}
    c_mov = clips_by_id["clip_moving_01"]
    assert c_mov["behavior"] == "Locomotion"
    assert c_mov["canonical_behavior"] == "locomotion"
    assert c_mov["annotation_behavior"] == "Moving"
    assert c_mov["primary_correct"] is True
    assert c_mov["pf_target_correct"] is True
    assert c_mov["pf_joint_correct"] is True
    assert c_mov["coverage_a2_correct"] is True
    assert c_mov["scale_source"] == "default_scale_config_body_length"

    c_sta = clips_by_id["clip_static_scaled"]
    assert c_sta["cm_per_pixel"] == pytest.approx(0.09900655529473237)
    assert c_sta["scale_source"] == "metadata_distance_scale"

    loco_row = report["cohorts"]["dataset_c"]["per_behavior"]["Locomotion"]
    assert loco_row["support"] == 1
    assert loco_row["primary_correct"] == 1
    assert loco_row["pf_target_correct"] == 1
    assert loco_row["pf_joint_correct"] == 1


def test_evaluate_cleaned_dataset_and_merge_dedup_priority_and_metrics(tmp_path: Path) -> None:
    import sys

    scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import evaluate_cleaned_dataset_and_merge as merge_cli

    ds_a = tmp_path / "a"
    ds_b = tmp_path / "b"
    ds_c = tmp_path / "c"
    fps = 30.0

    # A has 2 clips: clip_ab (also in B), clip_ac (also in C, A-only in Union(A,B)), plus clip_a_only
    together_frames = [
        {
            "frame": f,
            "detections": [
                _mouse_detection(1, (30.0, 30.0), (1.0, 0.0)),
                _mouse_detection(2, (30.0, 37.2), (1.0, 0.0)),
            ],
        }
        for f in range(45)
    ]
    _write_clip(ds_a, "Together", "clip_ab", mouse_ids=[1, 2], tracks=together_frames, fps=fps)
    _write_clip(ds_b, "Together", "clip_ab", mouse_ids=[1, 2], tracks=together_frames, fps=fps)
    _write_clip(ds_a, "Together", "clip_ac", mouse_ids=[1, 2], tracks=together_frames, fps=fps)
    _write_clip(ds_c, "Together", "clip_ac", mouse_ids=[1, 2], tracks=together_frames, fps=fps)
    _write_clip(ds_a, "Together", "clip_a_only", mouse_ids=[1, 2], tracks=together_frames, fps=fps)

    # C also has Moving (60 frames) and Static (45 frames)
    moving_frames = [
        {"frame": f, "detections": [_mouse_detection(12, (15.0 + 0.45 * f, 30.0), (1.0, 0.0))]}
        for f in range(60)
    ]
    static_frames = [
        {"frame": f, "detections": [_mouse_detection(7, (40.0, 40.0), (1.0, 0.0))]}
        for f in range(45)
    ]
    _write_clip(ds_c, "Moving", "clip_c_moving", mouse_ids=[12], tracks=moving_frames, fps=fps)
    _write_clip(ds_b, "Static", "clip_b_static", mouse_ids=[7], tracks=static_frames, fps=fps)

    ab_report = evaluate_and_reconcile_datasets(
        dataset_a_root=ds_a, dataset_b_root=ds_b, workers=1, all_behaviors=True,
    )
    ab_quote2_path = tmp_path / "ab_quote2.json"
    ab_current_path = tmp_path / "ab_current.json"
    ab_quote2_path.write_text(json.dumps(ab_report), encoding="utf-8")
    ab_current_path.write_text(json.dumps(ab_report), encoding="utf-8")

    out_dir = tmp_path / "eval_out"
    rc = merge_cli.main([
        "--dataset-c", str(ds_c),
        "--ab-quote2-report", str(ab_quote2_path),
        "--ab-current-report", str(ab_current_path),
        "--output-dir", str(out_dir),
        "--workers", "1",
    ])
    assert rc == 0

    c_report_file = out_dir / "dataset_c_report.json"
    summary_json_file = out_dir / "comparison_and_merged_summary.json"
    summary_md_file = out_dir / "comparison_and_merged_summary.md"
    assert c_report_file.is_file()
    assert summary_json_file.is_file()
    assert summary_md_file.is_file()

    summary = json.loads(summary_json_file.read_text(encoding="utf-8"))
    assert summary["ab_quote2_vs_current_diff"]["exact_match"] is True
    recon = summary["baselines"]["quote2"]["reconciliation_abc"]
    assert recon["dataset_a_total"] == 3
    assert recon["dataset_b_total"] == 2
    assert recon["union_ab_total"] == 4  # B(2) + A-only(2: clip_ac, clip_a_only)
    assert recon["dataset_c_total"] == 2  # clip_ac + clip_c_moving
    assert recon["overlap_ac_count"] == 1
    assert recon["overlap_bc_count"] == 0
    assert recon["merged_union_abc_count"] == 5  # 4 + 2 - 1 = 5
    assert recon["merged_raw_union_plus_c_count"] == 6  # 4 + 2 = 6
    assert recon["merged_union_abc_source_breakdown"] == {"A": 1, "B": 2, "C": 2}

    c_loco = summary["baselines"]["quote2"]["cohorts"]["dataset_c"]["per_behavior"]["Locomotion"]
    assert c_loco["support"] == 1
    assert c_loco["primary_correct"] == 1
    assert c_loco["pf_target_correct"] == 1
    assert c_loco["pf_joint_correct"] == 1

    md_text = summary_md_file.read_text(encoding="utf-8")
    assert "pf_target" in md_text and "pf_joint" in md_text and "primary" in md_text
    assert "B > C > A" in md_text

    # Verify --skip-eval-c reuses dataset_c_report.json cleanly
    rc_skip = merge_cli.main([
        "--dataset-c", str(ds_c),
        "--ab-quote2-report", str(ab_quote2_path),
        "--output-dir", str(out_dir),
        "--skip-eval-c",
    ])
    assert rc_skip == 0
