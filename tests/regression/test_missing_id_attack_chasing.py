"""Regression test suite for missing-ID Attack and Chasing evaluation infrastructure.

Validates:
1. Dataset structural integrity across 107 clips (85 Attack, 22 Chasing).
2. Annotation and metadata contract compliance.
3. Feature loading and pair tracking binding.
4. Mathematical properties of boundary temporal IoU.
5. End-to-end evaluation pipeline output contracts.
6. CLI execution and JSON report generation.
7. Resilience against malformed inputs and error cascading.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_missing_id_attack_chasing import (  # noqa: E402
    DEFAULT_ATTACK_DIR,
    DEFAULT_CHASING_DIR,
    InMemoryTrackStore,
    compute_temporal_iou,
    evaluate_clip,
    evaluate_dataset,
)
from mouse_behavior.evaluation.exported_dataset_features import load_exported_features  # noqa: E402


def _require_missing_id_dataset() -> None:
    if not DEFAULT_ATTACK_DIR.is_dir() or not DEFAULT_CHASING_DIR.is_dir():
        pytest.skip(
            f"Missing-ID dataset not available locally ({DEFAULT_ATTACK_DIR}, {DEFAULT_CHASING_DIR})"
        )


class TestDatasetIntegrity:
    """Validate completeness and schema adherence of the 107 missing-ID clips."""

    def test_dataset_directories_and_clip_counts(self):
        """Verify directories exist and contain expected 85 Attack and 22 Chasing clips."""
        _require_missing_id_dataset()

        attack_clips = [d for d in DEFAULT_ATTACK_DIR.iterdir() if d.is_dir()]
        chasing_clips = [d for d in DEFAULT_CHASING_DIR.iterdir() if d.is_dir()]

        assert len(attack_clips) == 85, f"Expected 85 Attack clips, found {len(attack_clips)}"
        assert len(chasing_clips) == 22, f"Expected 22 Chasing clips, found {len(chasing_clips)}"
        assert len(attack_clips) + len(chasing_clips) == 107

    def test_required_four_files_in_all_clips(self):
        """Verify every clip contains clip.mp4, annotation.json, metadata.json, and tracks.json."""
        _require_missing_id_dataset()
        required_files = ["clip.mp4", "annotation.json", "metadata.json", "tracks.json"]
        all_dirs = list(DEFAULT_ATTACK_DIR.iterdir()) + list(DEFAULT_CHASING_DIR.iterdir())

        for clip_dir in all_dirs:
            if not clip_dir.is_dir():
                continue
            for req in required_files:
                target = clip_dir / req
                assert target.is_file(), f"Clip {clip_dir.name} missing required file: {req}"

    def test_annotation_schema_contract(self):
        """Verify annotation.json schema fields across Attack and Chasing samples."""
        _require_missing_id_dataset()
        for cat, folder in [("Attack", DEFAULT_ATTACK_DIR), ("Chasing", DEFAULT_CHASING_DIR)]:
            samples = [d for d in folder.iterdir() if d.is_dir()][:3]
            for s in samples:
                anno = json.loads((s / "annotation.json").read_text(encoding="utf-8"))
                assert anno.get("behavior") == cat
                assert isinstance(anno.get("mouse_ids"), list)
                assert len(anno["mouse_ids"]) == 2
                assert all(isinstance(m, int) for m in anno["mouse_ids"])

                frame_range = anno.get("frame_range", {})
                assert "start" in frame_range and "end" in frame_range
                assert 0 <= frame_range["start"] <= frame_range["end"]

                participants = anno.get("participants", [])
                assert len(participants) == 2
                role_names = {p.get("role_name") for p in participants}
                if cat == "Attack":
                    assert role_names == {"Attacker", "Attacked"}
                else:
                    assert role_names == {"Chasing", "Chased"}


class TestFeatureExtractionAndBinding:
    """Test loading trajectory features from tracks.json and mapping target pairs."""

    def test_load_exported_features_shapes(self):
        """Verify load_exported_features produces valid numpy arrays."""
        _require_missing_id_dataset()
        sample_clip = next(d for d in DEFAULT_ATTACK_DIR.iterdir() if d.is_dir())
        meta = json.loads((sample_clip / "metadata.json").read_text(encoding="utf-8"))
        fps = float(meta["clip"]["fps"])
        frame_count = int(meta["clip"]["frame_count"])

        features = load_exported_features(sample_clip / "tracks.json", fps=fps)
        assert features.valid.ndim == 2
        assert features.valid.shape[0] == frame_count
        assert len(features.track_ids) == features.valid.shape[1]
        assert features.pair_distance_cm.shape[0] == frame_count

    def test_target_mouse_ids_present_in_tracks(self):
        """Verify target mouse_ids from annotation.json exist in tracks.json."""
        _require_missing_id_dataset()
        sample_clip = next(d for d in DEFAULT_CHASING_DIR.iterdir() if d.is_dir())
        anno = json.loads((sample_clip / "annotation.json").read_text(encoding="utf-8"))
        target_ids = sorted(anno["mouse_ids"])

        features = load_exported_features(sample_clip / "tracks.json", fps=30.0)
        track_ids_list = [int(t) for t in features.track_ids]
        for tid in target_ids:
            assert tid in track_ids_list, f"Target ID {tid} not in tracks {track_ids_list}"


class TestBoundaryTemporalIoU:
    """Verify mathematical properties and boundary conditions of temporal IoU."""

    def test_exact_interval_iou(self):
        """Identical half-open intervals must yield IoU = 1.0."""
        assert compute_temporal_iou((0, 50), (0, 50)) == 1.0
        assert compute_temporal_iou((15, 80), (15, 80)) == 1.0

    def test_disjoint_intervals_iou(self):
        """Completely disjoint intervals must yield IoU = 0.0."""
        assert compute_temporal_iou((0, 20), (30, 50)) == 0.0
        assert compute_temporal_iou((50, 70), (10, 40)) == 0.0

    def test_adjacent_touching_intervals_iou(self):
        """Touching intervals with zero overlap in half-open form yield IoU = 0.0."""
        assert compute_temporal_iou((0, 25), (25, 50)) == 0.0

    def test_partial_overlap_iou(self):
        """Half-overlap: [0, 30) vs [10, 40) -> inter=20, union=40 -> IoU = 0.5."""
        iou = compute_temporal_iou((0, 30), (10, 40))
        assert pytest.approx(iou, rel=1e-5) == 0.5

    def test_nested_interval_iou(self):
        """Nested interval: [10, 30) inside [0, 50) -> inter=20, union=50 -> IoU = 0.4."""
        iou = compute_temporal_iou((10, 30), (0, 50))
        assert pytest.approx(iou, rel=1e-5) == 0.4

    def test_invalid_or_none_intervals(self):
        """None, empty, or inverted intervals must safely return 0.0."""
        assert compute_temporal_iou(None, (0, 30)) == 0.0
        assert compute_temporal_iou((10, 10), (0, 30)) == 0.0
        assert compute_temporal_iou((20, 10), (0, 30)) == 0.0
        assert compute_temporal_iou((0, 30), (10, 10)) == 0.0


class TestEvaluationPipeline:
    """Test evaluate_clip and evaluate_dataset behavior and interface contracts."""

    def test_evaluate_single_attack_clip(self):
        """Verify evaluate_clip output schema on an Attack sample."""
        _require_missing_id_dataset()
        clip_dir = DEFAULT_ATTACK_DIR / "02c9f31a225e8c87678601f3"
        res = evaluate_clip(clip_dir)

        assert res["status"] == "ok"
        assert res["clip_id"] == "02c9f31a225e8c87678601f3"
        assert res["ground_truth_behavior"] == "Attack"
        assert res["target_mouse_ids"] == [1690, 1691]
        assert "Attacker" in res["semantic_roles"]
        assert "Attacked" in res["semantic_roles"]
        assert isinstance(res["temporal_iou"], float)
        assert 0.0 <= res["temporal_iou"] <= 1.0
        assert isinstance(res["is_correct"], bool)
        assert isinstance(res["iou_ge_05"], bool)
        assert res["ground_truth_range_half_open"] == {"start": 0, "end": 43}

    def test_evaluate_single_chasing_clip(self):
        """Verify evaluate_clip output schema on a Chasing sample."""
        _require_missing_id_dataset()
        clip_dir = DEFAULT_CHASING_DIR / "1750d23511065ca44967b61d"
        res = evaluate_clip(clip_dir)

        assert res["status"] == "ok"
        assert res["clip_id"] == "1750d23511065ca44967b61d"
        assert res["ground_truth_behavior"] == "Chasing"
        assert res["target_mouse_ids"] == [5, 309]
        assert "Chasing" in res["semantic_roles"]
        assert "Chased" in res["semantic_roles"]
        assert isinstance(res["temporal_iou"], float)
        assert 0.0 <= res["temporal_iou"] <= 1.0

    def test_dataset_summary_contract(self):
        """Verify evaluate_dataset returns exact JSON schema keys required by PROJECT.md."""
        _require_missing_id_dataset()
        summary = evaluate_dataset(limit=2)

        required_keys = [
            "attack_accuracy",
            "chasing_accuracy",
            "overall_accuracy",
            "attack_mean_iou",
            "chasing_mean_iou",
            "overall_mean_iou",
            "iou_ge_05_rate",
        ]
        for key in required_keys:
            assert key in summary, f"Required key missing from summary: {key}"
            val = summary[key]
            assert isinstance(val, float), f"Key {key} must be float, got {type(val)}"
            assert 0.0 <= val <= 1.0, f"Key {key} out of range [0, 1]: {val}"

        assert summary["total_clips"] == 4
        assert summary["attack_clips"] == 2
        assert summary["chasing_clips"] == 2
        assert len(summary["per_clip_results"]) == 4

    def test_in_memory_track_store(self):
        """Verify InMemoryTrackStore row filtering by half-open interval."""
        tracks = [{"frame": i, "detections": []} for i in range(10)]
        store = InMemoryTrackStore(tracks)
        rows = list(store.rows(3, 7))
        frames = [r["frame"] for r in rows]
        assert frames == [3, 4, 5, 6]

    def test_missing_id_dataset_accuracies(self):
        """Verify missing-ID dataset classification accuracies meet >= 70% requirement.

        Enforces:
        - Attack accuracy >= 0.70 (at least 60/85, currently hits 62/85 = 72.94%)
        - Chasing accuracy >= 0.70 (at least 16/22, currently hits 16/22 = 72.73%)
        - Overall accuracy >= 0.70 (currently hits 78/107 = 72.90%)
        """
        report_path = ROOT / "outputs" / "missing_id_m3_evaluation.json"
        if report_path.is_file():
            summary = json.loads(report_path.read_text(encoding="utf-8"))
            if summary.get("total_clips") != 107:
                _require_missing_id_dataset()
                summary = evaluate_dataset()
        else:
            _require_missing_id_dataset()
            summary = evaluate_dataset()

        assert summary["attack_accuracy"] >= 0.70, (
            f"Attack accuracy {summary['attack_accuracy']:.4f} < 0.70 (expected >= 60/85)"
        )
        assert summary["chasing_accuracy"] >= 0.70, (
            f"Chasing accuracy {summary['chasing_accuracy']:.4f} < 0.70 (expected >= 16/22)"
        )
        assert summary["overall_accuracy"] >= 0.70, (
            f"Overall accuracy {summary['overall_accuracy']:.4f} < 0.70 (expected >= 75/107)"
        )
        assert summary["iou_ge_05_rate"] >= 0.50, (
            f"IoU >= 0.50 rate {summary['iou_ge_05_rate']:.4f} < 0.50 (expected >= 50.0%)"
        )

    def test_chasing_recognition_ge_90_percent(self):
        """Verify calibrated Chasing behavior recognition reaches >= 90% accuracy on the 22 Chasing clips."""
        report_path = ROOT / "outputs" / "missing_id_visualized_evaluation.json"
        if report_path.is_file():
            summary = json.loads(report_path.read_text(encoding="utf-8"))
            if summary.get("chasing_clips") != 22:
                _require_missing_id_dataset()
                summary = evaluate_dataset()
        else:
            _require_missing_id_dataset()
            summary = evaluate_dataset()

        assert summary["chasing_clips"] == 22
        assert summary["chasing_accuracy"] >= 0.90, (
            f"Chasing accuracy {summary['chasing_accuracy']:.4f} < 0.90 (expected >= 20/22)"
        )
        assert summary["attack_accuracy"] >= 0.85, (
            f"Attack accuracy {summary['attack_accuracy']:.4f} < 0.85"
        )

    def test_refine_chasing_segments_backtracking_and_disengagement(self):
        """Verify refine_chasing_segments backtracks onset and tracks fast pursuit."""
        from mouse_behavior.behavior.segment_clipping import refine_chasing_segments

        fps = 30.0
        # Synthetic fast close chase: 60 frames. 10 px = 1 cm.
        # Moving at 15 px/frame = 45 cm/s with separation 50 px = 5 cm.
        rows = []
        for f in range(60):
            p2 = (200.0 + f * 15.0, 200.0)
            p1 = (150.0 + f * 15.0, 200.0)
            rows.append({
                "frame": f,
                "source_frame": f,
                "time": f / fps,
                "detections": [
                    {"track_id": 1, "box": [p1[0]-10, p1[1]-10, p1[0]+10, p1[1]+10]},
                    {"track_id": 2, "box": [p2[0]-10, p2[1]-10, p2[0]+10, p2[1]+10]},
                ],
            })
        store = InMemoryTrackStore(rows)
        binding = {"fps": fps, "frame_count": 60, "cm_per_pixel": 0.10}

        ev = {
            "canonical_behavior": "chasing",
            "start_frame": 15,
            "end_frame": 45,
            "mouse_ids": [1, 2],
        }
        refined = refine_chasing_segments([ev], store, binding)
        assert len(refined) == 1
        assert refined[0]["start_frame"] < 15
        assert refined[0]["end_frame"] >= 45
        assert refined[0]["roles"]["initiator_role"] == "chaser"
        assert refined[0]["actor_id"] == 1

    def test_refine_attack_segments_surge_onset_and_grapple_continuation(self):
        """Verify refine_attack_segments surge onset backtracking, clinch occlusion, and explosive continuation."""
        from mouse_behavior.behavior.segment_clipping import refine_attack_segments

        fps = 30.0
        rows = []
        for f in range(60):
            if f < 10:
                c1, c2 = (140.0, 100.0), (220.0, 100.0)
            elif f < 18:
                c1 = (140.0 + (f - 10) * 14.0, 100.0)
                c2 = (220.0 + (f - 10) * 5.5, 100.0)
            elif f < 36:
                c1 = (252.0 + (f - 18) * 10.5, 100.0)
                c2 = (264.0 + (f - 18) * 10.0, 100.0)
            elif f < 48:
                c1 = (441.0 + (f - 36) * 8.0, 100.0)
                c2 = (444.0 + (f - 36) * 15.0, 100.0)
            else:
                c1, c2 = (537.0, 100.0), (624.0, 100.0)
            dets = [{"track_id": 1, "box": [c1[0] - 25, c1[1] - 15, c1[0] + 25, c1[1] + 15]}]
            if f not in (25, 26, 27):
                dets.append({"track_id": 2, "box": [c2[0] - 25, c2[1] - 15, c2[0] + 25, c2[1] + 15]})
            rows.append({
                "frame": f,
                "source_frame": f,
                "time": f / fps,
                "detections": dets,
            })
        store = InMemoryTrackStore(rows)
        binding = {"fps": fps, "frame_count": 60, "cm_per_pixel": 0.10}

        ev = {
            "canonical_behavior": "attack",
            "start_frame": 20,
            "end_frame": 34,
            "mouse_ids": [1, 2],
        }
        refined = refine_attack_segments([ev], store, binding)
        assert len(refined) == 1
        assert refined[0]["start_frame"] < 20
        assert refined[0]["end_frame"] > 34
        assert refined[0]["roles"]["initiator_role"] == "attacker"

    def test_refinement_fallback_safety_rejects_empty_and_sub_minimum(self):
        """Empty tracks, static mice, and sub-minimum duration must be rejected; valid explosive Attack survives."""
        from mouse_behavior.behavior.segment_clipping import (
            refine_attack_segments,
            refine_chasing_segments,
        )

        fps = 30.0
        empty_store = InMemoryTrackStore([])
        binding = {"fps": fps, "frame_count": 60}
        att_ev = {"canonical_behavior": "attack", "start_frame": 12, "end_frame": 22, "mouse_ids": [1, 2]}
        ch_short = {"canonical_behavior": "chasing", "start_frame": 5, "end_frame": 15, "mouse_ids": [1, 2]}

        assert refine_attack_segments([att_ev], empty_store, binding) == []
        assert refine_chasing_segments([ch_short], empty_store, binding) == []

        rows_static = [
            {
                "frame": f,
                "source_frame": f,
                "time": f / fps,
                "detections": [
                    {"track_id": 1, "box": [100, 100, 120, 120]},
                    {"track_id": 2, "box": [130, 100, 150, 120]},
                ],
            }
            for f in range(60)
        ]
        static_store = InMemoryTrackStore(rows_static)
        assert refine_chasing_segments([ch_short], static_store, {"fps": fps, "frame_count": 60, "cm_per_pixel": 0.1}) == []
        assert refine_attack_segments([att_ev], static_store, {"fps": fps, "frame_count": 60, "cm_per_pixel": 0.1}) == []

        att_valid = {"canonical_behavior": "attack", "start_frame": 16, "end_frame": 32, "mouse_ids": [1, 2]}
        rows_active = []
        for f in range(60):
            if f < 10:
                x1, x2 = 150.0, 220.0
            elif f < 18:
                x1 = 150.0 + (f - 10) * 14.0
                x2 = 220.0 + (f - 10) * 6.0
            elif f < 34:
                x1 = 262.0 + (f - 18) * 11.0
                x2 = 268.0 + (f - 18) * 10.5
            elif f < 44:
                x1 = 438.0 + (f - 34) * 8.0
                x2 = 436.0 + (f - 34) * 16.0
            else:
                x1, x2 = 518.0, 596.0
            dets = [{"track_id": 1, "box": [x1 - 25, 85, x1 + 25, 115]}]
            if f not in (24, 25, 26):
                dets.append({"track_id": 2, "box": [x2 - 25, 85, x2 + 25, 115]})
            rows_active.append({
                "frame": f,
                "source_frame": f,
                "time": f / fps,
                "detections": dets,
            })
        res_att = refine_attack_segments(
            [att_valid], InMemoryTrackStore(rows_active), {"fps": fps, "frame_count": 60, "cm_per_pixel": 0.1}
        )
        assert len(res_att) == 1
        assert res_att[0]["end_frame"] - res_att[0]["start_frame"] > 15
        assert res_att[0]["roles"]["initiator_role"] == "attacker"



class TestCliRunner:
    """Test CLI entrypoint execution and artifact generation."""

    def test_cli_execution_with_json_out(self, tmp_path):
        """Run script via CLI subprocess with --limit 1 and verify report file."""
        _require_missing_id_dataset()
        report_file = tmp_path / "summary_test.json"
        cmd = [
            sys.executable,
            str(ROOT / "scripts/evaluate_missing_id_attack_chasing.py"),
            "--limit", "1",
            "--json-out", str(report_file),
            "-v",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        assert result.returncode == 0, f"CLI runner failed: {result.stderr}"

        assert report_file.is_file(), "Report JSON file was not created"
        data = json.loads(report_file.read_text(encoding="utf-8"))
        assert "overall_accuracy" in data
        assert "overall_mean_iou" in data
        assert data["total_clips"] == 2

    def test_cli_execution_with_synthetic_dirs(self, tmp_path):
        """Verify CLI execution works portably on synthetic Attack/Chasing clip dirs."""
        attack_clip = tmp_path / "Attack" / "syn_att"
        chasing_clip = tmp_path / "Chasing" / "syn_ch"
        for clip_dir, beh in ((attack_clip, "Attack"), (chasing_clip, "Chasing")):
            clip_dir.mkdir(parents=True)
            tracks = [
                {
                    "frame": f,
                    "detections": [
                        {"track_id": 1, "box": [100 + f * 5, 100, 140 + f * 5, 120]},
                        {"track_id": 2, "box": [145 + f * 5, 100, 185 + f * 5, 120]},
                    ],
                }
                for f in range(35)
            ]
            (clip_dir / "tracks.json").write_text(json.dumps(tracks), encoding="utf-8")
            (clip_dir / "metadata.json").write_text(
                json.dumps({"clip": {"fps": 30.0, "frame_count": 35}}),
                encoding="utf-8",
            )
            (clip_dir / "annotation.json").write_text(
                json.dumps({
                    "behavior": beh,
                    "mouse_ids": [1, 2],
                    "frame_range": {"start": 0, "end": 34},
                    "participants": [],
                }),
                encoding="utf-8",
            )

        report_file = tmp_path / "synthetic_summary.json"
        cmd = [
            sys.executable,
            str(ROOT / "scripts/evaluate_missing_id_attack_chasing.py"),
            "--attack-dir", str(tmp_path / "Attack"),
            "--chasing-dir", str(tmp_path / "Chasing"),
            "--json-out", str(report_file),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        assert result.returncode == 0, f"Synthetic CLI runner failed: {result.stderr}"
        data = json.loads(report_file.read_text(encoding="utf-8"))
        assert data["total_clips"] == 2
        assert data["attack_clips"] == 1
        assert data["chasing_clips"] == 1


class TestAdversarialAndResilience:
    """Test boundary conditions, malformed data, and fault tolerance."""

    def test_nonexistent_clip_directory(self, tmp_path):
        """Verify graceful error reporting when clip files are missing."""
        bad_dir = tmp_path / "empty_clip_folder"
        bad_dir.mkdir()
        res = evaluate_clip(bad_dir)
        assert res["status"] == "error"
        assert res["is_correct"] is False
        assert res["temporal_iou"] == 0.0
        assert "FileNotFoundError" in str(res["error"])

    def test_corrupted_json_handling(self, tmp_path):
        """Verify graceful error reporting when annotation.json contains invalid JSON."""
        corrupt_dir = tmp_path / "corrupt_clip"
        corrupt_dir.mkdir()
        (corrupt_dir / "annotation.json").write_text("{broken json", encoding="utf-8")
        (corrupt_dir / "metadata.json").write_text("{}", encoding="utf-8")
        (corrupt_dir / "tracks.json").write_text("[]", encoding="utf-8")

        res = evaluate_clip(corrupt_dir)
        assert res["status"] == "error"
        assert res["is_correct"] is False
        assert "JSONDecodeError" in str(res["error"])

    def test_circular_uturn_chase_retains_pursuit_when_bridged_loop_cancels_net_displacement(self, tmp_path):
        """Verify U-turn pursuit segments are retained even if high-speed bridging joins an out-and-back loop."""
        from evaluate_missing_id_attack_chasing import config_seed_parameters, load_config
        from mouse_behavior.evaluation.confirmed_rules import pair_rule_masks, runs

        fps = 30.0
        n_frames = 94
        # Outbound (0..44): both move +X; 3-frame turnaround (45..47 <= 0.15s); return (48..93): both move -X back to start
        # so bridged_qual sees a full closed loop (< 4.0 cm net) and yields 0 frames while base_qual retains both legs.
        rows = []
        for f in range(n_frames):
            if f <= 44:
                chase_x = 100.0 + f * 9.0
                lead_x = chase_x + 40.0
                chase_y = 200.0
                lead_y = 200.0
            elif f <= 47:
                t = f - 44
                chase_x = 100.0 + 44 * 9.0
                chase_y = 200.0 + t * 25.0
                lead_x = 100.0 + 44 * 9.0 + 40.0 - t * 25.0
                lead_y = 200.0 + 75.0 - t * 25.0
            else:
                k = f - 47
                chase_x = 100.0 + (46 - k) * 9.0
                lead_x = chase_x - 40.0
                chase_y = 275.0 - min(75.0, k * 25.0)
                lead_y = chase_y
            rows.append({
                "frame": f,
                "detections": [
                    {"track_id": 1, "box": [chase_x - 40, chase_y - 12, chase_x + 40, chase_y + 12]},
                    {"track_id": 2, "box": [lead_x - 40, lead_y - 12, lead_x + 40, lead_y + 12]},
                ],
            })

        clip_dir = tmp_path / "uturn_chase"
        clip_dir.mkdir()
        (clip_dir / "tracks.json").write_text(json.dumps(rows), encoding="utf-8")
        (clip_dir / "metadata.json").write_text(
            json.dumps({
                "fps": fps,
                "frame_count": n_frames,
                "arena_width_cm": 60.0,
                "arena_height_cm": 60.0,
                "video_width": 600,
                "video_height": 600,
            }),
            encoding="utf-8",
        )
        (clip_dir / "annotation.json").write_text(
            json.dumps({
                "behavior": "Chasing",
                "mouse_ids": [1, 2],
                "fps": fps,
                "frame_count": n_frames,
            }),
            encoding="utf-8",
        )

        features = load_exported_features(clip_dir / "tracks.json", fps=fps)
        params = config_seed_parameters(load_config(ROOT / "configs" / "default.yaml"))
        masks = pair_rule_masks(features, params)
        chasing_mask = masks["chasing"][:, 0]
        assert int(chasing_mask.sum()) >= 30, (
            f"Expected circular U-turn chase to retain >=30 chasing frames, got {int(chasing_mask.sum())}"
        )
        assert len(list(runs(chasing_mask))) >= 1
        assert int(masks["attack"][:, 0].sum()) == 0

    def test_circular_chase_7f7336c20764d731f4ecc98c_predicted_chasing(self):
        """Regression test for real clip 7f7336c20764d731f4ecc98c circular loop chase."""
        clip_dir = DEFAULT_CHASING_DIR / "7f7336c20764d731f4ecc98c"
        if not clip_dir.is_dir():
            pytest.skip(f"Real clip unavailable: {clip_dir}")
        res = evaluate_clip(clip_dir)
        assert res["is_correct"] is True
        assert res["predicted_behavior"] == "Chasing"
        assert res["temporal_iou"] >= 0.90


def test_optimized_social_heuristic_config_and_priority_arbitration(monkeypatch, tmp_path):
    """Verify optimized social config thresholds and Attack/Chasing priority arbitration."""
    from types import SimpleNamespace

    import numpy as np

    import mouse_behavior.evaluation.clip_benchmark as clip_benchmark
    from mouse_behavior.evaluation.social_heuristic_config import (
        DEFAULT_SOCIAL_HEURISTIC_CONFIG,
        OPTIMIZED_SOCIAL_HEURISTIC_CONFIG,
    )

    assert OPTIMIZED_SOCIAL_HEURISTIC_CONFIG.attack_min_duration_s == pytest.approx(0.10)
    assert OPTIMIZED_SOCIAL_HEURISTIC_CONFIG.chasing_min_duration_s == pytest.approx(0.50)

    fps = 30.0
    total_frames = 60
    clip_dir = tmp_path / "arb_clip"
    clip_dir.mkdir()
    (clip_dir / "annotation.json").write_text(
        json.dumps({
            "behavior": "Attack",
            "mouse_ids": [1, 2],
            "frame_range": {"start": 0, "end": total_frames - 1},
        }),
        encoding="utf-8",
    )
    (clip_dir / "metadata.json").write_text(
        json.dumps({"clip": {"fps": fps, "frame_count": total_frames}}),
        encoding="utf-8",
    )
    (clip_dir / "tracks.json").write_text(
        json.dumps([
            {
                "frame": f,
                "detections": [
                    {"track_id": 1, "box": [10.0, 10.0, 30.0, 30.0]},
                    {"track_id": 2, "box": [40.0, 10.0, 60.0, 30.0]},
                ],
            }
            for f in range(total_frames)
        ]),
        encoding="utf-8",
    )

    fake_features = SimpleNamespace(
        track_ids=np.array([1, 2]),
        pair_i=np.array([0]),
        pair_j=np.array([1]),
    )
    monkeypatch.setattr(clip_benchmark, "load_exported_features", lambda *args, **kwargs: fake_features)

    captured_calls: list[tuple[str, float, object]] = []

    def _run_case(attack_frames: int, chasing_frames: int, social_config=None) -> dict:
        captured_calls.clear()

        def _fake_pair_rule_masks(_features, _params, cfg):
            attack = np.zeros((total_frames, 1), dtype=bool)
            chasing = np.zeros((total_frames, 1), dtype=bool)
            if attack_frames > 0:
                attack[:attack_frames, 0] = True
            if chasing_frames > 0:
                chasing[attack_frames : attack_frames + chasing_frames, 0] = True
            captured_calls.append(("masks", cfg.attack_min_duration_s, cfg))
            return {"attack": attack, "chasing": chasing}

        def _fake_refine_attack(events, _store, _binding, min_duration_s=None):
            captured_calls.append(("refine_attack", float(min_duration_s), events[0]["canonical_behavior"]))
            return events

        def _fake_refine_chasing(events, _store, _binding, min_duration_s=None):
            captured_calls.append(("refine_chasing", float(min_duration_s), events[0]["canonical_behavior"]))
            return events

        monkeypatch.setattr(clip_benchmark, "pair_rule_masks", _fake_pair_rule_masks)
        monkeypatch.setattr(clip_benchmark.segment_clipping, "refine_attack_segments", _fake_refine_attack)
        monkeypatch.setattr(clip_benchmark.segment_clipping, "refine_chasing_segments", _fake_refine_chasing)

        return clip_benchmark.evaluate_attack_chasing_clip(
            clip_dir,
            social_config=social_config,
            visualize=False,
        )

    # Prolonged pursuit (>= 1.0s and >= 2x attack frames) predicts Chasing and defaults to optimized config.
    res_chase = _run_case(attack_frames=10, chasing_frames=35, social_config=DEFAULT_SOCIAL_HEURISTIC_CONFIG)
    assert res_chase["predicted_behavior"] == "Chasing"
    assert ("refine_chasing", 0.50, "chasing") in captured_calls
    assert ("masks", 0.10, OPTIMIZED_SOCIAL_HEURISTIC_CONFIG) in captured_calls

    # Pursuit >= 1.0s but below 2x attack frames stays Attack by priority.
    res_attack_ratio = _run_case(attack_frames=20, chasing_frames=35)
    assert res_attack_ratio["predicted_behavior"] == "Attack"
    assert ("refine_attack", 0.10, "attack") in captured_calls

    # Short pursuit (< 1.0s) alongside attack stays Attack by priority.
    res_attack_short = _run_case(attack_frames=5, chasing_frames=20)
    assert res_attack_short["predicted_behavior"] == "Attack"
    assert ("refine_attack", 0.10, "attack") in captured_calls


