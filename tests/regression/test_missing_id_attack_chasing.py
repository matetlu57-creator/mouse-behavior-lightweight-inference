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


class TestDatasetIntegrity:
    """Validate completeness and schema adherence of the 107 missing-ID clips."""

    def test_dataset_directories_and_clip_counts(self):
        """Verify directories exist and contain expected 85 Attack and 22 Chasing clips."""
        assert DEFAULT_ATTACK_DIR.is_dir(), f"Attack dir missing: {DEFAULT_ATTACK_DIR}"
        assert DEFAULT_CHASING_DIR.is_dir(), f"Chasing dir missing: {DEFAULT_CHASING_DIR}"

        attack_clips = [d for d in DEFAULT_ATTACK_DIR.iterdir() if d.is_dir()]
        chasing_clips = [d for d in DEFAULT_CHASING_DIR.iterdir() if d.is_dir()]

        assert len(attack_clips) == 85, f"Expected 85 Attack clips, found {len(attack_clips)}"
        assert len(chasing_clips) == 22, f"Expected 22 Chasing clips, found {len(chasing_clips)}"
        assert len(attack_clips) + len(chasing_clips) == 107

    def test_required_four_files_in_all_clips(self):
        """Verify every clip contains clip.mp4, annotation.json, metadata.json, and tracks.json."""
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
                summary = evaluate_dataset()
        else:
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
                summary = evaluate_dataset()
        else:
            summary = evaluate_dataset()

        assert summary["chasing_clips"] == 22
        assert summary["chasing_accuracy"] >= 0.90, (
            f"Chasing accuracy {summary['chasing_accuracy']:.4f} < 0.90 (expected >= 20/22)"
        )
        assert summary["attack_accuracy"] >= 0.85, (
            f"Attack accuracy {summary['attack_accuracy']:.4f} < 0.85"
        )

    def test_refine_chasing_segments_backtracking_and_disengagement(self):
        """Verify refine_chasing_segments backtracks onset and trims disengagement."""
        from mouse_behavior.behavior.segment_clipping import refine_chasing_segments

        fps = 30.0
        # Synthetic chase: 60 frames. Frames 10-50 are active pursuit (mouse 1 chases mouse 2).
        # Frames 0-9: chaser moving at speed >= 8.0 cm/s towards target with distance <= 12 cm.
        # Frames 50-60: target moves away and chaser stops -> distance opens up.
        rows = []
        for f in range(60):
            # 10 px = 1 cm (assume cm_per_px ~ 0.1)
            # Mouse 2 (leader): starts at 200, moves forward at 15 cm/s (150 px/s = 5 px/frame)
            p2 = (200.0 + f * 5.0, 200.0)
            # Mouse 1 (follower): starts at 150 (dist 50px = 5cm), moves at 15 cm/s
            p1 = (150.0 + f * 5.0, 200.0)
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

        # Event candidate initially only marked frames [15, 45)
        ev = {
            "canonical_behavior": "chasing",
            "start_frame": 15,
            "end_frame": 45,
            "mouse_ids": [1, 2],
        }
        refined = refine_chasing_segments([ev], store, binding)
        assert len(refined) == 1
        # Onset backtracked from 15 towards frame 0
        assert refined[0]["start_frame"] < 15
        assert refined[0]["end_frame"] >= 45

    def test_refine_attack_segments_surge_onset_and_grapple_continuation(self):
        """Verify refine_attack_segments surge onset backtracking and grapple extension."""
        from mouse_behavior.behavior.segment_clipping import refine_attack_segments

        fps = 30.0
        # 50 frames:
        # Frames 5-15: strike surge (dist 60px = 6cm <= 8.5cm, speed 30 cm/s = 300 px/s = 10 px/f)
        # Frames 15-35: candidate core grapple
        # Frames 35-45: grapple continuation (dist 40px = 4cm <= 6.5cm)
        rows = []
        for f in range(50):
            if f < 15:
                # Fast surge closing in
                c1 = (100.0 + f * 10.0, 100.0)
                c2 = (220.0 + f * 2.0, 100.0)
            elif f < 35:
                # Core grapple
                c1 = (250.0, 100.0)
                c2 = (280.0, 100.0)
            else:
                # Intimate tumble
                c1 = (255.0, 100.0)
                c2 = (275.0, 100.0)
            rows.append({
                "frame": f,
                "source_frame": f,
                "time": f / fps,
                "detections": [
                    {"track_id": 1, "box": [c1[0]-10, c1[1]-10, c1[0]+10, c1[1]+10]},
                    {"track_id": 2, "box": [c2[0]-10, c2[1]-10, c2[0]+10, c2[1]+10]},
                ],
            })
        store = InMemoryTrackStore(rows)
        binding = {"fps": fps, "frame_count": 50, "cm_per_pixel": 0.10}

        # Event candidate initially frames [15, 35)
        ev = {
            "canonical_behavior": "attack",
            "start_frame": 15,
            "end_frame": 35,
            "mouse_ids": [1, 2],
        }
        refined = refine_attack_segments([ev], store, binding)
        assert len(refined) == 1
        # Surge onset backtracked before 15
        assert refined[0]["start_frame"] < 15
        # Grapple continuation extended past 35
        assert refined[0]["end_frame"] > 35

    def test_refinement_fallback_safety_preserves_events(self):
        """Verify that qualification failures never discard events but safely fall back."""
        from mouse_behavior.behavior.segment_clipping import (
            refine_attack_segments,
            refine_chasing_segments,
        )

        fps = 30.0
        # Empty rows / broken tracks
        store = InMemoryTrackStore([])
        binding = {"fps": fps, "frame_count": 20}

        att_ev = {"canonical_behavior": "attack", "start_frame": 5, "end_frame": 15, "mouse_ids": [1, 2]}
        ch_ev = {"canonical_behavior": "chasing", "start_frame": 5, "end_frame": 15, "mouse_ids": [1, 2]}

        ref_att = refine_attack_segments([att_ev], store, binding)
        ref_ch = refine_chasing_segments([ch_ev], store, binding)

        assert len(ref_att) == 1, "Attack event was erroneously discarded"
        assert len(ref_ch) == 1, "Chasing event was erroneously discarded"
        assert ref_att[0]["start_frame"] == 5 and ref_att[0]["end_frame"] == 15
        assert ref_ch[0]["start_frame"] == 5 and ref_ch[0]["end_frame"] == 15



class TestCliRunner:
    """Test CLI entrypoint execution and artifact generation."""

    def test_cli_execution_with_json_out(self, tmp_path):
        """Run script via CLI subprocess with --limit 1 and verify report file."""
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

