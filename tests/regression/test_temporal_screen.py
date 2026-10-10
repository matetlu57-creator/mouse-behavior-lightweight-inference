from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from mouse_behavior.evaluation.exported_dataset import ExportedFeatures, HeuristicParameters, SampleRecord
from mouse_behavior.evaluation.screening_signals import ScreenPolicy
from mouse_behavior.evaluation.temporal_screen import contiguous_windows, group_phase_masks, screen_record


def features(n=40, mice=2, fps=10.):
    i, j = np.triu_indices(mice, 1)
    pair = np.full((n, len(i)), 6.)
    speed = np.zeros((n, mice))
    return ExportedFeatures(fps=fps, track_ids=np.arange(mice) + 100, valid=np.ones((n, mice), bool),
                            centers_cm=np.zeros((n, mice, 2)), speed_cm_s=speed,
                            velocity_cm_s=np.zeros((n, mice, 2)), pair_i=i, pair_j=j,
                            pair_distance_cm=pair, pair_closing_speed_cm_s=np.zeros_like(pair),
                            pair_direction_similarity=np.ones_like(pair),
                            pair_nose_head_cm=np.full_like(pair, np.inf),
                            pair_nose_tail_cm=np.full_like(pair, np.inf),
                            nearest_distance_cm=np.full_like(speed, 6.),
                            mean_nearest_distance_cm=np.full(n, 6.), cm_per_pixel=.1,
                            track_selection_truncated=False, selected_track_count=mice,
                            reference_body_cm=10.7, nose_acceleration_cm_s2=np.zeros_like(speed))


def record(label, f, ids=None):
    return SampleRecord("sample", label, label.lower(), "sample", "annotation", "metadata",
                        "tracks", "clip", "recording", "session", f.fps, len(f.valid),
                        tuple(f.track_ids) if ids is None else ids, "high", 0, len(f.valid)-1,
                        0., len(f.valid)/f.fps)


def test_attack_keeps_attack_contact_and_removes_quiet_contact_lead_in():
    f = features()
    f.pair_nose_head_cm[:20] = 1.
    f.speed_cm_s[10:20, 0] = 20.
    plans, _ = screen_record(record("Attack", f), f, HeuristicParameters())
    assert [(p.start_frame, p.end_frame) for p in plans] == [(10, 20)]
    contacts, _ = screen_record(record("Snout-head_contact", f), f, HeuristicParameters())
    assert [(p.start_frame, p.end_frame) for p in contacts] == [(0, 10)]


def test_chase_allows_tail_contact_but_head_contact_splits_bouts():
    f = features()
    f.speed_cm_s[:] = 30.
    f.pair_nose_tail_cm[20] = 1.
    plans, _ = screen_record(record("Chasing", f), f, HeuristicParameters())
    assert [(p.start_frame, p.end_frame) for p in plans] == [(0, 40)]
    f.pair_nose_head_cm[20] = 1.
    plans, _ = screen_record(record("Chasing", f), f, HeuristicParameters())
    assert [(p.start_frame, p.end_frame) for p in plans] == [(0, 20), (21, 40)]
    assert all(not (p.start_frame <= 20 < p.end_frame) for p in plans)


def test_approach_ends_before_contact_and_together_excludes_approach_phase():
    f = features(n=60)
    f.speed_cm_s[:20, 0] = 10.
    nn = np.r_[np.linspace(12., 3., 20), np.ones(10), np.full(30, 6.)][:, None]
    f.pair_nose_nearest_left_to_right_cm = nn
    f.pair_nose_nearest_right_to_left_cm = np.full_like(nn, 100.)
    f.pair_nose_head_cm[20:30] = 1.
    plans, _ = screen_record(record("Approach", f), f, HeuristicParameters())
    approach_plans = [p for p in plans if p.behavior == 'Approach']
    assert approach_plans and all(p.end_frame <= 20 for p in approach_plans)
    together, _ = screen_record(record("Together", f), f, HeuristicParameters())
    assert together and all(p.start_frame >= max(w.end_frame for w in approach_plans) for p in together)


def test_social_clustering_is_formation_before_huddle_not_whole_group_travel():
    f = features(n=100, mice=3)
    f.centers_cm[:, 0, 0] = np.maximum(20.-np.arange(100)*2., 0.)
    f.centers_cm[:, 2, 0] = 1.
    f.velocity_cm_s[1:11, 0, 0] = -20.
    f.speed_cm_s[1:11, 0] = 20.
    f.pair_distance_cm = np.linalg.norm(f.centers_cm[:, f.pair_i]-f.centers_cm[:, f.pair_j], axis=2)
    compact, formation = group_phase_masks(f, [0, 1, 2], HeuristicParameters())
    onset = np.flatnonzero(compact)[0]
    assert np.any(formation[:onset])
    assert not np.any(formation[onset:])
    plans, _ = screen_record(record("Social clustering", f), f, HeuristicParameters())
    assert plans and all(p.end_frame <= onset for p in plans)
    f.speed_cm_s[:] = 20.
    _, formation = group_phase_masks(f, [0, 1, 2], HeuristicParameters())
    assert not formation.any()


def test_posture_probe_requires_body_static_and_human_head_turn_negative_wins():
    f = features(mice=2)
    f.speed_cm_s[20:, 1] = 30.
    episodes = [{"track_id": 101, "candidate": "grooming", "start_frame": 0, "end_frame": 40}]
    audit = [{"track_id": "101", "observed_behavior": "static", "decision": "confirmed",
              "start_frame": "0", "end_frame": "10", "review_id": "015"}]
    plans, _ = screen_record(record("Static", f, (100,)), f, HeuristicParameters(),
                             posture_episodes=episodes, audit_rows=audit,
                             policy=ScreenPolicy(posture_export_mode="review_candidates"))
    groom = [p for p in plans if p.behavior == "Grooming"]
    assert len(groom) == 1 and groom[0].mouse_ids == (101,)
    assert (groom[0].start_frame, groom[0].end_frame) == (10, 20)
    assert groom[0].status == "posture_candidate_needs_review"


def test_missing_annotated_target_cannot_be_replaced_by_visible_mouse():
    f = features()
    plans, reason = screen_record(record("Together", f, (100, 999)), f, HeuristicParameters())
    assert plans == [] and "target_missing" in reason


def test_unchanged_labels_keep_full_clip_and_locomotion_merge():
    f = features()
    plans, _ = screen_record(record("Running", f, (100,)), f, HeuristicParameters())
    assert plans[0].behavior == "Locomotion"
    assert (plans[0].start_frame, plans[0].end_frame) == (0, 40)
    assert plans[0].status == "original_label_preserved"


def test_contiguous_windows_half_open_and_fractional_fps_boundary():
    assert contiguous_windows(np.array([False, True, True, True, False]), 29.97, .1) == [(1, 4)]
    assert contiguous_windows(np.array([True, True]), 29.97, .1) == []
    f = features()
    bad = replace(record("Together", f), frame_count=41)
    assert screen_record(bad, f, HeuristicParameters())[1] == "track_video_frame_count_mismatch"


def test_cropped_contact_cannot_borrow_duration_from_separate_touch():
    f = features()
    f.pair_nose_head_cm[0:3] = 1.
    f.pair_nose_head_cm[10:13] = 1.
    plans, _ = screen_record(record("Snout-head_contact", f), f, HeuristicParameters())
    assert plans == []  # 0.6 s overall, but each standalone bout is only 0.3 s.


def test_static_dataset_cli_probes_use_actual_audit_contract(tmp_path):
    root = Path(__file__).resolve().parents[2]
    dataset = tmp_path / "input"
    sample = dataset / "Static" / "sample"
    sample.mkdir(parents=True)
    points = [[20, 50, 1], [28, 44, 1], [28, 56, 1], [50, 50, 1],
              [78, 30, 1], [78, 70, 1], [120, 50, 1]]
    frames = [{"frame": i, "detections": [{"track_id": 7, "keypoints": points}]} for i in range(60)]
    (sample / "tracks.json").write_text(json.dumps(frames), encoding="utf-8")
    (sample / "annotation.json").write_text(json.dumps({"behavior": "Static", "mouse_ids": [7],
        "frame_range": {"start": 0, "end": 59}, "time_range": {"start": 0, "end": 2}}), encoding="utf-8")
    (sample / "metadata.json").write_text(json.dumps({"clip": {"filename": "clip.mp4", "fps": 30,
        "frame_count": 60}, "source": {"start_frame": 100, "end_frame": 159}}), encoding="utf-8")
    (sample / "clip.mp4").write_bytes(b"planning reads metadata and tracks only")
    output = tmp_path / "output"
    subprocess.run([sys.executable, str(root / "scripts/screen_behavior_segments.py"),
                    "--dataset-root", str(dataset), "--output-dir", str(output), "--plan-only"],
                   cwd=root, check=True, capture_output=True, text=True)
    summary = json.loads((output / "screening_summary.json").read_text(encoding="utf-8"))
    assert summary["errors"] == []
    assert summary["selected_by_behavior"]["Static"] == 1


def test_non_static_human_posture_window_is_rendered_without_uncertain_labels(tmp_path):
    import csv
    import cv2
    root = Path(__file__).resolve().parents[2]
    dataset = tmp_path / "input"
    sample = dataset / "Dispersal" / "human_example"
    sample.mkdir(parents=True)
    points = [[20, 50, 1], [28, 44, 1], [28, 56, 1], [50, 50, 1],
              [78, 30, 1], [78, 70, 1], [120, 50, 1]]
    tracks = [{"frame": i, "time": i/30, "detections": [{"track_id": 7, "keypoints": points}]} for i in range(60)]
    (sample / "tracks.json").write_text(json.dumps(tracks), encoding="utf-8")
    (sample / "annotation.json").write_text(json.dumps({"behavior": "Dispersal", "mouse_ids": [7],
        "frame_range": {"start": 0, "end": 59}, "time_range": {"start": 0, "end": 2}}), encoding="utf-8")
    (sample / "metadata.json").write_text(json.dumps({"clip": {"filename": "clip.mp4", "fps": 30,
        "frame_count": 60}, "source": {"start_frame": 100, "end_frame": 159}}), encoding="utf-8")
    writer = cv2.VideoWriter(str(sample / "clip.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30., (192, 128))
    assert writer.isOpened()
    try:
        for i in range(60):
            writer.write(np.full((128, 192, 3), i*3, np.uint8))
    finally:
        writer.release()
    audit = tmp_path / "audit.csv"
    fields = ["clip_path", "track_id", "start_frame", "end_frame", "review_id", "decision", "observed_behavior"]
    with audit.open("w", encoding="utf-8", newline="") as stream:
        out = csv.DictWriter(stream, fieldnames=fields)
        out.writeheader()
        for review, decision, label, track in (("true", "confirmed", "grooming", 7),
                                               ("unsure", "uncertain", "rearing", 7),
                                               ("not_mouse", "rejected", "rearing", 99)):
            out.writerow(dict(clip_path=str(sample), track_id=track, start_frame=10, end_frame=40,
                              review_id=review, decision=decision, observed_behavior=label))
    output = tmp_path / "output"
    subprocess.run([sys.executable, str(root / "scripts/screen_behavior_segments.py"),
                    "--dataset-root", str(dataset), "--output-dir", str(output), "--human-audit", str(audit)],
                   cwd=root, check=True, capture_output=True, text=True)
    rows = json.loads((output / "review_manifest.json").read_text(encoding="utf-8"))
    windows = rows[0]["selected_windows"]
    assert len(windows) == 1 and windows[0]["behavior"] == "Grooming"
    assert windows[0]["status"] == "human_confirmed"
    assert (windows[0]["start_frame"], windows[0]["end_frame"], windows[0]["mouse_ids"]) == (10, 40, [7])
    target = output / windows[0]["export_path"]
    annotation = json.loads((target / "annotation.json").read_text())
    metadata = json.loads((target / "metadata.json").read_text())
    assert annotation["requires_review"] is False
    assert annotation["frame_range"] == {"start": 0, "end": 29}
    assert metadata["source"]["start_frame"] == 110 and metadata["source"]["end_frame"] == 139
