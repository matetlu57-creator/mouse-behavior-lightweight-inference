"""Persisted event slots must never be relabeled by a later RFID CSV edit."""

from __future__ import annotations

import hashlib
import json

import cv2
import numpy as np
import pytest

from mouse_behavior.visualization.rendering import render_behavior_video


@pytest.fixture
def render_inputs(tmp_path, write_pose_cache, pose_payload_factory):
    cache = write_pose_cache(tmp_path / 'cache', [
        {'frame': frame, 'pose_detections': [pose_payload_factory(42, 20)]}
        for frame in range(2)
    ])
    (cache / 'yolo_results_status.json').write_text(json.dumps({
        'status': 'complete', 'total_frames': 2, 'next_frame': 2, 'trajectory_schema_version': 1,
    }))
    video = tmp_path / 'input.mp4'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), 10, (64, 48))
    assert writer.isOpened()
    try:
        for _ in range(2):
            writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    finally:
        writer.release()
    mapping = tmp_path / 'identity.csv'
    mapping.write_text('frame,visual_track_id,animal_id\n0,42,mouse-A\n1,42,mouse-A\n')
    metadata = {
        'identity_source': 'rfid_mtrack', 'identity_labels': ['mouse-A'],
        'identity_map': str(mapping),
        'identity_map_sha256': hashlib.sha256(mapping.read_bytes()).hexdigest(),
    }
    (tmp_path / 'lightweight_analysis_metadata.json').write_text(json.dumps(metadata))
    events = tmp_path / 'events.csv'
    events.write_text('start_frame,end_frame,candidate_level,behavior,actor_id\n0,1,individual,stationary,0\n')
    return video, cache, events, mapping, metadata


@pytest.mark.parametrize('explicit', [False, True])
def test_render_uses_sealed_identity_and_writes_receipt(render_inputs, tmp_path, explicit):
    video, cache, events, mapping, metadata = render_inputs
    output = tmp_path / 'rendered.mp4'
    render_behavior_video(video, cache, events, output, expected_mice=1,
                          identity_map_path=mapping if explicit else None)
    receipt = json.loads(output.with_suffix('.identity.json').read_text(encoding='utf-8'))
    assert receipt['identity_labels'] == ['mouse-A']
    assert receipt['identity_map_sha256'] == metadata['identity_map_sha256']
    assert receipt['events_sha256'] == hashlib.sha256(events.read_bytes()).hexdigest()
    capture = cv2.VideoCapture(str(output))
    try:
        assert capture.isOpened() and capture.read()[0]
    finally:
        capture.release()


@pytest.mark.parametrize('change', ['mapping', 'labels', 'missing_digest', 'invalid_json'])
def test_render_rejects_identity_drift_without_publishing(render_inputs, tmp_path, change):
    video, cache, events, mapping, metadata = render_inputs
    if change == 'mapping':
        # Same biological label set, different frame assignment confidence: labels alone cannot bind this.
        mapping.write_text('frame,visual_track_id,animal_id,identity_confidence\n0,42,mouse-A,0.2\n')
    elif change == 'labels':
        metadata['identity_labels'] = ['mouse-B']
    elif change == 'missing_digest':
        metadata.pop('identity_map_sha256')
    (tmp_path / 'lightweight_analysis_metadata.json').write_text(
        '{bad' if change == 'invalid_json' else json.dumps(metadata),
    )
    output = tmp_path / 'bad.mp4'
    with pytest.raises(ValueError):
        render_behavior_video(video, cache, events, output, expected_mice=1)
    assert not output.exists()
    assert not output.with_suffix('.identity.json').exists()
