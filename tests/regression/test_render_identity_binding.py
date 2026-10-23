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
        'video': str(tmp_path / 'input.mp4'), 'fps': 10,
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
        'source_video': str(video), 'yolo_cache': str(cache), 'source_frames': 2, 'source_fps': 10,
        'identity_source': 'rfid_mtrack', 'identity_labels': ['mouse-A'],
        'identity_map': str(mapping),
        'identity_map_sha256': hashlib.sha256(mapping.read_bytes()).hexdigest(),
    }
    (tmp_path / 'lightweight_analysis_metadata.json').write_text(json.dumps(metadata))
    events = tmp_path / 'events.csv'
    events.write_text('start_frame,end_frame,candidate_level,behavior,actor_id\n0,1,individual,stationary,0\n')
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    status_path = cache / 'yolo_results_status.json'
    status = json.loads(status_path.read_text())
    status.update(source_video_sha256=digest(video),
                  chunk_sha256={path.name: digest(path) for path in cache.glob('*.pkl.gz')})
    status_path.write_text(json.dumps(status))
    metadata.update(source_binding={
        'source_video_sha256': digest(video), 'cache_status_sha256': digest(status_path),
        'cache_chunk_sha256': status['chunk_sha256'],
    }, event_files_sha256={events.name: digest(events)})
    (tmp_path / 'lightweight_analysis_metadata.json').write_text(json.dumps(metadata))
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


@pytest.mark.parametrize('change', ['source', 'fps', 'frames', 'analysis_source', 'missing_source'])
def test_render_rejects_wrong_source_or_timebase_before_writer(render_inputs, tmp_path, change):
    video, cache, events, _, metadata = render_inputs
    status_path = cache / 'yolo_results_status.json'
    status = json.loads(status_path.read_text())
    if change == 'source':
        wrong = tmp_path / 'different.mp4'
        writer = cv2.VideoWriter(str(wrong), cv2.VideoWriter_fourcc(*'mp4v'), 20, (64, 48))
        for _ in range(5):
            writer.write(np.full((48, 64, 3), 200, np.uint8))
        writer.release()
        video = wrong
    elif change == 'fps':
        status['fps'] = 20
    elif change == 'frames':
        status['total_frames'] = status['next_frame'] = 3
    elif change == 'missing_source':
        status.pop('video')
    else:
        metadata['source_video'] = str(tmp_path / 'unrelated.mp4')
        (tmp_path / 'lightweight_analysis_metadata.json').write_text(json.dumps(metadata))
    status_path.write_text(json.dumps(status))
    with pytest.raises(ValueError, match='source|FPS|frame|binding'):
        render_behavior_video(video, cache, events, tmp_path / 'bad.mp4', expected_mice=1)
    assert not (tmp_path / 'bad.mp4').exists()


def test_render_explicit_preview_records_actual_range(render_inputs, tmp_path):
    video, cache, events, _, _ = render_inputs
    output = tmp_path / 'preview.mp4'
    render_behavior_video(video, cache, events, output, expected_mice=1, max_frames=1)
    receipt = json.loads(output.with_suffix('.identity.json').read_text())
    assert receipt['render_contract'] == 'explicit_preview'
    assert receipt['source_frame_count'] == 2 and receipt['rendered_frame_count'] == 1
    assert receipt['source_frame_range_half_open'] == [0, 1]


def test_render_early_eof_never_publishes_partial_media(render_inputs, tmp_path, monkeypatch):
    from mouse_behavior.visualization import rendering
    video, cache, events, _, _ = render_inputs
    original = cv2.VideoCapture

    class EarlyEOF:
        def __init__(self, *args):
            self.capture = original(*args)
            self.reads = 0

        def __getattr__(self, name):
            return getattr(self.capture, name)

        def read(self):
            self.reads += 1
            return self.capture.read() if self.reads == 1 else (False, None)

    monkeypatch.setattr(rendering.cv2, 'VideoCapture', EarlyEOF)
    output = tmp_path / 'truncated.mp4'
    with pytest.raises((ValueError, RuntimeError), match='decode|frames|EOF'):
        render_behavior_video(video, cache, events, output, expected_mice=1)
    assert not output.exists() and not output.with_suffix('.identity.json').exists()


@pytest.mark.parametrize('change', ['missing_sha', 'event_bytes', 'chunk_bytes', 'metadata_binding', 'same_path_video'])
def test_render_rejects_content_and_analysis_binding_drift(render_inputs, tmp_path, change):
    video, cache, events, _, metadata = render_inputs
    if change == 'missing_sha':
        path = cache / 'yolo_results_status.json'
        status = json.loads(path.read_text())
        status.pop('source_video_sha256')
        path.write_text(json.dumps(status))
    elif change == 'event_bytes':
        events.write_text(events.read_text().replace('stationary', 'attack'))
    elif change == 'chunk_bytes':
        path = next(cache.glob('*.pkl.gz'))
        path.write_bytes(path.read_bytes() + b'changed')
    elif change == 'metadata_binding':
        metadata.pop('source_binding')
        (tmp_path / 'lightweight_analysis_metadata.json').write_text(json.dumps(metadata))
    else:
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), 10, (64, 48))
        for _ in range(2):
            writer.write(np.full((48, 64, 3), 220, np.uint8))
        writer.release()
    with pytest.raises(ValueError, match='binding|SHA|changed|rerun'):
        render_behavior_video(video, cache, events, tmp_path / 'unsafe.mp4', expected_mice=1)
    assert not (tmp_path / 'unsafe.mp4').exists()
