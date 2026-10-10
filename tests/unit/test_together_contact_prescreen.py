from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))

from screen_together_contact_rejections import _frozen_screen_inputs, _run_summary


def test_contact_duration_unions_alternating_mouse_directions():
    nan = float('nan')
    features = SimpleNamespace(
        track_ids=np.array([11, 22]),
        pair_i=np.array([0]),
        pair_j=np.array([1]),
        pair_distance_cm=np.full((10, 1), 5.0),
        pair_nose_head_left_to_right_cm=np.array([[1.0], [1.0], [nan], [nan], [nan],
                                                    [nan], [nan], [nan], [nan], [nan]]),
        pair_nose_head_right_to_left_cm=np.array([[nan], [nan], [nan], [1.0], [1.0],
                                                    [1.0], [nan], [nan], [nan], [nan]]),
        pair_nose_tail_left_to_right_cm=np.full((10, 1), 10.0),
        pair_nose_tail_right_to_left_cm=np.full((10, 1), 10.0),
        fps=10.0,
    )

    evidence = _run_summary(
        features, (11, 22), 0, 10,
        head_threshold=2.675,
        tail_threshold=2.675,
        minimum_seconds=0.5,
    )

    assert evidence['nose_head_contact_seconds'] == pytest.approx(0.5)
    assert evidence['nose_head_contact_bouts'] == 2
    assert evidence['suggested_behavior'] == 'Snout-head_contact'
    assert evidence['nose_tail_contact_seconds'] == 0


def _frozen_inputs_fixture(tmp_path):
    calibration = tmp_path / 'rig.json'
    calibration.write_text(json.dumps({
        'subject_metrics': {
            'unified_body_length_cm': 10.7,
            'segments_cm': {'hip_width': {'mean_cm': 4.1467}},
        },
    }), encoding='utf-8')
    config_path = tmp_path / 'default.yaml'
    config_path.write_text(
        f"""scale:
  mode: body_length
  calibration_file: {calibration.as_posix()}
  assumed_mouse_body_length_cm: 10.7
  assumed_mouse_body_width_cm: 4.1467
contact_detection:
  enabled: true
  nose_head_distance_cm: 2.675
  nose_tail_distance_cm: 2.675
  nose_head_min_cumulative_duration_seconds: 0.5
  nose_tail_min_cumulative_duration_seconds: 0.5
unrelated:
  changed_after_screening: true
""",
        encoding='utf-8',
    )
    provenance = {
        'config_sha256': 'frozen-whole-config-hash',
        'calibration_sha256': hashlib.sha256(calibration.read_bytes()).hexdigest(),
        'parameters': {
            'contact_distance_cm': 2.675,
            'contact_min_cumulative_seconds': 0.5,
        },
        'parameter_units': {
            'reference_body_length_cm': 10.7,
            'reference_body_width_cm': 4.1467,
        },
    }
    return provenance, config_path


def test_frozen_contact_inputs_allow_unrelated_config_drift(tmp_path):
    provenance, config_path = _frozen_inputs_fixture(tmp_path)

    _, head, tail, minimum, calibration = _frozen_screen_inputs(provenance, config_path)

    assert (head, tail, minimum) == (2.675, 2.675, 0.5)
    assert calibration == tmp_path / 'rig.json'


def test_frozen_contact_inputs_reject_changed_contact_threshold(tmp_path):
    provenance, config_path = _frozen_inputs_fixture(tmp_path)
    config_path.write_text(config_path.read_text(encoding='utf-8').replace(
        'nose_tail_distance_cm: 2.675', 'nose_tail_distance_cm: 3.0'
    ), encoding='utf-8')

    with pytest.raises(ValueError, match='nose_tail_distance_cm'):
        _frozen_screen_inputs(provenance, config_path)
