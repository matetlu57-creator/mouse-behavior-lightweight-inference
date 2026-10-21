"""Keypoint-screen rejected Together clips for human nose-contact reclassification."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import shutil
import uuid
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np

from _bootstrap import REPO_ROOT  # noqa: F401
from mouse_behavior.config import load_config
from mouse_behavior.evaluation.exported_dataset import (
    SampleRecord,
    exported_dataset_source_digest,
    load_exported_features,
)
from mouse_behavior.evaluation.manual_review import ReviewStore
from mouse_behavior.evaluation.review_transaction import safe_path, tree_hashes, write_json
from mouse_behavior.evaluation.segment_export import export_segment
from mouse_behavior.preprocessing.scale_resolution import resolve_calibration_path


BATCH_ID = 'together_contact_geometry_screen_v1'
MANIFEST_NAME = 'together_contact_reclassification_v1.json'


@contextmanager
def _open_review_store(package_root: Path, confirmed_root: Path, source_root: Path):
    store = ReviewStore(package_root, confirmed_root, source_root)
    try:
        yield store
    finally:
        store.close()


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def _source_sample(store: ReviewStore, key: str) -> Path:
    row = store.sources[key]
    record = row['record']
    backup_dir = store.root / 'backups' / key
    backup = backup_dir / 'sample'
    if backup_dir.exists():
        receipt_path = backup_dir / 'receipt.json'
        if not receipt_path.is_file() or not backup.is_dir():
            raise ValueError(f'incomplete immutable source backup: {backup_dir}')
        receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        if receipt.get('sample_id') != record['sample_id'] or tree_hashes(backup) != receipt.get('sha256'):
            raise ValueError(f'immutable source backup changed: {backup_dir}')
        return backup

    if store._state().get('sources', {}).get(key, {}).get('managed'):
        raise ValueError(f'managed source has no immutable backup: {key}')
    for field in ('clip_path', 'tracks_path', 'annotation_path', 'metadata_path'):
        path = Path(record[field])
        expected = row['source_fingerprints'].get(field)
        if not isinstance(expected, dict) or not path.is_file():
            raise ValueError(f'missing source fingerprint or file: {path}')
        stat = path.stat()
        if expected != {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}:
            raise ValueError(f'source changed since screening: {path}')
    return Path(record['sample_dir'])


def _record_for_source(row: dict, source: Path) -> SampleRecord:
    data = copy.deepcopy(row['record'])
    data.update(
        sample_dir=str(source),
        clip_path=str(source / 'clip.mp4'),
        tracks_path=str(source / 'tracks.json'),
        annotation_path=str(source / 'annotation.json'),
        metadata_path=str(source / 'metadata.json'),
        mouse_ids=tuple(data['mouse_ids']),
    )
    return SampleRecord(**data)


def _poster(video: Path, destination: Path) -> None:
    capture = cv2.VideoCapture(str(video))
    try:
        ok, frame = capture.read()
        if not ok or not cv2.imwrite(str(destination), frame):
            raise RuntimeError(f'cannot create contact-screen poster: {destination}')
    finally:
        capture.release()


def _frozen_screen_inputs(provenance: dict, config_path: Path) -> tuple[dict, float, float, float, Path]:
    """Use receipt-frozen contact thresholds and verify every live scale input."""
    parameters = provenance.get('parameters')
    units = provenance.get('parameter_units')
    if not isinstance(parameters, dict) or not isinstance(units, dict):
        raise ValueError('frozen review provenance is missing parameters or parameter units')

    head_threshold = tail_threshold = float(parameters['contact_distance_cm'])
    minimum_seconds = float(parameters['contact_min_cumulative_seconds'])
    reference_length = float(units['reference_body_length_cm'])
    reference_width = float(units['reference_body_width_cm'])
    if any(not math.isfinite(value) or value <= 0 for value in (
            head_threshold, tail_threshold, minimum_seconds, reference_length, reference_width)):
        raise ValueError('frozen contact or scale parameters must be finite and positive')

    config = load_config(config_path)
    scale = config.get('scale', {})
    contact = config.get('contact_detection', {})
    if str(scale.get('mode', '')).lower() != 'body_length':
        raise ValueError('current scale mode differs from the frozen body-length review scale')
    if contact.get('enabled') is not True:
        raise ValueError('current contact detection is disabled; refusing to infer screening parameters')
    for key, expected in (
        ('assumed_mouse_body_length_cm', reference_length),
        ('assumed_mouse_body_width_cm', reference_width),
    ):
        actual = float(scale.get(key, float('nan')))
        if not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-9):
            raise ValueError(f'current scale.{key} differs from the frozen review units')

    for key, expected in (
        ('nose_head_distance_cm', head_threshold),
        ('nose_tail_distance_cm', tail_threshold),
        ('nose_head_min_cumulative_duration_seconds', minimum_seconds),
        ('nose_tail_min_cumulative_duration_seconds', minimum_seconds),
    ):
        actual = float(contact.get(key, float('nan')))
        if not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-9):
            raise ValueError(f'current contact_detection.{key} differs from the frozen review parameters')

    calibration_path = resolve_calibration_path(scale.get('calibration_file', ''))
    expected_calibration_sha = provenance.get('calibration_sha256')
    if not isinstance(expected_calibration_sha, str) or _digest(calibration_path) != expected_calibration_sha:
        raise ValueError('scale calibration does not match the frozen review package')
    calibration = json.loads(calibration_path.read_text(encoding='utf-8'))
    subject = calibration.get('subject_metrics', {})
    measured_length = float(subject.get('unified_body_length_cm', float('nan')))
    measured_width = float(subject.get('segments_cm', {}).get('hip_width', {}).get('mean_cm', float('nan')))
    if (not math.isclose(measured_length, reference_length, rel_tol=0, abs_tol=1e-9)
            or not math.isclose(measured_width, reference_width, rel_tol=0, abs_tol=1e-9)):
        raise ValueError('calibration body dimensions differ from the frozen review units')
    return config, head_threshold, tail_threshold, minimum_seconds, calibration_path


def _run_summary(features, mouse_ids, start, end, *, head_threshold, tail_threshold,
                 minimum_seconds):
    result = {
        'suggested_behavior': None,
        'screen_status': 'keypoints_insufficient',
        'nose_head_contact_seconds': 0.0,
        'nose_tail_contact_seconds': 0.0,
        'nose_head_contact_bouts': 0,
        'nose_tail_contact_bouts': 0,
        'keypoint_valid_fraction': 0.0,
    }
    slots = {int(track_id): index for index, track_id in enumerate(features.track_ids)}
    if len(mouse_ids) != 2 or any(mouse_id not in slots for mouse_id in mouse_ids):
        return result
    pair_index = next((
        index for index, (left, right) in enumerate(zip(features.pair_i, features.pair_j))
        if {int(features.track_ids[left]), int(features.track_ids[right])} == set(mouse_ids)
    ), None)
    if pair_index is None:
        return result

    stop = min(end, len(features.pair_distance_cm))
    if start >= stop:
        return result
    valid_pair = np.isfinite(features.pair_distance_cm[start:stop, pair_index])
    geometry_channels = (
        features.pair_nose_head_left_to_right_cm,
        features.pair_nose_head_right_to_left_cm,
        features.pair_nose_tail_left_to_right_cm,
        features.pair_nose_tail_right_to_left_cm,
    )
    finite_geometry = [
        np.isfinite(channel[start:stop, pair_index])
        for channel in geometry_channels if channel is not None
    ]
    geometry_valid = (np.logical_or.reduce(finite_geometry) & valid_pair
                      if finite_geometry else np.zeros_like(valid_pair))
    result['keypoint_valid_fraction'] = (
        float(geometry_valid.mean()) if len(geometry_valid) else 0.0
    )
    if not geometry_valid.any():
        return result

    def measure(first, second, threshold):
        if first is None or second is None:
            return 0.0, 0
        a = np.asarray(first[start:stop, pair_index], dtype=float)
        b = np.asarray(second[start:stop, pair_index], dtype=float)
        contact_a = valid_pair & np.isfinite(a) & (a < threshold)
        contact_b = valid_pair & np.isfinite(b) & (b < threshold)
        union = contact_a | contact_b
        seconds = float(union.sum()) / features.fps
        padded = np.r_[False, union, False].astype(np.int8)
        transitions = np.diff(padded)
        bouts = int(np.sum(transitions == 1))
        return seconds, bouts

    head_seconds, head_bouts = measure(
        features.pair_nose_head_left_to_right_cm,
        features.pair_nose_head_right_to_left_cm,
        head_threshold,
    )
    tail_seconds, tail_bouts = measure(
        features.pair_nose_tail_left_to_right_cm,
        features.pair_nose_tail_right_to_left_cm,
        tail_threshold,
    )
    result.update(
        nose_head_contact_seconds=head_seconds,
        nose_tail_contact_seconds=tail_seconds,
        nose_head_contact_bouts=head_bouts,
        nose_tail_contact_bouts=tail_bouts,
    )
    head_qualifies = head_seconds >= minimum_seconds
    tail_qualifies = tail_seconds >= minimum_seconds
    if head_qualifies and not tail_qualifies:
        result.update(suggested_behavior='Snout-head_contact', screen_status='nose_head_supported')
    elif tail_qualifies and not head_qualifies:
        result.update(suggested_behavior='Snout-rear_contact', screen_status='nose_tail_supported')
    elif head_qualifies and tail_qualifies:
        result['screen_status'] = 'both_contact_types_supported'
    elif head_seconds > 0 or tail_seconds > 0:
        result['screen_status'] = 'subthreshold_contact_evidence'
    else:
        result['screen_status'] = 'no_contact_threshold_met'
    return result


def _screen_reason(evidence: dict, head_threshold: float, tail_threshold: float,
                   minimum_seconds: float) -> str:
    head = evidence['nose_head_contact_seconds']
    tail = evidence['nose_tail_contact_seconds']
    suggested = evidence['suggested_behavior']
    if suggested == 'Snout-head_contact':
        hint = '关键点初筛建议鼻头接触'
    elif suggested == 'Snout-rear_contact':
        hint = '关键点初筛建议鼻尾接触'
    elif evidence['screen_status'] == 'both_contact_types_supported':
        hint = '鼻头、鼻尾两类几何证据都达到阈值，类型待人工判断'
    elif evidence['screen_status'] == 'subthreshold_contact_evidence':
        hint = '存在低于累计时长阈值的接触几何证据，类型待人工判断'
    elif evidence['screen_status'] == 'no_contact_threshold_met':
        hint = '当前裁剪范围内没有鼻头或鼻尾几何达到时长阈值，需人工核对'
    else:
        hint = '目标关键点缺失或覆盖不足，需人工核对'
    return (
        f'{hint}；鼻头估计 {head:.3f} 秒（距离阈值 {head_threshold:.3f} cm），'
        f'鼻尾估计 {tail:.3f} 秒（距离阈值 {tail_threshold:.3f} cm），'
        f'接触累计时长阈值 {minimum_seconds:.3f} 秒；关键点有效帧比例 '
        f'{evidence["keypoint_valid_fraction"]:.1%}。初筛只用于排序，不能替代视频人工确认。'
    )


def prepare(package_root: Path, confirmed_root: Path, source_root: Path,
            output_root: Path, config_path: Path | None = None) -> dict:
    package_root, confirmed_root, source_root = (
        path.expanduser().resolve() for path in (package_root, confirmed_root, source_root)
    )
    output_root = output_root.expanduser().resolve()
    if any(output_root.is_relative_to(root) or root.is_relative_to(output_root)
           for root in (package_root, confirmed_root, source_root)):
        raise ValueError('contact-screen output must be separate from package, source and confirmed roots')

    with _open_review_store(package_root, confirmed_root, source_root) as store:
        manifest_path = store.package / 'combined_review_manifest.json'
        provenance = json.loads((store.package / 'provenance.json').read_text(encoding='utf-8'))
        config_path = (config_path or Path(provenance['config'])).expanduser().resolve()
        config, head_threshold, tail_threshold, minimum_seconds, calibration_path = (
            _frozen_screen_inputs(provenance, config_path)
        )

        state_bytes = store.state_path.read_bytes()
        state = json.loads(state_bytes)
        decisions = state.get('decisions', {})
        rejected = []
        for review_id, card in store.base_cards.items():
            decision = decisions.get(review_id, {})
            if (card['behavior'] == 'Together'
                    and decision.get('decision') == 'rejected'
                    and decision.get('behavior') == 'Together'):
                rejected.append((review_id, card, decision))
        if not rejected:
            raise ValueError('no rejected Together cards are available for contact screening')

        package_manifest = store.package / MANIFEST_NAME
        batch_root = store.package / 'review_sources' / BATCH_ID
        if package_manifest.exists() or batch_root.exists() or output_root.exists():
            raise FileExistsError('Together contact-screen outputs already exist; refusing to overwrite')

        token = uuid.uuid4().hex
        staging_root = batch_root.with_name(f'.{BATCH_ID}.staging-{token}')
        output_stage = output_root.with_name(f'.{output_root.name}.staging-{token}')
        manifest_stage = store.package / f'.{MANIFEST_NAME}.staging-{token}'
        batch_committed = output_committed = False
        try:
            staging_root.mkdir(parents=True, exist_ok=False)
            output_stage.mkdir(parents=True, exist_ok=False)
            cards = []
            feature_cache = {}
            for review_id, card, decision in rejected:
                start, end = decision['start_frame'], decision['end_frame']
                if (type(start) is not int or type(end) is not int
                        or not 0 <= start < end <= card['frame_count']):
                    raise ValueError(f'invalid saved Together frame range: {review_id}')
                source_row = store.sources[card['source_key']]
                source = _source_sample(store, card['source_key'])
                record = _record_for_source(source_row, source)
                if card['source_key'] not in feature_cache:
                    feature_cache[card['source_key']] = load_exported_features(
                        record.tracks_path, fps=record.fps, scale_config=config.get('scale', {})
                    )
                features = feature_cache[card['source_key']]
                evidence = _run_summary(
                    features, tuple(decision['mouse_ids']), start, end,
                    head_threshold=head_threshold,
                    tail_threshold=tail_threshold,
                    minimum_seconds=minimum_seconds,
                )
                reason = _screen_reason(evidence, head_threshold, tail_threshold, minimum_seconds)
                stage_target = staging_root / review_id
                export_segment(
                    record, stage_target,
                    start_frame=start, end_frame=end, behavior='Together',
                    mouse_ids=tuple(decision['mouse_ids']),
                    provenance={
                        'candidate': {'status': 'keypoint_prescreen_needs_human_review', **evidence},
                        'source_rejection': {
                            'review_id': review_id,
                            'decision': 'rejected',
                            'behavior': 'Together',
                            'note': decision.get('note', ''),
                        },
                        'screening': {
                            'batch_id': BATCH_ID,
                            'method': 'direct nose-to-opposing-head/rear keypoint geometry in the saved clip-local window',
                            'thresholds_cm': {
                                'nose_head': head_threshold,
                                'nose_tail': tail_threshold,
                            },
                            'minimum_cumulative_seconds': minimum_seconds,
                            'human_confirmation_required': True,
                        },
                    },
                )
                _poster(stage_target / 'clip.mp4', stage_target / 'target_poster.jpg')
                relative = Path('review_sources') / BATCH_ID / review_id
                cards.append({
                    'review_id': review_id,
                    'source_key': card['source_key'],
                    'sample_id': card['sample_id'],
                    'mouse_ids': list(decision['mouse_ids']),
                    'start_frame': start,
                    'end_frame': end,
                    'fps': float(card['fps']),
                    'frame_count': int(card['frame_count']),
                    'export_path': relative.as_posix(),
                    'poster_path': (relative / 'target_poster.jpg').as_posix(),
                    'candidate_status': 'together_contact_geometry_prescreen',
                    'candidate_reason': reason,
                    'screening_evidence': evidence,
                    'review_origin_behavior': 'Together',
                    'review_origin_decision_updated_at': decision.get('updated_at'),
                    'output_sha256': tree_hashes(stage_target),
                })

            document = {
                'schema_version': 1,
                'batch_id': BATCH_ID,
                'status': 'pending_human_review',
                'automatic_detector_enabled': False,
                'base_manifest_sha256': store.manifest_sha,
                'source_decisions_revision': state['revision'],
                'source_decisions_sha256': hashlib.sha256(state_bytes).hexdigest(),
                'source_manifest_sha256': _digest(manifest_path),
                'config_sha256': _digest(config_path),
                'frozen_config_sha256': provenance['config_sha256'],
                'frozen_calibration_sha256': provenance['calibration_sha256'],
                'feature_loader_sha256': exported_dataset_source_digest(REPO_ROOT),
                'frozen_feature_loader_sha256': provenance.get('exported_dataset_sha256'),
                'calibration_path': str(calibration_path),
                'contact_distance_cm': {
                    'nose_head': head_threshold,
                    'nose_tail': tail_threshold,
                },
                'minimum_cumulative_seconds': minimum_seconds,
                'candidate_count': len(cards),
                'screen_status_counts': {},
                'cards': cards,
            }
            document['screen_status_counts'] = dict(Counter(
                item['screening_evidence']['screen_status'] for item in cards
            ))
            write_json(output_stage / 'candidate_manifest.json', document)
            write_json(output_stage / 'summary.json', {
                'batch_id': BATCH_ID,
                'source_rejections': len(rejected),
                'candidate_count': len(cards),
                'screen_status_counts': document['screen_status_counts'],
                'source_decisions_revision': state['revision'],
                'automatic_detector_enabled': False,
            })
            write_json(manifest_stage, document)
            staging_root.rename(batch_root)
            batch_committed = True
            output_stage.rename(output_root)
            output_committed = True
            manifest_stage.replace(package_manifest)
        except BaseException:
            if batch_committed and batch_root.exists():
                safe_path(batch_root, [store.package / 'review_sources'])
                shutil.rmtree(batch_root)
            if output_committed and output_root.exists():
                shutil.rmtree(output_root)
            if staging_root.exists():
                safe_path(staging_root, [store.package / 'review_sources'])
                shutil.rmtree(staging_root)
            if output_stage.exists():
                shutil.rmtree(output_stage)
            if manifest_stage.exists():
                manifest_stage.unlink()
            raise

    return document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package-root', type=Path, required=True)
    parser.add_argument('--confirmed-root', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--config', type=Path)
    args = parser.parse_args()
    document = prepare(
        args.package_root, args.confirmed_root, args.source_root,
        args.output_root, args.config,
    )
    print(
        f"Prepared {document['candidate_count']} contact-screen cards; "
        f"evidence buckets: {json.dumps(document['screen_status_counts'], sort_keys=True)}"
    )
    print(args.output_root / 'candidate_manifest.json')


if __name__ == '__main__':
    main()
