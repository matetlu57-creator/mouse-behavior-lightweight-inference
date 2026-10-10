"""Prepare frame-exact, human-review-only Rearing crops from rejected Locomotion cards."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from pathlib import Path

import cv2

from _bootstrap import REPO_ROOT  # noqa: F401
from mouse_behavior.evaluation.exported_dataset import SampleRecord
from mouse_behavior.evaluation.manual_review import review_catalog, source_key
from mouse_behavior.evaluation.review_transaction import tree_hashes, write_json
from mouse_behavior.evaluation.segment_export import export_segment


BATCH_ID = 'rearing_from_rejected_locomotion_v1'
STATUS = 'rejected_locomotion_rearing_review_candidate'
PAYLOADS = ('clip.mp4', 'tracks.json', 'annotation.json', 'metadata.json')


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def _source_for(row: dict, confirmed_root: Path) -> Path:
    record = row['record']
    key = source_key(record['sample_id'])
    backup = confirmed_root / 'backups' / key
    if backup.exists():
        receipt_path = backup / 'receipt.json'
        sample = backup / 'sample'
        if not receipt_path.is_file() or not sample.is_dir():
            raise ValueError(f'incomplete immutable source backup: {backup}')
        receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        if (receipt.get('sample_id') != record['sample_id']
                or tree_hashes(sample) != receipt.get('sha256')):
            raise ValueError(f'immutable source backup changed: {backup}')
        return sample
    sample = Path(record['sample_dir'])
    for field in ('clip_path', 'tracks_path', 'annotation_path', 'metadata_path'):
        path = Path(record[field])
        expected = row['source_fingerprints'].get(field)
        if not isinstance(expected, dict) or not path.is_file():
            raise ValueError(f'missing source fingerprint or file: {path}')
        stat = path.stat()
        if expected != {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}:
            raise ValueError(f'source changed since screening: {path}')
    return sample


def _validate_window(source: Path, record: dict, mouse_id: int, start: int, end: int) -> None:
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= record['frame_count']:
        raise ValueError(f'invalid source frame range [{start}, {end}) for {record["sample_id"]}')
    tracks = json.loads((source / 'tracks.json').read_text(encoding='utf-8'))
    if not isinstance(tracks, list) or len(tracks) != record['frame_count']:
        raise ValueError(f'tracks do not cover the source clip: {record["sample_id"]}')
    if [frame.get('frame') for frame in tracks] != list(range(record['frame_count'])):
        raise ValueError(f'track frame indices differ from source clip: {record["sample_id"]}')
    visible = sum(
        any(int(d.get('track_id', -1)) == mouse_id for d in frame.get('detections', []))
        for frame in tracks[start:end]
    )
    if visible < max(1, round((end - start) * 0.75)):
        raise ValueError(f'visual ID {mouse_id} is too sparse in [{start}, {end})')
    metadata = json.loads((source / 'metadata.json').read_text(encoding='utf-8'))
    if (metadata.get('clip', {}).get('frame_count') != record['frame_count']
            or float(metadata.get('clip', {}).get('fps', 0)) != float(record['fps'])):
        raise ValueError(f'metadata timeline differs from frozen manifest: {record["sample_id"]}')


def _poster(path: Path, destination: Path) -> None:
    capture = cv2.VideoCapture(str(path))
    try:
        ok, frame = capture.read()
        if not ok or not cv2.imwrite(str(destination), frame):
            raise RuntimeError(f'cannot create review poster: {destination}')
    finally:
        capture.release()


def prepare(package_root: Path, confirmed_root: Path, spec_path: Path, output_root: Path) -> dict:
    package_root, confirmed_root = package_root.resolve(), confirmed_root.resolve()
    spec = json.loads(spec_path.read_text(encoding='utf-8'))
    batch_id = spec.get('batch_id')
    if (spec.get('schema_version') != 1 or not isinstance(batch_id, str)
            or re.fullmatch(r'rearing_from_rejected_locomotion_v[1-9][0-9]*', batch_id) is None):
        raise ValueError('unsupported candidate specification')
    manifest_path = package_root / 'combined_review_manifest.json'
    decisions_path = confirmed_root / 'decisions.json'
    manifest_bytes, decisions_bytes = manifest_path.read_bytes(), decisions_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    state = json.loads(decisions_bytes)
    if (state.get('revision') != spec.get('source_decisions_revision')
            or _digest(decisions_path) != spec.get('source_decisions_sha256')):
        raise ValueError('review decisions changed since candidate windows were screened')
    if _digest(manifest_path) != spec.get('base_manifest_sha256'):
        raise ValueError('frozen review manifest changed')
    rows = {source_key(row['record']['sample_id']): row for row in manifest}
    decisions = state['decisions']
    prepared = []
    for candidate in spec.get('candidates', []):
        source_id = candidate['source_review_id']
        rejected = decisions.get(source_id)
        if (not rejected or rejected.get('decision') != 'rejected'
                or rejected.get('behavior') != 'Locomotion'
                or '攀爬' not in rejected.get('note', '')):
            raise ValueError(f'candidate source is no longer a climbing-related Locomotion rejection: {source_id}')
        row = rows.get(rejected['source_key'])
        if (not row or source_key(row['record']['sample_id']) != rejected['source_key']):
            raise ValueError(f'source key does not resolve in frozen manifest: {source_id}')
        record = row['record']
        if len(rejected.get('mouse_ids', [])) != 1:
            raise ValueError(f'expected one rejected visual track: {source_id}')
        mouse_id = rejected['mouse_ids'][0]
        start, end = candidate['start_frame'], candidate['end_frame']
        source = _source_for(row, confirmed_root)
        _validate_window(source, record, mouse_id, start, end)
        candidate_reason = candidate.get('candidate_reason', '').strip()
        if not candidate_reason or len(candidate_reason) > 2000:
            raise ValueError(f'candidate reason is required and must be concise: {source_id}')
        window = {
            'behavior': 'Rearing', 'mouse_ids': [mouse_id], 'start_frame': start,
            'end_frame': end, 'status': STATUS,
            'reason': f'{candidate_reason}；来源退回备注：{rejected["note"]}',
        }
        card = review_catalog([{'record': record, 'selected_windows': [window]}])[0]
        if card['review_id'] in {item['review_id'] for item in prepared}:
            raise ValueError(f'duplicate candidate window: {card["review_id"]}')
        target = package_root / 'dataset' / 'Rearing' / batch_id / card['review_id']
        if target.exists():
            raise FileExistsError(f'refusing to overwrite existing candidate: {target}')
        record_for_export = copy.deepcopy(record)
        record_for_export.update(
            sample_dir=str(source),
            clip_path=str(source / 'clip.mp4'),
            tracks_path=str(source / 'tracks.json'),
            annotation_path=str(source / 'annotation.json'),
            metadata_path=str(source / 'metadata.json'),
            mouse_ids=tuple(record['mouse_ids']),
        )
        export_segment(
            SampleRecord(**record_for_export), target,
            start_frame=start, end_frame=end, behavior='Rearing', mouse_ids=(mouse_id,),
            provenance={
                'candidate': {'status': 'screening_candidate_needs_review'},
                'source_rejection': {'review_id': source_id, 'decision': 'rejected',
                                     'behavior': 'Locomotion', 'note': rejected['note']},
                'screening': {'batch_id': batch_id, 'method': 'human visual prescreen',
                              'interval_semantics': 'clip-local half-open [start, end)'},
            },
        )
        _poster(target / 'clip.mp4', target / 'target_poster.jpg')
        prepared.append({
            'review_id': card['review_id'], 'source_key': card['source_key'],
            'sample_id': card['sample_id'], 'behavior': 'Rearing',
            'mouse_ids': [mouse_id], 'start_frame': start, 'end_frame': end,
            'fps': float(record['fps']), 'frame_count': int(record['frame_count']),
            'export_path': (Path('dataset') / 'Rearing' / batch_id / card['review_id']).as_posix(),
            'poster_path': (Path('dataset') / 'Rearing' / batch_id / card['review_id']
                            / 'target_poster.jpg').as_posix(),
            'candidate_status': STATUS, 'candidate_reason': window['reason'],
            'output_sha256': tree_hashes(target),
        })
    if not prepared:
        raise ValueError('candidate specification contains no windows')
    document = {
        'schema_version': 1, 'batch_id': batch_id,
        'base_manifest_sha256': spec['base_manifest_sha256'],
        'source_decisions_revision': state['revision'],
        'source_decisions_sha256': spec['source_decisions_sha256'],
        'candidate_spec_sha256': _digest(spec_path),
        'cards': prepared,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / 'supplemental_review_candidates.json', document)
    write_json(output_root / 'candidate_manifest.json', {
        'schema_version': 1, 'batch_id': batch_id, 'status': 'pending_human_review',
        'automatic_detector_enabled': False, 'source_decisions_revision': state['revision'],
        'source_decisions_sha256': document['source_decisions_sha256'],
        'base_manifest_sha256': document['base_manifest_sha256'],
        'candidate_spec_sha256': document['candidate_spec_sha256'],
        'frame_interval': 'clip-local half-open [start_frame, end_frame)',
        'candidate_count': len(prepared), 'cards': prepared,
        'omitted_candidates': spec.get('omitted_candidates', []),
    })
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package-root', type=Path, required=True)
    parser.add_argument('--confirmed-root', type=Path, required=True)
    parser.add_argument('--candidate-spec', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    document = prepare(args.package_root, args.confirmed_root, args.candidate_spec, args.output_root)
    print(f'Prepared {len(document["cards"])} pending Rearing review candidates.')
    print(args.output_root / 'supplemental_review_candidates.json')


if __name__ == '__main__':
    main()
