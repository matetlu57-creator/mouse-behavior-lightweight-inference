"""Prepare review-only Together previews from contact-rejected Approach records."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path

import cv2

from _bootstrap import REPO_ROOT  # noqa: F401
from mouse_behavior.evaluation.exported_dataset import SampleRecord
from mouse_behavior.evaluation.manual_review import ReviewStore
from mouse_behavior.evaluation.review_transaction import tree_hashes, write_json
from mouse_behavior.evaluation.segment_export import export_segment


BATCH_ID = 'together_contact_transfer_v1'
MANIFEST_NAME = 'together_contact_candidates_v1.json'
MIN_DURATION_SECONDS = 1.0


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
            raise RuntimeError(f'cannot create Together review poster: {destination}')
    finally:
        capture.release()


def prepare(package_root: Path, confirmed_root: Path, source_root: Path,
            output_root: Path) -> dict:
    package_root, confirmed_root, source_root = (
        path.expanduser().resolve() for path in (package_root, confirmed_root, source_root)
    )
    output_root = output_root.expanduser().resolve()
    with _open_review_store(package_root, confirmed_root, source_root) as store:
        manifest_path = store.package / 'combined_review_manifest.json'
        state_bytes = store.state_path.read_bytes()
        state = json.loads(state_bytes)
        decisions = state.get('decisions', {})
        rejected = []
        for review_id, card in store.base_cards.items():
            decision = decisions.get(review_id, {})
            if (card['behavior'] == 'Approach'
                    and decision.get('decision') == 'rejected'
                    and decision.get('behavior') == 'Approach'
                    and '出现身体接触' in decision.get('note', '')):
                rejected.append((review_id, card, decision))
        if not rejected:
            raise ValueError('no body-contact Approach rejections are available for Together review')
        package_manifest = store.package / MANIFEST_NAME
        batch_root = store.package / 'dataset' / 'Together' / BATCH_ID
        if package_manifest.exists() or batch_root.exists() or output_root.exists():
            raise FileExistsError('Together contact candidate outputs already exist; refusing to overwrite')

        cards, omitted = [], []
        for review_id, card, decision in rejected:
            start, end = decision['start_frame'], decision['end_frame']
            if (type(start) is not int or type(end) is not int
                    or not 0 <= start < end <= card['frame_count']):
                raise ValueError(f'invalid saved Approach frame range: {review_id}')
            duration = (end - start) / float(card['fps'])
            reason = (
                '接近审核因出现身体接触未通过；请复核接触后是否进入静止、无互动的一起行为，'
                '并将通过片段裁剪为至少1秒。'
            )
            if duration < MIN_DURATION_SECONDS:
                omitted.append({
                    'review_id': review_id,
                    'source_key': card['source_key'],
                    'sample_id': card['sample_id'],
                    'mouse_ids': list(decision['mouse_ids']),
                    'start_frame': start,
                    'end_frame': end,
                    'fps': float(card['fps']),
                    'frame_count': int(card['frame_count']),
                    'candidate_status': 'approach_contact_source_shorter_than_1s',
                    'candidate_reason': (
                        f'{reason} 当前原片片段只有{duration:.3f}秒，短于1秒；'
                        '现有样本没有更长上下文，不能作为合格一起片段通过。'
                    ),
                    'duration_seconds': duration,
                })
                continue

            source_row = store.sources[card['source_key']]
            source = _source_sample(store, card['source_key'])
            record = _record_for_source(source_row, source)
            relative = Path('dataset') / 'Together' / BATCH_ID / review_id
            target = store.package / relative
            if target.exists():
                raise FileExistsError(f'refusing to overwrite existing Together candidate: {target}')
            export_segment(
                record, target,
                start_frame=start, end_frame=end, behavior='Together',
                mouse_ids=tuple(decision['mouse_ids']),
                provenance={
                    'candidate': {'status': 'screening_candidate_needs_review'},
                    'source_rejection': {
                        'review_id': review_id,
                        'decision': 'rejected',
                        'behavior': 'Approach',
                        'note': decision['note'],
                    },
                    'screening': {
                        'batch_id': BATCH_ID,
                        'method': 'review-only preview of the rejected source interval; reviewer must select a Together-only crop',
                        'interval_semantics': 'clip-local half-open [start_frame, end_frame)',
                        'minimum_approved_duration_seconds': MIN_DURATION_SECONDS,
                    },
                },
            )
            _poster(target / 'clip.mp4', target / 'target_poster.jpg')
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
                'candidate_status': 'approach_contact_to_together_review',
                'candidate_reason': reason,
                'duration_seconds': duration,
                'output_sha256': tree_hashes(target),
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
            'minimum_approved_duration_seconds': MIN_DURATION_SECONDS,
            'candidate_count': len(cards),
            'omitted_count': len(omitted),
            'cards': cards,
            'omitted_candidates': omitted,
        }
        if len(cards) + len(omitted) != len(rejected):
            raise RuntimeError('Together transfer did not account for every rejected source')
        output_root.mkdir(parents=True, exist_ok=False)
        write_json(package_manifest, document)
        write_json(output_root / 'candidate_manifest.json', document)
        write_json(output_root / 'summary.json', {
            'batch_id': BATCH_ID,
            'source_rejections': len(rejected),
            'reviewable_previews_at_least_1s': len(cards),
            'omitted_sources_shorter_than_1s': len(omitted),
            'source_decisions_revision': state['revision'],
            'candidate_manifest_sha256': _digest(package_manifest),
        })
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package-root', type=Path, required=True)
    parser.add_argument('--confirmed-root', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    document = prepare(args.package_root, args.confirmed_root, args.source_root, args.output_root)
    print(
        f'Prepared {document["candidate_count"]} Together previews at least 1 second; '
        f'{document["omitted_count"]} source clips are too short.'
    )
    print(args.output_root / 'candidate_manifest.json')


if __name__ == '__main__':
    main()
