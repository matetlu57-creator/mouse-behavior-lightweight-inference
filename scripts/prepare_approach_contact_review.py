"""Prepare human-review previews for contact-rejected Approach samples."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path

import cv2

from _bootstrap import REPO_ROOT  # noqa: F401
from mouse_behavior.evaluation.exported_dataset import SampleRecord
from mouse_behavior.evaluation.manual_review import ReviewStore
from mouse_behavior.evaluation.review_transaction import safe_path, tree_hashes, write_json
from mouse_behavior.evaluation.segment_export import export_segment


BATCH_ID = 'approach_contact_review_v1'
MANIFEST_NAME = 'approach_contact_review_candidates_v1.json'


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
            raise RuntimeError(f'cannot create Approach review poster: {destination}')
    finally:
        capture.release()


def prepare(package_root: Path, confirmed_root: Path, source_root: Path,
            output_root: Path) -> dict:
    package_root, confirmed_root, source_root = (
        path.expanduser().resolve() for path in (package_root, confirmed_root, source_root)
    )
    output_root = output_root.expanduser().resolve()
    if any(output_root.is_relative_to(root) or root.is_relative_to(output_root)
           for root in (package_root, confirmed_root, source_root)):
        raise ValueError('Approach review output must be separate from package, source and confirmed roots')

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
            raise ValueError('no body-contact Approach rejections are available for review')

        package_manifest = store.package / MANIFEST_NAME
        batch_root = store.package / 'dataset' / 'Approach' / BATCH_ID
        if package_manifest.exists() or batch_root.exists() or output_root.exists():
            raise FileExistsError('Approach contact review outputs already exist; refusing to overwrite')

        token = uuid.uuid4().hex
        staging_root = batch_root.with_name(f'.{BATCH_ID}.staging-{token}')
        output_stage = output_root.with_name(f'.{output_root.name}.staging-{token}')
        manifest_stage = store.package / f'.{MANIFEST_NAME}.staging-{token}'
        batch_committed = output_committed = False
        try:
            staging_root.mkdir(parents=True, exist_ok=False)
            output_stage.mkdir(parents=True, exist_ok=False)
            cards = []
            durations = []
            for review_id, card, decision in rejected:
                start, end = decision['start_frame'], decision['end_frame']
                if (type(start) is not int or type(end) is not int
                        or not 0 <= start < end <= card['frame_count']):
                    raise ValueError(f'invalid saved Approach frame range: {review_id}')
                if not isinstance(decision.get('updated_at'), str):
                    raise ValueError(f'missing source decision timestamp: {review_id}')
                duration = (end - start) / float(card['fps'])
                durations.append(duration)
                source_row = store.sources[card['source_key']]
                source = _source_sample(store, card['source_key'])
                record = _record_for_source(source_row, source)
                stage_target = staging_root / review_id
                export_segment(
                    record, stage_target,
                    start_frame=start, end_frame=end, behavior='Approach',
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
                            'method': 'review-only preview of the saved Approach interval; reviewer must crop any valid approach before body contact',
                            'interval_semantics': 'clip-local half-open [start_frame, end_frame)',
                        },
                    },
                )
                _poster(stage_target / 'clip.mp4', stage_target / 'target_poster.jpg')
                relative = Path('dataset') / 'Approach' / BATCH_ID / review_id
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
                    'candidate_status': 'approach_contact_to_approach_review',
                    'candidate_reason': (
                        '原接近审核因片段含身体接触而退回；现重新审核接触前的接近过程。'
                        '请把终点裁在身体接触之前，并按接近定义复核。'
                    ),
                    'duration_seconds': duration,
                    'review_origin_behavior': 'Approach',
                    'review_origin_decision_updated_at': decision['updated_at'],
                    'output_sha256': tree_hashes(stage_target),
                })

            if len(cards) != len(rejected):
                raise RuntimeError('Approach contact review did not export every source rejection')
            document = {
                'schema_version': 1,
                'batch_id': BATCH_ID,
                'status': 'pending_human_review',
                'automatic_detector_enabled': False,
                'base_manifest_sha256': store.manifest_sha,
                'source_decisions_revision': state['revision'],
                'source_decisions_sha256': hashlib.sha256(state_bytes).hexdigest(),
                'source_manifest_sha256': _digest(manifest_path),
                'candidate_count': len(cards),
                'cards': cards,
            }
            write_json(output_stage / 'candidate_manifest.json', document)
            write_json(output_stage / 'summary.json', {
                'batch_id': BATCH_ID,
                'source_rejections': len(rejected),
                'approach_review_previews': len(cards),
                'previews_shorter_than_one_second': sum(value < 1.0 for value in durations),
                'duration_seconds_min': min(durations),
                'duration_seconds_max': max(durations),
                'source_decisions_revision': state['revision'],
            })
            write_json(manifest_stage, document)
            staging_root.rename(batch_root)
            batch_committed = True
            output_stage.rename(output_root)
            output_committed = True
            manifest_stage.replace(package_manifest)
        except BaseException:
            if batch_committed and batch_root.exists():
                safe_path(batch_root, [store.package / 'dataset' / 'Approach'])
                shutil.rmtree(batch_root)
            if output_committed and output_root.exists():
                shutil.rmtree(output_root)
            if staging_root.exists():
                safe_path(staging_root, [store.package / 'dataset' / 'Approach'])
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
    args = parser.parse_args()
    document = prepare(args.package_root, args.confirmed_root, args.source_root, args.output_root)
    print(
        f'Prepared {document["candidate_count"]} Approach contact-review previews; '
        f'source decision revision {document["source_decisions_revision"]}.'
    )
    print(args.output_root / 'candidate_manifest.json')


if __name__ == '__main__':
    main()
