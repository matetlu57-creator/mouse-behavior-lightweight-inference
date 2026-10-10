"""Persistent human decisions and transactional write-back to annotation samples."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import shutil
import threading
import uuid
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from mouse_behavior.evaluation.exported_dataset import SampleRecord
from mouse_behavior.evaluation.review_transaction import (
    SwapTransaction, safe_path, sha, tree_hashes, write_json,
)
from mouse_behavior.evaluation.segment_export import export_segment
from mouse_behavior.evaluation.review_lock import ReviewOwnershipLock


LABELS = ('Approach', 'Attack', 'Chasing', 'Dispersal', 'Grooming', 'Rearing',
          'Group locomotion', 'Huddling', 'Isolation', 'Locomotion', 'Snout-head_contact',
          'Snout-rear_contact', 'Social clustering', 'Static', 'Together')
PAYLOADS = ('clip.mp4', 'tracks.json', 'annotation.json', 'metadata.json')
DECISIONS = ('pending', 'approved', 'rejected', 'uncertain')


class ReviewConflict(RuntimeError):
    pass


def source_key(sample_id):
    return hashlib.sha256(sample_id.encode('utf-8')).hexdigest()[:16]


def review_catalog(rows):
    cards = []
    for row in rows:
        raw = row['record']
        for window in row.get('selected_windows') or [None]:
            identity = {'sample_id': raw['sample_id'], 'window': None if window is None else {
                k: window[k] for k in ('behavior', 'mouse_ids', 'start_frame', 'end_frame')}}
            review_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
            label = window['behavior'] if window else raw['behavior']
            label = 'Locomotion' if label in {'Walking', 'Running'} else label
            cards.append({'review_id': review_id, 'source_key': source_key(raw['sample_id']),
                          'sample_id': raw['sample_id'], 'behavior': label,
                          'mouse_ids': list(window['mouse_ids'] if window else raw['mouse_ids']),
                          'start_frame': window['start_frame'] if window else 0,
                          'end_frame': window['end_frame'] if window else raw['frame_count'],
                          'fps': raw['fps'], 'frame_count': raw['frame_count'],
                          'export_path': window.get('export_path') if window else None,
                          'poster_path': window.get('poster_path') if window else None,
                          'candidate_status': window['status'] if window else 'no_qualified_candidate',
                          'candidate_reason': window['reason'] if window else row['reason']})
    if len({c['review_id'] for c in cards}) != len(cards):
        raise ValueError('duplicate review card identity')
    return cards


def huddling_recheck_catalog(base_cards, decisions):
    """Return Social clustering rejections whose note explicitly requests huddling review."""
    cards = []
    for review_id, base in base_cards.items():
        decision = decisions.get(review_id, {})
        note = decision.get('note', '')
        if (base['behavior'] != 'Social clustering'
                or decision.get('decision') != 'rejected'
                or not note.strip().startswith('扎堆')):
            continue
        updated_at = decision.get('updated_at')
        if not isinstance(updated_at, str) or not updated_at:
            raise ValueError(f'missing rejection timestamp for huddling re-review: {review_id}')
        card = copy.deepcopy(base)
        card.update(
            behavior='Huddling',
            candidate_status='social_clustering_huddling_recheck',
            candidate_reason=(
                f'原社会聚集不通过备注：{note}。此卡仅因备注提到扎堆而重新列入复核；'
                '请核实实际行为、目标视觉 ID 和起止帧，不会自动通过。'
            ),
            review_from_behavior='Social clustering',
            review_origin_behavior='Social clustering',
            review_origin_decision_updated_at=updated_at,
            review_default_behavior='Huddling',
            review_display_behavior='扎堆待复核',
            review_suggestion_label=f'原备注：{note}',
            review_transfer_pending_label='待扎堆复核（原社会聚集未通过）',
        )
        cards.append(card)
    return cards


class ReviewStore:
    def __init__(self, package_root, confirmed_root, source_root):
        self.ownership_locks = []
        self.closed = False
        try:
            self._initialize(package_root, confirmed_root, source_root)
        except BaseException:
            self.close()
            raise

    def close(self):
        self.closed = True
        for lock in reversed(self.ownership_locks):
            lock.close()

    def _initialize(self, package_root, confirmed_root, source_root):
        self.package = Path(package_root).resolve()
        self.root = Path(confirmed_root).resolve()
        self.source_root = Path(source_root).resolve()
        for a, b in ((self.root, self.source_root), (self.root, self.package),
                     (self.package, self.source_root)):
            if a.is_relative_to(b) or b.is_relative_to(a):
                raise ValueError('package, source and confirmed roots must be separate')
        safe_path(self.root, [self.root])
        safe_path(self.source_root, [self.source_root])
        if not self.source_root.is_dir():
            raise FileNotFoundError('original dataset root is missing')
        self.root.mkdir(parents=True, exist_ok=True)
        for path in (self.source_root/'.manual-review.lock', self.root/'.manual-review.lock'):
            safe_path(path, [self.source_root, self.root])
            self.ownership_locks.append(ReviewOwnershipLock(path))
        safe_path(self.root/'dataset', [self.root])
        (self.root/'dataset').mkdir(exist_ok=True)
        manifest = self.package / 'combined_review_manifest.json'
        self.manifest_sha = sha(manifest)
        self.rows = json.loads(manifest.read_text(encoding='utf-8'))
        self.cards = {c['review_id']: c for c in review_catalog(self.rows)}
        self.sources = {source_key(r['record']['sample_id']): r for r in self.rows}
        for row in self.rows:
            record = row['record']
            sample = safe_path(record['sample_dir'], [self.source_root])
            if len(sample.relative_to(self.source_root).parts) != 2:
                raise ValueError('expected behavior/sample source layout')
            for name, field in (('clip.mp4', 'clip_path'), ('tracks.json', 'tracks_path'),
                                ('annotation.json', 'annotation_path'), ('metadata.json', 'metadata_path')):
                if Path(record[field]).resolve() != sample / name:
                    raise ValueError('unsupported source payload filenames')
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.transaction = SwapTransaction(self.root, self.source_root)
        self.recovery_error = None
        try:
            self.transaction.recover()
        except (OSError, ValueError, RuntimeError) as error:
            self.recovery_error = str(error)
        self.state_path = self.root / 'decisions.json'
        if not self.state_path.exists() and not self.recovery_error:
            write_json(self.state_path, {'schema_version': 1, 'manifest_sha256': self.manifest_sha,
                       'revision': 0, 'decisions': {}, 'sources': {}, 'events': [], 'requests': {}})
            write_json(self.root / 'dataset_manifest.json', [])
        if self.state_path.exists():
            if self._state()['manifest_sha256'] != self.manifest_sha:
                raise ReviewConflict('review package changed; use a new confirmed directory for this batch')
        self.candidate_sha = None
        if self.state_path.exists() and self._state().get('candidate_updates'):
            pointer = self._state()['candidate_updates']
            if (not isinstance(pointer, dict) or set(pointer) != {'path', 'sha256'}
                    or pointer['path'] != f"candidate_updates/{pointer['sha256']}.json"):
                raise ReviewConflict('invalid candidate revision pointer')
            path = safe_path(self.root / pointer['path'], [self.root])
            if sha(path) != pointer['sha256']:
                raise ReviewConflict('candidate revision changed')
            self.cards = self._updated_cards(json.loads(path.read_text(encoding='utf-8')))
            self.candidate_sha = pointer['sha256']
        self.base_cards = dict(self.cards)
        self.supplemental_sha = None
        self.supplemental_cards = {}
        if self.state_path.exists() and self._state().get('supplemental_review_candidates'):
            pointer = self._state()['supplemental_review_candidates']
            digest = pointer.get('sha256') if isinstance(pointer, dict) else None
            if (not isinstance(pointer, dict) or set(pointer) != {'path', 'sha256'}
                    or not isinstance(digest, str) or len(digest) != 64
                    or any(value not in '0123456789abcdef' for value in digest)
                    or pointer['path'] != f"candidate_updates/supplemental-{digest}.json"):
                raise ReviewConflict('invalid supplemental review candidate pointer')
            path = safe_path(self.root / pointer['path'], [self.root])
            if sha(path) != pointer['sha256']:
                raise ReviewConflict('supplemental review candidates changed')
            self.supplemental_cards = self._supplemental_cards(
                json.loads(path.read_text(encoding='utf-8')))
            self.cards.update(self.supplemental_cards)
            self.supplemental_sha = pointer['sha256']

    def _updated_cards(self, document):
        """Layer checked defaults over stable frozen identities, not decisions."""
        if (not isinstance(document, dict) or document.get('schema_version') != 1
                or document.get('base_manifest_sha256') != self.manifest_sha
                or not isinstance(document.get('updates'), list)):
            raise ValueError('invalid candidate revision schema or base manifest')
        cards = {c['review_id']: c for c in review_catalog(self.rows)}
        seen = set()
        fields = {'review_id', 'start_frame', 'end_frame', 'export_path', 'poster_path',
                  'candidate_status', 'candidate_reason', 'message', 'output_sha256'}
        for update in document['updates']:
            if not isinstance(update, dict) or set(update) != fields:
                raise ValueError('invalid candidate update fields')
            rid = update['review_id']
            if rid not in cards or rid in seen or cards[rid]['behavior'] != 'Approach':
                raise ValueError('unknown, duplicate or non-Approach candidate update')
            seen.add(rid)
            card = cards[rid]
            a, b = update['start_frame'], update['end_frame']
            if type(a) is not int or type(b) is not int or not 0 <= a < b <= card['frame_count']:
                raise ValueError('invalid revised candidate frame range')
            for field in ('candidate_status', 'candidate_reason', 'message'):
                if not isinstance(update[field], str) or len(update[field]) > 2000:
                    raise ValueError('invalid candidate description')
            export = update['export_path']
            if not isinstance(update['output_sha256'], dict):
                raise ValueError('invalid candidate output hashes')
            if export is not None:
                if (not isinstance(export, str) or Path(export).is_absolute()
                        or Path(export).parts[:2] != ('dataset', 'Approach')):
                    raise ValueError('candidate media must be inside dataset/Approach')
                target = safe_path(self.package / export, [self.package / 'dataset' / 'Approach'])
                if (not all(name in update['output_sha256'] for name in (*PAYLOADS, 'target_poster.jpg'))
                        or tree_hashes(target) != update['output_sha256']):
                    raise ReviewConflict('revised candidate payload changed')
                if update['poster_path'] != (Path(export) / 'target_poster.jpg').as_posix():
                    raise ValueError('invalid candidate poster path')
                annotation = json.loads((target/'annotation.json').read_text(encoding='utf-8'))
                tracks = json.loads((target/'tracks.json').read_text(encoding='utf-8'))
                metadata = json.loads((target/'metadata.json').read_text(encoding='utf-8'))
                provenance = metadata.get('provenance', {})
                if (not isinstance(tracks, list) or len(tracks) != b-a
                        or [f.get('frame') for f in tracks] != list(range(b-a))
                        or annotation.get('mouse_ids') != card['mouse_ids']
                        or annotation.get('behavior') != 'Approach'
                        or annotation.get('frame_range') != {'start': 0, 'end': b-a-1}
                        or metadata.get('clip', {}).get('frame_count') != b-a
                        or provenance.get('source_sample_id') != card['sample_id']
                        or provenance.get('offsets', {}).get('source_clip_local_half_open') != {'start': a, 'end': b}):
                    raise ValueError('candidate export frames or IDs do not match')
            elif update['poster_path'] is not None or update['output_sha256']:
                raise ValueError('missing candidate cannot have media hashes')
            card.update({k: copy.deepcopy(update[k]) for k in fields
                         if k not in {'review_id', 'message', 'output_sha256'}})
            card['candidate_update_message'] = update['message']
        return cards

    def publish_candidate_updates(self, document, *, expected_revision):
        """Commit immutable candidate defaults without touching either dataset.

        The existing state transaction makes a crash recover the old pointer.
        Media is prepared first; unreferenced prepared media is not published.
        """
        with self.lock:
            if self.closed:
                raise RuntimeError('review store ownership was released')
            self.transaction.recover()
            before = self.state_path.read_bytes()
            state = json.loads(before)
            if type(expected_revision) is not int or state['revision'] != expected_revision:
                raise ReviewConflict('review state changed while preparing candidates')
            cards = self._updated_cards(document)
            content = (json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode('utf-8')
            digest = hashlib.sha256(content).hexdigest()
            path = safe_path(self.root / 'candidate_updates' / (digest+'.json'), [self.root])
            if path.exists():
                if path.read_bytes() != content:
                    raise ReviewConflict('candidate revision receipt changed')
            else:
                write_json(path, document)
            assert sha(path) == digest
            state['candidate_updates'] = {'path': path.relative_to(self.root).as_posix(), 'sha256': digest}
            state['revision'] += 1
            state.setdefault('candidate_update_history', []).append({
                'sha256': digest, 'revision': state['revision'],
                'updated_at': datetime.now(timezone.utc).isoformat(),
                'review_ids': [u['review_id'] for u in document['updates']]})
            txid = uuid.uuid4().hex
            stage = self.root / f'.review-stage-{txid}-decisions.json'
            write_json(stage, state)
            self.transaction.apply(txid, [(self.state_path, stage,
                {'.': hashlib.sha256(before).hexdigest()}, tree_hashes(stage))])
            self.cards, self.candidate_sha = cards, digest
            return self.snapshot()

    def _supplemental_cards(self, document):
        """Validate a human-review-only card batch layered over the frozen manifest."""
        batch_id = document.get('batch_id') if isinstance(document, dict) else None
        if (not isinstance(document, dict) or document.get('schema_version') != 1
                or document.get('base_manifest_sha256') != self.manifest_sha
                or not isinstance(batch_id, str)
                or re.fullmatch(
                    r'(?:rearing_from_rejected_locomotion|approach_no_contact_from_rejected)_v[1-9][0-9]*',
                    batch_id,
                ) is None
                or not isinstance(document.get('cards'), list) or not document['cards']):
            raise ValueError('invalid supplemental review candidate schema or base manifest')
        cards = {}
        fields = {'review_id', 'source_key', 'sample_id', 'behavior', 'mouse_ids',
                  'start_frame', 'end_frame', 'fps', 'frame_count', 'export_path',
                  'poster_path', 'candidate_status', 'candidate_reason', 'output_sha256'}
        for item in document['cards']:
            if not isinstance(item, dict) or set(item) != fields:
                raise ValueError('invalid supplemental review candidate fields')
            key = item['source_key']
            if key not in self.sources or item['behavior'] not in {'Rearing', 'Approach'}:
                raise ValueError('supplemental cards must target a known source as Rearing or Approach')
            raw = self.sources[key]['record']
            if item['sample_id'] != raw['sample_id'] or key != source_key(item['sample_id']):
                raise ValueError('supplemental source identity mismatch')
            if item['review_id'] in self.base_cards or item['review_id'] in cards:
                raise ValueError('supplemental review ID collides with a frozen card or duplicate')
            prior_card = self.supplemental_cards.get(item['review_id'])
            if prior_card is not None:
                prior_decision = self._state().get('decisions', {}).get(item['review_id'], {})
                identity_fields = ('source_key', 'sample_id', 'behavior', 'mouse_ids',
                                   'start_frame', 'end_frame', 'fps', 'frame_count')
                # The content-derived ID stays stable for an unchanged pending interval;
                # the supplemental digest still rejects writes from a stale review page.
                if any(item[field] != prior_card[field] for field in identity_fields):
                    raise ValueError('a supplemental review ID can only be reused for the same identity and range')
                prior_status = prior_decision.get('decision')
                if prior_status not in (None, 'pending', 'approved', 'rejected', 'uncertain'):
                    raise ValueError('supplemental review ID has an unsupported prior decision')
            a, b = item['start_frame'], item['end_frame']
            if (type(a) is not int or type(b) is not int
                    or not 0 <= a < b <= raw['frame_count']):
                raise ValueError('invalid supplemental candidate frame range')
            if (not isinstance(item['mouse_ids'], list) or not item['mouse_ids']
                    or any(type(value) is not int or value < 0 for value in item['mouse_ids'])
                    or item['mouse_ids'] != sorted(set(item['mouse_ids']))):
                raise ValueError('invalid supplemental candidate visual IDs')
            if (type(item['frame_count']) is not int or item['frame_count'] != raw['frame_count']
                    or not isinstance(item['fps'], (int, float))
                    or not math.isfinite(item['fps']) or item['fps'] <= 0
                    or item['fps'] != raw['fps']):
                raise ValueError('supplemental source timeline differs from frozen manifest')
            expected_status = {
                'Rearing': {'rejected_locomotion_rearing_review_candidate'},
                'Approach': {
                    'approach_no_contact_recut_review_candidate',
                    'approach_no_contact_recrop_v1_review_candidate',
                    'approach_no_contact_recrop_v2_review_candidate',
                },
            }[item['behavior']]
            if (item['candidate_status'] not in expected_status
                    or not isinstance(item['candidate_reason'], str)
                    or not item['candidate_reason'].strip()
                    or len(item['candidate_reason']) > 2000):
                raise ValueError('invalid supplemental candidate description')
            export = item['export_path']
            poster = item['poster_path']
            if (not isinstance(export, str) or Path(export).is_absolute()
                    or Path(export).parts[:2] != ('dataset', item['behavior'])
                    or not isinstance(poster, str)
                    or Path(poster).as_posix() != (Path(export) / 'target_poster.jpg').as_posix()):
                raise ValueError('supplemental candidate media must be under dataset/Rearing')
            target = safe_path(self.package / export, [self.package / 'dataset' / item['behavior']])
            expected_files = {*PAYLOADS, 'target_poster.jpg'}
            hashes = item['output_sha256']
            if (not isinstance(hashes, dict) or set(hashes) != expected_files
                    or tree_hashes(target) != hashes):
                raise ReviewConflict('supplemental candidate payload changed')
            annotation = json.loads((target/'annotation.json').read_text(encoding='utf-8'))
            tracks = json.loads((target/'tracks.json').read_text(encoding='utf-8'))
            metadata = json.loads((target/'metadata.json').read_text(encoding='utf-8'))
            provenance = metadata.get('provenance', {})
            if item['candidate_status'] in {
                    'approach_no_contact_recrop_v1_review_candidate',
                    'approach_no_contact_recrop_v2_review_candidate'}:
                source_metadata_path = self._source_tracks_path(key).parent / 'metadata.json'
                source_metadata = json.loads(source_metadata_path.read_text(encoding='utf-8'))
                source_start = source_metadata.get('source', {}).get('start_frame')
                crop_source = metadata.get('source', {})
                offsets = provenance.get('offsets', {}).get('source_clip_local_half_open')
                range_matches_source = (
                    offsets in ({'start': 0, 'end': b-a}, {'start': a, 'end': b})
                    and type(source_start) is int
                    and crop_source.get('start_frame') == source_start + a
                    and crop_source.get('end_frame') == source_start + b - 1
                )
            else:
                range_matches_source = (
                    provenance.get('offsets', {}).get('source_clip_local_half_open')
                    == {'start': a, 'end': b}
                )
            if (not isinstance(tracks, list) or len(tracks) != b-a
                    or [frame.get('frame') for frame in tracks] != list(range(b-a))
                    or annotation.get('mouse_ids') != item['mouse_ids']
                    or annotation.get('behavior') != item['behavior']
                    or annotation.get('frame_range') != {'start': 0, 'end': b-a-1}
                    or metadata.get('clip', {}).get('frame_count') != b-a
                    or provenance.get('source_sample_id') != raw['sample_id']
                    or not range_matches_source):
                raise ValueError('supplemental crop does not match its source window')
            target_frames = [
                any(int(d.get('track_id', -1)) in item['mouse_ids']
                    for d in frame.get('detections', [])) for frame in tracks
            ]
            crop_ids = {int(d['track_id']) for frame in tracks for d in frame.get('detections', [])}
            if not set(item['mouse_ids']) <= crop_ids:
                raise ValueError('supplemental visual ID is absent from candidate crop')
            if sum(target_frames) < max(1, round(len(tracks) * 0.75)):
                raise ValueError('supplemental visual ID is too sparse in candidate crop')
            identity = {'sample_id': raw['sample_id'], 'window': {
                name: item[name] for name in ('behavior', 'mouse_ids', 'start_frame', 'end_frame')}}
            if item['behavior'] == 'Approach':
                identity['candidate_family'] = 'approach_no_contact'
            expected_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
            if item['review_id'] != expected_id:
                raise ValueError('supplemental review ID does not match its source window')
            cards[item['review_id']] = {
                key: copy.deepcopy(item[key]) for key in fields if key != 'output_sha256'
            }
        return cards

    def publish_supplemental_candidates(self, document, *, expected_revision):
        """Publish immutable review cards without touching prior decisions or source data."""
        with self.lock:
            if self.closed:
                raise RuntimeError('review store ownership was released')
            self.transaction.recover()
            before = self.state_path.read_bytes()
            state = json.loads(before)
            if type(expected_revision) is not int or state['revision'] != expected_revision:
                raise ReviewConflict('review state changed while preparing supplemental candidates')
            cards = self._supplemental_cards(document)
            pinned_revision = document.get('source_decisions_revision')
            pinned_sha256 = document.get('source_decisions_sha256')
            if (type(pinned_revision) is not int or pinned_revision != state['revision']
                    or not isinstance(pinned_sha256, str)
                    or hashlib.sha256(before).hexdigest() != pinned_sha256):
                raise ReviewConflict('source decisions changed since supplemental candidates were prepared')
            prior_pointer = state.get('supplemental_review_candidates')
            prior_ids = set(self.supplemental_cards)
            new_ids = set(cards)
            final_ids = {
                review_id for review_id in prior_ids
                if state.get('decisions', {}).get(review_id, {}).get('decision')
                in {'approved', 'rejected', 'uncertain'}
            }
            if final_ids - new_ids:
                raise ReviewConflict('replacement must retain every reviewed supplemental card')
            for review_id in final_ids:
                prior = self.supplemental_cards[review_id]
                updated = cards[review_id]
                identity_fields = ('source_key', 'sample_id', 'behavior', 'mouse_ids',
                                   'start_frame', 'end_frame', 'fps', 'frame_count')
                if any(prior[field] != updated[field] for field in identity_fields):
                    raise ReviewConflict('replacement cannot change a reviewed supplemental card')
            superseded_ids = sorted(prior_ids - new_ids)
            superseded_pending = {}
            for review_id in superseded_ids:
                prior_decision = state.get('decisions', {}).get(review_id)
                if prior_decision and prior_decision.get('decision') in {'approved', 'rejected', 'uncertain'}:
                    raise ReviewConflict('replacement cannot remove a reviewed supplemental card')
                if prior_decision:
                    superseded_pending[review_id] = state['decisions'].pop(review_id)
            content = (json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode('utf-8')
            digest = hashlib.sha256(content).hexdigest()
            path = safe_path(self.root / 'candidate_updates' / f'supplemental-{digest}.json', [self.root])
            if path.exists():
                if path.read_bytes() != content:
                    raise ReviewConflict('supplemental candidate receipt changed')
            else:
                write_json(path, document)
            state['supplemental_review_candidates'] = {
                'path': path.relative_to(self.root).as_posix(), 'sha256': digest}
            state['revision'] += 1
            state.setdefault('supplemental_candidate_history', []).append({
                'batch_id': document['batch_id'], 'sha256': digest,
                'revision': state['revision'], 'review_ids': list(cards),
                'previous_sha256': prior_pointer.get('sha256') if prior_pointer else None,
                'superseded_review_ids': superseded_ids,
                'superseded_pending_decisions': superseded_pending,
                'retained_pending_ids': sorted((prior_ids & new_ids) - final_ids),
                'preserved_reviewed_ids': sorted(final_ids)})
            txid = uuid.uuid4().hex
            stage = self.root / f'.review-stage-{txid}-decisions.json'
            write_json(stage, state)
            self.transaction.apply(txid, [(self.state_path, stage,
                {'.': hashlib.sha256(before).hexdigest()}, tree_hashes(stage))])
            self.supplemental_cards = cards
            self.cards = {**self.base_cards, **cards}
            self.supplemental_sha = digest
            return self.snapshot()

    def _state(self):
        return json.loads(self.state_path.read_text(encoding='utf-8'))

    def snapshot(self):
        with self.lock:
            state = self._state() if self.state_path.exists() else {'revision': 0, 'decisions': {}}
            counts = Counter(d['decision'] for d in state['decisions'].values())
            counts['pending'] += len(self.cards) - len(state['decisions'])
            return {'revision': state['revision'], 'decisions': state['decisions'],
                    'counts': dict(counts), 'total': len(self.cards),
                    'confirmed_dataset': str(self.root / 'dataset'),
                    'source_dataset': str(self.source_root), 'recovery_error': self.recovery_error,
                    'candidate_sha256': self.candidate_sha,
                    'supplemental_sha256': self.supplemental_sha}

    def _backup(self, key):
        row = self.sources[key]
        source = safe_path(row['record']['sample_dir'], [self.source_root])
        directory = self.root / 'backups' / key
        receipt_path = directory / 'receipt.json'
        backup = directory / 'sample'
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
            if tree_hashes(backup) != receipt['sha256']:
                raise ReviewConflict('immutable source backup changed')
            return backup
        if directory.exists():
            raise ReviewConflict('incomplete backup; inspect it before retrying')
        for field, fp in row['source_fingerprints'].items():
            stat = Path(row['record'][field]).stat()
            if fp != {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}:
                raise ReviewConflict('source changed after screening; re-review the source')
        before = tree_hashes(source)
        temporary = self.root / 'backups' / ('.' + key + '-' + uuid.uuid4().hex)
        temporary.mkdir(parents=True)
        try:
            shutil.copytree(source, temporary / 'sample')
            if tree_hashes(temporary / 'sample') != before or tree_hashes(source) != before:
                raise ReviewConflict('source changed while taking backup')
            write_json(temporary / 'receipt.json', {'source': str(source), 'sha256': before,
                        'sample_id': row['record']['sample_id'], 'manifest_sha256': self.manifest_sha})
            temporary.rename(directory)
        except BaseException:
            safe_path(temporary, [self.root])
            if temporary.exists():
                shutil.rmtree(temporary)
            raise
        return backup

    def _source_tracks_path(self, key):
        backup = self.root / 'backups' / key / 'sample'
        if backup.exists():
            return safe_path(backup / 'tracks.json', [self.root])
        return safe_path(Path(self.sources[key]['record']['tracks_path']), [self.source_root])

    def source_track_boxes(self, key):
        if key not in self.sources:
            raise ValueError('unknown source')
        with self.lock:
            record = self.sources[key]['record']
            tracks = json.loads(self._source_tracks_path(key).read_text(encoding='utf-8'))
            expected = int(record['frame_count'])
            if not isinstance(tracks, list) or len(tracks) != expected:
                raise ValueError('source tracks do not match the frozen video frame count')
            result = []
            for frame_index, frame in enumerate(tracks):
                if not isinstance(frame, dict) or frame.get('frame') != frame_index:
                    raise ValueError('source tracks are not aligned to the original video timeline')
                detections = []
                for detection in frame.get('detections', []):
                    if not isinstance(detection, dict):
                        continue
                    try:
                        track_id = int(detection['track_id'])
                        box = [float(value) for value in detection['box']]
                    except (KeyError, TypeError, ValueError):
                        continue
                    if (track_id < 0 or len(box) != 4
                            or not all(math.isfinite(value) for value in box)
                            or box[2] <= box[0] or box[3] <= box[1]):
                        continue
                    detections.append({'track_id': track_id, 'box': box})
                result.append({'frame': frame_index, 'detections': detections})
            return result

    def source_details(self, key):
        if key not in self.sources:
            raise ValueError('unknown source')
        with self.lock:
            # Before first approval the frozen tracks are still at their original location.
            tracks = json.loads(self._source_tracks_path(key).read_text(encoding='utf-8'))
            return {'mouse_ids': sorted({int(d['track_id']) for frame in tracks for d in frame['detections']})}

    def _validate(self, payload, card, reference):
        label, ids = payload['behavior'], payload['mouse_ids']
        a, b = payload['start_frame'], payload['end_frame']
        if label not in LABELS or type(a) is not int or type(b) is not int:
            raise ValueError('choose a valid behavior and integer frame bounds')
        if not 0 <= a < b <= card['frame_count']:
            raise ValueError('frame bounds must satisfy 0 <= start < end <= source frames')
        if label == 'Together' and b-a < float(card['fps']):
            raise ValueError('Together（一起）片段必须至少1秒，请延长裁剪范围')
        if not isinstance(ids, list) or not ids or any(type(i) is not int or i < 0 for i in ids):
            raise ValueError('choose actual nonnegative visual track IDs')
        if len(set(ids)) != len(ids):
            raise ValueError('target IDs must be unique')
        single = {'Static', 'Grooming', 'Rearing', 'Locomotion', 'Isolation'}
        pairs = {'Approach', 'Attack', 'Chasing', 'Together', 'Snout-head_contact', 'Snout-rear_contact'}
        if (label in single and len(ids) != 1) or (label in pairs and len(ids) != 2):
            raise ValueError('individual behavior needs one ID; pair behavior needs exactly two IDs')
        if label not in single | pairs and len(ids) < (3 if label == 'Social clustering' else 2):
            raise ValueError('insufficient group target IDs')
        tracks = json.loads((reference / 'tracks.json').read_text(encoding='utf-8'))
        sliced = [f for f in tracks if a <= f['frame'] < b]
        if sorted(f['frame'] for f in sliced) != list(range(a, b)):
            raise ValueError('source tracks do not cover the selected frame range')
        observed = {int(d['track_id']) for frame in sliced for d in frame['detections']}
        if not set(ids) <= observed:
            raise ValueError('selected IDs are absent from the selected original frames')

    def _check_managed(self, state, key, backup):
        source = self.sources[key]['record']['sample_dir']
        managed = state['sources'].get(key, {}).get('managed')
        if managed is None:
            if tree_hashes(Path(source)) != tree_hashes(backup):
                raise ReviewConflict('source changed since backup; refusing overwrite')
            return {source: tree_hashes(backup)}
        for path, expected in managed.items():
            safe_path(path, [self.source_root])
            if not Path(path).exists() or tree_hashes(path) != expected:
                raise ReviewConflict('previously synchronized source changed externally')
        return managed

    def save(self, payload):
        with self.lock:
            if self.closed:
                raise RuntimeError('review store ownership was released')
            self.transaction.recover()
            self.recovery_error = None
            state_bytes = self.state_path.read_bytes()
            state = json.loads(state_bytes)
            state_hashes = {'.': hashlib.sha256(state_bytes).hexdigest()}
            manifest_hashes = tree_hashes(self.root / 'dataset_manifest.json')
            request_id = payload.get('request_id')
            if not isinstance(request_id, str) or not 8 <= len(request_id) <= 100:
                raise ValueError('request_id is required')
            digest = hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()
            if request_id in state['requests']:
                if state['requests'][request_id]['sha256'] != digest:
                    raise ReviewConflict('request identifier reused with different content')
                return self.snapshot()
            if type(payload.get('revision')) is not int or payload['revision'] != state['revision']:
                raise ReviewConflict('another review changed this dataset; refresh and retry')
            if self.candidate_sha and payload.get('candidate_sha256') != self.candidate_sha:
                raise ReviewConflict('candidate defaults changed; reload the review page before saving')
            card = self.cards.get(payload.get('review_id'))
            if not card or payload.get('decision') not in DECISIONS:
                raise ValueError('unknown review card or decision')
            if (payload.get('review_id') in self.supplemental_cards
                    and payload.get('supplemental_sha256') != self.supplemental_sha):
                raise ReviewConflict('supplemental candidates changed; reload the review page before saving')
            note = payload.get('note', '')
            if not isinstance(note, str) or len(note) > 2000:
                raise ValueError('note must be at most 2000 characters')
            fields = ('decision', 'behavior', 'mouse_ids', 'start_frame', 'end_frame')
            if any(k not in payload for k in fields):
                raise ValueError('missing review fields')
            if (not isinstance(payload['behavior'], str) or len(payload['behavior']) > 80
                    or not isinstance(payload['mouse_ids'], list)
                    or any(type(i) is not int or i < 0 for i in payload['mouse_ids'])
                    or type(payload['start_frame']) is not int or type(payload['end_frame']) is not int):
                raise ValueError('invalid review field types')
            decision = {k: copy.deepcopy(payload[k]) for k in fields}
            decision.update(note=note, source_key=card['source_key'],
                            updated_at=datetime.now(timezone.utc).isoformat())
            prior = state['decisions'].get(card['review_id'], {})
            meaningful = (*fields, 'note')
            if prior and all(prior.get(k) == decision.get(k) for k in meaningful):
                return self.snapshot()
            key = card['source_key']
            needs_sync = decision['decision'] == 'approved' or prior.get('decision') == 'approved'
            backup = self._backup(key) if needs_sync else None
            if decision['decision'] == 'approved':
                self._validate(payload, card, backup)
            updated = copy.deepcopy(state)
            updated['decisions'][card['review_id']] = decision
            approved = [(rid, d) for rid, d in updated['decisions'].items()
                        if d['source_key'] == key and d['decision'] == 'approved']
            def signature(d):
                return d['behavior'], tuple(sorted(d['mouse_ids'])), d['start_frame'], d['end_frame']
            if len({signature(d) for _, d in approved}) != len(approved):
                raise ValueError('the same source/behavior/IDs/frame range is already approved on another card')
            txid = uuid.uuid4().hex
            swaps, created = [], []
            try:
                if needs_sync:
                    self._prepare_source(updated, state, key, backup, approved, txid, swaps, created)
                updated['revision'] += 1
                updated['requests'][request_id] = {'sha256': digest, 'revision': updated['revision']}
                updated['events'].append({'review_id': card['review_id'], 'request_id': request_id,
                                          'before': prior, 'after': decision, 'revision': updated['revision']})
                manifest = [dict(review_id=rid, **d) for rid, d in updated['decisions'].items()
                            if d['decision'] == 'approved']
                for target, value, expected in ((self.root / 'dataset_manifest.json', manifest, manifest_hashes),
                                                 (self.state_path, updated, state_hashes)):
                    stage = target.with_name(f'.review-stage-{txid}-{target.name}')
                    write_json(stage, value)
                    created.append(stage)
                    swaps.append((target, stage, expected, tree_hashes(stage)))
                self.transaction.apply(txid, swaps)
            except BaseException:
                if not self.transaction.journal.exists():
                    for path in created:
                        safe_path(path, [self.root, self.source_root])
                        if path.exists():
                            shutil.rmtree(path) if path.is_dir() else path.unlink()
                else:
                    self.recovery_error = '文件同步尚未恢复，请关闭占用文件的程序后重试。'
                raise
            return self.snapshot()

    def _prepare_source(self, updated, state, key, backup, approved, txid, swaps, created):
        raw = self.sources[key]['record']
        original = Path(raw['sample_dir'])
        managed = self._check_managed(state, key, backup)
        previous = [(rid, d) for rid, d in state['decisions'].items()
                    if d['source_key'] == key and d['decision'] == 'approved']
        for rid, d in previous:
            accepted = safe_path(d['dataset_path'], [self.root / 'dataset'])
            if not accepted.exists() or tree_hashes(accepted) != d['dataset_sha256']:
                raise ReviewConflict('confirmed sample changed externally')
        primary = state['sources'].get(key, {}).get('primary_review_id')
        if primary not in {rid for rid, _ in approved}:
            primary = next((rid for rid, _ in sorted(approved)), None)
        installed, accepted_targets = {}, set()
        previous_targets = {d['dataset_path'] for _, d in previous}
        for rid, d in sorted(approved):
            target = self.root / 'dataset' / d['behavior'] / rid
            safe_path(target, [self.root])
            if target.exists() and str(target) not in previous_targets:
                raise ReviewConflict('confirmed destination is occupied by an unmanaged sample')
            target.parent.mkdir(parents=True, exist_ok=True)
            stage = target.with_name(f'.review-stage-{txid}-{rid}')
            safe_path(stage, [self.root])
            created.append(stage)
            record = SampleRecord(**{**raw, 'mouse_ids': tuple(raw['mouse_ids'])})
            record = replace(record, sample_dir=str(backup), clip_path=str(backup/'clip.mp4'),
                             tracks_path=str(backup/'tracks.json'), annotation_path=str(backup/'annotation.json'),
                             metadata_path=str(backup/'metadata.json'))
            receipt = export_segment(record, stage, start_frame=d['start_frame'], end_frame=d['end_frame'],
                                     behavior=d['behavior'], mouse_ids=tuple(d['mouse_ids']),
                                     provenance={'candidate': {'status': 'human_confirmed'},
                                                 'manual_review': {'review_id': rid, 'decision': d},
                                                 'screen_manifest_sha256': self.manifest_sha})
            annotation = json.loads((stage/'annotation.json').read_text(encoding='utf-8'))
            annotation.update(confidence='human_confirmed', requires_review=False,
                              screening_status='human_confirmed', manual_review_id=rid)
            write_json(stage/'annotation.json', annotation)
            receipt['output_sha256']['annotation.json'] = sha(stage/'annotation.json')
            receipt['output_dir'] = str(target)
            write_json(stage/'review_receipt.json', receipt)
            d['dataset_path'], d['dataset_sha256'] = str(target), tree_hashes(stage)
            expected = next((d['dataset_sha256'] for _, d in previous if d['dataset_path'] == str(target)), None)
            swaps.append((target, stage, expected, d['dataset_sha256']))
            accepted_targets.add(str(target))
            basename = original.name if rid == primary else f'{original.name}__confirmed__{rid}'
            source_target = self.source_root / d['behavior'] / basename
            if source_target.exists() and str(source_target) not in managed:
                source_target = source_target.with_name(basename + '__source_' + key)
            if source_target.exists() and str(source_target) not in managed:
                raise ReviewConflict('original destination collides with another sample')
            safe_path(source_target, [self.source_root])
            source_target.parent.mkdir(parents=True, exist_ok=True)
            source_stage = source_target.with_name(f'.review-stage-{txid}-{rid}')
            created.append(source_stage)
            # Authorization covers four payloads. Preserve any other source artifacts.
            shutil.copytree(backup, source_stage)
            for name in PAYLOADS:
                shutil.copy2(stage/name, source_stage/name)
            d['original_synced_path'] = str(source_target)
            installed[str(source_target)] = tree_hashes(source_stage)
            swaps.append((source_target, source_stage, managed.get(str(source_target)), installed[str(source_target)]))
        if not approved:
            source_stage = original.with_name(f'.review-stage-{txid}-{key}')
            created.append(source_stage)
            if original.exists() and str(original) not in managed:
                raise ReviewConflict('original restoration path is occupied')
            shutil.copytree(backup, source_stage)
            installed[str(original)] = tree_hashes(source_stage)
            swaps.append((original, source_stage, managed.get(str(original)), installed[str(original)]))
        for path in managed:
            if path not in installed:
                swaps.append((Path(path), None, managed[path], {}))
        for _, d in previous:
            if d['dataset_path'] not in accepted_targets:
                swaps.append((Path(d['dataset_path']), None, d['dataset_sha256'], {}))
        updated['sources'][key] = {'primary_review_id': primary, 'managed': installed,
                                   'original_path': str(original), 'backup_path': str(backup)}

    def confirmed_video(self, review_id):
        with self.lock:
            d = self._state()['decisions'].get(review_id)
            if not d or d['decision'] != 'approved':
                raise FileNotFoundError('sample has not been approved')
            source = safe_path(Path(d['dataset_path'])/'clip.mp4', [self.root / 'dataset'])
            digest = d['dataset_sha256']['clip.mp4']
            media = self.root / 'media' / (digest + '.mp4')
            safe_path(media, [self.root])
            if not media.exists():
                if sha(source) != digest:
                    raise ReviewConflict('confirmed video changed externally')
                media.parent.mkdir(parents=True, exist_ok=True)
                temporary = media.with_suffix('.tmp')
                shutil.copy2(source, temporary)
                if sha(temporary) != digest:
                    raise ReviewConflict('confirmed video changed during media copy')
                temporary.rename(media)
            return media

    def confirmed_tracks(self, review_id):
        with self.lock:
            d = self._state()['decisions'].get(review_id)
            if not d or d['decision'] != 'approved':
                raise FileNotFoundError('sample has not been approved')
            source = safe_path(Path(d['dataset_path'])/'tracks.json', [self.root/'dataset'])
            digest = d['dataset_sha256'].get('tracks.json')
            if not digest or sha(source) != digest:
                raise ReviewConflict('confirmed tracks changed externally')
            media = self.root/'media'/f'{digest}.json'
            safe_path(media, [self.root])
            if not media.exists() or sha(media) != digest:
                media.parent.mkdir(parents=True, exist_ok=True)
                temporary = media.with_name(f'.{media.name}.{uuid.uuid4().hex}.tmp')
                try:
                    shutil.copy2(source, temporary)
                    if sha(temporary) != digest:
                        raise ReviewConflict('confirmed tracks changed during media copy')
                    temporary.replace(media)
                finally:
                    temporary.unlink(missing_ok=True)
            return media

    def dataset_manifest(self):
        with self.lock:
            return json.loads((self.root/'dataset_manifest.json').read_text(encoding='utf-8'))

    def retry_recovery(self):
        with self.lock:
            if self.closed:
                raise RuntimeError('review store ownership was released')
            try:
                self.transaction.recover()
            except (OSError, RuntimeError, ValueError) as error:
                self.recovery_error = str(error)
                raise
            self.recovery_error = None
            return self.snapshot()
