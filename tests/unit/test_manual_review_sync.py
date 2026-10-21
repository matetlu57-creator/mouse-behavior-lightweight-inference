"""Approval is not successful until both datasets and durable decisions agree."""

from __future__ import annotations

import copy
import json
import shutil
import threading
import uuid
from dataclasses import asdict
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from mouse_behavior.evaluation.exported_dataset import discover_samples
from mouse_behavior.evaluation.manual_review import (
    ReviewConflict, ReviewStore, huddling_recheck_catalog, review_catalog, source_key,
)
from mouse_behavior.evaluation.manual_review_page import render_review_page
from mouse_behavior.evaluation.manual_review_server import create_server
from mouse_behavior.evaluation.review_transaction import tree_hashes, write_json
from tests.unit.test_segment_export import sample as _sample_fixture

sample = _sample_fixture  # Reuse the real MP4/track fixture without importing site-packages tests.


@pytest.fixture
def review(sample, tmp_path):
    source = Path(sample.sample_dir)
    (source/'user_notes.txt').write_text('keep ancillary files', encoding='utf-8')
    package = tmp_path/'package'
    package.mkdir()
    (package/'review_sources').mkdir()
    shutil.copy2(source/'clip.mp4', package/'review_sources'/f'{source_key(sample.sample_id)}.mp4')
    row = {'record': asdict(sample), 'reason': 'test', 'source_fingerprints': {}, 'selected_windows': [
        {'behavior': 'Approach', 'mouse_ids': [1, 2], 'start_frame': 0, 'end_frame': 6,
         'status': 'rule_candidate_needs_review', 'reason': 'test'},
        {'behavior': 'Static', 'mouse_ids': [9], 'start_frame': 4, 'end_frame': 8,
         'status': 'rule_candidate_needs_review', 'reason': 'test'}]}
    for field in ('clip_path', 'tracks_path', 'annotation_path', 'metadata_path'):
        stat = Path(row['record'][field]).stat()
        row['source_fingerprints'][field] = {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
    write_json(package/'combined_review_manifest.json', [row])
    render_review_page(package, [row])
    store = ReviewStore(package, tmp_path/'confirmed', tmp_path/'source')
    try:
        yield store, source, tree_hashes(source)
    finally:
        store.close()


def payload(store, *, index=0, decision='approved', **changes):
    card = list(store.cards.values())[index]
    result = {k: card[k] for k in ('review_id', 'behavior', 'mouse_ids', 'start_frame', 'end_frame')}
    result.update(decision=decision, revision=store.snapshot()['revision'], note='visible motion',
                  request_id=uuid.uuid4().hex)
    result.update(changes)
    return result


def assert_synced(store):
    state = store._state()
    approved = [d for d in state['decisions'].values() if d['decision'] == 'approved']
    assert len(discover_samples(store.root/'dataset')) == len(approved)
    for d in approved:
        a, b = Path(d['dataset_path']), Path(d['original_synced_path'])
        for name in ('clip.mp4', 'tracks.json', 'annotation.json', 'metadata.json'):
            assert (a/name).read_bytes() == (b/name).read_bytes()
        assert (b/'user_notes.txt').read_text() == 'keep ancillary files'
        annotation = json.loads((a/'annotation.json').read_text())
        assert annotation['confidence'] == 'human_confirmed' and annotation['requires_review'] is False
        assert annotation['frame_range']['end'] == d['end_frame']-d['start_frame']-1
    discover_samples(store.source_root)  # Includes relabeling: parent behavior must agree.


def test_together_approval_requires_at_least_one_second(review):
    store, source, _ = review
    card = dict(next(iter(store.cards.values())))
    # Keep the fixture's eight frames while making the boundary exact and deterministic.
    card['fps'] = 8.0
    short = payload(
        store, decision='approved', behavior='Together', mouse_ids=[1, 2],
        start_frame=0, end_frame=7,
    )
    with pytest.raises(ValueError, match='Together.*1秒'):
        store._validate(short, card, source)

    one_second = payload(
        store, decision='approved', behavior='Together', mouse_ids=[1, 2],
        start_frame=0, end_frame=8,
    )
    store._validate(one_second, card, source)


def candidate_revision(store, *, missing=False):
    from dataclasses import replace
    from mouse_behavior.evaluation.exported_dataset import SampleRecord
    from mouse_behavior.evaluation.segment_export import export_segment

    card = list(store.cards.values())[0]
    update = {'review_id': card['review_id'], 'start_frame': 1, 'end_frame': 6,
              'export_path': None, 'poster_path': None, 'output_sha256': {},
              'candidate_status': 'no_qualified_candidate' if missing else 'rule_candidate_needs_review',
              'candidate_reason': 'actual first frame checked', 'message': 'Revised start; saved decisions retained'}
    if not missing:
        record = SampleRecord(**store.rows[0]['record'])
        backup = store.root/'backups'/card['source_key']/'sample'
        if backup.exists():
            record = replace(record, **{field: str(backup/name) for field, name in
                [('clip_path', 'clip.mp4'), ('tracks_path', 'tracks.json'),
                 ('annotation_path', 'annotation.json'), ('metadata_path', 'metadata.json')]})
        target = store.package/'dataset'/'Approach'/'revised'
        export_segment(record, target, behavior='Approach', mouse_ids=[1, 2], start_frame=1, end_frame=6,
                       provenance={'test': 'candidate revision'})
        import cv2
        capture = cv2.VideoCapture(str(target/'clip.mp4'))
        try:
            ok, frame = capture.read()
            assert ok and cv2.imwrite(str(target/'target_poster.jpg'), frame)
        finally:
            capture.release()
        update.update(export_path='dataset/Approach/revised',
                      poster_path='dataset/Approach/revised/target_poster.jpg', output_sha256=tree_hashes(target))
    return {'schema_version': 1, 'base_manifest_sha256': store.manifest_sha, 'updates': [update]}


def supplemental_candidates(store, *, windows=((1, 6),), batch_id='rearing_from_rejected_locomotion_v1'):
    import hashlib
    import cv2

    from dataclasses import replace
    from mouse_behavior.evaluation.exported_dataset import SampleRecord
    from mouse_behavior.evaluation.segment_export import export_segment

    raw = store.rows[0]['record']
    record = SampleRecord(**raw)
    backup = store.root/'backups'/source_key(raw['sample_id'])/'sample'
    if backup.exists():
        record = replace(record, **{field: str(backup/name) for field, name in
            [('clip_path', 'clip.mp4'), ('tracks_path', 'tracks.json'),
             ('annotation_path', 'annotation.json'), ('metadata_path', 'metadata.json')]})
    items = []
    for start, end in windows:
        window = {
            'behavior': 'Rearing', 'mouse_ids': [1], 'start_frame': start, 'end_frame': end,
            'status': 'rejected_locomotion_rearing_review_candidate',
            'reason': '站立攀爬候选，须逐帧人工确认。',
        }
        card = review_catalog([{'record': raw, 'selected_windows': [window]}])[0]
        relative = Path('dataset')/'Rearing'/batch_id/card['review_id']
        target = store.package/relative
        export_segment(
            record, target, start_frame=start, end_frame=end, behavior='Rearing', mouse_ids=(1,),
            provenance={'candidate': {'status': 'screening_candidate_needs_review'},
                        'source_sample_id': raw['sample_id'],
                        'screening': {'batch_id': batch_id},
                        'offsets': {'source_clip_local_half_open': {'start': start, 'end': end}}},
        )
        capture = cv2.VideoCapture(str(target/'clip.mp4'))
        try:
            ok, frame = capture.read()
            assert ok and cv2.imwrite(str(target/'target_poster.jpg'), frame)
        finally:
            capture.release()
        items.append({
            'review_id': card['review_id'], 'source_key': card['source_key'],
            'sample_id': card['sample_id'], 'behavior': card['behavior'],
            'mouse_ids': card['mouse_ids'], 'start_frame': card['start_frame'],
            'end_frame': card['end_frame'], 'fps': card['fps'],
            'frame_count': card['frame_count'], 'export_path': relative.as_posix(),
            'poster_path': (relative/'target_poster.jpg').as_posix(),
            'candidate_status': window['status'], 'candidate_reason': window['reason'],
            'output_sha256': tree_hashes(target),
        })
    state_bytes = store.state_path.read_bytes()
    state = json.loads(state_bytes)
    return {'schema_version': 1, 'batch_id': batch_id,
            'base_manifest_sha256': store.manifest_sha,
            'source_decisions_revision': state['revision'],
            'source_decisions_sha256': hashlib.sha256(state_bytes).hexdigest(),
            'cards': items}


def approach_supplemental_candidates(store, *, window=(1, 6), batch_id='approach_no_contact_from_rejected_v1'):
    import hashlib
    import cv2

    from mouse_behavior.evaluation.exported_dataset import SampleRecord
    from mouse_behavior.evaluation.segment_export import export_segment

    raw = store.rows[0]['record']
    record = SampleRecord(**raw)
    start, end = window
    ids = [1, 2]
    identity = {
        'sample_id': raw['sample_id'],
        'window': {'behavior': 'Approach', 'mouse_ids': ids,
                   'start_frame': start, 'end_frame': end},
        'candidate_family': 'approach_no_contact',
    }
    review_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    relative = Path('dataset')/'Approach'/batch_id/review_id
    target = store.package/relative
    export_segment(
        record, target, start_frame=start, end_frame=end, behavior='Approach', mouse_ids=tuple(ids),
        provenance={'candidate': {'status': 'approach_no_contact_recut_review_candidate'},
                    'screening': {'batch_id': batch_id},
                    'offsets': {'source_clip_local_half_open': {'start': start, 'end': end}}},
    )
    capture = cv2.VideoCapture(str(target/'clip.mp4'))
    try:
        ok, frame = capture.read()
        assert ok and cv2.imwrite(str(target/'target_poster.jpg'), frame)
    finally:
        capture.release()
    state_bytes = store.state_path.read_bytes()
    state = json.loads(state_bytes)
    item = {
        'review_id': review_id, 'source_key': source_key(raw['sample_id']),
        'sample_id': raw['sample_id'], 'behavior': 'Approach', 'mouse_ids': ids,
        'start_frame': start, 'end_frame': end, 'fps': raw['fps'],
        'frame_count': raw['frame_count'], 'export_path': relative.as_posix(),
        'poster_path': (relative/'target_poster.jpg').as_posix(),
        'candidate_status': 'approach_no_contact_recut_review_candidate',
        'candidate_reason': '无身体接触的接近重裁，待人工复核。',
        'output_sha256': tree_hashes(target),
    }
    return {
        'schema_version': 1, 'batch_id': batch_id,
        'base_manifest_sha256': store.manifest_sha,
        'source_decisions_revision': state['revision'],
        'source_decisions_sha256': hashlib.sha256(state_bytes).hexdigest(),
        'cards': [item],
    }


def test_supplemental_rearing_cards_are_revisioned_and_sync_through_review_store(review, tmp_path):
    store, source, original = review
    before = copy.deepcopy(store._state())
    manifest_before = (store.root/'dataset_manifest.json').read_bytes()
    document = supplemental_candidates(store)
    snapshot = store.publish_supplemental_candidates(document, expected_revision=0)
    card_id = document['cards'][0]['review_id']
    assert snapshot['revision'] == 1
    assert snapshot['supplemental_sha256']
    assert card_id in store.supplemental_cards and card_id in store.cards
    assert store._state()['decisions'] == before['decisions']
    assert store._state()['events'] == before['events']
    assert store._state()['requests'] == before['requests']
    assert (store.root/'dataset_manifest.json').read_bytes() == manifest_before
    assert tree_hashes(source) == original

    render_review_page(
        store.package, store.rows, catalog=list(store.cards.values()),
        candidate_sha256=store.candidate_sha,
        supplemental_sha256=store.supplemental_sha,
    )
    assert 'href="/rearing_review.html?behavior=Rearing"' in (
        store.package/'index.html').read_text(encoding='utf-8')

    page = render_review_page(
        store.package, store.rows, catalog=list(store.supplemental_cards.values()),
        candidate_sha256=store.candidate_sha,
        supplemental_sha256=store.supplemental_sha, filename='rearing_review.html',
    )
    assert page is None
    html = (store.package/'rearing_review.html').read_text(encoding='utf-8')
    assert '站立攀爬待审核候选裁剪' in html
    assert '候选审核 ID' in html and card_id in html
    assert '视觉轨迹 ID' in html and 'tracking-overlay' in html
    assert 'data-track-path=' in html and '轨迹标记暂不可用' in html
    assert 'supplemental_sha256:supplementalSha' in html

    card = next(iter(store.supplemental_cards.values()))
    server = create_server(store, 0)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        with urlopen(f'http://127.0.0.1:{server.server_port}/{card["export_path"]}/tracks.json') as response:
            assert 'application/json' in response.headers['Content-Type']
            assert isinstance(json.load(response), list)
    finally:
        server.shutdown()
        server.server_close()
        worker.join(5)

    store.close()
    restarted = ReviewStore(store.package, store.root, store.source_root)
    try:
        assert restarted.snapshot()['supplemental_sha256'] == snapshot['supplemental_sha256']
        payload_value = payload(restarted, decision='approved', index=0)
        card = restarted.supplemental_cards[card_id]
        payload_value.update({key: card[key] for key in
                              ('review_id', 'behavior', 'mouse_ids', 'start_frame', 'end_frame')})
        payload_value.update(
            revision=restarted.snapshot()['revision'],
            candidate_sha256=restarted.candidate_sha,
            supplemental_sha256=restarted.supplemental_sha,
        )
        restarted.save(payload_value)
        assert_synced(restarted)
        assert restarted._state()['decisions'][card_id]['behavior'] == 'Rearing'
        assert Path(restarted._state()['decisions'][card_id]['original_synced_path']).parent.name == 'Rearing'
    finally:
        restarted.close()


def test_approach_supplemental_cards_preserve_rearing_decisions_and_have_dedicated_page(review):
    store, source, original = review
    rearing_doc = supplemental_candidates(store)
    store.publish_supplemental_candidates(rearing_doc, expected_revision=0)
    rearing_id = rearing_doc['cards'][0]['review_id']
    rearing = store.supplemental_cards[rearing_id]
    store.save({
        **{key: rearing[key] for key in ('review_id', 'behavior', 'mouse_ids', 'start_frame', 'end_frame')},
        'decision': 'rejected', 'revision': store.snapshot()['revision'],
        'candidate_sha256': store.candidate_sha,
        'supplemental_sha256': store.supplemental_sha,
        'note': '保持既有攀爬复核结论', 'request_id': uuid.uuid4().hex,
    })
    before = copy.deepcopy(store._state())
    active = before['supplemental_review_candidates']
    previous = json.loads((store.root/active['path']).read_text(encoding='utf-8'))
    approach_doc = approach_supplemental_candidates(store)
    approach_doc['cards'] = previous['cards'] + approach_doc['cards']

    snapshot = store.publish_supplemental_candidates(
        approach_doc, expected_revision=store.snapshot()['revision'])
    approach_id = approach_doc['cards'][-1]['review_id']
    assert snapshot['revision'] == before['revision'] + 1
    assert approach_id in store.supplemental_cards and approach_id not in store.base_cards
    assert store._state()['decisions'][rearing_id] == before['decisions'][rearing_id]
    assert store._state()['decisions'].get(approach_id) is None
    assert tree_hashes(source) == original

    render_review_page(
        store.package, store.rows, catalog=list(store.cards.values()),
        candidate_sha256=store.candidate_sha, supplemental_sha256=store.supplemental_sha,
        approach_no_contact_review_available=True,
    )
    assert '/approach_no_contact_review.html?behavior=Approach' in (
        store.package/'index.html').read_text(encoding='utf-8')
    card = store.supplemental_cards[approach_id]
    render_review_page(
        store.package, store.rows, catalog=[card], candidate_sha256=store.candidate_sha,
        supplemental_sha256=store.supplemental_sha,
        filename='approach_no_contact_review.html',
        page_title='接近无身体接触重裁复核',
        page_intro='候选尚未自动通过。',
        page_rules='片段内不得出现身体接触；不设置最短时长。',
    )
    page = (store.package/'approach_no_contact_review.html').read_text(encoding='utf-8')
    assert 'pageTitleOverride='+json.dumps('接近无身体接触重裁复核') in page
    assert 'pageRulesOverride='+json.dumps('片段内不得出现身体接触；不设置最短时长。') in page
    assert '无身体接触的接近重裁（待人工复核）' in page
    assert approach_id in page and '视觉轨迹 ID' in page and 'data-target-ids="1,2"' in page
    assert 'review_sources/' in page and card['export_path']+'/clip.mp4' in page

    server = create_server(store, 0)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        with urlopen(f'http://127.0.0.1:{server.server_port}/approach_no_contact_review.html?behavior=Approach') as response:
            assert response.status == 200
            body = response.read().decode('utf-8')
            assert approach_id in body
            assert 'pageIntroOverride='+json.dumps('候选尚未自动通过。') in body
        with urlopen(f'http://127.0.0.1:{server.server_port}/{card["export_path"]}/tracks.json') as response:
            assert 'application/json' in response.headers['Content-Type']
            assert len(json.load(response)) == card['end_frame'] - card['start_frame']
    finally:
        server.shutdown()
        server.server_close()
        worker.join(5)


def test_approach_recrop_candidates_have_separate_page_and_keep_baked_id_boxes(review):
    store, source, original = review
    first = approach_supplemental_candidates(store, window=(1, 6))
    store.publish_supplemental_candidates(first, expected_revision=0)
    first_id = first['cards'][0]['review_id']
    before = copy.deepcopy(store._state())

    recrop = approach_supplemental_candidates(
        store, window=(2, 7), batch_id='approach_no_contact_from_rejected_v2')
    recrop_card = recrop['cards'][0]
    recrop_card['candidate_status'] = 'approach_no_contact_recrop_v1_review_candidate'
    recrop_card['candidate_reason'] = '接近无接触重裁候选；目标视觉 ID 框已绘制。'
    recrop_media = store.package / recrop_card['export_path']
    recrop_metadata_path = recrop_media / 'metadata.json'
    recrop_metadata = json.loads(recrop_metadata_path.read_text(encoding='utf-8'))
    recrop_metadata['provenance']['offsets']['source_clip_local_half_open'] = {
        'start': 0, 'end': recrop_card['end_frame'] - recrop_card['start_frame'],
    }
    recrop_metadata['provenance']['target_id_overlay'] = {'method': 'fixture'}
    write_json(recrop_metadata_path, recrop_metadata)
    recrop_card['output_sha256'] = tree_hashes(recrop_media)
    recrop['cards'] = first['cards'] + recrop['cards']
    snapshot = store.publish_supplemental_candidates(
        recrop, expected_revision=store.snapshot()['revision'])
    recrop_id = recrop_card['review_id']

    assert snapshot['revision'] == before['revision'] + 1
    assert first_id in store.supplemental_cards
    assert recrop_id in store.supplemental_cards and recrop_id not in store.base_cards
    assert store._state()['decisions'] == before['decisions']
    assert tree_hashes(source) == original

    render_review_page(
        store.package, store.rows, catalog=list(store.cards.values()),
        candidate_sha256=store.candidate_sha, supplemental_sha256=store.supplemental_sha,
        approach_no_contact_review_available=True,
        approach_no_contact_recrop_review_available=True,
    )
    index = (store.package/'index.html').read_text(encoding='utf-8')
    assert '/approach_no_contact_recrop_review.html?behavior=Approach' in index
    assert '/approach_no_contact_review.html?behavior=Approach' in index

    render_review_page(
        store.package, store.rows, catalog=[store.supplemental_cards[recrop_id]],
        candidate_sha256=store.candidate_sha, supplemental_sha256=store.supplemental_sha,
        filename='approach_no_contact_recrop_review.html',
        page_title='接近无接触重裁审核（60条）',
        page_intro='候选不是自动通过结果。',
        page_rules='身体接触帧和目标框重叠帧均须为零。',
    )
    page = (store.package/'approach_no_contact_recrop_review.html').read_text(encoding='utf-8')
    assert '接近无接触裁剪片段（目标 ID 框已绘制，待人工审核）' in page
    assert '原始样本片段（目标视觉 ID 框）' in page
    assert '完整原片（目标视觉 ID 框）' not in page
    assert recrop_id in page and 'review_sources/' in page
    assert recrop_card['export_path']+'/clip.mp4' in page
    assert page.count('data-track-path=') == 1  # Full source only; candidate IDs are baked in.

    server = create_server(store, 0)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        with urlopen(
            f'http://127.0.0.1:{server.server_port}/approach_no_contact_recrop_review.html?behavior=Approach'
        ) as response:
            body = response.read().decode('utf-8')
            assert response.status == 200 and recrop_id in body
        with urlopen(
            f'http://127.0.0.1:{server.server_port}/{recrop_card["export_path"]}/clip.mp4'
        ) as response:
            assert response.status == 200
            assert 'video/mp4' in response.headers['Content-Type']
    finally:
        server.shutdown()
        server.server_close()
        worker.join(5)


def test_approach_recrop_v2_candidates_use_live_track_overlay(review):
    store, _, _ = review
    first = approach_supplemental_candidates(store, window=(1, 6))
    store.publish_supplemental_candidates(first, expected_revision=0)

    recrop = approach_supplemental_candidates(
        store, window=(2, 7), batch_id='approach_no_contact_from_rejected_v2')
    card = recrop['cards'][0]
    card['candidate_status'] = 'approach_no_contact_recrop_v2_review_candidate'
    card['candidate_reason'] = '从连续源帧回补的接近无接触候选；目标 ID 框由轨迹动态显示。'
    recrop['cards'] = first['cards'] + recrop['cards']
    store.publish_supplemental_candidates(
        recrop, expected_revision=store.snapshot()['revision'])

    render_review_page(
        store.package, store.rows, catalog=[store.supplemental_cards[card['review_id']]],
        candidate_sha256=store.candidate_sha, supplemental_sha256=store.supplemental_sha,
        filename='approach_no_contact_recrop_review.html',
        approach_no_contact_recrop_review_available=True,
    )
    page = (store.package/'approach_no_contact_recrop_review.html').read_text(encoding='utf-8')
    assert '接近无接触裁剪片段（动态显示目标 ID 框，待人工审核）' in page
    assert card['review_id'] in page and card['export_path']+'/clip.mp4' in page
    assert card['export_path']+'/tracks.json' in page
    assert page.count('data-track-path=') == 2  # Source and new candidate both show live target IDs.


def test_supplemental_revision_replaces_pending_cards_and_preserves_human_decisions(review):
    store, _, _ = review
    first = supplemental_candidates(store, windows=((1, 6), (2, 7), (1, 7)))
    store.publish_supplemental_candidates(first, expected_revision=0)
    first_supplemental_sha = store.supplemental_sha
    approved_id = first['cards'][0]['review_id']
    superseded_id = first['cards'][1]['review_id']
    retained_pending_id = first['cards'][2]['review_id']
    approved = store.supplemental_cards[approved_id]
    store.save({
        **{key: approved[key] for key in
           ('review_id', 'behavior', 'mouse_ids', 'start_frame', 'end_frame')},
        'decision': 'approved', 'revision': store.snapshot()['revision'],
        'candidate_sha256': store.candidate_sha,
        'supplemental_sha256': store.supplemental_sha,
        'note': 'reviewed before crop revision', 'request_id': uuid.uuid4().hex,
    })
    before = copy.deepcopy(store._state())
    replacement = supplemental_candidates(
        store, windows=((1, 6), (3, 8), (1, 7)),
        batch_id='rearing_from_rejected_locomotion_v2')
    without_reviewed = copy.deepcopy(replacement)
    without_reviewed['cards'] = without_reviewed['cards'][1:]
    with pytest.raises(ReviewConflict, match='retain every reviewed supplemental card'):
        store.publish_supplemental_candidates(without_reviewed, expected_revision=before['revision'])
    snapshot = store.publish_supplemental_candidates(replacement, expected_revision=before['revision'])
    new_pending_id = replacement['cards'][1]['review_id']
    assert snapshot['revision'] == before['revision'] + 1
    assert set(store.supplemental_cards) == {approved_id, new_pending_id, retained_pending_id}
    assert superseded_id not in store.supplemental_cards
    assert store._state()['decisions'][approved_id] == before['decisions'][approved_id]
    history = store._state()['supplemental_candidate_history'][-1]
    assert history['superseded_review_ids'] == [superseded_id]
    assert history['retained_pending_ids'] == [retained_pending_id]
    assert history['preserved_reviewed_ids'] == [approved_id]
    assert store.snapshot()['counts']['approved'] == 1
    assert store.snapshot()['counts']['pending'] == len(store.cards) - 1
    retained = store.supplemental_cards[retained_pending_id]
    with pytest.raises(ReviewConflict, match='supplemental candidates changed'):
        store.save({
            **{key: retained[key] for key in
               ('review_id', 'behavior', 'mouse_ids', 'start_frame', 'end_frame')},
            'decision': 'rejected', 'revision': snapshot['revision'],
            'candidate_sha256': store.candidate_sha,
            'supplemental_sha256': first_supplemental_sha,
            'note': 'stale-page guard', 'request_id': uuid.uuid4().hex,
        })
    assert retained_pending_id not in store._state()['decisions']
    store.close()
    restarted = ReviewStore(store.package, store.root, store.source_root)
    try:
        assert set(restarted.supplemental_cards) == {approved_id, new_pending_id, retained_pending_id}
        assert restarted._state()['decisions'][approved_id] == before['decisions'][approved_id]
        assert superseded_id not in restarted.cards
    finally:
        restarted.close()


def test_supplemental_candidate_publication_rejects_stale_revision_and_tampered_media(review):
    store, source, original = review
    document = supplemental_candidates(store)
    with pytest.raises(ValueError, match='schema'):
        store.publish_supplemental_candidates([], expected_revision=0)
    with pytest.raises(ReviewConflict, match='state changed'):
        store.publish_supplemental_candidates(document, expected_revision=1)
    document['source_decisions_revision'] += 1
    with pytest.raises(ReviewConflict, match='source decisions changed'):
        store.publish_supplemental_candidates(document, expected_revision=0)
    document['source_decisions_revision'] -= 1
    document['cards'][0]['output_sha256']['clip.mp4'] = '0'*64
    with pytest.raises(ReviewConflict, match='payload changed'):
        store.publish_supplemental_candidates(document, expected_revision=0)
    assert store.snapshot()['revision'] == 0
    assert tree_hashes(source) == original


def test_candidate_revision_preserves_decisions_and_both_datasets_on_restart(review):
    store, source, _ = review
    store.save(payload(store, start_frame=2, end_frame=6))
    store.save(payload(store, index=1, decision='rejected'))
    before = copy.deepcopy(store._state())
    original = tree_hashes(source)
    confirmed = tree_hashes(store.root/'dataset')
    manifest = (store.root/'dataset_manifest.json').read_bytes()
    catalog_ids = set(store.cards)
    document = candidate_revision(store)
    snapshot = store.publish_candidate_updates(document, expected_revision=2)
    assert snapshot['revision'] == 3 and snapshot['candidate_sha256']
    assert set(store.cards) == catalog_ids and list(store.cards.values())[0]['start_frame'] == 1
    for field in ('decisions', 'sources', 'events', 'requests'):
        assert store._state()[field] == before[field]
    assert tree_hashes(source) == original and tree_hashes(store.root/'dataset') == confirmed
    assert (store.root/'dataset_manifest.json').read_bytes() == manifest
    store.close()
    restarted = ReviewStore(store.package, store.root, store.source_root)
    try:
        assert restarted.snapshot() == snapshot
        request = payload(restarted)
        with pytest.raises(ReviewConflict, match='candidate'):
            restarted.save(request)
        request['candidate_sha256'] = snapshot['candidate_sha256']
        restarted.save(request)
        assert_synced(restarted)
        assert restarted._state()['decisions'][request['review_id']]['start_frame'] == 1
    finally:
        restarted.close()


@pytest.mark.parametrize('invalid', ['base', 'range', 'wrong_range_same_length', 'unknown', 'duplicate', 'non_approach', 'path', 'hash'])
def test_invalid_candidate_revision_cannot_publish_or_change_decisions(review, invalid):
    store, source, original = review
    document = candidate_revision(store)
    update = document['updates'][0]
    if invalid == 'base':
        document['base_manifest_sha256'] = 'invalid'
    elif invalid == 'range':
        update['end_frame'] = 99
    elif invalid == 'wrong_range_same_length':
        update.update(start_frame=2, end_frame=7)
    elif invalid == 'unknown':
        update['review_id'] = 'unknown'
    elif invalid == 'duplicate':
        document['updates'].append(copy.deepcopy(update))
    elif invalid == 'non_approach':
        update['review_id'] = list(store.cards)[1]
    elif invalid == 'path':
        update['export_path'] = '../source'
    elif invalid == 'hash':
        update['output_sha256']['clip.mp4'] = 'invalid'
    before = store.state_path.read_bytes()
    with pytest.raises((ValueError, ReviewConflict)):
        store.publish_candidate_updates(document, expected_revision=0)
    assert store.state_path.read_bytes() == before and tree_hashes(source) == original


def test_missing_candidate_revision_shows_original_and_never_discards_rejection(review):
    store, source, original = review
    store.save(payload(store, decision='rejected'))
    document = candidate_revision(store, missing=True)
    store.publish_candidate_updates(document, expected_revision=1)
    card = list(store.cards.values())[0]
    assert card['export_path'] is None and store.snapshot()['counts']['rejected'] == 1
    render_review_page(store.package, store.rows, catalog=list(store.cards.values()),
                       candidate_sha256=store.candidate_sha)
    page = (store.package/'index.html').read_text(encoding='utf-8')
    assert 'Revised start; saved decisions retained' in page
    assert store.candidate_sha in page and 'const candidateSha=' in page
    assert tree_hashes(source) == original


def test_candidate_revision_receipt_tampering_blocks_restart(review):
    store, _, _ = review
    store.publish_candidate_updates(candidate_revision(store, missing=True), expected_revision=0)
    pointer = store._state()['candidate_updates']
    (store.root/pointer['path']).write_text('{}', encoding='utf-8')
    store.close()
    with pytest.raises(ReviewConflict, match='candidate revision changed'):
        ReviewStore(store.package, store.root, store.source_root)


@pytest.mark.parametrize('fail_at', [1, 2])
def test_candidate_pointer_crash_restores_prior_catalog_and_decisions(review, monkeypatch, fail_at):
    store, source, original = review
    from mouse_behavior.evaluation import review_transaction as module
    actual_replace = module.os.replace
    actual_recover = store.transaction.recover
    count, crashed = 0, False
    before = store.state_path.read_bytes()
    document = candidate_revision(store, missing=True)

    def injected(src, dst):
        nonlocal count, crashed
        actual_replace(src, dst)
        if (Path(src).name.startswith('.review-stage-') and not Path(src).name.endswith('.tmp')) or '.review-rollback-' in Path(dst).name:
            count += 1
            if count == fail_at:
                crashed = True
                raise RuntimeError('simulated candidate pointer crash')

    def no_recovery():
        if crashed:
            raise RuntimeError('process exited')
        actual_recover()

    monkeypatch.setattr(module.os, 'replace', injected)
    monkeypatch.setattr(store.transaction, 'recover', no_recovery)
    with pytest.raises(RuntimeError):
        store.publish_candidate_updates(document, expected_revision=0)
    monkeypatch.setattr(module.os, 'replace', actual_replace)
    store.close()
    restarted = ReviewStore(store.package, store.root, store.source_root)
    try:
        assert restarted.state_path.read_bytes() == before
        assert restarted.candidate_sha is None and restarted.snapshot()['revision'] == 0
        assert tree_hashes(source) == original and not restarted.transaction.journal.exists()
    finally:
        restarted.close()


def test_approve_updates_both_datasets_and_restart_persists(review):
    store, source, original = review
    request = payload(store, start_frame=2, end_frame=6)
    result = store.save(request)
    assert result['counts']['approved'] == 1 and result['revision'] == 1
    assert_synced(store)
    assert tree_hashes(source) != original
    backup = store.root/'backups'/source_key('Approach/example')/'sample'
    assert tree_hashes(backup) == original
    snapshot = store.snapshot()
    store.close()
    restarted = ReviewStore(store.package, store.root, store.source_root)
    try:
        assert restarted.snapshot() == snapshot
        assert restarted.save(request)['revision'] == 1  # Lost response retry does not crop twice.
    finally:
        restarted.close()


def test_multiple_windows_relabel_primary_undo_and_restore_all_original_bytes(review):
    store, source, original = review
    store.save(payload(store, behavior='Grooming', mouse_ids=[1], start_frame=2, end_frame=5))
    assert not source.exists()
    store.save(payload(store, index=1))
    assert_synced(store)
    assert len(discover_samples(store.source_root)) == 2
    store.save(payload(store, decision='rejected'))
    assert_synced(store)
    assert len(discover_samples(store.source_root)) == 1
    store.save(payload(store, index=1, decision='pending'))
    assert tree_hashes(source) == original
    assert not list((store.root/'dataset').glob('*/*/clip.mp4'))
    assert json.loads((store.root/'dataset_manifest.json').read_text()) == []
    assert store.snapshot()['revision'] == 4


def test_rejected_and_uncertain_do_not_enter_dataset_or_touch_originals(review):
    store, source, original = review
    store.save(payload(store, decision='rejected'))
    store.save(payload(store, index=1, decision='uncertain'))
    assert tree_hashes(source) == original
    assert not (store.root/'backups').exists()
    assert json.loads((store.root/'dataset_manifest.json').read_text()) == []


@pytest.mark.parametrize('changes', [{'mouse_ids': [1, 999]}, {'start_frame': -1},
                                    {'end_frame': 99}, {'behavior': '../Static'},
                                    {'mouse_ids': [1, 1]}, {'start_frame': True},
                                    {'behavior': 'Grooming', 'mouse_ids': [1, 2]}])
def test_invalid_approval_never_mutates_original_or_accepts_sample(review, changes):
    store, source, original = review
    with pytest.raises(ValueError):
        store.save(payload(store, **changes))
    assert tree_hashes(source) == original and store.snapshot()['revision'] == 0
    assert not list((store.root/'dataset').glob('*/*/clip.mp4'))


def test_stale_tab_and_changed_source_are_rejected(review):
    store, source, original = review
    stale = payload(store, index=1)
    store.save(payload(store, decision='uncertain'))
    with pytest.raises(ReviewConflict, match='another review'):
        store.save(stale)
    (source/'annotation.json').write_text('{"external":true}')
    with pytest.raises(ReviewConflict, match='source changed'):
        store.save(payload(store))
    assert store.snapshot()['counts'].get('approved', 0) == 0


@pytest.mark.parametrize('fail_at', range(1, 8))
def test_crash_after_each_payload_rename_recovers_on_restart(review, monkeypatch, fail_at):
    store, source, original = review
    from mouse_behavior.evaluation import review_transaction as module
    actual_replace = module.os.replace
    actual_recover = store.transaction.recover
    count, crashed = 0, False

    def injected(src, dst):
        nonlocal count, crashed
        actual_replace(src, dst)
        if (Path(src).name.startswith('.review-stage-') and not Path(src).name.endswith('.tmp')) or '.review-rollback-' in Path(dst).name:
            count += 1
            if count == fail_at:
                crashed = True
                raise RuntimeError('simulated abrupt exit after completed rename')

    def no_recovery_after_crash():
        if crashed:
            raise RuntimeError('process exited before recovery')
        actual_recover()

    monkeypatch.setattr(module.os, 'replace', injected)
    monkeypatch.setattr(store.transaction, 'recover', no_recovery_after_crash)
    with pytest.raises(RuntimeError):
        store.save(payload(store, start_frame=2, end_frame=6))
    assert store.transaction.journal.exists()
    monkeypatch.setattr(module.os, 'replace', actual_replace)
    store.close()
    restarted = ReviewStore(store.package, store.root, store.source_root)
    try:
        assert not restarted.recovery_error
        assert tree_hashes(source) == original
        assert restarted.snapshot()['revision'] == 0
        assert not list((store.root/'dataset').glob('*/*/clip.mp4'))
        assert not restarted.transaction.journal.exists()
    finally:
        restarted.close()


def test_external_changes_after_approval_block_overwrite_and_preserve_edit(review):
    store, _, _ = review
    store.save(payload(store))
    synced = Path(next(iter(store._state()['decisions'].values()))['original_synced_path'])
    (synced/'annotation.json').write_text('{"external":true}')
    with pytest.raises(ReviewConflict, match='changed externally'):
        store.save(payload(store, decision='pending'))
    assert (synced/'annotation.json').read_text() == '{"external":true}'
    assert store.snapshot()['revision'] == 1


def test_api_persists_acceptance_and_ranges_and_blocks_cross_origin(review):
    store, source, original = review
    server = create_server(store, 0)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    url = f'http://127.0.0.1:{server.server_port}'
    try:
        with urlopen(url+'/api/state') as response:
            state = json.load(response)
        headers = {'Content-Type': 'application/json', 'X-Review-Token': state['token']}
        with urlopen(Request(url+'/api/recover', data=b'{}', headers=headers)) as response:
            assert json.load(response)['revision'] == 0
        foreign = {**headers, 'Origin': 'http://untrusted.example'}
        with pytest.raises(HTTPError) as error:
            urlopen(Request(url+'/api/decision', data=json.dumps(payload(store)).encode(), headers=foreign))
        assert error.value.code == 403 and tree_hashes(source) == original
        with urlopen(Request(url+'/api/decision', data=json.dumps(payload(store)).encode(), headers=headers)) as response:
            assert json.load(response)['counts']['approved'] == 1
        assert_synced(store)
        rid = list(store.cards)[0]
        video = store.confirmed_video(rid).read_bytes()
        tracks = store.confirmed_tracks(rid).read_bytes()
        with urlopen(url+f'/api/confirmed/{rid}/tracks.json') as response:
            assert 'application/json' in response.headers['Content-Type']
            assert response.read() == tracks
        with urlopen(Request(url+f'/api/confirmed/{rid}/clip.mp4', headers={'Range': 'bytes=4-15'})) as response:
            assert response.status == 206 and response.read() == video[4:16]
            assert response.headers['Content-Range'] == f'bytes 4-15/{len(video)}'
        with urlopen(Request(url+f'/api/confirmed/{rid}/clip.mp4', method='HEAD')) as response:
            assert int(response.headers['Content-Length']) == len(video)
        for path in ('/api/not-a-real-route', '/decisions.json'):
            with pytest.raises(HTTPError) as error:
                urlopen(url+path)
            assert error.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
        worker.join(5)


def test_all_source_cards_have_manual_controls_even_without_candidates(review):
    store, _, _ = review
    rows = copy.deepcopy(store.rows)
    rows[0]['selected_windows'] = []
    render_review_page(store.package, rows)
    page = (store.package/'index.html').read_text(encoding='utf-8')
    assert page.count('<article ') == 1
    assert '确认通过并同步' in page and '暂不确定' in page
    assert '/api/decision' in page and '/api/state' in page
    assert '暂无合格自动片段' in page and '播放和筛选仍可使用' in page


def test_bottom_pagination_returns_to_top_after_page_change(review):
    store, _, _ = review
    render_review_page(
        store.package, [], catalog=[next(iter(store.cards.values()))],
        candidate_sha256='test-candidate',
    )
    page = (store.package/'index.html').read_text(encoding='utf-8')
    assert '<nav class="pagination-bottom" aria-label="审核分页">' in page
    assert 'id="prev-bottom"' in page and 'id="next-bottom"' in page
    assert 'id="count-bottom" aria-live="polite"' in page
    assert "function changePage(delta)" in page
    assert "window.scrollTo(0,0)" in page
    assert "document.getElementById('next'+suffix).onclick=()=>changePage(1)" in page


def test_behavior_scoped_page_locks_queue_and_summarizes_only_that_behavior(review):
    store, _, _ = review
    catalog = [dict(card, behavior='Locomotion' if index == 0 else 'Static')
               for index, card in enumerate(store.cards.values())]
    render_review_page(store.package, [], catalog=catalog, candidate_sha256='test-candidate')
    page = (store.package/'index.html').read_text(encoding='utf-8')
    assert "new URLSearchParams(location.search).get('behavior')" in page
    assert "behaviorAliases={'移动':'Locomotion','运动':'Locomotion','站立攀爬':'Rearing','静止':'Static','一起':'Together','接触行为复核':'ContactReview'}" in page
    assert '此页面不会切换到其他行为类别' in page
    assert 'for(const card of scopedCards)' in page and 'scopedIds.has(id)' in page


def test_contact_rejections_are_returned_to_approach_review(review):
    store, _, _ = review
    card = {
        **next(iter(store.cards.values())),
        'behavior': 'Approach',
        'candidate_status': 'approach_contact_to_approach_review',
        'candidate_reason': '原接近审核因片段含身体接触而退回；现重新审核接触前的接近过程。',
        'review_from_behavior': 'Approach',
        'review_default_behavior': 'Approach',
        'review_origin_behavior': 'Approach',
        'review_origin_decision_updated_at': 'source-rejection-time',
        'review_transfer_pending_label': '待重新审核（原接近因身体接触退回）',
        'export_path': 'dataset/Approach/contact-preview',
        'poster_path': 'dataset/Approach/contact-preview/target_poster.jpg',
    }
    render_review_page(
        store.package, [], catalog=[card], candidate_sha256=store.candidate_sha,
        filename='approach_contact_review.html',
    )
    page = (store.package/'approach_contact_review.html').read_text(encoding='utf-8')
    assert '接近退回预览（重新审核接近过程）' in page
    assert '接近接触退回复核：这些片段仍归接近审核，不预先改成一起。' in page
    assert 'function isUnreviewedTransfer(base,saved)' in page
    assert 'saved.updated_at===base.review_origin_decision_updated_at' in page
    assert '待重新审核（原接近因身体接触退回）' in page
    assert 'review_default_behavior' in page
    assert '待Together复核（原接近未通过）' not in page


def test_social_clustering_huddling_notes_return_as_pending_huddling_review(review):
    store, _, _ = review
    base = copy.deepcopy(next(iter(store.base_cards.values())))
    base.update(review_id='huddle-one', behavior='Social clustering', export_path=None,
                poster_path=None, mouse_ids=[1, 2, 9])
    base_cards = {
        'huddle-one': base,
        'huddle-two': {**base, 'review_id': 'huddle-two'},
        'not-huddling': {**base, 'review_id': 'not-huddling'},
        'already-approved': {**base, 'review_id': 'already-approved'},
        'other-behavior': {**base, 'review_id': 'other-behavior', 'behavior': 'Together'},
    }
    decisions = {
        'huddle-one': {'decision': 'rejected', 'behavior': 'Social clustering',
                       'note': '扎堆', 'updated_at': 'rejection-one'},
        'huddle-two': {'decision': 'rejected', 'behavior': 'Social clustering',
                       'note': '扎堆时间太长', 'updated_at': 'rejection-two'},
        'not-huddling': {'decision': 'rejected', 'behavior': 'Social clustering',
                         'note': '不是扎堆', 'updated_at': 'rejection-three'},
        'already-approved': {'decision': 'approved', 'behavior': 'Social clustering',
                             'note': '扎堆', 'updated_at': 'approval-one'},
        'other-behavior': {'decision': 'rejected', 'behavior': 'Together',
                           'note': '扎堆', 'updated_at': 'rejection-four'},
    }

    cards = huddling_recheck_catalog(base_cards, decisions)

    assert [card['review_id'] for card in cards] == ['huddle-one', 'huddle-two']
    assert all(card['behavior'] == 'Huddling' for card in cards)
    assert all(card['review_origin_behavior'] == 'Social clustering' for card in cards)
    assert cards[0]['review_default_behavior'] == 'Huddling'
    assert '原社会聚集不通过备注：扎堆' in cards[0]['candidate_reason']

    render_review_page(
        store.package, store.rows, catalog=cards,
        candidate_sha256=store.candidate_sha, supplemental_sha256=store.supplemental_sha,
        filename='huddling_recheck_review.html', page_title='扎堆复核（2条）',
        page_intro='社会聚集未通过记录重新列为扎堆复核候选。',
        page_rules='原不通过决定保留；只有人工确认后才同步。',
    )
    page = (store.package/'huddling_recheck_review.html').read_text(encoding='utf-8')
    assert 'PAGE_TITLE_OVERRIDE' not in page
    assert '原始样本片段（目标视觉 ID 框）' in page
    assert '待扎堆复核（原社会聚集未通过）' in page
    assert 'review_default_behavior' in page

    server = create_server(store, 0)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        with urlopen(
            f'http://127.0.0.1:{server.server_port}/huddling_recheck_review.html?behavior=Huddling'
        ) as response:
            body = response.read().decode('utf-8')
            assert response.status == 200 and 'huddle-one' in body and 'huddle-two' in body
    finally:
        server.shutdown()
        server.server_close()
        worker.join(5)


def test_contact_prescreen_page_labels_keypoint_results_as_hints(review):
    store, _, _ = review
    card = {
        **next(iter(store.cards.values())),
        'behavior': 'Together',
        'candidate_status': 'together_contact_geometry_prescreen',
        'candidate_reason': '鼻头初筛建议；需人工核实。',
        'review_display_behavior': '初筛建议 Snout-head_contact',
        'review_suggestion_label': '关键点建议：Snout-head_contact',
        'review_origin_behavior': 'Together',
        'review_origin_decision_updated_at': 'source-rejection-time',
        'review_default_behavior': 'Snout-head_contact',
        'review_transfer_pending_label': '待关键点初筛复核（Together未通过）',
        'export_path': 'review_sources/contact-screen/example',
        'poster_path': 'review_sources/contact-screen/example/target_poster.jpg',
    }
    render_review_page(
        store.package, [], catalog=[card], candidate_sha256=store.candidate_sha,
        filename='contact_reclassification_review.html',
    )
    page = (store.package/'contact_reclassification_review.html').read_text(encoding='utf-8')
    assert '关键点初筛预览（待人工确认）' in page
    assert '建议行为仅作线索，不是确认标签' in page
    assert "scopedBehavior==='ContactReview'?catalog" in page
    assert 'function isUnreviewedTransfer(base,saved)' in page
    assert '待关键点初筛复核（Together未通过）' in page


def test_together_review_page_is_served_but_other_html_remains_private(review):
    store, _, _ = review
    render_review_page(
        store.package, [], catalog=[next(iter(store.cards.values()))],
        candidate_sha256=store.candidate_sha, filename='together_review.html',
    )
    render_review_page(
        store.package, [], catalog=[next(iter(store.cards.values()))],
        candidate_sha256=store.candidate_sha, filename='approach_contact_review.html',
    )
    render_review_page(
        store.package, [], catalog=[next(iter(store.cards.values()))],
        candidate_sha256=store.candidate_sha, filename='contact_reclassification_review.html',
    )
    server = create_server(store, 0)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        with urlopen(f'http://127.0.0.1:{server.server_port}/together_review.html?behavior=Together') as response:
            assert response.status == 200
            assert 'text/html' in response.headers['Content-Type']
        with urlopen(f'http://127.0.0.1:{server.server_port}/approach_contact_review.html?behavior=Approach') as response:
            assert response.status == 200
            assert 'text/html' in response.headers['Content-Type']
        with urlopen(f'http://127.0.0.1:{server.server_port}/contact_reclassification_review.html?behavior=ContactReview') as response:
            assert response.status == 200
            assert 'text/html' in response.headers['Content-Type']
        with pytest.raises(HTTPError) as error:
            urlopen(f'http://127.0.0.1:{server.server_port}/private.html')
        assert error.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
        worker.join(5)


def test_exported_candidate_shows_full_review_and_visual_ids(review):
    store, _, _ = review
    card = {**next(iter(store.cards.values())), 'export_path': 'dataset/Static/example'}
    render_review_page(store.package, store.rows, catalog=[card],
                       candidate_sha256=store.candidate_sha)
    page = (store.package/'index.html').read_text(encoding='utf-8')
    article = page.split(f'<article data-id="{card["review_id"]}"', 1)[1].split('</article>', 1)[0]

    assert f'<strong>候选审核 ID：</strong><code>{card["review_id"]}</code>' in article
    assert f'视觉轨迹 ID 1,2 · 审核 ID {card["review_id"]}' in article
    assert 'src="dataset/Static/example/clip.mp4"' in article
    assert 'data-track-path="dataset/Static/example/tracks.json"' in article
    assert 'data-target-ids="1,2"' in article and 'class="tracking-overlay"' in article
    assert "'/api/confirmed/'+c.dataset.id+'/tracks.json?revision='+state.revision" in page
    assert "v.dataset.targetIds=d.mouse_ids.join(',')" in page
    assert "input.addEventListener('input'" in page
    assert "videos.forEach(video=>{video.dataset.targetIds=ids;drawTracking(video);});" in page


def test_static_review_page_explains_head_turn_policy_in_plain_chinese(review):
    store, _, _ = review
    card = {
        **next(iter(store.cards.values())),
        'behavior': 'Static',
        'candidate_reason': 'body_static_head_turn_allowed',
        'export_path': 'dataset/Static/example',
    }
    render_review_page(store.package, store.rows, catalog=[card],
                       candidate_sha256=store.candidate_sha)
    page = (store.package/'index.html').read_text(encoding='utf-8')
    article = page.split(f'<article data-id="{card["review_id"]}"', 1)[1].split('</article>', 1)[0]

    assert '静止判定提示：持续至少1秒，体心距片段中位位置不超过4 cm' in page
    assert '短暂挪动最多1秒且偏移不超过4 cm' in page
    assert '1.3375 cm' not in page
    assert '0.2秒' not in page
    assert '短鼻尾二维投影本身不会取消符合中心静止边界的静止资格' in page
    assert '顶视投影不能证明真实抬身高度' in page
    assert '候选说明：静止判定持续至少1秒，体心距片段中位位置不超过4 cm；允许不超过1秒、偏移不超过4 cm的短暂挪动' in article
    assert 'data-track-path="dataset/Static/example/tracks.json"' in article
    assert 'data-target-ids="1,2"' in article and 'class="tracking-overlay"' in article
    assert 'body_static_head_turn_allowed' not in article


def test_second_server_cannot_recover_or_write_a_live_store(review):
    store, _, _ = review
    with pytest.raises(RuntimeError, match='another manual-review server'):
        ReviewStore(store.package, store.root, store.source_root)
    with pytest.raises(RuntimeError, match='another manual-review server'):
        ReviewStore(store.package, store.root.with_name('other-confirmed'), store.source_root)
    assert store.snapshot()['revision'] == 0


def test_external_edit_during_export_is_not_adopted_as_transaction_baseline(review, monkeypatch):
    store, source, _ = review
    from mouse_behavior.evaluation import manual_review as module
    real_export = module.export_segment

    def edit_during_export(*args, **kwargs):
        result = real_export(*args, **kwargs)
        (source/'user_notes.txt').write_text('edit during slow export')
        return result

    monkeypatch.setattr(module, 'export_segment', edit_during_export)
    with pytest.raises(RuntimeError, match='changed while preparing'):
        store.save(payload(store))
    assert (source/'user_notes.txt').read_text() == 'edit during slow export'
    assert store.snapshot()['revision'] == 0
    assert not list((store.root/'dataset').glob('*/*/clip.mp4'))


@pytest.mark.parametrize('phase', ['prepared', 'committed'])
def test_partial_cleanup_sharing_violation_can_resume_after_restart(review, monkeypatch, phase):
    store, source, original = review
    from mouse_behavior.evaluation import review_transaction as module
    actual_remove, actual_replace = module.shutil.rmtree, module.os.replace
    interrupted = False

    def remove_one_then_fail(path, *args, **kwargs):
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            next(p for p in Path(path).iterdir() if p.is_file()).unlink()
            raise PermissionError('simulated sharing violation after one file was deleted')
        return actual_remove(path, *args, **kwargs)

    def interrupt_after_original_install(src, dst):
        actual_replace(src, dst)
        if phase == 'prepared' and Path(dst) == source and Path(src).name.startswith('.review-stage-'):
            raise RuntimeError('interrupted before commit')

    monkeypatch.setattr(module.shutil, 'rmtree', remove_one_then_fail)
    monkeypatch.setattr(module.os, 'replace', interrupt_after_original_install)
    with pytest.raises((RuntimeError, PermissionError)):
        store.save(payload(store, start_frame=2, end_frame=6))
    assert store.transaction.journal.exists()
    assert json.loads(store.transaction.journal.read_text())['phase'] == phase
    monkeypatch.setattr(module.shutil, 'rmtree', actual_remove)
    monkeypatch.setattr(module.os, 'replace', actual_replace)
    store.close()
    restarted = ReviewStore(store.package, store.root, store.source_root)
    try:
        assert not restarted.recovery_error and not restarted.transaction.journal.exists()
        if phase == 'prepared':
            assert tree_hashes(source) == original and restarted.snapshot()['revision'] == 0
        else:
            assert restarted.snapshot()['revision'] == 1
            assert_synced(restarted)
    finally:
        restarted.close()


def test_confirmed_video_stream_is_immutable_and_does_not_lock_swap_target(review):
    store, source, original = review
    store.save(payload(store))
    rid = list(store.cards)[0]
    media = store.confirmed_video(rid)
    assert media.is_relative_to(store.root/'media')
    with media.open('rb') as streaming_handle:
        store.save(payload(store, decision='pending'))
        assert streaming_handle.read(12)[4:8] == b'ftyp'
    assert tree_hashes(source) == original


def test_stage_changed_after_export_cannot_diverge_from_decision_hashes(review, monkeypatch):
    store, source, original = review
    actual_apply = store.transaction.apply

    def altered_stage(txid, swaps):
        next(stage for target, stage, old, new in swaps if stage and stage.is_dir()).joinpath('annotation.json').write_text('{"tampered":true}')
        actual_apply(txid, swaps)

    monkeypatch.setattr(store.transaction, 'apply', altered_stage)
    with pytest.raises(RuntimeError, match='staged sample changed'):
        store.save(payload(store))
    assert tree_hashes(source) == original and store.snapshot()['revision'] == 0


@pytest.mark.parametrize('changes', [{'mouse_ids': {}}, {'start_frame': 'bad'}, {'behavior': []}])
def test_nonapproved_malformed_fields_cannot_break_saved_frontend_state(review, changes):
    store, _, _ = review
    with pytest.raises(ValueError, match='field types'):
        store.save(payload(store, decision='rejected', **changes))
    assert store.snapshot()['revision'] == 0



def test_full_source_video_shows_target_track_boxes(review):
    store, _, _ = review
    card = dict(next(iter(store.cards.values())))
    render_review_page(store.package, store.rows, catalog=[card],
                       candidate_sha256=store.candidate_sha)
    page = (store.package/'index.html').read_text(encoding='utf-8')
    article = page.split(f'<article data-id="{card["review_id"]}"', 1)[1].split('</article>', 1)[0]

    assert f'data-track-path="/api/source-tracks/{card["source_key"]}.json"' in article
    assert f'data-target-ids="{",".join(map(str, card["mouse_ids"]))}"' in article
    assert 'class="original"' in article and 'class="tracking-overlay"' in article
    assert "document.querySelectorAll('video[data-track-path]')" in page


def test_source_tracks_api_returns_full_original_boxes_even_after_sync(review):
    store, _, _ = review
    card = next(iter(store.cards.values()))
    key = card['source_key']
    source_track_path = Path(store.sources[key]['record']['tracks_path'])
    source_tracks = json.loads(source_track_path.read_text(encoding='utf-8'))
    expected = [
        {'frame': frame['frame'], 'detections': [
            {'track_id': int(detection['track_id']), 'box': detection['box']}
            for detection in frame.get('detections', [])
            if isinstance(detection.get('box'), list) and len(detection['box']) == 4
        ]}
        for frame in source_tracks
    ]
    server = create_server(store, 0)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        def fetch_tracks():
            with urlopen(f'http://127.0.0.1:{server.server_port}/api/source-tracks/{key}.json') as response:
                assert 'application/json' in response.headers['Content-Type']
                return json.load(response)

        assert fetch_tracks() == expected
        store.save(payload(store, decision='approved'))
        assert fetch_tracks() == expected
    finally:
        server.shutdown()
        server.server_close()
        worker.join(5)
