"""Serve manual confirmation and backed-up original dataset write-back locally."""

from __future__ import annotations

import argparse
import copy
import json
import logging
from pathlib import Path

from _bootstrap import REPO_ROOT  # noqa: F401
from mouse_behavior.evaluation.manual_review import ReviewStore, huddling_recheck_catalog
from mouse_behavior.evaluation.manual_review_page import render_review_page
from mouse_behavior.evaluation.manual_review_server import create_server
from mouse_behavior.evaluation.review_transaction import safe_path, tree_hashes


def _together_review_catalog(store):
    return [copy.deepcopy(card) for card in store.base_cards.values()
            if card['behavior'] == 'Together']


def _approach_no_contact_review_catalog(store):
    return [copy.deepcopy(card) for card in store.supplemental_cards.values()
            if card['behavior'] == 'Approach'
            and card['candidate_status'] == 'approach_no_contact_recut_review_candidate']


def _approach_no_contact_recrop_review_catalog(store):
    return [copy.deepcopy(card) for card in store.supplemental_cards.values()
            if card['behavior'] == 'Approach'
            and card['candidate_status'] in {
                'approach_no_contact_recrop_v1_review_candidate',
                'approach_no_contact_recrop_v2_review_candidate',
            }]


def _approach_contact_review_catalog(store):
    manifest_path = store.package / 'approach_contact_review_candidates_v1.json'
    if not manifest_path.is_file():
        return None
    document = json.loads(manifest_path.read_text(encoding='utf-8'))
    if (document.get('schema_version') != 1
            or document.get('base_manifest_sha256') != store.manifest_sha
            or document.get('batch_id') != 'approach_contact_review_v1'
            or not isinstance(document.get('cards'), list)
            or document.get('candidate_count') != len(document['cards'])):
        raise ValueError('Approach contact-review manifest does not match the frozen review package')

    cards = []
    seen = set()
    for item in document['cards']:
        review_id = item.get('review_id')
        base = store.base_cards.get(review_id)
        if (not base or base['behavior'] != 'Approach'
                or base['source_key'] != item.get('source_key')
                or base['sample_id'] != item.get('sample_id')
                or base['frame_count'] != item.get('frame_count')
                or float(base['fps']) != float(item.get('fps', 0))
                or not isinstance(item.get('review_origin_decision_updated_at'), str)):
            raise ValueError(f'Approach review candidate no longer matches its source: {review_id}')
        if review_id in seen:
            raise ValueError(f'duplicate Approach review candidate: {review_id}')
        seen.add(review_id)
        export_path = item.get('export_path')
        target = safe_path(store.package / export_path, [store.package / 'dataset' / 'Approach'])
        if tree_hashes(target) != item.get('output_sha256'):
            raise ValueError(f'Approach review media changed: {review_id}')
        card = copy.deepcopy(base)
        card.update(
            behavior='Approach',
            mouse_ids=list(item['mouse_ids']),
            start_frame=item['start_frame'],
            end_frame=item['end_frame'],
            export_path=export_path,
            poster_path=item['poster_path'],
            candidate_status='approach_contact_to_approach_review',
            candidate_reason=item['candidate_reason'],
            review_from_behavior='Approach',
            review_default_behavior='Approach',
            review_origin_behavior='Approach',
            review_origin_decision_updated_at=item['review_origin_decision_updated_at'],
            review_transfer_pending_label='待重新审核（原接近因身体接触退回）',
        )
        cards.append(card)
    if len(cards) != document['candidate_count']:
        raise ValueError('Approach review catalog contains missing candidates')
    return cards


def _together_contact_reclassification_catalog(store):
    manifest_path = store.package / 'together_contact_reclassification_v1.json'
    if not manifest_path.is_file():
        return None
    document = json.loads(manifest_path.read_text(encoding='utf-8'))
    if (document.get('schema_version') != 1
            or document.get('base_manifest_sha256') != store.manifest_sha
            or document.get('batch_id') != 'together_contact_geometry_screen_v1'
            or not isinstance(document.get('cards'), list)
            or document.get('candidate_count') != len(document['cards'])):
        raise ValueError('Together contact-screen manifest does not match the frozen review package')

    cards = []
    seen = set()
    contact_labels = {'Snout-head_contact', 'Snout-rear_contact'}
    for item in document['cards']:
        review_id = item.get('review_id')
        base = store.base_cards.get(review_id)
        if (not base or base['behavior'] != 'Together'
                or base['source_key'] != item.get('source_key')
                or base['sample_id'] != item.get('sample_id')
                or base['frame_count'] != item.get('frame_count')
                or float(base['fps']) != float(item.get('fps', 0))
                or item.get('review_origin_behavior') != 'Together'
                or not isinstance(item.get('review_origin_decision_updated_at'), str)):
            raise ValueError(f'contact-screen candidate no longer matches its Together source: {review_id}')
        if review_id in seen:
            raise ValueError(f'duplicate Together contact-screen candidate: {review_id}')
        seen.add(review_id)
        evidence = item.get('screening_evidence')
        if not isinstance(evidence, dict):
            raise ValueError(f'missing contact-screen evidence: {review_id}')
        suggestion = evidence.get('suggested_behavior')
        if suggestion is not None and suggestion not in contact_labels:
            raise ValueError(f'invalid contact-screen suggestion: {review_id}')
        export_path = item.get('export_path')
        target = safe_path(store.package / export_path, [store.package / 'review_sources'])
        if tree_hashes(target) != item.get('output_sha256'):
            raise ValueError(f'Together contact-screen media changed: {review_id}')
        card = copy.deepcopy(base)
        card.update(
            mouse_ids=list(item['mouse_ids']),
            start_frame=item['start_frame'],
            end_frame=item['end_frame'],
            export_path=export_path,
            poster_path=item['poster_path'],
            candidate_status='together_contact_geometry_prescreen',
            candidate_reason=item['candidate_reason'],
            review_from_behavior='Together',
            review_origin_behavior='Together',
            review_origin_decision_updated_at=item['review_origin_decision_updated_at'],
            review_default_behavior=suggestion or '',
            review_suggestion_label=(
                f'关键点建议：{suggestion}' if suggestion else '关键点初筛：类型待确认'
            ),
            review_display_behavior=(
                f'初筛建议 {suggestion}' if suggestion else '接触类型待确认'
            ),
            review_transfer_pending_label='待关键点初筛复核（Together未通过）',
        )
        cards.append(card)
    if len(cards) != document['candidate_count']:
        raise ValueError('contact-screen catalog contains missing candidates')
    return cards


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package-root', type=Path, required=True)
    parser.add_argument('--confirmed-root', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8768)
    args = parser.parse_args()
    store = ReviewStore(args.package_root, args.confirmed_root, args.source_root)
    server = None
    try:
        together_catalog = _together_review_catalog(store)
        huddling_recheck = huddling_recheck_catalog(
            store.base_cards, store._state().get('decisions', {}))
        approach_no_contact_catalog = _approach_no_contact_review_catalog(store)
        approach_no_contact_recrop_catalog = _approach_no_contact_recrop_review_catalog(store)
        approach_contact_catalog = _approach_contact_review_catalog(store)
        contact_reclassification_catalog = _together_contact_reclassification_catalog(store)
        render_review_page(store.package, store.rows, catalog=list(store.cards.values()),
                           candidate_sha256=store.candidate_sha,
                           supplemental_sha256=store.supplemental_sha,
                           together_review_available=bool(together_catalog),
                           approach_contact_review_available=bool(approach_contact_catalog),
                           huddling_recheck_review_available=bool(huddling_recheck),
                           approach_no_contact_review_available=bool(approach_no_contact_catalog),
                           approach_no_contact_recrop_review_available=bool(approach_no_contact_recrop_catalog),
                           contact_reclassification_review_available=bool(contact_reclassification_catalog))
        rearing_catalog = [card for card in store.supplemental_cards.values()
                           if card['behavior'] == 'Rearing']
        if rearing_catalog:
            render_review_page(
                store.package, store.rows, catalog=rearing_catalog,
                candidate_sha256=store.candidate_sha,
                supplemental_sha256=store.supplemental_sha,
                filename='rearing_review.html',
            )
        if together_catalog:
            render_review_page(
                store.package, store.rows, catalog=together_catalog,
                candidate_sha256=store.candidate_sha,
                supplemental_sha256=store.supplemental_sha,
                filename='together_review.html',
            )
        if huddling_recheck:
            render_review_page(
                store.package, store.rows, catalog=huddling_recheck,
                candidate_sha256=store.candidate_sha,
                supplemental_sha256=store.supplemental_sha,
                filename='huddling_recheck_review.html',
                page_title=f'扎堆复核（{len(huddling_recheck)}条）',
                page_intro=(
                    f'本页有 {len(huddling_recheck)} 条社会聚集未通过记录，原备注提到“扎堆”。'
                    '这些条目仅重新列为人工复核候选，原不通过决定保留；没有自动改标签或写入已通过数据集。'
                    '视频为现有原始样本片段，完整连续录像当前未提供。请核对实际行为、目标视觉 ID 和起止帧；'
                    '确认是扎堆后可保留 Huddling 并调整裁剪范围，也可选择其他实际行为。'
                ),
                page_rules=(
                    '复核状态在本页显示为待扎堆复核；只有点击“确认通过并同步”后，样本才进入对应行为目录并同步原始四类文件。'
                    '未完成确认的条目不会进入新数据集。'
                ),
            )
        if approach_contact_catalog:
            render_review_page(
                store.package, store.rows, catalog=approach_contact_catalog,
                candidate_sha256=store.candidate_sha,
                supplemental_sha256=store.supplemental_sha,
                filename='approach_contact_review.html',
            )
        if contact_reclassification_catalog:
            render_review_page(
                store.package, store.rows, catalog=contact_reclassification_catalog,
                candidate_sha256=store.candidate_sha,
                supplemental_sha256=store.supplemental_sha,
                filename='contact_reclassification_review.html',
            )
        if approach_no_contact_catalog:
            render_review_page(
                store.package, store.rows, catalog=approach_no_contact_catalog,
                candidate_sha256=store.candidate_sha,
                supplemental_sha256=store.supplemental_sha,
                filename='approach_no_contact_review.html',
                page_title='接近无身体接触重裁复核',
                page_intro='本页仅含从接近拒绝片段中重裁出的无身体接触候选。请对照完整原片与候选片段，逐段确认行为、视觉 ID 和起止帧；候选尚未自动通过。',
                page_rules='候选按当前接近规则截取至鼻头到对方身体表面约 3.5±0.5 cm，要求片段内无身体接触；不设置最短时长。只有人工确认后才写入新数据集并同步原始视频、轨迹、标注和 metadata。',
            )
        if approach_no_contact_recrop_catalog:
            render_review_page(
                store.package, store.rows, catalog=approach_no_contact_recrop_catalog,
                candidate_sha256=store.candidate_sha,
                supplemental_sha256=store.supplemental_sha,
                filename='approach_no_contact_recrop_review.html',
                approach_no_contact_review_available=bool(approach_no_contact_catalog),
                page_title=f'接近无接触重裁审核（{len(approach_no_contact_recrop_catalog)}条）',
                page_intro=(
                    f'本页共 {len(approach_no_contact_recrop_catalog)} 条待复核候选，按第一个视频的起点间距约 49.14 cm 作为参照。'
                    '仅使用现有源样本片段；完整连续录像当前未提供。并列视频供核对行为、目标视觉 ID 与起止帧，候选不是自动通过结果。'
                ),
                page_rules='本轮保留起点间距达到约 49.14 cm 的候选，并补入两条可由连续同源片段回补的候选；起点距离按来源样本的体长刻度计算。片段内不得出现身体接触或目标框重叠；不设最短时长。新补候选通过轨迹动态显示目标 ID 框，短暂插值框以虚线和 * 标记，tracks.json 保留原轨迹。只有人工确认后才会进入新数据集并同步原始视频、轨迹、标注和 metadata。',
            )
        server = create_server(store, args.port)
        logging.basicConfig(level=logging.INFO, format='%(message)s')
        logging.info('Manual review: http://127.0.0.1:%s/', server.server_port)
        logging.info('Approved dataset: %s', store.root/'dataset')
        logging.info('Original write-back: %s (verified backup before approval)', store.source_root)
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if server:
            server.server_close()
        store.close()


if __name__ == '__main__':
    main()
