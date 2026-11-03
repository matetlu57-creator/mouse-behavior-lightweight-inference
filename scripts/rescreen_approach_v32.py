"""Bounded derived-stage rescreen of frozen Approach parents; never full inference."""
from __future__ import annotations

import argparse
import copy
import json
import math
import tempfile
from pathlib import Path

import numpy as np

from _bootstrap import REPO_ROOT  # noqa: F401
from mouse_behavior.canonical_video_pipeline import TrackStore, rule_execution_binding, _runs
from mouse_behavior.evaluation.confirmed_rules import _open_approach_mask
from mouse_behavior.evaluation.exported_dataset_features import load_exported_features
from mouse_behavior.evaluation.exported_dataset_models import HeuristicParameters
from mouse_behavior.evaluation.segment_export import _sha256, _write_json
from mouse_behavior.evaluation.social_event_geometry import observed_body_separation, observed_path_and_net
from mouse_behavior.evaluation.social_heuristic_config import DEFAULT_SOCIAL_HEURISTIC_CONFIG as CONFIG
from mouse_behavior.visualization.behavior_packages import export_packages, write_manifests


def actor_candidate_mask(features, parent, params):
    """Choose the parent's measured actor direction, never the union of roles."""
    slots = {int(track): i for i, track in enumerate(features.track_ids)}
    actor, target = slots[parent['actor_id']], slots[parent['target_id']]
    edges = [i for i, (a, b) in enumerate(zip(features.pair_i, features.pair_j))
             if {int(a), int(b)} == {actor, target}]
    if len(edges) != 1:
        raise ValueError('parent pair must have exactly one feature edge')
    edge = edges[0]
    evidence = {}
    _open_approach_mask(features, edge, params, direction_evidence=evidence)
    direction = 0 if int(features.pair_i[edge]) == actor else 1
    return evidence['candidates'][direction], edge, actor, target


def clipped_child(parent, start, end):
    """Intersect every upstream support ledger with the new source bounds."""
    child = copy.deepcopy(parent)
    child.update(start_frame=int(start), end_frame=int(end), upstream_parent_event_id=parent['event_id'])
    for key, value in parent.items():
        if key.endswith('_frame_intervals'):
            spans = [(max(start, a), min(end, b)) for a, b in value if max(start, a) < min(end, b)]
            child[key] = spans
            child[key.replace('_intervals', '_count')] = sum(b-a for a, b in spans)
    child['inferred_or_partially_unobserved_frame_count'] = end-start-child.get('observed_all_participants_frame_count', end-start)
    child['event_id'] = f"{parent['event_id']}_v32_{start:08d}_{end:08d}"
    return child


def rescreen_parent(parent, store, fps, scale, params):
    start, end = parent['start_frame'], parent['end_frame']
    rows = store.window(start, end)
    if len(rows) != end-start or [row['frame'] for row in rows] != list(range(end-start)):
        raise ValueError('missing parent frames: refusing padding')
    # Only the source parent pair participates; other mice cannot lend roles.
    for row in rows:
        row['detections'] = [d for d in row['detections'] if d['track_id'] in parent['mouse_ids']]
    with tempfile.TemporaryDirectory(prefix='approach_v32_features_') as temporary:
        tracks = Path(temporary)/'tracks.json'
        _write_json(tracks, rows)
        f = load_exported_features(tracks, fps=fps, max_tracks=2,
            scale_config=dict(mode='fixed', cm_per_pixel=scale['cm_per_pixel'],
                              assumed_mouse_body_length_cm=scale['reference_body_cm']))
    if not {parent['actor_id'], parent['target_id']}.issubset(set(map(int, f.track_ids))):
        return []
    mask, edge, actor, target = actor_candidate_mask(f, parent, params)
    separated = observed_body_separation(f, edge, max_body_lengths=CONFIG.approach_max_footprint_body_lengths)
    children = []
    for low, high in _runs(mask):
        visible = f.valid[low:high, actor] & f.valid[low:high, target]
        path, net = observed_path_and_net(f.centers_cm[low:high, actor], visible)
        if (high-low)/fps <= CONFIG.approach_min_duration_s or path+1e-9 < CONFIG.approach_min_actor_path_cm:
            continue
        if not separated[low:high].all():
            raise ValueError('current Approach candidate contains unmeasured/contact geometry')
        child = clipped_child(parent, start+low, start+high)
        child.update(actor_observed_path_lower_bound_cm=float(path), actor_observed_net_cm=float(net),
                     derived_stage='v32_approach_parent_rescreen', known_separated_frame_count=high-low,
                     event_duration_s=(high-low)/fps, requires_review=True)
        children.append(child)
    return children


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--clip-context-seconds', type=float, default=0.)
    args = parser.parse_args()
    if not math.isfinite(args.clip_context_seconds) or args.clip_context_seconds < 0:
        raise ValueError('clip context seconds must be finite and nonnegative')
    if args.output.exists():
        raise FileExistsError(args.output)
    names = ['events.json', 'tracks.sqlite', 'input_binding.json', 'scale_receipt.json', 'run_receipt_compact.json']
    hashes = {name: _sha256(args.source/name) for name in names}
    def read(name):
        return json.loads((args.source/name).read_text(encoding='utf-8'))
    upstream, scale, receipt = read('input_binding.json'), read('scale_receipt.json'), read('run_receipt_compact.json')
    if receipt['status'] != 'complete':
        raise ValueError('upstream run is not complete')
    for name in names[:-1]:
        if receipt['payload_sha256'].get(name) != hashes[name]:
            raise ValueError(f'upstream payload mismatch: {name}')
    params = HeuristicParameters(**upstream['effective_parameters'])
    binding = {**upstream, **rule_execution_binding(params, 'disabled')}
    binding['derived_stage_provenance'] = dict(stage='bounded_approach_parent_rescreen',
        fresh_full_inference=False, source=str(args.source.resolve()), upstream_payload_sha256=hashes,
        rescreen_script_sha256=_sha256(Path(__file__)),
        playback_scope='pure_qualified_event' if args.clip_context_seconds == 0 else 'event_with_leading_context',
        upstream_rule_source_sha256=upstream.get('rule_source_sha256'))
    binding['screening_stage'] = copy.deepcopy(binding['derived_stage_provenance'])
    source_video = Path(binding['source_video']['path'])
    if _sha256(source_video) != binding['source_video']['sha256']:
        raise ValueError('source video hash mismatch')
    parents = [e for e in read('events.json') if e['canonical_behavior'] == 'approach']
    store, children, audit = TrackStore(args.source/'tracks.sqlite'), [], []
    fps = binding['source_video']['fps']
    args.output.mkdir(parents=True)
    for index, parent in enumerate(parents):
        kept = rescreen_parent(parent, store, fps, scale, params)
        children.extend(kept)
        audit.append(dict(parent_event_id=parent['event_id'], source_bounds=[parent['start_frame'],parent['end_frame']],
                          retained=[e['event_id'] for e in kept]))
        if index % 25 == 0:
            print(f'Approach parents {index+1}/{len(parents)}; retained {len(children)}', flush=True)
    children.sort(key=lambda e: (e['start_frame'],e['end_frame'],e['event_id']))
    selected = [children[int(i)] for i in np.linspace(0,len(children)-1,min(30,len(children)),dtype=int)] if children else []
    _write_json(args.output/'input_binding.json', binding)
    _write_json(args.output/'upstream_input_binding.json', upstream)
    _write_json(args.output/'scale_receipt.json', scale)
    _write_json(args.output/'events.json', children)
    _write_json(args.output/'selected_events.json', selected)
    _write_json(args.output/'parent_audit.json', audit)
    packages = export_packages(args.output/'packages', source_video, store, selected, binding, scale,
        workers=2, original_video=True, clip_codec='h264_nvenc',
        clip_context_seconds=args.clip_context_seconds) if selected else []
    write_manifests(args.output, packages)
    after = {name: _sha256(args.source/name) for name in names}
    source_hash = _sha256(source_video)
    if hashes != after or source_hash != binding['source_video']['sha256']:
        raise ValueError('frozen inputs changed during rescreen')
    _write_json(args.output/'run_receipt.json', dict(status='complete', stage='derived_stage_only',
        fresh_full_inference=False, parent_count=len(parents), retained_count=len(children), clip_count=len(packages),
        source_preserved_sha256=source_hash, upstream_payload_sha256=hashes,
        actor_path_minimum_cm=CONFIG.approach_min_actor_path_cm, duration_strictly_greater_than_s=CONFIG.approach_min_duration_s,
        clip_context_before_seconds=args.clip_context_seconds, clip_context_after_seconds=0., review_required=True,
        playback_scope='pure_qualified_event' if args.clip_context_seconds == 0 else 'event_with_leading_context',
        rejection_policy='Current endpoints/duration/path and observed body separation; no retuning',
        limitations=['Reuses upstream observed tracks and source-parent scope; not fresh full inference',
                     'No-contact qualification applies to true event bounds; leading playback context is not qualified event support',
                     'Visual track IDs are not biological identity; geometric heuristic candidates require human review']))
    print(json.dumps(dict(parents=len(parents),retained=len(children),clips=len(packages)),ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
