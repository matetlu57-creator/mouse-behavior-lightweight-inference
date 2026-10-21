"""Attack30 is a scoring choice, independent of action detection and other labels."""
import sys

import pytest

from scripts import evaluate_active_dataset_v17 as evaluation


def test_attack_coverage_defaults_to_30_percent_without_changing_other_behaviors():
    args = evaluation.build_parser().parse_args(['--output-dir', 'out'])
    assert args.threshold == .5
    assert args.attack_threshold == .3
    assert evaluation.coverage_acceptance_threshold('attack', args.threshold, args.attack_threshold) == .3
    for name in ('nose_head_contact', 'nose_tail_contact', 'chase', 'together', 'dispersal'):
        assert evaluation.coverage_acceptance_threshold(name, args.threshold, args.attack_threshold) == .5


def test_coverage_boundary_changes_score_only_and_preserves_predictions():
    candidates = [{'target_ids': [1, 2], 'coverage': .30, 'evidence_frames': 3}]
    old = evaluation.score_targets(candidates, [1, 2], .5)
    new = evaluation.score_targets(candidates, [1, 2], .3)
    assert not old['a2_hit'] and new['a2_hit']
    assert old['coverage_on_true_target'] == new['coverage_on_true_target'] == .3
    assert candidates == [{'target_ids': [1, 2], 'coverage': .3, 'evidence_frames': 3}]
    assert not evaluation.score_targets([{'target_ids': [1, 2], 'coverage': .29999}], [1, 2], .3)['a2_hit']


@pytest.mark.parametrize('bad', ['0', '-.1', '1.1', 'nan', 'inf'])
def test_invalid_attack_threshold_stops_before_dataset_work(monkeypatch, bad):
    monkeypatch.setattr(sys, 'argv', ['evaluation', '--output-dir', 'out', '--attack-threshold', bad])
    monkeypatch.setattr(evaluation, 'evaluate', lambda args: pytest.fail('invalid threshold reached evaluation'))
    with pytest.raises(SystemExit) as exc:
        evaluation.main()
    assert exc.value.code == 2
