"""A fitted rule tree must not pass on resubstitution accuracy alone."""

from mouse_behavior.evaluation.induced_rule_tree import passes_session_generalization_gate


def test_high_training_accuracy_does_not_unlock_validation_when_session_holdout_fails():
    # [STEP4 2026-10-04] 个体层合并：Walking+Running -> Locomotion
    training = {"per_behavior": {"approach": {"joint_top1_accuracy": 0.98},
                                 "locomotion": {"joint_top1_accuracy": 0.96}}}
    holdout = {"per_behavior": {"approach": {"joint_top1_accuracy": 0.25},
                                "locomotion": {"joint_top1_accuracy": 0.92}}}
    assert not passes_session_generalization_gate(training, holdout, 0.9)
    holdout["per_behavior"]["approach"]["joint_top1_accuracy"] = 0.91
    assert passes_session_generalization_gate(training, holdout, 0.9)
