from __future__ import annotations

import numpy as np

from mouse_behavior.parallel_behavior_fsm import (
    BooleanBehaviorFSM,
    CategoricalBehaviorFSM,
    ParallelBehaviorFSM,
)


def test_boolean_fsm_preserves_gap_fill_and_minimum_duration_semantics():
    # The false frame in the middle is a one-frame recovery gap.  The two
    # frames at either edge stay outside the event, just as the legacy helper
    # did, so the finalized span is exactly frames 2..8.
    raw = np.array([False, False, True, True, False, True, True, True, True, False])
    result = BooleanBehaviorFSM(min_duration_frames=4, max_gap_frames=1).run(raw)

    assert [(span.start, span.end) for span in result.spans] == [(2, 8)]
    np.testing.assert_array_equal(
        result.active_mask,
        [False, False, True, True, True, True, True, True, True, False],
    )
    assert result.state[4] == "RECOVERY"
    assert result.state[2] == "ACTIVE"


def test_boolean_fsm_rejects_short_candidate_after_gap_processing():
    raw = np.array([False, True, False, False, True, False])
    result = BooleanBehaviorFSM(min_duration_frames=3, max_gap_frames=1).run(raw)

    assert result.spans == ()
    assert not bool(result.active_mask.any())
    assert result.state[1] == "CANDIDATE"


def test_parallel_regions_can_be_active_at_the_same_frame():
    coordinator = ParallelBehaviorFSM({"mode": "active", "collect_diagnostics": True})
    chase = coordinator.run_boolean_region(
        scope="pair",
        region_id="1_2",
        behavior="chase",
        mask=np.array([True, True, True]),
        min_duration_frames=1,
        max_gap_frames=0,
    )
    contact = coordinator.run_boolean_region(
        scope="contact",
        region_id="1_2",
        behavior="nose_head_contact",
        mask=np.array([False, True, True]),
        min_duration_frames=1,
        max_gap_frames=0,
    )

    assert bool(chase.active_mask[1])
    assert bool(contact.active_mask[1])
    assert len(coordinator.regions) == 2


def test_categorical_fsm_keeps_contact_direction_and_geometry_as_state_key():
    states = [
        {"type": "nose_head", "direction": "a_to_b"},
        {"type": "nose_head", "direction": "a_to_b"},
        {"type": "nose_tail", "direction": "a_to_b"},
        None,
        {"type": "nose_tail", "direction": "b_to_a"},
    ]
    result = CategoricalBehaviorFSM().run(
        states,
        state_key=lambda value: None
        if value is None
        else (value["type"], value["direction"]),
    )

    assert [(span.start, span.end, span.key) for span in result.spans] == [
        (0, 1, ("nose_head", "a_to_b")),
        (2, 2, ("nose_tail", "a_to_b")),
        (4, 4, ("nose_tail", "b_to_a")),
    ]
