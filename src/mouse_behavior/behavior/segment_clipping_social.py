"""Social/pair-layer behavior segment clipping (Approach, Attack, Chasing, Contact, Together).

The implementation is separated by responsibility into:
- ``segment_clipping_social_approach``: approach segment completion, duration enforcement, and source requalification
- ``segment_clipping_social_attack_chasing``: attack and chasing segment onset/continuation cropping and strict duration gates
- ``segment_clipping_social_contact``: snout-head/rear contact and together bout completion and duration gates

Existing imports from ``mouse_behavior.behavior.segment_clipping_social`` remain supported.
"""
from __future__ import annotations

from mouse_behavior.behavior.segment_clipping_social_approach import (
    refine_approach_segments as refine_approach_segments,
)
from mouse_behavior.behavior.segment_clipping_social_attack_chasing import (
    discover_fast_chasing_candidates as discover_fast_chasing_candidates,
    discover_fight_ball_attack_episodes as discover_fight_ball_attack_episodes,
    discover_id_switch_chasing_candidates as discover_id_switch_chasing_candidates,
    refine_attack_segments as refine_attack_segments,
    refine_chasing_segments as refine_chasing_segments,
)
from mouse_behavior.behavior.segment_clipping_social_contact import (
    _box_gap_cm as _box_gap_cm,
    _box_iou as _box_iou,
    _directed_head_distance_cm as _directed_head_distance_cm,
    attach_social_roles as attach_social_roles,
    build_social_roles as build_social_roles,
    refine_contact_and_together_segments as refine_contact_and_together_segments,
)

__all__ = [
    "attach_social_roles",
    "build_social_roles",
    "discover_fast_chasing_candidates",
    "discover_fight_ball_attack_episodes",
    "discover_id_switch_chasing_candidates",
    "refine_approach_segments",
    "refine_attack_segments",
    "refine_chasing_segments",
    "refine_contact_and_together_segments",
]
