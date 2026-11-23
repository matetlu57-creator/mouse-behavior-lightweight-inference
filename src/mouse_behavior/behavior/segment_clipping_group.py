"""Group/collective-layer behavior segment clipping (Huddling, Social clustering, Dispersal, Group locomotion, Isolation).

The implementation is separated by responsibility into:
- ``segment_clipping_group_dispersal``: dispersal segment clipping and origin-member recovery
- ``segment_clipping_group_clustering``: social clustering clipping, occlusion verification, and scene-cluster veto
- ``segment_clipping_group_motion``: huddling, group locomotion, and isolation segment clipping

Existing imports from ``mouse_behavior.behavior.segment_clipping_group`` remain supported.
"""
from __future__ import annotations

from mouse_behavior.behavior.segment_clipping_group_clustering import (
    _has_preexisting_scene_cluster_veto as _has_preexisting_scene_cluster_veto,
    _sc_occluded_members_verified as _sc_occluded_members_verified,
    refine_social_clustering_segments as refine_social_clustering_segments,
)
from mouse_behavior.behavior.segment_clipping_group_dispersal import (
    _recover_dispersal_origin_members as _recover_dispersal_origin_members,
    refine_dispersal_segments as refine_dispersal_segments,
)
from mouse_behavior.behavior.segment_clipping_group_motion import (
    refine_group_locomotion_segments as refine_group_locomotion_segments,
    refine_huddle_segments as refine_huddle_segments,
    refine_isolation_segments as refine_isolation_segments,
)

__all__ = [
    "refine_dispersal_segments",
    "refine_group_locomotion_segments",
    "refine_huddle_segments",
    "refine_isolation_segments",
    "refine_social_clustering_segments",
]
