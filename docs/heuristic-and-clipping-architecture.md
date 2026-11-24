# Heuristic Recognition and Segment Clipping Modular Architecture

This note documents the responsibility split across the 15-behavior heuristic
recognition engine, the modular segment clipping pipeline, and the unified
[`mouse_behavior.heuristics`](../src/mouse_behavior/heuristics/__init__.py)
public facade.

All modules enforce the repository boundaries in [AGENTS.md](../AGENTS.md) and
[index.md](index.md): Ultralytics `YOLO.track()` owns `visual_track_id`,
mTrack/RFID mappings are the sole source of `animal_id`, missing identities or
endpoints are never inferred from spatial/kinematic cues, and `Together` vs.
contact labels never receive dual-label credit.

## 1. Heuristic Recognition Engine

### Configuration & Parameter Models

| Module | Primary Symbols | Responsibility |
| --- | --- | --- |
| [`behavior/individual_rules.py`](../src/mouse_behavior/behavior/individual_rules.py) | `IndividualHeuristicConfig`, `DEFAULT_INDIVIDUAL_CONFIG` | Era-invariant physical thresholds for individual motion and opt-in posture screens (`enable_posture_screen=False` by default). |
| [`evaluation/social_heuristic_config.py`](../src/mouse_behavior/evaluation/social_heuristic_config.py) | `SocialHeuristicConfig`, `DEFAULT_SOCIAL_HEURISTIC_CONFIG` | Versioned pair-layer distance, speed, acceleration, contact, and priority settings. |
| [`evaluation/group_heuristic_config.py`](../src/mouse_behavior/evaluation/group_heuristic_config.py) | `GroupHeuristicConfig`, `DEFAULT_GROUP_HEURISTIC_CONFIG` | Fixed collective geometry, stability, dispersal, synchronous motion, and isolation defaults. |
| [`evaluation/exported_dataset_models.py`](../src/mouse_behavior/evaluation/exported_dataset_models.py) | `HeuristicParameters`, `CANONICAL_BEHAVIORS`, `LABEL_TO_CANONICAL` | Tunable calibration parameters and the 15 retained canonical behavior mappings. |

### Recognition Layers

1. **Individual layer (`behavior/individual_rules.py`)**
   - `individual_rule_masks` returns an `IndividualRuleResult` covering
     `locomotion`, `static`, and opt-in `grooming` / `rearing` masks and
     diagnostics, delegating bounded stationary geometry to `behavior/static_rule.py`.
2. **Social / pair layer (`evaluation/confirmed_rules.py`)**
   - `pair_rule_masks`, `pair_rule_roles`, `policy_receipt`,
     `confirmed_profiles`, and `social_profiles` compute observed pair evidence,
     directed actor/target roles, and priority arbitration across `attack`,
     `chasing`, `approach`, `together`, `head_contact`, and `tail_contact`,
     supported by `social_event_geometry.py`, `pair_roles.py`, and
     `screening_signals.py`.
3. **Collective / group formation & rules**
   - [`evaluation/clustering_formation.py`](../src/mouse_behavior/evaluation/clustering_formation.py):
     `formation_masks` qualifies joining/convergence into dense formations and
     supports bounded top-down pile occlusion verification (`k >= 3`) while
     emitting only observed frames.
   - [`evaluation/group_event_geometry.py`](../src/mouse_behavior/evaluation/group_event_geometry.py):
     `_GroupProcess`, `group_separation`, `_components`, `_geometry`,
     `_inside_pile_envelope`, `_quiet_at`, `_huddle_geometry`, `_proposal_sets`,
     `_window_positions`, and `_region_gap`.
   - [`evaluation/group_event_clustering.py`](../src/mouse_behavior/evaluation/group_event_clustering.py):
     `_huddle_window` / `_huddle_event` / `_huddle_windows`,
     `_clustering_process` / `_clustering_event` (with `_members_in_preexisting_scene_cluster`
     and `_is_parallel_group_locomotion` vetoes), and `_synchronous_process` /
     `_synchronous_event`.
   - [`evaluation/group_event_dispersal.py`](../src/mouse_behavior/evaluation/group_event_dispersal.py):
     `dispersal_phase`, `_dispersal_process`, and departure-origin / attempt /
     velocity helpers.
   - [`evaluation/group_event_rules.py`](../src/mouse_behavior/evaluation/group_event_rules.py):
     Orchestrates `group_rule_candidates` (including `group_locomotion` vs.
     `social_clustering` deferral), `isolation_summaries`, and
     `isolation_candidates`, while re-exporting all symbols moved to
     `group_event_geometry`, `group_event_clustering`, and `group_event_dispersal`.
4. **Clip-level prediction & provenance (`evaluation/exported_dataset*.py`)**
   - `exported_dataset_prediction.py` (`predict_features`) aggregates all
     individual, pair, and group evidence for clip evaluation.
   - `exported_dataset.py` maintains `EXPORTED_DATASET_SOURCE_FILES` (strictly
     sorted) so provenance digests track every heuristic module, including
     `group_event_clustering.py`, `group_event_dispersal.py`, and
     `group_event_geometry.py`.

## 2. Segment Clipping Layers

| Module | Primary Symbols | Responsibility |
| --- | --- | --- |
| [`behavior/segment_clipping_common.py`](../src/mouse_behavior/behavior/segment_clipping_common.py) | `InMemoryTrackStore`, `MIN_PARTICIPANTS_BY_BEHAVIOR`, `MIN_CLIP_DURATION_S_BY_BEHAVIOR`, `MAX_DURATION_S_BY_BEHAVIOR`, `min_clip_duration_policy_for_behavior`, `qualify_clip_duration`, `endpoint_observations`, `endpoints_fully_observed`, `update_segment_bounds` | Shared track storage, authoritative 15-behavior participant and minimum/maximum clip duration policy tables, endpoint visibility checks, and boundary closure helpers. |
| [`behavior/segment_clipping_individual.py`](../src/mouse_behavior/behavior/segment_clipping_individual.py) | `refine_static_segments`, `refine_locomotion_segments`, `refine_rearing_segments`, `refine_grooming_segments` | Fine-cropping for single-mouse behaviors centered on measured kinematic/posture support. |
| [`behavior/segment_clipping_social_approach.py`](../src/mouse_behavior/behavior/segment_clipping_social_approach.py) | `refine_approach_segments` | Approach bout completion from distant non-contact approach up to initial box touch, crowd exclusion, final directionality checks, directed roles, and `>= 1.0 s` enforcement. |
| [`behavior/segment_clipping_social_attack_chasing.py`](../src/mouse_behavior/behavior/segment_clipping_social_attack_chasing.py) | `refine_attack_segments`, `discover_fight_ball_attack_episodes`, `refine_chasing_segments`, `discover_fast_chasing_candidates`, `discover_id_switch_chasing_candidates` | High-intensity clinch/fight-ball Attack (`> 0.5 s`) and high-speed direction-aligned Chasing (`> 0.7 s`) cropping and discovery with directed roles and zero sub-threshold fallbacks. |
| [`behavior/segment_clipping_social_contact.py`](../src/mouse_behavior/behavior/segment_clipping_social_contact.py) | `refine_contact_and_together_segments`, `attach_social_roles`, `_directed_head_distance_cm`, `_box_gap_cm` | Context-extended snout-head/rear contact (`> 0.4 s`) and Together (`>= 1.0 s`) bout completion with directed sniffer roles (`together` stays undirected) and observed endpoint gates. |
| [`behavior/segment_clipping_social.py`](../src/mouse_behavior/behavior/segment_clipping_social.py) | Re-export facade | Backward-compatible facade over the three `segment_clipping_social_*` submodules. |
| [`behavior/segment_clipping_group_dispersal.py`](../src/mouse_behavior/behavior/segment_clipping_group_dispersal.py) | `refine_dispersal_segments`, `_recover_dispersal_origin_members` | Recovers `>= 3` observed compact origin members, trims reconvergence/quiet tails, and enforces `> 5.0 s` minimum and `15.0 s` maximum duration. |
| [`behavior/segment_clipping_group_clustering.py`](../src/mouse_behavior/behavior/segment_clipping_group_clustering.py) | `refine_social_clustering_segments`, `_sc_occluded_members_verified`, `_has_preexisting_scene_cluster_veto` | Enforces `>= 3` participants, vetoes pre-existing scene clusters, supports bounded pile occlusion, trims idle prefixes, and enforces `> 3.0 s` minimum total duration (including post-arrival stable gathering hold capped at `3.0 s`). |
| [`behavior/segment_clipping_group_motion.py`](../src/mouse_behavior/behavior/segment_clipping_group_motion.py) | `refine_huddle_segments`, `refine_group_locomotion_segments`, `refine_isolation_segments` | Huddle sliding (`>= 5.0 s`, `<= 15.0 s`), synchronous-motion bout extraction for `group_locomotion` (`> 3.0 s`, `<= 9.0 s`), and `isolation` (`> 10.0 s`, `<= 20.0 s`) clipping. |
| [`behavior/segment_clipping_group.py`](../src/mouse_behavior/behavior/segment_clipping_group.py) | Re-export facade | Backward-compatible facade over the three `segment_clipping_group_*` submodules. |
| [`behavior/segment_clipping.py`](../src/mouse_behavior/behavior/segment_clipping.py) | `refine_behavior_segments`, `select_behavior_segments` | Dispatches all individual/social/group refiners, enforces `min_participants_for_behavior` and `qualify_clip_duration`, deduplicates with `strict_non_overlapping_shared_mouse=True`, and leaves `max_per_behavior=None` (unlimited). |

### Per-Behavior Clipping Duration & Unlimited Export Policy

All 15 retained canonical behaviors enforce an explicit minimum clip duration via `MIN_CLIP_DURATION_S_BY_BEHAVIOR` and `qualify_clip_duration()`, and default selectors (`select_behavior_segments`, `select_social_review_clips`, `select_group_review_clips`, `infer_all_behavior_events`, and long-video validation scripts) impose **no default output count cap** (`max_per_behavior=None` / `max_events_per_behavior=None` / `max_clips_per_behavior=None`):

| Layer | Canonical Behavior | Min Duration Gate | Max Duration Cap | Directed Role Attribution | Default Output Count Limit |
| --- | --- | --- | --- | --- | --- |
| Individual | `stationary` (`static`) | `>= 1.0 s` | — | N/A | Unlimited (`None`) |
| Individual | `locomotion` | `>= 0.5 s` | — | N/A | Unlimited (`None`) |
| Individual | `grooming` | `>= 0.3 s` | — | N/A | Unlimited (`None`) |
| Individual | `rearing` | `>= 0.5 s` | — | N/A | Unlimited (`None`) |
| Social | `approach` | `>= 1.0 s` | — | `approacher -> approached` | Unlimited (`None`) |
| Social | `attack` | `> 0.5 s` | — | `attacker -> victim` | Unlimited (`None`) |
| Social | `chase` (`chasing`) | `> 0.7 s` | — | `chaser -> chased` | Unlimited (`None`) |
| Social | `nose_head_contact` (`head_contact`) | `> 0.4 s` | — | `sniffer -> sniffed` | Unlimited (`None`) |
| Social | `nose_tail_contact` (`tail_contact`) | `> 0.4 s` | — | `sniffer -> sniffed` | Unlimited (`None`) |
| Social | `together` | `>= 1.0 s` | — | Undirected (`actor_id=-1`) | Unlimited (`None`) |
| Group | `huddle` | `>= 5.0 s` | `<= 15.0 s` | N/A (`>= 3` mice) | Unlimited (`None`) |
| Group | `social_clustering` | `> 3.0 s` (incl. stable gathering) | `3.0 s` post-arrival hold cap | N/A (`>= 3` mice) | Unlimited (`None`) |
| Group | `dispersal` | `> 5.0 s` | `<= 15.0 s` | N/A (`>= 3` mice) | Unlimited (`None`) |
| Group | `group_locomotion` | `> 3.0 s` | `<= 9.0 s` | N/A (`>= 3` mice) | Unlimited (`None`) |
| Group | `isolation` | `> 10.0 s` | `<= 20.0 s` | Target mouse | Unlimited (`None`) |

`canonical_video_pipeline.py` tracks all `segment_clipping_social_*`, `segment_clipping_group_*`, and
`EXPORTED_DATASET_SOURCE_FILES` modules inside `rule_execution_binding()` so
provenance receipts detect changes in any submodule.

## 3. Using the `mouse_behavior.heuristics` Facade

`mouse_behavior.heuristics` uses lazy `__getattr__` lookups over an explicit
`__all__` export list, so importing it adds no eager heavyweight dependencies or
import cycles while preserving `is` identity with the canonical defining
modules:

```python
from mouse_behavior.heuristics import (
    DEFAULT_GROUP_HEURISTIC_CONFIG,
    DEFAULT_INDIVIDUAL_HEURISTIC_CONFIG,
    DEFAULT_SOCIAL_HEURISTIC_CONFIG,
    HeuristicParameters,
    InMemoryTrackStore,
    group_rule_candidates,
    individual_rule_masks,
    pair_rule_masks,
    refine_behavior_segments,
    select_behavior_segments,
)
```

Existing deep imports from `mouse_behavior.behavior.*` and
`mouse_behavior.evaluation.*` remain fully supported for backward
compatibility and test monkeypatching.
