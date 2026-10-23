# Architecture and data contracts

## Responsibility boundary

See the [2026-10-23 boundary and media audit repair](behavior-boundary-media-audit-2026-10-23.md)
for event arbitration, role changes, late group origins, sealed media inputs,
recropping, historical rerender compatibility and validation limits.

The system has three explicit layers:

1. Ultralytics reads video frames and emits Pose detections with a persistent `visual_track_id`.
2. An external mTrack/RFID process associates visual tracklets with stable `animal_id` values and exports the normalized CSV contract.
3. `mouse_behavior` validates those inputs, constructs trajectory arrays, computes kinematics, and emits behavior events.

The behavior layer must not assign identities from spatial order, nearest position, coat brightness, pose similarity, or motion. Visual-only analysis is supported for diagnostics and is marked provisional.

## Tracked Pose cache

Each gzip-pickled cache chunk contains a list of frame records. A record has:

- `frame`: zero-based source-video frame index;
- `timestamp_seconds`: source-video timestamp;
- `pose_detections`: tracked Pose detections.

Each usable detection contains `visual_track_id`, `bbox_xyxy`, `box_conf`, seven `keypoints_px`, seven `keypoint_conf`, and `pose_quality`. `yolo_results_status.json` must declare `status=complete`, a matching `total_frames`/`next_frame`, and `trajectory_schema_version >= 1`. Older caches without visual IDs are rejected and must be rebuilt.

## Identity CSV

The adapter accepts exact-frame rows or inclusive frame intervals. Required concepts are a non-negative visual track ID, a non-empty animal ID, and either one frame or a start/end range. Confidence is optional and defaults to 1.0. Conflicting overlapping mappings are rejected.

In RFID mode, a detection without a matching assignment is counted as unresolved and excluded. If two visual detections map to the same animal in one frame, only the highest confidence-weighted detection is retained and the conflict is reported in trajectory statistics.

## Internal arrays and outputs

Vectorized behavior algorithms use integer slot indices for array efficiency. `tracks.identity_labels[slot]` is the authoritative label for that slot. The arrays also expose frame-level `visual_track_ids` and `identity_confidence`, plus `visual_track_ids_by_slot` for auditability.

Event tables add label fields next to numeric actor/target/contact/member slots. Website track export contains `track_id` (array slot), `visual_track_id`, `animal_id`, `identity_source`, and `identity_confidence`. Consumers must use `animal_id` for biological identity and treat `track_id` only as an internal row/slot key.

## Modules

- `pose_cache.py`: sequential Ultralytics `model.track(..., persist=True)` cache creation.
- `tracking/cache.py`: validation and trajectory-array normalization; no tracker implementation.
- `identity/mtrack.py`: normalized mTrack/RFID CSV adapter.
- `identity/labels.py`: label annotation for behavior records.
- `behavior/` and `preprocessing/`: behavior features, FSMs, and event extraction.
- `evaluation/exported_dataset.py` is the stable compatibility façade for exported
  dataset analysis. Its implementation is separated into `exported_dataset_models.py`,
  `exported_dataset_records.py`, `exported_dataset_features.py`,
  `exported_dataset_temporal.py`, `exported_dataset_collective.py`,
  `exported_dataset_prediction.py`, `exported_dataset_metrics.py`, and
  `exported_dataset_config.py`; provenance hashes cover the façade and its static
  local dependency closure, including current pair roles, group formation,
  individual rules and scale resolution. A regression checks coverage whenever
  a new local dependency is introduced.
  Its historical target-ID hit metrics are conditional on annotated IDs, not joint Top-1.
- `evaluation/blind_top1.py` and `evaluation/train_only_tuning.py`: independent
  behavior-plus-visual-ID prediction, exact scoring, and train-session-only
  parameter optimization. See [the blind calibration protocol](train-only-blind-calibration.md).
  This diagnostic path does not replace the production Ultralytics + mTrack/RFID flow.
- `evaluation/calibrated_rule_ranker.py` and `scripts/calibrate_existing_split.py`:
  supervised calibration of weighted heuristic trajectory evidence on the existing
  training split, followed by frozen-model evaluation on the existing validation
  split. No extra test set or training-accuracy gate is required. Training-internal
  recording holdouts select regularization; validation labels only score locked predictions.
- [`behavior-definitions.md`](behavior-definitions.md): operational definitions, duration gates, roles, and output semantics for the extended ethogram.
- `visualization/`: rendering and annotation-package export.

## Independent posture review and standalone video packages

`behavior/posture_measurements.py`, `posture_rules.py`, and `posture_adapter.py`
separate shared pixel measurements from independent Grooming/Rearing evidence,
candidate events and conflict resolution. Rearing includes wall-supported
standing. Both classes remain suspended in default canonical prediction;
`--posture-mode v3_review` explicitly enables unconfirmed review candidates.

`canonical_video_pipeline.py` preserves raw visual IDs and binds video/cache
provenance. `behavior/duration_policy.py` resolves effective duration gates;
`visualization/behavior_packages.py` verifies standalone intervals before export
and again after decoding. Video, annotation, metadata and tracks share the same
half-open source interval. Legacy fixed-window showcase clips are explicitly
distinguished from qualified standalone behavior events.

See [the implementation and merge record](posture-v3-duration-review-2026-10-21.md)
and [the architecture decision](adr/2026-10-21-independent-posture-and-standalone-duration.md).
The repository CI executes complete quality gates on Python 3.10 and 3.11 under
the existing required check names. CI success does not validate scientific
classification accuracy or alter human-reviewed A1/A2 results.

RFID analysis now seals the exact mapping CSV bytes as `identity_map_sha256`.
Re-rendering checks this digest, source and slot labels before publishing; legacy
RFID metadata without the digest requires re-analysis. Visual-only website exports
retain raw visual IDs and emit no biological `animal_id`. All group review approvals
require at least three observed IDs. The historical viewer's correct/error navigation
uses strict matches; old diagnostic values never grant current credit. See the
[pre-merge audit and migration notes](posture-v3-duration-review-2026-10-21.md#父-pr-16-合并前补查与修复)
and [explicit frozen-media packaging](../tools/video_review_viewer/README.md).

Historical blind inference checkpoints seal the frozen model, provenance, exact
manifest, trajectory contents, implementation and flushed evidence prefix. Resume
requires `--historical-model`; old unsealed partial runs must restart. Split-based
scoring verifies the manifest receipt before label joining. Non-isolation blind
group candidates need at least three target IDs; this does not change the
lower-level two-leavers-from-a-group process condition. Tracked private data and
credential filenames are rejected by the repository checker as well as ignored.
