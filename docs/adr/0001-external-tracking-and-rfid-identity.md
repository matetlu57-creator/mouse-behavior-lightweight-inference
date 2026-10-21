# ADR 0001: External visual tracking and RFID identity

- Status: accepted
- Date: 2026-08-31

## Context

The previous repository implemented a second identity tracker inside the behavior pipeline. Most of the code and configuration handled Hungarian assignment, pose/position recovery, appearance heuristics, bright-object detection, occlusion recovery, and long-term ReID. This duplicated functionality already available in Ultralytics and still could not establish true biological identity after visual ID switches or long occlusion. The experimental workflow requires RFID-backed identity compatible with mTrack.

## Decision

Ultralytics `YOLO.track()` is the only in-repository source of visual tracklets. Its ID is stored as `visual_track_id` in the Pose cache. mTrack or another external RFID integration owns sensor acquisition and visual-to-RFID association, and exports a small normalized CSV mapping. The behavior package consumes that mapping as authoritative `animal_id` evidence.

mTrack and this package may run in separate Python environments. This boundary is intentional: the bundled `Pose26` model requires the newer Ultralytics runtime pinned by this project, while an upstream RFID workflow may carry a different dependency set.

The behavior layer may choose among duplicate detections already mapped to the same animal, but it may not create or infer an identity. Unresolved RFID detections stay unresolved. Without an identity map, slots are bound directly to Ultralytics IDs and labeled as provisional visual identities.

## Consequences

- The full custom-tracking pipeline and its configuration are removed.
- Existing caches without `visual_track_id` are incompatible and must be rebuilt.
- Behavior outputs distinguish internal numeric slots, visual track IDs, and animal IDs.
- Upstream mTrack code and hardware access remain outside this repository; integration occurs through the documented CSV boundary.
- Validation now focuses on ID-switch handling, missing-map behavior, conflict rejection, and preservation of animal-labeled trajectories.
