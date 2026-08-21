# Tracking boundary

Tracking/cache responsibilities are implemented in
`mouse_behavior.tracking.cache`:

- cache record iteration and frame-count discovery;
- detection payload normalization and duplicate suppression;
- arena-membership checks;
- position/keypoint assignment and lightweight track-cache construction.

`mouse_behavior.lightweight_behavior_inference` remains the compatibility
orchestrator. It re-exports the historical private helper names, while new
code should import tracking helpers from `tracking.cache` only when it needs
that lower-level boundary.

Do not add a second `tracker_v2.py` or copy the tracker into another directory.
Future identity/occlusion changes should be made behind this boundary and
protected by regression tests before changing the public facade.
