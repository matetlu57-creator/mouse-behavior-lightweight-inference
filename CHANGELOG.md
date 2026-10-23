# Changelog

## Unreleased

- Requalify Approach after pursuit arbitration and retain independently qualified
  pursuit role changes; detect Dispersal from later observed group formations.
- Preserve endpoint-sensitive qualified events across video core boundaries.
- Seal lightweight video/cache/event inputs, reject drifting historical rule
  snapshots, and publish only verified frame-exact media with consistent crop
  annotations. Forward the measured body reference into pair features.
- Add cross-module regression reproductions; keep reviewed datasets, numeric
  thresholds and historical human-reviewed A1/A2 unchanged. See
  [the audit repair record](docs/behavior-boundary-media-audit-2026-10-23.md).

- Added Following events with explicit Follower/Leader roles, a 5–30 cm distance gate, coordinated direction, a three-second minimum, and suppression when chase/attack or leader-escape evidence is present.
- Added Group locomotion, Social clustering, and Dispersal events with stable group membership traces and minimum durations of 3, 5, and 10 seconds.
- Added Chinese labels, colors, priorities, focus aliases, and annotation-website export support for all four behaviors.
- Kept the new behavior rules downstream of identified trajectories; no visual tracking or identity-recovery logic was added.

- Replaced behavior-layer ID assignment with Ultralytics tracked pose caches.
- Added an external mTrack/RFID identity-map adapter and identity labels in behavior outputs.
- Removed the legacy full pipeline, heuristic bright-object/ReID tracking configuration, and custom Hungarian tracking path.
- Added trajectory-contract regression tests, repository validation, CI, and architecture documentation.
