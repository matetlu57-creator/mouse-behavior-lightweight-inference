# Changelog

## Unreleased

- Added Following events with explicit Follower/Leader roles, a 5–30 cm distance gate, coordinated direction, a three-second minimum, and suppression when chase/attack or leader-escape evidence is present.
- Added Group locomotion, Social clustering, and Dispersal events with stable group membership traces and minimum durations of 3, 5, and 10 seconds.
- Added Chinese labels, colors, priorities, focus aliases, and annotation-website export support for all four behaviors.
- Kept the new behavior rules downstream of identified trajectories; no visual tracking or identity-recovery logic was added.

- Replaced behavior-layer ID assignment with Ultralytics tracked pose caches.
- Added an external mTrack/RFID identity-map adapter and identity labels in behavior outputs.
- Removed the legacy full pipeline, heuristic bright-object/ReID tracking configuration, and custom Hungarian tracking path.
- Added trajectory-contract regression tests, repository validation, CI, and architecture documentation.
