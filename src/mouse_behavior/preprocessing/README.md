# Preprocessing boundary

Preprocessing and feature-construction responsibilities are split into focused
modules:

- `constants.py`: shared keypoint, skeleton and behavior-label constants;
- `geometry.py`: finite-point, vector, angle, cosine and IoU primitives;
- `kinematics.py`: smoothing, pose-deformation and individual motion features;
- `pair_features.py`: pair geometry, distance/heading prefiltering, padded
  valuable-frame windows and candidate-only pair metrics;
- `arena_learning.py`: per-video cage-boundary learning from model-independent
  YOLO cache records.

The lightweight path consumes completed Pose caches, so it does not duplicate
the upstream detector preprocessing. The compatibility entry point imports
these modules and keeps its historical callable names available for old
scripts and notebooks.
