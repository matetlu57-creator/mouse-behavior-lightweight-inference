# Tracking boundary

The current lightweight tracker is still implemented inside
`mouse_behavior.lightweight_behavior_inference` because its position,
keypoint and cache-window operations share the same analysis arrays. This
directory is the stable boundary for the next extraction: association, motion,
occlusion and identity management will move here one component at a time,
with regression tests before each move.

Do not add a second `tracker_v2.py` or copy the current tracker into this
directory. Use a feature branch and preserve the old import facade while the
split is validated.
