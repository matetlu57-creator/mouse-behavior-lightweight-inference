"""Deprecated package alias for the renamed lightweight analysis module.

New code should import :mod:`mouse_behavior.lightweight_behavior_inference`.
The alias remains inside the package so older notebooks can migrate without a
repository-root Python shim.
"""

from __future__ import annotations

from ._compat import reexport


_IMPLEMENTATION = reexport("mouse_behavior.lightweight_behavior_inference", globals())
