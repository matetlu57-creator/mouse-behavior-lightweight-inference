"""Trajectory adapters for tracker-produced multi-mouse pose records."""

from .cache import TrajectoryCacheError, load_trajectory_cache

__all__ = ["TrajectoryCacheError", "load_trajectory_cache"]
