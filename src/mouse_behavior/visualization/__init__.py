"""Visualization boundaries for optional rendering and inspection outputs."""

from mouse_behavior.visualization.clip_overlay_renderer import (
    ROLE_COLORS,
    normalize_roles_map,
    render_annotated_clip_from_dir,
    render_annotated_original_video,
    render_cropped_comparison_video,
)
from mouse_behavior.visualization.comparison_site import (
    ComparisonServerHandler,
    create_comparison_server,
    render_comparison_page,
)

__all__ = [
    "ComparisonServerHandler",
    "ROLE_COLORS",
    "create_comparison_server",
    "normalize_roles_map",
    "render_annotated_clip_from_dir",
    "render_annotated_original_video",
    "render_comparison_page",
    "render_cropped_comparison_video",
]
