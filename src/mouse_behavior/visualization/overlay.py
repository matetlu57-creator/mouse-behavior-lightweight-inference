"""Behavior-aware text and hierarchy overlays for rendered videos.

The inference events remain unchanged.  This module only converts persisted
events into a compact, human-readable overlay for visual review.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..preprocessing.constants import (
    BEHAVIOR_NAMES_ZH,
    GROUP_BEHAVIORS,
    INDIVIDUAL_BEHAVIORS,
    SOCIAL_BEHAVIORS,
)


CONTACT_ALIASES = {
    "nose_head": "nose_head_contact",
    "nose_tail": "nose_tail_contact",
}
SOCIAL_DISPLAY_BEHAVIORS = frozenset(SOCIAL_BEHAVIORS) | frozenset(CONTACT_ALIASES)
GROUP_DISPLAY_BEHAVIORS = frozenset(GROUP_BEHAVIORS)
INDIVIDUAL_DISPLAY_BEHAVIORS = frozenset(INDIVIDUAL_BEHAVIORS)

DISPLAY_NAMES_ZH = {
    **BEHAVIOR_NAMES_ZH,
    "nose_head": "鼻头接触",
    "nose_tail": "鼻尾接触",
    "nose_head_contact": "鼻头接触",
    "nose_tail_contact": "鼻尾接触",
}

FOCUS_BEHAVIORS = frozenset({"chase", "avoidance", "attack"})
FOCUS_NAMES_ZH = {
    "chase": "追逐/被追逐",
    "avoidance": "回避/被回避",
    "attack": "攻击/被攻击",
}

ROLE_NAMES_ZH = {
    "approach": {"actor": "主动接近", "target": "被接近", "pair": "接近"},
    "chase": {"actor": "追逐", "target": "被追逐", "pair": "追逐"},
    "avoidance": {"actor": "回避", "target": "被回避", "pair": "回避"},
    "attack": {"actor": "攻击", "target": "被攻击", "pair": "攻击"},
}

# OpenCV uses BGR; the Pillow drawing helper below converts these values to
# RGB only at the final text-rendering step.
DEFAULT_TRACK_COLOR_BGR = (80, 220, 80)
UNKNOWN_BEHAVIOR_COLOR_BGR = (170, 170, 170)
EVENT_COLORS_BGR = {
    "together": (80, 190, 190),
    "approach": {"actor": (0, 165, 255), "target": (255, 200, 80), "pair": (80, 190, 255)},
    "chase": {"actor": (0, 215, 255), "target": (255, 140, 0), "pair": (80, 190, 255)},
    "avoidance": {"actor": (255, 220, 0), "target": (200, 80, 220), "pair": (220, 160, 0)},
    "attack": {"actor": (0, 0, 255), "target": (255, 0, 255), "pair": (120, 80, 255)},
    "nose_head": (80, 220, 120),
    "nose_tail": (220, 160, 60),
    "nose_head_contact": (80, 220, 120),
    "nose_tail_contact": (220, 160, 60),
    "huddle": (180, 120, 255),
    "isolation": (180, 120, 255),
    "running": (80, 220, 80),
    "walking": (0, 200, 255),
    "stationary": (180, 180, 180),
}

CATEGORY_NAMES_ZH = {
    "social": "社交行为",
    "group": "群体行为",
    "social_group": "社交行为/群体行为",
    "mixed": "高层行为+个体行为",
    "individual": "个体行为",
    "none": "个体行为",
}

# A frame can legitimately contain several orthogonal event streams.  The
# renderer therefore resolves conflicts per mouse rather than hiding every
# individual event whenever another mouse is in a social/group event.
# Semantic priority is intentionally independent of the numerical confidence
# score: a contact score of 1.0 must not overwrite a high-confidence attack.
DISPLAY_PRIORITY = {
    "attack": 100,
    "chase": 90,
    "avoidance": 80,
    "approach": 70,
    "together": 60,
    "nose_head_contact": 50,
    "nose_tail_contact": 50,
    "huddle": 40,
    "isolation": 40,
    "running": 30,
    "walking": 20,
    "stationary": 10,
}


def format_mouse_id(mouse_id: int) -> str:
    """Format a stable two-digit ID for boxes and the side behavior panel."""

    value = int(mouse_id)
    return f"ID {value:02d}" if value >= 0 else f"ID {value}"


@dataclass(frozen=True)
class MouseOverlay:
    """Text and box style for one tracked mouse in one rendered frame."""

    text: str
    color_bgr: tuple[int, int, int]
    priority: tuple[int, float]


@dataclass(frozen=True)
class BoxLabel:
    """A label anchored to one already-rendered bounding box."""

    bbox: tuple[int, int, int, int]
    text: str
    color_bgr: tuple[int, int, int]


def canonical_behavior(value: Any) -> str:
    """Return one display behavior name without mutating the source event."""

    behavior = str(value or "").strip().lower()
    return CONTACT_ALIASES.get(behavior, behavior)


def normalize_focus_behavior(value: Any) -> str | None:
    """Normalize an externally supplied render focus without changing inference.

    The Beiyi validation manifest may provide the expected folder label to the
    renderer.  This value controls only the persistent review heading; event
    CSVs and behavior decisions never read it.
    """

    aliases = {
        "追逐": "chase",
        "追逐/被追逐": "chase",
        "追逐-被追逐": "chase",
        "回避": "avoidance",
        "回避/被回避": "avoidance",
        "回避-被回避": "avoidance",
        "攻击": "attack",
        "攻击行为": "attack",
    }
    raw = str(value or "").strip().lower()
    behavior = aliases.get(raw, canonical_behavior(raw))
    return behavior if behavior in FOCUS_BEHAVIORS else None


def focus_behavior_name_zh(value: Any) -> str:
    """Return the stable Chinese display name for a render focus."""

    behavior = normalize_focus_behavior(value)
    return FOCUS_NAMES_ZH.get(behavior or "", str(value or "重点行为"))


def event_category(event: Mapping[str, Any]) -> str | None:
    behavior = canonical_behavior(event.get("behavior"))
    if behavior in SOCIAL_DISPLAY_BEHAVIORS or behavior in {
        "nose_head_contact",
        "nose_tail_contact",
    }:
        return "social"
    if behavior in GROUP_DISPLAY_BEHAVIORS:
        return "group"
    if behavior in INDIVIDUAL_DISPLAY_BEHAVIORS:
        return "individual"
    return None


def normalize_contact_events(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Adapt contact CSV rows to the renderer's common event shape."""

    normalized: list[dict[str, Any]] = []
    for row in rows:
        contact_type = canonical_behavior(row.get("contact_type"))
        behaviors = (
            ("nose_head_contact", "nose_tail_contact")
            if contact_type == "nose_head_and_nose_tail"
            else (contact_type,)
        )
        for behavior in behaviors:
            if behavior not in {"nose_head_contact", "nose_tail_contact"}:
                continue
            event = dict(row)
            event.update(
                {
                    "behavior": behavior,
                    "behavior_name_zh": DISPLAY_NAMES_ZH[behavior],
                    "candidate_level": "extended",
                    "event_scope": "pair",
                    "actor_id": row.get("contact_actor_id", -1),
                    "target_id": row.get("contact_target_id", -1),
                    "role_ambiguous": row.get("role_ambiguous", False),
                    "peak_score": row.get("peak_score", 1.0),
                }
            )
            normalized.append(event)
    return normalized


def _safe_int(value: Any, default: int = -1) -> int:
    try:
        result = int(float(value))
    except (TypeError, ValueError, OverflowError):
        return default
    return result


def pair_ids(value: Any) -> tuple[int, ...]:
    """Extract non-negative mouse IDs from keys such as ``0_1`` or ``mouse_3``."""

    found: list[int] = []
    for raw in re.findall(r"-?\d+", str(value or "")):
        number = _safe_int(raw)
        if number >= 0 and number not in found:
            found.append(number)
    return tuple(found)


def event_score(event: Mapping[str, Any]) -> float:
    for key in ("peak_score", "mean_score", "mean_behavior_confidence"):
        try:
            score = float(event.get(key, 0.0))
        except (TypeError, ValueError):
            continue
        if np.isfinite(score):
            return score
    return 0.0


def _deduplicate(events: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    best: dict[tuple[str, str, str, int, int], dict[str, Any]] = {}
    for source in events:
        event = dict(source)
        behavior = canonical_behavior(event.get("behavior"))
        category = event_category({"behavior": behavior})
        if category is None:
            continue
        event["_display_behavior"] = behavior
        key = (
            category,
            behavior,
            str(event.get("pair_key", "")),
            _safe_int(event.get("actor_id")),
            _safe_int(event.get("target_id")),
        )
        previous = best.get(key)
        if previous is None or event_score(event) > event_score(previous):
            best[key] = event
    return list(best.values())


def _display_priority(event: Mapping[str, Any]) -> int:
    return int(DISPLAY_PRIORITY.get(canonical_behavior(event.get("behavior")), 0))


def select_display_events(
    active_events: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    """Select frame events while preserving orthogonal behavior channels.

    Social/group events are high-level only for the mice that participate in
    them.  Individual events for other tracked mice remain visible, so a
    nearby attack does not turn the other 18 mice into an uninformative
    ``仅追踪`` label.
    """

    events = _deduplicate(active_events)
    selected = events
    categories = {event_category(event) for event in events if event_category(event) is not None}
    high_level = categories.intersection({"social", "group"})
    if high_level and "individual" in categories:
        layer = "mixed"
    elif high_level:
        layer = "social_group" if len(high_level) == 2 else next(iter(high_level))
    elif "individual" in categories:
        layer = "individual"
    else:
        layer = "none"
    selected.sort(
        key=lambda event: (
            -_display_priority(event),
            -event_score(event),
            int(event.get("start_frame", 0) or 0),
        )
    )
    return selected, layer


def _event_role(event: Mapping[str, Any], mouse_id: int) -> str | None:
    actor = _safe_int(event.get("actor_id"))
    target = _safe_int(event.get("target_id"))
    if actor == mouse_id:
        return "actor"
    if target == mouse_id:
        return "target"
    return None


def _event_ids(event: Mapping[str, Any]) -> tuple[int, ...]:
    ids: list[int] = []
    for value in (event.get("actor_id"), event.get("target_id")):
        number = _safe_int(value)
        if number >= 0 and number not in ids:
            ids.append(number)
    for number in pair_ids(event.get("pair_key")):
        if number not in ids:
            ids.append(number)
    # Group events produced by the extended ethogram carry their actual
    # participants.  ``pair_ids`` also handles CSV round-trips such as
    # ``"[1, 4, 9]"`` and ``"1;4;9"`` without introducing a second schema.
    for field in ("member_ids", "member_ids_at_peak"):
        for number in pair_ids(event.get(field)):
            if number not in ids:
                ids.append(number)
    return tuple(ids)


def _event_color(behavior: str, role: str | None = None) -> tuple[int, int, int]:
    value = EVENT_COLORS_BGR.get(behavior, DEFAULT_TRACK_COLOR_BGR)
    if isinstance(value, dict):
        return tuple(value.get(role or "pair", value.get("pair", DEFAULT_TRACK_COLOR_BGR)))
    return tuple(value)


def _event_label(event: Mapping[str, Any], mouse_id: int) -> str:
    behavior = canonical_behavior(event.get("behavior"))
    role = _event_role(event, mouse_id)
    role_names = ROLE_NAMES_ZH.get(behavior)
    if role_names and role in {"actor", "target"}:
        return role_names[role]
    return DISPLAY_NAMES_ZH.get(behavior, behavior or "未判定")


def build_mouse_overlays(
    display_events: Sequence[Mapping[str, Any]],
    layer: str,
    valid_ids: Iterable[int],
) -> dict[int, MouseOverlay]:
    """Build an ID+behavior label for every valid track in the frame."""

    ids = tuple(sorted({int(value) for value in valid_ids if int(value) >= 0}))
    overlays = {
        mouse_id: MouseOverlay(
            text=f"{format_mouse_id(mouse_id)}｜仅追踪",
            color_bgr=UNKNOWN_BEHAVIOR_COLOR_BGR,
            priority=(0, 0.0),
        )
        for mouse_id in ids
    }
    for event in display_events:
        behavior = canonical_behavior(event.get("behavior"))
        category = event_category(event)
        if category is None:
            continue
        score = event_score(event)
        event_ids = set(_event_ids(event))
        if category == "group":
            # Do not invent participants for a group event.  New analysis
            # rows contain member_ids; legacy CSVs without that field remain
            # visible in the sidebar summary but leave per-mouse labels intact.
            group_ids = set()
            for field in ("member_ids", "member_ids_at_peak"):
                group_ids.update(pair_ids(event.get(field)))
            for mouse_id in ids:
                if mouse_id not in group_ids:
                    continue
                candidate = MouseOverlay(
                    text=f"{format_mouse_id(mouse_id)}｜群体：{DISPLAY_NAMES_ZH.get(behavior, behavior)}",
                    color_bgr=_event_color(behavior),
                    priority=(_display_priority(event), score),
                )
                if candidate.priority > overlays[mouse_id].priority:
                    overlays[mouse_id] = candidate
            continue

        for mouse_id in event_ids.intersection(ids):
            role = _event_role(event, mouse_id)
            label = _event_label(event, mouse_id)
            candidate = MouseOverlay(
                text=f"{format_mouse_id(mouse_id)}｜{label}",
                color_bgr=_event_color(behavior, role),
                priority=(_display_priority(event), score),
            )
            if candidate.priority > overlays[mouse_id].priority:
                overlays[mouse_id] = candidate
    return overlays


def _summary_ids(event: Mapping[str, Any]) -> str:
    actor = _safe_int(event.get("actor_id"))
    target = _safe_int(event.get("target_id"))
    if actor >= 0 and target >= 0:
        return f"ID{actor}→ID{target}"
    ids = _event_ids(event)
    if ids:
        return "+".join(f"ID{item}" for item in ids)
    return "群体"


def event_summary(event: Mapping[str, Any]) -> str:
    behavior = canonical_behavior(event.get("behavior"))
    role_names = ROLE_NAMES_ZH.get(behavior)
    if (
        role_names
        and _safe_int(event.get("actor_id")) >= 0
        and _safe_int(event.get("target_id")) >= 0
    ):
        return f"{role_names['actor']} {_summary_ids(event)}"
    return f"{DISPLAY_NAMES_ZH.get(behavior, behavior or '未判定')} {_summary_ids(event)}"


def _truncate(text: str, max_chars: int = 56) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def build_panel_lines(display_events: Sequence[Mapping[str, Any]], layer: str) -> list[str]:
    """Return at most two compact Chinese lines for the top-left status tag."""

    first = f"行为层级：{CATEGORY_NAMES_ZH.get(layer, layer)}"
    if layer == "mixed":
        first += "（按参与ID分别显示）"
    if display_events:
        second = "；".join(event_summary(event) for event in display_events[:5])
        if len(display_events) > 5:
            second += f"；另有 {len(display_events) - 5} 项"
        return [first, _truncate(second)]
    return [first, "当前帧无已判定事件，框上显示“仅追踪”"]


def resolve_font_path(font_path: Path | None = None) -> Path | None:
    candidates = []
    if font_path is not None:
        candidates.append(Path(font_path))
    configured = os.environ.get("MOUSE_BEHAVIOR_FONT_PATH", "").strip()
    if configured:
        candidates.append(Path(configured))
    candidates.extend(
        [
            Path(r"C:\Windows\Fonts\msyh.ttc"),
            Path(r"C:\Windows\Fonts\simhei.ttf"),
            Path(r"C:\Windows\Fonts\simsun.ttc"),
            Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
            Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
        ]
    )
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def load_font(font_path: Path | None, font_size: int):
    resolved = resolve_font_path(font_path)
    if resolved is None:
        return ImageFont.load_default()
    return ImageFont.truetype(str(resolved), size=max(int(font_size), 10))


def font_size_for_frame(width: int, height: int) -> int:
    return max(12, min(24, int(round(min(width, height) / 80))))


def sidebar_width_for_frame(width: int, height: int) -> int:
    """Return a readable right-panel width without shrinking source imagery."""

    del height  # Kept in the signature so callers can use frame dimensions.
    panel_width = max(420, min(560, int(round(int(width) * 0.24))))
    return panel_width if panel_width % 2 == 0 else panel_width + 1


def _rgb_from_bgr(color_bgr: tuple[int, int, int]) -> tuple[int, int, int]:
    return int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0])


def draw_text_overlay(
    frame: np.ndarray,
    box_labels: Sequence[BoxLabel],
    panel_lines: Sequence[str],
    font: Any,
) -> np.ndarray:
    """Draw Chinese text with Pillow while preserving the OpenCV frame API."""

    height, width = frame.shape[:2]
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
    underlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    underlay_draw = ImageDraw.Draw(underlay)
    draw = ImageDraw.Draw(image)
    padding = 5
    header_bottom = 0
    if panel_lines:
        text_boxes = [
            draw.textbbox((0, 0), line, font=font, stroke_width=1) for line in panel_lines
        ]
        panel_width = min(
            max(width - 16, 1),
            max((box[2] - box[0] for box in text_boxes), default=0) + padding * 2,
        )
        line_height = max((box[3] - box[1] for box in text_boxes), default=font.size) + 4
        panel_height = line_height * len(panel_lines) + padding * 2
        underlay_draw.rectangle(
            (8, 8, min(width - 8, 8 + panel_width), min(height - 8, 8 + panel_height)),
            fill=(0, 0, 0, 145),
        )
        image = Image.alpha_composite(image, underlay)
        draw = ImageDraw.Draw(image)

        for index, line in enumerate(panel_lines):
            draw.text(
                (8 + padding, 8 + padding + index * line_height),
                line,
                font=font,
                fill=(255, 255, 255, 255),
                stroke_width=1,
                stroke_fill=(0, 0, 0, 255),
            )

        header_bottom = 8 + panel_height
    for item in box_labels:
        x1, y1, x2, y2 = item.bbox
        bounds = draw.textbbox((0, 0), item.text, font=font, stroke_width=1)
        text_width = bounds[2] - bounds[0]
        text_height = bounds[3] - bounds[1]
        x = max(2, min(width - text_width - padding * 2 - 2, x1))
        y = y1 - text_height - padding * 2 - 2
        if y < header_bottom + 2:
            y = y2 + 2
        if y + text_height + padding * 2 > height - 2:
            y = max(2, y1 - text_height - padding * 2 - 2)
        draw.rectangle(
            (x, y, x + text_width + padding * 2, y + text_height + padding * 2),
            fill=(0, 0, 0, 175),
        )
        draw.text(
            (x + padding, y + padding),
            item.text,
            font=font,
            fill=(*_rgb_from_bgr(item.color_bgr), 255),
            stroke_width=1,
            stroke_fill=(0, 0, 0, 255),
        )

    return cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)


def _split_overlay_text(text: str) -> tuple[str, str]:
    identifier, separator, behavior = str(text).partition("｜")
    if not separator:
        return str(text), ""
    return identifier, behavior


def draw_behavior_sidebar(
    frame: np.ndarray,
    *,
    frame_index: int,
    total_frames: int,
    fps: float,
    active_event_count: int,
    display_events: Sequence[Mapping[str, Any]],
    display_layer: str,
    mouse_overlays: Mapping[int, MouseOverlay],
    valid_ids: Iterable[int],
    font: Any | None = None,
    header_font: Any | None = None,
    small_font: Any | None = None,
    panel_width: int | None = None,
    focus_behavior: str | None = None,
    focus_active: bool = False,
) -> np.ndarray:
    """Append a separate ID-to-behavior panel to the right of ``frame``.

    The source image is kept at its original size. All persistent status text
    is drawn in this new panel so it cannot cover cage pixels or keypoints.
    """

    height, width = frame.shape[:2]
    sidebar_width = int(panel_width or sidebar_width_for_frame(width, height))
    base_size = font_size_for_frame(width, height)
    body_font = font or load_font(None, max(base_size, 14))
    title_font = header_font or load_font(None, max(base_size + 4, 18))
    detail_font = small_font or load_font(None, max(base_size - 1, 12))
    body_size = int(getattr(body_font, "size", max(base_size, 14)))
    title_size = int(getattr(title_font, "size", max(base_size + 4, 18)))
    detail_size = int(getattr(detail_font, "size", max(base_size - 1, 12)))

    panel = Image.new("RGB", (sidebar_width, height), (25, 22, 20))
    draw = ImageDraw.Draw(panel)
    padding = max(12, int(round(sidebar_width * 0.035)))

    normalized_focus = normalize_focus_behavior(focus_behavior)
    title = "小鼠 ID 与当前行为"
    if normalized_focus:
        title = f"{title}｜重点：{focus_behavior_name_zh(normalized_focus)}"
    draw.text(
        (padding, 14),
        _truncate(title, 31),
        font=title_font,
        fill=(245, 245, 245),
    )
    safe_fps = float(fps) if np.isfinite(fps) and fps > 0 else 1.0
    progress = (
        f"帧 {int(frame_index) + 1}/{max(int(total_frames), 1)}  "
        f"时间 {int(frame_index) / safe_fps:.2f}秒  事件 {int(active_event_count)}"
    )
    draw.text(
        (padding, 14 + title_size + 5),
        progress,
        font=detail_font,
        fill=(220, 220, 220),
    )
    focus_status_height = 0
    if normalized_focus:
        focus_status = "当前证据" if bool(focus_active) else "当前帧无证据，保留重点关注"
        focus_y = 14 + title_size + detail_size + 11
        draw.text(
            (padding, focus_y),
            _truncate(f"重点行为状态：{focus_status}", 27),
            font=detail_font,
            fill=(255, 190, 80) if not focus_active else (100, 235, 145),
        )
        focus_status_height = detail_size + 5
    summary_lines = build_panel_lines(display_events, display_layer)
    summary_y = 14 + title_size + detail_size + 15 + focus_status_height
    for index, line in enumerate(summary_lines[:2]):
        draw.text(
            (padding, summary_y + index * (detail_size + 4)),
            _truncate(line, 44),
            font=detail_font,
            fill=(210, 210, 210),
        )

    ids = tuple(sorted({int(value) for value in valid_ids if int(value) >= 0}))
    row_top = summary_y + 2 * (detail_size + 4) + 12
    row_height = max(38, int(round(body_size * 2.35)))
    max_rows = max((height - row_top - 8) // row_height, 0)
    if len(ids) > max_rows and max_rows > 0:
        ids = ids[:max_rows]

    for row_index, mouse_id in enumerate(ids):
        overlay = mouse_overlays.get(
            mouse_id,
            MouseOverlay(
                text=f"{format_mouse_id(mouse_id)}｜仅追踪",
                color_bgr=UNKNOWN_BEHAVIOR_COLOR_BGR,
                priority=(0, 0.0),
            ),
        )
        y1 = row_top + row_index * row_height
        y2 = y1 + row_height - 5
        active = overlay.priority[0] > 0
        background = (56, 45, 40) if active else (33, 30, 28)
        draw.rounded_rectangle((8, y1, sidebar_width - 8, y2), radius=3, fill=background)
        draw.rectangle((8, y1, 14, y2), fill=_rgb_from_bgr(overlay.color_bgr))
        identifier, behavior = _split_overlay_text(overlay.text)
        text_y = y1 + max(2, (row_height - body_size) // 2 - 1)
        draw.text(
            (20, text_y),
            identifier,
            font=body_font,
            fill=_rgb_from_bgr(overlay.color_bgr),
        )
        draw.text(
            (105, text_y),
            _truncate(behavior or "仅追踪", 22),
            font=body_font,
            fill=(245, 245, 245),
        )

    if not ids:
        draw.text(
            (padding, row_top),
            "当前帧未检测到有效小鼠",
            font=body_font,
            fill=(220, 220, 220),
        )

    panel_bgr = cv2.cvtColor(np.asarray(panel), cv2.COLOR_RGB2BGR)
    return np.concatenate((frame, panel_bgr), axis=1)


__all__ = [
    "BoxLabel",
    "MouseOverlay",
    "build_mouse_overlays",
    "build_panel_lines",
    "canonical_behavior",
    "draw_behavior_sidebar",
    "draw_text_overlay",
    "event_category",
    "event_summary",
    "focus_behavior_name_zh",
    "font_size_for_frame",
    "format_mouse_id",
    "load_font",
    "normalize_focus_behavior",
    "normalize_contact_events",
    "pair_ids",
    "sidebar_width_for_frame",
    "select_display_events",
]
