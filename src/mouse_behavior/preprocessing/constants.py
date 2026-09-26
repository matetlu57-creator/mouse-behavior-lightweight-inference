"""Shared pose and ethogram constants for the lightweight pipeline."""

from __future__ import annotations

PROJECT_NAME = "mouse-behavior-lightweight-inference"

SOCIAL_BEHAVIORS = (
    "together",
    "approach",
    "following",
    "chase",
    "avoidance",
    "attack",
    "nose_head_contact",
    "nose_tail_contact",
)
GROUP_BEHAVIORS = (
    "huddle",
    "social_clustering",
    "group_locomotion",
    "dispersal",
    "isolation",
)
INDIVIDUAL_BEHAVIORS = ("running", "walking", "stationary")
EXTENDED_BEHAVIORS = SOCIAL_BEHAVIORS + GROUP_BEHAVIORS + INDIVIDUAL_BEHAVIORS
BEHAVIOR_LAYERS = {
    "individual": INDIVIDUAL_BEHAVIORS,
    "social": SOCIAL_BEHAVIORS,
    "group": GROUP_BEHAVIORS,
}
BEHAVIOR_LAYER_ORDER = ("individual", "social", "group")
# Rendering consumes these only within a layer. Category priority is applied
# separately so a group event never destroys co-occurring lower-layer output.
BEHAVIOR_DISPLAY_PRIORITY = {
    # The user's explicit within-group order.
    "huddle": 115,
    "social_clustering": 110,
    "dispersal": 105,
    "group_locomotion": 100,
    "isolation": 95,
    # Social order is a display order; train-calibrated Top-1 scoring is free
    # to prefer a more specific event such as Attack over Approach.
    "approach": 190,
    "together": 180,
    "chase": 170,
    "avoidance": 160,
    "attack": 150,
    "nose_head_contact": 140,
    "nose_tail_contact": 130,
    "following": 120,
    # Individual labels have no biological precedence over one another.
    "running": 55,
    "walking": 55,
    "stationary": 55,
}
BEHAVIOR_NAMES_ZH = {
    "together": "一起",
    "approach": "接近",
    "following": "跟随",
    "chase": "追逐",
    "avoidance": "回避",
    "attack": "攻击",
    "nose_head_contact": "鼻头接触",
    "nose_tail_contact": "鼻尾接触",
    "huddle": "扎堆",
    "social_clustering": "社会聚集",
    "group_locomotion": "群体同步运动",
    "dispersal": "群体分散",
    "isolation": "孤立",
    "running": "奔跑",
    "walking": "行走",
    "stationary": "静止",
}

KP_NOSE = 0
KP_LEFT_EAR = 1
KP_RIGHT_EAR = 2
KP_NECK = 3
KP_LEFT_HIP = 4
KP_RIGHT_HIP = 5
KP_TAIL = 6
KEYPOINTS = 7

SKELETON_EDGES = (
    (KP_NOSE, KP_LEFT_EAR),
    (KP_NOSE, KP_RIGHT_EAR),
    (KP_LEFT_EAR, KP_NECK),
    (KP_RIGHT_EAR, KP_NECK),
    (KP_NECK, KP_LEFT_HIP),
    (KP_NECK, KP_RIGHT_HIP),
    (KP_LEFT_HIP, KP_TAIL),
    (KP_RIGHT_HIP, KP_TAIL),
)

FOUR_CLASS_NAMES = {
    0: "00_非追逐非攻击",
    1: "01_非攻击性追逐",
    2: "02_非追逐攻击",
    3: "03_攻击性追逐",
}

__all__ = [
    "PROJECT_NAME",
    "SOCIAL_BEHAVIORS",
    "GROUP_BEHAVIORS",
    "INDIVIDUAL_BEHAVIORS",
    "EXTENDED_BEHAVIORS",
    "BEHAVIOR_LAYERS",
    "BEHAVIOR_LAYER_ORDER",
    "BEHAVIOR_DISPLAY_PRIORITY",
    "BEHAVIOR_NAMES_ZH",
    "KP_NOSE",
    "KP_LEFT_EAR",
    "KP_RIGHT_EAR",
    "KP_NECK",
    "KP_LEFT_HIP",
    "KP_RIGHT_HIP",
    "KP_TAIL",
    "KEYPOINTS",
    "SKELETON_EDGES",
    "FOUR_CLASS_NAMES",
]
