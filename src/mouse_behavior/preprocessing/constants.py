"""Shared pose and ethogram constants for the lightweight pipeline."""

from __future__ import annotations

PROJECT_NAME = "mouse-behavior-lightweight-inference"

SOCIAL_BEHAVIORS = (
    "together",
    "approach",
    "chase",
    "avoidance",
    "attack",
    "nose_head_contact",
    "nose_tail_contact",
)
GROUP_BEHAVIORS = ("huddle", "isolation")
INDIVIDUAL_BEHAVIORS = ("running", "walking", "stationary")
EXTENDED_BEHAVIORS = SOCIAL_BEHAVIORS + GROUP_BEHAVIORS + INDIVIDUAL_BEHAVIORS
BEHAVIOR_NAMES_ZH = {
    "together": "一起",
    "approach": "接近",
    "chase": "追逐",
    "avoidance": "回避",
    "attack": "攻击",
    "nose_head_contact": "鼻头接触",
    "nose_tail_contact": "鼻尾接触",
    "huddle": "扎堆",
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
