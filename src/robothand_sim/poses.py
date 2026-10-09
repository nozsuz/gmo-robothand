"""Actuator targets for the Shadow Hand (right).

Only the thumb and index finger operate the tweezers. Middle, ring and little fingers are
curled out of the way and the wrist is held neutral.

Finger motion is expressed as pad travel along the tweezer closing axis, in millimetres
from the Menagerie "two finger pinch" keyframe (positive = towards the other finger).
The joint mixes below were obtained from the pad-point Jacobians in the tweezer frame
(see scripts/inspect_pinch.py): the thumb combines THJ4 and THJ5 so that the pad moves
across the leaf without sliding off its width.
"""

from __future__ import annotations

STATIC = {
    "rh_A_WRJ2": 0.0,
    "rh_A_WRJ1": 0.0,
    "rh_A_MFJ4": 0.0, "rh_A_MFJ3": 1.2, "rh_A_MFJ0": 2.0,
    "rh_A_RFJ4": 0.0, "rh_A_RFJ3": 1.2, "rh_A_RFJ0": 2.0,
    "rh_A_LFJ5": 0.4, "rh_A_LFJ4": 0.0, "rh_A_LFJ3": 1.2, "rh_A_LFJ0": 2.0,
}

# Thumb-index pinch, from the Menagerie "two finger pinch" keyframe.
PINCH = {
    "rh_A_THJ5": 0.031, "rh_A_THJ4": 1.08, "rh_A_THJ3": 0.16, "rh_A_THJ2": 0.584, "rh_A_THJ1": 0.307,
    "rh_A_FFJ4": -0.301, "rh_A_FFJ3": 0.977, "rh_A_FFJ0": 1.21,
}
# Joint change per millimetre of pad travel towards the other finger [rad/mm].
THUMB_MIX = {"rh_A_THJ4": -1 / 89.1, "rh_A_THJ5": 0.478 / 89.1}
INDEX_MIX = {"rh_A_FFJ3": 1 / 72.4}

OPEN_MM = -16.0  # both pads clear of the spring-opened leaves


def hand_ctrl(thumb_mm: float = OPEN_MM, index_mm: float = OPEN_MM) -> dict[str, float]:
    ctrl = {**STATIC, **PINCH}
    for name, k in THUMB_MIX.items():
        ctrl[name] += k * thumb_mm
    for name, k in INDEX_MIX.items():
        ctrl[name] += k * index_mm
    return ctrl
