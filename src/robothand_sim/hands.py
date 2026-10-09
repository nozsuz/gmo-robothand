"""Hand models: everything that differs between the Shadow Hand and the BrainCo Revo 2.

A `Hand` knows how to attach itself to the gantry mount, which bodies are the palm and the
thumb / index pads, and how to turn "pad travel along the tweezer closing axis" into
actuator targets.

* Shadow Hand (MuJoCo Menagerie): 20 actuators, hand-tuned pinch pose and joint mixes
  (see poses.py).
* BrainCo Revo 2 (the five-finger hand offered for the Unitree G1-D): 11 joints, 6
  actuators (thumb rotation, thumb flexion, four finger flexions); each distal joint is
  coupled to its proximal joint. The official MJCF has neither actuators nor couplings, so
  they are added here. The pinch pose is found by a search over the thumb/index joints, and
  the joint mixes are computed from pad-point Jacobians in the tweezer frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import mujoco
import numpy as np

from robothand_sim import poses
from robothand_sim.paths import REVO2_RIGHT_XML, SHADOW_RIGHT_HAND_XML


@dataclass
class Hand:
    name: str
    palm_body: str
    pad_bodies: dict[str, str]  # {"th": ..., "ff": ...}: collision geoms become th_pad / ff_pad
    web_bodies: tuple[str, str]  # thumb base / index base: the tool's back end rests between them
    static: dict[str, float]  # unused fingers (curled away) and other fixed targets
    pinch: dict[str, float]  # thumb-index pinch
    thumb_mix: dict[str, float]  # rad per mm of thumb pad travel towards the index pad
    index_mix: dict[str, float]
    across_mix: dict[str, float] = field(default_factory=dict)  # thumb shift across the leaf
    open_mm: float = poses.OPEN_MM
    thumb_actuators: list[str] = field(default_factory=list)  # for fit_mixes
    # Bodies whose mutual contact defines the pinch (pads, plus the links they sit on).
    pinch_bodies: dict[str, tuple[str, ...]] = field(default_factory=dict)
    index_actuators: list[str] = field(default_factory=list)

    def ctrl(self, thumb_mm: float | None = None, index_mm: float | None = None, across_mm: float = 0.0) -> dict[str, float]:
        thumb_mm = self.open_mm if thumb_mm is None else thumb_mm
        index_mm = self.open_mm if index_mm is None else index_mm
        out = {**self.static, **self.pinch}
        for mix, amount in ((self.thumb_mix, thumb_mm), (self.index_mix, index_mm), (self.across_mix, across_mm)):
            for name, k in mix.items():
                out[name] += k * amount
        return out

    def pinch_ctrl(self) -> dict[str, float]:
        return {**self.static, **self.pinch}

    def squeeze_ctrl(self) -> dict[str, float]:
        """Targets that drive the thumb and index pads into each other (pinch calibration)."""
        return self.ctrl(15.0, 15.0)

    # The methods below are hand specific.
    def attach(self, world: mujoco.MjSpec, mount: mujoco.MjsBody) -> None:
        raise NotImplementedError

    def set_qpos(self, model: mujoco.MjModel, data: mujoco.MjData, ctrl: dict[str, float]) -> None:
        raise NotImplementedError


def _name_pads(world: mujoco.MjSpec, pad_bodies: dict[str, str]) -> None:
    for finger, body in pad_bodies.items():
        geoms = [g for g in world.body(body).geoms if g.contype]
        geoms[0].name = f"{finger}_pad"


# --------------------------------------------------------------------------------------
# Shadow Hand


class ShadowHand(Hand):
    def __init__(self) -> None:
        super().__init__(
            name="shadow",
            palm_body="rh_palm",
            pad_bodies={"th": "rh_thdistal", "ff": "rh_ffdistal"},
            web_bodies=("rh_thproximal", "rh_ffproximal"),
            static=poses.STATIC,
            pinch=poses.PINCH,
            thumb_mix=poses.THUMB_MIX,
            index_mix=poses.INDEX_MIX,
            across_mix=poses.THUMB_ACROSS_MIX,
            pinch_bodies={"th": ("rh_thdistal",), "ff": ("rh_ffdistal",)},
        )

    def attach(self, world: mujoco.MjSpec, mount: mujoco.MjsBody) -> None:
        hand = mujoco.MjSpec.from_file(str(SHADOW_RIGHT_HAND_XML))
        mount.add_frame().attach_body(hand.body("rh_forearm"), "", "")
        for body in world.bodies:
            if body.name.startswith("rh_"):
                body.gravcomp = 1  # the real arm/hand controller compensates gravity
        # Lock the wrist at neutral: its servos are weak (5-10 Nm), so under fast arm motion the
        # hand swings and the tweezer tip lags the arm by several centimetres.
        for joint in ("rh_WRJ1", "rh_WRJ2"):
            world.add_equality(type=mujoco.mjtEq.mjEQ_JOINT, name1=joint, data=[0, 0, 0, 0, 0] + [0] * 6)
        _name_pads(world, self.pad_bodies)

    def set_qpos(self, model: mujoco.MjModel, data: mujoco.MjData, ctrl: dict[str, float]) -> None:
        """Coupled J1+J2 tendons are split like the real hand: the middle joint (J2) flexes first."""
        for name, value in ctrl.items():
            act = model.actuator(name)
            if act.trntype[0] == mujoco.mjtTrn.mjTRN_JOINT:
                data.qpos[model.jnt_qposadr[act.trnid[0]]] = value
            else:
                base = model.tendon(act.trnid[0]).name[:-1]  # "rh_FFJ0" -> "rh_FFJ"
                j2 = min(value, np.pi / 2)
                data.qpos[model.jnt_qposadr[model.joint(base + "2").id]] = j2
                data.qpos[model.jnt_qposadr[model.joint(base + "1").id]] = value - j2


# --------------------------------------------------------------------------------------
# BrainCo Revo 2

REVO2_FINGERS = ("index", "middle", "ring", "pinky")
# Distal = coef * proximal (brainco-description URDF mimic tags; revo2_description differs
# slightly, see README).
REVO2_COUPLING = {"thumb": 1.0, **{f: 1.155 for f in REVO2_FINGERS}}
REVO2_OTHERS = {f"right_{f}_proximal_joint": 0.15 for f in ("middle", "ring", "pinky")}
REVO2_ACTUATORS = {  # actuator (= proximal joint) -> (kp, torque limit from the MJCF)
    "right_thumb_metacarpal_joint": (1.0, 0.5),
    "right_thumb_proximal_joint": (2.0, 1.1),
    **{f"right_{f}_proximal_joint": (2.0, 2.0) for f in REVO2_FINGERS},
}


class Revo2Hand(Hand):
    def __init__(self, pinch: dict[str, float], thumb_mix, index_mix, across_mix) -> None:
        # Middle, ring and little fingers stay extended ("OK sign"): curled, they block the thumb.
        curled = dict(REVO2_OTHERS)
        super().__init__(
            name="revo2",
            palm_body="right_hand_base_link",
            pad_bodies={"th": "right_thumb_touch_link", "ff": "right_index_touch_link"},
            web_bodies=("right_thumb_proximal_link", "right_index_proximal_link"),
            static=curled,
            pinch=pinch,
            thumb_mix=thumb_mix,
            index_mix=index_mix,
            across_mix=across_mix,
            thumb_actuators=["right_thumb_metacarpal_joint", "right_thumb_proximal_joint"],
            index_actuators=["right_index_proximal_joint"],
            # The index pad is slightly recessed in its distal link: the thumb pad meets both.
            pinch_bodies={"th": ("right_thumb_touch_link", "right_thumb_distal_link"),
                          "ff": ("right_index_touch_link", "right_index_distal_link")},
        )

    def squeeze_ctrl(self) -> dict[str, float]:
        ctrl = self.pinch_ctrl()
        ctrl["right_thumb_proximal_joint"] += 0.25
        ctrl["right_index_proximal_joint"] += 0.25
        return ctrl

    def attach(self, world: mujoco.MjSpec, mount: mujoco.MjsBody) -> None:
        attach_revo2(world, mount)

    def set_qpos(self, model: mujoco.MjModel, data: mujoco.MjData, ctrl: dict[str, float]) -> None:
        for name, value in ctrl.items():
            data.qpos[model.joint(name).qposadr[0]] = value
            finger = name.split("_")[1]
            if name.endswith("_proximal_joint"):
                distal = model.joint(f"right_{finger}_distal_joint")
                data.qpos[distal.qposadr[0]] = REVO2_COUPLING[finger] * value


def attach_revo2(world: mujoco.MjSpec, mount: mujoco.MjsBody) -> None:
    hand = mujoco.MjSpec.from_file(str(REVO2_RIGHT_XML))
    mount.add_frame().attach_body(hand.body("right_hand_base_link"), "", "")
    for body in world.bodies:
        if body.name.startswith("right_"):
            body.gravcomp = 1
    for joint in world.joints:
        if joint.name.startswith("right_"):
            joint.damping = [0.05, 0.0, 0.0]  # (MuJoCo 3.15: damping is a polynomial)
            joint.armature = 0.0005
            joint.frictionloss = 0.005
    for finger, coef in REVO2_COUPLING.items():
        world.add_equality(
            type=mujoco.mjtEq.mjEQ_JOINT,
            name1=f"right_{finger}_distal_joint",
            name2=f"right_{finger}_proximal_joint",
            data=[0, coef, 0, 0, 0] + [0] * 6,
        )
    for joint, (kp, torque) in REVO2_ACTUATORS.items():
        lo, hi = world.joint(joint).range
        world.add_actuator(
            name=joint,
            target=joint,
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
            gaintype=mujoco.mjtGain.mjGAIN_FIXED,
            biastype=mujoco.mjtBias.mjBIAS_AFFINE,
            gainprm=[kp] + [0] * 9,
            biasprm=[0, -kp, 0] + [0] * 7,
            ctrlrange=[lo, hi],
            forcerange=[-torque, torque],
        )
    _name_pads(world, {"th": "right_thumb_touch_link", "ff": "right_index_touch_link"})


@lru_cache(maxsize=1)
def revo2_pinch() -> dict[str, float]:
    """Thumb rotation / thumb flexion / index flexion at which the two touch pads meet.

    Kinematic grid search. A candidate is valid when the pads touch (within 1 mm) while the
    proximal parts of each finger stay at least 2 mm from the other finger (otherwise the
    index tip ends up pressed into the thumb's proximal link). Among valid
    candidates, the one whose pad centres are closest (pads facing each other) wins.
    """
    world = mujoco.MjSpec()
    world.compiler.degree = False
    attach_revo2(world, world.worldbody.add_body(name="mount"))
    model = world.compile()
    data = mujoco.MjData(model)
    hand = Revo2Hand({}, {}, {}, {})
    th, ff = model.geom("th_pad").id, model.geom("ff_pad").id

    def collision_geoms(prefix: str) -> list[int]:
        return [g for g in range(model.ngeom) if model.geom_contype[g]
                and model.body(model.geom_bodyid[g]).name.startswith(prefix)]

    # Pads sit on the distal links, so distal/pad parts may come close; what must not happen is
    # a fingertip pressing into the other finger's proximal links.
    tips = ("_distal_link", "_touch_link")
    thumb, index = collision_geoms("right_thumb"), collision_geoms("right_index")
    rest = collision_geoms("right_middle") + collision_geoms("right_ring") + collision_geoms("right_pinky")
    is_tip = {g: model.body(model.geom_bodyid[g]).name.endswith(tips) for g in thumb + index}
    others = [(g, h) for g in thumb for h in index if not (is_tip[g] and is_tip[h])]
    others += [(g, h) for g in thumb for h in rest]
    fromto = np.zeros(6)
    best = (np.inf, None)
    for rot in np.linspace(0.6, 1.57, 14):
        for tf in np.linspace(0.0, 1.03, 16):
            for fi in np.linspace(0.2, 1.41, 18):
                ctrl = {"right_thumb_metacarpal_joint": rot, "right_thumb_proximal_joint": tf,
                        "right_index_proximal_joint": fi}
                hand.set_qpos(model, data, {**REVO2_OTHERS, **ctrl})
                mujoco.mj_kinematics(model, data)
                if abs(mujoco.mj_geomDistance(model, data, th, ff, 0.05, fromto)) > 0.001:
                    continue
                if min(mujoco.mj_geomDistance(model, data, g, h, 0.05, fromto) for g, h in others) < 0.002:
                    continue
                centre_gap = np.linalg.norm(data.geom_xpos[th] - data.geom_xpos[ff])
                if centre_gap < best[0]:
                    best = (centre_gap, ctrl)
    if best[1] is None:
        raise RuntimeError("no Revo 2 pinch pose found")
    return best[1]


@lru_cache(maxsize=1)
def revo2() -> Revo2Hand:
    """Revo 2 with pinch pose and joint mixes fitted to this hand (see scene.fit_mixes)."""
    from robothand_sim import scene  # local import: scene uses hands

    pinch = revo2_pinch()
    mixes = scene.fit_mixes(Revo2Hand(pinch, {}, {}, {}))
    return Revo2Hand(pinch, *mixes)


def get(name: str) -> Hand:
    if name == "shadow":
        return ShadowHand()
    if name == "revo2":
        return revo2()
    raise ValueError(f"unknown hand {name!r}")
