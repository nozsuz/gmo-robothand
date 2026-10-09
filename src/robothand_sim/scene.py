"""Scene construction: Shadow Hand (5 fingers) + tweezers + table + two plates + beans.

The hand is carried by a 3-axis gantry (three slide joints with stiff position servos, a
stand-in for a robot arm; the wrist orientation is fixed). The tweezers are a
two-leaf tool whose back end sits in a holder fixed to the palm; each leaf is a spring
loaded hinge that opens by itself, and the thumb and index finger pads close it by
pressing on the leaves. The tweezer placement is derived from the hand's own pinch
geometry at build time, so changing the pinch pose re-fits the tool automatically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import mujoco
import numpy as np

from robothand_sim import poses
from robothand_sim.paths import SHADOW_RIGHT_HAND_XML

# Contact groups (contype / conaffinity bits).
#   bit 1: hand        bit 2: tool and world objects
# Hand-vs-tweezer contacts are restricted to the thumb / index distal pads via explicit pairs.
HAND_BITS = 1
TOOL_BITS = 2
WORLD_BITS = HAND_BITS | TOOL_BITS
ARM_KP = 1.0e4  # gantry servo stiffness [N/m]
ARM_DAMPING = 400.0  # gantry joint damping [N s/m]
ARM_ARMATURE = 0.5
PAD = 0.0008  # tip pad thickness; the pads meet on the centre plane when closed
# Near-hard contact for the bean. MuJoCo's constraint regularisation scales with 1/mass, so a
# 0.25 g bean squeezed with ~1 N sinks centimetres into default-soft contacts; dmax -> 1 fixes it.
HARD_SOLREF = [0.004, 1.0]
HARD_SOLIMP = [0.95, 0.9999, 0.001, 0.5, 2]


@dataclass
class SceneConfig:
    n_beans: int = 5
    bean_radius: float = 0.004  # roasted soybean ~8 mm diameter
    bean_radius_jitter: float = 0.0005  # per-bean uniform +- [m]
    bean_mass: float = 0.25e-3  # ~0.25 g
    bean_friction: tuple[float, float, float] = (0.6, 0.005, 0.0005)
    tip_friction: float = 0.8  # tweezer tip pad friction (material lever: coating / silicone)
    tweezer_length: float = 0.125
    tweezer_thickness: float = 0.002
    tweezer_width: float = 0.009
    tweezer_open_angle: float = 0.10  # spring rest angle of each leaf [rad]
    tweezer_stiffness: float = 0.03  # leaf spring [Nm/rad]
    plate_a: tuple[float, float] = (0.0, -0.09)
    plate_b: tuple[float, float] = (0.0, 0.09)
    plate_radius: float = 0.05
    table_height: float = 0.0
    timestep: float = 0.001
    seed: int = 0
    tweezer_tilt_deg: float = 10.0  # tweezer axis angle from vertical (towards +x)
    extra: dict = field(default_factory=dict)


@dataclass
class SceneInfo:
    """Quantities the controller needs, expressed in world / mount frames."""

    mount_quat: np.ndarray  # world orientation of the hand mount
    tip_offset: np.ndarray  # tip centre relative to mount position, world frame (wrist at 0)
    bean_names: list[str]
    plate_a: np.ndarray  # plate centre top surface (x, y, z)
    plate_b: np.ndarray
    plate_radius: float
    gantry_mass: float = 0.0  # moving mass per gantry axis, for feedforward [kg]
    tweezer_length: float = 0.125


# --------------------------------------------------------------------------------------
# Hand + mount


def _hand_world_spec(timestep: float) -> mujoco.MjSpec:
    world = mujoco.MjSpec()
    world.modelname = "robothand_bean_transfer"
    world.compiler.degree = False
    world.option.timestep = timestep
    world.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    world.option.impratio = 10
    world.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    world.option.noslip_iterations = 3

    hand = mujoco.MjSpec.from_file(str(SHADOW_RIGHT_HAND_XML))
    # A kinematic (mocap) mount would teleport the hand each step: contacts then see zero
    # velocity and friction cannot carry the bean. Slide joints give the tool real velocity.
    gantry = world.worldbody.add_body(name="gantry", pos=[0, 0, 0])
    for axis, vec in zip("xyz", np.eye(3)):
        gantry.add_joint(
            name=f"arm_{axis}", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=vec, damping=ARM_DAMPING, armature=ARM_ARMATURE
        )
        world.add_actuator(
            name=f"arm_{axis}",
            target=f"arm_{axis}",
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
            gaintype=mujoco.mjtGain.mjGAIN_FIXED,
            biastype=mujoco.mjtBias.mjBIAS_AFFINE,
            gainprm=[ARM_KP] + [0] * 9,
            biasprm=[0, -ARM_KP, 0] + [0] * 7,
            ctrlrange=[-1, 1],
            forcerange=[-1000, 1000],
        )
    gantry.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.01, 0, 0], mass=0.5, contype=0, conaffinity=0, group=4)
    gantry.gravcomp = 1
    mount = gantry.add_body(name="hand_mount")
    mount.add_frame().attach_body(hand.body("rh_forearm"), "", "")
    for body in world.bodies:
        if body.name.startswith("rh_"):
            body.gravcomp = 1  # the real arm/hand controller compensates gravity
    # Lock the wrist at neutral: its servos are weak (5-10 Nm), so under fast arm motion the
    # hand swings and the tweezer tip lags the arm by several centimetres.
    for joint in ("rh_WRJ1", "rh_WRJ2"):
        world.add_equality(type=mujoco.mjtEq.mjEQ_JOINT, name1=joint, data=[0, 0, 0, 0, 0] + [0] * 6)
    # Name the distal pad collision geoms so we can pair them with the tweezer leaves.
    for finger in ("th", "ff"):
        for geom in world.body(f"rh_{finger}distal").geoms:
            if geom.contype:
                geom.name = f"{finger}_pad"
    return world


def _apply_ctrl(model: mujoco.MjModel, data: mujoco.MjData, ctrl: dict[str, float]) -> None:
    for name, value in ctrl.items():
        data.ctrl[model.actuator(name).id] = value


def _set_qpos_from_ctrl(model: mujoco.MjModel, data: mujoco.MjData, ctrl: dict[str, float]) -> None:
    """Initialise joint positions consistently with actuator targets.

    Coupled J1+J2 tendons are split like the real hand: the middle joint (J2) flexes first.
    """
    for name, value in ctrl.items():
        act = model.actuator(name)
        if act.trntype[0] == mujoco.mjtTrn.mjTRN_JOINT:
            data.qpos[model.jnt_qposadr[act.trnid[0]]] = value
        else:
            base = model.tendon(act.trnid[0]).name[:-1]  # "rh_FFJ0" -> "rh_FFJ"
            j2 = min(value, np.pi / 2)
            data.qpos[model.jnt_qposadr[model.joint(base + "2").id]] = j2
            data.qpos[model.jnt_qposadr[model.joint(base + "1").id]] = value - j2


@lru_cache(maxsize=4)
def calibrate_pinch() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (pinch point, closing normal thumb->index, web point) in the palm frame.

    Squeezes thumb and index together without any tool and reads the pad contact.
    """
    spec = _hand_world_spec(timestep=0.001)
    model = spec.compile()
    data = mujoco.MjData(model)
    target = poses.hand_ctrl(15.0, 15.0)  # drive the pads into each other
    _set_qpos_from_ctrl(model, data, poses.hand_ctrl(0.0, 0.0))
    _apply_ctrl(model, data, target)
    for _ in range(1500):
        mujoco.mj_step(model, data)

    th = model.geom("th_pad").id
    ff = model.geom("ff_pad").id
    pts, normals = [], []
    for c in data.contact[: data.ncon]:
        pair = {c.geom1, c.geom2}
        if pair == {th, ff}:
            pts.append(c.pos.copy())
            n = c.frame[:3].copy()  # points from geom1 to geom2
            normals.append(n if c.geom1 == th else -n)
    if not pts:
        raise RuntimeError("pinch calibration failed: thumb and index pads do not touch")
    p_world = np.mean(pts, axis=0)
    n_world = np.mean(normals, axis=0)
    n_world /= np.linalg.norm(n_world)
    web_world = 0.5 * (data.body("rh_thproximal").xpos + data.body("rh_ffproximal").xpos)

    rot = data.body("rh_palm").xmat.reshape(3, 3)
    pos = data.body("rh_palm").xpos
    return rot.T @ (p_world - pos), rot.T @ n_world, rot.T @ (web_world - pos)


# --------------------------------------------------------------------------------------
# Tweezers


def _add_tweezers(spec: mujoco.MjSpec, cfg: SceneConfig) -> None:
    pinch, normal, web = calibrate_pinch()
    axis = pinch - web
    axis -= axis.dot(normal) * normal
    reach = np.linalg.norm(axis)  # hinge -> finger pad distance along the tool
    axis /= reach
    hinge_axis = np.cross(axis, normal)
    rot = np.column_stack([axis, normal, hinge_axis])  # tweezer frame: x=tool, y=closing, z=hinge
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rot.flatten())

    # Hinge sits on the web; shift it so the closed leaves meet exactly at the pinch point.
    hinge_pos = pinch - reach * axis
    L, t, w = cfg.tweezer_length, cfg.tweezer_thickness, cfg.tweezer_width
    holder = spec.body("rh_palm").add_body(name="tweezers", pos=hinge_pos, quat=quat)
    holder.childclass = "main"  # do not inherit the hand's joint friction / damping defaults
    holder.add_geom(
        name="tweezer_back",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.008, t * 1.5, w / 2],
        pos=[-0.006, 0, 0],
        material="steel",
        contype=0,
        conaffinity=0,
        mass=0.002,
    )
    for side, sign in (("th", -1.0), ("ff", 1.0)):
        leaf = holder.add_body(name=f"tweezer_{side}")
        leaf.add_joint(
            name=f"tweezer_{side}_hinge",
            type=mujoco.mjtJoint.mjJNT_HINGE,
            axis=[0, 0, sign],  # positive angle opens the leaf away from the centre line
            range=[0, 2.5 * cfg.tweezer_open_angle],
            stiffness=cfg.tweezer_stiffness,
            springref=cfg.tweezer_open_angle,
            damping=0.002,
            armature=1e-5,
        )
        # Leaf body: the inner face lies on the centre plane when the tweezers are closed.
        leaf.add_geom(
            name=f"tweezer_{side}_leaf",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[L / 2, t / 2, w / 2],
            pos=[L / 2, sign * (PAD + t / 2), 0],
            material="steel",
            contype=TOOL_BITS,
            conaffinity=TOOL_BITS,
            friction=[0.3, 0.005, 0.0001],  # bare stainless steel
            priority=1,
            solref=HARD_SOLREF,
            solimp=HARD_SOLIMP,
            mass=0.004,
        )
        # Tip pad: a thin high-friction layer on the inner face (coating / rubber).
        leaf.add_geom(
            name=f"tweezer_{side}_tip",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.008, PAD / 2, w / 2],
            pos=[L - 0.008, sign * PAD / 2, 0],
            rgba=[0.85, 0.35, 0.2, 1],
            contype=TOOL_BITS,
            conaffinity=TOOL_BITS,
            friction=[cfg.tip_friction, 0.01, 0.001],
            condim=4,
            priority=2,
            solref=HARD_SOLREF,
            solimp=HARD_SOLIMP,
            mass=0.0001,
        )
        leaf.add_site(name=f"tweezer_{side}_tipsite", pos=[L, 0, 0], size=[0.001] * 3)
        spec.add_pair(
            name=f"{side}_pad_vs_leaf",
            geomname1=f"{side}_pad",
            geomname2=f"tweezer_{side}_leaf",
            condim=3,
            friction=[1.0, 1.0, 0.005, 0.0001, 0.0001],
        )
    holder.add_site(name="tweezer_tip", pos=[L, 0, 0], size=[0.0015] * 3, rgba=[1, 0, 0, 1])
    spec.add_exclude(bodyname1="tweezer_th", bodyname2="tweezer_ff")
    # ...but the tip pads do meet when the tweezers are fully closed.
    spec.add_pair(name="tip_vs_tip", geomname1="tweezer_th_tip", geomname2="tweezer_ff_tip", condim=3)


# --------------------------------------------------------------------------------------
# World: table, plates, beans


def _add_plate(spec: mujoco.MjSpec, name: str, xy, cfg: SceneConfig, rgba) -> None:
    top = cfg.table_height
    body = spec.worldbody.add_body(name=name, pos=[xy[0], xy[1], top])
    body.add_geom(
        name=f"{name}_base",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[cfg.plate_radius, 0.002, 0],
        pos=[0, 0, 0.002],
        rgba=rgba,
        contype=WORLD_BITS,
        conaffinity=WORLD_BITS,
        friction=[0.6, 0.005, 0.001],
        condim=6,
    )
    # Lip: tangential capsules forming a low ring, like the raised edge of a dish.
    n_seg = 36
    r = cfg.plate_radius
    for i in range(n_seg):
        ang = 2 * np.pi * i / n_seg
        yaw = np.array([np.cos(ang / 2), 0, 0, np.sin(ang / 2)])
        q = np.zeros(4)  # capsule axis (local z) along the tangent
        mujoco.mju_mulQuat(q, yaw, np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0]))
        body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            size=[0.003, r * np.pi / n_seg, 0],
            pos=[r * np.cos(ang), r * np.sin(ang), 0.006],
            quat=q,
            rgba=rgba,
            contype=WORLD_BITS,
            conaffinity=WORLD_BITS,
            friction=[0.6, 0.005, 0.001],
        )


def _bean_positions(cfg: SceneConfig) -> np.ndarray:
    rng = np.random.default_rng(cfg.seed)
    pts: list[np.ndarray] = []
    max_r = cfg.plate_radius - 0.015
    while len(pts) < cfg.n_beans:
        r = max_r * np.sqrt(rng.uniform())
        a = rng.uniform(0, 2 * np.pi)
        p = np.array([r * np.cos(a), r * np.sin(a)])
        if all(np.linalg.norm(p - q) > 4 * cfg.bean_radius for q in pts):
            pts.append(p)
    return np.array(pts) + np.array(cfg.plate_a)


def _add_world(spec: mujoco.MjSpec, cfg: SceneConfig) -> list[str]:
    spec.add_material(name="steel", rgba=[0.85, 0.87, 0.9, 1], specular=0.9, shininess=0.8, reflectance=0.2)
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 720
    spec.visual.quality.shadowsize = 4096
    spec.stat.extent = 0.4
    spec.stat.center = [0.0, 0.0, 0.08]
    spec.worldbody.add_light(pos=[0.3, -0.3, 1.0], dir=[-0.3, 0.3, -1.0], diffuse=[0.7, 0.7, 0.7])
    spec.worldbody.add_light(pos=[-0.4, 0.3, 0.8], dir=[0.4, -0.3, -0.8], diffuse=[0.35, 0.35, 0.35])
    spec.worldbody.add_geom(
        name="table",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.3, 0.3, 0.01],
        pos=[0, 0, cfg.table_height - 0.01],
        rgba=[0.55, 0.42, 0.3, 1],
        contype=WORLD_BITS,
        conaffinity=WORLD_BITS,
    )
    spec.add_texture(
        name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
        rgb1=[0.42, 0.52, 0.62], rgb2=[0.08, 0.1, 0.13], width=512, height=3072,
    )
    spec.add_texture(
        name="floor", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        rgb1=[0.22, 0.25, 0.28], rgb2=[0.18, 0.2, 0.23], width=300, height=300,
    )
    spec.add_material(name="floor", textures=["", "floor"], texrepeat=[8, 8], texuniform=True)
    spec.worldbody.add_geom(
        name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[2, 2, 0.05], pos=[0, 0, -0.75],
        material="floor", contype=0, conaffinity=0,
    )
    # "overview": palm side, both plates in view. Video close-ups use a camera that tracks the tip.
    spec.worldbody.add_camera(name="overview", pos=[-0.38, 0.0, 0.2], xyaxes=[0, -1, 0, 0.35, 0, 0.94])
    _add_plate(spec, "plate_a", cfg.plate_a, cfg, rgba=[0.92, 0.92, 0.9, 1])
    _add_plate(spec, "plate_b", cfg.plate_b, cfg, rgba=[0.75, 0.85, 0.95, 1])

    names = []
    plate_top = cfg.table_height + 0.004
    radii = cfg.bean_radius + np.random.default_rng(cfg.seed + 10_000).uniform(
        -cfg.bean_radius_jitter, cfg.bean_radius_jitter, cfg.n_beans
    )
    for i, ((x, y), radius) in enumerate(zip(_bean_positions(cfg), radii)):
        name = f"bean{i}"
        bean = spec.worldbody.add_body(name=name, pos=[x, y, plate_top + radius + 0.0005])
        bean.add_freejoint(name=f"{name}_free")
        bean.add_geom(
            name=f"{name}_geom",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[radius, 0, 0],
            mass=cfg.bean_mass * (radius / cfg.bean_radius) ** 3,
            rgba=[0.78, 0.6, 0.3, 1],
            contype=WORLD_BITS,
            conaffinity=WORLD_BITS,
            friction=list(cfg.bean_friction),
            condim=6,
            solref=HARD_SOLREF,
            solimp=HARD_SOLIMP,
        )
        names.append(name)
    return names


# --------------------------------------------------------------------------------------


def build(cfg: SceneConfig | None = None) -> tuple[mujoco.MjModel, SceneInfo]:
    cfg = cfg or SceneConfig()
    spec = _hand_world_spec(cfg.timestep)
    _add_tweezers(spec, cfg)
    bean_names = _add_world(spec, cfg)

    # Orientation of the mount: tweezer axis points down (tilted towards +x), leaves close along y.
    model = spec.compile()
    data = mujoco.MjData(model)
    _set_qpos_from_ctrl(model, data, poses.hand_ctrl())
    mujoco.mj_kinematics(model, data)  # gantry at the origin, mount at identity
    rot_rel = data.body("tweezers").xmat.reshape(3, 3).copy()
    tip_rel = data.site("tweezer_tip").xpos.copy()

    tilt = np.deg2rad(cfg.tweezer_tilt_deg)
    a_w = np.array([-np.sin(tilt), 0.0, -np.cos(tilt)])  # tips point down, hand leans back (+x side)
    n_w = np.array([0.0, 1.0, 0.0])
    rot_des = np.column_stack([a_w, n_w, np.cross(a_w, n_w)])
    rot_mount = rot_des @ rot_rel.T
    mount_quat = np.zeros(4)
    mujoco.mju_mat2Quat(mount_quat, rot_mount.flatten())

    plate_top = cfg.table_height + 0.004
    info = SceneInfo(
        mount_quat=mount_quat,
        tip_offset=rot_mount @ tip_rel,
        bean_names=bean_names,
        plate_a=np.array([*cfg.plate_a, plate_top]),
        plate_b=np.array([*cfg.plate_b, plate_top]),
        plate_radius=cfg.plate_radius,
        tweezer_length=cfg.tweezer_length,
        gantry_mass=float(model.body_subtreemass[model.body("gantry").id]) + ARM_ARMATURE,
    )
    model.body("hand_mount").pos[:] = 0
    model.body("hand_mount").quat[:] = mount_quat
    return model, info


def reset(model: mujoco.MjModel, data: mujoco.MjData, info: SceneInfo, tip_pos: np.ndarray) -> None:
    """Put the hand in the open pose with the tweezer tip at `tip_pos`."""
    mujoco.mj_resetData(model, data)
    ctrl = poses.hand_ctrl()
    _set_qpos_from_ctrl(model, data, ctrl)
    _apply_ctrl(model, data, ctrl)
    set_tip_target(model, data, info, tip_pos)
    for axis, value in zip("xyz", data.ctrl[[model.actuator(f"arm_{a}").id for a in "xyz"]]):
        data.qpos[model.joint(f"arm_{axis}").qposadr[0]] = value
    mujoco.mj_forward(model, data)


def set_tip_target(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    info: SceneInfo,
    tip_pos: np.ndarray,
    tip_vel: np.ndarray | None = None,
    tip_acc: np.ndarray | None = None,
) -> None:
    """Command the gantry so that the (closed) tweezer tip goes to `tip_pos` (wrist neutral).

    With the reference velocity / acceleration, the servo target is shifted so that damping
    and inertia are compensated (feedforward), as an industrial arm controller would do.
    """
    target = tip_pos - info.tip_offset
    if tip_vel is not None:
        target = target + (ARM_DAMPING / ARM_KP) * tip_vel
    if tip_acc is not None:
        target = target + (info.gantry_mass / ARM_KP) * tip_acc
    for axis, value in zip("xyz", target):
        data.ctrl[model.actuator(f"arm_{axis}").id] = value


apply_ctrl = _apply_ctrl
