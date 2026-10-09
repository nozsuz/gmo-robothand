"""Scene construction: Shadow Hand (5 fingers) + tweezers + table + two plates + beans.

The hand is carried by a 3-axis gantry (three slide joints with stiff position servos, a
stand-in for a robot arm; the wrist orientation is fixed). The tweezers are a two-leaf
tool; each leaf is a spring-loaded hinge that opens by itself, and the thumb and index
finger pads close it by pressing on the leaves. The tweezer placement is derived from the
hand's own pinch geometry at build time, so changing the pinch pose re-fits the tool.

Two ways of holding the tool (`SceneConfig.tool_mount`):
  * "holder":  the back end sits in a holder fixed to the palm (the tool cannot move).
  * "fingers": the tool is a free body held only by contact: the thumb and index pads
               pinch the leaves (the leaf springs keep pressing back on the pads, as with
               real tweezers) and the back end rests against the hand. At the start of
               an episode the tool is held at its nominal pose while the fingers close
               on it (`place_tool`), then released.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import mujoco
import numpy as np

from robothand_sim import hands
from robothand_sim.hands import Hand

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
    tool_mount: str = "holder"  # "holder" (fixed to the palm) or "fingers" (held by contact only)
    hand: str = "shadow"  # "shadow" (Shadow Hand) or "revo2" (BrainCo Revo 2, offered for the G1-D)
    # Finger-held tweezers need a stiffer spring: it is what keeps the pads pressed on the tool.
    held_open_angle: float = 0.10
    held_stiffness: float = 0.6
    pad_friction: float = 1.2  # fingertip skin / rubber on steel
    pad_torsion: float = 0.01  # torsional friction of a soft fingertip contact patch [m]
    grip_width: float = 0.016  # widened finger-grip section of each leaf (finger-held only)
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
    tool_mount: str = "holder"
    hand_name: str = "shadow"
    palm_body: str = "rh_palm"
    leaf_stiffness: float = 0.03  # [Nm/rad]
    leaf_springref: float = 0.10  # [rad]
    pad_lever: float = 0.08  # hinge -> finger pad distance along the tool [m]


# --------------------------------------------------------------------------------------
# Hand + mount


def _hand_world_spec(hand: Hand, timestep: float) -> mujoco.MjSpec:
    world = mujoco.MjSpec()
    world.modelname = "robothand_bean_transfer"
    world.compiler.degree = False
    world.option.timestep = timestep
    world.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    world.option.impratio = 10
    world.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    world.option.noslip_iterations = 3

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
    # The hand also names its thumb / index pad collision geoms th_pad / ff_pad.
    hand.attach(world, mount)
    return world


def _apply_ctrl(model: mujoco.MjModel, data: mujoco.MjData, ctrl: dict[str, float]) -> None:
    for name, value in ctrl.items():
        data.ctrl[model.actuator(name).id] = value


_PINCH_CACHE: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def calibrate_pinch(hand: Hand) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (pinch point, closing normal thumb->index, web point) in the palm frame.

    Squeezes thumb and index together without any tool and reads the pad contact.
    """
    if hand.name in _PINCH_CACHE:
        return _PINCH_CACHE[hand.name]
    spec = _hand_world_spec(hand, timestep=0.001)
    model = spec.compile()
    data = mujoco.MjData(model)
    hand.set_qpos(model, data, hand.pinch_ctrl())
    _apply_ctrl(model, data, hand.squeeze_ctrl())  # drive the pads into each other
    for _ in range(1500):
        mujoco.mj_step(model, data)

    th_bodies = {model.body(b).id for b in hand.pinch_bodies["th"]}
    ff_bodies = {model.body(b).id for b in hand.pinch_bodies["ff"]}
    pts, normals = [], []
    for c in data.contact[: data.ncon]:
        b1, b2 = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
        if (b1 in th_bodies and b2 in ff_bodies) or (b1 in ff_bodies and b2 in th_bodies):
            pts.append(c.pos.copy())
            n = c.frame[:3].copy()  # points from geom1 to geom2
            normals.append(n if b1 in th_bodies else -n)
    if not pts:
        raise RuntimeError(f"pinch calibration failed for {hand.name}: thumb and index pads do not touch")
    p_world = np.mean(pts, axis=0)
    n_world = np.mean(normals, axis=0)
    n_world /= np.linalg.norm(n_world)
    web_world = 0.5 * (data.body(hand.web_bodies[0]).xpos + data.body(hand.web_bodies[1]).xpos)

    rot = data.body(hand.palm_body).xmat.reshape(3, 3)
    pos = data.body(hand.palm_body).xpos
    out = rot.T @ (p_world - pos), rot.T @ n_world, rot.T @ (web_world - pos)
    _PINCH_CACHE[hand.name] = out
    return out


def tool_frame(hand: Hand) -> tuple[np.ndarray, np.ndarray, float]:
    """Tweezer frame in the palm frame: (hinge position, rotation [tool, closing, hinge], reach).

    The tool runs from the thumb-index web through the pinch point; the closed leaves meet
    exactly at the pinch point.
    """
    pinch, normal, web = calibrate_pinch(hand)
    axis = pinch - web
    axis -= axis.dot(normal) * normal
    reach = float(np.linalg.norm(axis))  # hinge -> finger pad distance along the tool
    axis /= reach
    rot = np.column_stack([axis, normal, np.cross(axis, normal)])
    return pinch - reach * axis, rot, reach


def fit_mixes(hand: Hand) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Joint mixes (rad per mm) from the pad-point Jacobians in the tweezer frame.

    thumb: moves the thumb pad towards the index pad without moving it across the leaf;
    across: moves it across the leaf (hinge axis) without closing; index: index pad towards
    the thumb. Only for hands whose fingers are driven joint by joint (Revo 2).
    """
    _, tool_rot, _ = tool_frame(hand)
    model = _hand_world_spec(hand, 0.001).compile()
    data = mujoco.MjData(model)
    hand.set_qpos(model, data, hand.pinch_ctrl())
    mujoco.mj_forward(model, data)
    to_tool = tool_rot.T @ data.body(hand.palm_body).xmat.reshape(3, 3).T  # world -> tool frame

    def pad_jacobian(finger: str, actuators: list[str]) -> np.ndarray:
        other = "ff" if finger == "th" else "th"
        fromto = np.zeros(6)
        mujoco.mj_geomDistance(model, data, model.geom(f"{finger}_pad").id, model.geom(f"{other}_pad").id, 0.05, fromto)
        jacp = np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jacp, None, fromto[:3], model.geom(f"{finger}_pad").bodyid[0])
        cols = []
        for act in actuators:
            col = jacp[:, model.joint(act).dofadr[0]].copy()
            for eq in range(model.neq):  # add coupled (follower) joints
                if model.eq_type[eq] == mujoco.mjtEq.mjEQ_JOINT and model.eq_obj2id[eq] == model.joint(act).id:
                    col += model.eq_data[eq][1] * jacp[:, model.jnt_dofadr[model.eq_obj1id[eq]]]
            cols.append(col)
        return to_tool @ np.column_stack(cols) * 1000  # mm per rad

    th_acts, ff_acts = hand.thumb_actuators, hand.index_actuators
    j_th = pad_jacobian("th", th_acts)[1:3]  # rows: closing, across
    thumb = np.linalg.solve(j_th, [1.0, 0.0])
    across = np.linalg.solve(j_th, [0.0, 1.0])
    j_ff = pad_jacobian("ff", ff_acts)[1]
    index = -1.0 / j_ff[0]
    return (
        dict(zip(th_acts, thumb.tolist())),
        {ff_acts[0]: float(index)},
        dict(zip(th_acts, across.tolist())),
    )


# --------------------------------------------------------------------------------------
# Tweezers


def _add_tweezers(spec: mujoco.MjSpec, cfg: SceneConfig, hand: Hand) -> None:
    # Tweezer frame: x = tool, y = closing (thumb -> index), z = hinge axis.
    hinge_pos, rot, reach = tool_frame(hand)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rot.flatten())
    L, t, w = cfg.tweezer_length, cfg.tweezer_thickness, cfg.tweezer_width
    held = cfg.tool_mount == "fingers"
    stiffness = cfg.held_stiffness if held else cfg.tweezer_stiffness
    open_angle = cfg.held_open_angle if held else cfg.tweezer_open_angle
    # Nominal tool pose in the hand (where the holder puts it / where the fingers should keep it).
    palm = spec.body(hand.palm_body)
    palm.add_site(name="tool_ref", pos=hinge_pos, quat=quat, size=[0.001] * 3, group=4)
    if held:
        holder = spec.worldbody.add_body(name="tweezers")
        holder.add_freejoint(name="tweezers_free")
    else:
        holder = palm.add_body(name="tweezers", pos=hinge_pos, quat=quat)
    holder.childclass = "main"  # do not inherit the hand's joint friction / damping defaults
    holder.add_geom(
        name="tweezer_back",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.008, t * 1.5, w / 2],
        pos=[-0.006, 0, 0],
        material="steel",
        # No contact with the hand: at the pinch-fitted pose the back end overlaps the palm, and
        # the resulting push makes a free tool jump out of the fingers.
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
            range=[0, 2.5 * open_angle],
            stiffness=stiffness,
            springref=open_angle,
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
        if held:
            # Widened grip section where the finger pads press (like the ridged grip of real
            # tweezers), so a pad does not slide off the edge of a narrow leaf.
            leaf.add_geom(
                name=f"tweezer_{side}_grip",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[0.0225, t / 2, cfg.grip_width / 2],
                pos=[max(reach, 0.0225), sign * (PAD + t / 2), 0],  # centred on the finger pads
                rgba=[0.45, 0.47, 0.5, 1],
                contype=0,
                conaffinity=0,
                mass=0.001,
            )
        spec.add_pair(
            name=f"{side}_pad_vs_leaf",
            geomname1=f"{side}_pad",
            geomname2=f"tweezer_{side}_grip" if held else f"tweezer_{side}_leaf",
            condim=4 if held else 3,
            friction=[cfg.pad_friction, cfg.pad_friction, cfg.pad_torsion, 0.0001, 0.0001],
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
    for _ in range(100_000):
        if len(pts) == cfg.n_beans:
            break
        r = max_r * np.sqrt(rng.uniform())
        a = rng.uniform(0, 2 * np.pi)
        p = np.array([r * np.cos(a), r * np.sin(a)])
        if all(np.linalg.norm(p - q) > 4 * cfg.bean_radius for q in pts):
            pts.append(p)
    else:
        raise ValueError(f"cannot place {cfg.n_beans} beans 16 mm apart on the plate (about 10 fit)")
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
    hand = hands.get(cfg.hand)
    spec = _hand_world_spec(hand, cfg.timestep)
    _add_tweezers(spec, cfg, hand)
    bean_names = _add_world(spec, cfg)

    # Orientation of the mount: tweezer axis points down (tilted towards +x), leaves close along y.
    model = spec.compile()
    data = mujoco.MjData(model)
    hand.set_qpos(model, data, hand.ctrl())
    mujoco.mj_kinematics(model, data)  # gantry at the origin, mount at identity
    rot_rel = data.site("tool_ref").xmat.reshape(3, 3).copy()
    tip_rel = data.site("tool_ref").xpos + rot_rel[:, 0] * cfg.tweezer_length

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
        tool_mount=cfg.tool_mount,
        pad_lever=tool_frame(hand)[2],
        hand_name=hand.name,
        palm_body=hand.palm_body,
        leaf_stiffness=cfg.held_stiffness if cfg.tool_mount == "fingers" else cfg.tweezer_stiffness,
        leaf_springref=cfg.held_open_angle if cfg.tool_mount == "fingers" else cfg.tweezer_open_angle,
        gantry_mass=float(model.body_subtreemass[model.body("gantry").id]) + ARM_ARMATURE,
    )
    model.body("hand_mount").pos[:] = 0
    model.body("hand_mount").quat[:] = mount_quat
    return model, info


def reset(model: mujoco.MjModel, data: mujoco.MjData, info: SceneInfo, tip_pos: np.ndarray) -> None:
    """Put the hand in the open pose with the tweezer tip at `tip_pos`."""
    mujoco.mj_resetData(model, data)
    hand = hands.get(info.hand_name)
    ctrl = hand.ctrl()
    hand.set_qpos(model, data, ctrl)
    _apply_ctrl(model, data, ctrl)
    set_tip_target(model, data, info, tip_pos)
    for axis, value in zip("xyz", data.ctrl[[model.actuator(f"arm_{a}").id for a in "xyz"]]):
        data.qpos[model.joint(f"arm_{axis}").qposadr[0]] = value
    if info.tool_mount == "fingers":
        mujoco.mj_kinematics(model, data)
        place_tool(model, data)
        for side in ("th", "ff"):
            data.qpos[model.joint(f"tweezer_{side}_hinge").qposadr[0]] = info.leaf_springref
    mujoco.mj_forward(model, data)


def place_tool(model: mujoco.MjModel, data: mujoco.MjData, leaf_angle: float | None = None) -> None:
    """Put the free tool at its nominal pose in the hand, at rest (hand-over only).

    Called every step while the fingers close on the tool at the start of an episode; after
    that, only contacts hold it. With `leaf_angle`, the leaves are pinned too, so a finger
    closing on a leaf cannot push it past the pad. Uses the poses from the last kinematics pass.
    """
    adr = model.joint("tweezers_free").qposadr[0]
    dof = model.joint("tweezers_free").dofadr[0]
    data.qpos[adr : adr + 3] = data.site("tool_ref").xpos
    mujoco.mju_mat2Quat(data.qpos[adr + 3 : adr + 7], data.site("tool_ref").xmat)
    data.qvel[dof : dof + 6] = 0
    if leaf_angle is not None:
        for side in ("th", "ff"):
            joint = model.joint(f"tweezer_{side}_hinge")
            data.qpos[joint.qposadr[0]] = leaf_angle
            data.qvel[joint.dofadr[0]] = 0


def tool_drift(data: mujoco.MjData) -> tuple[float, float]:
    """Tool pose error relative to its nominal pose in the hand: (mm, degrees)."""
    ref_pos, ref_rot = data.site("tool_ref").xpos, data.site("tool_ref").xmat.reshape(3, 3)
    pos, rot = data.body("tweezers").xpos, data.body("tweezers").xmat.reshape(3, 3)
    cos = np.clip((np.trace(ref_rot.T @ rot) - 1) / 2, -1.0, 1.0)
    return 1000 * float(np.linalg.norm(pos - ref_pos)), float(np.degrees(np.arccos(cos)))


def measured_tip_offset(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Actual tweezer tip position relative to the gantry (what tracking the tool would give)."""
    mount = np.array([data.qpos[model.joint(f"arm_{a}").qposadr[0]] for a in "xyz"])
    return data.site("tweezer_tip").xpos - mount


def set_tip_target(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    info: SceneInfo,
    tip_pos: np.ndarray,
    tip_vel: np.ndarray | None = None,
    tip_acc: np.ndarray | None = None,
    offset: np.ndarray | None = None,
) -> None:
    """Command the gantry so that the (closed) tweezer tip goes to `tip_pos` (wrist neutral).

    With the reference velocity / acceleration, the servo target is shifted so that damping
    and inertia are compensated (feedforward), as an industrial arm controller would do.
    `offset` (tip relative to the gantry) defaults to the nominal one; pass a measured offset
    when the tool can move in the hand.
    """
    target = tip_pos - (info.tip_offset if offset is None else offset)
    if tip_vel is not None:
        target = target + (ARM_DAMPING / ARM_KP) * tip_vel
    if tip_acc is not None:
        target = target + (info.gantry_mass / ARM_KP) * tip_acc
    for axis, value in zip("xyz", target):
        data.ctrl[model.actuator(f"arm_{axis}").id] = value


apply_ctrl = _apply_ctrl
