"""Scripted (non-learned) controller: move beans one at a time from plate A to plate B.

The controller drives two things:
  * the tweezer tip position, through the 3-axis gantry carrying the hand (stand-in for an arm)
  * the thumb / index pads, which open and close the tweezers
Tip motion follows minimum-jerk profiles. Gripping is a guarded move: each finger closes at
a fixed rate until its pad touches the tweezer leaf, then an integral loop holds the
pad-to-leaf contact force (a stand-in for a fingertip tactile sensor) at a target, so both
leaves press the bean equally even though the two fingers have different kinematics.

With finger-held tweezers (no holder) the pads are always on the leaves: each finger
regulates its pad force to `hold_force` while open (the leaf springs then keep the tool
pinched) and to `squeeze_force` while gripping. The tool settles into its own grasp after
the hand-over; its tip position is then taken from tool tracking (`measured_tip_offset`),
and slip of the tool in the hand is recorded per transfer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from robothand_sim import hands, poses, scene


@dataclass
class TransferParams:
    approach_height: float = 0.05  # tip height above the bean before descending
    grasp_depth: float = -0.0025  # tip end height relative to the bean centre (pads cover the equator)
    min_tip_clearance: float = 0.0012  # ...but never closer than this to the plate surface [m]
    carry_height: float = 0.07
    release_height: float = 0.012  # bean drop height above plate B surface
    grip_force: float = 0.25  # target pad-on-leaf normal force per finger [N]
    force_gain: float = 120.0  # pad travel rate per unit force error [mm / (N s)]
    approach_rate: float = 80.0  # pad travel rate while the leaf tip is far from the bean [mm/s]
    final_rate: float = 20.0  # ...and once it is within `slow_zone` of the centre line [mm/s]
    slow_zone: float = 7.5  # leaf tip distance from the centre line where closing slows [mm]
    contact_force: float = 0.03  # pad force regarded as contact [N]
    max_mm: float = 8.0
    open_mm: float = poses.OPEN_MM
    perception_noise: float = 0.0  # std of the observed bean position error, per axis in xy [m]
    # finger-held tweezers
    hold_force: float = 0.25  # pad force while open: keeps the tool pinched [N]
    squeeze_force: float = 0.6  # pad force while gripping a bean (includes the leaf spring) [N]
    held_mm: tuple[float, float] = (-30.0, 10.0)  # pad travel range
    offset_tau: float = 0.05  # filter time constant of the measured tip offset [s]
    t_pin_leaves: float = 0.5  # hand-over 1: tool and leaves pinned, the fingers close on the leaves
    t_pin_tool: float = 0.3  # hand-over 2: leaves free, tool still pinned
    t_regrasp: float = 0.6  # hand-over 3: released, the tool settles into the fingers' grasp
    # The tool counts as dropped when both pads lose it, or when it has slipped so far that
    # tool tracking can no longer be trusted (the episode then ends).
    drop_slip_mm: float = 40.0
    drop_slip_deg: float = 60.0
    drop_contact_time: float = 0.2  # both pad forces below `contact_force` for this long [s]
    max_offset_correction: float = 0.03  # limit on tool-tracking correction of the tip target [m]
    # In-hand stabilisation from tool tracking (finger-held): roll of the tool about its own
    # axis is fed back to a thumb shift across the leaf; sideways drift along the closing axis
    # to a common shift of both fingers.
    roll_gain: float = 2.0  # thumb across-shift rate per degree of roll [mm/(deg s)]
    center_gain: float = 10.0  # common finger shift rate per mm of sideways drift [1/s]
    max_across_mm: float = 6.0
    filter_omega: float = 30.0  # [rad/s] reference filter for "filtered" policy commands
    speed: float = 1.0  # all phase durations are divided by this
    # phase durations [s] at speed 1
    t_settle: float = 0.3
    t_approach: float = 0.6
    t_descend: float = 0.5
    t_close: float = 0.5
    t_hold: float = 0.2
    t_lift: float = 0.5
    t_carry: float = 0.9
    t_lower: float = 0.4
    t_open: float = 0.25
    t_retreat: float = 0.5


@dataclass
class Phase:
    name: str
    duration: float
    tip_from: np.ndarray
    tip_to: np.ndarray
    grip: bool  # True: force-controlled squeeze, False: fingers go to the open pose
    pin: str = ""  # hand-over: "tool+leaves" or "tool" are held at their nominal pose
    # Tip interpolation: "minjerk"; "linear" or "filtered" for 20 Hz policy commands ("filtered"
    # passes the command through a critically damped 2nd-order filter: smooth velocity and
    # acceleration, which a finger-held tool needs).
    profile: str = "minjerk"


def _min_jerk(s: float) -> tuple[float, float, float]:
    """Minimum-jerk profile: position, d/ds and d2/ds2 at normalised time s in [0, 1]."""
    s = min(max(s, 0.0), 1.0)
    return (
        s**3 * (10 - 15 * s + 6 * s**2),
        30 * s**2 * (1 - s) ** 2,
        60 * s * (1 - s) * (1 - 2 * s),
    )


def plan(
    start_tip: np.ndarray, bean_pos: np.ndarray, surface_z: float, drop: np.ndarray, p: TransferParams
) -> list[Phase]:
    """Waypoints for one transfer.

    `surface_z` is the plate A surface height and `drop` the point on plate B's surface to
    release over. Small beans would otherwise put the tip corner into the plate, which
    presses the bean down instead of pinching it.
    """
    grasp = bean_pos.copy()
    grasp[2] = max(bean_pos[2] + p.grasp_depth, surface_z + p.min_tip_clearance)
    T = 1.0 / p.speed
    above_bean = bean_pos + np.array([0, 0, p.approach_height])
    lift = bean_pos.copy()
    lift[2] = drop[2] + p.carry_height
    above_drop = drop + np.array([0, 0, p.carry_height])
    release = drop + np.array([0, 0, p.release_height + 0.004])
    return [
        Phase("approach", T * p.t_approach, start_tip, above_bean, grip=False),
        Phase("descend", T * p.t_descend, above_bean, grasp, grip=False),
        Phase("close", T * p.t_close, grasp, grasp, grip=True),
        Phase("hold", T * p.t_hold, grasp, grasp, grip=True),
        Phase("lift", T * p.t_lift, grasp, lift, grip=True),
        Phase("carry", T * p.t_carry, lift, above_drop, grip=True),
        Phase("lower", T * p.t_lower, above_drop, release, grip=True),
        Phase("open", T * p.t_open, release, release, grip=False),
        Phase("retreat", T * p.t_retreat, release, above_drop, grip=False),
    ]


def pad_forces(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Normal force of the thumb pad and index pad on their tweezer leaves [N]."""
    ids = {}
    for k, side in enumerate(("th", "ff")):
        for part in ("leaf", "grip"):  # finger-held tweezers are pressed on the grip section
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"tweezer_{side}_{part}")
            if gid >= 0:
                ids[frozenset((model.geom(f"{side}_pad").id, gid))] = k
    out = np.zeros(2)
    f6 = np.zeros(6)
    for i, c in enumerate(data.contact[: data.ncon]):
        k = ids.get(frozenset((c.geom1, c.geom2)))
        if k is not None:
            mujoco.mj_contactForce(model, data, i, f6)
            out[k] += f6[0]
    return out


def where(pos: np.ndarray, info: scene.SceneInfo) -> str:
    """'A', 'B' or 'lost' for a bean position."""
    for name, centre in (("A", info.plate_a), ("B", info.plate_b)):
        if np.linalg.norm(pos[:2] - centre[:2]) < info.plate_radius and pos[2] < centre[2] + 0.02:
            return name
    return "lost"


def pick_target_bean(data: mujoco.MjData, info: scene.SceneInfo) -> str | None:
    """Among beans on plate A, choose the one with the most free space around it."""
    pos = {n: data.body(n).xpos.copy() for n in info.bean_names}
    on_a = [n for n in info.bean_names if where(pos[n], info) == "A"]
    if not on_a:
        return None

    def clearance(name: str) -> float:
        others = [np.linalg.norm(pos[name][:2] - pos[k][:2]) for k in on_a if k != name]
        edge = info.plate_radius - np.linalg.norm(pos[name][:2] - info.plate_a[:2])
        return min(others + [edge * 1.5])

    return max(on_a, key=clearance)


def drop_point(k: int, info: scene.SceneInfo) -> np.ndarray:
    """Spread released beans over plate B so they do not land on each other."""
    if k == 0:
        return info.plate_b.copy()
    ang = 2 * np.pi * (k - 1) / 6
    return info.plate_b + 0.022 * np.array([np.cos(ang), np.sin(ang), 0.0])


@dataclass
class TransferResult:
    bean: str
    success: bool  # the bean ended up on plate B
    lifted: bool
    final_pos: np.ndarray
    disturbed: int  # other beans on plate A pushed by more than 1 cm
    t_start: float
    t_end: float
    slip_mm: float = 0.0  # finger-held: max tool displacement in the hand during the transfer
    slip_deg: float = 0.0
    tool_dropped: bool = False


@dataclass
class EpisodeResult:
    transfers: list[TransferResult]
    n_in_b: int
    n_lost: int
    duration: float
    log: dict = field(default_factory=dict)
    tool_dropped: bool = False

    @property
    def success(self) -> bool:
        return bool(self.transfers) and all(t.success for t in self.transfers)


class _Runner:
    """Steps the simulation through phases, keeping finger state, logs and the frame clock."""

    def __init__(self, model, data, info, p: TransferParams, on_frame, frame_dt: float):
        self.model, self.data, self.info, self.p = model, data, info, p
        self.on_frame, self.frame_dt = on_frame, frame_dt
        self.mm = np.array([p.open_mm, p.open_mm])  # thumb, index pad travel
        self.log = {"t": [], "phase": [], "force": [], "mm": []}
        self.t0 = data.time
        self.next_frame = 0.0
        self.held = info.tool_mount == "fingers"
        self.hand = hands.get(info.hand_name)
        self.offset: np.ndarray | None = None  # filtered measured tip offset (finger-held)
        self.grasp_ref: tuple[np.ndarray, np.ndarray] | None = None  # tool pose in the palm
        self.slip = np.zeros(2)  # max (mm, deg) since the last reset_slip()
        self.dropped = False
        self.across = 0.0  # thumb shift across the leaf [mm]
        self.no_contact = 0.0  # time both pads have been off the tool [s]
        self.ref: tuple[np.ndarray, np.ndarray] | None = None  # filtered reference (pos, vel)
        # Leaf angle at which the spring pushes back on the pads with `hold_force`.
        self.open_leaf_angle = info.leaf_springref - p.hold_force * info.pad_lever / info.leaf_stiffness

    def _tool_in_palm(self) -> tuple[np.ndarray, np.ndarray]:
        d = self.data
        rp, pp = d.body(self.info.palm_body).xmat.reshape(3, 3), d.body(self.info.palm_body).xpos
        return rp.T @ (d.body("tweezers").xpos - pp), rp.T @ d.body("tweezers").xmat.reshape(3, 3)

    def set_grasp_reference(self) -> None:
        """The tool's settled pose in the hand; slip is measured from here."""
        self.grasp_ref = self._tool_in_palm()

    def reset_slip(self) -> np.ndarray:
        out, self.slip = self.slip, np.zeros(2)
        return out

    def _track_slip(self, force: np.ndarray) -> None:
        if self.grasp_ref is None:
            return
        lost = bool(np.all(force < self.p.contact_force))
        self.no_contact = self.no_contact + self.model.opt.timestep if lost else 0.0
        if self.no_contact > self.p.drop_contact_time:
            self.dropped = True
        p, r = self._tool_in_palm()
        p0, r0 = self.grasp_ref
        cos = np.clip((np.trace(r0.T @ r) - 1) / 2, -1.0, 1.0)
        self.slip = np.maximum(self.slip, [1000 * np.linalg.norm(p - p0), np.degrees(np.arccos(cos))])
        if self.slip[0] > self.p.drop_slip_mm or self.slip[1] > self.p.drop_slip_deg:
            self.dropped = True

    def _in_hand_error(self) -> tuple[float, float]:
        """(roll about the tool axis [deg], sideways drift at the pads along the closing axis [mm])
        relative to the settled grasp, both in the settled tool frame."""
        if self.grasp_ref is None:
            return 0.0, 0.0
        p, r = self._tool_in_palm()
        p0, r0 = self.grasp_ref
        rel = r0.T @ r
        roll = np.degrees(np.arctan2(rel[2, 1], rel[1, 1]))  # rotation about the tool x axis
        pinch, pinch0 = p + r[:, 0] * self.info.pad_lever, p0 + r0[:, 0] * self.info.pad_lever
        return float(roll), float(1000 * r0[:, 1] @ (pinch - pinch0))

    def _finger_update(self, ph: Phase, force: np.ndarray, s: float, mm_start: np.ndarray) -> None:
        p, d = self.p, self.data
        dt = self.model.opt.timestep
        # Leaf tip distance from the centre line, from the leaf hinge angles.
        leaf_mm = 1000 * self.info.tweezer_length * np.array(
            [d.joint("tweezer_th_hinge").qpos[0], d.joint("tweezer_ff_hinge").qpos[0]]
        )
        free_rate = np.where(leaf_mm > p.slow_zone, p.approach_rate, p.final_rate)
        if self.held:
            # The pads never leave the leaves: regulate force, and close no faster than the
            # guarded-move rates so the leaves do not flick the bean away.
            target = p.squeeze_force if ph.grip else p.hold_force
            rate = np.minimum(p.force_gain * (target - force), free_rate)
            if not ph.pin:
                roll, side = self._in_hand_error()
                rate = rate + p.center_gain * side * np.array([-1.0, 1.0])
                self.across = float(np.clip(self.across + p.roll_gain * roll * dt, -p.max_across_mm, p.max_across_mm))
            self.mm = np.clip(self.mm + rate * dt, *p.held_mm)
        elif ph.grip:
            rate = np.where(force < p.contact_force, free_rate, p.force_gain * (p.grip_force - force))
            self.mm = np.clip(self.mm + rate * dt, p.open_mm, p.max_mm)
        else:
            self.mm = mm_start + s * (p.open_mm - mm_start)

    def _tip_offset(self) -> np.ndarray | None:
        if not self.held:
            return None  # the holder keeps the tool at its nominal pose
        measured = scene.measured_tip_offset(self.model, self.data)
        correction = measured - self.info.tip_offset
        norm = np.linalg.norm(correction)
        if norm > self.p.max_offset_correction:  # never chase a tool that is falling out
            measured = self.info.tip_offset + correction * self.p.max_offset_correction / norm
        if self.offset is None:
            self.offset = measured
        a = self.model.opt.timestep / self.p.offset_tau
        self.offset = (1 - a) * self.offset + a * measured
        return self.offset

    def _frame(self, phase: str) -> None:
        if self.on_frame is not None and self.data.time - self.t0 >= self.next_frame:
            self.on_frame(self.data, phase)
            self.next_frame += self.frame_dt

    def run(self, phases: list[Phase], target: str | None = None) -> bool:
        m, d, p = self.model, self.data, self.p
        dt = m.opt.timestep
        lifted = False
        for ph in phases:
            n_steps = max(1, int(round(ph.duration / dt)))
            T = n_steps * dt
            delta = ph.tip_to - ph.tip_from
            mm_start = self.mm.copy()
            for k in range(n_steps):
                if ph.profile == "filtered":
                    if self.ref is None:
                        self.ref = (ph.tip_from.copy(), np.zeros(3))
                    pos, vel = self.ref
                    w = p.filter_omega
                    acc = w * w * (ph.tip_to - pos) - 2 * w * vel
                    vel = vel + acc * dt
                    pos = pos + vel * dt
                    self.ref = (pos, vel)
                    tip, tip_vel, tip_acc = pos, vel, acc
                    s = (k + 1) / n_steps
                else:
                    if ph.profile == "linear":
                        s, ds, dds = (k + 1) / n_steps, 1.0, 0.0
                    else:
                        s, ds, dds = _min_jerk((k + 1) / n_steps)
                    tip = ph.tip_from + s * delta
                    tip_vel, tip_acc = delta * ds / T, delta * dds / T**2
                force = pad_forces(m, d)
                self._finger_update(ph, force, s, mm_start)
                offset = self._tip_offset()
                scene.set_tip_target(m, d, self.info, tip, tip_vel, tip_acc, offset)
                scene.apply_ctrl(m, d, self.hand.ctrl(*self.mm, self.across))
                mujoco.mj_step(m, d)
                if ph.pin:
                    scene.place_tool(m, d, self.open_leaf_angle if ph.pin == "tool+leaves" else None)
                elif self.held:
                    self._track_slip(force)
                    if self.dropped:
                        return lifted
                if target and ph.name in ("lift", "carry") and d.body(target).xpos[2] > self.info.plate_a[2] + 0.03:
                    lifted = True
                if k % 10 == 0:
                    self.log["t"].append(d.time - self.t0)
                    self.log["phase"].append(ph.name)
                    self.log["force"].append(force)
                    self.log["mm"].append(self.mm.copy())
                self._frame(ph.name)
        return lifted


def run_episode(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    info: scene.SceneInfo,
    params: TransferParams | None = None,
    n_transfers: int = 1,
    on_frame=None,
    frame_dt: float = 1 / 30,
    on_transfer=None,
    rng: np.random.Generator | None = None,
) -> EpisodeResult:
    """Transfer up to `n_transfers` beans from plate A to plate B, one at a time.

    `on_frame(data, phase)` is called at ~1/frame_dt Hz (for video); `on_transfer(result)`
    after each transfer. `rng` drives the perception noise.
    """
    p = params or TransferParams()
    rng = rng or np.random.default_rng(0)
    tip = info.plate_a + np.array([0, 0, p.approach_height + 0.03])
    scene.reset(model, data, info, tip)
    runner = _Runner(model, data, info, p, on_frame, frame_dt)
    if runner.held:
        # Hand the tool over: pinned while the fingers close on it, then released.
        runner.run(
            [
                Phase("handover", p.t_pin_leaves, tip, tip, grip=False, pin="tool+leaves"),
                Phase("handover", p.t_pin_tool, tip, tip, grip=False, pin="tool"),
                Phase("regrasp", p.t_regrasp, tip, tip, grip=False),
            ]
        )
        runner.set_grasp_reference()
    runner.run([Phase("settle", p.t_settle, tip, tip, grip=False)])  # let the beans settle

    transfers: list[TransferResult] = []
    for k in range(n_transfers):
        target = pick_target_bean(data, info)
        if target is None:
            break
        others0 = {
            n: data.body(n).xpos.copy()
            for n in info.bean_names
            if n != target and where(data.body(n).xpos, info) == "A"
        }
        t_start = data.time - runner.t0
        runner.reset_slip()
        observed = data.body(target).xpos.copy()
        observed[:2] += rng.normal(0.0, p.perception_noise, 2)
        phases = plan(tip, observed, info.plate_a[2], drop_point(k, info), p)
        lifted = runner.run(phases, target)
        tip = phases[-1].tip_to
        final = data.body(target).xpos.copy()
        slip = runner.reset_slip()
        result = TransferResult(
            bean=target,
            success=where(final, info) == "B",
            lifted=lifted,
            final_pos=final,
            disturbed=int(sum(np.linalg.norm(data.body(n).xpos[:2] - q[:2]) > 0.01 for n, q in others0.items())),
            t_start=t_start,
            t_end=data.time - runner.t0,
            slip_mm=float(slip[0]),
            slip_deg=float(slip[1]),
            tool_dropped=runner.dropped,
        )
        transfers.append(result)
        if on_transfer is not None:
            on_transfer(result)
        if runner.dropped:
            break  # a dropped tool ends the run
    if not runner.dropped:
        runner.run([Phase("done", 0.5, tip, tip, grip=False)])  # let things come to rest

    # Re-check after settling: a released bean can still roll off the plate. This may only
    # downgrade a transfer; a missed bean that a later attempt moved stays a failure here.
    for tr in transfers:
        tr.final_pos = data.body(tr.bean).xpos.copy()
        tr.success = tr.success and where(tr.final_pos, info) == "B"
    places = [where(data.body(n).xpos, info) for n in info.bean_names]
    return EpisodeResult(
        transfers=transfers,
        n_in_b=places.count("B"),
        n_lost=places.count("lost"),
        duration=data.time - runner.t0,
        log=runner.log,
        tool_dropped=runner.dropped,
    )
