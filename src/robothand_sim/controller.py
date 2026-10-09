"""Scripted (non-learned) controller: move beans one at a time from plate A to plate B.

The controller drives two things:
  * the tweezer tip position, through the 3-axis gantry carrying the hand (stand-in for an arm)
  * the thumb / index pads, which open and close the tweezers
Tip motion follows minimum-jerk profiles. Gripping is a guarded move: each finger closes at
a fixed rate until its pad touches the tweezer leaf, then an integral loop holds the
pad-to-leaf contact force (a stand-in for a fingertip tactile sensor) at a target, so both
leaves press the bean equally even though the two fingers have different kinematics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from robothand_sim import poses, scene


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
    ids = {
        frozenset((model.geom("th_pad").id, model.geom("tweezer_th_leaf").id)): 0,
        frozenset((model.geom("ff_pad").id, model.geom("tweezer_ff_leaf").id)): 1,
    }
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


@dataclass
class EpisodeResult:
    transfers: list[TransferResult]
    n_in_b: int
    n_lost: int
    duration: float
    log: dict = field(default_factory=dict)

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

    def _frame(self, phase: str) -> None:
        if self.on_frame is not None and self.data.time - self.t0 >= self.next_frame:
            self.on_frame(self.data, phase)
            self.next_frame += self.frame_dt

    def run(self, phases: list[Phase], target: str) -> bool:
        m, d, p = self.model, self.data, self.p
        dt = m.opt.timestep
        lifted = False
        for ph in phases:
            n_steps = max(1, int(round(ph.duration / dt)))
            T = n_steps * dt
            delta = ph.tip_to - ph.tip_from
            mm_start = self.mm.copy()
            for k in range(n_steps):
                s, ds, dds = _min_jerk((k + 1) / n_steps)
                tip = ph.tip_from + s * delta
                force = pad_forces(m, d)
                if ph.grip:
                    # Leaf tip distance from the centre line, from the leaf hinge angles.
                    leaf_mm = 1000 * self.info.tweezer_length * np.array(
                        [d.joint("tweezer_th_hinge").qpos[0], d.joint("tweezer_ff_hinge").qpos[0]]
                    )
                    free_rate = np.where(leaf_mm > p.slow_zone, p.approach_rate, p.final_rate)
                    rate = np.where(force < p.contact_force, free_rate, p.force_gain * (p.grip_force - force))
                    self.mm = np.clip(self.mm + rate * dt, p.open_mm, p.max_mm)
                else:
                    self.mm = mm_start + s * (p.open_mm - mm_start)
                scene.set_tip_target(m, d, self.info, tip, delta * ds / T, delta * dds / T**2)
                scene.apply_ctrl(m, d, poses.hand_ctrl(*self.mm))
                mujoco.mj_step(m, d)
                if ph.name in ("lift", "carry") and d.body(target).xpos[2] > self.info.plate_a[2] + 0.03:
                    lifted = True
                if k % 10 == 0:
                    self.log["t"].append(d.time - self.t0)
                    self.log["phase"].append(ph.name)
                    self.log["force"].append(force)
                    self.log["mm"].append(self.mm.copy())
                self._frame(ph.name)
        return lifted

    def idle(self, seconds: float, phase: str = "done") -> None:
        for _ in range(int(seconds / self.model.opt.timestep)):
            mujoco.mj_step(self.model, self.data)
            self._frame(phase)


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
    runner.idle(p.t_settle, "settle")  # let the beans settle before choosing one

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
        observed = data.body(target).xpos.copy()
        observed[:2] += rng.normal(0.0, p.perception_noise, 2)
        phases = plan(tip, observed, info.plate_a[2], drop_point(k, info), p)
        lifted = runner.run(phases, target)
        tip = phases[-1].tip_to
        final = data.body(target).xpos.copy()
        result = TransferResult(
            bean=target,
            success=where(final, info) == "B",
            lifted=lifted,
            final_pos=final,
            disturbed=int(sum(np.linalg.norm(data.body(n).xpos[:2] - q[:2]) > 0.01 for n, q in others0.items())),
            t_start=t_start,
            t_end=data.time - runner.t0,
        )
        transfers.append(result)
        if on_transfer is not None:
            on_transfer(result)
    runner.idle(0.5)  # let things come to rest

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
    )
