"""A 20 Hz command interface on top of the low-level controller, for imitation learning.

A policy sees a low-dimensional observation and outputs, every 50 ms, the next tweezer-tip
target position and an open/close command. The low-level loops (finger force control,
in-hand stabilisation of finger-held tweezers, gantry tracking) stay as in the scripted
controller. `ScriptedPolicy` drives the same interface with the scripted plan, so its
rollouts are demonstrations in exactly the form a learned policy has to reproduce.

Observation (float32, OBS_DIM):
  tip position (3), tip velocity (3), target bean - tip (3), drop point - tip (3),
  pad forces (2), leaf tip offsets from the centre line in mm (2), last grip command (1)
Action (float32, ACT_DIM): next tip target - current tip (3), grip command 0/1 (1)
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import mujoco
import numpy as np

from robothand_sim import controller, scene
from robothand_sim.controller import Phase, TransferParams

DT = 0.05  # policy period [s]
OBS_DIM = 17
ACT_DIM = 4


@dataclass
class EnvConfig:
    scene: scene.SceneConfig = field(default_factory=scene.SceneConfig)
    params: TransferParams = field(default_factory=TransferParams)
    n_transfers: int = 5
    attempt_timeout: float = 12.0  # [s] per bean (scripted takes 4.35 s at speed 1)
    profile: str = ""  # command interpolation; default: "linear" with the holder, "filtered" when finger-held


class TransferEnv:
    def __init__(self, cfg: EnvConfig, rng: np.random.Generator | None = None):
        self.cfg = cfg
        self.model, self.info = scene.build(cfg.scene)
        self.data = mujoco.MjData(self.model)
        self.rng = rng or np.random.default_rng(cfg.scene.seed)

    # ----------------------------------------------------------------------------------
    def reset(self) -> np.ndarray:
        p = self.cfg.params
        m, d, info = self.model, self.data, self.info
        tip = info.plate_a + np.array([0, 0, p.approach_height + 0.03])
        scene.reset(m, d, info, tip)
        self.runner = controller._Runner(m, d, info, p, None, 1.0)
        if self.runner.held:
            self.runner.run([
                Phase("handover", p.t_pin_leaves, tip, tip, grip=False, pin="tool+leaves"),
                Phase("handover", p.t_pin_tool, tip, tip, grip=False, pin="tool"),
                Phase("regrasp", p.t_regrasp, tip, tip, grip=False),
            ])
            self.runner.set_grasp_reference()
        self.runner.run([Phase("settle", p.t_settle, tip, tip, grip=False)])
        self.command = tip.copy()
        self.grip = 0.0
        self.results: list[controller.TransferResult] = []
        self.done = False
        self._next_target()
        return self.observe()

    def _next_target(self) -> None:
        d, info = self.data, self.info
        self.target = controller.pick_target_bean(d, info)
        if self.target is None or len(self.results) >= self.cfg.n_transfers:
            self.done = True
            return
        observed = d.body(self.target).xpos.copy()
        observed[:2] += self.rng.normal(0.0, self.cfg.params.perception_noise, 2)
        self.target_obs = observed
        self.drop = controller.drop_point(len(self.results), info)
        self.others0 = {n: d.body(n).xpos.copy() for n in info.bean_names
                        if n != self.target and controller.where(d.body(n).xpos, info) == "A"}
        self.t_start = d.time
        self.closed_once = False
        self.lifted = False
        self.runner.reset_slip()

    # ----------------------------------------------------------------------------------
    def tip(self) -> np.ndarray:
        return self.data.site("tweezer_tip").xpos.copy()

    def observe(self) -> np.ndarray:
        d, m = self.data, self.model
        tip = self.tip()
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_SITE, m.site("tweezer_tip").id, vel, 0)
        leaf_mm = 1000 * self.info.tweezer_length * np.array(
            [d.joint("tweezer_th_hinge").qpos[0], d.joint("tweezer_ff_hinge").qpos[0]])
        return np.concatenate([
            tip, vel[3:], self.target_obs - tip, self.drop - tip,
            controller.pad_forces(m, d), leaf_mm, [self.grip],
        ]).astype(np.float32)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, bool]:
        """Apply one 50 ms command: tip target relative to the current tip, and grip (> 0.5 = close)."""
        d, info = self.data, self.info
        new_command = self.tip() + np.asarray(action[:3], dtype=float)
        # Safety floor (like a workspace limit on a real arm): the tip never goes closer to the
        # plates than the grasp clearance; pushing the hard tip pads into a plate jams the leaves.
        floor = self.info.plate_a[2] + self.cfg.params.min_tip_clearance
        new_command[2] = max(new_command[2], floor)
        grip = bool(action[3] > 0.5)
        self.closed_once |= grip
        profile = self.cfg.profile or ("filtered" if self.runner.held else "linear")
        ph = Phase("policy", DT, self.command, new_command, grip=grip, profile=profile)
        self.lifted |= self.runner.run([ph], self.target)
        self.command, self.grip = new_command, float(grip)

        # An attempt ends once the tweezers have closed and then opened again high above
        # plate B (the bean was released), or on timeout / a dropped tool.
        tip = self.tip()
        released = (self.closed_once and not grip and tip[2] > info.plate_b[2] + 0.04
                    and np.linalg.norm(tip[:2] - self.drop[:2]) < 0.03)
        if released or d.time - self.t_start > self.cfg.attempt_timeout or self.runner.dropped:
            self._finish_attempt()
        return self.observe(), self.done

    def _finish_attempt(self) -> None:
        d, info = self.data, self.info
        final = d.body(self.target).xpos.copy()
        slip = self.runner.reset_slip()
        self.results.append(controller.TransferResult(
            bean=self.target,
            success=controller.where(final, info) == "B",
            lifted=self.lifted,
            final_pos=final,
            disturbed=int(sum(np.linalg.norm(d.body(n).xpos[:2] - q[:2]) > 0.01 for n, q in self.others0.items())),
            t_start=self.t_start,
            t_end=d.time,
            slip_mm=float(slip[0]),
            slip_deg=float(slip[1]),
            tool_dropped=self.runner.dropped,
        ))
        if self.runner.dropped:
            self.done = True
            return
        self._next_target()

    def finish(self) -> list[controller.TransferResult]:
        """Let things settle and re-check the transferred beans (as run_episode does)."""
        if not self.runner.dropped:
            self.runner.run([Phase("done", 0.5, self.command, self.command, grip=False)])
        for tr in self.results:
            tr.final_pos = self.data.body(tr.bean).xpos.copy()
            tr.success = tr.success and controller.where(tr.final_pos, self.info) == "B"
        return self.results


class ScriptedPolicy:
    """The scripted plan, sampled every 50 ms through the policy interface."""

    def __init__(self, env: TransferEnv):
        self.env = env
        self.plan: list[Phase] | None = None
        self.target = None

    def act(self, obs: np.ndarray) -> np.ndarray:
        env = self.env
        if env.target != self.target:  # a new attempt: plan from the current command
            self.target = env.target
            p = env.cfg.params
            self.plan = controller.plan(env.command, env.target_obs, env.info.plate_a[2], env.drop, p)
            self.t = 0.0
        self.t += DT
        tip_target, grip = self._sample(self.t)
        return np.concatenate([tip_target - env.tip(), [1.0 if grip else 0.0]]).astype(np.float32)

    def _sample(self, t: float) -> tuple[np.ndarray, bool]:
        for ph in self.plan:
            if t <= ph.duration:
                s = controller._min_jerk(t / ph.duration)[0]
                return ph.tip_from + s * (ph.tip_to - ph.tip_from), ph.grip
            t -= ph.duration
        last = self.plan[-1]
        return last.tip_to.copy(), last.grip


def rollout(env: TransferEnv, policy, action_noise: float = 0.0, max_steps: int = 4000,
            rng: np.random.Generator | None = None, record: bool = False):
    """Run an episode. With `action_noise` (m), the executed tip command is perturbed while the
    recorded label stays the policy's clean action (DART-style data for imitation learning)."""
    rng = rng or np.random.default_rng(0)
    obs = env.reset()
    observations, actions = [], []
    for _ in range(max_steps):
        if env.done:
            break
        act = policy.act(obs)
        if record:
            observations.append(obs)
            actions.append(act)
        executed = act.copy()
        if action_noise > 0:
            executed[:3] += rng.normal(0.0, action_noise, 3)
        obs, done = env.step(executed)
    results = env.finish()
    if record:
        return results, np.array(observations), np.array(actions)
    return results


def scripted_env_config(seed: int, speed: float = 1.0, noise: float = 0.0, **scene_kw) -> EnvConfig:
    return EnvConfig(
        scene=scene.SceneConfig(seed=seed, **scene_kw),
        params=replace(TransferParams(), speed=speed, perception_noise=noise),
    )
