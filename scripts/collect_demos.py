"""Collect scripted demonstrations through the 20 Hz policy interface (src/robothand_sim/env.py).

Half of the episodes add small noise to the executed tip commands while recording the clean
scripted action (DART), so the data also covers recovering from being slightly off the plan.

    uv run python scripts/collect_demos.py --name holder --episodes 200
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from robothand_sim import env as envlib
from robothand_sim.paths import OUTPUT_DIR

# Training layouts: seeds 1000+ (dev 0-99, scripted evaluation 100-399, policy evaluation 500+).
SEED_OFFSET = 1000
NOISE_XY, NOISE_Z = 0.0007, 0.0003  # executed-command noise for DART episodes [m]


def collect(job: dict) -> dict:
    seed = job["seed"]
    cfg = envlib.scripted_env_config(seed, speed=job["speed"], noise=job["perception_noise"],
                                     tool_mount=job["tool_mount"], hand=job["hand"])
    e = envlib.TransferEnv(cfg)
    rng = np.random.default_rng(seed)
    dart = seed % 2 == 1
    policy = envlib.ScriptedPolicy(e)
    obs = e.reset()
    observations, actions = [], []
    while not e.done and len(actions) < 4000:
        act = policy.act(obs)
        observations.append(obs)
        actions.append(act)
        executed = act.copy()
        if dart:
            executed[:2] += rng.normal(0.0, NOISE_XY, 2)
            executed[2] += rng.normal(0.0, NOISE_Z)
        obs, _ = e.step(executed)
    results = e.finish()
    return {
        "seed": seed,
        "obs": np.array(observations, dtype=np.float32),
        "act": np.array(actions, dtype=np.float32),
        "success": sum(r.success for r in results),
        "attempts": len(results),
        "dart": dart,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="holder")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--perception-noise", type=float, default=0.0)
    ap.add_argument("--tool-mount", default="holder")
    ap.add_argument("--hand", default="shadow")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()

    jobs = [{"seed": SEED_OFFSET + i, "speed": args.speed, "perception_noise": args.perception_noise,
             "tool_mount": args.tool_mount, "hand": args.hand} for i in range(args.episodes)]
    t = time.perf_counter()
    with ProcessPoolExecutor(args.workers) as ex:
        eps = list(ex.map(collect, jobs))
    ok = sum(e["success"] for e in eps)
    att = sum(e["attempts"] for e in eps)
    print(f"{len(eps)} episodes, scripted success {ok}/{att}, {sum(len(e['act']) for e in eps)} samples, "
          f"{time.perf_counter() - t:.0f}s")
    OUTPUT_DIR.mkdir(exist_ok=True)
    np.savez_compressed(
        OUTPUT_DIR / f"demos_{args.name}.npz",
        obs=np.concatenate([e["obs"] for e in eps]),
        act=np.concatenate([e["act"] for e in eps]),
        episode=np.concatenate([np.full(len(e["act"]), i) for i, e in enumerate(eps)]),
        seed=np.array([e["seed"] for e in eps]),
        success=np.array([e["success"] for e in eps]),
        attempts=np.array([e["attempts"] for e in eps]),
        dart=np.array([e["dart"] for e in eps]),
    )
    print(f"saved {OUTPUT_DIR / f'demos_{args.name}.npz'}")


if __name__ == "__main__":
    main()
