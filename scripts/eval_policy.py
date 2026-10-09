"""Closed-loop evaluation of a learned policy (and of the scripted plan through the same
20 Hz interface, for reference) on layouts never used for training.

    uv run python scripts/eval_policy.py --policy holder --episodes 50
    uv run python scripts/eval_policy.py --policy holder --video 1   # also render one episode
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from robothand_sim import env as envlib
from robothand_sim.paths import OUTPUT_DIR

EVAL_SEED_OFFSET = 500  # policy evaluation layouts (training uses 1000+)


def find_policy(name: str) -> Path:
    for c in [OUTPUT_DIR / f"policy_{name}.pt", *map(Path, glob.glob(f"/kaggle/input/**/policy_{name}.pt", recursive=True))]:
        if c.exists():
            return c
    raise FileNotFoundError(f"policy_{name}.pt not found")


def run(job: dict) -> list[dict]:
    import torch

    torch.set_num_threads(1)
    from robothand_sim.policy import DiffusionPolicy, LearnedPolicy

    seed = job["seed"]
    cfg = envlib.scripted_env_config(seed, speed=job["speed"], noise=job["perception_noise"],
                                     tool_mount=job["tool_mount"], hand=job["hand"])
    e = envlib.TransferEnv(cfg)
    if job["kind"] == "scripted":
        pol = envlib.ScriptedPolicy(e)
    else:
        pol = LearnedPolicy(DiffusionPolicy.load(job["policy_path"]), e, seed=seed)
    t = time.perf_counter()
    results = envlib.rollout(e, pol)
    wall = time.perf_counter() - t
    rows = []
    for k in range(cfg.n_transfers):
        r = results[k] if k < len(results) else None
        rows.append({"kind": job["kind"], "seed": seed, "index": k,
                     "success": int(bool(r and r.success)),
                     "seconds": round(r.t_end - r.t_start, 3) if r else "",
                     "tool_dropped": int(bool(r and r.tool_dropped)), "wall": round(wall, 1)})
    return rows


def render_episode(policy_path: Path, seed: int, out: Path, job: dict) -> None:
    import subprocess

    subprocess.run("apt-get -qq update > /dev/null 2>&1; apt-get -qq install -y libosmesa6 > /dev/null 2>&1", shell=True)
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    from robothand_sim.policy import DiffusionPolicy, LearnedPolicy
    from robothand_sim.video import TwoPanelRecorder

    cfg = envlib.scripted_env_config(seed, speed=job["speed"], tool_mount=job["tool_mount"], hand=job["hand"])
    e = envlib.TransferEnv(cfg)
    pol = LearnedPolicy(DiffusionPolicy.load(policy_path), e, seed=seed)
    rec = TwoPanelRecorder(e.model, title="Shadow Hand + tweezers  (learned: Diffusion Policy)")
    obs = e.reset()
    e.runner.on_frame, e.runner.frame_dt, e.runner.next_frame = rec, 1 / 30, 0.0
    e.runner.t0 = e.data.time
    while not e.done:
        rec.counter = f"transferred {sum(r.success for r in e.results)}/{cfg.n_transfers}"
        obs, _ = e.step(pol.act(obs))
    e.finish()
    rec.counter = f"transferred {sum(r.success for r in e.results)}/{cfg.n_transfers}"
    rec.save(out)
    print(f"saved {out} ({len(rec.frames)} frames)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="holder")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--perception-noise", type=float, default=0.0)
    ap.add_argument("--tool-mount", default="holder")
    ap.add_argument("--hand", default="shadow")
    ap.add_argument("--no-scripted", action="store_true")
    ap.add_argument("--video", type=int, default=0, help="render this many episodes")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()

    path = find_policy(args.policy)
    base = {"speed": args.speed, "perception_noise": args.perception_noise, "tool_mount": args.tool_mount,
            "hand": args.hand, "policy_path": str(path)}
    kinds = ["learned"] + ([] if args.no_scripted else ["scripted"])
    jobs = [{**base, "kind": k, "seed": EVAL_SEED_OFFSET + i} for k in kinds for i in range(args.episodes)]
    t = time.perf_counter()
    with ProcessPoolExecutor(args.workers) as ex:
        rows = [r for batch in ex.map(run, jobs) for r in batch]
    print(f"{len(jobs)} episodes in {time.perf_counter() - t:.0f}s")

    OUTPUT_DIR.mkdir(exist_ok=True)
    name = f"policy_eval_{args.policy}_x{args.speed:g}"
    with (OUTPUT_DIR / f"{name}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    lines = [f"# {name}", "", f"layout seeds {EVAL_SEED_OFFSET}-{EVAL_SEED_OFFSET + args.episodes - 1}, "
             f"5 beans each, speed x{args.speed:g}, tool {args.tool_mount}, hand {args.hand}", "",
             "| controller | success | s/bean (successful) | tool dropped (episodes) |", "|---|---|---|---|"]
    for k in kinds:
        sel = [r for r in rows if r["kind"] == k]
        ok = sum(r["success"] for r in sel)
        secs = [float(r["seconds"]) for r in sel if r["success"]]
        drops = len({r["seed"] for r in sel if r["tool_dropped"]})
        lines.append(f"| {k} | {ok}/{len(sel)} = {ok / len(sel):.0%} | {np.mean(secs) if secs else float('nan'):.2f} | {drops} |")
    md = "\n".join(lines) + "\n"
    (OUTPUT_DIR / f"{name}.md").write_text(md, encoding="utf-8")
    print(md)
    for i in range(args.video):
        render_episode(path, EVAL_SEED_OFFSET + i, OUTPUT_DIR / f"{name}_seed{EVAL_SEED_OFFSET + i}.mp4", base)


if __name__ == "__main__":
    main()
