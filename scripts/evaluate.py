"""Success-rate evaluation of the scripted controller over bean layouts and conditions.

Each episode transfers every bean on plate A (5 by default) for one layout seed. A grid
of conditions is swept; results go to outputs/eval_<name>.csv and a Markdown summary.

Layout seeds 0-99 are the development set (used while debugging the controller). Reported
numbers use seeds from 100 upwards, which were never looked at during development.

    uv run python scripts/evaluate.py --name baseline --seeds 20
    uv run python scripts/evaluate.py --name friction --seeds 20 --tip-friction 0.2 0.3 0.5 0.8 1.2
    uv run python scripts/evaluate.py --name speed --seeds 20 --speed 1.0 1.5 2.0 2.5
    uv run python scripts/evaluate.py --name noise --seeds 20 --noise 0 0.001 0.002 0.003
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np

from robothand_sim import controller, scene
from robothand_sim.paths import OUTPUT_DIR


def run_one(job: dict) -> list[dict]:
    cfg = scene.SceneConfig(
        seed=job["seed"], tip_friction=job["tip_friction"], n_beans=job["n_beans"], tool_mount=job["tool_mount"],
        hand=job["hand"],
    )
    model, info = scene.build(cfg)
    data = mujoco.MjData(model)
    params = replace(controller.TransferParams(), speed=job["speed"], perception_noise=job["noise"])
    res = controller.run_episode(
        model, data, info, params=params, n_transfers=job["n_beans"], rng=np.random.default_rng(job["seed"])
    )
    rows = []
    for k, tr in enumerate(res.transfers):
        rows.append(
            {
                **{key: job[key] for key in ("seed", "tip_friction", "speed", "noise")},
                "index": k,
                "bean": tr.bean,
                "success": int(tr.success),
                "lifted": int(tr.lifted),
                "disturbed": tr.disturbed,
                "seconds": round(tr.t_end - tr.t_start, 3),
                "tool_mount": job["tool_mount"],
                "hand": job["hand"],
                "slip_mm": round(tr.slip_mm, 2),
                "slip_deg": round(tr.slip_deg, 2),
            }
        )
    # Beans never attempted because the tool was dropped count as failures.
    for k in range(len(res.transfers), job["n_beans"]):
        rows.append(
            {
                **{key: job[key] for key in ("seed", "tip_friction", "speed", "noise")},
                "index": k,
                "bean": "",
                "success": 0,
                "lifted": 0,
                "disturbed": 0,
                "seconds": "",
                "tool_mount": job["tool_mount"],
                "hand": job["hand"],
                "slip_mm": "",
                "slip_deg": "",
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="baseline")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--seed-offset", type=int, default=100, help="first layout seed (>=100: held out)")
    ap.add_argument("--n-beans", type=int, default=5)
    ap.add_argument("--tip-friction", type=float, nargs="+", default=[scene.SceneConfig.tip_friction])
    ap.add_argument("--speed", type=float, nargs="+", default=[1.0])
    ap.add_argument("--noise", type=float, nargs="+", default=[0.0], help="perception noise std [m]")
    ap.add_argument("--tool-mount", default="holder", choices=["holder", "fingers"])
    ap.add_argument("--hand", default="shadow", choices=["shadow", "revo2"])
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    args = ap.parse_args()

    jobs = [
        {"seed": s, "tip_friction": f, "speed": v, "noise": n, "n_beans": args.n_beans, "tool_mount": args.tool_mount,
         "hand": args.hand}
        for f, v, n, s in itertools.product(
            args.tip_friction, args.speed, args.noise, range(args.seed_offset, args.seed_offset + args.seeds)
        )
    ]
    t = time.perf_counter()
    with ProcessPoolExecutor(args.workers) as ex:
        rows = [r for batch in ex.map(run_one, jobs) for r in batch]
    print(f"{len(jobs)} episodes, {len(rows)} transfers in {time.perf_counter() - t:.0f}s")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / f"eval_{args.name}.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    seeds = f"layout seeds {args.seed_offset}-{args.seed_offset + args.seeds - 1} x {args.n_beans} beans per condition"
    md = summarize(rows, f"eval: {args.name}", seeds)
    (args.out_dir / f"eval_{args.name}.md").write_text(md, encoding="utf-8")
    print(md)


def summarize(rows: list[dict], title: str, subtitle: str = "") -> str:
    """Markdown table, one line per (tip friction, speed, noise) condition."""
    key = lambda r: (float(r["tip_friction"]), float(r["speed"]), float(r["noise"]))  # noqa: E731
    lines = [
        f"# {title}",
        "",
        subtitle,
        "",
        "| tip friction | speed | perception noise [mm] | success | 95% CI | lifted | disturbed/transfer | s/bean | beans/5 min |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for cond in sorted({key(r) for r in rows}):
        sel = [r for r in rows if key(r) == cond]
        k, N = sum(int(r["success"]) for r in sel), len(sel)
        lo, hi = wilson(k, N)
        done = [float(r["seconds"]) for r in sel if int(r["success"])]  # time of successful transfers
        sec = float(np.mean(done)) if done else float("nan")
        f_, v, n = cond
        lines.append(
            f"| {f_:g} | {v:g} | {n * 1000:.1f} | {k}/{N} = {k / N:.0%} | {lo:.0%}-{hi:.0%} | "
            f"{np.mean([int(r['lifted']) for r in sel]):.0%} | {np.mean([int(r['disturbed']) for r in sel]):.2f} | "
            f"{sec:.2f} | {(300 / sec * k / N) if done else 0:.0f} |"
        )
    return "\n".join(lines) + "\n"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    den = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


if __name__ == "__main__":
    main()
