"""Scripted transfer of beans from plate A to plate B, optionally saving a video.

    uv run python scripts/transfer_beans.py                       # one bean
    uv run python scripts/transfer_beans.py -n 5 --video outputs/transfer_5.mp4
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np

from robothand_sim import controller, scene
from robothand_sim.video import TwoPanelRecorder


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--n-beans", type=int, default=1, help="beans to transfer")
    ap.add_argument("--hand", default="shadow", choices=["shadow", "revo2"])
    ap.add_argument("--tool-mount", default="holder", choices=["holder", "fingers"])
    ap.add_argument("--seed", type=int, default=0, help="bean layout seed")
    ap.add_argument("--speed", type=float, default=1.0, help="motion speed factor")
    ap.add_argument("--tip-friction", type=float, default=scene.SceneConfig.tip_friction)
    ap.add_argument("--video", type=Path, default=None, help="mp4 path (omit for no video)")
    args = ap.parse_args()

    cfg = scene.SceneConfig(seed=args.seed, tip_friction=args.tip_friction, n_beans=max(5, args.n_beans),
                            hand=args.hand, tool_mount=args.tool_mount)
    model, info = scene.build(cfg)
    data = mujoco.MjData(model)
    params = replace(controller.TransferParams(), speed=args.speed)
    title = f"Shadow Hand + tweezers  (scripted, speed x{args.speed:g})"
    rec = TwoPanelRecorder(model, title=title) if args.video else None
    done = []

    def on_transfer(tr: controller.TransferResult) -> None:
        done.append(tr.success)
        print(
            f"  {tr.bean}: success={tr.success} lifted={tr.lifted} disturbed={tr.disturbed} "
            f"t={tr.t_start:.2f}->{tr.t_end:.2f}s"
        )
        if rec:
            rec.counter = f"transferred {sum(done)}/{args.n_beans}"

    if rec:
        rec.counter = f"transferred 0/{args.n_beans}"
    t = time.perf_counter()
    res = controller.run_episode(
        model, data, info, params=params, n_transfers=args.n_beans, on_frame=rec, on_transfer=on_transfer
    )
    wall = time.perf_counter() - t
    n_ok = sum(tr.success for tr in res.transfers)
    per_bean = np.mean([tr.t_end - tr.t_start for tr in res.transfers])
    print(
        f"result: {n_ok}/{args.n_beans} on plate B, lost={res.n_lost}, "
        f"{per_bean:.2f} s/bean (sim), wall={wall:.1f}s"
    )
    if rec:
        rec.save(args.video)
        print(f"saved {args.video} ({len(rec.frames)} frames)")


if __name__ == "__main__":
    main()
