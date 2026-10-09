"""Interactive MuJoCo viewer running the scripted controller in (roughly) real time.

    uv run python scripts/view.py -n 5
"""

from __future__ import annotations

import argparse
import time

import mujoco
import mujoco.viewer

from robothand_sim import controller, scene


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--n-beans", type=int, default=5)
    ap.add_argument("--hand", default="shadow", choices=["shadow", "revo2"])
    ap.add_argument("--tool-mount", default="holder", choices=["holder", "fingers"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    model, info = scene.build(scene.SceneConfig(seed=args.seed, n_beans=max(5, args.n_beans),
                                                hand=args.hand, tool_mount=args.tool_mount))
    data = mujoco.MjData(model)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0, 0, 0.06]
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 0.5, 0, -22
        wall0, sim0 = time.perf_counter(), data.time

        def on_frame(d: mujoco.MjData, phase: str) -> None:
            if not viewer.is_running():
                raise SystemExit
            viewer.sync()
            lag = (d.time - sim0) - (time.perf_counter() - wall0)
            if lag > 0:
                time.sleep(lag)

        res = controller.run_episode(model, data, info, n_transfers=args.n_beans, on_frame=on_frame, frame_dt=1 / 60)
        print(f"{sum(t.success for t in res.transfers)}/{args.n_beans} transferred")
        while viewer.is_running():
            time.sleep(0.05)


if __name__ == "__main__":
    main()
