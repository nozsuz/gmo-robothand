"""Train the low-dimensional Diffusion Policy on collected demonstrations.

    uv run python scripts/train_policy.py --demos holder --steps 30000
The demos file is looked up as outputs/demos_<name>.npz or under /kaggle/input (when the
collecting kernel is attached as a kernel source).
"""

from __future__ import annotations

import argparse
import glob
import time
from pathlib import Path

import numpy as np
import torch

from robothand_sim.paths import OUTPUT_DIR
from robothand_sim.policy import DiffusionPolicy, Normalizer, PolicyConfig, make_windows


def find_demos(name: str) -> Path:
    candidates = [OUTPUT_DIR / f"demos_{name}.npz", *map(Path, glob.glob(f"/kaggle/input/**/demos_{name}.npz", recursive=True))]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"demos_{name}.npz not found in {candidates}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", default="holder")
    ap.add_argument("--name", default=None, help="checkpoint name (default: the demos name)")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--successful-only", action="store_true", help="drop episodes with failed transfers")
    args = ap.parse_args()
    name = args.name or args.demos

    path = find_demos(args.demos)
    data = np.load(path)
    obs, act, episode = data["obs"], data["act"], data["episode"]
    if args.successful_only:
        good = np.flatnonzero(data["success"] == data["attempts"])
        keep = np.isin(episode, good)
        obs, act, episode = obs[keep], act[keep], episode[keep]
    cfg = PolicyConfig()
    x, y = make_windows(obs, act, episode, cfg)
    print(f"{path}: {len(np.unique(episode))} episodes, {len(x)} windows")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DiffusionPolicy(cfg, Normalizer(x, y)).to(device)
    print(f"device={device} params={sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    xt, yt = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    ema = torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999))
    gen = torch.Generator(device=device).manual_seed(0)
    t0, running = time.perf_counter(), []
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, len(xt), (args.batch,), device=device, generator=gen)
        loss = model.loss(xt[idx], yt[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        ema.update_parameters(model)
        running.append(loss.item())
        if step % 2000 == 0 or step == args.steps:
            print(f"step {step}: loss {np.mean(running):.4f} ({time.perf_counter() - t0:.0f}s)", flush=True)
            running = []
    final = ema.module.cpu().eval()
    OUTPUT_DIR.mkdir(exist_ok=True)
    final.save(OUTPUT_DIR / f"policy_{name}.pt")
    print(f"saved {OUTPUT_DIR / f'policy_{name}.pt'}")


if __name__ == "__main__":
    main()
