"""Low-dimensional Diffusion Policy (Chi et al., 2023) for the 20 Hz transfer interface.

The policy sees the last `n_obs` observations and predicts a chunk of `horizon` actions.
Each chunk action k is a tip target relative to the tip position at the time the chunk was
predicted (so chunk targets do not drift with execution), plus a grip command. A residual
MLP predicts the noise (DDPM training, deterministic DDIM sampling at run time).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from robothand_sim.env import ACT_DIM, OBS_DIM


@dataclass
class PolicyConfig:
    n_obs: int = 2
    horizon: int = 16
    n_exec: int = 8  # actions executed before re-planning
    hidden: int = 512
    depth: int = 4
    train_steps: int = 100  # diffusion steps
    infer_steps: int = 10  # DDIM steps at run time


# ------------------------------------------------------------------------------------------
# Data


def chunk_targets(obs: np.ndarray, act: np.ndarray) -> np.ndarray:
    """Per-step absolute tip targets (target_{t+1}) and grips from recorded (obs, act)."""
    return np.concatenate([obs[:, :3] + act[:, :3], act[:, 3:4]], axis=1)


def make_windows(obs: np.ndarray, act: np.ndarray, episode: np.ndarray, cfg: PolicyConfig):
    """(observation history, action chunk) training pairs; chunks are relative to the anchor tip."""
    xs, ys = [], []
    for ep in np.unique(episode):
        o, a = obs[episode == ep], act[episode == ep]
        targets = chunk_targets(o, a)
        n = len(o)
        for t in range(n):
            hist = [o[max(t - k, 0)] for k in reversed(range(cfg.n_obs))]
            idx = np.minimum(np.arange(t, t + cfg.horizon), n - 1)
            chunk = targets[idx].copy()
            chunk[:, :3] -= o[t, :3]
            xs.append(np.concatenate(hist))
            ys.append(chunk.reshape(-1))
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


class Normalizer(nn.Module):
    """Observations: zero mean / unit std. Actions: per-dimension min/max to [-1, 1]."""

    def __init__(self, x: np.ndarray | None = None, y: np.ndarray | None = None, x_dim: int = 0, y_dim: int = 0):
        super().__init__()
        x_dim = x.shape[1] if x is not None else x_dim
        y_dim = y.shape[1] if y is not None else y_dim
        self.register_buffer("x_mean", torch.zeros(x_dim))
        self.register_buffer("x_std", torch.ones(x_dim))
        self.register_buffer("y_lo", -torch.ones(y_dim))
        self.register_buffer("y_hi", torch.ones(y_dim))
        if x is not None:
            self.x_mean.copy_(torch.from_numpy(x.mean(0)))
            self.x_std.copy_(torch.from_numpy(x.std(0) + 1e-6))
            lo, hi = y.min(0), y.max(0)
            span = np.maximum(hi - lo, 1e-4)
            self.y_lo.copy_(torch.from_numpy(lo - 0.05 * span))
            self.y_hi.copy_(torch.from_numpy(hi + 0.05 * span))

    def x(self, x):
        return (x - self.x_mean) / self.x_std

    def y(self, y):
        return 2 * (y - self.y_lo) / (self.y_hi - self.y_lo) - 1

    def y_inv(self, y):
        return (y + 1) / 2 * (self.y_hi - self.y_lo) + self.y_lo


# ------------------------------------------------------------------------------------------
# Model


class Block(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, x):
        return x + self.net(x)


class NoiseNet(nn.Module):
    def __init__(self, cond_dim: int, act_dim: int, hidden: int, depth: int):
        super().__init__()
        self.t_dim = 64
        self.inp = nn.Linear(act_dim + cond_dim + self.t_dim, hidden)
        self.blocks = nn.Sequential(*[Block(hidden) for _ in range(depth)])
        self.out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, act_dim))

    def t_embed(self, t):
        half = self.t_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        ang = t.float()[:, None] * freqs[None]
        return torch.cat([ang.sin(), ang.cos()], dim=1)

    def forward(self, noisy, t, cond):
        return self.out(self.blocks(self.inp(torch.cat([noisy, cond, self.t_embed(t)], dim=1))))


def cosine_alphas_cumprod(steps: int) -> torch.Tensor:
    s = 0.008
    x = torch.linspace(0, steps, steps + 1)
    f = torch.cos(((x / steps) + s) / (1 + s) * math.pi / 2) ** 2
    ac = f / f[0]
    betas = (1 - ac[1:] / ac[:-1]).clamp(max=0.999)
    return torch.cumprod(1 - betas, dim=0)


class DiffusionPolicy(nn.Module):
    def __init__(self, cfg: PolicyConfig, normalizer: Normalizer):
        super().__init__()
        self.cfg = cfg
        self.norm = normalizer
        self.act_dim = cfg.horizon * ACT_DIM
        self.net = NoiseNet(cfg.n_obs * OBS_DIM, self.act_dim, cfg.hidden, cfg.depth)
        self.register_buffer("ac", cosine_alphas_cumprod(cfg.train_steps))

    def loss(self, x, y):
        cond, y0 = self.norm.x(x), self.norm.y(y)
        t = torch.randint(0, self.cfg.train_steps, (len(y0),), device=y0.device)
        noise = torch.randn_like(y0)
        a = self.ac[t][:, None]
        noisy = a.sqrt() * y0 + (1 - a).sqrt() * noise
        return nn.functional.mse_loss(self.net(noisy, t, cond), noise)

    @torch.no_grad()
    def sample(self, x):
        """DDIM (eta = 0) from noise to an action chunk, for a batch of observation histories."""
        cond = self.norm.x(x)
        y = torch.randn(len(x), self.act_dim, device=x.device)
        ts = torch.linspace(self.cfg.train_steps - 1, 0, self.cfg.infer_steps, device=x.device).long()
        for i, t in enumerate(ts):
            a = self.ac[t]
            a_prev = self.ac[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=x.device)
            eps = self.net(y, t.expand(len(y)), cond)
            y0 = ((y - (1 - a).sqrt() * eps) / a.sqrt()).clamp(-1.5, 1.5)
            y = a_prev.sqrt() * y0 + (1 - a_prev).sqrt() * eps
        return self.norm.y_inv(y).view(len(x), self.cfg.horizon, ACT_DIM)

    def save(self, path) -> None:
        torch.save({"cfg": asdict(self.cfg), "state": self.state_dict(),
                    "x_dim": self.norm.x_mean.numel(), "y_dim": self.norm.y_lo.numel()}, path)

    @staticmethod
    def load(path, device="cpu") -> "DiffusionPolicy":
        ck = torch.load(path, map_location=device)
        cfg = PolicyConfig(**ck["cfg"])
        pol = DiffusionPolicy(cfg, Normalizer(x_dim=ck["x_dim"], y_dim=ck["y_dim"]))
        pol.load_state_dict(ck["state"])
        return pol.to(device).eval()


# ------------------------------------------------------------------------------------------
# Running in the env


class LearnedPolicy:
    """Receding-horizon execution: predict a chunk, execute `n_exec` of its actions, re-plan."""

    def __init__(self, model: DiffusionPolicy, env, seed: int = 0):
        self.model, self.env = model, env
        self.history: list[np.ndarray] = []
        self.queue: list[np.ndarray] = []
        self.target = None
        torch.manual_seed(seed)

    def act(self, obs: np.ndarray) -> np.ndarray:
        cfg = self.model.cfg
        if self.env.target != self.target:  # a new bean: drop the rest of the old chunk
            self.target, self.queue = self.env.target, []
        self.history = (self.history + [obs])[-cfg.n_obs:]
        if not self.queue:
            hist = [self.history[0]] * (cfg.n_obs - len(self.history)) + self.history
            x = torch.from_numpy(np.concatenate(hist)[None]).float()
            chunk = self.model.sample(x)[0].numpy()
            anchor = obs[:3].copy()
            self.queue = [np.concatenate([anchor + c[:3], c[3:]]) for c in chunk[: cfg.n_exec]]
        target = self.queue.pop(0)
        return np.concatenate([target[:3] - obs[:3], [float(target[3] > 0.5)]]).astype(np.float32)
