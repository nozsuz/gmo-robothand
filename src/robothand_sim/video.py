"""Two-panel video: fixed overview camera + a close-up camera that follows the tweezer tip."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class TwoPanelRecorder:
    def __init__(self, model: mujoco.MjModel, panel_w: int = 640, panel_h: int = 540, title: str = ""):
        self.model = model
        self.renderer = mujoco.Renderer(model, panel_h, panel_w)
        self.close = mujoco.MjvCamera()
        self.close.distance = 0.13
        self.close.azimuth = 200
        self.close.elevation = -12
        self.title = title
        self.frames: list[np.ndarray] = []
        self.counter = ""  # e.g. "transferred 1/1"
        self.font = ImageFont.load_default()
        for name in ("arial.ttf", "DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
            try:
                self.font = ImageFont.truetype(name, 22)
                break
            except OSError:
                continue

    def __call__(self, data: mujoco.MjData, phase: str) -> None:
        self.renderer.update_scene(data, camera="overview")
        left = self.renderer.render()
        tip = data.site("tweezer_tip").xpos
        # follow the tip smoothly, but keep the close-up above the table
        self.close.lookat[:] = [tip[0], tip[1], max(tip[2], 0.02)]
        self.renderer.update_scene(data, camera=self.close)
        right = self.renderer.render()
        img = Image.fromarray(np.hstack([left, right]))
        draw = ImageDraw.Draw(img)
        draw.text((14, 10), self.title, fill=(255, 255, 255), font=self.font)
        draw.text((14, 38), f"t = {data.time:5.2f} s   phase: {phase}", fill=(255, 255, 255), font=self.font)
        if self.counter:
            draw.text((14, 66), self.counter, fill=(255, 220, 120), font=self.font)
        draw.text((left.shape[1] + 14, 10), "close-up (tracks tweezer tip)", fill=(255, 255, 255), font=self.font)
        self.frames.append(np.asarray(img))

    def save(self, path: Path, fps: int = 30) -> None:
        import imageio.v2 as imageio

        path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(path, self.frames, fps=fps, quality=8, macro_block_size=1)
