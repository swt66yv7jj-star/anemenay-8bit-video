from __future__ import annotations

import math

import numpy as np

from generator.styles.base import VisualStyle

C_BG = np.array([12, 12, 16], dtype=np.uint8)
C_FG = np.array([220, 220, 230], dtype=np.uint8)
C_ACCENT = np.array([0, 200, 255], dtype=np.uint8)


class BlankStyle(VisualStyle):
    """
    Starter template — copy this file to create a new style.

    1. Duplicate this file as generator/styles/your_style.py
    2. Subclass VisualStyle, set name + description
    3. Implement render_frame()
    4. Register it in generator/styles/__init__.py
    """

    name = "blank"
    description = "Minimal template — edit to build your own look"

    def render_frame(self, t, rms, peak, bars, title, artist) -> np.ndarray:
        c = self.canvas()
        bass = float(rms)

        # Background pulse
        pulse = 0.5 + 0.5 * math.sin(t * 2)
        bg = (C_BG * (1 - bass * 0.3) + C_ACCENT * bass * 0.15).astype(np.uint8)
        c.img[:] = bg

        # Center crosshair that reacts to beat
        cx, cy = c.pw // 2, c.ph // 2
        size = int(10 + peak * 20)
        c.rect(cx - size, cy, size * 2, 1, C_FG)
        c.rect(cx, cy - size, 1, size * 2, C_FG)

        c.bars(bars, bass, [C_ACCENT, C_FG, C_BG + 40])
        c.title_card(title, artist, C_BG, C_FG, C_ACCENT, peak)
        c.scanlines(0.1)
        return c.img
