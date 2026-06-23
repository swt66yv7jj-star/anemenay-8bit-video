from __future__ import annotations

import math

import numpy as np

from generator.styles.base import VisualStyle

C_NIGHT = np.array([26, 26, 46], dtype=np.uint8)
C_SKY_TOP = np.array([255, 107, 157], dtype=np.uint8)
C_SKY_MID = np.array([196, 77, 255], dtype=np.uint8)
C_SKY_BOT = np.array([78, 205, 196], dtype=np.uint8)
C_MOON = np.array([255, 248, 220], dtype=np.uint8)
C_INK = np.array([20, 16, 36], dtype=np.uint8)
C_SKIN = np.array([255, 210, 180], dtype=np.uint8)
C_HAIR = np.array([40, 28, 68], dtype=np.uint8)
C_HAIR_HI = np.array([120, 80, 180], dtype=np.uint8)
C_EYE = np.array([60, 180, 255], dtype=np.uint8)
C_SHINE = np.array([255, 255, 255], dtype=np.uint8)
C_COAT = np.array([48, 48, 88], dtype=np.uint8)
C_ACCENT = np.array([255, 230, 102], dtype=np.uint8)
C_SAKURA = np.array([255, 170, 190], dtype=np.uint8)
C_WINDOW = np.array([255, 220, 120], dtype=np.uint8)
C_BUILD = np.array([16, 12, 32], dtype=np.uint8)


class AnimeStyle(VisualStyle):
    name = "anime"
    description = "Dusk rooftop anime — pixel character, sakura, city"

    def render_frame(self, t, rms, peak, bars, title, artist) -> np.ndarray:
        c = self.canvas()
        bass = float(rms)
        horizon = 108

        for y in range(horizon):
            p = y / horizon
            if p < 0.45:
                col = (C_SKY_TOP * (1 - p * 2) + C_SKY_MID * p * 2).astype(np.uint8)
            else:
                q = (p - 0.45) / 0.55
                col = (C_SKY_MID * (1 - q) + C_SKY_BOT * q).astype(np.uint8)
            c.img[y, :] = col

        c.rect(248, 24, 20, 20, C_MOON)
        c.rect(258, 26, 10, 16, C_SKY_MID)

        c.rect(0, horizon, c.pw, c.ph - horizon, C_NIGHT)
        heights = [42, 58, 36, 72, 48, 64, 40, 52, 44, 68, 38, 56]
        x = 0
        for i, h in enumerate(heights):
            w = 26 + (i % 3) * 2
            c.rect(x, horizon - h, w, h + 72, C_BUILD)
            for wy in range(horizon - h + 8, 106, 10):
                if (i + wy) % 3 and abs(math.sin(t * 2.5 + i + wy * 0.1)) > 0.3:
                    c.rect(x + 6, wy, 4, 4, C_WINDOW)
            x += w - 2

        c.rect(0, 118, c.pw, 4, C_INK)
        for i in range(24):
            seed = i * 3571
            px = (seed + int(t * (22 + i % 6))) % c.pw
            py = (seed // 11 + int(t * (35 + bass * 20 + i % 5))) % (c.ph - 10)
            c.rect(px, py, 2, 2, C_SAKURA)

        if peak > 0.35:
            cx, cy = c.pw // 2, 78
            for i in range(int(6 + peak * 14)):
                ang = (i / max(int(6 + peak * 14), 1)) * math.pi * 2 + t * 2
                for step in range(0, int(30 + peak * 50), 3):
                    c.rect(int(cx + math.cos(ang) * step), int(cy + math.sin(ang) * step), 2, 1, C_ACCENT)

        self._draw_character(c, 140, 62, t, peak, bass)
        c.bars(bars, bass, [C_SKY_MID, C_EYE, C_SAKURA])
        c.title_card(title, artist, C_INK, C_SHINE, C_ACCENT, peak)
        c.scanlines()
        c.vignette()
        return c.img

    def _draw_character(self, c, ox: int, oy: int, t: float, peak: float, bass: float) -> None:
        oy += int(math.sin(t * 6) * bass * 3)
        if peak > 0.65:
            oy -= int(peak * 4)
        for sx, sy, sw, sh in [(0, -14, 4, 6), (8, -16, 4, 8), (16, -18, 4, 10), (24, -16, 4, 8), (32, -14, 4, 6)]:
            c.rect(ox + sx, oy + sy, sw, sh, C_HAIR)
        c.rect(ox + 4, oy - 8, 28, 12, C_HAIR)
        c.rect(ox + 6, oy + 2, 24, 20, C_SKIN)
        for ex in (ox + 10, ox + 22):
            c.rect(ex, oy + 10, 8, 8, C_SHINE)
            c.rect(ex + 1, oy + 11, 6, 6, C_EYE)
        c.rect(ox + 2, oy + 22, 32, 28, C_COAT)
        c.rect(ox + 10, oy + 22, 16, 4, C_ACCENT)
