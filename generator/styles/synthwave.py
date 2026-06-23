from __future__ import annotations

import math

import numpy as np

from generator.canvas import Canvas
from generator.styles.base import VisualStyle

# Neon synthwave palette
C_VOID = np.array([8, 4, 20], dtype=np.uint8)
C_SKY_TOP = np.array([20, 0, 48], dtype=np.uint8)
C_SKY_BOT = np.array([255, 60, 120], dtype=np.uint8)
C_SUN1 = np.array([255, 200, 60], dtype=np.uint8)
C_SUN2 = np.array([255, 120, 40], dtype=np.uint8)
C_SUN3 = np.array([255, 40, 100], dtype=np.uint8)
C_GRID = np.array([255, 0, 128], dtype=np.uint8)
C_CYAN = np.array([0, 255, 255], dtype=np.uint8)
C_PURPLE = np.array([180, 60, 255], dtype=np.uint8)
C_WHITE = np.array([240, 240, 255], dtype=np.uint8)
C_MOUNT = np.array([12, 8, 28], dtype=np.uint8)


class SynthwaveStyle(VisualStyle):
    name = "synthwave"
    description = "Neon sunset grid — retro 80s drive at night"

    def render_frame(self, t, rms, peak, bars, title, artist) -> np.ndarray:
        c = self.canvas()
        bass = float(rms)

        # Sky gradient
        horizon = 88
        for y in range(horizon):
            p = y / horizon
            c.img[y, :] = (C_SKY_TOP * (1 - p) + C_SKY_BOT * p).astype(np.uint8)

        # Retro sun
        sun_y = 52 + int(math.sin(t * 0.8) * 2)
        for i, (col, r) in enumerate([(C_SUN1, 22), (C_SUN2, 17), (C_SUN3, 12)]):
            cx, cy = 160, sun_y + i * 2
            for deg in range(0, 360, 8):
                rad = math.radians(deg)
                px = int(cx + math.cos(rad) * r)
                py = int(cy + math.sin(rad) * r * 0.55)
                c.rect(px, py, 2, 2, col)
        for cut in range(6):
            cy = sun_y + 8 + cut * 5
            c.rect(130, cy, 60, 3, C_SKY_BOT)

        # Mountains
        pts = [0, 28, 48, 18, 88, 40, 130, 14, 180, 36, 220, 20, 260, 38, 320, 24]
        for i in range(0, len(pts) - 2, 2):
            x0, h0 = pts[i], pts[i + 1]
            x1, h1 = pts[i + 2], pts[i + 3]
            steps = max(abs(x1 - x0), 1)
            for s in range(steps):
                f = s / steps
                x = x0 + int((x1 - x0) * f)
                h = int(h0 + (h1 - h0) * f)
                c.rect(x, horizon - h, 2, h + 20, C_MOUNT)

        # Floor + perspective grid
        c.rect(0, horizon, c.pw, c.ph - horizon, C_VOID)
        vanish_x, vanish_y = c.pw // 2, horizon
        scroll = (t * 40 + bass * 20) % 14
        for row in range(0, 14):
            depth = row + scroll / 14
            y = horizon + int((depth / 14) ** 1.6 * (c.ph - horizon))
            if y >= c.ph:
                continue
            fade = 0.4 + 0.6 * (1 - depth / 14)
            col = (C_GRID * fade + C_PURPLE * (1 - fade) * 0.3).astype(np.uint8)
            c.rect(0, y, c.pw, 1, col)
        for col_x in range(-c.pw, c.pw * 2, 18):
            x_base = col_x + int(t * 12) % 18
            steps = 10
            for s in range(steps):
                f0, f1 = s / steps, (s + 1) / steps
                y0 = horizon + int(f0 ** 1.5 * (c.ph - horizon))
                y1 = horizon + int(f1 ** 1.5 * (c.ph - horizon))
                x0 = int(vanish_x + (x_base - vanish_x) * f0)
                x1 = int(vanish_x + (x_base - vanish_x) * f1)
                steps_line = max(abs(x1 - x0), abs(y1 - y0), 1)
                for k in range(steps_line):
                    f = k / steps_line
                    px = int(x0 + (x1 - x0) * f)
                    py = int(y0 + (y1 - y0) * f)
                    c.rect(px, py, 1, 1, C_GRID if s % 2 else C_CYAN)

        # Beat pulse ring
        if peak > 0.45:
            ring_r = int(20 + peak * 30 + bass * 10)
            for deg in range(0, 360, 14):
                rad = math.radians(deg + t * 80)
                px = int(160 + math.cos(rad) * ring_r)
                py = int(110 + math.sin(rad) * ring_r * 0.35)
                c.rect(px, py, 2, 2, C_CYAN if peak > 0.65 else C_PURPLE)

        c.bars(bars, bass, [C_CYAN, C_PURPLE, C_GRID, C_SUN3])
        c.title_card(title, artist, C_VOID, C_WHITE, C_CYAN, peak)
        c.scanlines(0.12)
        c.vignette(0.35)
        return c.img
