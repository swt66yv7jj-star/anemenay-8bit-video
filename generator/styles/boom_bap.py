from __future__ import annotations

import math

import numpy as np

from generator.styles.base import VisualStyle

# Dusty boom bap palette
C_WALL = np.array([72, 48, 36], dtype=np.uint8)
C_BRICK = np.array([96, 60, 44], dtype=np.uint8)
C_MORTAR = np.array([48, 32, 24], dtype=np.uint8)
C_WARM = np.array([180, 140, 88], dtype=np.uint8)
C_GOLD = np.array([212, 168, 72], dtype=np.uint8)
C_DUST = np.array([140, 110, 72], dtype=np.uint8)
C_VINYL = np.array([16, 12, 12], dtype=np.uint8)
C_GROOVE = np.array([32, 24, 20], dtype=np.uint8)
C_LABEL = np.array([168, 48, 40], dtype=np.uint8)
C_LABEL2 = np.array([220, 200, 160], dtype=np.uint8)
C_SMOKE = np.array([80, 68, 56], dtype=np.uint8)
C_KICK = np.array([255, 120, 40], dtype=np.uint8)
C_SNARE = np.array([255, 220, 140], dtype=np.uint8)
C_WAVE = np.array([200, 160, 100], dtype=np.uint8)
C_INK = np.array([20, 14, 10], dtype=np.uint8)


class BoomBapStyle(VisualStyle):
    name = "boombap"
    description = "Dusty vinyl crate — brick wall, spinning record, sample waveforms"

    def render_frame(self, t, rms, peak, bars, title, artist) -> np.ndarray:
        c = self.canvas()
        bass = float(rms)

        self._draw_brick_wall(c, t)
        self._draw_crates(c)
        self._draw_smoke(c, t, bass)
        self._draw_vinyl(c, 118, 42, t, peak, bass)
        self._draw_waveform(c, bars, t, peak)
        self._draw_drums(c, t, peak, bass)
        self._draw_mpc_pads(c, peak, bass)

        c.bars(bars, bass, [C_GOLD, C_WARM, C_DUST, C_KICK], base_y=c.ph - 8)
        c.title_card(title, artist, C_INK, C_LABEL2, C_GOLD, peak)
        self._draw_vinyl_dust(c, t)
        c.scanlines(0.18)
        c.vignette(0.4)
        return c.img

    def _draw_brick_wall(self, c, t: float) -> None:
        c.img[:] = C_MORTAR
        for row in range(0, c.ph, 8):
            offset = 6 if (row // 8) % 2 else 0
            for col in range(-12 + offset, c.pw + 12, 24):
                shade = C_BRICK if (row + col) % 17 else C_WALL
                c.rect(col, row, 22, 6, shade)

    def _draw_crates(self, c) -> None:
        # Record crates left + right
        for base_x in (8, 248):
            c.rect(base_x, 118, 52, 40, C_WALL)
            c.rect(base_x + 2, 120, 48, 2, C_DUST)
            for i in range(5):
                c.rect(base_x + 4 + i * 9, 124, 7, 28, C_VINYL)
                c.rect(base_x + 6 + i * 9, 130, 3, 3, C_LABEL)

    def _draw_smoke(self, c, t: float, bass: float) -> None:
        for i in range(12):
            x = 40 + i * 22 + int(math.sin(t * 0.7 + i) * 8)
            y = 90 - int(abs(math.sin(t * 0.5 + i * 0.8)) * 20 + bass * 8)
            c.rect(x, y, 6, 2, C_SMOKE)
            c.rect(x + 2, y - 2, 4, 2, C_SMOKE)

    def _draw_vinyl(self, c, ox: int, oy: int, t: float, peak: float, bass: float) -> None:
        cx, cy = ox + 42, oy + 42
        r = 38 + int(bass * 4)
        spin = t * (1.8 + bass * 0.8)

        for deg in range(0, 360, 6):
            rad = math.radians(deg + spin * 40)
            px = int(cx + math.cos(rad) * r)
            py = int(cy + math.sin(rad) * r)
            col = C_GROOVE if deg % 12 else C_VINYL
            c.rect(px, py, 2, 2, col)

        for ring in (32, 24, 16):
            for deg in range(0, 360, 18):
                rad = math.radians(deg + spin * 40)
                px = int(cx + math.cos(rad) * ring)
                py = int(cy + math.sin(rad) * ring)
                c.rect(px, py, 1, 1, C_GROOVE)

        c.rect(cx - 18, cy - 18, 36, 36, C_LABEL)
        c.rect(cx - 12, cy - 12, 24, 24, C_LABEL2)
        c.rect(cx - 4, cy - 4, 8, 8, C_VINYL)
        if peak > 0.5:
            c.rect(cx - 2, cy - 2, 4, 4, C_GOLD)

    def _draw_waveform(self, c, bars: np.ndarray, t: float, peak: float) -> None:
        base_y = 108
        w = c.pw - 40
        x0 = 20
        for i, lv in enumerate(bars):
            x = x0 + int(i * w / len(bars))
            h = int(lv * 28 + peak * 8)
            col = C_WAVE if lv < 0.6 else C_KICK if lv < 0.85 else C_SNARE
            c.rect(x, base_y - h, 3, h, col)
            if peak > 0.6 and i % 4 == 0:
                c.rect(x, base_y - h - 2, 3, 2, C_GOLD)

    def _draw_drums(self, c, t: float, peak: float, bass: float) -> None:
        # Kick / snare indicators — boom bap pocket
        beat = t * 2.0
        kick_on = (beat % 1.0) < 0.12 or peak > 0.7
        snare_on = abs((beat + 0.5) % 1.0 - 0.5) < 0.1 or (peak > 0.55 and int(t * 4) % 2 == 1)

        if kick_on or bass > 0.55:
            c.rect(24, 148, 20, 10, C_KICK)
            c.text("K", 30, 150, C_INK, 1)
        if snare_on:
            c.rect(276, 148, 20, 10, C_SNARE)
            c.text("S", 282, 150, C_INK, 1)

    def _draw_mpc_pads(self, c, peak: float, bass: float) -> None:
        ox, oy = 196, 130
        for row in range(4):
            for col in range(4):
                lit = (row + col + int(peak * 4)) % 3 == 0 and (peak > 0.4 or bass > 0.45)
                col_c = C_GOLD if lit else C_WALL
                c.rect(ox + col * 10, oy + row * 8, 8, 6, col_c)

    def _draw_vinyl_dust(self, c, t: float) -> None:
        for i in range(30):
            seed = i * 2347
            if abs(math.sin(t * 3 + i)) > 0.82:
                x = (seed + int(t * 5)) % c.pw
                y = (seed // 19) % c.ph
                c.rect(x, y, 1, 1, C_DUST)
