from __future__ import annotations

import math

import numpy as np

FONT: dict[str, list[int]] = {
    "A": [0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11],
    "B": [0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E],
    "C": [0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E],
    "D": [0x1E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x1E],
    "E": [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F],
    "F": [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10],
    "G": [0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0E],
    "H": [0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11],
    "I": [0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E],
    "L": [0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F],
    "M": [0x11, 0x1B, 0x15, 0x11, 0x11, 0x11, 0x11],
    "N": [0x11, 0x19, 0x15, 0x13, 0x11, 0x11, 0x11],
    "O": [0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E],
    "R": [0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11],
    "S": [0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E],
    "T": [0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04],
    "U": [0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E],
    "V": [0x11, 0x11, 0x11, 0x11, 0x0A, 0x0A, 0x04],
    "W": [0x11, 0x11, 0x11, 0x15, 0x15, 0x1B, 0x11],
    "Y": [0x11, 0x11, 0x0A, 0x04, 0x04, 0x04, 0x04],
    " ": [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
}


class Canvas:
    def __init__(self, pw: int, ph: int) -> None:
        self.pw = pw
        self.ph = ph
        self.img = np.zeros((ph, pw, 3), dtype=np.uint8)

    def rect(self, x: int, y: int, w: int, h: int, color: np.ndarray) -> None:
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(self.pw, x + w), min(self.ph, y + h)
        if x0 < x1 and y0 < y1:
            self.img[y0:y1, x0:x1] = color

    def text(self, text: str, x: int, y: int, color: np.ndarray, scale: int = 1) -> None:
        cx = x
        for ch in text.upper():
            glyph = FONT.get(ch, FONT[" "])
            for row, bits in enumerate(glyph):
                for col in range(5):
                    if bits & (1 << (4 - col)):
                        self.rect(cx + col * scale, y + row * scale, scale, scale, color)
            cx += 6 * scale

    def bars(self, levels: np.ndarray, bass: float, colors: list[np.ndarray], base_y: int | None = None) -> None:
        n = len(levels)
        gap = 2
        bar_w = (self.pw - gap * (n + 1)) // n
        base = base_y if base_y is not None else self.ph - 10
        for i, lv in enumerate(levels):
            h = int(2 + lv * 22 + bass * 5)
            x = gap + i * (bar_w + gap)
            self.rect(x, base - h, bar_w, h, colors[i % len(colors)])

    def scanlines(self, strength: float = 0.15) -> None:
        self.img[0::2] = (self.img[0::2].astype(np.float32) * (1 - strength)).astype(np.uint8)

    def vignette(self, strength: float = 0.45) -> None:
        yy, xx = np.mgrid[0 : self.ph, 0 : self.pw]
        dist = np.sqrt(
            ((xx - self.pw / 2) / (self.pw * 0.55)) ** 2 + ((yy - self.ph / 2) / (self.ph * 0.55)) ** 2
        )
        self.img[:] = (
            self.img.astype(np.float32) * (1.0 - strength * np.clip(dist, 0, 1))[..., None]
        ).astype(np.uint8)

    def title_card(self, title: str, artist: str, bg: np.ndarray, fg: np.ndarray, accent: np.ndarray, peak: float) -> None:
        self.rect(72, 2, 176, 22, bg)
        self.rect(74, 4, 172, 18, bg // 2)
        self.text(title, max(4, (self.pw - len(title) * 6) // 2), 6, accent if peak > 0.5 else fg, 1)
        self.text(artist, max(4, (self.pw - len(artist) * 6) // 2), 14, fg, 1)
