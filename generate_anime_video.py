#!/usr/bin/env python3
"""Anime-style 8-bit music video — no photos, pure pixel art."""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import imageio
import imageio_ffmpeg
import numpy as np
from PIL import Image

PW, PH = 320, 180
OUT_W, OUT_H = 1920, 1080
FPS = 30

# Anime dusk palette
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

FONT: dict[str, list[int]] = {
    "A": [0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11],
    "D": [0x1E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x1E],
    "E": [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F],
    "F": [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10],
    "H": [0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11],
    "I": [0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E],
    "M": [0x11, 0x1B, 0x15, 0x11, 0x11, 0x11, 0x11],
    "N": [0x11, 0x19, 0x15, 0x13, 0x11, 0x11, 0x11],
    "O": [0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E],
    "R": [0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11],
    "S": [0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E],
    "T": [0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04],
    "Y": [0x11, 0x11, 0x0A, 0x04, 0x04, 0x04, 0x04],
    " ": [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
}


def ffmpeg_exe() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()


def load_audio_pcm(path: Path, sample_rate: int = 22050) -> tuple[np.ndarray, float]:
    cmd = [
        ffmpeg_exe(), "-i", str(path), "-f", "s16le", "-acodec", "pcm_s16le",
        "-ac", "1", "-ar", str(sample_rate), "-v", "error", "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=True)
    samples = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    return samples, len(samples) / sample_rate


def build_envelope(samples: np.ndarray, sample_rate: int, frame_count: int) -> tuple[np.ndarray, np.ndarray]:
    spf = sample_rate / FPS
    rms = np.zeros(frame_count, dtype=np.float32)
    peak = np.zeros(frame_count, dtype=np.float32)
    for i in range(frame_count):
        chunk = samples[int(i * spf) : int((i + 1) * spf)]
        if chunk.size == 0:
            continue
        rms[i] = float(np.sqrt(np.mean(chunk * chunk)))
        peak[i] = float(np.max(np.abs(chunk)))
    kernel = np.ones(5, dtype=np.float32) / 5.0
    smooth = np.convolve(rms, kernel, mode="same")
    smooth /= float(smooth.max()) or 1.0
    peak /= float(peak.max()) or 1.0
    return smooth, peak


def draw_rect(img: np.ndarray, x: int, y: int, w: int, h: int, color: np.ndarray) -> None:
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(PW, x + w), min(PH, y + h)
    if x0 < x1 and y0 < y1:
        img[y0:y1, x0:x1] = color


def draw_text(img: np.ndarray, text: str, x: int, y: int, color: np.ndarray, scale: int = 1) -> None:
    cx = x
    for ch in text.upper():
        glyph = FONT.get(ch, FONT[" "])
        for row, bits in enumerate(glyph):
            for col in range(5):
                if bits & (1 << (4 - col)):
                    draw_rect(img, cx + col * scale, y + row * scale, scale, scale, color)
        cx += 6 * scale


def draw_anime_sky(img: np.ndarray, t: float, bass: float) -> None:
    horizon = 108
    for y in range(horizon):
        p = y / horizon
        if p < 0.45:
            c = (C_SKY_TOP * (1 - p * 2) + C_SKY_MID * p * 2).astype(np.uint8)
        else:
            q = (p - 0.45) / 0.55
            c = (C_SKY_MID * (1 - q) + C_SKY_BOT * q + C_NIGHT * bass * 0.15).astype(np.uint8)
        img[y, :] = c

    # Moon
    mx, my = 248, 24
    draw_rect(img, mx, my, 20, 20, C_MOON)
    draw_rect(img, mx + 10, my + 2, 10, 16, C_SKY_MID)


def draw_city(img: np.ndarray, t: float) -> None:
    draw_rect(img, 0, 108, PW, PH - 108, C_NIGHT)
    heights = [42, 58, 36, 72, 48, 64, 40, 52, 44, 68, 38, 56]
    x = 0
    for i, h in enumerate(heights):
        w = 26 + (i % 3) * 2
        draw_rect(img, x, 108 - h, w, h + 72, C_BUILD)
        for wy in range(108 - h + 8, 106, 10):
            if (i + wy) % 3 != 0:
                flicker = abs(math.sin(t * 2.5 + i + wy * 0.1)) > 0.3
                if flicker:
                    draw_rect(img, x + 6, wy, 4, 4, C_WINDOW)
                    if i % 4 == 0:
                        draw_rect(img, x + 14, wy, 4, 4, C_WINDOW)
        x += w - 2


def draw_rooftop(img: np.ndarray) -> None:
    draw_rect(img, 0, 118, PW, 4, C_INK)
    for x in range(0, PW, 8):
        draw_rect(img, x, 122, 6, 2, C_COAT)


def draw_speed_lines(img: np.ndarray, intensity: float, t: float) -> None:
    if intensity < 0.35:
        return
    cx, cy = PW // 2, 78
    count = int(6 + intensity * 14)
    for i in range(count):
        ang = (i / count) * math.pi * 2 + t * 2
        length = int(30 + intensity * 50)
        for step in range(0, length, 3):
            px = int(cx + math.cos(ang) * step)
            py = int(cy + math.sin(ang) * step)
            if 0 <= px < PW and 0 <= py < PH:
                draw_rect(img, px, py, 2, 1, C_SHINE if step % 6 == 0 else C_ACCENT)


def draw_sakura(img: np.ndarray, t: float, bass: float) -> None:
    for i in range(24):
        seed = i * 3571
        x = (seed + int(t * (22 + i % 6))) % PW
        y = (seed // 11 + int(t * (35 + bass * 20 + i % 5))) % (PH - 10)
        draw_rect(img, x, y, 2, 2, C_SAKURA)
        draw_rect(img, x + 1, y - 1, 1, 1, C_SHINE)


def draw_anime_character(img: np.ndarray, ox: int, oy: int, t: float, peak: float, bass: float) -> None:
    """Procedural pixel anime protagonist — no photo."""
    bounce = int(math.sin(t * 6) * bass * 3)
    oy += bounce
    if peak > 0.65:
        oy -= int(peak * 4)

    # Spiky hair back
    spikes = [(0, -14, 4, 6), (8, -16, 4, 8), (16, -18, 4, 10), (24, -16, 4, 8), (32, -14, 4, 6)]
    for sx, sy, sw, sh in spikes:
        draw_rect(img, ox + sx, oy + sy, sw, sh, C_HAIR)
    draw_rect(img, ox + 4, oy - 8, 28, 12, C_HAIR)
    draw_rect(img, ox + 8, oy - 4, 20, 4, C_HAIR_HI)

    # Face
    draw_rect(img, ox + 6, oy + 2, 24, 20, C_SKIN)
    draw_rect(img, ox + 6, oy + 2, 24, 2, C_INK)

    # Anime eyes
    eye_y = oy + 10
    for ex in (ox + 10, ox + 22):
        draw_rect(img, ex, eye_y, 8, 8, C_SHINE)
        draw_rect(img, ex + 1, eye_y + 1, 6, 6, C_EYE)
        draw_rect(img, ex + 1, eye_y + 1, 2, 6, C_INK)
        draw_rect(img, ex + 4, eye_y + 2, 2, 2, C_SHINE)
        if peak > 0.55:
            draw_rect(img, ex, eye_y - 1, 8, 2, C_INK)

    # Mouth / expression
    if peak > 0.6:
        draw_rect(img, ox + 14, oy + 18, 8, 2, C_INK)
    else:
        draw_rect(img, ox + 15, oy + 18, 6, 1, C_INK)

    # Coat / body
    draw_rect(img, ox + 2, oy + 22, 32, 28, C_COAT)
    draw_rect(img, ox + 14, oy + 24, 8, 24, C_INK)
    draw_rect(img, ox + 4, oy + 28, 6, 16, C_COAT)
    draw_rect(img, ox + 26, oy + 28, 6, 16, C_COAT)

    # Scarf accent
    draw_rect(img, ox + 10, oy + 22, 16, 4, C_ACCENT)
    if bass > 0.5:
        draw_rect(img, ox + 8, oy + 26, 4, 8, C_SAKURA)


def draw_manga_bars(img: np.ndarray, levels: np.ndarray, bass: float) -> None:
    n = len(levels)
    gap = 2
    bar_w = (PW - gap * (n + 1)) // n
    base = PH - 10
    for i, lv in enumerate(levels):
        h = int(2 + lv * 22 + bass * 5)
        x = gap + i * (bar_w + gap)
        c = C_SKY_MID if i % 3 == 0 else C_EYE if i % 3 == 1 else C_SAKURA
        draw_rect(img, x, base - h, bar_w, h, c)


def draw_title_card(img: np.ndarray, title: str, artist: str, peak: float) -> None:
    draw_rect(img, 72, 2, 176, 22, C_INK)
    draw_rect(img, 74, 4, 172, 18, C_NIGHT)
    draw_text(img, title, 98, 6, C_ACCENT if peak > 0.5 else C_SHINE, 1)
    draw_text(img, artist, 108, 14, C_SAKURA, 1)


def apply_scanlines(img: np.ndarray) -> None:
    img[0::2] = (img[0::2].astype(np.float32) * 0.88).astype(np.uint8)


def apply_vignette(img: np.ndarray) -> None:
    yy, xx = np.mgrid[0:PH, 0:PW]
    dist = np.sqrt(((xx - PW / 2) / (PW * 0.58)) ** 2 + ((yy - PH / 2) / (PH * 0.58)) ** 2)
    img[:] = (img.astype(np.float32) * (1.0 - 0.45 * np.clip(dist, 0, 1))[..., None]).astype(np.uint8)


def render_frame(
    t: float,
    rms: float,
    peak: float,
    bars: np.ndarray,
    title: str,
    artist: str,
) -> np.ndarray:
    img = np.zeros((PH, PW, 3), dtype=np.uint8)
    bass = float(rms)

    draw_anime_sky(img, t, bass)
    draw_city(img, t)
    draw_rooftop(img)
    draw_sakura(img, t, bass)
    draw_speed_lines(img, peak, t)
    draw_anime_character(img, 140, 62, t, peak, bass)
    draw_manga_bars(img, bars, bass)
    draw_title_card(img, title, artist, peak)
    apply_scanlines(img)
    apply_vignette(img)
    return img


def upscale(img: np.ndarray) -> np.ndarray:
    return np.asarray(Image.fromarray(img).resize((OUT_W, OUT_H), Image.Resampling.NEAREST))


def mux_audio(video: Path, audio: Path, output: Path) -> None:
    subprocess.run(
        [ffmpeg_exe(), "-y", "-i", str(video), "-i", str(audio),
         "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", str(output)],
        check=True, capture_output=True,
    )


def generate(
    audio: Path,
    output: Path,
    title: str = "ART OF SIN",
    artist: str = "ANEMENAY",
) -> Path:
    print(f"Loading audio: {audio}")
    samples, duration = load_audio_pcm(audio)
    frames = int(duration * FPS) + 1
    print(f"Duration: {duration:.1f}s | Frames: {frames}")

    rms_env, peak_env = build_envelope(samples, 22050, frames)
    spf = int(22050 / FPS)
    bar_levels = np.zeros((frames, 16), dtype=np.float32)
    for i in range(frames):
        chunk = samples[i * spf : (i + 1) * spf]
        if chunk.size >= 16:
            parts = np.array_split(chunk, 16)
            bar_levels[i] = [np.sqrt(np.mean(p * p)) for p in parts]
            bar_levels[i] /= float(bar_levels[i].max()) or 1.0

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        silent = Path(tmp) / "silent.mp4"
        print("Rendering anime style...")
        writer = imageio.get_writer(
            silent, fps=FPS, codec="libx264", quality=8, ffmpeg_params=["-pix_fmt", "yuv420p"]
        )
        try:
            for i in range(frames):
                frame = render_frame(
                    i / FPS, float(rms_env[i]), float(peak_env[i]),
                    bar_levels[i], title, artist,
                )
                writer.append_data(upscale(frame))
                if i % 180 == 0:
                    print(f"  {100 * i / max(frames - 1, 1):5.1f}%")
        finally:
            writer.close()
        print("Muxing audio...")
        mux_audio(silent, audio, output)

    print(f"Done: {output}")
    return output


def main() -> None:
    p = argparse.ArgumentParser(description="Anime 8-bit music video (no photos)")
    p.add_argument("--audio", type=Path, default=Path("assets/Art_of_sin_M_3.wav"))
    p.add_argument("--output", type=Path, default=Path("output/Art_of_Sin_anime_8bit.mp4"))
    p.add_argument("--title", default="ART OF SIN")
    p.add_argument("--artist", default="ANEMENAY")
    args = p.parse_args()
    if not args.audio.exists():
        sys.exit(f"Audio not found: {args.audio}")
    generate(args.audio, args.output, args.title, args.artist)


if __name__ == "__main__":
    main()
