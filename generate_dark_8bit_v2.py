#!/usr/bin/env python3
"""Dark 8-bit music video — cathedral dungeon style with portrait, halo, and horns."""

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
from PIL import Image, ImageEnhance

PW, PH = 320, 180
OUT_W, OUT_H = 1920, 1080
FPS = 30
PORTRAIT_SIZE = 104

# Heat-of-the-moment dark palette
C_BLACK = np.array([8, 0, 12], dtype=np.uint8)
C_MAROON = np.array([102, 0, 0], dtype=np.uint8)
C_CRIMSON = np.array([153, 0, 51], dtype=np.uint8)
C_WINE = np.array([95, 2, 31], dtype=np.uint8)
C_BLOOD = np.array([140, 0, 26], dtype=np.uint8)
C_AMBER = np.array([255, 144, 0], dtype=np.uint8)
C_GOLD = np.array([255, 196, 64], dtype=np.uint8)
C_ASH = np.array([54, 38, 48], dtype=np.uint8)
C_STONE = np.array([72, 56, 64], dtype=np.uint8)
C_FOG = np.array([36, 20, 32], dtype=np.uint8)

DARK_PALETTE = np.array(
    [C_BLACK, C_MAROON, C_CRIMSON, C_WINE, C_BLOOD, C_AMBER, C_GOLD, C_ASH, C_STONE, C_FOG],
    dtype=np.float32,
)

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


def quantize_dark(arr: np.ndarray) -> np.ndarray:
    flat = arr.reshape(-1, 3).astype(np.float32)
    dist = ((flat[:, None, :] - DARK_PALETTE[None, :, :]) ** 2).sum(axis=2)
    return DARK_PALETTE[dist.argmin(axis=1)].astype(np.uint8).reshape(arr.shape)


def load_portrait(path: Path, size: int = PORTRAIT_SIZE) -> np.ndarray:
    """Fixed-resolution portrait — no gradual pixelation."""
    im = Image.open(path).convert("RGB")
    w, h = im.size
    side = min(w, h)
    left = (w - side) // 2
    top = max(0, (h - side) // 2 - int(side * 0.06))
    im = im.crop((left, top, left + side, top + side))
    im = ImageEnhance.Contrast(im).enhance(1.25)
    im = ImageEnhance.Brightness(im).enhance(0.82)
    im = im.resize((size, size), Image.Resampling.LANCZOS)
    arr = np.array(im, dtype=np.uint8)
    grade = arr.astype(np.float32)
    grade[:, :, 0] = np.clip(grade[:, :, 0] * 1.12 + 6, 0, 255)
    grade[:, :, 1] *= 0.82
    grade[:, :, 2] *= 0.78
    return quantize_dark(grade.astype(np.uint8))


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


def draw_halo_above(img: np.ndarray, head_cx: int, head_top: int, pw: int, t: float) -> None:
    """Golden halo floating above the head."""
    cy = head_top - 18
    rx = int(pw * 0.34)
    ry = max(4, int(pw * 0.09))
    pulse = 0.75 + 0.25 * math.sin(t * 2.8)
    col = (C_GOLD * pulse + C_AMBER * (1 - pulse)).astype(np.uint8)
    for deg in range(0, 360, 10):
        rad = math.radians(deg)
        px = int(head_cx + math.cos(rad) * rx)
        py = int(cy + math.sin(rad) * ry)
        draw_rect(img, px, py, 2, 2, col)
        if deg % 30 == 0:
            draw_rect(img, px - 1, py - 1, 3, 3, col)


def draw_horns_above(img: np.ndarray, head_cx: int, head_top: int, pw: int, peak: float) -> None:
    """Devil horns curving up from above the head, flanking the halo."""
    col = C_AMBER if peak > 0.55 else C_CRIMSON
    spread = int(pw * 0.30)
    base_y = head_top - 6

    # Left horn
    lx = head_cx - spread
    draw_rect(img, lx, base_y, 3, 4, col)
    draw_rect(img, lx - 3, base_y - 6, 3, 6, col)
    draw_rect(img, lx - 5, base_y - 12, 2, 6, col)
    draw_rect(img, lx - 6, base_y - 16, 2, 4, col)

    # Right horn
    rx = head_cx + spread - 2
    draw_rect(img, rx, base_y, 3, 4, col)
    draw_rect(img, rx + 3, base_y - 6, 3, 6, col)
    draw_rect(img, rx + 5, base_y - 12, 2, 6, col)
    draw_rect(img, rx + 6, base_y - 16, 2, 4, col)


def blit_portrait(
    img: np.ndarray,
    portrait: np.ndarray,
    ox: int,
    oy: int,
    peak: float,
    bass: float,
    t: float,
) -> None:
    ph, pw = portrait.shape[:2]
    shake = int(math.sin(t * 20) * peak * 2)
    ox += shake

    # Dark ornate frame
    glow = (C_BLOOD * (0.5 + 0.5 * bass) + C_AMBER * peak * 0.4).astype(np.uint8)
    draw_rect(img, ox - 5, oy - 22, pw + 10, ph + 28, glow)
    draw_rect(img, ox - 3, oy - 20, pw + 6, ph + 24, C_BLACK)
    draw_rect(img, ox - 1, oy - 1, pw + 2, ph + 2, C_WINE)

    for y in range(ph):
        for x in range(pw):
            px, py = ox + x, oy + y
            if 0 <= px < PW and 0 <= py < PH:
                img[py, px] = portrait[y, x]

    head_cx = ox + pw // 2
    head_top = oy + int(ph * 0.10)
    draw_halo_above(img, head_cx, head_top, pw, t)
    draw_horns_above(img, head_cx, head_top, pw, peak)


def draw_cathedral(img: np.ndarray, t: float, bass: float) -> None:
    img[:] = C_BLACK
    for y in range(PH):
        fog = C_FOG if y < PH // 2 else C_BLACK
        img[y, :] = (fog * (1 - bass * 0.2) + C_MAROON * bass * 0.2).astype(np.uint8)

    # Stone arches
    for i, x in enumerate(range(0, PW, 40)):
        h = 50 + (i * 11) % 30
        draw_rect(img, x + 4, 130 - h, 12, h, C_STONE)
        draw_rect(img, x, 130 - h - 8, 20, 8, C_ASH)

    # Stained-glass window pulse
    pulse = 0.4 + 0.6 * abs(math.sin(t * 1.8))
    glass = (C_CRIMSON * pulse + C_AMBER * (1 - pulse) * 0.4).astype(np.uint8)
    draw_rect(img, 136, 36, 48, 40, glass)
    draw_rect(img, 146, 46, 28, 24, C_BLACK)

    # Torches
    for i, x in enumerate([28, 280]):
        flicker = 0.5 + 0.5 * abs(math.sin(t * 8 + i * 2.1))
        flame = (C_AMBER * flicker + C_GOLD * (1 - flicker)).astype(np.uint8)
        draw_rect(img, x, 100, 3, 14, C_STONE)
        draw_rect(img, x - 2, 94, 7, 6, flame)


def draw_pixel_rain(img: np.ndarray, t: float) -> None:
    for i in range(40):
        seed = i * 4447
        x = (seed + int(t * (30 + i % 8))) % PW
        y = (seed // 13 + int(t * (60 + i % 12))) % (PH - 20)
        draw_rect(img, x, y, 1, 3, C_WINE if i % 2 else C_CRIMSON)


def draw_crosses(img: np.ndarray, t: float) -> None:
    for i, x in enumerate([60, 180, 240]):
        bob = int(math.sin(t * 1.5 + i) * 2)
        cy = 52 + bob
        draw_rect(img, x, cy, 2, 10, C_ASH)
        draw_rect(img, x - 3, cy + 2, 8, 2, C_ASH)


def draw_flame_bars(img: np.ndarray, levels: np.ndarray, bass: float) -> None:
    n = len(levels)
    gap = 2
    bar_w = (PW - gap * (n + 1)) // n
    base = PH - 12
    for i, lv in enumerate(levels):
        h = int(2 + lv * 24 + bass * 6)
        x = gap + i * (bar_w + gap)
        c = C_AMBER if lv > 0.6 else C_BLOOD if lv > 0.3 else C_WINE
        draw_rect(img, x, base - h, bar_w, h, c)
        if lv > 0.75:
            draw_rect(img, x, base - h - 2, bar_w, 2, C_GOLD)


def apply_scanlines(img: np.ndarray) -> None:
    img[0::2] = (img[0::2].astype(np.float32) * 0.72).astype(np.uint8)


def apply_vignette(img: np.ndarray) -> None:
    yy, xx = np.mgrid[0:PH, 0:PW]
    dist = np.sqrt(((xx - PW / 2) / (PW * 0.5)) ** 2 + ((yy - PH / 2) / (PH * 0.52)) ** 2)
    img[:] = (img.astype(np.float32) * (1.0 - 0.65 * np.clip(dist, 0, 1))[..., None]).astype(np.uint8)


def apply_glitch(img: np.ndarray, intensity: float) -> None:
    if intensity < 0.6:
        return
    for _ in range(int(1 + intensity * 4)):
        y = np.random.randint(0, PH - 2)
        img[y : y + 2] = np.roll(img[y : y + 2], np.random.randint(-8, 9), axis=0)


def render_frame(
    portrait: np.ndarray,
    t: float,
    rms: float,
    peak: float,
    bars: np.ndarray,
    title: str,
    artist: str,
) -> np.ndarray:
    img = np.zeros((PH, PW, 3), dtype=np.uint8)
    bass = float(rms)

    draw_cathedral(img, t, bass)
    draw_pixel_rain(img, t)
    draw_crosses(img, t)

    ph, pw = portrait.shape[:2]
    # Extra top margin so halo/horns sit clearly above portrait
    blit_portrait(img, portrait, (PW - pw) // 2, 42, peak, bass, t)

    draw_flame_bars(img, bars, bass)
    draw_text(img, title, 98, 6, C_AMBER if peak > 0.5 else C_GOLD, 1)
    draw_text(img, artist, 108, 16, C_CRIMSON, 1)

    apply_glitch(img, peak)
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
    portrait_path: Path,
    output: Path,
    title: str = "ART OF SIN",
    artist: str = "ANEMENAY",
) -> Path:
    print(f"Loading portrait: {portrait_path}")
    portrait = load_portrait(portrait_path)
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
        print("Rendering dark 8-bit style...")
        writer = imageio.get_writer(
            silent, fps=FPS, codec="libx264", quality=8, ffmpeg_params=["-pix_fmt", "yuv420p"]
        )
        try:
            for i in range(frames):
                frame = render_frame(
                    portrait, i / FPS, float(rms_env[i]), float(peak_env[i]),
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
    p = argparse.ArgumentParser(description="Dark cathedral 8-bit music video")
    p.add_argument("--audio", type=Path, default=Path("assets/Art_of_sin_M_3.wav"))
    p.add_argument("--portrait", type=Path, default=Path("assets/portrait_dark.jpg"))
    p.add_argument("--output", type=Path, default=Path("output/Art_of_Sin_dark_8bit.mp4"))
    p.add_argument("--title", default="ART OF SIN")
    p.add_argument("--artist", default="ANEMENAY")
    args = p.parse_args()
    if not args.audio.exists():
        sys.exit(f"Audio not found: {args.audio}")
    if not args.portrait.exists():
        sys.exit(f"Portrait not found: {args.portrait}")
    generate(args.audio, args.portrait, args.output, args.title, args.artist)


if __name__ == "__main__":
    main()
