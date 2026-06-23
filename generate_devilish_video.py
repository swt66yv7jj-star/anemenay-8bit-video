#!/usr/bin/env python3
"""Generate devilish 8-bit music video with artist portrait incorporated."""

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

# Devilish palette (Art of Sin tones + hell accents)
C_VOID = np.array([4, 0, 6], dtype=np.uint8)
C_DEEP = np.array([66, 40, 65], dtype=np.uint8)
C_WINE = np.array([92, 44, 60], dtype=np.uint8)
C_BLOOD = np.array([168, 12, 32], dtype=np.uint8)
C_BLOOD_HOT = np.array([240, 32, 48], dtype=np.uint8)
C_HELL = np.array([255, 68, 0], dtype=np.uint8)
C_PURPLE = np.array([108, 24, 120], dtype=np.uint8)
C_GLOW = np.array([180, 60, 200], dtype=np.uint8)
C_BONE = np.array([210, 196, 170], dtype=np.uint8)
C_ASH = np.array([48, 36, 44], dtype=np.uint8)
C_STAR = np.array([240, 240, 255], dtype=np.uint8)
C_GOLD = np.array([255, 210, 64], dtype=np.uint8)
C_UFO = np.array([160, 255, 200], dtype=np.uint8)
C_ALIEN = np.array([120, 255, 140], dtype=np.uint8)
C_SPACE = np.array([8, 4, 24], dtype=np.uint8)
C_NEBULA = np.array([40, 16, 72], dtype=np.uint8)

FONT: dict[str, list[int]] = {
    "A": [0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11],
    "B": [0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E],
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


def load_portrait(path: Path, size: int = 104) -> np.ndarray:
    """Fixed-resolution portrait — no gradual pixelation."""
    im = Image.open(path).convert("RGB")
    w, h = im.size
    side = min(w, h)
    left = (w - side) // 2
    top = max(0, (h - side) // 2 - int(side * 0.08))
    im = im.crop((left, top, left + side, top + side))
    im = ImageEnhance.Contrast(im).enhance(1.2)
    im = ImageEnhance.Color(im).enhance(0.92)
    im = im.resize((size, size), Image.Resampling.LANCZOS)
    arr = np.array(im, dtype=np.uint8)
    tint = arr.astype(np.float32)
    tint[:, :, 0] = np.clip(tint[:, :, 0] * 1.08, 0, 255)
    tint[:, :, 2] = np.clip(tint[:, :, 2] * 0.88, 0, 255)
    return tint.astype(np.uint8)


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


def draw_pentagram(img: np.ndarray, cx: float, cy: float, radius: float, rotation: float, color: np.ndarray, t: int = 2) -> None:
    pts = []
    for i in range(5):
        a = rotation + i * (2 * math.pi / 5) - math.pi / 2
        pts.append((cx + math.cos(a) * radius, cy + math.sin(a) * radius))
    order = [0, 2, 4, 1, 3, 0]
    for a, b in zip(order[:-1], order[1:]):
        x0, y0 = pts[a]
        x1, y1 = pts[b]
        steps = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
        for s in range(steps + 1):
            f = s / max(steps, 1)
            px, py = int(x0 + (x1 - x0) * f), int(y0 + (y1 - y0) * f)
            draw_rect(img, px - t // 2, py - t // 2, t, t, color)


def draw_halo_above(img: np.ndarray, head_cx: int, head_top: int, pw: int, t: float) -> None:
    cy = head_top - 18
    rx = int(pw * 0.34)
    ry = max(4, int(pw * 0.09))
    pulse = 0.7 + 0.3 * math.sin(t * 3.0)
    col = (C_GOLD * pulse + C_BONE * (1 - pulse)).astype(np.uint8)
    for deg in range(0, 360, 10):
        rad = math.radians(deg)
        px = int(head_cx + math.cos(rad) * rx)
        py = int(cy + math.sin(rad) * ry)
        draw_rect(img, px, py, 2, 2, col)


def draw_horns_above(img: np.ndarray, head_cx: int, head_top: int, pw: int, peak: float) -> None:
    col = C_BLOOD_HOT if peak > 0.5 else C_BLOOD
    spread = int(pw * 0.30)
    base_y = head_top - 6
    lx = head_cx - spread
    rx = head_cx + spread - 2
    draw_rect(img, lx, base_y, 3, 4, col)
    draw_rect(img, lx - 3, base_y - 6, 3, 6, col)
    draw_rect(img, lx - 5, base_y - 12, 2, 6, col)
    draw_rect(img, rx, base_y, 3, 4, col)
    draw_rect(img, rx + 3, base_y - 6, 3, 6, col)
    draw_rect(img, rx + 5, base_y - 12, 2, 6, col)


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
    shake_x = int(math.sin(t * 24) * peak * 3)
    shake_y = int(math.cos(t * 19) * peak * 2)
    ox += shake_x
    oy += shake_y

    # Glowing frame
    frame_col = (C_BLOOD_HOT * (0.4 + 0.6 * peak) + C_GLOW * (0.3 + 0.4 * bass)).astype(np.uint8)
    draw_rect(img, ox - 4, oy - 4, pw + 8, ph + 8, frame_col)
    draw_rect(img, ox - 2, oy - 2, pw + 4, ph + 4, C_VOID)

    for y in range(ph):
        for x in range(pw):
            px, py = ox + x, oy + y
            if 0 <= px < PW and 0 <= py < PH:
                img[py, px] = portrait[y, x]

    head_cx = ox + pw // 2
    head_top = oy + int(ph * 0.10)
    draw_halo_above(img, head_cx, head_top, pw, t)
    draw_horns_above(img, head_cx, head_top, pw, peak)


def draw_stars(img: np.ndarray, t: float) -> None:
    img[:] = C_SPACE
    for y in range(PH // 2 + 20):
        blend = y / (PH // 2 + 20)
        img[y, :] = (C_SPACE * (1 - blend) + C_NEBULA * blend).astype(np.uint8)

    for i in range(90):
        seed = i * 6271
        x = seed % PW
        y = (seed // 17) % (PH - 40)
        twinkle = 0.35 + 0.65 * abs(math.sin(t * 2.5 + i * 0.7))
        if twinkle > 0.75:
            size = 2 if i % 11 == 0 else 1
            col = C_GOLD if i % 13 == 0 else C_STAR
            draw_rect(img, x, y, size, size, (col * twinkle).astype(np.uint8))


def draw_ufo(img: np.ndarray, t: float, bass: float) -> None:
    x = int((t * 28) % (PW + 40)) - 20
    y = 18 + int(math.sin(t * 1.4) * 6)
    body = C_UFO
    draw_rect(img, x + 4, y + 4, 22, 4, body)
    draw_rect(img, x + 8, y + 2, 14, 3, (body * 0.8 + C_GLOW * 0.2).astype(np.uint8))
    draw_rect(img, x + 12, y, 6, 2, C_STAR)
    if bass > 0.45:
        beam = (C_GLOW * 0.5 + C_UFO * 0.5).astype(np.uint8)
        draw_rect(img, x + 14, y + 8, 2, 10 + int(bass * 8), beam)


def draw_alien(img: np.ndarray, ox: int, oy: int, scale: int = 2) -> None:
    col = C_ALIEN
    draw_rect(img, ox + 2 * scale, oy, 3 * scale, 4 * scale, col)
    draw_rect(img, ox + 1 * scale, oy + 1 * scale, scale, scale, C_VOID)
    draw_rect(img, ox + 4 * scale, oy + 1 * scale, scale, scale, C_VOID)
    draw_rect(img, ox + 2 * scale, oy + 4 * scale, 3 * scale, scale, col)


def draw_god_figure(img: np.ndarray, ox: int, oy: int, scale: int = 2) -> None:
    col = C_GOLD
    draw_rect(img, ox + 2 * scale, oy, 3 * scale, scale, col)
    draw_rect(img, ox + 1 * scale, oy + scale, 5 * scale, 4 * scale, col)
    draw_rect(img, ox, oy + 2 * scale, scale, 3 * scale, col)
    draw_rect(img, ox + 6 * scale, oy + 2 * scale, scale, 3 * scale, col)


def draw_cosmic_background(img: np.ndarray, t: float, bass: float) -> None:
    draw_stars(img, t)
    draw_ufo(img, t, bass)
    draw_ufo(img, t + 4.7, bass * 0.8)  # second UFO offset

    draw_god_figure(img, 24, 58, 2)
    draw_god_figure(img, 268, 64, 2)
    draw_alien(img, 52, 72)
    draw_alien(img, 238, 78)
    draw_alien(img, 148, 28)

    draw_text(img, "GODS AND ALIENS", 78, 164, C_GLOW, 1)


def draw_hell_floor(img: np.ndarray, t: float) -> None:
    draw_rect(img, 0, 128, PW, PH - 128, C_DEEP)
    for x in range(0, PW, 12):
        off = int(math.sin(t * 2 + x * 0.1) * 2)
        draw_rect(img, x, 128 + off, 8, PH - 128, C_ASH)
    # Lava cracks
    for i in range(8):
        lx = 20 + i * 36 + int(math.sin(t * 3 + i) * 4)
        glow = (C_HELL * 0.6 + C_BLOOD * 0.4).astype(np.uint8)
        draw_rect(img, lx, 140, 14, 2, glow)


def draw_bars(img: np.ndarray, levels: np.ndarray, bass: float) -> None:
    n = len(levels)
    gap = 2
    bar_w = (PW - gap * (n + 1)) // n
    base = PH - 14
    for i, lv in enumerate(levels):
        h = int(3 + lv * 28 + bass * 8)
        x = gap + i * (bar_w + gap)
        c = C_HELL if i % 4 == 0 else C_BLOOD if i % 4 == 1 else C_GLOW if i % 4 == 2 else C_WINE
        draw_rect(img, x, base - h, bar_w, h, c)


def apply_scanlines(img: np.ndarray) -> None:
    img[0::2] = (img[0::2].astype(np.float32) * 0.78).astype(np.uint8)


def apply_vignette(img: np.ndarray) -> None:
    yy, xx = np.mgrid[0:PH, 0:PW]
    dist = np.sqrt(((xx - PW / 2) / (PW * 0.55)) ** 2 + ((yy - PH / 2) / (PH * 0.55)) ** 2)
    img[:] = (img.astype(np.float32) * (1.0 - 0.6 * np.clip(dist, 0, 1))[..., None]).astype(np.uint8)


def apply_glitch(img: np.ndarray, intensity: float) -> None:
    if intensity < 0.5:
        return
    for _ in range(int(2 + intensity * 6)):
        y = np.random.randint(0, PH - 3)
        h = np.random.randint(1, 3)
        img[y : y + h] = np.roll(img[y : y + h], np.random.randint(-12, 13), axis=0)


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

    draw_cosmic_background(img, t, bass)
    draw_hell_floor(img, t)

    rot = t * 0.5
    draw_pentagram(img, 48, 78, 18, rot, C_PURPLE, 1)
    draw_pentagram(img, 272, 82, 16, -rot, C_BLOOD, 1)

    ph, pw = portrait.shape[:2]
    blit_portrait(img, portrait, (PW - pw) // 2, 42, peak, bass, t)

    draw_bars(img, bars, bass)
    draw_text(img, title, 98, 4, C_BLOOD_HOT if peak > 0.55 else C_BONE, 1)
    draw_text(img, artist, 108, 14, C_GLOW, 1)

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
        print("Rendering...")
        writer = imageio.get_writer(silent, fps=FPS, codec="libx264", quality=8, ffmpeg_params=["-pix_fmt", "yuv420p"])
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
    p = argparse.ArgumentParser(description="Devilish 8-bit video with portrait")
    p.add_argument("--audio", type=Path, default=Path("assets/Art_of_sin_M_3.wav"))
    p.add_argument("--portrait", type=Path, default=Path("assets/portrait.jpg"))
    p.add_argument("--output", type=Path, default=Path("output/Art_of_Sin_devilish_8bit.mp4"))
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
