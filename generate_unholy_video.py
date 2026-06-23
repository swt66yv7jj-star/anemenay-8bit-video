#!/usr/bin/env python3
"""Generate a 16:9 8-bit 'unholy' music video synced to an audio track."""

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

# Internal pixel grid (scaled up with nearest-neighbor for crisp 8-bit look)
PW, PH = 320, 180
OUT_W, OUT_H = 1920, 1080
FPS = 30

# Unholy palette
C_VOID = np.array([6, 2, 10], dtype=np.uint8)
C_DEEP = np.array([22, 6, 28], dtype=np.uint8)
C_BLOOD = np.array([148, 10, 28], dtype=np.uint8)
C_BLOOD_HOT = np.array([220, 36, 48], dtype=np.uint8)
C_PURPLE = np.array([88, 20, 112], dtype=np.uint8)
C_PURPLE_GLOW = np.array([148, 52, 188], dtype=np.uint8)
C_BONE = np.array([214, 202, 176], dtype=np.uint8)
C_SICK = np.array([52, 196, 72], dtype=np.uint8)
C_ASH = np.array([58, 48, 62], dtype=np.uint8)
C_FLAME = np.array([236, 118, 24], dtype=np.uint8)

# 12x14 pixel skull (1 = bone, 2 = eye, 3 = blood)
SKULL = np.array(
    [
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 1],
        [0, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1, 1],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0],
        [0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0],
        [0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ],
    dtype=np.uint8,
)

# 5x7 pixel font (rows as bits)
FONT: dict[str, list[int]] = {
    "A": [0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11],
    "D": [0x1E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x1E],
    "E": [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F],
    "H": [0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11],
    "M": [0x11, 0x1B, 0x15, 0x11, 0x11, 0x11, 0x11],
    "N": [0x11, 0x19, 0x15, 0x13, 0x11, 0x11, 0x11],
    "O": [0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E],
    "Y": [0x11, 0x11, 0x0A, 0x04, 0x04, 0x04, 0x04],
    " ": [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
}


def ffmpeg_exe() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()


def load_audio_pcm(path: Path, sample_rate: int = 22050) -> tuple[np.ndarray, float]:
    cmd = [
        ffmpeg_exe(),
        "-i",
        str(path),
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-v",
        "error",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=True)
    samples = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    duration = len(samples) / sample_rate
    return samples, duration


def build_envelope(samples: np.ndarray, sample_rate: int, frame_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame RMS (smooth) and transient peaks for beat hits."""
    spf = sample_rate / FPS
    rms = np.zeros(frame_count, dtype=np.float32)
    peak = np.zeros(frame_count, dtype=np.float32)

    for i in range(frame_count):
        start = int(i * spf)
        end = int((i + 1) * spf)
        chunk = samples[start:end]
        if chunk.size == 0:
            continue
        rms[i] = float(np.sqrt(np.mean(chunk * chunk)))
        peak[i] = float(np.max(np.abs(chunk)))

    # Smooth RMS for visuals
    kernel = np.ones(5, dtype=np.float32) / 5.0
    smooth = np.convolve(rms, kernel, mode="same")
    mx = float(smooth.max()) or 1.0
    smooth /= mx
    peak /= float(peak.max()) or 1.0
    return smooth, peak


def draw_rect(img: np.ndarray, x: int, y: int, w: int, h: int, color: np.ndarray) -> None:
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(PW, x + w), min(PH, y + h)
    if x0 >= x1 or y0 >= y1:
        return
    img[y0:y1, x0:x1] = color


def draw_text(img: np.ndarray, text: str, x: int, y: int, color: np.ndarray, scale: int = 2) -> None:
    cx = x
    for ch in text.upper():
        glyph = FONT.get(ch, FONT[" "])
        for row, bits in enumerate(glyph):
            for col in range(5):
                if bits & (1 << (4 - col)):
                    draw_rect(img, cx + col * scale, y + row * scale, scale, scale, color)
        cx += 6 * scale


def draw_pentagram(
    img: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
    rotation: float,
    color: np.ndarray,
    thickness: int = 2,
) -> None:
    points = []
    for i in range(5):
        angle = rotation + i * (2 * math.pi / 5) - math.pi / 2
        points.append((cx + math.cos(angle) * radius, cy + math.sin(angle) * radius))

    order = [0, 2, 4, 1, 3, 0]
    for a, b in zip(order[:-1], order[1:]):
        x0, y0 = points[a]
        x1, y1 = points[b]
        steps = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
        for s in range(steps + 1):
            t = s / max(steps, 1)
            px = int(x0 + (x1 - x0) * t)
            py = int(y0 + (y1 - y0) * t)
            draw_rect(img, px - thickness // 2, py - thickness // 2, thickness, thickness, color)


def draw_skull(img: np.ndarray, ox: int, oy: int, scale: int, flash: float) -> None:
    bone = C_BONE if flash < 0.55 else C_BLOOD_HOT
    eye = C_VOID
    mouth = C_BLOOD if flash < 0.7 else C_FLAME
    for y, row in enumerate(SKULL):
        for x, val in enumerate(row):
            if val == 0:
                continue
            color = bone if val == 1 else eye if val == 2 else mouth
            draw_rect(img, ox + x * scale, oy + y * scale, scale, scale, color)


def draw_cathedral(img: np.ndarray, t: float) -> None:
    """Hellish cathedral silhouette with flickering candles."""
    draw_rect(img, 0, 118, PW, PH - 118, C_DEEP)
    for i, x in enumerate(range(8, PW, 26)):
        h = 44 + (i * 7) % 28
        draw_rect(img, x, 118 - h, 10, h, C_ASH)
        draw_rect(img, x - 2, 118 - h - 6, 14, 6, C_PURPLE)

    # Arch window glow pulse
    pulse = 0.5 + 0.5 * math.sin(t * 2.4)
    glow = (
        C_PURPLE * (0.35 + 0.25 * pulse)
        + C_BLOOD * (0.15 + 0.2 * (1 - pulse))
    ).astype(np.uint8)
    draw_rect(img, 138, 52, 44, 52, glow)
    draw_rect(img, 148, 62, 24, 32, C_VOID)

    # Candles
    for i, x in enumerate([36, 92, 228, 284]):
        flicker = 0.4 + 0.6 * abs(math.sin(t * 9.0 + i * 1.7))
        flame = (C_FLAME * flicker + C_BLOOD * (1 - flicker)).astype(np.uint8)
        draw_rect(img, x, 108, 2, 10, C_BONE)
        draw_rect(img, x - 1, 104, 4, 4, flame)


def draw_bars(img: np.ndarray, levels: np.ndarray, bass: float) -> None:
    n = len(levels)
    gap = 2
    bar_w = (PW - gap * (n + 1)) // n
    base_y = PH - 18
    for i, level in enumerate(levels):
        h = int(4 + level * 34 + bass * 10)
        x = gap + i * (bar_w + gap)
        color = C_SICK if i % 3 == 0 else C_BLOOD if i % 3 == 1 else C_PURPLE_GLOW
        draw_rect(img, x, base_y - h, bar_w, h, color)


def draw_hell_sky(img: np.ndarray, t: float, bass: float) -> None:
    img[:] = C_VOID
    # Distant embers
    for i in range(48):
        seed = i * 9973
        x = (seed + int(t * (12 + (i % 5)))) % PW
        y = 8 + (seed // 17) % 72
        flicker = 0.3 + 0.7 * abs(math.sin(t * 3.1 + i))
        if flicker > 0.82:
            color = C_FLAME if i % 2 else C_BLOOD_HOT
            draw_rect(img, x, y, 2, 2, color)

    # Unholy moon / eye
    mx, my = 252, 28
    pulse = 0.55 + 0.45 * math.sin(t * 1.6)
    moon = (C_BLOOD * (0.4 + 0.3 * bass) + C_PURPLE_GLOW * pulse).astype(np.uint8)
    draw_rect(img, mx, my, 22, 22, moon)
    draw_rect(img, mx + 8, my + 8, 6, 6, C_VOID)


def apply_scanlines(img: np.ndarray, strength: float = 0.22) -> None:
    for y in range(0, PH, 2):
        img[y] = (img[y].astype(np.float32) * (1.0 - strength)).astype(np.uint8)


def apply_vignette(img: np.ndarray) -> None:
    yy, xx = np.mgrid[0:PH, 0:PW]
    cx, cy = PW / 2, PH / 2
    dist = np.sqrt(((xx - cx) / (PW * 0.65)) ** 2 + ((yy - cy) / (PH * 0.65)) ** 2)
    mask = np.clip(dist, 0, 1)
    factor = (1.0 - 0.55 * mask)[..., None]
    img[:] = (img.astype(np.float32) * factor).astype(np.uint8)


def apply_glitch(img: np.ndarray, intensity: float) -> None:
    if intensity < 0.55:
        return
    bands = int(2 + intensity * 5)
    for _ in range(bands):
        y = np.random.randint(0, PH - 4)
        h = np.random.randint(1, 4)
        shift = np.random.randint(-10, 11)
        img[y : y + h] = np.roll(img[y : y + h], shift, axis=0)
        if intensity > 0.75:
            img[y : y + h] = (img[y : y + h].astype(np.float32) * np.array([1.2, 0.7, 1.3])).clip(0, 255).astype(
                np.uint8
            )


def render_frame(
    frame_idx: int,
    t: float,
    rms: float,
    peak: float,
    bar_levels: np.ndarray,
    title: str,
    artist: str,
) -> np.ndarray:
    img = np.zeros((PH, PW, 3), dtype=np.uint8)
    bass = float(rms)

    draw_hell_sky(img, t, bass)
    draw_cathedral(img, t)

    rot = t * 0.55
    pent_color = (
        C_BLOOD_HOT * (0.35 + 0.65 * peak) + C_PURPLE_GLOW * (0.4 + 0.3 * bass)
    ).astype(np.uint8)
    draw_pentagram(img, 160, 96, 34 + bass * 8, rot, pent_color, thickness=2)
    draw_pentagram(img, 160, 96, 20 + bass * 4, -rot * 1.4, C_BLOOD, thickness=1)

    skull_scale = 3 if peak < 0.65 else 4
    draw_skull(img, 146, 72, skull_scale, peak)

    # Floating unholy runes
    for i in range(6):
        rx = int(24 + i * 48 + math.sin(t * 1.8 + i) * 8)
        ry = int(34 + math.cos(t * 2.2 + i * 0.8) * 6)
        col = C_SICK if i % 2 else C_FLAME
        draw_rect(img, rx, ry, 3, 3, col)
        draw_rect(img, rx + 1, ry - 2, 1, 2, col)

    draw_bars(img, bar_levels, bass)

    title_color = C_BLOOD_HOT if peak > 0.6 else C_BONE
    draw_text(img, title, 96, 8, title_color, scale=2)
    draw_text(img, artist, 88, 26, C_PURPLE_GLOW, scale=1)

    if peak > 0.72:
        draw_text(img, "UNHOLY", 118, 150, C_SICK, scale=1)

    apply_glitch(img, peak)
    apply_scanlines(img)
    apply_vignette(img)
    return img


def upscale(img: np.ndarray) -> np.ndarray:
    pil = Image.fromarray(img, mode="RGB")
    pil = pil.resize((OUT_W, OUT_H), Image.Resampling.NEAREST)
    return np.asarray(pil)


def mux_audio(video_path: Path, audio_path: Path, output_path: Path) -> None:
    cmd = [
        ffmpeg_exe(),
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def generate(
    audio_path: Path,
    output_path: Path,
    title: str = "DOH",
    artist: str = "ANEMENAY",
) -> Path:
    print(f"Loading audio: {audio_path}")
    samples, duration = load_audio_pcm(audio_path)
    frame_count = int(duration * FPS) + 1
    print(f"Duration: {duration:.2f}s | Frames: {frame_count} @ {FPS}fps")

    rms_env, peak_env = build_envelope(samples, 22050, frame_count)

    # 16 frequency-ish bands from chunked FFT proxy (simple split of waveform)
    bar_levels = np.zeros((frame_count, 16), dtype=np.float32)
    spf = int(22050 / FPS)
    for i in range(frame_count):
        chunk = samples[i * spf : (i + 1) * spf]
        if chunk.size < 16:
            continue
        parts = np.array_split(chunk, 16)
        bar_levels[i] = np.array([np.sqrt(np.mean(p * p)) for p in parts], dtype=np.float32)
        mx = float(bar_levels[i].max()) or 1.0
        bar_levels[i] /= mx

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        silent_video = Path(tmp) / "silent.mp4"
        print("Rendering frames...")
        writer = imageio.get_writer(
            silent_video,
            fps=FPS,
            codec="libx264",
            quality=8,
            ffmpeg_params=["-pix_fmt", "yuv420p"],
        )
        try:
            for i in range(frame_count):
                t = i / FPS
                frame = render_frame(
                    i,
                    t,
                    float(rms_env[i]),
                    float(peak_env[i]),
                    bar_levels[i],
                    title,
                    artist,
                )
                writer.append_data(upscale(frame))
                if i % 120 == 0:
                    pct = 100.0 * i / max(frame_count - 1, 1)
                    print(f"  {pct:5.1f}% ({i}/{frame_count})")
        finally:
            writer.close()

        print("Muxing audio...")
        mux_audio(silent_video, audio_path, output_path)

    print(f"Done: {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate unholy 8-bit music video")
    parser.add_argument(
        "--audio",
        type=Path,
        default=Path("/Users/anemenay/Desktop/DOH/DOH/DOH.mp3"),
        help="Path to input audio file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/DOH_unholy_8bit.mp4"),
        help="Output MP4 path",
    )
    parser.add_argument("--title", default="DOH", help="Song title on screen")
    parser.add_argument("--artist", default="ANEMENAY", help="Artist name on screen")
    args = parser.parse_args()

    if not args.audio.exists():
        print(f"Audio not found: {args.audio}", file=sys.stderr)
        sys.exit(1)

    generate(args.audio, args.output, args.title, args.artist)


if __name__ == "__main__":
    main()
