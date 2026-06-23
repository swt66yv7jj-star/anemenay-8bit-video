from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import imageio
import imageio_ffmpeg
import numpy as np
from PIL import Image

from generator.config import VideoConfig
from generator.styles import get_style


def ffmpeg_exe() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()


def load_audio_pcm(path: Path, sample_rate: int) -> tuple[np.ndarray, float]:
    cmd = [
        ffmpeg_exe(), "-i", str(path), "-f", "s16le", "-acodec", "pcm_s16le",
        "-ac", "1", "-ar", str(sample_rate), "-v", "error", "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=True)
    samples = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    return samples, len(samples) / sample_rate


def build_envelope(samples: np.ndarray, sample_rate: int, fps: int, frame_count: int) -> tuple[np.ndarray, np.ndarray]:
    spf = sample_rate / fps
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


def build_bar_levels(samples: np.ndarray, sample_rate: int, fps: int, frames: int, bar_count: int) -> np.ndarray:
    spf = int(sample_rate / fps)
    levels = np.zeros((frames, bar_count), dtype=np.float32)
    for i in range(frames):
        chunk = samples[i * spf : (i + 1) * spf]
        if chunk.size >= bar_count:
            parts = np.array_split(chunk, bar_count)
            levels[i] = [np.sqrt(np.mean(p * p)) for p in parts]
            levels[i] /= float(levels[i].max()) or 1.0
    return levels


def upscale(img: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    return np.asarray(Image.fromarray(img).resize((out_w, out_h), Image.Resampling.NEAREST))


def mux_audio(video: Path, audio: Path, output: Path) -> None:
    subprocess.run(
        [ffmpeg_exe(), "-y", "-i", str(video), "-i", str(audio),
         "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", str(output)],
        check=True, capture_output=True,
    )


def render_video(config: VideoConfig) -> Path:
    style = get_style(config.style, config.pixel_w, config.pixel_h)
    print(f"Style: {style.name} — {style.description}")
    print(f"Loading audio: {config.audio}")

    samples, duration = load_audio_pcm(config.audio, config.sample_rate)
    frames = int(duration * config.fps) + 1
    print(f"Duration: {duration:.1f}s | Frames: {frames}")

    rms_env, peak_env = build_envelope(samples, config.sample_rate, config.fps, frames)
    bar_levels = build_bar_levels(samples, config.sample_rate, config.fps, frames, config.bar_count)

    config.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        silent = Path(tmp) / "silent.mp4"
        print("Rendering...")
        writer = imageio.get_writer(
            silent, fps=config.fps, codec="libx264", quality=8, ffmpeg_params=["-pix_fmt", "yuv420p"]
        )
        try:
            for i in range(frames):
                frame = style.render_frame(
                    t=i / config.fps,
                    rms=float(rms_env[i]),
                    peak=float(peak_env[i]),
                    bars=bar_levels[i],
                    title=config.title,
                    artist=config.artist,
                )
                writer.append_data(upscale(frame, config.out_w, config.out_h))
                if i % 180 == 0:
                    print(f"  {100 * i / max(frames - 1, 1):5.1f}%")
        finally:
            writer.close()
        print("Muxing audio...")
        mux_audio(silent, config.audio, config.output)

    print(f"Done: {config.output}")
    return config.output
