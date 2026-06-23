from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class VideoConfig:
    audio: Path
    output: Path
    style: str = "synthwave"
    title: str = "ART OF SIN"
    artist: str = "ANEMENAY"
    fps: int = 30
    pixel_w: int = 320
    pixel_h: int = 180
    out_w: int = 1920
    out_h: int = 1080
    sample_rate: int = 22050
    bar_count: int = 16
