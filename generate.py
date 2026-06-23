#!/usr/bin/env python3
"""
Modular music video generator.

Usage:
  python generate.py --list-styles
  python generate.py --style synthwave --audio assets/Art_of_sin_M_3.wav
  python generate.py --style anime --title "ART OF SIN" --artist ANEMENAY
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from generator.config import VideoConfig
from generator.core import render_video
from generator.styles import list_styles


def main() -> None:
    p = argparse.ArgumentParser(description="Modular 8-bit music video generator")
    p.add_argument("--list-styles", action="store_true", help="Show available visual styles")
    p.add_argument("--style", default="synthwave", help="Visual style name (default: synthwave)")
    p.add_argument("--audio", type=Path, default=Path("assets/Art_of_sin_M_3.wav"))
    p.add_argument("--output", type=Path, default=None, help="Output MP4 (auto-named if omitted)")
    p.add_argument("--title", default="ART OF SIN")
    p.add_argument("--artist", default="ANEMENAY")
    args = p.parse_args()

    if args.list_styles:
        print("Available styles:\n")
        for name, desc in list_styles():
            print(f"  {name:12} {desc}")
        return

    if not args.audio.exists():
        sys.exit(f"Audio not found: {args.audio}")

    output = args.output or Path(f"output/{args.title.replace(' ', '_')}_{args.style}.mp4")

    config = VideoConfig(
        audio=args.audio,
        output=output,
        style=args.style,
        title=args.title,
        artist=args.artist,
    )
    render_video(config)


if __name__ == "__main__":
    main()
