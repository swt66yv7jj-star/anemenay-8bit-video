from __future__ import annotations

from generator.styles.anime import AnimeStyle
from generator.styles.base import VisualStyle
from generator.styles.blank import BlankStyle
from generator.styles.synthwave import SynthwaveStyle

STYLES: dict[str, type[VisualStyle]] = {
    SynthwaveStyle.name: SynthwaveStyle,
    AnimeStyle.name: AnimeStyle,
    BlankStyle.name: BlankStyle,
}


def list_styles() -> list[tuple[str, str]]:
    return [(name, cls.description) for name, cls in STYLES.items()]


def get_style(name: str, pw: int, ph: int) -> VisualStyle:
    key = name.lower()
    if key not in STYLES:
        available = ", ".join(STYLES)
        raise ValueError(f"Unknown style '{name}'. Available: {available}")
    return STYLES[key](pw, ph)
