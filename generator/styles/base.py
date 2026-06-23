from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from generator.canvas import Canvas


class VisualStyle(ABC):
    name: str = "base"
    description: str = ""

    def __init__(self, pw: int, ph: int) -> None:
        self.pw = pw
        self.ph = ph

    @abstractmethod
    def render_frame(
        self,
        t: float,
        rms: float,
        peak: float,
        bars: np.ndarray,
        title: str,
        artist: str,
    ) -> np.ndarray:
        ...

    def canvas(self) -> Canvas:
        return Canvas(self.pw, self.ph)
