"""Rasterizer creating a pixelated image of the cart's position.
"""

from dataclasses import dataclass

import numpy as np

__all__ = ["SceneSpec", "SceneRenderer"]


@dataclass
class SceneSpec:
    """Geometry of the rendered scene.

    Attributes
    ----------
    width, height : int
        Frame size in pixels.
    span : float
        Half-width of the visible track, in metres.
    dot_radius_px : float
        Radius of the dot, in pixels.
    """

    width: int = 96
    height: int = 96
    span: float = 12.0
    dot_radius_px: float = 2.0

    @property
    def metres_per_pixel(self) -> float:
        """Track length covered by one pixel column."""
        return 2.0 * self.span / self.width

    def column(self, x: float) -> float:
        """Fractional pixel column of track position ``x``."""
        return (x + self.span) / (2.0 * self.span) * self.width


class SceneRenderer:
    """Draws a white dot on black at the cart's position, as a uint8 frame.

    Parameters
    ----------
    spec : SceneSpec
        Geometry of the scene.
    """

    def __init__(self, spec: SceneSpec):
        if spec.dot_radius_px <= 0.0:
            raise ValueError(f"Dot radius must be positive, got {spec.dot_radius_px}.")

        self.spec = spec

        # The dot's shape, stamped out once.
        self._reach = int(np.floor(spec.dot_radius_px))
        offsets = np.arange(-self._reach, self._reach + 1, dtype=float)
        self._stamp = np.hypot(offsets[:, None], offsets[None, :]) <= spec.dot_radius_px
        self._row0 = int(round(spec.height / 2.0)) - self._reach

    def frame(self, x: float) -> np.ndarray:
        """The frame of shape ``(height, width)``; empty if the dot is out of view."""
        h, w = self.spec.height, self.spec.width
        frame = np.zeros((h, w), dtype=np.uint8)

        column = self.spec.column(float(x))
        if not np.isfinite(column):
            return frame

        size = self._stamp.shape[0]
        r0 = self._row0
        c0 = int(round(column)) - self._reach

        r_lo, r_hi = max(r0, 0), min(r0 + size, h)
        c_lo, c_hi = max(c0, 0), min(c0 + size, w)
        if r_lo >= r_hi or c_lo >= c_hi:
            return frame

        stamp = self._stamp[r_lo - r0:r_hi - r0, c_lo - c0:c_hi - c0]
        frame[r_lo:r_hi, c_lo:c_hi][stamp] = 255

        return frame

    def is_visible(self, x: float) -> bool:
        """Whether any part of the dot is still in the frame."""
        if not np.isfinite(x):
            return False

        column = self.spec.column(float(x))
        return bool(-self.spec.dot_radius_px < column < self.spec.width + self.spec.dot_radius_px)

    def rgb(
        self,
        x: float,
        velocity: float = 0.0,
        thrust: np.ndarray | None = None,
        angles: np.ndarray | None = None,
        scale: int = 8,
    ) -> np.ndarray:
        """The policy's frame magnified, with thrust axes and a velocity line drawn in.

        Returns
        -------
        np.ndarray
            Of shape ``(height * scale, width * scale, 3)`` and dtype uint8.
        """
        big = np.kron(self.frame(x), np.ones((scale, scale), dtype=np.uint8))
        rgb = np.repeat(big[:, :, None], 3, axis=2)

        col = self.spec.column(float(x)) * scale
        row = self.spec.height / 2.0 * scale

        if thrust is not None and angles is not None:
            length = 5.0 * self.spec.dot_radius_px * scale
            for command, angle in zip(np.ravel(thrust), np.ravel(angles)):
                mag = min(abs(float(command)), 1.0)
                colour = (int(60 + 195 * mag), int(60 * (1 - mag)), int(60 * (1 - mag)))
                dc, dr = np.cos(angle) * length, -np.sin(angle) * length
                _draw_line(rgb, row - dr, col - dc, row + dr, col + dc, colour)

        if velocity != 0.0:
            end = col + velocity / self.spec.metres_per_pixel * scale
            _draw_line(rgb, row, col, row, end, (40, 200, 90))

        return rgb


def _draw_line(rgb: np.ndarray, r0: float, c0: float, r1: float, c1: float, colour: tuple) -> None:
    """Draw a one-pixel line onto an RGB image, in place."""
    if not np.all(np.isfinite([r0, c0, r1, c1])):
        return

    steps = int(max(abs(r1 - r0), abs(c1 - c0))) + 1
    rows = np.rint(np.linspace(r0, r1, steps)).astype(int)
    cols = np.rint(np.linspace(c0, c1, steps)).astype(int)

    inside = (rows >= 0) & (rows < rgb.shape[0]) & (cols >= 0) & (cols < rgb.shape[1])
    rgb[rows[inside], cols[inside]] = colour
