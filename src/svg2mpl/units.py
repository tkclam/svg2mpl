"""Parse SVG/CSS lengths with units into user units (CSS pixels).

Absolute units use the CSS reference of 96 px per inch. Relative units need a
context: ``em``/``ex``/``rem`` resolve against a font size, and percentages
against a viewport dimension chosen by ``axis`` (``"x"`` for widths, ``"y"`` for
heights, anything else for the normalized diagonal, as the SVG spec requires
for radii and stroke widths).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

__all__ = [
    "NUMBER_RE",
    "UNITS",
    "LengthContext",
    "parse_length",
    "parse_number",
    "parse_numbers",
    "split_length",
]

NUMBER_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?")
_LENGTH_RE = re.compile(
    r"\s*([-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?)\s*([a-zA-Z%]*)\s*"
)

# Absolute units expressed in CSS pixels.
UNITS = {
    "": 1.0,
    "px": 1.0,
    "pt": 96.0 / 72.0,
    "pc": 16.0,
    "in": 96.0,
    "cm": 96.0 / 2.54,
    "mm": 96.0 / 25.4,
    "q": 96.0 / 101.6,
}

DEFAULT_FONT_SIZE = 16.0


@dataclass(frozen=True)
class LengthContext:
    """What relative lengths resolve against."""

    viewport_width: float = 100.0
    viewport_height: float = 100.0
    font_size: float = DEFAULT_FONT_SIZE

    @property
    def diagonal(self) -> float:
        return math.hypot(self.viewport_width, self.viewport_height) / math.sqrt(2)

    def percent_base(self, axis: str) -> float:
        if axis == "x":
            return self.viewport_width
        if axis == "y":
            return self.viewport_height
        return self.diagonal


def split_length(value: str | None) -> tuple[float, str] | None:
    """Split ``"3.5mm"`` into ``(3.5, "mm")``; ``None`` if it is not a length."""
    if value is None:
        return None
    m = _LENGTH_RE.match(value)
    if not m:
        return None
    return float(m.group(1)), m.group(2).lower()


def parse_length(
    value: str | None,
    default: float = 0.0,
    *,
    axis: str = "other",
    ctx: LengthContext | None = None,
) -> float:
    """Convert an SVG length to user units.

    Unknown units fall back to treating the number as user units, so malformed
    input degrades gracefully instead of raising.
    """
    parts = split_length(value)
    if parts is None:
        return default
    number, unit = parts
    if unit in UNITS:
        return number * UNITS[unit]
    ctx = ctx or LengthContext()
    if unit == "%":
        return number / 100.0 * ctx.percent_base(axis)
    if unit in ("em", "rem"):
        return number * ctx.font_size
    if unit == "ex":
        return number * ctx.font_size * 0.5
    if unit == "ch":
        return number * ctx.font_size * 0.5
    if unit == "vw":
        return number / 100.0 * ctx.viewport_width
    if unit == "vh":
        return number / 100.0 * ctx.viewport_height
    return number


def parse_number(value: str | None, default: float = 0.0) -> float:
    """Return the first number in ``value`` (ignoring any unit)."""
    if value is None:
        return default
    m = NUMBER_RE.search(value)
    return float(m.group()) if m else default


def parse_numbers(value: str | None) -> list[float]:
    """Return every number in a comma/whitespace separated list."""
    return [float(m) for m in NUMBER_RE.findall(value or "")]
