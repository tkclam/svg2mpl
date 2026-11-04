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
from collections.abc import Callable
from dataclasses import dataclass

__all__ = [
    "NUMBER_RE",
    "UNITS",
    "LengthContext",
    "parse_fraction",
    "parse_length",
    "parse_number",
    "parse_number_pair",
    "parse_numbers",
    "resolve_region",
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


def parse_number_pair(
    value: str | None, default: tuple[float, float] = (0.0, 0.0)
) -> tuple[float, float]:
    """Parse ``"a [b]"`` into ``(a, b)``; a single number is used for both."""
    nums = parse_numbers(value)
    if not nums:
        return default
    return nums[0], nums[1] if len(nums) > 1 else nums[0]


def parse_fraction(value: str | None, default: float = 0.0) -> float:
    """Parse an ``objectBoundingBox`` coordinate: ``"0.5"`` or ``"50%"`` -> 0.5."""
    if value is None:
        return default
    v = value.strip()
    try:
        return float(v[:-1]) / 100.0 if v.endswith("%") else float(v)
    except ValueError:
        return default


Rect = tuple[float, float, float, float]


def resolve_region(
    get: Callable[[str], str | None],
    units: str,
    bbox: Rect | None,
    ctx: LengthContext,
    default: tuple[str, str, str, str] = ("-10%", "-10%", "120%", "120%"),
) -> Rect | None:
    """Resolve an ``x/y/width/height`` region (masks, filters) to user space.

    ``get(name)`` returns an attribute value. With ``objectBoundingBox`` units
    the values are fractions of ``bbox``; otherwise they are lengths. Returns
    ``(x0, y0, x1, y1)``, or ``None`` for an empty region or missing bbox.
    """
    names = ("x", "y", "width", "height")
    vals = [get(n) or d for n, d in zip(names, default, strict=True)]
    if units == "objectBoundingBox":
        if bbox is None:
            return None
        bx0, by0, bx1, by1 = bbox
        bw, bh = bx1 - bx0, by1 - by0
        x = bx0 + parse_fraction(vals[0]) * bw
        y = by0 + parse_fraction(vals[1]) * bh
        w = parse_fraction(vals[2]) * bw
        h = parse_fraction(vals[3]) * bh
    else:
        x = parse_length(vals[0], 0.0, axis="x", ctx=ctx)
        y = parse_length(vals[1], 0.0, axis="y", ctx=ctx)
        w = parse_length(vals[2], 0.0, axis="x", ctx=ctx)
        h = parse_length(vals[3], 0.0, axis="y", ctx=ctx)
    if w <= 0 or h <= 0:
        return None
    return x, y, x + w, y + h
