"""Convert SVG shape elements into matplotlib :class:`~matplotlib.path.Path` objects.

Supports ``path``, ``rect`` (including rounded corners), ``circle``, ``ellipse``,
``line``, ``polyline`` and ``polygon``. Each converter reads the element's
geometric attributes and returns a single :class:`Path` in the element's own
coordinate system; transforms are applied later by the scene traversal.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

import numpy as np
from matplotlib.path import Path

from .pathdata import (
    ParsedPath,
    empty_path,
    parse_path_data,
    polyline_path_data,
)
from .units import LengthContext, parse_length

__all__ = [
    "SHAPE_TAGS",
    "MARKABLE_TAGS",
    "shape_to_path",
    "shape_to_parsed",
    "strip_ns",
]

SHAPE_TAGS = frozenset(
    {"path", "rect", "circle", "ellipse", "line", "polyline", "polygon"}
)
# Elements that accept marker-start/mid/end.
MARKABLE_TAGS = frozenset({"path", "line", "polyline", "polygon"})


def strip_ns(tag: str) -> str:
    """Drop an XML namespace from a tag, e.g. ``"{...svg}rect"`` -> ``"rect"``."""
    return tag.rsplit("}", 1)[-1]


_LENGTH_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?")
# Cubic-Bezier circle approximation constant (4/3 * (sqrt(2) - 1)).
_KAPPA = 0.5522847498307936


def _length(
    value: str | None,
    default: float = 0.0,
    axis: str = "other",
    ctx: LengthContext | None = None,
) -> float:
    """Parse an SVG length into user units (``"3px"`` -> ``3.0``, ``"1in"`` -> 96)."""
    return parse_length(value, default, axis=axis, ctx=ctx)


def _points(value: str | None) -> np.ndarray:
    """Parse a ``points`` list into an ``(n, 2)`` array."""
    nums = [float(m) for m in _LENGTH_RE.findall(value or "")]
    nums = nums[: len(nums) - (len(nums) % 2)]  # drop a dangling coordinate
    return np.array(nums, dtype=float).reshape(-1, 2)


def _rounded_rect(x, y, w, h, rx, ry) -> Path:
    """Build a rounded-rectangle path with cubic-Bezier corners."""
    rx = min(rx, w / 2)
    ry = min(ry, h / 2)
    cx, cy = _KAPPA * rx, _KAPPA * ry
    x1, y1 = x + w, y + h  # far corner
    verts = [
        (x + rx, y),  # start on the bottom edge
        (x1 - rx, y),  # bottom edge
        (x1 - rx + cx, y),
        (x1, y + ry - cy),
        (x1, y + ry),  # bottom-right corner
        (x1, y1 - ry),  # right edge
        (x1, y1 - ry + cy),
        (x1 - rx + cx, y1),
        (x1 - rx, y1),  # top-right corner
        (x + rx, y1),  # top edge
        (x + rx - cx, y1),
        (x, y1 - ry + cy),
        (x, y1 - ry),  # top-left corner
        (x, y + ry),  # left edge
        (x, y + ry - cy),
        (x + rx - cx, y),
        (x + rx, y),  # bottom-left corner, back to start
        (x + rx, y),
    ]
    codes = [
        Path.MOVETO,
        Path.LINETO,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.LINETO,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.LINETO,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.LINETO,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CLOSEPOLY,
    ]
    return Path(verts, codes)


def _rect(a: Mapping[str, str], ctx=None) -> Path:
    x, y = _length(a.get("x"), 0, "x", ctx), _length(a.get("y"), 0, "y", ctx)
    w, h = _length(a.get("width"), 0, "x", ctx), _length(a.get("height"), 0, "y", ctx)
    if w <= 0 or h <= 0:
        return empty_path()

    def radius(name, axis):
        v = a.get(name)
        if v is None or v.strip().lower() == "auto":
            return None
        r = _length(v, 0, axis, ctx)
        return r if r >= 0 else None

    rx, ry = radius("rx", "x"), radius("ry", "y")
    if rx is None and ry is None:
        return Path.unit_rectangle().transformed(_box_transform(x, y, w, h))
    rx = ry if rx is None else rx
    ry = rx if ry is None else ry
    if rx <= 0 or ry <= 0:
        return Path.unit_rectangle().transformed(_box_transform(x, y, w, h))
    return _rounded_rect(x, y, w, h, rx, ry)


def _box_transform(x, y, w, h):
    from matplotlib.transforms import Affine2D

    return Affine2D().scale(w, h).translate(x, y)


def _ellipse(cx, cy, rx, ry) -> Path:
    """An ellipse starting at ``(cx + rx, cy)`` and running in the positive
    angle direction, as the SVG spec defines (this sets the dash phase)."""
    if rx <= 0 or ry <= 0:
        return empty_path()
    k = _KAPPA
    verts = [
        (cx + rx, cy),
        (cx + rx, cy + k * ry),
        (cx + k * rx, cy + ry),
        (cx, cy + ry),
        (cx - k * rx, cy + ry),
        (cx - rx, cy + k * ry),
        (cx - rx, cy),
        (cx - rx, cy - k * ry),
        (cx - k * rx, cy - ry),
        (cx, cy - ry),
        (cx + k * rx, cy - ry),
        (cx + rx, cy - k * ry),
        (cx + rx, cy),
        (cx + rx, cy),
    ]
    codes = [Path.MOVETO] + [Path.CURVE4] * 12 + [Path.CLOSEPOLY]
    return Path(verts, codes)


def _line_points(a: Mapping[str, str], ctx=None) -> np.ndarray:
    return np.array(
        [
            (_length(a.get("x1"), 0, "x", ctx), _length(a.get("y1"), 0, "y", ctx)),
            (_length(a.get("x2"), 0, "x", ctx), _length(a.get("y2"), 0, "y", ctx)),
        ]
    )


def _line(a: Mapping[str, str], ctx=None) -> Path:
    return Path(_line_points(a, ctx), [Path.MOVETO, Path.LINETO])


def _poly(a: Mapping[str, str], close: bool) -> Path:
    return polyline_path_data(_points(a.get("points")), close).path


def shape_to_parsed(
    tag: str, attrib: Mapping[str, str], ctx: LengthContext | None = None
) -> ParsedPath | None:
    """Like :func:`shape_to_path` but also returns vertex data (for markers)."""
    tag = strip_ns(tag)
    if tag == "path":
        return parse_path_data(attrib.get("d"))
    if tag == "line":
        return polyline_path_data(_line_points(attrib, ctx), closed=False)
    if tag in ("polyline", "polygon"):
        return polyline_path_data(_points(attrib.get("points")), tag == "polygon")
    path = shape_to_path(tag, attrib, ctx)
    return None if path is None else ParsedPath(path, [])


def shape_to_path(
    tag: str, attrib: Mapping[str, str], ctx: LengthContext | None = None
) -> Path | None:
    """Convert one SVG shape element to a :class:`Path`.

    Parameters
    ----------
    tag : str
        The element tag, with or without an XML namespace.
    attrib : Mapping
        The element's attributes.
    ctx : LengthContext, optional
        Resolves percentages and font-relative units.

    Returns
    -------
    Path or None
        ``None`` if ``tag`` is not a recognised shape element.
    """
    tag = strip_ns(tag)
    if tag == "path":
        return parse_path_data(attrib.get("d")).path
    if tag == "rect":
        return _rect(attrib, ctx)
    cx = _length(attrib.get("cx"), 0, "x", ctx)
    cy = _length(attrib.get("cy"), 0, "y", ctx)
    if tag == "circle":
        r = _length(attrib.get("r"), 0, "other", ctx)
        return _ellipse(cx, cy, r, r)
    if tag == "ellipse":
        rx_attr, ry_attr = attrib.get("rx"), attrib.get("ry")
        auto = ("auto", None)
        rx = (
            None
            if (rx_attr or "auto").strip() in auto
            else _length(rx_attr, 0, "x", ctx)
        )
        ry = (
            None
            if (ry_attr or "auto").strip() in auto
            else _length(ry_attr, 0, "y", ctx)
        )
        rx = ry if rx is None else rx
        ry = rx if ry is None else ry
        return _ellipse(cx, cy, rx or 0.0, ry or 0.0)
    if tag == "line":
        return _line(attrib, ctx)
    if tag == "polyline":
        return _poly(attrib, close=False)
    if tag == "polygon":
        return _poly(attrib, close=True)
    return None
