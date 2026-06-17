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

from .pathdata import empty_path, parse_path

__all__ = ["SHAPE_TAGS", "shape_to_path", "strip_ns"]

SHAPE_TAGS = frozenset(
    {"path", "rect", "circle", "ellipse", "line", "polyline", "polygon"}
)


def strip_ns(tag: str) -> str:
    """Drop an XML namespace from a tag, e.g. ``"{...svg}rect"`` -> ``"rect"``."""
    return tag.rsplit("}", 1)[-1]


_LENGTH_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?")
# Cubic-Bezier circle approximation constant (4/3 * (sqrt(2) - 1)).
_KAPPA = 0.5522847498307936


def _length(value: str | None, default: float = 0.0) -> float:
    """Parse an SVG length, ignoring a trailing unit (e.g. ``"3px"`` -> ``3.0``)."""
    if value is None:
        return default
    m = _LENGTH_RE.match(value.strip())
    return float(m.group()) if m else default


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


def _rect(a: Mapping[str, str]) -> Path:
    x, y = _length(a.get("x")), _length(a.get("y"))
    w, h = _length(a.get("width")), _length(a.get("height"))
    if w <= 0 or h <= 0:
        return empty_path()
    rx_attr, ry_attr = a.get("rx"), a.get("ry")
    rx = _length(rx_attr) if rx_attr is not None else None
    ry = _length(ry_attr) if ry_attr is not None else None
    if rx is None and ry is None:
        return Path.unit_rectangle().transformed(_box_transform(x, y, w, h))
    rx = ry if rx is None else rx
    ry = rx if ry is None else ry
    if rx <= 0 and ry <= 0:
        return Path.unit_rectangle().transformed(_box_transform(x, y, w, h))
    return _rounded_rect(x, y, w, h, rx, ry)


def _box_transform(x, y, w, h):
    from matplotlib.transforms import Affine2D

    return Affine2D().scale(w, h).translate(x, y)


def _ellipse(cx, cy, rx, ry) -> Path:
    from matplotlib.transforms import Affine2D

    if rx <= 0 or ry <= 0:
        return empty_path()
    return Path.unit_circle().transformed(Affine2D().scale(rx, ry).translate(cx, cy))


def _line(a: Mapping[str, str]) -> Path:
    p0 = (_length(a.get("x1")), _length(a.get("y1")))
    p1 = (_length(a.get("x2")), _length(a.get("y2")))
    return Path([p0, p1], [Path.MOVETO, Path.LINETO])


def _poly(a: Mapping[str, str], close: bool) -> Path:
    pts = _points(a.get("points"))
    if len(pts) == 0:
        return empty_path()
    codes = [Path.MOVETO] + [Path.LINETO] * (len(pts) - 1)
    if close:
        pts = np.vstack([pts, pts[0]])
        codes.append(Path.CLOSEPOLY)
    return Path(pts, codes)


def shape_to_path(tag: str, attrib: Mapping[str, str]) -> Path | None:
    """Convert one SVG shape element to a :class:`Path`.

    Parameters
    ----------
    tag : str
        The element tag, with or without an XML namespace.
    attrib : Mapping
        The element's attributes.

    Returns
    -------
    Path or None
        ``None`` if ``tag`` is not a recognised shape element.
    """
    tag = strip_ns(tag)
    if tag == "path":
        return parse_path(attrib.get("d"))
    if tag == "rect":
        return _rect(attrib)
    if tag == "circle":
        r = _length(attrib.get("r"))
        return _ellipse(_length(attrib.get("cx")), _length(attrib.get("cy")), r, r)
    if tag == "ellipse":
        return _ellipse(
            _length(attrib.get("cx")),
            _length(attrib.get("cy")),
            _length(attrib.get("rx")),
            _length(attrib.get("ry")),
        )
    if tag == "line":
        return _line(attrib)
    if tag == "polyline":
        return _poly(attrib, close=False)
    if tag == "polygon":
        return _poly(attrib, close=True)
    return None
