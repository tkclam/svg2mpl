"""Parse SVG ``transform`` attributes and ``viewBox`` into matplotlib affines.

SVG and matplotlib disagree on two things this module reconciles:

* **Composition order.** In an SVG ``transform="A B"`` the left-most function is
  the outermost: a point is mapped as ``A @ (B @ p)``. matplotlib's ``+``
  operator composes the other way -- ``(P + Q).transform(p)`` applies ``P``
  first -- so the list is folded right-to-left with ``+`` (see
  :func:`parse_transform`).
* **Y axis direction.** SVG's y axis points *down*; matplotlib data coordinates
  point *up*. The optional flip is handled by :func:`flip_y_transform`, applied
  once at the document level rather than baked into every element.
"""

from __future__ import annotations

import math
import re

import numpy as np
from matplotlib.transforms import Affine2D

from .units import parse_numbers

__all__ = [
    "parse_angles",
    "parse_transform",
    "parse_viewbox",
    "parse_preserve_aspect_ratio",
    "viewbox_transform",
    "flip_y_transform",
]

# One transform function, e.g. ``translate(1, 2)`` or ``matrix(1 0 0 1 0 0)``.
# The CSS spellings (``translateX``, ``skew``, units such as ``deg``/``px``) are
# accepted too, since SVG 2 allows ``transform`` as a CSS property.
_FUNC_RE = re.compile(
    r"(matrix|translateX|translateY|translate|scaleX|scaleY|scale|rotate|"
    r"skewX|skewY|skew)\s*\(([^)]*)\)",
    re.IGNORECASE,
)
_NUM_UNIT_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?([a-zA-Z%]*)")
_ANGLE_UNITS = {"": 1.0, "deg": 1.0, "rad": 180.0 / math.pi, "grad": 0.9, "turn": 360.0}


def parse_angles(text: str | None) -> list[float]:
    """Return every number in ``text`` (commas and whitespace are separators).

    Angle units are converted to degrees; length units are dropped.
    """
    out = []
    for m in _NUM_UNIT_RE.finditer(text or ""):
        unit = m.group(1).lower()
        num = float(m.group()[: len(m.group()) - len(m.group(1))])
        out.append(num * _ANGLE_UNITS.get(unit, 1.0))
    return out


def _func_to_affine(name: str, args: list[float]) -> Affine2D:
    """Build the :class:`Affine2D` for a single SVG transform function."""
    name = name.lower()
    if not args:
        raise ValueError(f"Missing arguments for {name!r}")
    if name == "matrix":
        a, b, c, d, e, f = args[:6]
        return Affine2D.from_values(a, b, c, d, e, f)
    if name == "translatex":
        return Affine2D().translate(args[0], 0.0)
    if name == "translatey":
        return Affine2D().translate(0.0, args[0])
    if name == "scalex":
        return Affine2D().scale(args[0], 1.0)
    if name == "scaley":
        return Affine2D().scale(1.0, args[0])
    if name == "skew":
        ax_ = math.tan(math.radians(args[0]))
        ay_ = math.tan(math.radians(args[1])) if len(args) > 1 else 0.0
        return Affine2D.from_values(1, ay_, ax_, 1, 0, 0)
    if name == "translate":
        tx = args[0]
        ty = args[1] if len(args) > 1 else 0.0
        return Affine2D().translate(tx, ty)
    if name == "scale":
        sx = args[0]
        sy = args[1] if len(args) > 1 else sx
        return Affine2D().scale(sx, sy)
    if name == "rotate":
        if len(args) >= 3:
            angle, cx, cy = args[0], args[1], args[2]
            return Affine2D().translate(-cx, -cy).rotate_deg(angle).translate(cx, cy)
        return Affine2D().rotate_deg(args[0])
    if name == "skewx":
        from math import radians, tan

        return Affine2D.from_values(1, 0, tan(radians(args[0])), 1, 0, 0)
    if name == "skewy":
        from math import radians, tan

        return Affine2D.from_values(1, tan(radians(args[0])), 0, 1, 0, 0)
    raise ValueError(f"Unknown transform function: {name!r}")


def parse_transform(value: str | None) -> Affine2D:
    """Parse an SVG ``transform`` attribute into an :class:`Affine2D`.

    The returned affine maps a point in the element's local coordinates to its
    parent's coordinates, matching SVG semantics (left-most function outermost).

    Parameters
    ----------
    value : str or None
        The raw ``transform`` attribute, e.g. ``"translate(3 4) rotate(90)"``.

    Returns
    -------
    Affine2D
        Identity when ``value`` is empty or ``None``.
    """
    matrix = np.eye(3)
    if not value:
        return Affine2D()
    for name, args in _FUNC_RE.findall(value):
        try:
            m = _func_to_affine(name, parse_angles(args))
        except ValueError:
            return Affine2D()  # an invalid list disables the whole attribute
        # The left-most SVG function is the outermost, so each new (inner)
        # function multiplies on the right.
        matrix = matrix @ m.get_matrix()
    return Affine2D(matrix)


def parse_viewbox(value: str | None) -> tuple[float, float, float, float] | None:
    """Parse a ``viewBox`` attribute into ``(min_x, min_y, width, height)``."""
    if not value:
        return None
    nums = parse_numbers(value)
    if len(nums) != 4:
        return None
    return nums[0], nums[1], nums[2], nums[3]


def parse_preserve_aspect_ratio(value: str | None) -> tuple[str, str]:
    """Parse ``preserveAspectRatio`` into ``(align, "meet"|"slice")``."""
    align, mode = "xMidYMid", "meet"
    if not value:
        return align, mode
    tokens = value.split()
    if tokens and tokens[0] == "defer":
        tokens = tokens[1:]
    if tokens:
        if tokens[0] == "none" or (
            len(tokens[0]) == 8 and tokens[0][0] == "x" and tokens[0][4] == "Y"
        ):
            align = tokens[0]
    if len(tokens) > 1 and tokens[1] in ("meet", "slice"):
        mode = tokens[1]
    return align, mode


def viewbox_transform(
    viewbox: tuple[float, float, float, float] | None,
    width: float | None,
    height: float | None,
    preserve_aspect_ratio: str | None = None,
    x: float = 0.0,
    y: float = 0.0,
) -> Affine2D:
    """Return the affine mapping ``viewBox`` user units to the viewport.

    The viewport is the rectangle ``(x, y, width, height)``. When no explicit
    viewport size is given the viewBox units are used directly (only the origin
    offset is removed). ``preserve_aspect_ratio`` follows the SVG attribute of
    the same name (default ``xMidYMid meet``).
    """
    affine = Affine2D()
    if viewbox is None:
        return affine.translate(x, y)
    min_x, min_y, vb_w, vb_h = viewbox
    affine.translate(-min_x, -min_y)
    if not (width and height and vb_w > 0 and vb_h > 0):
        return affine.translate(x, y)
    sx, sy = width / vb_w, height / vb_h
    align, mode = parse_preserve_aspect_ratio(preserve_aspect_ratio)
    if align == "none":
        return affine.scale(sx, sy).translate(x, y)
    s = min(sx, sy) if mode == "meet" else max(sx, sy)
    affine.scale(s, s)
    extra_w, extra_h = width - vb_w * s, height - vb_h * s
    fx = {"xMin": 0.0, "xMid": 0.5, "xMax": 1.0}.get(align[:4], 0.5)
    fy = {"YMin": 0.0, "YMid": 0.5, "YMax": 1.0}.get(align[4:], 0.5)
    return affine.translate(x + extra_w * fx, y + extra_h * fy)


def flip_y_transform(height: float) -> Affine2D:
    """Return the affine that flips the y axis about ``height``.

    Converts SVG's y-down convention to matplotlib's y-up, keeping content in
    the ``[0, height]`` band so it stays in the same place after the flip.
    """
    return Affine2D().scale(1, -1).translate(0, height)
