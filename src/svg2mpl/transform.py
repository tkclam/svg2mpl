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

import re

from matplotlib.transforms import Affine2D

__all__ = ["parse_transform", "parse_viewbox", "viewbox_transform", "flip_y_transform"]

# One transform function, e.g. ``translate(1, 2)`` or ``matrix(1 0 0 1 0 0)``.
_FUNC_RE = re.compile(
    r"(matrix|translate|scale|rotate|skewX|skewY)\s*\(([^)]*)\)", re.IGNORECASE
)
_NUM_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?")


def _numbers(text: str) -> list[float]:
    """Return every number in ``text`` (commas and whitespace are separators)."""
    return [float(m) for m in _NUM_RE.findall(text)]


def _func_to_affine(name: str, args: list[float]) -> Affine2D:
    """Build the :class:`Affine2D` for a single SVG transform function."""
    name = name.lower()
    if name == "matrix":
        a, b, c, d, e, f = args
        return Affine2D.from_values(a, b, c, d, e, f)
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
    result = Affine2D()
    if not value:
        return result
    for name, args in _FUNC_RE.findall(value):
        m = _func_to_affine(name, _numbers(args))
        # Fold so that the left-most SVG function is applied last (outermost):
        # ``result`` accumulates the already-seen (more inner) functions.
        result = m + result
    return result


def parse_viewbox(value: str | None) -> tuple[float, float, float, float] | None:
    """Parse a ``viewBox`` attribute into ``(min_x, min_y, width, height)``."""
    if not value:
        return None
    nums = _numbers(value)
    if len(nums) != 4:
        return None
    return nums[0], nums[1], nums[2], nums[3]


def viewbox_transform(
    viewbox: tuple[float, float, float, float] | None,
    width: float | None,
    height: float | None,
) -> Affine2D:
    """Return the affine mapping ``viewBox`` user units to the viewport.

    When no explicit viewport ``width``/``height`` is given the viewBox units are
    used directly (only the origin offset is removed). ``preserveAspectRatio`` is
    treated as the default ``xMidYMid meet`` (uniform scaling). Downstream
    placement (:func:`svg2mpl.render.add_svg`) re-normalizes by the rendered
    bounding box, so the absolute scale rarely matters in practice.
    """
    affine = Affine2D()
    if viewbox is None:
        return affine
    min_x, min_y, vb_w, vb_h = viewbox
    affine.translate(-min_x, -min_y)
    if width and height and vb_w and vb_h:
        s = min(width / vb_w, height / vb_h)
        affine.scale(s, s)
    return affine


def flip_y_transform(height: float) -> Affine2D:
    """Return the affine that flips the y axis about ``height``.

    Converts SVG's y-down convention to matplotlib's y-up, keeping content in
    the ``[0, height]`` band so it stays in the same place after the flip.
    """
    return Affine2D().scale(1, -1).translate(0, height)
