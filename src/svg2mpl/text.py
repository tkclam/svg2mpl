"""Render SVG ``<text>`` as matplotlib :class:`~matplotlib.text.Text` artists.

This is best-effort: the anchor point, fill color, ``text-anchor`` and an
approximate font size and rotation are honored. Font matching falls back to
matplotlib's defaults, and advanced layout (``tspan`` positioning, ``textPath``,
kerning) is not attempted.
"""

from __future__ import annotations

import math
import re

import numpy as np
from matplotlib.transforms import Affine2D

from .color import parse_paint

__all__ = ["add_texts"]

_NUM_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?")
_ANCHOR_TO_HA = {"start": "left", "middle": "center", "end": "right"}


def _font_size(value: str | None, default: float = 16.0) -> float:
    if not value:
        return default
    m = _NUM_RE.match(value.strip())
    return float(m.group()) if m else default


def _rotation_deg(affine: Affine2D) -> float:
    """Extract the rotation angle (degrees) of an affine, ignoring reflection."""
    a, b = affine.get_matrix()[0, 0], affine.get_matrix()[1, 0]
    return math.degrees(math.atan2(b, a))


def add_texts(ax, scene, *, affine2d, flip, base_transform):
    """Render every text drawable in ``scene`` onto ``ax``.

    Parameters
    ----------
    ax : Axes
    scene : Scene
    affine2d : Affine2D
        The placement affine (identity when not placing).
    flip : Affine2D or None
        The y-flip transform, or ``None``.
    base_transform : Transform
        Usually ``ax.transData``.
    """
    artists = []
    for text in scene.texts:
        # Anchor: element coords -> document space -> flip -> placement.
        point = text.transform.transform((text.x, text.y))
        if flip is not None:
            point = flip.transform(point)
        point = affine2d.transform(point)

        style = text.style
        color = parse_paint(style.get("fill"), default="black")
        if color == "none":
            color = "black"
        ha = _ANCHOR_TO_HA.get((style.get("text-anchor") or "start").strip(), "left")
        size = _font_size(style.get("font-size"))
        # Glyphs stay upright: combine placement + content rotation, drop the flip.
        rotation = _rotation_deg(text.transform + affine2d)

        artist = ax.text(
            point[0],
            point[1],
            text.content,
            color=color,
            fontsize=size,
            ha=ha,
            va="baseline",
            rotation=rotation,
            rotation_mode="anchor",
            transform=base_transform,
        )
        if style.get("font-family"):
            artist.set_fontfamily(style["font-family"].split(",")[0].strip())
        if style.get("font-weight"):
            artist.set_fontweight(style["font-weight"])
        if style.get("font-style"):
            artist.set_fontstyle(style["font-style"])
        artists.append(artist)
    return artists


def text_anchor_points(scene, *, affine2d, flip) -> np.ndarray:
    """Return the placed anchor points of all texts (for bbox inclusion)."""
    pts = []
    for text in scene.texts:
        p = text.transform.transform((text.x, text.y))
        if flip is not None:
            p = flip.transform(p)
        pts.append(affine2d.transform(p))
    return np.array(pts) if pts else np.empty((0, 2))
