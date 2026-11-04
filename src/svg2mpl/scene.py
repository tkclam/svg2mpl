"""The intermediate representation produced by parsing an SVG.

A :class:`Scene` is an ordered list of drawables -- :class:`Shape`,
:class:`Image` and :class:`Group` -- in painter's order, with geometry already
in *document space* (element ``transform`` and ``viewBox`` baked in, but *not*
the optional y-flip, which is applied at render time). Clipping is resolved to a
single document-space clip path per drawable.

Stroke widths and dash lengths in :attr:`Shape.kwargs` are in document units;
the renderer converts them to points so strokes scale with the drawing.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from matplotlib.path import Path
from matplotlib.transforms import Affine2D

__all__ = ["Shape", "Image", "Group", "Mask", "Text", "Scene", "BoundPaint"]


@dataclass
class BoundPaint:
    """A paint server (gradient/pattern) bound to the element that uses it.

    ``transform`` maps the element's user space to document space and ``bbox``
    is the element's user-space bounding box ``(x0, y0, x1, y1)`` (for
    ``objectBoundingBox`` units). ``opacity`` multiplies the paint's alpha.
    """

    server: Any
    transform: Affine2D
    bbox: tuple[float, float, float, float] | None
    opacity: float = 1.0
    # For patterns: the tile content, resolved for this element.
    tile: Any = None

    def average_color(self):
        color = self.server.average_color()
        if isinstance(color, tuple) and len(color) == 4:
            return (*color[:3], color[3] * self.opacity)
        return color


@dataclass
class Text:
    """A ``<text>`` element: its content, first anchor point and style.

    The glyph outlines are emitted as :class:`Shape` items (``Shape.text`` points
    back here); this record keeps the logical text for inspection and for the
    ``text_as_paths=False`` rendering mode.
    """

    id: str
    x: float
    y: float
    content: str
    style: dict[str, str]
    transform: Affine2D
    runs: list = field(default_factory=list)


@dataclass
class Shape:
    """A filled/stroked path with its resolved matplotlib style."""

    id: str
    path: Path
    style: dict[str, str]
    kwargs: dict[str, Any]
    clip: Path | None = None
    fill_paint: BoundPaint | None = None
    stroke_paint: BoundPaint | None = None
    scale_stroke: bool = True
    text: Text | None = None


@dataclass
class Image:
    """A raster image drawn through an affine pixel-to-document transform.

    ``data`` is an ``(h, w, 4)`` RGBA array whose row 0 is the *top* row;
    ``transform`` maps pixel coordinates (``x`` right, ``y`` down, the image
    spanning ``[0, w] x [0, h]``) to document space.
    """

    id: str
    data: np.ndarray
    transform: Affine2D
    clip: Path | None = None
    alpha: float = 1.0
    interpolation: str = "auto"

    @property
    def path(self) -> Path:
        """The image outline in document space (for bounding boxes)."""
        h, w = self.data.shape[:2]
        return Path([(0, 0), (w, 0), (w, h), (0, h), (0, 0)], closed=True).transformed(
            self.transform
        )


@dataclass
class Mask:
    """A ``<mask>``: its content items and document-space region."""

    items: list
    region: tuple[float, float, float, float]
    kind: str = "luminance"  # or "alpha"


@dataclass
class Group:
    """A subtree composited off-screen (for masks and filters).

    The renderer rasterizes ``items`` over ``region`` (document space), applies
    ``filter`` then ``mask`` and ``opacity``, and draws the result as an image
    clipped to ``clip``.
    """

    id: str
    items: list
    region: tuple[float, float, float, float]
    clip: Path | None = None
    mask: Mask | None = None
    filter: Any = None
    opacity: float = 1.0

    @property
    def path(self) -> Path:
        x0, y0, x1, y1 = self.region
        return Path([(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)], closed=True)


def iter_shapes(items) -> Iterator[Shape]:
    """Yield every :class:`Shape` in ``items``, descending into groups."""
    for item in items:
        if isinstance(item, Shape):
            yield item
        elif isinstance(item, Group):
            yield from iter_shapes(item.items)


@dataclass
class Scene:
    """A parsed SVG: its drawables, size, and paint-server definitions."""

    items: list = field(default_factory=list)
    texts: list[Text] = field(default_factory=list)
    width: float = 0.0
    height: float = 0.0
    gradients: dict[str, Any] = field(default_factory=dict)

    @property
    def shapes(self) -> list[Shape]:
        """All shapes in painter's order (including those inside groups)."""
        return list(iter_shapes(self.items))

    def __len__(self) -> int:
        return len(self.items)
