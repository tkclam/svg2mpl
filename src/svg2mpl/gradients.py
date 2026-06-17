"""Parse SVG gradients and render them as clipped images.

matplotlib has no native gradient fill for arbitrary paths, so a gradient is
drawn as a small RGBA image (evaluated from the gradient stops) clipped to the
shape's outline. ``objectBoundingBox`` gradients (the default) are reproduced
faithfully under translation/scale; ``userSpaceOnUse`` and rotated placements are
best-effort. :meth:`Gradient.average_color` provides a solid-color fallback for
the batched fast path.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from matplotlib.transforms import Affine2D

from .color import parse_paint
from .shapes import strip_ns
from .transform import parse_transform

__all__ = ["Gradient", "parse_gradients", "add_gradient_fill"]

_XLINK = "{http://www.w3.org/1999/xlink}href"


def _stop_rgba(element) -> tuple[float, tuple[float, float, float, float]]:
    """Return ``(offset, rgba)`` for a ``<stop>`` element."""
    from matplotlib.colors import to_rgba

    attrib = element.attrib
    style = {}
    for decl in (attrib.get("style") or "").split(";"):
        if ":" in decl:
            k, _, v = decl.partition(":")
            style[k.strip()] = v.strip()
    offset = attrib.get("offset", style.get("offset", "0"))
    off = float(offset[:-1]) / 100.0 if offset.strip().endswith("%") else float(offset)
    color = parse_paint(
        attrib.get("stop-color", style.get("stop-color", "black")), default="black"
    )
    op = attrib.get("stop-opacity", style.get("stop-opacity", "1"))
    alpha = float(op[:-1]) / 100.0 if op.strip().endswith("%") else float(op)
    rgba = to_rgba(color, alpha)
    return min(max(off, 0.0), 1.0), rgba


@dataclass
class Gradient:
    """A parsed linear or radial gradient."""

    kind: str  # "linear" or "radial"
    stops: list[tuple[float, tuple]] = field(default_factory=list)
    units: str = "objectBoundingBox"
    transform: Affine2D = field(default_factory=Affine2D)
    coords: dict = field(default_factory=dict)

    def average_color(self):
        """A representative solid color (mean of the stops), for fallbacks."""
        if not self.stops:
            return "none"
        arr = np.array([rgba for _, rgba in self.stops], dtype=float)
        return tuple(arr.mean(axis=0))

    def color_at(self, t: np.ndarray) -> np.ndarray:
        """Evaluate the stop ramp at parameter(s) ``t`` in ``[0, 1]``."""
        if not self.stops:
            return np.zeros(t.shape + (4,))
        offs = np.array([o for o, _ in self.stops])
        cols = np.array([c for _, c in self.stops])
        t = np.clip(t, 0.0, 1.0)
        out = np.empty(t.shape + (4,))
        for ch in range(4):
            out[..., ch] = np.interp(t, offs, cols[:, ch])
        return out

    def _param(self, pts: np.ndarray) -> np.ndarray:
        """Map gradient-space points to the ramp parameter ``t``."""
        if self.kind == "linear":
            x1 = self.coords.get("x1", 0.0)
            y1 = self.coords.get("y1", 0.0)
            x2 = self.coords.get("x2", 1.0)
            y2 = self.coords.get("y2", 0.0)
            d = np.array([x2 - x1, y2 - y1])
            denom = float(d @ d) or 1.0
            return ((pts - [x1, y1]) @ d) / denom
        cx = self.coords.get("cx", 0.5)
        cy = self.coords.get("cy", 0.5)
        r = self.coords.get("r", 0.5) or 1e-9
        return np.hypot(pts[..., 0] - cx, pts[..., 1] - cy) / r

    def render_image(self, bbox, res: int = 256) -> np.ndarray:
        """Render the gradient as an ``(res, res, 4)`` RGBA image over ``bbox``."""
        x0, y0, x1, y1 = bbox.x0, bbox.y0, bbox.x1, bbox.y1
        xs = np.linspace(x0, x1, res)
        ys = np.linspace(y0, y1, res)
        gx, gy = np.meshgrid(xs, ys)
        if self.units == "objectBoundingBox":
            w = (x1 - x0) or 1.0
            h = (y1 - y0) or 1.0
            pts = np.stack([(gx - x0) / w, (gy - y0) / h], axis=-1)
        else:  # userSpaceOnUse (best-effort: assume image space == user space)
            pts = np.stack([gx, gy], axis=-1)
        # Undo the gradientTransform to evaluate in the gradient's own space.
        try:
            inv = self.transform.inverted()
            flat = inv.transform(pts.reshape(-1, 2))
            pts = flat.reshape(pts.shape)
        except Exception:
            pass
        return self.color_at(self._param(pts))


def _coords(element, kind) -> dict:
    out = {}
    names = (
        ("x1", "y1", "x2", "y2") if kind == "linear" else ("cx", "cy", "r", "fx", "fy")
    )
    for name in names:
        if name in element.attrib:
            val = element.attrib[name].strip()
            out[name] = float(val[:-1]) / 100.0 if val.endswith("%") else float(val)
    return out


def parse_gradients(root) -> dict[str, Gradient]:
    """Parse every ``<linearGradient>``/``<radialGradient>`` keyed by id.

    Handles ``xlink:href``/``href`` inheritance of stops and attributes.
    """
    raw = {}
    for el in root.iter():
        tag = strip_ns(el.tag)
        if tag in ("linearGradient", "radialGradient") and el.get("id"):
            raw[el.get("id")] = (tag, el)

    resolved: dict[str, Gradient] = {}

    def build(gid, seen):
        if gid in resolved:
            return resolved[gid]
        if gid not in raw or gid in seen:
            return None
        seen.add(gid)
        tag, el = raw[gid]
        kind = "linear" if tag == "linearGradient" else "radial"

        parent = None
        href = el.get(_XLINK) or el.get("href")
        if href and href.startswith("#"):
            parent = build(href[1:], seen)

        stops = [_stop_rgba(c) for c in el if strip_ns(c.tag) == "stop"]
        if not stops and parent is not None:
            stops = list(parent.stops)

        units = el.get("gradientUnits") or (
            parent.units if parent else "objectBoundingBox"
        )
        if "gradientTransform" in el.attrib:
            transform = parse_transform(el.attrib["gradientTransform"])
        else:
            transform = parent.transform if parent else Affine2D()

        coords = dict(parent.coords) if parent else {}
        coords.update(_coords(el, kind))

        grad = Gradient(
            kind=kind, stops=stops, units=units, transform=transform, coords=coords
        )
        resolved[gid] = grad
        return grad

    for gid in raw:
        build(gid, set())
    return resolved


def add_gradient_fill(
    ax, path, gradient: Gradient, *, transform=None, zorder=None, res: int = 256
):
    """Fill ``path`` (in ``transform`` coordinates) with ``gradient`` via an image.

    The gradient is rendered as an RGBA image over the path's bounding box and
    clipped to the path outline. ``transform`` defaults to ``ax.transData``.

    Returns the created :class:`~matplotlib.image.AxesImage`.
    """
    from matplotlib.patches import PathPatch

    if transform is None:
        transform = ax.transData
    bbox = path.get_extents()
    if bbox.width == 0 or bbox.height == 0:
        return None
    img = gradient.render_image(bbox, res=res)
    im = ax.imshow(
        img,
        extent=(bbox.x0, bbox.x1, bbox.y0, bbox.y1),
        origin="lower",
        interpolation="bilinear",
        zorder=zorder if zorder is not None else 1,
    )
    im.set_transform(transform)
    clip = PathPatch(path, transform=transform, facecolor="none", edgecolor="none")
    ax.add_patch(clip)
    im.set_clip_path(clip)
    return im
