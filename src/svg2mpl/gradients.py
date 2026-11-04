"""Parse SVG paint servers (gradients and patterns) and render gradients.

matplotlib has no native gradient fill for arbitrary paths, so a gradient is
drawn as an RGBA image evaluated per pixel from the gradient definition and
clipped to the shape's outline. Evaluation happens in the referencing
element's user space, so ``objectBoundingBox`` and ``userSpaceOnUse`` units,
``gradientTransform``, rotated/skewed elements, ``spreadMethod``
(``pad``/``reflect``/``repeat``) and radial focal points (``fx``/``fy``/``fr``)
are all reproduced. :meth:`Gradient.average_color` provides a solid-color
fallback.

Patterns are parsed into :class:`Pattern` definitions here; their tile content
is resolved by the parser and rasterized by the renderer.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.transforms import Affine2D

from ._xml import href, strip_ns
from .color import NONE, parse_color
from .style import compute_style, declarations, opacity
from .transform import parse_transform, parse_viewbox
from .units import LengthContext, parse_fraction, parse_length

__all__ = ["Gradient", "Pattern", "parse_gradients"]

_LINEAR_DEFAULTS = {"x1": "0%", "y1": "0%", "x2": "100%", "y2": "0%"}
_RADIAL_DEFAULTS = {"cx": "50%", "cy": "50%", "r": "50%", "fr": "0%"}
_AXIS = {"x1": "x", "x2": "x", "cx": "x", "fx": "x", "y1": "y", "y2": "y"}
_AXIS.update({"cy": "y", "fy": "y"})


def _stop(element, style) -> tuple[float, tuple[float, float, float, float]]:
    """Return ``(offset, rgba)`` for a ``<stop>`` with its computed style."""
    offset = element.attrib.get("offset", style.get("offset", "0")).strip()
    try:
        off = float(offset[:-1]) / 100.0 if offset.endswith("%") else float(offset)
    except ValueError:
        off = 0.0
    current = parse_color(style.get("color"), default="black")
    if current == "none":
        current = "black"
    color = parse_color(
        style.get("stop-color", "black"), current_color=current, default="black"
    )
    if color == "none":
        color = (0.0, 0.0, 0.0, 0.0)
    r, g, b, a = to_rgba(color)
    a *= opacity(style.get("stop-opacity"))
    return min(max(off, 0.0), 1.0), (r, g, b, a)


@dataclass
class Gradient:
    """A parsed linear or radial gradient."""

    kind: str  # "linear" or "radial"
    stops: list[tuple[float, tuple]] = field(default_factory=list)
    units: str = "objectBoundingBox"
    transform: Affine2D = field(default_factory=Affine2D)
    coords: dict = field(default_factory=dict)
    spread: str = "pad"
    id: str = ""
    # Resolved defaults for omitted coordinates (userSpaceOnUse percentages).
    defaults: dict = field(default_factory=dict)

    def average_color(self):
        """A representative solid color (mean of the stops), for fallbacks."""
        if not self.stops:
            return "none"
        arr = np.array([rgba for _, rgba in self.stops], dtype=float)
        return tuple(arr.mean(axis=0))

    def color_at(self, t: np.ndarray) -> np.ndarray:
        """Evaluate the stop ramp at parameter(s) ``t`` (spread applied)."""
        t = np.asarray(t, dtype=float)
        if not self.stops:
            return np.zeros((*t.shape, 4))
        if self.spread == "repeat":
            t = np.mod(t, 1.0)
        elif self.spread == "reflect":
            t = np.abs(np.mod(t + 1.0, 2.0) - 1.0)
        offs = np.array([o for o, _ in self.stops])
        cols = np.array([c for _, c in self.stops])
        t = np.clip(t, 0.0, 1.0)
        out = np.empty((*t.shape, 4))
        for ch in range(4):
            out[..., ch] = np.interp(t, offs, cols[:, ch])
        return out

    def _coord(self, name: str) -> float:
        if name in self.coords:
            return self.coords[name]
        if name == "fx":
            return self._coord("cx")
        if name == "fy":
            return self._coord("cy")
        if name in self.defaults:
            return self.defaults[name]
        defaults = _LINEAR_DEFAULTS if self.kind == "linear" else _RADIAL_DEFAULTS
        return parse_fraction(defaults[name])

    def _param(self, pts: np.ndarray) -> np.ndarray:
        """Map gradient-space points to the ramp parameter ``t`` (NaN = no paint)."""
        if self.kind == "linear":
            x1, y1 = self._coord("x1"), self._coord("y1")
            x2, y2 = self._coord("x2"), self._coord("y2")
            d = np.array([x2 - x1, y2 - y1])
            denom = float(d @ d)
            if denom == 0:
                return np.ones(pts.shape[:-1])  # degenerate: last stop color
            return ((pts - [x1, y1]) @ d) / denom
        cx, cy, r = self._coord("cx"), self._coord("cy"), self._coord("r")
        fx, fy, fr = self._coord("fx"), self._coord("fy"), self._coord("fr")
        if r <= 0:
            return np.ones(pts.shape[:-1])
        # SVG 1.1: a focal point outside the end circle is moved onto it.
        fdx, fdy = fx - cx, fy - cy
        fd = np.hypot(fdx, fdy)
        if fr == 0 and fd > r * 0.999:
            k = r * 0.999 / fd
            fx, fy = cx + fdx * k, cy + fdy * k
        if fx == cx and fy == cy and fr == 0:
            return np.hypot(pts[..., 0] - cx, pts[..., 1] - cy) / r
        # Two-point conical gradient: largest t with |p - c(t)| = r(t) >= 0.
        cdx, cdy, dr = cx - fx, cy - fy, r - fr
        px, py = pts[..., 0] - fx, pts[..., 1] - fy
        a = cdx * cdx + cdy * cdy - dr * dr
        b = px * cdx + py * cdy + fr * dr
        c = px * px + py * py - fr * fr
        if abs(a) < 1e-12:
            with np.errstate(divide="ignore", invalid="ignore"):
                t = c / (2 * b)
        else:
            disc = b * b - a * c
            with np.errstate(invalid="ignore"):
                sq = np.sqrt(np.where(disc >= 0, disc, np.nan))
            t1, t2 = (b + sq) / a, (b - sq) / a
            t = np.fmax(t1, t2)
            bad = fr + t * dr < 0
            t = np.where(bad, np.fmin(t1, t2), t)
        return t

    def evaluate(self, user_pts: np.ndarray, bbox=None) -> np.ndarray:
        """RGBA at points given in the referencing element's user space."""
        pts = np.asarray(user_pts, dtype=float)
        if self.units == "objectBoundingBox":
            if bbox is None:
                return np.zeros((*pts.shape[:-1], 4))
            x0, y0, x1, y1 = bbox
            w, h = (x1 - x0), (y1 - y0)
            if w == 0 or h == 0:
                return np.zeros((*pts.shape[:-1], 4))
            pts = np.stack([(pts[..., 0] - x0) / w, (pts[..., 1] - y0) / h], axis=-1)
        try:
            inv = self.transform.inverted()
        except np.linalg.LinAlgError:  # degenerate gradientTransform: no paint
            return np.zeros((*pts.shape[:-1], 4))
        pts = inv.transform(pts.reshape(-1, 2)).reshape(pts.shape)
        t = self._param(pts)
        out = self.color_at(np.nan_to_num(t, nan=0.0))
        out[np.isnan(t)] = 0.0
        return out

    def render_image(self, bbox, res: int = 256) -> np.ndarray:
        """Render the gradient as an ``(res, res, 4)`` RGBA image over ``bbox``.

        ``bbox`` (a matplotlib ``Bbox``) is both the sampled area and, for
        ``objectBoundingBox`` gradients, the object box. Row 0 is ``bbox.y0``.
        """
        x0, y0, x1, y1 = bbox.x0, bbox.y0, bbox.x1, bbox.y1
        xs = np.linspace(x0, x1, res)
        ys = np.linspace(y0, y1, res)
        gx, gy = np.meshgrid(xs, ys)
        return self.evaluate(np.stack([gx, gy], axis=-1), (x0, y0, x1, y1))


@dataclass
class Pattern:
    """A parsed ``<pattern>`` (attributes resolved through ``href`` chains)."""

    id: str
    element: ET.Element  # the element whose children form the tile
    x: str = "0"
    y: str = "0"
    width: str = "0"
    height: str = "0"
    units: str = "objectBoundingBox"
    content_units: str = "userSpaceOnUse"
    viewbox: tuple | None = None
    preserve_aspect_ratio: str | None = None
    transform: Affine2D = field(default_factory=Affine2D)
    color: tuple | str = "none"

    def average_color(self):
        return self.color


def _coords(raw: dict, units: str, ctx: LengthContext) -> dict:
    """Resolve coordinate attributes; unparsable values are left out."""
    out = {}
    for name, val in raw.items():
        if units == "objectBoundingBox":
            value = parse_fraction(val, math.nan)
        else:
            value = parse_length(val, math.nan, axis=_AXIS.get(name, "other"), ctx=ctx)
        if not math.isnan(value):
            out[name] = value
    return out


def _attribute_style(el: ET.Element) -> dict:
    """An element's style from its own attributes only (no CSS, no cascade)."""
    return compute_style(declarations(el.attrib), None)


def parse_gradients(
    root: ET.Element,
    style_of: Callable[[ET.Element], dict] | None = None,
    ctx: LengthContext | None = None,
) -> dict[str, Gradient | Pattern]:
    """Parse every gradient and pattern keyed by id.

    Handles ``xlink:href``/``href`` inheritance of stops and attributes.
    ``style_of(element)`` returns an element's computed style (used for stop
    colors set via CSS); by default only attributes and inline styles are used.
    """
    ctx = ctx or LengthContext()
    style_of = style_of or _attribute_style

    raw = {}
    for el in root.iter():
        if not isinstance(el.tag, str):
            continue
        tag = strip_ns(el.tag)
        if tag in ("linearGradient", "radialGradient", "pattern") and el.get("id"):
            raw[el.get("id")] = (tag, el)

    resolved: dict = {}

    def chain(gid):
        """The href chain starting at ``gid`` (cycle-safe)."""
        out, seen = [], set()
        while gid in raw and gid not in seen:
            seen.add(gid)
            out.append(raw[gid])
            ref = href(raw[gid][1])
            gid = ref[1:] if ref and ref.startswith("#") else None
        return out

    def first_attr(links, name, kinds=None):
        for tag, el in links:
            if kinds and tag not in kinds:
                continue
            if name in el.attrib:
                return el.attrib[name]
        return None

    for gid, (tag, _el) in raw.items():
        links = chain(gid)
        if tag == "pattern":
            resolved[gid] = _build_pattern(gid, links, first_attr, style_of)
            continue
        kind = "linear" if tag == "linearGradient" else "radial"
        stops = []
        for _t, link in links:
            kids = [
                c for c in link if isinstance(c.tag, str) and strip_ns(c.tag) == "stop"
            ]
            if kids:
                stops = [_stop(c, style_of(c)) for c in kids]
                break
        # Offsets must be monotonic.
        last = 0.0
        for i, (off, rgba) in enumerate(stops):
            last = max(last, off)
            stops[i] = (last, rgba)
        grad_kinds = ("linearGradient", "radialGradient")
        units = first_attr(links, "gradientUnits", grad_kinds) or "objectBoundingBox"
        tr = first_attr(links, "gradientTransform", grad_kinds)
        transform = parse_transform(tr) if tr else Affine2D()
        spread = (first_attr(links, "spreadMethod", grad_kinds) or "pad").strip()
        names = (
            ("x1", "y1", "x2", "y2")
            if kind == "linear"
            else ("cx", "cy", "r", "fx", "fy", "fr")
        )
        raw_coords = {}
        for n in names:
            v = first_attr(links, n, (tag,))
            if v is not None:
                raw_coords[n] = v
        table = _LINEAR_DEFAULTS if kind == "linear" else _RADIAL_DEFAULTS
        defaults = _coords(table, units, ctx) if units == "userSpaceOnUse" else {}
        resolved[gid] = Gradient(
            defaults=defaults,
            kind=kind,
            stops=stops,
            units=units,
            transform=transform,
            coords=_coords(raw_coords, units, ctx),
            spread=spread if spread in ("pad", "reflect", "repeat") else "pad",
            id=gid,
        )
    return resolved


def _build_pattern(gid, links, first_attr, style_of) -> Pattern:
    kinds = ("pattern",)
    content = links[0][1]
    for _t, el in links:
        if any(isinstance(c.tag, str) for c in el):
            content = el
            break
    tr = first_attr(links, "patternTransform", kinds)
    pattern = Pattern(
        id=gid,
        element=content,
        x=first_attr(links, "x", kinds) or "0",
        y=first_attr(links, "y", kinds) or "0",
        width=first_attr(links, "width", kinds) or "0",
        height=first_attr(links, "height", kinds) or "0",
        units=first_attr(links, "patternUnits", kinds) or "objectBoundingBox",
        content_units=first_attr(links, "patternContentUnits", kinds)
        or "userSpaceOnUse",
        viewbox=parse_viewbox(first_attr(links, "viewBox", kinds)),
        preserve_aspect_ratio=first_attr(links, "preserveAspectRatio", kinds),
        transform=parse_transform(tr) if tr else Affine2D(),
    )
    # A representative color for stroke fallbacks: the first filled child.
    for el in content.iter():
        if el is content or not isinstance(el.tag, str):
            continue
        color = parse_color(style_of(el).get("fill"), default=NONE)
        if color != NONE:
            pattern.color = color
            break
    return pattern
