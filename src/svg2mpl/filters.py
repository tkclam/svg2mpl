"""Resolve SVG filters into primitive chains (evaluated by :mod:`._filter_ops`).

An element with ``filter`` is rasterized off-screen over its filter region; the
primitives then run on that RGBA image, which is drawn back as an image.
Implemented primitives:

``feBlend``, ``feColorMatrix``, ``feComponentTransfer``, ``feComposite``,
``feConvolveMatrix``, ``feDiffuseLighting``, ``feDisplacementMap``,
``feDropShadow``, ``feFlood``, ``feGaussianBlur``, ``feImage``, ``feMerge``,
``feMorphology``, ``feOffset``, ``feSpecularLighting``, ``feTile`` and
``feTurbulence``, plus the CSS filter functions (``blur()``, ``drop-shadow()``,
``grayscale()``, ``sepia()``, ``saturate()``, ``hue-rotate()``, ``invert()``,
``opacity()``, ``brightness()``, ``contrast()``).

Inputs ``SourceGraphic``, ``SourceAlpha`` and named results are supported;
``BackgroundImage``/``BackgroundAlpha``/``FillPaint``/``StrokePaint`` evaluate
to transparent black. ``color-interpolation-filters`` (``linearRGB`` by
default, or ``sRGB``) is honored per filter. Lengths are converted through the
element's transform into document units; filters on rotated or skewed elements
are approximated axis-aligned. ``feTurbulence``'s ``stitchTiles`` is ignored.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import numpy as np
from matplotlib.colors import to_rgb, to_rgba
from matplotlib.transforms import Affine2D

from ._filter_ops import apply_filter as _apply
from ._state import State
from ._xml import href, strip_ns
from .color import parse_color
from .geometry import Rect, axis_scales, transform_rect, transform_vector
from .style import opacity
from .transform import parse_angles
from .units import (
    LengthContext,
    parse_fraction,
    parse_length,
    parse_number,
    parse_number_pair,
    parse_numbers,
    resolve_region,
)

if TYPE_CHECKING:
    from xml.etree.ElementTree import Element

    from .parser import Parser

__all__ = ["Filter", "apply_filter", "css_filter", "parse_filter"]


@dataclass
class Filter:
    """A resolved filter chain (all lengths in document units)."""

    primitives: list[dict]
    region: Rect
    color_space: str = "linearRGB"
    # feImage content keyed by primitive index: ("items", items) or
    # ("raster", rgba, subregion).
    images: dict = field(default_factory=dict)


def apply_filter(flt: Filter, source, region, ppx, ppy, rasterize):
    """Run ``flt`` on a straight-alpha sRGB image (see :mod:`._filter_ops`)."""
    return _apply(
        flt.primitives,
        source,
        region,
        ppx,
        ppy,
        rasterize,
        color_space=flt.color_space,
        images=flt.images,
    )


# --------------------------------------------------------------------------- #
# <filter> elements
# --------------------------------------------------------------------------- #
@dataclass
class _PrimitiveContext:
    """Everything a primitive parser needs besides the element."""

    parser: Parser
    st: State
    ctm: Affine2D
    style: dict
    prim_units: str
    bbox: Rect | None
    filter: Filter

    def length(self, value: float, axis: str) -> float:
        """A primitive length (a number in primitive units) in user units."""
        if self.prim_units == "objectBoundingBox" and self.bbox is not None:
            span = (
                self.bbox[2] - self.bbox[0]
                if axis == "x"
                else self.bbox[3] - self.bbox[1]
            )
            return value * span
        return value

    def doc_lengths(self, value: str | None, default=(0.0, 0.0)) -> tuple[float, float]:
        """An ``"x [y]"`` length pair in document units (per axis)."""
        kx, ky = axis_scales(self.ctm)
        x, y = parse_number_pair(value, default)
        return self.length(x, "x") * kx, self.length(y, "y") * ky


def _href_chain(parser: Parser, f_el: Element) -> list[Element]:
    """``f_el`` followed by the filters it inherits from through ``href``."""
    chain: list[Element] = []
    el: Element | None = f_el
    while el is not None and el not in chain:
        chain.append(el)
        ref = href(el)
        el = parser.id_index.get(ref[1:]) if ref and ref.startswith("#") else None
    return chain


def parse_filter(
    parser: Parser, f_el: Element, bbox_user: Rect | None, ctm: Affine2D, st: State
) -> Filter | None:
    """Resolve a ``<filter>`` element for one referencing element."""
    chain = _href_chain(parser, f_el)

    def attr(name: str) -> str | None:
        return next((e.attrib[name] for e in chain if name in e.attrib), None)

    fstyle = parser.style_of(f_el)
    units = attr("filterUnits") or "objectBoundingBox"
    rect = resolve_region(attr, units, bbox_user, parser.ctx(fstyle, st))
    if rect is None:
        return None
    space = (fstyle.get("color-interpolation-filters") or "linearRGB").strip()
    out = Filter(primitives=[], region=transform_rect(rect, ctm), color_space=space)
    # Primitives come from the first filter in the chain that has children.
    host = next((e for e in chain if any(isinstance(c.tag, str) for c in e)), f_el)
    prim_units = attr("primitiveUnits") or "userSpaceOnUse"
    for index, child in enumerate(c for c in host if isinstance(c.tag, str)):
        tag = strip_ns(child.tag)
        if not tag.startswith("fe"):
            continue
        pctx = _PrimitiveContext(
            parser,
            st,
            ctm,
            parser.child_style(child, fstyle),
            prim_units,
            bbox_user,
            out,
        )
        prim: dict = {
            "type": tag,
            "in": child.get("in"),
            "in2": child.get("in2"),
            "result": child.get("result"),
        }
        subregion = _subregion(child, pctx, rect)
        if subregion is not None:
            prim["subregion"] = subregion
        handler = _PRIMITIVE_PARSERS.get(tag)
        if handler is not None:
            handler(child, prim, pctx, index)
        out.primitives.append(prim)
    return out


def _subregion(child: Element, pctx: _PrimitiveContext, rect: Rect) -> Rect | None:
    """A primitive's ``x/y/width/height`` subregion in document space."""
    a = child.attrib
    if not any(k in a for k in ("x", "y", "width", "height")):
        return None
    if pctx.prim_units == "objectBoundingBox" and pctx.bbox is not None:
        bx0, by0, bx1, by1 = pctx.bbox
        bw, bh = bx1 - bx0, by1 - by0
        x0 = bx0 + parse_fraction(a["x"]) * bw if "x" in a else rect[0]
        y0 = by0 + parse_fraction(a["y"]) * bh if "y" in a else rect[1]
        w = parse_fraction(a["width"]) * bw if "width" in a else rect[2] - x0
        h = parse_fraction(a["height"]) * bh if "height" in a else rect[3] - y0
    else:
        ctx = pctx.parser.ctx(pctx.style, pctx.st)
        x0 = parse_length(a.get("x"), rect[0], axis="x", ctx=ctx)
        y0 = parse_length(a.get("y"), rect[1], axis="y", ctx=ctx)
        w = parse_length(a.get("width"), rect[2] - x0, axis="x", ctx=ctx)
        h = parse_length(a.get("height"), rect[3] - y0, axis="y", ctx=ctx)
    if w <= 0 or h <= 0:
        return None
    return transform_rect((x0, y0, x0 + w, y0 + h), pctx.ctm)


def _flood_color(style: dict) -> tuple[float, float, float, float]:
    color = parse_color(style.get("flood-color", "black"), default="black")
    r, g, b, a = to_rgba(color if color != "none" else (0, 0, 0, 0))
    return r, g, b, a * opacity(style.get("flood-opacity"))


def _p_blur(el, prim, pctx, _i):
    prim["std"] = pctx.doc_lengths(el.get("stdDeviation"))


def _p_offset(el, prim, pctx, _i, default=0.0):
    dx = pctx.length(parse_number(el.get("dx"), default), "x")
    dy = pctx.length(parse_number(el.get("dy"), default), "y")
    prim["offset"] = transform_vector(pctx.ctm, dx, dy)


def _p_flood(_el, prim, pctx, _i):
    prim["color"] = _flood_color(pctx.style)


def _p_drop_shadow(el, prim, pctx, i):
    prim["std"] = pctx.doc_lengths(el.get("stdDeviation"), (2.0, 2.0))
    _p_offset(el, prim, pctx, i, default=2.0)
    prim["color"] = _flood_color(pctx.style)


def _p_color_matrix(el, prim, _pctx, _i):
    prim["mtype"] = (el.get("type") or "matrix").strip()
    prim["values"] = parse_numbers(el.get("values"))


def _p_component_transfer(el, prim, _pctx, _i):
    funcs = {}
    for f in el:
        if not isinstance(f.tag, str):
            continue
        kind = strip_ns(f.tag)
        if kind in ("feFuncR", "feFuncG", "feFuncB", "feFuncA"):
            funcs[kind[-1]] = {
                "type": (f.get("type") or "identity").strip(),
                "table": parse_numbers(f.get("tableValues")),
                "slope": parse_number(f.get("slope"), 1.0),
                "intercept": parse_number(f.get("intercept"), 0.0),
                "amplitude": parse_number(f.get("amplitude"), 1.0),
                "exponent": parse_number(f.get("exponent"), 1.0),
                "offset": parse_number(f.get("offset"), 0.0),
            }
    prim["funcs"] = funcs


def _p_composite(el, prim, _pctx, _i):
    prim["operator"] = (el.get("operator") or "over").strip()
    prim["k"] = tuple(parse_number(el.get(f"k{i}"), 0.0) for i in range(1, 5))


def _p_blend(el, prim, _pctx, _i):
    prim["mode"] = (el.get("mode") or "normal").strip()


def _p_merge(el, prim, _pctx, _i):
    prim["nodes"] = [
        n.get("in")
        for n in el
        if isinstance(n.tag, str) and strip_ns(n.tag) == "feMergeNode"
    ]


def _p_morphology(el, prim, pctx, _i):
    prim["operator"] = (el.get("operator") or "erode").strip()
    prim["radius"] = pctx.doc_lengths(el.get("radius"))


def _p_convolve(el, prim, _pctx, _i):
    ox, oy = (int(v) for v in parse_number_pair(el.get("order"), (3.0, 3.0)))
    prim["order"] = (ox, oy)
    prim["kernel"] = parse_numbers(el.get("kernelMatrix"))
    prim["divisor"] = parse_number(el.get("divisor"), 0.0)
    prim["bias"] = parse_number(el.get("bias"), 0.0)
    prim["target"] = (
        int(parse_number(el.get("targetX"), ox // 2)),
        int(parse_number(el.get("targetY"), oy // 2)),
    )
    prim["edge"] = (el.get("edgeMode") or "duplicate").strip()
    prim["preserve_alpha"] = (el.get("preserveAlpha") or "false").strip() == "true"


def _p_displacement(el, prim, pctx, _i):
    kx, _ky = axis_scales(pctx.ctm)
    prim["scale"] = pctx.length(parse_number(el.get("scale"), 0.0), "x") * kx
    prim["xch"] = (el.get("xChannelSelector") or "A").strip()
    prim["ych"] = (el.get("yChannelSelector") or "A").strip()


def _p_turbulence(el, prim, pctx, _i):
    kx, ky = axis_scales(pctx.ctm)
    fx, fy = parse_number_pair(el.get("baseFrequency"))
    prim["freq"] = (fx / max(kx, 1e-12), fy / max(ky, 1e-12))
    prim["octaves"] = int(parse_number(el.get("numOctaves"), 1))
    prim["seed"] = parse_number(el.get("seed"), 0.0)
    prim["fractal"] = (el.get("type") or "turbulence").strip() == "fractalNoise"
    prim["origin_doc"] = pctx.ctm


def _p_lighting(el, prim, pctx, _i):
    color = parse_color(pctx.style.get("lighting-color", "white"), default="white")
    prim["color"] = to_rgb(color if color != "none" else "black")
    prim["surface_scale"] = parse_number(el.get("surfaceScale"), 1.0)
    prim["kd"] = parse_number(el.get("diffuseConstant"), 1.0)
    prim["ks"] = parse_number(el.get("specularConstant"), 1.0)
    prim["exp"] = parse_number(el.get("specularExponent"), 1.0)
    prim["light"] = _light_source(el, pctx)


def _light_source(el: Element, pctx: _PrimitiveContext) -> dict:
    """The first light source child, positions in document units."""
    kx, _ky = axis_scales(pctx.ctm)

    def point(prefix_x, prefix_y, prefix_z):
        x = pctx.length(parse_number(el_light.get(prefix_x), 0.0), "x")
        y = pctx.length(parse_number(el_light.get(prefix_y), 0.0), "y")
        z = pctx.length(parse_number(el_light.get(prefix_z), 0.0), "x") * kx
        px, py = pctx.ctm.transform((x, y))
        return (float(px), float(py), z)

    for el_light in el:
        if not isinstance(el_light.tag, str):
            continue
        kind = strip_ns(el_light.tag)
        if kind == "feDistantLight":
            return {
                "type": "distant",
                "azimuth": parse_number(el_light.get("azimuth"), 0.0),
                "elevation": parse_number(el_light.get("elevation"), 0.0),
            }
        if kind == "fePointLight":
            return {"type": "point", "pos": point("x", "y", "z")}
        if kind == "feSpotLight":
            cone = el_light.get("limitingConeAngle")
            return {
                "type": "spot",
                "pos": point("x", "y", "z"),
                "at": point("pointsAtX", "pointsAtY", "pointsAtZ"),
                "exp": parse_number(el_light.get("specularExponent"), 1.0),
                "cone": parse_number(cone, 0.0) if cone else None,
            }
    return {"type": "distant", "azimuth": 0.0, "elevation": 0.0}


def _p_image(el, prim, pctx, index):
    ref = href(el)
    prim["image_index"] = index
    if not ref:
        return
    if ref.startswith("#"):
        target = pctx.parser.id_index.get(ref[1:])
        if target is not None:
            items: list = []
            state = replace(pctx.st, ctm=pctx.ctm, out=items, clip=None, opacity=1.0)
            pctx.parser.walk(target, state)
            pctx.filter.images[index] = ("items", items)
        return
    loaded = pctx.parser.load_image(ref)
    if isinstance(loaded, np.ndarray):
        sub = prim.get("subregion") or pctx.filter.region
        pctx.filter.images[index] = ("raster", loaded, sub)


_PRIMITIVE_PARSERS: dict[str, Callable] = {
    "feBlend": _p_blend,
    "feColorMatrix": _p_color_matrix,
    "feComponentTransfer": _p_component_transfer,
    "feComposite": _p_composite,
    "feConvolveMatrix": _p_convolve,
    "feDiffuseLighting": _p_lighting,
    "feDisplacementMap": _p_displacement,
    "feDropShadow": _p_drop_shadow,
    "feFlood": _p_flood,
    "feGaussianBlur": _p_blur,
    "feImage": _p_image,
    "feMerge": _p_merge,
    "feMorphology": _p_morphology,
    "feOffset": _p_offset,
    "feSpecularLighting": _p_lighting,
    "feTurbulence": _p_turbulence,
}


# --------------------------------------------------------------------------- #
# CSS filter functions
# --------------------------------------------------------------------------- #
_CSS_FUNC = re.compile(r"([a-z-]+)\(([^()]*(?:\([^()]*\)[^()]*)*)\)", re.IGNORECASE)
_TRAILING_COLOR = re.compile(
    r"(#[0-9a-fA-F]+|rgba?\([^)]*\)|hsla?\([^)]*\)|[a-zA-Z]+)\s*$"
)
_SEPIA = np.array([[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]])


def _amount(args: str, default: float = 1.0) -> float:
    if not args:
        return default
    return parse_fraction(args.split(maxsplit=1)[0], default)


def _css_color_primitive(name: str, a: float) -> dict:
    """The primitive equivalent to a CSS color filter function."""
    clamped = min(max(a, 0.0), 1.0)
    if name == "grayscale":
        return {"type": "feColorMatrix", "mtype": "saturate", "values": [1 - clamped]}
    if name == "saturate":
        return {"type": "feColorMatrix", "mtype": "saturate", "values": [a]}
    if name == "sepia":
        m = (1 - clamped) * np.eye(3) + clamped * _SEPIA
        rows = [[*m[r], 0.0, 0.0] for r in range(3)]
        values = [v for row in rows for v in row] + [0, 0, 0, 1, 0]
        return {"type": "feColorMatrix", "mtype": "matrix", "values": values}
    funcs: dict = {}
    if name == "invert":
        funcs = {ch: {"type": "table", "table": [a, 1 - a]} for ch in "RGB"}
    elif name == "opacity":
        funcs = {"A": {"type": "table", "table": [0, clamped]}}
    elif name == "brightness":
        funcs = {ch: {"type": "linear", "slope": a, "intercept": 0.0} for ch in "RGB"}
    elif name == "contrast":
        funcs = {
            ch: {"type": "linear", "slope": a, "intercept": 0.5 - 0.5 * a}
            for ch in "RGB"
        }
    return {"type": "feComponentTransfer", "funcs": funcs}


_CSS_COLOR_FUNCS = (
    "grayscale",
    "sepia",
    "saturate",
    "invert",
    "opacity",
    "brightness",
    "contrast",
)


def css_filter(
    value: str, bbox_user: Rect | None, ctm: Affine2D, ctx: LengthContext
) -> Filter | None:
    """Translate CSS filter functions into an equivalent primitive chain.

    Returns ``None`` for an empty list, an unknown function, or an element
    without geometry.
    """
    if bbox_user is None:
        return None
    kx, ky = axis_scales(ctm)
    prims: list[dict] = []
    margin = 0.0  # how far the effect can reach beyond the element
    for raw_name, raw_args in _CSS_FUNC.findall(value):
        name, args = raw_name.lower(), raw_args.strip()
        if name == "blur":
            s = parse_length(args or "0", 0.0, ctx=ctx)
            prims.append({"type": "feGaussianBlur", "std": (s * kx, s * ky)})
            margin = max(margin, 3 * s)
        elif name == "drop-shadow":
            color = "black"
            m = _TRAILING_COLOR.search(args)
            if m and not m.group(1)[0].isdigit():
                color, args = m.group(1), args[: m.start()]
            nums = [parse_length(t, 0.0, ctx=ctx) for t in args.split()] + [0.0] * 3
            dx, dy, blur = nums[:3]
            parsed = parse_color(color, default="black")
            prims.append(
                {
                    "type": "feDropShadow",
                    "offset": transform_vector(ctm, dx, dy),
                    "std": (blur / 2 * kx, blur / 2 * ky),
                    "color": to_rgba(parsed if parsed != "none" else (0, 0, 0, 0)),
                }
            )
            margin = max(margin, abs(dx) + abs(dy) + 3 * blur)
        elif name == "hue-rotate":
            deg = (parse_angles(args) or [0.0])[0]
            prims.append(
                {"type": "feColorMatrix", "mtype": "hueRotate", "values": [deg]}
            )
        elif name in _CSS_COLOR_FUNCS:
            prims.append(_css_color_primitive(name, _amount(args)))
        else:
            return None
    if not prims:
        return None
    x0, y0, x1, y1 = bbox_user
    pad_x = 0.1 * (x1 - x0) + margin
    pad_y = 0.1 * (y1 - y0) + margin
    region = transform_rect((x0 - pad_x, y0 - pad_y, x1 + pad_x, y1 + pad_y), ctm)
    # CSS filter functions operate in sRGB.
    return Filter(primitives=prims, region=region, color_space="sRGB")
