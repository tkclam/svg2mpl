"""Turn an element's fill and stroke into :class:`~svg2mpl.scene.Shape` items.

This resolves ``url(#id) [fallback]`` paints to gradients or pattern tiles,
folds the opacities into colors, converts stroke widths and dashes from user
units to document units, applies the fill rule and splits shapes according to
``paint-order``.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

from matplotlib.path import Path
from matplotlib.transforms import Affine2D

from ._state import State
from .color import NONE, parse_color, split_paint, with_alpha
from .geometry import Rect, area_scale, evenodd_to_nonzero, rect_is_empty
from .gradients import Gradient, Pattern
from .scene import BoundPaint, PatternTile, Shape, Text
from .style import opacity, style_to_kwargs
from .transform import viewbox_transform
from .units import parse_fraction, parse_length

if TYPE_CHECKING:
    from .parser import Parser

__all__ = ["PAINT_ORDER", "bind_paint", "has_fill", "paint_order", "paint_shape"]

PAINT_ORDER = ("fill", "stroke", "markers")


def paint_order(value: str | None) -> list[str]:
    """Parse ``paint-order`` into the full order of fill, stroke and markers."""
    if not value or value.strip().lower() == "normal":
        return list(PAINT_ORDER)
    given = [t for t in value.split() if t in PAINT_ORDER]
    return given + [t for t in PAINT_ORDER if t not in given]


def _is_url(value: Any) -> bool:
    return isinstance(value, str) and value.lower().startswith("url(")


def bind_paint(
    parser: Parser,
    value: str,
    ctm: Affine2D,
    bbox_user: Rect | None,
    paint_opacity: float,
    st: State,
    current_color: Any,
) -> tuple[Any, BoundPaint | None]:
    """Resolve a ``url(#id) [fallback]`` paint into ``(color, BoundPaint|None)``.

    The color is the solid color to use (``"none"`` when the paint server
    draws the fill itself); a missing or unusable server falls back to the
    fallback color, as the spec requires.
    """
    ref, fallback = split_paint(value)
    server = parser.servers.get(ref) if ref else None

    def fallback_color():
        if not fallback:
            return NONE
        color = parse_color(fallback, current_color=current_color, default=NONE)
        return with_alpha(color, paint_opacity)

    if isinstance(server, Gradient):
        if not server.stops:
            return NONE, None
        if server.units == "objectBoundingBox" and rect_is_empty(bbox_user):
            return fallback_color(), None
        if len(server.stops) == 1:
            return with_alpha(server.stops[0][1], paint_opacity), None
        return NONE, BoundPaint(server, ctm, bbox_user, paint_opacity)
    if isinstance(server, Pattern):
        bound = _bind_pattern(parser, server, bbox_user, ctm, paint_opacity, st)
        return (NONE, bound) if bound is not None else (fallback_color(), None)
    return fallback_color(), None


def _bind_pattern(
    parser: Parser,
    pat: Pattern,
    bbox_user: Rect | None,
    ctm: Affine2D,
    paint_opacity: float,
    st: State,
) -> BoundPaint | None:
    """Resolve a pattern's tile for one referencing element."""
    if id(pat.element) in st.chain:
        return None
    pstyle = parser.style_of(pat.element)
    if pat.units == "objectBoundingBox":
        if bbox_user is None:
            return None
        bx0, by0, bx1, by1 = bbox_user
        bw, bh = bx1 - bx0, by1 - by0
        x = bx0 + parse_fraction(pat.x) * bw
        y = by0 + parse_fraction(pat.y) * bh
        w, h = parse_fraction(pat.width) * bw, parse_fraction(pat.height) * bh
    else:
        ctx = parser.ctx(pstyle, st)
        x = parse_length(pat.x, 0.0, axis="x", ctx=ctx)
        y = parse_length(pat.y, 0.0, axis="y", ctx=ctx)
        w = parse_length(pat.width, 0.0, axis="x", ctx=ctx)
        h = parse_length(pat.height, 0.0, axis="y", ctx=ctx)
    if w <= 0 or h <= 0:
        return None
    if pat.viewbox is not None:
        content = viewbox_transform(pat.viewbox, w, h, pat.preserve_aspect_ratio)
    elif pat.content_units == "objectBoundingBox" and bbox_user is not None:
        content = Affine2D().scale(
            bbox_user[2] - bbox_user[0], bbox_user[3] - bbox_user[1]
        )
    else:
        content = Affine2D()
    items: list = []
    tile_state = State(
        ctm=content,
        style=pstyle,
        out=items,
        vp=(pat.viewbox[2], pat.viewbox[3]) if pat.viewbox else st.vp,
        chain=st.chain | {id(pat.element)},
        depth=st.depth + 1,
    )
    for child in pat.element:
        parser.walk(child, tile_state)
    tile_to_doc = Affine2D().translate(x, y) + pat.transform + ctm
    tile = PatternTile(items=items, width=w, height=h, transform=tile_to_doc)
    return BoundPaint(pat, ctm, bbox_user, paint_opacity, tile=tile)


def paint_shape(
    parser: Parser,
    el_id: str,
    path_user: Path,
    style: dict,
    ctm: Affine2D,
    st: State,
    bbox_user: Rect | None,
    text: Text | None = None,
) -> tuple[list[Shape], list[str]]:
    """Build the shapes painting ``path_user`` (in the element's user space).

    Returns the shapes (two when ``paint-order`` puts the stroke under the
    fill) and the element's paint order, which also places its markers.
    """
    ctx = parser.ctx(style, st)
    kwargs = style_to_kwargs(style, ctx, st.opacity)
    current = parse_color(style.get("color"), default="black")
    if current == NONE:
        current = "black"
    element_opacity = opacity(style.get("opacity")) * st.opacity

    fill_paint = stroke_paint = None
    fill_url, stroke_url = _is_url(kwargs["facecolor"]), _is_url(kwargs["edgecolor"])
    if fill_url or stroke_url:
        # Resolved paints carry their own opacity; fold any patch alpha into
        # the other paint so a single alpha does not clobber them.
        alpha = kwargs.pop("alpha", None)
        if alpha is not None:
            for key in ("facecolor", "edgecolor"):
                kwargs[key] = with_alpha(kwargs[key], alpha)
    if fill_url:
        pop = element_opacity * opacity(style.get("fill-opacity"))
        kwargs["facecolor"], fill_paint = bind_paint(
            parser, kwargs["facecolor"], ctm, bbox_user, pop, st, current
        )
    if stroke_url:
        pop = element_opacity * opacity(style.get("stroke-opacity"))
        color, stroke_paint = bind_paint(
            parser, kwargs["edgecolor"], ctm, bbox_user, pop, st, current
        )
        if stroke_paint is not None:
            # Fallback solid color; the renderer draws gradient strokes itself.
            color = with_alpha(stroke_paint.server.average_color(), pop)
        kwargs["edgecolor"] = color

    # Stroke geometry: user units -> document units.
    vector_effect = (style.get("vector-effect") or "").strip().lower()
    non_scaling = vector_effect == "non-scaling-stroke"
    k = 1.0 if non_scaling else area_scale(ctm)
    kwargs["linewidth"] = kwargs.get("linewidth", 1.0) * k
    if "linestyle" in kwargs:
        offset, dashes = kwargs["linestyle"]
        kwargs["linestyle"] = (offset * k, [d * k for d in dashes])

    if (style.get("fill-rule") or "nonzero").strip().lower() == "evenodd":
        path_user = evenodd_to_nonzero(path_user)
    shape = Shape(
        id=el_id,
        path=path_user.transformed(ctm),
        style=dict(style),
        kwargs=kwargs,
        clip=st.clip,
        fill_paint=fill_paint,
        stroke_paint=stroke_paint,
        scale_stroke=not non_scaling,
        text=text,
    )

    order = paint_order(style.get("paint-order"))
    has_fill = kwargs["facecolor"] != NONE or fill_paint is not None
    has_stroke = kwargs["edgecolor"] != NONE or stroke_paint is not None
    if has_fill and has_stroke and order.index("stroke") < order.index("fill"):
        stroke_only = replace(
            shape, kwargs={**kwargs, "facecolor": NONE}, fill_paint=None
        )
        fill_only = replace(
            shape, kwargs={**kwargs, "edgecolor": NONE}, stroke_paint=None
        )
        return [stroke_only, fill_only], order
    return [shape], order


def has_fill(shape: Shape) -> bool:
    """Whether a shape paints its interior (solid color or paint server)."""
    return shape.kwargs.get("facecolor") != NONE or shape.fill_paint is not None
