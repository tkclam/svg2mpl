"""Place ``marker-start``/``marker-mid``/``marker-end`` on path vertices."""

from __future__ import annotations

import contextlib
from dataclasses import replace
from typing import TYPE_CHECKING

from matplotlib.transforms import Affine2D

from ._state import State
from .geometry import intersect_clip, rect_path
from .pathdata import ParsedPath, marker_vertices
from .style import opacity
from .transform import compose, parse_angles, parse_viewbox, viewbox_transform
from .units import LengthContext, parse_length

if TYPE_CHECKING:
    from xml.etree.ElementTree import Element

    from .parser import Parser

__all__ = ["MARKER_PROPERTIES", "emit_markers", "has_markers"]

MARKER_PROPERTIES = ("marker-start", "marker-mid", "marker-end")
_ROLE_PROPERTIES = {
    "start": ("marker-start",),
    "mid": ("marker-mid",),
    "end": ("marker-end",),
    "both": ("marker-start", "marker-end"),
}
_REF_KEYWORDS = {"left": 0.0, "top": 0.0, "center": 0.5, "right": 1.0, "bottom": 1.0}


def has_markers(style: dict) -> bool:
    """Whether a style references any marker."""
    return any(
        style.get(p, "none").strip().lower() not in ("none", "")
        for p in MARKER_PROPERTIES
    )


def emit_markers(
    parser: Parser, parsed: ParsedPath, style: dict, ctm: Affine2D, st: State
) -> None:
    """Instantiate the element's markers at its path vertices into ``st.out``."""
    stroke_width = parse_length(
        style.get("stroke-width"), 1.0, ctx=parser.ctx(style, st)
    )
    # Markers are painted as part of their element, so they share its opacity.
    mst = replace(st, opacity=opacity(style.get("opacity")) * st.opacity)
    for x, y, angle, role in marker_vertices(parsed):
        for prop in _ROLE_PROPERTIES[role]:
            marker = parser.ref(style.get(prop), ("marker",))
            if marker is None or id(marker) in st.chain:
                continue
            is_start = prop == "marker-start"
            _place(parser, marker, (x, y), angle, is_start, stroke_width, ctm, mst)


def _orientation(marker: Element, auto_angle: float, is_start: bool) -> float:
    orient = (marker.get("orient") or "0").strip()
    if orient == "auto":
        return auto_angle
    if orient == "auto-start-reverse":
        return auto_angle + 180.0 if is_start else auto_angle
    angles = parse_angles(orient)
    return angles[0] if angles else 0.0


def _ref_coord(value, vb, axis: int, size: float, ctx: LengthContext) -> float:
    """``refX``/``refY``: a length in viewBox units or a keyword (SVG 2)."""
    if value is None:
        return 0.0
    v = value.strip().lower()
    if v in _REF_KEYWORDS:
        origin = vb[axis] if vb else 0.0
        span = vb[2 + axis] if vb else size
        return origin + _REF_KEYWORDS[v] * span
    return parse_length(v, 0.0, axis="xy"[axis], ctx=ctx)


def _place(
    parser: Parser,
    marker: Element,
    vertex: tuple[float, float],
    auto_angle: float,
    is_start: bool,
    stroke_width: float,
    ctm: Affine2D,
    st: State,
) -> None:
    mstyle = parser.style_of(marker)
    ctx = parser.ctx(mstyle, st)
    mw = parse_length(marker.get("markerWidth") or "3", 3.0, axis="x", ctx=ctx)
    mh = parse_length(marker.get("markerHeight") or "3", 3.0, axis="y", ctx=ctx)
    if mw <= 0 or mh <= 0:
        return
    units = marker.get("markerUnits") or "strokeWidth"
    scale = stroke_width if units == "strokeWidth" else 1.0
    vb = parse_viewbox(marker.get("viewBox"))
    view = (
        viewbox_transform(vb, mw, mh, marker.get("preserveAspectRatio"))
        if vb
        else Affine2D()
    )
    ref = view.transform(
        (
            _ref_coord(marker.get("refX"), vb, 0, mw, ctx),
            _ref_coord(marker.get("refY"), vb, 1, mh, ctx),
        )
    )
    # Viewport space (after viewBox) -> the shape's user space -> document.
    placement = (
        Affine2D()
        .translate(-ref[0], -ref[1])
        .scale(scale)
        .rotate_deg(_orientation(marker, auto_angle, is_start))
        .translate(*vertex)
    )
    outer = compose(placement, ctm)
    clip = st.clip
    overflow = (mstyle.get("overflow") or "hidden").strip().lower()
    if overflow not in ("visible", "auto"):
        with contextlib.suppress(ValueError):
            clip = intersect_clip(clip, rect_path(0, 0, mw, mh).transformed(outer))
    marker_state = replace(
        st,
        ctm=compose(view, outer),
        style=mstyle,
        clip=clip,
        vp=(vb[2], vb[3]) if vb else (mw, mh),
        chain=st.chain | {id(marker)},
    )
    for child in marker:
        parser.walk(child, marker_state)
