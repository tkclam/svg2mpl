"""Resolve ``<clipPath>`` and ``<mask>`` references.

Clip regions become a single document-space path using the nonzero rule (see
:mod:`svg2mpl.geometry` for how ``clip-rule`` and nested clips are folded in).
Masks become a :class:`~svg2mpl.scene.Mask` whose content is composited
off-screen by the renderer.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING

from matplotlib.path import Path
from matplotlib.transforms import Affine2D

from ._state import State
from ._xml import href, is_svg_element, strip_ns
from .color import NONE
from .geometry import (
    Rect,
    empty_path,
    evenodd_to_nonzero,
    intersect_clip,
    join_paths,
    normalize_orientation,
    path_extents,
    rect_intersection,
    rect_is_empty,
    rect_path,
    transform_rect,
)
from .scene import Mask, Shape
from .shapes import SHAPE_TAGS, shape_to_parsed
from .style import is_hidden, is_invisible
from .units import parse_length, resolve_region

if TYPE_CHECKING:
    from xml.etree.ElementTree import Element

    from .parser import Parser

__all__ = ["any_overlap", "build_mask", "clip_region", "painted_extents"]


def clip_region(
    parser: Parser,
    cp_el: Element,
    bbox_fn: Callable[[], Rect | None],
    ctm: Affine2D,
    st: State,
    seen: frozenset = frozenset(),
) -> Path | None:
    """A ``<clipPath>``'s region in document space (nonzero rule).

    ``bbox_fn`` lazily computes the referencing element's user-space bounding
    box, needed only for ``clipPathUnits="objectBoundingBox"``. An empty path
    means everything is clipped away.
    """
    if id(cp_el) in seen:
        return None
    cp_style = parser.style_of(cp_el)
    base = parser.transform_of(cp_el)
    if (cp_el.get("clipPathUnits") or "userSpaceOnUse") == "objectBoundingBox":
        bb = bbox_fn()
        if bb is None or rect_is_empty(bb):
            return empty_path()
        unit_box = (
            Affine2D().scale(bb[2] - bb[0], bb[3] - bb[1]).translate(bb[0], bb[1])
        )
        base = base + unit_box
    total = base + ctm
    pieces = []
    for child in cp_el:
        geom = _clip_child_geometry(parser, child, cp_style, total, st)
        if geom is not None and len(geom.vertices):
            # Orient every piece the same way so the nonzero union is a union.
            pieces.append(normalize_orientation(geom))
    region = join_paths(pieces)
    nested = parser.ref(cp_style.get("clip-path"), ("clipPath",))
    if nested is not None:
        other = clip_region(parser, nested, bbox_fn, ctm, st, seen | {id(cp_el)})
        with contextlib.suppress(ValueError):  # two non-convex clips: keep ours
            region = intersect_clip(region, other)
    return region


def _clip_child_geometry(
    parser: Parser, child: Element, cp_style: dict, total: Affine2D, st: State
) -> Path | None:
    """Document-space geometry of one ``<clipPath>`` child (or ``None``)."""
    if not is_svg_element(child) or not parser.conditions_pass(child):
        return None
    tag = strip_ns(child.tag)
    target = child
    style = parser.child_style(child, cp_style)
    to_doc = parser.transform_of(child) + total
    if tag == "use":
        # A clipPath may only <use> shapes and text directly.
        ref = parser.ref(href(child))
        if ref is None:
            return None
        ctx = parser.ctx(style, st)
        offset = Affine2D().translate(
            parse_length(child.get("x"), 0.0, axis="x", ctx=ctx),
            parse_length(child.get("y"), 0.0, axis="y", ctx=ctx),
        )
        target, style = ref, parser.child_style(ref, style)
        tag = strip_ns(target.tag)
        to_doc = parser.transform_of(target) + offset + to_doc
    if (tag not in SHAPE_TAGS and tag != "text") or is_hidden(style):
        return None
    if is_invisible(style):
        return None
    if tag == "text":
        measured = replace(
            st, style=cp_style, ctm=to_doc, clip=None, out=[], measuring=True
        )
        parser.text(target, style, to_doc, measured)
        return join_paths(item.path for item in measured.out)
    parsed = shape_to_parsed(tag, target.attrib, parser.ctx(style, st))
    if parsed is None or len(parsed.path.vertices) == 0:
        return None
    geom = parsed.path
    if (style.get("clip-rule") or "nonzero").strip() == "evenodd":
        geom = evenodd_to_nonzero(geom)
    return geom.transformed(to_doc)


def build_mask(
    parser: Parser, m_el: Element, bbox_user: Rect | None, ctm: Affine2D, st: State
) -> Mask | None:
    """Resolve a ``<mask>`` for one referencing element (``None`` hides it)."""
    if id(m_el) in st.chain:
        return None
    mstyle = parser.style_of(m_el)
    units = m_el.get("maskUnits") or "objectBoundingBox"
    rect = resolve_region(m_el.get, units, bbox_user, parser.ctx(mstyle, st))
    if rect is None:
        return None
    content_ctm = ctm
    if (m_el.get("maskContentUnits") or "userSpaceOnUse") == "objectBoundingBox":
        if bbox_user is None:
            return None
        bx0, by0, bx1, by1 = bbox_user
        content_ctm = Affine2D().scale(bx1 - bx0, by1 - by0).translate(bx0, by0) + ctm
    items: list = []
    mask_state = replace(
        st,
        ctm=content_ctm,
        style=mstyle,
        out=items,
        opacity=1.0,
        clip=rect_path(*rect).transformed(ctm),
        chain=st.chain | {id(m_el)},
    )
    for child in m_el:
        parser.walk(child, mask_state)
    kind = (mstyle.get("mask-type") or "luminance").strip().lower()
    return Mask(
        items=items,
        region=transform_rect(rect, ctm),
        kind=kind if kind in ("alpha", "luminance") else "luminance",
    )


def painted_extents(item) -> Rect | None:
    """Document-space extents of a drawable, padded for its stroke and clipped."""
    ext = path_extents(item.path)
    if ext is None:
        return None
    if isinstance(item, Shape) and item.kwargs.get("edgecolor", NONE) != NONE:
        # Generous padding: miter joins can reach past half the stroke width.
        pad = (
            item.kwargs.get("linewidth", 0.0) / 2 * (1.5 if item.scale_stroke else 1.0)
        )
        ext = (ext[0] - pad, ext[1] - pad, ext[2] + pad, ext[3] + pad)
    clip = getattr(item, "clip", None)
    if clip is not None:
        clip_ext = path_extents(clip)
        if clip_ext is None:
            return None
        ext = rect_intersection(ext, clip_ext)
        if rect_is_empty(ext):
            return None
    return ext


def any_overlap(boxes: list[Rect]) -> bool:
    """Whether any two rectangles overlap (a sweep over x)."""
    active: list[Rect] = []
    for b in sorted(boxes):
        active = [a for a in active if a[2] > b[0]]
        if any(a[1] < b[3] and b[1] < a[3] for a in active):
            return True
        active.append(b)
    return False
