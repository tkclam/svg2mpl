"""Traverse an SVG document into a :class:`~svg2mpl.scene.Scene`.

The walk maintains, per element, the current transformation matrix (CTM, user
space -> document space), the inherited computed style, the accumulated group
opacity, the active clip region and the current viewport. It descends through
containers (``g``/``a``/``switch``/nested ``svg``), instantiates ``<use>``
references (including ``<symbol>`` viewports), places markers, resolves paint
servers, lays out text and decodes images. Elements that need off-screen
compositing (``mask``/``filter``) become :class:`~svg2mpl.scene.Group` items.

Non-rendered subtrees (``defs``, ``symbol``, ``clipPath``, gradients, metadata,
animation elements, ``foreignObject`` ...) are skipped during the normal walk.
"""

from __future__ import annotations

import base64
import io
import math
import os
import re
import urllib.parse
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path as FilePath
from xml.etree import ElementTree

import numpy as np
from matplotlib.path import Path
from matplotlib.transforms import Affine2D

from .color import NONE, parse_color, parse_url_ref, split_paint
from .css import compute_css_declarations
from .geometry import (
    evenodd_to_nonzero,
    intersect_clip,
    join_paths,
    normalize_orientation,
    path_extents,
    rect_path,
    reverse_path,
)
from .gradients import Gradient, Pattern, parse_gradients
from .pathdata import parse_path_data
from .scene import BoundPaint, Group, Image, Mask, Scene, Shape, Text
from .shapes import MARKABLE_TAGS, SHAPE_TAGS, shape_to_parsed, strip_ns
from .style import (
    compute_style,
    declarations,
    font_size,
    is_hidden,
    is_invisible,
    opacity,
    parse_style_attr,
    style_to_kwargs,
)
from .transform import (
    parse_preserve_aspect_ratio,
    parse_transform,
    parse_viewbox,
    viewbox_transform,
)
from .units import LengthContext, parse_length, split_length

__all__ = ["parse", "parse_string"]

SVG_NS = "http://www.w3.org/2000/svg"
_XLINK = "{http://www.w3.org/1999/xlink}href"
_CONTAINER_TAGS = frozenset({"g", "a"})
# Subtrees that define resources or metadata rather than drawing anything.
_SKIP_TAGS = frozenset(
    {
        "defs",
        "symbol",
        "clipPath",
        "mask",
        "marker",
        "pattern",
        "linearGradient",
        "radialGradient",
        "style",
        "title",
        "desc",
        "metadata",
        "filter",
        "script",
        "view",
        "cursor",
        "font",
        "font-face",
        "glyph",
        "animate",
        "animateMotion",
        "animateTransform",
        "animateColor",
        "set",
        "mpath",
        "foreignObject",
        "solidcolor",
        "hatch",
        "meshgradient",
    }
)
_MAX_DEPTH = 64
_SUPPORTED_LANGUAGES = ("en",)


def _length(value, default=0.0, axis="other", ctx=None):
    return parse_length(value, default, axis=axis, ctx=ctx)


def parse(source: str | FilePath) -> Scene:
    """Parse an SVG file path into a :class:`Scene`."""
    root = ElementTree.parse(str(source)).getroot()
    return _Parser(root, base_dir=os.path.dirname(os.path.abspath(str(source)))).build()


def parse_string(text: str, base_dir: str | None = None) -> Scene:
    """Parse an SVG document from a string into a :class:`Scene`."""
    return _Parser(ElementTree.fromstring(text), base_dir=base_dir).build()


@dataclass
class _State:
    """Per-element traversal state (copied, never mutated, when descending)."""

    ctm: Affine2D
    style: dict | None
    out: list
    vp: tuple[float, float]
    opacity: float = 1.0
    clip: Path | None = None
    depth: int = 0
    chain: frozenset = frozenset()
    measuring: bool = False


@dataclass
class PatternTile:
    """A pattern's tile content, resolved for one referencing element."""

    items: list
    width: float
    height: float
    transform: Affine2D  # tile space -> document space
    extras: dict = field(default_factory=dict)


def _is_svg_tag(el) -> bool:
    if not isinstance(el.tag, str):
        return False
    if el.tag.startswith("{"):
        return el.tag[1:].split("}", 1)[0] == SVG_NS
    return True


def _linear_scale(affine: Affine2D) -> float:
    m = affine.get_matrix()
    return math.sqrt(abs(m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0]))


def _transformed_bbox(rect, affine: Affine2D):
    x0, y0, x1, y1 = rect
    pts = affine.transform([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
    return (pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max())


def _union(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _href(el):
    return el.get(_XLINK) or el.get("href")


class _Parser:
    def __init__(self, root, base_dir: str | None = None, depth: int = 0):
        self.root = root
        self.base_dir = base_dir
        self.nest_depth = depth
        self.css = compute_css_declarations(root)
        self.id_index = {}
        for e in root.iter():
            if (
                isinstance(e.tag, str)
                and e.get("id")
                and e.get("id") not in self.id_index
            ):
                self.id_index[e.get("id")] = e
        self.parent_of = {id(c): p for p in root.iter() for c in p}
        self._style_cache: dict[int, dict] = {}
        self.warned: set[str] = set()

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def warn(self, key: str, msg: str) -> None:
        if key not in self.warned:
            self.warned.add(key)
            warnings.warn(msg, stacklevel=4)

    def own_decls(self, el) -> dict:
        return declarations(el.attrib, self.css.get(id(el)))

    def child_style(self, el, parent_style):
        return compute_style(self.own_decls(el), parent_style)

    def style_of(self, el) -> dict:
        """The computed style of ``el`` in its document context."""
        key = id(el)
        if key not in self._style_cache:
            parent = self.parent_of.get(key)
            ps = self.style_of(parent) if parent is not None else None
            self._style_cache[key] = compute_style(self.own_decls(el), ps)
        return self._style_cache[key]

    def transform_of(self, el) -> Affine2D:
        """The element's own transform (CSS ``transform`` beats the attribute)."""
        inline = parse_style_attr(el.get("style")).get("transform")
        css = (self.css.get(id(el)) or {}).get("transform")
        value = inline or css or el.get("transform")
        if value and value.strip().lower() == "none":
            return Affine2D()
        return parse_transform(value)

    def ctx(self, style, st: _State) -> LengthContext:
        return LengthContext(st.vp[0], st.vp[1], font_size(style))

    def ref(self, value, tags=None):
        """Resolve ``url(#id)`` or ``#id`` to an element (optionally by tag)."""
        if not value:
            return None
        value = value.strip()
        ref = value[1:] if value.startswith("#") else parse_url_ref(value)
        el = self.id_index.get(ref) if ref else None
        if el is None or (tags and strip_ns(el.tag) not in tags):
            return None
        return el

    @staticmethod
    def conditions_pass(el) -> bool:
        ext = el.get("requiredExtensions")
        if ext is not None and ext.strip():
            return False
        lang = el.get("systemLanguage")
        if lang is not None:
            langs = [x.strip().lower() for x in lang.split(",") if x.strip()]
            return any(
                x == s or x.startswith(s + "-")
                for x in langs
                for s in _SUPPORTED_LANGUAGES
            )
        return True

    # ------------------------------------------------------------------ #
    # Document
    # ------------------------------------------------------------------ #
    def document_viewport(self):
        root = self.root
        vb = parse_viewbox(root.get("viewBox"))

        def absolute(name):
            v = root.get(name)
            parts = split_length(v) if v else None
            if parts is None or parts[1] == "%" or parts[0] <= 0:
                return None
            return _length(v)

        w, h = absolute("width"), absolute("height")
        if vb is not None:
            if w and not h:
                h = w * vb[3] / vb[2] if vb[2] else None
            elif h and not w:
                w = h * vb[2] / vb[3] if vb[3] else None
            if not (w and h):
                w, h = vb[2], vb[3]
        return vb, w or 0.0, h or 0.0

    def build(self, outer_ctm=None, out=None, opacity_=1.0, clip=None) -> Scene:
        vb, w, h = self.document_viewport()
        root_ctm = viewbox_transform(vb, w, h, self.root.get("preserveAspectRatio"))
        if outer_ctm is not None:
            root_ctm = root_ctm + outer_ctm
        vp = (vb[2], vb[3]) if vb else (w or 100.0, h or 100.0)
        style_ctx = LengthContext(vp[0], vp[1])
        self.servers = parse_gradients(self.root, style_of=self.style_of, ctx=style_ctx)
        self.scene = Scene(width=w, height=h, gradients=self.servers)
        if out is not None:
            self.scene.items = out
        st = _State(
            ctm=root_ctm,
            style=None,
            out=self.scene.items,
            vp=vp,
            opacity=opacity_,
            clip=clip,
        )
        root = self.root
        style = compute_style(self.own_decls(root), None)
        if is_hidden(style):
            return self.scene
        ctm = self.transform_of(root) + st.ctm
        self.with_effects(root, style, ctm, st, lambda s: self.children(root, s, style))
        return self.scene

    # ------------------------------------------------------------------ #
    # Traversal
    # ------------------------------------------------------------------ #
    def children(self, el, st: _State, style) -> None:
        inner = replace(st, style=style)
        for child in el:
            self.walk(child, inner)

    def walk(self, el, st: _State) -> None:
        if not _is_svg_tag(el) or st.depth > _MAX_DEPTH:
            return
        tag = strip_ns(el.tag)
        if tag in _SKIP_TAGS or not self.conditions_pass(el):
            return
        style = compute_style(self.own_decls(el), st.style)
        if is_hidden(style):
            return
        ctm = self.transform_of(el) + st.ctm
        self.render(el, tag, style, ctm, replace(st, depth=st.depth + 1))

    def render(self, el, tag, style, ctm, st: _State) -> None:
        if tag in _CONTAINER_TAGS:
            self.with_effects(el, style, ctm, st, lambda s: self.children(el, s, style))
        elif tag == "switch":
            self.with_effects(el, style, ctm, st, lambda s: self.switch(el, s, style))
        elif tag == "svg":
            self.nested_svg(el, style, ctm, st)
        elif tag == "use":
            self.use(el, style, ctm, st)
        elif tag in SHAPE_TAGS:
            self.with_effects(
                el,
                style,
                ctm,
                st,
                lambda s: self.shape(el, tag, style, ctm, s),
                leaf=True,
            )
        elif tag == "text":
            self.with_effects(
                el, style, ctm, st, lambda s: self.text(el, style, ctm, s), leaf=True
            )
        elif tag == "image":
            self.with_effects(
                el, style, ctm, st, lambda s: self.image(el, style, ctm, s), leaf=True
            )

    def switch(self, el, st: _State, style) -> None:
        inner = replace(st, style=style)
        for child in el:
            if not _is_svg_tag(child):
                continue
            tag = strip_ns(child.tag)
            if tag in _SKIP_TAGS or not self.conditions_pass(child):
                continue
            self.walk(child, inner)
            return

    # ------------------------------------------------------------------ #
    # Group effects: opacity, clip-path, mask, filter
    # ------------------------------------------------------------------ #
    def with_effects(self, el, style, ctm, st: _State, body, leaf=False) -> None:
        st = replace(st, ctm=ctm)
        if st.measuring:
            body(st)
            return
        op = opacity(style.get("opacity"))
        clip = st.clip
        cp = style.get("clip-path")
        if cp and cp.strip().lower() != "none":
            cp_el = self.ref(cp, ("clipPath",))
            if cp_el is not None:
                region = self.clip_region(
                    cp_el, lambda: self.bbox(el, style, st), ctm, st
                )
                if region is not None:
                    try:
                        clip = intersect_clip(st.clip, region)
                    except ValueError:
                        # Two non-convex clips: composite off-screen, clip twice.
                        self.raster_group(
                            el, style, ctm, st, body, leaf, outer=st.clip, inner=region
                        )
                        return
                if clip is not None and len(clip.vertices) == 0:
                    return  # clipped away entirely
        mask_el = self.ref(style.get("mask"), ("mask",))
        flt = style.get("filter")
        flt = None if not flt or flt.strip().lower() == "none" else flt
        if mask_el is not None or flt is not None:
            self.effect_group(
                el, style, ctm, replace(st, clip=clip), body, leaf, mask_el, flt, op
            )
            return
        if leaf:
            body(replace(st, clip=clip))
        elif op < 1.0:
            self.group_opacity(el, replace(st, clip=clip), body, op)
        else:
            body(replace(st, clip=clip, opacity=st.opacity * op))

    def group_opacity(self, el, st: _State, body, op) -> None:
        """Apply a container's ``opacity``.

        Multiplying the opacity into each child is exact when the children do
        not overlap; otherwise the group is composited off-screen first, as a
        browser does, so overlaps do not show through each other.
        """
        n_texts = len(self.scene.texts)
        tmp: list = []
        body(replace(st, opacity=st.opacity * op, out=tmp))
        boxes = [b for b in (_painted_extents(i) for i in tmp) if b is not None]
        if not _any_overlap(boxes):
            st.out.extend(tmp)
            return
        del self.scene.texts[n_texts:]
        region = None
        for b in boxes:
            region = _union(region, b)
        if st.clip is not None:
            ce = path_extents(st.clip)
            if ce is None:
                return
            region = _intersect_rect(region, ce)
        if region[2] <= region[0] or region[3] <= region[1]:
            return
        group = Group(
            id=el.get("id", ""),
            items=[],
            region=region,
            clip=st.clip,
            opacity=st.opacity * op,
        )
        st.out.append(group)
        body(replace(st, opacity=1.0, out=group.items, clip=None))

    def raster_group(self, el, style, ctm, st, body, leaf, outer, inner):
        bb = self.bbox(el, style, st)
        if bb is None:
            return
        region = path_extents(inner)
        if region is None:
            return
        group = Group(
            id=el.get("id", ""), items=[], region=region, clip=outer, opacity=1.0
        )
        st.out.append(group)
        inner_state = replace(st, out=group.items, clip=inner)
        if leaf:
            body(inner_state)
        else:
            body(
                replace(inner_state, opacity=st.opacity * opacity(style.get("opacity")))
            )

    def effect_group(self, el, style, ctm, st, body, leaf, mask_el, flt, op):
        from .filters import css_filter, parse_filter

        bbox_user = self.bbox(el, style, st)
        region = None
        mask = None
        filt = None
        if flt is not None:
            f_el = self.ref(flt, ("filter",))
            if f_el is not None:
                filt = parse_filter(self, f_el, bbox_user, ctm, st)
            elif "url(" not in flt:
                filt = css_filter(flt, bbox_user, ctm, self.ctx(style, st))
            if filt is None and "url(" in flt:
                return  # a reference to a missing/invalid filter disables rendering
            if filt is not None:
                region = filt.region
        if mask_el is not None:
            mask = self.mask(mask_el, bbox_user, ctm, st)
            if mask is None:
                return
            region = (
                mask.region if region is None else _intersect_rect(region, mask.region)
            )
        if region is None:
            if bbox_user is None:
                return
            region = _transformed_bbox(bbox_user, ctm)
        if region is None or region[2] <= region[0] or region[3] <= region[1]:
            return
        group = Group(
            id=el.get("id", ""),
            items=[],
            region=region,
            clip=st.clip,
            mask=mask,
            filter=filt,
            opacity=st.opacity * op,
        )
        st.out.append(group)
        inner = replace(st, out=group.items, clip=None, opacity=1.0)
        if leaf:
            # The element's own opacity is applied to the composited group; a
            # leaf multiplies its own opacity into its colors, so cancel it out.
            body(replace(inner, opacity=1.0 / op if op > 0 else 1.0))
        else:
            body(inner)

    def bbox(self, el, style, st: _State):
        """The element's geometry bounding box in its own user space."""
        tmp = replace(
            st, out=[], measuring=True, clip=None, ctm=Affine2D(), opacity=1.0
        )
        self.render(el, strip_ns(el.tag), style, Affine2D(), tmp)
        box = None
        for item in tmp.out:
            box = _union(box, path_extents(item.path))
        return box

    def clip_region(self, cp_el, bbox_fn, ctm, st: _State, seen=frozenset()):
        """A ``<clipPath>``'s region in document space (nonzero rule)."""
        key = id(cp_el)
        if key in seen:
            return None
        cp_style = self.style_of(cp_el)
        base = self.transform_of(cp_el)
        if (cp_el.get("clipPathUnits") or "userSpaceOnUse") == "objectBoundingBox":
            bb = bbox_fn()
            if bb is None or bb[2] <= bb[0] or bb[3] <= bb[1]:
                return Path(np.empty((0, 2)))
            base = base + Affine2D().scale(bb[2] - bb[0], bb[3] - bb[1]).translate(
                bb[0], bb[1]
            )
        total = base + ctm
        pieces = []
        cst = replace(st, style=cp_style, ctm=total, clip=None, out=[], measuring=True)
        for child in cp_el:
            if not _is_svg_tag(child) or not self.conditions_pass(child):
                continue
            tag = strip_ns(child.tag)
            target, extra = child, Affine2D()
            cstyle = self.child_style(child, cp_style)
            if tag == "use":
                target = self.ref(_href(child))
                if target is None:
                    continue
                ctx = self.ctx(cstyle, st)
                extra = Affine2D().translate(
                    _length(child.get("x"), 0, "x", ctx),
                    _length(child.get("y"), 0, "y", ctx),
                )
                ustyle = cstyle
                cstyle = self.child_style(target, ustyle)
                tag = strip_ns(target.tag)
                extra = self.transform_of(target) + extra
            if tag not in SHAPE_TAGS and tag != "text":
                continue
            if is_hidden(cstyle) or is_invisible(cstyle):
                continue
            ctrans = (
                self.transform_of(child) + total
                if target is child
                else extra + self.transform_of(child) + total
            )
            if tag == "text":
                tmp = replace(cst, out=[], ctm=ctrans)
                self.text(target, cstyle, ctrans, tmp)
                geom = join_paths([s.path for s in tmp.out])
            else:
                parsed = shape_to_parsed(tag, target.attrib, self.ctx(cstyle, st))
                if parsed is None or len(parsed.path.vertices) == 0:
                    continue
                geom = parsed.path
                if (cstyle.get("clip-rule") or "nonzero").strip() == "evenodd":
                    geom = evenodd_to_nonzero(geom)
                geom = geom.transformed(ctrans)
            if len(geom.vertices):
                pieces.append(normalize_orientation(geom))
        region = join_paths(pieces)
        nested = self.ref(cp_style.get("clip-path"), ("clipPath",))
        if nested is not None:
            other = self.clip_region(nested, bbox_fn, ctm, st, seen | {key})
            try:
                region = intersect_clip(region, other)
            except ValueError:
                pass
        return region

    def mask(self, m_el, bbox_user, ctm, st: _State):
        units = m_el.get("maskUnits") or "objectBoundingBox"
        content_units = m_el.get("maskContentUnits") or "userSpaceOnUse"
        mstyle = self.style_of(m_el)
        rect = _region(
            m_el,
            units,
            bbox_user,
            self.ctx(mstyle, st),
            default=("-10%", "-10%", "120%", "120%"),
        )
        if rect is None:
            return None
        content_ctm = ctm
        if content_units == "objectBoundingBox":
            if bbox_user is None:
                return None
            bx0, by0, bx1, by1 = bbox_user
            content_ctm = (
                Affine2D().scale(bx1 - bx0, by1 - by0).translate(bx0, by0) + ctm
            )
        region_doc = _transformed_bbox(rect, ctm)
        items: list = []
        mst = replace(
            st,
            ctm=content_ctm,
            style=mstyle,
            out=items,
            opacity=1.0,
            clip=rect_path(*rect).transformed(ctm),
            chain=st.chain | {id(m_el)},
        )
        if id(m_el) in st.chain:
            return None
        for child in m_el:
            self.walk(child, mst)
        kind = (mstyle.get("mask-type") or "luminance").strip().lower()
        return Mask(
            items=items,
            region=region_doc,
            kind=kind if kind in ("alpha", "luminance") else "luminance",
        )

    # ------------------------------------------------------------------ #
    # Viewports
    # ------------------------------------------------------------------ #
    def viewport_clip(self, style, rect, ctm, st):
        overflow = (style.get("overflow") or "hidden").strip().lower()
        if overflow in ("visible", "auto"):
            return st.clip, True
        region = rect_path(*rect).transformed(ctm)
        try:
            return intersect_clip(st.clip, region), True
        except ValueError:
            return st.clip, True

    def nested_svg(self, el, style, ctm, st: _State, size=None) -> None:
        ctx = self.ctx(style, st)
        x = _length(el.get("x"), 0, "x", ctx)
        y = _length(el.get("y"), 0, "y", ctx)
        w = _length(el.get("width") or "100%", 0, "x", ctx)
        h = _length(el.get("height") or "100%", 0, "y", ctx)
        if size is not None:
            w = size[0] if size[0] is not None else w
            h = size[1] if size[1] is not None else h
        if w <= 0 or h <= 0:
            return
        vb = parse_viewbox(el.get("viewBox"))
        inner_ctm = (
            viewbox_transform(vb, w, h, el.get("preserveAspectRatio"), x, y) + ctm
        )
        vp = (vb[2], vb[3]) if vb else (w, h)

        def body(s):
            clip, _ = self.viewport_clip(style, (x, y, x + w, y + h), ctm, s)
            if clip is not None and len(clip.vertices) == 0:
                return
            self.children(el, replace(s, ctm=inner_ctm, vp=vp, clip=clip), style)

        if st.measuring:
            self.children(el, replace(st, ctm=inner_ctm, vp=vp), style)
            return
        self.with_effects(el, style, ctm, st, body)

    def use(self, el, style, ctm, st: _State) -> None:
        target = self.ref(_href(el))
        if target is None or id(target) in st.chain or el is target:
            return
        ctx = self.ctx(style, st)
        x = _length(el.get("x"), 0, "x", ctx)
        y = _length(el.get("y"), 0, "y", ctx)
        use_ctm = Affine2D().translate(x, y) + ctm
        tag = strip_ns(target.tag)
        chain = st.chain | {id(target)}

        def body(s):
            s = replace(s, chain=chain, ctm=use_ctm)
            if tag == "symbol":
                self.symbol(el, target, style, use_ctm, s)
            elif tag == "svg":
                tstyle = self.child_style(target, style)
                if is_hidden(tstyle):
                    return
                w = el.get("width")
                h = el.get("height")
                size = (
                    _length(w, 0, "x", ctx) if w else None,
                    _length(h, 0, "y", ctx) if h else None,
                )
                self.nested_svg(
                    target, tstyle, self.transform_of(target) + use_ctm, s, size=size
                )
            else:
                self.walk(target, replace(s, style=style))

        self.with_effects(el, style, ctm, st, body)

    def symbol(self, use_el, sym, use_style, use_ctm, st: _State) -> None:
        sstyle = self.child_style(sym, use_style)
        if is_hidden(sstyle):
            return
        ctx = self.ctx(sstyle, st)
        w = use_el.get("width") or sym.get("width") or "100%"
        h = use_el.get("height") or sym.get("height") or "100%"
        w, h = _length(w, 0, "x", ctx), _length(h, 0, "y", ctx)
        sx = _length(sym.get("x"), 0, "x", ctx)
        sy = _length(sym.get("y"), 0, "y", ctx)
        if w <= 0 or h <= 0:
            return
        vb = parse_viewbox(sym.get("viewBox"))
        inner_ctm = (
            viewbox_transform(vb, w, h, sym.get("preserveAspectRatio"), sx, sy)
            + use_ctm
        )
        vp = (vb[2], vb[3]) if vb else (w, h)
        clip = st.clip
        if not st.measuring:
            clip, _ = self.viewport_clip(sstyle, (sx, sy, sx + w, sy + h), use_ctm, st)
            if clip is not None and len(clip.vertices) == 0:
                return
        self.children(sym, replace(st, ctm=inner_ctm, vp=vp, clip=clip), sstyle)

    # ------------------------------------------------------------------ #
    # Painting
    # ------------------------------------------------------------------ #
    def bind_paint(
        self, value, style, ctm, bbox_user, paint_opacity, st, current_color
    ):
        """Resolve a ``url(#id) [fallback]`` paint into ``(color, BoundPaint|None)``."""
        ref, fallback = split_paint(value)
        server = self.servers.get(ref) if ref else None

        def fallback_color():
            if not fallback:
                return NONE
            c = parse_color(fallback, current_color=current_color, default=NONE)
            return _with_alpha(c, paint_opacity)

        if server is None:
            return fallback_color(), None
        if isinstance(server, Gradient):
            if not server.stops:
                return NONE, None
            if server.units == "objectBoundingBox" and (
                bbox_user is None
                or bbox_user[2] <= bbox_user[0]
                or bbox_user[3] <= bbox_user[1]
            ):
                return fallback_color(), None
            if len(server.stops) == 1:
                return _with_alpha(server.stops[0][1], paint_opacity), None
            return NONE, BoundPaint(server, ctm, bbox_user, paint_opacity)
        if isinstance(server, Pattern):
            bound = self.bind_pattern(server, bbox_user, ctm, paint_opacity, st)
            if bound is None:
                return fallback_color(), None
            return NONE, bound
        return fallback_color(), None

    def bind_pattern(self, pat: Pattern, bbox_user, ctm, paint_opacity, st):
        if id(pat.element) in st.chain:
            return None
        pstyle = self.style_of(pat.element)
        ctx = self.ctx(pstyle, st)
        if pat.units == "objectBoundingBox":
            if bbox_user is None:
                return None
            bx0, by0, bx1, by1 = bbox_user
            bw, bh = bx1 - bx0, by1 - by0

            def frac(v):
                v = v.strip()
                return float(v[:-1]) / 100 if v.endswith("%") else float(v)

            try:
                x, y = bx0 + frac(pat.x) * bw, by0 + frac(pat.y) * bh
                w, h = frac(pat.width) * bw, frac(pat.height) * bh
            except ValueError:
                return None
        else:
            x, y = _length(pat.x, 0, "x", ctx), _length(pat.y, 0, "y", ctx)
            w, h = _length(pat.width, 0, "x", ctx), _length(pat.height, 0, "y", ctx)
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
        tst = _State(
            ctm=content,
            style=pstyle,
            out=items,
            vp=(pat.viewbox[2], pat.viewbox[3]) if pat.viewbox else st.vp,
            chain=st.chain | {id(pat.element)},
            depth=st.depth + 1,
        )
        for child in pat.element:
            self.walk(child, tst)
        tile_to_doc = Affine2D().translate(x, y) + pat.transform + ctm
        tile = PatternTile(items=items, width=w, height=h, transform=tile_to_doc)
        return BoundPaint(pat, ctm, bbox_user, paint_opacity, tile=tile)

    def paint_shape(self, el_id, path_user, style, ctm, st, bbox_user, text=None):
        """Emit fill/stroke shapes for a path in the element's user space."""
        ctx = self.ctx(style, st)
        kwargs = style_to_kwargs(style, ctx, st.opacity)
        current = parse_color(style.get("color"), default="black")
        if current == NONE:
            current = "black"
        el_op = opacity(style.get("opacity")) * st.opacity
        fill_paint = stroke_paint = None
        fc, ec = kwargs.get("facecolor"), kwargs.get("edgecolor")
        fill_url = isinstance(fc, str) and fc.lower().startswith("url(")
        stroke_url = isinstance(ec, str) and ec.lower().startswith("url(")
        if fill_url:
            pop = el_op * opacity(style.get("fill-opacity"))
            fc, fill_paint = self.bind_paint(
                fc, style, ctm, bbox_user, pop, st, current
            )
        if stroke_url:
            pop = el_op * opacity(style.get("stroke-opacity"))
            ec, stroke_paint = self.bind_paint(
                ec, style, ctm, bbox_user, pop, st, current
            )
            if stroke_paint is not None:
                ec = _with_alpha(stroke_paint.server.average_color(), pop)
        if fill_url or stroke_url:
            # Resolved paints carry their opacity; fold any patch alpha into
            # the other paint so a single alpha does not clobber them.
            alpha = kwargs.pop("alpha", None)
            if alpha is not None:
                if not fill_url:
                    kwargs["facecolor"] = _with_alpha(kwargs["facecolor"], alpha)
                if not stroke_url:
                    kwargs["edgecolor"] = _with_alpha(kwargs["edgecolor"], alpha)
            if fill_url:
                kwargs["facecolor"] = fc
            if stroke_url:
                kwargs["edgecolor"] = ec

        # Stroke geometry: user units -> document units.
        non_scaling = (
            style.get("vector-effect") or ""
        ).strip().lower() == "non-scaling-stroke"
        k = 1.0 if non_scaling else _linear_scale(ctm)
        kwargs["linewidth"] = kwargs.get("linewidth", 1.0) * k
        if "linestyle" in kwargs:
            off, dashes = kwargs["linestyle"]
            kwargs["linestyle"] = (off * k, [d * k for d in dashes])

        if (style.get("fill-rule") or "nonzero").strip().lower() == "evenodd":
            path_user = evenodd_to_nonzero(path_user)
        path_doc = path_user.transformed(ctm)

        def make(kw, fp, sp):
            return Shape(
                id=el_id,
                path=path_doc,
                style=dict(style),
                kwargs=kw,
                clip=st.clip,
                fill_paint=fp,
                stroke_paint=sp,
                scale_stroke=not non_scaling,
                text=text,
            )

        has_fill = kwargs.get("facecolor") != NONE or fill_paint is not None
        has_stroke = kwargs.get("edgecolor") != NONE or stroke_paint is not None
        order = _paint_order(style.get("paint-order"))
        shapes = []
        if has_fill and has_stroke and order.index("stroke") < order.index("fill"):
            shapes.append(make({**kwargs, "facecolor": NONE}, None, stroke_paint))
            shapes.append(make({**kwargs, "edgecolor": NONE}, fill_paint, None))
        else:
            shapes.append(make(kwargs, fill_paint, stroke_paint))
        return shapes, order

    def shape(self, el, tag, style, ctm, st: _State) -> None:
        ctx = self.ctx(style, st)
        parsed = shape_to_parsed(tag, el.attrib, ctx)
        if parsed is None or len(parsed.path.vertices) == 0:
            return
        if st.measuring:
            st.out.append(Shape(el.get("id", ""), parsed.path.transformed(ctm), {}, {}))
            return
        if is_invisible(style):
            return
        bbox_user = path_extents(parsed.path)
        shapes, order = self.paint_shape(
            el.get("id", ""), parsed.path, style, ctm, st, bbox_user
        )
        has_markers = tag in MARKABLE_TAGS and any(
            style.get(p, "none").strip().lower() not in ("none", "")
            for p in ("marker-start", "marker-mid", "marker-end")
        )
        if not has_markers:
            st.out.extend(shapes)
            return
        markers: list = []
        self.markers(parsed, style, ctm, replace(st, out=markers))
        for part in order:
            if part == "markers":
                st.out.extend(markers)
            elif part == "fill":
                st.out.extend(
                    s
                    for s in shapes
                    if s.kwargs.get("facecolor") != NONE or s.fill_paint
                )
            else:
                st.out.extend(
                    s
                    for s in shapes
                    if not (s.kwargs.get("facecolor") != NONE or s.fill_paint)
                )

    # ------------------------------------------------------------------ #
    # Markers
    # ------------------------------------------------------------------ #
    def markers(self, parsed, style, ctm, st: _State) -> None:
        from .pathdata import marker_vertices

        ctx = self.ctx(style, st)
        sw = parse_length(style.get("stroke-width"), 1.0, ctx=ctx)
        el_op = opacity(style.get("opacity")) * st.opacity
        for x, y, angle, role in marker_vertices(parsed):
            props = {
                "start": ("marker-start",),
                "mid": ("marker-mid",),
                "end": ("marker-end",),
                "both": ("marker-start", "marker-end"),
            }[role]
            for prop in props:
                m = self.ref(style.get(prop), ("marker",))
                if m is None or id(m) in st.chain:
                    continue
                r = "start" if prop == "marker-start" else role
                self.place_marker(
                    m, x, y, angle, r, sw, ctm, replace(st, opacity=el_op)
                )

    def place_marker(self, m, x, y, auto_angle, role, sw, ctm, st: _State) -> None:
        mstyle = self.style_of(m)
        ctx = self.ctx(mstyle, st)
        orient = (m.get("orient") or "0").strip()
        if orient == "auto":
            angle = auto_angle
        elif orient == "auto-start-reverse":
            angle = auto_angle + 180.0 if role == "start" else auto_angle
        else:
            from .transform import _numbers

            nums = _numbers(orient)
            angle = nums[0] if nums else 0.0
        scale = sw if (m.get("markerUnits") or "strokeWidth") == "strokeWidth" else 1.0
        mw = _length(m.get("markerWidth") or "3", 3.0, "x", ctx)
        mh = _length(m.get("markerHeight") or "3", 3.0, "y", ctx)
        if mw <= 0 or mh <= 0:
            return
        vb = parse_viewbox(m.get("viewBox"))
        view = (
            viewbox_transform(vb, mw, mh, m.get("preserveAspectRatio"))
            if vb
            else Affine2D()
        )
        refx = _ref_coord(m.get("refX"), vb, 0, mw, ctx, "x")
        refy = _ref_coord(m.get("refY"), vb, 1, mh, ctx, "y")
        rx, ry = view.transform((refx, refy))
        outer = (
            Affine2D()
            .translate(-rx, -ry)
            .scale(scale)
            .rotate_deg(angle)
            .translate(x, y)
            + ctm
        )
        content_ctm = view + outer
        clip = st.clip
        overflow = (mstyle.get("overflow") or "hidden").strip().lower()
        if overflow not in ("visible", "auto"):
            region = rect_path(0, 0, mw, mh).transformed(outer)
            try:
                clip = intersect_clip(clip, region)
            except ValueError:
                pass
        mst = replace(
            st,
            ctm=content_ctm,
            style=mstyle,
            clip=clip,
            vp=(vb[2], vb[3]) if vb else (mw, mh),
            chain=st.chain | {id(m)},
        )
        for child in m:
            self.walk(child, mst)

    # ------------------------------------------------------------------ #
    # Text
    # ------------------------------------------------------------------ #
    def textpath_geometry(self, tp):
        """The path a ``<textPath>`` follows, in the text's user space."""
        d = tp.get("path")
        if d:
            path = parse_path_data(d).path
        else:
            target = self.ref(_href(tp))
            if target is None or strip_ns(target.tag) not in SHAPE_TAGS:
                return None
            parsed = shape_to_parsed(strip_ns(target.tag), target.attrib)
            if parsed is None:
                return None
            path = parsed.path.transformed(self.transform_of(target))
        if (tp.get("side") or "left").strip() == "right":
            path = reverse_path(path)
        return path

    def text(self, el, style, ctm, st: _State) -> None:
        from .text import layout_text

        runs, content, first = layout_text(
            el,
            style,
            child_style=self.child_style,
            ctx_for=lambda s: self.ctx(s, st),
            resolve_textpath=self.textpath_geometry,
        )
        if not runs:
            return
        if st.measuring:
            for run in runs:
                st.out.append(Shape(run.id, run.path.transformed(ctm), {}, {}))
            return
        record = Text(
            id=el.get("id", ""),
            x=first[0],
            y=first[1],
            content=content,
            style=dict(style),
            transform=ctm,
        )
        self.scene.texts.append(record)
        bbox_user = None
        for run in runs:
            bbox_user = _union(bbox_user, path_extents(run.path))
        for run in runs:
            if is_invisible(run.style) or len(run.path.vertices) == 0:
                continue
            shapes, _ = self.paint_shape(
                run.id, run.path, run.style, ctm, st, bbox_user, text=record
            )
            record.runs.extend(shapes)
            st.out.extend(shapes)

    # ------------------------------------------------------------------ #
    # Images
    # ------------------------------------------------------------------ #
    def image(self, el, style, ctm, st: _State) -> None:
        ctx = self.ctx(style, st)
        x = _length(el.get("x"), 0, "x", ctx)
        y = _length(el.get("y"), 0, "y", ctx)
        w_attr, h_attr = el.get("width"), el.get("height")
        w = _length(w_attr, 0, "x", ctx) if w_attr and w_attr != "auto" else None
        h = _length(h_attr, 0, "y", ctx) if h_attr and h_attr != "auto" else None
        if (w is not None and w <= 0) or (h is not None and h <= 0):
            return
        href = _href(el)
        if not href:
            return
        loaded = self.load_image(href)
        if loaded is None:
            return
        kind, payload = loaded
        par = el.get("preserveAspectRatio")
        if kind == "svg":
            sub_root = payload
            sub = _Parser(sub_root, base_dir=self.base_dir, depth=self.nest_depth + 1)
            vb, iw, ih = sub.document_viewport()
            iw = iw or (vb[2] if vb else 100.0)
            ih = ih or (vb[3] if vb else 100.0)
        else:
            ih, iw = payload.shape[:2]
        w, h = _auto_size(w, h, iw, ih)
        if st.measuring:
            st.out.append(
                Shape(
                    el.get("id", ""),
                    rect_path(x, y, x + w, y + h).transformed(ctm),
                    {},
                    {},
                )
            )
            return
        if is_invisible(style):
            return
        clip = st.clip
        _align, mode = parse_preserve_aspect_ratio(par)
        if mode == "slice" or kind == "svg":
            try:
                clip = intersect_clip(
                    clip, rect_path(x, y, x + w, y + h).transformed(ctm)
                )
            except ValueError:
                pass
        alpha = opacity(style.get("opacity")) * st.opacity
        if kind == "svg":
            if self.nest_depth >= 8:
                return
            # Map the sub-document's own document space into the image viewport.
            mapping = viewbox_transform((0, 0, iw, ih), w, h, par, x, y) + ctm
            sub.build(outer_ctm=mapping, out=st.out, opacity_=alpha, clip=clip)
            return
        mapping = viewbox_transform((0, 0, iw, ih), w, h, par, x, y) + ctm
        rendering = (style.get("image-rendering") or "auto").strip().lower()
        interp = (
            "nearest"
            if rendering in ("pixelated", "optimizespeed", "crisp-edges")
            else "auto"
        )
        st.out.append(
            Image(
                id=el.get("id", ""),
                data=payload,
                transform=mapping,
                clip=clip,
                alpha=alpha,
                interpolation=interp,
            )
        )

    def load_image(self, href: str):
        """Decode an image reference into ``("raster", rgba)`` or ``("svg", root)``."""
        try:
            data, mime = _read_href(href, self.base_dir)
        except Exception as exc:  # noqa: BLE001 - report and skip unreadable images
            self.warn(
                "image:" + href[:40],
                f"svg2mpl: could not load image {href[:60]!r}: {exc}",
            )
            return None
        if data is None:
            return None
        if (
            "svg" in (mime or "")
            or data.lstrip()[:5] in (b"<?xml", b"<svg ")
            or b"<svg" in data[:512]
        ):
            try:
                return "svg", ElementTree.fromstring(data)
            except ElementTree.ParseError:
                pass
        try:
            from PIL import Image as PILImage

            img = PILImage.open(io.BytesIO(data)).convert("RGBA")
            return "raster", np.asarray(img)
        except Exception as exc:  # noqa: BLE001
            self.warn("image:" + href[:40], f"svg2mpl: could not decode image: {exc}")
            return None


# ---------------------------------------------------------------------- #
# Module helpers
# ---------------------------------------------------------------------- #
def _paint_order(value: str | None) -> list[str]:
    default = ["fill", "stroke", "markers"]
    if not value or value.strip().lower() == "normal":
        return default
    given = [t for t in value.split() if t in default]
    return given + [t for t in default if t not in given]


def _with_alpha(color, alpha: float):
    if color == NONE or alpha == 1.0:
        return color
    from matplotlib.colors import to_rgba

    r, g, b, a = to_rgba(color)
    return (r, g, b, a * alpha)


def _painted_extents(item):
    """Document-space extents of a drawable, padded by half its stroke."""
    ext = path_extents(item.path)
    if ext is None:
        return None
    if isinstance(item, Shape) and item.kwargs.get("edgecolor", NONE) != NONE:
        pad = (
            item.kwargs.get("linewidth", 0.0) / 2 * (1.5 if item.scale_stroke else 1.0)
        )
        ext = (ext[0] - pad, ext[1] - pad, ext[2] + pad, ext[3] + pad)
    clip = getattr(item, "clip", None)
    if clip is not None:
        ce = path_extents(clip)
        if ce is None:
            return None
        ext = _intersect_rect(ext, ce)
        if ext[2] <= ext[0] or ext[3] <= ext[1]:
            return None
    return ext


def _any_overlap(boxes) -> bool:
    """Whether any two boxes overlap (sweep over x)."""
    boxes = sorted(boxes)
    active: list = []
    for b in boxes:
        active = [a for a in active if a[2] > b[0]]
        for a in active:
            if a[1] < b[3] and b[1] < a[3]:
                return True
        active.append(b)
    return False


def _intersect_rect(a, b):
    r = (max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]))
    return r


def _region(el, units, bbox_user, ctx, default):
    """An ``x/y/width/height`` region (masks, filters) in user space."""
    names = ("x", "y", "width", "height")
    vals = [el.get(n) or d for n, d in zip(names, default, strict=False)]
    if units == "objectBoundingBox":
        if bbox_user is None:
            return None
        bx0, by0, bx1, by1 = bbox_user
        bw, bh = bx1 - bx0, by1 - by0

        def frac(v):
            v = v.strip()
            try:
                return float(v[:-1]) / 100 if v.endswith("%") else float(v)
            except ValueError:
                return 0.0

        x, y = bx0 + frac(vals[0]) * bw, by0 + frac(vals[1]) * bh
        w, h = frac(vals[2]) * bw, frac(vals[3]) * bh
    else:
        x = _length(vals[0], 0, "x", ctx)
        y = _length(vals[1], 0, "y", ctx)
        w = _length(vals[2], 0, "x", ctx)
        h = _length(vals[3], 0, "y", ctx)
    if w <= 0 or h <= 0:
        return None
    return (x, y, x + w, y + h)


def _ref_coord(value, vb, axis, size, ctx, ax):
    if value is None:
        return 0.0
    v = value.strip().lower()
    span = vb[2 + axis] if vb else size
    origin = vb[axis] if vb else 0.0
    keywords = {"left": 0.0, "top": 0.0, "center": 0.5, "right": 1.0, "bottom": 1.0}
    if v in keywords:
        return origin + keywords[v] * span
    return _length(v, 0.0, ax, ctx)


def _auto_size(w, h, iw, ih):
    if w is None and h is None:
        return float(iw), float(ih)
    if w is None:
        return h * iw / ih if ih else h, h
    if h is None:
        return w, w * ih / iw if iw else w
    return w, h


def _read_href(href: str, base_dir: str | None):
    """Return ``(bytes, mime)`` for a ``data:`` URI or local file reference."""
    href = href.strip()
    if href.startswith("data:"):
        header, _, payload = href[5:].partition(",")
        mime = header.split(";")[0].strip().lower()
        if ";base64" in header.lower():
            payload = re.sub(r"\s+", "", payload)
            return base64.b64decode(payload + "=" * (-len(payload) % 4)), mime
        return urllib.parse.unquote_to_bytes(payload), mime
    scheme = urllib.parse.urlparse(href).scheme
    if scheme in ("http", "https", "ftp"):
        raise ValueError("remote images are not fetched")
    if scheme == "file":
        path = urllib.parse.unquote(urllib.parse.urlparse(href).path)
    else:
        path = urllib.parse.unquote(href.split("#")[0])
        if not os.path.isabs(path):
            if base_dir is None:
                raise ValueError("relative image path without a base directory")
            path = os.path.join(base_dir, path)
    with open(path, "rb") as fh:
        data = fh.read()
    mime = "image/svg+xml" if path.lower().endswith((".svg", ".svgz")) else ""
    if path.lower().endswith(".svgz"):
        import gzip

        data = gzip.decompress(data)
    return data, mime
