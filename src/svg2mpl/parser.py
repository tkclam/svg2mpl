"""Traverse an SVG document into a :class:`~svg2mpl.scene.Scene`.

The walk maintains, per element, the current transformation matrix (CTM, user
space -> document space), the inherited computed style, the accumulated group
opacity, the active clip region and the current viewport (see
:class:`~svg2mpl._state.State`). It descends through containers (``g``, ``a``,
``switch``, nested ``svg``), instantiates ``<use>`` references (including
``<symbol>`` viewports), lays out text and decodes images. Painting, clipping,
markers and filters live in their own modules; elements that need off-screen
compositing (``mask``, ``filter``, overlapping group opacity) become
:class:`~svg2mpl.scene.Group` items.

Non-rendered subtrees (``defs``, ``symbol``, ``clipPath``, gradients, metadata,
animation elements, ``foreignObject`` ...) are skipped during the normal walk.
"""

from __future__ import annotations

import contextlib
import warnings
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path as FilePath

import numpy as np
from matplotlib.path import Path
from matplotlib.transforms import Affine2D

from ._state import State
from ._xml import href, is_svg_element, strip_ns
from .clipping import any_overlap, build_mask, clip_region, painted_extents
from .color import parse_url_ref
from .css import compute_css_declarations
from .filters import Filter, css_filter, parse_filter
from .geometry import (
    Rect,
    intersect_clip,
    is_empty,
    path_extents,
    rect_intersection,
    rect_is_empty,
    rect_path,
    rect_union,
    reverse_path,
    transform_rect,
)
from .gradients import parse_gradients
from .images import ImageError, auto_size, load_image
from .markers import emit_markers, has_markers
from .paint import has_fill, paint_shape
from .pathdata import parse_path_data
from .scene import Group, Image, Mask, Scene, Shape, Text
from .shapes import MARKABLE_TAGS, SHAPE_TAGS, shape_to_parsed
from .style import (
    compute_style,
    declarations,
    font_size,
    is_hidden,
    is_invisible,
    opacity,
    parse_style_attr,
)
from .text import layout_text
from .transform import (
    compose,
    parse_preserve_aspect_ratio,
    parse_transform,
    parse_viewbox,
    viewbox_transform,
)
from .units import LengthContext, parse_length, split_length

__all__ = ["Parser", "parse", "parse_string"]

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
_MAX_NESTED_DOCUMENTS = 8  # <image> referencing SVG referencing SVG ...
_SUPPORTED_LANGUAGES = ("en",)
_PIXELATED = ("pixelated", "optimizespeed", "crisp-edges")

Body = Callable[[State], None]


def parse(source: str | FilePath) -> Scene:
    """Parse an SVG file path into a :class:`Scene`."""
    path = FilePath(source)
    root = ET.parse(path).getroot()
    return Parser(root, base_dir=str(path.resolve().parent)).build()


def parse_string(text: str, base_dir: str | None = None) -> Scene:
    """Parse an SVG document from a string into a :class:`Scene`.

    ``base_dir`` resolves relative ``<image>`` references.
    """
    root = ET.fromstring(text)
    return Parser(root, base_dir=base_dir).build()


def _length(value: str | None, axis: str, ctx: LengthContext, default=0.0) -> float:
    return parse_length(value, default, axis=axis, ctx=ctx)


class Parser:
    """Walks one SVG document; see the module docstring."""

    def __init__(self, root: ET.Element, base_dir: str | None = None, depth: int = 0):
        self.root = root
        self.base_dir = base_dir
        self.nest_depth = depth
        self.css = compute_css_declarations(root)
        self.id_index: dict[str, ET.Element] = {}
        for e in root.iter():
            el_id = e.get("id") if isinstance(e.tag, str) else None
            if el_id and el_id not in self.id_index:
                self.id_index[el_id] = e
        self.parent_of = {id(c): p for p in root.iter() for c in p}
        self.servers: dict = {}
        self.scene = Scene()
        self._style_cache: dict[int, dict] = {}
        self._warned: set[str] = set()

    # ------------------------------------------------------------------ #
    # Helpers used across the parser modules
    # ------------------------------------------------------------------ #
    def warn(self, key: str, msg: str) -> None:
        """Warn once per ``key``."""
        if key not in self._warned:
            self._warned.add(key)
            warnings.warn(msg, stacklevel=4)

    def own_decls(self, el: ET.Element) -> dict:
        return declarations(el.attrib, self.css.get(id(el)))

    def child_style(self, el: ET.Element, parent_style: dict | None) -> dict:
        return compute_style(self.own_decls(el), parent_style)

    def style_of(self, el: ET.Element) -> dict:
        """The computed style of ``el`` in its document context."""
        key = id(el)
        if key not in self._style_cache:
            parent = self.parent_of.get(key)
            parent_style = self.style_of(parent) if parent is not None else None
            self._style_cache[key] = compute_style(self.own_decls(el), parent_style)
        return self._style_cache[key]

    def transform_of(self, el: ET.Element) -> Affine2D:
        """The element's own transform (CSS ``transform`` beats the attribute)."""
        inline = parse_style_attr(el.get("style")).get("transform")
        css = (self.css.get(id(el)) or {}).get("transform")
        value = inline or css or el.get("transform")
        if value and value.strip().lower() == "none":
            return Affine2D()
        return parse_transform(value)

    @staticmethod
    def ctx(style: dict | None, st: State) -> LengthContext:
        return LengthContext(st.vp[0], st.vp[1], font_size(style))

    def ref(self, value: str | None, tags: tuple[str, ...] | None = None):
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
    def conditions_pass(el: ET.Element) -> bool:
        """Evaluate the conditional processing attributes."""
        ext = el.get("requiredExtensions")
        if ext is not None and ext.strip():
            return False  # no extensions are supported
        lang = el.get("systemLanguage")
        if lang is None:
            return True
        langs = [x.strip().lower() for x in lang.split(",") if x.strip()]
        return any(
            x == s or x.startswith(s + "-") for x in langs for s in _SUPPORTED_LANGUAGES
        )

    # ------------------------------------------------------------------ #
    # Document
    # ------------------------------------------------------------------ #
    def document_viewport(self):
        """``(viewBox, width, height)`` of the root, in CSS pixels."""
        root = self.root
        vb = parse_viewbox(root.get("viewBox"))

        def absolute(name):
            value = root.get(name)
            parts = split_length(value)
            if parts is None or parts[1] == "%" or parts[0] <= 0:
                return None
            return parse_length(value)

        w, h = absolute("width"), absolute("height")
        if vb is not None:
            if w and not h:
                h = w * vb[3] / vb[2] if vb[2] else None
            elif h and not w:
                w = h * vb[2] / vb[3] if vb[3] else None
            if not (w and h):
                w, h = vb[2], vb[3]
        return vb, w or 0.0, h or 0.0

    def build(
        self,
        outer_ctm: Affine2D | None = None,
        out: list | None = None,
        group_opacity: float = 1.0,
        clip: Path | None = None,
    ) -> Scene:
        """Parse the document; optionally into ``out`` through ``outer_ctm``
        (how ``<image>`` embeds another SVG document)."""
        vb, w, h = self.document_viewport()
        root_ctm = viewbox_transform(vb, w, h, self.root.get("preserveAspectRatio"))
        if outer_ctm is not None:
            root_ctm = compose(root_ctm, outer_ctm)
        vp = (vb[2], vb[3]) if vb else (w or 100.0, h or 100.0)
        self.servers = parse_gradients(
            self.root, style_of=self.style_of, ctx=LengthContext(*vp)
        )
        self.scene = Scene(width=w, height=h, gradients=self.servers)
        if out is not None:
            self.scene.items = out
        st = State(
            ctm=root_ctm,
            style=None,
            out=self.scene.items,
            vp=vp,
            opacity=group_opacity,
            clip=clip,
        )
        root = self.root
        style = compute_style(self.own_decls(root), None)
        if not is_hidden(style):
            ctm = compose(self.transform_of(root), st.ctm)
            self.with_effects(root, style, ctm, st, self._children_body(root, style))
        return self.scene

    # ------------------------------------------------------------------ #
    # Traversal
    # ------------------------------------------------------------------ #
    def _children_body(self, el: ET.Element, style: dict) -> Body:
        def body(st: State) -> None:
            self.children(el, st, style)

        return body

    def children(self, el: ET.Element, st: State, style: dict) -> None:
        inner = replace(st, style=style)
        for child in el:
            self.walk(child, inner)

    def walk(self, el: ET.Element, st: State) -> None:
        """Render ``el`` (and its subtree) into ``st.out``."""
        if not is_svg_element(el) or st.depth > _MAX_DEPTH:
            return
        tag = strip_ns(el.tag)
        if tag in _SKIP_TAGS or not self.conditions_pass(el):
            return
        style = compute_style(self.own_decls(el), st.style)
        if is_hidden(style):
            return
        ctm = compose(self.transform_of(el), st.ctm)
        self.render(el, tag, style, ctm, replace(st, depth=st.depth + 1))

    def render(self, el: ET.Element, tag: str, style: dict, ctm: Affine2D, st: State):
        """Dispatch on the element type (after style and transform)."""
        if tag in _CONTAINER_TAGS:
            self.with_effects(el, style, ctm, st, self._children_body(el, style))
        elif tag == "switch":
            self.with_effects(el, style, ctm, st, lambda s: self.switch(el, s, style))
        elif tag == "svg":
            self.nested_svg(el, style, ctm, st)
        elif tag == "use":
            self.use(el, style, ctm, st)
        elif tag in SHAPE_TAGS:
            leaf = lambda s: self.shape(el, tag, style, ctm, s)  # noqa: E731
            self.with_effects(el, style, ctm, st, leaf, leaf=True)
        elif tag == "text":
            leaf = lambda s: self.text(el, style, ctm, s)  # noqa: E731
            self.with_effects(el, style, ctm, st, leaf, leaf=True)
        elif tag == "image":
            leaf = lambda s: self.image(el, style, ctm, s)  # noqa: E731
            self.with_effects(el, style, ctm, st, leaf, leaf=True)

    def switch(self, el: ET.Element, st: State, style: dict) -> None:
        """Render the first direct child whose conditions pass."""
        inner = replace(st, style=style)
        for child in el:
            if not is_svg_element(child):
                continue
            if strip_ns(child.tag) in _SKIP_TAGS or not self.conditions_pass(child):
                continue
            self.walk(child, inner)
            return

    # ------------------------------------------------------------------ #
    # Group effects: opacity, clip-path, mask, filter
    # ------------------------------------------------------------------ #
    def with_effects(
        self,
        el: ET.Element,
        style: dict,
        ctm: Affine2D,
        st: State,
        body: Body,
        leaf: bool = False,
    ) -> None:
        """Run ``body`` with the element's clip, mask, filter and opacity.

        ``leaf`` elements (shapes, text, images) fold their own ``opacity``
        into their colors; containers pass it down or composite off-screen.
        """
        st = replace(st, ctm=ctm)
        if st.measuring:
            body(st)
            return
        op = opacity(style.get("opacity"))
        clip = st.clip
        cp_el = self.ref(style.get("clip-path"), ("clipPath",))
        if cp_el is not None:
            region = clip_region(self, cp_el, lambda: self.bbox(el, style, st), ctm, st)
            if region is not None:
                try:
                    clip = intersect_clip(st.clip, region)
                except ValueError:
                    # Two non-convex clips: composite off-screen, clip twice.
                    self._double_clip(el, style, st, body, leaf, region)
                    return
            if clip is not None and is_empty(clip):
                return  # clipped away entirely
        st = replace(st, clip=clip)
        mask_el = self.ref(style.get("mask"), ("mask",))
        flt = style.get("filter")
        flt = None if not flt or flt.strip().lower() == "none" else flt
        if mask_el is not None or flt is not None:
            self._effect_group(el, style, st, body, leaf, mask_el, flt, op)
        elif leaf:
            body(st)
        elif op < 1.0:
            self._group_opacity(el, st, body, op)
        else:
            body(st)

    def _group_opacity(self, el: ET.Element, st: State, body: Body, op: float) -> None:
        """Apply a container's ``opacity``.

        Multiplying the opacity into each child is exact when the children do
        not overlap; otherwise the group is composited off-screen first, as a
        browser does, so overlaps do not show through each other.
        """
        n_texts = len(self.scene.texts)
        tmp: list = []
        body(replace(st, opacity=st.opacity * op, out=tmp))
        boxes = [b for b in (painted_extents(i) for i in tmp) if b is not None]
        if not any_overlap(boxes):
            st.out.extend(tmp)
            return
        del self.scene.texts[n_texts:]  # the second pass records them again
        region = None
        for b in boxes:
            region = rect_union(region, b)
        if region is not None and st.clip is not None:
            clip_ext = path_extents(st.clip)
            region = rect_intersection(region, clip_ext) if clip_ext else None
        if region is None or rect_is_empty(region):
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

    def _double_clip(
        self,
        el: ET.Element,
        style: dict,
        st: State,
        body: Body,
        leaf: bool,
        inner: Path,
    ) -> None:
        """Clip by two non-convex regions: the inner one inside an off-screen
        group, the outer one on the group's image."""
        region = path_extents(inner)
        if region is None:
            return
        group = Group(id=el.get("id", ""), items=[], region=region, clip=st.clip)
        st.out.append(group)
        inner_state = replace(st, out=group.items, clip=inner)
        if not leaf:
            inner_state = replace(
                inner_state, opacity=st.opacity * opacity(style.get("opacity"))
            )
        body(inner_state)

    def _resolve_filter(self, flt: str, style: dict, bbox_user, st: State):
        """``(filter, ok)``: ``ok`` is False when a ``url()`` reference is broken,
        which disables rendering of the element."""
        f_el = self.ref(flt, ("filter",))
        if f_el is not None:
            filt = parse_filter(self, f_el, bbox_user, st.ctm, st)
            return filt, filt is not None
        if "url(" in flt:
            return None, False
        return css_filter(flt, bbox_user, st.ctm, self.ctx(style, st)), True

    def _effect_group(
        self,
        el: ET.Element,
        style: dict,
        st: State,
        body: Body,
        leaf: bool,
        mask_el: ET.Element | None,
        flt: str | None,
        op: float,
    ) -> None:
        """Render ``el`` into an off-screen group with its filter and mask."""
        bbox_user = self.bbox(el, style, st)
        region: Rect | None = None
        filt: Filter | None = None
        mask: Mask | None = None
        if flt is not None:
            filt, ok = self._resolve_filter(flt, style, bbox_user, st)
            if not ok:
                return
            if filt is not None:
                region = filt.region
        if mask_el is not None:
            mask = build_mask(self, mask_el, bbox_user, st.ctm, st)
            if mask is None:
                return
            region = (
                mask.region
                if region is None
                else rect_intersection(region, mask.region)
            )
        if region is None and bbox_user is not None:
            region = transform_rect(bbox_user, st.ctm)
        if region is None or rect_is_empty(region):
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
        if leaf and op > 0:
            # The group applies the element's opacity; a leaf would also fold
            # its own opacity into its colors, so cancel that out.
            inner = replace(inner, opacity=1.0 / op)
        body(inner)

    def bbox(self, el: ET.Element, style: dict, st: State) -> Rect | None:
        """The element's geometry bounding box in its own user space."""
        measured = replace(
            st, out=[], measuring=True, clip=None, ctm=Affine2D(), opacity=1.0
        )
        self.render(el, strip_ns(el.tag), style, Affine2D(), measured)
        box = None
        for item in measured.out:
            box = rect_union(box, path_extents(item.path))
        return box

    # ------------------------------------------------------------------ #
    # Viewports
    # ------------------------------------------------------------------ #
    @staticmethod
    def _viewport_clip(style: dict, rect: Rect, ctm: Affine2D, st: State):
        """The clip after entering a viewport (``overflow`` hides by default)."""
        overflow = (style.get("overflow") or "hidden").strip().lower()
        if overflow in ("visible", "auto"):
            return st.clip
        try:
            return intersect_clip(st.clip, rect_path(*rect).transformed(ctm))
        except ValueError:
            return st.clip

    def nested_svg(
        self,
        el: ET.Element,
        style: dict,
        ctm: Affine2D,
        st: State,
        size: tuple[float | None, float | None] | None = None,
    ) -> None:
        """A nested ``<svg>`` viewport (``size`` overrides from a ``<use>``)."""
        ctx = self.ctx(style, st)
        x, y = _length(el.get("x"), "x", ctx), _length(el.get("y"), "y", ctx)
        w = _length(el.get("width") or "100%", "x", ctx)
        h = _length(el.get("height") or "100%", "y", ctx)
        if size is not None:
            w = size[0] if size[0] is not None else w
            h = size[1] if size[1] is not None else h
        if w <= 0 or h <= 0:
            return
        vb = parse_viewbox(el.get("viewBox"))
        par = el.get("preserveAspectRatio")
        inner_ctm = compose(viewbox_transform(vb, w, h, par, x, y), ctm)
        vp = (vb[2], vb[3]) if vb else (w, h)

        def body(s: State) -> None:
            clip = s.clip
            if not s.measuring:
                clip = self._viewport_clip(style, (x, y, x + w, y + h), ctm, s)
                if clip is not None and is_empty(clip):
                    return
            self.children(el, replace(s, ctm=inner_ctm, vp=vp, clip=clip), style)

        self.with_effects(el, style, ctm, st, body)

    def use(self, el: ET.Element, style: dict, ctm: Affine2D, st: State) -> None:
        """Instantiate a ``<use>`` reference."""
        target = self.ref(href(el))
        if target is None or id(target) in st.chain or el is target:
            return
        ctx = self.ctx(style, st)
        offset = Affine2D().translate(
            _length(el.get("x"), "x", ctx), _length(el.get("y"), "y", ctx)
        )
        use_ctm = compose(offset, ctm)
        tag = strip_ns(target.tag)

        def body(s: State) -> None:
            s = replace(s, chain=s.chain | {id(target)}, ctm=use_ctm)
            if tag == "symbol":
                self._symbol(el, target, style, s)
            elif tag == "svg":
                target_style = self.child_style(target, style)
                if is_hidden(target_style):
                    return
                w, h = el.get("width"), el.get("height")
                size = (
                    _length(w, "x", ctx) if w else None,
                    _length(h, "y", ctx) if h else None,
                )
                target_ctm = compose(self.transform_of(target), use_ctm)
                self.nested_svg(target, target_style, target_ctm, s, size=size)
            else:
                self.walk(target, replace(s, style=style))

        self.with_effects(el, style, ctm, st, body)

    def _symbol(self, use_el: ET.Element, sym: ET.Element, use_style: dict, st: State):
        """A ``<symbol>`` instantiated by ``use_el`` (``st.ctm`` is the use's)."""
        sym_style = self.child_style(sym, use_style)
        if is_hidden(sym_style):
            return
        ctx = self.ctx(sym_style, st)
        w = _length(use_el.get("width") or sym.get("width") or "100%", "x", ctx)
        h = _length(use_el.get("height") or sym.get("height") or "100%", "y", ctx)
        if w <= 0 or h <= 0:
            return
        x, y = _length(sym.get("x"), "x", ctx), _length(sym.get("y"), "y", ctx)
        vb = parse_viewbox(sym.get("viewBox"))
        par = sym.get("preserveAspectRatio")
        inner_ctm = compose(viewbox_transform(vb, w, h, par, x, y), st.ctm)
        clip = st.clip
        if not st.measuring:
            clip = self._viewport_clip(sym_style, (x, y, x + w, y + h), st.ctm, st)
            if clip is not None and is_empty(clip):
                return
        vp = (vb[2], vb[3]) if vb else (w, h)
        self.children(sym, replace(st, ctm=inner_ctm, vp=vp, clip=clip), sym_style)

    # ------------------------------------------------------------------ #
    # Leaves: shapes, text, images
    # ------------------------------------------------------------------ #
    def shape(
        self, el: ET.Element, tag: str, style: dict, ctm: Affine2D, st: State
    ) -> None:
        parsed = shape_to_parsed(tag, el.attrib, self.ctx(style, st))
        if parsed is None or is_empty(parsed.path):
            return
        if st.measuring:
            st.out.append(Shape(el.get("id", ""), parsed.path.transformed(ctm), {}, {}))
            return
        if is_invisible(style):
            return
        shapes, order = paint_shape(
            self,
            el.get("id", ""),
            parsed.path,
            style,
            ctm,
            st,
            path_extents(parsed.path),
        )
        if tag not in MARKABLE_TAGS or not has_markers(style):
            st.out.extend(shapes)
            return
        markers: list = []
        emit_markers(self, parsed, style, ctm, replace(st, out=markers))
        for part in order:
            if part == "markers":
                st.out.extend(markers)
            elif part == "fill":
                st.out.extend(s for s in shapes if has_fill(s))
            else:
                st.out.extend(s for s in shapes if not has_fill(s))

    def textpath_geometry(self, tp: ET.Element) -> Path | None:
        """The path a ``<textPath>`` follows, in the text's user space."""
        d = tp.get("path")
        if d:
            path = parse_path_data(d).path
        else:
            target = self.ref(href(tp))
            if target is None or strip_ns(target.tag) not in SHAPE_TAGS:
                return None
            parsed = shape_to_parsed(strip_ns(target.tag), target.attrib)
            if parsed is None:
                return None
            path = parsed.path.transformed(self.transform_of(target))
        if (tp.get("side") or "left").strip() == "right":
            path = reverse_path(path)
        return path

    def text(self, el: ET.Element, style: dict, ctm: Affine2D, st: State) -> None:
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
            st.out.extend(Shape(r.id, r.path.transformed(ctm), {}, {}) for r in runs)
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
            bbox_user = rect_union(bbox_user, path_extents(run.path))
        for run in runs:
            if is_invisible(run.style) or is_empty(run.path):
                continue
            shapes, _ = paint_shape(
                self, run.id, run.path, run.style, ctm, st, bbox_user, text=record
            )
            record.runs.extend(shapes)
            st.out.extend(shapes)

    def load_image(self, ref: str):
        """Decode an image reference, warning (once) and returning ``None`` on
        failure."""
        try:
            return load_image(ref, self.base_dir)
        except (ImageError, OSError, ValueError) as exc:
            self.warn(
                "image:" + ref[:40],
                f"svg2mpl: could not load image {ref[:60]!r}: {exc}",
            )
            return None

    def image(self, el: ET.Element, style: dict, ctm: Affine2D, st: State) -> None:
        ctx = self.ctx(style, st)
        x, y = _length(el.get("x"), "x", ctx), _length(el.get("y"), "y", ctx)
        w_attr, h_attr = el.get("width"), el.get("height")
        w = _length(w_attr, "x", ctx) if w_attr and w_attr != "auto" else None
        h = _length(h_attr, "y", ctx) if h_attr and h_attr != "auto" else None
        if (w is not None and w <= 0) or (h is not None and h <= 0):
            return
        ref = href(el)
        loaded = self.load_image(ref) if ref else None
        if loaded is None:
            return
        pixels: np.ndarray | None = None
        sub: Parser | None = None
        if isinstance(loaded, np.ndarray):
            pixels = loaded
            ih, iw = pixels.shape[:2]
        else:
            sub = Parser(loaded, base_dir=self.base_dir, depth=self.nest_depth + 1)
            vb, iw, ih = sub.document_viewport()
            iw = iw or (vb[2] if vb else 100.0)
            ih = ih or (vb[3] if vb else 100.0)
        w, h = auto_size(w, h, iw, ih)
        viewport = rect_path(x, y, x + w, y + h).transformed(ctm)
        if st.measuring:
            st.out.append(Shape(el.get("id", ""), viewport, {}, {}))
            return
        if is_invisible(style):
            return
        par = el.get("preserveAspectRatio")
        clip = st.clip
        if sub is not None or parse_preserve_aspect_ratio(par)[1] == "slice":
            with contextlib.suppress(ValueError):
                clip = intersect_clip(clip, viewport)
        alpha = opacity(style.get("opacity")) * st.opacity
        # Map the image's own pixel (or document) space into its viewport.
        mapping = compose(viewbox_transform((0, 0, iw, ih), w, h, par, x, y), ctm)
        if pixels is None:
            if sub is not None and self.nest_depth < _MAX_NESTED_DOCUMENTS:
                sub.build(outer_ctm=mapping, out=st.out, group_opacity=alpha, clip=clip)
            return
        rendering = (style.get("image-rendering") or "auto").strip().lower()
        st.out.append(
            Image(
                id=el.get("id", ""),
                data=pixels,
                transform=mapping,
                clip=clip,
                alpha=alpha,
                interpolation="nearest" if rendering in _PIXELATED else "auto",
            )
        )
