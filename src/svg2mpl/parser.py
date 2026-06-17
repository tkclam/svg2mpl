"""Traverse an SVG document into a :class:`~svg2mpl.scene.Scene`.

The walk maintains a current transformation matrix (CTM) and the inherited
computed style, descending through containers (``g``/``a``/``svg``), baking each
shape's geometry into document space, resolving ``<use>`` references, and
skipping non-rendered subtrees (``defs``, gradients, ``clipPath`` ...).
"""

from __future__ import annotations

from pathlib import Path as FilePath
from xml.etree import ElementTree

from matplotlib.transforms import Affine2D

from .css import compute_css_declarations
from .gradients import parse_gradients
from .scene import Scene, Shape, Text
from .shapes import SHAPE_TAGS, shape_to_path, strip_ns
from .style import compute_style, declarations, is_hidden, style_to_kwargs
from .transform import parse_transform, parse_viewbox, viewbox_transform

__all__ = ["parse", "parse_string"]

_XLINK = "{http://www.w3.org/1999/xlink}href"
_CONTAINER_TAGS = frozenset({"g", "a", "svg"})
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
    }
)
_MAX_USE_DEPTH = 64


def _length(value):
    from .shapes import _length as _len

    return _len(value)


def parse(source: str | FilePath) -> Scene:
    """Parse an SVG file path into a :class:`Scene`."""
    root = ElementTree.parse(str(source)).getroot()
    return _build(root)


def parse_string(text: str) -> Scene:
    """Parse an SVG document from a string into a :class:`Scene`."""
    return _build(ElementTree.fromstring(text))


def _document_size(root, viewbox):
    """Return the document ``(width, height)`` in the baked path's coordinates."""
    width = _length(root.get("width")) if root.get("width") else None
    height = _length(root.get("height")) if root.get("height") else None
    if viewbox is not None:
        _, _, vb_w, vb_h = viewbox
        if width and height:
            return width, height
        return vb_w, vb_h
    return width or 0.0, height or 0.0


def _build(root) -> Scene:
    css_decls = compute_css_declarations(root)
    id_index = {e.get("id"): e for e in root.iter() if e.get("id")}
    viewbox = parse_viewbox(root.get("viewBox"))
    width = _length(root.get("width")) if root.get("width") else None
    height = _length(root.get("height")) if root.get("height") else None
    root_ctm = viewbox_transform(viewbox, width, height)

    doc_w, doc_h = _document_size(root, viewbox)
    scene = Scene(width=doc_w, height=doc_h, gradients=parse_gradients(root))

    ctx = _Context(scene=scene, css_decls=css_decls, id_index=id_index)
    ctx.walk(root, root_ctm, parent_style=None, depth=0)
    return scene


class _Context:
    """Carries the shared, read-only state through the recursive walk."""

    def __init__(self, scene, css_decls, id_index):
        self.scene = scene
        self.css_decls = css_decls
        self.id_index = id_index

    def walk(self, element, ctm: Affine2D, parent_style, depth: int) -> None:
        tag = strip_ns(element.tag)
        if tag in _SKIP_TAGS:
            return

        own = declarations(element.attrib, self.css_decls.get(id(element)))
        style = compute_style(own, parent_style)
        ctm = parse_transform(element.attrib.get("transform")) + ctm

        if is_hidden(style):
            return

        if tag in _CONTAINER_TAGS:
            for child in element:
                self.walk(child, ctm, style, depth)
            return

        if tag == "use":
            self._expand_use(element, ctm, style, depth)
            return

        if tag in SHAPE_TAGS:
            self._add_shape(element, ctm, style)
            return

        if tag in ("text",):
            self._add_text(element, ctm, style)
            return

    def _add_shape(self, element, ctm, style) -> None:
        path = shape_to_path(element.tag, element.attrib)
        if path is None or len(path.vertices) == 0:
            return
        self.scene.shapes.append(
            Shape(
                id=element.get("id", ""),
                path=path.transformed(ctm),
                style=dict(style),
                kwargs=style_to_kwargs(style),
            )
        )

    def _add_text(self, element, ctm, style) -> None:
        content = "".join(element.itertext()).strip()
        if not content:
            return
        self.scene.texts.append(
            Text(
                id=element.get("id", ""),
                x=_length(element.get("x")),
                y=_length(element.get("y")),
                content=content,
                style=dict(style),
                transform=ctm,
            )
        )

    def _expand_use(self, element, ctm, style, depth) -> None:
        if depth >= _MAX_USE_DEPTH:
            return
        href = element.get(_XLINK) or element.get("href")
        if not href or not href.startswith("#"):
            return
        target = self.id_index.get(href[1:])
        if target is None:
            return
        x, y = _length(element.get("x")), _length(element.get("y"))
        use_ctm = Affine2D().translate(x, y) + ctm
        # A referenced <symbol>/<svg> acts as a group; anything else is cloned.
        if strip_ns(target.tag) in ("symbol", "svg"):
            for child in target:
                self.walk(child, use_ctm, style, depth + 1)
        else:
            self.walk(target, use_ctm, style, depth + 1)
