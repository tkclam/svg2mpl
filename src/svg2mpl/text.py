"""Lay out SVG ``<text>`` into glyph outlines.

Text is converted to :class:`~matplotlib.path.Path` outlines using matplotlib's
font machinery, so it scales, rotates, clips and accepts gradients exactly like
any other shape. Supported:

* ``<tspan>``, ``<a>`` and ``<textPath>`` children with their own styles;
* per-character ``x``/``y``/``dx``/``dy``/``rotate`` lists;
* text chunks and ``text-anchor`` (``start``/``middle``/``end``);
* ``xml:space`` / ``white-space`` whitespace handling;
* ``letter-spacing``, ``word-spacing``, kerning, ``textLength`` with
  ``lengthAdjust``;
* ``baseline-shift`` (``sub``/``super``/lengths) and ``dominant-baseline`` /
  ``alignment-baseline``;
* ``text-decoration`` (``underline``/``overline``/``line-through``);
* ``font-family`` lists with generic families, ``font-weight``, ``font-style``,
  ``font-stretch`` and the ``font`` shorthand.

Fonts are matched against the fonts matplotlib knows about; missing families
fall back to the generic family or matplotlib's default sans-serif. Vertical
writing modes and bidirectional reordering are not implemented.

:func:`add_texts` renders :class:`~svg2mpl.scene.Text` records as matplotlib
``Text`` artists instead (the ``text_as_paths=False`` mode).
"""

from __future__ import annotations

import itertools
import logging
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties, findfont, get_font
from matplotlib.path import Path
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D

from ._xml import strip_ns
from .color import parse_paint
from .geometry import empty_path, flatten, is_empty, join_paths, rect_path
from .style import font_size
from .units import LengthContext, parse_length, parse_numbers, split_length

try:  # private, but the same layout matplotlib's own TextPath uses
    from matplotlib import _text_helpers
except ImportError:  # pragma: no cover
    _text_helpers = None

try:  # matplotlib >= 3.10
    from matplotlib.ft2font import LoadFlags

    _NO_HINTING = LoadFlags.NO_HINTING
except ImportError:  # pragma: no cover - older matplotlib
    from matplotlib.ft2font import (
        LOAD_NO_HINTING as _NO_HINTING,  # pyright: ignore[reportAttributeAccessIssue]
    )

__all__ = ["TextRun", "add_texts", "layout_text"]

_XML_SPACE = "{http://www.w3.org/XML/1998/namespace}space"
_ANCHOR_TO_HA = {"start": "left", "middle": "center", "end": "right"}
_GENERIC = {
    "serif": "serif",
    "sans-serif": "sans-serif",
    "monospace": "monospace",
    "cursive": "cursive",
    "fantasy": "fantasy",
    "system-ui": "sans-serif",
    "ui-sans-serif": "sans-serif",
    "ui-serif": "serif",
    "ui-monospace": "monospace",
    "ui-rounded": "sans-serif",
    "emoji": "sans-serif",
    "math": "serif",
}
_REF = 100.0  # reference size glyphs are generated at (then scaled)
_CONTENT_TAGS = {"tspan", "a", "textPath", "tref"}


@dataclass
class TextRun:
    """A styled run of glyph outlines in the text element's user space."""

    id: str
    style: dict
    path: Path
    element: ET.Element | None = None


@dataclass
class _Char:
    ch: str
    style: dict
    element: ET.Element
    entries: tuple
    textpath: ET.Element | None
    decorations: tuple
    shift: float
    preserve: bool
    x: float | None = None
    y: float | None = None
    dx: float = 0.0
    dy: float = 0.0
    rotate: float = 0.0


@dataclass
class _PosEntry:
    x: list | None = None
    y: list | None = None
    dx: list | None = None
    dy: list | None = None
    rotate: list | None = None
    count: int = 0
    element: ET.Element | None = None
    chars: list = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Fonts
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _known_families() -> frozenset:
    return frozenset(f.name.lower() for f in font_manager.fontManager.ttflist)


def _families(value: str | None) -> tuple[str, ...]:
    out = []
    for raw in (value or "").split(","):
        name = raw.strip().strip("'\"").strip()
        if not name:
            continue
        low = name.lower()
        if low in _GENERIC:
            out.append(_GENERIC[low])
        elif low in _known_families():
            out.append(name)
    if not any(f in _GENERIC.values() for f in out):
        out.append("sans-serif")
    return tuple(dict.fromkeys(out))


def _weight(value: str | None):
    v = (value or "normal").strip().lower()
    if v == "normal":
        return 400
    if v == "bold":
        return 700
    try:
        return int(float(v))
    except ValueError:
        return 400


def _font_key(style) -> tuple:
    fstyle = (style.get("font-style") or "normal").strip().lower()
    if fstyle not in ("normal", "italic", "oblique"):
        fstyle = "normal"
    stretch = (style.get("font-stretch") or "normal").strip().lower()
    return (
        _families(style.get("font-family")),
        _weight(style.get("font-weight")),
        fstyle,
        stretch,
    )


@lru_cache(maxsize=256)
def _props(key):
    families, weight, fstyle, stretch = key
    try:
        return FontProperties(
            family=list(families), weight=weight, style=fstyle, stretch=stretch
        )
    except ValueError:
        return FontProperties(family=list(families), weight=weight, style=fstyle)


@lru_cache(maxsize=256)
def _ft_font(key):
    # Families were pre-filtered to installed ones, but generic families can
    # still make findfont log "not found" for each rcParams candidate.
    logger = logging.getLogger("matplotlib.font_manager")
    level = logger.level
    logger.setLevel(logging.ERROR)
    try:
        return get_font(findfont(_props(key)))
    finally:
        logger.setLevel(level)


@lru_cache(maxsize=256)
def _metrics(key) -> dict:
    """Font metrics as fractions of the em size."""
    font = _ft_font(key)
    em = font.units_per_EM or 1000
    os2 = font.get_sfnt_table("OS/2") or {}
    x_height = os2.get("sxHeight") or 0.5 * em
    return {
        "asc": font.ascender / em,
        "desc": font.descender / em,
        "upos": font.underline_position / em,
        "uthick": max(font.underline_thickness / em, 0.02),
        "xh": x_height / em,
    }


@lru_cache(maxsize=4096)
def _glyph(key, ch: str) -> Path:
    """The outline of ``ch`` at size :data:`_REF`, baseline at y=0, y down."""
    if ch.isspace():
        return empty_path()
    try:
        p = TextPath((0, 0), ch, size=_REF, prop=_props(key))
    except (RuntimeError, ValueError):  # no usable glyph in any font
        return empty_path()
    return p.transformed(Affine2D().scale(1, -1))


def _layout_advances(font, text: str) -> list[float] | None:
    """Kerned advances from matplotlib's text layout (private API, so any
    mismatch falls back to unkerned advances)."""
    if _text_helpers is None:
        return None
    items = list(_text_helpers.layout(text, font))
    if len(items) != len(text):
        return None
    last = items[-1]
    # The field was renamed from ``glyph_idx`` to ``glyph_index`` in mpl 3.11.
    index = getattr(last, "glyph_index", getattr(last, "glyph_idx", None))
    if index is None:
        return None
    glyph = last.ft_object.load_glyph(index, flags=_NO_HINTING)
    xs = [it.x for it in items] + [last.x + glyph.linearHoriAdvance / 65536]
    return [b - a for a, b in itertools.pairwise(xs)]


@lru_cache(maxsize=4096)
def _advances(key, text: str) -> tuple[float, ...]:
    """Per-character advances (kerning included) at size :data:`_REF`."""
    font = _ft_font(key)
    font.set_size(_REF, 72)
    kerned = _layout_advances(font, text)
    if kerned is not None:
        return tuple(kerned)
    return tuple(_char_advance(font, ch) for ch in text)


def _char_advance(font, ch: str) -> float:
    try:
        glyph = font.load_char(ord(ch), flags=_NO_HINTING)
    except RuntimeError:  # glyph missing from this font
        return _REF * 0.5
    return glyph.linearHoriAdvance / 65536


# --------------------------------------------------------------------------- #
# Collection
# --------------------------------------------------------------------------- #
def _lengths(value, ctx, axis):
    if value is None:
        return None
    tokens = [t for t in re.split(r"[\s,]+", value.strip()) if t]
    out = [parse_length(t, 0.0, axis=axis, ctx=ctx) for t in tokens]
    return out or None


def _preserve(el, style, parent_preserve):
    ws = (style.get("white-space") or "").strip().lower()
    if ws in ("pre", "pre-wrap", "break-spaces", "pre-line"):
        return True
    if ws == "normal":
        return False
    sp = el.get(_XML_SPACE)
    if sp == "preserve":
        return True
    if sp == "default":
        return False
    return parent_preserve


def _baseline_shift(style, size) -> float:
    v = (style.get("baseline-shift") or "").strip().lower()
    if not v or v == "baseline":
        return 0.0
    if v == "sub":
        return -0.2 * size
    if v == "super":
        return 0.4 * size
    parts = split_length(v)
    if parts is None:
        return 0.0
    if parts[1] == "%":
        return parts[0] / 100.0 * size
    return parse_length(v, 0.0, ctx=LengthContext(font_size=size))


def _decorations(style):
    v = (style.get("text-decoration") or "").lower()
    return tuple(k for k in ("underline", "overline", "line-through") if k in v)


class _Collector:
    def __init__(self, child_style, ctx_for):
        self.child_style = child_style
        self.ctx_for = ctx_for
        self.chars: list[_Char] = []
        self.stack: list[_PosEntry] = []
        self.entries: list[_PosEntry] = []

    def collect(self, el, style, textpath, decorations, shift, preserve):
        ctx = self.ctx_for(style)
        entry = _PosEntry(
            x=_lengths(el.get("x"), ctx, "x")
            if strip_ns(el.tag) != "textPath"
            else None,
            y=_lengths(el.get("y"), ctx, "y")
            if strip_ns(el.tag) != "textPath"
            else None,
            dx=_lengths(el.get("dx"), ctx, "x"),
            dy=_lengths(el.get("dy"), ctx, "y"),
            rotate=parse_numbers(el.get("rotate")) or None,
            element=el,
        )
        self.stack.append(entry)
        self.entries.append(entry)
        own_decor = _decorations(style)
        if own_decor:
            decorations = decorations + tuple((k, style) for k in own_decor)
        shift = shift + _baseline_shift(style, font_size(style))
        preserve = _preserve(el, style, preserve)
        self._add(el.text, el, style, textpath, decorations, shift, preserve)
        for child in el:
            if isinstance(child.tag, str):
                tag = strip_ns(child.tag)
                if tag in _CONTENT_TAGS and tag != "tref":
                    cs = self.child_style(child, style)
                    if (cs.get("display") or "").strip().lower() != "none":
                        tp = child if tag == "textPath" else textpath
                        self.collect(child, cs, tp, decorations, shift, preserve)
            self._add(child.tail, el, style, textpath, decorations, shift, preserve)
        self.stack.pop()

    def _add(self, text, el, style, textpath, decorations, shift, preserve):
        if not text:
            return
        entries = tuple(self.stack)
        for ch in text:
            self.chars.append(
                _Char(ch, style, el, entries, textpath, decorations, shift, preserve)
            )


def _collapse(chars: list[_Char]) -> list[_Char]:
    out: list[_Char] = []
    for c in chars:
        if c.preserve:
            if c.ch in "\n\r\t":
                c.ch = " "
            out.append(c)
            continue
        if c.ch in "\n\r":
            continue
        if c.ch == "\t":
            c.ch = " "
        if c.ch == " " and (not out or out[-1].ch == " "):
            continue
        out.append(c)
    # Strip trailing (non-preserved) spaces.
    while out and out[-1].ch == " " and not out[-1].preserve:
        out.pop()
    return out


def _assign_positions(chars: list[_Char]) -> None:
    for c in chars:
        inner = list(reversed(c.entries))
        for attr in ("x", "y", "dx", "dy"):
            for e in inner:
                lst = getattr(e, attr)
                if lst is not None and e.count < len(lst):
                    setattr(c, attr, lst[e.count])
                    break
        for e in inner:
            if e.rotate is not None:
                c.rotate = e.rotate[min(e.count, len(e.rotate) - 1)]
                break
        for e in c.entries:
            e.count += 1


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #
@dataclass
class _Glyph:
    char: _Char
    key: tuple
    size: float
    x: float
    y: float
    advance: float
    chunk: int
    on_path: ET.Element | None = None


def _baseline_offset(style, key, size) -> float:
    """Downward offset of the alphabetic baseline for the alignment baseline."""
    v = style.get("alignment-baseline") or style.get("dominant-baseline") or "auto"
    v = v.strip().lower()
    m = _metrics(key)
    if v == "middle":
        return m["xh"] * size / 2
    if v == "central":
        return (m["asc"] + m["desc"]) * size / 2
    if v == "hanging":
        return 0.8 * m["asc"] * size
    if v in ("text-before-edge", "text-top", "before-edge"):
        return m["asc"] * size
    if v in ("text-after-edge", "text-bottom", "after-edge", "ideographic"):
        return m["desc"] * size
    if v == "mathematical":
        return 0.5 * m["asc"] * size
    return 0.0


def _letter_spacing(style, size) -> float:
    v = (style.get("letter-spacing") or "normal").strip().lower()
    if v == "normal":
        return 0.0
    return parse_length(v, 0.0, ctx=LengthContext(font_size=size))


def _word_spacing(style, size) -> float:
    v = (style.get("word-spacing") or "normal").strip().lower()
    if v == "normal":
        return 0.0
    return parse_length(v, 0.0, ctx=LengthContext(font_size=size))


class _PathMeasure:
    """Arc-length parameterization of a (flattened) path."""

    def __init__(self, path: Path):
        # Subpaths are chained; the jump between them (a moveto) has no length.
        rings = flatten(path)
        self.pts = np.vstack(rings) if rings else np.empty((0, 2))
        if len(self.pts) >= 2:
            seg = np.diff(self.pts, axis=0)
            self.seglen = np.hypot(seg[:, 0], seg[:, 1])
            ends = np.cumsum([len(r) for r in rings])[:-1]
            self.seglen[ends - 1] = 0.0  # segment from one ring's end to the next
            self.cum = np.concatenate([[0.0], np.cumsum(self.seglen)])
        else:
            self.seglen = np.empty(0)
            self.cum = np.zeros(1)
        self.length = float(self.cum[-1])

    def at(self, s: float):
        if self.length <= 0 or s < 0 or s > self.length:
            return None
        i = int(np.searchsorted(self.cum, s, side="right") - 1)
        i = min(max(i, 0), len(self.seglen) - 1)
        while i < len(self.seglen) - 1 and self.seglen[i] == 0:
            i += 1
        t = (s - self.cum[i]) / self.seglen[i] if self.seglen[i] else 0.0
        p = self.pts[i] + t * (self.pts[i + 1] - self.pts[i])
        d = self.pts[i + 1] - self.pts[i]
        return p, math.degrees(math.atan2(d[1], d[0]))


def layout_text(element, style, child_style, ctx_for, resolve_textpath):
    """Lay out a ``<text>`` element.

    Parameters
    ----------
    element : Element
        The ``<text>`` element.
    style : dict
        Its computed style.
    child_style : callable
        ``child_style(child, parent_style)`` -> computed style of a child.
    ctx_for : callable
        ``ctx_for(style)`` -> :class:`LengthContext` for that style.
    resolve_textpath : callable
        ``resolve_textpath(textpath_element)`` -> ``Path`` in the text's user
        space (or ``None``).

    Returns
    -------
    (list[TextRun], str, (x, y))
        The glyph/decoration runs, the collapsed text content, and the first
        character's anchor.
    """
    col = _Collector(child_style, ctx_for)
    col.collect(element, style, None, (), 0.0, _preserve(element, style, False))
    chars = _collapse(col.chars)
    # Leading spaces are stripped too (non-preserved).
    while chars and chars[0].ch == " " and not chars[0].preserve:
        chars.pop(0)
    if not chars:
        return [], "", (0.0, 0.0)
    _assign_positions(chars)

    advances = _char_advances(chars)
    measures: dict[int, _PathMeasure | None] = {}
    glyphs: list[_Glyph] = []
    chunks: list[dict] = []
    cx = cy = 0.0
    current_tp = None
    for i, c in enumerate(chars):
        key = _font_key(c.style)
        size = font_size(c.style)
        tp = c.textpath
        new_chunk = i == 0 or c.x is not None or c.y is not None
        if tp is not current_tp:
            new_chunk = True
            if tp is not None:
                if id(tp) not in measures:
                    p = resolve_textpath(tp)
                    measures[id(tp)] = _PathMeasure(p) if p is not None else None
                m = measures[id(tp)]
                off = tp.get("startOffset")
                parts = split_length(off) if off else None
                if parts and parts[1] == "%" and m is not None:
                    cx = parts[0] / 100.0 * m.length
                else:
                    cx = parse_length(off, 0.0, ctx=ctx_for(c.style)) if off else 0.0
                cy = 0.0
            current_tp = tp
        if c.x is not None:
            cx = c.x
        if c.y is not None:
            cy = c.y
        cx += c.dx
        cy += c.dy
        if new_chunk:
            anchor = (c.style.get("text-anchor") or "start").strip().lower()
            chunks.append({"anchor": anchor, "start": cx, "glyphs": [], "tp": tp})
        adv = advances[i]
        g = _Glyph(c, key, size, cx, cy, adv, len(chunks) - 1, on_path=tp)
        glyphs.append(g)
        chunks[-1]["glyphs"].append(g)
        spacing = _letter_spacing(c.style, size)
        if c.ch == " ":
            spacing += _word_spacing(c.style, size)
        cx += adv + spacing

    _apply_text_length(element, ctx_for(style), glyphs)

    for chunk in chunks:
        gs = chunk["glyphs"]
        if not gs:
            continue
        start = gs[0].x
        end = max(g.x + g.advance for g in gs)
        width = end - start
        shift = {"middle": -width / 2, "end": -width}.get(chunk["anchor"], 0.0)
        if shift:
            for g in gs:
                g.x += shift

    runs = _build_runs(element, glyphs, measures)
    content = "".join(c.ch for c in chars)
    first = (chunks[0]["start"], glyphs[0].y) if glyphs else (0.0, 0.0)
    return runs, content, first


def _char_advances(chars: list[_Char]) -> list[float]:
    """Advance of each character, with pair kerning inside same-font runs."""
    out: list[float] = []
    i = 0
    while i < len(chars):
        key, size = _font_key(chars[i].style), font_size(chars[i].style)
        j = i
        while (
            j + 1 < len(chars)
            and chars[j + 1].x is None
            and _font_key(chars[j + 1].style) == key
            and font_size(chars[j + 1].style) == size
        ):
            j += 1
        text = "".join(c.ch for c in chars[i : j + 1])
        out.extend(a * size / _REF for a in _advances(key, text))
        i = j + 1
    return out


def _apply_text_length(element, ctx, glyphs):
    target = element.get("textLength")
    if not target or len(glyphs) < 1:
        return
    length = parse_length(target, 0.0, axis="x", ctx=ctx)
    if length <= 0:
        return
    start = glyphs[0].x
    natural = max(g.x + g.advance for g in glyphs) - start
    if natural <= 0:
        return
    if (element.get("lengthAdjust") or "spacing") == "spacingAndGlyphs":
        k = length / natural
        for g in glyphs:
            g.x = start + (g.x - start) * k
            g.advance *= k
            g.char.style = dict(g.char.style)
            g.char.style["_xscale"] = repr(k)
    elif len(glyphs) > 1:
        extra = (length - natural) / (len(glyphs) - 1)
        for n, g in enumerate(glyphs):
            g.x += n * extra


def _glyph_transform(g: _Glyph, measures) -> Affine2D | None:
    scale = g.size / _REF
    xscale = float(g.char.style.get("_xscale", 1.0))
    base = _baseline_offset(g.char.style, g.key, g.size) - g.char.shift
    t = Affine2D().scale(scale * xscale, scale)
    if g.on_path is not None:
        m = measures.get(id(g.on_path))
        if m is None:
            return None
        mid = g.x + g.advance / 2
        hit = m.at(mid)
        if hit is None:
            return None
        (px, py), ang = hit
        t.translate(-g.advance / 2, base + g.y)
        if g.char.rotate:
            t.rotate_deg(g.char.rotate)
        return t.rotate_deg(ang).translate(px, py)
    if g.char.rotate:
        t.rotate_deg(g.char.rotate)
    return t.translate(g.x, g.y + base)


def _build_runs(element, glyphs, measures) -> list[TextRun]:
    text_id = element.get("id", "")
    runs: list[TextRun] = []
    current = None
    pieces: list[Path] = []

    def flush():
        if current is not None and pieces:
            el, style = current
            runs.append(
                TextRun(
                    id=el.get("id") or text_id,
                    style=style,
                    path=join_paths(pieces),
                    element=el,
                )
            )

    for g in glyphs:
        t = _glyph_transform(g, measures)
        if t is None:
            continue
        glyph = _glyph(g.key, g.char.ch)
        k = (g.char.element, id(g.char.style))
        if current is None or (current[0], id(current[1])) != k:
            flush()
            current = (g.char.element, g.char.style)
            pieces = []
        if not is_empty(glyph):
            pieces.append(glyph.transformed(t))
    flush()

    # Decorations: one bar per (decoration, declaring style, chunk) span.
    spans: dict = {}
    for g in glyphs:
        if g.on_path is not None:
            continue
        for kind, dstyle in g.char.decorations:
            spans.setdefault((kind, id(dstyle), g.chunk), (dstyle, []))[1].append(g)
    for (kind, _sid, _chunk), (dstyle, gs) in spans.items():
        size = font_size(dstyle)
        key = _font_key(dstyle)
        m = _metrics(key)
        x0 = min(g.x for g in gs)
        x1 = max(g.x + g.advance for g in gs)
        y = gs[0].y - gs[0].char.shift
        thick = m["uthick"] * size
        if kind == "underline":
            yc = y - m["upos"] * size
        elif kind == "overline":
            yc = y - m["asc"] * size + thick / 2
        else:
            yc = y - m["xh"] * size / 2
        runs.append(
            TextRun(
                id=element.get("id", ""),
                style=dstyle,
                path=rect_path(x0, yc - thick / 2, x1, yc + thick / 2),
                element=element,
            )
        )
    return runs


# --------------------------------------------------------------------------- #
# matplotlib Text artists (text_as_paths=False)
# --------------------------------------------------------------------------- #
def _rotation_deg(affine: Affine2D) -> float:
    """Extract the rotation angle (degrees) of an affine, ignoring reflection."""
    a, b = affine.get_matrix()[0, 0], affine.get_matrix()[1, 0]
    return math.degrees(math.atan2(b, a))


def add_texts(ax, texts, *, placement, base, zorder=3):
    """Render text records as matplotlib ``Text`` artists.

    ``placement`` maps document space to data coordinates (y-flip and
    placement) and ``base`` is usually ``ax.transData``. Glyph sizes are in
    points and do not follow the drawing's scale; the default path rendering
    keeps faithful sizes.
    """
    artists = []
    for text in texts:
        x, y = (text.transform + placement).transform((text.x, text.y))
        style = text.style
        color = parse_paint(style.get("fill"), default="black")
        if color == "none" or (isinstance(color, str) and color.startswith("url(")):
            color = "black"
        ha = _ANCHOR_TO_HA.get((style.get("text-anchor") or "start").strip(), "left")
        size = font_size(style)
        artist = ax.text(
            x,
            y,
            text.content,
            color=color,
            ha=ha,
            va="baseline",
            # Glyphs stay upright: combine placement and content rotation.
            rotation=_rotation_deg(text.transform + placement),
            rotation_mode="anchor",
            transform=base,
            zorder=zorder,
            fontproperties=_props(_font_key(style)),
        )
        artist.set_fontsize(size)
        artists.append(artist)
    return artists
