"""Compute SVG presentation styles and map them to matplotlib artist kwargs.

The cascade implemented here is, in increasing priority:

1. inherited values from the parent element (for inheritable properties),
2. the element's presentation *attributes* (``fill="red"`` ...),
3. author ``<style>`` rules matched by :mod:`svg2mpl.css` (passed in already
   resolved per element),
4. the element's inline ``style="..."`` attribute.

:func:`style_to_kwargs` then turns a computed style into kwargs accepted by
:class:`~matplotlib.patches.PathPatch` / :class:`~matplotlib.collections.PathCollection`.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from .color import NONE, parse_paint

__all__ = [
    "INHERITED",
    "PRESENTATION_PROPERTIES",
    "parse_style_attr",
    "compute_style",
    "style_to_kwargs",
    "is_hidden",
]

# Properties that inherit from parent to child per the SVG spec (the subset we
# act on). Notably ``opacity``/``display`` do *not* inherit.
INHERITED = frozenset(
    {
        "fill",
        "fill-opacity",
        "fill-rule",
        "stroke",
        "stroke-opacity",
        "stroke-width",
        "stroke-linecap",
        "stroke-linejoin",
        "stroke-dasharray",
        "color",
        "visibility",
        "font-family",
        "font-size",
        "font-style",
        "font-weight",
        "text-anchor",
    }
)

# Presentation attributes we read off elements (in addition to inline style/CSS).
PRESENTATION_PROPERTIES = INHERITED | frozenset({"opacity", "display"})

_LINECAP = {"butt": "butt", "round": "round", "square": "projecting"}
_LINEJOIN = {"miter": "miter", "round": "round", "bevel": "bevel"}
_NUM_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?")


def parse_style_attr(text: str | None) -> dict[str, str]:
    """Parse an inline ``style="a:b; c:d"`` attribute into a dict."""
    out: dict[str, str] = {}
    if not text:
        return out
    for decl in text.split(";"):
        if ":" not in decl:
            continue
        name, _, value = decl.partition(":")
        name = name.strip().lower()
        if name:
            out[name] = value.strip()
    return out


def declarations(
    attrib: Mapping[str, str], css_decls: Mapping[str, str] | None = None
) -> dict[str, str]:
    """Collect an element's own declarations in cascade priority order.

    Presentation attributes < author ``<style>`` rules < inline ``style``.
    """
    own: dict[str, str] = {}
    for prop in PRESENTATION_PROPERTIES:
        if prop in attrib:
            own[prop] = attrib[prop]
    if css_decls:
        own.update(css_decls)
    own.update(parse_style_attr(attrib.get("style")))
    return own


def compute_style(
    own: Mapping[str, str], parent: Mapping[str, str] | None
) -> dict[str, str]:
    """Merge an element's own declarations with the inherited parent style."""
    computed: dict[str, str] = {}
    if parent:
        computed.update({k: v for k, v in parent.items() if k in INHERITED})
    computed.update(own)
    return computed


def is_hidden(style: Mapping[str, str]) -> bool:
    """Return whether the element should not be drawn at all."""
    return style.get("display", "").strip().lower() == "none" or style.get(
        "visibility", ""
    ).strip().lower() in ("hidden", "collapse")


def _clamp01(x: float) -> float:
    return min(max(x, 0.0), 1.0)


def _float(value: str | None, default: float) -> float:
    if value is None:
        return default
    m = _NUM_RE.match(value.strip())
    return float(m.group()) if m else default


def _opacity(value: str | None, default: float = 1.0) -> float:
    if value is None:
        return default
    v = value.strip()
    if v.endswith("%"):
        return _clamp01(_float(v[:-1], default * 100.0) / 100.0)
    return _clamp01(_float(v, default))


def _apply_alpha(color, alpha: float):
    """Fold ``alpha`` into a color, leaving ``none`` and ``url(...)`` untouched."""
    if color == NONE or (isinstance(color, str) and color.lower().startswith("url(")):
        return color
    from matplotlib.colors import to_rgba

    return to_rgba(color, alpha)


def _dashes(value: str | None):
    if not value or value.strip().lower() == "none":
        return None
    nums = [float(m) for m in _NUM_RE.findall(value)]
    if not nums:
        return None
    if len(nums) % 2:  # odd count is repeated per SVG spec
        nums = nums * 2
    return (0.0, nums)


def style_to_kwargs(style: Mapping[str, str]) -> dict[str, Any]:
    """Convert a computed style into matplotlib artist keyword arguments.

    The common case (no per-paint opacity, no dashes/caps/joins) yields exactly
    ``{facecolor, edgecolor, linewidth, alpha}`` so that homogeneous drawings
    can be batched into a single ``PathCollection``. Extra keys are added only
    when the corresponding SVG properties are present.
    """
    current_color = parse_paint(style.get("color"), default="black")
    facecolor = parse_paint(
        style.get("fill"), current_color=current_color, default="black"
    )
    edgecolor = parse_paint(
        style.get("stroke"), current_color=current_color, default=NONE
    )
    linewidth = _float(style.get("stroke-width"), 1.0)

    opacity = _opacity(style.get("opacity"))
    fill_opacity = _opacity(style.get("fill-opacity"))
    stroke_opacity = _opacity(style.get("stroke-opacity"))

    kwargs: dict[str, Any] = {
        "facecolor": facecolor,
        "edgecolor": edgecolor,
        "linewidth": linewidth,
    }
    if fill_opacity == 1.0 and stroke_opacity == 1.0:
        # Simple, batchable case: element opacity maps to a single patch alpha.
        kwargs["alpha"] = opacity
    else:
        # Per-paint opacity must be baked into the colors (a single patch alpha
        # would clobber them), so this element falls off the fast path.
        kwargs["facecolor"] = _apply_alpha(facecolor, opacity * fill_opacity)
        kwargs["edgecolor"] = _apply_alpha(edgecolor, opacity * stroke_opacity)

    if "stroke-linecap" in style:
        kwargs["capstyle"] = _LINECAP.get(style["stroke-linecap"].strip().lower())
    if "stroke-linejoin" in style:
        kwargs["joinstyle"] = _LINEJOIN.get(style["stroke-linejoin"].strip().lower())
    dashes = _dashes(style.get("stroke-dasharray"))
    if dashes is not None:
        kwargs["linestyle"] = dashes

    return {k: v for k, v in kwargs.items() if v is not None}
