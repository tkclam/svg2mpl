"""Compute SVG presentation styles and map them to matplotlib artist kwargs.

The cascade implemented here is, in increasing priority:

1. inherited values from the parent element (for inheritable properties),
2. the element's presentation *attributes* (``fill="red"`` ...),
3. author ``<style>`` rules matched by :mod:`svg2mpl.css` (passed in already
   resolved per element),
4. the element's inline ``style="..."`` attribute,
5. ``!important`` author rules, then ``!important`` inline declarations.

The CSS-wide keywords ``inherit``, ``initial`` and ``unset`` are honored, the
``font`` and ``marker`` shorthands are expanded, and ``font-size`` is resolved to
an absolute size so ``em``/``%`` work for descendants.

:func:`style_to_kwargs` then turns a computed style into kwargs accepted by
:class:`~matplotlib.patches.PathPatch` / :class:`~matplotlib.collections.PathCollection`.
Stroke widths and dash lengths are expressed in the element's *user units*; the
parser scales them into document units and the renderer into points.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from .color import NONE, parse_color, parse_paint, with_alpha
from .units import DEFAULT_FONT_SIZE, LengthContext, parse_length, split_length

__all__ = [
    "INHERITED",
    "PRESENTATION_PROPERTIES",
    "parse_style_attr",
    "declarations",
    "compute_style",
    "style_to_kwargs",
    "is_hidden",
    "is_invisible",
    "font_size",
    "opacity",
]

# Properties that inherit from parent to child per the SVG/CSS specs.
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
        "stroke-miterlimit",
        "stroke-dasharray",
        "stroke-dashoffset",
        "color",
        "visibility",
        "font-family",
        "font-size",
        "font-style",
        "font-weight",
        "font-variant",
        "font-stretch",
        "text-anchor",
        "letter-spacing",
        "word-spacing",
        "direction",
        "writing-mode",
        "dominant-baseline",
        "clip-rule",
        "marker-start",
        "marker-mid",
        "marker-end",
        "paint-order",
        "image-rendering",
        "shape-rendering",
        "text-rendering",
        "color-interpolation",
        "color-interpolation-filters",
        "white-space",
        "pointer-events",
        "cursor",
    }
)

# Presentation attributes that do not inherit.
NON_INHERITED = frozenset(
    {
        "opacity",
        "display",
        "clip-path",
        "mask",
        "filter",
        "stop-color",
        "stop-opacity",
        "flood-color",
        "flood-opacity",
        "lighting-color",
        "overflow",
        "baseline-shift",
        "alignment-baseline",
        "text-decoration",
        "mix-blend-mode",
        "isolation",
        "vector-effect",
        "mask-type",
        "unicode-bidi",
        "transform-origin",
    }
)

# Presentation attributes we read off elements (in addition to inline style/CSS).
# The ``font`` and ``marker`` shorthands are CSS-only, as in browsers.
PRESENTATION_PROPERTIES = INHERITED | NON_INHERITED

_LINECAP = {"butt": "butt", "round": "round", "square": "projecting"}
_LINEJOIN = {
    "miter": "miter",
    "miter-clip": "miter",
    "arcs": "miter",
    "round": "round",
    "bevel": "bevel",
}
_NUM_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?")
_IMPORTANT_RE = re.compile(r"\s*!\s*important\s*$", re.IGNORECASE)

_FONT_SIZE_KEYWORDS = {
    "xx-small": 9.0,
    "x-small": 10.0,
    "small": 13.0,
    "medium": 16.0,
    "large": 18.0,
    "x-large": 24.0,
    "xx-large": 32.0,
    "xxx-large": 48.0,
}
_FONT_STYLES = {"italic", "oblique"}
_FONT_WEIGHTS = {"bold", "bolder", "lighter"} | {str(w) for w in range(100, 1000, 100)}
_FONT_STRETCH = {
    "ultra-condensed",
    "extra-condensed",
    "condensed",
    "semi-condensed",
    "semi-expanded",
    "expanded",
    "extra-expanded",
    "ultra-expanded",
}


def parse_style_attr(text: str | None) -> dict[str, str]:
    """Parse an inline ``style="a:b; c:d"`` attribute into a dict."""
    out: dict[str, str] = {}
    if not text:
        return out
    # Strip CSS comments first.
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    for decl in _split_decls(text):
        if ":" not in decl:
            continue
        name, _, value = decl.partition(":")
        name = name.strip().lower()
        if name:
            out[name] = value.strip()
    return out


def _split_decls(text: str) -> list[str]:
    """Split on ``;`` outside parentheses/quotes (``url(data:...;base64)``)."""
    out, depth, quote, buf = [], 0, "", []
    for ch in text:
        if quote:
            if ch == quote:
                quote = ""
        elif ch in "'\"":
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(depth - 1, 0)
        elif ch == ";" and depth == 0:
            out.append("".join(buf))
            buf = []
            continue
        buf.append(ch)
    out.append("".join(buf))
    return out


def _split_important(decls: Mapping[str, str] | None):
    normal: dict[str, str] = {}
    important: dict[str, str] = {}
    for k, v in (decls or {}).items():
        if _IMPORTANT_RE.search(v):
            important[k] = _IMPORTANT_RE.sub("", v).strip()
        else:
            normal[k] = v
    return normal, important


def _expand_shorthands(decls: dict[str, str]) -> dict[str, str]:
    """Expand ``marker`` and ``font`` in place (keeping declaration order)."""
    if "marker" not in decls and "font" not in decls:
        return decls
    out: dict[str, str] = {}
    for k, v in decls.items():
        if k == "marker":
            for side in ("marker-start", "marker-mid", "marker-end"):
                out[side] = v
        elif k == "font":
            out.update(_parse_font_shorthand(v))
        else:
            out[k] = v
    return out


def _parse_font_shorthand(value: str) -> dict[str, str]:
    value = value.strip()
    if value.lower() in ("inherit", "initial", "unset"):
        return {
            p: value for p in ("font-style", "font-weight", "font-size", "font-family")
        }
    out = {
        "font-style": "normal",
        "font-variant": "normal",
        "font-weight": "normal",
        "font-stretch": "normal",
    }
    tokens = re.findall(r"\"[^\"]*\"|'[^']*'|[^\s]+", value)
    i = 0
    while i < len(tokens):
        t = tokens[i].lower()
        if t in _FONT_STYLES:
            out["font-style"] = t
        elif t in _FONT_WEIGHTS:
            out["font-weight"] = t
        elif t == "small-caps":
            out["font-variant"] = t
        elif t in _FONT_STRETCH:
            out["font-stretch"] = t
        elif t == "normal":
            pass
        else:
            break
        i += 1
    if i < len(tokens):
        size = tokens[i]
        if "/" in size:  # font-size/line-height
            size = size.split("/")[0]
        out["font-size"] = size
        i += 1
        if i < len(tokens) and tokens[i].startswith("/"):
            i += 1
        family = " ".join(tokens[i:]).strip()
        if family:
            out["font-family"] = family
    return out


def declarations(
    attrib: Mapping[str, str], css_decls: Mapping[str, str] | None = None
) -> dict[str, str]:
    """Collect an element's own declarations in cascade priority order.

    Presentation attributes < author ``<style>`` rules < inline ``style`` <
    ``!important`` author rules < ``!important`` inline declarations.
    """
    own: dict[str, str] = {}
    for prop in PRESENTATION_PROPERTIES:
        if prop in attrib:
            own[prop] = attrib[prop]
    own = _expand_shorthands(own)
    css_normal, css_important = _split_important(css_decls)
    inline_normal, inline_important = _split_important(
        parse_style_attr(attrib.get("style"))
    )
    for layer in (css_normal, inline_normal, css_important, inline_important):
        if layer:
            own.update(_expand_shorthands(dict(layer)))
    return own


def font_size(style: Mapping[str, str] | None) -> float:
    """The computed (absolute) font size of a style, in user units."""
    if not style:
        return DEFAULT_FONT_SIZE
    parts = split_length(style.get("font-size"))
    return parts[0] if parts else DEFAULT_FONT_SIZE


def _resolve_font_size(value: str, parent_size: float) -> float:
    v = value.strip().lower()
    if v in _FONT_SIZE_KEYWORDS:
        return _FONT_SIZE_KEYWORDS[v]
    if v == "smaller":
        return parent_size / 1.2
    if v == "larger":
        return parent_size * 1.2
    parts = split_length(v)
    if parts is None:
        return parent_size
    num, unit = parts
    if unit == "%":
        return num / 100.0 * parent_size
    if unit in ("em", "rem"):
        return num * parent_size
    if unit == "ex":
        return num * parent_size * 0.5
    return parse_length(v, parent_size)


def _resolve_font_weight(value: str, parent: str | None) -> str:
    v = value.strip().lower()
    if v not in ("bolder", "lighter"):
        return v
    try:
        base = int({"normal": "400", "bold": "700"}.get(parent or "400", parent))
    except (TypeError, ValueError):
        base = 400
    if v == "bolder":
        return str(400 if base < 400 else 700 if base < 600 else 900)
    return str(100 if base < 600 else 400 if base < 800 else 700)


def compute_style(
    own: Mapping[str, str], parent: Mapping[str, str] | None
) -> dict[str, str]:
    """Merge an element's own declarations with the inherited parent style."""
    computed: dict[str, str] = {}
    if parent:
        computed.update({k: v for k, v in parent.items() if k in INHERITED})
    parent_size = font_size(parent)
    for k, v in own.items():
        keyword = v.strip().lower() if isinstance(v, str) else v
        if keyword == "inherit":
            if parent and k in parent:
                computed[k] = parent[k]
            else:
                computed.pop(k, None)
            continue
        if keyword == "initial" or (keyword == "unset" and k not in INHERITED):
            computed.pop(k, None)
            continue
        if keyword == "unset":
            continue  # inherited value (already copied) stays
        computed[k] = v
    if "font-size" in own and own["font-size"].strip().lower() != "inherit":
        if "font-size" in computed:
            size = _resolve_font_size(computed["font-size"], parent_size)
            computed["font-size"] = repr(float(size))
    if "font-weight" in computed and "font-weight" in own:
        computed["font-weight"] = _resolve_font_weight(
            computed["font-weight"], (parent or {}).get("font-weight")
        )
    return computed


def is_hidden(style: Mapping[str, str]) -> bool:
    """Whether the element (and its subtree) is not rendered (``display:none``)."""
    return style.get("display", "").strip().lower() == "none"


def is_invisible(style: Mapping[str, str]) -> bool:
    """Whether the element itself is not painted (``visibility:hidden``)."""
    return style.get("visibility", "").strip().lower() in ("hidden", "collapse")


def _clamp01(x: float) -> float:
    return min(max(x, 0.0), 1.0)


def _float(value: str | None, default: float) -> float:
    if value is None:
        return default
    m = _NUM_RE.match(value.strip())
    return float(m.group()) if m else default


def opacity(value: str | None, default: float = 1.0) -> float:
    """Parse an opacity (number or percentage), clamped to ``[0, 1]``."""
    if value is None:
        return default
    v = value.strip()
    if v.endswith("%"):
        return _clamp01(_float(v[:-1], default * 100.0) / 100.0)
    return _clamp01(_float(v, default))


def _is_url(color) -> bool:
    return isinstance(color, str) and color.lower().startswith("url(")


def _own_alpha(color) -> float:
    if color == NONE or _is_url(color):
        return 1.0
    from matplotlib.colors import to_rgba

    return to_rgba(color)[3]


def dash_array(value: str | None, ctx: LengthContext | None = None):
    """Parse ``stroke-dasharray`` into a list of user-unit lengths, or ``None``."""
    if not value or value.strip().lower() == "none":
        return None
    tokens = [t for t in re.split(r"[\s,]+", value.strip()) if t]
    nums = [parse_length(t, float("nan"), ctx=ctx) for t in tokens]
    if not nums or any(n != n or n < 0 for n in nums) or sum(nums) <= 0:
        return None
    if len(nums) % 2:  # odd count is repeated per SVG spec
        nums = nums * 2
    return nums


def _dashes(value: str | None, offset: float = 0.0, ctx=None):
    nums = dash_array(value, ctx)
    return None if nums is None else (offset, nums)


def style_to_kwargs(
    style: Mapping[str, str],
    ctx: LengthContext | None = None,
    group_opacity: float = 1.0,
) -> dict[str, Any]:
    """Convert a computed style into matplotlib artist keyword arguments.

    The common case (no per-paint opacity, no dashes/caps/joins) yields exactly
    ``{facecolor, edgecolor, linewidth, alpha}`` so that homogeneous drawings
    can be batched into a single ``PathCollection``. Extra keys are added only
    when the corresponding SVG properties are present.

    ``linewidth`` and dash lengths are in the element's user units.
    ``group_opacity`` is the product of the ancestors' ``opacity``.
    """
    ctx = ctx or LengthContext(font_size=font_size(style))
    current_color = parse_color(style.get("color"), default="black")
    if current_color == NONE:
        current_color = "black"
    facecolor = parse_paint(
        style.get("fill"), current_color=current_color, default="black"
    )
    edgecolor = parse_paint(
        style.get("stroke"), current_color=current_color, default=NONE
    )
    linewidth = parse_length(style.get("stroke-width"), 1.0, ctx=ctx)
    if linewidth <= 0:
        edgecolor, linewidth = NONE, 0.0

    element_opacity = opacity(style.get("opacity")) * group_opacity
    fill_opacity = opacity(style.get("fill-opacity"))
    stroke_opacity = opacity(style.get("stroke-opacity"))

    kwargs: dict[str, Any] = {
        "facecolor": facecolor,
        "edgecolor": edgecolor,
        "linewidth": linewidth,
    }
    if (
        fill_opacity == 1.0
        and stroke_opacity == 1.0
        and _own_alpha(facecolor) == 1.0
        and _own_alpha(edgecolor) == 1.0
    ):
        # Simple, batchable case: element opacity maps to a single patch alpha.
        kwargs["alpha"] = element_opacity
    else:
        # Per-paint opacity must be baked into the colors (a single patch alpha
        # would clobber them), so this element falls off the fast path.
        kwargs["facecolor"] = with_alpha(facecolor, element_opacity * fill_opacity)
        kwargs["edgecolor"] = with_alpha(edgecolor, element_opacity * stroke_opacity)

    if "stroke-linecap" in style:
        kwargs["capstyle"] = _LINECAP.get(style["stroke-linecap"].strip().lower())
    if "stroke-linejoin" in style:
        kwargs["joinstyle"] = _LINEJOIN.get(style["stroke-linejoin"].strip().lower())
    offset = parse_length(style.get("stroke-dashoffset"), 0.0, ctx=ctx)
    dashes = _dashes(style.get("stroke-dasharray"), offset, ctx)
    if dashes is not None:
        kwargs["linestyle"] = dashes

    return {k: v for k, v in kwargs.items() if v is not None}
