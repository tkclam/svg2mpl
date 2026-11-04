"""Parse SVG paint/color values into matplotlib-friendly forms.

matplotlib already understands named colors and ``#rgb``/``#rrggbb`` hex, so the
job here is mostly to recognise the SVG-specific spellings (``rgb()``, ``hsl()``,
``none``, ``currentColor``, ``url(#id)``) and normalise them.
"""

from __future__ import annotations

import re
from colorsys import hls_to_rgb

__all__ = [
    "NONE",
    "parse_color",
    "parse_paint",
    "parse_url_ref",
    "split_paint",
    "to_grayscale",
]

# Sentinel for "no paint" -- matplotlib understands the string "none".
NONE = "none"

_URL_RE = re.compile(r"url\(\s*#([^)\s]+)\s*\)", re.IGNORECASE)
_RGB_RE = re.compile(r"rgba?\(([^)]*)\)", re.IGNORECASE)
_HSL_RE = re.compile(r"hsla?\(([^)]*)\)", re.IGNORECASE)
_NUM_RE = re.compile(r"[-+]?\d*\.?\d+%?")


def parse_url_ref(value: str | None) -> str | None:
    """Return the fragment id of a ``url(#id)`` reference, else ``None``."""
    if not value:
        return None
    m = _URL_RE.search(value)
    return m.group(1) if m else None


def split_paint(value: str | None) -> tuple[str | None, str | None]:
    """Split a paint into ``(url_ref_id, fallback)``.

    ``"url(#g) red"`` -> ``("g", "red")``; ``"red"`` -> ``(None, "red")``.
    """
    if not value:
        return None, None
    value = value.strip()
    m = _URL_RE.match(value) if value.lower().startswith("url(") else None
    if m:
        rest = value[m.end() :].strip()
        return m.group(1), (rest or None)
    if value.lower().startswith("url("):
        # url() that is not a local fragment reference (external documents).
        end = value.find(")")
        rest = value[end + 1 :].strip() if end >= 0 else ""
        return "", (rest or None)
    return None, value


def _channel(token: str) -> float:
    """Parse one ``rgb()`` channel (``0-255`` or ``0%-100%``) to ``0-1``."""
    token = token.strip()
    if token.endswith("%"):
        return min(max(float(token[:-1]) / 100.0, 0.0), 1.0)
    return min(max(float(token) / 255.0, 0.0), 1.0)


def parse_paint(
    value: str | None,
    *,
    current_color: str | tuple = "black",
    default: str = NONE,
):
    """Normalise an SVG paint value to something matplotlib can consume.

    Parameters
    ----------
    value : str or None
        A ``fill``/``stroke`` value: a named color, hex, ``rgb()``/``rgba()``,
        ``hsl()``/``hsla()``, ``none``, ``currentColor`` or ``url(#id)``.
    current_color : color, optional
        The value substituted for ``currentColor`` (the inherited ``color``
        property). Defaults to ``"black"``.
    default : str, optional
        Returned when ``value`` is ``None`` or empty.

    Returns
    -------
    str or tuple
        ``"none"``, a matplotlib color string, or an ``(r, g, b)`` /
        ``(r, g, b, a)`` tuple in ``0-1``. ``url(#id)`` references are returned
        verbatim so the caller can resolve the gradient.
    """
    if value is None:
        return default
    value = value.strip()
    if not value:
        return default
    low = value.lower()
    if low == "none":
        return NONE
    if low == "currentcolor":
        return current_color
    if low.startswith("url("):
        return value  # gradient/pattern reference, resolved by the caller
    return parse_color(value, current_color=current_color, default=default)


def _alpha_token(a: str) -> float:
    a = a.strip()
    alpha = float(a[:-1]) / 100.0 if a.endswith("%") else float(a)
    return min(max(alpha, 0.0), 1.0)


def parse_color(
    value: str | None,
    *,
    current_color: str | tuple = "black",
    default=NONE,
):
    """Parse a CSS/SVG ``<color>`` (no ``url()``) into a matplotlib color.

    Invalid colors yield ``default`` rather than raising, mirroring how a
    browser ignores an invalid declaration.
    """
    if value is None:
        return default
    value = value.strip()
    if not value:
        return default
    # Drop an SVG 1.1 ``icc-color(...)`` suffix: the sRGB fallback comes first.
    if "icc-color" in value.lower():
        value = value[: value.lower().index("icc-color")].strip()
    low = value.lower()
    if low == "none":
        return NONE
    if low == "transparent":
        return (0.0, 0.0, 0.0, 0.0)
    if low == "currentcolor":
        return current_color
    try:
        return _parse_color(value, low)
    except (ValueError, IndexError):
        return default


def _parse_color(value: str, low: str):

    m = _RGB_RE.match(value)
    if m:
        parts = [p for p in re.split(r"[,\s/]+", m.group(1).strip()) if p]
        if len(parts) >= 3:
            r, g, b = (_channel(p) for p in parts[:3])
            if len(parts) >= 4:
                return (r, g, b, _alpha_token(parts[3]))
            return (r, g, b)
        raise ValueError(value)

    m = _HSL_RE.match(value)
    if m:
        parts = [p for p in re.split(r"[,\s/]+", m.group(1).strip()) if p]
        if len(parts) >= 3:
            h = _hue(parts[0])
            s = float(parts[1].rstrip("%")) / 100.0
            ln = float(parts[2].rstrip("%")) / 100.0
            r, g, b = hls_to_rgb(h % 1.0, ln, s)
            if len(parts) >= 4:
                return (r, g, b, _alpha_token(parts[3]))
            return (r, g, b)
        raise ValueError(value)

    from matplotlib.colors import is_color_like

    if low.startswith("#"):
        digits = low[1:]
        if len(digits) in (3, 4) and all(c in "0123456789abcdef" for c in digits):
            vals = [int(c * 2, 16) / 255.0 for c in digits]
            return tuple(vals)
        if len(digits) in (6, 8) and is_color_like(low):
            return low
        raise ValueError(value)
    # Named color: SVG/CSS names are case-insensitive; matplotlib knows CSS4.
    if is_color_like(low):
        return low
    raise ValueError(value)


def _hue(token: str) -> float:
    """Parse a CSS hue (``deg`` default, also ``rad``/``grad``/``turn``) to 0-1."""
    t = token.strip().lower()
    if t.endswith("deg"):
        return float(t[:-3]) / 360.0
    if t.endswith("grad"):
        return float(t[:-4]) / 400.0
    if t.endswith("rad"):
        import math

        return float(t[:-3]) / (2 * math.pi)
    if t.endswith("turn"):
        return float(t[:-4])
    return float(t) / 360.0


def to_grayscale(color) -> tuple[float, float, float]:
    """Convert a matplotlib color to its luminance-preserving gray equivalent."""
    from matplotlib.colors import to_rgb

    r, g, b = to_rgb(color)
    v = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return (v, v, v)
