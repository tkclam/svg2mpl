"""Parse SVG paint/color values into matplotlib-friendly forms.

matplotlib already understands named colors and ``#rgb``/``#rrggbb`` hex, so the
job here is mostly to recognise the SVG-specific spellings (``rgb()``, ``hsl()``,
``none``, ``currentColor``, ``url(#id)``) and normalise them.
"""

from __future__ import annotations

import re
from colorsys import hls_to_rgb

__all__ = ["NONE", "parse_paint", "parse_url_ref", "to_grayscale"]

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

    m = _RGB_RE.match(value)
    if m:
        parts = [p for p in re.split(r"[,\s/]+", m.group(1).strip()) if p]
        if len(parts) >= 3:
            r, g, b = (_channel(p) for p in parts[:3])
            if len(parts) >= 4:
                a = parts[3]
                alpha = float(a[:-1]) / 100.0 if a.endswith("%") else float(a)
                return (r, g, b, min(max(alpha, 0.0), 1.0))
            return (r, g, b)

    m = _HSL_RE.match(value)
    if m:
        parts = [p for p in re.split(r"[,\s/]+", m.group(1).strip()) if p]
        if len(parts) >= 3:
            h = float(parts[0].rstrip("deg")) / 360.0
            s = float(parts[1].rstrip("%")) / 100.0
            ln = float(parts[2].rstrip("%")) / 100.0
            r, g, b = hls_to_rgb(h % 1.0, ln, s)
            if len(parts) >= 4:
                a = parts[3]
                alpha = float(a[:-1]) / 100.0 if a.endswith("%") else float(a)
                return (r, g, b, min(max(alpha, 0.0), 1.0))
            return (r, g, b)

    # Named color or hex -- hand straight to matplotlib.
    return value


def to_grayscale(color) -> tuple[float, float, float]:
    """Convert a matplotlib color to its luminance-preserving gray equivalent."""
    from matplotlib.colors import to_rgb

    r, g, b = to_rgb(color)
    v = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return (v, v, v)
