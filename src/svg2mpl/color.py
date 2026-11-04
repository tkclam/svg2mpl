"""Parse SVG paint/color values into matplotlib-friendly forms.

matplotlib already understands named colors and ``#rgb``/``#rrggbb`` hex, so the
job here is mostly to recognise the SVG-specific spellings (``rgb()``, ``hsl()``,
``none``, ``currentColor``, ``url(#id)``) and normalise them.
"""

from __future__ import annotations

import math
import re
from colorsys import hls_to_rgb

import numpy as np
from matplotlib.colors import is_color_like, to_rgb, to_rgba
from matplotlib.typing import ColorType

__all__ = [
    "NONE",
    "gray_image",
    "gray_rgba",
    "parse_color",
    "parse_paint",
    "parse_url_ref",
    "split_paint",
    "to_grayscale",
    "with_alpha",
]

# Sentinel for "no paint" -- matplotlib understands the string "none".
NONE = "none"
# Rec. 601 luma weights used for grayscale conversion.
_LUMA = (0.2989, 0.5870, 0.1140)

_URL_RE = re.compile(r"url\(\s*#([^)\s]+)\s*\)", re.IGNORECASE)
_RGB_RE = re.compile(r"rgba?\(([^)]*)\)", re.IGNORECASE)
_HSL_RE = re.compile(r"hsla?\(([^)]*)\)", re.IGNORECASE)


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
    current_color: ColorType = "black",
    default: ColorType = NONE,
) -> ColorType:
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
    current_color: ColorType = "black",
    default: ColorType = NONE,
) -> ColorType:
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


def _parse_color(value: str, low: str) -> ColorType:

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

    if low.startswith("#"):
        digits = low[1:]
        if len(digits) in (3, 4) and all(c in "0123456789abcdef" for c in digits):
            r, g, b, *a = (int(c * 2, 16) / 255.0 for c in digits)
            return (r, g, b, a[0]) if a else (r, g, b)
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
        return float(t[:-3]) / (2 * math.pi)
    if t.endswith("turn"):
        return float(t[:-4])
    return float(t) / 360.0


def with_alpha(color: ColorType, alpha: float) -> ColorType:
    """Multiply ``alpha`` into a color (``"none"`` and ``url()`` pass through)."""
    if color == NONE or (isinstance(color, str) and color.lower().startswith("url(")):
        return color
    if alpha == 1.0:
        return color
    r, g, b, a = to_rgba(color)
    return (r, g, b, a * alpha)


def to_grayscale(color: ColorType) -> tuple[float, float, float]:
    """Convert a matplotlib color to its luminance-preserving gray equivalent."""
    r, g, b = to_rgb(color)
    v = _LUMA[0] * r + _LUMA[1] * g + _LUMA[2] * b
    return (v, v, v)


def gray_rgba(color):
    """Grayscale version of a color, keeping its alpha (``"none"`` passes)."""
    if color is None or (isinstance(color, str) and color.lower() == "none"):
        return color
    r, g, b, a = to_rgba(color)
    v = to_grayscale((r, g, b))[0]
    return (v, v, v, a)


def gray_image(rgba: np.ndarray) -> np.ndarray:
    """Grayscale copy of an RGBA image (uint8 input is scaled to ``[0, 1]``)."""
    out = np.array(rgba, dtype=float)
    if out.max() > 1.0:
        out /= 255.0
    v = out[..., :3] @ np.asarray(_LUMA)
    out[..., 0] = out[..., 1] = out[..., 2] = v
    return out
