"""Parse the SVG path ``d`` attribute into a matplotlib :class:`~matplotlib.path.Path`.

This currently delegates to :func:`svgpath2mpl.parse_path`, which is a small,
battle-tested implementation of the full SVG path grammar (M/L/H/V/C/S/Q/T/A/Z,
absolute and relative, with arcs approximated by Beziers). It is isolated behind
this seam so it can be vendored or replaced later without touching the rest of
the package.
"""

from __future__ import annotations

import numpy as np
from matplotlib.path import Path
from svgpath2mpl import parse_path as _parse_path

__all__ = ["parse_path", "empty_path"]


def empty_path() -> Path:
    """Return a vertex-less path (shape ``(0, 2)``) for degenerate shapes."""
    return Path(np.empty((0, 2)))


def parse_path(d: str | None) -> Path:
    """Parse a path ``d`` string into a matplotlib :class:`~matplotlib.path.Path`.

    An empty or missing ``d`` yields an empty path (no vertices), so callers can
    treat every shape uniformly.
    """
    if not d or not d.strip():
        return empty_path()
    return _parse_path(d)
