"""svg2mpl: render SVG files into matplotlib artists.

Public API
----------
- :func:`add_svg` -- render an SVG file (or markup string) onto a matplotlib axes.
- :func:`load` / :func:`load_string` -- parse an SVG into a :class:`Scene`.
- :class:`Scene`, :class:`Shape`, :class:`Text` -- the parsed representation.

The renderer aims to be *faithful*: it honors element transforms, ``<g>``
inheritance, ``viewBox``, the basic shape elements, and CSS (presentation
attributes, inline ``style``, and -- with the ``css`` extra -- ``<style>``
blocks).
"""

from importlib.metadata import PackageNotFoundError, version

from .render import (
    add_svg,
    get_affine_matrix,
    get_bbox,
    get_is_included,
    load,
    load_string,
    vectorize_dicts,
)
from .scene import Scene, Shape, Text

try:
    __version__ = version("svg2mpl")
except PackageNotFoundError:  # not installed (e.g. running from a source tree)
    __version__ = "0.0.0+unknown"

__all__ = [
    "add_svg",
    "load",
    "load_string",
    "Scene",
    "Shape",
    "Text",
    "get_bbox",
    "get_affine_matrix",
    "get_is_included",
    "vectorize_dicts",
    "__version__",
]
