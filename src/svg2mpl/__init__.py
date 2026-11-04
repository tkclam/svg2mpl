"""svg2mpl: render SVG files into matplotlib artists.

Public API
----------
- :func:`add_svg` -- render an SVG file (or markup string) onto a matplotlib axes.
- :func:`load` / :func:`load_string` -- parse an SVG into a :class:`Scene`.
- :class:`Scene`, :class:`Shape`, :class:`Image`, :class:`Group`, :class:`Text`
  -- the parsed representation.

The renderer aims to be *faithful* to static SVG: transforms and viewports,
CSS, gradients, patterns, clipping, masks, markers, text (as glyph outlines),
images and filters are all honored.
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
from .scene import Group, Image, Scene, Shape, Text

try:
    __version__ = version("svg2mpl")
except PackageNotFoundError:  # not installed (e.g. running from a source tree)
    __version__ = "0.0.0+unknown"

__all__ = [
    "Group",
    "Image",
    "Scene",
    "Shape",
    "Text",
    "__version__",
    "add_svg",
    "get_affine_matrix",
    "get_bbox",
    "get_is_included",
    "load",
    "load_string",
    "vectorize_dicts",
]
