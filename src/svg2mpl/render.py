"""Public API: load an SVG and add its artists to a matplotlib axes.

``add_svg`` is the high-level entry point. It positions a drawing by a
normalized anchor, scales it to a target size, rotates it, excludes paths by id,
recolors per id or converts to grayscale, while rendering a *faithful* scene
underneath (see :mod:`svg2mpl._draw`): drawables are emitted in document
(painter's) order, strokes scale with the drawing, clip paths are applied,
gradients and patterns become clipped images, and masked or filtered groups
are composited off-screen.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import replace
from functools import cache
from pathlib import Path as FilePath
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import Affine2D, Bbox

from ._draw import DrawOptions, Renderer, emit_collection, get_is_included, is_batchable
from .geometry import Rect, path_extents, rect_intersection, rect_is_empty, rect_union
from .parser import parse, parse_string
from .scene import Group, Scene, Shape
from .transform import flip_y_transform

__all__ = [
    "DEFAULT_RASTER_RESOLUTION",
    "add_svg",
    "get_affine_matrix",
    "get_bbox",
    "get_is_included",
    "load",
    "load_string",
    "vectorize_dicts",
]

# Pixels across the document's larger side for off-screen effects.
DEFAULT_RASTER_RESOLUTION = 1024


# --------------------------------------------------------------------------- #
# Loading (cached on path + mtime)
# --------------------------------------------------------------------------- #
@cache
def _load_cached(path_str: str, _mtime: float) -> Scene:
    """Parse and cache a Scene; ``_mtime`` invalidates on file change."""
    return parse(path_str)


def load(source: str | FilePath) -> Scene:
    """Parse an SVG file into a (cached) :class:`~svg2mpl.scene.Scene`.

    Paths in the returned scene are in SVG user space (y points down); the
    optional y-flip is applied by :func:`add_svg`. The result is shared and
    cached -- callers must not mutate it.
    """
    p = FilePath(source)
    return _load_cached(str(p), p.stat().st_mtime)


def load_string(text: str) -> Scene:
    """Parse an SVG document from a string (not cached)."""
    return parse_string(text)


# --------------------------------------------------------------------------- #
# Generic geometry helpers
# --------------------------------------------------------------------------- #
def get_bbox(paths: Iterable[Any]) -> Bbox:
    """Return the bounding box enclosing a collection of paths."""
    extents = np.array([p.get_extents() for p in paths])
    if extents.size == 0:
        raise ValueError("Cannot compute a bounding box of an empty collection.")
    mins = extents[:, 0].min(axis=0)
    maxs = extents[:, 1].max(axis=0)
    return Bbox([mins, maxs])


def get_affine_matrix(
    bbox: Bbox,
    source_xy: tuple[float, float],
    target_xy: tuple[float, float],
    rotation: float,
    width: float | None,
    height: float | None,
    scale: tuple[float, float] | float = (1.0, 1.0),
) -> Affine2D:
    """Return an affine that places ``bbox`` at ``target_xy`` in data coords.

    ``source_xy`` is the anchor inside the bounding box in normalized coordinates
    (``(0, 0)`` lower-left, ``(1, 1)`` upper-right). ``width``/``height`` set the
    final size (``None`` preserves aspect ratio); ``scale`` is an extra factor;
    ``rotation`` is in degrees.
    """
    (x0, y0), (x1, y1) = np.asarray(bbox)
    w, h = x1 - x0, y1 - y0

    affine2d = Affine2D()
    affine2d.translate(-x0 - w * source_xy[0], -y0 - h * source_xy[1])

    if width is not None and height is not None:
        affine2d.scale(width / w, height / h)
    elif width is not None:
        affine2d.scale(width / w)
    elif height is not None:
        affine2d.scale(height / h)

    sx, sy = (scale, scale) if np.ndim(scale) == 0 else scale  # type: ignore[misc]
    affine2d.scale(sx, sy)

    affine2d.rotate_deg(rotation)
    affine2d.translate(*target_xy)
    return affine2d


def vectorize_dicts(dicts: Mapping[str, Mapping[str, Any]]) -> dict[str, list] | None:
    """Transpose a mapping of dicts into a dict of lists for ``PathCollection``.

    Returns ``None`` if the input is empty or the property dicts do not share the
    same keys (so a single ``PathCollection`` cannot represent them).
    """
    if not dicts:
        return None
    it = iter(dicts.values())
    k0 = set(next(it).keys())
    if any(set(d.keys()) != k0 for d in it):
        return None
    return {k: [d[k] for d in dicts.values()] for k in k0}


# --------------------------------------------------------------------------- #
# Scene helpers
# --------------------------------------------------------------------------- #
def _as_tuple(x: str | Iterable[str]) -> tuple[str, ...]:
    return (x,) if isinstance(x, str) else tuple(x)


def _item_extents(item) -> Rect | None:
    """Document-space extents of a drawable, limited by its clip."""
    ext = path_extents(item.path)
    if ext is None or item.clip is None:
        return ext
    clip_ext = path_extents(item.clip)
    if clip_ext is None:
        return None
    ext = rect_intersection(ext, clip_ext)
    return None if rect_is_empty(ext) else ext


def _geometry_extents(items, exclude=(), bbox_exclude=()) -> Rect | None:
    box = None
    for item in items:
        if get_is_included(item.id, exclude) and get_is_included(item.id, bbox_exclude):
            box = rect_union(box, _item_extents(item))
    return box


def _flip_height(scene: Scene) -> float:
    """Height to flip about: the document height, else the geometry extent."""
    if scene.height:
        return scene.height
    box = _geometry_extents(scene.items)
    return float(box[1] + box[3]) if box is not None else 0.0


def _filter_items(items: list, exclude: tuple[str, ...]) -> list:
    """Drop items (recursively inside groups) whose id matches ``exclude``."""
    out = []
    for item in items:
        if not get_is_included(item.id, exclude):
            continue
        if isinstance(item, Group):
            kids = _filter_items(item.items, exclude)
            if not kids:
                continue
            if len(kids) != len(item.items):
                item = replace(item, items=kids)  # noqa: PLW2901
        out.append(item)
    return out


def _placement_bbox(items, flip: Affine2D, bbox_exclude) -> Bbox | None:
    box = _geometry_extents(items, (), bbox_exclude)
    if box is None:
        return None
    pts = flip.transform([(box[0], box[1]), (box[2], box[3])])
    return Bbox([pts.min(axis=0), pts.max(axis=0)])


@cache
def _cached_bbox(
    path_str: str,
    _mtime: float,
    flip_y: bool,
    exclude: tuple[str, ...],
    bbox_exclude: tuple[str, ...],
) -> Bbox | None:
    """Cached placement bbox of the drawn, non-excluded items of a file."""
    scene = _load_cached(path_str, _mtime)
    flip = flip_y_transform(_flip_height(scene)) if flip_y else Affine2D()
    return _placement_bbox(_filter_items(scene.items, exclude), flip, bbox_exclude)


def _resolve_source(source) -> tuple[Scene, tuple[str, float] | None]:
    """Return ``(scene, cache_key)`` for a path or inline-markup source.

    ``cache_key`` is ``(path_str, mtime)`` for files (enabling bbox caching) or
    ``None`` for inline markup.
    """
    if isinstance(source, str) and source.lstrip().startswith("<"):
        return load_string(source), None
    p = FilePath(source)
    if not p.exists():
        raise FileNotFoundError(f"SVG file not found: {source}")
    return _load_cached(str(p), p.stat().st_mtime), (str(p), p.stat().st_mtime)


# --------------------------------------------------------------------------- #
# add_svg
# --------------------------------------------------------------------------- #
def add_svg(
    source: str | FilePath,
    xy: tuple[float, float] = (0, 0),
    width: float | None = None,
    height: float | None = None,
    scale: tuple[float, float] | float = (1, 1),
    origin: tuple[float, float] = (0.5, 0.5),
    rotation: float = 0,
    transform=None,
    flip_y: bool = True,
    place: bool = True,
    exclude: str | Iterable[str] = (),
    bbox_exclude: str | Iterable[str] = (),
    grayscale: bool = False,
    force_patches: bool = False,
    ax=None,
    per_path_kwargs: Mapping[str, Mapping[str, Any]] | None = None,
    text_as_paths: bool = True,
    raster_resolution: int = DEFAULT_RASTER_RESOLUTION,
    **global_kwargs: Any,
):
    """Render an SVG into a matplotlib axes.

    Parameters
    ----------
    source : str or Path
        Path to an SVG file, or a string of SVG markup (must start with ``<``).
    xy : tuple, optional
        Where to place the drawing's ``origin`` anchor, in data coordinates.
    width, height : float or None, optional
        Final size of the drawing's bounding box. If only one is given the other
        is derived to preserve the aspect ratio. If both are ``None`` and
        ``place`` is ``True``, the drawing keeps its intrinsic size.
    scale : float or tuple, optional
        An extra scaling factor applied on top of ``width``/``height``.
    origin : tuple, optional
        Anchor point inside the bounding box, normalized (default center).
    rotation : float, optional
        Rotation in degrees.
    transform : Transform, optional
        A matplotlib transform to compose with (default ``ax.transData``).
    flip_y : bool, optional
        Flip SVG's y-down axis to matplotlib's y-up so the drawing appears as in
        a browser. Set ``False`` to keep raw SVG coordinates. Default ``True``.
    place : bool, optional
        When ``True`` (default) the drawing is positioned/scaled via ``xy``,
        ``width``/``height``, ``origin``, ``rotation``. When ``False`` the
        drawing is rendered in its own coordinates (only ``flip_y`` applies),
        which faithfully reproduces the SVG's absolute layout.
    exclude : str or Iterable[str], optional
        Regular expressions; paths whose id matches are not drawn.
    bbox_exclude : str or Iterable[str], optional
        Regular expressions for paths to ignore when computing the placement
        bounding box.
    grayscale : bool, optional
        Convert all colors (fills, strokes, gradients, images) to grayscale.
    force_patches : bool, optional
        Return individual :class:`~matplotlib.patches.PathPatch` objects instead
        of a single :class:`~matplotlib.collections.PathCollection`.
    ax : Axes, optional
        Axes to draw on (default current axes).
    per_path_kwargs : Mapping of Mapping, optional
        Per-path overrides: ``{property: {path_id: value}}``.
    text_as_paths : bool, optional
        Draw text as glyph outlines (default), which follow the drawing's scale,
        clipping and paint exactly. ``False`` uses matplotlib ``Text`` artists
        instead (selectable text in vector output, sizes in points).
    raster_resolution : int, optional
        Pixels across the document's larger side used for off-screen effects
        (masks, filters, patterns). Default 1024.
    **global_kwargs
        Keyword arguments applied to every path. ``linewidth`` given here (or in
        ``per_path_kwargs``) is in points and does not scale with the drawing.

    Returns
    -------
    PathCollection or dict
        A ``PathCollection`` when the paths can be batched, else a dict mapping
        path id to ``PathPatch``.
    """
    scene, cache_key = _resolve_source(source)
    exclude = _as_tuple(exclude)
    bbox_exclude = _as_tuple(bbox_exclude)
    items = _filter_items(scene.items, exclude)
    if not items:
        raise ValueError(f"No paths to draw from {source!r} (all were excluded?).")

    if ax is None:
        ax = plt.gca()
    base_transform = ax.transData if transform is None else transform
    flip = flip_y_transform(_flip_height(scene)) if flip_y else Affine2D()

    affine2d = Affine2D()  # identity when not placing
    if place:
        if cache_key is not None:
            bbox = _cached_bbox(*cache_key, flip_y, exclude, bbox_exclude)
        else:
            bbox = _placement_bbox(items, flip, bbox_exclude)
        if bbox is not None and bbox.width > 0 and bbox.height > 0:
            affine2d = get_affine_matrix(
                bbox=bbox,
                source_xy=origin,
                target_xy=xy,
                rotation=rotation,
                width=width,
                height=height,
                scale=scale,
            )

    size = max(scene.width, scene.height)
    if not size:
        box = _geometry_extents(items)
        size = max(box[2] - box[0], box[3] - box[1]) if box else 1.0
    options = DrawOptions(
        zorder=global_kwargs.pop("zorder", 1),
        density=raster_resolution / (size or 1.0),
        grayscale=grayscale,
        text_as_paths=text_as_paths,
        per_path_kwargs=per_path_kwargs or {},
        global_kwargs=global_kwargs,
    )
    renderer = Renderer(ax, flip + affine2d, base_transform, options, flip=flip)

    result = None
    has_text = any(isinstance(s, Shape) and s.text is not None for s in items)
    if not force_patches and is_batchable(items) and (text_as_paths or not has_text):
        result = emit_collection(renderer, items)
    if result is None:
        renderer.draw(items)
        renderer.finish()
        result = renderer.patches

    ax.autoscale_view()
    return result
