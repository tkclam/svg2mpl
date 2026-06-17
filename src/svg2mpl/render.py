"""Public API: load an SVG and add its artists to a matplotlib axes.

``add_svg`` is the high-level entry point. It mirrors the placement / exclude /
recolor conveniences that were previously baked into flyplotlib (positioning a
drawing by a normalized anchor, scaling to a target size, rotating, excluding
paths by id, recoloring per id, grayscale) while rendering a *faithful* scene
graph underneath.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from functools import cache
from pathlib import Path as FilePath
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PathCollection
from matplotlib.patches import PathPatch
from matplotlib.transforms import Affine2D, Bbox

from .color import parse_url_ref, to_grayscale
from .parser import parse, parse_string
from .scene import Scene
from .transform import flip_y_transform

__all__ = [
    "load",
    "load_string",
    "add_svg",
    "get_bbox",
    "get_affine_matrix",
    "get_is_included",
    "vectorize_dicts",
]


# --------------------------------------------------------------------------- #
# Loading (cached on path + mtime, like flyplotlib's _parse_svg)
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
# Generic geometry helpers (ported verbatim from flyplotlib.core)
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

    if np.ndim(scale) == 0:
        affine2d.scale(scale, scale)
    else:
        affine2d.scale(*scale)

    affine2d.rotate_deg(rotation)
    affine2d.translate(*target_xy)
    return affine2d


def get_is_included(id_: str, exclude: str | Iterable[str]) -> bool:
    """Return whether ``id_`` survives the ``exclude`` regular expressions."""
    if isinstance(exclude, str):
        exclude = (exclude,)
    return not any(re.match(r, id_) for r in exclude)


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
# Scene -> matplotlib
# --------------------------------------------------------------------------- #
def _as_tuple(x: str | Iterable[str]) -> tuple[str, ...]:
    return (x,) if isinstance(x, str) else tuple(x)


def _flip_height(scene: Scene) -> float:
    """Height to flip about: the document height, else the geometry extent."""
    if scene.height:
        return scene.height
    if scene.shapes:
        bb = get_bbox([s.path for s in scene.shapes])
        return float(bb.y0 + bb.y1)
    return 0.0


def _is_url(value) -> bool:
    return isinstance(value, str) and value.lower().startswith("url(")


def _resolve_fill(value, scene: Scene):
    """Replace an unresolved ``url(#id)`` paint with a usable color (edge paints)."""
    if _is_url(value):
        ref = parse_url_ref(value)
        grad = scene.gradients.get(ref) if ref else None
        if grad is not None:
            return grad.average_color()
        return "none"
    return value


def get_path_dicts(
    scene: Scene,
    *,
    flip_y: bool = True,
    exclude: str | Iterable[str] = (),
    grayscale: bool = False,
    per_path_kwargs: Mapping[str, Mapping[str, Any]] | None = None,
    **global_kwargs: Any,
) -> dict[str, dict[str, Any]]:
    """Build per-shape matplotlib property dicts from a scene.

    The returned dicts are keyed by path id and carry the matplotlib kwargs plus
    a ``path`` entry (geometry in display space, with the optional y-flip
    applied).
    """
    per_path_kwargs = per_path_kwargs or {}
    flip = flip_y_transform(_flip_height(scene)) if flip_y else None

    dicts: dict[str, dict[str, Any]] = {}
    for shape in scene.shapes:
        if not get_is_included(shape.id, exclude):
            continue
        entry = dict(shape.kwargs)

        # Gradient fills can't be a matplotlib color: blank the face and tag the
        # entry so add_svg can overlay a clipped gradient image later.
        grad_ref = None
        if _is_url(entry.get("facecolor")):
            ref = parse_url_ref(entry["facecolor"])
            grad_ref = ref if ref and ref in scene.gradients else None
            entry["facecolor"] = "none"
        entry["edgecolor"] = _resolve_fill(entry.get("edgecolor"), scene)
        if grayscale and entry.get("facecolor") not in (None, "none"):
            entry["facecolor"] = to_grayscale(entry["facecolor"])
        entry["path"] = shape.path.transformed(flip) if flip else shape.path

        entry.update(global_kwargs)
        for prop, mapping in per_path_kwargs.items():
            if shape.id in mapping:
                entry[prop] = mapping[shape.id]
        # Only fall back to a gradient overlay if the user didn't set a fill.
        if grad_ref is not None and entry.get("facecolor") == "none":
            entry["_gradient"] = grad_ref
        # Keep duplicate ids from clobbering each other in the collection.
        key = shape.id or f"__shape_{len(dicts)}"
        while key in dicts:
            key += "_"
        dicts[key] = entry
    return dicts


@cache
def _cached_bbox(
    path_str: str,
    _mtime: float,
    flip_y: bool,
    exclude: tuple[str, ...],
    bbox_exclude: tuple[str, ...],
) -> Bbox:
    """Cached display-space bbox of the drawn, non-excluded paths."""
    scene = _load_cached(path_str, _mtime)
    flip = flip_y_transform(_flip_height(scene)) if flip_y else None
    paths = [
        (s.path.transformed(flip) if flip else s.path)
        for s in scene.shapes
        if get_is_included(s.id, exclude) and get_is_included(s.id, bbox_exclude)
    ]
    return get_bbox(paths)


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
        Convert fill colors to grayscale.
    force_patches : bool, optional
        Return individual :class:`~matplotlib.patches.PathPatch` objects instead
        of a single :class:`~matplotlib.collections.PathCollection`.
    ax : Axes, optional
        Axes to draw on (default current axes).
    per_path_kwargs : Mapping of Mapping, optional
        Per-path overrides: ``{property: {path_id: value}}``.
    **global_kwargs
        Keyword arguments applied to every path.

    Returns
    -------
    PathCollection or dict
        A ``PathCollection`` when the paths can be batched, else a dict mapping
        path id to ``PathPatch``.
    """
    scene, cache_key = _resolve_source(source)

    exclude = _as_tuple(exclude)
    bbox_exclude = _as_tuple(bbox_exclude)

    path_dicts = get_path_dicts(
        scene,
        flip_y=flip_y,
        exclude=exclude,
        grayscale=grayscale,
        per_path_kwargs=per_path_kwargs,
        **global_kwargs,
    )
    if not path_dicts and not scene.texts:
        raise ValueError(f"No paths to draw from {source!r} (all were excluded?).")

    if ax is None:
        ax = plt.gca()
    base_transform = ax.transData if transform is None else transform

    affine2d = Affine2D()  # identity when not placing
    if place and path_dicts:
        if cache_key is not None:
            bbox = _cached_bbox(*cache_key, flip_y, exclude, bbox_exclude)
        else:
            bbox = get_bbox([d["path"] for d in path_dicts.values()])
        affine2d = get_affine_matrix(
            bbox=bbox,
            source_xy=origin,
            target_xy=xy,
            rotation=rotation,
            width=width,
            height=height,
            scale=scale,
        )
    full_transform = affine2d + base_transform

    # Gradient-filled shapes can't be batched (they need a clipped image each).
    has_gradients = any("_gradient" in d for d in path_dicts.values())

    result = _emit(
        ax,
        path_dicts,
        full_transform,
        force_patches=force_patches or has_gradients,
    )

    if has_gradients:
        _add_gradient_fills(ax, scene, path_dicts, affine2d, base_transform)
    if scene.texts:
        from .text import add_texts

        flip = flip_y_transform(_flip_height(scene)) if flip_y else None
        add_texts(
            ax, scene, affine2d=affine2d, flip=flip, base_transform=base_transform
        )

    ax.autoscale_view()
    return result


def _emit(ax, path_dicts, transform, *, force_patches: bool):
    """Draw the shapes as a single PathCollection or as individual patches."""
    vectorized = None if force_patches else vectorize_dicts(path_dicts)
    if vectorized is not None:
        vectorized["paths"] = vectorized.pop("path")
        collection = PathCollection(**vectorized)
        collection.set_transform(transform)
        ax.add_collection(collection)
        return collection

    patches = {}
    for key, value in path_dicts.items():
        value = {k: v for k, v in value.items() if k != "_gradient"}
        patch = PathPatch(**value)
        patch.set_transform(transform)
        ax.add_patch(patch)
        patches[key] = patch
    return patches


def _add_gradient_fills(ax, scene, path_dicts, affine2d, base_transform) -> None:
    """Overlay a clipped gradient image for every gradient-filled shape."""
    from .gradients import add_gradient_fill

    for value in path_dicts.values():
        ref = value.get("_gradient")
        if ref is None:
            continue
        grad = scene.gradients.get(ref)
        if grad is None:
            continue
        final_path = value["path"].transformed(affine2d)
        add_gradient_fill(ax, final_path, grad, transform=base_transform)


def _resolve_source(source):
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
