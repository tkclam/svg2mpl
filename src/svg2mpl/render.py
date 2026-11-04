"""Public API: load an SVG and add its artists to a matplotlib axes.

``add_svg`` is the high-level entry point. It positions a drawing by a
normalized anchor, scales it to a target size, rotates it, excludes paths by id,
recolors per id or converts to grayscale, while rendering a *faithful* scene
underneath: drawables are emitted in document (painter's) order, strokes scale
with the drawing, clip paths are applied, gradients/patterns become clipped
images, and masked/filtered groups are composited off-screen.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from functools import cache
from pathlib import Path as FilePath
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import PathPatch
from matplotlib.transforms import Affine2D, Bbox

from .artists import SVGLineCollection, SVGPathCollection, SVGPathPatch
from .color import to_grayscale
from .geometry import intersect_clip, path_extents
from .parser import parse, parse_string
from .scene import Group, Image, Scene, Shape
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

# Off-screen compositing resolution (pixels per document unit is derived from
# this and the document size) and per-image pixel cap.
DEFAULT_RASTER_RESOLUTION = 1024
_MAX_RASTER_PX = 4096


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
# Scene helpers
# --------------------------------------------------------------------------- #
def _as_tuple(x: str | Iterable[str]) -> tuple[str, ...]:
    return (x,) if isinstance(x, str) else tuple(x)


def _item_extents(item):
    """Document-space extents of a drawable, limited by its clip."""
    ext = path_extents(item.path)
    if ext is None:
        return None
    clip = getattr(item, "clip", None)
    if clip is not None:
        ce = path_extents(clip)
        if ce is None:
            return None
        ext = (
            max(ext[0], ce[0]),
            max(ext[1], ce[1]),
            min(ext[2], ce[2]),
            min(ext[3], ce[3]),
        )
        if ext[2] < ext[0] or ext[3] < ext[1]:
            return None
    return ext


def _geometry_extents(items, exclude=(), bbox_exclude=()):
    box = None
    for item in items:
        if not get_is_included(item.id, exclude) or not get_is_included(
            item.id, bbox_exclude
        ):
            continue
        ext = _item_extents(item)
        if ext is None:
            continue
        box = (
            ext
            if box is None
            else (
                min(box[0], ext[0]),
                min(box[1], ext[1]),
                max(box[2], ext[2]),
                max(box[3], ext[3]),
            )
        )
    return box


def _flip_height(scene: Scene) -> float:
    """Height to flip about: the document height, else the geometry extent."""
    if scene.height:
        return scene.height
    box = _geometry_extents(scene.items)
    if box is not None:
        return float(box[1] + box[3])
    return 0.0


def _filter_items(items, exclude):
    out = []
    for item in items:
        if not get_is_included(item.id, exclude):
            continue
        if isinstance(item, Group):
            kids = _filter_items(item.items, exclude)
            if not kids:
                continue
            if len(kids) != len(item.items):
                from dataclasses import replace

                item = replace(item, items=kids)
        out.append(item)
    return out


def _placement_bbox(items, flip, exclude, bbox_exclude) -> Bbox | None:
    box = _geometry_extents(items, exclude, bbox_exclude)
    if box is None:
        return None
    pts = flip.transform([(box[0], box[1]), (box[2], box[3])])
    return Bbox([pts.min(axis=0), pts.max(axis=0)])


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
    a ``path`` entry (geometry with the optional y-flip applied). Stroke widths
    are in document units unless overridden.
    """
    flip = flip_y_transform(_flip_height(scene)) if flip_y else None
    out = {}
    for shape in scene.shapes:
        if not get_is_included(shape.id, exclude):
            continue
        entry = _shape_entry(shape, grayscale, per_path_kwargs or {}, global_kwargs)
        entry.pop("_paint", None)
        entry.pop("_svg", None)
        entry["path"] = shape.path.transformed(flip) if flip else shape.path
        out[_unique_key(out, shape.id)] = entry
    return out


def _unique_key(d, key, counts=None):
    """A dict key for ``key`` that does not clobber an earlier duplicate id.

    Duplicates get ``_1``, ``_2`` ... suffixes; ``counts`` (a dict reused across
    calls) keeps this linear for drawings with many repeated ids.
    """
    key = key or f"__shape_{len(d)}"
    if key not in d:
        return key
    counts = {} if counts is None else counts
    n = counts.get(key, 0) + 1
    while f"{key}_{n}" in d:
        n += 1
    counts[key] = n
    return f"{key}_{n}"


def _gray(color):
    if color is None or (isinstance(color, str) and color.lower() == "none"):
        return color
    from matplotlib.colors import to_rgba

    r, g, b, a = to_rgba(color)
    v = to_grayscale((r, g, b))[0]
    return (v, v, v, a)


def _gray_image(rgba: np.ndarray) -> np.ndarray:
    out = np.array(rgba, dtype=float)
    if out.max() > 1.0:
        out /= 255.0
    v = out[..., 0] * 0.2989 + out[..., 1] * 0.5870 + out[..., 2] * 0.1140
    out[..., 0] = out[..., 1] = out[..., 2] = v
    return out


def _shape_entry(shape: Shape, grayscale, per_path_kwargs, global_kwargs) -> dict:
    """matplotlib kwargs for a shape, with user overrides applied."""
    entry = dict(shape.kwargs)
    lw_doc = entry.pop("linewidth", 1.0)
    dashes = entry.pop("linestyle", None)
    if grayscale:
        entry["facecolor"] = _gray(entry.get("facecolor"))
        entry["edgecolor"] = _gray(entry.get("edgecolor"))
    entry.update(global_kwargs)
    for prop, mapping in per_path_kwargs.items():
        if shape.id in mapping:
            entry[prop] = mapping[shape.id]
    user_lw = "linewidth" in entry or "lw" in entry
    if "lw" in entry:
        entry["linewidth"] = entry.pop("lw")
    user_ls = "linestyle" in entry or "ls" in entry
    if "ls" in entry:
        entry["linestyle"] = entry.pop("ls")
    entry["_svg"] = {
        "lw": None if user_lw else lw_doc,
        "dashes": None if user_ls else dashes,
        "scaled": shape.scale_stroke,
    }
    if not user_lw:
        entry["linewidth"] = lw_doc  # replaced at draw time
    fc = entry.get("facecolor")
    paint = (
        shape.fill_paint
        if (fc is None or (isinstance(fc, str) and fc == "none"))
        else None
    )
    entry["_paint"] = (paint, shape.stroke_paint)
    return entry


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #
class _Renderer:
    """Emit scene items as matplotlib artists on one axes."""

    def __init__(
        self,
        ax,
        doc_to_data,
        base,
        *,
        zorder,
        density,
        grayscale,
        text_as_paths,
        per_path_kwargs,
        global_kwargs,
        flip=None,
    ):
        self.ax = ax
        self.doc_to_data = doc_to_data
        self.base = base
        self.full = doc_to_data + base
        # Shape paths are stored with the y-flip baked in (as earlier releases
        # did), so their artists use the remaining placement transform.
        self.flip = flip if flip is not None else Affine2D()
        self.shape_transform = (self.flip.inverted() + doc_to_data) + base
        self.zorder = zorder
        self.density = density
        self.grayscale = grayscale
        self.text_as_paths = text_as_paths
        self.per_path_kwargs = per_path_kwargs or {}
        self.global_kwargs = dict(global_kwargs)
        self.global_kwargs.pop("zorder", None)
        self.patches: dict = {}
        self._key_counts: dict = {}
        self._limits: list = []
        self.extra: list = []
        self._texts_done: set = set()

    # -- helpers -------------------------------------------------------- #
    def _clip_patch(self, path):
        clip = PathPatch(path, transform=self.full, facecolor="none", edgecolor="none")
        return clip

    def _set_clip(self, artist, clip):
        if clip is not None:
            artist.set_clip_path(self._clip_patch(clip))

    def _add(self, artist):
        artist.set_zorder(self.zorder)
        self.extra.append(artist)
        return artist

    def _add_image(
        self,
        data,
        extent,
        to_doc=None,
        interpolation="antialiased",
        alpha=None,
        clip=None,
    ):
        """Add an image whose ``extent`` is in coordinates mapped by ``to_doc``.

        ``Axes.imshow`` would reset the axes limits and aspect from the extent,
        so the artist is built directly and only the data limits are updated.
        """
        from matplotlib.image import AxesImage

        im = AxesImage(
            self.ax, interpolation=interpolation, origin="upper", extent=extent
        )
        im.set_data(data)
        if alpha is not None:
            im.set_alpha(alpha)
        to_doc = Affine2D() if to_doc is None else Affine2D(to_doc.get_matrix())
        im.set_transform(to_doc + self.full)
        if clip is not None:
            im.set_clip_path(self._clip_patch(clip))
        self.ax.add_image(im)
        if self.base is self.ax.transData:
            x0, x1, y1, y0 = extent
            corners = (to_doc + self.doc_to_data).transform(
                [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
            )
            self.ax.update_datalim(corners)
        return self._add(im)

    # -- items ------------------------------------------------------------ #
    def draw(self, items, collect=True):
        for item in items:
            clip = getattr(item, "clip", None)
            if clip is not None and len(clip.vertices) == 0:
                continue
            if isinstance(item, Shape):
                if item.text is not None and not self.text_as_paths:
                    self._text_artist(item.text)
                    continue
                self._shape(item, collect)
            elif isinstance(item, Image):
                self._image(item)
            elif isinstance(item, Group):
                self._group(item)

    def finish(self):
        """Fold the drawn shapes' extents into the axes data limits."""
        if not self._limits:
            return
        ext = np.array(self._limits)
        corners = np.array(
            [
                [ext[:, 0].min(), ext[:, 1].min()],
                [ext[:, 2].max(), ext[:, 3].max()],
                [ext[:, 0].min(), ext[:, 3].max()],
                [ext[:, 2].max(), ext[:, 1].min()],
            ]
        )
        self._limits = []
        # Shape paths are in flipped document space; map through placement.
        to_data = self.shape_transform - self.ax.transData
        if self.base is self.ax.transData:
            self.ax.update_datalim(to_data.transform(corners))

    def _text_artist(self, record):
        if id(record) in self._texts_done:
            return
        self._texts_done.add(id(record))
        from .text import add_texts

        flip_place = self.doc_to_data
        artists = add_texts(
            self.ax,
            None,
            affine2d=flip_place,
            flip=None,
            base_transform=self.base,
            texts=[record],
        )
        self.extra.extend(artists)

    def _shape(self, shape: Shape, collect=True):
        entry = _shape_entry(
            shape, self.grayscale, self.per_path_kwargs, self.global_kwargs
        )
        svg = entry.pop("_svg")
        fill_paint, stroke_paint = entry.pop("_paint")
        if fill_paint is not None:
            self._paint_fill(shape, fill_paint)
        if stroke_paint is not None and entry.get("edgecolor") not in (None, "none"):
            if self._paint_stroke(shape, stroke_paint, entry, svg):
                entry["edgecolor"] = "none"
        path = shape.path.transformed(self.flip)
        entry.setdefault("snap", False)  # browsers do not snap edges to pixels
        patch = SVGPathPatch(path, **entry)
        if svg["lw"] is not None:
            patch.set_svg_stroke(svg["lw"], svg["dashes"], svg["scaled"], _ref(path))
        patch.set_transform(self.shape_transform)
        self._set_clip(patch, shape.clip)
        patch.set_zorder(self.zorder)
        # add_artist + one bulk data-limit update: Axes.add_patch computes exact
        # curve extents per patch, which dominates on large drawings.
        self.ax.add_artist(patch)
        ext = path_extents(path)
        if ext is not None:
            self._limits.append(ext)
        if collect:
            self.patches[_unique_key(self.patches, shape.id, self._key_counts)] = patch
        else:
            self.extra.append(patch)

    def _image(self, item: Image):
        data = item.data
        if self.grayscale:
            data = _gray_image(data)
        h, w = data.shape[:2]
        interp = item.interpolation
        if interp == "auto":
            # Browsers smooth up-scaled images; antialias down-scaled ones.
            m = item.transform.get_matrix()
            px = math.sqrt(abs(m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0])) * self.density
            interp = "bilinear" if px >= 1.0 else "antialiased"
        self._add_image(
            data,
            (0, w, h, 0),
            to_doc=item.transform,
            interpolation=interp,
            alpha=item.alpha if item.alpha < 1 else None,
            clip=item.clip,
        )

    # -- paint servers ----------------------------------------------------- #
    def _px_grid(self, ext, density=None):
        density = density or self.density
        x0, y0, x1, y1 = ext
        w = max(2, min(_MAX_RASTER_PX, int(math.ceil((x1 - x0) * density))))
        h = max(2, min(_MAX_RASTER_PX, int(math.ceil((y1 - y0) * density))))
        return w, h

    def _paint_fill(self, shape: Shape, bound):
        from .gradients import Gradient

        region = shape.path
        clip_extra = None
        if shape.clip is not None:
            try:
                region = intersect_clip(shape.path, shape.clip)
            except ValueError:
                clip_extra = shape.clip
        ext = path_extents(region)
        if ext is None or ext[2] <= ext[0] or ext[3] <= ext[1]:
            return
        if isinstance(bound.server, Gradient):
            w, h = self._px_grid(
                ext,
                density=min(self.density, 512 / max(ext[2] - ext[0], ext[3] - ext[1])),
            )
            xs = ext[0] + (np.arange(w) + 0.5) * (ext[2] - ext[0]) / w
            ys = ext[1] + (np.arange(h) + 0.5) * (ext[3] - ext[1]) / h
            gx, gy = np.meshgrid(xs, ys)
            pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
            try:
                user = bound.transform.inverted().transform(pts)
            except Exception:
                return
            rgba = bound.server.evaluate(user.reshape(h, w, 2), bound.bbox)
            rgba[..., 3] *= bound.opacity
            if clip_extra is not None:
                rgba[..., 3] *= _coverage(clip_extra, pts).reshape(h, w)
            if self.grayscale:
                rgba = _gray_image(rgba)
            self._add_image(
                np.clip(rgba, 0, 1),
                (ext[0], ext[2], ext[3], ext[1]),
                interpolation="bilinear",
                clip=shape.path if clip_extra is not None else region,
            )
            return
        tile = bound.tile
        if tile is None:
            return
        self._pattern(
            tile, ext, shape.path if clip_extra is not None else region, bound.opacity
        )

    def _pattern(self, tile, ext, clip_path, opacity):
        try:
            inv = tile.transform.inverted()
        except Exception:
            return
        x0, y0, x1, y1 = ext
        corners = inv.transform([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
        i0 = math.floor(corners[:, 0].min() / tile.width)
        i1 = math.ceil(corners[:, 0].max() / tile.width)
        j0 = math.floor(corners[:, 1].min() / tile.height)
        j1 = math.ceil(corners[:, 1].max() / tile.height)
        ni, nj = max(i1 - i0, 1), max(j1 - j0, 1)
        if ni * nj > 250000:
            return
        # Tile pixel size from the doc-space density.
        m = tile.transform.get_matrix()
        sx = math.hypot(m[0, 0], m[1, 0]) * tile.width
        sy = math.hypot(m[0, 1], m[1, 1]) * tile.height
        tw = max(2, int(math.ceil(sx * self.density)))
        th = max(2, int(math.ceil(sy * self.density)))
        budget = _MAX_RASTER_PX * 2
        if tw * ni > budget:
            tw = max(2, budget // ni)
        if th * nj > budget:
            th = max(2, budget // nj)
        img = rasterize(tile.items, (0.0, 0.0, tile.width, tile.height), (th, tw))
        img[..., 3] *= opacity
        if self.grayscale:
            img = _gray_image(img)
        big = np.tile(img, (nj, ni, 1))
        ox, oy = i0 * tile.width, j0 * tile.height
        self._add_image(
            np.clip(big, 0, 1),
            (ox, ox + ni * tile.width, oy + nj * tile.height, oy),
            to_doc=tile.transform,
            clip=clip_path,
        )

    def _paint_stroke(self, shape: Shape, bound, entry, svg) -> bool:
        """Draw a gradient stroke as short colored segments. Returns success."""
        from .gradients import Gradient

        if not isinstance(bound.server, Gradient) or svg["lw"] is None:
            return False
        segs, mids = [], []
        for ring in _stroke_polylines(shape.path):
            if len(ring) < 2:
                continue
            fine = _subdivide(ring, n=400)
            s = np.stack([fine[:-1], fine[1:]], axis=1)
            segs.append(s)
            mids.append((fine[:-1] + fine[1:]) / 2)
        if not segs:
            return False
        segs = np.concatenate(segs)
        mids = np.concatenate(mids)
        try:
            user = bound.transform.inverted().transform(mids)
        except Exception:
            return False
        colors = bound.server.evaluate(user, bound.bbox)
        colors[..., 3] *= bound.opacity
        if self.grayscale:
            colors = _gray_image(colors[None])[0]
        lc = SVGLineCollection(
            segs, colors=np.clip(colors, 0, 1), capstyle="round", joinstyle="round"
        )
        lc.set_svg_stroke(svg["lw"], None, svg["scaled"], _ref(shape.path))
        lc.set_transform(self.full)
        self._set_clip(lc, shape.clip)
        self.ax.add_collection(lc, autolim=False)
        self._add(lc)
        return True

    # -- off-screen groups -------------------------------------------------- #
    def _group(self, group: Group):
        x0, y0, x1, y1 = group.region
        if x1 <= x0 or y1 <= y0:
            return
        w, h = self._px_grid(group.region)
        img = rasterize(group.items, group.region, (h, w))
        ppx, ppy = w / (x1 - x0), h / (y1 - y0)
        if group.filter is not None:
            from .filters import apply_filter

            img = apply_filter(
                group.filter,
                img,
                group.region,
                ppx,
                ppy,
                lambda items, region, shape: rasterize(items, region, shape),
            )
        if group.mask is not None:
            m = rasterize(group.mask.items, group.region, (h, w))
            if group.mask.kind == "alpha":
                factor = m[..., 3]
            else:
                lum = m[..., 0] * 0.2125 + m[..., 1] * 0.7154 + m[..., 2] * 0.0721
                factor = lum * m[..., 3]
            mx0, my0, mx1, my1 = group.mask.region
            xs = x0 + (np.arange(w) + 0.5) / ppx
            ys = y0 + (np.arange(h) + 0.5) / ppy
            inside = ((ys >= my0) & (ys <= my1))[:, None] & ((xs >= mx0) & (xs <= mx1))[
                None, :
            ]
            img[..., 3] *= factor * inside
        img[..., 3] *= group.opacity
        if self.grayscale:
            img = _gray_image(img)
        self._add_image(np.clip(img, 0, 1), (x0, x1, y1, y0), clip=group.clip)


def _coverage(path, pts) -> np.ndarray:
    """1.0 where points fall inside ``path``, else 0.0 (hard-edged coverage)."""
    return path.contains_points(pts).astype(float)


def _ref(path):
    ext = path_extents(path)
    if ext is None:
        return (0.0, 0.0)
    return ((ext[0] + ext[2]) / 2, (ext[1] + ext[3]) / 2)


def _stroke_polylines(path):
    """Open polylines of every subpath (closed subpaths repeat their start)."""
    from matplotlib.path import Path as MPath

    from .geometry import flatten, split_subpaths

    out = []
    for sub in split_subpaths(path):
        rings = flatten(sub)
        closed = sub.codes is not None and sub.codes[-1] == MPath.CLOSEPOLY
        for r in rings:
            out.append(np.vstack([r, r[:1]]) if closed else r)
    return out


def _subdivide(poly: np.ndarray, n: int) -> np.ndarray:
    seg = np.diff(poly, axis=0)
    length = np.hypot(seg[:, 0], seg[:, 1])
    total = length.sum()
    if total <= 0:
        return poly
    out = [poly[:1]]
    for p, d, L in zip(poly[:-1], seg, length, strict=False):
        k = max(1, int(round(n * L / total)))
        t = np.linspace(0, 1, k + 1)[1:, None]
        out.append(p + t * d)
    return np.vstack(out)


def rasterize(items, region, shape) -> np.ndarray:
    """Render ``items`` (document space) over ``region`` into straight RGBA.

    ``region`` is ``(x0, y0, x1, y1)`` in the items' coordinates (y down) and
    ``shape`` the output ``(height, width)``. Row 0 is ``y0``.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    h, w = int(shape[0]), int(shape[1])
    dpi = 100.0
    fig = Figure(figsize=((w + 0.01) / dpi, (h + 0.01) / dpi), dpi=dpi)
    canvas = FigureCanvasAgg(fig)
    fig.patch.set_alpha(0.0)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_axis_off()
    ax.patch.set_alpha(0.0)
    x0, y0, x1, y1 = region
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    density = w / (x1 - x0) if x1 > x0 else 1.0
    r = _Renderer(
        ax,
        Affine2D(),
        ax.transData,
        zorder=1,
        density=density,
        grayscale=False,
        text_as_paths=True,
        per_path_kwargs=None,
        global_kwargs={},
    )
    r.draw(items, collect=False)
    canvas.draw()
    buf = np.asarray(canvas.buffer_rgba(), dtype=float) / 255.0
    out = np.zeros((h, w, 4))
    hh, ww = min(h, buf.shape[0]), min(w, buf.shape[1])
    out[:hh, :ww] = buf[:hh, :ww]
    return out


def _batchable(items) -> bool:
    for item in items:
        if not isinstance(item, Shape):
            return False
        if (
            item.clip is not None
            or item.fill_paint is not None
            or item.stroke_paint is not None
        ):
            return False
    return True


def _emit_collection(ax, r: _Renderer, shapes, transform):
    entries = {}
    counts: dict = {}
    svg_lw, svg_dash, scaled = [], [], []
    for shape in shapes:
        entry = _shape_entry(shape, r.grayscale, r.per_path_kwargs, r.global_kwargs)
        svg = entry.pop("_svg")
        entry.pop("_paint")
        entry["path"] = shape.path.transformed(r.flip)
        entries[_unique_key(entries, shape.id, counts)] = (entry, svg)
    dicts = {k: v[0] for k, v in entries.items()}
    vectorized = vectorize_dicts(dicts)
    if vectorized is None:
        return None
    svgs = [v[1] for v in entries.values()]
    if len({s["scaled"] for s in svgs}) > 1:
        return None
    if any(s["lw"] is None for s in svgs) and any(s["lw"] is not None for s in svgs):
        return None
    for s in svgs:
        svg_lw.append(s["lw"])
        svg_dash.append(s["dashes"])
        scaled.append(s["scaled"])
    vectorized["paths"] = vectorized.pop("path")
    vectorized.setdefault("snap", False)
    if "snap" in vectorized and isinstance(vectorized["snap"], list):
        vectorized["snap"] = vectorized["snap"][0]
    if "linestyle" in vectorized and svg_lw[0] is not None:
        vectorized.pop("linestyle")
    collection = SVGPathCollection(**vectorized)
    if svg_lw[0] is not None:
        has_dash = any(d is not None for d in svg_dash)
        ref = _ref(vectorized["paths"][0]) if vectorized["paths"] else (0.0, 0.0)
        collection.set_svg_stroke(
            svg_lw,
            [d if d is not None else (0.0, [1.0, 0.0]) for d in svg_dash]
            if has_dash
            else None,
            scaled[0],
            ref,
        )
    collection.set_transform(r.shape_transform)
    collection.set_zorder(r.zorder)
    ax.add_collection(collection)
    return collection


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
            bbox = _placement_bbox(items, flip, exclude, bbox_exclude)
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
    doc_to_data = flip + affine2d
    full_transform = doc_to_data + base_transform

    size = max(scene.width, scene.height)
    if not size:
        box = _geometry_extents(items)
        size = max(box[2] - box[0], box[3] - box[1]) if box else 1.0
    density = raster_resolution / (size or 1.0)
    renderer = _Renderer(
        ax,
        doc_to_data,
        base_transform,
        zorder=global_kwargs.get("zorder", 1),
        density=density,
        grayscale=grayscale,
        text_as_paths=text_as_paths,
        per_path_kwargs=per_path_kwargs,
        global_kwargs=global_kwargs,
        flip=flip,
    )

    result = None
    if (
        not force_patches
        and _batchable(items)
        and (text_as_paths or not any(s.text is not None for s in items))
    ):
        result = _emit_collection(ax, renderer, items, full_transform)
    if result is None:
        renderer.draw(items)
        renderer.finish()
        result = renderer.patches

    ax.autoscale_view()
    return result


@cache
def _cached_bbox(
    path_str: str,
    _mtime: float,
    flip_y: bool,
    exclude: tuple[str, ...],
    bbox_exclude: tuple[str, ...],
) -> Bbox | None:
    """Cached display-space bbox of the drawn, non-excluded items."""
    scene = _load_cached(path_str, _mtime)
    flip = flip_y_transform(_flip_height(scene)) if flip_y else Affine2D()
    return _placement_bbox(_filter_items(scene.items, exclude), flip, (), bbox_exclude)


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
