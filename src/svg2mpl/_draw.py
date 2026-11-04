"""Emit scene items as matplotlib artists (the machinery behind ``add_svg``).

Items are drawn in document (painter's) order at one ``zorder``. Shapes become
:class:`~svg2mpl.artists.SVGPathPatch` objects (or one batched collection),
paint servers become clipped images, and off-screen groups (masks, filters,
overlapping opacity) are rasterized with :func:`rasterize` and drawn back as
images.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from matplotlib.transforms import Affine2D, Transform

from .artists import SVGLineCollection, SVGPathCollection, SVGPathPatch
from .color import NONE, gray_image, gray_rgba
from .filters import apply_filter
from .geometry import (
    Rect,
    area_scale,
    flatten,
    intersect_clip,
    is_empty,
    path_extents,
    rect_is_empty,
    split_subpaths,
)
from .gradients import Gradient
from .scene import BoundPaint, Group, Image, PatternTile, Shape, Text
from .text import add_texts

__all__ = [
    "MAX_RASTER_PX",
    "DrawOptions",
    "Renderer",
    "emit_collection",
    "get_is_included",
    "is_batchable",
    "rasterize",
    "unique_key",
]

# Cap on any off-screen image's width or height, in pixels.
MAX_RASTER_PX = 4096
# Gradient images are evaluated at most this many pixels across.
_MAX_GRADIENT_PX = 512
# Give up on patterns that would need more tiles than this.
_MAX_PATTERN_TILES = 250_000
# Segments a gradient stroke is split into.
_GRADIENT_STROKE_SEGMENTS = 400
# Rec. 709 luminance weights, used for luminance masks.
_MASK_LUMA = np.array([0.2125, 0.7154, 0.0721])


def get_is_included(id_: str, exclude: str | Iterable[str]) -> bool:
    """Return whether ``id_`` survives the ``exclude`` regular expressions."""
    if isinstance(exclude, str):
        exclude = (exclude,)
    return not any(re.match(r, id_) for r in exclude)


def unique_key(d: Mapping, key: str, counts: dict | None = None) -> str:
    """A dict key for ``key`` that does not clobber an earlier duplicate id.

    Missing ids become ``__shape_N``; duplicates get ``_1``, ``_2`` ...
    suffixes. ``counts`` (a dict reused across calls) keeps this linear for
    drawings with many repeated ids.
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


# --------------------------------------------------------------------------- #
# Per-shape styling with user overrides
# --------------------------------------------------------------------------- #
@dataclass
class DrawOptions:
    """Render-time options shared by every item."""

    zorder: float = 1
    density: float = 1.0  # off-screen pixels per document unit
    grayscale: bool = False
    text_as_paths: bool = True
    per_path_kwargs: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    global_kwargs: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class _Stroke:
    """How an artist's line width is managed.

    ``width`` is in document units (rescaled on every draw), or ``None`` when
    the user gave a width in points that must be left alone.
    """

    width: float | None
    dashes: tuple | None
    scaled: bool


@dataclass
class _Entry:
    kwargs: dict
    stroke: _Stroke
    fill_paint: BoundPaint | None
    stroke_paint: BoundPaint | None


def _entry(shape: Shape, opts: DrawOptions) -> _Entry:
    """A shape's matplotlib kwargs with user overrides applied."""
    kwargs = dict(shape.kwargs)
    width = kwargs.pop("linewidth", 1.0)
    dashes = kwargs.pop("linestyle", None)
    if opts.grayscale:
        kwargs["facecolor"] = gray_rgba(kwargs.get("facecolor"))
        kwargs["edgecolor"] = gray_rgba(kwargs.get("edgecolor"))
    kwargs.update(opts.global_kwargs)
    for prop, mapping in opts.per_path_kwargs.items():
        if shape.id in mapping:
            kwargs[prop] = mapping[shape.id]
    for short, long in (("lw", "linewidth"), ("ls", "linestyle")):
        if short in kwargs:
            kwargs[long] = kwargs.pop(short)
    user_width = "linewidth" in kwargs
    user_style = "linestyle" in kwargs
    if not user_width:
        kwargs["linewidth"] = width  # replaced at draw time
    kwargs.setdefault("snap", False)  # browsers do not snap edges to pixels
    # A fill color given by the user replaces the gradient/pattern.
    fill_paint = shape.fill_paint if kwargs.get("facecolor") == NONE else None
    return _Entry(
        kwargs=kwargs,
        stroke=_Stroke(
            width=None if user_width else width,
            dashes=None if user_style else dashes,
            scaled=shape.scale_stroke,
        ),
        fill_paint=fill_paint,
        stroke_paint=shape.stroke_paint,
    )


def _center(path: Path) -> tuple[float, float]:
    ext = path_extents(path)
    if ext is None:
        return (0.0, 0.0)
    return ((ext[0] + ext[2]) / 2, (ext[1] + ext[3]) / 2)


# --------------------------------------------------------------------------- #
# The renderer
# --------------------------------------------------------------------------- #
class Renderer:
    """Emit scene items as matplotlib artists on one axes.

    ``doc_to_data`` maps document space (y down) to data coordinates (flip and
    placement); ``base`` is usually ``ax.transData``. Shape paths are handed
    to matplotlib with the y-flip ``flip`` baked in, as earlier releases did,
    so ``patch.get_path()`` reads in flipped coordinates.
    """

    def __init__(
        self,
        ax,
        doc_to_data: Affine2D,
        base: Transform,
        options: DrawOptions,
        flip: Affine2D | None = None,
    ):
        self.ax = ax
        self.opts = options
        self.doc_to_data = doc_to_data
        self.base = base
        self.full = doc_to_data + base
        self.flip = flip if flip is not None else Affine2D()
        self.shape_transform = (self.flip.inverted() + doc_to_data) + base
        self.patches: dict[str, SVGPathPatch] = {}
        self.extra: list = []
        self._key_counts: dict = {}
        self._limits: list[Rect] = []
        self._texts_done: set[int] = set()

    # -- helpers -------------------------------------------------------- #
    def _clip_patch(self, path: Path) -> PathPatch:
        return PathPatch(path, transform=self.full, facecolor="none", edgecolor="none")

    def _set_clip(self, artist, clip: Path | None) -> None:
        if clip is not None:
            artist.set_clip_path(self._clip_patch(clip))

    def _add(self, artist):
        artist.set_zorder(self.opts.zorder)
        self.extra.append(artist)
        return artist

    def _add_image(
        self,
        data: np.ndarray,
        extent: tuple[float, float, float, float],
        to_doc: Affine2D | None = None,
        interpolation: str = "antialiased",
        alpha: float | None = None,
        clip: Path | None = None,
    ) -> AxesImage:
        """Add an image whose ``extent`` is in coordinates mapped by ``to_doc``.

        ``Axes.imshow`` would reset the axes limits and aspect from the extent,
        so the artist is built directly and only the data limits are updated.
        """
        im = AxesImage(
            self.ax, interpolation=interpolation, origin="upper", extent=extent
        )
        im.set_data(data)
        if alpha is not None:
            im.set_alpha(alpha)
        to_doc = Affine2D() if to_doc is None else Affine2D(to_doc.get_matrix())
        im.set_transform(to_doc + self.full)
        self._set_clip(im, clip)
        self.ax.add_image(im)
        if self.base is self.ax.transData:
            x0, x1, y1, y0 = extent
            corners = (to_doc + self.doc_to_data).transform(
                [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
            )
            self.ax.update_datalim(corners)
        return self._add(im)

    def _px_grid(self, ext: Rect, density: float | None = None) -> tuple[int, int]:
        density = density or self.opts.density
        x0, y0, x1, y1 = ext
        w = max(2, min(MAX_RASTER_PX, math.ceil((x1 - x0) * density)))
        h = max(2, min(MAX_RASTER_PX, math.ceil((y1 - y0) * density)))
        return w, h

    def _gray(self, img: np.ndarray) -> np.ndarray:
        return gray_image(img) if self.opts.grayscale else img

    # -- items ------------------------------------------------------------ #
    def draw(self, items: Iterable, collect: bool = True) -> None:
        """Draw ``items``; with ``collect`` patches are recorded by id."""
        for item in items:
            clip = getattr(item, "clip", None)
            if clip is not None and is_empty(clip):
                continue
            if isinstance(item, Shape):
                if item.text is not None and not self.opts.text_as_paths:
                    self._text_artist(item.text)
                else:
                    self._shape(item, collect)
            elif isinstance(item, Image):
                self._image(item)
            elif isinstance(item, Group):
                self._group(item)

    def finish(self) -> None:
        """Fold the drawn shapes' extents into the axes data limits."""
        if not self._limits or self.base is not self.ax.transData:
            self._limits = []
            return
        ext = np.array(self._limits)
        x0, y0 = ext[:, 0].min(), ext[:, 1].min()
        x1, y1 = ext[:, 2].max(), ext[:, 3].max()
        corners = np.array([[x0, y0], [x1, y1], [x0, y1], [x1, y0]])
        self._limits = []
        to_data = self.shape_transform - self.ax.transData
        self.ax.update_datalim(to_data.transform(corners))

    def _text_artist(self, record: Text) -> None:
        if id(record) in self._texts_done:
            return
        self._texts_done.add(id(record))
        self.extra.extend(
            add_texts(self.ax, [record], placement=self.doc_to_data, base=self.base)
        )

    def _shape(self, shape: Shape, collect: bool = True) -> None:
        entry = _entry(shape, self.opts)
        if entry.fill_paint is not None:
            self._paint_fill(shape, entry.fill_paint)
        gradient_stroke = entry.stroke_paint
        if (
            gradient_stroke is None
            or not isinstance(gradient_stroke.server, Gradient)
            or entry.stroke.width is None
            or entry.kwargs.get("edgecolor") == NONE
        ):
            gradient_stroke = None  # solid fallback color (or user override)
        else:
            entry.kwargs["edgecolor"] = NONE  # drawn as segments after the fill
        path = shape.path.transformed(self.flip)
        patch = SVGPathPatch(path, **entry.kwargs)
        if entry.stroke.width is not None:
            patch.set_svg_stroke(
                entry.stroke.width,
                entry.stroke.dashes,
                entry.stroke.scaled,
                _center(path),
            )
        patch.set_transform(self.shape_transform)
        self._set_clip(patch, shape.clip)
        patch.set_zorder(self.opts.zorder)
        # add_artist + one bulk data-limit update: Axes.add_patch computes exact
        # curve extents per patch, which dominates on large drawings.
        self.ax.add_artist(patch)
        ext = path_extents(path)
        if ext is not None:
            self._limits.append(ext)
        if collect:
            self.patches[unique_key(self.patches, shape.id, self._key_counts)] = patch
        else:
            self.extra.append(patch)
        if gradient_stroke is not None:
            self._gradient_stroke(shape, gradient_stroke, entry.stroke)

    def _image(self, item: Image) -> None:
        data = self._gray(item.data)
        h, w = data.shape[:2]
        interp = item.interpolation
        if interp == "auto":
            # Browsers smooth up-scaled images; antialias down-scaled ones.
            px = area_scale(item.transform) * self.opts.density
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
    def _paint_fill(self, shape: Shape, bound: BoundPaint) -> None:
        region: Path = shape.path
        clip_extra = None  # a clip that could not be merged into the outline
        if shape.clip is not None:
            try:
                region = intersect_clip(shape.path, shape.clip) or region
            except ValueError:
                clip_extra = shape.clip
        ext = path_extents(region)
        if ext is None or rect_is_empty(ext):
            return
        outline = shape.path if clip_extra is not None else region
        if isinstance(bound.server, Gradient):
            self._gradient_fill(bound, ext, outline, clip_extra)
        elif bound.tile is not None:
            self._pattern(bound.tile, ext, outline, bound.opacity)

    def _gradient_fill(
        self, bound: BoundPaint, ext: Rect, outline: Path, clip_extra: Path | None
    ) -> None:
        size = max(ext[2] - ext[0], ext[3] - ext[1])
        w, h = self._px_grid(
            ext, density=min(self.opts.density, _MAX_GRADIENT_PX / size)
        )
        xs = ext[0] + (np.arange(w) + 0.5) * (ext[2] - ext[0]) / w
        ys = ext[1] + (np.arange(h) + 0.5) * (ext[3] - ext[1]) / h
        gx, gy = np.meshgrid(xs, ys)
        pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
        try:
            user = bound.transform.inverted().transform(pts)
        except np.linalg.LinAlgError:
            return  # degenerate transform: nothing visible
        rgba = bound.server.evaluate(user.reshape(h, w, 2), bound.bbox)
        rgba[..., 3] *= bound.opacity
        if clip_extra is not None:
            # Bake the second clip into the alpha (hard-edged at this resolution).
            rgba[..., 3] *= clip_extra.contains_points(pts).reshape(h, w)
        self._add_image(
            np.clip(self._gray(rgba), 0, 1),
            (ext[0], ext[2], ext[3], ext[1]),
            interpolation="bilinear",
            clip=outline,
        )

    def _pattern(self, tile: PatternTile, ext: Rect, outline: Path, opacity: float):
        try:
            inv = tile.transform.inverted()
        except np.linalg.LinAlgError:
            return
        x0, y0, x1, y1 = ext
        corners = inv.transform([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
        i0 = math.floor(corners[:, 0].min() / tile.width)
        i1 = math.ceil(corners[:, 0].max() / tile.width)
        j0 = math.floor(corners[:, 1].min() / tile.height)
        j1 = math.ceil(corners[:, 1].max() / tile.height)
        ni, nj = max(i1 - i0, 1), max(j1 - j0, 1)
        if ni * nj > _MAX_PATTERN_TILES:
            return
        # Tile pixel size from the doc-space density, within a total budget.
        m = tile.transform.get_matrix()
        sx = math.hypot(m[0, 0], m[1, 0]) * tile.width
        sy = math.hypot(m[0, 1], m[1, 1]) * tile.height
        budget = MAX_RASTER_PX * 2
        tw = min(max(2, math.ceil(sx * self.opts.density)), max(2, budget // ni))
        th = min(max(2, math.ceil(sy * self.opts.density)), max(2, budget // nj))
        img = rasterize(tile.items, (0.0, 0.0, tile.width, tile.height), (th, tw))
        img[..., 3] *= opacity
        big = np.tile(self._gray(img), (nj, ni, 1))
        ox, oy = i0 * tile.width, j0 * tile.height
        self._add_image(
            np.clip(big, 0, 1),
            (ox, ox + ni * tile.width, oy + nj * tile.height, oy),
            to_doc=tile.transform,
            clip=outline,
        )

    def _gradient_stroke(
        self, shape: Shape, bound: BoundPaint, stroke: _Stroke
    ) -> None:
        """Draw a gradient stroke as short segments colored by the gradient."""
        segs, mids = [], []
        for line in _stroke_polylines(shape.path):
            if len(line) < 2:
                continue
            fine = _subdivide(line, _GRADIENT_STROKE_SEGMENTS)
            segs.append(np.stack([fine[:-1], fine[1:]], axis=1))
            mids.append((fine[:-1] + fine[1:]) / 2)
        if not segs:
            return
        try:
            user = bound.transform.inverted().transform(np.concatenate(mids))
        except np.linalg.LinAlgError:
            return
        colors = bound.server.evaluate(user, bound.bbox)
        colors[..., 3] *= bound.opacity
        colors = self._gray(colors[None])[0]
        lc = SVGLineCollection(
            list(np.concatenate(segs)),
            colors=np.clip(colors, 0, 1),
            capstyle="round",
            joinstyle="round",
        )
        lc.set_svg_stroke(stroke.width, None, stroke.scaled, _center(shape.path))
        lc.set_transform(self.full)
        self._set_clip(lc, shape.clip)
        self.ax.add_collection(lc, autolim=False)
        self._add(lc)

    # -- off-screen groups -------------------------------------------------- #
    def _group(self, group: Group) -> None:
        if rect_is_empty(group.region) or not group.items:
            return
        x0, y0, x1, y1 = group.region
        w, h = self._px_grid(group.region)
        img = rasterize(group.items, group.region, (h, w))
        ppx, ppy = w / (x1 - x0), h / (y1 - y0)
        if group.filter is not None:
            img = apply_filter(group.filter, img, group.region, ppx, ppy, rasterize)
        if group.mask is not None:
            m = rasterize(group.mask.items, group.region, (h, w))
            if group.mask.kind == "alpha":
                factor = m[..., 3]
            else:
                factor = (m[..., :3] @ _MASK_LUMA) * m[..., 3]
            mx0, my0, mx1, my1 = group.mask.region
            xs = x0 + (np.arange(w) + 0.5) / ppx
            ys = y0 + (np.arange(h) + 0.5) / ppy
            in_y = (ys >= my0) & (ys <= my1)
            in_x = (xs >= mx0) & (xs <= mx1)
            img[..., 3] *= factor * (in_y[:, None] & in_x[None, :])
        img[..., 3] *= group.opacity
        self._add_image(
            np.clip(self._gray(img), 0, 1), (x0, x1, y1, y0), clip=group.clip
        )


def _stroke_polylines(path: Path) -> list[np.ndarray]:
    """Open polylines of every subpath (closed subpaths repeat their start)."""
    out = []
    for sub in split_subpaths(path):
        closed = sub.codes is not None and sub.codes[-1] == Path.CLOSEPOLY
        out.extend(np.vstack([r, r[:1]]) if closed else r for r in flatten(sub))
    return out


def _subdivide(poly: np.ndarray, n: int) -> np.ndarray:
    """Resample a polyline into about ``n`` pieces proportional to length."""
    seg = np.diff(poly, axis=0)
    length = np.hypot(seg[:, 0], seg[:, 1])
    total = length.sum()
    if total <= 0:
        return poly
    out = [poly[:1]]
    for p, d, seg_len in zip(poly[:-1], seg, length, strict=True):
        k = max(1, round(n * seg_len / total))
        t = np.linspace(0, 1, k + 1)[1:, None]
        out.append(p + t * d)
    return np.vstack(out)


def rasterize(items: list, region: Rect, shape: tuple[int, int]) -> np.ndarray:
    """Render ``items`` (document space) over ``region`` into straight RGBA.

    ``region`` is ``(x0, y0, x1, y1)`` in the items' coordinates (y down) and
    ``shape`` the output ``(height, width)``. Row 0 is ``y0``.
    """
    h, w = int(shape[0]), int(shape[1])
    dpi = 100.0
    # A hair extra so float rounding cannot lose the last pixel row/column.
    fig = Figure(figsize=((w + 0.01) / dpi, (h + 0.01) / dpi), dpi=dpi)
    canvas = FigureCanvasAgg(fig)
    fig.patch.set_alpha(0.0)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_axis_off()
    ax.patch.set_alpha(0.0)
    x0, y0, x1, y1 = region
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    options = DrawOptions(density=w / (x1 - x0) if x1 > x0 else 1.0)
    Renderer(ax, Affine2D(), ax.transData, options).draw(items, collect=False)
    canvas.draw()
    buf = np.asarray(canvas.buffer_rgba(), dtype=float) / 255.0
    out = np.zeros((h, w, 4))
    hh, ww = min(h, buf.shape[0]), min(w, buf.shape[1])
    out[:hh, :ww] = buf[:hh, :ww]
    return out


def is_batchable(items: list) -> bool:
    """Whether every item is a plain shape that a single collection can draw."""
    return all(
        isinstance(item, Shape)
        and item.clip is None
        and item.fill_paint is None
        and item.stroke_paint is None
        for item in items
    )


def emit_collection(r: Renderer, shapes: list[Shape]) -> SVGPathCollection | None:
    """Draw ``shapes`` as one collection, or return ``None`` if they cannot be
    batched (differing property sets or stroke handling)."""
    entries = [_entry(shape, r.opts) for shape in shapes]
    keys = set(entries[0].kwargs)
    if any(set(e.kwargs) != keys for e in entries):
        return None
    if len({e.stroke.scaled for e in entries}) > 1:
        return None
    managed = {e.stroke.width is not None for e in entries}
    if len(managed) > 1:
        return None
    props = {k: [e.kwargs[k] for e in entries] for k in keys}
    props["snap"] = props["snap"][0]  # a collection snaps as a whole
    paths = [shape.path.transformed(r.flip) for shape in shapes]
    collection = SVGPathCollection(paths, **props)
    if managed == {True}:
        dashes = [e.stroke.dashes for e in entries]
        if any(d is not None for d in dashes):
            solid = (0.0, [1.0, 0.0])
            dash_list = [d if d is not None else solid for d in dashes]
        else:
            dash_list = None
        collection.set_svg_stroke(
            [e.stroke.width or 0.0 for e in entries],
            dash_list,
            entries[0].stroke.scaled,
            _center(paths[0]),
        )
    collection.set_transform(r.shape_transform)
    collection.set_zorder(r.opts.zorder)
    r.ax.add_collection(collection)
    return collection
