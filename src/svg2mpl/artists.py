"""matplotlib artists whose stroke width follows the drawing's scale.

SVG stroke widths are lengths in user space: a 2-unit stroke on a drawing that
is scaled up 10x is 20 units wide. matplotlib line widths are in *points*,
independent of the data scale. These artists store the stroke width (and dash
pattern) in document units and convert them to points on every draw, using the
artist's current transform -- so strokes stay proportional when the drawing is
placed, resized, zoomed or saved at another DPI.
"""

from __future__ import annotations

import numpy as np
from matplotlib import rcParams
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.patches import PathPatch

__all__ = ["SVGPathPatch", "SVGPathCollection", "SVGLineCollection", "points_per_unit"]

# CSS px -> pt, for strokes that do not scale (vector-effect=non-scaling-stroke).
PX_TO_PT = 72.0 / 96.0


def points_per_unit(transform, renderer, ref_point) -> float:
    """Points per document unit of ``transform`` near ``ref_point``."""
    x, y = ref_point
    eps = 1.0
    try:
        p = transform.transform(np.array([[x, y], [x + eps, y], [x, y + eps]], float))
    except Exception:
        return 1.0
    jx, jy = p[1] - p[0], p[2] - p[0]
    det = abs(jx[0] * jy[1] - jx[1] * jy[0])
    if not np.isfinite(det) or det <= 0:
        return 0.0
    px_per_unit = np.sqrt(det) / eps
    return float(px_per_unit / renderer.points_to_pixels(1.0))


def _dash(dashes, k, lw_pt):
    """Dash ``(offset, [on, off, ...])`` in doc units -> matplotlib linestyle."""
    if dashes is None:
        return "solid"
    off, seq = dashes
    seq = [max(d * k, 1e-6) for d in seq]
    off = off * k
    if rcParams["lines.scale_dashes"] and lw_pt > 0:
        seq = [d / lw_pt for d in seq]
        off = off / lw_pt
    return (off, seq)


class _ScaledStroke:
    """Mixin: recompute line width / dashes from document units at draw time."""

    _svg_lw = None  # scalar or per-item list, in document units
    _svg_dashes = None  # (offset, seq) or per-item list, in document units
    _svg_scaled = True  # False: _svg_lw is in CSS px (non-scaling strokes)
    _svg_ref = (0.0, 0.0)

    def set_svg_stroke(self, linewidth, dashes=None, scaled=True, ref=(0.0, 0.0)):
        self._svg_lw = linewidth
        self._svg_dashes = dashes
        self._svg_scaled = scaled
        self._svg_ref = ref
        self._svg_last = None

    def _svg_update(self, renderer) -> None:
        if self._svg_lw is None:
            return
        if self._svg_scaled:
            k = points_per_unit(self.get_transform(), renderer, self._svg_ref)
        else:
            k = PX_TO_PT
        if getattr(self, "_svg_last", None) == k:
            return
        self._svg_last = k
        self._svg_apply(k)

    def draw(self, renderer):
        if self.get_visible():
            stale = self.stale
            self._svg_update(renderer)
            self.stale = stale
        return super().draw(renderer)


class SVGPathPatch(_ScaledStroke, PathPatch):
    """A :class:`~matplotlib.patches.PathPatch` with a scale-aware stroke."""

    def _svg_apply(self, k):
        lw = self._svg_lw * k
        self.set_linewidth(lw)
        if self._svg_dashes is not None:
            self.set_linestyle(_dash(self._svg_dashes, k, lw))


class SVGPathCollection(_ScaledStroke, PathCollection):
    """A :class:`~matplotlib.collections.PathCollection` with scale-aware strokes."""

    def _svg_apply(self, k):
        lws = np.atleast_1d(np.asarray(self._svg_lw, float)) * k
        self.set_linewidth(lws)
        if self._svg_dashes is not None:
            dashes = self._svg_dashes
            if not isinstance(dashes, list):
                dashes = [dashes] * len(lws)
            styles = [_dash(d, k, lw) for d, lw in zip(dashes, lws, strict=False)]
            self.set_linestyle(styles)
            self.set_linewidth(lws)


class SVGLineCollection(_ScaledStroke, LineCollection):
    """A :class:`~matplotlib.collections.LineCollection` with scale-aware widths."""

    def _svg_apply(self, k):
        lw = np.atleast_1d(np.asarray(self._svg_lw, float)) * k
        self.set_linewidth(lw)
