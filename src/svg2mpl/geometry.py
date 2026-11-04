"""Geometry helpers on matplotlib paths.

matplotlib always fills with the *nonzero* winding rule and accepts only one
clip path per artist. This module bridges both gaps:

* :func:`evenodd_to_nonzero` re-orients subpaths by nesting depth so that a
  nonzero fill reproduces SVG's ``evenodd`` rule (exact for subpaths that do
  not self-intersect, which covers glyphs and icons with holes);
* :func:`intersect_clip` intersects two clip regions, exactly when either is
  convex (rectangles, circles, ellipses -- the overwhelmingly common clips) via
  Sutherland-Hodgman clipping of every ring, which preserves winding numbers.
"""

from __future__ import annotations

import numpy as np
from matplotlib.path import Path

__all__ = [
    "split_subpaths",
    "flatten",
    "evenodd_to_nonzero",
    "normalize_orientation",
    "reverse_path",
    "signed_area",
    "intersect_clip",
    "is_convex",
    "join_paths",
    "rect_path",
    "path_extents",
]

_SEG_LEN = {Path.MOVETO: 1, Path.LINETO: 1, Path.CURVE3: 2, Path.CURVE4: 3}


def rect_path(x0: float, y0: float, x1: float, y1: float) -> Path:
    """An axis-aligned closed rectangle."""
    return Path(
        [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)],
        [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY],
    )


def join_paths(paths) -> Path:
    """Concatenate paths into one compound path (empty paths are skipped)."""
    paths = [p for p in paths if p is not None and len(p.vertices)]
    if not paths:
        return Path(np.empty((0, 2)))
    if len(paths) == 1:
        return paths[0]
    return Path.make_compound_path(*paths)


def _curve_extrema(p: np.ndarray) -> np.ndarray:
    """Points at the axis-aligned extrema of Bezier segments ``p`` (n, k, 2)."""
    out = [p[:, 0], p[:, -1]]
    # Only segments whose control points leave their endpoints' range can
    # bulge past them; skip the root finding for the rest.
    lo = np.minimum(p[:, 0], p[:, -1])
    hi = np.maximum(p[:, 0], p[:, -1])
    ctrl = p[:, 1:-1]
    bulge = np.any((ctrl < lo[:, None]) | (ctrl > hi[:, None]), axis=(1, 2))
    if not bulge.any():
        return np.concatenate(out)
    p = p[bulge]
    ts = []
    if p.shape[1] == 3:  # quadratic: B'(t) = 0 -> t = (p0 - p1) / (p0 - 2p1 + p2)
        den = p[:, 0] - 2 * p[:, 1] + p[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            ts.append((p[:, 0] - p[:, 1]) / den)
    else:  # cubic: a t^2 + b t + c = 0 per axis
        a = -p[:, 0] + 3 * p[:, 1] - 3 * p[:, 2] + p[:, 3]
        b = 2 * (p[:, 0] - 2 * p[:, 1] + p[:, 2])
        c = p[:, 1] - p[:, 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            disc = np.sqrt(np.where(b * b - 4 * a * c >= 0, b * b - 4 * a * c, np.nan))
            lin = np.where(np.abs(a) < 1e-12, -c / b, np.nan)
            ts += [
                np.where(np.abs(a) < 1e-12, lin, (-b + disc) / (2 * a)),
                np.where(np.abs(a) < 1e-12, np.nan, (-b - disc) / (2 * a)),
            ]
    for t in ts:  # t: (n, 2), one root per axis
        for axis in (0, 1):
            tt = t[:, axis]
            ok = np.isfinite(tt) & (tt > 0) & (tt < 1)
            if not ok.any():
                continue
            seg = p[ok]
            u = tt[ok][:, None]
            if p.shape[1] == 3:
                pt = (
                    (1 - u) ** 2 * seg[:, 0]
                    + 2 * (1 - u) * u * seg[:, 1]
                    + u**2 * seg[:, 2]
                )
            else:
                pt = (
                    (1 - u) ** 3 * seg[:, 0]
                    + 3 * (1 - u) ** 2 * u * seg[:, 1]
                    + 3 * (1 - u) * u**2 * seg[:, 2]
                    + u**3 * seg[:, 3]
                )
            out.append(pt)
    return np.concatenate(out)


def path_extents(path: Path):
    """Exact bounding box ``(x0, y0, x1, y1)`` of a path, or ``None`` if empty.

    A vectorized replacement for ``Path.get_extents`` (which evaluates curve
    extrema segment by segment and dominates parse time on large drawings).
    """
    if path is None or len(path.vertices) == 0:
        return None
    cached = getattr(path, "_svg2mpl_extents", None)
    if cached is not None:
        return cached
    verts = np.asarray(path.vertices, float)
    codes = path.codes
    if codes is None:
        pts = verts
    else:
        is_curve = (codes == Path.CURVE3) | (codes == Path.CURVE4)
        keep = (codes != Path.CLOSEPOLY) & ~is_curve & (codes != Path.STOP)
        pieces = [verts[keep]]
        for code, k in ((Path.CURVE3, 2), (Path.CURVE4, 3)):
            idx = np.flatnonzero(codes == code)
            if len(idx) == 0:
                continue
            # Consecutive codes of one kind come in groups of k per segment;
            # a segment starts at the vertex just before its first control.
            runs = idx.reshape(-1, k) if len(idx) % k == 0 else None
            if runs is None:
                pieces.append(verts[idx])
                continue
            starts = runs[:, 0] - 1
            seg = np.stack(
                [verts[starts]] + [verts[runs[:, j]] for j in range(k)], axis=1
            )
            pieces.append(_curve_extrema(seg))
        pts = np.concatenate(pieces) if pieces else verts
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if len(pts) == 0:
        return None
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    result = (float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1]))
    try:
        path._svg2mpl_extents = result
    except AttributeError:
        pass
    return result


def split_subpaths(path: Path) -> list[Path]:
    """Split a path into its subpaths (each starting with a ``MOVETO``)."""
    verts = path.vertices
    codes = path.codes
    if codes is None:
        return [path] if len(verts) else []
    starts = list(np.flatnonzero(codes == Path.MOVETO))
    if not starts or starts[0] != 0:
        starts = [0] + starts
    bounds = starts + [len(verts)]
    out = []
    for a, b in zip(bounds[:-1], bounds[1:], strict=False):
        if b > a:
            out.append(Path(verts[a:b], codes[a:b]))
    return out


def _bezier_points(p: np.ndarray, n: int) -> np.ndarray:
    """Sample a Bezier with control points ``p`` at ``n`` points (excluding t=0)."""
    t = np.linspace(0.0, 1.0, n + 1)[1:, None]
    if len(p) == 3:
        return (1 - t) ** 2 * p[0] + 2 * (1 - t) * t * p[1] + t**2 * p[2]
    return (
        (1 - t) ** 3 * p[0]
        + 3 * (1 - t) ** 2 * t * p[1]
        + 3 * (1 - t) * t**2 * p[2]
        + t**3 * p[3]
    )


def flatten(path: Path, tolerance: float | None = None) -> list[np.ndarray]:
    """Flatten a path into polygons (one ``(n, 2)`` array per subpath).

    Every subpath is treated as closed (as for filling). ``tolerance`` is an
    approximate chord deviation; by default it scales with the path size.
    """
    verts = path.vertices
    codes = path.codes
    if len(verts) == 0:
        return []
    if codes is None:
        return [np.asarray(verts, float)]
    if tolerance is None:
        ext = np.ptp(verts, axis=0).max() if len(verts) > 1 else 1.0
        tolerance = max(ext, 1e-9) * 1e-3
    rings: list[np.ndarray] = []
    cur: list[np.ndarray] = []
    i = 0
    n = len(verts)
    while i < n:
        code = codes[i]
        if code == Path.MOVETO:
            if len(cur) > 1:
                rings.append(np.array(cur))
            cur = [verts[i]]
            i += 1
        elif code == Path.LINETO:
            if not cur:
                cur = [verts[i]]
            else:
                cur.append(verts[i])
            i += 1
        elif code in (Path.CURVE3, Path.CURVE4):
            k = 2 if code == Path.CURVE3 else 3
            p0 = cur[-1] if cur else verts[i]
            ctrl = np.vstack([p0, verts[i : i + k]])
            size = np.ptp(ctrl, axis=0).max()
            m = int(np.clip(np.sqrt(size / tolerance) * 2, 2, 64))
            cur.extend(_bezier_points(ctrl, m))
            i += k
        elif code == Path.CLOSEPOLY:
            if len(cur) > 1:
                rings.append(np.array(cur))
            cur = [cur[0]] if cur else []
            # A following non-MOVETO continues from the subpath start.
            i += 1
            if i < n and codes[i] != Path.MOVETO:
                continue
            cur = []
        else:  # STOP or unknown
            i += 1
    if len(cur) > 1:
        rings.append(np.array(cur))
    return rings


def _signed_area(ring: np.ndarray) -> float:
    x, y = ring[:, 0], ring[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _point_in_ring(pt, ring: np.ndarray) -> bool:
    return Path(ring).contains_point(pt)


def _reverse_subpath(sub: Path) -> Path:
    """Reverse a subpath's direction, keeping curve segments intact."""
    verts = sub.vertices
    codes = sub.codes if sub.codes is not None else np.full(len(verts), Path.LINETO)
    closed = codes[-1] == Path.CLOSEPOLY
    if closed:
        verts, codes = verts[:-1], codes[:-1]
    # Decompose into segments: (code, points-after-start).
    segs = []
    i = 1
    while i < len(verts):
        k = _SEG_LEN.get(int(codes[i]), 1)
        segs.append((int(codes[i]), verts[i : i + k]))
        i += k
    pts = [verts[0]] + [s[1][-1] for s in segs]
    new_v = [pts[-1]]
    new_c = [Path.MOVETO]
    for j in range(len(segs) - 1, -1, -1):
        code, seg = segs[j]
        start = pts[j]
        if code in (Path.CURVE3, Path.CURVE4):
            ctrl = list(seg[:-1])[::-1]
            new_v.extend(ctrl + [start])
            new_c.extend([code] * (len(ctrl) + 1))
        else:
            new_v.append(start)
            new_c.append(Path.LINETO)
    if closed:
        new_v.append(new_v[0])
        new_c.append(Path.CLOSEPOLY)
    return Path(np.array(new_v, float), new_c)


def evenodd_to_nonzero(path: Path) -> Path:
    """Re-orient subpaths so a nonzero fill matches the ``evenodd`` rule."""
    subs = split_subpaths(path)
    if len(subs) < 2:
        return path
    flats = [flatten(s) for s in subs]
    rings = [f[0] if f else None for f in flats]
    out = []
    for k, (sub, ring) in enumerate(zip(subs, rings, strict=False)):
        if ring is None or len(ring) < 3:
            out.append(sub)
            continue
        # Depth = number of other rings containing a representative point.
        probe = ring[0] * 0.999 + ring[len(ring) // 2] * 0.001
        depth = sum(
            1
            for j, other in enumerate(rings)
            if j != k
            and other is not None
            and len(other) >= 3
            and _point_in_ring(probe, other)
        )
        want_ccw = depth % 2 == 0
        is_ccw = _signed_area(ring) > 0
        out.append(sub if want_ccw == is_ccw else _reverse_subpath(sub))
    return Path.make_compound_path(*out)


def signed_area(path: Path) -> float:
    """Total signed area of a path's (flattened) rings; positive = CCW."""
    return sum(_signed_area(r) for r in flatten(path) if len(r) >= 3)


def reverse_path(path: Path) -> Path:
    """Reverse the direction of every subpath."""
    subs = split_subpaths(path)
    if not subs:
        return path
    return join_paths([_reverse_subpath(s) for s in subs])


def normalize_orientation(path: Path) -> Path:
    """Orient a path so its filled area winds positively (for clip unions)."""
    return reverse_path(path) if signed_area(path) < 0 else path


def is_convex(path: Path) -> bool:
    """Whether a path is a single convex ring."""
    rings = flatten(path)
    if len(rings) != 1:
        return False
    return _ring_convex(rings[0])


def _dedupe(ring: np.ndarray) -> np.ndarray:
    if len(ring) < 2:
        return ring
    keep = np.ones(len(ring), bool)
    keep[1:] = np.any(np.abs(np.diff(ring, axis=0)) > 1e-12, axis=1)
    ring = ring[keep]
    if len(ring) > 1 and np.allclose(ring[0], ring[-1]):
        ring = ring[:-1]
    return ring


def _ring_convex(ring: np.ndarray) -> bool:
    ring = _dedupe(ring)
    if len(ring) < 3:
        return False
    d = np.roll(ring, -1, axis=0) - ring
    cross = d[:, 0] * np.roll(d[:, 1], -1) - d[:, 1] * np.roll(d[:, 0], -1)
    scale = np.abs(cross).max() or 1.0
    cross = cross[np.abs(cross) > 1e-9 * scale]
    return bool(np.all(cross > 0) or np.all(cross < 0))


def _clip_ring(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    """Sutherland-Hodgman: clip ``subject`` by the convex CCW polygon ``clip``."""
    out = subject
    m = len(clip)
    for k in range(m):
        if len(out) == 0:
            break
        a, b = clip[k], clip[(k + 1) % m]
        edge = b - a
        inp = out
        side = edge[0] * (inp[:, 1] - a[1]) - edge[1] * (inp[:, 0] - a[0])
        res = []
        n = len(inp)
        for i in range(n):
            p, q = inp[i], inp[(i + 1) % n]
            sp, sq = side[i], side[(i + 1) % n]
            if sp >= 0:
                res.append(p)
            if (sp >= 0) != (sq >= 0):
                t = sp / (sp - sq)
                res.append(p + t * (q - p))
        out = np.array(res) if res else np.empty((0, 2))
    return out


def _clip_path_by_convex(subject: Path, convex: Path) -> Path:
    ring = _dedupe(flatten(convex)[0])
    if _signed_area(ring) < 0:
        ring = ring[::-1]
    pieces = []
    for r in flatten(subject):
        r = _dedupe(r)
        if len(r) < 3:
            continue
        c = _clip_ring(r, ring)
        if len(c) >= 3:
            pieces.append(
                Path(
                    np.vstack([c, c[:1]]),
                    [Path.MOVETO] + [Path.LINETO] * (len(c) - 1) + [Path.CLOSEPOLY],
                )
            )
    return join_paths(pieces)


def intersect_clip(a: Path | None, b: Path | None) -> Path | None:
    """Intersect two (nonzero-rule) clip regions.

    Returns the intersection, ``a``/``b`` when the other is ``None``, or raises
    :class:`ValueError` when neither region is convex (no exact method).
    """
    if a is None:
        return b
    if b is None:
        return a
    ea, eb = path_extents(a), path_extents(b)
    if ea is None or eb is None:
        return Path(np.empty((0, 2)))
    # Disjoint boxes: empty intersection.
    if ea[2] < eb[0] or eb[2] < ea[0] or ea[3] < eb[1] or eb[3] < ea[1]:
        return Path(np.empty((0, 2)))
    if is_convex(b):
        return _clip_path_by_convex(a, b)
    if is_convex(a):
        return _clip_path_by_convex(b, a)
    raise ValueError("cannot intersect two non-convex clip regions exactly")
