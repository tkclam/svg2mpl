"""Parse the SVG path ``d`` attribute into a matplotlib :class:`~matplotlib.path.Path`.

Implements the full SVG 1.1/2 path grammar: ``M L H V C S Q T A Z`` in absolute
and relative forms, implicit command repetition, compact numbers (``1.5.5``,
``1e-3``) and compact arc flags (``a1 1 0 00 1 1``). Elliptical arcs are converted
to cubic Beziers using the endpoint-to-center conversion of SVG appendix B.2.4.

Following the spec's error handling, parsing stops at the first malformed token
and everything parsed up to that point is kept.

Besides the geometry, the parser records each path *vertex* (the end point of
every command) with its incoming/outgoing tangent so markers can be placed at
the vertices of the original commands rather than at intermediate Bezier
joints introduced by arc conversion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from matplotlib.path import Path

from .geometry import empty_path
from .units import NUMBER_RE

__all__ = [
    "ParsedPath",
    "Subpath",
    "empty_path",
    "marker_vertices",
    "parse_path",
    "parse_path_data",
    "polyline_path_data",
]

_WS = " \t\r\n\f,"
_ARGC = {"M": 2, "L": 2, "H": 1, "V": 1, "C": 6, "S": 4, "Q": 4, "T": 2, "A": 7}


Point = tuple[float, float]
Direction = tuple[float, float]  # a unit vector

# Path codes as plain ints (matplotlib defines them as numpy uint8).
_MOVETO, _LINETO = int(Path.MOVETO), int(Path.LINETO)
_CURVE3, _CURVE4, _CLOSEPOLY = int(Path.CURVE3), int(Path.CURVE4), int(Path.CLOSEPOLY)


@dataclass
class Subpath:
    """Vertex bookkeeping for one subpath (used for marker placement)."""

    start: tuple[float, float]
    # (end_point, direction_at_segment_start, direction_at_segment_end)
    segments: list[tuple[Point, Direction | None, Direction | None]] = field(
        default_factory=list
    )
    closed: bool = False


@dataclass
class ParsedPath:
    """A parsed path: matplotlib geometry plus per-command vertex data."""

    path: Path
    subpaths: list[Subpath]


class _Scanner:
    def __init__(self, text: str):
        self.s = text
        self.i = 0
        self.n = len(text)

    def skip(self) -> None:
        s, i, n = self.s, self.i, self.n
        while i < n and s[i] in _WS:
            i += 1
        self.i = i

    def done(self) -> bool:
        self.skip()
        return self.i >= self.n

    def peek_command(self) -> str | None:
        self.skip()
        if self.i < self.n and self.s[self.i].isalpha() and self.s[self.i] not in "eE":
            return self.s[self.i]
        return None

    def number(self) -> float:
        self.skip()
        m = NUMBER_RE.match(self.s, self.i)
        if not m:
            raise ValueError("expected number")
        self.i = m.end()
        return float(m.group())

    def flag(self) -> bool:
        self.skip()
        if self.i < self.n and self.s[self.i] in "01":
            self.i += 1
            return self.s[self.i - 1] == "1"
        raise ValueError("expected flag")

    def args(self, cmd: str) -> list[float]:
        if cmd.upper() == "A":
            rx, ry, rot = self.number(), self.number(), self.number()
            fa, fs = self.flag(), self.flag()
            return [rx, ry, rot, float(fa), float(fs), self.number(), self.number()]
        return [self.number() for _ in range(_ARGC[cmd.upper()])]

    def has_number(self) -> bool:
        self.skip()
        return self.i < self.n and NUMBER_RE.match(self.s, self.i) is not None


def _unit(dx: float, dy: float):
    n = math.hypot(dx, dy)
    return (dx / n, dy / n) if n > 1e-12 else None


def _first_dir(p0, *pts):
    for p in pts:
        d = _unit(p[0] - p0[0], p[1] - p0[1])
        if d is not None:
            return d
    return None


def _last_dir(p3, *pts):
    for p in pts:
        d = _unit(p3[0] - p[0], p3[1] - p[1])
        if d is not None:
            return d
    return None


def _arc_to_cubics(x1, y1, rx, ry, phi_deg, large, sweep, x2, y2):
    """Return a list of ``(c1, c2, p3)`` cubic segments approximating an arc."""
    rx, ry = abs(rx), abs(ry)
    phi = math.radians(phi_deg % 360.0)
    cos_p, sin_p = math.cos(phi), math.sin(phi)
    dx2, dy2 = (x1 - x2) / 2.0, (y1 - y2) / 2.0
    x1p = cos_p * dx2 + sin_p * dy2
    y1p = -sin_p * dx2 + cos_p * dy2
    lam = (x1p * x1p) / (rx * rx) + (y1p * y1p) / (ry * ry)
    if lam > 1.0:
        s = math.sqrt(lam)
        rx, ry = rx * s, ry * s
    num = rx * rx * ry * ry - rx * rx * y1p * y1p - ry * ry * x1p * x1p
    den = rx * rx * y1p * y1p + ry * ry * x1p * x1p
    coef = math.sqrt(max(0.0, num / den)) if den else 0.0
    if large == sweep:
        coef = -coef
    cxp = coef * rx * y1p / ry
    cyp = -coef * ry * x1p / rx
    cx = cos_p * cxp - sin_p * cyp + (x1 + x2) / 2.0
    cy = sin_p * cxp + cos_p * cyp + (y1 + y2) / 2.0

    def angle(ux, uy, vx, vy):
        return math.atan2(ux * vy - uy * vx, ux * vx + uy * vy)

    ux, uy = (x1p - cxp) / rx, (y1p - cyp) / ry
    vx, vy = (-x1p - cxp) / rx, (-y1p - cyp) / ry
    theta1 = angle(1.0, 0.0, ux, uy)
    dtheta = angle(ux, uy, vx, vy)
    if not sweep and dtheta > 0:
        dtheta -= 2 * math.pi
    elif sweep and dtheta < 0:
        dtheta += 2 * math.pi

    n = max(1, math.ceil(abs(dtheta) / (math.pi / 2) - 1e-9))
    delta = dtheta / n
    t = 4.0 / 3.0 * math.tan(delta / 4.0)

    def to_user(px, py):
        return (
            cx + rx * cos_p * px - ry * sin_p * py,
            cy + rx * sin_p * px + ry * cos_p * py,
        )

    out = []
    a1 = theta1
    for k in range(n):
        a2 = a1 + delta
        c1 = to_user(math.cos(a1) - t * math.sin(a1), math.sin(a1) + t * math.cos(a1))
        c2 = to_user(math.cos(a2) + t * math.sin(a2), math.sin(a2) - t * math.cos(a2))
        p3 = (x2, y2) if k == n - 1 else to_user(math.cos(a2), math.sin(a2))
        out.append((c1, c2, p3))
        a1 = a2
    return out


class _Builder:
    def __init__(self):
        self.verts: list[tuple[float, float]] = []
        self.codes: list[int] = []
        self.subpaths: list[Subpath] = []
        self.cur = (0.0, 0.0)
        self.start = (0.0, 0.0)
        self.need_move = True

    def move(self, p):
        self.verts.append(p)
        self.codes.append(_MOVETO)
        self.cur = self.start = p
        self.subpaths.append(Subpath(start=p))
        self.need_move = False

    def _ensure(self):
        if self.need_move:
            self.move(self.cur)

    def line(self, p):
        self._ensure()
        p0 = self.cur
        d = _unit(p[0] - p0[0], p[1] - p0[1])
        self.verts.append(p)
        self.codes.append(_LINETO)
        self.subpaths[-1].segments.append((p, d, d))
        self.cur = p

    def quad(self, c, p):
        self._ensure()
        p0 = self.cur
        self.verts += [c, p]
        self.codes += [_CURVE3, _CURVE3]
        self.subpaths[-1].segments.append(
            (p, _first_dir(p0, c, p), _last_dir(p, c, p0))
        )
        self.cur = p

    def cubic(self, c1, c2, p, record=True):
        self._ensure()
        p0 = self.cur
        self.verts += [c1, c2, p]
        self.codes += [_CURVE4] * 3
        if record:
            self.subpaths[-1].segments.append(
                (p, _first_dir(p0, c1, c2, p), _last_dir(p, c2, c1, p0))
            )
        self.cur = p

    def arc(self, rx, ry, rot, large, sweep, p):
        self._ensure()
        p0 = self.cur
        if p0 == p:
            return
        if rx == 0 or ry == 0:
            self.line(p)
            return
        cubics = _arc_to_cubics(p0[0], p0[1], rx, ry, rot, large, sweep, p[0], p[1])
        first = _first_dir(p0, cubics[0][0], cubics[0][1], cubics[0][2])
        prev = cubics[-2][2] if len(cubics) > 1 else p0
        last = _last_dir(p, cubics[-1][1], cubics[-1][0], prev)
        for c1, c2, p3 in cubics:
            self.cubic(c1, c2, p3, record=False)
        self.subpaths[-1].segments.append((p, first, last))

    def close(self):
        if self.need_move:
            return
        p0, s = self.cur, self.start
        d = _unit(s[0] - p0[0], s[1] - p0[1])
        self.verts.append(s)
        self.codes.append(_CLOSEPOLY)
        sub = self.subpaths[-1]
        sub.segments.append((s, d, d))
        sub.closed = True
        self.cur = s
        self.need_move = True

    def result(self) -> ParsedPath:
        if not self.verts:
            return ParsedPath(empty_path(), [])
        return ParsedPath(Path(np.array(self.verts, float), self.codes), self.subpaths)


@lru_cache(maxsize=4096)
def parse_path_data(d: str | None) -> ParsedPath:
    """Parse ``d`` into geometry plus vertex data (see :class:`ParsedPath`).

    Results are cached (``<use>`` instances share their definition's ``d``);
    treat the returned object as read-only.
    """
    b = _Builder()
    if not d or not d.strip():
        return b.result()
    sc = _Scanner(d)
    cmd = None
    last_ctrl = None  # reflected control point for S/T
    last_kind = ""
    try:
        while not sc.done():
            c = sc.peek_command()
            if c is not None:
                sc.i += 1
                cmd = c
                if cmd.upper() not in _ARGC and cmd.upper() != "Z":
                    break
            elif cmd is None or cmd.upper() == "Z":
                break  # numbers without a command
            if cmd in "Zz":
                b.close()
                last_kind = "Z"
                last_ctrl = None
                # Allow numbers directly after Z only if a new command follows.
                if c is None:
                    break
                continue
            if c is None and not sc.has_number():
                break
            a = sc.args(cmd)
            rel = cmd.islower()
            up = cmd.upper()
            x0, y0 = b.cur
            if up == "M":
                p = (x0 + a[0], y0 + a[1]) if rel else (a[0], a[1])
                b.move(p)
                # Subsequent pairs are implicit lineto commands.
                cmd = "l" if rel else "L"
                last_ctrl, last_kind = None, "L"
                continue
            if up == "L":
                b.line((x0 + a[0], y0 + a[1]) if rel else (a[0], a[1]))
                last_ctrl, last_kind = None, "L"
            elif up == "H":
                b.line((x0 + a[0] if rel else a[0], y0))
                last_ctrl, last_kind = None, "L"
            elif up == "V":
                b.line((x0, y0 + a[0] if rel else a[0]))
                last_ctrl, last_kind = None, "L"
            elif up == "C":
                off = (x0, y0) if rel else (0.0, 0.0)
                c1 = (off[0] + a[0], off[1] + a[1])
                c2 = (off[0] + a[2], off[1] + a[3])
                p = (off[0] + a[4], off[1] + a[5])
                b.cubic(c1, c2, p)
                last_ctrl, last_kind = c2, "C"
            elif up == "S":
                off = (x0, y0) if rel else (0.0, 0.0)
                c1 = (
                    (2 * x0 - last_ctrl[0], 2 * y0 - last_ctrl[1])
                    if last_kind == "C" and last_ctrl
                    else (x0, y0)
                )
                c2 = (off[0] + a[0], off[1] + a[1])
                p = (off[0] + a[2], off[1] + a[3])
                b.cubic(c1, c2, p)
                last_ctrl, last_kind = c2, "C"
            elif up == "Q":
                off = (x0, y0) if rel else (0.0, 0.0)
                ctrl = (off[0] + a[0], off[1] + a[1])
                p = (off[0] + a[2], off[1] + a[3])
                b.quad(ctrl, p)
                last_ctrl, last_kind = ctrl, "Q"
            elif up == "T":
                off = (x0, y0) if rel else (0.0, 0.0)
                ctrl = (
                    (2 * x0 - last_ctrl[0], 2 * y0 - last_ctrl[1])
                    if last_kind == "Q" and last_ctrl
                    else (x0, y0)
                )
                p = (off[0] + a[0], off[1] + a[1])
                b.quad(ctrl, p)
                last_ctrl, last_kind = ctrl, "Q"
            elif up == "A":
                p = (x0 + a[5], y0 + a[6]) if rel else (a[5], a[6])
                b.arc(a[0], a[1], a[2], bool(a[3]), bool(a[4]), p)
                last_ctrl, last_kind = None, "A"
    except ValueError:
        pass  # keep everything parsed before the error
    return b.result()


def parse_path(d: str | None) -> Path:
    """Parse a path ``d`` string into a matplotlib :class:`~matplotlib.path.Path`.

    An empty or missing ``d`` yields an empty path (no vertices), so callers can
    treat every shape uniformly.
    """
    return parse_path_data(d).path


def polyline_path_data(points: np.ndarray, closed: bool) -> ParsedPath:
    """Build a :class:`ParsedPath` for a ``polyline``/``polygon``/``line``."""
    b = _Builder()
    if len(points) == 0:
        return b.result()
    b.move((float(points[0][0]), float(points[0][1])))
    for x, y in points[1:]:
        b.line((float(x), float(y)))
    if closed:
        b.close()
    return b.result()


def _bisect(d_in, d_out):
    if d_in is None:
        return d_out
    if d_out is None:
        return d_in
    bx, by = d_in[0] + d_out[0], d_in[1] + d_out[1]
    if math.hypot(bx, by) < 1e-12:  # a cusp: fall back to the incoming direction
        return d_in
    return (bx, by)


def _deg(d) -> float:
    return math.degrees(math.atan2(d[1], d[0])) if d is not None else 0.0


def marker_vertices(parsed: ParsedPath) -> list[tuple[float, float, float, str]]:
    """Return ``(x, y, auto_angle_deg, role)`` for each path vertex.

    ``role`` is ``"start"`` for the very first vertex, ``"end"`` for the last,
    ``"both"`` for a single-vertex path and ``"mid"`` otherwise, matching
    ``marker-start``/``marker-mid``/``marker-end``.
    The ``auto`` angle is the bisector of the incoming and outgoing directions.
    """
    out: list[tuple[float, float, float, str]] = []
    for sub in parsed.subpaths:
        segs = sub.segments
        if not segs:
            out.append((sub.start[0], sub.start[1], 0.0, "mid"))
            continue
        # Start vertex.
        first_out = segs[0][1]
        start_in = segs[-1][2] if sub.closed else None
        out.append(
            (sub.start[0], sub.start[1], _deg(_bisect(start_in, first_out)), "mid")
        )
        for k, (end, _d0, d1) in enumerate(segs):
            if k + 1 < len(segs):
                d_out = segs[k + 1][1]
            elif sub.closed:
                d_out = first_out
            else:
                d_out = None
            out.append((end[0], end[1], _deg(_bisect(d1, d_out)), "mid"))
    if out:
        x, y, a, _ = out[0]
        out[0] = (x, y, a, "start")
        x, y, a, _ = out[-1]
        out[-1] = (x, y, a, "end") if len(out) > 1 else (x, y, a, "both")
    return out


def start_direction(parsed: ParsedPath) -> float:
    """Outgoing direction (degrees) at the first vertex, for ``auto-start-reverse``."""
    for sub in parsed.subpaths:
        if sub.segments:
            return _deg(sub.segments[0][1])
    return 0.0
