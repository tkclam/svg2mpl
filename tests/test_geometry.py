import numpy as np
import pytest
from matplotlib.path import Path

from svg2mpl.geometry import (
    evenodd_to_nonzero,
    flatten,
    intersect_clip,
    is_convex,
    path_extents,
    rect_path,
)
from svg2mpl.pathdata import parse_path
from svg2mpl.units import LengthContext, parse_length


def test_units():
    assert parse_length("1in") == pytest.approx(96)
    assert parse_length("25.4mm") == pytest.approx(96)
    assert parse_length("72pt") == pytest.approx(96)
    assert parse_length("1pc") == pytest.approx(16)
    assert parse_length("3") == 3
    assert parse_length("junk", 7.0) == 7.0


def test_relative_units():
    ctx = LengthContext(viewport_width=200, viewport_height=100, font_size=10)
    assert parse_length("50%", axis="x", ctx=ctx) == pytest.approx(100)
    assert parse_length("50%", axis="y", ctx=ctx) == pytest.approx(50)
    diag = np.hypot(200, 100) / np.sqrt(2)
    assert parse_length("100%", ctx=ctx) == pytest.approx(diag)
    assert parse_length("2em", ctx=ctx) == pytest.approx(20)


def winding(path, pt):
    """Nonzero winding number of ``pt`` (what matplotlib fills with)."""
    total = 0
    for ring in flatten(path):
        x, y = ring[:, 0] - pt[0], ring[:, 1] - pt[1]
        x2, y2 = np.roll(x, -1), np.roll(y, -1)
        up = (y <= 0) & (y2 > 0)
        down = (y > 0) & (y2 <= 0)
        cross = x * y2 - x2 * y
        total += int(np.sum(up & (cross > 0)) - np.sum(down & (cross < 0)))
    return total


def test_evenodd_makes_holes():
    # Two squares with the same winding: nonzero fills the inner one.
    p = parse_path("M0 0 H10 V10 H0 Z M2 2 H8 V8 H2 Z")
    assert winding(p, (5, 5)) != 0
    q = evenodd_to_nonzero(p)
    assert winding(q, (5, 5)) == 0
    assert winding(q, (1, 5)) != 0


def test_evenodd_three_levels():
    p = parse_path("M0 0 H30 V30 H0 Z M5 5 H25 V25 H5 Z M10 10 H20 V20 H10 Z")
    q = evenodd_to_nonzero(p)
    assert winding(q, (2, 15)) != 0
    assert winding(q, (7, 15)) == 0
    assert winding(q, (15, 15)) != 0


def test_is_convex():
    assert is_convex(rect_path(0, 0, 1, 1))
    assert is_convex(Path.unit_circle())
    assert not is_convex(parse_path("M0 0 L10 0 L5 2 L10 10 L0 10 Z"))


def test_intersect_with_convex():
    a = rect_path(0, 0, 10, 10)
    b = rect_path(5, 5, 15, 15)
    c = intersect_clip(a, b)
    assert path_extents(c) == pytest.approx((5, 5, 10, 10))
    assert intersect_clip(None, b) is b
    assert len(intersect_clip(a, rect_path(20, 20, 30, 30)).vertices) == 0


def test_intersect_two_nonconvex_raises():
    star = parse_path("M0 0 L10 0 L5 2 L10 10 L0 10 Z")
    with pytest.raises(ValueError, match="non-convex"):
        intersect_clip(star, star)


def test_fast_extents_match_matplotlib():
    for d in [
        "M0 0 C 10 20 20 -20 30 0",
        "M0 0 Q 5 10 10 0 L 20 5",
        "M10 10 A5 5 0 1 1 20 10Z",
    ]:
        p = parse_path(d)
        bb = p.get_extents()
        assert path_extents(p) == pytest.approx((bb.x0, bb.y0, bb.x1, bb.y1))
