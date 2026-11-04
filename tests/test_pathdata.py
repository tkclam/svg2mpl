import numpy as np
import pytest
from matplotlib.path import Path

from svg2mpl.geometry import path_extents
from svg2mpl.pathdata import marker_vertices, parse_path, parse_path_data


def ends(d):
    """Vertices at the end of each command (excluding curve controls)."""
    return [
        (round(x, 6), round(y, 6))
        for x, y, _a, _r in marker_vertices(parse_path_data(d))
    ]


def test_empty():
    assert len(parse_path("").vertices) == 0
    assert len(parse_path(None).vertices) == 0


def test_absolute_and_relative_lines():
    assert ends("M1 2 L3 4 l1 1 H10 h-2 V0 v5") == [
        (1, 2),
        (3, 4),
        (4, 5),
        (10, 5),
        (8, 5),
        (8, 0),
        (8, 5),
    ]


def test_implicit_lineto_after_moveto():
    assert ends("M0 0 10 0 10 10") == [(0, 0), (10, 0), (10, 10)]
    assert ends("m1 1 2 0 0 2") == [(1, 1), (3, 1), (3, 3)]


def test_compact_numbers():
    assert ends("M1e1-1e1L.5.5.6.6") == [(10, -10), (0.5, 0.5), (0.6, 0.6)]


def test_compact_arc_flags():
    # "00" are two flags, not the number 0.
    p = parse_path("M0 0a1 1 0 00 1 1")
    assert np.allclose(p.vertices[-1], (1, 1))


def test_arc_semicircle_extents():
    p = parse_path("M0 0 A 5 5 0 0 1 10 0")
    x0, y0, x1, y1 = path_extents(p)
    # sweep-flag 1 goes through negative y (upwards in SVG space)
    assert (x0, x1) == pytest.approx((0, 10))
    assert y0 == pytest.approx(-5, abs=1e-6)
    assert y1 == pytest.approx(0, abs=1e-6)


def test_arc_radius_scaled_up_when_too_small():
    p = parse_path("M0 0 A 1 1 0 0 1 10 0")
    assert np.allclose(p.vertices[-1], (10, 0))
    assert path_extents(p)[1] == pytest.approx(-5, abs=1e-6)


def test_zero_radius_arc_is_line():
    p = parse_path("M0 0 A 0 5 0 0 1 10 0")
    assert list(p.codes) == [Path.MOVETO, Path.LINETO]


def test_segment_after_close_starts_at_subpath_start():
    p = parse_path("M5 5 L10 5 Z L5 10")
    # After Z a new subpath starts (implicit moveto) at the start point.
    assert list(p.codes[-2:]) == [Path.MOVETO, Path.LINETO]
    assert np.allclose(p.vertices[-2], (5, 5))


def test_smooth_curves_reflect_control_points():
    p = parse_path("M0 0 C 0 10 10 10 10 0 S 20 -10 20 0")
    # first control of S is the reflection of (10, 10) about (10, 0)
    assert np.allclose(p.vertices[4], (10, -10))
    q = parse_path("M0 0 Q 5 5 10 0 T 20 0")
    assert np.allclose(q.vertices[3], (15, -5))


def test_error_keeps_parsed_prefix():
    p = parse_path("M0 0 L10 0 L10 10 # junk")
    assert len(p.vertices) == 3


def test_marker_vertices_roles_and_angles():
    v = marker_vertices(parse_path_data("M0 0 L10 0 L10 10"))
    assert [r for *_rest, r in v] == ["start", "mid", "end"]
    assert v[0][2] == pytest.approx(0)
    assert v[1][2] == pytest.approx(45)  # bisector of 0 and 90 degrees
    assert v[2][2] == pytest.approx(90)


def test_marker_vertices_skip_arc_joints():
    # An arc becomes several Beziers, but has a single end vertex.
    v = marker_vertices(parse_path_data("M0 0 A 5 5 0 1 1 0 0.01"))
    assert len(v) == 2
