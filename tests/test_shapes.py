import numpy as np
import pytest

from svg2mpl.shapes import shape_to_path, strip_ns


def test_strip_ns():
    assert strip_ns("{http://www.w3.org/2000/svg}rect") == "rect"
    assert strip_ns("rect") == "rect"


def test_unknown_tag_returns_none():
    assert shape_to_path("nope", {}) is None


def test_rect_extents():
    p = shape_to_path("rect", {"x": "1", "y": "2", "width": "4", "height": "6"})
    (x0, y0), (x1, y1) = p.get_extents().get_points()
    assert (x0, y0, x1, y1) == (1.0, 2.0, 5.0, 8.0)


def test_rect_zero_size_is_empty():
    p = shape_to_path("rect", {"width": "0", "height": "10"})
    assert len(p.vertices) == 0


def test_rounded_rect_stays_within_box():
    p = shape_to_path("rect", {"width": "10", "height": "10", "rx": "2", "ry": "2"})
    (x0, y0), (x1, y1) = p.get_extents().get_points()
    assert (round(x0), round(y0), round(x1), round(y1)) == (0, 0, 10, 10)
    # corners are rounded => the box corner (0,0) is not a vertex
    assert not np.any(np.all(np.isclose(p.vertices, [0, 0]), axis=1))


def test_circle_extents():
    p = shape_to_path("circle", {"cx": "5", "cy": "5", "r": "3"})
    (x0, y0), (x1, y1) = p.get_extents().get_points()
    assert np.allclose([x0, y0, x1, y1], [2, 2, 8, 8])


def test_ellipse_extents():
    p = shape_to_path("ellipse", {"cx": "0", "cy": "0", "rx": "4", "ry": "2"})
    (x0, y0), (x1, y1) = p.get_extents().get_points()
    assert np.allclose([x0, y0, x1, y1], [-4, -2, 4, 2])


def test_line():
    p = shape_to_path("line", {"x1": "0", "y1": "0", "x2": "3", "y2": "4"})
    assert np.allclose(p.vertices, [[0, 0], [3, 4]])


def test_polyline_open():
    p = shape_to_path("polyline", {"points": "0,0 1,1 2,0"})
    assert len(p.vertices) == 3
    assert p.codes[-1] != p.CLOSEPOLY


def test_polygon_closed():
    p = shape_to_path("polygon", {"points": "0,0 1,1 2,0"})
    assert p.codes[-1] == p.CLOSEPOLY


def test_length_ignores_units():
    p = shape_to_path("rect", {"width": "4px", "height": "6px"})
    (x0, y0), (x1, y1) = p.get_extents().get_points()
    assert (x1 - x0, y1 - y0) == (4.0, 6.0)


@pytest.mark.parametrize("tag", ["path"])
def test_empty_path(tag):
    assert len(shape_to_path(tag, {"d": ""}).vertices) == 0


def test_path_with_namespace_tag():
    p = shape_to_path("{http://www.w3.org/2000/svg}path", {"d": "M0 0 L10 0"})
    assert np.allclose(p.vertices[:2], [[0, 0], [10, 0]])
