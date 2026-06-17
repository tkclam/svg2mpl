import numpy as np
import pytest

from svg2mpl.transform import (
    flip_y_transform,
    parse_transform,
    parse_viewbox,
    viewbox_transform,
)


def _apply(affine, point):
    return tuple(np.round(affine.transform(point), 6))


def test_identity_for_empty():
    assert _apply(parse_transform(None), (3, 4)) == (3.0, 4.0)
    assert _apply(parse_transform(""), (3, 4)) == (3.0, 4.0)


def test_translate():
    t = parse_transform("translate(10, 5)")
    assert _apply(t, (1, 2)) == (11.0, 7.0)


def test_translate_single_arg():
    t = parse_transform("translate(10)")
    assert _apply(t, (1, 2)) == (11.0, 2.0)


def test_scale_uniform_and_xy():
    assert _apply(parse_transform("scale(2)"), (3, 4)) == (6.0, 8.0)
    assert _apply(parse_transform("scale(2, 3)"), (3, 4)) == (6.0, 12.0)


def test_rotate_about_origin():
    t = parse_transform("rotate(90)")
    assert _apply(t, (1, 0)) == (0.0, 1.0)


def test_rotate_about_center():
    t = parse_transform("rotate(180, 5, 5)")
    assert _apply(t, (5, 6)) == (5.0, 4.0)


def test_matrix():
    t = parse_transform("matrix(1 0 0 1 7 8)")
    assert _apply(t, (0, 0)) == (7.0, 8.0)


def test_skewx():
    t = parse_transform("skewX(45)")
    assert _apply(t, (0, 2)) == (2.0, 2.0)


def test_composition_order_left_is_outermost():
    # SVG: translate first conceptually outermost; a point is T(R(p)).
    t = parse_transform("translate(10, 0) scale(2)")
    assert _apply(t, (1, 1)) == (12.0, 2.0)


def test_parse_viewbox():
    assert parse_viewbox("0 0 100 50") == (0.0, 0.0, 100.0, 50.0)
    assert parse_viewbox("bad") is None
    assert parse_viewbox(None) is None


def test_viewbox_transform_offset_only_without_viewport():
    t = viewbox_transform((10, 20, 100, 50), None, None)
    assert _apply(t, (10, 20)) == (0.0, 0.0)


def test_viewbox_transform_scales_to_viewport():
    t = viewbox_transform((0, 0, 100, 100), 50, 50)
    assert _apply(t, (100, 100)) == (50.0, 50.0)


@pytest.mark.parametrize("y,expect", [(0, 100), (100, 0), (25, 75)])
def test_flip_y(y, expect):
    t = flip_y_transform(100)
    assert _apply(t, (3, y)) == (3.0, float(expect))
