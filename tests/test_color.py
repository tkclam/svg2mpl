import pytest

from svg2mpl.color import NONE, parse_paint, parse_url_ref, to_grayscale


def test_none_and_default():
    assert parse_paint("none") == NONE
    assert parse_paint(None, default="black") == "black"
    assert parse_paint("  ", default=NONE) == NONE


def test_named_and_hex_passthrough():
    assert parse_paint("red") == "red"
    assert parse_paint("#abcdef") == "#abcdef"


def test_current_color():
    assert parse_paint("currentColor", current_color="teal") == "teal"


def test_rgb_int():
    assert parse_paint("rgb(255, 0, 128)") == pytest.approx((1.0, 0.0, 128 / 255))


def test_rgb_percent():
    assert parse_paint("rgb(100%, 0%, 50%)") == pytest.approx((1.0, 0.0, 0.5))


def test_rgba():
    r, g, b, a = parse_paint("rgba(0, 0, 0, 0.5)")
    assert (r, g, b, a) == pytest.approx((0.0, 0.0, 0.0, 0.5))


def test_hsl():
    r, g, b = parse_paint("hsl(0, 100%, 50%)")
    assert (r, g, b) == pytest.approx((1.0, 0.0, 0.0))


def test_url_passthrough_and_ref():
    assert parse_paint("url(#grad)") == "url(#grad)"
    assert parse_url_ref("url(#grad)") == "grad"
    assert parse_url_ref("red") is None


def test_to_grayscale_is_gray():
    r, g, b = to_grayscale("red")
    assert r == g == b
