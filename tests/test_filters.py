import numpy as np
import pytest

from svg2mpl._filter_ops import gaussian_blur
from svg2mpl.filters import Filter, apply_filter

REGION = (0.0, 0.0, 10.0, 10.0)


def run(prims, img, space="sRGB"):
    flt = Filter(primitives=prims, region=REGION, color_space=space)
    return apply_filter(flt, img, REGION, 2.0, 2.0, lambda *_: np.zeros(img.shape))


def square():
    img = np.zeros((20, 20, 4))
    img[5:15, 5:15] = (1.0, 0.0, 0.0, 1.0)
    return img


def test_blur_preserves_mass():
    img = np.zeros((60, 60, 4))
    img[25:35, 25:35] = (1.0, 0.0, 0.0, 1.0)
    out = gaussian_blur(img, 2.0, 2.0)
    assert out[..., 3].sum() == pytest.approx(img[..., 3].sum(), rel=1e-6)
    edge = out[30, 23:28, 3]
    assert np.all(np.diff(edge) > 0)
    assert 0 < edge[0] < edge[-1] < 1


def test_offset_moves_pixels():
    out = run([{"type": "feOffset", "offset": (1.0, 0.0)}], square())
    # 1 document unit = 2 pixels to the right
    assert out[10, 6, 3] == 0
    assert out[10, 16, 3] == 1


def test_saturate_zero_is_gray():
    out = run(
        [{"type": "feColorMatrix", "mtype": "saturate", "values": [0.0]}], square()
    )
    r, g, b, _alpha = out[10, 10]
    assert r == pytest.approx(g) == pytest.approx(b)


def test_flood_and_composite_in():
    prims = [
        {"type": "feFlood", "color": (0.0, 0.0, 1.0, 1.0), "result": "f"},
        {
            "type": "feComposite",
            "in": "f",
            "in2": "SourceAlpha",
            "operator": "in",
            "k": (0, 0, 0, 0),
        },
    ]
    out = run(prims, square())
    assert tuple(out[10, 10]) == pytest.approx((0, 0, 1, 1))
    assert out[0, 0, 3] == 0


def test_merge_and_named_results():
    prims = [
        {"type": "feFlood", "color": (0.0, 1.0, 0.0, 1.0), "result": "bg"},
        {"type": "feMerge", "nodes": ["bg", "SourceGraphic"]},
    ]
    out = run(prims, square())
    assert tuple(out[0, 0]) == pytest.approx((0, 1, 0, 1))
    assert tuple(out[10, 10]) == pytest.approx((1, 0, 0, 1))


def test_component_transfer_table_inverts():
    prims = [
        {
            "type": "feComponentTransfer",
            "funcs": {"R": {"type": "table", "table": [1, 0]}},
        }
    ]
    out = run(prims, square())
    assert out[10, 10, 0] == pytest.approx(0)


def test_linear_rgb_round_trip_is_identity():
    img = square() * 0.5
    img[..., 3] = 1.0
    out = run([{"type": "feOffset", "offset": (0.0, 0.0)}], img, space="linearRGB")
    assert out[10, 10, :3] == pytest.approx(img[10, 10, :3], abs=1e-6)
