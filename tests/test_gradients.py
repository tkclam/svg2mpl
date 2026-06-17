import matplotlib

matplotlib.use("Agg")
from xml.etree import ElementTree as ET  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import svg2mpl  # noqa: E402
from svg2mpl.gradients import parse_gradients  # noqa: E402

LINEAR = """<svg xmlns="http://www.w3.org/2000/svg"
                 xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 10 10">
  <defs>
    <linearGradient id="base">
      <stop offset="0" stop-color="black"/>
      <stop offset="1" stop-color="white"/>
    </linearGradient>
    <linearGradient id="ref" xlink:href="#base" x1="0" y1="0" x2="1" y2="1"/>
    <radialGradient id="rad">
      <stop offset="0" stop-color="red" stop-opacity="1"/>
      <stop offset="1" stop-color="red" stop-opacity="0"/>
    </radialGradient>
  </defs>
  <rect id="r" x="0" y="0" width="10" height="10" fill="url(#ref)"/>
</svg>"""


@pytest.fixture
def ax():
    fig, ax = plt.subplots()
    yield ax
    plt.close(fig)


def test_parse_collects_gradients():
    grads = parse_gradients(ET.fromstring(LINEAR))
    assert set(grads) == {"base", "ref", "rad"}


def test_href_inherits_stops():
    grads = parse_gradients(ET.fromstring(LINEAR))
    assert len(grads["ref"].stops) == 2  # inherited from #base
    assert grads["ref"].coords == {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0}


def test_average_color():
    grads = parse_gradients(ET.fromstring(LINEAR))
    avg = grads["base"].average_color()
    assert avg[:3] == pytest.approx((0.5, 0.5, 0.5))


def test_color_at_interpolates():
    grads = parse_gradients(ET.fromstring(LINEAR))
    mid = grads["base"].color_at(np.array(0.5))
    assert mid[:3] == pytest.approx((0.5, 0.5, 0.5))


def test_render_image_shape_and_endpoints():
    grads = parse_gradients(ET.fromstring(LINEAR))
    from matplotlib.transforms import Bbox

    img = grads["base"].render_image(Bbox([[0, 0], [1, 1]]), res=16)
    assert img.shape == (16, 16, 4)
    # default linear gradient runs left->right: left dark, right light
    assert img[0, 0, 0] < img[0, -1, 0]


def test_radial_opacity_fades_outward():
    grads = parse_gradients(ET.fromstring(LINEAR))
    from matplotlib.transforms import Bbox

    img = grads["rad"].render_image(Bbox([[0, 0], [1, 1]]), res=32)
    center = img[16, 16, 3]
    corner = img[0, 0, 3]
    assert center > corner  # opaque center, transparent edge


def test_add_svg_gradient_forces_patches_and_adds_image(ax):
    res = svg2mpl.add_svg(LINEAR, ax=ax, place=False)
    assert isinstance(res, dict)  # gradients can't be batched
    assert res["r"].get_facecolor()[3] == 0.0  # face blanked
    assert len(ax.images) == 1  # one clipped gradient image


def test_unresolved_gradient_falls_back_to_none(ax):
    svg = '<svg xmlns="http://www.w3.org/2000/svg"><rect width="2" height="2" fill="url(#missing)"/></svg>'
    res = svg2mpl.add_svg(svg, ax=ax, place=False, force_patches=True)
    assert res["__shape_0"].get_facecolor()[3] == 0.0
    assert len(ax.images) == 0
