import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

import svg2mpl  # noqa: E402

TEXT = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
  <text id="t" x="10" y="20" font-size="12" fill="navy" text-anchor="middle">Hi</text>
</svg>"""


@pytest.fixture
def ax():
    fig, ax = plt.subplots()
    yield ax
    plt.close(fig)


def test_text_is_collected():
    scene = svg2mpl.load_string(TEXT)
    assert [t.content for t in scene.texts] == ["Hi"]
    assert (scene.texts[0].x, scene.texts[0].y) == (10.0, 20.0)


def test_text_rendered_with_style(ax):
    svg2mpl.add_svg(TEXT, ax=ax, place=False)
    assert len(ax.texts) == 1
    artist = ax.texts[0]
    assert artist.get_text() == "Hi"
    assert artist.get_horizontalalignment() == "center"  # text-anchor=middle
    assert artist.get_fontsize() == 12


def test_text_anchor_flipped(ax):
    # y=20 in a 100-tall doc flips to 80 with flip_y
    svg2mpl.add_svg(TEXT, ax=ax, place=False, flip_y=True)
    assert ax.texts[0].get_position()[1] == pytest.approx(80.0)


def test_empty_text_ignored():
    svg = '<svg xmlns="http://www.w3.org/2000/svg"><text x="0" y="0">   </text></svg>'
    assert svg2mpl.load_string(svg).texts == []
