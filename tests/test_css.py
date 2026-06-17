import pytest

from svg2mpl.css import css_available
from svg2mpl.parser import parse_string

pytestmark = pytest.mark.skipif(
    not css_available(), reason="optional CSS deps (tinycss2/cssselect2) not installed"
)

STYLED = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10">
  <style>
    .box { fill: rebeccapurple; }
    #special { fill: gold; }
    rect { stroke: black; stroke-width: 0.5; }
  </style>
  <rect id="a" class="box" x="0" y="0" width="4" height="4"/>
  <rect id="special" class="box" x="5" y="5" width="4" height="4"/>
</svg>"""


def test_class_selector_applies():
    a = next(s for s in parse_string(STYLED).shapes if s.id == "a")
    assert a.style["fill"] == "rebeccapurple"


def test_id_selector_beats_class():
    special = next(s for s in parse_string(STYLED).shapes if s.id == "special")
    assert special.style["fill"] == "gold"


def test_type_selector_applies_to_all():
    for s in parse_string(STYLED).shapes:
        assert s.style["stroke"] == "black"
        assert s.kwargs["linewidth"] == 0.5


def test_inline_style_beats_css():
    svg = """<svg xmlns="http://www.w3.org/2000/svg">
      <style>rect { fill: red; }</style>
      <rect id="r" width="1" height="1" style="fill: lime"/>
    </svg>"""
    assert parse_string(svg).shapes[0].style["fill"] == "lime"
