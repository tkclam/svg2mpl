import numpy as np

from svg2mpl.parser import parse, parse_string

BASIC = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
  <g transform="translate(10, 10)">
    <rect id="r" x="0" y="0" width="40" height="20" fill="#3366cc"/>
  </g>
  <circle id="c" cx="50" cy="50" r="5" fill="red"/>
</svg>"""


def test_document_size_from_viewbox():
    scene = parse_string(BASIC)
    assert (scene.width, scene.height) == (100.0, 100.0)


def test_collects_shapes_with_ids():
    scene = parse_string(BASIC)
    assert [s.id for s in scene.shapes] == ["r", "c"]


def test_group_transform_is_baked():
    scene = parse_string(BASIC)
    rect = next(s for s in scene.shapes if s.id == "r")
    (x0, y0), (x1, y1) = rect.path.get_extents().get_points()
    # rect 0,0..40,20 shifted by the group's translate(10,10)
    assert np.allclose([x0, y0, x1, y1], [10, 10, 50, 30])


def test_hidden_elements_skipped():
    svg = """<svg xmlns="http://www.w3.org/2000/svg">
      <rect id="a" width="1" height="1"/>
      <rect id="b" width="1" height="1" display="none"/>
      <rect id="c" width="1" height="1" visibility="hidden"/>
    </svg>"""
    assert [s.id for s in parse_string(svg).shapes] == ["a"]


def test_defs_not_drawn_but_use_instantiates():
    svg = """<svg xmlns="http://www.w3.org/2000/svg"
                  xmlns:xlink="http://www.w3.org/1999/xlink">
      <defs><rect id="tile" width="4" height="4" fill="teal"/></defs>
      <use xlink:href="#tile" x="10" y="20"/>
    </svg>"""
    scene = parse_string(svg)
    # the <defs> rect is not drawn; only the <use> instance is
    assert len(scene.shapes) == 1
    (x0, y0), (x1, y1) = scene.shapes[0].path.get_extents().get_points()
    assert np.allclose([x0, y0, x1, y1], [10, 20, 14, 24])


def test_nested_inheritance():
    svg = """<svg xmlns="http://www.w3.org/2000/svg">
      <g fill="orange">
        <g stroke="black">
          <rect id="r" width="2" height="2"/>
        </g>
      </g>
    </svg>"""
    rect = parse_string(svg).shapes[0]
    assert rect.kwargs["facecolor"] == "orange"
    assert rect.kwargs["edgecolor"] == "black"


def test_parse_from_file(tmp_path):
    f = tmp_path / "x.svg"
    f.write_text(BASIC)
    assert len(parse(f).shapes) == 2
