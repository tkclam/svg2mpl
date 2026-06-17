from svg2mpl.style import (
    compute_style,
    declarations,
    is_hidden,
    parse_style_attr,
    style_to_kwargs,
)


def test_parse_style_attr():
    assert parse_style_attr("fill: red; stroke:blue") == {
        "fill": "red",
        "stroke": "blue",
    }
    assert parse_style_attr(None) == {}
    assert parse_style_attr("garbage") == {}


def test_declarations_priority_inline_over_attr():
    attrib = {"fill": "red", "style": "fill: green"}
    assert declarations(attrib)["fill"] == "green"


def test_declarations_css_over_attr_under_inline():
    attrib = {"fill": "red", "style": "fill: green"}
    css = {"fill": "blue"}
    # inline (green) wins over css (blue) wins over attr (red)
    assert declarations(attrib, css)["fill"] == "green"
    assert declarations({"fill": "red"}, css)["fill"] == "blue"


def test_compute_style_inherits_only_inheritable():
    parent = {"fill": "red", "opacity": "0.5"}
    child = compute_style({}, parent)
    assert child["fill"] == "red"  # inherited
    assert "opacity" not in child  # not inherited


def test_compute_style_child_overrides_parent():
    child = compute_style({"fill": "blue"}, {"fill": "red"})
    assert child["fill"] == "blue"


def test_default_fill_is_black():
    assert style_to_kwargs({})["facecolor"] == "black"


def test_default_stroke_is_none():
    assert style_to_kwargs({})["edgecolor"] == "none"


def test_simple_opacity_maps_to_alpha():
    kw = style_to_kwargs({"fill": "red", "opacity": "0.3"})
    assert kw["alpha"] == 0.3
    assert kw["facecolor"] == "red"


def test_fill_opacity_bakes_into_color_and_drops_alpha():
    kw = style_to_kwargs({"fill": "red", "fill-opacity": "0.5"})
    assert "alpha" not in kw
    assert kw["facecolor"][3] == 0.5  # rgba with baked alpha


def test_linecap_join_dash():
    kw = style_to_kwargs(
        {
            "stroke": "black",
            "stroke-linecap": "round",
            "stroke-linejoin": "bevel",
            "stroke-dasharray": "4 2",
        }
    )
    assert kw["capstyle"] == "round"
    assert kw["joinstyle"] == "bevel"
    assert kw["linestyle"] == (0.0, [4.0, 2.0])


def test_dasharray_odd_count_is_doubled():
    kw = style_to_kwargs({"stroke": "black", "stroke-dasharray": "3"})
    assert kw["linestyle"] == (0.0, [3.0, 3.0])


def test_is_hidden():
    assert is_hidden({"display": "none"})
    assert is_hidden({"visibility": "hidden"})
    assert not is_hidden({"fill": "red"})
