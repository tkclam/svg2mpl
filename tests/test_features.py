"""Scene-level tests for the SVG features handled by the parser."""

import base64
import io

import numpy as np
import pytest
from matplotlib.colors import to_rgba
from PIL import Image as PILImage

import svg2mpl
import svg2mpl.css as css
from svg2mpl.geometry import flatten, path_extents
from svg2mpl.pathdata import parse_path
from svg2mpl.scene import Group, Image
from svg2mpl.text import _PathMeasure

NS = 'xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"'


def scene(body, attrs='width="100" height="100"'):
    return svg2mpl.load_string(f"<svg {NS} {attrs}>{body}</svg>")


def shape(body, **kw):
    return scene(body, **kw).shapes[0]


def ext(item):
    return path_extents(item.path)


def winding(path, pt):
    """Nonzero winding number of ``pt`` (what matplotlib fills with)."""
    total = 0
    for ring in flatten(path):
        x, y = ring[:, 0] - pt[0], ring[:, 1] - pt[1]
        x2, y2 = np.roll(x, -1), np.roll(y, -1)
        up = (y <= 0) & (y2 > 0)
        down = (y > 0) & (y2 <= 0)
        cross = x * y2 - x2 * y
        total += int(np.sum(up & (cross > 0)) - np.sum(down & (cross < 0)))
    return total


# --- structure ---------------------------------------------------------- #
def test_nested_svg_viewbox_and_clip():
    s = shape(
        '<svg x="10" y="10" width="20" height="20" viewBox="0 0 2 2">'
        '<rect width="4" height="4"/></svg>'
    )
    assert ext(s) == pytest.approx((10, 10, 50, 50))
    assert path_extents(s.clip) == pytest.approx((10, 10, 30, 30))


def test_preserve_aspect_ratio_alignment():
    s = shape(
        '<svg width="100" height="50" viewBox="0 0 10 10" preserveAspectRatio="xMaxYMin meet">'
        '<rect width="10" height="10"/></svg>'
    )
    assert ext(s) == pytest.approx((50, 0, 100, 50))


def test_symbol_viewport_from_use():
    s = shape(
        '<defs><symbol id="s" viewBox="0 0 1 1"><rect width="1" height="1"/></symbol></defs>'
        '<use href="#s" x="5" y="5" width="10" height="10"/>'
    )
    assert ext(s) == pytest.approx((5, 5, 15, 15))


def test_use_inherits_style():
    s = shape('<defs><circle id="c" r="2"/></defs><use href="#c" fill="red"/>')
    assert s.kwargs["facecolor"] == "red"


def test_use_cycle_is_safe():
    sc = scene('<g id="a"><use href="#a"/><rect width="1" height="1"/></g>')
    assert len(sc.shapes) >= 1


def test_switch_picks_first_match():
    sc = scene(
        '<switch><rect id="fr" systemLanguage="fr" width="1" height="1"/>'
        '<rect id="ext" requiredExtensions="http://x" width="1" height="1"/>'
        '<rect id="ok" width="1" height="1"/><rect id="later" width="1" height="1"/></switch>'
    )
    assert [s.id for s in sc.shapes] == ["ok"]


def test_visibility_can_be_overridden_by_child():
    sc = scene(
        '<g visibility="hidden"><rect id="a" width="1" height="1"/>'
        '<rect id="b" width="1" height="1" visibility="visible"/></g>'
    )
    assert [s.id for s in sc.shapes] == ["b"]


def test_root_units_and_viewbox():
    sc = scene(
        '<rect width="10" height="10"/>',
        attrs='width="1in" height="1in" viewBox="0 0 10 10"',
    )
    assert (sc.width, sc.height) == pytest.approx((96, 96))
    assert ext(sc.shapes[0]) == pytest.approx((0, 0, 96, 96))


# --- style -------------------------------------------------------------- #
def test_inherit_transparent_and_fallback():
    sc = scene(
        '<g fill="red"><rect id="a" width="1" height="1" fill="inherit"/></g>'
        '<rect id="b" width="1" height="1" fill="transparent"/>'
        '<rect id="c" width="1" height="1" fill="url(#nope) blue"/>'
    )
    a, b, c = sc.shapes
    assert a.kwargs["facecolor"] == "red"
    assert to_rgba(b.kwargs["facecolor"])[3] == 0
    assert to_rgba(c.kwargs["facecolor"])[:3] == to_rgba("blue")[:3]


def test_rgba_alpha_is_not_overwritten():
    s = shape('<rect width="1" height="1" fill="rgba(255,0,0,0.25)"/>')
    assert "alpha" not in s.kwargs
    assert to_rgba(s.kwargs["facecolor"])[3] == pytest.approx(0.25)


def test_group_opacity_without_overlap_is_multiplied():
    sc = scene(
        '<g opacity="0.5"><rect width="10" height="10"/>'
        '<rect x="20" width="10" height="10" opacity="0.5"/></g>'
    )
    assert [s.kwargs["alpha"] for s in sc.items] == [0.5, 0.25]


def test_group_opacity_with_overlap_composites():
    sc = scene(
        '<g opacity="0.5"><rect width="10" height="10"/><rect x="5" width="10" height="10"/></g>'
    )
    assert len(sc.items) == 1
    assert isinstance(sc.items[0], Group)
    assert sc.items[0].opacity == 0.5
    assert all(s.kwargs["alpha"] == 1.0 for s in sc.items[0].items)


def test_stroke_width_scales_with_transform():
    s = shape(
        '<g transform="scale(3)"><rect width="1" height="1" stroke="red" stroke-width="2"/></g>'
    )
    assert s.kwargs["linewidth"] == pytest.approx(6)
    n = shape(
        '<g transform="scale(3)"><rect width="1" height="1" stroke="red" stroke-width="2"'
        ' vector-effect="non-scaling-stroke"/></g>'
    )
    assert n.kwargs["linewidth"] == pytest.approx(2)
    assert not n.scale_stroke


def test_units_and_percentages_in_geometry():
    s = shape('<rect x="10%" width="50%" height="1cm"/>')
    assert ext(s) == pytest.approx((10, 0, 60, 96 / 2.54))


def test_paint_order_stroke_first():
    sc = scene(
        '<rect width="5" height="5" fill="red" stroke="blue" paint-order="stroke"/>'
    )
    first, second = sc.shapes
    assert first.kwargs["facecolor"] == "none"
    assert first.kwargs["edgecolor"] == "blue"
    assert second.kwargs["edgecolor"] == "none"
    assert second.kwargs["facecolor"] == "red"


def test_evenodd_fill_rule():
    s = shape('<path fill-rule="evenodd" d="M0 0H10V10H0Z M2 2H8V8H2Z"/>')
    assert winding(s.path, (5, 5)) == 0
    assert winding(s.path, (1, 5)) != 0


# --- clipping / masking / filters -------------------------------------- #
def test_clip_path_user_space():
    s = shape(
        '<clipPath id="c"><rect width="5" height="5"/></clipPath>'
        '<rect width="10" height="10" clip-path="url(#c)"/>'
    )
    assert path_extents(s.clip) == pytest.approx((0, 0, 5, 5))


def test_clip_path_object_bounding_box_and_nesting():
    s = shape(
        '<clipPath id="half" clipPathUnits="objectBoundingBox"><rect width="0.5" height="1"/></clipPath>'
        '<clipPath id="top"><rect width="100" height="5"/></clipPath>'
        '<g clip-path="url(#top)"><rect x="10" width="20" height="10" clip-path="url(#half)"/></g>'
    )
    assert path_extents(s.clip) == pytest.approx((10, 0, 20, 5))


def test_empty_clip_hides_element():
    sc = scene('<clipPath id="e"/><rect width="1" height="1" clip-path="url(#e)"/>')
    assert sc.shapes == []


def test_non_convex_nested_clips_use_offscreen_group():
    star = "M50 0 L61 35 L98 35 L68 57 L79 91 L50 70 L21 91 L32 57 L2 35 L39 35 Z"
    sc = scene(
        f'<clipPath id="a"><path d="{star}"/></clipPath>'
        f'<clipPath id="b"><path d="{star}" transform="translate(10 0)"/></clipPath>'
        '<g clip-path="url(#a)"><rect width="100" height="100" clip-path="url(#b)"/></g>'
    )
    assert isinstance(sc.items[0], Group)
    assert sc.items[0].clip is not None
    assert sc.items[0].items[0].clip is not None


def test_mask_and_filter_create_groups():
    sc = scene(
        '<mask id="m"><rect width="50" height="100" fill="white"/></mask>'
        '<filter id="f"><feGaussianBlur stdDeviation="2"/></filter>'
        '<rect width="100" height="100" mask="url(#m)"/>'
        '<circle cx="50" cy="50" r="10" filter="url(#f)"/>'
    )
    m, f = sc.items
    assert isinstance(m, Group)
    assert m.mask is not None
    assert m.region == pytest.approx((-10, -10, 110, 110))
    assert isinstance(f, Group)
    assert f.filter is not None
    assert f.filter.primitives[0]["std"] == pytest.approx((2, 2))


def test_css_filter_functions():
    sc = scene('<rect width="10" height="10" style="filter: blur(1px) grayscale(1)"/>')
    types = [p["type"] for p in sc.items[0].filter.primitives]
    assert types == ["feGaussianBlur", "feColorMatrix"]


# --- markers ------------------------------------------------------------ #
def test_markers_placed_and_oriented():
    sc = scene(
        '<marker id="m" markerWidth="2" markerHeight="2" orient="auto" markerUnits="userSpaceOnUse">'
        '<rect width="2" height="1"/></marker>'
        '<polyline points="0,0 10,0 10,10" fill="none" stroke="black"'
        ' marker-start="url(#m)" marker-mid="url(#m)" marker-end="url(#m)"/>'
    )
    _line, *marks = sc.shapes
    assert len(marks) == 3
    # The end marker is rotated 90 degrees at (10, 10).
    assert ext(marks[2]) == pytest.approx((9, 10, 10, 12))


# --- paint servers ------------------------------------------------------ #
def test_gradient_bound_to_shape():
    s = shape(
        '<linearGradient id="g"><stop offset="0" stop-color="red"/><stop offset="1" stop-color="blue"/>'
        '</linearGradient><rect x="10" width="20" height="10" fill="url(#g)"/>'
    )
    assert s.kwargs["facecolor"] == "none"
    rgba = s.fill_paint.server.evaluate(
        np.array([[10.0, 5.0], [30.0, 5.0]]), s.fill_paint.bbox
    )
    assert rgba[0][:3] == pytest.approx((1, 0, 0))
    assert rgba[1][:3] == pytest.approx((0, 0, 1))


def test_gradient_spread_and_focal():
    sc = scene(
        '<linearGradient id="r" spreadMethod="reflect" x2="0.5"><stop offset="0" stop-color="black"/>'
        '<stop offset="1" stop-color="white"/></linearGradient>'
        '<radialGradient id="f" fx="0.2"><stop offset="0" stop-color="black"/>'
        '<stop offset="1" stop-color="white"/></radialGradient>'
        '<rect width="10" height="10" fill="url(#r)"/><rect width="10" height="10" fill="url(#f)"/>'
    )
    r, f = sc.gradients["r"], sc.gradients["f"]
    t = r.evaluate(np.array([[0.75, 0.5]]), (0, 0, 1, 1))[0]
    assert t[0] == pytest.approx(0.5)  # reflected back halfway
    # The focal point is darkest.
    near = f.evaluate(np.array([[0.2, 0.5]]), (0, 0, 1, 1))[0][0]
    center = f.evaluate(np.array([[0.5, 0.5]]), (0, 0, 1, 1))[0][0]
    assert near < center


def test_single_stop_gradient_is_solid():
    s = shape(
        '<linearGradient id="g"><stop stop-color="lime"/></linearGradient><rect width="1" height="1" fill="url(#g)"/>'
    )
    assert s.fill_paint is None
    assert to_rgba(s.kwargs["facecolor"])[:3] == (0, 1, 0)


def test_pattern_tile_resolved():
    s = shape(
        '<pattern id="p" width="10" height="10" patternUnits="userSpaceOnUse">'
        '<circle cx="5" cy="5" r="3" fill="red"/></pattern>'
        '<rect width="50" height="50" fill="url(#p)"/>'
    )
    tile = s.fill_paint.tile
    assert (tile.width, tile.height) == (10, 10)
    assert len(tile.items) == 1


# --- text ---------------------------------------------------------------- #
def test_text_becomes_glyph_shapes():
    sc = scene(
        '<text x="10" y="50" font-size="20" fill="navy">Hi <tspan fill="red">you</tspan></text>'
    )
    assert [t.content for t in sc.texts] == ["Hi you"]
    navy, red = sc.shapes
    assert navy.kwargs["facecolor"] == "navy"
    assert red.kwargs["facecolor"] == "red"
    x0, _y0, x1, y1 = ext(navy)
    assert x0 >= 10
    assert y1 <= 51
    assert ext(red)[0] > x1  # the tspan continues after "Hi "


def test_text_anchor_and_positions():
    start = ext(shape('<text x="50" y="50" font-size="10">abc</text>'))
    middle = ext(
        shape('<text x="50" y="50" font-size="10" text-anchor="middle">abc</text>')
    )
    end = ext(shape('<text x="50" y="50" font-size="10" text-anchor="end">abc</text>'))
    assert start[0] >= 50
    assert end[2] <= 50.5
    assert (middle[0] + middle[2]) / 2 == pytest.approx(50, abs=1)


def test_text_whitespace_collapsed():
    sc = scene("<text>  a \n   b  </text>")
    assert sc.texts[0].content == "a b"


def test_text_path():
    sc = scene(
        '<path id="p" d="M0 50 L100 50"/>'
        '<text font-size="10"><textPath href="#p" startOffset="50">xx</textPath></text>'
    )
    glyphs = sc.shapes[-1]
    assert ext(glyphs)[0] >= 49


# --- images --------------------------------------------------------------- #
def _png_uri():
    buf = io.BytesIO()
    PILImage.fromarray(np.full((2, 4, 4), 255, np.uint8)).save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def test_raster_image_placed_with_aspect_ratio():
    sc = scene(f'<image x="0" y="0" width="40" height="40" href="{_png_uri()}"/>')
    img = sc.items[0]
    assert isinstance(img, Image)
    assert img.data.shape == (2, 4, 4)
    assert ext(img) == pytest.approx((0, 10, 40, 30))  # 2:1 image, meet


def test_svg_image_is_vector():
    inner = base64.b64encode(
        b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><rect width="10" height="10"/></svg>'
    ).decode()
    sc = scene(
        f'<image width="50" height="50" href="data:image/svg+xml;base64,{inner}"/>'
    )
    assert ext(sc.shapes[0]) == pytest.approx((0, 0, 50, 50))


# --- CSS ---------------------------------------------------------------- #
def test_css_important_beats_inline():
    s = shape(
        '<style>rect{fill:red !important}</style><rect width="1" height="1" style="fill:blue"/>'
    )
    assert s.kwargs["facecolor"] == "red"


def test_builtin_css_fallback(monkeypatch):
    monkeypatch.setattr(css, "css_available", lambda: False)
    sc = scene(
        "<style>.a{fill:red} g > .b{fill:green} #c{fill:blue} rect[data-x]{stroke:black}</style>"
        '<rect class="a" width="1" height="1"/><g><rect class="b" width="1" height="1"/></g>'
        '<rect id="c" class="a" width="1" height="1" data-x="1"/>'
    )
    a, b, c = sc.shapes
    assert a.kwargs["facecolor"] == "red"
    assert b.kwargs["facecolor"] == "green"
    assert c.kwargs["facecolor"] == "blue"
    assert c.kwargs["edgecolor"] == "black"


# --- rendering ------------------------------------------------------------ #
def test_strokes_scale_with_placement(ax):
    svg = f'<svg {NS} viewBox="0 0 10 10"><line x2="10" y2="10" stroke="black"/></svg>'
    small = svg2mpl.add_svg(svg, ax=ax, width=1, force_patches=True)
    big = svg2mpl.add_svg(svg, ax=ax, width=10, force_patches=True)
    ax.figure.canvas.draw()
    (s,), (b,) = small.values(), big.values()
    assert b.get_linewidth() == pytest.approx(10 * s.get_linewidth(), rel=1e-3)


def test_user_linewidth_is_in_points(ax):
    svg = f'<svg {NS} viewBox="0 0 10 10"><line x2="10" y2="10" stroke="black"/></svg>'
    res = svg2mpl.add_svg(svg, ax=ax, width=100, linewidth=2, force_patches=True)
    ax.figure.canvas.draw()
    assert next(iter(res.values())).get_linewidth() == 2


def test_images_do_not_reset_axes(ax):
    ax.set_aspect("auto")
    svg2mpl.add_svg(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10">'
        '<rect width="10" height="10" filter="blur(1px)" style="filter: blur(1px)"/></svg>',
        ax=ax,
        xy=(100, 100),
        width=5,
    )
    assert ax.get_aspect() == "auto"
    assert len(ax.images) == 1
    x0, x1 = ax.get_xlim()
    assert x0 < 100 < x1


def test_mask_renders_image(ax):
    svg2mpl.add_svg(
        f'<svg {NS} viewBox="0 0 10 10"><mask id="m"><rect width="5" height="10" fill="white"/></mask>'
        '<rect width="10" height="10" fill="red" mask="url(#m)"/></svg>',
        ax=ax,
        place=False,
    )
    ax.figure.canvas.draw()
    img = ax.images[0].get_array()
    # The group region spans -1..11; the left half is opaque, the right not.
    h, w = img.shape[:2]
    assert img[h // 2, int(w * 3 / 12), 3] > 0.9
    assert img[h // 2, int(w * 9 / 12), 3] < 0.1


def test_text_as_artists(ax):
    svg2mpl.add_svg(
        f'<svg {NS} viewBox="0 0 100 100"><text x="10" y="20" font-size="12">Hi</text></svg>',
        ax=ax,
        text_as_paths=False,
    )
    assert [t.get_text() for t in ax.texts] == ["Hi"]


# --- regressions --------------------------------------------------------- #
def test_text_path_ignores_gap_between_subpaths():

    m = _PathMeasure(parse_path("M0 0 L50 0 M0 50 L50 50"))
    assert m.length == pytest.approx(100)
    (x, y), _angle = m.at(75)
    assert (x, y) == pytest.approx((25, 50))


def test_deeply_nested_styles_do_not_recurse():
    depth = 3000
    body = (
        '<g fill="red">' * depth
        + '<linearGradient id="g"><stop stop-color="currentColor"/></linearGradient>'
        + "</g>" * depth
    )
    sc = scene(body + '<rect width="1" height="1" fill="url(#g)"/>')
    assert sc.shapes


def test_pattern_image_is_bounded(ax):
    svg = (
        f'<svg {NS} viewBox="0 0 100 100"><pattern id="p" width="0.5" height="0.5" '
        'patternUnits="userSpaceOnUse"><rect width="0.25" height="0.5"/></pattern>'
        '<rect width="100" height="100" fill="url(#p)"/></svg>'
    )
    svg2mpl.add_svg(svg, ax=ax, place=False, raster_resolution=8192)
    h, w = ax.images[0].get_array().shape[:2]
    assert h * w <= 4096**2
