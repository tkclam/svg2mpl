from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.collections import PathCollection  # noqa: E402
from matplotlib.patches import PathPatch  # noqa: E402

import svg2mpl  # noqa: E402
from svg2mpl.render import get_affine_matrix, get_bbox, get_is_included  # noqa: E402

ASSET = Path(__file__).parent / "assets" / "basic.svg"


@pytest.fixture
def ax():
    fig, ax = plt.subplots()
    yield ax
    plt.close(fig)


def test_load_is_cached():
    a = svg2mpl.load(ASSET)
    b = svg2mpl.load(ASSET)
    assert a is b  # same cached object


def test_load_returns_scene():
    scene = svg2mpl.load(ASSET)
    assert [s.id for s in scene.shapes] == ["r", "c", "p", "l"]


def test_add_svg_returns_collection_for_uniform(ax):
    res = svg2mpl.add_svg(ASSET, ax=ax)
    assert isinstance(res, PathCollection)


def test_force_patches_returns_dict(ax):
    res = svg2mpl.add_svg(ASSET, ax=ax, force_patches=True)
    assert isinstance(res, dict)
    assert all(isinstance(p, PathPatch) for p in res.values())
    assert set(res) == {"r", "c", "p", "l"}


def test_exclude_drops_paths(ax):
    res = svg2mpl.add_svg(ASSET, ax=ax, force_patches=True, exclude=r"[cp]")
    assert set(res) == {"r", "l"}


def test_exclude_everything_raises(ax):
    with pytest.raises(ValueError):
        svg2mpl.add_svg(ASSET, ax=ax, exclude=r".*")


def test_grayscale(ax):
    res = svg2mpl.add_svg(ASSET, ax=ax, force_patches=True, grayscale=True)
    fc = res["r"].get_facecolor()
    assert fc[0] == fc[1] == fc[2]


def test_per_path_kwargs(ax):
    res = svg2mpl.add_svg(
        ASSET, ax=ax, force_patches=True, per_path_kwargs={"facecolor": {"r": "black"}}
    )
    assert tuple(res["r"].get_facecolor()) == (0.0, 0.0, 0.0, 1.0)


def test_global_kwargs_applied(ax):
    res = svg2mpl.add_svg(ASSET, ax=ax, force_patches=True, edgecolor="magenta")
    assert tuple(res["c"].get_edgecolor())[:3] == pytest.approx((1.0, 0.0, 1.0))


def test_inline_markup_source(ax):
    svg = '<svg xmlns="http://www.w3.org/2000/svg"><rect width="2" height="2"/></svg>'
    res = svg2mpl.add_svg(svg, ax=ax)
    assert isinstance(res, PathCollection)


def test_missing_file_raises(ax):
    with pytest.raises(FileNotFoundError):
        svg2mpl.add_svg("does-not-exist.svg", ax=ax)


def test_flip_y_inverts_geometry(ax):
    no_flip = svg2mpl.add_svg(
        ASSET, ax=ax, place=False, flip_y=False, force_patches=True
    )
    flipped = svg2mpl.add_svg(
        ASSET, ax=ax, place=False, flip_y=True, force_patches=True
    )
    y_raw = no_flip["r"].get_path().get_extents().get_points()[0][1]
    y_flip = flipped["r"].get_path().get_extents().get_points()[0][1]
    assert y_raw != pytest.approx(y_flip)


def test_placement_positions_anchor(ax):
    # place the center of the drawing at (5, 5) with width 1
    svg2mpl.add_svg(ASSET, ax=ax, xy=(5, 5), width=1, origin=(0.5, 0.5))
    # the collection's transformed bbox should be centered near (5, 5)
    coll = ax.collections[0]
    bbox = coll.get_datalim(ax.transData)
    cx = (bbox.x0 + bbox.x1) / 2
    assert cx == pytest.approx(5, abs=0.5)


# --- helper unit tests ---------------------------------------------------- #
def test_get_bbox_empty_raises():
    with pytest.raises(ValueError):
        get_bbox([])


def test_get_is_included():
    assert get_is_included("leg_1", "leg") is False
    assert get_is_included("wing", ("leg", "haltere")) is True


def test_get_affine_matrix_scales_to_width():
    bbox = svg2mpl.render.Bbox([[0, 0], [10, 5]])
    affine = get_affine_matrix(bbox, (0, 0), (0, 0), 0, 2, None, (1, 1))
    assert np.allclose(affine.transform((10, 5)), (2, 1))
