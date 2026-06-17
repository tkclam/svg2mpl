"""Image-comparison tests (pytest-mpl).

Run the comparison with ``uv run pytest --mpl``. Without ``--mpl`` the figures
are still built (smoke test) but not compared. Regenerate baselines with::

    uv run pytest --mpl-generate-path=tests/baseline
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

import svg2mpl  # noqa: E402

ASSET = Path(__file__).parent / "assets" / "basic.svg"

GRADIENT = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 60">
  <defs>
    <linearGradient id="lg" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0" stop-color="red"/>
      <stop offset="1" stop-color="blue"/>
    </linearGradient>
    <radialGradient id="rg">
      <stop offset="0" stop-color="yellow"/>
      <stop offset="1" stop-color="green"/>
    </radialGradient>
  </defs>
  <rect x="5" y="5" width="40" height="20" fill="url(#lg)"/>
  <circle cx="70" cy="20" r="15" fill="url(#rg)" stroke="black" stroke-width="1"/>
</svg>"""


@pytest.mark.mpl_image_compare(baseline_dir="baseline", tolerance=20)
def test_image_basic_shapes():
    fig, ax = plt.subplots(figsize=(4, 4))
    svg2mpl.add_svg(ASSET, ax=ax, place=False)
    ax.set_aspect("equal")
    ax.set_axis_off()
    return fig


@pytest.mark.mpl_image_compare(baseline_dir="baseline", tolerance=20)
def test_image_transforms_and_groups():
    svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
      <g transform="translate(50,50)">
        <g transform="rotate(30)">
          <rect x="-20" y="-10" width="40" height="20" fill="coral"/>
        </g>
        <circle r="6" fill="black"/>
      </g>
    </svg>"""
    fig, ax = plt.subplots(figsize=(4, 4))
    svg2mpl.add_svg(svg, ax=ax, place=False)
    ax.set_aspect("equal")
    ax.set_axis_off()
    return fig


@pytest.mark.mpl_image_compare(baseline_dir="baseline", tolerance=20)
def test_image_gradients():
    fig, ax = plt.subplots(figsize=(5, 3))
    svg2mpl.add_svg(GRADIENT, ax=ax, place=False)
    ax.set_aspect("equal")
    ax.set_axis_off()
    return fig
