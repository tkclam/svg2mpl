"""Image-comparison tests (pytest-mpl).

Run the comparison with ``uv run pytest --mpl``. Without ``--mpl`` the figures
are still built (smoke test) but not compared. Regenerate baselines with::

    uv run pytest --mpl-generate-path=tests/baseline
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pytest

import svg2mpl

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


FEATURES = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
  <defs>
    <clipPath id="c"><circle cx="50" cy="50" r="40"/></clipPath>
    <pattern id="dots" width="10" height="10" patternUnits="userSpaceOnUse">
      <circle cx="5" cy="5" r="3" fill="crimson"/>
    </pattern>
    <marker id="arrow" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="6"
            markerHeight="6" orient="auto-start-reverse">
      <path d="M0 0 L10 5 L0 10 z" fill="navy"/>
    </marker>
    <linearGradient id="fade"><stop offset="0" stop-color="white"/>
      <stop offset="1" stop-color="black"/></linearGradient>
    <mask id="m"><rect x="110" y="10" width="80" height="80" fill="url(#fade)"/></mask>
    <filter id="shadow"><feDropShadow dx="3" dy="3" stdDeviation="2"/></filter>
  </defs>
  <g clip-path="url(#c)">
    <rect width="100" height="100" fill="orange"/>
    <rect width="50" height="50" fill="navy"/>
  </g>
  <rect x="110" y="10" width="80" height="80" fill="teal" mask="url(#m)"/>
  <rect x="10" y="110" width="80" height="40" fill="url(#dots)" stroke="black"/>
  <path d="M20 180 L80 180" stroke="black" stroke-width="2"
        marker-start="url(#arrow)" marker-end="url(#arrow)"/>
  <circle cx="150" cy="130" r="20" fill="gold" filter="url(#shadow)"/>
  <text x="150" y="185" font-family="DejaVu Sans" font-size="18"
        text-anchor="middle" fill="purple">Text</text>
</svg>"""


@pytest.mark.mpl_image_compare(baseline_dir="baseline", tolerance=20)
def test_image_features():
    fig, ax = plt.subplots(figsize=(4, 4))
    svg2mpl.add_svg(FEATURES, ax=ax, place=False)
    ax.set_aspect("equal")
    ax.set_axis_off()
    return fig
