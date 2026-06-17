# svg2mpl

Render SVG files into [matplotlib](https://matplotlib.org/) artists.

`svg2mpl` parses an SVG into matplotlib `Path`/`Patch` objects and draws them on
an `Axes`. Unlike a one-off "extract the paths" helper, it aims to be *faithful*
to the SVG: it honors element `transform`s, `<g>` group inheritance, `viewBox`,
the basic shape elements, and CSS styling.

📖 **[Documentation](https://tkclam.github.io/svg2mpl/)** — guide, runnable
example notebooks with rendered plots, and the full API reference.

## Install

```bash
pip install svg2mpl
# with <style>-block CSS support:
pip install "svg2mpl[css]"
```

## Quick start

```python
import matplotlib.pyplot as plt
import svg2mpl

fig, ax = plt.subplots()
svg2mpl.add_svg("drawing.svg", ax=ax)
ax.set_aspect("equal")
plt.show()
```

`add_svg` can also place a drawing at a position/size/rotation, exclude paths by
id, recolor per id, or convert to grayscale:

```python
svg2mpl.add_svg(
    "logo.svg",
    xy=(2, 3),          # data coordinates of the anchor
    width=1.5,          # final width (height kept to aspect ratio)
    origin=(0.5, 0.5),  # anchor at the drawing's center
    rotation=30,
    exclude=r"grid.*",  # skip paths whose id matches
)
```

To inspect the parsed scene without drawing:

```python
scene = svg2mpl.load("drawing.svg")
for shape in scene.shapes:
    print(shape.id, shape.kwargs)
```

## Coverage

| Area | Supported |
| --- | --- |
| Shapes | `path`, `rect` (incl. rounded), `circle`, `ellipse`, `line`, `polyline`, `polygon` |
| Structure | `<g>` groups & inheritance, `<use>`/`<defs>`/`<symbol>`, `viewBox` |
| Transforms | `matrix`, `translate`, `scale`, `rotate`, `skewX`, `skewY` |
| Styling | presentation attributes, inline `style`, `<style>` CSS (with `[css]` extra) |
| Paint | named/hex/`rgb()`/`rgba()`/`hsl()` colors, `currentColor`, per-paint opacity |

Gradients and `<text>` have best-effort support; see the docstrings for the
current limitations (matplotlib has no native gradient fill, and font matching is
approximate).

## Coordinate system

SVG's y axis points down; matplotlib's points up. By default `add_svg` flips the
y axis so a drawing looks the same as in a browser. Pass `flip_y=False` to keep
raw SVG coordinates.

## License

MIT
