# svg2mpl

Render SVG files into [matplotlib](https://matplotlib.org/) artists.

`svg2mpl` parses an SVG into matplotlib `Path`/`Patch` objects and draws them on
an `Axes`. Unlike a one-off "extract the paths" helper, it aims to be *faithful*
to the SVG: it honors element `transform`s, `<g>` group inheritance, `viewBox`,
the basic shape elements, and CSS styling.

```python
import matplotlib.pyplot as plt
import svg2mpl

fig, ax = plt.subplots()
svg2mpl.add_svg("drawing.svg", ax=ax)
ax.set_aspect("equal")
plt.show()
```

## Why svg2mpl?

- **Faithful rendering** — transforms, group inheritance, `viewBox`, and CSS are
  applied so a drawing looks like it does in a browser.
- **Drop it into a plot** — place a drawing at any data coordinate, scale it to a
  target size, rotate it, or stamp it repeatedly as a custom marker.
- **Style at render time** — exclude paths by id, recolor per id, convert to
  grayscale, or pass any matplotlib patch keyword.
- **Inspect the scene** — load an SVG into a plain dataclass and read off the
  parsed shapes and styles without drawing anything.

## Where to next

<div class="grid cards" markdown>

- :material-download: **[Installation](installation.md)** — install from PyPI,
  with or without the CSS extra.
- :material-book-open-variant: **[User guide](guide.md)** — the full tour of
  `add_svg`, placement, styling, and the scene model.
- :material-notebook: **[Examples](examples/quickstart.ipynb)** — runnable
  notebooks with rendered plots.
- :material-api: **[API reference](api.md)** — every public function and
  dataclass.

</div>

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

## License

Released under the [MIT License](https://github.com/tkclam/svg2mpl/blob/main/LICENSE).
