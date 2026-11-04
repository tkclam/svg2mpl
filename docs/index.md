# svg2mpl

Render SVG files into [matplotlib](https://matplotlib.org/) artists.

`svg2mpl` parses an SVG into matplotlib `Path`/`Patch` objects and draws them on
an `Axes`. Unlike a one-off "extract the paths" helper, it aims to be *faithful*
to the SVG: it honors transforms, viewports, CSS, gradients, patterns, clipping, masks,
markers, text, images and filters, so a drawing looks the way it does in a
browser.

```python
import matplotlib.pyplot as plt
import svg2mpl

fig, ax = plt.subplots()
svg2mpl.add_svg("drawing.svg", ax=ax)
ax.set_aspect("equal")
plt.show()
```

## Why svg2mpl?

- **Faithful rendering** — transforms, viewports, CSS, gradients, patterns,
  clip paths, masks, markers, text and filters are applied so a drawing looks
  like it does in a browser. Strokes and text scale with the drawing.
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

svg2mpl targets static SVG 1.1 / SVG 2 rendering. Output is checked against
Chromium on a suite of feature drawings.

| Area | Supported |
| --- | --- |
| Shapes | `path` (full grammar, incl. arcs and compact syntax), `rect` (rounded), `circle`, `ellipse`, `line`, `polyline`, `polygon` |
| Structure | `<g>`, `<a>`, `<switch>` (`systemLanguage`, `requiredExtensions`), nested `<svg>` viewports, `<use>`, `<symbol>`, `<defs>` |
| Coordinates | `viewBox`, `preserveAspectRatio` (all alignments, `meet`/`slice`/`none`), units (`px pt pc in cm mm q em ex %`) |
| Transforms | `matrix`, `translate`, `scale`, `rotate`, `skewX`, `skewY`, CSS `transform` (`deg`/`rad`/`turn`, `translateX` ...) |
| Styling | presentation attributes, inline `style`, `<style>` CSS (simple selectors built in; full selectors with the `[css]` extra), `!important`, `@media`, `inherit`/`initial`/`unset`, `font` and `marker` shorthands |
| Paint | named/hex (`#rgb[a]`, `#rrggbb[aa]`)/`rgb()`/`hsl()` colors, `transparent`, `currentColor`, `url(#id) fallback` paints, `fill-rule`, `opacity`/`fill-opacity`/`stroke-opacity`, group opacity (composited when children overlap), `paint-order` |
| Strokes | width (scales with the drawing), caps, joins, dashes with `stroke-dashoffset`, `vector-effect: non-scaling-stroke` |
| Gradients | linear and radial, `objectBoundingBox`/`userSpaceOnUse`, `gradientTransform`, `spreadMethod` (`pad`/`reflect`/`repeat`), focal point (`fx`/`fy`/`fr`), `href` inheritance, gradient strokes |
| Patterns | `patternUnits`, `patternContentUnits`, `viewBox`, `patternTransform`, `href` inheritance |
| Clipping & masking | `clip-path` (incl. `objectBoundingBox`, `clip-rule`, nested clips, text clips), `<mask>` (luminance and alpha) |
| Markers | `marker-start`/`mid`/`end`, `orient` (`auto`, `auto-start-reverse`, angles), `markerUnits`, `viewBox`, `refX`/`refY`, overflow clipping |
| Text | rendered as glyph outlines: `<tspan>`, `<textPath>`, `x`/`y`/`dx`/`dy`/`rotate` lists, `text-anchor`, baselines, `baseline-shift`, `letter-`/`word-spacing`, `textLength`, `text-decoration`, whitespace handling, kerning |
| Images | `<image>` with PNG/JPEG/GIF/... (data URIs or local files) and nested SVG documents (kept as vectors), `preserveAspectRatio`, `image-rendering` |
| Filters | `feBlend`, `feColorMatrix`, `feComponentTransfer`, `feComposite`, `feConvolveMatrix`, `feDiffuseLighting`, `feDisplacementMap`, `feDropShadow`, `feFlood`, `feGaussianBlur`, `feImage`, `feMerge`, `feMorphology`, `feOffset`, `feSpecularLighting`, `feTile`, `feTurbulence`, plus CSS filter functions |

### Limitations

matplotlib is a vector library without compositing, so a few things are
approximated:

- **Masks, filters and overlapping group opacity** are composited off-screen
  and drawn back as an image (resolution set by `raster_resolution`).
- **Gradients and patterns** are drawn as images clipped to the shape.
- **Non-uniformly scaled strokes** (e.g. under `scale(2, 1)`) use a single,
  averaged width; matplotlib strokes cannot be anisotropic.
- **Fonts** are matched against the fonts matplotlib knows about; generic
  families map to matplotlib's defaults, which may differ from a browser's.
- **Not supported:** animation (SMIL), scripting, `<foreignObject>`,
  `mix-blend-mode`, vertical writing modes and bidirectional text, external
  (`http://`) resources, `BackgroundImage` filter inputs.

## License

Released under the [MIT License](https://github.com/tkclam/svg2mpl/blob/main/LICENSE).
