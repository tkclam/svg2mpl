# User guide

This guide walks through the public API. For runnable, plotted versions of these
snippets see the [examples](examples/quickstart.ipynb).

## The two entry points

`svg2mpl` has two layers:

- [`load`][svg2mpl.load] / [`load_string`][svg2mpl.load_string] parse an SVG into
  a [`Scene`][svg2mpl.Scene] — a plain dataclass you can inspect.
- [`add_svg`][svg2mpl.add_svg] does the same and draws the result onto a
  matplotlib `Axes`, with conveniences for placement and styling.

Most of the time you want `add_svg`.

```python
import matplotlib.pyplot as plt
import svg2mpl

fig, ax = plt.subplots()
svg2mpl.add_svg("drawing.svg", ax=ax)
ax.set_aspect("equal")
plt.show()
```

The `source` argument accepts a file path **or** a string of SVG markup (anything
starting with `<`), so you can render inline drawings without touching disk:

```python
svg = '<svg viewBox="0 0 10 10"><circle cx="5" cy="5" r="4" fill="tomato"/></svg>'
svg2mpl.add_svg(svg, ax=ax)
```

## Placing a drawing

By default `add_svg` keeps the drawing's intrinsic size and position. Pass
placement arguments to drop it at a specific spot in **data coordinates**:

```python
svg2mpl.add_svg(
    "logo.svg",
    xy=(2, 3),          # data coordinates of the anchor
    width=1.5,          # final width (height kept to aspect ratio)
    origin=(0.5, 0.5),  # anchor at the drawing's center
    rotation=30,        # degrees
)
```

- `xy` — where the anchor lands, in data coordinates.
- `width` / `height` — the final bounding-box size. Give one and the other is
  derived to preserve the aspect ratio; give both to stretch.
- `origin` — the anchor point *inside* the drawing, normalized so `(0, 0)` is the
  lower-left corner and `(1, 1)` the upper-right. The default `(0.5, 0.5)` centers
  the drawing on `xy`.
- `scale` — an extra multiplier on top of `width`/`height`.
- `rotation` — degrees, counter-clockwise.

Set `place=False` to render the drawing in its own coordinate system instead,
faithfully reproducing the SVG's absolute layout (only `flip_y` still applies).

!!! tip "Stamping a drawing many times"
    Because `add_svg` re-uses a cached parse of each file, calling it in a loop to
    place the same drawing at many `xy` positions (custom markers, icons on a map)
    is cheap. See the [placement example](examples/placement.ipynb).

## Coordinate system & `flip_y`

SVG's y axis points **down**; matplotlib's points **up**. By default `add_svg`
flips the y axis so a drawing looks the same as in a browser. Pass `flip_y=False`
to keep raw SVG coordinates (text will be upside down relative to the browser, but
the numbers match the SVG source).

## Styling

### Stroke widths

SVG stroke widths are lengths in the drawing's own units, so they scale with it:
a drawing placed at `width=1` has thinner lines than the same drawing at
`width=10`, exactly as when you resize it in a browser. svg2mpl recomputes the
matplotlib line widths whenever the figure is drawn, so strokes stay in
proportion when you zoom, resize the figure or save at another DPI.
Strokes marked `vector-effect="non-scaling-stroke"` keep a fixed width instead.

### Global keywords

Any extra keyword is forwarded to every drawn artist, so the full matplotlib patch
vocabulary is available:

```python
svg2mpl.add_svg("drawing.svg", alpha=0.6, linewidth=0.5, edgecolor="black")
```

A `linewidth` you pass (globally or per path) is in **points**, like any
matplotlib line width, and does not scale with the drawing.

### Per-path overrides

`per_path_kwargs` maps a matplotlib property to a `{path_id: value}` dict, letting
you restyle individual elements by their SVG `id`:

```python
svg2mpl.add_svg(
    "chart.svg",
    per_path_kwargs={
        "facecolor": {"bar1": "tab:blue", "bar2": "tab:orange"},
    },
)
```

### Excluding paths

`exclude` takes one or more regular expressions; any path whose `id` matches is not
drawn. `bbox_exclude` works the same way but only affects which paths count toward
the placement bounding box (handy for ignoring a frame or backdrop when centering):

```python
svg2mpl.add_svg("drawing.svg", exclude=r"grid.*", bbox_exclude="background")
```

### Grayscale

```python
svg2mpl.add_svg("drawing.svg", grayscale=True)
```

This converts every color: fills, strokes, gradients, patterns and images.

## Text

Text is converted to glyph outlines with matplotlib's font engine, so it scales,
rotates, clips and takes gradients exactly like the rest of the drawing.
`<tspan>`, `<textPath>`, per-character positioning, anchors, baselines, spacing
and decorations are laid out as in a browser. Fonts are matched against the
fonts matplotlib can find; generic families (`sans-serif`, `serif`,
`monospace`) map to matplotlib's defaults.

If you need real text objects instead (for example selectable text in a PDF),
pass `text_as_paths=False`. Each `<text>` element then becomes a matplotlib
`Text` artist whose font size is in points and does not follow the drawing's
scale.

## Effects drawn as images

matplotlib has no compositing, so some SVG effects are evaluated off-screen and
drawn back as images, positioned and clipped exactly:

- gradients and patterns (clipped to the shape's outline),
- `<mask>` and `filter` (including CSS filter functions such as `blur()`),
- `opacity` on a group whose children overlap (so they don't show through each
  other, as in a browser; non-overlapping children stay vectors).

`raster_resolution` sets the pixel count across the document's larger side used
for these (default 1024):

```python
svg2mpl.add_svg("drawing.svg", raster_resolution=2048)
```

## Return value & `force_patches`

`add_svg` batches shapes into a single
[`PathCollection`](https://matplotlib.org/stable/api/collections_api.html#matplotlib.collections.PathCollection)
when they share the same properties — fast to draw and a single artist to manage.
When the shapes can't be batched (or you pass `force_patches=True`) it returns a
`dict` mapping each path `id` to its
[`PathPatch`](https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.PathPatch.html),
which you can then tweak individually:

```python
patches = svg2mpl.add_svg("drawing.svg", force_patches=True)
patches["circle1"].set_alpha(0.3)
```

Drawings that use clipping, gradients, patterns, masks, filters or images always
return the `dict` form, because those need more than one artist. The extra
artists (images) are added to the axes in document order alongside the patches.

## Inspecting the scene

To read the parsed representation without drawing, use `load`:

```python
scene = svg2mpl.load("drawing.svg")
print(scene.width, scene.height)        # document size
for shape in scene.shapes:
    print(shape.id, shape.kwargs)       # matplotlib kwargs per shape
for text in scene.texts:
    print(text.id, text.content)
```

A [`Scene`][svg2mpl.Scene] holds its drawables in painter's order in
`scene.items`: [`Shape`][svg2mpl.Shape] (a path with its matplotlib style, clip
path and any gradient/pattern paint), [`Image`][svg2mpl.Image] and
[`Group`][svg2mpl.Group] (a masked or filtered subtree). `scene.shapes` lists
every shape, and `scene.texts` holds one [`Text`][svg2mpl.Text] record per
`<text>` element. Geometry is in document space (element transforms and
`viewBox` baked in, but **not** the y-flip — that is applied at render time).
The result of `load` is cached and shared, so treat it as read-only.

See the [coverage table](index.md#coverage) for what is supported and how the
few approximations behave.
