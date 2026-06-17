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

### Global keywords

Any extra keyword is forwarded to every drawn artist, so the full matplotlib patch
vocabulary is available:

```python
svg2mpl.add_svg("drawing.svg", alpha=0.6, linewidth=0.5, edgecolor="black")
```

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

Gradient-filled shapes always fall back to individual patches because each needs a
clipped gradient image.

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

A [`Scene`][svg2mpl.Scene] holds [`Shape`][svg2mpl.Shape] and
[`Text`][svg2mpl.Text] drawables with their geometry already in document space
(element transforms and `viewBox` baked in, but **not** the y-flip — that is applied
at render time). The result of `load` is cached and shared, so treat it as
read-only.

!!! warning "Best-effort features"
    **Gradients** are approximated with a clipped image because matplotlib has no
    native gradient fill, and **`<text>`** uses approximate font matching. See the
    docstrings in the [API reference](api.md) for current limitations.
