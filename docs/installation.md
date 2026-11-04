# Installation

`svg2mpl` requires Python 3.10 or newer.

## From PyPI

```bash
pip install svg2mpl
```

This pulls in `matplotlib`, `numpy`, and `pillow` (for embedded images).

### CSS `<style>`-block support

Presentation attributes (`fill="red"`), inline styles (`style="fill:red"`) and
`<style>` blocks with simple selectors (type, `.class`, `#id`, attribute
selectors, descendant/child combinators) work out of the box. For the full CSS
selector grammar (pseudo-classes, sibling combinators ...), install the
optional `css` extra:

```bash
pip install "svg2mpl[css]"
```

This adds `tinycss2` and `cssselect2`.

## With uv

```bash
uv add svg2mpl
# with the CSS extra:
uv add "svg2mpl[css]"
```

## From source

```bash
git clone https://github.com/tkclam/svg2mpl
cd svg2mpl
uv sync --extra css
```

To build the documentation locally, sync the `docs` dependency group and serve
it with live reload:

```bash
uv sync --group docs --extra css
uv run mkdocs serve
```

## Verify the install

```python
import svg2mpl

print(svg2mpl.__version__)
```
