"""SVG filter effects, evaluated with numpy on a rasterized element.

An element with ``filter`` is rasterized off-screen over its filter region;
the primitives below then run on that RGBA image, which is drawn back as an
image. Implemented primitives:

``feBlend``, ``feColorMatrix``, ``feComponentTransfer``, ``feComposite``,
``feConvolveMatrix``, ``feDiffuseLighting``, ``feDisplacementMap``,
``feDropShadow``, ``feFlood``, ``feGaussianBlur``, ``feImage``, ``feMerge``,
``feMorphology``, ``feOffset``, ``feSpecularLighting``, ``feTile`` and
``feTurbulence``, plus the CSS filter functions (``blur()``, ``drop-shadow()``,
``grayscale()``, ``sepia()``, ``saturate()``, ``hue-rotate()``, ``invert()``,
``opacity()``, ``brightness()``, ``contrast()``).

Inputs ``SourceGraphic``, ``SourceAlpha`` and named results are supported;
``BackgroundImage``/``BackgroundAlpha``/``FillPaint``/``StrokePaint`` evaluate
to transparent black. ``color-interpolation-filters`` (``linearRGB`` by
default, or ``sRGB``) is honored per filter. Lengths are converted through the
element's transform; filters on rotated or skewed elements are approximated
axis-aligned.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np
from matplotlib.transforms import Affine2D

from .color import parse_color
from .shapes import strip_ns
from .units import parse_length, parse_numbers

__all__ = ["Filter", "parse_filter", "css_filter", "apply_filter"]

_XLINK = "{http://www.w3.org/1999/xlink}href"


@dataclass
class Filter:
    """A resolved filter chain (all lengths in document units)."""

    primitives: list[dict]
    region: tuple[float, float, float, float]
    color_space: str = "linearRGB"
    # Extra items to rasterize (feImage referencing elements), keyed by index.
    images: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def _scales(ctm: Affine2D) -> tuple[float, float]:
    m = ctm.get_matrix()
    return math.hypot(m[0, 0], m[1, 0]), math.hypot(m[0, 1], m[1, 1])


def _vec(ctm: Affine2D, dx: float, dy: float) -> tuple[float, float]:
    m = ctm.get_matrix()
    return m[0, 0] * dx + m[0, 1] * dy, m[1, 0] * dx + m[1, 1] * dy


def _bbox_rect(rect, ctm):
    x0, y0, x1, y1 = rect
    pts = ctm.transform([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
    return (pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max())


def _frac(v: str) -> float:
    v = v.strip()
    try:
        return float(v[:-1]) / 100 if v.endswith("%") else float(v)
    except ValueError:
        return 0.0


def parse_filter(parser, f_el, bbox_user, ctm, st) -> Filter | None:
    """Resolve a ``<filter>`` element for one referencing element."""
    from .parser import _region

    # href inheritance: attributes and primitives from referenced filters.
    chain, seen, el = [], set(), f_el
    while el is not None and id(el) not in seen:
        seen.add(id(el))
        chain.append(el)
        href = el.get(_XLINK) or el.get("href")
        el = parser.id_index.get(href[1:]) if href and href.startswith("#") else None

    def attr(name):
        for e in chain:
            if name in e.attrib:
                return e.attrib[name]
        return None

    fstyle = parser.style_of(f_el)
    ctx = parser.ctx(fstyle, st)
    units = attr("filterUnits") or "objectBoundingBox"
    proxy = _Attrs({n: attr(n) for n in ("x", "y", "width", "height") if attr(n)})
    rect = _region(proxy, units, bbox_user, ctx, ("-10%", "-10%", "120%", "120%"))
    if rect is None:
        return None
    prim_units = attr("primitiveUnits") or "userSpaceOnUse"
    host = next((e for e in chain if any(isinstance(c.tag, str) for c in e)), f_el)
    space = (fstyle.get("color-interpolation-filters") or "linearRGB").strip()
    out = Filter(primitives=[], region=_bbox_rect(rect, ctm), color_space=space)
    kx, ky = _scales(ctm)
    bw = (bbox_user[2] - bbox_user[0]) if bbox_user else 1.0
    bh = (bbox_user[3] - bbox_user[1]) if bbox_user else 1.0

    def num_pair(value, default=(0.0, 0.0)):
        nums = parse_numbers(value)
        if not nums:
            return default
        return (nums[0], nums[1] if len(nums) > 1 else nums[0])

    def plen(v, axis):
        """A primitive length (number) in user units."""
        if prim_units == "objectBoundingBox":
            return v * (bw if axis == "x" else bh)
        return v

    for idx, child in enumerate(c for c in host if isinstance(c.tag, str)):
        tag = strip_ns(child.tag)
        if not tag.startswith("fe"):
            continue
        cstyle = parser.child_style(child, fstyle)
        a = child.attrib
        prim: dict = {
            "type": tag,
            "in": a.get("in"),
            "in2": a.get("in2"),
            "result": a.get("result"),
        }
        # Primitive subregion.
        if any(k in a for k in ("x", "y", "width", "height")):
            if prim_units == "objectBoundingBox" and bbox_user:
                sx0 = bbox_user[0] + _frac(a["x"]) * bw if "x" in a else rect[0]
                sy0 = bbox_user[1] + _frac(a["y"]) * bh if "y" in a else rect[1]
                sw = _frac(a["width"]) * bw if "width" in a else rect[2] - sx0
                sh = _frac(a["height"]) * bh if "height" in a else rect[3] - sy0
            else:
                sx0 = parse_length(a.get("x"), rect[0], axis="x", ctx=ctx)
                sy0 = parse_length(a.get("y"), rect[1], axis="y", ctx=ctx)
                sw = parse_length(a.get("width"), rect[2] - sx0, axis="x", ctx=ctx)
                sh = parse_length(a.get("height"), rect[3] - sy0, axis="y", ctx=ctx)
            if sw > 0 and sh > 0:
                prim["subregion"] = _bbox_rect((sx0, sy0, sx0 + sw, sy0 + sh), ctm)
        if tag in ("feGaussianBlur", "feDropShadow"):
            sx, sy = num_pair(
                a.get("stdDeviation"),
                (2.0, 2.0) if tag == "feDropShadow" else (0.0, 0.0),
            )
            prim["std"] = (plen(sx, "x") * kx, plen(sy, "y") * ky)
        if tag in ("feOffset", "feDropShadow"):
            d = 2.0 if tag == "feDropShadow" else 0.0
            dx = parse_numbers(a.get("dx")) or [d]
            dy = parse_numbers(a.get("dy")) or [d]
            prim["offset"] = _vec(ctm, plen(dx[0], "x"), plen(dy[0], "y"))
        if tag in ("feFlood", "feDropShadow"):
            from matplotlib.colors import to_rgba

            from .style import opacity

            c = parse_color(cstyle.get("flood-color", "black"), default="black")
            r, g, b, al = to_rgba(c if c != "none" else (0, 0, 0, 0))
            prim["color"] = (r, g, b, al * opacity(cstyle.get("flood-opacity")))
        if tag == "feColorMatrix":
            prim["mtype"] = (a.get("type") or "matrix").strip()
            prim["values"] = parse_numbers(a.get("values"))
        if tag == "feComponentTransfer":
            funcs = {}
            for f in child:
                if not isinstance(f.tag, str):
                    continue
                ft = strip_ns(f.tag)
                if ft in ("feFuncR", "feFuncG", "feFuncB", "feFuncA"):
                    funcs[ft[-1]] = {
                        "type": (f.get("type") or "identity").strip(),
                        "table": parse_numbers(f.get("tableValues")),
                        "slope": _num(f.get("slope"), 1.0),
                        "intercept": _num(f.get("intercept"), 0.0),
                        "amplitude": _num(f.get("amplitude"), 1.0),
                        "exponent": _num(f.get("exponent"), 1.0),
                        "offset": _num(f.get("offset"), 0.0),
                    }
            prim["funcs"] = funcs
        if tag == "feComposite":
            prim["operator"] = (a.get("operator") or "over").strip()
            prim["k"] = tuple(_num(a.get(f"k{i}"), 0.0) for i in range(1, 5))
        if tag == "feBlend":
            prim["mode"] = (a.get("mode") or "normal").strip()
        if tag == "feMerge":
            prim["nodes"] = [
                n.get("in")
                for n in child
                if isinstance(n.tag, str) and strip_ns(n.tag) == "feMergeNode"
            ]
        if tag == "feMorphology":
            rx, ry = num_pair(a.get("radius"))
            prim["operator"] = (a.get("operator") or "erode").strip()
            prim["radius"] = (plen(rx, "x") * kx, plen(ry, "y") * ky)
        if tag == "feConvolveMatrix":
            ox, oy = num_pair(a.get("order"), (3.0, 3.0))
            prim["order"] = (int(ox), int(oy))
            prim["kernel"] = parse_numbers(a.get("kernelMatrix"))
            prim["divisor"] = _num(a.get("divisor"), 0.0)
            prim["bias"] = _num(a.get("bias"), 0.0)
            prim["target"] = (
                int(_num(a.get("targetX"), int(ox) // 2)),
                int(_num(a.get("targetY"), int(oy) // 2)),
            )
            prim["edge"] = (a.get("edgeMode") or "duplicate").strip()
            prim["preserve_alpha"] = (
                a.get("preserveAlpha") or "false"
            ).strip() == "true"
        if tag == "feDisplacementMap":
            prim["scale"] = plen(_num(a.get("scale"), 0.0), "x") * kx
            prim["xch"] = (a.get("xChannelSelector") or "A").strip()
            prim["ych"] = (a.get("yChannelSelector") or "A").strip()
        if tag == "feTurbulence":
            fx, fy = num_pair(a.get("baseFrequency"))
            prim["freq"] = (fx / max(kx, 1e-12), fy / max(ky, 1e-12))
            prim["octaves"] = int(_num(a.get("numOctaves"), 1))
            prim["seed"] = _num(a.get("seed"), 0.0)
            prim["stitch"] = (a.get("stitchTiles") or "noStitch") == "stitch"
            prim["fractal"] = (a.get("type") or "turbulence").strip() == "fractalNoise"
            prim["origin_doc"] = ctm
        if tag in ("feDiffuseLighting", "feSpecularLighting"):
            from matplotlib.colors import to_rgb

            lc = parse_color(cstyle.get("lighting-color", "white"), default="white")
            prim["color"] = to_rgb(lc if lc != "none" else "black")
            prim["surface_scale"] = _num(a.get("surfaceScale"), 1.0)
            prim["kd"] = _num(a.get("diffuseConstant"), 1.0)
            prim["ks"] = _num(a.get("specularConstant"), 1.0)
            prim["exp"] = _num(a.get("specularExponent"), 1.0)
            prim["light"] = _light(child, ctm, plen)
            prim["unit_scale"] = (kx + ky) / 2
        if tag == "feImage":
            href = a.get(_XLINK) or a.get("href")
            prim["href"] = href
            if href and href.startswith("#"):
                target = parser.id_index.get(href[1:])
                if target is not None:
                    from dataclasses import replace

                    items: list = []
                    parser.walk(
                        target, replace(st, ctm=ctm, out=items, clip=None, opacity=1.0)
                    )
                    out.images[idx] = ("items", items)
            elif href:
                loaded = parser.load_image(href)
                if loaded is not None and loaded[0] == "raster":
                    sub = prim.get("subregion") or out.region
                    out.images[idx] = ("raster", loaded[1], sub)
            prim["image_index"] = idx
        out.primitives.append(prim)
    return out


class _Attrs:
    """A minimal element stand-in exposing ``get`` for :func:`_region`."""

    def __init__(self, attrs):
        self.attrs = attrs

    def get(self, name, default=None):
        return self.attrs.get(name, default)


def _num(value, default):
    nums = parse_numbers(value)
    return nums[0] if nums else default


def _light(el, ctm, plen):
    for c in el:
        if not isinstance(c.tag, str):
            continue
        t = strip_ns(c.tag)
        if t == "feDistantLight":
            return {
                "type": "distant",
                "azimuth": _num(c.get("azimuth"), 0.0),
                "elevation": _num(c.get("elevation"), 0.0),
            }
        if t in ("fePointLight", "feSpotLight"):
            x, y = plen(_num(c.get("x"), 0.0), "x"), plen(_num(c.get("y"), 0.0), "y")
            px, py = ctm.transform((x, y))
            z = plen(_num(c.get("z"), 0.0), "x") * _scales(ctm)[0]
            light = {"type": "point", "pos": (px, py, z)}
            if t == "feSpotLight":
                tx = plen(_num(c.get("pointsAtX"), 0.0), "x")
                ty = plen(_num(c.get("pointsAtY"), 0.0), "y")
                qx, qy = ctm.transform((tx, ty))
                qz = plen(_num(c.get("pointsAtZ"), 0.0), "x") * _scales(ctm)[0]
                light.update(
                    type="spot",
                    at=(qx, qy, qz),
                    exp=_num(c.get("specularExponent"), 1.0),
                    cone=_num(c.get("limitingConeAngle"), None)
                    if c.get("limitingConeAngle")
                    else None,
                )
            return light
    return {"type": "distant", "azimuth": 0.0, "elevation": 0.0}


_CSS_FUNC = re.compile(r"([a-z-]+)\(([^()]*(?:\([^()]*\)[^()]*)*)\)", re.IGNORECASE)


def css_filter(value: str, bbox_user, ctm, ctx) -> Filter | None:
    """Translate CSS filter functions into an equivalent primitive chain."""
    if bbox_user is None:
        return None
    kx, ky = _scales(ctm)
    prims: list[dict] = []
    margin = 0.0
    for name, args in _CSS_FUNC.findall(value):
        name = name.lower()
        args = args.strip()

        def amount(default=1.0, args=args):
            if not args:
                return default
            v = args.split()[0]
            return float(v[:-1]) / 100 if v.endswith("%") else float(v)

        if name == "blur":
            s = parse_length(args or "0", 0.0, ctx=ctx)
            prims.append({"type": "feGaussianBlur", "std": (s * kx, s * ky)})
            margin = max(margin, 3 * s)
        elif name == "drop-shadow":
            color_m = re.search(
                r"(#[0-9a-fA-F]+|rgba?\([^)]*\)|hsla?\([^)]*\)|[a-zA-Z]+)\s*$", args
            )
            color = "black"
            if color_m and not re.match(r"^[\d.]", color_m.group(1)):
                color = color_m.group(1)
                args = args[: color_m.start()]
            nums = [parse_length(t, 0.0, ctx=ctx) for t in args.split()]
            nums += [0.0] * (3 - len(nums))
            from matplotlib.colors import to_rgba

            c = parse_color(color, default="black")
            prims.append(
                {
                    "type": "feDropShadow",
                    "offset": _vec(ctm, nums[0], nums[1]),
                    "std": (nums[2] / 2 * kx, nums[2] / 2 * ky),
                    "color": to_rgba(c if c != "none" else (0, 0, 0, 0)),
                }
            )
            margin = max(margin, abs(nums[0]) + abs(nums[1]) + 3 * nums[2])
        elif name in (
            "grayscale",
            "sepia",
            "saturate",
            "hue-rotate",
            "invert",
            "opacity",
            "brightness",
            "contrast",
        ):
            if name == "hue-rotate":
                from .transform import _numbers

                deg = (_numbers(args) or [0.0])[0]
                prims.append(
                    {"type": "feColorMatrix", "mtype": "hueRotate", "values": [deg]}
                )
                continue
            a = amount()
            prims.append(_css_color_prim(name, a))
        else:
            return None
    if not prims:
        return None
    x0, y0, x1, y1 = bbox_user
    pad_x = 0.1 * (x1 - x0) + margin
    pad_y = 0.1 * (y1 - y0) + margin
    region = _bbox_rect((x0 - pad_x, y0 - pad_y, x1 + pad_x, y1 + pad_y), ctm)
    return Filter(primitives=prims, region=region, color_space="sRGB")


def _css_color_prim(name, a):
    if name == "grayscale":
        a = min(max(a, 0.0), 1.0)
        return {"type": "feColorMatrix", "mtype": "saturate", "values": [1 - a]}
    if name == "saturate":
        return {"type": "feColorMatrix", "mtype": "saturate", "values": [a]}
    if name == "sepia":
        a = min(max(a, 0.0), 1.0)
        m = np.array(
            [[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]]
        )
        m = (1 - a) * np.eye(3) + a * m
        vals = []
        for r in range(3):
            vals += list(m[r]) + [0, 0]
        vals += [0, 0, 0, 1, 0]
        return {"type": "feColorMatrix", "mtype": "matrix", "values": vals}
    funcs = {}
    if name == "invert":
        for ch in "RGB":
            funcs[ch] = {"type": "table", "table": [a, 1 - a]}
    elif name == "opacity":
        funcs["A"] = {"type": "table", "table": [0, min(max(a, 0.0), 1.0)]}
    elif name == "brightness":
        for ch in "RGB":
            funcs[ch] = {"type": "linear", "slope": a, "intercept": 0.0}
    elif name == "contrast":
        for ch in "RGB":
            funcs[ch] = {"type": "linear", "slope": a, "intercept": -(0.5 * a) + 0.5}
    return {"type": "feComponentTransfer", "funcs": funcs}


# --------------------------------------------------------------------------- #
# Evaluation (premultiplied RGBA float images, row 0 = top)
# --------------------------------------------------------------------------- #
def _to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _to_srgb(c):
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def _premul(img):
    out = img.copy()
    out[..., :3] *= out[..., 3:4]
    return out


def _unpremul(img):
    out = img.copy()
    a = out[..., 3:4]
    with np.errstate(divide="ignore", invalid="ignore"):
        out[..., :3] = np.where(a > 1e-9, out[..., :3] / a, 0.0)
    return out


def _box_blur_axis(img, d, axis):
    """Box blur of width ``d`` pixels along ``axis`` (transparent edges)."""
    if d <= 1:
        return img
    n = img.shape[axis]
    lo = d // 2
    hi = d - lo - 1
    pad = [(0, 0)] * img.ndim
    pad[axis] = (d, d)
    c = np.cumsum(np.pad(img, pad), axis=axis)
    zero = np.zeros_like(np.take(c, [0], axis=axis))
    c = np.concatenate([zero, c], axis=axis)
    idx = np.arange(n) + d
    return (np.take(c, idx + hi + 1, axis=axis) - np.take(c, idx - lo, axis=axis)) / d


def _gauss_axis(img, sigma, axis):
    if sigma <= 0:
        return img
    if sigma < 2.0:
        r = max(1, int(math.ceil(sigma * 3)))
        x = np.arange(-r, r + 1)
        k = np.exp(-0.5 * (x / sigma) ** 2)
        k /= k.sum()
        pad = [(0, 0)] * img.ndim
        pad[axis] = (r, r)
        p = np.pad(img, pad)
        out = np.zeros_like(img)
        n = img.shape[axis]
        for i, w in enumerate(k):
            out += w * np.take(p, np.arange(i, i + n), axis=axis)
        return out
    d = int(math.floor(sigma * 3 * math.sqrt(2 * math.pi) / 4 + 0.5))
    for _ in range(3):
        img = _box_blur_axis(img, d, axis)
    return img


def gaussian_blur(img, sx_px, sy_px):
    """Separable Gaussian blur of an ``(h, w, c)`` array (sigmas in pixels)."""
    out = _gauss_axis(img, sy_px, 0)
    return _gauss_axis(out, sx_px, 1)


def _shift(img, dx_px, dy_px):
    """Shift by (possibly fractional) pixels, filling with transparency."""
    out = img
    ix, iy = int(math.floor(dx_px)), int(math.floor(dy_px))
    fx, fy = dx_px - ix, dy_px - iy

    def int_shift(a, sx, sy):
        res = np.zeros_like(a)
        h, w = a.shape[:2]
        if abs(sx) >= w or abs(sy) >= h:
            return res
        ys = slice(max(sy, 0), h + min(sy, 0))
        yd = slice(max(-sy, 0), h + min(-sy, 0))
        xs = slice(max(sx, 0), w + min(sx, 0))
        xd = slice(max(-sx, 0), w + min(-sx, 0))
        res[ys, xs] = a[yd, xd]
        return res

    out = int_shift(img, ix, iy)
    if fx > 1e-6 or fy > 1e-6:
        right = int_shift(out, 1, 0)
        down = int_shift(out, 0, 1)
        diag = int_shift(out, 1, 1)
        out = (
            (1 - fx) * (1 - fy) * out
            + fx * (1 - fy) * right
            + (1 - fx) * fy * down
            + fx * fy * diag
        )
    return out


def _color_matrix(img_unpre, mtype, values):
    rgb = img_unpre[..., :3]
    a = img_unpre[..., 3:4]
    if mtype == "matrix":
        if len(values) != 20:
            return img_unpre
        m = np.array(values).reshape(4, 5)
        src = np.concatenate([rgb, a, np.ones_like(a)], axis=-1)
        out = src @ m.T
        return np.clip(out, 0.0, 1.0)
    if mtype == "saturate":
        s = values[0] if values else 1.0
        m = np.array(
            [
                [0.213 + 0.787 * s, 0.715 - 0.715 * s, 0.072 - 0.072 * s],
                [0.213 - 0.213 * s, 0.715 + 0.285 * s, 0.072 - 0.072 * s],
                [0.213 - 0.213 * s, 0.715 - 0.715 * s, 0.072 + 0.928 * s],
            ]
        )
        return np.concatenate([np.clip(rgb @ m.T, 0, 1), a], axis=-1)
    if mtype == "hueRotate":
        h = math.radians(values[0] if values else 0.0)
        c, s = math.cos(h), math.sin(h)
        m = np.array(
            [
                [
                    0.213 + c * 0.787 - s * 0.213,
                    0.715 - c * 0.715 - s * 0.715,
                    0.072 - c * 0.072 + s * 0.928,
                ],
                [
                    0.213 - c * 0.213 + s * 0.143,
                    0.715 + c * 0.285 + s * 0.140,
                    0.072 - c * 0.072 - s * 0.283,
                ],
                [
                    0.213 - c * 0.213 - s * 0.787,
                    0.715 - c * 0.715 + s * 0.715,
                    0.072 + c * 0.928 + s * 0.072,
                ],
            ]
        )
        return np.concatenate([np.clip(rgb @ m.T, 0, 1), a], axis=-1)
    if mtype == "luminanceToAlpha":
        lum = rgb @ np.array([0.2125, 0.7154, 0.0721])
        out = np.zeros_like(img_unpre)
        out[..., 3] = np.clip(lum, 0, 1)
        return out
    return img_unpre


def _transfer(c, f):
    t = f.get("type", "identity")
    if t == "table":
        vals = f.get("table") or []
        if not vals:
            return c
        n = len(vals) - 1
        if n == 0:
            return np.full_like(c, vals[0])
        return np.interp(c, np.linspace(0, 1, n + 1), vals)
    if t == "discrete":
        vals = f.get("table") or []
        if not vals:
            return c
        n = len(vals)
        idx = np.clip(np.floor(c * n).astype(int), 0, n - 1)
        return np.asarray(vals)[idx]
    if t == "linear":
        return f.get("slope", 1.0) * c + f.get("intercept", 0.0)
    if t == "gamma":
        return f.get("amplitude", 1.0) * np.power(
            np.clip(c, 0, None), f.get("exponent", 1.0)
        ) + f.get("offset", 0.0)
    return c


def _composite(a, b, op, k):
    """Porter-Duff on premultiplied images: ``a`` = in, ``b`` = in2."""
    aa, ab = a[..., 3:4], b[..., 3:4]
    if op == "over":
        return a + b * (1 - aa)
    if op == "in":
        return a * ab
    if op == "out":
        return a * (1 - ab)
    if op == "atop":
        return a * ab + b * (1 - aa)
    if op == "xor":
        return a * (1 - ab) + b * (1 - aa)
    if op == "lighter":
        return np.clip(a + b, 0, 1)
    if op == "arithmetic":
        k1, k2, k3, k4 = k
        out = np.clip(k1 * a * b + k2 * a + k3 * b + k4, 0, 1)
        out[..., :3] = np.minimum(out[..., :3], out[..., 3:4])
        return out
    return a + b * (1 - aa)


def _blend(a, b, mode):
    """``feBlend``: ``a`` (in) over ``b`` (in2) with a separable blend mode."""
    ua, ub = _unpremul(a)[..., :3], _unpremul(b)[..., :3]
    aa, ab = a[..., 3:4], b[..., 3:4]
    if mode == "multiply":
        f = ua * ub
    elif mode == "screen":
        f = ua + ub - ua * ub
    elif mode == "darken":
        f = np.minimum(ua, ub)
    elif mode == "lighten":
        f = np.maximum(ua, ub)
    elif mode == "overlay":
        f = np.where(ub <= 0.5, 2 * ua * ub, 1 - 2 * (1 - ua) * (1 - ub))
    elif mode == "hard-light":
        f = np.where(ua <= 0.5, 2 * ua * ub, 1 - 2 * (1 - ua) * (1 - ub))
    elif mode == "color-dodge":
        with np.errstate(divide="ignore", invalid="ignore"):
            f = np.where(ua >= 1, 1.0, np.minimum(1.0, ub / (1 - ua)))
    elif mode == "color-burn":
        with np.errstate(divide="ignore", invalid="ignore"):
            f = np.where(ua <= 0, 0.0, 1 - np.minimum(1.0, (1 - ub) / ua))
    elif mode == "difference":
        f = np.abs(ua - ub)
    elif mode == "exclusion":
        f = ua + ub - 2 * ua * ub
    elif mode == "soft-light":
        d = np.where(ub <= 0.25, ((16 * ub - 12) * ub + 4) * ub, np.sqrt(ub))
        f = np.where(
            ua <= 0.5, ub - (1 - 2 * ua) * ub * (1 - ub), ub + (2 * ua - 1) * (d - ub)
        )
    else:
        f = ua
    out_a = aa + ab - aa * ab
    rgb = (1 - ab) * a[..., :3] + (1 - aa) * b[..., :3] + aa * ab * f
    return np.concatenate([rgb, out_a], axis=-1)


def _morphology(img, op, rx, ry):
    fn = np.minimum if op == "erode" else np.maximum
    out = img
    for axis, r in ((1, int(round(rx))), (0, int(round(ry)))):
        if r <= 0:
            continue
        pad = [(0, 0)] * img.ndim
        pad[axis] = (r, r)
        p = np.pad(
            out, pad, mode="constant", constant_values=0.0 if op == "dilate" else 0.0
        )
        n = out.shape[axis]
        acc = np.take(p, np.arange(0, n), axis=axis)
        for i in range(1, 2 * r + 1):
            acc = fn(acc, np.take(p, np.arange(i, i + n), axis=axis))
        out = acc
    return out


def _convolve(img, prim):
    ox, oy = prim["order"]
    k = prim["kernel"]
    if ox <= 0 or oy <= 0 or len(k) != ox * oy:
        return np.zeros_like(img)
    kern = np.array(k).reshape(oy, ox)
    div = prim["divisor"] or kern.sum() or 1.0
    tx, ty = prim["target"]
    mode = {"duplicate": "edge", "wrap": "wrap", "none": "constant"}.get(
        prim["edge"], "edge"
    )
    src = img if not prim["preserve_alpha"] else _unpremul(img)
    p = np.pad(src, ((oy, oy), (ox, ox), (0, 0)), mode=mode)
    h, w = img.shape[:2]
    out = np.zeros_like(img)
    for j in range(oy):
        for i in range(ox):
            # Kernel is applied rotated 180 degrees (true convolution).
            wgt = kern[oy - 1 - j, ox - 1 - i]
            if wgt == 0:
                continue
            ys = oy + j - ty
            xs = ox + i - tx
            out += wgt * p[ys : ys + h, xs : xs + w]
    out = out / div + prim["bias"]
    if prim["preserve_alpha"]:
        out[..., 3] = src[..., 3]
        out = np.clip(out, 0, 1)
        return _premul(out)
    out = np.clip(out, 0, 1)
    out[..., :3] = np.minimum(out[..., :3], out[..., 3:4])
    return out


def _displace(img, mapimg, scale_px, xch, ych):
    um = _unpremul(mapimg)
    chan = {"R": 0, "G": 1, "B": 2, "A": 3}
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    sx = xx + scale_px * (um[..., chan.get(xch, 3)] - 0.5)
    sy = yy + scale_px * (um[..., chan.get(ych, 3)] - 0.5)
    ix = np.clip(np.round(sx).astype(int), 0, w - 1)
    iy = np.clip(np.round(sy).astype(int), 0, h - 1)
    out = img[iy, ix]
    out[(sx < 0) | (sx > w - 1) | (sy < 0) | (sy > h - 1)] = 0
    return out


# Reference Perlin turbulence from the SVG specification, vectorized.
_BSIZE, _BM, _PN, _NM = 0x100, 0xFF, 4096, 0xFFF


def _turb_init(seed):
    rand_m, rand_a, rand_q, rand_r = 2147483647, 16807, 127773, 2836

    def setup(s):
        if s <= 0:
            s = -(s % (rand_m - 1)) + 1
        if s > rand_m - 1:
            s = rand_m - 1
        return s

    def rnd(s):
        s = rand_a * (s % rand_q) - rand_r * (s // rand_q)
        if s <= 0:
            s += rand_m
        return s

    s = setup(int(seed))
    lattice = np.zeros(_BSIZE + _BSIZE + 2, dtype=int)
    grad = np.zeros((4, _BSIZE + _BSIZE + 2, 2))
    for k in range(4):
        for i in range(_BSIZE):
            lattice[i] = i
            for j in range(2):
                s = rnd(s)
                grad[k, i, j] = ((s % (_BSIZE + _BSIZE)) - _BSIZE) / _BSIZE
            n = math.hypot(grad[k, i, 0], grad[k, i, 1]) or 1.0
            grad[k, i] /= n
    for i in range(_BSIZE - 1, 0, -1):
        k = lattice[i]
        s = rnd(s)
        j = s % _BSIZE
        lattice[i] = lattice[j]
        lattice[j] = k
    for i in range(_BSIZE + 2):
        lattice[_BSIZE + i] = lattice[i]
        grad[:, _BSIZE + i] = grad[:, i]
    return lattice, grad


def _noise2(ch, x, y, lattice, grad, stitch):
    t = x + _PN
    bx0 = t.astype(int) & _BM
    bx1 = (bx0 + 1) & _BM
    rx0 = t - t.astype(int)
    rx1 = rx0 - 1.0
    t = y + _PN
    by0 = t.astype(int) & _BM
    by1 = (by0 + 1) & _BM
    ry0 = t - t.astype(int)
    ry1 = ry0 - 1.0
    if stitch is not None:
        wrap_x, wrap_y, width, height = stitch
        bx0 = np.where(bx0 >= wrap_x, bx0 - width, bx0)
        bx1 = np.where(bx1 >= wrap_x, bx1 - width, bx1)
        by0 = np.where(by0 >= wrap_y, by0 - height, by0)
        by1 = np.where(by1 >= wrap_y, by1 - height, by1)
    bx0 &= _BM
    bx1 &= _BM
    by0 &= _BM
    by1 &= _BM
    i = lattice[bx0]
    j = lattice[bx1]
    b00 = lattice[i + by0]
    b10 = lattice[j + by0]
    b01 = lattice[i + by1]
    b11 = lattice[j + by1]
    sx = rx0 * rx0 * (3.0 - 2.0 * rx0)
    sy = ry0 * ry0 * (3.0 - 2.0 * ry0)
    g = grad[ch]
    u = rx0 * g[b00, 0] + ry0 * g[b00, 1]
    v = rx1 * g[b10, 0] + ry0 * g[b10, 1]
    a = u + sx * (v - u)
    u = rx0 * g[b01, 0] + ry1 * g[b01, 1]
    v = rx1 * g[b11, 0] + ry1 * g[b11, 1]
    b = u + sx * (v - u)
    return a + sy * (b - a)


def _turbulence(prim, shape, region, ppx, ppy):
    h, w = shape
    lattice, grad = _turb_init(prim["seed"])
    x0, y0 = region[0], region[1]
    ys, xs = np.mgrid[0:h, 0:w].astype(float)
    # Pixel centres in document space, then back to the element's user space.
    dx = x0 + (xs + 0.5) / ppx
    dy = y0 + (ys + 0.5) / ppy
    try:
        inv = prim["origin_doc"].inverted()
        pts = inv.transform(np.stack([dx.ravel(), dy.ravel()], axis=1))
        ux, uy = pts[:, 0].reshape(h, w), pts[:, 1].reshape(h, w)
    except Exception:
        ux, uy = dx, dy
    kx, ky = _scales(prim["origin_doc"])
    fx, fy = prim["freq"][0] * kx, prim["freq"][1] * ky
    out = np.zeros((h, w, 4))
    for ch in range(4):
        total = np.zeros((h, w))
        ratio = 1.0
        vx, vy = ux * fx, uy * fy
        for _ in range(max(1, prim["octaves"])):
            n = _noise2(ch, vx, vy, lattice, grad, None)
            total += (n if prim["fractal"] else np.abs(n)) / ratio
            vx, vy = vx * 2, vy * 2
            ratio *= 2
        out[..., ch] = (total + 1) / 2 if prim["fractal"] else total
    out = np.clip(out, 0, 1)
    return _premul(out)


def _lighting(img, prim, ppx, ppy, region, specular):
    alpha = img[..., 3]
    ss = prim["surface_scale"]
    # Surface normals from the alpha channel (Sobel, as in the spec).
    pad = np.pad(alpha, 1, mode="edge")
    nx = (
        -ss
        / 4
        * (
            (pad[:-2, 2:] + 2 * pad[1:-1, 2:] + pad[2:, 2:])
            - (pad[:-2, :-2] + 2 * pad[1:-1, :-2] + pad[2:, :-2])
        )
    )
    ny = (
        -ss
        / 4
        * (
            (pad[2:, :-2] + 2 * pad[2:, 1:-1] + pad[2:, 2:])
            - (pad[:-2, :-2] + 2 * pad[:-2, 1:-1] + pad[:-2, 2:])
        )
    )
    # Slopes per document unit (kernelUnitLength ~ one user unit), so the
    # result does not depend on the off-screen resolution.
    nx, ny = nx * ppx, ny * ppy
    norm = np.sqrt(nx * nx + ny * ny + 1)
    n = np.stack([nx / norm, ny / norm, 1 / norm], axis=-1)
    light = prim["light"]
    color = np.array(prim["color"], float)
    h, w = alpha.shape
    if light["type"] == "distant":
        az, el = math.radians(light["azimuth"]), math.radians(light["elevation"])
        lv = np.array(
            [math.cos(az) * math.cos(el), math.sin(az) * math.cos(el), math.sin(el)]
        )
        L = np.broadcast_to(lv, (h, w, 3))
        lc = np.broadcast_to(color, (h, w, 3))
    else:
        ys, xs = np.mgrid[0:h, 0:w].astype(float)
        px = (region[0] + (xs + 0.5) / ppx) * ppx
        py = (region[1] + (ys + 0.5) / ppy) * ppy
        lx, ly, lz = light["pos"]
        lx, ly, lz = lx * ppx, ly * ppy, lz * ppx
        L = np.stack([lx - px, ly - py, lz - ss * alpha], axis=-1)
        L /= np.linalg.norm(L, axis=-1, keepdims=True) + 1e-12
        lc = np.broadcast_to(color, (h, w, 3)).copy()
        if light["type"] == "spot":
            ax_, ay_, az_ = light["at"]
            S = np.array([ax_ * ppx - lx, ay_ * ppy - ly, az_ * ppx - lz])
            S /= np.linalg.norm(S) + 1e-12
            minus_l_dot_s = -(L @ S)
            fac = np.power(np.clip(minus_l_dot_s, 0, None), light["exp"])
            if light["cone"] is not None:
                fac = np.where(
                    minus_l_dot_s < math.cos(math.radians(light["cone"])), 0.0, fac
                )
            lc = lc * fac[..., None]
    if specular:
        H = L + np.array([0, 0, 1.0])
        H /= np.linalg.norm(H, axis=-1, keepdims=True) + 1e-12
        f = prim["ks"] * np.power(np.clip((n * H).sum(-1), 0, None), prim["exp"])
        rgb = np.clip(f[..., None] * lc, 0, 1)
        a = rgb.max(axis=-1, keepdims=True)
        return np.concatenate([rgb, a], axis=-1)
    f = prim["kd"] * np.clip((n * L).sum(-1), 0, None)
    rgb = np.clip(f[..., None] * lc, 0, 1)
    return np.concatenate([rgb, np.ones((h, w, 1))], axis=-1)


def _subregion_mask(prim, region, shape, ppx, ppy):
    sub = prim.get("subregion")
    if sub is None:
        return None
    h, w = shape
    x0 = int(round((sub[0] - region[0]) * ppx))
    x1 = int(round((sub[2] - region[0]) * ppx))
    y0 = int(round((sub[1] - region[1]) * ppy))
    y1 = int(round((sub[3] - region[1]) * ppy))
    m = np.zeros((h, w, 1))
    m[max(y0, 0) : max(min(y1, h), 0), max(x0, 0) : max(min(x1, w), 0)] = 1.0
    return m, (max(x0, 0), max(y0, 0), min(x1, w), min(y1, h))


def apply_filter(
    flt: Filter, source: np.ndarray, region, ppx, ppy, rasterize
) -> np.ndarray:
    """Run a filter on a straight-alpha sRGB image; returns straight sRGB RGBA.

    ``rasterize(items, region, shape)`` renders extra items (for ``feImage``).
    """
    linear = flt.color_space != "sRGB"
    src = source.astype(float).copy()
    if linear:
        src[..., :3] = _to_linear(src[..., :3])
    src = _premul(src)
    shape = src.shape[:2]
    zeros = np.zeros_like(src)
    results: dict = {}
    sub_boxes: dict = {}
    last = src
    last_box = None

    def inp(name):
        if name in (None, ""):
            return last
        if name == "SourceGraphic":
            return src
        if name == "SourceAlpha":
            out = np.zeros_like(src)
            out[..., 3] = src[..., 3]
            return out
        if name in results:
            return results[name]
        return (
            zeros
            if name
            in ("BackgroundImage", "BackgroundAlpha", "FillPaint", "StrokePaint")
            else last
        )

    for prim in flt.primitives:
        t = prim["type"]
        a = inp(prim.get("in"))
        if t == "feGaussianBlur":
            sx, sy = prim["std"]
            out = (
                gaussian_blur(a, sx * ppx, sy * ppy) if (sx > 0 or sy > 0) else a.copy()
            )
            if sx < 0 or sy < 0:
                out = zeros
        elif t == "feOffset":
            dx, dy = prim["offset"]
            out = _shift(a, dx * ppx, dy * ppy)
        elif t == "feFlood":
            c = np.array(prim["color"], float)
            if linear:
                c[:3] = _to_linear(c[:3])
            c[:3] *= c[3]
            out = np.broadcast_to(c, src.shape).copy()
        elif t == "feDropShadow":
            sx, sy = prim["std"]
            alpha = np.zeros_like(a)
            alpha[..., 3] = a[..., 3]
            blur = gaussian_blur(alpha, sx * ppx, sy * ppy)
            blur = _shift(blur, prim["offset"][0] * ppx, prim["offset"][1] * ppy)
            c = np.array(prim["color"], float)
            if linear:
                c[:3] = _to_linear(c[:3])
            shadow = np.zeros_like(a)
            shadow[..., 3] = blur[..., 3] * c[3]
            shadow[..., :3] = c[:3] * shadow[..., 3:4]
            out = a + shadow * (1 - a[..., 3:4])
        elif t == "feColorMatrix":
            out = _premul(_color_matrix(_unpremul(a), prim["mtype"], prim["values"]))
        elif t == "feComponentTransfer":
            u = _unpremul(a)
            for i, ch in enumerate("RGBA"):
                f = prim["funcs"].get(ch)
                if f:
                    u[..., i] = np.clip(_transfer(u[..., i], f), 0, 1)
            out = _premul(u)
        elif t == "feComposite":
            out = _composite(a, inp(prim.get("in2")), prim["operator"], prim["k"])
        elif t == "feBlend":
            out = _blend(a, inp(prim.get("in2")), prim["mode"])
        elif t == "feMerge":
            out = np.zeros_like(src)
            for node in prim["nodes"] or [None]:
                layer = inp(node)
                out = layer + out * (1 - layer[..., 3:4])
        elif t == "feMorphology":
            rx, ry = prim["radius"]
            if rx < 0 or ry < 0:
                out = zeros
            else:
                out = _morphology(a, prim["operator"], rx * ppx, ry * ppy)
        elif t == "feConvolveMatrix":
            out = _convolve(a, prim)
        elif t == "feDisplacementMap":
            out = _displace(
                a, inp(prim.get("in2")), prim["scale"] * ppx, prim["xch"], prim["ych"]
            )
        elif t == "feTurbulence":
            out = _turbulence(prim, shape, region, ppx, ppy)
        elif t in ("feDiffuseLighting", "feSpecularLighting"):
            p2 = dict(prim)
            if linear:  # only the light color changes space; results do not
                p2["color"] = tuple(_to_linear(np.array(prim["color"], float)))
            lit = _lighting(a, p2, ppx, ppy, region, t == "feSpecularLighting")
            out = lit  # diffuse is opaque; specular already has A = max(RGB)
        elif t == "feTile":
            box = sub_boxes.get(prim.get("in")) if prim.get("in") else last_box
            out = _tile(a, box, shape)
        elif t == "feImage":
            out = _fe_image(flt, prim, region, shape, ppx, ppy, rasterize, linear)
        else:
            out = a.copy()
        sm = _subregion_mask(prim, region, shape, ppx, ppy)
        box = None
        if sm is not None:
            out = out * sm[0]
            box = sm[1]
        if prim.get("result"):
            results[prim["result"]] = out
            sub_boxes[prim["result"]] = box
        last, last_box = out, box
    out = _unpremul(np.clip(last, 0, 1))
    if linear:
        out[..., :3] = _to_srgb(out[..., :3])
    return np.clip(out, 0, 1)


def _tile(img, box, shape):
    if box is None:
        return img.copy()
    x0, y0, x1, y1 = box
    if x1 <= x0 or y1 <= y0:
        return np.zeros_like(img)
    tile = img[y0:y1, x0:x1]
    h, w = shape
    th, tw = tile.shape[:2]
    reps = (math.ceil(h / th) + 2, math.ceil(w / tw) + 2, 1)
    big = np.tile(tile, reps)
    oy, ox = y0 % th, x0 % tw
    return big[th - oy : th - oy + h, tw - ox : tw - ox + w].copy()


def _fe_image(flt, prim, region, shape, ppx, ppy, rasterize, linear):
    entry = flt.images.get(prim.get("image_index"))
    h, w = shape
    if entry is None:
        return np.zeros((h, w, 4))
    if entry[0] == "items":
        img = rasterize(entry[1], region, shape)
    else:
        data, sub = entry[1], entry[2]
        img = np.zeros((h, w, 4))
        x0 = int(round((sub[0] - region[0]) * ppx))
        y0 = int(round((sub[1] - region[1]) * ppy))
        x1 = int(round((sub[2] - region[0]) * ppx))
        y1 = int(round((sub[3] - region[1]) * ppy))
        if x1 > x0 and y1 > y0:
            from PIL import Image as PILImage

            pil = PILImage.fromarray(np.asarray(data, dtype=np.uint8))
            pil = pil.resize((x1 - x0, y1 - y0))
            arr = np.asarray(pil, dtype=float) / 255.0
            ys, xs = slice(max(y0, 0), min(y1, h)), slice(max(x0, 0), min(x1, w))
            img[ys, xs] = arr[
                ys.start - y0 : ys.stop - y0, xs.start - x0 : xs.stop - x0
            ]
    img = np.asarray(img, dtype=float)
    if linear:
        img[..., :3] = _to_linear(img[..., :3])
    return _premul(img)
