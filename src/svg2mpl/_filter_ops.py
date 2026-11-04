"""numpy implementations of the SVG filter primitives.

Images are ``(h, w, 4)`` float arrays with row 0 at the top. Primitives work on
*premultiplied* RGBA in the filter's working color space (``linearRGB`` by
default); :func:`apply_filter` converts from and back to straight-alpha sRGB.
Lengths arrive in document units and are scaled by the pixel densities
``ppx``/``ppy`` (pixels per document unit).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from PIL import Image as PILImage

from .geometry import Rect, axis_scales

__all__ = ["apply_filter", "gaussian_blur", "to_linear", "to_srgb"]

Image = np.ndarray
Rasterize = Callable[[list, Rect, tuple[int, int]], Image]


def to_linear(c):
    """sRGB -> linearRGB (per channel, values in ``[0, 1]``)."""
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def to_srgb(c):
    """linearRGB -> sRGB (per channel, values in ``[0, 1]``)."""
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
        r = max(1, math.ceil(sigma * 3))
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
    d = math.floor(sigma * 3 * math.sqrt(2 * math.pi) / 4 + 0.5)
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
    ix, iy = math.floor(dx_px), math.floor(dy_px)
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
    for axis, r in ((1, round(rx)), (0, round(ry))):
        if r <= 0:
            continue
        pad = [(0, 0)] * img.ndim
        pad[axis] = (r, r)
        p = np.pad(out, pad)  # outside the image is transparent black
        n = out.shape[axis]
        acc = np.take(p, np.arange(0, n), axis=axis)
        for i in range(1, 2 * r + 1):
            acc = fn(acc, np.take(p, np.arange(i, i + n), axis=axis))
        out = acc
    return out


_EDGE_MODES: dict[str, Literal["edge", "wrap", "constant"]] = {
    "duplicate": "edge",
    "wrap": "wrap",
    "none": "constant",
}


def _convolve(img, prim):
    ox, oy = prim["order"]
    k = prim["kernel"]
    if ox <= 0 or oy <= 0 or len(k) != ox * oy:
        return np.zeros_like(img)
    kern = np.array(k).reshape(oy, ox)
    div = prim["divisor"] or kern.sum() or 1.0
    tx, ty = prim["target"]
    mode = _EDGE_MODES.get(prim["edge"], "edge")
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
# (``stitchTiles`` is not implemented.)
_BSIZE, _BM, _PN = 0x100, 0xFF, 4096


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


def _noise2(ch, x, y, lattice, grad):
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
    except np.linalg.LinAlgError:
        ux, uy = dx, dy
    else:
        pts = inv.transform(np.stack([dx.ravel(), dy.ravel()], axis=1))
        ux, uy = pts[:, 0].reshape(h, w), pts[:, 1].reshape(h, w)
    kx, ky = axis_scales(prim["origin_doc"])
    fx, fy = prim["freq"][0] * kx, prim["freq"][1] * ky
    out = np.zeros((h, w, 4))
    for ch in range(4):
        total = np.zeros((h, w))
        ratio = 1.0
        vx, vy = ux * fx, uy * fy
        for _ in range(max(1, prim["octaves"])):
            n = _noise2(ch, vx, vy, lattice, grad)
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
        to_light = np.broadcast_to(lv, (h, w, 3))
        lc = np.broadcast_to(color, (h, w, 3))
    else:
        ys, xs = np.mgrid[0:h, 0:w].astype(float)
        px = (region[0] + (xs + 0.5) / ppx) * ppx
        py = (region[1] + (ys + 0.5) / ppy) * ppy
        lx, ly, lz = light["pos"]
        lx, ly, lz = lx * ppx, ly * ppy, lz * ppx
        to_light = np.stack([lx - px, ly - py, lz - ss * alpha], axis=-1)
        to_light /= np.linalg.norm(to_light, axis=-1, keepdims=True) + 1e-12
        lc = np.broadcast_to(color, (h, w, 3)).copy()
        if light["type"] == "spot":
            ax_, ay_, az_ = light["at"]
            spot_dir = np.array([ax_ * ppx - lx, ay_ * ppy - ly, az_ * ppx - lz])
            spot_dir /= np.linalg.norm(spot_dir) + 1e-12
            minus_l_dot_s = -(to_light @ spot_dir)
            fac = np.power(np.clip(minus_l_dot_s, 0, None), light["exp"])
            if light["cone"] is not None:
                fac = np.where(
                    minus_l_dot_s < math.cos(math.radians(light["cone"])), 0.0, fac
                )
            lc = lc * fac[..., None]
    if specular:
        halfway = to_light + np.array([0, 0, 1.0])
        halfway /= np.linalg.norm(halfway, axis=-1, keepdims=True) + 1e-12
        f = prim["ks"] * np.power(np.clip((n * halfway).sum(-1), 0, None), prim["exp"])
        rgb = np.clip(f[..., None] * lc, 0, 1)
        a = rgb.max(axis=-1, keepdims=True)
        return np.concatenate([rgb, a], axis=-1)
    f = prim["kd"] * np.clip((n * to_light).sum(-1), 0, None)
    rgb = np.clip(f[..., None] * lc, 0, 1)
    return np.concatenate([rgb, np.ones((h, w, 1))], axis=-1)


def _subregion_mask(prim, region, shape, ppx, ppy):
    sub = prim.get("subregion")
    if sub is None:
        return None
    h, w = shape
    x0 = round((sub[0] - region[0]) * ppx)
    x1 = round((sub[2] - region[0]) * ppx)
    y0 = round((sub[1] - region[1]) * ppy)
    y1 = round((sub[3] - region[1]) * ppy)
    m = np.zeros((h, w, 1))
    m[max(y0, 0) : max(min(y1, h), 0), max(x0, 0) : max(min(x1, w), 0)] = 1.0
    return m, (max(x0, 0), max(y0, 0), min(x1, w), min(y1, h))


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


def _fe_image(images, prim, region, shape, ppx, ppy, rasterize, linear):
    entry = images.get(prim.get("image_index"))
    h, w = shape
    if entry is None:
        return np.zeros((h, w, 4))
    if entry[0] == "items":
        img = rasterize(entry[1], region, shape)
    else:
        data, sub = entry[1], entry[2]
        img = np.zeros((h, w, 4))
        x0 = round((sub[0] - region[0]) * ppx)
        y0 = round((sub[1] - region[1]) * ppy)
        x1 = round((sub[2] - region[0]) * ppx)
        y1 = round((sub[3] - region[1]) * ppy)
        if x1 > x0 and y1 > y0:
            pil = PILImage.fromarray(np.asarray(data, dtype=np.uint8))
            pil = pil.resize((x1 - x0, y1 - y0))
            arr = np.asarray(pil, dtype=float) / 255.0
            ys, xs = slice(max(y0, 0), min(y1, h)), slice(max(x0, 0), min(x1, w))
            img[ys, xs] = arr[
                ys.start - y0 : ys.stop - y0, xs.start - x0 : xs.stop - x0
            ]
    img = np.asarray(img, dtype=float)
    if linear:
        img[..., :3] = to_linear(img[..., :3])
    return _premul(img)


# --------------------------------------------------------------------------- #
# Primitive dispatch
# --------------------------------------------------------------------------- #
_TRANSPARENT_INPUTS = ("BackgroundImage", "BackgroundAlpha", "FillPaint", "StrokePaint")


@dataclass
class _Env:
    """What a primitive can read: inputs, results and the pixel grid."""

    src: Image
    linear: bool
    region: Rect
    ppx: float
    ppy: float
    rasterize: Rasterize
    images: dict
    results: dict = field(default_factory=dict)
    boxes: dict = field(default_factory=dict)
    last: Image | None = None
    last_box: tuple | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return self.src.shape[0], self.src.shape[1]

    def input(self, name: str | None) -> Image:
        """Resolve an ``in``/``in2`` reference."""
        if not name:
            return self.src if self.last is None else self.last
        if name == "SourceGraphic":
            return self.src
        if name == "SourceAlpha":
            out = np.zeros_like(self.src)
            out[..., 3] = self.src[..., 3]
            return out
        if name in self.results:
            return self.results[name]
        if name in _TRANSPARENT_INPUTS:
            return np.zeros_like(self.src)
        return self.input(None)  # unknown reference: the previous result

    def color(self, rgba) -> np.ndarray:
        """A straight sRGB color in the working space, premultiplied."""
        c = np.array(rgba, float)
        if self.linear:
            c[:3] = to_linear(c[:3])
        c[:3] *= c[3]
        return c


def _op_blur(prim, a, env):
    sx, sy = prim["std"]
    if sx < 0 or sy < 0:
        return np.zeros_like(a)
    if sx == 0 and sy == 0:
        return a.copy()
    return gaussian_blur(a, sx * env.ppx, sy * env.ppy)


def _op_offset(prim, a, env):
    dx, dy = prim["offset"]
    return _shift(a, dx * env.ppx, dy * env.ppy)


def _op_flood(prim, _a, env):
    return np.broadcast_to(env.color(prim["color"]), env.src.shape).copy()


def _op_drop_shadow(prim, a, env):
    alpha = np.zeros_like(a)
    alpha[..., 3] = a[..., 3]
    sx, sy = prim["std"]
    blur = gaussian_blur(alpha, sx * env.ppx, sy * env.ppy)
    blur = _shift(blur, prim["offset"][0] * env.ppx, prim["offset"][1] * env.ppy)
    c = env.color(prim["color"])
    shadow = blur[..., 3:4] * c  # c is premultiplied
    return a + shadow * (1 - a[..., 3:4])


def _op_color_matrix(prim, a, _env):
    return _premul(_color_matrix(_unpremul(a), prim["mtype"], prim["values"]))


def _op_component_transfer(prim, a, _env):
    u = _unpremul(a)
    for i, ch in enumerate("RGBA"):
        f = prim["funcs"].get(ch)
        if f:
            u[..., i] = np.clip(_transfer(u[..., i], f), 0, 1)
    return _premul(u)


def _op_composite(prim, a, env):
    return _composite(a, env.input(prim.get("in2")), prim["operator"], prim["k"])


def _op_blend(prim, a, env):
    return _blend(a, env.input(prim.get("in2")), prim["mode"])


def _op_merge(prim, _a, env):
    out = np.zeros_like(env.src)
    for node in prim["nodes"] or [None]:
        layer = env.input(node)
        out = layer + out * (1 - layer[..., 3:4])
    return out


def _op_morphology(prim, a, env):
    rx, ry = prim["radius"]
    if rx < 0 or ry < 0:
        return np.zeros_like(a)
    return _morphology(a, prim["operator"], rx * env.ppx, ry * env.ppy)


def _op_convolve(prim, a, _env):
    return _convolve(a, prim)


def _op_displacement(prim, a, env):
    mapimg = env.input(prim.get("in2"))
    return _displace(a, mapimg, prim["scale"] * env.ppx, prim["xch"], prim["ych"])


def _op_turbulence(prim, _a, env):
    return _turbulence(prim, env.shape, env.region, env.ppx, env.ppy)


def _op_lighting(prim, a, env):
    lit_prim = dict(prim)
    if env.linear:  # only the light color changes space; the result does not
        lit_prim["color"] = tuple(to_linear(np.array(prim["color"], float)))
    specular = prim["type"] == "feSpecularLighting"
    # Diffuse light is opaque; specular already has A = max(R, G, B), so both
    # are valid premultiplied images.
    return _lighting(a, lit_prim, env.ppx, env.ppy, env.region, specular)


def _op_tile(prim, a, env):
    box = env.boxes.get(prim.get("in")) if prim.get("in") else env.last_box
    return _tile(a, box, env.shape)


def _op_image(prim, _a, env):
    return _fe_image(
        env.images,
        prim,
        env.region,
        env.shape,
        env.ppx,
        env.ppy,
        env.rasterize,
        env.linear,
    )


_OPS: dict[str, Callable[[dict, Image, _Env], Image]] = {
    "feBlend": _op_blend,
    "feColorMatrix": _op_color_matrix,
    "feComponentTransfer": _op_component_transfer,
    "feComposite": _op_composite,
    "feConvolveMatrix": _op_convolve,
    "feDiffuseLighting": _op_lighting,
    "feDisplacementMap": _op_displacement,
    "feDropShadow": _op_drop_shadow,
    "feFlood": _op_flood,
    "feGaussianBlur": _op_blur,
    "feImage": _op_image,
    "feMerge": _op_merge,
    "feMorphology": _op_morphology,
    "feOffset": _op_offset,
    "feSpecularLighting": _op_lighting,
    "feTile": _op_tile,
    "feTurbulence": _op_turbulence,
}
SUPPORTED_PRIMITIVES = frozenset(_OPS)


def apply_filter(
    primitives: list[dict],
    source: Image,
    region: Rect,
    ppx: float,
    ppy: float,
    rasterize: Rasterize,
    *,
    color_space: str = "linearRGB",
    images: dict | None = None,
) -> Image:
    """Run a primitive chain on a straight-alpha sRGB image.

    Returns straight-alpha sRGB RGBA. ``rasterize(items, region, shape)``
    renders the elements an ``feImage`` references (``images`` maps primitive
    index to ``("items", items)`` or ``("raster", rgba, subregion)``).
    Unknown primitives pass their input through.
    """
    linear = color_space != "sRGB"
    src = source.astype(float).copy()
    if linear:
        src[..., :3] = to_linear(src[..., :3])
    env = _Env(
        src=_premul(src),
        linear=linear,
        region=region,
        ppx=ppx,
        ppy=ppy,
        rasterize=rasterize,
        images=images or {},
    )
    for prim in primitives:
        a = env.input(prim.get("in"))
        op = _OPS.get(prim["type"])
        out = op(prim, a, env) if op is not None else a.copy()
        box = None
        sub = _subregion_mask(prim, region, env.shape, ppx, ppy)
        if sub is not None:
            out = out * sub[0]
            box = sub[1]
        if prim.get("result"):
            env.results[prim["result"]] = out
            env.boxes[prim["result"]] = box
        env.last, env.last_box = out, box
    final = env.src if env.last is None else env.last
    out = _unpremul(np.clip(final, 0, 1))
    if linear:
        out[..., :3] = to_srgb(out[..., :3])
    return np.clip(out, 0, 1)
