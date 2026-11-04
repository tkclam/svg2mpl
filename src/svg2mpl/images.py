"""Load the resources referenced by ``<image>`` (and ``feImage``).

References may be ``data:`` URIs or local file paths (absolute, ``file://`` or
relative to the SVG file). Remote URLs are never fetched. Raster formats are
decoded with Pillow; SVG documents are returned as XML trees so the caller can
render them as vectors.
"""

from __future__ import annotations

import base64
import gzip
import io
import re
import urllib.parse
from pathlib import Path as FilePath
from xml.etree import ElementTree
from xml.etree.ElementTree import Element

import numpy as np

__all__ = ["ImageError", "auto_size", "load_image", "read_href"]


class ImageError(ValueError):
    """Raised when an image reference cannot be read or decoded."""


def read_href(href: str, base_dir: str | None) -> tuple[bytes, str]:
    """Return ``(bytes, mime_type)`` for a ``data:`` URI or a local file."""
    href = href.strip()
    if href.startswith("data:"):
        header, _, payload = href[5:].partition(",")
        mime = header.split(";")[0].strip().lower()
        if ";base64" in header.lower():
            payload = re.sub(r"\s+", "", payload)
            return base64.b64decode(payload + "=" * (-len(payload) % 4)), mime
        return urllib.parse.unquote_to_bytes(payload), mime
    parsed = urllib.parse.urlparse(href)
    if parsed.scheme in ("http", "https", "ftp"):
        raise ImageError("remote images are not fetched")
    if parsed.scheme == "file":
        path = FilePath(urllib.parse.unquote(parsed.path))
    else:
        path = FilePath(urllib.parse.unquote(href.split("#")[0]))
        if not path.is_absolute():
            if base_dir is None:
                raise ImageError("relative image path without a base directory")
            path = FilePath(base_dir) / path
    data = path.read_bytes()
    suffix = path.suffix.lower()
    if suffix == ".svgz":
        data = gzip.decompress(data)
    return data, "image/svg+xml" if suffix in (".svg", ".svgz") else ""


def _looks_like_svg(data: bytes, mime: str) -> bool:
    return (
        "svg" in mime
        or data.lstrip()[:5] in (b"<?xml", b"<svg ")
        or b"<svg" in data[:512]
    )


def load_image(
    href: str, base_dir: str | None
) -> tuple[str, np.ndarray] | tuple[str, Element]:
    """Decode an image reference.

    Returns ``("raster", rgba_uint8_array)`` or ``("svg", xml_root)``; raises
    :class:`ImageError` (or :class:`OSError`) when it cannot be loaded.
    """
    from PIL import Image as PILImage
    from PIL import UnidentifiedImageError

    data, mime = read_href(href, base_dir)
    if _looks_like_svg(data, mime):
        try:
            return "svg", ElementTree.fromstring(data)  # noqa: S314 - local input
        except ElementTree.ParseError:
            pass
    try:
        img = PILImage.open(io.BytesIO(data)).convert("RGBA")
    except (UnidentifiedImageError, OSError) as exc:
        raise ImageError(f"could not decode image: {exc}") from exc
    return "raster", np.asarray(img)


def auto_size(
    w: float | None, h: float | None, iw: float, ih: float
) -> tuple[float, float]:
    """Fill in a missing ``width``/``height`` from the intrinsic aspect ratio."""
    if w is None:
        if h is None:
            return float(iw), float(ih)
        return (h * iw / ih if ih else h), h
    if h is None:
        return w, (w * ih / iw if iw else w)
    return w, h
