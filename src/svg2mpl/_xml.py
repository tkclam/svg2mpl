"""Small helpers for working with SVG elements from :mod:`xml.etree`."""

from __future__ import annotations

from xml.etree.ElementTree import Element

__all__ = ["SVG_NS", "XLINK_HREF", "href", "is_svg_element", "strip_ns"]

SVG_NS = "http://www.w3.org/2000/svg"
XLINK_HREF = "{http://www.w3.org/1999/xlink}href"


def strip_ns(tag: str) -> str:
    """Drop an XML namespace from a tag, e.g. ``"{...svg}rect"`` -> ``"rect"``."""
    return tag.rsplit("}", 1)[-1]


def is_svg_element(el: Element) -> bool:
    """Whether ``el`` is an element (not a comment) in the SVG namespace.

    Elements without a namespace count as SVG, so documents that omit
    ``xmlns`` still render; foreign elements (Inkscape metadata ...) do not.
    """
    if not isinstance(el.tag, str):
        return False
    if el.tag.startswith("{"):
        return el.tag[1:].split("}", 1)[0] == SVG_NS
    return True


def href(el: Element) -> str | None:
    """The element's ``href`` (SVG 2) or ``xlink:href`` (SVG 1.1) reference."""
    return el.get("href") or el.get(XLINK_HREF)
