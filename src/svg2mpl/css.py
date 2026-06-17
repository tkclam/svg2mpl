"""Resolve author ``<style>`` blocks against the SVG tree (optional feature).

Requires the ``css`` extra (``tinycss2`` + ``cssselect2``). When those packages
are not installed this module quietly returns no declarations, so presentation
attributes and inline ``style="..."`` continue to work on their own.
"""

from __future__ import annotations

import warnings
from typing import Any

from .shapes import strip_ns

__all__ = ["compute_css_declarations", "css_available"]


def css_available() -> bool:
    """Return whether the optional CSS dependencies are importable."""
    try:
        import cssselect2  # noqa: F401
        import tinycss2  # noqa: F401
    except ImportError:
        return False
    return True


def _parse_declarations(content) -> dict[str, str]:
    import tinycss2

    out: dict[str, str] = {}
    for decl in tinycss2.parse_declaration_list(
        content, skip_comments=True, skip_whitespace=True
    ):
        if decl.type != "declaration":
            continue
        out[decl.lower_name] = tinycss2.serialize(decl.value).strip()
    return out


def compute_css_declarations(root) -> dict[int, dict[str, Any]]:
    """Map each element (by ``id()``) to its CSS-resolved declarations.

    Declarations from all ``<style>`` elements are matched against every element
    and merged by ascending specificity then source order, so the winning
    declaration is applied last.

    Returns an empty mapping when there are no ``<style>`` blocks or the optional
    dependencies are missing (a one-time warning is emitted in the latter case).
    """
    style_texts = [
        (e.text or "") for e in root.iter() if strip_ns(e.tag) == "style" and e.text
    ]
    if not style_texts:
        return {}
    if not css_available():
        warnings.warn(
            "This SVG has <style> blocks but the optional CSS dependencies "
            "(tinycss2, cssselect2) are not installed; install svg2mpl[css] to "
            "honor them. Presentation attributes and inline styles still apply.",
            stacklevel=2,
        )
        return {}

    import cssselect2
    import tinycss2

    matcher = cssselect2.Matcher()
    for css in style_texts:
        for rule in tinycss2.parse_stylesheet(
            css, skip_comments=True, skip_whitespace=True
        ):
            if rule.type != "qualified-rule":
                continue
            try:
                selectors = cssselect2.compile_selector_list(rule.prelude)
            except cssselect2.SelectorError:
                continue
            decls = _parse_declarations(rule.content)
            for selector in selectors:
                matcher.add_selector(selector, decls)

    result: dict[int, dict[str, Any]] = {}
    wrapper = cssselect2.ElementWrapper.from_xml_root(root)
    for element in wrapper.iter_subtree():
        matches = matcher.match(element)
        if not matches:
            continue
        # match() yields (specificity, order, pseudo, payload); apply lowest
        # priority first so higher-specificity / later rules win the dict merge.
        matches.sort(key=lambda m: (m[0], m[1]))
        merged: dict[str, Any] = {}
        for *_unused, decls in matches:
            merged.update(decls)
        result[id(element.etree_element)] = merged
    return result
