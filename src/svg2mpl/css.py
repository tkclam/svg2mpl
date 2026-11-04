"""Resolve author ``<style>`` blocks against the SVG tree.

With the ``css`` extra (``tinycss2`` + ``cssselect2``) the full selector grammar
is supported. Without it a built-in matcher handles the selectors that SVG
exporters actually emit: type, ``.class``, ``#id``, ``*``, attribute selectors
(``[a]``, ``[a=v]``, ``[a~=v]``, ``[a^=v]``, ``[a$=v]``, ``[a*=v]``), compound
selectors and the descendant/child combinators, with proper specificity.
Selectors it cannot parse are skipped.

``!important`` is preserved on the returned values (as a ``!important``
suffix) so :func:`svg2mpl.style.declarations` can order it correctly.
"""

from __future__ import annotations

import re
from typing import Any

from ._xml import strip_ns

__all__ = ["compute_css_declarations", "css_available"]


def css_available() -> bool:
    """Return whether the optional CSS dependencies are importable."""
    try:
        import cssselect2  # noqa: F401
        import tinycss2  # noqa: F401
    except ImportError:
        return False
    return True


def _style_texts(root) -> list[str]:
    out = []
    for e in root.iter():
        if not isinstance(e.tag, str) or strip_ns(e.tag) != "style":
            continue
        kind = (e.get("type") or "text/css").strip().lower()
        if kind not in ("text/css", ""):
            continue
        text = "".join(e.itertext())
        if text.strip():
            out.append(text)
    return out


def compute_css_declarations(root) -> dict[int, dict[str, Any]]:
    """Map each element (by ``id()``) to its CSS-resolved declarations.

    Declarations from all ``<style>`` elements are matched against every element
    and merged by ascending specificity then source order, so the winning
    declaration is applied last.
    """
    texts = _style_texts(root)
    if not texts:
        return {}
    if css_available():
        return _compute_cssselect2(root, texts)
    return _compute_builtin(root, texts)


# --------------------------------------------------------------------------- #
# tinycss2 / cssselect2 backend
# --------------------------------------------------------------------------- #
def _parse_declarations(content) -> dict[str, str]:
    import tinycss2

    out: dict[str, str] = {}
    for decl in tinycss2.parse_declaration_list(
        content, skip_comments=True, skip_whitespace=True
    ):
        if decl.type != "declaration":
            continue
        value = tinycss2.serialize(decl.value).strip()
        if decl.important:
            value += " !important"
        out[decl.lower_name] = value
    return out


def _media_applies(prelude_text: str) -> bool:
    q = prelude_text.strip().lower()
    return "print" not in q or "screen" in q


def _compute_cssselect2(root, texts) -> dict[int, dict[str, Any]]:
    import cssselect2
    import tinycss2

    matcher = cssselect2.Matcher()

    def add_rules(rules):
        for rule in rules:
            if rule.type == "at-rule" and rule.lower_at_keyword == "media":
                if rule.content is not None and _media_applies(
                    tinycss2.serialize(rule.prelude)
                ):
                    add_rules(
                        tinycss2.parse_rule_list(
                            rule.content, skip_comments=True, skip_whitespace=True
                        )
                    )
                continue
            if rule.type != "qualified-rule":
                continue
            try:
                selectors = cssselect2.compile_selector_list(rule.prelude)
            except cssselect2.SelectorError:
                continue
            decls = _parse_declarations(rule.content)
            for selector in selectors:
                matcher.add_selector(selector, decls)

    for css in texts:
        add_rules(
            tinycss2.parse_stylesheet(css, skip_comments=True, skip_whitespace=True)
        )

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


# --------------------------------------------------------------------------- #
# Built-in fallback backend
# --------------------------------------------------------------------------- #
_COMPOUND_RE = re.compile(
    r"(?P<type>\*|[a-zA-Z_][\w-]*(?:\|[\w-]+)?)?"
    r"(?P<rest>(?:#[\w-]+|\.[\w-]+|\[[^\]]+\]|:[\w-]+(?:\([^)]*\))?)*)$"
)
_PART_RE = re.compile(r"#[\w-]+|\.[\w-]+|\[[^\]]+\]|:[\w-]+(?:\([^)]*\))?")
_ATTR_RE = re.compile(
    r"\[\s*([\w:-]+)\s*(?:([~^$*|]?=)\s*(\"[^\"]*\"|'[^']*'|[^\]\s]+)\s*(i)?\s*)?\]"
)


class _Compound:
    def __init__(self, text: str):
        m = _COMPOUND_RE.match(text)
        if not m or not text:
            raise ValueError(text)
        t = m.group("type")
        self.type = None if t in (None, "*") else t.split("|")[-1]
        self.ids, self.classes, self.attrs, self.pseudos = [], [], [], []
        for part in _PART_RE.findall(m.group("rest") or ""):
            if part[0] == "#":
                self.ids.append(part[1:])
            elif part[0] == ".":
                self.classes.append(part[1:])
            elif part[0] == "[":
                am = _ATTR_RE.match(part)
                if not am:
                    raise ValueError(part)
                val = am.group(3)
                if val and val[0] in "'\"":
                    val = val[1:-1]
                self.attrs.append((am.group(1), am.group(2), val))
            else:
                if part not in (":first-child", ":root", ":last-child"):
                    raise ValueError(part)
                self.pseudos.append(part)

    def specificity(self):
        return (
            len(self.ids),
            len(self.classes) + len(self.attrs) + len(self.pseudos),
            1 if self.type else 0,
        )

    def matches(self, el, parent_of) -> bool:
        if self.type is not None and strip_ns(el.tag) != self.type:
            return False
        if self.ids and any(el.get("id") != i for i in self.ids):
            return False
        if self.classes:
            have = set((el.get("class") or "").split())
            if not all(c in have for c in self.classes):
                return False
        for name, op, val in self.attrs:
            got = _get_attr(el, name)
            if got is None:
                return False
            if op is None:
                continue
            if op == "=" and got != val:
                return False
            if op == "~=" and val not in got.split():
                return False
            if op == "^=" and not got.startswith(val):
                return False
            if op == "$=" and not got.endswith(val):
                return False
            if op == "*=" and val not in got:
                return False
            if op == "|=" and not (got == val or got.startswith(val + "-")):
                return False
        for p in self.pseudos:
            parent = parent_of.get(id(el))
            if p == ":root":
                if parent is not None:
                    return False
                continue
            if parent is None:
                return False
            kids = [k for k in parent if isinstance(k.tag, str)]
            if p == ":first-child" and kids[0] is not el:
                return False
            if p == ":last-child" and kids[-1] is not el:
                return False
        return True


def _get_attr(el, name):
    if name in el.attrib:
        return el.attrib[name]
    for k, v in el.attrib.items():
        if strip_ns(k) == name.split(":")[-1]:
            return v
    return None


def _parse_selector(text: str):
    """Parse a complex selector into ``[(compound, combinator_to_left), ...]``."""
    text = re.sub(r"\s*>\s*", " > ", text.strip())
    tokens = text.split()
    parts = []
    comb = None
    for tok in tokens:
        if tok == ">":
            comb = ">"
            continue
        parts.append((_Compound(tok), comb if parts else None))
        comb = " "
    if not parts:
        raise ValueError(text)
    # Re-associate: each entry's combinator links it to the previous compound.
    return parts


def _selector_matches(parts, el, parent_of) -> bool:
    def match_at(idx, element) -> bool:
        compound, comb = parts[idx]
        if not compound.matches(element, parent_of):
            return False
        if idx == 0:
            return True
        if comb == ">":
            parent = parent_of.get(id(element))
            return parent is not None and match_at(idx - 1, parent)
        anc = parent_of.get(id(element))
        while anc is not None:
            if match_at(idx - 1, anc):
                return True
            anc = parent_of.get(id(anc))
        return False

    return match_at(len(parts) - 1, el)


def _builtin_rules(css: str):
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    css = css.replace("<![CDATA[", "").replace("]]>", "")
    i, n = 0, len(css)
    while i < n:
        brace = css.find("{", i)
        if brace < 0:
            break
        prelude = css[i:brace].strip()
        # Find the matching close brace (one level of nesting for @media).
        depth, j = 1, brace + 1
        while j < n and depth:
            if css[j] == "{":
                depth += 1
            elif css[j] == "}":
                depth -= 1
            j += 1
        body = css[brace + 1 : j - 1]
        i = j
        if prelude.startswith("@"):
            if prelude.lower().startswith("@media") and _media_applies(prelude[6:]):
                yield from _builtin_rules(body)
            continue
        yield prelude, body


def _compute_builtin(root, texts) -> dict[int, dict[str, Any]]:
    from .style import parse_style_attr

    rules = []
    order = 0
    for css in texts:
        for prelude, body in _builtin_rules(css):
            decls = parse_style_attr(body)
            if not decls:
                continue
            for sel in prelude.split(","):
                try:
                    parts = _parse_selector(sel)
                except ValueError:
                    continue
                spec = tuple(
                    sum(c.specificity()[k] for c, _ in parts) for k in range(3)
                )
                rules.append((spec, order, parts, decls))
                order += 1
    if not rules:
        return {}
    parent_of = {id(c): p for p in root.iter() for c in p}
    result: dict[int, dict[str, Any]] = {}
    for el in root.iter():
        if not isinstance(el.tag, str):
            continue
        hits = [r for r in rules if _selector_matches(r[2], el, parent_of)]
        if not hits:
            continue
        hits.sort(key=lambda r: (r[0], r[1]))
        merged: dict[str, Any] = {}
        for *_unused, decls in hits:
            merged.update(decls)
        result[id(el)] = merged
    return result
