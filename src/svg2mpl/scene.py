"""The intermediate representation produced by parsing an SVG.

A :class:`Scene` is a flat list of drawables (shapes and text) with their
geometry already in document space (element ``transform`` and ``viewBox`` baked
in, but *not* the optional y-flip, which is applied at render time). It also
carries the document size and any gradient definitions referenced by fills.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from matplotlib.path import Path
from matplotlib.transforms import Affine2D

__all__ = ["Shape", "Text", "Scene"]


@dataclass
class Shape:
    """A filled/stroked path with its resolved matplotlib style."""

    id: str
    path: Path
    style: dict[str, str]
    kwargs: dict[str, Any]


@dataclass
class Text:
    """A piece of text with its anchor point and resolved style."""

    id: str
    x: float
    y: float
    content: str
    style: dict[str, str]
    transform: Affine2D


@dataclass
class Scene:
    """A parsed SVG: its drawables, size, and gradient definitions."""

    shapes: list[Shape] = field(default_factory=list)
    texts: list[Text] = field(default_factory=list)
    width: float = 0.0
    height: float = 0.0
    gradients: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.shapes) + len(self.texts)
