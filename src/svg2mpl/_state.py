"""Traversal state shared by the parser modules."""

from __future__ import annotations

from dataclasses import dataclass

from matplotlib.path import Path
from matplotlib.transforms import Affine2D

__all__ = ["State"]


@dataclass(frozen=True)
class State:
    """Per-element traversal state, copied (never mutated) when descending.

    Attributes
    ----------
    ctm : Affine2D
        Current transformation matrix: user space -> document space.
    style : dict or None
        The parent's computed style (what children inherit from).
    out : list
        Where drawables are appended (the scene, a group, a pattern tile ...).
    vp : tuple
        Width and height of the nearest viewport, for percentage lengths.
    opacity : float
        Product of the ancestors' group opacities folded into children.
    clip : Path or None
        Active clip region in document space (nonzero rule).
    depth : int
        Nesting depth, guarding against runaway recursion.
    chain : frozenset
        ``id()`` of referenced elements being expanded (``use``, markers,
        patterns, masks), guarding against reference cycles.
    measuring : bool
        Only collect geometry (for bounding boxes); no paint, clip or effects.
    """

    ctm: Affine2D
    style: dict | None
    out: list
    vp: tuple[float, float]
    opacity: float = 1.0
    clip: Path | None = None
    depth: int = 0
    chain: frozenset = frozenset()
    measuring: bool = False
