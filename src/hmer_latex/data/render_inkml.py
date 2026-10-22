"""Render CROHME InkML stroke trajectories to offline raster images.

CROHME expressions are stored as InkML: XML files with per-stroke lists of
(x, y[, t]) points plus a ground-truth LaTeX/symbol-label annotation. We
render these to grayscale images ourselves (rather than using any
pre-rendered version) so we control resolution and stroke width, which lets
us later generate synthetic "photographic" variants (thicker/thinner
strokes, scaled, rotated, etc.) from the same source data.

This is a stub: fill in `parse_inkml` and `render_strokes` once we've
pulled a sample InkML file and looked at its actual structure.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass
class InkExpression:
    """A single parsed InkML expression."""

    strokes: list[np.ndarray]  # each array is (N, 2) of (x, y) points
    label: str  # ground-truth LaTeX string


def parse_inkml(path: Path) -> InkExpression:
    """Parse a single .inkml file into stroke coordinates + LaTeX label."""
    raise NotImplementedError("TODO: parse InkML XML (trace elements + annotation)")


def render_strokes(
    strokes: list[np.ndarray],
    image_size: tuple[int, int] = (256, 256),
    stroke_width: int = 3,
    padding: int = 10,
) -> Image.Image:
    """Rasterize a list of strokes to a single grayscale image.

    Strokes are normalized to fit within `image_size` (minus `padding`)
    while preserving aspect ratio, then drawn as connected line segments.
    """
    raise NotImplementedError("TODO: normalize stroke coordinates and draw with PIL.ImageDraw")


def render_inkml_file(path: Path, **render_kwargs) -> tuple[Image.Image, str]:
    """Convenience wrapper: parse + render a single InkML file."""
    expr = parse_inkml(path)
    image = render_strokes(expr.strokes, **render_kwargs)
    return image, expr.label


def render_dataset(inkml_dir: Path, out_dir: Path, **render_kwargs) -> None:
    """Render every .inkml file in `inkml_dir` to PNGs in `out_dir`,
    alongside a labels file mapping image filename -> LaTeX string.
    """
    raise NotImplementedError("TODO: walk inkml_dir, call render_inkml_file, write labels.tsv")
