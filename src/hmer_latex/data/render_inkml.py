"""Render CROHME InkML stroke trajectories to offline raster images.

CROHME expressions are stored as InkML: XML files with per-stroke lists of
(x, y[, t]) points plus a ground-truth LaTeX/symbol-label annotation. We
render these to grayscale images ourselves (rather than using any
pre-rendered version) so we control resolution and stroke width, which lets
us later generate synthetic "photographic" variants (thicker/thinner
strokes, scaled, rotated, etc.) from the same source data.

Usage:
    uv run python -m hmer_latex.data.render_inkml \
        --inkml-dirs <dir> [<dir> ...] --out-dir data/processed/crohme_train

Normally invoked via scripts/prepare_data.py, which renders every split and
builds the vocab in one go.

Facts about the CROHME 2023 corpus that this module encodes (all measured,
see the plan file):

- The y axis needs no flip: InkML y grows downward, same as image rows.
  Verified against the reference IMG/ PNGs shipped with the dataset.
- 5.4% of files contain at least one single-point trace (dots: periods,
  \\cdot, the dot on an i). PIL's draw.line renders *nothing* for a
  one-point path, so those are drawn as filled ellipses instead.
- Ground-truth annotations are inconsistently formatted, so every label
  goes through vocab.tokenize_latex before being written to labels.tsv.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

from hmer_latex.data.vocab import tokenize_latex

INKML_NS = {"ink": "http://www.w3.org/2003/InkML"}

BACKGROUND = 255  # white
INK = 0  # black -- matches the dataset's own reference renders


@dataclass
class InkExpression:
    """A single parsed InkML expression."""

    strokes: list[np.ndarray]  # each array is (N, 2) of (x, y) points
    label: str  # ground-truth LaTeX string


def parse_inkml(path: Path) -> InkExpression:
    """Parse a single .inkml file into stroke coordinates + LaTeX label."""
    root = ET.parse(path).getroot()

    strokes: list[np.ndarray] = []
    for trace in root.findall("ink:trace", INKML_NS):
        points = []
        for chunk in (trace.text or "").split(","):
            fields = chunk.split()
            # Some CROHME years carry a third (time) channel; x, y come first.
            if len(fields) >= 2:
                points.append((float(fields[0]), float(fields[1])))
        if points:
            strokes.append(np.asarray(points, dtype=np.float64))

    truth = root.find('ink:annotation[@type="truth"]', INKML_NS)
    label = tokenize_latex(truth.text or "" if truth is not None else "")

    return InkExpression(strokes=strokes, label=label)


def render_strokes(
    strokes: list[np.ndarray],
    target_height: int = 64,
    max_width: int = 512,
    stroke_width: int = 3,
    margin: int = 4,
    min_side: int = 16,
) -> Image.Image:
    """Rasterize a list of strokes to a single grayscale image.

    One scale factor is computed across *all* strokes together, so relative
    size and position between strokes is preserved -- that spatial relation
    is the entire signal for superscripts, subscripts and fractions.

    The ink is scaled to `target_height` tall and the width follows the
    aspect ratio, capped at `max_width`. Every image therefore comes out a
    different size; collate_batch pads each batch to a common one and passes
    a mask so the model ignores the padding. Expressions wider than
    `max_width / target_height` (8:1 at the defaults) are width-limited and
    come out shorter than `target_height`.
    """
    if not strokes:
        # Blank, and kept small rather than letting an empty expression pad
        # every batch it lands in out to max_width.
        return Image.new("L", (min_side, min_side), BACKGROUND)

    all_points = np.concatenate(strokes, axis=0)
    lo = all_points.min(axis=0)
    span = np.maximum(all_points.max(axis=0) - lo, 1e-6)

    # Fit the height, then back off if that would overflow max_width.
    scale = (target_height - 2 * margin) / span[1]
    scale = min(scale, (max_width - 2 * margin) / span[0])

    # The clamps are load-bearing, not defensive: `span` is floored at 1e-6,
    # so an expression with no vertical extent at all (a lone minus sign, a
    # single dot) would otherwise scale to a canvas of zero height.
    width = int(np.clip(round(span[0] * scale) + 2 * margin, min_side, max_width))
    height = int(np.clip(round(span[1] * scale) + 2 * margin, min_side, target_height))
    # No centring: padding is added on the right/bottom at batch time.
    offset = np.array([margin, margin], dtype=np.float64)

    image = Image.new("L", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(image)
    radius = stroke_width / 2.0
    for stroke in strokes:
        points = (stroke - lo) * scale + offset
        # A single point, or a stroke that never moves, is a dot. draw.line
        # renders 0 pixels for a one-point path and a single pixel for a
        # degenerate two-point one, which silently loses periods and \cdot.
        if len(points) == 1 or np.ptp(points, axis=0).max() < 1e-9:
            x, y = points[0]
            draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=INK)
        else:
            draw.line(
                [(float(x), float(y)) for x, y in points],
                fill=INK,
                width=stroke_width,
                joint="curve",
            )

    return image


def render_inkml_file(path: Path, **render_kwargs) -> tuple[Image.Image, str]:
    """Convenience wrapper: parse + render a single InkML file."""
    expr = parse_inkml(path)
    image = render_strokes(expr.strokes, **render_kwargs)
    return image, expr.label


def render_dataset(
    inkml_dirs: list[Path],
    out_dir: Path,
    show_progress: bool = True,
    **render_kwargs,
) -> tuple[int, int]:
    """Render every .inkml file under `inkml_dirs` to PNGs in `out_dir`,
    alongside a labels.tsv mapping image filename -> tokenized LaTeX.

    Takes a *list* of directories so one split can be assembled from several
    source dirs (the real-ink CROHME train split lives in two). Naming the
    dirs explicitly is also what keeps the 147k-file Artificial_data corpus
    out -- never point this at a whole `train/` tree.

    Returns (n_written, n_skipped).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    for inkml_dir in inkml_dirs:
        paths.extend(sorted(Path(inkml_dir).rglob("*.inkml")))

    rows: list[tuple[str, str]] = []
    skipped = 0
    for path in tqdm(paths, desc=out_dir.name, disable=not show_progress):
        try:
            image, label = render_inkml_file(path, **render_kwargs)
        except Exception as exc:  # noqa: BLE001 -- one bad file shouldn't kill a 12k render
            tqdm.write(f"  skipped {path.name}: {type(exc).__name__}: {exc}")
            skipped += 1
            continue
        if not label:
            tqdm.write(f"  skipped {path.name}: empty label")
            skipped += 1
            continue
        filename = f"{path.stem}.png"
        image.save(out_dir / filename)
        rows.append((filename, label))

    # newline="" to match HMEDataset._load_labels, which reads with csv.reader.
    with open(out_dir / "labels.tsv", "w", newline="") as f:
        csv.writer(f, delimiter="\t").writerows(rows)

    return len(rows), skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inkml-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--target-height", type=int, default=64)
    parser.add_argument("--max-width", type=int, default=512)
    parser.add_argument("--stroke-width", type=int, default=3)
    args = parser.parse_args()

    written, skipped = render_dataset(
        args.inkml_dirs,
        args.out_dir,
        target_height=args.target_height,
        max_width=args.max_width,
        stroke_width=args.stroke_width,
    )
    print(f"{args.out_dir}: wrote {written}, skipped {skipped}")


if __name__ == "__main__":
    main()
