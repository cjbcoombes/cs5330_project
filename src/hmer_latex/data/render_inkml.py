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
    image_size: tuple[int, int] = (384, 128),
    stroke_width: int = 3,
    padding: int = 10,
) -> Image.Image:
    """Rasterize a list of strokes to a single grayscale image.

    Strokes are normalized to fit within `image_size` (minus `padding`)
    while preserving aspect ratio, then drawn as connected line segments.

    `image_size` is (width, height) -- PIL's order, not the (H, W) of the
    resulting tensor. The default 384x128 is wide because 93% of CROHME
    expressions are wider than tall (median aspect 3.13:1).

    One scale factor is computed across *all* strokes together, so relative
    size and position between strokes is preserved -- that spatial relation
    is the entire signal for superscripts, subscripts and fractions.
    """
    width, height = image_size
    image = Image.new("L", image_size, BACKGROUND)
    if not strokes:
        return image

    all_points = np.concatenate(strokes, axis=0)
    lo = all_points.min(axis=0)
    span = np.maximum(all_points.max(axis=0) - lo, 1e-6)

    avail = np.array([width - 2 * padding, height - 2 * padding], dtype=np.float64)
    scale = float(np.min(avail / span))

    # Centre the scaled ink in the canvas; the limiting axis ends up with
    # exactly `padding` on each side, the other with padding + letterboxing.
    offset = (np.array([width, height], dtype=np.float64) - span * scale) / 2.0

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
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--stroke-width", type=int, default=3)
    parser.add_argument("--padding", type=int, default=10)
    args = parser.parse_args()

    written, skipped = render_dataset(
        args.inkml_dirs,
        args.out_dir,
        image_size=(args.width, args.height),
        stroke_width=args.stroke_width,
        padding=args.padding,
    )
    print(f"{args.out_dir}: wrote {written}, skipped {skipped}")


if __name__ == "__main__":
    main()
