"""Render the CROHME splits, build the vocab, and carve the MVP overfit slice.

Usage:
    uv run python scripts/prepare_data.py
    uv run python scripts/prepare_data.py --tiny-size 128
    uv run python scripts/prepare_data.py --splits train    # just one split

Reads data/raw/crohme/crohme_2023/ (already downloaded by download_data.py)
and writes data/processed/:

    crohme_train/   labels.tsv + PNGs   12024 real-ink expressions
    crohme_val/                          1702
    crohme_test/                         3499
    crohme_tiny/                           64   slice of train, for overfitting
    vocab.json                            142 tokens, built from train only

Deliberately excluded: INKML/train/Artificial_data (147062 synthetically
generated files). It dwarfs the real ink, is irrelevant to the MVP, and
would pull synthetic-only tokens into the vocab. Phase 2 can opt in.

This exists as one command rather than three render invocations plus a
vocab one-liner because the ordering matters -- the vocab must be built
from the *train* labels, and that is easy to get silently wrong by hand.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

from hmer_latex.data.render_inkml import render_dataset
from hmer_latex.data.vocab import UNK, Vocab

REPO_ROOT = Path(__file__).resolve().parent.parent
CROHME_2023 = REPO_ROOT / "data" / "raw" / "crohme" / "crohme_2023" / "TC11_CROHME23" / "INKML"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

# split name -> source dirs. The train split is assembled from two dirs;
# naming them explicitly is what keeps Artificial_data out.
SPLITS: dict[str, list[Path]] = {
    "train": [CROHME_2023 / "train" / "CROHME2023_train", CROHME_2023 / "train" / "CROHME2019"],
    "val": [CROHME_2023 / "val"],
    "test": [CROHME_2023 / "test"],
}


def read_labels(split_dir: Path) -> list[str]:
    with open(split_dir / "labels.tsv", newline="") as f:
        return [row[1] for row in csv.reader(f, delimiter="\t") if row]


def unk_rate(labels: list[str], vocab: Vocab) -> float:
    """Fraction of tokens in `labels` that the vocab doesn't know.

    A high rate on val/test means the tokenizer is splitting inconsistently,
    not that the data is exotic -- it's the cheapest check we have on it.
    """
    unk_id = vocab.stoi[UNK]
    ids = [i for label in labels for i in vocab.encode(label)]
    if not ids:
        return 0.0
    return sum(i == unk_id for i in ids) / len(ids)


def carve_tiny(train_dir: Path, tiny_dir: Path, n: int) -> int:
    """Copy the first `n` train samples into their own split directory.

    The first n rather than a random n: the overfit test wants a fixed,
    inspectable set, and reproducing it shouldn't depend on a seed.
    """
    if tiny_dir.exists():
        shutil.rmtree(tiny_dir)
    tiny_dir.mkdir(parents=True)

    with open(train_dir / "labels.tsv", newline="") as f:
        rows = [row for row in csv.reader(f, delimiter="\t") if row][:n]

    for filename, _ in rows:
        shutil.copyfile(train_dir / filename, tiny_dir / filename)
    with open(tiny_dir / "labels.tsv", "w", newline="") as f:
        csv.writer(f, delimiter="\t").writerows(rows)

    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", choices=list(SPLITS), default=list(SPLITS))
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--stroke-width", type=int, default=3)
    parser.add_argument("--padding", type=int, default=10)
    parser.add_argument("--tiny-size", type=int, default=64)
    args = parser.parse_args()

    missing = [d for s in args.splits for d in SPLITS[s] if not d.is_dir()]
    if missing:
        raise SystemExit(
            "Missing raw data:\n  "
            + "\n  ".join(str(d) for d in missing)
            + "\n\nRun: uv run python scripts/download_data.py crohme"
        )

    render_kwargs = dict(
        image_size=(args.width, args.height),
        stroke_width=args.stroke_width,
        padding=args.padding,
    )

    counts: dict[str, tuple[int, int]] = {}
    for split in args.splits:
        out_dir = PROCESSED_DIR / f"crohme_{split}"
        counts[split] = render_dataset(SPLITS[split], out_dir, **render_kwargs)

    # Vocab from the train split only; val/test unknowns are expected and
    # should map to <unk> rather than inflating the vocab.
    vocab = None
    train_dir = PROCESSED_DIR / "crohme_train"
    if (train_dir / "labels.tsv").exists():
        vocab = Vocab.build_from_labels(read_labels(train_dir))
        vocab.save(PROCESSED_DIR / "vocab.json")

    n_tiny = 0
    if "train" in args.splits:
        n_tiny = carve_tiny(train_dir, PROCESSED_DIR / "crohme_tiny", args.tiny_size)

    print(f"\n{'=' * 58}\nprepare_data summary  (canvas {args.width}x{args.height})\n{'=' * 58}")
    for split, (written, skipped) in counts.items():
        print(f"  crohme_{split:<6} {written:>6} rendered   {skipped:>4} skipped")
    if n_tiny:
        print(f"  crohme_tiny   {n_tiny:>6} copied from train")
    if vocab is not None:
        print(f"\n  vocab         {len(vocab):>6} tokens -> {PROCESSED_DIR / 'vocab.json'}")
        for split in args.splits:
            labels = read_labels(PROCESSED_DIR / f"crohme_{split}")
            print(f"  <unk> rate    {unk_rate(labels, vocab) * 100:>6.3f}%  on {split}")


if __name__ == "__main__":
    main()
