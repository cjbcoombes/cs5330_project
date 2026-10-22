"""PyTorch Dataset classes for CROHME / HME100K (+ synthetic augmented) data.

Expects data/processed/<split>/ to contain rendered images plus a
labels.tsv (filename<TAB>latex) produced by render_inkml.render_dataset
(for CROHME) or an equivalent prep step for HME100K.
"""

from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from hmer_latex.data.vocab import Vocab


class HMEDataset(Dataset):
    """A directory of (image, LaTeX label) pairs."""

    def __init__(self, data_dir: Path, vocab: Vocab, transform=None):
        self.data_dir = Path(data_dir)
        self.vocab = vocab
        self.transform = transform
        self.samples: list[tuple[str, str]] = self._load_labels(self.data_dir / "labels.tsv")

    @staticmethod
    def _load_labels(labels_path: Path) -> list[tuple[str, str]]:
        with open(labels_path, newline="") as f:
            return [(row[0], row[1]) for row in csv.reader(f, delimiter="\t")]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        filename, latex = self.samples[idx]
        image = Image.open(self.data_dir / filename).convert("L")
        if self.transform is not None:
            image = self.transform(image)
        token_ids = self.vocab.encode(latex)
        return image, token_ids


def collate_batch(batch, pad_id: int):
    """TODO: pad images to a common size and token sequences to a common
    length, return (image_batch, image_lengths?, token_batch, token_lengths).
    """
    raise NotImplementedError("TODO: implement once image transform/sizing strategy is decided")
