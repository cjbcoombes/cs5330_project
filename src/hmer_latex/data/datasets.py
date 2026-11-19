"""PyTorch Dataset classes for CROHME / HME100K (+ synthetic augmented) data.

Expects data/processed/<split>/ to contain rendered images plus a
labels.tsv (filename<TAB>latex) produced by render_inkml.render_dataset
(for CROHME) or an equivalent prep step for HME100K.
"""

from __future__ import annotations

import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from hmer_latex.data.vocab import Vocab


def default_transform():
    """PIL grayscale image -> float tensor (1, H, W) with ink=1, background=0.

    The renderer writes black ink on white so the PNGs stay human-readable,
    but the model wants the opposite polarity: the encoder's convs use
    `padding=1`, i.e. *zero* padding, so background-as-zero means the
    implicit border matches real background instead of reading as ink.
    """
    return transforms.Compose(
        [
            transforms.ToTensor(),  # (1, H, W) float32 in [0, 1]
            transforms.Lambda(lambda x: 1.0 - x),
        ]
    )


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
            return [(row[0], row[1]) for row in csv.reader(f, delimiter="\t") if row]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        filename, latex = self.samples[idx]
        image = Image.open(self.data_dir / filename).convert("L")
        if self.transform is not None:
            image = self.transform(image)
        token_ids = self.vocab.encode(latex)
        return image, token_ids


def collate_batch(batch, pad_id: int, sos_id: int, eos_id: int):
    """Stack a list of (image, token_ids) samples into padded batch tensors.

    Returns (images, tgt_in, tgt_out, tgt_pad_mask):

        images        (B, 1, H, W)  float32
        tgt_in        (B, T)        int64   [sos] + ids, pad-filled
        tgt_out       (B, T)        int64   ids + [eos], pad-filled
        tgt_pad_mask  (B, T)        bool    True where tgt_in is padding

    `tgt_in` and `tgt_out` are the same ids offset by one position -- that
    offset *is* teacher forcing. Position t of tgt_in is the input the
    decoder conditions on to predict position t of tgt_out.

    The sos/eos convention lives here rather than in Vocab.encode (which
    adds neither) or in the training loop, so that one function owns it.
    `model.generate` has to agree with it; nothing enforces that statically.

    Images need no padding -- the renderer emits a fixed canvas, so they are
    already a common size. `T` is per-batch, not global.

    `tgt_pad_mask` is passed straight through to the model as
    `target_ids_mask` and ends up as the decoder's `tgt_key_padding_mask`
    (True = ignore, which is why the mask is `tgt_in == pad_id` rather than
    its negation). It is what keeps padded positions out of *attention*;
    CrossEntropyLoss(ignore_index=pad_id) separately keeps them out of the
    loss. Both are needed -- without the mask, padded positions still take
    part in attention normalization, so a sample's activations depend
    slightly on what else happens to be in its batch.
    """
    images, token_lists = zip(*batch)

    image_batch = torch.stack(images, dim=0)

    max_len = max(len(ids) for ids in token_lists) + 1
    tgt_in = torch.full((len(token_lists), max_len), pad_id, dtype=torch.long)
    tgt_out = torch.full((len(token_lists), max_len), pad_id, dtype=torch.long)
    for i, ids in enumerate(token_lists):
        tgt_in[i, 0] = sos_id
        if ids:
            seq = torch.tensor(ids, dtype=torch.long)
            tgt_in[i, 1 : len(ids) + 1] = seq
            tgt_out[i, : len(ids)] = seq
        tgt_out[i, len(ids)] = eos_id

    tgt_pad_mask = tgt_in == pad_id

    return image_batch, tgt_in, tgt_out, tgt_pad_mask
