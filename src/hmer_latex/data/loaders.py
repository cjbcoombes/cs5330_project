"""DataLoader construction for the CROHME splits, in one place.

Training, per-epoch validation and final scoring all need loaders built the
same way -- same collate function, same sos/eos convention, same padding
multiple. Building them here rather than in each caller is what keeps the
train and eval paths from drifting apart: a loader whose `size_multiple`
disagrees with the encoder's stride fails with a shape error deep inside the
model, and one that shuffles silently misaligns predictions from labels.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

from torch.utils.data import DataLoader, Subset

from hmer_latex.config import ExperimentConfig
from hmer_latex.data.datasets import (
    HMEDataset,
    collate_batch,
    default_transform,
    train_transform,
)
from hmer_latex.data.vocab import EOS, PAD, SOS, Vocab


def make_loader(
    data_dir: str | Path,
    vocab: Vocab,
    config: ExperimentConfig,
    shuffle: bool = False,
    augment: bool = False,
    limit: int | None = None,
) -> DataLoader:
    """One split -> a DataLoader of 5-tuples (see datasets.collate_batch).

    `limit` takes the first N samples, for the per-epoch ExpRate estimate:
    greedy-decoding the whole val split every epoch costs more than the
    epoch itself. The *first* N rather than a random N so the number is
    comparable across epochs and across runs.
    """
    dataset = HMEDataset(
        data_dir=data_dir,
        vocab=vocab,
        transform=train_transform() if augment else default_transform(),
    )
    if limit is not None and limit < len(dataset):
        dataset = Subset(dataset, range(limit))

    # functools.partial rather than a closure: DataLoader workers are spawned
    # on macOS, so collate_fn has to be picklable.
    collate = partial(
        collate_batch,
        pad_id=vocab.stoi[PAD],
        sos_id=vocab.stoi[SOS],
        eos_id=vocab.stoi[EOS],
        size_multiple=config.data.size_multiple,
    )
    return DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=shuffle,
        collate_fn=collate,
        num_workers=config.data.num_workers,
    )


def train_loader(config: ExperimentConfig, vocab: Vocab) -> DataLoader:
    return make_loader(
        config.data.train_dir, vocab, config, shuffle=True, augment=config.data.augment
    )


def val_loader(
    config: ExperimentConfig, vocab: Vocab, limit: int | None = None
) -> DataLoader:
    return make_loader(config.data.val_dir, vocab, config, limit=limit)


def test_loader(config: ExperimentConfig, vocab: Vocab) -> DataLoader:
    return make_loader(config.data.test_dir, vocab, config)


def label_lengths(loader: DataLoader) -> list[tuple[int, str]]:
    """(token count, label) for every sample behind `loader`, longest first."""
    dataset = loader.dataset
    if isinstance(dataset, Subset):
        samples = [dataset.dataset.samples[i] for i in dataset.indices]
    else:
        samples = dataset.samples
    return sorted(((len(latex.split()), latex) for _, latex in samples), reverse=True)
