"""Training entry point.

Usage:
    uv run python -m hmer_latex.train --config configs/base.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from hmer_latex.config import ExperimentConfig


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train(config: ExperimentConfig) -> None:
    device = pick_device(config.train.device)
    print(f"Training on device: {device}")

    # TODO:
    # - load Vocab from config.data.vocab_path
    # - build HMEDataset/DataLoader for train + val
    # - build HMERModel, move to device
    # - optimizer (AdamW), LR schedule
    # - training loop with teacher forcing, cross-entropy loss (ignore pad)
    # - periodic validation: ExpRate (exact match) on held-out set
    # - checkpoint to config.train.checkpoint_dir, logs to config.train.log_dir
    raise NotImplementedError("TODO: implement training loop")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    config = ExperimentConfig.from_yaml(args.config)
    train(config)


if __name__ == "__main__":
    main()
