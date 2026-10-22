"""Evaluation entry point: run a trained checkpoint against a dataset and
report ExpRate (exact-match accuracy) and symbol-level error rate.

Usage:
    uv run python -m hmer_latex.eval --config configs/base.yaml --checkpoint checkpoints/baseline/best.pt --data-dir data/processed/crohme_test
"""

from __future__ import annotations

import argparse
from pathlib import Path

from hmer_latex.config import ExperimentConfig


def evaluate(config: ExperimentConfig, checkpoint_path: Path, data_dir: Path) -> dict:
    # TODO:
    # - load Vocab, build HMEDataset for data_dir
    # - load HMERModel from checkpoint_path
    # - run model.generate over the dataset
    # - compute ExpRate (exact string match after normalization) and
    #   edit-distance-based symbol error rate
    # - optionally dump per-sample predictions for error analysis /
    #   attention visualization
    raise NotImplementedError("TODO: implement evaluation loop")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()

    config = ExperimentConfig.from_yaml(args.config)
    metrics = evaluate(config, args.checkpoint, args.data_dir)
    print(metrics)


if __name__ == "__main__":
    main()
