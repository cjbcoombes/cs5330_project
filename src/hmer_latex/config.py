"""Experiment configuration, loaded from YAML (see configs/base.yaml).

We'll be running many variants (different datasets, augmentations,
preprocessing options) so every run should be driven by a config file
rather than hardcoded args -- makes runs reproducible and diffable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class DataConfig:
    train_dir: str = "data/processed/crohme_train"
    val_dir: str = "data/processed/crohme_val"
    vocab_path: str = "data/processed/vocab.json"
    image_size: tuple[int, int] = (256, 256)
    batch_size: int = 16
    num_workers: int = 4


@dataclass
class ModelConfig:
    d_model: int = 256
    nhead: int = 8
    num_decoder_layers: int = 3


@dataclass
class TrainConfig:
    epochs: int = 50
    lr: float = 1e-4
    weight_decay: float = 1e-4
    device: str = "auto"  # "auto" picks cuda > mps > cpu
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "runs"
    run_name: str = "baseline"


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> "ExperimentConfig":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        return cls(
            data=DataConfig(**raw.get("data", {})),
            model=ModelConfig(**raw.get("model", {})),
            train=TrainConfig(**raw.get("train", {})),
        )
