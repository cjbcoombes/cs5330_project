"""Experiment configuration.

Plain dataclasses with defaults -- for now, one variant at a time, edited
directly in code. If/when we're running many variants side by side, revisit
moving this to YAML configs.
"""

from __future__ import annotations

from dataclasses import dataclass, field


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
