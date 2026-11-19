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
    test_dir: str = "data/processed/crohme_test"
    tiny_dir: str = "data/processed/crohme_tiny"  # 64-sample MVP overfit slice
    vocab_path: str = "data/processed/vocab.json"

    # (width, height), PIL's order -- the resulting tensor is (1, height, width).
    # Wide because 93% of CROHME expressions are wider than tall (median
    # aspect 3.13:1); a square canvas spends ~68% of its pixels on whitespace.
    # With SimpleCNNEncoder's /32 downsampling this gives a 12x4 = 48-position
    # feature grid.
    image_size: tuple[int, int] = (384, 128)
    batch_size: int = 16
    num_workers: int = 0


@dataclass
class ModelConfig:
    # MVP defaults (SimpleCNNEncoder + 1-layer/1-head decoder, see README
    # Roadmap Phase 1a). Bump these once the MVP loop runs end to end and
    # you swap in DenseNetEncoder.
    d_model: int = 64
    nhead: int = 1
    num_decoder_layers: int = 1


@dataclass
class TrainConfig:
    epochs: int = 1000
    lr: float = 1e-4
    weight_decay: float = 1e-4
    device: str = "auto"  # "auto" picks cuda > mps > cpu
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "runs"
    run_name: str = "baseline_1"


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
