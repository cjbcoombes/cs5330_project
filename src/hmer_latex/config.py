"""Experiment configuration.

Plain dataclasses with defaults, plus `--preset` / `--set` overrides so a run
can be described entirely by its command line:

    uv run python -m hmer_latex.train --preset densenet_small
    uv run python -m hmer_latex.train --preset simple_mvp \
        --set data.train_dir=data/processed/crohme_tiny data.augment=false

The preset is applied first, then `--set` on top. Both use dotted
`section.field` keys matching the dataclasses below.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field


@dataclass
class DataConfig:
    train_dir: str = "data/processed/crohme_train"
    val_dir: str = "data/processed/crohme_val"
    test_dir: str = "data/processed/crohme_test"
    tiny_dir: str = "data/processed/crohme_tiny"  # 64-sample overfit slice
    vocab_path: str = "data/processed/vocab.json"

    # collate_batch pads each batch up to a multiple of this. 32 satisfies
    # both encoders (SimpleCNN downsamples by 32, DenseNet by 16), which is
    # what makes the feature-mask arithmetic in HMERModel exact.
    size_multiple: int = 32

    augment: bool = True
    batch_size: int = 16
    num_workers: int = 0


@dataclass
class ModelConfig:
    encoder: str = "densenet"  # "densenet" | "simple"
    d_model: int = 128
    nhead: int = 4
    num_decoder_layers: int = 3
    dim_feedforward: int = 512
    dropout: float = 0.2

    # DenseNet encoder only.
    growth_rate: int = 16
    encoder_layers: int = 8  # per dense block, of which there are 3

    # Caps the causal mask and the decoder's positional encoding. The longest
    # real training label is 204 tokens, so this has to clear that.
    max_seq_len: int = 256


@dataclass
class TrainConfig:
    epochs: int = 80
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 500
    device: str = "auto"  # "auto" picks cuda > mps > cpu
    seed: int = 0
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "runs"
    run_name: str = "densenet_small"

    # Greedy decoding the whole val split every epoch is slow; score the
    # first this-many samples each epoch and the full split once at the end.
    # The *first* N rather than a random N so the per-epoch number is
    # comparable across epochs and across runs.
    val_exprate_samples: int = 512


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def from_cli(cls, argv: list[str] | None = None) -> "ExperimentConfig":
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--preset", default="densenet_small", choices=sorted(PRESETS))
        parser.add_argument(
            "--set",
            nargs="*",
            default=[],
            metavar="section.field=value",
            help="override any config field, e.g. --set train.lr=1e-4 data.batch_size=8",
        )
        args = parser.parse_args(argv)

        config = cls()
        for key, value in PRESETS[args.preset].items():
            config.override(key, value)
        for pair in args.set:
            key, sep, value = pair.partition("=")
            if not sep:
                parser.error(f"--set expects section.field=value, got {pair!r}")
            config.override(key, value)
        return config

    def override(self, dotted_key: str, value) -> None:
        """Set one field by "section.field" name, casting to the field's type."""
        section_name, _, field_name = dotted_key.partition(".")
        section = getattr(self, section_name, None)
        if section is None or not hasattr(section, field_name):
            raise KeyError(f"no config field {dotted_key!r}")
        setattr(section, field_name, _cast(value, getattr(section, field_name)))

    def flat(self) -> dict:
        """Flattened "section.field" -> value, for TensorBoard add_hparams."""
        return {
            f"{section}.{name}": v
            for section, fields in asdict(self).items()
            for name, v in fields.items()
        }


def _cast(value, current):
    """Cast a command-line string to the type of the value it replaces."""
    if not isinstance(value, str):
        return value
    if isinstance(current, bool):
        if value.lower() not in {"true", "false", "1", "0"}:
            raise ValueError(f"expected a boolean, got {value!r}")
        return value.lower() in {"true", "1"}
    if isinstance(current, int):
        return int(value)
    if isinstance(current, float):
        return float(value)
    return value


# Starting points, as flat dotted-key overrides on the defaults above.
#
# `densenet_small` is sized for MPS. `densenet_bttr` is BTTR's configuration
# (Zhao et al. 2021) -- worth re-checking against the paper before the first
# CUDA run. Both use AdamW + warmup + cosine rather than BTTR's Adadelta at
# lr=1.0; if the big config plateaus well below ~50% ExpRate, their optimizer
# is the first thing to try.
PRESETS: dict[str, dict] = {
    # The Phase 1a MVP: tiny model, tiny decoder. Fast enough to debug the
    # training loop itself on.
    "simple_mvp": {
        "data.batch_size": 32,
        "model.encoder": "simple",
        "model.d_model": 64,
        "model.nhead": 1,
        "model.num_decoder_layers": 1,
        "model.dim_feedforward": 256,
        "model.dropout": 0.1,
        "train.warmup_steps": 0,
        "train.run_name": "simple_mvp",
    },
    "densenet_small": {},  # the dataclass defaults
    "densenet_bttr": {
        "data.batch_size": 32,
        "model.growth_rate": 24,
        "model.encoder_layers": 16,
        "model.d_model": 256,
        "model.nhead": 8,
        "model.num_decoder_layers": 3,
        "model.dim_feedforward": 1024,
        "model.dropout": 0.3,
        "train.epochs": 200,
        "train.run_name": "densenet_bttr",
    },
}
