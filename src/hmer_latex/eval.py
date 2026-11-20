"""Scoring a model against a split: ExpRate plus symbol-level error rate.

Called from train.py each epoch and at the end of a run, and importable
into a notebook:

    from hmer_latex.data.loaders import test_loader
    from hmer_latex.eval import load_for_inference, evaluate, format_metrics

    model, vocab, config = load_for_inference("checkpoints/run/best.pt", device)
    metrics, samples = evaluate(model, test_loader(config, vocab), vocab, device)
    print(format_metrics(metrics))
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from hmer_latex.config import ExperimentConfig
from hmer_latex.data.vocab import EOS, PAD, SOS, Vocab
from hmer_latex.metrics import expression_metrics
from hmer_latex.models.model import HMERModel


def build_model(config: ExperimentConfig, vocab_size: int) -> HMERModel:
    """The one place that maps ModelConfig onto HMERModel's constructor."""
    return HMERModel(
        vocab_size=vocab_size,
        encoder=config.model.encoder,
        d_model=config.model.d_model,
        nhead=config.model.nhead,
        num_decoder_layers=config.model.num_decoder_layers,
        dim_feedforward=config.model.dim_feedforward,
        dropout=config.model.dropout,
        max_seq_len=config.model.max_seq_len,
        growth_rate=config.model.growth_rate,
        encoder_layers=config.model.encoder_layers,
    )


def load_for_inference(
    checkpoint_path: str | Path, device: torch.device
) -> tuple[HMERModel, Vocab, ExperimentConfig]:
    """Rebuild a model from a checkpoint written by train.save_checkpoint.

    The stored config is what makes this possible: d_model/nhead/layers are
    needed to *construct* the model before weights will load, and the stored
    vocab_path because token id 47 means nothing on its own.
    """
    blob = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = ExperimentConfig()
    for section, fields in blob["config"].items():
        for name, value in fields.items():
            config.override(f"{section}.{name}", value)

    vocab = Vocab.load(blob.get("vocab_path", config.data.vocab_path))
    model = build_model(config, len(vocab)).to(device)
    model.load_state_dict(blob["model"])
    model.eval()

    return model, vocab, config


def _tokens_up_to_eos(ids: list[int], eos_id: int, special: set[int]) -> list[str]:
    """Token ids -> symbols, stopping at the first eos and dropping specials.

    Greedy decoding keeps emitting past a row's eos (every row in the batch
    runs until the whole batch is done), so the tail has to be cut here or
    the junk counts against every metric.
    """
    out = []
    for i in ids:
        if i == eos_id:
            break
        if i not in special:
            out.append(i)
    return out


@torch.no_grad()
def evaluate(
    model: HMERModel,
    loader: DataLoader,
    vocab: Vocab,
    device: torch.device,
) -> tuple[dict, list[tuple[str, str]]]:
    """Greedy-decode every batch in `loader` and score the predictions.

    References come from `tgt_out`, i.e. the ids the model was trained to
    produce. Out-of-vocabulary tokens are therefore scored as the <unk> the
    model actually sees, not as the original symbol -- which flatters the
    number slightly, but keeps this consistent with the training loss. The
    <unk> rate is 0.02% on val and 0.10% on test (scripts/prepare_data.py
    prints it), so the difference is well below the noise.

    Returns (metrics, samples) where samples is a few (ref, pred) pairs for
    eyeballing.
    """
    pad_id, sos_id, eos_id = vocab.stoi[PAD], vocab.stoi[SOS], vocab.stoi[EOS]
    special = {pad_id, sos_id}

    model.eval()
    preds: list[list[str]] = []
    refs: list[list[str]] = []
    for images, img_pad_mask, _, tgt_out, _ in loader:
        out = model.generate(
            images.to(device), img_pad_mask.to(device), sos_id, eos_id
        )
        for row in out.cpu():
            ids = _tokens_up_to_eos(row.tolist(), eos_id, special)
            preds.append([vocab.itos[i] for i in ids])
        for row in tgt_out:
            ids = _tokens_up_to_eos(row.tolist(), eos_id, special)
            refs.append([vocab.itos[i] for i in ids])

    samples = [(" ".join(r), " ".join(p)) for r, p in list(zip(refs, preds))[:8]]
    return expression_metrics(preds, refs), samples


def format_metrics(metrics: dict) -> str:
    return (
        f"ExpRate {metrics['exprate'] * 100:5.2f}%  "
        f"@1 {metrics['exprate_leq1'] * 100:5.2f}%  "
        f"@2 {metrics['exprate_leq2'] * 100:5.2f}%  "
        f"SER {metrics['symbol_error_rate'] * 100:5.2f}%  "
        f"(n={metrics['n']})"
    )
