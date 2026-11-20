"""Training entry point.

Usage:
    uv run python -m hmer_latex.train --preset densenet_small
    uv run python -m hmer_latex.train --preset simple_mvp \
        --set data.train_dir=data/processed/crohme_tiny data.augment=false

Logs scalars to TensorBoard under `{train.log_dir}/{train.run_name}`:

    uv run tensorboard --logdir runs

Each epoch logs teacher-forced train/val loss plus greedy-decoded val
ExpRate, and `best.pt` is selected on **val ExpRate** -- not on loss, which
can keep improving while exact-match accuracy stalls. Expect ExpRate to sit
at 0 for the first several epochs even as token accuracy looks healthy:
labels average 15 tokens, so exact match needs almost every token right.
"""

from __future__ import annotations

import random
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from hmer_latex.config import ExperimentConfig
from hmer_latex.data.loaders import (
    label_lengths,
    make_loader,
    train_loader,
    val_loader,
)
from hmer_latex.data.vocab import PAD, Vocab
from hmer_latex.eval import build_model, evaluate, format_metrics, load_for_inference


def pick_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """Seed everything that affects a run: init, shuffling, augmentation.

    Without this, two configs that differ by a percent or two of ExpRate
    can't be told apart from run-to-run noise.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    opt: torch.optim.Optimizer,
    lr_sched: torch.optim.lr_scheduler.LRScheduler,
    config: ExperimentConfig,
    epoch: int,
    best_exprate: float,
) -> None:
    """Write everything needed to either resume training or run inference.

    `config` is here because you need d_model/nhead/num_decoder_layers to
    *construct* the model before load_state_dict will accept the weights;
    `vocab_path` because token id 47 means nothing on its own, and a
    rebuilt vocab reorders ids and silently produces fluent nonsense.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": lr_sched.state_dict(),
            "epoch": epoch,
            "best_exprate": best_exprate,
            "config": asdict(config),
            "vocab_path": config.data.vocab_path,
        },
        path,
    )


def check_label_lengths(loader: DataLoader, max_seq_len: int) -> None:
    """Fail now, with a useful message, rather than on a shape assert later.

    collate adds one position for sos/eos, and the decoder's causal mask and
    positional encoding are only built out to max_seq_len. The longest real
    CROHME train label is 204 tokens, so a default of 100 blows up partway
    into the first epoch -- on whichever batch happens to contain it.
    """
    lengths = label_lengths(loader)
    if not lengths:
        return
    longest, label = lengths[0]
    if longest + 1 > max_seq_len:
        raise SystemExit(
            f"the training data contains a {longest}-token label, which needs "
            f"max_seq_len >= {longest + 1} (currently {max_seq_len}).\n"
            f"Either raise --set model.max_seq_len={longest + 1} or drop the "
            f"sample.\n  {label[:120]}"
        )


def build_schedule(
    opt: torch.optim.Optimizer, warmup: int, total_steps: int
) -> torch.optim.lr_scheduler.LRScheduler:
    """Linear warmup then cosine decay, stepped once per batch.

    Post-norm transformers are famously unstable in the first few hundred
    steps without the warmup. Cosine's T_max is in *steps*, not epochs --
    get that wrong and the LR anneals to zero in the first epoch.
    """
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, total_steps - warmup)
    if warmup <= 0:
        return cosine
    return torch.optim.lr_scheduler.SequentialLR(
        opt,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(opt, start_factor=1e-3, total_iters=warmup),
            cosine,
        ],
        milestones=[warmup],
    )


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    lr_sched: torch.optim.lr_scheduler.LRScheduler,
    ce_loss: torch.nn.Module,
    pad_id: int,
    device: torch.device,
    writer: SummaryWriter,
    step: int,
    desc: str,
) -> tuple[int, float, float]:
    """One pass over the training split -> (next step, epoch loss, token acc).

    Both epoch figures are token-weighted rather than a mean of batch means:
    T varies per batch, so averaging batch losses over-weights batches of
    short formulas.
    """
    model.train()
    loss_sum, correct, tokens = 0.0, 0, 0

    bar = tqdm(loader, desc=desc, leave=False)
    for imgs, img_mask, tgt_in, tgt_out, tgt_mask in bar:
        imgs, img_mask = imgs.to(device), img_mask.to(device)
        tgt_in, tgt_out = tgt_in.to(device), tgt_out.to(device)
        tgt_mask = tgt_mask.to(device)

        opt.zero_grad()
        logits = model(imgs, img_mask, tgt_in, tgt_mask)
        loss = ce_loss(logits.transpose(1, 2), tgt_out)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        lr_sched.step()

        real = tgt_out != pad_id
        n_tokens = int(real.sum())
        with torch.no_grad():
            n_correct = int(((logits.argmax(-1) == tgt_out) & real).sum())

        loss_sum += loss.item() * n_tokens
        correct += n_correct
        tokens += n_tokens

        batch_acc = n_correct / max(n_tokens, 1)
        writer.add_scalar("train/loss", loss.item(), step)
        writer.add_scalar("train/token_acc", batch_acc, step)
        writer.add_scalar("train/lr", lr_sched.get_last_lr()[0], step)
        bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{batch_acc:.3f}")
        step += 1

    return step, loss_sum / max(tokens, 1), correct / max(tokens, 1)


@torch.no_grad()
def validation_loss(
    model: torch.nn.Module,
    loader: DataLoader,
    ce_loss: torch.nn.Module,
    pad_id: int,
    device: torch.device,
) -> float:
    """Teacher-forced loss over the val split -- cheap, and the signal for
    overfitting. ExpRate is the metric that matters, but it is slow and sits
    at 0 early in training, when loss is still the only thing moving."""
    model.eval()
    loss_sum, tokens = 0.0, 0
    for imgs, img_mask, tgt_in, tgt_out, tgt_mask in loader:
        imgs, img_mask = imgs.to(device), img_mask.to(device)
        tgt_in, tgt_out = tgt_in.to(device), tgt_out.to(device)
        tgt_mask = tgt_mask.to(device)
        logits = model(imgs, img_mask, tgt_in, tgt_mask)
        n = int((tgt_out != pad_id).sum())
        loss_sum += ce_loss(logits.transpose(1, 2), tgt_out).item() * n
        tokens += n
    return loss_sum / max(tokens, 1)


def log_validation(
    writer: SummaryWriter,
    epoch: int,
    val_loss: float,
    metrics: dict,
    samples: list[tuple[str, str]],
) -> None:
    writer.add_scalar("val/loss", val_loss, epoch)
    writer.add_scalar("val/exprate", metrics["exprate"], epoch)
    writer.add_scalar("val/exprate_leq1", metrics["exprate_leq1"], epoch)
    writer.add_scalar("val/exprate_leq2", metrics["exprate_leq2"], epoch)
    writer.add_scalar("val/ser", metrics["symbol_error_rate"], epoch)
    writer.add_text(
        "val/predictions",
        "\n\n".join(f"ref  `{r}`\n\npred `{p}`" for r, p in samples),
        epoch,
    )


def score_best_checkpoint(
    checkpoint: Path, config: ExperimentConfig, vocab: Vocab, device: torch.device
) -> dict:
    """Reload the best checkpoint and score it on the *full* val and test
    splits. The per-epoch ExpRate is a subset estimate; these are the numbers
    to report."""
    print(f"\nScoring '{checkpoint}' on the full val and test splits...")
    model, _, _ = load_for_inference(checkpoint, device)

    results: dict = {}
    for split, data_dir in (("val", config.data.val_dir), ("test", config.data.test_dir)):
        metrics, _ = evaluate(model, make_loader(data_dir, vocab, config), vocab, device)
        print(f"  {split:<5} {format_metrics(metrics)}")
        results[f"final_{split}/exprate"] = metrics["exprate"]
        results[f"final_{split}/exprate_leq1"] = metrics["exprate_leq1"]
        results[f"final_{split}/exprate_leq2"] = metrics["exprate_leq2"]
        results[f"final_{split}/ser"] = metrics["symbol_error_rate"]
    return results


def train(config: ExperimentConfig) -> None:
    set_seed(config.train.seed)
    device = pick_device(config.train.device)
    print(f"Training on device: {device}")

    vocab = Vocab.load(config.data.vocab_path)
    pad_id = vocab.stoi[PAD]
    print(f"Loaded vocab from '{config.data.vocab_path}' with {len(vocab)} symbols")

    train_dl = train_loader(config, vocab)
    val_dl = val_loader(config, vocab)
    exprate_dl = val_loader(config, vocab, limit=config.train.val_exprate_samples)
    check_label_lengths(train_dl, config.model.max_seq_len)
    print(
        f"Train '{config.data.train_dir}' {len(train_dl.dataset)} samples, "
        f"val '{config.data.val_dir}' {len(val_dl.dataset)} samples, "
        f"batch_size={config.data.batch_size}, batches={len(train_dl)}"
    )

    model = build_model(config, len(vocab)).to(device)
    print(
        f"Built {config.model.encoder} encoder + {config.model.num_decoder_layers}-layer "
        f"decoder, d_model={config.model.d_model}, nhead={config.model.nhead}"
    )
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    num_epochs = config.train.epochs
    total_steps = max(num_epochs * len(train_dl), 1)
    opt = torch.optim.AdamW(
        model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay
    )
    lr_sched = build_schedule(
        opt, min(config.train.warmup_steps, total_steps - 1), total_steps
    )
    ce_loss = torch.nn.CrossEntropyLoss(ignore_index=pad_id)

    ckpt_dir = Path(config.train.checkpoint_dir) / config.train.run_name
    writer = SummaryWriter(log_dir=Path(config.train.log_dir) / config.train.run_name)
    writer.add_text("config", f"```\n{config}\n```")
    print(f"Logging to '{writer.log_dir}', checkpointing to '{ckpt_dir}'")

    step = 0
    best_exprate = -1.0
    best_val_loss = float("inf")
    final_metrics: dict = {}

    for epoch_i in range(num_epochs):
        step, epoch_loss, epoch_acc = train_one_epoch(
            model, train_dl, opt, lr_sched, ce_loss, pad_id, device, writer,
            step, desc=f"epoch {epoch_i + 1}/{num_epochs}",
        )
        writer.add_scalar("train/loss_epoch", epoch_loss, epoch_i)
        writer.add_scalar("train/token_acc_epoch", epoch_acc, epoch_i)

        val_loss = validation_loss(model, val_dl, ce_loss, pad_id, device)
        val_metrics, samples = evaluate(model, exprate_dl, vocab, device)
        log_validation(writer, epoch_i, val_loss, val_metrics, samples)
        print(
            f"epoch {epoch_i + 1}/{num_epochs}  train_loss={epoch_loss:.4f}  "
            f"token_acc={epoch_acc:.4f}  val_loss={val_loss:.4f}  "
            f"{format_metrics(val_metrics)}  lr={lr_sched.get_last_lr()[0]:.2e}"
        )

        save_checkpoint(ckpt_dir / "last.pt", model, opt, lr_sched, config, epoch_i, best_exprate)
        # ExpRate first, val loss as the tie-break -- early on every epoch
        # ties at 0.0 ExpRate, and loss is the only thing distinguishing them.
        exprate = val_metrics["exprate"]
        if exprate > best_exprate or (exprate == best_exprate and val_loss < best_val_loss):
            best_exprate, best_val_loss = exprate, val_loss
            save_checkpoint(ckpt_dir / "best.pt", model, opt, lr_sched, config, epoch_i, best_exprate)

        final_metrics = {"val/exprate": exprate, "val/loss": val_loss}

    final_metrics.update(score_best_checkpoint(ckpt_dir / "best.pt", config, vocab, device))
    writer.add_hparams(config.flat(), final_metrics)
    writer.close()
    print(f"\nDone. best val ExpRate={best_exprate:.4f}, checkpoints in '{ckpt_dir}'")


def main() -> None:
    train(ExperimentConfig.from_cli(sys.argv[1:]))


if __name__ == "__main__":
    main()
