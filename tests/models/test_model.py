"""End-to-end shape and gradient tests for HMERModel.

These are the cheapest possible stand-in for "the training loop runs":
real MVP dimensions, one forward, one backward. Anything that breaks the
image -> encoder -> 2D pos-enc -> memory -> decoder -> logits chain shows
up here without needing data on disk.
"""

import pytest
import torch

from hmer_latex.models.model import HMERModel

# Real MVP numbers: 384x128 canvas, /32 downsampling -> 4x12 = 48 memory
# positions, d_model=64, vocab 138. L=48 != C=64 on purpose, so a missing
# .permute() between the encoder and the decoder is a shape error.
V, D = 138, 64
IMG_H, IMG_W = 128, 384


def test_model_forward_shape():
    model = HMERModel(vocab_size=V, d_model=D, nhead=1, num_decoder_layers=1)

    logits = model(torch.randn(2, 1, IMG_H, IMG_W), torch.randint(0, V, (2, 7)))

    assert logits.shape == (2, 7, V)


def test_model_encoder_grid_is_48_positions():
    """Pins the canvas/downsampling arithmetic the rest of the docs assume.

    If this changes, `max_img_len` and the memory-shape notes in
    docs/training-and-eval-loop.md are stale.
    """
    model = HMERModel(vocab_size=V, d_model=D)

    features = model.encoder(torch.randn(1, 1, IMG_H, IMG_W))

    assert features.shape == (1, D, 4, 12)


@pytest.mark.parametrize("T", [1, 2, 13, 52])
def test_model_forward_shape_for_varying_T(T):
    """T is per-batch (collate pads to the batch's longest label)."""
    model = HMERModel(vocab_size=V, d_model=D)

    logits = model(torch.randn(2, 1, IMG_H, IMG_W), torch.randint(0, V, (2, T)))

    assert logits.shape == (2, T, V)


def test_model_backward_reaches_every_parameter():
    """One backward pass; every parameter must get a gradient.

    A parameter with `grad is None` is disconnected from the loss -- the
    classic symptom of a submodule that was built but never called, or of
    a tensor rebuilt inside forward() instead of being a buffer. A zero
    gradient is also suspicious, but legitimately happens (dead ReLU,
    unused embedding rows), so only None is treated as a failure.
    """
    model = HMERModel(vocab_size=V, d_model=D)

    logits = model(torch.randn(2, 1, IMG_H, IMG_W), torch.randint(0, V, (2, 7)))
    logits.sum().backward()

    missing = [name for name, p in model.named_parameters() if p.grad is None]
    assert missing == []


def test_model_cross_entropy_step_reduces_loss_on_one_sample():
    """The smallest possible overfit test: can the model learn one pair?

    This is Phase 4 (train loss -> ~0 on crohme_tiny) compressed into a
    unit test -- same loss, same optimizer, one fixed (image, label) pair
    and no data on disk. If this can't drive the loss down, nothing about
    the real overfit run will work either, and the bug is in the wiring,
    not the architecture.
    """
    torch.manual_seed(0)
    model = HMERModel(vocab_size=V, d_model=D)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

    images = torch.randn(1, 1, IMG_H, IMG_W)
    tgt_in = torch.tensor([[1, 10, 11, 12]])   # <sos> a b c
    tgt_out = torch.tensor([[10, 11, 12, 2]])  # a b c <eos>

    losses = []
    for _ in range(30):
        opt.zero_grad()
        logits = model(images, tgt_in)
        loss = criterion(logits.reshape(-1, V), tgt_out.reshape(-1))
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0] / 2, f"loss went {losses[0]:.3f} -> {losses[-1]:.3f}"


def test_model_state_dict_roundtrips():
    """Checkpointing depends on this: rebuild from config, load weights,
    get identical outputs. Anything held as a plain tensor attribute
    instead of a buffer silently drops out of state_dict.
    """
    torch.manual_seed(0)
    model = HMERModel(vocab_size=V, d_model=D).eval()

    clone = HMERModel(vocab_size=V, d_model=D).eval()
    clone.load_state_dict(model.state_dict())

    images = torch.randn(1, 1, IMG_H, IMG_W)
    target = torch.randint(0, V, (1, 7))
    with torch.no_grad():
        torch.testing.assert_close(model(images, target), clone(images, target))
