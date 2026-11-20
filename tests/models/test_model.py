"""End-to-end shape and gradient tests for HMERModel.

These are the cheapest possible stand-in for "the training loop runs":
real MVP dimensions, one forward, one backward. Anything that breaks the
image -> encoder -> 2D pos-enc -> memory -> decoder -> logits chain shows
up here without needing data on disk.
"""

import pytest
import torch

from hmer_latex.models.model import HMERModel

# A 384x128 image through SimpleCNNEncoder's /32 downsampling -> 4x12 = 48
# memory positions, d_model=64, vocab 138. Real images vary in size; this is
# just a fixed one to assert shapes against.
# L=48 != C=64 on purpose, so a missing .permute() between the encoder and
# the decoder is a shape error.
V, D = 138, 64
IMG_H, IMG_W = 128, 384
PAD, SOS, EOS = 0, 1, 2


def mvp(**kwargs) -> HMERModel:
    """A `simple_mvp`-shaped model: the cheap one these tests run on."""
    return HMERModel(vocab_size=V, encoder="simple", d_model=D, **kwargs)


def no_pad(target_ids: torch.Tensor) -> torch.Tensor:
    """The `target_ids_mask` for a batch with no padding in it.

    True marks a position to be ignored (`tgt_key_padding_mask` semantics,
    matching collate_batch), so an unpadded batch is all-False.
    """
    return torch.zeros(target_ids.shape, dtype=torch.bool)


def test_model_forward_shape():
    model = mvp(nhead=1, num_decoder_layers=1)

    target = torch.randint(0, V, (2, 7))
    logits = model(torch.randn(2, 1, IMG_H, IMG_W), None, target, no_pad(target))

    assert logits.shape == (2, 7, V)


def test_model_encoder_grid_is_48_positions():
    """Pins the downsampling arithmetic the docs assume.

    If this changes, the memory-shape notes in
    docs/training-and-eval-loop.md are stale.
    """
    model = mvp()

    features = model.encoder(torch.randn(1, 1, IMG_H, IMG_W))

    assert features.shape == (1, D, 4, 12)


@pytest.mark.parametrize("T", [1, 2, 13, 52])
def test_model_forward_shape_for_varying_T(T):
    """T is per-batch (collate pads to the batch's longest label)."""
    model = mvp()

    target = torch.randint(0, V, (2, T))
    logits = model(torch.randn(2, 1, IMG_H, IMG_W), None, target, no_pad(target))

    assert logits.shape == (2, T, V)


def test_model_backward_reaches_every_parameter():
    """One backward pass; every parameter must get a gradient.

    A parameter with `grad is None` is disconnected from the loss -- the
    classic symptom of a submodule that was built but never called, or of
    a tensor rebuilt inside forward() instead of being a buffer. A zero
    gradient is also suspicious, but legitimately happens (dead ReLU,
    unused embedding rows), so only None is treated as a failure.
    """
    model = mvp()

    target = torch.randint(0, V, (2, 7))
    logits = model(torch.randn(2, 1, IMG_H, IMG_W), None, target, no_pad(target))
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
    model = mvp()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD)

    images = torch.randn(1, 1, IMG_H, IMG_W)
    tgt_in = torch.tensor([[SOS, 10, 11, 12]])   # <sos> a b c
    tgt_out = torch.tensor([[10, 11, 12, EOS]])  # a b c <eos>
    tgt_mask = tgt_in == PAD

    losses = []
    for _ in range(30):
        opt.zero_grad()
        logits = model(images, None, tgt_in, tgt_mask)
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
    model = mvp().eval()

    clone = mvp().eval()
    clone.load_state_dict(model.state_dict())

    images = torch.randn(1, 1, IMG_H, IMG_W)
    target = torch.randint(0, V, (1, 7))
    mask = no_pad(target)
    with torch.no_grad():
        torch.testing.assert_close(
            model(images, None, target, mask), clone(images, None, target, mask)
        )


def test_model_forward_accepts_a_padded_batch():
    """What the training loop actually hands forward(): collate's tensors.

    tgt_in / tgt_out / tgt_pad_mask all come out of collate_batch at (B, T)
    with True on the padded tail, so the mask argument has to line up with
    target_ids position for position.
    """
    model = mvp()

    tgt_in = torch.tensor(
        [
            [SOS, 10, 11, 12, 13],
            [SOS, 20, 21, PAD, PAD],
        ]
    )
    logits = model(torch.randn(2, 1, IMG_H, IMG_W), None, tgt_in, tgt_in == PAD)

    assert logits.shape == (2, 5, V)


def test_model_padding_does_not_leak_across_the_batch():
    """A sample's logits must not depend on what else is in its batch.

    Without `target_ids_mask`, a shorter label's padded tail still takes
    part in attention normalization, so the same (image, label) pair scores
    differently depending on how long the batch's longest label happened to
    be. Same image and same 3-token label, once alone and once batched
    beside a longer one: the first sample's real positions must match.
    """
    torch.manual_seed(0)
    model = mvp().eval()

    image = torch.randn(1, 1, IMG_H, IMG_W)
    short = torch.tensor([[SOS, 10, 11]])

    padded = torch.tensor(
        [
            [SOS, 10, 11, PAD, PAD, PAD],
            [SOS, 20, 21, 22, 23, 24],
        ]
    )
    images = torch.cat([image, torch.randn(1, 1, IMG_H, IMG_W)], dim=0)

    with torch.no_grad():
        alone = model(image, None, short, no_pad(short))
        batched = model(images, None, padded, padded == PAD)

    torch.testing.assert_close(alone[0], batched[0, :3], rtol=1e-4, atol=1e-5)


def test_model_generate_greedy_shape():
    """generate builds its own all-False target mask as the sequence grows
    (nothing is target padding during decode), so it takes no target mask.
    """
    model = mvp(max_seq_len=6).eval()

    out = model.generate(torch.randn(2, 1, IMG_H, IMG_W), None, sos_id=SOS, eos_id=EOS)

    assert out.ndim == 2 and out.shape[0] == 2
    assert 2 <= out.shape[1] <= 7
    assert (out[:, 0] == SOS).all()


# --- variable-size images: padding must not leak between batch members ---


def densenet(**kwargs) -> HMERModel:
    """A deliberately tiny DenseNet model, for mask arithmetic only."""
    return HMERModel(
        vocab_size=V, encoder="densenet", d_model=D, nhead=2,
        num_decoder_layers=1, growth_rate=4, encoder_layers=2, **kwargs
    )


@pytest.mark.parametrize("encoder", ["simple", "densenet"])
@pytest.mark.parametrize("size", [(64, 128), (128, 384), (96, 256)])
def test_mask_grid_matches_the_feature_grid(encoder, size):
    """memory_pad_mask pools img_pad_mask down by encoder.total_stride and
    expects to land exactly on the encoder's output grid. If the two
    disagree the memory and its mask describe different cells, and
    attention silently masks the wrong positions -- so it raises instead.
    """
    model = densenet() if encoder == "densenet" else mvp()
    H, W = size

    features = model.encoder(torch.randn(1, 1, H, W))
    mem_mask = model.memory_pad_mask(torch.zeros(1, H, W, dtype=torch.bool), features)

    assert mem_mask.shape[1] == features.shape[-1] * features.shape[-2]
    assert mem_mask.shape[1] == (H // model.encoder.total_stride) * (
        W // model.encoder.total_stride
    )


@pytest.mark.parametrize("encoder", ["simple", "densenet"])
def test_model_image_padding_does_not_leak_across_the_batch(encoder):
    """A sample's logits must not depend on how wide its batch-mates were.

    This is the image-side twin of test_model_padding_does_not_leak_across_
    the_batch, and the core correctness property of variable-size input: the
    same image, once alone and once padded out beside a wider one, has to
    produce identical logits.

    Two conditions make this hold, and both matter if you extend this test.
    The sample's own size is a multiple of the encoder's total stride (which
    collate_batch guarantees): zero padding then matches what the
    convolutions pad with anyway, and no pooling window straddles the
    boundary. And the model is in eval(), so BatchNorm uses running
    statistics rather than batch statistics that the padded region would
    pollute.

    The comparison is still only approximate: cuDNN/MPS pick different
    tiling for different input widths, so the float accumulation order
    changes even where the arithmetic is identical. SimpleCNN happens to
    come out bit-exact, DenseNet lands ~1e-5 away.
    """
    model = (densenet() if encoder == "densenet" else mvp()).eval()
    stride = model.encoder.total_stride
    h, w = 2 * stride, 4 * stride

    image = torch.randn(1, 1, h, w)
    target = torch.randint(0, V, (1, 5))

    # The same image, padded out to twice the width beside a wider sample.
    wide = torch.zeros(2, 1, h, 2 * w)
    wide[0, :, :, :w] = image[0]
    wide[1] = torch.randn(1, h, 2 * w)
    pad_mask = torch.zeros(2, h, 2 * w, dtype=torch.bool)
    pad_mask[0, :, w:] = True

    targets = target.repeat(2, 1)

    with torch.no_grad():
        alone = model(image, torch.zeros(1, h, w, dtype=torch.bool), target, no_pad(target))
        batched = model(wide, pad_mask, targets, no_pad(targets))

    torch.testing.assert_close(alone[0], batched[0], rtol=1e-3, atol=1e-4)


def test_model_ignoring_the_image_mask_would_change_the_answer():
    """The sanity half of the test above: the mask has to reach attention.

    Without it, the padded sample's logits would be computed over blank
    padding as if it were image content, so passing the mask must make a
    difference.
    """
    model = mvp().eval()
    stride = model.encoder.total_stride
    h, w = 2 * stride, 4 * stride

    images = torch.randn(1, 1, h, 2 * w)
    pad_mask = torch.zeros(1, h, 2 * w, dtype=torch.bool)
    pad_mask[0, :, w:] = True
    target = torch.randint(0, V, (1, 5))

    with torch.no_grad():
        masked = model(images, pad_mask, target, no_pad(target))
        unmasked = model(images, None, target, no_pad(target))

    assert not torch.allclose(masked, unmasked)


def test_generate_stays_within_max_seq_len():
    """The sequence must not outgrow the decoder's causal mask / positional
    encoding. An untrained head rarely emits eos -- the epoch-1 regime --
    so the loop bound is what keeps the first validation pass alive."""
    model = mvp(max_seq_len=8).eval()

    out = model.generate(torch.randn(3, 1, IMG_H, IMG_W), None, SOS, EOS)

    assert out.shape[1] <= model.max_seq_len + 1


def test_unknown_encoder_is_rejected():
    with pytest.raises(ValueError, match="unknown encoder"):
        HMERModel(vocab_size=V, encoder="resnet")
