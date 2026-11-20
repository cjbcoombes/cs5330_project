import pytest
import torch

from math import cos, sin
from hmer_latex.models.decoder import (
    PositionalEncoding1D,
    PositionalEncoding2D,
    TransformerDecoder,
)
from hmer_latex.models.encoder import SimpleCNNEncoder


def no_pad(target_ids: torch.Tensor) -> torch.Tensor:
    """The `target_ids_mask` for a batch with no padding in it.

    `tgt_key_padding_mask` semantics (and collate_batch's): True marks a
    position to be *ignored*, so an unpadded batch is all-False. Most tests
    below are about something other than padding, and pass this.
    """
    return torch.zeros(target_ids.shape, dtype=torch.bool)


def pe2d_expected(H, W, C, seq_denom=10000.0):
    """Hand-build the expected 2D encoding, independently of the module.

    Row h gets position h and column w gets position w -- absolute indices,
    not normalized by the grid extent (see
    test_positional_encoding_2d_is_independent_of_grid_extent). Channels
    [0, C/2) carry the row encoding and [C/2, C) the column encoding,
    interleaved sin/cos, with frequency i using 1 / seq_denom ** (2i / C).
    """
    half = C // 2
    inv_freq = [1.0 / seq_denom ** (2 * i / C) for i in range(C // 4)]

    expected = torch.zeros(C, H, W)
    for h in range(H):
        for i, f in enumerate(inv_freq):
            expected[2 * i, h, :] = sin(h * f)
            expected[2 * i + 1, h, :] = cos(h * f)
    for w in range(W):
        for i, f in enumerate(inv_freq):
            expected[half + 2 * i, :, w] = sin(w * f)
            expected[half + 2 * i + 1, :, w] = cos(w * f)
    return expected


def test_positional_encoding_2d_explicit_values_small():
    """Checks PositionalEncoding2D against hand-computed sin/cos values.

    With C=4 there is a single frequency per axis (i=0, so denom = 1
    whatever seq_denom is), which pins down the row/column channel split
    without involving frequency scaling.
    """
    B, C, H, W = 2, 4, 2, 3

    pe = PositionalEncoding2D(d_model=C, max_len=max(H, W))
    expected = pe2d_expected(H, W, C).expand(B, C, H, W)

    torch.testing.assert_close(pe(torch.zeros(B, C, H, W)), expected)
    # ...and the encoding is *added* to the features, not substituted for them.
    torch.testing.assert_close(pe(torch.ones(B, C, H, W)), expected + 1.0)


def test_positional_encoding_2d_explicit_values_multi_frequency():
    """Same, but C=8 gives two sin/cos pairs per axis, pinning down the
    per-channel frequency scaling (and so the seq_denom base and exponent).
    """
    B, C, H, W = 2, 8, 2, 3

    pe = PositionalEncoding2D(d_model=C, max_len=max(H, W))
    out = pe(torch.zeros(B, C, H, W))

    torch.testing.assert_close(out, pe2d_expected(H, W, C).expand(B, C, H, W))
    # Pin one value numerically, so a self-consistent-but-wrong helper can't
    # make this pass: channel 0, row 1 is sin(1 * 1.0).
    assert out[0, 0, 1, 0].item() == pytest.approx(sin(1.0))


def test_positional_encoding_2d_is_independent_of_grid_extent():
    """A cell's encoding must not depend on how big its grid is.

    Images in a batch are padded to the batch maximum (collate_batch), so
    an encoding that divided positions by H and W -- as this one used to --
    would give the same pixel a different encoding depending on which other
    samples happened to share its batch.
    """
    pe = PositionalEncoding2D(d_model=8)

    padded = pe(torch.zeros(1, 8, 4, 6))
    tight = pe(torch.zeros(1, 8, 4, 3))

    torch.testing.assert_close(padded[:, :, :, :3], tight)


def test_positional_encoding_2d_covers_the_real_feature_grid():
    """max_len caps the precomputed table, and the encoder divides image
    size by 16 or 32 -- so the default has to clear a 512x64 image."""
    pe = PositionalEncoding2D(d_model=8)

    assert pe(torch.zeros(1, 8, 64 // 16, 512 // 16)).shape == (1, 8, 4, 32)


def test_positional_encoding_2d_does_not_mutate_its_input():
    """A pos-enc layer must be a pure function of its input.

    If it writes into `features` in place, two things break: the caller's
    tensor silently changes under it, and (worse) autograd refuses to
    backprop through the encoder, because the thing being overwritten is
    the saved output of a ReLU.
    """
    pe = PositionalEncoding2D(d_model=8)
    features = torch.randn(2, 8, 4, 6)
    snapshot = features.clone()

    pe(features)

    torch.testing.assert_close(features, snapshot)


def test_positional_encoding_2d_backward_through_encoder():
    """The real failure mode of an in-place pos-enc: loss.backward() raises.

    Shape tests pass and the forward pass looks fine; this is the first
    thing that actually catches it.
    """
    enc = SimpleCNNEncoder(out_channels=8)
    pe = PositionalEncoding2D(d_model=8)

    out = pe(enc(torch.randn(1, 1, 128, 384)))
    out.sum().backward()

    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in enc.parameters())


def test_positional_encoding_1d_explicit_values():
    """Hand-computed sin/cos for PositionalEncoding1D, C=4, L=3.

    Mirrors the 2D tests: channel pairs are interleaved sin/cos, and
    channel pair i uses frequency 1 / seq_denom ** (2i / C). With C=4
    there are two pairs (i = 0, 1); seq_denom=100 makes denom_1 = 0.1,
    so the second frequency is a clean tenth of the first.

    Unlike the 2D encoding, positions here are *not* normalized by L --
    a token's encoding must not depend on how long its sequence happens
    to be (see test_positional_encoding_1d_is_prefix_consistent).

    Expected values are plain math.sin/cos, computed independently.
    """
    C, L = 4, 3
    pe = PositionalEncoding1D(d_model=C, seq_denom=100, max_len=L)

    # denom_0 = 100 ** (-0/4) = 1.0 ; denom_1 = 100 ** (-2/4) = 0.1
    expected = torch.tensor(
        [
            [sin(0 * 1.0), sin(1 * 1.0), sin(2 * 1.0)],  # ch0: sin, freq 1.0
            [cos(0 * 1.0), cos(1 * 1.0), cos(2 * 1.0)],  # ch1: cos, freq 1.0
            [sin(0 * 0.1), sin(1 * 0.1), sin(2 * 0.1)],  # ch2: sin, freq 0.1
            [cos(0 * 0.1), cos(1 * 0.1), cos(2 * 0.1)],  # ch3: cos, freq 0.1
        ]
    )

    out = pe(torch.zeros(2, C, L))
    torch.testing.assert_close(out, expected.expand(2, C, L).contiguous())


def test_positional_encoding_1d_is_additive_and_pure():
    """Output = input + (the same) encoding, and the input is untouched."""
    pe = PositionalEncoding1D(d_model=8, max_len=20)

    zeros = torch.zeros(1, 8, 5)
    encoding = pe(zeros).clone()

    features = torch.randn(3, 8, 5)
    snapshot = features.clone()
    out = pe(features)

    torch.testing.assert_close(out, features + encoding)
    torch.testing.assert_close(features, snapshot)


def test_positional_encoding_1d_is_prefix_consistent():
    """Position t's encoding must be the same at any sequence length.

    `T` is per-batch (collate pads to the longest label in the batch) and
    generate() grows the sequence one token at a time, so if the encoding
    of position 2 depends on L, the same token gets a different embedding
    in training than at decode time -- and a different one again depending
    on which batch it landed in.
    """
    pe = PositionalEncoding1D(d_model=8, max_len=20)

    short = pe(torch.zeros(1, 8, 3))
    long = pe(torch.zeros(1, 8, 12))

    torch.testing.assert_close(short, long[:, :, :3])


def test_positional_encodings_are_registered_buffers():
    """Pos-enc tables must be registered buffers, not plain attributes.

    `.to(device)` only moves parameters and buffers. A plain tensor
    attribute stays on the CPU, so `model.to("mps")` leaves it behind and
    the first forward pass dies on a device mismatch -- which is invisible
    on a CPU-only test run. Checking `named_buffers()` catches it without
    needing an accelerator.

    (Buffers are also what puts the table in `state_dict()`, though
    `persistent=False` is the right call for a table that is fully
    determined by d_model/seq_denom.)
    """
    pe1 = PositionalEncoding1D(d_model=8, max_len=10)
    pe2 = PositionalEncoding2D(d_model=8)

    assert [t for _, t in pe1.named_buffers()], "PositionalEncoding1D holds no buffers"
    assert [t for _, t in pe2.named_buffers()], "PositionalEncoding2D holds no buffers"

    # Nothing float and persistent should be hiding as a plain attribute.
    for module in (pe1, pe2):
        registered = {id(t) for _, t in module.named_buffers()}
        stray = [
            name
            for name, value in vars(module).items()
            if isinstance(value, torch.Tensor) and id(value) not in registered
        ]
        assert stray == [], f"{type(module).__name__} has unregistered tensors: {stray}"


def test_transformer_decoder_output_shape():
    """(B, L, C) memory + (B, T) target ids -> (B, T, V) logits.

    L=48 and C=64 are the real MVP numbers, and deliberately differ, so a
    transposed `memory` is a shape error rather than a silent bug.
    """
    B, L, C, T, V = 2, 48, 64, 7, 138

    dec = TransformerDecoder(vocab_size=V, d_model=C, nhead=1, num_layers=1)
    target = torch.randint(0, V, (B, T))
    logits = dec(torch.randn(B, L, C), target, no_pad(target))

    assert logits.shape == (B, T, V)


def test_transformer_decoder_shape_for_varying_T():
    """T is per-batch, so forward() has to handle a changing target length."""
    B, L, C, V = 3, 48, 64, 138
    dec = TransformerDecoder(vocab_size=V, d_model=C, nhead=1, num_layers=1)
    memory = torch.randn(B, L, C)

    for T in (1, 2, 13, 64):
        target = torch.randint(0, V, (B, T))
        logits = dec(memory, target, no_pad(target))
        assert logits.shape == (B, T, V), f"T={T}"


def test_transformer_decoder_is_causal():
    """Logits at position t must not depend on target tokens after t.

    This is the one property the causal mask exists to guarantee, and the
    one whose absence is invisible in training (loss just drops faster
    than it should) and fatal at generation time. Changing the *last*
    target token must leave every earlier position's logits identical.
    """
    B, L, C, T, V = 1, 48, 64, 6, 138

    dec = TransformerDecoder(vocab_size=V, d_model=C, nhead=1, num_layers=1).eval()
    memory = torch.randn(B, L, C)
    target = torch.randint(0, V, (B, T))

    tampered = target.clone()
    tampered[0, -1] = (target[0, -1] + 1) % V

    with torch.no_grad():
        a = dec(memory, target, no_pad(target))
        b = dec(memory, tampered, no_pad(tampered))

    torch.testing.assert_close(a[:, :-1, :], b[:, :-1, :])
    # ...and the sanity half: the tampered position itself *should* differ,
    # otherwise this test would also pass on a decoder that ignores input.
    assert not torch.allclose(a[:, -1, :], b[:, -1, :])


def test_transformer_decoder_prefix_matches_full_sequence():
    """Greedy decode relies on this: feeding a prefix of length t gives the
    same logits at position t-1 as feeding the full sequence does.

    If it doesn't hold, train-time and generate-time paths diverge and the
    model scores well on teacher-forced loss while decoding nonsense.
    """
    B, L, C, T, V = 1, 48, 64, 8, 138

    dec = TransformerDecoder(vocab_size=V, d_model=C, nhead=1, num_layers=1).eval()
    memory = torch.randn(B, L, C)
    target = torch.randint(0, V, (B, T))

    with torch.no_grad():
        full = dec(memory, target, no_pad(target))
        prefix = dec(memory, target[:, :4], no_pad(target[:, :4]))

    torch.testing.assert_close(full[:, :4, :], prefix, rtol=1e-4, atol=1e-5)


def test_transformer_decoder_ignores_masked_padding():
    """A masked tail must not affect the logits of the real prefix.

    This is the whole point of `target_ids_mask`: collate pads every label
    out to the batch's longest one, so without the mask a sample's
    activations depend on what else landed in its batch (padded positions
    still take part in attention normalization). Feeding a 4-token sequence
    padded to 9 with the pad positions masked must reproduce the unpadded
    4-token logits.
    """
    B, L, C, T, V, PAD = 1, 48, 64, 4, 138, 0

    dec = TransformerDecoder(vocab_size=V, d_model=C, nhead=1, num_layers=1).eval()
    memory = torch.randn(B, L, C)
    target = torch.randint(1, V, (B, T))

    padded = torch.cat([target, torch.full((B, 5), PAD)], dim=1)
    mask = padded == PAD

    with torch.no_grad():
        short = dec(memory, target, no_pad(target))
        long = dec(memory, padded, mask)

    torch.testing.assert_close(short, long[:, :T, :], rtol=1e-4, atol=1e-5)


def test_transformer_decoder_padding_mask_changes_output():
    """The sanity half: the mask has to actually reach the attention.

    Without this, test_transformer_decoder_ignores_masked_padding would
    also pass on a decoder that silently drops `target_ids_mask` -- the
    causal mask alone already keeps position t from seeing the tail. Here
    the *masked* positions are early ones, which causality does not hide,
    so ignoring the argument changes the result.
    """
    B, L, C, T, V = 1, 48, 64, 6, 138

    dec = TransformerDecoder(vocab_size=V, d_model=C, nhead=1, num_layers=1).eval()
    memory = torch.randn(B, L, C)
    target = torch.randint(1, V, (B, T))

    mask = torch.zeros(B, T, dtype=torch.bool)
    mask[0, 1:3] = True  # pretend positions 1-2 are padding

    with torch.no_grad():
        unmasked = dec(memory, target, no_pad(target))
        masked = dec(memory, target, mask)

    assert not torch.allclose(unmasked[:, -1, :], masked[:, -1, :])


def test_transformer_decoder_accepts_mask_for_varying_T():
    """Mask shape is (B, T) and T is per-batch, like target_ids."""
    B, L, C, V = 3, 48, 64, 138
    dec = TransformerDecoder(vocab_size=V, d_model=C, nhead=1, num_layers=1)
    memory = torch.randn(B, L, C)

    for T in (1, 2, 13, 64):
        target = torch.randint(0, V, (B, T))
        mask = torch.zeros(B, T, dtype=torch.bool)
        mask[:, T // 2 :] = True
        assert dec(memory, target, mask).shape == (B, T, V), f"T={T}"
