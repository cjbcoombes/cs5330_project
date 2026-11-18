import torch

from math import sin, cos
from hmer_latex.models.decoder import (
    PositionalEncoding1D,
    PositionalEncoding2D,
    TransformerDecoder,
)
from hmer_latex.models.encoder import SimpleCNNEncoder


def test_positional_encoding_2d_explicit_values_small():
    """Checks PositionalEncoding2D against hand-computed sin/cos values.

    Per BTTR (Zhao et al. 2021): row/col positions are normalized to [0, 1)
    by dividing by H/W, then channels [0, C/2) get the row (H) encoding and
    channels [C/2, C) get the column (W) encoding, each as an interleaved
    sin/cos pair. With C=4 there's only one frequency (i=0), so denom =
    seq_denom**0 = 1 regardless of seq_denom -- this test only pins down
    the normalization and H/W-channel-split structure, not frequency
    scaling across channels.

    Expected sin/cos values below are plain trig evaluations (math.sin/cos
    of h/H and w/W), computed independently of the module under test.
    """
    B, C, H, W = 2, 4, 2, 3

    pe = PositionalEncoding2D(d_model=C, max_len=max(H, W))

    th = 1 / 3
    tth = 2 / 3
    test_pe = [
        [[sin(0   / 10000**(0/4)), sin(0   / 10000**(0/4)), sin(0   / 10000**(0/4))],
         [sin(0.5 / 10000**(0/4)), sin(0.5 / 10000**(0/4)), sin(0.5 / 10000**(0/4))],],
        [[cos(0   / 10000**(0/4)), cos(0   / 10000**(0/4)), cos(0   / 10000**(0/4))],
         [cos(0.5 / 10000**(0/4)), cos(0.5 / 10000**(0/4)), cos(0.5 / 10000**(0/4))],],
        [[sin(0   / 10000**(0/4)), sin(th  / 10000**(0/4)), sin(tth / 10000**(0/4))],
         [sin(0   / 10000**(0/4)), sin(th  / 10000**(0/4)), sin(tth / 10000**(0/4))],],
        [[cos(0   / 10000**(0/4)), cos(th  / 10000**(0/4)), cos(tth / 10000**(0/4))],
         [cos(0   / 10000**(0/4)), cos(th  / 10000**(0/4)), cos(tth / 10000**(0/4))],],
    ]

    in1 = torch.zeros(B, C, H, W)
    out1 = pe(in1)
    exp1 = torch.Tensor([test_pe] * B)

    test = PositionalEncoding2D(d_model=4, max_len=3).cross_prod
    print(test.shape)
    print(test)

    print(out1)
    print(exp1)
    torch.testing.assert_close(out1, exp1)

    in2 = torch.ones(B, C, H, W)
    out2 = pe(in2)
    exp2 = torch.Tensor([test_pe] * B)
    exp2 += torch.ones_like(exp2)
    torch.testing.assert_close(out2, exp2)



def test_positional_encoding_2d_explicit_values_multi_frequency():
    """Same structure as above, but C=8 gives two sin/cos frequency pairs
    per axis (i=0, 1), so this additionally pins down the per-channel
    frequency scaling (and thus the seq_denom base).

    With seq_denom=10000 and C=8: denom_i = 1/seq_denom**(2*i/C), so
    denom_0 = 1 and denom_1 = 1/10000**(1/4) = 0.1 -- deliberately chosen
    so the second frequency is a clean multiple of the first.

    Row half (channels 0-3): ch0/1 = sin/cos(x * denom_0),
    ch2/3 = sin/cos(x * denom_1), x = h/H.
    Column half (channels 4-7): ch4/5 = sin/cos(y * denom_0),
    ch6/7 = sin/cos(y * denom_1), y = w/W.

    Expected values are plain trig evaluations (math.sin/cos), computed
    independently of the module under test.
    """
    B, C, H, W = 2, 8, 2, 3

    pe = PositionalEncoding2D(d_model=C, max_len=max(H, W))
    features = torch.zeros(B, C, H, W)
    out = pe(features)

    # row positions normalized to [0, 1): h/H for h in {0, 1}
    # columns: [sin(x*denom_0), cos(x*denom_0), sin(x*denom_1), cos(x*denom_1)]
    row_channels = [
        [0.0, 1.0, 0.0, 1.0],
        [0.479425538604203, 0.8775825618903728, 0.04997916927067833, 0.9987502603949663],
    ]
    # column positions normalized to [0, 1): w/W for w in {0, 1, 2}
    col_channels = [
        [0.0, 1.0, 0.0, 1.0],
        [0.3271946967961522, 0.9449569463147377, 0.03332716083675362, 0.9994444958828685],
        [0.618369803069737, 0.785887260776948, 0.066617294923393, 0.9977786007011223],
    ]

    expected = torch.zeros(B, C, H, W)
    for h in range(H):
        for c in range(4):
            expected[:, c, h, :] = row_channels[h][c]
    for w in range(W):
        for c in range(4):
            expected[:, 4 + c, :, w] = col_channels[w][c]

    torch.testing.assert_close(out, expected)


def test_positional_encoding_2d_does_not_mutate_its_input():
    """A pos-enc layer must be a pure function of its input.

    If it writes into `features` in place, two things break: the caller's
    tensor silently changes under it, and (worse) autograd refuses to
    backprop through the encoder, because the thing being overwritten is
    the saved output of a ReLU.
    """
    pe = PositionalEncoding2D(d_model=8, max_len=10)
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
    pe = PositionalEncoding2D(d_model=8, max_len=100)

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
    determined by d_model/seq_denom/max_len.)
    """
    pe1 = PositionalEncoding1D(d_model=8, max_len=10)
    pe2 = PositionalEncoding2D(d_model=8, max_len=10)

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
    logits = dec(torch.randn(B, L, C), torch.randint(0, V, (B, T)))

    assert logits.shape == (B, T, V)


def test_transformer_decoder_shape_for_varying_T():
    """T is per-batch, so forward() has to handle a changing target length."""
    B, L, C, V = 3, 48, 64, 138
    dec = TransformerDecoder(vocab_size=V, d_model=C, nhead=1, num_layers=1)
    memory = torch.randn(B, L, C)

    for T in (1, 2, 13, 64):
        logits = dec(memory, torch.randint(0, V, (B, T)))
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
        a = dec(memory, target)
        b = dec(memory, tampered)

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
        full = dec(memory, target)
        prefix = dec(memory, target[:, :4])

    torch.testing.assert_close(full[:, :4, :], prefix, rtol=1e-4, atol=1e-5)
