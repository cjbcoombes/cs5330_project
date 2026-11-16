import torch

from math import sin, cos
from hmer_latex.models.decoder import PositionalEncoding2D


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
