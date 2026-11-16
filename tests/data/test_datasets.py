"""collate_batch tests -- the sos/eos contract and padding live here."""

import torch

from hmer_latex.data.datasets import collate_batch, default_transform
from hmer_latex.data.vocab import Vocab, tokenize_latex

PAD, SOS, EOS = 0, 1, 2


def make_batch(lengths, h=128, w=384):
    return [(torch.zeros(1, h, w), list(range(4, 4 + n))) for n in lengths]


def test_collate_shapes_and_dtypes():
    images, tgt_in, tgt_out, mask = collate_batch(make_batch([3, 7, 5]), PAD, SOS, EOS)
    assert images.shape == (3, 1, 128, 384)
    assert images.dtype == torch.float32
    # T = longest label + 1, to make room for sos/eos
    assert tgt_in.shape == tgt_out.shape == mask.shape == (3, 8)
    assert tgt_in.dtype == tgt_out.dtype == torch.int64
    assert mask.dtype == torch.bool


def test_teacher_forcing_offset():
    """tgt_in and tgt_out are the same ids shifted by one -- that *is* the
    teacher-forcing signal. Position t of tgt_in predicts position t of tgt_out."""
    ids = [4, 5, 6]
    _, tgt_in, tgt_out, _ = collate_batch([(torch.zeros(1, 4, 4), ids)], PAD, SOS, EOS)
    assert tgt_in[0].tolist() == [SOS, 4, 5, 6]
    assert tgt_out[0].tolist() == [4, 5, 6, EOS]


def test_shorter_sequences_are_pad_filled():
    _, tgt_in, tgt_out, _ = collate_batch(make_batch([2, 5]), PAD, SOS, EOS)
    assert tgt_in[0].tolist() == [SOS, 4, 5, PAD, PAD, PAD]
    assert tgt_out[0].tolist() == [4, 5, EOS, PAD, PAD, PAD]


def test_mask_marks_padding_of_tgt_in():
    _, tgt_in, _, mask = collate_batch(make_batch([2, 5]), PAD, SOS, EOS)
    assert torch.equal(mask, tgt_in == PAD)
    assert mask[0].tolist() == [False, False, False, True, True, True]
    assert not mask[1].any()


def test_default_transform_inverts_polarity():
    """The renderer writes black ink on white; the tensor must be the opposite,
    so that the encoder's zero-padding matches background rather than ink."""
    from PIL import Image

    white = default_transform()(Image.new("L", (8, 8), 255))
    black = default_transform()(Image.new("L", (8, 8), 0))
    assert white.shape == (1, 8, 8)
    assert white.max().item() == 0.0  # background -> 0
    assert black.min().item() == 1.0  # ink -> 1


def test_collate_consumes_real_vocab_output():
    """End-to-end on the actual seam: tokenize -> Vocab.encode -> collate."""
    vocab = Vocab.build_from_labels([tokenize_latex("$x^{2}+1$")])
    ids = vocab.encode(tokenize_latex("x^{2}"))
    _, tgt_in, tgt_out, _ = collate_batch([(torch.zeros(1, 4, 4), ids)], PAD, SOS, EOS)
    assert tgt_in[0, 0].item() == SOS
    assert tgt_out[0, len(ids)].item() == EOS
    assert vocab.decode(tgt_out[0].tolist()) == "x ^ { 2 }"
