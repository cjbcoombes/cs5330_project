import torch

from hmer_latex.models.encoder import DenseNetEncoder


def test_densenet_encoder_output_shape():
    enc = DenseNetEncoder(out_channels=256)
    out = enc(torch.randn(2, 1, 128, 256))
    assert out.shape == (2, 256, 8, 16)
