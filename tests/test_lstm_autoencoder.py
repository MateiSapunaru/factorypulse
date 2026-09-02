from __future__ import annotations

import torch

from factorypulse.models.deep.lstm_autoencoder import LSTMAutoencoder, LSTMAutoencoderConfig


def _make_model(num_layers: int = 1) -> LSTMAutoencoder:
    config = LSTMAutoencoderConfig(
        input_size=6,
        hidden_size=16,
        latent_size=4,
        num_layers=num_layers,
        dropout=0.1,
    )
    return LSTMAutoencoder(config)


def test_forward_pass_preserves_input_shape() -> None:
    model = _make_model()
    batch_size, seq_len, input_size = 8, 20, 6
    x = torch.randn(batch_size, seq_len, input_size)

    output = model(x)

    assert output.shape == (batch_size, seq_len, input_size)


def test_forward_pass_with_multiple_layers() -> None:
    model = _make_model(num_layers=2)
    x = torch.randn(4, 10, 6)

    output = model(x)

    assert output.shape == x.shape


def test_forward_pass_handles_variable_sequence_lengths() -> None:
    model = _make_model()
    for seq_len in (1, 5, 30):
        x = torch.randn(2, seq_len, 6)
        output = model(x)
        assert output.shape == (2, seq_len, 6)


def test_forward_pass_output_is_finite() -> None:
    model = _make_model()
    x = torch.randn(3, 15, 6)

    output = model(x)

    assert torch.isfinite(output).all()
