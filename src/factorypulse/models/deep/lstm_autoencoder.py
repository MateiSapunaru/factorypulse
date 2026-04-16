from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class LSTMAutoencoderConfig:
    input_size: int
    hidden_size: int
    latent_size: int
    num_layers: int
    dropout: float


class LSTMAutoencoder(nn.Module):
    def __init__(self, config: LSTMAutoencoderConfig) -> None:
        super().__init__()
        self.config = config

        encoder_dropout = config.dropout if config.num_layers > 1 else 0.0
        decoder_dropout = config.dropout if config.num_layers > 1 else 0.0

        self.encoder = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=encoder_dropout,
        )

        self.to_latent = nn.Linear(config.hidden_size, config.latent_size)
        self.from_latent = nn.Linear(config.latent_size, config.hidden_size)

        self.decoder = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=decoder_dropout,
        )

        self.output_layer = nn.Linear(config.hidden_size, config.input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden_n, _) = self.encoder(x)
        encoded = hidden_n[-1]

        latent = self.to_latent(encoded)
        decoded_seed = self.from_latent(latent)

        repeated = decoded_seed.unsqueeze(1).repeat(1, x.size(1), 1)
        decoded_seq, _ = self.decoder(repeated)
        reconstructed = self.output_layer(decoded_seq)

        return reconstructed