import torch
import torch.nn as nn


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size=88, hidden_size=256, latent_size=128, num_layers=2):
        super().__init__()

        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.to_latent = nn.Linear(hidden_size, latent_size)
        self.from_latent = nn.Linear(latent_size, hidden_size)

        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.output_layer = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        _, (hidden, _) = self.encoder(x)

        # use last layer hidden state
        hidden_last = hidden[-1]                     # (batch, hidden_size)
        z = self.to_latent(hidden_last)              # (batch, latent_size)

        decoder_input = self.from_latent(z)          # (batch, hidden_size)
        decoder_input = decoder_input.unsqueeze(1)   # (batch, 1, hidden_size)
        decoder_input = decoder_input.repeat(1, x.size(1), 1)

        decoded, _ = self.decoder(decoder_input)
        out = self.output_layer(decoded)             # (batch, seq_len, input_size)

        return out, z