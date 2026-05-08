import torch
import torch.nn as nn


class MusicVAE(nn.Module):
    def __init__(self, input_size=88, hidden_size=256, latent_size=128,
                 enc_num_layers=2, dec_num_layers=1):
        super().__init__()

        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=enc_num_layers,
            batch_first=True
        )

        self.fc_mu = nn.Linear(hidden_size, latent_size)
        self.fc_logvar = nn.Linear(hidden_size, latent_size)

        self.latent_to_hidden = nn.Linear(latent_size, hidden_size)

        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=dec_num_layers,
            batch_first=True
        )

        self.output_layer = nn.Linear(hidden_size, input_size)

    def encode(self, x):
        _, (hidden, _) = self.encoder(x)
        hidden_last = hidden[-1]
        mu = self.fc_mu(hidden_last)
        logvar = self.fc_logvar(hidden_last)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, seq_len=128):
        hidden = self.latent_to_hidden(z)
        decoder_input = hidden.unsqueeze(1).repeat(1, seq_len, 1)

        decoded, _ = self.decoder(decoder_input)
        output = self.output_layer(decoded)
        return output

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        output = self.decode(z, seq_len=x.size(1))
        return output, mu, logvar