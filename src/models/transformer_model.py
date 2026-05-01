import torch
import torch.nn as nn


class MusicTransformer(nn.Module):
    def __init__(self, vocab_size=129, d_model=128, nhead=4, num_layers=3, dim_feedforward=256, max_seq_len=128):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        batch_size, seq_len = x.shape

        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).repeat(batch_size, 1)

        token_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(positions)

        x_emb = token_emb + pos_emb

        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device),
            diagonal=1
        ).bool()

        output = self.transformer(
            x_emb,
            mask=causal_mask
        )

        logits = self.output_layer(output)
        return logits