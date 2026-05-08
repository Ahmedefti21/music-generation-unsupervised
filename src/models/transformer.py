import torch
import torch.nn as nn
import math


class MusicTransformer(nn.Module):
    """GPT-style decoder-only Transformer for autoregressive music generation.
    
    Architecture per guideline Step 7:
    - Token embedding + positional embedding
    - Stack of Transformer decoder layers with causal masked self-attention
    - Final linear projection to vocab size
    """
    def __init__(self, vocab_size=284, d_model=256, nhead=8, num_layers=6,
                 dim_feedforward=1024, max_seq_len=512, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.output_layer = nn.Linear(d_model, vocab_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, padding_mask=None):
        """
        x: (batch, seq_len) — token IDs
        padding_mask: (batch, seq_len) — True where padded
        """
        batch_size, seq_len = x.shape

        # Positional embeddings (guideline: added before first Transformer layer)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        x_emb = self.token_embedding(x) * math.sqrt(self.d_model) + self.position_embedding(positions)
        x_emb = self.dropout(x_emb)

        # Causal mask (guideline: applied at every self-attention layer)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device),
            diagonal=1
        ).bool()

        output = self.transformer(
            x_emb,
            mask=causal_mask,
            src_key_padding_mask=padding_mask
        )

        logits = self.output_layer(output)
        return logits