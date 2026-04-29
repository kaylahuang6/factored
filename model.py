import torch
import torch.nn as nn


class TinyTransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=128,
        n_heads=3,
        n_layers=3,
        d_ff=512,
        max_len=8,
        dropout=0.0,
    ):
        super().__init__()

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x, return_hidden=False):
        B, T = x.shape
        device = x.device

        pos = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        h = self.token_emb(x) + self.pos_emb(pos)

        causal_mask = torch.triu(
            torch.ones(T, T, device=device, dtype=torch.bool),
            diagonal=1
        )

        h = self.transformer(h, mask=causal_mask)
        h = self.ln(h)
        logits = self.head(h)

        if return_hidden:
            return logits, h
        return logits