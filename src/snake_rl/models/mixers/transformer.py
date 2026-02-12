# src/snake_rl/models/mixers/transformer.py
from __future__ import annotations

import torch
from torch import nn


class TransformerMixer(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        n_layers: int = 4,
        n_heads: int = 4,
        ffn_dim: int | None = None,
        dropout: float = 0.1,
        norm_first: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.n_layers = int(n_layers)
        self.n_heads = int(n_heads)
        self.ffn_dim = int(ffn_dim) if ffn_dim is not None else 4 * int(d_model)
        self.dropout = float(dropout)

        enc = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.ffn_dim,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=bool(norm_first),
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=self.n_layers)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.encoder(tokens, src_key_padding_mask=mask)
