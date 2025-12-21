# src/snake_rl/models/vits/tile_vit_extractor.py
from __future__ import annotations

from typing import Tuple

import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


def _infer_chw(space: spaces.Box) -> Tuple[int, int, int]:
    """
    Return (C, H, W) for supported tile-id observations.
    Supported shapes:
      - (H, W)
      - (C, H, W)   (e.g. VecFrameStack -> channels-first stacking)
    """
    if not isinstance(space, spaces.Box):
        raise TypeError(f"TileViTExtractor expects spaces.Box, got {type(space)!r}")

    shape = space.shape
    if shape is None:
        raise ValueError("observation_space.shape is None")

    if len(shape) == 2:
        h, w = int(shape[0]), int(shape[1])
        return 1, h, w

    if len(shape) == 3:
        c, h, w = int(shape[0]), int(shape[1]), int(shape[2])
        return c, h, w

    raise ValueError(f"Unsupported observation shape {shape!r}. Expected (H,W) or (C,H,W).")


class TileViTExtractor(BaseFeaturesExtractor):
    """
    ViT-like encoder-only Transformer features extractor for symbolic tile-ID grids.

    Input (SB3):
      observations: torch.Tensor
        - [B, H, W] or [B, C, H, W]
      Values are tile ids in [0, num_tiles-1] (uint8 or float from wrappers; casted to long).

    Output:
      features: [B, features_dim]
    """

    def __init__(
            self,
            observation_space: spaces.Box,
            *,
            num_tiles: int,
            features_dim: int = 512,
            d_model: int = 128,
            n_layers: int = 4,
            n_heads: int = 4,
            ffn_dim: int | None = None,
            dropout: float = 0.1,
            use_cls_token: bool = True,
            use_2d_positional: bool = True,
            use_frame_embed: bool = True,
            frame_fuse: str = "sum",  # "sum" or "concat"
    ) -> None:
        super().__init__(observation_space, features_dim)

        if int(num_tiles) <= 1:
            raise ValueError(f"num_tiles must be >= 2, got {num_tiles}")
        if int(d_model) <= 0:
            raise ValueError(f"d_model must be > 0, got {d_model}")
        if int(n_layers) <= 0:
            raise ValueError(f"n_layers must be > 0, got {n_layers}")
        if int(n_heads) <= 0 or (int(d_model) % int(n_heads) != 0):
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        if frame_fuse not in {"sum", "concat"}:
            raise ValueError(f"frame_fuse must be 'sum' or 'concat', got {frame_fuse!r}")

        self.num_tiles = int(num_tiles)
        self.d_model = int(d_model)
        self.n_layers = int(n_layers)
        self.n_heads = int(n_heads)
        self.ffn_dim = int(ffn_dim) if ffn_dim is not None else 4 * int(d_model)
        self.dropout = float(dropout)
        self.use_cls_token = bool(use_cls_token)
        self.use_2d_positional = bool(use_2d_positional)
        self.use_frame_embed = bool(use_frame_embed)
        self.frame_fuse = str(frame_fuse)

        c, h, w = _infer_chw(observation_space)
        self.in_channels = int(c)
        self.h = int(h)
        self.w = int(w)
        self.seq_len = int(self.h * self.w)

        # Token embedding for tile IDs.
        self.tile_emb = nn.Embedding(self.num_tiles, self.d_model)

        # Optional: embed the "frame/channel" index if we have C>1.
        self.frame_emb = None
        if self.in_channels > 1 and self.use_frame_embed:
            self.frame_emb = nn.Embedding(self.in_channels, self.d_model)

        # Positional embeddings for the grid.
        self.pos_1d = None
        self.pos_row = None
        self.pos_col = None
        if self.use_2d_positional:
            self.pos_row = nn.Embedding(self.h, self.d_model)
            self.pos_col = nn.Embedding(self.w, self.d_model)
        else:
            self.pos_1d = nn.Embedding(self.seq_len, self.d_model)

        # Optional CLS token.
        self.cls_token = None
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))

        self.drop = nn.Dropout(self.dropout)

        # Encoder-only transformer (pre-LN).
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.ffn_dim,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=self.n_layers)

        # Pool projection to SB3 features_dim.
        self.out_proj = nn.Linear(self.d_model, int(features_dim), bias=True)

        # Init (simple, consistent).
        nn.init.normal_(self.tile_emb.weight, mean=0.0, std=0.02)
        if self.frame_emb is not None:
            nn.init.normal_(self.frame_emb.weight, mean=0.0, std=0.02)
        if self.pos_row is not None:
            nn.init.normal_(self.pos_row.weight, mean=0.0, std=0.02)
        if self.pos_col is not None:
            nn.init.normal_(self.pos_col.weight, mean=0.0, std=0.02)
        if self.pos_1d is not None:
            nn.init.normal_(self.pos_1d.weight, mean=0.0, std=0.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

    def _add_positional(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D], where T == H*W
        if self.use_2d_positional:
            assert self.pos_row is not None and self.pos_col is not None
            device = x.device
            row_idx = torch.arange(self.h, device=device)
            col_idx = torch.arange(self.w, device=device)
            pe_row = self.pos_row(row_idx)  # [H, D]
            pe_col = self.pos_col(col_idx)  # [W, D]
            pe = (pe_row[:, None, :] + pe_col[None, :, :]).reshape(self.seq_len, self.d_model)  # [T, D]
            return x + pe.unsqueeze(0)
        else:
            assert self.pos_1d is not None
            device = x.device
            idx = torch.arange(self.seq_len, device=device)
            pe = self.pos_1d(idx)  # [T, D]
            return x + pe.unsqueeze(0)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Accept [B,H,W] or [B,C,H,W]
        if observations.ndim == 3:
            obs = observations.unsqueeze(1)  # [B,1,H,W]
        elif observations.ndim == 4:
            obs = observations
        else:
            raise ValueError(f"expected obs [B,H,W] or [B,C,H,W], got shape={tuple(observations.shape)}")

        b, c, h, w = obs.shape
        if h != self.h or w != self.w:
            raise ValueError(f"obs H,W mismatch: got {(h, w)} expected {(self.h, self.w)}")
        if c != self.in_channels:
            raise ValueError(f"obs C mismatch: got C={c} expected C={self.in_channels}")

        # Cast to long for embedding lookup (SB3 might hand us float obs).
        tile_ids = obs.to(dtype=torch.long)  # [B,C,H,W]

        if torch.any(tile_ids < 0) or torch.any(tile_ids >= self.num_tiles):
            raise ValueError(
                f"tile id out of range: expected [0,{self.num_tiles - 1}] "
                f"got min={int(tile_ids.min())} max={int(tile_ids.max())}"
            )

        # Flatten grid -> tokens.
        # Base per-frame embeddings: [B, C, T, D]
        ids = tile_ids.reshape(b, c, self.seq_len)
        t = self.tile_emb(ids)  # [B, C, T, D]

        # Add optional frame embedding.
        if self.frame_emb is not None:
            frame_ids = torch.arange(c, device=obs.device, dtype=torch.long).view(1, c, 1)  # [1,C,1]
            t = t + self.frame_emb(frame_ids).expand(b, c, self.seq_len, self.d_model)

        # Fuse frames into a single token sequence [B, T, D].
        if c == 1:
            x = t[:, 0, :, :]
        elif self.frame_fuse == "sum":
            x = t.sum(dim=1)  # [B, T, D]
        else:
            # concat frames along token dimension -> [B, C*T, D]
            x = t.reshape(b, c * self.seq_len, self.d_model)

        # If we concatenated frames, positional embeddings need to be applied per-frame grid.
        if c > 1 and self.frame_fuse == "concat":
            # Apply grid positional per frame, then concat already did that by layout:
            # easiest: reshape back, add pos per frame, reshape again.
            x2 = x.reshape(b, c, self.seq_len, self.d_model)
            # add pos to each frame independently
            pos_added = []
            for fi in range(c):
                pos_added.append(self._add_positional(x2[:, fi, :, :]))
            x = torch.cat(pos_added, dim=1)  # [B, C*T, D]
        else:
            x = self._add_positional(x)

        x = self.drop(x)

        # Optional CLS token.
        if self.cls_token is not None:
            cls = self.cls_token.expand(b, 1, self.d_model)
            x = torch.cat([cls, x], dim=1)

        # Transformer encoder.
        x = self.encoder(x)

        # Pool.
        if self.cls_token is not None:
            pooled = x[:, 0, :]
        else:
            pooled = x.mean(dim=1)

        return self.out_proj(pooled)
