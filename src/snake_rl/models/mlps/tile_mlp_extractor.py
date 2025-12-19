# src/snake_rl/models/mlps/tile_mlp_extractor.py
from __future__ import annotations

from typing import Iterable, Tuple

import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


def _infer_chw(space: spaces.Box) -> Tuple[int, int, int]:
    if not isinstance(space, spaces.Box):
        raise TypeError(f"TileMLPExtractor expects spaces.Box, got {type(space)!r}")
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


def _build_mlp(in_dim: int, hidden: Iterable[int], out_dim: int, dropout: float) -> nn.Module:
    layers: list[nn.Module] = []
    d = int(in_dim)
    for h in hidden:
        h = int(h)
        layers.append(nn.Linear(d, h))
        layers.append(nn.GELU())
        if float(dropout) > 0:
            layers.append(nn.Dropout(float(dropout)))
        d = h
    layers.append(nn.Linear(d, int(out_dim)))
    return nn.Sequential(*layers)


class TileMLPExtractor(BaseFeaturesExtractor):
    """
    Embedding + (optional) 2D positional embeddings + MLP over flattened token embeddings.

    Input:
      obs: [B,H,W] or [B,C,H,W] with uint8 tile ids in [0, num_tiles-1]

    Output:
      features: [B, features_dim]
    """

    def __init__(
            self,
            observation_space: spaces.Box,
            *,
            num_tiles: int,
            features_dim: int = 512,
            d_emb: int = 128,
            mlp_hidden: list[int] | tuple[int, ...] = (1024, 512),
            dropout: float = 0.0,
            use_2d_positional: bool = True,
            use_frame_embed: bool = False,
            frame_fuse: str = "sum",  # "sum" or "concat"
            pre_ln: bool = True,
    ) -> None:
        super().__init__(observation_space, features_dim)

        if int(num_tiles) <= 1:
            raise ValueError(f"num_tiles must be >= 2, got {num_tiles}")
        if int(d_emb) <= 0:
            raise ValueError(f"d_emb must be > 0, got {d_emb}")
        if frame_fuse not in {"sum", "concat"}:
            raise ValueError(f"frame_fuse must be 'sum' or 'concat', got {frame_fuse!r}")

        c, h, w = _infer_chw(observation_space)
        self.in_channels = int(c)
        self.h = int(h)
        self.w = int(w)
        self.seq_len = int(self.h * self.w)

        self.num_tiles = int(num_tiles)
        self.d_emb = int(d_emb)
        self.use_2d_positional = bool(use_2d_positional)
        self.use_frame_embed = bool(use_frame_embed)
        self.frame_fuse = str(frame_fuse)
        self.pre_ln = bool(pre_ln)

        self.tile_emb = nn.Embedding(self.num_tiles, self.d_emb)

        self.frame_emb = None
        if self.in_channels > 1 and self.use_frame_embed:
            self.frame_emb = nn.Embedding(self.in_channels, self.d_emb)

        self.pos_row = None
        self.pos_col = None
        if self.use_2d_positional:
            self.pos_row = nn.Embedding(self.h, self.d_emb)
            self.pos_col = nn.Embedding(self.w, self.d_emb)

        self.ln = nn.LayerNorm(self.d_emb) if self.pre_ln else nn.Identity()

        # flatten dimension depends on frame fuse
        if self.in_channels == 1:
            flat_in = self.seq_len * self.d_emb
        else:
            if self.frame_fuse == "sum":
                flat_in = self.seq_len * self.d_emb
            else:
                flat_in = (self.in_channels * self.seq_len) * self.d_emb

        self.mlp = _build_mlp(flat_in, mlp_hidden, int(features_dim), float(dropout))

        # simple init (consistent with your ViT)
        nn.init.normal_(self.tile_emb.weight, mean=0.0, std=0.02)
        if self.frame_emb is not None:
            nn.init.normal_(self.frame_emb.weight, mean=0.0, std=0.02)
        if self.pos_row is not None:
            nn.init.normal_(self.pos_row.weight, mean=0.0, std=0.02)
        if self.pos_col is not None:
            nn.init.normal_(self.pos_col.weight, mean=0.0, std=0.02)

    def _add_pos(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D] where T=H*W
        if not self.use_2d_positional:
            return x
        assert self.pos_row is not None and self.pos_col is not None
        device = x.device
        row_idx = torch.arange(self.h, device=device)
        col_idx = torch.arange(self.w, device=device)
        pe = (self.pos_row(row_idx)[:, None, :] + self.pos_col(col_idx)[None, :, :]).reshape(self.seq_len, self.d_emb)
        return x + pe.unsqueeze(0)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.ndim == 3:
            obs = observations.unsqueeze(1)  # [B,1,H,W]
        elif observations.ndim == 4:
            obs = observations
        else:
            raise ValueError(f"expected obs [B,H,W] or [B,C,H,W], got {tuple(observations.shape)}")

        b, c, h, w = obs.shape
        if h != self.h or w != self.w:
            raise ValueError(f"obs H,W mismatch: got {(h, w)} expected {(self.h, self.w)}")
        if c != self.in_channels:
            raise ValueError(f"obs C mismatch: got C={c} expected C={self.in_channels}")

        ids = obs.to(dtype=torch.long).reshape(b, c, self.seq_len)  # [B,C,T]
        t = self.tile_emb(ids)  # [B,C,T,D]

        if self.frame_emb is not None:
            frame_ids = torch.arange(c, device=obs.device, dtype=torch.long).view(1, c, 1)
            t = t + self.frame_emb(frame_ids).expand(b, c, self.seq_len, self.d_emb)

        # add pos per-frame
        if c == 1:
            x = self._add_pos(t[:, 0, :, :])  # [B,T,D]
            x = self.ln(x)
            flat = x.reshape(b, self.seq_len * self.d_emb)
        else:
            if self.frame_fuse == "sum":
                xs = []
                for fi in range(c):
                    xi = self._add_pos(t[:, fi, :, :])
                    xs.append(xi)
                x = torch.stack(xs, dim=1).sum(dim=1)  # [B,T,D]
                x = self.ln(x)
                flat = x.reshape(b, self.seq_len * self.d_emb)
            else:
                # concat: keep per-frame pos, then concat tokens
                xs = []
                for fi in range(c):
                    xi = self._add_pos(t[:, fi, :, :])
                    xs.append(xi)
                x = torch.cat(xs, dim=1)  # [B, C*T, D]
                x = self.ln(x)
                flat = x.reshape(b, (c * self.seq_len) * self.d_emb)

        return self.mlp(flat)
