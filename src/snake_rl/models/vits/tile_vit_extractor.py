# src/snake_rl/models/vits/tile_vit_extractor.py
from __future__ import annotations

from typing import Optional, Tuple

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

    Positional modes (single source of truth via `pos_mode`):
      - "abs_2d"     : learned row+col embeddings (classic ViT-ish)
      - "abs_1d"     : learned 1D index embedding over flattened tokens
      - "pov_center" : learned offsets from grid center (anchor at center)

    Optional masking (attention padding mask):
      - if use_token_mask=True, tokens with tile_id == mask_token_id are ignored by attention
      - CLS token is never masked
      - for pooling="mean"/"cls_mean", masked tokens can be excluded from the mean (mask_pool=True)

    Pooling modes:
      - "cls"      : use CLS token only (requires use_cls_token=True)
      - "mean"     : mean over tokens (excludes CLS if present)
      - "cls_mean" : concatenate [CLS, mean(tokens)] then project
    """

    POS_MODES = {"abs_2d", "abs_1d", "pov_center"}

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
            pooling: str = "cls",  # "cls" | "mean" | "cls_mean"
            use_frame_embed: bool = True,
            frame_fuse: str = "sum",  # "sum" or "concat"
            pos_mode: str = "abs_2d",
            # token masking
            use_token_mask: bool = False,
            mask_token_id: int = 0,
            mask_pool: bool = True,
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

        pooling = str(pooling)
        if pooling not in {"cls", "mean", "cls_mean"}:
            raise ValueError(f"pooling must be one of {{'cls','mean','cls_mean'}}, got {pooling!r}")
        if pooling in {"cls", "cls_mean"} and not use_cls_token:
            raise ValueError(f"pooling={pooling!r} requires use_cls_token=True")

        pos_mode = str(pos_mode)
        if pos_mode not in self.POS_MODES:
            raise ValueError(f"pos_mode must be one of {sorted(self.POS_MODES)}, got {pos_mode!r}")

        self.num_tiles = int(num_tiles)
        self.d_model = int(d_model)
        self.n_layers = int(n_layers)
        self.n_heads = int(n_heads)
        self.ffn_dim = int(ffn_dim) if ffn_dim is not None else 4 * int(d_model)
        self.dropout = float(dropout)
        self.use_cls_token = bool(use_cls_token)
        self.pooling = pooling
        self.use_frame_embed = bool(use_frame_embed)
        self.frame_fuse = str(frame_fuse)

        self.pos_mode = pos_mode

        self.use_token_mask = bool(use_token_mask)
        self.mask_token_id = int(mask_token_id)
        self.mask_pool = bool(mask_pool)

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

        # Absolute positional embeddings.
        self.pos_1d = None
        self.pos_row = None
        self.pos_col = None

        # Anchored (center-relative) positional embeddings.
        self.rel_row = None
        self.rel_col = None

        if self.pos_mode == "abs_1d":
            self.pos_1d = nn.Embedding(self.seq_len, self.d_model)
        elif self.pos_mode == "abs_2d":
            self.pos_row = nn.Embedding(self.h, self.d_model)
            self.pos_col = nn.Embedding(self.w, self.d_model)
        else:
            # pov_center: learn embeddings over center-relative offsets (via indices in [0..H-1],[0..W-1]).
            self.rel_row = nn.Embedding(self.h, self.d_model)
            self.rel_col = nn.Embedding(self.w, self.d_model)

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
        in_dim = self.d_model * (2 if self.pooling == "cls_mean" else 1)
        self.out_proj = nn.Linear(in_dim, int(features_dim), bias=True)

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
        if self.rel_row is not None:
            nn.init.normal_(self.rel_row.weight, mean=0.0, std=0.02)
        if self.rel_col is not None:
            nn.init.normal_(self.rel_col.weight, mean=0.0, std=0.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

        # Cache base offset grids (registered buffers so they move with .to(device)).
        cy = self.h // 2
        cx = self.w // 2
        dy = (torch.arange(self.h) - cy).view(self.h, 1).expand(self.h, self.w)  # [H,W]
        dx = (torch.arange(self.w) - cx).view(1, self.w).expand(self.h, self.w)  # [H,W]
        self.register_buffer("_dy_base", dy.reshape(self.seq_len), persistent=False)  # [T]
        self.register_buffer("_dx_base", dx.reshape(self.seq_len), persistent=False)  # [T]

    def _add_positional_absolute_2d(self, x: torch.Tensor) -> torch.Tensor:
        assert self.pos_row is not None and self.pos_col is not None
        device = x.device
        row_idx = torch.arange(self.h, device=device)
        col_idx = torch.arange(self.w, device=device)
        pe_row = self.pos_row(row_idx)  # [H, D]
        pe_col = self.pos_col(col_idx)  # [W, D]
        pe = (pe_row[:, None, :] + pe_col[None, :, :]).reshape(self.seq_len, self.d_model)  # [T, D]
        return x + pe.unsqueeze(0)

    def _add_positional_absolute_1d(self, x: torch.Tensor) -> torch.Tensor:
        assert self.pos_1d is not None
        device = x.device
        idx = torch.arange(self.seq_len, device=device)
        pe = self.pos_1d(idx)  # [T, D]
        return x + pe.unsqueeze(0)

    def _add_positional_pov_center(self, x: torch.Tensor) -> torch.Tensor:
        """
        Center-anchored PE: position is encoded as (dy, dx) relative to grid center.
        """
        assert self.rel_row is not None and self.rel_col is not None

        dy = self._dy_base.view(1, -1)  # [1,T]
        dx = self._dx_base.view(1, -1)  # [1,T]

        cy = self.h // 2
        cx = self.w // 2
        iy = (dy + cy).clamp(0, self.h - 1).to(dtype=torch.long)
        ix = (dx + cx).clamp(0, self.w - 1).to(dtype=torch.long)

        pe = self.rel_row(iy) + self.rel_col(ix)  # [1,T,D]
        return x + pe

    def _add_positional(self, x: torch.Tensor) -> torch.Tensor:
        if self.pos_mode == "abs_2d":
            return self._add_positional_absolute_2d(x)
        if self.pos_mode == "abs_1d":
            return self._add_positional_absolute_1d(x)
        return self._add_positional_pov_center(x)

    def _build_token_mask(self, tile_ids: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Build src_key_padding_mask for attention.

        Returns:
          mask: [B, T] bool, True means "ignore this token" (padding/OOB)
        """
        if not self.use_token_mask:
            return None

        b, c, h, w = tile_ids.shape
        assert h == self.h and w == self.w

        if c == 1:
            m = (tile_ids[:, 0, :, :] == self.mask_token_id).reshape(b, self.seq_len)
            return m

        if self.frame_fuse == "sum":
            # One token per position; mark masked only if all frames are masked there.
            m = (tile_ids == self.mask_token_id).all(dim=1)  # [B,H,W]
            return m.reshape(b, self.seq_len)

        # frame_fuse == "concat": each frame contributes its own tokens
        m = (tile_ids == self.mask_token_id).reshape(b, c * self.seq_len)  # [B,C*T]
        return m

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Mean over tokens excluding masked positions.
        x:    [B,T,D]
        mask: [B,T] bool (True => exclude)
        """
        keep = ~mask  # True => include
        denom = keep.sum(dim=1).clamp(min=1).to(dtype=x.dtype)  # [B]
        keep_f = keep.to(dtype=x.dtype).unsqueeze(-1)  # [B,T,1]
        return (x * keep_f).sum(dim=1) / denom.unsqueeze(-1)  # [B,D]

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

        tile_ids = obs.to(dtype=torch.long)  # [B,C,H,W]

        # Build token mask in the same token order as x (before CLS).
        token_mask = self._build_token_mask(tile_ids)  # [B,T] or [B,C*T] or None

        # Flatten grid -> tokens.
        ids = tile_ids.reshape(b, c, self.seq_len)
        t = self.tile_emb(ids)  # [B, C, T, D]

        # Add optional frame embedding.
        if self.frame_emb is not None:
            frame_ids = torch.arange(c, device=obs.device, dtype=torch.long).view(1, c, 1)  # [1,C,1]
            t = t + self.frame_emb(frame_ids).expand(b, c, self.seq_len, self.d_model)

        # Fuse frames into a single token sequence [B, T, D] or [B, C*T, D].
        if c == 1:
            x = t[:, 0, :, :]
        elif self.frame_fuse == "sum":
            x = t.sum(dim=1)  # [B, T, D]
        else:
            x = t.reshape(b, c * self.seq_len, self.d_model)  # [B, C*T, D]

        # Positional embeddings.
        if c > 1 and self.frame_fuse == "concat":
            x2 = x.reshape(b, c, self.seq_len, self.d_model)
            pos_added = []
            for fi in range(c):
                pos_added.append(self._add_positional(x2[:, fi, :, :]))
            x = torch.cat(pos_added, dim=1)  # [B, C*T, D]
        else:
            x = self._add_positional(x)

        x = self.drop(x)

        # Optional CLS token (never masked).
        if self.cls_token is not None:
            cls = self.cls_token.expand(b, 1, self.d_model)
            x = torch.cat([cls, x], dim=1)
            if token_mask is not None:
                cls_mask = torch.zeros((b, 1), device=token_mask.device, dtype=torch.bool)
                token_mask = torch.cat([cls_mask, token_mask], dim=1)  # [B,1+T]

        # Attention masking: True means "ignore".
        x = self.encoder(x, src_key_padding_mask=token_mask)

        # Pool (always return [B, features_dim])
        if self.pooling == "cls":
            pooled = x[:, 0, :]

        elif self.pooling == "mean":
            if self.cls_token is not None:
                tok = x[:, 1:, :]
                m = token_mask[:, 1:] if (token_mask is not None and self.mask_pool) else None
            else:
                tok = x
                m = token_mask if (token_mask is not None and self.mask_pool) else None

            pooled = self._masked_mean(tok, m) if m is not None else tok.mean(dim=1)

        else:  # "cls_mean"
            cls = x[:, 0, :]
            tok = x[:, 1:, :]
            m = token_mask[:, 1:] if (token_mask is not None and self.mask_pool) else None
            mean = self._masked_mean(tok, m) if m is not None else tok.mean(dim=1)
            pooled = torch.cat([cls, mean], dim=-1)

        return self.out_proj(pooled)
