# src/snake_rl/models/vits/vit_utils.py
from __future__ import annotations

from typing import Optional

import torch
from torch import nn

POS_MODES: set[str] = {"abs_2d", "abs_1d", "pov_center"}


class GridPositionalEncoding(nn.Module):
    """
    Positional encoding for tokens that come from a 2D grid of size (H,W).

    Modes:
      - "abs_2d"     : learned row+col embeddings (ViT-style)
      - "abs_1d"     : learned 1D index embedding over flattened tokens
      - "pov_center" : learned offsets from grid center (anchor at center)

    Input/Output:
      x: [B, T=H*W, D]
      returns x + pe
    """

    def __init__(self, *, h: int, w: int, d_model: int, pos_mode: str = "abs_2d") -> None:
        super().__init__()
        self.h = int(h)
        self.w = int(w)
        self.d_model = int(d_model)
        self.seq_len = int(self.h * self.w)

        pos_mode = str(pos_mode)
        if pos_mode not in POS_MODES:
            raise ValueError(f"pos_mode must be one of {sorted(POS_MODES)}, got {pos_mode!r}")
        self.pos_mode = pos_mode

        self.pos_1d: Optional[nn.Embedding] = None
        self.pos_row: Optional[nn.Embedding] = None
        self.pos_col: Optional[nn.Embedding] = None
        self.rel_row: Optional[nn.Embedding] = None
        self.rel_col: Optional[nn.Embedding] = None

        if self.pos_mode == "abs_1d":
            self.pos_1d = nn.Embedding(self.seq_len, self.d_model)
        elif self.pos_mode == "abs_2d":
            self.pos_row = nn.Embedding(self.h, self.d_model)
            self.pos_col = nn.Embedding(self.w, self.d_model)
        else:
            self.rel_row = nn.Embedding(self.h, self.d_model)
            self.rel_col = nn.Embedding(self.w, self.d_model)

        # Cache base offset grids (move with .to(device)).
        cy = self.h // 2
        cx = self.w // 2
        dy = (torch.arange(self.h) - cy).view(self.h, 1).expand(self.h, self.w)  # [H,W]
        dx = (torch.arange(self.w) - cx).view(1, self.w).expand(self.h, self.w)  # [H,W]
        self.register_buffer("_dy_base", dy.reshape(self.seq_len), persistent=False)  # [T]
        self.register_buffer("_dx_base", dx.reshape(self.seq_len), persistent=False)  # [T]

        # Init (consistent with your ViT extractors)
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

    def _add_abs_2d(self, x: torch.Tensor) -> torch.Tensor:
        assert self.pos_row is not None and self.pos_col is not None
        device = x.device
        row_idx = torch.arange(self.h, device=device)
        col_idx = torch.arange(self.w, device=device)
        pe = (self.pos_row(row_idx)[:, None, :] + self.pos_col(col_idx)[None, :, :]).reshape(self.seq_len, self.d_model)
        return x + pe.unsqueeze(0)

    def _add_abs_1d(self, x: torch.Tensor) -> torch.Tensor:
        assert self.pos_1d is not None
        device = x.device
        idx = torch.arange(self.seq_len, device=device)
        pe = self.pos_1d(idx)
        return x + pe.unsqueeze(0)

    def _add_pov_center(self, x: torch.Tensor) -> torch.Tensor:
        assert self.rel_row is not None and self.rel_col is not None
        dy = self._dy_base.view(1, -1)  # [1,T]
        dx = self._dx_base.view(1, -1)  # [1,T]
        cy = self.h // 2
        cx = self.w // 2
        iy = (dy + cy).clamp(0, self.h - 1).to(dtype=torch.long)
        ix = (dx + cx).clamp(0, self.w - 1).to(dtype=torch.long)
        pe = self.rel_row(iy) + self.rel_col(ix)  # [1,T,D]
        return x + pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != self.seq_len or x.shape[2] != self.d_model:
            raise ValueError(
                f"expected x [B,{self.seq_len},{self.d_model}], got shape={tuple(x.shape)}"
            )
        if self.pos_mode == "abs_2d":
            return self._add_abs_2d(x)
        if self.pos_mode == "abs_1d":
            return self._add_abs_1d(x)
        return self._add_pov_center(x)


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Mean over tokens excluding masked positions.
    x:    [B,T,D]
    mask: [B,T] bool (True => exclude)
    """
    keep = ~mask
    denom = keep.sum(dim=1).clamp(min=1).to(dtype=x.dtype)  # [B]
    keep_f = keep.to(dtype=x.dtype).unsqueeze(-1)  # [B,T,1]
    return (x * keep_f).sum(dim=1) / denom.unsqueeze(-1)


def pool_tokens(
        tokens: torch.Tensor,
        *,
        pooling: str,
        has_cls: bool,
        token_mask: Optional[torch.Tensor] = None,
        mask_pool: bool = True,
) -> torch.Tensor:
    """
    Pool transformer outputs into [B,D].

    tokens:    [B, T, D] (T includes CLS if has_cls=True)
    pooling:   "cls" | "mean" | "cls_mean"
    token_mask: [B, T] bool where True means "ignore" (optional)
    """
    pooling = str(pooling)
    if pooling not in {"cls", "mean", "cls_mean"}:
        raise ValueError(f"pooling must be one of {{'cls','mean','cls_mean'}}, got {pooling!r}")

    if pooling in {"cls", "cls_mean"} and not has_cls:
        raise ValueError(f"pooling={pooling!r} requires has_cls=True")

    if pooling == "cls":
        return tokens[:, 0, :]

    if pooling == "mean":
        if has_cls:
            tok = tokens[:, 1:, :]
            m = token_mask[:, 1:] if (token_mask is not None and mask_pool) else None
        else:
            tok = tokens
            m = token_mask if (token_mask is not None and mask_pool) else None
        return masked_mean(tok, m) if m is not None else tok.mean(dim=1)

    # cls_mean
    cls = tokens[:, 0, :]
    tok = tokens[:, 1:, :]
    m = token_mask[:, 1:] if (token_mask is not None and mask_pool) else None
    mean = masked_mean(tok, m) if m is not None else tok.mean(dim=1)
    return torch.cat([cls, mean], dim=-1)
