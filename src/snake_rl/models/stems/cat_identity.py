# src/snake_rl/models/stems/cat_identity.py
from __future__ import annotations

import torch
from torch import nn

from snake_rl.models.stems.base import TokenStem, TokenStemOutput


class CatIdentityStem(TokenStem):
    def __init__(
        self,
        *,
        num_tiles: int,
        d_model: int,
        frame_fuse: str = "sum",
        use_token_mask: bool = False,
        mask_token_id: int = 0,
    ) -> None:
        super().__init__()
        self.num_tiles = int(num_tiles)
        self.d_model = int(d_model)
        self.frame_fuse = str(frame_fuse)
        if self.frame_fuse not in {"sum", "concat"}:
            raise ValueError(f"frame_fuse must be 'sum' or 'concat', got {frame_fuse!r}")
        self.use_token_mask = bool(use_token_mask)
        self.mask_token_id = int(mask_token_id)

        self.emb = nn.Embedding(self.num_tiles, self.d_model)
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.02)

    def forward(self, obs: torch.Tensor) -> TokenStemOutput:
        if obs.ndim == 3:
            obs = obs.unsqueeze(1)
        if obs.ndim != 4:
            raise ValueError(f"expected obs [B,C,H,W], got {tuple(obs.shape)}")

        b, c, h, w = obs.shape
        tile_ids = obs.to(dtype=torch.long)

        ids = tile_ids.reshape(b, c, h * w)
        t = self.emb(ids)  # [B, C, T, D]

        if self.frame_fuse == "sum":
            tokens = t.sum(dim=1)
            frame_count = 1
            frame_fuse = "sum"
        else:
            tokens = t.reshape(b, c * h * w, self.d_model)
            frame_count = int(c)
            frame_fuse = "concat"

        mask = None
        if self.use_token_mask:
            if self.frame_fuse == "sum":
                m = (tile_ids == self.mask_token_id).all(dim=1)  # [B,H,W]
                mask = m.reshape(b, h * w)
            else:
                mask = (tile_ids == self.mask_token_id).reshape(b, c * h * w)

        return TokenStemOutput(
            tokens=tokens,
            mask=mask,
            grid=(int(h), int(w)),
            frame_count=frame_count,
            frame_fuse=frame_fuse,
        )
