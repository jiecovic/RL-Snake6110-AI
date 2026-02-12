# src/snake_rl/models/stems/cat_patch.py
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from snake_rl.models.stems.base import TokenStem, TokenStemOutput


class CatPatchStem(TokenStem):
    def __init__(
        self,
        *,
        num_tiles: int,
        d_model: int,
        patch: int = 2,
        stride: int | None = None,
        frame_fuse: str = "sum",
        use_token_mask: bool = False,
        mask_token_id: int = 0,
    ) -> None:
        super().__init__()
        self.num_tiles = int(num_tiles)
        self.d_model = int(d_model)
        self.patch = int(patch)
        self.stride = int(stride) if stride is not None else int(patch)
        self.frame_fuse = str(frame_fuse)
        if self.frame_fuse != "sum":
            raise ValueError("CatPatchStem only supports frame_fuse='sum'")
        self.use_token_mask = bool(use_token_mask)
        self.mask_token_id = int(mask_token_id)

        self.emb = nn.Embedding(self.num_tiles, self.d_model)
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.02)

        self.conv = nn.Conv2d(
            self.d_model,
            self.d_model,
            kernel_size=self.patch,
            stride=self.stride,
            padding=0,
        )

    def forward(self, obs: torch.Tensor) -> TokenStemOutput:
        if obs.ndim == 3:
            obs = obs.unsqueeze(1)
        if obs.ndim != 4:
            raise ValueError(f"expected obs [B,C,H,W], got {tuple(obs.shape)}")

        b, c, h, w = obs.shape
        tile_ids = obs.to(dtype=torch.long)

        ids = tile_ids.reshape(b, c, h * w)
        t = self.emb(ids).reshape(b, c, h, w, self.d_model)
        x = t.sum(dim=1).permute(0, 3, 1, 2).contiguous()  # [B, D, H, W]

        y = self.conv(x)
        _, d, h2, w2 = y.shape
        tokens = y.flatten(2).transpose(1, 2)

        mask = None
        if self.use_token_mask:
            m = (tile_ids == self.mask_token_id).any(dim=1).to(dtype=torch.float32)
            m2 = F.max_pool2d(m, kernel_size=self.patch, stride=self.stride, padding=0)
            mask = m2.to(dtype=torch.bool).reshape(b, h2 * w2)

        return TokenStemOutput(
            tokens=tokens,
            mask=mask,
            grid=(int(h2), int(w2)),
            frame_count=1,
            frame_fuse="sum",
        )
