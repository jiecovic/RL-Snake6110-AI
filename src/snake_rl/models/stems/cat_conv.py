# src/snake_rl/models/stems/cat_conv.py
from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from snake_rl.models.stems.base import TokenStem, TokenStemOutput


def _act(name: str) -> nn.Module:
    n = str(name).strip().lower()
    if n in {"relu", "r"}:
        return nn.ReLU()
    if n in {"gelu", "g"}:
        return nn.GELU()
    if n in {"silu", "swish"}:
        return nn.SiLU()
    if n in {"tanh"}:
        return nn.Tanh()
    raise ValueError(f"Unknown activation {name!r}")


class CatConvStem(TokenStem):
    def __init__(
        self,
        *,
        num_tiles: int,
        d_model: int,
        layers: list[dict[str, Any]],
        frame_fuse: str = "sum",
        use_token_mask: bool = False,
        mask_token_id: int = 0,
    ) -> None:
        super().__init__()
        if not layers:
            raise ValueError("CatConvStem requires non-empty layers")
        self.num_tiles = int(num_tiles)
        self.d_model = int(d_model)
        self.frame_fuse = str(frame_fuse)
        if self.frame_fuse != "sum":
            raise ValueError("CatConvStem only supports frame_fuse='sum'")
        self.use_token_mask = bool(use_token_mask)
        self.mask_token_id = int(mask_token_id)

        self.emb = nn.Embedding(self.num_tiles, self.d_model)
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.02)

        modules: list[nn.Module] = []
        in_ch = self.d_model
        for layer in layers:
            out_raw = layer.get("out")
            if out_raw is None:
                raise ValueError("CatConvStem layer missing required key 'out'")
            out_ch = int(out_raw)
            k = int(layer.get("k", 3))
            s = int(layer.get("s", 1))
            p = int(layer.get("p", 0))
            act = layer.get("act", "relu")
            modules.append(nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p))
            modules.append(_act(act))
            in_ch = out_ch

        self.conv = nn.Sequential(*modules)
        self.proj: nn.Module
        if int(in_ch) != int(self.d_model):
            self.proj = nn.Conv2d(int(in_ch), int(self.d_model), kernel_size=1, stride=1)
        else:
            self.proj = nn.Identity()

        self._mask_layers = [
            (int(layer.get("k", 3)), int(layer.get("s", 1)), int(layer.get("p", 0)))
            for layer in layers
        ]

    def forward(self, obs: torch.Tensor) -> TokenStemOutput:
        if obs.ndim == 3:
            obs = obs.unsqueeze(1)
        if obs.ndim != 4:
            raise ValueError(f"expected obs [B,C,H,W], got {tuple(obs.shape)}")

        b, c, h, w = obs.shape
        tile_ids = obs.to(dtype=torch.long)
        ids = tile_ids.reshape(b, c, h * w)
        t = self.emb(ids).reshape(b, c, h, w, self.d_model)
        x = t.sum(dim=1).permute(0, 3, 1, 2).contiguous()

        y = self.conv(x)
        y = self.proj(y)
        _, d, h2, w2 = y.shape
        tokens = y.flatten(2).transpose(1, 2)

        mask = None
        if self.use_token_mask:
            m = (tile_ids == self.mask_token_id).any(dim=1).to(dtype=torch.float32)
            for k, s, p in self._mask_layers:
                if s > 1 or k > 1 or p > 0:
                    m = F.max_pool2d(m, kernel_size=k, stride=s, padding=p)
            mask = m.to(dtype=torch.bool).reshape(b, h2 * w2)

        return TokenStemOutput(
            tokens=tokens,
            mask=mask,
            grid=(int(h2), int(w2)),
            frame_count=1,
            frame_fuse="sum",
        )
