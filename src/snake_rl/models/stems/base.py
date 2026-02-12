# src/snake_rl/models/stems/base.py
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(slots=True)
class TokenStemOutput:
    tokens: torch.Tensor  # [B, T, D]
    mask: torch.Tensor | None
    grid: tuple[int, int] | None  # (H, W) if tokens come from a grid
    frame_count: int
    frame_fuse: str  # "sum" | "concat"


class TokenStem(nn.Module):
    d_model: int

    def forward(self, obs: torch.Tensor) -> TokenStemOutput:  # pragma: no cover - interface
        raise NotImplementedError
