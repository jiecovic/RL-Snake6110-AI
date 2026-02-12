# src/snake_rl/models/mixers/identity.py
from __future__ import annotations

import torch
from torch import nn


class IdentityMixer(nn.Module):
    def forward(self, tokens: torch.Tensor, _mask: torch.Tensor | None = None) -> torch.Tensor:
        return tokens
