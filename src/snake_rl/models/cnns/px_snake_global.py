# src/snake_rl/models/cnns/px_snake_global.py
from __future__ import annotations

from gymnasium import spaces
from torch import nn

from snake_rl.models.cnns.base import BaseCNNExtractor


class PxSnakeGlobal(BaseCNNExtractor):
    """
    Strided pixel CNN with tile-aligned downsampling.

    Architecture (base channels = c):
      - Conv(k=4, s=4): tile-aligned downsampling
      - Conv(k=3, s=1): local refinement
      - Conv(k=3, s=1): local refinement

    Channel scaling:
      - Base channels c are computed via BaseCNNExtractor.c(1) using c_mult
      - Effective channels: c â†’ 2c â†’ 2c

    Notes:
      - Strong inductive bias toward tile-level structure.
      - Useful when pixel grid is a clean rendering of symbolic tiles.
      - Complements PxStridedCNN_L3K8 (larger receptive field, less alignment).

    Output:
      - build_stem() returns a spatial feature map [B, 2c, H', W']
      - BaseCNNExtractor applies: Flatten â†’ Linear â†’ ReLU to produce features_dim
    """

    def build_stem(self, observation_space: spaces.Box) -> nn.Module:
        c = self.c(1)

        return nn.Sequential(
            nn.Conv2d(self.in_ch, c, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(c, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
