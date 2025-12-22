# src/snake_rl/models/cnns/px_tilealign_cnn_c4.py
from __future__ import annotations

from gymnasium import spaces
from torch import nn

from snake_rl.models.cnns.base import BaseCNNExtractor


class PxTileAlignedCNN_C4(BaseCNNExtractor):
    """
    Pixel CNN with tile-aligned downsampling.

    Architecture (base channels = 16):
      - Conv(k=4, s=4): tile-aligned downsampling
      - Conv(k=3, s=1) x2

    Channel scaling:
      - base channels are scaled via BaseCNNExtractor.c(base) using c_mult
      - Effective channels: c=16*c_mult, then 2c, then 2c

    Output:
      - build_stem() returns spatial map [B, 2c, H', W'] (NO Flatten)
      - BaseCNNExtractor adds: Flatten -> Linear -> ReLU
    """

    def build_stem(self, observation_space: spaces.Box) -> nn.Module:
        c = self.c(1)  # 16 * c_mult

        return nn.Sequential(
            nn.Conv2d(self.in_ch, c, kernel_size=4, stride=4),
            nn.ReLU(),
            nn.Conv2d(c, 2 * c, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(2 * c, 2 * c, kernel_size=3, stride=1),
            nn.ReLU(),
        )
