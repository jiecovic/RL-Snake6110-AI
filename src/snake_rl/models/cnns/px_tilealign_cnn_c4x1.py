# src/snake_rl/models/cnns/px_tilealign_cnn_c4x1.py
from __future__ import annotations

from gymnasium import spaces
from torch import nn

from snake_rl.models.cnns.base import BaseCNNExtractor


class PxTileAlignedCNN_C4x1(BaseCNNExtractor):
    """
    Pixel CNN with tile-aligned downsampling (+1 extra conv layer).

    Architecture (base channels = 16):
      - Conv(k=4, s=4): tile-aligned downsampling
      - Conv(k=3, s=1) x3

    Channel scaling:
      - base channels scaled via self.c(16) => c = 16*c_mult
      - Effective channels: c, 2c, 2c, 4c

    Output:
      - build_stem() returns spatial map [B, 4c, H', W'] (NO Flatten)
      - BaseCNNExtractor adds: Flatten -> Linear -> ReLU
    """

    def build_stem(self, observation_space: spaces.Box) -> nn.Module:
        c = self.c(16)  # 16 * c_mult

        return nn.Sequential(
            nn.Conv2d(self.in_ch, c, kernel_size=4, stride=4),
            nn.ReLU(),
            nn.Conv2d(c, 2 * c, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(2 * c, 2 * c, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(2 * c, 4 * c, kernel_size=3, stride=1),
            nn.ReLU(),
        )
