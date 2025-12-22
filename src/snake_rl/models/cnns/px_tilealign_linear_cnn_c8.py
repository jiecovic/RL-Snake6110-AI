# src/snake_rl/models/cnns/px_tilealign_linear_cnn_c8.py
from __future__ import annotations

from gymnasium import spaces
from torch import nn

from snake_rl.models.cnns.base import BaseCNNExtractor


class PxTileAlignLinearCNN_C8(BaseCNNExtractor):
    """
    Single-layer tile-aligned CNN baseline.

    Architecture (base channels = 8):
      - Conv(k=4, s=4)
      - ReLU

    Channel scaling:
      - output channels = self.c(8) => 8*c_mult

    Output:
      - build_stem() returns spatial map [B, c, H', W'] (NO Flatten)
      - BaseCNNExtractor adds: Flatten -> Linear -> ReLU
    """

    def build_stem(self, observation_space: spaces.Box) -> nn.Module:
        c = self.c(8)  # 8 * c_mult

        return nn.Sequential(
            nn.Conv2d(self.in_ch, c, kernel_size=4, stride=4),
            nn.ReLU(),
            # nn.Conv2d(c, 2 * c, kernel_size=3, stride=1),
            # nn.ReLU(),
        )
