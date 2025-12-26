# src/snake_rl/models/cnns/px_strided_cnn_l3k4.py
from __future__ import annotations

from gymnasium import spaces
from torch import nn

from snake_rl.models.cnns.base import BaseCNNExtractor


class PxStridedCNN_L3K4(BaseCNNExtractor):
    """
    Strided pixel CNN with tile-aligned downsampling.

    Architecture (base channels = c):
      - Conv(k=4, s=4): tile-aligned downsampling
      - Conv(k=3, s=1): local refinement
      - Conv(k=3, s=1): local refinement

    Channel scaling:
      - Base channels c are computed via BaseCNNExtractor.c(1) using c_mult
      - Effective channels: c → 2c → 2c

    Notes:
      - Strong inductive bias toward tile-level structure.
      - Useful when pixel grid is a clean rendering of symbolic tiles.
      - Complements PxStridedCNN_L3K8 (larger receptive field, less alignment).

    Output:
      - build_stem() returns a spatial feature map [B, 2c, H', W']
      - BaseCNNExtractor applies: Flatten → Linear → ReLU to produce features_dim
    """

    def build_stem(self, observation_space: spaces.Box) -> nn.Module:
        c = self.c(1)

        return nn.Sequential(
            nn.Conv2d(self.in_ch, c, kernel_size=4, stride=4),
            nn.ReLU(),
            nn.Conv2d(c, 2 * c, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(2 * c, 2 * c, kernel_size=3, stride=1),
            nn.ReLU(),
        )
