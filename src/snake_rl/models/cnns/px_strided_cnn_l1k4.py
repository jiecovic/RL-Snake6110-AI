# src/snake_rl/models/cnns/px_strided_cnn_l1k4.py
from __future__ import annotations

from gymnasium import spaces
from torch import nn

from snake_rl.models.cnns.base import BaseCNNExtractor


class PxStridedCNN_L1K4(BaseCNNExtractor):
    """
    Single-layer strided pixel CNN with tile-aligned downsampling.

    Architecture (base channels = c):
      - Conv(k=4, s=4): tile-aligned downsampling + embedding

    Channel scaling:
      - Base channels c are computed via BaseCNNExtractor.c(1) using c_mult
      - Output channels: c

    Notes:
      - Acts like a learned per-tile embedding (stride == kernel).
      - Minimal spatial mixing (only within a tile-sized patch).
      - Good baseline when you want the MLP/head to do most of the work.

    Output:
      - build_stem() returns a spatial feature map [B, c, H', W']
      - BaseCNNExtractor applies: Flatten → Linear → ReLU to produce features_dim
    """

    def build_stem(self, observation_space: spaces.Box) -> nn.Module:
        c = self.c(1)
        return nn.Sequential(
            nn.Conv2d(self.in_ch, c, kernel_size=4, stride=4),
            nn.ReLU(),
        )
