# src/snake_rl/models/cnns/px_strided_cnn_l2_s2s2_k4.py
from __future__ import annotations

from gymnasium import spaces
from torch import nn

from snake_rl.models.cnns.base import BaseCNNExtractor


class PxStridedCNN_L2_S2S2_K4(BaseCNNExtractor):
    """
    Two-layer strided pixel CNN with gentler downsampling than L1K4.

    Architecture (base channels = c):
      - Conv(k=4, s=2): early downsampling + local mixing
      - Conv(k=4, s=2): second downsampling to reach ~tile scale (overall stride 4)

    Channel scaling:
      - Base channels c are computed via BaseCNNExtractor.c(1) using c_mult
      - Effective channels: c -> 2c

    Notes:
      - Compared to PxStridedCNN_L1K4 (k=4, s=4), this adds one extra mixing stage
        before reaching the same total downsampling factor (4).
      - Useful when L1K4 is too "shallow" but you still want tile-aligned output.

    Output:
      - build_stem() returns a spatial feature map [B, 2c, H', W']
      - BaseCNNExtractor applies: Flatten → Linear → ReLU to produce features_dim
    """

    def build_stem(self, observation_space: spaces.Box) -> nn.Module:
        c = self.c(1)
        return nn.Sequential(
            nn.Conv2d(self.in_ch, c, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(c, 2 * c, kernel_size=4, stride=2),
            nn.ReLU(),
        )
