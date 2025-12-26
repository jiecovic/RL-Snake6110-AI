# src/snake_rl/models/cnns/px_strided_cnn_l3k8.py
from __future__ import annotations

from gymnasium import spaces
from torch import nn

from snake_rl.models.cnns.base import BaseCNNExtractor


class PxStridedCNN_L3K8(BaseCNNExtractor):
    """
    Strided pixel CNN with aggressive early downsampling.

    Architecture (base channels = c):
      - Conv(k=8, s=4): large-kernel strided downsampling
      - Conv(k=4, s=2): further spatial reduction
      - Conv(k=3, s=1): local feature refinement

    Channel scaling:
      - Base channels c are computed via BaseCNNExtractor.c(1) using c_mult
      - Effective channels: c → 2c → 2c

    Notes:
      - This is a general-purpose pixel encoder (not tile- or ViT-specific).
      - Large initial stride encourages fast spatial abstraction.
      - Suitable for PPO-style CNN + MLP or CNN + Transformer heads.

    Output:
      - build_stem() returns a spatial feature map [B, 2c, H', W']
      - BaseCNNExtractor applies: Flatten → Linear → ReLU to produce features_dim
    """

    def build_stem(self, observation_space: spaces.Box) -> nn.Module:
        c = self.c(1)

        return nn.Sequential(
            nn.Conv2d(self.in_ch, c, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(c, 2 * c, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(2 * c, 2 * c, kernel_size=3, stride=1),
            nn.ReLU(),
        )
