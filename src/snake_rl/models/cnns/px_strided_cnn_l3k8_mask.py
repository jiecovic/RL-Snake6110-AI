# src/snake_rl/models/cnns/px_strided_cnn_l3k8_mask.py
from __future__ import annotations

from gymnasium import spaces
from torch import nn

from snake_rl.models.cnns.base import BaseCNNExtractor


class PxStridedCNN_L3K8_Mask(BaseCNNExtractor):
    """
    PxStridedCNN_L3K8 extended for (pixel + OOB mask) input.

    Expected input:
      - C=2 with channels: [pixel, oob_mask]
        * pixel: uint8-ish (0..255) or float
        * oob_mask: 0/1 (or 0/255) indicating in-bounds / out-of-bounds

    Strategy:
      - 1x1 "fusion" conv learns a good mixing of pixel+mask before aggressive downsampling.
      - Then the same L3K8 strided stem as the base model.

    Output:
      - build_stem() returns spatial map [B, 2c, H', W']
      - BaseCNNExtractor applies: Flatten → Linear → ReLU to features_dim
    """

    def build_stem(self, observation_space: spaces.Box) -> nn.Module:
        c = self.c(1)

        return nn.Sequential(
            # learn how to combine pixel + mask early
            nn.Conv2d(self.in_ch, c, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),

            # same as PxStridedCNN_L3K8
            nn.Conv2d(c, c, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(c, 2 * c, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(2 * c, 2 * c, kernel_size=3, stride=1),
            nn.ReLU(),
        )
