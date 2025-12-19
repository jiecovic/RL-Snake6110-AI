# src/snake_rl/models/cnns/px_tilealign_linear_cnn_c8.py
from gymnasium import spaces
from torch import nn

from snake_rl.models.cnns.base import BaseCNNExtractor


class PxTileAlignLinearCNN_C8(BaseCNNExtractor):
    """
    Single-layer tile-aligned CNN baseline.

    Architecture:
      - Conv(k=4, s=4, c=8)
      - ReLU
      - Flatten

    Acts as a minimal patchifying baseline (no spatial mixing beyond tiles).
    """

    def build_cnn(self, observation_space: spaces.Box) -> nn.Module:
        in_ch = int(observation_space.shape[0])
        c = 8

        return nn.Sequential(
            nn.Conv2d(in_ch, c, kernel_size=4, stride=4),
            nn.ReLU(),
            nn.Flatten(),
        )
