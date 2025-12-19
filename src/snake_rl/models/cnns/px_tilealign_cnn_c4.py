# src/snake_rl/models/cnns/px_tilealign_cnn_c4.py
from gymnasium import spaces
from torch import nn

from snake_rl.models.cnns.base import BaseCNNExtractor


class PxTileAlignedCNN_C4(BaseCNNExtractor):
    """
    Pixel CNN with tile-aligned downsampling.

    Architecture:
      - Conv(k=4, s=4): aligns receptive field and stride to 4x4 rendered tiles
      - Conv(k=3, s=1) x2
      - Channels: C=4 -> 2C -> 2C
    """

    def build_cnn(self, observation_space: spaces.Box) -> nn.Module:
        in_ch = int(observation_space.shape[0])
        c = 4

        return nn.Sequential(
            nn.Conv2d(in_ch, c, kernel_size=4, stride=4),
            nn.ReLU(),
            nn.Conv2d(c, 2 * c, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(2 * c, 2 * c, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
