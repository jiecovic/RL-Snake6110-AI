# src/snake_rl/models/cnns/px_lite_cnn_c8.py
from gymnasium import spaces
from torch import nn

from snake_rl.models.cnns.base import BaseCNNExtractor


class PxLiteCNN_C8(BaseCNNExtractor):
    """
    Lightweight Nature-style CNN with reduced channel count.

    Base (c_mult=1):
      - Conv(k=8,s=4):  8
      - Conv(k=4,s=2): 16
      - Conv(k=3,s=1): 16

    c_mult scales all channels.
    """

    def build_stem(self, observation_space: spaces.Box) -> nn.Module:
        in_ch = int(observation_space.shape[0])
        c = 8 * self.c_mult

        return nn.Sequential(
            nn.Conv2d(in_ch, c, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(c, 2 * c, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(2 * c, 2 * c, kernel_size=3, stride=1),
            nn.ReLU(),
        )
