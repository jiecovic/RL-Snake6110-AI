# src/snake_rl/models/cnns/tile_cnn4.py
from gymnasium import spaces
from torch import nn

from .base_cnn_extractor import BaseCNNExtractor


class TileCNN4(BaseCNNExtractor):
    """
    Lightweight tile-aligned CNN.
    Kernel size matches tile size (4x4).
    Proven baseline from earlier experiments.
    """

    def build_cnn(self, observation_space: spaces.Box) -> nn.Module:
        in_ch = observation_space.shape[0]
        C = 4

        return nn.Sequential(
            nn.Conv2d(in_ch, C, kernel_size=4, stride=4),
            nn.ReLU(),
            nn.Conv2d(C, 2 * C, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(2 * C, 2 * C, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
