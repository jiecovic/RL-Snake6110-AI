# src/snake_rl/models/cnns/nature_cnn8.py
from gymnasium import spaces
from torch import nn

from .base_cnn_extractor import BaseCNNExtractor


class NatureCNN8(BaseCNNExtractor):
    """
    Classic Nature-DQN CNN.
    Kernel sizes: 8-4-3.
    Reference baseline.
    """

    def build_cnn(self, observation_space: spaces.Box) -> nn.Module:
        in_ch = observation_space.shape[0]

        return nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
