# src/snake_rl/models/cnns/lite_cnn8.py
from gymnasium import spaces
from torch import nn

from .base_cnn_extractor import BaseCNNExtractor


class LiteCNN8(BaseCNNExtractor):
    """
    Lightweight kernel-8 CNN.
    Fewer channels than NatureCNN.
    Experimental but controlled baseline.
    """

    def build_cnn(self, observation_space: spaces.Box) -> nn.Module:
        in_ch = observation_space.shape[0]
        C = 8

        return nn.Sequential(
            nn.Conv2d(in_ch, C, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(C, 2 * C, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(2 * C, 2 * C, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
