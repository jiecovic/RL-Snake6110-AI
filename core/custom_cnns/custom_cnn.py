from gymnasium import spaces
from torch import nn

from .base_cnn_extractor import BaseCNNExtractor

class SnakeCNN_3Layers(BaseCNNExtractor):
    """
    Classic Nature-DQN style CNN.
    """

    def build_cnn(self, observation_space: spaces.Box) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten()
        )

class SnakeCNN_Deep(BaseCNNExtractor):
    """
    Classic Nature-DQN style CNN.
    """

    def build_cnn(self, observation_space: spaces.Box) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten()
        )

