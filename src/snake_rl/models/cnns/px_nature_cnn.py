# src/snake_rl/models/cnns/px_nature_cnn.py
from __future__ import annotations

from gymnasium import spaces
from torch import nn

from snake_rl.models.cnns.base import BaseCNNExtractor


class PxNatureCNN(BaseCNNExtractor):
    """
    Classic Nature-DQN CNN (fixed reference).

    IMPORTANT:
      - This extractor deliberately ignores any channel scaling (e.g. c_mult).
      - It is meant to be a stable baseline architecture: 32 -> 64 -> 64.
    """

    def build_stem(self, observation_space: spaces.Box) -> nn.Module:
        in_ch = int(observation_space.shape[0])

        # Fixed channels: do NOT scale with c_mult.
        return nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
