# src/snake_rl/models/cnns/base.py
from __future__ import annotations

import numpy as np
import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.preprocessing import is_image_space
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class BaseCNNExtractor(BaseFeaturesExtractor):
    """
    Base class for CNN feature extractors compatible with Stable-Baselines3.

    Design:
    - Expects channel-first image observations: (C, H, W)
    - Subclasses implement build_cnn() only
    - Feature dimensionality is inferred automatically via a forward probe
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 256,
        normalized_image: bool = False,
    ):
        if not isinstance(observation_space, spaces.Box):
            raise TypeError(f"Expected Box space, got {type(observation_space)}")

        if not is_image_space(
            observation_space,
            check_channels=False,
            normalized_image=normalized_image,
        ):
            raise ValueError(
                "CNN extractors require an image space "
                "(3D Box with dtype uint8 or float).\n"
                f"Got: {observation_space}"
            )

        super().__init__(observation_space, features_dim)
        self.normalized_image = normalized_image

        # Subclass provides CNN backbone (must end with Flatten)
        self.cnn = self.build_cnn(observation_space)

        # Infer flattened feature size via a dummy forward pass
        with th.no_grad():
            sample = observation_space.sample()
            if not isinstance(sample, np.ndarray):
                raise ValueError(
                    f"Sampled observation was not a numpy array: {type(sample)}"
                )

            sample_tensor = th.as_tensor(sample[None]).float()
            try:
                n_flatten = self.cnn(sample_tensor).shape[1]
            except Exception as e:
                raise RuntimeError(
                    f"Error computing flattened CNN size: {e}"
                ) from e

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

    def build_cnn(self, observation_space: spaces.Box) -> nn.Module:
        """Return CNN backbone ending in nn.Flatten()."""
        raise NotImplementedError
