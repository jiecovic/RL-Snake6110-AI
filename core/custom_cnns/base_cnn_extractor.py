import gym
import torch as th
import torch.nn as nn
from typing import Optional
from gymnasium import spaces
import numpy as np
from stable_baselines3.common.preprocessing import is_image_space
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class BaseCNNExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 256,
        normalized_image: bool = False,
    ):
        if not isinstance(observation_space, spaces.Box):
            raise TypeError(f"Expected Box space, got {type(observation_space)}")
        if not is_image_space(observation_space, check_channels=False, normalized_image=normalized_image):
            raise ValueError(
                f"CNN extractors require an image space (3D Box with dtype uint8 or float).\n"
                f"Got: {observation_space}"
            )

        super().__init__(observation_space, features_dim)
        self.normalized_image = normalized_image

        # Subclass provides CNN architecture
        self.cnn = self.build_cnn(observation_space)

        # Compute number of features after CNN
        with th.no_grad():
            sample = observation_space.sample()
            if isinstance(sample, np.ndarray):
                sample_tensor = th.as_tensor(sample[None]).float()
            else:
                raise ValueError(f"Sampled observation was not a numpy array: {type(sample)}")

            try:
                n_flatten = self.cnn(sample_tensor).shape[1]
            except Exception as e:
                raise RuntimeError(f"Error computing n_flatten in CNN: {e}")

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

    def build_cnn(self, observation_space: spaces.Box) -> nn.Module:
        raise NotImplementedError("Subclasses must implement build_cnn()")

