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
    Base CNN extractor for SB3.

    Contract:
      - Subclasses implement build_stem() returning a spatial map [B, C', H', W'] (NO Flatten).
      - Base adds a default head: Flatten -> Linear -> ReLU to produce features_dim.

    Channel scaling:
      - c_mult scales all "base channel" choices inside subclasses through self.c(base).
      - Subclasses should never cast shapes or validate c_mult.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 256,
        normalized_image: bool = False,
        *,
        c_mult: int = 1,
    ) -> None:
        if not isinstance(observation_space, spaces.Box):
            raise TypeError(f"Expected Box space, got {type(observation_space)!r}")

        if not is_image_space(
            observation_space,
            check_channels=False,
            normalized_image=bool(normalized_image),
        ):
            raise ValueError(
                "CNN extractors require an image space "
                "(3D Box with dtype uint8 or float).\n"
                f"Got: {observation_space}"
            )

        if int(c_mult) < 1:
            raise ValueError("c_mult must be >= 1")

        super().__init__(observation_space, int(features_dim))

        shape = observation_space.shape
        if shape is None or len(shape) != 3:
            raise ValueError(f"Expected image shape (C,H,W), got {shape!r}")

        # Freeze these as plain ints (type checkers + IDEs stay happy everywhere).
        self.in_ch: int = int(shape[0])
        self.c_mult: int = int(c_mult)
        self.normalized_image: bool = bool(normalized_image)

        # Subclass-defined stem (spatial map).
        self.stem: nn.Module = self.build_stem(observation_space)

        # Infer flattened size from the stem once.
        n_flatten = self._probe_flattened_dim_from_stem(self.stem, observation_space)

        # Default head used by plain CNN extractors.
        self.head: nn.Module = nn.Sequential(
            nn.Flatten(),
            nn.Linear(int(n_flatten), int(features_dim)),
            nn.ReLU(),
        )

    def c(self, base: int) -> int:
        """
        Scale a base channel count by c_mult.
        Use this instead of `base * self.c_mult` in subclasses.
        """
        return int(base) * self.c_mult

    def forward(self, observations: th.Tensor) -> th.Tensor:
        x = self.stem(observations)
        return self.head(x)

    def forward_stem(self, observations: th.Tensor) -> th.Tensor:
        """Expose spatial map for hybrid extractors (e.g., CNN->ViT)."""
        return self.stem(observations)

    def build_stem(self, observation_space: spaces.Box) -> nn.Module:
        """Return CNN stem producing [B,C,H,W] (NO Flatten)."""
        raise NotImplementedError

    @staticmethod
    def _sample_as_tensor(space: spaces.Box) -> th.Tensor:
        sample = space.sample()
        if not isinstance(sample, np.ndarray):
            raise ValueError(f"Sampled observation was not a numpy array: {type(sample)!r}")
        return th.as_tensor(sample[None], dtype=th.float32)

    @classmethod
    def _probe_flattened_dim_from_stem(cls, stem: nn.Module, space: spaces.Box) -> int:
        with th.no_grad():
            x = cls._sample_as_tensor(space)
            y = stem(x)
        if y.ndim != 4:
            raise RuntimeError(f"build_stem() must output [B,C,H,W]. Got shape={tuple(y.shape)}")
        return int(y.shape[1] * y.shape[2] * y.shape[3])
