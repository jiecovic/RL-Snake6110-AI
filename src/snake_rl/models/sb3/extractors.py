# src/snake_rl/models/sb3/extractors.py
from __future__ import annotations

from typing import Dict, Optional, Type

import torch as th
from gymnasium import spaces
from torch import nn

from stable_baselines3.common.preprocessing import get_flattened_obs_dim, is_image_space
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, NatureCNN
from stable_baselines3.common.type_aliases import TensorDict


class CustomCombinedExtractor(BaseFeaturesExtractor):
    """
    SB3 features extractor for Dict observations.

    - For image-like subspaces (Box with shape (C,H,W)), uses a CNN-style features extractor.
    - For non-image subspaces, flattens and concatenates.
    """

    def __init__(
            self,
            observation_space: spaces.Dict,
            cnn_features_dim: int = 512,
            normalized_image: bool = False,
            cnn_extractor_class: Optional[Type[BaseFeaturesExtractor]] = None,
    ) -> None:
        # features_dim is set after we build sub-extractors
        super().__init__(observation_space, features_dim=1)

        if cnn_extractor_class is None:
            cnn_extractor_class = NatureCNN

        extractors: Dict[str, nn.Module] = {}
        total_concat_size = 0

        for key, subspace in observation_space.spaces.items():
            if is_image_space(subspace, check_channels=False, normalized_image=normalized_image):
                # CNN-style features extractor (must be BaseFeaturesExtractor-compatible)
                extractors[key] = cnn_extractor_class(
                    subspace,  # type: ignore[arg-type]
                    features_dim=int(cnn_features_dim),
                    normalized_image=normalized_image,
                )
                total_concat_size += int(cnn_features_dim)
            else:
                extractors[key] = nn.Flatten()
                total_concat_size += get_flattened_obs_dim(subspace)

        self.extractors = nn.ModuleDict(extractors)
        self._features_dim = total_concat_size

    def forward(self, observations: TensorDict) -> th.Tensor:
        encoded_parts = []
        for key, extractor in self.extractors.items():
            encoded_parts.append(extractor(observations[key]))
        return th.cat(encoded_parts, dim=1)
