# src/snake_rl/models/registry.py
from __future__ import annotations

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from snake_rl.models.sb3.extractors import SnakeExtractor

FEATURE_EXTRACTOR_REGISTRY: dict[str, type[BaseFeaturesExtractor]] = {
    "snake_unified": SnakeExtractor,
}


def available_feature_extractors() -> list[str]:
    return sorted(FEATURE_EXTRACTOR_REGISTRY.keys())
