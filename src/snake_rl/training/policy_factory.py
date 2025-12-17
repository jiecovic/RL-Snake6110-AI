# src/snake_rl/training/policy_factory.py
from __future__ import annotations

from typing import Any

from gymnasium import spaces

from snake_rl.config.schema import TrainConfig
from snake_rl.models.cnns.registry import CNN_REGISTRY


def build_policy_kwargs(*, cfg: TrainConfig, observation_space) -> dict[str, Any]:
    """
    Build SB3 policy_kwargs from TrainConfig.

    - Always passes net_arch (post-CNN MLP).
    - Selects custom CNN feature extractor via CNN_REGISTRY using cfg.model.cnn.type.
    """
    policy_kwargs: dict[str, Any] = {
        "net_arch": list(cfg.model.net_arch),
    }

    if isinstance(observation_space, spaces.Box):
        cnn_key = str(cfg.model.cnn.type)
        try:
            cnn_cls = CNN_REGISTRY[cnn_key]
        except KeyError as e:
            raise ValueError(
                f"Unknown CNN type {cnn_key!r}. Available: {sorted(CNN_REGISTRY.keys())}"
            ) from e

        policy_kwargs |= {
            "features_extractor_class": cnn_cls,
            "features_extractor_kwargs": {
                "features_dim": int(cfg.model.cnn.features_dim),
            },
        }

    return policy_kwargs
