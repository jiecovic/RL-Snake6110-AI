# src/snake_rl/training/policy_factory.py
from __future__ import annotations

from typing import Any

from gymnasium import spaces

from snake_rl.config.schema import TrainConfig
from snake_rl.models.cnns.registry import CNN_REGISTRY
from snake_rl.models.mlps.tile_mlp_extractor import TileMLPExtractor
from snake_rl.models.vits.tile_vit_extractor import TileViTExtractor


def _infer_num_tiles_from_box(space: spaces.Box) -> int:
    """
    Infer vocab size from a tile-id Box space where high == num_tiles-1.
    Works for scalar high or array-like high.
    """
    hi = space.high
    max_hi = float(hi.max()) if hasattr(hi, "max") else float(hi)
    return int(max_hi) + 1


def _get_tiles_box(observation_space) -> spaces.Box:
    """
    Accept either:
      - Box directly (tile ids)
      - Dict with "tiles": Box (future-proof)
    """
    if isinstance(observation_space, spaces.Box):
        return observation_space
    if isinstance(observation_space, spaces.Dict):
        if "tiles" not in observation_space.spaces:
            raise ValueError("Dict observation_space is missing key 'tiles'")
        sub = observation_space.spaces["tiles"]
        if not isinstance(sub, spaces.Box):
            raise ValueError(f"observation_space['tiles'] must be Box, got {type(sub)!r}")
        return sub
    raise ValueError(f"Unsupported observation_space type: {type(observation_space)!r}")


def build_policy_kwargs(*, cfg: TrainConfig, observation_space) -> dict[str, Any]:
    """
    Build SB3 policy_kwargs from TrainConfig.

    - Always passes net_arch (post-extractor MLP).
    - Selects feature extractor via cfg.model.features_extractor.type:
        * CNN_REGISTRY keys -> custom CNN extractor
        * "tile_vit"       -> TileViTExtractor (symbolic tile-id ViT)
        * "tile_mlp"       -> TileMLPExtractor (symbolic tile-id embedding + MLP)
    - Passes cfg.model.features_extractor.params through to the extractor kwargs
      for tile_vit/tile_mlp.
    """
    policy_kwargs: dict[str, Any] = {
        "net_arch": list(cfg.model.net_arch),
    }

    fe = cfg.model.features_extractor
    extractor_key = str(fe.type).strip().lower()
    features_dim = int(fe.features_dim)
    extra_params = dict(fe.params)

    # --- NEW: ViT-like tile extractor ---------------------------------------
    if extractor_key == "tile_vit":
        tiles_box = _get_tiles_box(observation_space)
        num_tiles = _infer_num_tiles_from_box(tiles_box)

        policy_kwargs |= {
            "features_extractor_class": TileViTExtractor,
            "features_extractor_kwargs": {
                "num_tiles": int(num_tiles),
                "features_dim": int(features_dim),
                **extra_params,
            },
        }
        return policy_kwargs

    # --- NEW: MLP tile extractor --------------------------------------------
    if extractor_key == "tile_mlp":
        tiles_box = _get_tiles_box(observation_space)
        num_tiles = _infer_num_tiles_from_box(tiles_box)

        policy_kwargs |= {
            "features_extractor_class": TileMLPExtractor,
            "features_extractor_kwargs": {
                "num_tiles": int(num_tiles),
                "features_dim": int(features_dim),
                **extra_params,
            },
        }
        return policy_kwargs

    # --- Existing: CNN extractor --------------------------------------------
    if isinstance(observation_space, spaces.Box):
        try:
            cnn_cls = CNN_REGISTRY[extractor_key]
        except KeyError as e:
            raise ValueError(
                f"Unknown feature extractor type {extractor_key!r}. "
                f"Available CNNs: {sorted(CNN_REGISTRY.keys())}, plus: 'tile_vit', 'tile_mlp'"
            ) from e

        # For CNNs, we only pass features_dim by default.
        policy_kwargs |= {
            "features_extractor_class": cnn_cls,
            "features_extractor_kwargs": {
                "features_dim": int(features_dim),
            },
        }

    return policy_kwargs
