# src/snake_rl/training/policy_factory.py
from __future__ import annotations

from typing import Any

from gymnasium import spaces

from snake_rl.config.schema import TrainConfig
from snake_rl.models.registry import FEATURE_EXTRACTOR_REGISTRY, available_feature_extractors


def _infer_num_tiles_from_box(space: spaces.Box) -> int:
    """
    Infer vocab size from a tile-id Box space where high == num_tiles-1.
    Works for scalar high or array-like high.
    """
    hi = space.high
    max_hi = float(hi.max()) if hasattr(hi, "max") else float(hi)
    return int(max_hi) + 1


def _get_tiles_box(observation_space: spaces.Space) -> spaces.Box:
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


def build_policy_kwargs(*, cfg: TrainConfig, observation_space: spaces.Space) -> dict[str, Any]:
    """
    Build SB3 policy_kwargs from TrainConfig.

    In snake_rl, "models" are SB3 feature extractors. The PPO policy head remains SB3's
    default (ActorCritic*Policy), optionally with a post-extractor MLP (net_arch).

    Selection:
      cfg.model.features_extractor.type must be a key in FEATURE_EXTRACTOR_REGISTRY.

    Conventions:
      - px_* extractors: pixel-based (typically CNN); kwargs: {features_dim}
      - tile_* extractors: symbolic tile-id; kwargs: {num_tiles, features_dim, **params}

    IMPORTANT:
      For tile_* extractors, num_tiles is inferred from observation_space.high (+1).
      If the user provides num_tiles in config params, we validate it matches to avoid
      silent mismatches when swapping vocabs.
    """
    fe = cfg.model.features_extractor
    extractor_key = str(fe.type).strip().lower()
    features_dim = int(fe.features_dim)
    extra_params = dict(fe.params)

    try:
        extractor_cls = FEATURE_EXTRACTOR_REGISTRY[extractor_key]
    except KeyError as e:
        raise ValueError(
            f"Unknown feature extractor type {extractor_key!r}. "
            f"Available: {available_feature_extractors()}"
        ) from e

    policy_kwargs: dict[str, Any] = {
        "net_arch": list(cfg.model.net_arch),
        "features_extractor_class": extractor_cls,
    }

    # Tile-id models need vocab size inferred from the observation space.
    if extractor_key.startswith("tile_"):
        tiles_box = _get_tiles_box(observation_space)
        inferred_num_tiles = _infer_num_tiles_from_box(tiles_box)

        # Guard against silent override / mismatch.
        user_num_tiles = extra_params.pop("num_tiles", None)
        if user_num_tiles is not None and int(user_num_tiles) != int(inferred_num_tiles):
            raise ValueError(
                f"Config provided num_tiles={user_num_tiles}, but observation_space implies "
                f"num_tiles={inferred_num_tiles}. Remove num_tiles from model.features_extractor.params "
                f"or fix your env/vocab."
            )

        policy_kwargs["features_extractor_kwargs"] = {
            "num_tiles": int(inferred_num_tiles),
            "features_dim": int(features_dim),
            **extra_params,
        }
        return policy_kwargs

    # Pixel models (CNNs): keep kwargs minimal and strict.
    if extractor_key.startswith("px_"):
        if isinstance(observation_space, spaces.Dict):
            raise ValueError(
                f"Extractor {extractor_key!r} expects a Box observation_space (pixel), "
                f"but got Dict with keys={list(observation_space.spaces.keys())}."
            )
        if not isinstance(observation_space, spaces.Box):
            raise ValueError(
                f"Extractor {extractor_key!r} expects a Box observation_space (pixel), "
                f"but got {type(observation_space)!r}."
            )

        # Allow params for specific px_* hybrids (e.g., px_cnn_vit) while keeping others strict.
        allow_px_params = {"px_cnn_vit"}
        if extra_params and extractor_key not in allow_px_params:
            raise ValueError(
                f"Extractor {extractor_key!r} does not accept params; got: {sorted(extra_params.keys())}"
            )

        policy_kwargs["features_extractor_kwargs"] = {
            "features_dim": int(features_dim),
            **(extra_params if extractor_key in allow_px_params else {}),
        }
        return policy_kwargs

    # If we ever introduce other prefixes, force an explicit decision.
    raise ValueError(
        f"Unsupported extractor key prefix for {extractor_key!r}. "
        f"Expected 'px_*' or 'tile_*'. Available: {available_feature_extractors()}"
    )
