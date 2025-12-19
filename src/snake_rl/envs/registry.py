# src/snake_rl/envs/registry.py
from __future__ import annotations

from snake_rl.envs.base import BaseSnakeEnv
from snake_rl.envs.tile_id_env import GlobalTileIdEnv, PovTileIdEnv
from snake_rl.envs.variants import (
    GlobalPixelDirectionEnv,
    GlobalPixelEnv,
    PovPixelEnv,
    PovPixelFillEnv,
)

ENV_REGISTRY: dict[str, type[BaseSnakeEnv]] = {
    "global_pixel": GlobalPixelEnv,
    "global_pixel_dir": GlobalPixelDirectionEnv,
    "pov_pixel": PovPixelEnv,
    "pov_pixel_fill": PovPixelFillEnv,
    "global_tile_id": GlobalTileIdEnv,
    "pov_tile_id": PovTileIdEnv,
}


def get_env_cls(env_id: str) -> type[BaseSnakeEnv]:
    """
    Resolve an environment class from env_id.

    This is the single source of truth for env lookup/validation.
    """
    key = str(env_id)
    try:
        return ENV_REGISTRY[key]
    except KeyError as e:
        available = ", ".join(sorted(ENV_REGISTRY.keys()))
        raise ValueError(f"Unknown env.id={env_id!r}. Available: {available}") from e


def available_envs() -> list[str]:
    return sorted(ENV_REGISTRY.keys())
