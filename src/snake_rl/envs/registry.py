# src/snake_rl/envs/registry.py
from __future__ import annotations

from snake_rl.envs.base import BaseSnakeEnv
from snake_rl.envs.pixel_envs import (
    HeadPixelEnv,
    HeadPixelFillEnv,
    WorldPixelDirectionEnv,
    WorldPixelEnv,
)
from snake_rl.envs.tile_id_env import HeadTileIdEnv, WorldTileIdEnv

ENV_REGISTRY: dict[str, type[BaseSnakeEnv]] = {
    "world_pixel": WorldPixelEnv,
    "world_pixel_dir": WorldPixelDirectionEnv,
    "head_pixel": HeadPixelEnv,
    "head_pixel_fill": HeadPixelFillEnv,
    "world_tile_id": WorldTileIdEnv,
    "head_tile_id": HeadTileIdEnv,
}

# Back-compat aliases for older config names.
ENV_ALIASES: dict[str, str] = {
    "global_pixel": "world_pixel",
    "global_pixel_dir": "world_pixel_dir",
    "pov_pixel": "head_pixel",
    "pov_pixel_fill": "head_pixel_fill",
    "global_tile_id": "world_tile_id",
    "pov_tile_id": "head_tile_id",
}


def get_env_cls(env_id: str) -> type[BaseSnakeEnv]:
    """
    Resolve an environment class from env_id.

    This is the single source of truth for env lookup/validation.
    """
    key = str(env_id)
    if key in ENV_ALIASES:
        key = ENV_ALIASES[key]
    try:
        return ENV_REGISTRY[key]
    except KeyError as e:
        available = ", ".join(sorted(ENV_REGISTRY.keys()))
        raise ValueError(f"Unknown env.id={env_id!r}. Available: {available}") from e


def available_envs() -> list[str]:
    return sorted(ENV_REGISTRY.keys())
