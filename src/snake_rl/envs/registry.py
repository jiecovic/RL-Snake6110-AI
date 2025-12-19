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
