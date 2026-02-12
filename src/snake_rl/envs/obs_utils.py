# src/snake_rl/envs/obs_utils.py
from __future__ import annotations

import numpy as np


def world_pixel_frame(
    *,
    pixel_grid: np.ndarray,
    tile_size: int,
    remove_border: bool,
) -> np.ndarray:
    if not remove_border:
        return pixel_grid
    ts = int(tile_size)
    return pixel_grid[ts:-ts, ts:-ts]


def world_tile_frame(*, tile_grid: np.ndarray, remove_border: bool) -> np.ndarray:
    if not remove_border:
        return tile_grid
    return tile_grid[1:-1, 1:-1]


__all__ = [
    "world_pixel_frame",
    "world_tile_frame",
]
