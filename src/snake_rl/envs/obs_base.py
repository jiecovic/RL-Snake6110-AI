# src/snake_rl/envs/obs_base.py
from __future__ import annotations

from abc import ABC
from typing import Optional

import numpy as np

from snake_rl.game.geometry import Direction
from snake_rl.game.snakegame import SnakeGame


class PixelObsEnvBase(ABC):
    """
    Small helper base for pixel observations.

    This does NOT implement gym.Env. Concrete envs inherit this alongside BaseSnakeEnv.
    It exists to keep pixel-frame extraction logic (global vs POV) out of each env.
    """

    def __init__(self, game: SnakeGame):
        self.game = game
        self._tilesize = self.game.tileset.tile_size

    def _global_pixel_frame(self) -> np.ndarray:
        """
        Return a single global pixel frame as (H,W) uint8.
        """
        return self.game.pixel_buffer.astype(np.uint8, copy=False)

    def _pov_pixel_frame(self, *, view_radius: int, rotate_to_head: bool = True) -> np.ndarray:
        """
        Return a single POV pixel frame centered on the head.

        If rotate_to_head=True (default):
          - rotate so the head faces UP (egocentric POV).
        If rotate_to_head=False:
          - keep world orientation (allocentric POV).

        Output: (view_px, view_px) uint8
        """
        head_x, head_y = self.game.snake[0].x, self.game.snake[0].y
        tilesize = self._tilesize
        pixel_grid = self.game.pixel_buffer

        radius = int(view_radius)
        view_tiles = 2 * radius + 1
        view_size = view_tiles * tilesize

        # Full window bounds in pixel coordinates
        col_start = (head_x - radius) * tilesize
        col_end = (head_x + radius + 1) * tilesize
        row_start = (head_y - radius) * tilesize
        row_end = (head_y + radius + 1) * tilesize

        vision = np.zeros((view_size, view_size), dtype=pixel_grid.dtype)

        # Overlap with pixel_grid
        grid_row_start = max(0, row_start)
        grid_row_end = min(pixel_grid.shape[0], row_end)
        grid_col_start = max(0, col_start)
        grid_col_end = min(pixel_grid.shape[1], col_end)

        # Placement into vision buffer
        vision_row_start = grid_row_start - row_start
        vision_row_end = vision_row_start + (grid_row_end - grid_row_start)
        vision_col_start = grid_col_start - col_start
        vision_col_end = vision_col_start + (grid_col_end - grid_col_start)

        vision[vision_row_start:vision_row_end, vision_col_start:vision_col_end] = pixel_grid[
            grid_row_start:grid_row_end,
            grid_col_start:grid_col_end,
        ]

        if rotate_to_head:
            # Rotate so the head faces UP
            direction = self.game.direction
            if direction == Direction.RIGHT:
                vision = np.rot90(vision, k=1)
            elif direction == Direction.DOWN:
                vision = np.rot90(vision, k=2)
            elif direction == Direction.LEFT:
                vision = np.rot90(vision, k=3)

        return vision.astype(np.uint8, copy=False)


class FillFeature:
    """
    Global 'crampedness' / 'fill' feature helper.

    Interpretation:
    - Normalizes current snake length relative to playable tiles.
    - Optionally bins the value into a discrete number of bins.
    """

    def __init__(self, *, fill_bins: Optional[int] = None):
        self.fill_bins = None if fill_bins is None else int(fill_bins)

    def compute(self, *, snake_len: int, initial_len: int, max_playable: int):
        # Normalize "how far we are into filling the board" (0..1)
        denom = max(1, (max_playable - initial_len))
        x = (snake_len - initial_len) / denom
        x = float(np.clip(x, 0.0, 1.0))

        if self.fill_bins is None:
            # Scalar feature as Box(1,) float32 (SB3-friendly)
            return np.array([x], dtype=np.float32)

        # Discrete binned feature (0..bins-1)
        b = int(np.floor(x * self.fill_bins))
        return min(b, self.fill_bins - 1)
