# src/snake_rl/envs/tile_id_env.py
from __future__ import annotations

import numpy as np
from gymnasium import spaces

from snake_rl.envs.base import BaseSnakeEnv
from snake_rl.game.geometry import Direction, Point
from snake_rl.game.snakegame import SnakeGame
from snake_rl.game.tile_types import TileType


def _tile_vocab_size() -> int:
    # We store TileType.value directly in SnakeGame.tile_grid (uint8).
    # Vocab size is max enum value + 1, assuming values are 0..K.
    return int(max(int(t.value) for t in TileType) + 1)


class GlobalTileIdEnv(BaseSnakeEnv):
    """
    Global symbolic grid observation (no pixels).

    Observation:
      Box(shape=(1, H, W), dtype=uint8)
        value == TileType.value (TileType.EMPTY=0)

    Notes:
    - Intended for transformer / symbolic models.
    - Frame stacking works out-of-the-box (stacked along channel dimension).
    - Optional cropping (remove_border) is handled here (env-level), not in SnakeGame.
    """

    def __init__(self, game: SnakeGame, *, remove_border: bool = True):
        BaseSnakeEnv.__init__(self, game)

        self.remove_border = bool(remove_border)

        # Keep historical behavior: env ctor forces a clean state.
        self.game.reset()

        h, w = int(self.game.height), int(self.game.width)
        if self.remove_border:
            if h <= 2 or w <= 2:
                raise ValueError(f"remove_border=True requires h,w > 2; got h={h} w={w}")
            h -= 2
            w -= 2

        self.num_tile_ids: int = _tile_vocab_size()

        self.observation_space = spaces.Box(
            low=0,
            high=self.num_tile_ids - 1,
            shape=(1, h, w),
            dtype=np.uint8,
        )

    def _get_grid_view(self) -> np.ndarray:
        g = self.game.tile_grid  # (H,W) uint8, values are TileType.value
        if not self.remove_border:
            return g
        return g[1:-1, 1:-1]

    def get_obs(self):
        grid = self._get_grid_view()
        return grid[None, :, :].astype(np.uint8, copy=False)


class PovTileIdEnv(BaseSnakeEnv):
    """
    POV symbolic grid observation (tile ids), centered on head and rotated so forward is UP.

    Observation:
      Box(shape=(1, V, V), dtype=uint8)
        value == TileType.value (TileType.EMPTY=0)

    Where:
      V = 2*view_radius + 1

    """

    def __init__(
            self,
            game: SnakeGame,
            *,
            view_radius: int,
    ):
        BaseSnakeEnv.__init__(self, game)

        self.view_radius = int(view_radius)
        if self.view_radius < 0:
            raise ValueError(f"view_radius must be >= 0, got {self.view_radius}")

        # Keep historical behavior: env ctor forces a clean state.
        self.game.reset()

        v = 2 * self.view_radius + 1

        self.num_tile_ids: int = _tile_vocab_size()

        self.observation_space = spaces.Box(
            low=0,
            high=self.num_tile_ids - 1,
            shape=(1, v, v),
            dtype=np.uint8,
        )

    def _pov_tile_frame(self) -> np.ndarray:
        g = self.game.tile_grid  # (H,W) uint8

        head: Point = self.game.get_head_position()
        hx, hy = int(head.x), int(head.y)

        r = self.view_radius
        v = 2 * r + 1

        # Window bounds in grid coordinates
        x0, x1 = hx - r, hx + r + 1
        y0, y1 = hy - r, hy + r + 1

        out = np.full((v, v), int(TileType.EMPTY.value), dtype=np.uint8)

        # Overlap with g
        gy0 = max(0, y0)
        gy1 = min(g.shape[0], y1)
        gx0 = max(0, x0)
        gx1 = min(g.shape[1], x1)

        oy0 = gy0 - y0
        oy1 = oy0 + (gy1 - gy0)
        ox0 = gx0 - x0
        ox1 = ox0 + (gx1 - gx0)

        out[oy0:oy1, ox0:ox1] = g[gy0:gy1, gx0:gx1]

        # Rotate so the head faces UP (match PixelObsEnvBase convention)
        d = self.game.direction
        if d == Direction.RIGHT:
            out = np.rot90(out, k=1)
        elif d == Direction.DOWN:
            out = np.rot90(out, k=2)
        elif d == Direction.LEFT:
            out = np.rot90(out, k=3)

        return out

    def get_obs(self):
        frame = self._pov_tile_frame()
        return frame[None, :, :].astype(np.uint8, copy=False)
