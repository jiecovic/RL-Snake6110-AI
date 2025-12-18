# src/snake_rl/envs/tile_id_env.py
from __future__ import annotations

import numpy as np
from gymnasium import spaces

from snake_rl.envs.base import BaseSnakeEnv
from snake_rl.game.snakegame import SnakeGame
from snake_rl.game.tile_types import TileType


def _build_tile_id_lut() -> dict[TileType, int]:
    """
    Stable mapping TileType -> integer tile id.

    Convention:
      - 0 is reserved for EMPTY
      - TileType enum members are mapped to 1..N in definition order

    This avoids relying on Enum.auto() values and keeps IDs stable.
    """
    lut: dict[TileType, int] = {}
    idx = 1
    for tt in TileType:
        lut[tt] = idx
        idx += 1
    return lut


_TILE_ID_LUT = _build_tile_id_lut()
_EMPTY_ID = 0
_NUM_TILE_IDS = 1 + len(_TILE_ID_LUT)  # includes EMPTY


class GlobalTileIdEnv(BaseSnakeEnv):
    """
    Global symbolic grid observation (no pixels).

    Observation:
      Box(shape=(1, H, W), dtype=uint8)
        0          = empty
        1..N       = TileType ids (snake head/body/tail orientation, walls, food)

    Notes:
    - Intended for transformer / symbolic models.
    - Frame stacking works out-of-the-box (stacked along channel dimension).
    """

    def __init__(self, game: SnakeGame):
        BaseSnakeEnv.__init__(self, game)

        self.game.reset()
        h, w = int(self.game.height), int(self.game.width)

        # Exposed for model construction (e.g. embedding table size)
        self.num_tile_ids: int = int(_NUM_TILE_IDS)

        self.observation_space = spaces.Box(
            low=0,
            high=self.num_tile_ids - 1,
            shape=(1, h, w),
            dtype=np.uint8,
        )

    def _build_tile_id_grid(self) -> np.ndarray:
        """
        Build a (H, W) grid of tile IDs from the current game state.

        Priority order:
          snake > food > walls > empty
        """
        h, w = int(self.game.height), int(self.game.width)
        grid = np.full((h, w), _EMPTY_ID, dtype=np.uint8)

        # Walls
        for pos, tile_type in self.game.wall_tiles:
            grid[pos.y, pos.x] = _TILE_ID_LUT.get(tile_type, _EMPTY_ID)

        # Food
        if hasattr(TileType, "FOOD"):
            food_id = _TILE_ID_LUT[TileType.FOOD]
            for p in self.game.food:
                grid[p.y, p.x] = food_id

        # Snake layers (already encode orientation as TileType)
        for tile_type, layer in self.game.snake_layers.items():
            tid = _TILE_ID_LUT.get(tile_type, _EMPTY_ID)
            if tid == _EMPTY_ID:
                continue
            yxs = np.argwhere(layer > 0)
            for y, x in yxs:
                grid[int(y), int(x)] = tid

        return grid

    def get_obs(self):
        grid = self._build_tile_id_grid()
        return grid[None, :, :].astype(np.uint8, copy=False)
