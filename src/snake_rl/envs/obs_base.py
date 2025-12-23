# src/snake_rl/envs/obs_base.py
from __future__ import annotations

from abc import ABC
from typing import Any, Optional, Tuple, Union

import numpy as np

from snake_rl.game.geometry import Direction
from snake_rl.game.snakegame import SnakeGame

Radius = Union[int, Tuple[int, int]]


def _parse_view_radius(v: Any) -> Tuple[int, int]:
    """
    Parse POV radius.

    Accepted:
      - int r           -> (r, r)
      - (ry, rx) tuple  -> (ry, rx)
      - [ry, rx] list   -> (ry, rx)  (YAML)

    Returns:
      (ry, rx) with ry,rx >= 0
    """
    if isinstance(v, (int, np.integer)):
        r = int(v)
        if r < 0:
            raise ValueError(f"view_radius must be >= 0, got {r}")
        return (r, r)

    if isinstance(v, (tuple, list)) and len(v) == 2:
        ry = int(v[0])
        rx = int(v[1])
        if ry < 0 or rx < 0:
            raise ValueError(f"view_radius must be >= 0, got {(ry, rx)}")
        return (ry, rx)

    raise TypeError(
        "view_radius must be an int or a pair (ry, rx) / [ry, rx], "
        f"got {type(v).__name__}: {v}"
    )


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

    def _pov_pixel_frame(self, *, view_radius: Radius, rotate_to_head: bool = True) -> np.ndarray:
        """
        Return a single POV pixel frame centered on the head.

        view_radius:
          - int r        => square POV (r, r)
          - (ry, rx)     => rectangular POV

        rotate_to_head:
          - True  => egocentric (forward is UP). For rectangular radii, (ry, rx) is in egocentric axes:
                     ry = forward/back radius, rx = left/right radius.
          - False => world-oriented crop (ry vertical, rx horizontal).

        Output: (H,W) uint8
        """
        tilesize = int(self._tilesize)
        pixel_grid = self.game.pixel_buffer

        ry, rx = _parse_view_radius(view_radius)
        view_tiles_y = 2 * ry + 1
        view_tiles_x = 2 * rx + 1
        view_h = view_tiles_y * tilesize
        view_w = view_tiles_x * tilesize

        head_x, head_y = int(self.game.snake[0].x), int(self.game.snake[0].y)

        vision = np.zeros((view_h, view_w), dtype=pixel_grid.dtype)

        if not rotate_to_head:
            # World-oriented rectangular crop (ry vertical, rx horizontal).
            col_start = (head_x - rx) * tilesize
            col_end = (head_x + rx + 1) * tilesize
            row_start = (head_y - ry) * tilesize
            row_end = (head_y + ry + 1) * tilesize

            grid_row_start = max(0, row_start)
            grid_row_end = min(pixel_grid.shape[0], row_end)
            grid_col_start = max(0, col_start)
            grid_col_end = min(pixel_grid.shape[1], col_end)

            vision_row_start = grid_row_start - row_start
            vision_row_end = vision_row_start + (grid_row_end - grid_row_start)
            vision_col_start = grid_col_start - col_start
            vision_col_end = vision_col_start + (grid_col_end - grid_col_start)

            vision[vision_row_start:vision_row_end, vision_col_start:vision_col_end] = pixel_grid[
                grid_row_start:grid_row_end,
                grid_col_start:grid_col_end,
            ]

            return vision.astype(np.uint8, copy=False)

        # Egocentric crop (forward is UP). Implemented by sampling tiles into a fixed (ry, rx) window.
        d = self.game.direction
        assert d is not None, "SnakeGame.direction is None (did you call game.reset()?)"

        # For each ego tile offset (dx, dy) in the output window, map to world tile offset.
        for oy in range(-ry, ry + 1):
            for ox in range(-rx, rx + 1):
                dx_ego = ox
                dy_ego = oy

                if d == Direction.UP:
                    dx_w = dx_ego
                    dy_w = dy_ego
                elif d == Direction.RIGHT:
                    dx_w = -dy_ego
                    dy_w = dx_ego
                elif d == Direction.DOWN:
                    dx_w = -dx_ego
                    dy_w = -dy_ego
                elif d == Direction.LEFT:
                    dx_w = dy_ego
                    dy_w = -dx_ego
                else:
                    dx_w = dx_ego
                    dy_w = dy_ego

                wx = head_x + dx_w
                wy = head_y + dy_w

                if not (0 <= wx < self.game.width and 0 <= wy < self.game.height):
                    continue

                # Copy that tile's pixel block into the egocentric buffer.
                src_y0 = wy * tilesize
                src_y1 = src_y0 + tilesize
                src_x0 = wx * tilesize
                src_x1 = src_x0 + tilesize

                dst_y0 = (oy + ry) * tilesize
                dst_y1 = dst_y0 + tilesize
                dst_x0 = (ox + rx) * tilesize
                dst_x1 = dst_x0 + tilesize

                vision[dst_y0:dst_y1, dst_x0:dst_x1] = pixel_grid[src_y0:src_y1, src_x0:src_x1]

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
