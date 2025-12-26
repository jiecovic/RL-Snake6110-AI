# src/snake_rl/envs/pixel_obs.py
from __future__ import annotations

from abc import ABC
from typing import Any, Optional, Tuple, Union, overload

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


def _rotate_tile_block(block: np.ndarray, d: Direction) -> np.ndarray:
    """
    Rotate a tile's pixel block so the egocentric view has "forward == UP".

    The egocentric sampling already maps world coordinates into the ego window.
    But the *sprites inside each tile* (e.g. directional head, corners) are still
    in world orientation, so we rotate the pixel block itself.

    Convention (np.rot90): k>0 rotates CCW.
      - If snake faces UP    : no rotation
      - RIGHT (east)         : rotate CCW 90 (k=1)
      - DOWN (south)         : rotate 180 (k=2)
      - LEFT (west)          : rotate CW 90  (k=3)
    """
    if d == Direction.UP:
        k = 0
    elif d == Direction.RIGHT:
        k = 1
    elif d == Direction.DOWN:
        k = 2
    elif d == Direction.LEFT:
        k = 3
    else:
        k = 0

    if k == 0:
        return block
    return np.rot90(block, k=k).copy()


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

    @overload
    def _pov_pixel_frame(
            self,
            *,
            view_radius: Radius,
            rotate_to_head: bool = True,
            oob_fill_value: int = 0,
            return_valid: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        ...

    @overload
    def _pov_pixel_frame(
            self,
            *,
            view_radius: Radius,
            rotate_to_head: bool = True,
            oob_fill_value: int = 0,
            return_valid: bool = False,
    ) -> np.ndarray:
        ...

    def _pov_pixel_frame(
            self,
            *,
            view_radius: Radius,
            rotate_to_head: bool = True,
            oob_fill_value: int = 0,
            return_valid: bool = False,
    ):
        """
        Return a single POV pixel frame centered on the head.

        view_radius:
          - int r        => square POV (r, r)
          - (ry, rx)     => rectangular POV

        rotate_to_head:
          - True  => egocentric (forward is UP). For rectangular radii, (ry, rx) is in egocentric axes:
                     ry = forward/back radius, rx = left/right radius.
          - False => world-oriented crop (ry vertical, rx horizontal).

        oob_fill_value:
          - Pixel value used for padded OOB regions (uint8-ish int).

        return_valid:
          - If True, also return a boolean valid mask (H,W) where True means "in-bounds board pixels".

        Output:
          - frame: (H,W) uint8
          - optionally valid: (H,W) bool
        """
        tilesize = int(self._tilesize)
        pixel_grid = self.game.pixel_buffer  # (H,W), uint8-ish

        ry, rx = _parse_view_radius(view_radius)
        view_tiles_y = 2 * ry + 1
        view_tiles_x = 2 * rx + 1
        view_h = view_tiles_y * tilesize
        view_w = view_tiles_x * tilesize

        head_x, head_y = int(self.game.snake[0].x), int(self.game.snake[0].y)

        fill = np.uint8(int(oob_fill_value) & 0xFF)
        vision = np.full((view_h, view_w), fill, dtype=np.uint8)

        valid: Optional[np.ndarray] = None
        if return_valid:
            valid = np.zeros((view_h, view_w), dtype=np.bool_)

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

            if grid_row_end > grid_row_start and grid_col_end > grid_col_start:
                vision[vision_row_start:vision_row_end, vision_col_start:vision_col_end] = pixel_grid[
                    grid_row_start:grid_row_end,
                    grid_col_start:grid_col_end,
                ].astype(np.uint8, copy=False)
                if valid is not None:
                    valid[vision_row_start:vision_row_end, vision_col_start:vision_col_end] = True

            if return_valid:
                assert valid is not None
                return vision, valid
            return vision

        # Egocentric crop (forward is UP). We sample tiles into a fixed (ry, rx) window.
        # IMPORTANT: we must also rotate each tile's *pixel block* so directional sprites look correct.
        d = self.game.direction
        assert d is not None, "SnakeGame.direction is None (did you call game.reset()?)"

        for oy in range(-ry, ry + 1):
            for ox in range(-rx, rx + 1):
                dx_ego = ox
                dy_ego = oy

                # Map ego offsets -> world offsets (so the correct tile is selected)
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

                # Source tile block in the *world-oriented* pixel buffer
                src_y0 = wy * tilesize
                src_y1 = src_y0 + tilesize
                src_x0 = wx * tilesize
                src_x1 = src_x0 + tilesize

                # Destination tile block in the ego view
                dst_y0 = (oy + ry) * tilesize
                dst_y1 = dst_y0 + tilesize
                dst_x0 = (ox + rx) * tilesize
                dst_x1 = dst_x0 + tilesize

                block = pixel_grid[src_y0:src_y1, src_x0:src_x1].astype(np.uint8, copy=False)

                # Rotate the *tile pixels* so forward==UP visually
                block = _rotate_tile_block(block, d)

                vision[dst_y0:dst_y1, dst_x0:dst_x1] = block
                if valid is not None:
                    valid[dst_y0:dst_y1, dst_x0:dst_x1] = True

        if return_valid:
            assert valid is not None
            return vision, valid
        return vision


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
