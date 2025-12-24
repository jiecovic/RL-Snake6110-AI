# src/snake_rl/envs/tile_id_env.py
from __future__ import annotations

from typing import Any, Tuple

import numpy as np
from gymnasium import spaces

from snake_rl.envs.base import BaseSnakeEnv
from snake_rl.game.geometry import Direction, Point
from snake_rl.game.snakegame import SnakeGame
from snake_rl.game.tile_types import TileType
from snake_rl.vocab import load_tile_vocab


def _tile_vocab_size() -> int:
    # We store TileType.value directly in SnakeGame.tile_grid (uint8).
    # Vocab size is max enum value + 1, assuming values are 0..K.
    return int(max(int(t.value) for t in TileType) + 1)


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


class GlobalTileIdEnv(BaseSnakeEnv):
    """
    Global symbolic grid observation (no pixels).

    Observation:
      Box(shape=(1, H, W), dtype=uint8)
        value == class_id in [0, num_classes-1]

    Notes:
    - If tile_vocab is None (default), class_id == TileType.value (raw IDs).
    - Intended for transformer / symbolic models.
    - Frame stacking works out-of-the-box (stacked along channel dimension).
    - Optional cropping (remove_border) is handled here (env-level), not in SnakeGame.
    """

    def __init__(
            self,
            game: SnakeGame,
            *,
            remove_border: bool = True,
            tile_vocab: str | None = None,
    ):
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

        # Raw TileType vocab size (TileType.value)
        self.raw_vocab_size: int = _tile_vocab_size()

        # Optional: map raw TileType IDs -> compact class IDs.
        self._tile_vocab = None
        if tile_vocab is not None:
            self._tile_vocab = load_tile_vocab(tile_vocab)

        self.num_classes: int = (
            int(self._tile_vocab.num_classes) if self._tile_vocab is not None else int(self.raw_vocab_size)
        )

        # Expose for logging/debug
        self.tile_vocab_name: str | None = self._tile_vocab.name if self._tile_vocab is not None else None
        self.tile_vocab_sha256: str | None = self._tile_vocab.sha256 if self._tile_vocab is not None else None

        self.observation_space = spaces.Box(
            low=0,
            high=self.num_classes - 1,
            shape=(1, h, w),
            dtype=np.uint8,
        )

        # Optional: last grids for tooling (agent-view IDs later)
        self._last_raw_grid: np.ndarray | None = None
        self._last_class_grid: np.ndarray | None = None

    def _get_grid_view(self) -> np.ndarray:
        g = self.game.tile_grid  # (H,W) uint8, values are TileType.value
        if not self.remove_border:
            return g
        return g[1:-1, 1:-1]

    def get_obs(self):
        raw = self._get_grid_view()
        if self._tile_vocab is None:
            grid = raw
        else:
            grid = self._tile_vocab.lut[raw]

        self._last_raw_grid = raw
        self._last_class_grid = grid

        return grid[None, :, :].astype(np.uint8, copy=False)


class PovTileIdEnv(BaseSnakeEnv):
    """
    POV symbolic grid observation (tile ids), centered on head.

    Observation:
      Box(shape=(1, VY, VX), dtype=uint8)
        value == class_id in [0, num_classes-1]

    Where:
      VY = 2*ry + 1
      VX = 2*rx + 1

    Notes:
    - If tile_vocab is None (default), class_id == TileType.value (raw IDs).
    - rotate_to_head=True uses an egocentric view where:
        ry = forward/back radius
        rx = left/right radius
      (shape remains constant even when rx != ry)
    - rotate_to_head=False crops in world axes (ry vertical, rx horizontal).
    - If mask_oob=True:
        - 0 is reserved as the OOB sentinel
        - all in-bounds IDs are shifted by +1
        - num_classes increases by +1
    """

    def __init__(
            self,
            game: SnakeGame,
            *,
            view_radius: int | tuple[int, int],
            tile_vocab: str | None = None,
            rotate_to_head: bool = True,
            mask_oob: bool = False,
    ):
        BaseSnakeEnv.__init__(self, game)

        self.view_radius_y, self.view_radius_x = _parse_view_radius(view_radius)
        self.rotate_to_head = bool(rotate_to_head)
        self.mask_oob = bool(mask_oob)

        # Expose for debugging/logging.
        self.oob_id: int | None = 0 if self.mask_oob else None
        self._id_shift: int = 1 if self.mask_oob else 0

        # Keep historical behavior: env ctor forces a clean state.
        self.game.reset()

        vy = 2 * self.view_radius_y + 1
        vx = 2 * self.view_radius_x + 1

        # Raw TileType vocab size (TileType.value)
        self.raw_vocab_size: int = _tile_vocab_size()

        # Optional: map raw TileType IDs -> compact class IDs.
        self._tile_vocab = None
        if tile_vocab is not None:
            self._tile_vocab = load_tile_vocab(tile_vocab)

        base_num = int(self._tile_vocab.num_classes) if self._tile_vocab is not None else int(self.raw_vocab_size)
        self.num_classes: int = base_num + (1 if self.mask_oob else 0)

        # Expose for logging/debug
        self.tile_vocab_name: str | None = self._tile_vocab.name if self._tile_vocab is not None else None
        self.tile_vocab_sha256: str | None = self._tile_vocab.sha256 if self._tile_vocab is not None else None

        self.observation_space = spaces.Box(
            low=0,
            high=self.num_classes - 1,
            shape=(1, vy, vx),
            dtype=np.uint8,
        )

        # Optional: last grids for tooling (agent-view IDs later)
        self._last_raw_grid: np.ndarray | None = None
        self._last_class_grid: np.ndarray | None = None

    def _pov_tile_frame_with_valid(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (raw_frame, valid_mask), both (VY,VX).

        raw_frame contains TileType.value IDs. OOB is filled with EMPTY in raw_frame;
        valid_mask indicates which entries correspond to real board coordinates.
        """
        g = self.game.tile_grid  # (H,W) uint8

        head: Point = self.game.get_head_position()
        hx, hy = int(head.x), int(head.y)

        ry = int(self.view_radius_y)
        rx = int(self.view_radius_x)
        vy = 2 * ry + 1
        vx = 2 * rx + 1

        out = np.full((vy, vx), int(TileType.EMPTY.value), dtype=np.uint8)
        valid = np.zeros((vy, vx), dtype=np.bool_)

        if not self.rotate_to_head:
            # World-oriented rectangular crop (ry vertical, rx horizontal).
            x0, x1 = hx - rx, hx + rx + 1
            y0, y1 = hy - ry, hy + ry + 1

            gy0 = max(0, y0)
            gy1 = min(g.shape[0], y1)
            gx0 = max(0, x0)
            gx1 = min(g.shape[1], x1)

            oy0 = gy0 - y0
            oy1 = oy0 + (gy1 - gy0)
            ox0 = gx0 - x0
            ox1 = ox0 + (gx1 - gx0)

            out[oy0:oy1, ox0:ox1] = g[gy0:gy1, gx0:gx1]
            valid[oy0:oy1, ox0:ox1] = True
            return out, valid

        # Egocentric crop (forward is UP). Shape stays (vy, vx) even if ry != rx.
        d = self.game.direction
        assert d is not None, "SnakeGame.direction is None (did you call game.reset()?)"

        ys = np.arange(-ry, ry + 1, dtype=np.int32)  # ego dy (rows)
        xs = np.arange(-rx, rx + 1, dtype=np.int32)  # ego dx (cols)
        dy_ego, dx_ego = np.meshgrid(ys, xs, indexing="ij")  # (vy,vx)

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

        xw = hx + dx_w
        yw = hy + dy_w

        mask = (xw >= 0) & (xw < g.shape[1]) & (yw >= 0) & (yw < g.shape[0])
        out[mask] = g[yw[mask], xw[mask]]
        valid[mask] = True
        return out, valid

    def get_obs(self):
        raw, valid = self._pov_tile_frame_with_valid()

        # Map raw -> class IDs (or identity).
        if self._tile_vocab is None:
            frame = raw
        else:
            frame = self._tile_vocab.lut[raw]

        self._last_raw_grid = raw
        self._last_class_grid = frame

        if not self.mask_oob:
            return frame[None, :, :].astype(np.uint8, copy=False)

        # mask_oob=True: 0 is OOB, valid IDs shifted by +1.
        out = np.zeros_like(frame, dtype=np.uint8)
        out[valid] = (frame[valid].astype(np.uint16) + 1).astype(np.uint8, copy=False)
        return out[None, :, :].astype(np.uint8, copy=False)
