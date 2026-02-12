# src/snake_rl/envs/tile_id_env.py
from __future__ import annotations

import numpy as np
from gymnasium import spaces

from snake_rl.config.schema import RewardConfig
from snake_rl.envs.base import BaseSnakeEnv
from snake_rl.envs.obs_utils import global_tile_frame, pov_tile_frame_with_valid
from snake_rl.envs.view_radius import parse_view_radius
from snake_rl.game.snakegame import SnakeGame, tileset_tile_count
from snake_rl.vocab import load_tile_vocab


def _tile_vocab_size() -> int:
    # Raw tile ids come from Rust tileset (0..K-1).
    return int(tileset_tile_count())


class GlobalTileIdEnv(BaseSnakeEnv):
    """
    Global symbolic grid observation (no pixels).

    Observation:
      Box(shape=(1, H, W), dtype=uint8)
        value == class_id in [0, num_classes-1]

    Notes:
    - If tile_vocab is None (default), class_id == raw Rust tile ids.
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
        reward: RewardConfig | dict | None = None,
    ):
        BaseSnakeEnv.__init__(self, game, reward=reward)

        self.remove_border = bool(remove_border)

        # Keep historical behavior: env ctor forces a clean state.
        self.game.reset()

        h, w = int(self.game.height), int(self.game.width)
        if self.remove_border:
            if h <= 2 or w <= 2:
                raise ValueError(f"remove_border=True requires h,w > 2; got h={h} w={w}")
            h -= 2
            w -= 2

        # Raw tile vocab size (Rust tile ids)
        self.raw_vocab_size: int = _tile_vocab_size()

        # Optional: map raw tile IDs -> compact class IDs.
        self._tile_vocab = None
        if tile_vocab is not None:
            self._tile_vocab = load_tile_vocab(tile_vocab)

        if self._tile_vocab is not None:
            num_classes = int(self._tile_vocab.num_classes)
        else:
            num_classes = int(self.raw_vocab_size)
        self.num_classes: int = num_classes

        # Expose for logging/debug
        if self._tile_vocab is not None:
            self.tile_vocab_name = self._tile_vocab.name
            self.tile_vocab_sha256 = self._tile_vocab.sha256
        else:
            self.tile_vocab_name = None
            self.tile_vocab_sha256 = None

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
        return global_tile_frame(tile_grid=self.game.tile_grid, remove_border=self.remove_border)

    def get_obs(self):
        raw = self._get_grid_view()
        grid = raw if self._tile_vocab is None else self._tile_vocab.lut[raw]

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
    - If tile_vocab is None (default), class_id == raw Rust tile ids.
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
        reward: RewardConfig | dict | None = None,
    ):
        BaseSnakeEnv.__init__(self, game, reward=reward)

        self.view_radius_y, self.view_radius_x = parse_view_radius(view_radius)
        self.rotate_to_head = bool(rotate_to_head)
        self.mask_oob = bool(mask_oob)

        # Expose for debugging/logging.
        self.oob_id: int | None = 0 if self.mask_oob else None
        self._id_shift: int = 1 if self.mask_oob else 0

        # Keep historical behavior: env ctor forces a clean state.
        self.game.reset()

        vy = 2 * self.view_radius_y + 1
        vx = 2 * self.view_radius_x + 1

        # Raw tile vocab size (Rust tile ids)
        self.raw_vocab_size: int = _tile_vocab_size()

        # Optional: map raw tile IDs -> compact class IDs.
        self._tile_vocab = None
        if tile_vocab is not None:
            self._tile_vocab = load_tile_vocab(tile_vocab)

        if self._tile_vocab is not None:
            base_num = int(self._tile_vocab.num_classes)
        else:
            base_num = int(self.raw_vocab_size)
        self.num_classes: int = base_num + (1 if self.mask_oob else 0)

        # Expose for logging/debug
        if self._tile_vocab is not None:
            self.tile_vocab_name = self._tile_vocab.name
            self.tile_vocab_sha256 = self._tile_vocab.sha256
        else:
            self.tile_vocab_name = None
            self.tile_vocab_sha256 = None

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

        raw_frame contains raw Rust tile ids. OOB is filled with EMPTY in raw_frame;
        valid_mask indicates which entries correspond to real board coordinates.
        """
        d = self.game.direction
        assert d is not None, "SnakeGame.direction is None (did you call game.reset()?)"
        return pov_tile_frame_with_valid(
            tile_grid=self.game.tile_grid,
            head=self.game.get_head_position(),
            direction=d,
            view_radius=(self.view_radius_y, self.view_radius_x),
            rotate_to_head=bool(self.rotate_to_head),
        )

    def get_obs(self):
        raw, valid = self._pov_tile_frame_with_valid()

        # Map raw -> class IDs (or identity).
        frame = raw if self._tile_vocab is None else self._tile_vocab.lut[raw]

        self._last_raw_grid = raw
        self._last_class_grid = frame

        if not self.mask_oob:
            return frame[None, :, :].astype(np.uint8, copy=False)

        # mask_oob=True: 0 is OOB, valid IDs shifted by +1.
        out = np.zeros_like(frame, dtype=np.uint8)
        out[valid] = (frame[valid].astype(np.uint16) + 1).astype(np.uint8, copy=False)
        return out[None, :, :].astype(np.uint8, copy=False)
