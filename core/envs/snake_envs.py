# snake_pixel_dir_obs_env.py
import numpy as np
from gymnasium import spaces
from .base_snake_env import BaseSnakeEnv
from collections import deque
from typing import Dict, Optional, Tuple
from gymnasium.core import ObsType
from core.snake6110.geometry import Direction

import cv2
from core.snake6110.snakegame import SnakeGame, Point


class SnakePixelDirectionObsEnv(BaseSnakeEnv):
    def __init__(self, game, render_mode=None):
        super().__init__(game, render_mode)

        self.game.reset()
        height, width = self.game.pixel_buffer.shape

        self.observation_space = spaces.Dict({
            "pixel": spaces.Box(
                low=0,
                high=255,
                shape=(1, height, width),
                dtype=np.uint8
            ),
            "direction": spaces.Discrete(4)
        })

    def get_obs(self):
        pixel = self.game.pixel_buffer[None, :, :].astype(np.float32)
        direction = self.game.direction.value

        return {
            "pixel": pixel,
            "direction": np.array(direction, dtype=np.int64)
        }


class SnakePixelObsEnv(BaseSnakeEnv):
    def __init__(self, game, render_mode=None):
        super().__init__(game, render_mode)

        self.game.reset()
        height, width = self.game.pixel_buffer.shape

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(1, height, width),
            dtype=np.uint8
        )

    def get_obs(self):
        return self.game.pixel_buffer[None, :, :].astype(np.uint8)


class DiscreteLengthSnakeEnv(BaseSnakeEnv):
    def __init__(self, game, render_mode: Optional[str] = None, length_bins: int = 8):
        super().__init__(game, render_mode)
        self.length_bins = length_bins
        self.initial_snake_length = game.level.init_snake_length
        self.max_snake_length = (game.level.width - 2) * (game.level.height - 2)

    def _discretize_snake_length(self, length: Optional[int] = None) -> int:
        length = length if length is not None else len(self.game.snake)
        norm = (length - self.initial_snake_length) / (self.max_snake_length - self.initial_snake_length)
        norm = float(np.clip(norm, 0.0, 1.0))
        return min(int(norm * self.length_bins), self.length_bins - 1)


class SnakePixelStackedEnv(DiscreteLengthSnakeEnv):
    def __init__(self, game: SnakeGame, render_mode: Optional[str] = None, n_stack: int = 2, length_bins: int = 4):
        super().__init__(game, render_mode, length_bins=length_bins)
        self.n_stack = n_stack

        self.game.reset()
        h, w = self.game.pixel_buffer.shape

        self.pixel_stack: deque[np.ndarray] = deque(maxlen=n_stack)
        for _ in range(n_stack):
            self.pixel_stack.append(np.zeros((h, w), dtype=np.float32))

        self.observation_space = spaces.Dict({
            "pixel": spaces.Box(low=0, high=255, shape=(n_stack, h, w), dtype=np.uint8),
            "snake_length": spaces.Discrete(length_bins)
        })

    def get_obs(self) -> Dict[str, np.ndarray]:
        current = self.game.pixel_buffer.astype(np.float32)
        self.pixel_stack.append(current)
        stacked_pixel = np.stack(list(self.pixel_stack), axis=0).astype(np.uint8)

        return {
            "pixel": stacked_pixel,
            "snake_length": self._discretize_snake_length()
        }

    def reset(self, **kwargs):
        super().reset(**kwargs)
        self.pixel_stack.clear()
        for _ in range(self.n_stack):
            self.pixel_stack.append(self.game.pixel_buffer.astype(np.uint8))
        return self.get_obs(), {}


class SnakeFirstPersonEnv(DiscreteLengthSnakeEnv):
    def __init__(
            self,
            game: SnakeGame,
            render_mode: Optional[DiscreteLengthSnakeEnv.RenderMode] = None,
            n_stack: int = 2,
            view_radius: Optional[int] = None,
            length_bins: int = 8,
    ):
        super().__init__(game, render_mode, length_bins=length_bins)
        self.n_stack = n_stack
        self.tilesize = self.game.tileset.tile_size

        if view_radius is None:
            view_radius = max(self.game.width - 2, self.game.height - 2)
        self.view_radius = view_radius
        self.view_tiles = 2 * view_radius + 1
        self.view_px = self.view_tiles * self.tilesize

        self.obs_px_shape = (self.n_stack, self.view_px, self.view_px)

        self.frames: deque[np.ndarray] = deque(maxlen=self.n_stack)
        dummy_frame = np.zeros(self.obs_px_shape[1:], dtype=np.uint8)
        for _ in range(self.n_stack):
            self.frames.append(dummy_frame)

        self.observation_space = spaces.Dict({
            "pixel": spaces.Box(low=0, high=255, shape=self.obs_px_shape, dtype=np.uint8),
            "snake_length": spaces.Discrete(length_bins)
        })

    def _get_centered_rotated_view(self) -> np.ndarray:
        head_x, head_y = self.game.snake[0].x, self.game.snake[0].y
        radius = self.view_radius
        tilesize = self.tilesize
        pixel_grid = self.game.pixel_buffer

        # Compute the full vision window in pixel coordinates
        col_start = (head_x - radius) * tilesize
        col_end = (head_x + radius + 1) * tilesize
        row_start = (head_y - radius) * tilesize
        row_end = (head_y + radius + 1) * tilesize

        view_size = (2 * radius + 1) * tilesize
        vision = np.zeros((view_size, view_size), dtype=pixel_grid.dtype)

        # Compute overlapping bounds (clipping to pixel_grid)
        grid_row_start = max(0, row_start)
        grid_row_end = min(pixel_grid.shape[0], row_end)
        grid_col_start = max(0, col_start)
        grid_col_end = min(pixel_grid.shape[1], col_end)

        # Compute where to place in the vision buffer
        vision_row_start = grid_row_start - row_start
        vision_row_end = vision_row_start + (grid_row_end - grid_row_start)
        vision_col_start = grid_col_start - col_start
        vision_col_end = vision_col_start + (grid_col_end - grid_col_start)

        # Copy pixels into centered vision
        vision[vision_row_start:vision_row_end, vision_col_start:vision_col_end] = \
            pixel_grid[grid_row_start:grid_row_end, grid_col_start:grid_col_end]

        # Rotate if needed (head should face UP)
        direction = self.game.direction
        if direction == Direction.RIGHT:
            vision = np.rot90(vision, k=1)
        elif direction == Direction.DOWN:
            vision = np.rot90(vision, k=2)
        elif direction == Direction.LEFT:
            vision = np.rot90(vision, k=3)

        assert vision.shape == (view_size, view_size), f"Wrong view shape: {vision.shape}"
        return vision

    def reset(self, **kwargs) -> Tuple[Dict[str, np.ndarray], Dict]:
        _, info = super().reset(**kwargs)
        frame = self._get_centered_rotated_view()
        self.frames.clear()
        for _ in range(self.n_stack):
            self.frames.append(frame)
        return self.get_obs(), info

    def get_obs(self) -> Dict[str, np.ndarray]:
        frame = self._get_centered_rotated_view()
        self.frames.append(frame)

        stacked = np.stack(self.frames, axis=0).astype(np.uint8)

        return {
            "pixel": stacked,
            "snake_length": self._discretize_snake_length()
        }
