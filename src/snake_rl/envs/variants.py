# src/snake_rl/envs/variants.py
from __future__ import annotations

from typing import Optional

import numpy as np
from gymnasium import spaces

from snake_rl.envs.base import BaseSnakeEnv
from snake_rl.envs.obs_base import FillFeature, PixelObsEnvBase, PixelStacker
from snake_rl.game.snakegame import SnakeGame


class GlobalPixelEnv(BaseSnakeEnv, PixelObsEnvBase):
    """
    Baseline: global pixel only.

    Observation: Box (1,H,W) uint8
    Stacking: prefer SB3 VecFrameStack (keeps env simple).
    """

    def __init__(self, game: SnakeGame):
        BaseSnakeEnv.__init__(self, game)
        PixelObsEnvBase.__init__(self, game)

        self.game.reset()
        h, w = self.game.pixel_buffer.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(1, h, w), dtype=np.uint8)

    def get_obs(self):
        frame = self._global_pixel_frame()
        return frame[None, :, :].astype(np.uint8, copy=False)


class GlobalPixelDirectionEnv(BaseSnakeEnv, PixelObsEnvBase):
    """
    Baseline: global pixel + absolute direction.

    Observation: Dict(
      pixel=Box((n_stack,H,W), uint8),
      direction=Discrete(4)
    )

    Note: This env stacks only the pixel frames internally when n_stack>1.
    """

    def __init__(self, game: SnakeGame, *, n_stack: int = 1):
        BaseSnakeEnv.__init__(self, game)
        PixelObsEnvBase.__init__(self, game)

        self.n_stack = int(n_stack)

        self.game.reset()
        h, w = self.game.pixel_buffer.shape

        if self.n_stack <= 1:
            pixel_shape = (1, h, w)
            self._stacker: Optional[PixelStacker] = None
        else:
            pixel_shape = (self.n_stack, h, w)
            self._stacker = PixelStacker(n_stack=self.n_stack, frame_shape=(h, w))

        self.observation_space = spaces.Dict(
            {
                "pixel": spaces.Box(low=0, high=255, shape=pixel_shape, dtype=np.uint8),
                "direction": spaces.Discrete(4),
            }
        )

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        # Ensure the stack starts from the reset frame (no temporal leakage).
        if self._stacker is not None:
            frame = self._global_pixel_frame()
            self._stacker.reset_with(frame)
        return self.get_obs(), info

    def get_obs(self):
        frame = self._global_pixel_frame()
        if self._stacker is None:
            pixel = frame[None, :, :].astype(np.uint8, copy=False)
        else:
            pixel = self._stacker.push_and_stack(frame)

        # Absolute direction (UP/RIGHT/DOWN/LEFT) encoded as 0..3.
        d = self.game.direction
        assert d is not None, "SnakeGame.direction is None (did you call game.reset()?)"
        direction = int(d.value)

        return {
            "pixel": pixel,
            "direction": np.array(direction, dtype=np.int64),
        }


class PovPixelEnv(BaseSnakeEnv, PixelObsEnvBase):
    """
    Baseline: POV pixel only (centered on head, rotated so forward is UP).

    Observation: Box (1,view_px,view_px) uint8
    Stacking: prefer SB3 VecFrameStack.
    """

    def __init__(self, game: SnakeGame, *, view_radius: int):
        BaseSnakeEnv.__init__(self, game)
        PixelObsEnvBase.__init__(self, game)

        self.view_radius = int(view_radius)

        self.game.reset()
        frame = self._pov_pixel_frame(view_radius=self.view_radius)
        h, w = frame.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(1, h, w), dtype=np.uint8)

    def get_obs(self):
        frame = self._pov_pixel_frame(view_radius=self.view_radius)
        return frame[None, :, :].astype(np.uint8, copy=False)


class PovPixelFillEnv(BaseSnakeEnv, PixelObsEnvBase):
    """
    Baseline: POV pixel + global fill indicator.

    Observation: Dict(
      pixel=Box((n_stack,view_px,view_px), uint8),
      fill=Box((1,), float32) OR Discrete(fill_bins)
    )

    Note: This env stacks only the pixel frames internally when n_stack>1.
    """

    def __init__(
            self,
            game: SnakeGame,
            *,
            view_radius: int,
            n_stack: int = 1,
            fill_bins: Optional[int] = None,
    ):
        BaseSnakeEnv.__init__(self, game)
        PixelObsEnvBase.__init__(self, game)

        self.view_radius = int(view_radius)
        self.n_stack = int(n_stack)
        self._fill = FillFeature(fill_bins=fill_bins)

        self.game.reset()
        frame = self._pov_pixel_frame(view_radius=self.view_radius)
        h, w = frame.shape

        if self.n_stack <= 1:
            pixel_shape = (1, h, w)
            self._stacker: Optional[PixelStacker] = None
        else:
            pixel_shape = (self.n_stack, h, w)
            self._stacker = PixelStacker(n_stack=self.n_stack, frame_shape=(h, w))

        fill_space = (
            spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
            if fill_bins is None
            else spaces.Discrete(int(fill_bins))
        )

        self.observation_space = spaces.Dict(
            {
                "pixel": spaces.Box(low=0, high=255, shape=pixel_shape, dtype=np.uint8),
                "fill": fill_space,
            }
        )

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        if self._stacker is not None:
            frame = self._pov_pixel_frame(view_radius=self.view_radius)
            self._stacker.reset_with(frame)
        return self.get_obs(), info

    def get_obs(self):
        frame = self._pov_pixel_frame(view_radius=self.view_radius)
        if self._stacker is None:
            pixel = frame[None, :, :].astype(np.uint8, copy=False)
        else:
            pixel = self._stacker.push_and_stack(frame)

        fill = self._fill.compute(
            snake_len=len(self.game.snake),
            initial_len=self.initial_snake_length,
            max_playable=self.max_snake_length,
        )

        return {
            "pixel": pixel,
            "fill": fill,
        }
