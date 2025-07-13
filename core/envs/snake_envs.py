# snake_pixel_dir_obs_env.py
import numpy as np
from gymnasium import spaces
from .base_snake_env import BaseSnakeEnv
from core.snake6110.geometry import Direction


class SnakePixelDirectionObsEnv(BaseSnakeEnv):
    def __init__(self, game, render_mode=None):
        super().__init__(game, render_mode)

        self.game._update_pixel_buffer()
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
        self.game._update_pixel_buffer()
        pixel = self.game.pixel_buffer[None, :, :].astype(np.float32)
        direction = self.game.direction.value

        return {
            "pixel": pixel,
            "direction": np.array(direction, dtype=np.int64)
        }

