# src/snake_rl/envs/snake_env.py
from __future__ import annotations

from typing import Any

from snake_rl.config.schema import RewardConfig
from snake_rl.envs.base import BaseSnakeEnv
from snake_rl.envs.specs import ActionSpec, ObservationSpec
from snake_rl.game.snake_engine import SnakeEngine


class SnakeEnv(BaseSnakeEnv):
    """
    Unified Snake Gym environment with composable action + observation specs.
    """

    def __init__(
        self,
        game: SnakeEngine,
        *,
        obs: ObservationSpec,
        action: ActionSpec | None = None,
        reward: RewardConfig | dict[str, Any] | None = None,
    ):
        super().__init__(game, action=action, reward=reward)

        self.obs_spec = obs
        self._tile_vocab = self.obs_spec.load_tile_vocab()

        self.observation_space = self.obs_spec.make_space(
            width=int(self.game.width),
            height=int(self.game.height),
            tile_size=int(self.game.tile_size),
            tile_vocab=self._tile_vocab,
        )

    def get_obs(self):
        return self.obs_spec.observe(
            game=self.game,
            tile_vocab=self._tile_vocab,
            initial_snake_length=self.initial_snake_length,
            max_playable_tiles=self.max_snake_length,
        )


__all__ = ["SnakeEnv"]
