# src/snake_rl/envs/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import gymnasium as gym
import snake_rl._core as core
from gymnasium import spaces

from snake_rl.config.schema import RewardConfig
from snake_rl.game.snakegame import SnakeGame


class BaseSnakeEnv(gym.Env, ABC):
    """
    Shared RL environment mechanics for Snake.

    IMPORTANT:
    This base class is used together with mixins (e.g. PixelObsEnvBase).
    Therefore, DO NOT call `super().__init__()` here, because that would
    trigger mixin __init__ methods via MRO and break when they require args.
    """

    FATAL_MASK = (
        core.MOVE_HIT_SELF
        | core.MOVE_HIT_WALL
        | core.MOVE_HIT_BOUNDARY
        | core.MOVE_NOT_RUNNING
        | core.MOVE_WIN
    )

    TERMINATION_PRIORITY = [
        core.MOVE_WIN,
        core.MOVE_HIT_WALL,
        core.MOVE_HIT_SELF,
        core.MOVE_HIT_BOUNDARY,
        core.MOVE_TIMEOUT,
        core.MOVE_NOT_RUNNING,
    ]

    TERMINATION_CAUSES = {
        core.MOVE_WIN: "win",
        core.MOVE_HIT_WALL: "hit_wall",
        core.MOVE_HIT_SELF: "hit_self",
        core.MOVE_HIT_BOUNDARY: "hit_boundary",
        core.MOVE_NOT_RUNNING: "not_running",
        core.MOVE_TIMEOUT: "timeout",
    }

    def __init__(self, game: SnakeGame, *, reward: RewardConfig | dict[str, Any] | None = None):
        # Avoid cooperative super() because subclasses also mix in PixelObsEnvBase.
        gym.Env.__init__(self)

        self.game: SnakeGame = game

        # 0 = forward, 1 = left, 2 = right
        self.action_space: spaces.Space = spaces.Discrete(3)

        if reward is None:
            reward_cfg = RewardConfig()
        elif isinstance(reward, RewardConfig):
            reward_cfg = reward
        elif isinstance(reward, dict):
            reward_cfg = RewardConfig(**reward)
        else:
            raise TypeError(f"reward must be RewardConfig|dict|None, got {type(reward).__name__}")

        self.reward: RewardConfig = reward_cfg

        # Limits and rewards based on level dimensions (RL logic: keep unchanged)
        max_steps = int(self.game.max_playable_tiles * float(self.reward.max_steps_factor))
        self.max_steps: int = max(1, max_steps)
        self.max_snake_length: int = self.game.max_playable_tiles

        # Snake length is runtime state (spawn), so capture it on reset().
        self.initial_snake_length: int = 0

        self.tiny_reward: float = float(self.reward.step_penalty_scale) / float(self.max_steps)

        self.current_step_since_last_food: int = 0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed, options=options)

        # Bind the game's RNG to the env RNG. This makes seeding robust and
        # consistent with Gymnasium/SB3 behavior (per-env streams).
        self.game.reset(seed=seed)
        self.initial_snake_length = int(self.game.snake_len)

        self.current_step_since_last_food = 0

        obs = self.get_obs()
        return obs, {}

    def _get_min_food_distance(self) -> int | None:
        head = self.game.get_head_position()
        food = self.game.get_food_positions()
        if not food:
            return None
        hx, hy = int(head[0]), int(head[1])
        return min(abs(hx - int(f[0])) + abs(hy - int(f[1])) for f in food)

    def step(self, action: int):
        reward = 0.0
        if action not in (0, 1, 2):
            raise ValueError(f"action must be 0|1|2, got {action!r}")
        results = self.game.move(int(action))
        obs = self.get_obs()

        self.current_step_since_last_food += 1

        is_win = bool(results & core.MOVE_WIN)
        is_fatal = bool(results & self.FATAL_MASK)
        is_food = bool(results & core.MOVE_FOOD)
        is_truncated = self.current_step_since_last_food >= self.max_steps

        if is_win:
            reward += float(self.reward.win_reward)
            terminated = True
            truncated = False

            if is_food:
                self.current_step_since_last_food = 0
        else:
            reward -= self.tiny_reward

            if is_food:
                reward += float(self.reward.food_reward)
                reward += float(self.reward.food_speed_bonus) * (
                    1.0 - (self.current_step_since_last_food / self.max_steps)
                )
                self.current_step_since_last_food = 0

            if is_fatal:
                reward -= float(self.reward.fatal_penalty)

            terminated = is_fatal
            truncated = not terminated and is_truncated
            if truncated:
                reward -= float(self.reward.timeout_penalty)

        info: dict[str, Any] = {"move_results": int(results)}

        if terminated or truncated:
            result_for_cause: int | None = next(
                (r for r in self.TERMINATION_PRIORITY if (results & r)),
                None,
            )
            if truncated and result_for_cause is None:
                result_for_cause = core.MOVE_TIMEOUT

            if result_for_cause is None:
                cause = "unknown"
            else:
                cause = self.TERMINATION_CAUSES.get(result_for_cause, "unknown")

            info.update({"final_score": self.game.score, "termination_cause": cause})

        return obs, reward, terminated, truncated, info

    @abstractmethod
    def get_obs(self):
        raise NotImplementedError
