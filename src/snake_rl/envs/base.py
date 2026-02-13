# src/snake_rl/envs/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import gymnasium as gym
from gymnasium import spaces

from snake_rl import _core as core
from snake_rl.config.schema import RewardConfig
from snake_rl.envs.specs import ActionSpec
from snake_rl.game.snake_engine import SnakeEngine


class BaseSnakeEnv(gym.Env, ABC):
    """
    Shared RL environment mechanics for Snake.

    IMPORTANT:
    This base class is used together with mixins (e.g. PixelObsEnvBase).
    Therefore, DO NOT call `super().__init__()` here, because that would
    trigger mixin __init__ methods via MRO and break when they require args.
    """

    metadata = {"render_modes": []}
    render_mode = None

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

    def __init__(
        self,
        game: SnakeEngine,
        *,
        action: ActionSpec | None = None,
        reward: RewardConfig | dict[str, Any] | None = None,
    ):
        # Avoid cooperative super() because subclasses also mix in PixelObsEnvBase.
        gym.Env.__init__(self)

        self.game: SnakeEngine = game

        self.action_spec = action if action is not None else ActionSpec()
        self.action_space: spaces.Space = self.action_spec.action_space()
        self.render_mode = None

        if reward is None:
            reward_cfg = RewardConfig()
        elif isinstance(reward, RewardConfig):
            reward_cfg = reward
        elif isinstance(reward, dict):
            reward_cfg = RewardConfig(**reward)
        else:
            raise TypeError(f"reward must be RewardConfig|dict|None, got {type(reward).__name__}")

        self.reward: RewardConfig = reward_cfg

        # Limits and rewards based on board dimensions (RL logic: keep unchanged)
        max_steps = int(self.game.max_playable_tiles * float(self.reward.max_steps_factor))
        self.max_steps: int = max(1, max_steps)
        self.max_snake_length: int = self.game.max_playable_tiles

        # Snake length is runtime state (spawn), so capture it on reset().
        self.initial_snake_length: int = 0

        self.tiny_reward: float = float(self.reward.step_penalty_scale) / float(self.max_steps)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed, options=options)

        # Bind the game's RNG to the env RNG. This makes seeding robust and
        # consistent with Gymnasium/SB3 behavior (per-env streams).
        # Note: Rust treats seed=None as "do not reseed", so RNG stream continues.
        self.game.reset(seed=seed)
        self.initial_snake_length = int(self.game.snake_len)

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
        action_i = int(action)
        self.action_spec.validate_action(action_i)
        results = self.action_spec.step(self.game, action_i)
        obs = self.get_obs()

        steps_since_food = int(self.game.steps_since_food)

        is_win = bool(results & core.MOVE_WIN)
        is_fatal = bool(results & self.FATAL_MASK)
        is_food = bool(results & core.MOVE_FOOD)
        is_timeout = bool(results & core.MOVE_TIMEOUT)
        if is_win:
            reward += float(self.reward.win_reward)
            terminated = True
            truncated = False
        else:
            reward -= self.tiny_reward

            if is_food:
                reward += float(self.reward.food_reward)
                reward += float(self.reward.food_speed_bonus) * (
                    1.0 - (steps_since_food / self.max_steps)
                )

            if is_fatal:
                reward -= float(self.reward.fatal_penalty)

            terminated = is_fatal
            truncated = not terminated and is_timeout
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
