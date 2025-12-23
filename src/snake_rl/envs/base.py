# src/snake_rl/envs/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import gymnasium as gym
from gymnasium import spaces

from snake_rl.game.geometry import RelativeDirection
from snake_rl.game.snakegame import MoveResult, SnakeGame


class BaseSnakeEnv(gym.Env, ABC):
    """
    Shared RL environment mechanics for Snake.

    IMPORTANT:
    This base class is used together with mixins (e.g. PixelObsEnvBase).
    Therefore, DO NOT call `super().__init__()` here, because that would
    trigger mixin __init__ methods via MRO and break when they require args.
    """

    FATAL_RESULTS = {
        MoveResult.HIT_SELF,
        MoveResult.HIT_WALL,
        MoveResult.HIT_BOUNDARY,
        MoveResult.GAME_NOT_RUNNING,
        MoveResult.CYCLE_DETECTED,
        MoveResult.WIN,
    }

    TERMINATION_PRIORITY = [
        MoveResult.WIN,
        MoveResult.HIT_WALL,
        MoveResult.HIT_SELF,
        MoveResult.HIT_BOUNDARY,
        MoveResult.CYCLE_DETECTED,
        MoveResult.TIMEOUT,
        MoveResult.GAME_NOT_RUNNING,
    ]

    TERMINATION_CAUSES = {
        MoveResult.WIN: "win",
        MoveResult.HIT_WALL: "hit_wall",
        MoveResult.HIT_SELF: "hit_self",
        MoveResult.HIT_BOUNDARY: "hit_boundary",
        MoveResult.CYCLE_DETECTED: "cycle",
        MoveResult.GAME_NOT_RUNNING: "not_running",
        MoveResult.TIMEOUT: "timeout",
    }

    def __init__(self, game: SnakeGame):
        # Avoid cooperative super() because subclasses also mix in PixelObsEnvBase.
        gym.Env.__init__(self)

        self.game: SnakeGame = game

        # 0 = forward, 1 = left, 2 = right
        self.action_space: spaces.Discrete = spaces.Discrete(3)

        # Limits and rewards based on level dimensions (RL logic: keep unchanged)
        self.max_steps: int = int(self.game.max_playable_tiles * 1.3)
        self.max_snake_length: int = self.game.max_playable_tiles

        # Snake length is runtime state (spawn), so capture it on reset().
        self.initial_snake_length: int = 0

        self.tiny_reward: float = 1.0 / self.max_steps

        self.current_step_since_last_food: int = 0
        self.visited_nodes: set = set()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.game.reset()
        self.initial_snake_length = len(self.game.snake)

        self.current_step_since_last_food = 0
        self.visited_nodes.clear()

        obs = self.get_obs()
        return obs, {}

    def _get_min_food_distance(self) -> Optional[int]:
        head = self.game.get_head_position()
        food = self.game.get_food_positions()
        if not food:
            return None
        return min(abs(head.x - f.x) + abs(head.y - f.y) for f in food)

    def _compute_shaping_reward(self, before: int, after: int) -> float:
        if before is None or after is None:
            return 0.0
        if after < before:
            return 0.1 * self.tiny_reward
        elif after > before:
            return -0.1 * self.tiny_reward
        return 0.0

    def step(self, action: int):
        reward = 0.0
        direction = RelativeDirection(action)
        results = self.game.move(direction)
        obs = self.get_obs()

        self.current_step_since_last_food += 1

        is_win = MoveResult.WIN in results
        is_fatal = any(r in self.FATAL_RESULTS for r in results)
        is_food = MoveResult.FOOD_EATEN in results
        is_truncated = self.current_step_since_last_food >= self.max_steps

        if is_win:
            reward += 10.0
            terminated = True
            truncated = False

            if is_food:
                self.visited_nodes.clear()
                self.current_step_since_last_food = 0
        else:
            reward -= self.tiny_reward

            if is_food:
                reward += 1.5
                reward += 2 * (1.0 - (self.current_step_since_last_food / self.max_steps))
                self.visited_nodes.clear()
                self.current_step_since_last_food = 0

            if is_fatal:
                reward -= 5.0

            terminated = is_fatal
            truncated = not terminated and is_truncated
            # if truncated:
            #     reward -= 0.2  # small truncation loss

        info: dict[str, Any] = {"move_results": results}

        if terminated or truncated:
            result_for_cause: Optional[MoveResult] = next(
                (r for r in self.TERMINATION_PRIORITY if r in results),
                None,
            )
            if truncated and result_for_cause is None:
                result_for_cause = MoveResult.TIMEOUT

            if result_for_cause is None:
                cause = "unknown"
            else:
                cause = self.TERMINATION_CAUSES.get(result_for_cause, "unknown")

            info.update({"final_score": self.game.score, "termination_cause": cause})

        return obs, reward, terminated, truncated, info

    @abstractmethod
    def get_obs(self):
        raise NotImplementedError
