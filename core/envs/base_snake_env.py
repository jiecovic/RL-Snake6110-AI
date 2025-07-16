# base_snake_env.py
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from abc import ABC, abstractmethod
from typing import Literal, Optional
from core.snake6110.snakegame import SnakeGame, MoveResult
from core.snake6110.geometry import RelativeDirection


class BaseSnakeEnv(gym.Env, ABC):
    metadata = {
        "render_modes": ["human", "none"],
        "render_fps": 5
    }

    RenderMode = Literal["human", "none"]

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
        MoveResult.TIMEOUT: "timeout"
    }

    def __init__(self, game: SnakeGame, render_mode: RenderMode = "none"):
        """
        Initializes the base Snake RL environment.

        Args:
            game (SnakeGame): The game instance.
            render_mode (RenderMode): "human" or "none".
        """
        self.render_mode: BaseSnakeEnv.RenderMode = render_mode
        super().__init__()

        self.game: SnakeGame = game
        self.action_space: spaces.Discrete = spaces.Discrete(3)  # 0 = forward, 1 = left, 2 = right

        # Limits and rewards based on level dimensions
        self.max_steps: int = int(self.game.max_playable_tiles * 1.5)
        self.max_snake_length: int = self.game.max_playable_tiles
        self.initial_snake_length: int = self.game.level.init_snake_length
        self.tiny_reward: float = 1.0 / self.max_steps

        self.current_step_since_last_food: int = 0
        self.visited_nodes: set = set()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset()
        self.current_step_since_last_food = 0
        self.visited_nodes.clear()
        obs = self.get_obs()
        return obs, {}

    def _get_min_food_distance(self) -> Optional[int]:
        """Returns Manhattan distance from the snake head to the nearest food."""
        head = self.game.get_head_position()
        food = self.game.get_food_positions()
        if not food:
            return None
        return min(abs(head.x - f.x) + abs(head.y - f.y) for f in food)

    def _compute_shaping_reward(self, before: int, after: int) -> float:
        """Returns a small reward or penalty based on change in food distance."""
        if before is None or after is None:
            return 0.0
        if after < before:
            return 0.1 * self.tiny_reward  # reward for moving closer
        elif after > before:
            return -0.1 * self.tiny_reward  # optional penalty
        return 0.0

    def step(self, action: int):
        reward = 0.0
        direction = RelativeDirection(action)
        results = self.game.move(direction)
        obs = self.get_obs()

        self.current_step_since_last_food += 1

        # Flags for various result types
        is_win = MoveResult.WIN in results
        is_fatal = any(r in self.FATAL_RESULTS for r in results)
        is_food = MoveResult.FOOD_EATEN in results
        is_truncated = self.current_step_since_last_food >= self.max_steps

        if is_win:
            # Win condition: no penalties, high reward
            reward += 10.0
            terminated = True
            truncated = False

            # Optional: clear food-related state even on win
            if is_food:
                self.visited_nodes.clear()
                self.current_step_since_last_food = 0
        else:
            # Small step penalty to encourage efficiency
            reward -= self.tiny_reward

            if is_food:
                # Food eaten:
                # +1.0 base reward
                # +bonus scaled by how quickly it was found (up to +2.0)
                reward += 1.0
                reward += 2 * (1.0 - (self.current_step_since_last_food / self.max_steps))
                self.visited_nodes.clear()
                self.current_step_since_last_food = 0

            if is_fatal:
                # Major penalty for hitting wall, self, etc.
                reward -= 5.0

            # Terminate if fatal; truncate if max steps reached
            terminated = is_fatal
            truncated = not terminated and is_truncated

        info = {
            "move_results": results,
        }

        # Include termination cause and final score if episode ends
        if terminated or truncated:
            result_for_cause = next(
                (r for r in self.TERMINATION_PRIORITY if r in results),
                MoveResult.TIMEOUT if truncated else None
            )
            cause = self.TERMINATION_CAUSES.get(result_for_cause, "unknown")
            info.update({
                "final_score": self.game.score,
                "termination_cause": cause
            })

        self.render()
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            self.game.render()

    def close(self):
        import pygame
        pygame.quit()

    @abstractmethod
    def get_obs(self):
        """Return the current observation (must be implemented by subclasses)"""
        pass
