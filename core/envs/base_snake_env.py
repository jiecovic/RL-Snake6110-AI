# base_snake_env.py
import gymnasium as gym
from gymnasium import spaces
from abc import ABC, abstractmethod
from typing import Literal
from core.snake6110.snakegame import SnakeGame, MoveResult
from core.snake6110.geometry import RelativeDirection


class BaseSnakeEnv(gym.Env, ABC):
    metadata = {
        "render_modes": ["human", "none"],
        "render_fps": 5
    }

    RenderMode = Literal["human", "none"]

    def __init__(self, game: SnakeGame, render_mode: RenderMode = "none"):
        self.render_mode = render_mode
        super().__init__()
        self.game = game
        self.action_space = spaces.Discrete(3)

        self.max_steps = self.game.width * self.game.height + 10
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset()
        self.current_step = 0
        obs = self.get_obs()
        return obs, {}


    def _get_min_food_distance(self) -> int:
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
            return 0.01  # reward for moving closer
        elif after > before:
            return -0.01  # optional penalty
        return 0.0


    def step(self, action: int):
        # === 1. Compute pre-move distance to nearest food ===
        dist_before = self._get_min_food_distance()

        # === 2. Perform action and update environment ===
        direction = RelativeDirection(action)
        result = self.game.move(direction)
        obs = self.get_obs()
        self.current_step += 1

        # === 3. Compute post-move distance and shaping reward ===
        dist_after = self._get_min_food_distance()
        shaping_reward = self._compute_shaping_reward(dist_before, dist_after)

        # === 4. Determine episode end conditions ===
        terminated = result in {
            MoveResult.HIT_SELF,
            MoveResult.HIT_WALL,
            MoveResult.HIT_BOUNDARY,
            MoveResult.GAME_NOT_RUNNING
        }
        truncated = self.current_step >= self.max_steps

        # If episode was truncated (timeout), treat it as a timeout result
        if truncated and not terminated:
            result = MoveResult.TIMEOUT

        # === 5. Compute total reward ===
        reward = 1.0 if result == MoveResult.FOOD_EATEN else 0.0
        reward += shaping_reward

        # Apply penalty for fatal mistakes
        if result in {
            MoveResult.HIT_SELF,
            MoveResult.HIT_WALL,
            MoveResult.HIT_BOUNDARY
        }:
            reward -= 0.5

        # === 6. Construct info dict if episode ends ===
        info = {}
        if terminated or truncated:
            cause = {
                MoveResult.HIT_WALL: "hit_wall",
                MoveResult.HIT_SELF: "hit_self",
                MoveResult.HIT_BOUNDARY: "hit_boundary",
                MoveResult.GAME_NOT_RUNNING: "not_running",
                MoveResult.TIMEOUT: "timeout"
            }.get(result, "unknown")

            info.update({
                "final_score": self.game.score,
                "termination_cause": cause
            })

        # === 7. Optional rendering ===
        self.render()

        # === 8. Return step result ===
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

