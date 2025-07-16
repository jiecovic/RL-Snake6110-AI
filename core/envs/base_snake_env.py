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

    def __init__(self, game: SnakeGame, render_mode: RenderMode = "none"):
        self.render_mode = render_mode
        super().__init__()
        self.game = game
        self.action_space = spaces.Discrete(3)

        self.max_steps = (self.game.width - 2) * (self.game.height - 2) + 2
        self.current_step_since_last_food = 0
        self.visited_nodes = set()

        self.tiny_reward = 1.0 / self.max_steps
        self.initial_snake_length = self.game.level.init_snake_length
        self.max_snake_length = (self.game.level.width - 2) * (self.game.level.height - 2)

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
        # === 1. Compute pre-move distance to nearest food ===

        # dist_before = self._get_min_food_distance()
        # head_before = self.game.get_head_position()
        # dijkstra_path = self.game.find_path_to_closest_food_dijkstra()

        reward = 0.0

        # === 2. Perform action and update environment ===
        direction = RelativeDirection(action)
        results = self.game.move(direction)  # now a list of MoveResult

        # head_after = self.game.get_head_position()
        # if len(self.game.snake) <= (self.game.level.init_snake_length + 5) and dijkstra_path and len(dijkstra_path) > 1:
        # if dijkstra_path and len(dijkstra_path) > 1:
        #     if head_after in dijkstra_path:
        #         reward += 0 #self.tiny_reward  # tiny reward for being on path
        #     else:
        #         reward -= self.tiny_reward  # penalty for being off-path

        # n_visited_nodes_before = len(self.visited_nodes)
        # self.visited_nodes.add(self.game.get_head_position())
        # n_visited_nodes_after = len(self.visited_nodes)
        # if n_visited_nodes_after > n_visited_nodes_before:
        #     reward += 0.01
        # if len(self.game.snake) <= (self.game.level.init_snake_length + 5):
        reward -= self.tiny_reward

        obs = self.get_obs()
        self.current_step_since_last_food += 1

        # === 3. Compute post-move distance and shaping reward ===
        # dist_after = self._get_min_food_distance()
        # shaping_reward = self._compute_shaping_reward(dist_before, dist_after)
        # reward += shaping_reward

        # === 4. Determine episode end conditions ===
        fatal_results = {
            MoveResult.HIT_SELF,
            MoveResult.HIT_WALL,
            MoveResult.HIT_BOUNDARY,
            MoveResult.GAME_NOT_RUNNING,
            # CYCLE_DETECTED handled separately
        }

        is_cycle = MoveResult.CYCLE_DETECTED in results
        is_fatal = any(r in fatal_results for r in results)

        terminated = is_fatal  # is_cycle or is_fatal
        truncated = self.current_step_since_last_food >= self.max_steps

        # === 5. Compute total reward ===
        if MoveResult.FOOD_EATEN in results:
            reward += 1.0
            reward += (1.0 - (self.current_step_since_last_food / self.max_steps))  # bonus for shorter paths
            self.visited_nodes.clear()
            self.current_step_since_last_food = 0

        if is_fatal:
            reward -= 5.0
        elif is_cycle:
            reward -= 0.5  # Custom penalty for cycle detection

        # === 6. Construct info dict ===
        info = {
            "move_results": results,
        }

        if terminated or truncated:
            # Determine most severe cause
            priority = [
                MoveResult.HIT_WALL,
                MoveResult.HIT_SELF,
                MoveResult.HIT_BOUNDARY,
                MoveResult.CYCLE_DETECTED,
                MoveResult.GAME_NOT_RUNNING,
                MoveResult.TIMEOUT
            ]
            result_for_cause = next((r for r in priority if r in results), MoveResult.TIMEOUT if truncated else None)
            cause = {
                MoveResult.HIT_WALL: "hit_wall",
                MoveResult.HIT_SELF: "hit_self",
                MoveResult.HIT_BOUNDARY: "hit_boundary",
                MoveResult.CYCLE_DETECTED: "cycle",
                MoveResult.GAME_NOT_RUNNING: "not_running",
                MoveResult.TIMEOUT: "timeout"
            }.get(result_for_cause, "unknown")

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
