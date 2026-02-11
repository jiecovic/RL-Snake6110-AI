# src/snake_rl/game/snakegame.py
from __future__ import annotations

from enum import IntFlag

from snake_rl.core.game import RustGame
from snake_rl.game.geometry import Direction, Point, RelativeDirection
from snake_rl.game.level import BaseLevel
from snake_rl.game.tileset import Tileset


class MoveResult(IntFlag):
    OK = 1 << 0
    FOOD_EATEN = 1 << 1
    HIT_BOUNDARY = 1 << 2
    HIT_WALL = 1 << 3
    HIT_SELF = 1 << 4
    GAME_NOT_RUNNING = 1 << 5
    TIMEOUT = 1 << 6
    WIN = 1 << 7


class SnakeGame:
    """
    Rust-backed core game state + rules (headless, RL-safe).

    RNG contract
    - Rust owns all randomness.
    - env.reset(seed) passes a seed to Rust; resets without seed continue RNG stream.
    """

    def __init__(
        self,
        level: BaseLevel,
        food_count: int | None = None,
        tileset: Tileset | None = None,
        seed: int | None = None,
    ):
        if food_count is None:
            raise ValueError("food_count must be provided (level does not encode food).")
        self._impl = RustGame(
            level,
            food_count=int(food_count),
            tileset=tileset,
            seed=seed,
        )

    def reset(self, seed: int | None = None) -> None:
        self._impl.reset(seed=seed)

    def move(self, rel_dir: RelativeDirection = RelativeDirection.FORWARD) -> MoveResult:
        mask = self._impl.move(rel_dir)
        return MoveResult(int(mask))

    # ---- properties used by envs/renderers ----

    @property
    def width(self) -> int:
        return self._impl.width

    @property
    def height(self) -> int:
        return self._impl.height

    @property
    def tileset(self) -> Tileset:
        return self._impl.tileset

    @property
    def tile_grid(self):
        return self._impl.tile_grid

    @property
    def pixel_buffer(self):
        return self._impl.pixel_buffer

    @property
    def direction(self) -> Direction | None:
        return self._impl.direction

    @property
    def score(self) -> int:
        return self._impl.score

    @property
    def running(self) -> bool:
        return self._impl.running

    @property
    def snake_len(self) -> int:
        return self._impl.snake_len

    def get_head_position(self) -> Point:
        return self._impl.get_head_position()

    def get_food_positions(self) -> list[Point]:
        return self._impl.get_food_positions()

    @property
    def max_playable_tiles(self) -> int:
        return self._impl.max_playable_tiles

    @property
    def spawnable_count(self) -> int:
        return self._impl.spawnable_count


__all__ = ["MoveResult", "SnakeGame"]
