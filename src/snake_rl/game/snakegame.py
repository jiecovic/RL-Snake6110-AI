# src/snake_rl/game/snakegame.py
from __future__ import annotations

import random
from collections import deque
from enum import Enum, auto

import numpy as np

from snake_rl.game.geometry import DIRECTION_TO_POINT, Direction, Point, RelativeDirection
from snake_rl.game.level import BaseLevel
from snake_rl.game.tile_types import TileType
from snake_rl.game.tileset import Tileset


class MoveResult(Enum):
    OK = auto()
    FOOD_EATEN = auto()
    HIT_BOUNDARY = auto()
    HIT_WALL = auto()
    HIT_SELF = auto()
    GAME_NOT_RUNNING = auto()
    TIMEOUT = auto()
    CYCLE_DETECTED = auto()
    WIN = auto()


class SnakeGame:
    """
    Core game state + rules.

    Design rule: no pygame and no networkx imports here. Keep this module importable in
    headless RL training environments.
    """

    def __init__(
            self,
            level: BaseLevel,
            food_count: int | None = None,
            tileset: Tileset | None = None,
            seed: int | None = None,
    ):
        # === Basic config ===
        self.level = level
        self.width = level.width
        self.height = level.height
        self.tileset = tileset or Tileset()

        # === RNG ===
        self.seed_value = seed
        self.rng = random.Random(seed)

        # === Rendering-related buffers (tile-resolution; renderer decides scaling) ===
        self.pixel_buffer: np.ndarray = np.zeros((1, 1), dtype=np.uint8)

        # === Wall setup (static) ===
        self.wall_tiles: list[tuple[Point, TileType]] = level.get_wall_tiles()
        self.wall_positions: set[Point] = {pos for pos, _ in self.wall_tiles}

        self.wall_layer: np.ndarray = np.zeros((self.height, self.width), dtype=np.uint8)
        for pos in self.wall_positions:
            self.wall_layer[pos.y, pos.x] = 1

        # === Layers ===
        self.snake_layers: dict[TileType, np.ndarray] = {}
        self.food_layer: np.ndarray = np.zeros((self.height, self.width), dtype=np.uint8)

        # === Food target count ===
        existing_food = level.get_food_positions()
        if food_count is not None:
            self.target_food_count: int = int(food_count)
        elif existing_food:
            self.target_food_count = len(existing_food)
        else:
            raise ValueError("No food defined in level and no food_count specified.")

        # === Game state ===
        self.spawnable_tiles: set[Point] = set()
        self.snake: list[Point] = []
        self.snake_set: set[Point] = set()
        self.food: list[Point] = []
        self.direction: Direction | None = None

        self.score: int = 0
        self.running: bool = True

        # Optional trackers (kept from your original; used by analysis tooling)
        self.visited: set[Point] = set()

        # Now that wall_positions is known, set correct maxlen once (no double-init)
        self.recent_heads: deque[Point] = deque(maxlen=max(1, self.max_playable_tiles))

        # === Run reset ===
        self.reset()

    def reset(self) -> None:
        """Reset snake, food, score, state flags, and layers."""
        self.snake = self.level.get_snake_segments()
        self.snake_set = set(self.snake)
        self.direction = self.level.get_snake_direction()
        self.food = list(self.level.get_food_positions())
        self.score = 0
        self.running = True

        # spawnable tiles = non-wall minus snake minus food
        # (build from wall_positions; faster than repeated Point(...) checks against full grid)
        self.spawnable_tiles = {
            Point(x, y)
            for y in range(self.height)
            for x in range(self.width)
            if Point(x, y) not in self.wall_positions
        }
        self.spawnable_tiles -= self.snake_set
        self.spawnable_tiles -= set(self.food)

        self._spawn_food()

        self.visited.clear()
        self.recent_heads.clear()

        # Update layers/pixels
        self._update_wall_layer()
        self._update_snake_layers()
        self._update_food_layer()
        self._update_pixel_buffer()

    def set_seed(self, seed: int | None) -> None:
        self.seed_value = seed
        self.rng = random.Random(seed)

    @property
    def max_playable_tiles(self) -> int:
        return self.width * self.height - len(self.wall_positions)

    @property
    def won(self) -> bool:
        return len(self.snake) >= self.max_playable_tiles

    def get_head_position(self) -> Point:
        return self.snake[0]

    def get_food_positions(self) -> list[Point]:
        return self.food.copy()

    def _spawn_food(self) -> int:
        need = self.target_food_count - len(self.food)
        if need <= 0 or not self.spawnable_tiles:
            return 0

        candidates = self.rng.sample(list(self.spawnable_tiles), k=min(need, len(self.spawnable_tiles)))
        for p in candidates:
            self.food.append(p)
            self.spawnable_tiles.remove(p)
        return len(candidates)

    def move(self, rel_dir: RelativeDirection = RelativeDirection.FORWARD) -> list[MoveResult]:
        results = self._move(rel_dir)
        self._update_snake_layers()
        self._update_food_layer()
        self._update_pixel_buffer()
        return results

    def _move(self, rel_dir: RelativeDirection = RelativeDirection.FORWARD) -> list[MoveResult]:
        if not self.running:
            return [MoveResult.GAME_NOT_RUNNING]

        if self.direction is None:
            raise RuntimeError("Snake direction is not initialized (did you call reset?)")

        results: list[MoveResult] = []

        # 1) determine new direction
        new_direction = self.direction
        if rel_dir == RelativeDirection.LEFT:
            new_direction = self.direction.turn_left()
        elif rel_dir == RelativeDirection.RIGHT:
            new_direction = self.direction.turn_right()

        vec = DIRECTION_TO_POINT[new_direction]
        new_head = Point(self.snake[0].x + vec.x, self.snake[0].y + vec.y)

        # 2) collisions
        if not (0 <= new_head.x < self.width and 0 <= new_head.y < self.height):
            self.running = False
            return [MoveResult.HIT_BOUNDARY]

        if new_head in self.wall_positions:
            self.running = False
            return [MoveResult.HIT_WALL]

        # Use set for O(1) collision (big speed-up as snake grows)
        if new_head in self.snake_set:
            self.running = False
            return [MoveResult.HIT_SELF]

        # 3) apply move
        self.direction = new_direction
        self.snake.insert(0, new_head)
        self.snake_set.add(new_head)

        self.spawnable_tiles.discard(new_head)
        self.recent_heads.append(new_head)
        self.visited.add(new_head)

        # 4) eat or tail pop
        if new_head in self.food:
            self.food.remove(new_head)
            self.score += 1
            self.recent_heads.clear()
            self.visited.clear()
            results.append(MoveResult.FOOD_EATEN)

            self._spawn_food()

            if len(self.food) == 0 and self.won:
                self.running = False
                results.append(MoveResult.WIN)
        else:
            tail = self.snake.pop()
            self.snake_set.remove(tail)
            self.spawnable_tiles.add(tail)

        results.insert(0, MoveResult.OK)
        return results

    def _update_wall_layer(self) -> None:
        self.wall_layer.fill(0)
        for pos in self.wall_positions:
            self.wall_layer[pos.y, pos.x] = 1

    def _update_food_layer(self) -> None:
        self.food_layer.fill(0)
        for food in self.food:
            self.food_layer[food.y, food.x] = 1

    def _update_snake_layers(self) -> None:
        # NOTE: Correct but allocates a lot; optimize later by reusing arrays / single int-grid.
        self.snake_layers = {
            tile_type: np.zeros((self.height, self.width), dtype=np.float32)
            for tile_type in TileType
            if tile_type.name.startswith("SNAKE_")
        }

        if len(self.snake) < 2 or self.direction is None:
            return

        head = self.snake[0]
        head_type = {
            Direction.UP: TileType.SNAKE_HEAD_UP,
            Direction.DOWN: TileType.SNAKE_HEAD_DOWN,
            Direction.LEFT: TileType.SNAKE_HEAD_LEFT,
            Direction.RIGHT: TileType.SNAKE_HEAD_RIGHT,
        }[self.direction]
        self.snake_layers[head_type][head.y, head.x] = 1.0

        for i in range(1, len(self.snake) - 1):
            prev = self.snake[i - 1]
            curr = self.snake[i]
            next_ = self.snake[i + 1]

            if prev.x == next_.x:
                tile_type = TileType.SNAKE_BODY_VERTICAL
            elif prev.y == next_.y:
                tile_type = TileType.SNAKE_BODY_HORIZONTAL
            elif (prev.x < curr.x and next_.y > curr.y) or (next_.x < curr.x and prev.y > curr.y):
                tile_type = TileType.SNAKE_BODY_TR
            elif (prev.x > curr.x and next_.y > curr.y) or (next_.x > curr.x and prev.y > curr.y):
                tile_type = TileType.SNAKE_BODY_TL
            elif (prev.x < curr.x and next_.y < curr.y) or (next_.x < curr.x and prev.y < curr.y):
                tile_type = TileType.SNAKE_BODY_BR
            elif (prev.x > curr.x and next_.y < curr.y) or (next_.x > curr.x and prev.y < curr.y):
                tile_type = TileType.SNAKE_BODY_BL
            else:
                raise RuntimeError(
                    f"Could not determine body tile type at index {i}: prev={prev}, curr={curr}, next={next_}"
                )
            self.snake_layers[tile_type][curr.y, curr.x] = 1.0

        tail = self.snake[-1]
        prev = self.snake[-2]
        if prev.x < tail.x:
            tail_type = TileType.SNAKE_TAIL_RIGHT
        elif prev.x > tail.x:
            tail_type = TileType.SNAKE_TAIL_LEFT
        elif prev.y < tail.y:
            tail_type = TileType.SNAKE_TAIL_DOWN
        elif prev.y > tail.y:
            tail_type = TileType.SNAKE_TAIL_UP
        else:
            raise RuntimeError(f"Could not determine tail direction: prev={prev}, tail={tail}")

        self.snake_layers[tail_type][tail.y, tail.x] = 1.0

    def get_all_layers(self, channel_last: bool = False, merge_snake_layers: bool = False) -> np.ndarray:
        if merge_snake_layers:
            snake_combined = np.zeros((self.height, self.width), dtype=np.float32)
            for layer in self.snake_layers.values():
                snake_combined += layer
            layers: list[np.ndarray] = [snake_combined]
        else:
            layers = list(self.snake_layers.values())

        layers.append(self.wall_layer.astype(np.float32))
        layers.append(self.food_layer.astype(np.float32))

        stacked = np.stack(layers, axis=0)
        if channel_last:
            stacked = np.transpose(stacked, (1, 2, 0))
        return stacked

    def _update_pixel_buffer(self) -> None:
        tile_dim = self.tileset.tile_size
        pixel_height = self.height * tile_dim
        pixel_width = self.width * tile_dim
        self.pixel_buffer = np.zeros((pixel_height, pixel_width), dtype=np.uint8)

        for point, tile_type in self.wall_tiles:
            if tile_type in self.tileset:
                tile = np.array(self.tileset[tile_type], dtype=np.uint8)
                py, px = point.y * tile_dim, point.x * tile_dim
                self.pixel_buffer[py: py + tile_dim, px: px + tile_dim] = tile

        if TileType.FOOD in self.tileset:
            tile = np.array(self.tileset[TileType.FOOD], dtype=np.uint8)
            for point in self.food:
                py, px = point.y * tile_dim, point.x * tile_dim
                self.pixel_buffer[py: py + tile_dim, px: px + tile_dim] = tile

        for tile_type, layer in self.snake_layers.items():
            if tile_type in self.tileset:
                tile = np.array(self.tileset[tile_type], dtype=np.uint8)
                yxs = np.argwhere(layer > 0)
                for y, x in yxs:
                    py, px = y * tile_dim, x * tile_dim
                    self.pixel_buffer[py: py + tile_dim, px: px + tile_dim] = tile
