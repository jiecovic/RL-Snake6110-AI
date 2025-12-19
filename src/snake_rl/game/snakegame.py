# src/snake_rl/game/snakegame.py
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
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


@dataclass(frozen=True, slots=True)
class _MoveDelta:
    old_head: Point
    old_dir: Direction
    old_tail: Point
    new_head: Point
    new_dir: Direction
    ate_food: bool
    food_before: set[Point]
    food_after: set[Point]


class SnakeGame:
    """
    Core game state + rules.

    Design rule: no pygame and no networkx imports here. Keep this module importable in
    headless RL training environments.

    Incremental rendering (permanent):
    - Maintain tile_grid (H,W) of uint8 TileType values (TileType.EMPTY=0).
    - Maintain pixel_buffer (H*tile_size, W*tile_size) uint8.
    - On each successful move, update only the handful of cells that changed.

    NOTE:
    - If a move results in a collision (HIT_*), the snake state does NOT advance.
      In that case, we MUST NOT apply incremental deltas (otherwise we'd corrupt buffers).
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

        # === Rendering buffers ===
        self.pixel_buffer: np.ndarray = np.zeros((1, 1), dtype=np.uint8)

        # Canonical tile grid (TileType values, EMPTY=0)
        self.tile_grid: np.ndarray = np.zeros((self.height, self.width), dtype=np.uint8)

        # Optional trackers (kept from your original; used by analysis tooling)
        self.visited: set[Point] = set()

        # === Wall setup (static) ===
        self.wall_tiles: list[tuple[Point, TileType]] = level.get_wall_tiles()
        self.wall_positions: set[Point] = {pos for pos, _ in self.wall_tiles}

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

        # Optional: kept from your original; can be used by cycle detection tooling later
        self.recent_heads: deque[Point] = deque(maxlen=max(1, self.max_playable_tiles))

        # Debug-only verification (optional)
        self.enable_shadow_check: bool = False  # set True temporarily if you suspect a bug

        # Tile cache for fast blits (TileType -> (tile_size,tile_size) uint8)
        self._tile_cache: dict[TileType, np.ndarray] = {}
        self._empty_tile: np.ndarray = np.zeros((self.tileset.tile_size, self.tileset.tile_size), dtype=np.uint8)

        # === Run reset ===
        self.reset()

    def reset(self) -> None:
        """Reset snake, food, score, state flags, and buffers."""
        self.snake = self.level.get_snake_segments()
        self.snake_set = set(self.snake)
        self.direction = self.level.get_snake_direction()
        self.food = list(self.level.get_food_positions())
        self.score = 0
        self.running = True

        # spawnable tiles = non-wall minus snake minus food
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

        # Prepare tile cache + rebuild canonical grid + render full frame
        self._build_tile_cache()
        self._rebuild_tile_grid_full()
        self._render_full_from_tile_grid()

        if self.enable_shadow_check:
            self._shadow_check()

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

    # -------------------------------------------------------------------------
    # Incremental tile-grid + pixel rendering
    # -------------------------------------------------------------------------

    def _build_tile_cache(self) -> None:
        self._tile_cache.clear()
        for tt in TileType:
            if tt == TileType.EMPTY:
                continue
            if tt in self.tileset:
                self._tile_cache[tt] = np.array(self.tileset[tt], dtype=np.uint8)

        # In case tile_size changed (unlikely, but cheap to keep consistent)
        td = int(self.tileset.tile_size)
        if self._empty_tile.shape != (td, td):
            self._empty_tile = np.zeros((td, td), dtype=np.uint8)

    def _tile_at(self, tt: TileType) -> np.ndarray:
        if tt == TileType.EMPTY:
            return self._empty_tile
        return self._tile_cache.get(tt, self._empty_tile)

    def _blit_cell(self, p: Point, tt: TileType) -> None:
        td = int(self.tileset.tile_size)
        py, px = p.y * td, p.x * td
        self.pixel_buffer[py:py + td, px:px + td] = self._tile_at(tt)

    def _render_full_from_tile_grid(self) -> None:
        td = int(self.tileset.tile_size)
        ph, pw = self.height * td, self.width * td
        self.pixel_buffer = np.zeros((ph, pw), dtype=np.uint8)

        ys, xs = np.nonzero(self.tile_grid)
        for y, x in zip(ys.tolist(), xs.tolist()):
            tt = TileType(int(self.tile_grid[y, x]))
            self._blit_cell(Point(int(x), int(y)), tt)

    def _rebuild_tile_grid_full(self) -> None:
        self.tile_grid.fill(int(TileType.EMPTY.value))

        # walls
        for pos, tile_type in self.wall_tiles:
            self.tile_grid[pos.y, pos.x] = int(tile_type.value)

        # food
        for p in self.food:
            self.tile_grid[p.y, p.x] = int(TileType.FOOD.value)

        # snake
        self._paint_full_snake_into_grid()

    def _paint_full_snake_into_grid(self) -> None:
        if not self.snake:
            return
        if self.direction is None:
            raise RuntimeError("Snake direction is not initialized (did you call reset?)")

        h = self.snake[0]
        self.tile_grid[h.y, h.x] = int(self._head_tile(self.direction).value)

        if len(self.snake) == 1:
            return

        for i in range(1, len(self.snake) - 1):
            prev = self.snake[i - 1]
            curr = self.snake[i]
            nxt = self.snake[i + 1]
            self.tile_grid[curr.y, curr.x] = int(self._body_tile(prev, curr, nxt).value)

        tail = self.snake[-1]
        prev = self.snake[-2]
        self.tile_grid[tail.y, tail.x] = int(self._tail_tile(prev, tail).value)

    def _head_tile(self, d: Direction) -> TileType:
        return {
            Direction.UP: TileType.SNAKE_HEAD_UP,
            Direction.DOWN: TileType.SNAKE_HEAD_DOWN,
            Direction.LEFT: TileType.SNAKE_HEAD_LEFT,
            Direction.RIGHT: TileType.SNAKE_HEAD_RIGHT,
        }[d]

    def _tail_tile(self, prev: Point, tail: Point) -> TileType:
        if prev.x < tail.x:
            return TileType.SNAKE_TAIL_RIGHT
        if prev.x > tail.x:
            return TileType.SNAKE_TAIL_LEFT
        if prev.y < tail.y:
            return TileType.SNAKE_TAIL_DOWN
        if prev.y > tail.y:
            return TileType.SNAKE_TAIL_UP
        raise RuntimeError(f"Could not determine tail direction: prev={prev}, tail={tail}")

    def _body_tile(self, prev: Point, curr: Point, nxt: Point) -> TileType:
        if prev.x == nxt.x:
            return TileType.SNAKE_BODY_VERTICAL
        if prev.y == nxt.y:
            return TileType.SNAKE_BODY_HORIZONTAL

        if (prev.x < curr.x and nxt.y > curr.y) or (nxt.x < curr.x and prev.y > curr.y):
            return TileType.SNAKE_BODY_TR
        if (prev.x > curr.x and nxt.y > curr.y) or (nxt.x > curr.x and prev.y > curr.y):
            return TileType.SNAKE_BODY_TL
        if (prev.x < curr.x and nxt.y < curr.y) or (nxt.x < curr.x and prev.y < curr.y):
            return TileType.SNAKE_BODY_BR
        if (prev.x > curr.x and nxt.y < curr.y) or (nxt.x > curr.x and prev.y < curr.y):
            return TileType.SNAKE_BODY_BL

        raise RuntimeError(f"Could not determine body tile type: prev={prev}, curr={curr}, next={nxt}")

    def _apply_incremental_updates(self, delta: _MoveDelta) -> None:
        changed: list[Point] = []

        # Food changes
        added_food = delta.food_after - delta.food_before
        removed_food = delta.food_before - delta.food_after

        for p in removed_food:
            self.tile_grid[p.y, p.x] = int(TileType.EMPTY.value)
            changed.append(p)

        for p in added_food:
            self.tile_grid[p.y, p.x] = int(TileType.FOOD.value)
            changed.append(p)

        # Tail cell becomes empty only if we actually popped tail (i.e., no food eaten)
        if not delta.ate_food:
            self.tile_grid[delta.old_tail.y, delta.old_tail.x] = int(TileType.EMPTY.value)
            changed.append(delta.old_tail)

        # New head
        self.tile_grid[delta.new_head.y, delta.new_head.x] = int(self._head_tile(delta.new_dir).value)
        changed.append(delta.new_head)

        # Old head becomes body (only meaningful if length >= 3 after the move)
        if len(self.snake) >= 3:
            curr = self.snake[1]  # old head position
            prev = self.snake[0]  # new head position
            nxt = self.snake[2]
            self.tile_grid[curr.y, curr.x] = int(self._body_tile(prev, curr, nxt).value)
            changed.append(curr)

        # Tail + pre-tail refresh (cheap & safe)
        if len(self.snake) >= 2:
            tail = self.snake[-1]
            prev_tail = self.snake[-2]
            self.tile_grid[tail.y, tail.x] = int(self._tail_tile(prev_tail, tail).value)
            changed.append(tail)

            if len(self.snake) >= 3:
                curr = self.snake[-2]
                prev = self.snake[-3]
                nxt = self.snake[-1]
                self.tile_grid[curr.y, curr.x] = int(self._body_tile(prev, curr, nxt).value)
                changed.append(curr)

        if not changed:
            return

        seen: set[Point] = set()
        for p in changed:
            if p in seen:
                continue
            seen.add(p)
            tt = TileType(int(self.tile_grid[p.y, p.x]))
            self._blit_cell(p, tt)

    def _shadow_check(self) -> None:
        ref = np.zeros_like(self.tile_grid)
        ref.fill(int(TileType.EMPTY.value))

        for pos, tile_type in self.wall_tiles:
            ref[pos.y, pos.x] = int(tile_type.value)

        for p in self.food:
            ref[p.y, p.x] = int(TileType.FOOD.value)

        if self.snake:
            if self.direction is None:
                raise RuntimeError("Snake direction is not initialized (did you call reset?)")

            ref[self.snake[0].y, self.snake[0].x] = int(self._head_tile(self.direction).value)

            for i in range(1, len(self.snake) - 1):
                bt = self._body_tile(self.snake[i - 1], self.snake[i], self.snake[i + 1])
                ref[self.snake[i].y, self.snake[i].x] = int(bt.value)

            if len(self.snake) >= 2:
                tt = self._tail_tile(self.snake[-2], self.snake[-1])
                ref[self.snake[-1].y, self.snake[-1].x] = int(tt.value)

        if not np.array_equal(ref, self.tile_grid):
            ys, xs = np.where(ref != self.tile_grid)
            y0, x0 = int(ys[0]), int(xs[0])
            raise RuntimeError(
                f"tile_grid mismatch at (x={x0}, y={y0}): fast={int(self.tile_grid[y0, x0])} ref={int(ref[y0, x0])}"
            )

    # -------------------------------------------------------------------------
    # Core move logic (rules) + integration with incremental rendering
    # -------------------------------------------------------------------------

    def move(self, rel_dir: RelativeDirection = RelativeDirection.FORWARD) -> list[MoveResult]:
        if self.direction is None:
            raise RuntimeError("Snake direction is not initialized (did you call reset?)")
        if not self.snake:
            raise RuntimeError("Snake not initialized (did you call reset?)")

        old_head = self.snake[0]
        old_tail = self.snake[-1]
        old_dir = self.direction
        food_before = set(self.food)

        results = self._move(rel_dir)

        # If _move() returned early (collision / not running), the snake state did not advance.
        if MoveResult.OK not in results:
            if self.enable_shadow_check:
                self._shadow_check()
            return results

        new_dir = self.direction if self.direction is not None else old_dir
        new_head = self.snake[0]
        ate_food = MoveResult.FOOD_EATEN in results
        food_after = set(self.food)

        delta = _MoveDelta(
            old_head=old_head,
            old_dir=old_dir,
            old_tail=old_tail,
            new_head=new_head,
            new_dir=new_dir,
            ate_food=ate_food,
            food_before=food_before,
            food_after=food_after,
        )

        # Incremental tile_grid + incremental pixel blits (the one and only path)
        self._apply_incremental_updates(delta)

        if self.enable_shadow_check:
            self._shadow_check()

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

        # 2) collisions (EARLY RETURN: no state advance)
        if not (0 <= new_head.x < self.width and 0 <= new_head.y < self.height):
            self.running = False
            return [MoveResult.HIT_BOUNDARY]

        if new_head in self.wall_positions:
            self.running = False
            return [MoveResult.HIT_WALL]

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
