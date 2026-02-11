# src\snake_rl\game\snakegame.py
from __future__ import annotations

from dataclasses import dataclass
from enum import IntFlag
from typing import Any

import numpy as np

from snake_rl.game.geometry import Direction, Point, RelativeDirection
from snake_rl.game.level import BaseLevel
from snake_rl.game.tile_types import TileType
from snake_rl.game.tileset import Tileset

try:
    import snake_rl._core as rust_core
except Exception:  # pragma: no cover
    rust_core = None  # type: ignore[assignment]


class MoveResult(IntFlag):
    OK = 1 << 0
    FOOD_EATEN = 1 << 1
    HIT_BOUNDARY = 1 << 2
    HIT_WALL = 1 << 3
    HIT_SELF = 1 << 4
    GAME_NOT_RUNNING = 1 << 5
    TIMEOUT = 1 << 6
    WIN = 1 << 7


@dataclass(frozen=True)
class SpawnSpecData:
    x: int | None
    y: int | None
    length: int
    direction: Direction | None
    random_direction: bool
    jitter: int


def has_rust_core() -> bool:
    return rust_core is not None


def ensure_rust_core() -> Any:
    if rust_core is None:
        raise RuntimeError(
            "Rust core module not available. Run `maturin develop` or install with `pip install .`."
        )
    return rust_core


def spawn_from_level(level: BaseLevel) -> SpawnSpecData:
    sp = level.spawn
    return SpawnSpecData(
        x=sp.x,
        y=sp.y,
        length=int(sp.length),
        direction=sp.direction,
        random_direction=bool(sp.random_direction),
        jitter=int(sp.jitter),
    )


def build_level_grid(level: BaseLevel) -> np.ndarray:
    h = int(level.height)
    w = int(level.width)
    grid = np.zeros((h, w), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            grid[y, x] = np.uint8(int(level.grid[y][x].value))
    return grid


def build_tileset_array(tileset: Tileset) -> np.ndarray:
    max_id = int(max(int(t.value) for t in TileType))
    tile_size = int(tileset.tile_size)
    tiles = np.zeros((max_id + 1, tile_size, tile_size), dtype=np.uint8)
    for t in TileType:
        raw = np.array(tileset[t], dtype=np.uint8)
        if raw.size and int(raw.max()) <= 1:
            raw = (raw * np.uint8(255)).astype(np.uint8, copy=False)
        tiles[int(t.value)] = raw
    return tiles


def dir_to_i8(d: Direction | None) -> int:
    if d is None:
        return -1
    return int(d.value)


def i8_to_dir(v: int) -> Direction | None:
    iv = int(v)
    if iv < 0:
        return None
    return Direction(iv)


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

        core = ensure_rust_core()

        self.level = level
        self.tileset = tileset or Tileset()

        grid = build_level_grid(level)
        tiles = build_tileset_array(self.tileset)
        sp = spawn_from_level(level)

        self._core = core.Game(
            width=int(level.width),
            height=int(level.height),
            level_grid=grid,
            spawn_len=int(sp.length),
            spawn_dir=dir_to_i8(sp.direction),
            spawn_random_dir=bool(sp.random_direction),
            spawn_jitter=int(sp.jitter),
            food_count=int(food_count),
            tile_size=int(self.tileset.tile_size),
            tiles=tiles,
            spawn_x=sp.x,
            spawn_y=sp.y,
            seed=None if seed is None else int(seed),
        )

        self._tile_grid: np.ndarray | None = None
        self._pixel_buffer: np.ndarray | None = None
        self._direction: Direction | None = None
        self._score: int = 0
        self._running: bool = True
        self._snake_len: int = 0
        self._head_pos: Point = Point(0, 0)
        self._food_positions: list[Point] = []
        self._spawnable_count: int = 0

    def reset(self, seed: int | None = None) -> None:
        self._core.reset(None if seed is None else int(seed))
        self._refresh_state()

    def move(self, rel_dir: RelativeDirection = RelativeDirection.FORWARD) -> MoveResult:
        mask = int(self._core.step(int(rel_dir.value)))
        self._refresh_state()
        return MoveResult(mask)

    def _refresh_state(self) -> None:
        self._tile_grid = np.asarray(self._core.tile_grid(), dtype=np.uint8)
        self._pixel_buffer = np.asarray(self._core.pixel_grid(), dtype=np.uint8)
        self._direction = i8_to_dir(int(self._core.direction()))
        self._score = int(self._core.score)
        self._running = bool(self._core.running)
        self._snake_len = int(self._core.snake_len())
        hx, hy = self._core.head_pos()
        self._head_pos = Point(int(hx), int(hy))
        self._food_positions = [Point(int(x), int(y)) for x, y in self._core.food_positions()]
        self._spawnable_count = int(self._core.spawnable_count())

    # ---- properties used by envs/renderers ----

    @property
    def width(self) -> int:
        return int(self._core.width)

    @property
    def height(self) -> int:
        return int(self._core.height)

    @property
    def tile_grid(self) -> np.ndarray:
        if self._tile_grid is None:
            self._tile_grid = np.asarray(self._core.tile_grid(), dtype=np.uint8)
        return self._tile_grid

    @property
    def pixel_buffer(self) -> np.ndarray:
        if self._pixel_buffer is None:
            self._pixel_buffer = np.asarray(self._core.pixel_grid(), dtype=np.uint8)
        return self._pixel_buffer

    @property
    def direction(self) -> Direction | None:
        return self._direction

    @property
    def score(self) -> int:
        return self._score

    @property
    def running(self) -> bool:
        return self._running

    @property
    def snake_len(self) -> int:
        return self._snake_len

    def get_head_position(self) -> Point:
        return self._head_pos

    def get_food_positions(self) -> list[Point]:
        return list(self._food_positions)

    @property
    def max_playable_tiles(self) -> int:
        return int(self._core.max_playable_tiles())

    @property
    def spawnable_count(self) -> int:
        return self._spawnable_count


__all__ = [
    "MoveResult",
    "SnakeGame",
    "SpawnSpecData",
    "build_level_grid",
    "build_tileset_array",
    "dir_to_i8",
    "ensure_rust_core",
    "has_rust_core",
    "i8_to_dir",
    "spawn_from_level",
]
