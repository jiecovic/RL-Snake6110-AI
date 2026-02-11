# src/snake_rl/core/game.py
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
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


def _ensure_rust() -> Any:
    if rust_core is None:
        raise RuntimeError(
            "Rust core module not available. Run `maturin develop` or install with `pip install .`."
        )
    return rust_core


def _spawn_from_level(level: BaseLevel) -> SpawnSpecData:
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
    # Ensure tiles are 0..255 uint8 and aligned by TileType.value.
    max_id = int(max(int(t.value) for t in TileType))
    tile_size = int(tileset.tile_size)
    tiles = np.zeros((max_id + 1, tile_size, tile_size), dtype=np.uint8)
    for t in TileType:
        raw = np.array(tileset[t], dtype=np.uint8)
        if raw.size and int(raw.max()) <= 1:
            raw = (raw * np.uint8(255)).astype(np.uint8, copy=False)
        tiles[int(t.value)] = raw
    return tiles


def _dir_to_i8(d: Direction | None) -> int:
    if d is None:
        return -1
    return int(d.value)


def _i8_to_dir(v: int) -> Direction | None:
    if v is None:
        return None
    iv = int(v)
    if iv < 0:
        return None
    return Direction(iv)


class RustGame:
    def __init__(
        self,
        level: BaseLevel,
        *,
        food_count: int,
        tileset: Tileset | None = None,
        seed: int | None = None,
    ) -> None:
        core = _ensure_rust()

        self.level = level
        self.tileset = tileset or Tileset()

        grid = build_level_grid(level)
        tiles = build_tileset_array(self.tileset)
        sp = _spawn_from_level(level)

        self._core = core.Game(
            width=int(level.width),
            height=int(level.height),
            level_grid=grid,
            spawn_len=int(sp.length),
            spawn_dir=_dir_to_i8(sp.direction),
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

    def move(self, rel_dir: RelativeDirection = RelativeDirection.FORWARD) -> int:
        mask = int(self._core.step(int(rel_dir.value)))
        self._refresh_state()
        return mask

    def _refresh_state(self) -> None:
        self._tile_grid = np.asarray(self._core.tile_grid(), dtype=np.uint8)
        self._pixel_buffer = np.asarray(self._core.pixel_grid(), dtype=np.uint8)
        self._direction = _i8_to_dir(int(self._core.direction()))
        self._score = int(self._core.score)
        self._running = bool(self._core.running)
        self._snake_len = int(self._core.snake_len())
        hx, hy = self._core.head_pos()
        self._head_pos = Point(int(hx), int(hy))
        self._food_positions = [Point(int(x), int(y)) for x, y in self._core.food_positions()]
        self._spawnable_count = int(self._core.spawnable_count())

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


class RustVecGame:
    def __init__(
        self,
        *,
        n: int,
        level: BaseLevel,
        food_count: int,
        tileset: Tileset | None = None,
        seeds: Iterable[int] | None = None,
    ) -> None:
        core = _ensure_rust()

        self.level = level
        self.tileset = tileset or Tileset()

        grid = build_level_grid(level)
        tiles = build_tileset_array(self.tileset)
        sp = _spawn_from_level(level)

        seed_list = None
        if seeds is not None:
            seed_list = [int(s) for s in seeds]

        self._core = core.VecGame(
            n=int(n),
            width=int(level.width),
            height=int(level.height),
            level_grid=grid,
            spawn_len=int(sp.length),
            spawn_dir=_dir_to_i8(sp.direction),
            spawn_random_dir=bool(sp.random_direction),
            spawn_jitter=int(sp.jitter),
            food_count=int(food_count),
            tile_size=int(self.tileset.tile_size),
            tiles=tiles,
            spawn_x=sp.x,
            spawn_y=sp.y,
            seeds=seed_list,
        )

    def reset(self, seeds: Iterable[int] | None = None) -> None:
        seed_list = None
        if seeds is not None:
            seed_list = [int(s) for s in seeds]
        self._core.reset(seed_list)

    def reset_one(self, index: int, seed: int | None = None) -> None:
        self._core.reset_one(int(index), None if seed is None else int(seed))

    def step(self, actions: Iterable[int]) -> list[int]:
        return [int(x) for x in self._core.step([int(a) for a in actions])]

    def tile_grids(self) -> np.ndarray:
        return np.asarray(self._core.tile_grids(), dtype=np.uint8)

    def pixel_grids(self) -> np.ndarray:
        return np.asarray(self._core.pixel_grids(), dtype=np.uint8)

    def directions(self) -> np.ndarray:
        return np.asarray(self._core.directions(), dtype=np.int8)

    def head_positions(self) -> np.ndarray:
        return np.asarray(self._core.head_positions(), dtype=np.int32)

    def scores(self) -> np.ndarray:
        return np.asarray(self._core.scores(), dtype=np.int32)

    def running(self) -> np.ndarray:
        return np.asarray(self._core.running(), dtype=np.bool_)

    def snake_lens(self) -> np.ndarray:
        return np.asarray(self._core.snake_lens(), dtype=np.int32)

    def max_playable_tiles(self) -> int:
        return int(self._core.max_playable_tiles())

    def spawnable_counts(self) -> np.ndarray:
        return np.asarray(self._core.spawnable_counts(), dtype=np.int32)


__all__ = [
    "RustGame",
    "RustVecGame",
    "has_rust_core",
    "build_level_grid",
    "build_tileset_array",
]
