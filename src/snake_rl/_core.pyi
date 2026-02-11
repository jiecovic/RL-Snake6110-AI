from __future__ import annotations

from collections.abc import Iterable
from typing import Any

MOVE_OK: int
MOVE_FOOD: int
MOVE_HIT_BOUNDARY: int
MOVE_HIT_WALL: int
MOVE_HIT_SELF: int
MOVE_NOT_RUNNING: int
MOVE_TIMEOUT: int
MOVE_WIN: int

class Game:
    width: int
    height: int
    tile_size: int
    score: int
    running: bool

    def __init__(
        self,
        *,
        width: int,
        height: int,
        level_grid: Any,
        spawn_len: int,
        spawn_dir: int,
        spawn_random_dir: bool,
        spawn_jitter: int,
        food_count: int,
        tile_size: int,
        tiles: Any,
        spawn_x: int | None = ...,
        spawn_y: int | None = ...,
        seed: int | None = ...,
    ) -> None: ...
    def reset(self, seed: int | None = ...) -> None: ...
    def step(self, rel_dir: int) -> int: ...
    def tile_grid(self) -> Any: ...
    def pixel_grid(self) -> Any: ...
    def direction(self) -> int: ...
    def snake_len(self) -> int: ...
    def head_pos(self) -> tuple[int, int]: ...
    def food_positions(self) -> list[tuple[int, int]]: ...
    def max_playable_tiles(self) -> int: ...
    def spawnable_count(self) -> int: ...

class VecGame:
    def __init__(
        self,
        *,
        n: int,
        width: int,
        height: int,
        level_grid: Any,
        spawn_len: int,
        spawn_dir: int,
        spawn_random_dir: bool,
        spawn_jitter: int,
        food_count: int,
        tile_size: int,
        tiles: Any,
        spawn_x: int | None = ...,
        spawn_y: int | None = ...,
        seeds: list[int] | None = ...,
    ) -> None: ...
    def reset(self, seeds: list[int] | None = ...) -> None: ...
    def reset_one(self, index: int, seed: int | None = ...) -> None: ...
    def step(self, actions: Iterable[int]) -> list[int]: ...
    def tile_grids(self) -> Any: ...
    def pixel_grids(self) -> Any: ...
    def directions(self) -> list[int]: ...
    def head_positions(self) -> list[tuple[int, int]]: ...
    def scores(self) -> list[int]: ...
    def running(self) -> list[bool]: ...
    def snake_lens(self) -> list[int]: ...
    def max_playable_tiles(self) -> int: ...
    def spawnable_counts(self) -> list[int]: ...
