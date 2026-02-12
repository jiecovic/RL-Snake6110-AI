# src/snake_rl/game/snake_engine.py
from __future__ import annotations

from typing import Any

import numpy as np

from snake_rl import _core as rust_core

_TILESET_TILES: np.ndarray | None = None
_TILESET_TILE_SIZE: int | None = None
_TILE_ID_COUNT: int | None = None
_TILE_NAMES: list[str] | None = None


def has_rust_core() -> bool:
    return rust_core is not None


def ensure_rust_core() -> Any:
    if rust_core is None:
        raise RuntimeError(
            "Rust core module not available. Run `maturin develop` or install with `pip install .`."
        )
    return rust_core


def tileset_tile_size() -> int:
    global _TILESET_TILE_SIZE
    if _TILESET_TILE_SIZE is None:
        core = ensure_rust_core()
        _TILESET_TILE_SIZE = int(core.tileset_tile_size())
    return _TILESET_TILE_SIZE


def tileset_tiles() -> np.ndarray:
    global _TILESET_TILES
    if _TILESET_TILES is None:
        core = ensure_rust_core()
        _TILESET_TILES = np.asarray(core.tileset_tiles(), dtype=np.uint8)
    return _TILESET_TILES


def tile_empty_id() -> int:
    core = ensure_rust_core()
    return int(core.TILE_EMPTY)


def tileset_tile_count() -> int:
    global _TILE_ID_COUNT
    if _TILE_ID_COUNT is None:
        core = ensure_rust_core()
        _TILE_ID_COUNT = int(core.tileset_tile_count())
    return _TILE_ID_COUNT


def tileset_tile_names() -> list[str]:
    global _TILE_NAMES
    if _TILE_NAMES is None:
        core = ensure_rust_core()
        _TILE_NAMES = [str(x) for x in core.tileset_tile_names()]
    return list(_TILE_NAMES)


def _i8_to_dir(v: int) -> int | None:
    iv = int(v)
    if iv < 0:
        return None
    return iv


class SnakeEngine:
    """
    Rust-backed core game state + rules (headless, RL-safe).

    RNG contract
    - Rust owns all randomness.
    - env.reset(seed) passes a seed to Rust; resets without seed continue RNG stream.
    """

    def __init__(
        self,
        *,
        board: Any,
        food_count: int | None = None,
        seed: int | None = None,
    ):
        if food_count is None:
            raise ValueError("food_count must be provided (board does not encode food).")

        core = ensure_rust_core()
        if not isinstance(board, core.Board):
            raise TypeError("board must be a snake_rl._core.Board instance")
        self._core = core.SnakeEngine(
            board=board,
            food_count=int(food_count),
            seed=None,
        )

        self._tile_grid: np.ndarray | None = None
        self._pixel_buffer: np.ndarray | None = None
        self._direction: int | None = None
        self._score: int = 0
        self._running: bool = True
        self._snake_len: int = 0
        self._head_pos: tuple[int, int] = (0, 0)
        self._food_positions: list[tuple[int, int]] = []
        self._spawnable_count: int = 0

        self._core.reset(None if seed is None else int(seed))
        self._refresh_state()

    def reset(self, seed: int | None = None) -> None:
        self._core.reset(None if seed is None else int(seed))
        self._refresh_state()

    def move(self, rel_dir: int = 0) -> int:
        mask = int(self._core.step(int(rel_dir)))
        self._refresh_state()
        return mask

    def move_relative(self, rel_dir: int = 0) -> int:
        step = getattr(self._core, "step_relative", None)
        if step is None:
            step = self._core.step
        mask = int(step(int(rel_dir)))
        self._refresh_state()
        return mask

    def move_cardinal(self, abs_dir: int) -> int:
        step = getattr(self._core, "step_cardinal", None)
        if step is None:
            raise RuntimeError("Rust core does not expose step_cardinal (rebuild the extension).")
        mask = int(step(int(abs_dir)))
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
        self._head_pos = (int(hx), int(hy))
        self._food_positions = [(int(x), int(y)) for x, y in self._core.food_positions()]
        self._spawnable_count = int(self._core.spawnable_count())

    # ---- properties used by envs/renderers ----

    @property
    def width(self) -> int:
        return int(self._core.width)

    @property
    def height(self) -> int:
        return int(self._core.height)

    @property
    def tile_size(self) -> int:
        return int(self._core.tile_size)

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
    def direction(self) -> int | None:
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

    def get_head_position(self) -> tuple[int, int]:
        return self._head_pos

    def get_food_positions(self) -> list[tuple[int, int]]:
        return list(self._food_positions)

    @property
    def max_playable_tiles(self) -> int:
        return int(self._core.max_playable_tiles())

    @property
    def spawnable_count(self) -> int:
        return self._spawnable_count


__all__ = [
    "SnakeEngine",
    "ensure_rust_core",
    "has_rust_core",
    "tile_empty_id",
    "tileset_tile_count",
    "tileset_tile_names",
    "tileset_tile_size",
    "tileset_tiles",
]
