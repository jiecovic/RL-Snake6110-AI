# src/snake_rl/game/snake_engine.py
from __future__ import annotations

from typing import Any

import numpy as np

from snake_rl import _core as rust_core
from snake_rl.envs.view_radius import parse_view_radius


def has_rust_core() -> bool:
    return rust_core is not None


def ensure_rust_core() -> Any:
    if rust_core is None:
        raise RuntimeError(
            "Rust core module not available. Run `maturin develop` or install with `pip install .`."
        )
    return rust_core


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

        self._core.reset(None if seed is None else int(seed))

    def reset(self, seed: int | None = None) -> None:
        self._core.reset(None if seed is None else int(seed))

    def move(self, rel_dir: int = 0) -> int:
        mask = int(self._core.step(int(rel_dir)))
        return mask

    def move_relative(self, rel_dir: int = 0) -> int:
        step = getattr(self._core, "step_relative", None)
        if step is None:
            step = self._core.step
        mask = int(step(int(rel_dir)))
        return mask

    def move_cardinal(self, abs_dir: int) -> int:
        step = getattr(self._core, "step_cardinal", None)
        if step is None:
            raise RuntimeError("Rust core does not expose step_cardinal (rebuild the extension).")
        mask = int(step(int(abs_dir)))
        return mask

    def head_pixel_view(
        self,
        *,
        view_radius: int | tuple[int, int],
        rotate_to_head: bool = True,
        oob_fill_value: int = 0,
        return_valid: bool = False,
    ):
        ry, rx = parse_view_radius(view_radius)
        fn = getattr(self._core, "head_pixel_view", None)
        if fn is None:
            raise RuntimeError("Rust core does not expose head_pixel_view (rebuild the extension).")
        return fn(
            (int(ry), int(rx)),
            rotate_to_head=bool(rotate_to_head),
            oob_fill_value=int(oob_fill_value),
            return_valid=bool(return_valid),
        )

    def head_tile_view(
        self,
        *,
        view_radius: int | tuple[int, int],
        rotate_to_head: bool = True,
        empty_id: int | None = None,
        return_valid: bool = False,
    ):
        ry, rx = parse_view_radius(view_radius)
        fn = getattr(self._core, "head_tile_view", None)
        if fn is None:
            raise RuntimeError("Rust core does not expose head_tile_view (rebuild the extension).")
        return fn(
            (int(ry), int(rx)),
            rotate_to_head=bool(rotate_to_head),
            empty_id=None if empty_id is None else int(empty_id),
            return_valid=bool(return_valid),
        )

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
        return np.asarray(self._core.tile_grid(), dtype=np.uint8)

    @property
    def pixel_buffer(self) -> np.ndarray:
        return np.asarray(self._core.pixel_grid(), dtype=np.uint8)

    @property
    def direction(self) -> int | None:
        d = int(self._core.direction())
        return None if d < 0 else d

    @property
    def score(self) -> int:
        return int(self._core.score)

    @property
    def running(self) -> bool:
        return bool(self._core.running)

    @property
    def snake_len(self) -> int:
        return int(self._core.snake_len())

    def get_head_position(self) -> tuple[int, int]:
        hx, hy = self._core.head_pos()
        return (int(hx), int(hy))

    def get_food_positions(self) -> list[tuple[int, int]]:
        return [(int(x), int(y)) for x, y in self._core.food_positions()]

    @property
    def max_playable_tiles(self) -> int:
        return int(self._core.max_playable_tiles())

    @property
    def spawnable_count(self) -> int:
        return int(self._core.spawnable_count())


__all__ = [
    "SnakeEngine",
    "ensure_rust_core",
    "has_rust_core",
]
