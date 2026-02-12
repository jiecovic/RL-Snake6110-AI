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
        frame_stack_n: int = 1,
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
            frame_stack_n=int(frame_stack_n),
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

    def head_pixel_view_stacked(
        self,
        *,
        view_radius: int | tuple[int, int],
        rotate_to_head: bool = True,
        oob_fill_value: int = 0,
        return_valid: bool = False,
    ):
        ry, rx = parse_view_radius(view_radius)
        fn = getattr(self._core, "head_pixel_view_stacked", None)
        if fn is None:
            raise RuntimeError(
                "Rust core does not expose head_pixel_view_stacked (rebuild the extension)."
            )
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

    def head_tile_view_stacked(
        self,
        *,
        view_radius: int | tuple[int, int],
        rotate_to_head: bool = True,
        empty_id: int | None = None,
    ):
        ry, rx = parse_view_radius(view_radius)
        fn = getattr(self._core, "head_tile_view_stacked", None)
        if fn is None:
            raise RuntimeError(
                "Rust core does not expose head_tile_view_stacked (rebuild the extension)."
            )
        return fn(
            (int(ry), int(rx)),
            rotate_to_head=bool(rotate_to_head),
            empty_id=None if empty_id is None else int(empty_id),
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

    def tile_grid_stacked(self) -> np.ndarray:
        fn = getattr(self._core, "tile_grid_stacked", None)
        if fn is None:
            raise RuntimeError(
                "Rust core does not expose tile_grid_stacked (rebuild the extension)."
            )
        return np.asarray(fn(), dtype=np.uint8)

    @property
    def pixel_buffer(self) -> np.ndarray:
        return np.asarray(self._core.pixel_grid(), dtype=np.uint8)

    def pixel_buffer_stacked(self) -> np.ndarray:
        fn = getattr(self._core, "pixel_grid_stacked", None)
        if fn is None:
            raise RuntimeError(
                "Rust core does not expose pixel_grid_stacked (rebuild the extension)."
            )
        return np.asarray(fn(), dtype=np.uint8)

    @property
    def direction(self) -> int | None:
        d = int(self._core.direction())
        return None if d < 0 else d

    @property
    def direction_id(self) -> int:
        fn = getattr(self._core, "direction_id", None)
        if fn is None:
            raise RuntimeError("Rust core does not expose direction_id (rebuild the extension).")
        return int(fn())

    @property
    def score(self) -> int:
        return int(self._core.score)

    @property
    def running(self) -> bool:
        return bool(self._core.running)

    @property
    def episode_steps(self) -> int:
        fn = getattr(self._core, "episode_steps", None)
        if fn is None:
            raise RuntimeError("Rust core does not expose episode_steps (rebuild the extension).")
        return int(fn())

    @property
    def snake_len(self) -> int:
        return int(self._core.snake_len())

    @property
    def snake_progress(self) -> float:
        return float(self._core.snake_progress())

    def get_head_position(self) -> tuple[int, int]:
        hx, hy = self._core.head_pos()
        return (int(hx), int(hy))

    def get_food_positions(self) -> list[tuple[int, int]]:
        return [(int(x), int(y)) for x, y in self._core.food_positions()]

    def get_closest_food(self) -> tuple[int, int, int] | None:
        fn = getattr(self._core, "closest_food", None)
        if fn is None:
            raise RuntimeError("Rust core does not expose closest_food (rebuild the extension).")
        out = fn()
        if out is None:
            return None
        dx, dy, dist = out
        return (int(dx), int(dy), int(dist))

    def get_closest_food_norm(self, metric: str = "manhattan") -> tuple[float, float, float]:
        fn = getattr(self._core, "closest_food_norm", None)
        if fn is None:
            raise RuntimeError(
                "Rust core does not expose closest_food_norm (rebuild the extension)."
            )
        metric = str(metric).strip().lower()
        code = 0
        if metric == "euclidean":
            code = 1
        elif metric == "euclidean_sq":
            code = 2
        elif metric != "manhattan":
            raise ValueError("closest_food.metric must be manhattan|euclidean|euclidean_sq")
        dx, dy, dist = fn(int(code))
        return (float(dx), float(dy), float(dist))

    @property
    def steps_since_food(self) -> int:
        fn = getattr(self._core, "steps_since_food", None)
        if fn is None:
            raise RuntimeError(
                "Rust core does not expose steps_since_food (rebuild the extension)."
            )
        return int(fn())

    def time_since_food_norm(self, max_steps: int) -> float:
        fn = getattr(self._core, "time_since_food_norm", None)
        if fn is None:
            raise RuntimeError(
                "Rust core does not expose time_since_food_norm (rebuild the extension)."
            )
        return float(fn(int(max_steps)))

    def collision_flags(self) -> tuple[bool, bool, bool]:
        fn = getattr(self._core, "collision_flags", None)
        if fn is None:
            raise RuntimeError("Rust core does not expose collision_flags (rebuild the extension).")
        ahead, left, right = fn()
        return (bool(ahead), bool(left), bool(right))

    @property
    def max_playable_tiles(self) -> int:
        return int(self._core.max_playable_tiles())

    @property
    def spawnable_count(self) -> int:
        return int(self._core.spawnable_count())

    @property
    def frame_stack_n(self) -> int:
        return int(getattr(self._core, "frame_stack_n", 1))


__all__ = [
    "SnakeEngine",
    "ensure_rust_core",
    "has_rust_core",
]
