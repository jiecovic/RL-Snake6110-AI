# src/snake_rl/envs/specs.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from gymnasium import spaces

from snake_rl import _core as core
from snake_rl.envs.view_radius import parse_view_radius
from snake_rl.game.snake_engine import SnakeEngine
from snake_rl.vocab import TileVocab, load_tile_vocab


@dataclass(frozen=True)
class ActionSpec:
    type: str = "relative"

    def _kind(self) -> str:
        t = str(self.type).strip().lower()
        if t in {"relative", "rel"}:
            return "relative"
        if t in {"cardinal", "card", "absolute", "abs", "world"}:
            return "cardinal"
        raise ValueError(f"Unknown action.type={self.type!r}. Expected relative|cardinal.")

    def action_space_n(self) -> int:
        return 3 if self._kind() == "relative" else 4

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(self.action_space_n())

    def kind(self) -> str:
        return self._kind()

    def validate_action(self, action: int) -> None:
        a = int(action)
        n = self.action_space_n()
        if a < 0 or a >= n:
            raise ValueError(f"action must be in [0,{n - 1}], got {a!r}")

    def step(self, game: SnakeEngine, action: int) -> int:
        if self._kind() == "relative":
            return game.move_relative(int(action))
        return game.move_cardinal(int(action))


@dataclass(frozen=True)
class ObservationSpec:
    kind: str
    view: str
    params: dict[str, Any] = field(default_factory=dict)
    features: dict[str, Any] = field(default_factory=dict)

    def _kind(self) -> str:
        k = str(self.kind).strip().lower()
        if k in {"pixel", "pixels"}:
            return "pixel"
        if k in {"categorical", "cat"}:
            return "categorical"
        raise ValueError(f"Unknown obs.kind={self.kind!r}. Expected pixel|categorical.")

    def _view(self) -> str:
        v = str(self.view).strip().lower()
        if v in {"world", "global"}:
            return "world"
        if v in {"head", "egocentric", "ego"}:
            return "head"
        raise ValueError(f"Unknown obs.view={self.view!r}. Expected world|head.")

    def kind_norm(self) -> str:
        return self._kind()

    def view_norm(self) -> str:
        return self._view()

    def _feature_direction(self) -> bool:
        v = self.features.get("direction", False)
        return bool(v)

    def _feature_enabled(self, key: str) -> bool:
        v = self.features.get(key, False)
        if isinstance(v, dict):
            return bool(v.get("enabled", True))
        return bool(v)

    def _feature_snake_progress(self) -> bool:
        if self._feature_enabled("snake_progress"):
            return True
        # legacy alias
        return self._feature_enabled("fill")

    def _feature_time_since_food(self) -> bool:
        return self._feature_enabled("time_since_last_food")

    def _feature_closest_food(self) -> tuple[bool, str]:
        v = self.features.get("closest_food", False)
        if isinstance(v, dict):
            enabled = bool(v.get("enabled", True))
            metric = str(v.get("metric", "manhattan")).strip().lower()
            return enabled, metric
        if isinstance(v, bool):
            return bool(v), "manhattan"
        return False, "manhattan"

    def _feature_collision(self, key: str) -> bool:
        return self._feature_enabled(key)

    def frame_stack_key(self) -> str | None:
        if not self.uses_dict():
            return None
        return "pixel" if self._kind() == "pixel" else "categorical"

    def uses_dict(self) -> bool:
        return bool(
            self._feature_direction()
            or self._feature_snake_progress()
            or self._feature_time_since_food()
            or self._feature_closest_food()[0]
            or self._feature_collision("collision_ahead")
            or self._feature_collision("collision_left")
            or self._feature_collision("collision_right")
        )

    def tile_vocab_name(self) -> str | None:
        name = self.params.get("tile_vocab")
        if name is None:
            return None
        s = str(name).strip()
        return s if s else None

    def load_tile_vocab(self) -> TileVocab | None:
        name = self.tile_vocab_name()
        if name is None:
            return None
        return load_tile_vocab(name)

    def _remove_border(self) -> bool:
        return bool(self.params.get("remove_border", True))

    def _view_radius(self) -> tuple[int, int]:
        v = self.params.get("view_radius", None)
        if v is None:
            raise ValueError("obs.view_radius is required for head views")
        return parse_view_radius(v)

    def _rotate_to_head(self) -> bool:
        return bool(self.params.get("rotate_to_head", True))

    def _add_oob_mask(self) -> bool:
        return bool(self.params.get("add_oob_mask", False))

    def _pixel_oob_value(self) -> int:
        return int(self.params.get("pixel_oob_value", 255))

    def _mask_valid_value(self) -> int:
        return int(self.params.get("mask_valid_value", 255))

    def _mask_oob_value(self) -> int:
        return int(self.params.get("mask_oob_value", 0))

    def _tile_mask_oob(self) -> bool:
        return bool(self.params.get("mask_oob", False))

    def make_space(
        self,
        *,
        width: int,
        height: int,
        tile_size: int,
        tile_vocab: TileVocab | None = None,
        frame_stack_n: int = 1,
    ) -> spaces.Space:
        kind = self._kind()
        view = self._view()
        base_key = "pixel" if kind == "pixel" else "categorical"
        n_stack = max(1, int(frame_stack_n))

        if kind == "pixel":
            if view == "world":
                ph = int(height) * int(tile_size)
                pw = int(width) * int(tile_size)
                if self._remove_border():
                    ph -= 2 * int(tile_size)
                    pw -= 2 * int(tile_size)
                base_space = spaces.Box(
                    low=0,
                    high=255,
                    shape=(n_stack, ph, pw),
                    dtype=np.uint8,
                )
                frame_base_channels = int(core.pixel_frame_channels(False))
            else:
                ry, rx = self._view_radius()
                vh = (2 * ry + 1) * int(tile_size)
                vw = (2 * rx + 1) * int(tile_size)
                frame_base_channels = int(core.pixel_frame_channels(bool(self._add_oob_mask())))
                c = n_stack * frame_base_channels
                base_space = spaces.Box(low=0, high=255, shape=(c, vh, vw), dtype=np.uint8)
            base_space._frame_stack_n = int(n_stack)  # type: ignore[attr-defined]
            base_space._frame_base_channels = int(frame_base_channels)  # type: ignore[attr-defined]
        else:
            if tile_vocab is not None:
                base_num = int(tile_vocab.num_classes)
            else:
                base_num = int(core.tileset_tile_count())

            if view == "world":
                gh = int(height)
                gw = int(width)
                if self._remove_border():
                    if gh <= 2 or gw <= 2:
                        raise ValueError(f"remove_border=True requires h,w > 2; got h={gh} w={gw}")
                    gh -= 2
                    gw -= 2
                base_space = spaces.Box(
                    low=0,
                    high=base_num - 1,
                    shape=(n_stack, gh, gw),
                    dtype=np.uint8,
                )
            else:
                ry, rx = self._view_radius()
                vy = 2 * ry + 1
                vx = 2 * rx + 1
                base_space = spaces.Box(
                    low=0,
                    high=base_num - 1,
                    shape=(n_stack, vy, vx),
                    dtype=np.uint8,
                )
            base_space._frame_stack_n = int(n_stack)  # type: ignore[attr-defined]
            base_space._frame_base_channels = int(core.categorical_frame_channels())  # type: ignore[attr-defined]

        extras: dict[str, spaces.Space] = {}
        if self._feature_direction():
            extras["direction"] = spaces.Discrete(4)
        if self._feature_snake_progress():
            extras["snake_progress"] = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        if self._feature_time_since_food():
            extras["time_since_last_food"] = spaces.Box(
                low=0.0, high=1.0, shape=(1,), dtype=np.float32
            )
        if self._feature_closest_food()[0]:
            extras["closest_food_dx"] = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
            extras["closest_food_dy"] = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
            extras["closest_food_dist"] = spaces.Box(
                low=0.0, high=1.0, shape=(1,), dtype=np.float32
            )
        if self._feature_collision("collision_ahead"):
            extras["collision_ahead"] = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        if self._feature_collision("collision_left"):
            extras["collision_left"] = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        if self._feature_collision("collision_right"):
            extras["collision_right"] = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        if extras:
            return spaces.Dict({base_key: base_space, **extras})
        return base_space

    def observe(
        self,
        *,
        game: SnakeEngine,
        tile_vocab: TileVocab | None = None,
        initial_snake_length: int,
        max_playable_tiles: int,
        max_steps: int,
        frame_stack_n: int = 1,
    ):
        kind = self._kind()
        view = self._view()
        _ = int(initial_snake_length)
        _ = int(max_playable_tiles)
        _ = int(frame_stack_n)

        if kind == "pixel":
            if view == "world":
                frames = np.asarray(game.pixel_buffer_stacked(), dtype=np.uint8)
                if self._remove_border():
                    ts = int(game.tile_size)
                    frames = frames[:, ts:-ts, ts:-ts]
                base = frames
            else:
                if self._add_oob_mask():
                    frame, valid = game.head_pixel_view_stacked(
                        view_radius=self._view_radius(),
                        rotate_to_head=self._rotate_to_head(),
                        oob_fill_value=self._pixel_oob_value(),
                        return_valid=True,
                    )
                    frame_arr = np.asarray(frame, dtype=np.uint8)
                    valid_arr = np.asarray(valid, dtype=bool)
                    mask = np.full(
                        frame_arr.shape,
                        np.uint8(self._mask_oob_value()),
                        dtype=np.uint8,
                    )
                    mask[valid_arr] = np.uint8(self._mask_valid_value())
                    base = np.stack([frame_arr, mask], axis=1)
                    base = base.reshape(base.shape[0] * base.shape[1], base.shape[2], base.shape[3])
                else:
                    frame = game.head_pixel_view_stacked(
                        view_radius=self._view_radius(),
                        rotate_to_head=self._rotate_to_head(),
                        oob_fill_value=self._pixel_oob_value(),
                        return_valid=False,
                    )
                    base = np.asarray(frame, dtype=np.uint8)

            base_key = "pixel"
        else:
            grid = np.asarray(game.tile_grid_stacked(), dtype=np.uint8)
            if view == "world":
                if self._remove_border():
                    grid = grid[:, 1:-1, 1:-1]
                frame = grid if tile_vocab is None else tile_vocab.lut[grid]
                base = frame.astype(np.uint8, copy=False)
            else:
                raw = game.head_tile_view_stacked(
                    view_radius=self._view_radius(),
                    rotate_to_head=self._rotate_to_head(),
                    empty_id=None,
                )
                frame = raw if tile_vocab is None else tile_vocab.lut[raw]
                base = np.asarray(frame, dtype=np.uint8)

            base_key = "categorical"

        extras: dict[str, Any] = {}
        if self._feature_direction():
            extras["direction"] = np.int64(game.direction_id)

        if self._feature_snake_progress():
            extras["snake_progress"] = np.array(float(game.snake_progress), dtype=np.float32)

        if self._feature_time_since_food():
            extras["time_since_last_food"] = np.array(
                float(game.time_since_food_norm(int(max_steps))),
                dtype=np.float32,
            )

        closest_enabled, metric = self._feature_closest_food()
        if closest_enabled:
            dx_f, dy_f, dist_f = game.get_closest_food_norm(metric)
            extras["closest_food_dx"] = np.array(float(dx_f), dtype=np.float32)
            extras["closest_food_dy"] = np.array(float(dy_f), dtype=np.float32)
            extras["closest_food_dist"] = np.array(float(dist_f), dtype=np.float32)

        if (
            self._feature_collision("collision_ahead")
            or self._feature_collision("collision_left")
            or self._feature_collision("collision_right")
        ):
            c_ahead, c_left, c_right = game.collision_flags()
            if self._feature_collision("collision_ahead"):
                extras["collision_ahead"] = np.array(1.0 if c_ahead else 0.0, dtype=np.float32)
            if self._feature_collision("collision_left"):
                extras["collision_left"] = np.array(1.0 if c_left else 0.0, dtype=np.float32)
            if self._feature_collision("collision_right"):
                extras["collision_right"] = np.array(1.0 if c_right else 0.0, dtype=np.float32)

        if extras:
            return {base_key: base, **extras}
        return base


__all__ = ["ActionSpec", "ObservationSpec"]
