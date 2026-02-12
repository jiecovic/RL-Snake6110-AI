# src/snake_rl/envs/specs.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from gymnasium import spaces

from snake_rl.envs.obs_utils import (
    head_pixel_frame,
    head_tile_frame_with_valid,
    world_pixel_frame,
    world_tile_frame,
)
from snake_rl.envs.view_radius import parse_view_radius
from snake_rl.game.snake_engine import SnakeEngine, tileset_tile_count
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


def _compute_fill(
    *,
    snake_len: int,
    initial_len: int,
    max_playable: int,
    fill_bins: int | None,
) -> np.ndarray:
    denom = max(1, int(max_playable) - int(initial_len))
    x = (int(snake_len) - int(initial_len)) / float(denom)
    x = float(np.clip(x, 0.0, 1.0))

    if fill_bins is None:
        return np.array([x], dtype=np.float32)

    bins = int(fill_bins)
    b = int(np.floor(x * bins))
    b = min(b, bins - 1)
    return np.array(b, dtype=np.int64)


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
        if k in {"tile_id", "tile", "tiles"}:
            return "tile_id"
        raise ValueError(f"Unknown obs.kind={self.kind!r}. Expected pixel|tile_id.")

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

    def _feature_fill(self) -> tuple[bool, int | None]:
        fill = self.features.get("fill", None)
        if isinstance(fill, dict):
            enabled = bool(fill.get("enabled", True))
            bins = fill.get("bins", None)
            return enabled, None if bins is None else int(bins)
        if isinstance(fill, bool):
            return bool(fill), None
        if "fill_bins" in self.params:
            bins = self.params.get("fill_bins")
            return True, None if bins is None else int(bins)
        return False, None

    def frame_stack_key(self) -> str | None:
        if not self.uses_dict():
            return None
        return "pixel" if self._kind() == "pixel" else "tiles"

    def uses_dict(self) -> bool:
        return self._feature_direction() or self._feature_fill()[0]

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
    ) -> spaces.Space:
        kind = self._kind()
        view = self._view()
        base_key = "pixel" if kind == "pixel" else "tiles"

        if kind == "pixel":
            if view == "world":
                ph = int(height) * int(tile_size)
                pw = int(width) * int(tile_size)
                if self._remove_border():
                    ph -= 2 * int(tile_size)
                    pw -= 2 * int(tile_size)
                base_space = spaces.Box(low=0, high=255, shape=(1, ph, pw), dtype=np.uint8)
            else:
                ry, rx = self._view_radius()
                vh = (2 * ry + 1) * int(tile_size)
                vw = (2 * rx + 1) * int(tile_size)
                c = 2 if self._add_oob_mask() else 1
                base_space = spaces.Box(low=0, high=255, shape=(c, vh, vw), dtype=np.uint8)
        else:
            if tile_vocab is not None:
                base_num = int(tile_vocab.num_classes)
            else:
                base_num = int(tileset_tile_count())

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
                    shape=(1, gh, gw),
                    dtype=np.uint8,
                )
            else:
                ry, rx = self._view_radius()
                vy = 2 * ry + 1
                vx = 2 * rx + 1
                num_classes = base_num + (1 if self._tile_mask_oob() else 0)
                base_space = spaces.Box(
                    low=0,
                    high=num_classes - 1,
                    shape=(1, vy, vx),
                    dtype=np.uint8,
                )

        extras: dict[str, spaces.Space] = {}
        if self._feature_direction():
            extras["direction"] = spaces.Discrete(4)
        fill_enabled, fill_bins = self._feature_fill()
        if fill_enabled:
            extras["fill"] = (
                spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
                if fill_bins is None
                else spaces.Discrete(int(fill_bins))
            )

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
    ):
        kind = self._kind()
        view = self._view()

        if kind == "pixel":
            if view == "world":
                frame = world_pixel_frame(
                    pixel_grid=game.pixel_buffer.astype(np.uint8, copy=False),
                    tile_size=int(game.tile_size),
                    remove_border=self._remove_border(),
                )
                base = frame[None, :, :].astype(np.uint8, copy=False)
            else:
                d = game.direction
                assert d is not None, "SnakeEngine.direction is None (did you call game.reset()?)"
                if self._add_oob_mask():
                    frame, valid = head_pixel_frame(
                        pixel_grid=game.pixel_buffer.astype(np.uint8, copy=False),
                        tile_size=int(game.tile_size),
                        head=game.get_head_position(),
                        direction=int(d),
                        view_radius=self._view_radius(),
                        rotate_to_head=self._rotate_to_head(),
                        oob_fill_value=self._pixel_oob_value(),
                        return_valid=True,
                    )
                    mask = np.full(frame.shape, np.uint8(self._mask_oob_value()), dtype=np.uint8)
                    mask[valid] = np.uint8(self._mask_valid_value())
                    base = np.stack([frame.astype(np.uint8, copy=False), mask], axis=0)
                else:
                    frame = head_pixel_frame(
                        pixel_grid=game.pixel_buffer.astype(np.uint8, copy=False),
                        tile_size=int(game.tile_size),
                        head=game.get_head_position(),
                        direction=int(d),
                        view_radius=self._view_radius(),
                        rotate_to_head=self._rotate_to_head(),
                        oob_fill_value=self._pixel_oob_value(),
                        return_valid=False,
                    )
                    base = frame[None, :, :].astype(np.uint8, copy=False)

            base_key = "pixel"
        else:
            grid = game.tile_grid.astype(np.uint8, copy=False)
            if view == "world":
                raw = world_tile_frame(tile_grid=grid, remove_border=self._remove_border())
                frame = raw if tile_vocab is None else tile_vocab.lut[raw]
                base = frame[None, :, :].astype(np.uint8, copy=False)
            else:
                d = game.direction
                assert d is not None, "SnakeEngine.direction is None (did you call game.reset()?)"
                raw, valid = head_tile_frame_with_valid(
                    tile_grid=grid,
                    head=game.get_head_position(),
                    direction=int(d),
                    view_radius=self._view_radius(),
                    rotate_to_head=self._rotate_to_head(),
                )
                frame = raw if tile_vocab is None else tile_vocab.lut[raw]
                if self._tile_mask_oob():
                    out = np.zeros_like(frame, dtype=np.uint8)
                    out[valid] = (frame[valid].astype(np.uint16) + 1).astype(np.uint8, copy=False)
                    base = out[None, :, :].astype(np.uint8, copy=False)
                else:
                    base = frame[None, :, :].astype(np.uint8, copy=False)

            base_key = "tiles"

        extras: dict[str, Any] = {}
        if self._feature_direction():
            d = game.direction
            assert d is not None, "SnakeEngine.direction is None (did you call game.reset()?)"
            extras["direction"] = np.array(int(d), dtype=np.int64)

        fill_enabled, fill_bins = self._feature_fill()
        if fill_enabled:
            extras["fill"] = _compute_fill(
                snake_len=int(game.snake_len),
                initial_len=int(initial_snake_length),
                max_playable=int(max_playable_tiles),
                fill_bins=fill_bins,
            )

        if extras:
            return {base_key: base, **extras}
        return base


__all__ = ["ActionSpec", "ObservationSpec"]
