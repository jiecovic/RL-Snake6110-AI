# src/snake_rl/game/level/loader.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml

from snake_rl.game.geometry import Direction
from snake_rl.game.level.placement import parse_direction
from snake_rl.game.tile_types import TileType, WALL_TILES, is_body, is_head, is_tail
from snake_rl.game.tileset import Tileset
from snake_rl.utils.paths import asset_path


@dataclass(frozen=True, slots=True)
class SpawnSpec:
    """
    Spawn configuration for NEW world (dynamic snake state, static level).

    Semantics:
      - If x/y is None -> snakegame chooses a valid position (e.g. center or random valid).
      - If direction is None -> snakegame chooses direction (random if random_direction else default RIGHT).
      - jitter: optional random perturbation radius (Manhattan/box is up to SnakeGame; we just carry the number).
    """
    x: Optional[int]
    y: Optional[int]
    length: int
    direction: Optional[Direction]
    random_direction: bool
    jitter: int


def _has_any_dynamic_tiles(grid: list[list[TileType]]) -> bool:
    # NEW world: snake + food are runtime-only, therefore forbidden in level grids.
    return any(is_head(t) or is_body(t) or is_tail(t) or (t == TileType.FOOD) for row in grid for t in row)


def _is_static_level_tile(t: TileType) -> bool:
    return (t == TileType.EMPTY) or (t in WALL_TILES)


def load_level_yaml(
        path: str | Path,
        *,
        tileset: Tileset | None = None,
) -> tuple[int, int, list[list[TileType]], SpawnSpec]:
    """
    Load a YAML level file (NEW world only).

    Contract (NEW world):
      - The grid is STATIC ONLY: EMPTY + WALL_* tiles.
      - Snake + food are NOT allowed in the grid.
      - A `spawn:` block is REQUIRED and describes how SnakeGame should initialize.

    Level YAML example:

      width: 12
      height: 7
      grid:
        - "┌──────────┐"
        - "│          │"
        - "│          │"
        - "│          │"
        - "│          │"
        - "│          │"
        - "└==========┘"
      spawn:
        x: 6            # optional (omit for auto)
        y: 3            # optional
        length: 3       # optional, default 3
        direction: RIGHT  # optional (omit for auto)
        random_direction: false  # optional
        jitter: 0       # optional

    Notes:
      - glyph -> TileType mapping is owned by Tileset YAML.
    """
    p = Path(path)
    if not p.is_file():
        p = asset_path(str(path))
    if not p.is_file():
        raise FileNotFoundError(f"Level file not found: {path!r} (resolved to {p})")

    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping in YAML level file, got: {type(data).__name__}")

    width = int(data["width"])
    height = int(data["height"])
    ascii_grid: list[str] = data["grid"]

    if len(ascii_grid) != height:
        raise ValueError("Grid height does not match 'height'")
    if any(len(row) != width for row in ascii_grid):
        raise ValueError("Grid width mismatch")

    ts = tileset or Tileset()

    grid: list[list[TileType]] = []
    for row in ascii_grid:
        out_row: list[TileType] = []
        for ch in row:
            t = ts.tile_for_glyph(ch)
            out_row.append(t if t is not None else TileType.EMPTY)
        grid.append(out_row)

    # NEW world enforcement: no snake/food tiles in level grid
    if _has_any_dynamic_tiles(grid):
        raise ValueError(
            "Level grid contains dynamic tiles (snake and/or food). "
            "NEW world requires static-only grids (EMPTY + WALL_*). "
            "Move snake/food initialization to the 'spawn:' block / SnakeGame."
        )

    # Also ensure no weird non-wall tiles sneak in (future enums etc.)
    bad: list[TileType] = []
    for row in grid:
        for t in row:
            if not _is_static_level_tile(t):
                bad.append(t)
    if bad:
        uniq = ", ".join(sorted({t.name for t in bad}))
        raise ValueError(f"Level grid contains non-static tiles: {uniq}. Allowed: EMPTY + WALL_* only.")

    spawn = data.get("spawn")
    if not isinstance(spawn, dict):
        raise ValueError("Missing required 'spawn:' mapping in level YAML (NEW world).")

    # x/y optional (None means auto)
    x_v = spawn.get("x", None)
    y_v = spawn.get("y", None)
    x = int(x_v) if x_v is not None else None
    y = int(y_v) if y_v is not None else None

    length = int(spawn.get("length", 3))
    if length < 2:
        raise ValueError("spawn.length must be >= 2")

    random_direction = bool(spawn.get("random_direction", False))

    dir_v = spawn.get("direction", None)
    direction: Optional[Direction]
    if dir_v is None:
        direction = None
    else:
        direction = parse_direction(dir_v)

    jitter = int(spawn.get("jitter", 0))
    if jitter < 0:
        raise ValueError("spawn.jitter must be >= 0")

    spawn_spec = SpawnSpec(
        x=x,
        y=y,
        length=length,
        direction=direction,
        random_direction=random_direction,
        jitter=jitter,
    )

    return width, height, grid, spawn_spec
