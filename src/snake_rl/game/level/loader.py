# src/snake_rl/game/level/loader.py
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from snake_rl.game.geometry import Direction, Point
from snake_rl.game.level.placement import compute_spawn_cells, parse_direction, place_snake_linear
from snake_rl.game.tile_types import CHAR_TO_TILE, TileType, is_body, is_head, is_tail
from snake_rl.utils.paths import asset_path


def _has_any_snake_tiles(grid: list[list[TileType]]) -> bool:
    return any(is_head(t) or is_body(t) or is_tail(t) for row in grid for t in row)


def load_level_yaml(path: str | Path) -> tuple[int, int, list[list[TileType]], int]:
    """
    Load a YAML level file.

    Supports two modes:
      1) Fully declarative: grid contains the snake (head/body/tail). Snake length is inferred/validated.
      2) Hybrid: grid contains no snake tiles, and a `spawn:` block specifies x, y, direction, length.

    `path` can be:
      - absolute path
      - or assets-relative like "levels/test_level.yaml"
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

    grid: list[list[TileType]] = [[CHAR_TO_TILE.get(ch, TileType.EMPTY) for ch in row] for row in ascii_grid]

    # Mode A: snake embedded in grid
    if _has_any_snake_tiles(grid):
        # Infer by reconstructing segments (reuse BaseLevel algorithm logic via local reconstruction)
        # We do a light-weight reconstruction here to avoid importing BaseLevel from core.py.
        # If you want: move snake reconstruction to a shared helper later.
        head: Point | None = None
        for y in range(height):
            for x in range(width):
                if is_head(grid[y][x]):
                    if head is not None:
                        raise ValueError("Multiple snake heads found")
                    head = Point(x, y)
        if head is None:
            raise ValueError("No snake head found in grid")

        body_and_tail: set[Point] = {
            Point(x, y)
            for y in range(height)
            for x in range(width)
            if is_body(grid[y][x]) or is_tail(grid[y][x])
        }

        segments: list[Point] = [head]
        current = head

        def neighbors(p0: Point) -> list[Point]:
            return [
                Point(p0.x + 1, p0.y),
                Point(p0.x - 1, p0.y),
                Point(p0.x, p0.y + 1),
                Point(p0.x, p0.y - 1),
            ]

        while True:
            nxt = [p1 for p1 in neighbors(current) if p1 in body_and_tail]
            if not nxt:
                raise ValueError(f"Snake is disconnected at {current}")
            if len(nxt) > 1:
                raise ValueError(f"Ambiguous next segment from {current}: {nxt}")
            next_seg = nxt[0]
            segments.append(next_seg)
            body_and_tail.remove(next_seg)

            if is_tail(grid[next_seg.y][next_seg.x]):
                break
            current = next_seg

        if body_and_tail:
            raise ValueError(f"Unconnected segments left: {body_and_tail}")

        return width, height, grid, len(segments)

    # Mode B: spawn block required
    spawn = data.get("spawn")
    if not isinstance(spawn, dict):
        raise ValueError(
            "Level grid contains no snake tiles. Provide a 'spawn:' mapping with x, y, direction, length."
        )

    x = int(spawn["x"])
    y = int(spawn["y"])
    length = int(spawn.get("length", 3))
    direction: Direction = parse_direction(spawn.get("direction", "RIGHT"))

    head_pos = Point(x, y)

    # Border check
    if not (1 <= head_pos.x < width - 1 and 1 <= head_pos.y < height - 1):
        raise ValueError(f"Spawn head position {head_pos} is out of bounds or on border")

    cells = compute_spawn_cells(head_pos=head_pos, length=length, direction=direction)
    for pcell in cells:
        if not (0 <= pcell.x < width and 0 <= pcell.y < height):
            raise ValueError("Spawn snake does not fit in grid")
        if grid[pcell.y][pcell.x] != TileType.EMPTY:
            raise ValueError(f"Spawn cell not empty at {(pcell.x, pcell.y)} (tile={grid[pcell.y][pcell.x]})")

    place_snake_linear(grid=grid, head_pos=head_pos, length=length, direction=direction)
    return width, height, grid, length
