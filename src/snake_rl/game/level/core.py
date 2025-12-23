# src/snake_rl/game/level/core.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml

from snake_rl.game.geometry import Point
from snake_rl.game.level.loader import SpawnSpec, load_level_yaml
from snake_rl.game.tile_types import TileType, WALL_TILES
from snake_rl.game.tileset import Tileset


class BaseLevel:
    """
    NEW world level container (static only).

    - `grid` is STATIC ONLY: EMPTY + WALL_*.
    - Snake + food are runtime state (SnakeGame owns them).
    - `spawn` describes how SnakeGame should initialize the snake.
    """

    def __init__(self, width: int, height: int, grid: list[list[TileType]], spawn: SpawnSpec):
        self.width = width
        self.height = height
        self.grid = grid
        self.spawn = spawn

    def to_dict(self, *, tileset: Optional[Tileset] = None) -> dict[str, Any]:
        ts = tileset or Tileset()

        def cell_to_glyph(cell: TileType) -> str:
            g = ts.glyph_for(cell)
            return g if g is not None else "?"

        # Keep spawn in serialization (this is now part of the level definition)
        return {
            "width": self.width,
            "height": self.height,
            "grid": ["".join(cell_to_glyph(cell) for cell in row) for row in self.grid],
            "spawn": {
                "x": self.spawn.x,
                "y": self.spawn.y,
                "length": self.spawn.length,
                "direction": self.spawn.direction.name if self.spawn.direction is not None else None,
                "random_direction": self.spawn.random_direction,
                "jitter": self.spawn.jitter,
            },
        }

    def to_yaml(self, path: str | Path, *, tileset: Optional[Tileset] = None) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(tileset=tileset), f, sort_keys=False, allow_unicode=True)

    def __str__(self) -> str:
        # Best-effort debug printing using default tileset.
        ts = Tileset()

        def cell_to_glyph(cell: TileType) -> str:
            g = ts.glyph_for(cell)
            return g if g is not None else "?"

        return "\n".join("".join(cell_to_glyph(cell) for cell in row) for row in self.grid)

    def get_tile_type(self, pos: Point) -> TileType | None:
        if 0 <= pos.x < self.width and 0 <= pos.y < self.height:
            return self.grid[pos.y][pos.x]
        return None

    def get_wall_tiles(self) -> list[tuple[Point, TileType]]:
        return [
            (Point(x, y), tile)
            for y, row in enumerate(self.grid)
            for x, tile in enumerate(row)
            if tile in WALL_TILES
        ]

    def get_wall_positions(self) -> list[Point]:
        return [
            Point(x, y)
            for y, row in enumerate(self.grid)
            for x, tile in enumerate(row)
            if tile in WALL_TILES
        ]


class TemplateLevel(BaseLevel):
    def __init__(self, filepath: str | Path = "levels/test_level.yaml", *, tileset: Optional[Tileset] = None):
        width, height, grid, spawn = load_level_yaml(filepath, tileset=tileset)
        super().__init__(width, height, grid, spawn)


class EmptyLevel(BaseLevel):
    """
    Programmatic static level generator (walls + empty).

    NEW world:
    - Does NOT paint snake/food into the level grid.
    - Spawn is stored in `self.spawn` for SnakeGame.reset() to consume.
    """

    def __init__(
            self,
            height: int,
            width: int,
            *,
            spawn_x: int | None = None,
            spawn_y: int | None = None,
            spawn_length: int = 3,
            spawn_direction: str | None = "RIGHT",
            random_direction: bool = False,
            jitter: int = 0,
    ):
        grid: list[list[TileType]] = [[TileType.EMPTY for _ in range(width)] for _ in range(height)]

        for x in range(width):
            grid[0][x] = TileType.WALL_TOP
            grid[height - 1][x] = TileType.WALL_BOTTOM
        for y in range(height):
            grid[y][0] = TileType.WALL_LEFT
            grid[y][width - 1] = TileType.WALL_RIGHT

        grid[0][0] = TileType.WALL_TL
        grid[0][width - 1] = TileType.WALL_TR
        grid[height - 1][0] = TileType.WALL_BL
        grid[height - 1][width - 1] = TileType.WALL_BR

        # SpawnSpec wants a Direction or None; we keep the string here to avoid import churn.
        # SnakeGame can also accept None and choose.
        from snake_rl.game.level.placement import parse_direction  # local import to avoid cycles

        direction = parse_direction(spawn_direction) if spawn_direction is not None else None

        spawn = SpawnSpec(
            x=spawn_x,
            y=spawn_y,
            length=int(spawn_length),
            direction=direction,
            random_direction=bool(random_direction),
            jitter=int(jitter),
        )

        super().__init__(width, height, grid, spawn)
