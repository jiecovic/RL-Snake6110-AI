# src/snake_rl/game/level/core.py
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from snake_rl.game.geometry import Direction, Point
from snake_rl.game.level.loader import load_level_yaml
from snake_rl.game.level.placement import compute_spawn_cells, place_snake_linear
from snake_rl.game.tile_types import (
    TILE_TO_CHAR,
    TileType,
    WALL_TILES,
    direction_from_head_tile,
    is_body,
    is_head,
    is_tail,
)


def neighbor_positions(p: Point) -> list[Point]:
    return [
        Point(p.x + 1, p.y),
        Point(p.x - 1, p.y),
        Point(p.x, p.y + 1),
        Point(p.x, p.y - 1),
    ]


class BaseLevel:
    def __init__(self, width: int, height: int, grid: list[list[TileType]], init_snake_length: int):
        self.width = width
        self.height = height
        self.grid = grid
        self.init_snake_length = init_snake_length

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "grid": ["".join(TILE_TO_CHAR.get(cell, "?") for cell in row) for row in self.grid],
        }

    def to_yaml(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False, allow_unicode=True)

    def __str__(self) -> str:
        return "\n".join("".join(TILE_TO_CHAR.get(cell, "?") for cell in row) for row in self.grid)

    def get_tile_type(self, pos: Point) -> TileType | None:
        if 0 <= pos.x < self.width and 0 <= pos.y < self.height:
            return self.grid[pos.y][pos.x]
        return None

    def get_snake_segments(self) -> list[Point]:
        head: Point | None = None
        for y in range(self.height):
            for x in range(self.width):
                tile = self.grid[y][x]
                if is_head(tile):
                    if head is not None:
                        raise ValueError("Multiple snake heads found")
                    head = Point(x, y)

        if head is None:
            raise ValueError("No snake head found")

        body_and_tail: set[Point] = {
            Point(x, y)
            for y in range(self.height)
            for x in range(self.width)
            if is_body(self.grid[y][x]) or is_tail(self.grid[y][x])
        }

        segments: list[Point] = [head]
        current = head

        while True:
            neighbors = [p for p in neighbor_positions(current) if p in body_and_tail]
            if not neighbors:
                raise ValueError(f"Snake is disconnected at {current}")
            if len(neighbors) > 1:
                raise ValueError(f"Ambiguous next segment from {current}: {neighbors}")
            next_seg = neighbors[0]
            segments.append(next_seg)
            body_and_tail.remove(next_seg)

            if is_tail(self.grid[next_seg.y][next_seg.x]):
                break
            current = next_seg

        if body_and_tail:
            raise ValueError(f"Unconnected segments left: {body_and_tail}")

        return segments

    def get_snake_direction(self) -> Direction:
        for y in range(self.height):
            for x in range(self.width):
                tile = self.grid[y][x]
                try:
                    return direction_from_head_tile(tile)
                except ValueError:
                    continue
        raise ValueError("No snake head found in grid")

    def get_food_positions(self) -> list[Point]:
        return [
            Point(x, y)
            for y, row in enumerate(self.grid)
            for x, tile in enumerate(row)
            if tile == TileType.FOOD
        ]

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
    def __init__(self, filepath: str | Path = "levels/test_level.yaml"):
        width, height, grid, init_len = load_level_yaml(filepath)
        super().__init__(width, height, grid, init_len)


class EmptyLevel(BaseLevel):
    def __init__(
        self,
        height: int,
        width: int,
        head_pos: Point | None = None,
        snake_length: int = 3,
        direction: Direction = Direction.RIGHT,
    ):
        if head_pos is None:
            head_pos = Point(width // 2, height // 2)

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

        super().__init__(width, height, grid, snake_length)

        if not (1 <= head_pos.x < self.width - 1 and 1 <= head_pos.y < self.height - 1):
            raise ValueError(f"Head position {head_pos} is out of bounds or on border")

        cells = compute_spawn_cells(head_pos=head_pos, length=snake_length, direction=direction)
        for pcell in cells:
            if not (0 <= pcell.x < self.width and 0 <= pcell.y < self.height):
                raise ValueError("Snake does not fit in grid")
            if self.grid[pcell.y][pcell.x] != TileType.EMPTY:
                raise ValueError(f"Snake cell not empty at {(pcell.x, pcell.y)}")

        place_snake_linear(grid=self.grid, head_pos=head_pos, length=snake_length, direction=direction)
