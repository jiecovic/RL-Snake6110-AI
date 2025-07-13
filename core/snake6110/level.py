from typing import Any, Dict, List, Tuple, Optional, Set

import json

# Core Enums & Types
from core.snake6110.geometry import Direction
from core.snake6110.tile_types import TileType
from core.snake6110.geometry import Point

# Tile conversion utilities
from core.snake6110.tile_types import CHAR_TO_TILE, TILE_TO_CHAR

# Tile property checks
from core.snake6110.tile_types import (
    direction_from_head_tile,
    is_body,
    is_head,
    is_tail,
)

# Tile group sets
from core.snake6110.tile_types import WALL_TILES


def neighbor_positions(p: Point) -> List[Point]:
    return [
        Point(p.x + 1, p.y),
        Point(p.x - 1, p.y),
        Point(p.x, p.y + 1),
        Point(p.x, p.y - 1)
    ]


def is_adjacent(a: Point, b: Point) -> bool:
    return abs(a.x - b.x) + abs(a.y - b.y) == 1


class BaseLevel:
    def __init__(self, width: int, height: int, grid: List[List[TileType]]):
        self.width = width
        self.height = height
        self.grid = grid  # 2D list of TileType

    def to_dict(self) -> Dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "grid": ["".join(TILE_TO_CHAR.get(cell, "?") for cell in row) for row in self.grid],
        }

    def to_json(self, path: str):
        import json
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    def __str__(self):
        return "\n".join("".join(TILE_TO_CHAR.get(cell, "?") for cell in row) for row in self.grid)

    def validate_snake_placement(self, head_pos: Point, snake_length: int, direction: str = "left"):
        """Ensure the snake fits in the level grid based on head position and direction."""
        if not (1 <= head_pos.x < self.width - 1 and 1 <= head_pos.y < self.height - 1):
            raise ValueError(f"Head position {head_pos} is out of bounds or on border")

        if direction == "left":
            tail_x = head_pos.x - snake_length
            if tail_x < 1:
                raise ValueError(f"Snake length {snake_length} does not fit to the left of head at {head_pos}")
        else:
            raise NotImplementedError(f"Direction '{direction}' not supported yet")

    def get_tile_type(self, pos: Point) -> Optional[TileType]:
        if 0 <= pos.x < self.width and 0 <= pos.y < self.height:
            return self.grid[pos.y][pos.x]  # row-major
        return None

    def get_snake_segments(self) -> List[Point]:
        # Step 1: Find the head
        head: Optional[Point] = None
        for y in range(self.height):
            for x in range(self.width):
                tile = self.grid[y][x]
                if is_head(tile):
                    if head is not None:
                        raise ValueError("Multiple snake heads found")
                    head = Point(x, y)

        if head is None:
            raise ValueError("No snake head found")

        # Step 2: Collect all body and tail points
        body_and_tail: Set[Point] = {
            Point(x, y)
            for y in range(self.height)
            for x in range(self.width)
            if is_body(self.grid[y][x]) or is_tail(self.grid[y][x])
        }

        # Step 3: Reconstruct the path
        segments: List[Point] = [head]
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

    def get_food_positions(self) -> List[Point]:
        return [
            Point(x, y)
            for y, row in enumerate(self.grid)
            for x, tile in enumerate(row)
            if tile == TileType.FOOD
        ]

    def get_wall_tiles(self) -> List[Tuple[Point, TileType]]:
        return [
            (Point(x, y), tile)
            for y, row in enumerate(self.grid)
            for x, tile in enumerate(row)
            if tile in WALL_TILES
        ]

    def get_wall_positions(self) -> List[Tuple[Point, TileType]]:
        return [
            Point(x, y)
            for y, row in enumerate(self.grid)
            for x, tile in enumerate(row)
            if tile in WALL_TILES
        ]


class LevelFromTemplate(BaseLevel):
    def __init__(self, filepath: str = "assets/levels/empty-11x20.json"):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        width = data["width"]
        height = data["height"]
        ascii_grid: List[str] = data["grid"]

        if len(ascii_grid) != height:
            raise ValueError("Grid height does not match 'height'")
        if any(len(row) != width for row in ascii_grid):
            raise ValueError("Grid width mismatch")

        grid = [
            [CHAR_TO_TILE.get(char, TileType.EMPTY) for char in row]
            for row in ascii_grid
        ]

        super().__init__(width, height, grid)


class EmptyLevel(BaseLevel):
    def __init__(
            self,
            height: int,
            width: int,
            head_pos: Optional[Point] = None,
            snake_length: int = 3
    ):
        # Default head in center
        if head_pos is None:
            head_pos = Point(width // 2, height // 2)

        grid = [[TileType.EMPTY for _ in range(width)] for _ in range(height)]

        # Add borders
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

        super().__init__(width, height, grid)
        self.validate_snake_placement(head_pos, snake_length)

        # Place snake
        hx, hy = head_pos.x, head_pos.y
        grid[hy][hx] = TileType.SNAKE_HEAD_RIGHT
        for i in range(1, snake_length):
            grid[hy][hx - i] = TileType.SNAKE_BODY_HORIZONTAL
        grid[hy][hx - snake_length] = TileType.SNAKE_TAIL_LEFT
