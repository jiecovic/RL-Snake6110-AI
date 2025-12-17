# src/snake_rl/game/level/placement.py
from __future__ import annotations

from typing import Any

from snake_rl.game.geometry import Direction, Point
from snake_rl.game.tile_types import TileType


def parse_direction(v: Any) -> Direction:
    if isinstance(v, Direction):
        return v
    if isinstance(v, str):
        s = v.strip().upper()
        try:
            return Direction[s]
        except KeyError as e:
            raise ValueError(
                f"Invalid direction '{v}'. Use one of: UP, RIGHT, DOWN, LEFT."
            ) from e
    raise TypeError(f"direction must be a string like 'UP', got: {type(v).__name__}")


def compute_spawn_cells(*, head_pos: Point, length: int, direction: Direction) -> list[Point]:
    """
    Return the list of cells occupied by a straight snake of given length, starting at head_pos,
    extending opposite to movement direction.
    """
    if length < 2:
        raise ValueError("Snake length must be >= 2")

    dx, dy = 0, 0
    if direction == Direction.UP:
        dx, dy = 0, -1
    elif direction == Direction.DOWN:
        dx, dy = 0, 1
    elif direction == Direction.LEFT:
        dx, dy = -1, 0
    elif direction == Direction.RIGHT:
        dx, dy = 1, 0

    return [Point(head_pos.x - dx * i, head_pos.y - dy * i) for i in range(length)]


def place_snake_linear(
        *,
        grid: list[list[TileType]],
        head_pos: Point,
        length: int,
        direction: Direction,
) -> None:
    """
    Place a straight snake into an existing grid.

    Assumes:
      - caller validated bounds
      - caller validated destination cells are empty
    """
    if length < 2:
        raise ValueError("Snake length must be >= 2")

    if direction == Direction.UP:
        head_tile = TileType.SNAKE_HEAD_UP
        body_tile = TileType.SNAKE_BODY_VERTICAL
        tail_tile = TileType.SNAKE_TAIL_DOWN
    elif direction == Direction.DOWN:
        head_tile = TileType.SNAKE_HEAD_DOWN
        body_tile = TileType.SNAKE_BODY_VERTICAL
        tail_tile = TileType.SNAKE_TAIL_UP
    elif direction == Direction.LEFT:
        head_tile = TileType.SNAKE_HEAD_LEFT
        body_tile = TileType.SNAKE_BODY_HORIZONTAL
        tail_tile = TileType.SNAKE_TAIL_RIGHT
    elif direction == Direction.RIGHT:
        head_tile = TileType.SNAKE_HEAD_RIGHT
        body_tile = TileType.SNAKE_BODY_HORIZONTAL
        tail_tile = TileType.SNAKE_TAIL_LEFT
    else:
        raise ValueError(f"Unhandled direction: {direction}")

    cells = compute_spawn_cells(head_pos=head_pos, length=length, direction=direction)

    # Head
    grid[cells[0].y][cells[0].x] = head_tile

    # Body + tail
    for i, p in enumerate(cells[1:], start=1):
        grid[p.y][p.x] = tail_tile if i == length - 1 else body_tile
