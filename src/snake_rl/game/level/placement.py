# src/snake_rl/game/level/placement.py
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from snake_rl.game.geometry import Direction, Point


def parse_direction(v: Any) -> Direction:
    """Parse a direction value from config/YAML/etc."""
    if isinstance(v, Direction):
        return v
    if isinstance(v, str):
        s = v.strip().upper()
        try:
            return Direction[s]
        except KeyError as e:
            raise ValueError(f"Invalid direction '{v}'. Use one of: UP, RIGHT, DOWN, LEFT.") from e
    raise TypeError(f"direction must be a string like 'UP', got: {type(v).__name__}")


def compute_spawn_cells(*, head_pos: Point, length: int, direction: Direction) -> list[Point]:
    """
    Return the list of cells occupied by a straight snake of given length, starting at head_pos,
    extending opposite to movement direction.

    NEW world note:
    - This does NOT write into a level grid anymore; it's just geometry.
    """
    if length < 2:
        raise ValueError("Snake length must be >= 2")

    if direction == Direction.UP:
        dx, dy = 0, -1
    elif direction == Direction.DOWN:
        dx, dy = 0, 1
    elif direction == Direction.LEFT:
        dx, dy = -1, 0
    elif direction == Direction.RIGHT:
        dx, dy = 1, 0
    else:
        raise ValueError(f"Unhandled direction: {direction}")

    # Head is cells[0], then we extend "backwards" from movement direction.
    return [Point(head_pos.x - dx * i, head_pos.y - dy * i) for i in range(length)]


def is_straight_spawn_valid(
        *,
        cells: Iterable[Point],
        width: int,
        height: int,
        wall_positions: set[Point],
        require_interior: bool = True,
) -> bool:
    """
    Validate a candidate straight spawn.

    Requirements:
      - all cells are within bounds
      - none of the cells are walls
      - no duplicates

    If require_interior=True (default):
      - additionally require every cell to be strictly inside the border (not on x=0/x=w-1/y=0/y=h-1)
        This matches the common "walls are border" convention and avoids awkward spawns if the level
        ever has non-border holes.
    """
    seen: set[Point] = set()
    for p in cells:
        if not (0 <= p.x < width and 0 <= p.y < height):
            return False

        if require_interior:
            if p.x == 0 or p.x == width - 1 or p.y == 0 or p.y == height - 1:
                return False

        if p in wall_positions:
            return False
        if p in seen:
            return False
        seen.add(p)

    return True
