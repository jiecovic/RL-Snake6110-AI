from enum import Enum
from dataclasses import dataclass


class Direction(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    def turn_left(self) -> 'Direction':
        return Direction((self.value - 1) % 4)

    def turn_right(self) -> 'Direction':
        return Direction((self.value + 1) % 4)


class RelativeDirection(Enum):
    FORWARD = 0
    LEFT = 1
    RIGHT = 2

@dataclass(frozen=True)
class Point:
    x: int
    y: int

    def __add__(self, other: 'Point') -> 'Point':
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other: 'Point') -> 'Point':
        return Point(self.x - other.x, self.y - other.y)

    def __neg__(self) -> 'Point':
        return Point(-self.x, -self.y)

    def manhattan_distance(self, other: 'Point') -> int:
        return abs(self.x - other.x) + abs(self.y - other.y)

DIRECTION_TO_POINT = {
    Direction.UP: Point(0, -1),
    Direction.RIGHT: Point(1, 0),
    Direction.DOWN: Point(0, 1),
    Direction.LEFT: Point(-1, 0),
}
