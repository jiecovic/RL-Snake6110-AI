from enum import Enum
from typing import Dict
from snake_rl.game.geometry import Direction


class TileType(Enum):
    EMPTY = 0

    # Walls
    WALL_TL = 1
    WALL_TR = 2
    WALL_BL = 3
    WALL_BR = 4
    WALL_TOP = 5
    WALL_BOTTOM = 6
    WALL_LEFT = 7
    WALL_RIGHT = 8

    # Snake head
    SNAKE_HEAD_UP = 9
    SNAKE_HEAD_DOWN = 10
    SNAKE_HEAD_LEFT = 11
    SNAKE_HEAD_RIGHT = 12

    # Snake body
    SNAKE_BODY_VERTICAL = 13
    SNAKE_BODY_HORIZONTAL = 14
    SNAKE_BODY_BR = 15
    SNAKE_BODY_BL = 16
    SNAKE_BODY_TR = 17
    SNAKE_BODY_TL = 18

    # Snake tail
    SNAKE_TAIL_UP = 19
    SNAKE_TAIL_DOWN = 20
    SNAKE_TAIL_LEFT = 21
    SNAKE_TAIL_RIGHT = 22

    # Food
    FOOD = 23

    # Trackers
    TRACKER_VISITED = 24
    TRACKER_PATH = 25


HEAD_TILES = {
    TileType.SNAKE_HEAD_UP,
    TileType.SNAKE_HEAD_DOWN,
    TileType.SNAKE_HEAD_LEFT,
    TileType.SNAKE_HEAD_RIGHT,
}

BODY_TILES = {
    TileType.SNAKE_BODY_VERTICAL,
    TileType.SNAKE_BODY_HORIZONTAL,
    TileType.SNAKE_BODY_BR,
    TileType.SNAKE_BODY_BL,
    TileType.SNAKE_BODY_TR,
    TileType.SNAKE_BODY_TL,
}

TAIL_TILES = {
    TileType.SNAKE_TAIL_UP,
    TileType.SNAKE_TAIL_DOWN,
    TileType.SNAKE_TAIL_LEFT,
    TileType.SNAKE_TAIL_RIGHT,
}

WALL_TILES = {
    TileType.WALL_TL,
    TileType.WALL_TR,
    TileType.WALL_BL,
    TileType.WALL_BR,
    TileType.WALL_TOP,
    TileType.WALL_BOTTOM,
    TileType.WALL_LEFT,
    TileType.WALL_RIGHT,
}


def is_head(tile: TileType) -> bool:
    return tile in HEAD_TILES


def is_body(tile: TileType) -> bool:
    return tile in BODY_TILES


def is_tail(tile: TileType) -> bool:
    return tile in TAIL_TILES


def is_wall(tile: TileType) -> bool:
    return tile in WALL_TILES


def is_food(tile: TileType) -> bool:
    return tile == TileType.FOOD


# ASCII character to TileType mapping
CHAR_TO_TILE: Dict[str, TileType] = {
    ' ': TileType.EMPTY,

    # Walls
    '┌': TileType.WALL_TL,
    '┐': TileType.WALL_TR,
    '└': TileType.WALL_BL,
    '┘': TileType.WALL_BR,
    '─': TileType.WALL_TOP,
    '=': TileType.WALL_BOTTOM,
    '│': TileType.WALL_LEFT,
    '¦': TileType.WALL_RIGHT,

    # Snake head
    '▲': TileType.SNAKE_HEAD_UP,
    '▼': TileType.SNAKE_HEAD_DOWN,
    '◀': TileType.SNAKE_HEAD_LEFT,
    '▶': TileType.SNAKE_HEAD_RIGHT,

    # Snake body
    '║': TileType.SNAKE_BODY_VERTICAL,
    '═': TileType.SNAKE_BODY_HORIZONTAL,
    '╝': TileType.SNAKE_BODY_BR,
    '╚': TileType.SNAKE_BODY_BL,
    '╗': TileType.SNAKE_BODY_TR,
    '╔': TileType.SNAKE_BODY_TL,

    # Snake tail
    '╻': TileType.SNAKE_TAIL_UP,
    '╹': TileType.SNAKE_TAIL_DOWN,
    '╺': TileType.SNAKE_TAIL_LEFT,
    '╸': TileType.SNAKE_TAIL_RIGHT,

    # Food
    '*': TileType.FOOD,
}

# Optional reverse mapping for serialization or debugging
TILE_TO_CHAR: Dict[TileType, str] = {v: k for k, v in CHAR_TO_TILE.items()}


def direction_from_head_tile(tile: TileType) -> Direction:
    if tile == TileType.SNAKE_HEAD_UP:
        return Direction.UP
    elif tile == TileType.SNAKE_HEAD_RIGHT:
        return Direction.RIGHT
    elif tile == TileType.SNAKE_HEAD_DOWN:
        return Direction.DOWN
    elif tile == TileType.SNAKE_HEAD_LEFT:
        return Direction.LEFT
    else:
        raise ValueError(f"Tile {tile} is not a valid snake head tile")
