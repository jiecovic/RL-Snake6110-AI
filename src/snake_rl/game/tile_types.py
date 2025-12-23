# src/snake_rl/game/tile_types.py
from __future__ import annotations

from enum import Enum

from snake_rl.game.geometry import Direction


class TileType(Enum):
    """
    Canonical tile IDs used everywhere (tile_grid, vocabs, tilesets).

    NEW world note:
    - Glyph/ASCII mapping is NOT defined here anymore (owned by Tileset YAML).
    - This enum is the *single source of truth* for tile identities and stable IDs.
    """

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

    # Snake body (straight segments are direction-aware)
    SNAKE_BODY_VERTICAL_UP = 13
    SNAKE_BODY_VERTICAL_DOWN = 14
    SNAKE_BODY_HORIZONTAL_LEFT = 15
    SNAKE_BODY_HORIZONTAL_RIGHT = 16

    # Snake body corners
    SNAKE_BODY_BR = 17
    SNAKE_BODY_BL = 18
    SNAKE_BODY_TR = 19
    SNAKE_BODY_TL = 20

    # Snake tail
    SNAKE_TAIL_UP = 21
    SNAKE_TAIL_DOWN = 22
    SNAKE_TAIL_LEFT = 23
    SNAKE_TAIL_RIGHT = 24

    # Food
    FOOD = 25


# --- tile groups (fast helpers) ------------------------------------------------

HEAD_TILES = {
    TileType.SNAKE_HEAD_UP,
    TileType.SNAKE_HEAD_DOWN,
    TileType.SNAKE_HEAD_LEFT,
    TileType.SNAKE_HEAD_RIGHT,
}

# Includes straights + corners (but NOT head/tail)
BODY_TILES = {
    TileType.SNAKE_BODY_VERTICAL_UP,
    TileType.SNAKE_BODY_VERTICAL_DOWN,
    TileType.SNAKE_BODY_HORIZONTAL_LEFT,
    TileType.SNAKE_BODY_HORIZONTAL_RIGHT,
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


def direction_from_head_tile(tile: TileType) -> Direction:
    # Kept because it’s genuinely useful in a few places (debug, potential snapshots).
    if tile == TileType.SNAKE_HEAD_UP:
        return Direction.UP
    if tile == TileType.SNAKE_HEAD_RIGHT:
        return Direction.RIGHT
    if tile == TileType.SNAKE_HEAD_DOWN:
        return Direction.DOWN
    if tile == TileType.SNAKE_HEAD_LEFT:
        return Direction.LEFT
    raise ValueError(f"Tile {tile} is not a valid snake head tile")
