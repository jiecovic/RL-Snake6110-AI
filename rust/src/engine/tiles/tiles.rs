// rust/src/engine/tiles/tiles.rs
use crate::engine::constants::*;
use crate::engine::spatial::Point;

#[inline]
pub fn head_tile(direction: i8) -> u8 {
    match direction {
        0 => TILE_HEAD_UP,
        1 => TILE_HEAD_RIGHT,
        2 => TILE_HEAD_DOWN,
        3 => TILE_HEAD_LEFT,
        _ => TILE_HEAD_RIGHT,
    }
}

#[inline]
pub fn tail_tile(prev: Point, tail: Point) -> u8 {
    if prev.x < tail.x {
        return TILE_TAIL_RIGHT;
    }
    if prev.x > tail.x {
        return TILE_TAIL_LEFT;
    }
    if prev.y < tail.y {
        return TILE_TAIL_DOWN;
    }
    if prev.y > tail.y {
        return TILE_TAIL_UP;
    }
    TILE_TAIL_RIGHT
}

#[inline]
pub fn body_tile(prev: Point, curr: Point, nxt: Point) -> u8 {
    if prev.x == nxt.x {
        if prev.y < nxt.y {
            return TILE_BODY_VERTICAL_UP;
        }
        if prev.y > nxt.y {
            return TILE_BODY_VERTICAL_DOWN;
        }
    }

    if prev.y == nxt.y {
        if prev.x < nxt.x {
            return TILE_BODY_HORIZONTAL_LEFT;
        }
        if prev.x > nxt.x {
            return TILE_BODY_HORIZONTAL_RIGHT;
        }
    }

    if (prev.x < curr.x && nxt.y > curr.y) || (nxt.x < curr.x && prev.y > curr.y) {
        return TILE_BODY_TR;
    }
    if (prev.x > curr.x && nxt.y > curr.y) || (nxt.x > curr.x && prev.y > curr.y) {
        return TILE_BODY_TL;
    }
    if (prev.x < curr.x && nxt.y < curr.y) || (nxt.x < curr.x && prev.y < curr.y) {
        return TILE_BODY_BR;
    }
    if (prev.x > curr.x && nxt.y < curr.y) || (nxt.x > curr.x && prev.y < curr.y) {
        return TILE_BODY_BL;
    }

    TILE_BODY_HORIZONTAL_RIGHT
}
