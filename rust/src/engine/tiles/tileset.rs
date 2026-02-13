// rust/src/engine/tiles/tileset.rs
const TILE_SIZE: usize = 4;
const TILE_COUNT: usize = 27;
// Use O/X to keep the 2D tile literals aligned and readable.
const O: u8 = 0;
const X: u8 = 255;
// Keep tiles human-readable as 4x4 grids; flatten() runs at compile time.
const fn flatten(tile: [[u8; TILE_SIZE]; TILE_SIZE]) -> [u8; 16] {
    let mut out = [O; 16];
    let mut y = 0usize;
    while y < TILE_SIZE {
        let mut x = 0usize;
        while x < TILE_SIZE {
            out[y * TILE_SIZE + x] = tile[y][x];
            x += 1;
        }
        y += 1;
    }
    out
}

const TILESET: [[u8; 16]; TILE_COUNT] = [
    flatten([
        [X, O, O, X],
        [O, X, X, O],
        [O, X, X, O],
        [X, O, O, X],
    ]), // 0 oob
    flatten([
        [O, O, O, O],
        [O, O, O, O],
        [O, O, O, O],
        [O, O, O, O],
    ]), // 1 empty
    flatten([
        [O, O, O, O],
        [O, O, O, O],
        [O, O, O, O],
        [O, O, O, X],
    ]), // 2 wall_tl
    flatten([
        [O, O, O, O],
        [O, O, O, O],
        [O, O, O, O],
        [X, X, O, O],
    ]), // 3 wall_tr
    flatten([
        [O, O, O, X],
        [O, O, O, X],
        [O, O, O, O],
        [O, O, O, O],
    ]), // 4 wall_bl
    flatten([
        [O, X, O, O],
        [X, X, O, O],
        [O, O, O, O],
        [O, O, O, O],
    ]), // 5 wall_br
    flatten([
        [O, O, O, O],
        [O, O, O, O],
        [O, O, O, O],
        [X, X, X, X],
    ]), // 6 wall_top
    flatten([
        [O, O, O, O],
        [X, X, X, X],
        [O, O, O, O],
        [O, O, O, O],
    ]), // 7 wall_bottom
    flatten([
        [O, O, O, X],
        [O, O, O, X],
        [O, O, O, X],
        [O, O, O, X],
    ]), // 8 wall_left
    flatten([
        [O, X, O, O],
        [O, X, O, O],
        [O, X, O, O],
        [O, X, O, O],
    ]), // 9 wall_right
    flatten([
        [O, O, O, O],
        [O, X, X, X],
        [O, X, O, X],
        [O, X, X, X],
    ]), // 10 snake_head_up
    flatten([
        [O, X, X, X],
        [O, X, X, X],
        [O, X, O, X],
        [O, X, X, X],
    ]), // 11 snake_head_down
    flatten([
        [O, O, O, O],
        [O, X, X, X],
        [O, X, O, X],
        [O, X, X, X],
    ]), // 12 snake_head_left
    flatten([
        [O, O, O, O],
        [X, X, X, X],
        [X, X, O, X],
        [X, X, X, X],
    ]), // 13 snake_head_right
    flatten([
        [O, X, X, X],
        [O, X, X, X],
        [O, X, X, X],
        [O, X, X, X],
    ]), // 14 snake_body_vertical_up
    flatten([
        [O, X, X, X],
        [O, X, X, X],
        [O, X, X, X],
        [O, X, X, X],
    ]), // 15 snake_body_vertical_down
    flatten([
        [O, O, O, O],
        [X, X, X, X],
        [X, X, X, X],
        [X, X, X, X],
    ]), // 16 snake_body_horizontal_left
    flatten([
        [O, O, O, O],
        [X, X, X, X],
        [X, X, X, X],
        [X, X, X, X],
    ]), // 17 snake_body_horizontal_right
    flatten([
        [O, X, X, X],
        [X, X, X, X],
        [X, X, X, X],
        [X, X, X, X],
    ]), // 18 snake_body_br
    flatten([
        [O, X, X, X],
        [O, X, X, X],
        [O, X, X, X],
        [O, X, X, X],
    ]), // 19 snake_body_bl
    flatten([
        [O, O, O, O],
        [X, X, X, X],
        [X, X, X, X],
        [X, X, X, X],
    ]), // 20 snake_body_tr
    flatten([
        [O, O, O, O],
        [O, X, X, X],
        [O, X, X, X],
        [O, X, X, X],
    ]), // 21 snake_body_tl
    flatten([
        [O, O, O, O],
        [O, X, X, X],
        [O, X, X, X],
        [O, X, X, X],
    ]), // 22 snake_tail_up
    flatten([
        [O, X, X, X],
        [O, X, X, X],
        [O, X, X, X],
        [O, O, O, O],
    ]), // 23 snake_tail_down
    flatten([
        [O, O, O, O],
        [O, X, X, X],
        [O, X, X, X],
        [O, X, X, X],
    ]), // 24 snake_tail_left
    flatten([
        [O, O, O, O],
        [X, X, X, X],
        [X, X, X, X],
        [X, X, X, X],
    ]), // 25 snake_tail_right
    flatten([
        [O, O, O, O],
        [O, O, X, O],
        [O, X, O, X],
        [O, O, X, O],
    ]), // 26 food
];
const TILE_NAMES: [&str; TILE_COUNT] = [
    "OOB",
    "EMPTY",
    "WALL_TL",
    "WALL_TR",
    "WALL_BL",
    "WALL_BR",
    "WALL_TOP",
    "WALL_BOTTOM",
    "WALL_LEFT",
    "WALL_RIGHT",
    "SNAKE_HEAD_UP",
    "SNAKE_HEAD_DOWN",
    "SNAKE_HEAD_LEFT",
    "SNAKE_HEAD_RIGHT",
    "SNAKE_BODY_VERTICAL_UP",
    "SNAKE_BODY_VERTICAL_DOWN",
    "SNAKE_BODY_HORIZONTAL_LEFT",
    "SNAKE_BODY_HORIZONTAL_RIGHT",
    "SNAKE_BODY_BR",
    "SNAKE_BODY_BL",
    "SNAKE_BODY_TR",
    "SNAKE_BODY_TL",
    "SNAKE_TAIL_UP",
    "SNAKE_TAIL_DOWN",
    "SNAKE_TAIL_LEFT",
    "SNAKE_TAIL_RIGHT",
    "FOOD",
];

pub fn tileset_tile_size() -> usize {
    TILE_SIZE
}

pub fn tileset_tile_count() -> usize {
    TILE_COUNT
}

pub fn tileset_tile_names() -> Vec<String> {
    TILE_NAMES.iter().map(|n| n.to_string()).collect()
}

pub fn tileset_tile_names_raw() -> &'static [&'static str] {
    &TILE_NAMES
}

pub fn tileset_tiles() -> Vec<Vec<u8>> {
    TILESET.iter().map(|t| t.to_vec()).collect()
}
