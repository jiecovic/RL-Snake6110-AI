// rust/src/engine/tileset.rs
const TILE_SIZE: usize = 4;
const TILE_COUNT: usize = 26;
const TILESET: [[u8; 16]; TILE_COUNT] = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], // 0 empty
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255], // 1 wall_tl
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 0, 0], // 2 wall_tr
    [0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0], // 3 wall_bl
    [0, 255, 0, 0, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], // 4 wall_br
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255], // 5 wall_top
    [0, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0], // 6 wall_bottom
    [0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255], // 7 wall_left
    [0, 255, 0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255, 0, 0], // 8 wall_right
    [
        0, 0, 0, 0, 0, 255, 255, 255, 0, 255, 0, 255, 0, 255, 255, 255,
    ], // 9 snake_head_up
    [
        0, 255, 255, 255, 0, 255, 255, 255, 0, 255, 0, 255, 0, 255, 255, 255,
    ], // 10 snake_head_down
    [
        0, 0, 0, 0, 0, 255, 255, 255, 0, 255, 0, 255, 0, 255, 255, 255,
    ], // 11 snake_head_left
    [
        0, 0, 0, 0, 255, 255, 255, 255, 255, 255, 0, 255, 255, 255, 255, 255,
    ], // 12 snake_head_right
    [
        0, 255, 255, 255, 0, 255, 255, 255, 0, 255, 255, 255, 0, 255, 255, 255,
    ], // 13 snake_body_vertical_up
    [
        0, 255, 255, 255, 0, 255, 255, 255, 0, 255, 255, 255, 0, 255, 255, 255,
    ], // 14 snake_body_vertical_down
    [
        0, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    ], // 15 snake_body_horizontal_left
    [
        0, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    ], // 16 snake_body_horizontal_right
    [
        0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    ], // 17 snake_body_br
    [
        0, 255, 255, 255, 0, 255, 255, 255, 0, 255, 255, 255, 0, 255, 255, 255,
    ], // 18 snake_body_bl
    [
        0, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    ], // 19 snake_body_tr
    [
        0, 0, 0, 0, 0, 255, 255, 255, 0, 255, 255, 255, 0, 255, 255, 255,
    ], // 20 snake_body_tl
    [
        0, 0, 0, 0, 0, 255, 255, 255, 0, 255, 255, 255, 0, 255, 255, 255,
    ], // 21 snake_tail_up
    [
        0, 255, 255, 255, 0, 255, 255, 255, 0, 255, 255, 255, 0, 0, 0, 0,
    ], // 22 snake_tail_down
    [
        0, 0, 0, 0, 0, 255, 255, 255, 0, 255, 255, 255, 0, 255, 255, 255,
    ], // 23 snake_tail_left
    [
        0, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    ], // 24 snake_tail_right
    [0, 0, 0, 0, 0, 0, 255, 0, 0, 255, 0, 255, 0, 0, 255, 0], // 25 food
];
const TILE_NAMES: [&str; TILE_COUNT] = [
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

pub fn tileset_tiles() -> Vec<Vec<u8>> {
    TILESET.iter().map(|t| t.to_vec()).collect()
}
