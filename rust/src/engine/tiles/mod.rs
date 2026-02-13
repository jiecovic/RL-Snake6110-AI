// rust/src/engine/tiles/mod.rs

mod tileset;
mod tiles;

pub use tiles::{body_tile, head_tile, tail_tile};
pub use tileset::{
    tileset_tile_count,
    tileset_tile_names,
    tileset_tile_names_raw,
    tileset_tile_size,
    tileset_tiles,
};
