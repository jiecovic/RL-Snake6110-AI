// rust/src/engine/mod.rs

pub mod constants;
pub mod snake_engine;
pub mod geometry;
pub mod board;
pub mod obs_stack;
pub mod tiles;
pub mod tileset;
pub mod vocab;

pub use snake_engine::{EngineError, SnakeEngine};
pub use board::Board;
pub use tileset::{tileset_tile_count, tileset_tile_names, tileset_tile_size, tileset_tiles};
pub use vocab::{
    vocab_class_names,
    vocab_classes,
    vocab_defs,
    vocab_lut,
    vocab_num_classes,
};
