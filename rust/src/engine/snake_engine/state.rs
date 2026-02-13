// rust/src/engine/snake_engine/state.rs

use rand_chacha::ChaCha8Rng;
use std::collections::VecDeque;

use crate::engine::obs::{FrameStacker, HeadStacker};

#[derive(Debug)]
pub enum EngineError {
    InvalidGridLen,
    SpawnFailed,
    DirectionNone,
    PixelGridDisabled,
}

impl std::fmt::Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EngineError::InvalidGridLen => {
                write!(f, "invalid grid dimensions (width and height must be > 0)")
            }
            EngineError::SpawnFailed => write!(f, "could not find a valid snake spawn"),
            EngineError::DirectionNone => write!(f, "direction is None"),
            EngineError::PixelGridDisabled => write!(f, "pixel grid disabled"),
        }
    }
}

impl std::error::Error for EngineError {}

pub struct SnakeEngine {
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) tile_size: usize,
    pub(crate) frame_stack_n: usize,
    pub(crate) step_counter: u64,

    pub(crate) static_grid: Vec<u8>,
    pub(crate) wall_mask: Vec<bool>,
    pub(crate) wall_count: usize,
    pub(crate) spawnable: Vec<usize>,
    pub(crate) spawnable_pos: Vec<usize>,

    pub(crate) spawn_len: usize,
    pub(crate) spawn_dir: i8,
    pub(crate) spawn_random_dir: bool,
    pub(crate) spawn_jitter: i32,

    pub(crate) target_food_count: usize,

    pub(crate) rng: ChaCha8Rng,

    pub(crate) snake: VecDeque<usize>,
    pub(crate) snake_mask: Vec<bool>,
    pub(crate) food: Vec<usize>,
    pub(crate) food_mask: Vec<bool>,
    pub(crate) direction: Option<i8>,

    pub(crate) score: i32,
    pub(crate) running: bool,
    pub(crate) steps_since_food: usize,
    pub(crate) max_steps: Option<usize>,
    pub(crate) initial_snake_len: usize,

    pub(crate) tile_grid: Vec<u8>,
    pub(crate) pixel_grid: Vec<u8>,
    pub(crate) pixel_grid_enabled: bool,
    pub(crate) tile_cache: Vec<Vec<u8>>,

    pub(crate) world_tile_stack: Option<FrameStacker>,
    pub(crate) world_pixel_stack: Option<FrameStacker>,
    pub(crate) head_tile_stack: Option<HeadStacker>,
    pub(crate) head_pixel_stack: Option<HeadStacker>,
    pub(crate) head_pixel_valid_stack: Option<HeadStacker>,
}
