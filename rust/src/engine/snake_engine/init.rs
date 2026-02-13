// rust/src/engine/snake_engine/init.rs

use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::engine::constants::*;
use crate::engine::obs::{FrameStacker, HeadStacker};
use crate::engine::spatial::{dir_turn_left, dir_turn_right, idx};
use crate::engine::tiles::{tileset_tile_size, tileset_tiles};

use super::state::{EngineError, SnakeEngine};

impl SnakeEngine {
    pub fn new(
        width: usize,
        height: usize,
        food_count: usize,
        seed: Option<u64>,
        frame_stack_n: usize,
        enable_pixel_grid: bool,
        enable_world_tile_stack: bool,
        enable_world_pixel_stack: bool,
    ) -> Result<Self, EngineError> {
        if width == 0 || height == 0 {
            return Err(EngineError::InvalidGridLen);
        }

        let tile_size = tileset_tile_size();
        let tile_cache = tileset_tiles();

        let mut static_grid = vec![TILE_EMPTY; width * height];
        for x in 0..width {
            static_grid[idx(x as i32, 0, width)] = TILE_WALL_TOP;
            static_grid[idx(x as i32, (height - 1) as i32, width)] = TILE_WALL_BOTTOM;
        }
        for y in 0..height {
            static_grid[idx(0, y as i32, width)] = TILE_WALL_LEFT;
            static_grid[idx((width - 1) as i32, y as i32, width)] = TILE_WALL_RIGHT;
        }
        static_grid[idx(0, 0, width)] = TILE_WALL_TL;
        static_grid[idx((width - 1) as i32, 0, width)] = TILE_WALL_TR;
        static_grid[idx(0, (height - 1) as i32, width)] = TILE_WALL_BL;
        static_grid[idx((width - 1) as i32, (height - 1) as i32, width)] = TILE_WALL_BR;

        let mut wall_mask = vec![false; width * height];
        let mut wall_count = 0usize;
        for (i, v) in static_grid.iter().enumerate() {
            if *v != TILE_EMPTY {
                wall_mask[i] = true;
                wall_count += 1;
            }
        }

        let rng = match seed {
            Some(s) => ChaCha8Rng::seed_from_u64(s),
            None => ChaCha8Rng::seed_from_u64(rand::thread_rng().next_u64()),
        };

        let n_stack = frame_stack_n.max(1);
        let mut spawnable = Vec::with_capacity(width * height - wall_count);
        let mut spawnable_pos = vec![usize::MAX; width * height];
        for i in 0..(width * height) {
            if wall_mask[i] {
                continue;
            }
            spawnable_pos[i] = spawnable.len();
            spawnable.push(i);
        }

        let pixel_grid = if enable_pixel_grid {
            vec![0u8; width * height * tile_size * tile_size]
        } else {
            Vec::new()
        };

        Ok(Self {
            width,
            height,
            tile_size,
            frame_stack_n: n_stack,
            step_counter: 0,
            static_grid,
            wall_mask,
            wall_count,
            spawnable,
            spawnable_pos,
            spawn_len: 3,
            spawn_dir: 1,
            spawn_random_dir: false,
            spawn_jitter: 0,
            target_food_count: food_count,
            rng,
            snake: Vec::new(),
            snake_mask: vec![false; width * height],
            food: Vec::new(),
            food_mask: vec![false; width * height],
            direction: None,
            score: 0,
            running: true,
            steps_since_food: 0,
            max_steps: None,
            initial_snake_len: 0,
            tile_grid: vec![0u8; width * height],
            pixel_grid,
            pixel_grid_enabled: enable_pixel_grid,
            tile_cache,
            world_tile_stack: if n_stack > 1 && enable_world_tile_stack {
                Some(FrameStacker::new(n_stack))
            } else {
                None
            },
            world_pixel_stack: if n_stack > 1 && enable_pixel_grid && enable_world_pixel_stack {
                Some(FrameStacker::new(n_stack))
            } else {
                None
            },
            head_tile_stack: if n_stack > 1 { Some(HeadStacker::new(n_stack)) } else { None },
            head_pixel_stack: if n_stack > 1 { Some(HeadStacker::new(n_stack)) } else { None },
            head_pixel_valid_stack: if n_stack > 1 { Some(HeadStacker::new(n_stack)) } else { None },
        })
    }

    pub fn reset(&mut self, seed: Option<u64>) -> Result<(), EngineError> {
        if let Some(s) = seed {
            self.rng = ChaCha8Rng::seed_from_u64(s);
        }

        self.score = 0;
        self.running = true;
        self.steps_since_food = 0;
        self.snake.clear();
        self.food.clear();
        self.snake_mask.fill(false);
        self.food_mask.fill(false);
        self.direction = None;
        self.spawnable_reset();

        self.spawn_snake()?;
        self.initial_snake_len = self.snake.len();
        let _ = self.spawn_food();
        self.rebuild_grids();
        self.step_counter = 0;
        self.update_world_stacks(true);
        Ok(())
    }

    pub fn step(&mut self, rel_dir: i32) -> Result<u32, EngineError> {
        self.step_internal(rel_dir)
    }

    pub fn step_cardinal(&mut self, abs_dir: i32) -> Result<u32, EngineError> {
        if !self.running {
            return Ok(MOVE_NOT_RUNNING);
        }

        let dir = match self.direction {
            Some(d) => d,
            None => return Err(EngineError::DirectionNone),
        };

        let desired = ((abs_dir % 4) + 4) % 4;
        let desired = desired as i8;

        // Map absolute direction to a relative action (0=forward,1=left,2=right).
        // Opposite direction is treated as "forward" to avoid 180-degree reversals.
        let rel = if desired == dir {
            0
        } else if desired == dir_turn_left(dir) {
            1
        } else if desired == dir_turn_right(dir) {
            2
        } else {
            0
        };

        self.step_internal(rel)
    }
}
