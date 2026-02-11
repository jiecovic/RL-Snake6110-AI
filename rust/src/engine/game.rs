// rust\src\engine\game.rs
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use super::constants::*;
use super::geometry::{
    Point, compute_spawn_cells, dir_turn_left, dir_turn_right, dir_vec, idx, idx_to_point,
    is_straight_spawn_valid,
};
use super::tiles::{body_tile, head_tile, tail_tile};

#[derive(Debug)]
pub enum EngineError {
    InvalidGridLen,
    InvalidTileSize,
    InvalidTileCache,
    SpawnFailed,
    DirectionNone,
}

impl std::fmt::Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EngineError::InvalidGridLen => {
                write!(f, "static_grid length does not match width*height")
            }
            EngineError::InvalidTileSize => write!(f, "tile_size must be > 0"),
            EngineError::InvalidTileCache => write!(f, "tile_cache is invalid or too small"),
            EngineError::SpawnFailed => write!(f, "could not find a valid snake spawn"),
            EngineError::DirectionNone => write!(f, "direction is None"),
        }
    }
}

impl std::error::Error for EngineError {}

pub struct GameCore {
    width: usize,
    height: usize,
    tile_size: usize,

    static_grid: Vec<u8>,
    wall_mask: Vec<bool>,
    wall_count: usize,

    spawn_x: Option<i32>,
    spawn_y: Option<i32>,
    spawn_len: usize,
    spawn_dir: Option<i8>,
    spawn_random_dir: bool,
    spawn_jitter: i32,

    target_food_count: usize,

    rng: ChaCha8Rng,

    snake: Vec<usize>,
    snake_mask: Vec<bool>,
    food: Vec<usize>,
    food_mask: Vec<bool>,
    direction: Option<i8>,

    score: i32,
    running: bool,

    tile_grid: Vec<u8>,
    pixel_grid: Vec<u8>,
    tile_cache: Vec<Vec<u8>>,
}

impl GameCore {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        width: usize,
        height: usize,
        static_grid: Vec<u8>,
        spawn_x: Option<i32>,
        spawn_y: Option<i32>,
        spawn_len: usize,
        spawn_dir: Option<i8>,
        spawn_random_dir: bool,
        spawn_jitter: i32,
        food_count: usize,
        tile_size: usize,
        tile_cache: Vec<Vec<u8>>,
        seed: Option<u64>,
    ) -> Result<Self, EngineError> {
        if width == 0 || height == 0 {
            return Err(EngineError::InvalidGridLen);
        }
        if static_grid.len() != width * height {
            return Err(EngineError::InvalidGridLen);
        }
        if tile_size == 0 {
            return Err(EngineError::InvalidTileSize);
        }
        if tile_cache.is_empty() || tile_cache.len() <= TILE_FOOD as usize {
            return Err(EngineError::InvalidTileCache);
        }
        let tile_len = tile_size * tile_size;
        if tile_cache.iter().any(|t| t.len() != tile_len) {
            return Err(EngineError::InvalidTileCache);
        }

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
            None => ChaCha8Rng::seed_from_u64(0),
        };

        Ok(Self {
            width,
            height,
            tile_size,
            static_grid,
            wall_mask,
            wall_count,
            spawn_x,
            spawn_y,
            spawn_len,
            spawn_dir,
            spawn_random_dir,
            spawn_jitter,
            target_food_count: food_count,
            rng,
            snake: Vec::new(),
            snake_mask: vec![false; width * height],
            food: Vec::new(),
            food_mask: vec![false; width * height],
            direction: None,
            score: 0,
            running: true,
            tile_grid: vec![0u8; width * height],
            pixel_grid: vec![0u8; width * height * tile_size * tile_size],
            tile_cache,
        })
    }

    pub fn reset(&mut self, seed: Option<u64>) -> Result<(), EngineError> {
        if let Some(s) = seed {
            self.rng = ChaCha8Rng::seed_from_u64(s);
        }

        self.score = 0;
        self.running = true;
        self.snake.clear();
        self.food.clear();
        self.snake_mask.fill(false);
        self.food_mask.fill(false);
        self.direction = None;

        self.spawn_snake()?;
        self.spawn_food();
        self.rebuild_grids();
        Ok(())
    }

    pub fn step(&mut self, rel_dir: i32) -> Result<u32, EngineError> {
        self.step_internal(rel_dir)
    }

    pub fn tile_grid(&self) -> &[u8] {
        &self.tile_grid
    }

    pub fn pixel_grid(&self) -> &[u8] {
        &self.pixel_grid
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn tile_size(&self) -> usize {
        self.tile_size
    }

    pub fn score(&self) -> i32 {
        self.score
    }

    pub fn running(&self) -> bool {
        self.running
    }

    pub fn direction(&self) -> Option<i8> {
        self.direction
    }

    pub fn snake_len(&self) -> usize {
        self.snake.len()
    }

    pub fn head_pos(&self) -> (i32, i32) {
        if self.snake.is_empty() {
            return (0, 0);
        }
        let idx0 = self.snake[0];
        let x = (idx0 % self.width) as i32;
        let y = (idx0 / self.width) as i32;
        (x, y)
    }

    pub fn food_positions(&self) -> Vec<(i32, i32)> {
        self.food
            .iter()
            .map(|&i| ((i % self.width) as i32, (i / self.width) as i32))
            .collect()
    }

    pub fn max_playable_tiles(&self) -> usize {
        self.width * self.height - self.wall_count
    }

    pub fn spawnable_count(&self) -> usize {
        let max_playable = self.max_playable_tiles();
        let snake_len = self.snake.len();
        let food_len = self.food.len();
        if max_playable < snake_len + food_len {
            0
        } else {
            max_playable - snake_len - food_len
        }
    }
}

impl GameCore {
    fn spawn_snake(&mut self) -> Result<(), EngineError> {
        let length = if self.spawn_len < 2 {
            2
        } else {
            self.spawn_len
        };

        let direction = if self.spawn_random_dir {
            self.rng.gen_range(0..4) as i8
        } else {
            self.spawn_dir.unwrap_or(1)
        };

        let base = if self.spawn_x.is_none() || self.spawn_y.is_none() {
            Point {
                x: (self.width / 2) as i32,
                y: (self.height / 2) as i32,
            }
        } else {
            Point {
                x: self.spawn_x.unwrap(),
                y: self.spawn_y.unwrap(),
            }
        };

        let jitter = if self.spawn_jitter < 0 {
            0
        } else {
            self.spawn_jitter
        };

        let mut candidates: Vec<Point> = Vec::new();
        candidates.push(base);

        if jitter > 0 {
            for _ in 0..64 {
                let dx = self.rng.gen_range(-jitter..=jitter);
                let dy = self.rng.gen_range(-jitter..=jitter);
                candidates.push(Point {
                    x: base.x + dx,
                    y: base.y + dy,
                });
            }
        }

        for _ in 0..64 {
            candidates.push(Point {
                x: self.rng.gen_range(0..self.width) as i32,
                y: self.rng.gen_range(0..self.height) as i32,
            });
        }

        for head in candidates {
            let cells = compute_spawn_cells(head, length, direction);
            if is_straight_spawn_valid(
                &cells,
                self.width as i32,
                self.height as i32,
                &self.wall_mask,
            ) {
                self.snake.clear();
                self.snake_mask.fill(false);
                for p in cells.iter() {
                    let i = idx(p.x, p.y, self.width);
                    self.snake.push(i);
                    self.snake_mask[i] = true;
                }
                self.direction = Some(direction);
                return Ok(());
            }
        }

        Err(EngineError::SpawnFailed)
    }

    fn spawn_food(&mut self) {
        let need = if self.target_food_count > self.food.len() {
            self.target_food_count - self.food.len()
        } else {
            0
        };
        if need == 0 {
            return;
        }

        let mut spawnable: Vec<usize> = Vec::new();
        for i in 0..(self.width * self.height) {
            if self.wall_mask[i] {
                continue;
            }
            if self.snake_mask[i] {
                continue;
            }
            if self.food_mask[i] {
                continue;
            }
            spawnable.push(i);
        }

        if spawnable.is_empty() {
            return;
        }

        let k = std::cmp::min(need, spawnable.len());
        let indices = sample(&mut self.rng, spawnable.len(), k);
        for idx_i in indices.iter() {
            let p = spawnable[idx_i];
            self.food.push(p);
            self.food_mask[p] = true;
        }
    }

    fn won(&self) -> bool {
        self.snake.len() >= self.max_playable_tiles()
    }

    fn step_internal(&mut self, rel_dir: i32) -> Result<u32, EngineError> {
        if !self.running {
            return Ok(MOVE_NOT_RUNNING);
        }

        let dir = match self.direction {
            Some(d) => d,
            None => return Err(EngineError::DirectionNone),
        };

        let mut new_dir = dir;
        if rel_dir == 1 {
            new_dir = dir_turn_left(dir);
        } else if rel_dir == 2 {
            new_dir = dir_turn_right(dir);
        }

        let (dx, dy) = dir_vec(new_dir);
        let head_idx = self.snake[0];
        let head_x = (head_idx % self.width) as i32;
        let head_y = (head_idx / self.width) as i32;

        let new_head = Point {
            x: head_x + dx,
            y: head_y + dy,
        };

        if new_head.x < 0
            || new_head.x >= self.width as i32
            || new_head.y < 0
            || new_head.y >= self.height as i32
        {
            self.running = false;
            return Ok(MOVE_HIT_BOUNDARY);
        }

        let new_head_idx = idx(new_head.x, new_head.y, self.width);

        if self.wall_mask[new_head_idx] {
            self.running = false;
            return Ok(MOVE_HIT_WALL);
        }

        if self.snake_mask[new_head_idx] {
            self.running = false;
            return Ok(MOVE_HIT_SELF);
        }

        self.direction = Some(new_dir);
        self.snake.insert(0, new_head_idx);
        self.snake_mask[new_head_idx] = true;

        let mut result = MOVE_OK;

        if self.food_mask[new_head_idx] {
            self.food_mask[new_head_idx] = false;
            if let Some(pos) = self.food.iter().position(|&i| i == new_head_idx) {
                self.food.remove(pos);
            }
            self.score += 1;
            result |= MOVE_FOOD;
            self.spawn_food();

            if self.food.is_empty() && self.won() {
                self.running = false;
                result |= MOVE_WIN;
            }
        } else {
            let tail = self.snake.pop().unwrap();
            self.snake_mask[tail] = false;
        }

        self.rebuild_grids();
        Ok(result)
    }

    fn rebuild_grids(&mut self) {
        self.tile_grid.clone_from(&self.static_grid);

        for &i in self.food.iter() {
            self.tile_grid[i] = TILE_FOOD;
        }

        if !self.snake.is_empty() {
            let head_idx = self.snake[0];
            let head_dir = self.direction.unwrap_or(1);
            self.tile_grid[head_idx] = head_tile(head_dir);

            if self.snake.len() > 1 {
                for j in 1..(self.snake.len() - 1) {
                    let prev = self.snake[j - 1];
                    let curr = self.snake[j];
                    let nxt = self.snake[j + 1];
                    self.tile_grid[curr] = body_tile(
                        idx_to_point(prev, self.width),
                        idx_to_point(curr, self.width),
                        idx_to_point(nxt, self.width),
                    );
                }
                let tail = self.snake[self.snake.len() - 1];
                let prev = self.snake[self.snake.len() - 2];
                self.tile_grid[tail] = tail_tile(
                    idx_to_point(prev, self.width),
                    idx_to_point(tail, self.width),
                );
            }
        }

        self.rebuild_pixels();
    }

    fn rebuild_pixels(&mut self) {
        let ts = self.tile_size;
        let pw = self.width * ts;
        let ph = self.height * ts;
        let expected = pw * ph;
        if self.pixel_grid.len() != expected {
            self.pixel_grid = vec![0u8; expected];
        }

        for y in 0..self.height {
            for x in 0..self.width {
                let tile_id = self.tile_grid[y * self.width + x] as usize;
                let tile = &self.tile_cache[tile_id];
                for ty in 0..ts {
                    let row_start = (y * ts + ty) * pw + (x * ts);
                    let tile_row_start = ty * ts;
                    self.pixel_grid[row_start..row_start + ts]
                        .copy_from_slice(&tile[tile_row_start..tile_row_start + ts]);
                }
            }
        }
    }
}
