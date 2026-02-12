// rust/src/engine/snake_engine.rs
use rand::seq::index::sample;
use rand::{Rng, RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;

use super::constants::*;
use super::geometry::{
    Point, compute_spawn_cells, dir_turn_left, dir_turn_right, dir_vec, idx, idx_to_point,
    is_straight_spawn_valid,
};
use super::tiles::{body_tile, head_tile, tail_tile};
use super::tileset::{tileset_tile_size, tileset_tiles};

#[derive(Debug)]
pub enum EngineError {
    InvalidGridLen,
    SpawnFailed,
    DirectionNone,
}

impl std::fmt::Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EngineError::InvalidGridLen => {
                write!(f, "invalid grid dimensions (width and height must be > 0)")
            }
            EngineError::SpawnFailed => write!(f, "could not find a valid snake spawn"),
            EngineError::DirectionNone => write!(f, "direction is None"),
        }
    }
}

impl std::error::Error for EngineError {}

pub struct SnakeEngine {
    width: usize,
    height: usize,
    tile_size: usize,

    static_grid: Vec<u8>,
    wall_mask: Vec<bool>,
    wall_count: usize,

    spawn_len: usize,
    spawn_dir: i8,
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

impl SnakeEngine {
    pub fn new(
        width: usize,
        height: usize,
        food_count: usize,
        seed: Option<u64>,
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

        Ok(Self {
            width,
            height,
            tile_size,
            static_grid,
            wall_mask,
            wall_count,
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

    pub fn head_tile_view(
        &self,
        view_radius_y: i32,
        view_radius_x: i32,
        rotate_to_head: bool,
        empty_id: u8,
    ) -> Result<(Vec<u8>, Vec<bool>, usize, usize), EngineError> {
        let ry = view_radius_y.max(0);
        let rx = view_radius_x.max(0);
        let vy = (2 * ry + 1) as usize;
        let vx = (2 * rx + 1) as usize;

        let mut out = vec![empty_id; vy * vx];
        let mut valid = vec![false; vy * vx];

        let (head_x, head_y) = self.head_pos();
        let dir = match self.direction {
            Some(d) => d,
            None => return Err(EngineError::DirectionNone),
        };

        for oy in -ry..=ry {
            for ox in -rx..=rx {
                let (dx_w, dy_w) = if rotate_to_head {
                    match dir {
                        0 => (ox, oy),
                        1 => (-oy, ox),
                        2 => (-ox, -oy),
                        3 => (oy, -ox),
                        _ => (ox, oy),
                    }
                } else {
                    (ox, oy)
                };

                let wx = head_x + dx_w;
                let wy = head_y + dy_w;
                if wx < 0 || wy < 0 || wx >= self.width as i32 || wy >= self.height as i32 {
                    continue;
                }

                let dst_x = (ox + rx) as usize;
                let dst_y = (oy + ry) as usize;
                let dst_idx = dst_y * vx + dst_x;

                let src_idx = idx(wx, wy, self.width);
                out[dst_idx] = self.tile_grid[src_idx];
                valid[dst_idx] = true;
            }
        }

        Ok((out, valid, vy, vx))
    }

    pub fn head_pixel_view(
        &self,
        view_radius_y: i32,
        view_radius_x: i32,
        rotate_to_head: bool,
        oob_fill_value: u8,
    ) -> Result<(Vec<u8>, Vec<bool>, usize, usize), EngineError> {
        let ry = view_radius_y.max(0);
        let rx = view_radius_x.max(0);
        let vy = (2 * ry + 1) as usize;
        let vx = (2 * rx + 1) as usize;

        let tile_size = self.tile_size;
        let view_h = vy * tile_size;
        let view_w = vx * tile_size;

        let mut out = vec![oob_fill_value; view_h * view_w];
        let mut valid = vec![false; view_h * view_w];

        let (head_x, head_y) = self.head_pos();
        let dir = match self.direction {
            Some(d) => d,
            None => return Err(EngineError::DirectionNone),
        };

        let grid_px_w = self.width * tile_size;

        for oy in -ry..=ry {
            for ox in -rx..=rx {
                let (dx_w, dy_w) = if rotate_to_head {
                    match dir {
                        0 => (ox, oy),
                        1 => (-oy, ox),
                        2 => (-ox, -oy),
                        3 => (oy, -ox),
                        _ => (ox, oy),
                    }
                } else {
                    (ox, oy)
                };

                let wx = head_x + dx_w;
                let wy = head_y + dy_w;
                if wx < 0 || wy < 0 || wx >= self.width as i32 || wy >= self.height as i32 {
                    continue;
                }

                let src_x = (wx as usize) * tile_size;
                let src_y = (wy as usize) * tile_size;
                let dst_x = ((ox + rx) as usize) * tile_size;
                let dst_y = ((oy + ry) as usize) * tile_size;

                let k = if rotate_to_head { dir } else { 0 };

                for dy in 0..tile_size {
                    for dx in 0..tile_size {
                        let (sy, sx) = match k {
                            0 => (dy, dx),
                            1 => (dx, tile_size - 1 - dy),
                            2 => (tile_size - 1 - dy, tile_size - 1 - dx),
                            3 => (tile_size - 1 - dx, dy),
                            _ => (dy, dx),
                        };

                        let src_idx = (src_y + sy) * grid_px_w + (src_x + sx);
                        let dst_idx = (dst_y + dy) * view_w + (dst_x + dx);
                        out[dst_idx] = self.pixel_grid[src_idx];
                        valid[dst_idx] = true;
                    }
                }
            }
        }

        Ok((out, valid, view_h, view_w))
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

impl SnakeEngine {
    fn spawn_snake(&mut self) -> Result<(), EngineError> {
        let length = if self.spawn_len < 2 {
            2
        } else {
            self.spawn_len
        };

        let direction = if self.spawn_random_dir {
            self.rng.gen_range(0..4) as i8
        } else {
            self.spawn_dir
        };

        let base = Point {
            x: (self.width / 2) as i32,
            y: (self.height / 2) as i32,
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
