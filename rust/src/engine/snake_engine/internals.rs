// rust/src/engine/snake_engine/internals.rs

use rand::seq::index::sample;
use rand::Rng;

use crate::engine::constants::*;
use crate::engine::spatial::{
    Point, compute_spawn_cells, dir_turn_left, dir_turn_right, dir_vec, idx, idx_to_point,
    is_straight_spawn_valid,
};
use crate::engine::tiles::{body_tile, head_tile, tail_tile};

use super::state::{EngineError, SnakeEngine};

impl SnakeEngine {
    pub(crate) fn spawnable_reset(&mut self) {
        self.spawnable.clear();
        self.spawnable_pos.fill(usize::MAX);
        for i in 0..(self.width * self.height) {
            if self.wall_mask[i] {
                continue;
            }
            self.spawnable_pos[i] = self.spawnable.len();
            self.spawnable.push(i);
        }
    }

    pub(crate) fn spawnable_remove(&mut self, idx: usize) {
        let pos = self.spawnable_pos[idx];
        if pos == usize::MAX {
            return;
        }
        let last_idx = *self.spawnable.last().unwrap();
        self.spawnable[pos] = last_idx;
        self.spawnable.pop();
        self.spawnable_pos[last_idx] = pos;
        self.spawnable_pos[idx] = usize::MAX;
    }

    pub(crate) fn spawnable_add(&mut self, idx: usize) {
        if self.spawnable_pos[idx] != usize::MAX {
            return;
        }
        self.spawnable_pos[idx] = self.spawnable.len();
        self.spawnable.push(idx);
    }

    pub(crate) fn would_collide(&self, dir: i8) -> bool {
        if self.snake.is_empty() {
            return false;
        }
        let (dx, dy) = dir_vec(dir);
        let head_idx = self.snake[0];
        let head_x = (head_idx % self.width) as i32;
        let head_y = (head_idx / self.width) as i32;
        let nx = head_x + dx;
        let ny = head_y + dy;
        if nx < 0 || ny < 0 || nx >= self.width as i32 || ny >= self.height as i32 {
            return true;
        }
        let nidx = idx(nx, ny, self.width);
        if self.wall_mask[nidx] {
            return true;
        }
        if self.snake_mask[nidx] {
            return true;
        }
        false
    }

    pub(crate) fn spawn_snake(&mut self) -> Result<(), EngineError> {
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
                    self.snake.push_back(i);
                    self.snake_mask[i] = true;
                    self.spawnable_remove(i);
                }
                self.direction = Some(direction);
                return Ok(());
            }
        }

        Err(EngineError::SpawnFailed)
    }

    pub(crate) fn spawn_food(&mut self) -> Vec<usize> {
        let need = if self.target_food_count > self.food.len() {
            self.target_food_count - self.food.len()
        } else {
            0
        };
        if need == 0 {
            return Vec::new();
        }

        if self.spawnable.is_empty() {
            return Vec::new();
        }

        let k = std::cmp::min(need, self.spawnable.len());
        let indices = sample(&mut self.rng, self.spawnable.len(), k);
        let mut spawned = Vec::with_capacity(k);
        for idx_i in indices.iter() {
            spawned.push(self.spawnable[idx_i]);
        }
        for &p in spawned.iter() {
            self.food.push(p);
            self.food_mask[p] = true;
            self.spawnable_remove(p);
        }
        spawned
    }

    fn won(&self) -> bool {
        self.snake.len() >= self.max_playable_tiles()
    }

    pub(crate) fn step_internal(&mut self, rel_dir: i32) -> Result<u32, EngineError> {
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
        self.snake.push_front(new_head_idx);
        self.snake_mask[new_head_idx] = true;
        self.spawnable_remove(new_head_idx);

        let mut result = MOVE_OK;

        let ate_food = self.food_mask[new_head_idx];
        if ate_food {
            self.food_mask[new_head_idx] = false;
            if let Some(pos) = self.food.iter().position(|&i| i == new_head_idx) {
                self.food.remove(pos);
            }
            self.score += 1;
            result |= MOVE_FOOD;
            let spawned = self.spawn_food();

            if self.food.is_empty() && self.won() {
                self.running = false;
                result |= MOVE_WIN;
            }
            self.update_tiles_after_step(new_dir, None, &spawned);
        } else {
            let tail = self.snake.pop_back().unwrap();
            self.snake_mask[tail] = false;
            self.spawnable_add(tail);
            self.update_tiles_after_step(new_dir, Some(tail), &[]);
        }
        self.step_counter = self.step_counter.wrapping_add(1);
        if result & MOVE_FOOD != 0 {
            self.steps_since_food = 0;
        } else {
            self.steps_since_food = self.steps_since_food.saturating_add(1);
        }
        if self.running {
            if let Some(limit) = self.max_steps {
                if limit > 0 && self.steps_since_food >= limit {
                    self.running = false;
                    result |= MOVE_TIMEOUT;
                }
            }
        }
        self.update_world_stacks(false);
        Ok(result)
    }

    pub(crate) fn rebuild_grids(&mut self) {
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

        if self.pixel_grid_enabled {
            self.rebuild_pixels();
        }
    }

    fn update_tiles_after_step(
        &mut self,
        new_dir: i8,
        old_tail_idx: Option<usize>,
        spawned_food: &[usize],
    ) {
        let track_pixels = self.pixel_grid_enabled;
        let mut dirty = if track_pixels {
            Vec::with_capacity(4 + spawned_food.len())
        } else {
            Vec::new()
        };
        if let Some(tail_idx) = old_tail_idx {
            self.tile_grid[tail_idx] = self.static_grid[tail_idx];
            if track_pixels {
                dirty.push(tail_idx);
            }
        }

        if self.snake.is_empty() {
            return;
        }

        let head_idx = self.snake[0];
        self.tile_grid[head_idx] = head_tile(new_dir);
        if track_pixels {
            dirty.push(head_idx);
        }

        let len = self.snake.len();
        if len >= 2 {
            if len >= 3 {
                let prev = self.snake[0];
                let curr = self.snake[1];
                let next = self.snake[2];
                self.tile_grid[curr] = body_tile(
                    idx_to_point(prev, self.width),
                    idx_to_point(curr, self.width),
                    idx_to_point(next, self.width),
                );
                if track_pixels {
                    dirty.push(curr);
                }
            }
            let tail_idx = self.snake[len - 1];
            let prev_idx = self.snake[len - 2];
            self.tile_grid[tail_idx] = tail_tile(
                idx_to_point(prev_idx, self.width),
                idx_to_point(tail_idx, self.width),
            );
            if track_pixels {
                dirty.push(tail_idx);
            }
        }

        for &idx in spawned_food {
            self.tile_grid[idx] = TILE_FOOD;
            if track_pixels {
                dirty.push(idx);
            }
        }

        if track_pixels {
            self.update_pixels_for_tiles(&dirty);
        }
    }

    fn rebuild_pixels(&mut self) {
        if !self.pixel_grid_enabled {
            return;
        }
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

    fn update_pixels_for_tiles(&mut self, indices: &[usize]) {
        if !self.pixel_grid_enabled {
            return;
        }
        let ts = self.tile_size;
        let pw = self.width * ts;
        for &idx in indices {
            let tile_id = self.tile_grid[idx] as usize;
            let tile = &self.tile_cache[tile_id];
            let x = idx % self.width;
            let y = idx / self.width;
            let dst_x = x * ts;
            let dst_y = y * ts;
            for ty in 0..ts {
                let row_start = (dst_y + ty) * pw + dst_x;
                let tile_row_start = ty * ts;
                self.pixel_grid[row_start..row_start + ts]
                    .copy_from_slice(&tile[tile_row_start..tile_row_start + ts]);
            }
        }
    }

    pub(crate) fn update_world_stacks(&mut self, reset: bool) {
        let n = self.frame_stack_n;
        if n <= 1 {
            return;
        }
        if let Some(stack) = self.world_tile_stack.as_mut() {
            if reset {
                stack.reset_with(&self.tile_grid);
            } else {
                stack.push(&self.tile_grid);
            }
        }
        if self.pixel_grid_enabled {
            if let Some(stack) = self.world_pixel_stack.as_mut() {
                if reset {
                    stack.reset_with(&self.pixel_grid);
                } else {
                    stack.push(&self.pixel_grid);
                }
            }
        }
    }
}
