// rust/src/engine/snake_engine/info.rs

use crate::engine::spatial::{dir_turn_left, dir_turn_right};

use super::state::{EngineError, SnakeEngine};

impl SnakeEngine {
    pub fn tile_grid(&self) -> &[u8] {
        &self.tile_grid
    }

    pub fn tile_grid_stacked(&mut self) -> (Vec<u8>, usize, usize, usize) {
        let n = self.frame_stack_n;
        if n <= 1 {
            return (self.tile_grid.clone(), 1, self.height, self.width);
        }
        if let Some(stacker) = self.world_tile_stack.as_mut() {
            return (stacker.stacked().to_vec(), n, self.height, self.width);
        }
        (self.tile_grid.clone(), 1, self.height, self.width)
    }

    pub fn pixel_grid(&self) -> Result<&[u8], EngineError> {
        if !self.pixel_grid_enabled {
            return Err(EngineError::PixelGridDisabled);
        }
        Ok(&self.pixel_grid)
    }

    pub fn pixel_grid_stacked(&mut self) -> Result<(Vec<u8>, usize, usize, usize), EngineError> {
        if !self.pixel_grid_enabled {
            return Err(EngineError::PixelGridDisabled);
        }
        let n = self.frame_stack_n;
        let h = self.height * self.tile_size;
        let w = self.width * self.tile_size;
        if n <= 1 {
            return Ok((self.pixel_grid.clone(), 1, h, w));
        }
        if let Some(stacker) = self.world_pixel_stack.as_mut() {
            return Ok((stacker.stacked().to_vec(), n, h, w));
        }
        Ok((self.pixel_grid.clone(), 1, h, w))
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

    pub fn frame_stack_n(&self) -> usize {
        self.frame_stack_n
    }

    pub fn score(&self) -> i32 {
        self.score
    }

    pub fn running(&self) -> bool {
        self.running
    }

    pub fn episode_steps(&self) -> u64 {
        self.step_counter
    }

    pub fn direction(&self) -> Option<i8> {
        self.direction
    }

    pub fn direction_id(&self) -> u8 {
        match self.direction {
            Some(d) => ((d % 4 + 4) % 4) as u8,
            None => 0,
        }
    }

    pub fn snake_len(&self) -> usize {
        self.snake.len()
    }

    pub fn snake_progress(&self) -> f32 {
        let max_playable = self.max_playable_tiles();
        if max_playable <= self.initial_snake_len {
            return 0.0;
        }
        let denom = (max_playable - self.initial_snake_len) as f32;
        let num = (self.snake.len().saturating_sub(self.initial_snake_len)) as f32;
        let mut v = num / denom;
        if v < 0.0 {
            v = 0.0;
        }
        if v > 1.0 {
            v = 1.0;
        }
        v
    }

    pub fn time_since_food_norm(&self, max_steps: usize) -> f32 {
        let denom = max_steps.max(1) as f32;
        let v = (self.steps_since_food.min(max_steps)) as f32 / denom;
        if v < 0.0 {
            0.0
        } else if v > 1.0 {
            1.0
        } else {
            v
        }
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

    fn closest_food_by_metric(&self, metric: u8) -> Option<(i32, i32)> {
        if self.food.is_empty() {
            return None;
        }
        let (hx, hy) = self.head_pos();
        let mut best: Option<(i32, i32, i32)> = None;
        for &i in self.food.iter() {
            let fx = (i % self.width) as i32;
            let fy = (i / self.width) as i32;
            let dx = fx - hx;
            let dy = fy - hy;
            let dist = match metric {
                1 | 2 => dx * dx + dy * dy,
                _ => dx.abs() + dy.abs(),
            };
            match best {
                None => best = Some((dx, dy, dist)),
                Some((_bx, _by, bd)) if dist < bd => best = Some((dx, dy, dist)),
                _ => {}
            }
        }
        best.map(|(dx, dy, _)| (dx, dy))
    }

    pub fn closest_food(&self) -> Option<(i32, i32, i32)> {
        let (dx, dy) = self.closest_food_by_metric(0)?;
        Some((dx, dy, dx.abs() + dy.abs()))
    }

    pub fn closest_food_norm(&self, metric: u8) -> (f32, f32, f32) {
        let (dx, dy) = match self.closest_food_by_metric(metric) {
            Some(v) => v,
            None => (0, 0),
        };

        let w = (self.width.saturating_sub(1)).max(1) as f32;
        let h = (self.height.saturating_sub(1)).max(1) as f32;
        let dx_f = (dx as f32) / w;
        let dy_f = (dy as f32) / h;

        let dist_f = match metric {
            1 => {
                let denom = (w * w + h * h).sqrt();
                if denom <= 0.0 {
                    0.0
                } else {
                    let d2 = (dx * dx + dy * dy) as f32;
                    d2.sqrt() / denom
                }
            }
            2 => {
                let denom = w * w + h * h;
                if denom <= 0.0 {
                    0.0
                } else {
                    let d2 = (dx * dx + dy * dy) as f32;
                    d2 / denom
                }
            }
            _ => {
                let denom = w + h;
                if denom <= 0.0 {
                    0.0
                } else {
                    let d1 = (dx.abs() + dy.abs()) as f32;
                    d1 / denom
                }
            }
        };

        let dist_clamped = if dist_f < 0.0 {
            0.0
        } else if dist_f > 1.0 {
            1.0
        } else {
            dist_f
        };

        (dx_f, dy_f, dist_clamped)
    }

    pub fn steps_since_food(&self) -> usize {
        self.steps_since_food
    }

    pub fn set_max_steps(&mut self, max_steps: Option<usize>) {
        self.max_steps = max_steps;
    }

    pub fn set_spawn_random_dir(&mut self, enabled: bool) {
        self.spawn_random_dir = enabled;
    }

    pub fn collision_flags(&self) -> (bool, bool, bool) {
        let dir = match self.direction {
            Some(d) => d,
            None => return (false, false, false),
        };
        (
            self.would_collide(dir),
            self.would_collide(dir_turn_left(dir)),
            self.would_collide(dir_turn_right(dir)),
        )
    }

    pub fn max_playable_tiles(&self) -> usize {
        self.width * self.height - self.wall_count
    }

    pub fn spawnable_count(&self) -> usize {
        self.spawnable.len()
    }
}
