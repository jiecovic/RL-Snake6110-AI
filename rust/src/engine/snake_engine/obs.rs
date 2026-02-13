// rust/src/engine/snake_engine/obs.rs

use crate::engine::constants::TILE_OOB;
use crate::engine::obs::HeadViewKey;
use crate::engine::spatial::idx;

use super::state::{EngineError, SnakeEngine};

impl SnakeEngine {
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

    pub fn head_tile_view_stacked(
        &mut self,
        view_radius_y: i32,
        view_radius_x: i32,
        rotate_to_head: bool,
        empty_id: u8,
    ) -> Result<(Vec<u8>, usize, usize, usize), EngineError> {
        let (view, _valid, h, w) =
            self.head_tile_view(view_radius_y, view_radius_x, rotate_to_head, empty_id)?;

        let n = self.frame_stack_n;
        if n <= 1 {
            return Ok((view, 1, h, w));
        }

        let key = HeadViewKey {
            ry: view_radius_y,
            rx: view_radius_x,
            rotate: rotate_to_head,
            param: empty_id,
        };
        if let Some(stacker) = self.head_tile_stack.as_mut() {
            stacker.update(self.step_counter, key, &view);
            return Ok((stacker.stacked().to_vec(), n, h, w));
        }
        Ok((view, 1, h, w))
    }

    pub fn head_pixel_view(
        &self,
        view_radius_y: i32,
        view_radius_x: i32,
        rotate_to_head: bool,
        oob_fill_value: u8,
    ) -> Result<(Vec<u8>, Vec<bool>, usize, usize), EngineError> {
        if !self.pixel_grid_enabled {
            return Err(EngineError::PixelGridDisabled);
        }
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

        // If we have a tileset-defined OOB tile, prefill the view with it for consistency.
        let oob_id = TILE_OOB as usize;
        if oob_id < self.tile_cache.len() {
            let oob_tile = &self.tile_cache[oob_id];
            if oob_tile.len() == tile_size * tile_size {
                for ty in 0..vy {
                    for tx in 0..vx {
                        let dst_x0 = tx * tile_size;
                        let dst_y0 = ty * tile_size;
                        for dy in 0..tile_size {
                            let src_row = dy * tile_size;
                            let dst_idx = (dst_y0 + dy) * view_w + dst_x0;
                            out[dst_idx..dst_idx + tile_size]
                                .copy_from_slice(&oob_tile[src_row..src_row + tile_size]);
                        }
                    }
                }
            }
        }

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

    pub fn head_pixel_view_stacked(
        &mut self,
        view_radius_y: i32,
        view_radius_x: i32,
        rotate_to_head: bool,
        oob_fill_value: u8,
    ) -> Result<(Vec<u8>, Vec<u8>, usize, usize, usize), EngineError> {
        let (view, valid, h, w) =
            self.head_pixel_view(view_radius_y, view_radius_x, rotate_to_head, oob_fill_value)?;

        let n = self.frame_stack_n;
        let valid_u8: Vec<u8> = valid.iter().map(|v| if *v { 1 } else { 0 }).collect();
        if n <= 1 {
            return Ok((view, valid_u8, 1, h, w));
        }

        let key = HeadViewKey {
            ry: view_radius_y,
            rx: view_radius_x,
            rotate: rotate_to_head,
            param: oob_fill_value,
        };
        if let Some(stacker) = self.head_pixel_stack.as_mut() {
            stacker.update(self.step_counter, key, &view);
            let stacked = stacker.stacked().to_vec();

            if let Some(vs) = self.head_pixel_valid_stack.as_mut() {
                vs.update(self.step_counter, key, &valid_u8);
                let valid_stacked = vs.stacked().to_vec();
                return Ok((stacked, valid_stacked, n, h, w));
            }
        }
        Ok((view, valid_u8, 1, h, w))
    }
}
