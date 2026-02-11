use numpy::{ndarray, IntoPyArray, PyArray2, PyArray3, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::prelude::*;
use rand::seq::index::sample;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

const MOVE_OK: u32 = 1 << 0;
const MOVE_FOOD: u32 = 1 << 1;
const MOVE_HIT_BOUNDARY: u32 = 1 << 2;
const MOVE_HIT_WALL: u32 = 1 << 3;
const MOVE_HIT_SELF: u32 = 1 << 4;
const MOVE_NOT_RUNNING: u32 = 1 << 5;
const MOVE_TIMEOUT: u32 = 1 << 6;
const MOVE_WIN: u32 = 1 << 7;

const TILE_EMPTY: u8 = 0;
#[allow(dead_code)]
const TILE_WALL_TL: u8 = 1;
#[allow(dead_code)]
const TILE_WALL_TR: u8 = 2;
#[allow(dead_code)]
const TILE_WALL_BL: u8 = 3;
#[allow(dead_code)]
const TILE_WALL_BR: u8 = 4;
#[allow(dead_code)]
const TILE_WALL_TOP: u8 = 5;
#[allow(dead_code)]
const TILE_WALL_BOTTOM: u8 = 6;
#[allow(dead_code)]
const TILE_WALL_LEFT: u8 = 7;
#[allow(dead_code)]
const TILE_WALL_RIGHT: u8 = 8;
const TILE_HEAD_UP: u8 = 9;
const TILE_HEAD_DOWN: u8 = 10;
const TILE_HEAD_LEFT: u8 = 11;
const TILE_HEAD_RIGHT: u8 = 12;
const TILE_BODY_VERTICAL_UP: u8 = 13;
const TILE_BODY_VERTICAL_DOWN: u8 = 14;
const TILE_BODY_HORIZONTAL_LEFT: u8 = 15;
const TILE_BODY_HORIZONTAL_RIGHT: u8 = 16;
const TILE_BODY_BR: u8 = 17;
const TILE_BODY_BL: u8 = 18;
const TILE_BODY_TR: u8 = 19;
const TILE_BODY_TL: u8 = 20;
const TILE_TAIL_UP: u8 = 21;
const TILE_TAIL_DOWN: u8 = 22;
const TILE_TAIL_LEFT: u8 = 23;
const TILE_TAIL_RIGHT: u8 = 24;
const TILE_FOOD: u8 = 25;

#[derive(Clone, Copy)]
struct Point {
    x: i32,
    y: i32,
}

fn idx(x: i32, y: i32, width: usize) -> usize {
    (y as usize) * width + (x as usize)
}

fn dir_turn_left(d: i8) -> i8 {
    ((d + 3) % 4) as i8
}

fn dir_turn_right(d: i8) -> i8 {
    ((d + 1) % 4) as i8
}

fn dir_vec(d: i8) -> (i32, i32) {
    match d {
        0 => (0, -1),  // up
        1 => (1, 0),   // right
        2 => (0, 1),   // down
        3 => (-1, 0),  // left
        _ => (0, 0),
    }
}

#[pyclass]
struct Game {
    #[pyo3(get)]
    width: usize,
    #[pyo3(get)]
    height: usize,
    #[pyo3(get)]
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

    #[pyo3(get)]
    score: i32,
    #[pyo3(get)]
    running: bool,

    tile_grid: Vec<u8>,
    pixel_grid: Vec<u8>,
    tile_cache: Vec<Vec<u8>>,
}

#[pymethods]
impl Game {
    #[new]
    #[pyo3(
        signature = (
            width,
            height,
            level_grid,
            spawn_len,
            spawn_dir,
            spawn_random_dir,
            spawn_jitter,
            food_count,
            tile_size,
            tiles,
            spawn_x = None,
            spawn_y = None,
            seed = None
        )
    )]
    fn new(
        width: usize,
        height: usize,
        level_grid: PyReadonlyArray2<u8>,
        spawn_len: usize,
        spawn_dir: i8,
        spawn_random_dir: bool,
        spawn_jitter: i32,
        food_count: usize,
        tile_size: usize,
        tiles: PyReadonlyArray3<u8>,
        spawn_x: Option<i32>,
        spawn_y: Option<i32>,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        let grid = level_grid.as_array();
        if grid.shape() != [height, width] {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "level_grid shape does not match height/width",
            ));
        }

        let tiles_arr = tiles.as_array();
        let tiles_shape = tiles_arr.shape();
        if tiles_shape.len() != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "tiles must be a 3D array",
            ));
        }
        if tiles_shape[1] != tile_size || tiles_shape[2] != tile_size {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "tiles tile_size mismatch",
            ));
        }

        let mut static_grid = vec![0u8; width * height];
        let mut wall_mask = vec![false; width * height];
        let mut wall_count = 0usize;

        for y in 0..height {
            for x in 0..width {
                let v = grid[(y, x)];
                static_grid[y * width + x] = v;
                if v != TILE_EMPTY {
                    wall_mask[y * width + x] = true;
                    wall_count += 1;
                }
            }
        }

        let mut tile_cache: Vec<Vec<u8>> = Vec::with_capacity(tiles_shape[0]);
        for i in 0..tiles_shape[0] {
            let tile = tiles_arr.index_axis(ndarray::Axis(0), i);
            tile_cache.push(tile.iter().copied().collect());
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
            spawn_dir: if spawn_dir < 0 { None } else { Some(spawn_dir) },
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

    fn reset(&mut self, seed: Option<u64>) -> PyResult<()> {
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

    fn step(&mut self, rel_dir: u8) -> PyResult<u32> {
        let mask = self.step_internal(rel_dir as i32)?;
        Ok(mask)
    }

    fn tile_grid<'py>(&self, py: Python<'py>) -> Py<PyArray2<u8>> {
        let arr = ndarray::Array2::from_shape_vec(
            (self.height, self.width),
            self.tile_grid.clone(),
        )
        .unwrap();
        arr.into_pyarray_bound(py).unbind()
    }

    fn pixel_grid<'py>(&self, py: Python<'py>) -> Py<PyArray2<u8>> {
        let h = self.height * self.tile_size;
        let w = self.width * self.tile_size;
        let arr = ndarray::Array2::from_shape_vec((h, w), self.pixel_grid.clone()).unwrap();
        arr.into_pyarray_bound(py).unbind()
    }

    fn direction(&self) -> i8 {
        self.direction.unwrap_or(-1)
    }

    fn snake_len(&self) -> usize {
        self.snake.len()
    }

    fn head_pos(&self) -> (i32, i32) {
        if self.snake.is_empty() {
            return (0, 0);
        }
        let idx0 = self.snake[0];
        let x = (idx0 % self.width) as i32;
        let y = (idx0 / self.width) as i32;
        (x, y)
    }

    fn food_positions(&self) -> Vec<(i32, i32)> {
        self.food
            .iter()
            .map(|&i| ((i % self.width) as i32, (i / self.width) as i32))
            .collect()
    }

    fn max_playable_tiles(&self) -> usize {
        self.width * self.height - self.wall_count
    }

    fn spawnable_count(&self) -> usize {
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

impl Game {
    fn spawn_snake(&mut self) -> PyResult<()> {
        let length = if self.spawn_len < 2 { 2 } else { self.spawn_len };

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

        let jitter = if self.spawn_jitter < 0 { 0 } else { self.spawn_jitter };

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
            if is_straight_spawn_valid(&cells, self.width as i32, self.height as i32, &self.wall_mask)
            {
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

        Err(pyo3::exceptions::PyValueError::new_err(
            "Could not find a valid snake spawn",
        ))
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

    fn step_internal(&mut self, rel_dir: i32) -> PyResult<u32> {
        if !self.running {
            return Ok(MOVE_NOT_RUNNING);
        }

        let dir = match self.direction {
            Some(d) => d,
            None => return Err(pyo3::exceptions::PyRuntimeError::new_err("direction is None")),
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

        // Apply move
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

#[pyclass]
struct VecGame {
    games: Vec<Game>,
}

#[pymethods]
impl VecGame {
    #[new]
    #[pyo3(
        signature = (
            n,
            width,
            height,
            level_grid,
            spawn_len,
            spawn_dir,
            spawn_random_dir,
            spawn_jitter,
            food_count,
            tile_size,
            tiles,
            spawn_x = None,
            spawn_y = None,
            seeds = None
        )
    )]
    fn new(
        n: usize,
        width: usize,
        height: usize,
        level_grid: PyReadonlyArray2<u8>,
        spawn_len: usize,
        spawn_dir: i8,
        spawn_random_dir: bool,
        spawn_jitter: i32,
        food_count: usize,
        tile_size: usize,
        tiles: PyReadonlyArray3<u8>,
        spawn_x: Option<i32>,
        spawn_y: Option<i32>,
        seeds: Option<Vec<u64>>,
    ) -> PyResult<Self> {
        let mut games = Vec::with_capacity(n);
        for i in 0..n {
            let seed = seeds
                .as_ref()
                .and_then(|s| s.get(i))
                .copied();
            let mut g = Game::new(
                width,
                height,
                level_grid.clone(),
                spawn_len,
                spawn_dir,
                spawn_random_dir,
                spawn_jitter,
                food_count,
                tile_size,
                tiles.clone(),
                spawn_x,
                spawn_y,
                seed,
            )?;
            g.reset(None)?;
            games.push(g);
        }
        Ok(Self { games })
    }

    fn reset(&mut self, seeds: Option<Vec<u64>>) -> PyResult<()> {
        for (i, g) in self.games.iter_mut().enumerate() {
            let seed = seeds.as_ref().and_then(|s| s.get(i)).copied();
            g.reset(seed)?;
        }
        Ok(())
    }

    fn reset_one(&mut self, index: usize, seed: Option<u64>) -> PyResult<()> {
        if index >= self.games.len() {
            return Err(pyo3::exceptions::PyIndexError::new_err("index out of bounds"));
        }
        self.games[index].reset(seed)?;
        Ok(())
    }

    fn step(&mut self, actions: Vec<u8>) -> PyResult<Vec<u32>> {
        if actions.len() != self.games.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "actions length must match num envs",
            ));
        }
        let mut out: Vec<u32> = Vec::with_capacity(self.games.len());
        for (i, g) in self.games.iter_mut().enumerate() {
            let m = g.step_internal(actions[i] as i32)?;
            out.push(m);
        }
        Ok(out)
    }

    fn tile_grids<'py>(&self, py: Python<'py>) -> Py<PyArray3<u8>> {
        let n = self.games.len();
        if n == 0 {
            let arr = ndarray::Array3::<u8>::zeros((0, 0, 0));
            return arr.into_pyarray_bound(py).unbind();
        }
        let h = self.games[0].height;
        let w = self.games[0].width;
        let mut data = Vec::with_capacity(n * h * w);
        for g in self.games.iter() {
            data.extend_from_slice(&g.tile_grid);
        }
        let arr = ndarray::Array3::from_shape_vec((n, h, w), data).unwrap();
        arr.into_pyarray_bound(py).unbind()
    }

    fn pixel_grids<'py>(&self, py: Python<'py>) -> Py<PyArray3<u8>> {
        let n = self.games.len();
        if n == 0 {
            let arr = ndarray::Array3::<u8>::zeros((0, 0, 0));
            return arr.into_pyarray_bound(py).unbind();
        }
        let h = self.games[0].height * self.games[0].tile_size;
        let w = self.games[0].width * self.games[0].tile_size;
        let mut data = Vec::with_capacity(n * h * w);
        for g in self.games.iter() {
            data.extend_from_slice(&g.pixel_grid);
        }
        let arr = ndarray::Array3::from_shape_vec((n, h, w), data).unwrap();
        arr.into_pyarray_bound(py).unbind()
    }

    fn directions(&self) -> Vec<i8> {
        self.games
            .iter()
            .map(|g| g.direction.unwrap_or(-1))
            .collect()
    }

    fn head_positions(&self) -> Vec<(i32, i32)> {
        self.games.iter().map(|g| g.head_pos()).collect()
    }

    fn scores(&self) -> Vec<i32> {
        self.games.iter().map(|g| g.score).collect()
    }

    fn running(&self) -> Vec<bool> {
        self.games.iter().map(|g| g.running).collect()
    }

    fn snake_lens(&self) -> Vec<usize> {
        self.games.iter().map(|g| g.snake.len()).collect()
    }

    fn max_playable_tiles(&self) -> usize {
        if self.games.is_empty() {
            0
        } else {
            self.games[0].max_playable_tiles()
        }
    }

    fn spawnable_counts(&self) -> Vec<usize> {
        self.games.iter().map(|g| g.spawnable_count()).collect()
    }
}

fn compute_spawn_cells(head: Point, length: usize, direction: i8) -> Vec<Point> {
    let (dx, dy) = dir_vec(direction);
    let mut out: Vec<Point> = Vec::with_capacity(length);
    for i in 0..length {
        out.push(Point {
            x: head.x - dx * (i as i32),
            y: head.y - dy * (i as i32),
        });
    }
    out
}

fn is_straight_spawn_valid(cells: &[Point], width: i32, height: i32, wall_mask: &[bool]) -> bool {
    use std::collections::HashSet;
    let mut seen: HashSet<(i32, i32)> = HashSet::new();
    for p in cells.iter() {
        if p.x < 0 || p.x >= width || p.y < 0 || p.y >= height {
            return false;
        }
        if p.x == 0 || p.x == width - 1 || p.y == 0 || p.y == height - 1 {
            return false;
        }
        let i = idx(p.x, p.y, width as usize);
        if wall_mask[i] {
            return false;
        }
        if seen.contains(&(p.x, p.y)) {
            return false;
        }
        seen.insert((p.x, p.y));
    }
    true
}

fn idx_to_point(i: usize, width: usize) -> Point {
    Point {
        x: (i % width) as i32,
        y: (i / width) as i32,
    }
}

fn head_tile(direction: i8) -> u8 {
    match direction {
        0 => TILE_HEAD_UP,
        1 => TILE_HEAD_RIGHT,
        2 => TILE_HEAD_DOWN,
        3 => TILE_HEAD_LEFT,
        _ => TILE_HEAD_RIGHT,
    }
}

fn tail_tile(prev: Point, tail: Point) -> u8 {
    if prev.x < tail.x {
        return TILE_TAIL_RIGHT;
    }
    if prev.x > tail.x {
        return TILE_TAIL_LEFT;
    }
    if prev.y < tail.y {
        return TILE_TAIL_DOWN;
    }
    if prev.y > tail.y {
        return TILE_TAIL_UP;
    }
    TILE_TAIL_RIGHT
}

fn body_tile(prev: Point, curr: Point, nxt: Point) -> u8 {
    if prev.x == nxt.x {
        if prev.y < nxt.y {
            return TILE_BODY_VERTICAL_UP;
        }
        if prev.y > nxt.y {
            return TILE_BODY_VERTICAL_DOWN;
        }
    }

    if prev.y == nxt.y {
        if prev.x < nxt.x {
            return TILE_BODY_HORIZONTAL_LEFT;
        }
        if prev.x > nxt.x {
            return TILE_BODY_HORIZONTAL_RIGHT;
        }
    }

    if (prev.x < curr.x && nxt.y > curr.y) || (nxt.x < curr.x && prev.y > curr.y) {
        return TILE_BODY_TR;
    }
    if (prev.x > curr.x && nxt.y > curr.y) || (nxt.x > curr.x && prev.y > curr.y) {
        return TILE_BODY_TL;
    }
    if (prev.x < curr.x && nxt.y < curr.y) || (nxt.x < curr.x && prev.y < curr.y) {
        return TILE_BODY_BR;
    }
    if (prev.x > curr.x && nxt.y < curr.y) || (nxt.x > curr.x && prev.y < curr.y) {
        return TILE_BODY_BL;
    }

    TILE_BODY_HORIZONTAL_RIGHT
}

#[pymodule]
fn _core(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Game>()?;
    m.add_class::<VecGame>()?;

    m.add("MOVE_OK", MOVE_OK)?;
    m.add("MOVE_FOOD", MOVE_FOOD)?;
    m.add("MOVE_HIT_BOUNDARY", MOVE_HIT_BOUNDARY)?;
    m.add("MOVE_HIT_WALL", MOVE_HIT_WALL)?;
    m.add("MOVE_HIT_SELF", MOVE_HIT_SELF)?;
    m.add("MOVE_NOT_RUNNING", MOVE_NOT_RUNNING)?;
    m.add("MOVE_TIMEOUT", MOVE_TIMEOUT)?;
    m.add("MOVE_WIN", MOVE_WIN)?;

    Ok(())
}
