// rust\src\lib.rs

mod engine;

use engine::{
    EngineError, GameCore, MOVE_FOOD, MOVE_HIT_BOUNDARY, MOVE_HIT_SELF, MOVE_HIT_WALL,
    MOVE_NOT_RUNNING, MOVE_OK, MOVE_TIMEOUT, MOVE_WIN,
};
use numpy::{IntoPyArray, PyArray2, PyArray3, PyReadonlyArray2, PyReadonlyArray3, ndarray};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

fn map_engine_err(err: EngineError) -> PyErr {
    match err {
        EngineError::DirectionNone => PyRuntimeError::new_err(err.to_string()),
        _ => PyValueError::new_err(err.to_string()),
    }
}

fn build_static_grid(
    width: usize,
    height: usize,
    level_grid: PyReadonlyArray2<u8>,
) -> PyResult<Vec<u8>> {
    let grid = level_grid.as_array();
    if grid.shape() != [height, width] {
        return Err(PyValueError::new_err(
            "level_grid shape does not match height/width",
        ));
    }
    Ok(grid.iter().copied().collect())
}

fn build_tile_cache(tile_size: usize, tiles: PyReadonlyArray3<u8>) -> PyResult<Vec<Vec<u8>>> {
    let tiles_arr = tiles.as_array();
    let tiles_shape = tiles_arr.shape();
    if tiles_shape.len() != 3 {
        return Err(PyValueError::new_err("tiles must be a 3D array"));
    }
    if tiles_shape[1] != tile_size || tiles_shape[2] != tile_size {
        return Err(PyValueError::new_err("tiles tile_size mismatch"));
    }

    let mut tile_cache: Vec<Vec<u8>> = Vec::with_capacity(tiles_shape[0]);
    for i in 0..tiles_shape[0] {
        let tile = tiles_arr.index_axis(ndarray::Axis(0), i);
        tile_cache.push(tile.iter().copied().collect());
    }
    Ok(tile_cache)
}

#[pyclass]
struct Game {
    inner: GameCore,
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
    #[allow(clippy::too_many_arguments)]
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
        let static_grid = build_static_grid(width, height, level_grid)?;
        let tile_cache = build_tile_cache(tile_size, tiles)?;
        let spawn_dir = if spawn_dir < 0 { None } else { Some(spawn_dir) };

        let game = GameCore::new(
            width,
            height,
            static_grid,
            spawn_x,
            spawn_y,
            spawn_len,
            spawn_dir,
            spawn_random_dir,
            spawn_jitter,
            food_count,
            tile_size,
            tile_cache,
            seed,
        )
        .map_err(map_engine_err)?;

        Ok(Self { inner: game })
    }

    fn reset(&mut self, seed: Option<u64>) -> PyResult<()> {
        self.inner.reset(seed).map_err(map_engine_err)
    }

    fn step(&mut self, rel_dir: u8) -> PyResult<u32> {
        self.inner.step(rel_dir as i32).map_err(map_engine_err)
    }

    fn tile_grid<'py>(&self, py: Python<'py>) -> Py<PyArray2<u8>> {
        let arr = ndarray::Array2::from_shape_vec(
            (self.inner.height(), self.inner.width()),
            self.inner.tile_grid().to_vec(),
        )
        .unwrap();
        arr.into_pyarray_bound(py).unbind()
    }

    fn pixel_grid<'py>(&self, py: Python<'py>) -> Py<PyArray2<u8>> {
        let h = self.inner.height() * self.inner.tile_size();
        let w = self.inner.width() * self.inner.tile_size();
        let arr =
            ndarray::Array2::from_shape_vec((h, w), self.inner.pixel_grid().to_vec()).unwrap();
        arr.into_pyarray_bound(py).unbind()
    }

    #[getter]
    fn width(&self) -> usize {
        self.inner.width()
    }

    #[getter]
    fn height(&self) -> usize {
        self.inner.height()
    }

    #[getter]
    fn tile_size(&self) -> usize {
        self.inner.tile_size()
    }

    #[getter]
    fn score(&self) -> i32 {
        self.inner.score()
    }

    #[getter]
    fn running(&self) -> bool {
        self.inner.running()
    }

    fn direction(&self) -> i8 {
        self.inner.direction().unwrap_or(-1)
    }

    fn snake_len(&self) -> usize {
        self.inner.snake_len()
    }

    fn head_pos(&self) -> (i32, i32) {
        self.inner.head_pos()
    }

    fn food_positions(&self) -> Vec<(i32, i32)> {
        self.inner.food_positions()
    }

    fn max_playable_tiles(&self) -> usize {
        self.inner.max_playable_tiles()
    }

    fn spawnable_count(&self) -> usize {
        self.inner.spawnable_count()
    }
}

#[pyclass]
struct VecGame {
    games: Vec<GameCore>,
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
    #[allow(clippy::too_many_arguments)]
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
        let static_grid = build_static_grid(width, height, level_grid)?;
        let tile_cache = build_tile_cache(tile_size, tiles)?;
        let spawn_dir = if spawn_dir < 0 { None } else { Some(spawn_dir) };

        let mut games = Vec::with_capacity(n);
        for i in 0..n {
            let seed = seeds.as_ref().and_then(|s| s.get(i)).copied();
            let mut g = GameCore::new(
                width,
                height,
                static_grid.clone(),
                spawn_x,
                spawn_y,
                spawn_len,
                spawn_dir,
                spawn_random_dir,
                spawn_jitter,
                food_count,
                tile_size,
                tile_cache.clone(),
                seed,
            )
            .map_err(map_engine_err)?;
            g.reset(None).map_err(map_engine_err)?;
            games.push(g);
        }
        Ok(Self { games })
    }

    fn reset(&mut self, seeds: Option<Vec<u64>>) -> PyResult<()> {
        for (i, g) in self.games.iter_mut().enumerate() {
            let seed = seeds.as_ref().and_then(|s| s.get(i)).copied();
            g.reset(seed).map_err(map_engine_err)?;
        }
        Ok(())
    }

    fn reset_one(&mut self, index: usize, seed: Option<u64>) -> PyResult<()> {
        if index >= self.games.len() {
            return Err(PyValueError::new_err("index out of bounds"));
        }
        self.games[index].reset(seed).map_err(map_engine_err)?;
        Ok(())
    }

    fn step(&mut self, actions: Vec<u8>) -> PyResult<Vec<u32>> {
        if actions.len() != self.games.len() {
            return Err(PyValueError::new_err("actions length must match num envs"));
        }
        let mut out: Vec<u32> = Vec::with_capacity(self.games.len());
        for (i, g) in self.games.iter_mut().enumerate() {
            let m = g.step(actions[i] as i32).map_err(map_engine_err)?;
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
        let h = self.games[0].height();
        let w = self.games[0].width();
        let mut data = Vec::with_capacity(n * h * w);
        for g in self.games.iter() {
            data.extend_from_slice(g.tile_grid());
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
        let h = self.games[0].height() * self.games[0].tile_size();
        let w = self.games[0].width() * self.games[0].tile_size();
        let mut data = Vec::with_capacity(n * h * w);
        for g in self.games.iter() {
            data.extend_from_slice(g.pixel_grid());
        }
        let arr = ndarray::Array3::from_shape_vec((n, h, w), data).unwrap();
        arr.into_pyarray_bound(py).unbind()
    }

    fn directions(&self) -> Vec<i8> {
        self.games
            .iter()
            .map(|g| g.direction().unwrap_or(-1))
            .collect()
    }

    fn head_positions(&self) -> Vec<(i32, i32)> {
        self.games.iter().map(|g| g.head_pos()).collect()
    }

    fn scores(&self) -> Vec<i32> {
        self.games.iter().map(|g| g.score()).collect()
    }

    fn running(&self) -> Vec<bool> {
        self.games.iter().map(|g| g.running()).collect()
    }

    fn snake_lens(&self) -> Vec<usize> {
        self.games.iter().map(|g| g.snake_len()).collect()
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
