// rust/src/lib.rs
#![allow(unsafe_op_in_unsafe_fn)]

mod engine;

use engine::constants::*;
use engine::{
    Board as EngineBoard, EngineError, MOVE_FOOD, MOVE_HIT_BOUNDARY, MOVE_HIT_SELF, MOVE_HIT_WALL,
    MOVE_NOT_RUNNING, MOVE_OK, MOVE_TIMEOUT, MOVE_WIN, SnakeEngine as EngineSnakeEngine,
    tileset_tile_count, tileset_tile_names, tileset_tile_size, tileset_tiles, vocab_defs,
};
use numpy::{IntoPyArray, PyArray2, PyArray3, ndarray};
use pyo3::ToPyObject;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

fn map_engine_err(err: EngineError) -> PyErr {
    match err {
        EngineError::DirectionNone => PyRuntimeError::new_err(err.to_string()),
        _ => PyValueError::new_err(err.to_string()),
    }
}

#[pyclass(name = "Board")]
struct PyBoard {
    inner: EngineBoard,
}

#[pymethods]
impl PyBoard {
    #[new]
    fn new(width: usize, height: usize) -> PyResult<Self> {
        let inner = EngineBoard::new(width, height).map_err(map_engine_err)?;
        Ok(Self { inner })
    }

    #[getter]
    fn width(&self) -> usize {
        self.inner.width()
    }

    #[getter]
    fn height(&self) -> usize {
        self.inner.height()
    }
}

#[pyclass(name = "SnakeEngine")]
struct PySnakeEngine {
    inner: EngineSnakeEngine,
}

#[pymethods]
impl PySnakeEngine {
    #[new]
    #[pyo3(signature = (board, food_count, seed = None))]
    fn new(board: &PyBoard, food_count: usize, seed: Option<u64>) -> PyResult<Self> {
        let game = EngineSnakeEngine::new(
            board.inner.width(),
            board.inner.height(),
            food_count,
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

    fn step_relative(&mut self, rel_dir: u8) -> PyResult<u32> {
        self.inner.step(rel_dir as i32).map_err(map_engine_err)
    }

    fn step_cardinal(&mut self, abs_dir: u8) -> PyResult<u32> {
        self.inner.step_cardinal(abs_dir as i32).map_err(map_engine_err)
    }

    #[pyo3(signature = (view_radius, rotate_to_head = true, oob_fill_value = 0, return_valid = false))]
    fn head_pixel_view<'py>(
        &self,
        py: Python<'py>,
        view_radius: (i32, i32),
        rotate_to_head: bool,
        oob_fill_value: u8,
        return_valid: bool,
    ) -> PyResult<PyObject> {
        let (view, valid, h, w) = self
            .inner
            .head_pixel_view(view_radius.0, view_radius.1, rotate_to_head, oob_fill_value)
            .map_err(map_engine_err)?;
        let arr = ndarray::Array2::from_shape_vec((h, w), view).unwrap();
        if return_valid {
            let v = ndarray::Array2::from_shape_vec((h, w), valid).unwrap();
            let tup = (arr.into_pyarray_bound(py).unbind(), v.into_pyarray_bound(py).unbind())
                .to_object(py);
            Ok(tup)
        } else {
            Ok(arr.into_pyarray_bound(py).unbind().to_object(py))
        }
    }

    #[pyo3(signature = (view_radius, rotate_to_head = true, empty_id = None, return_valid = false))]
    fn head_tile_view<'py>(
        &self,
        py: Python<'py>,
        view_radius: (i32, i32),
        rotate_to_head: bool,
        empty_id: Option<u8>,
        return_valid: bool,
    ) -> PyResult<PyObject> {
        let empty = empty_id.unwrap_or(TILE_OOB);
        let (view, valid, h, w) = self
            .inner
            .head_tile_view(view_radius.0, view_radius.1, rotate_to_head, empty)
            .map_err(map_engine_err)?;
        let arr = ndarray::Array2::from_shape_vec((h, w), view).unwrap();
        if return_valid {
            let v = ndarray::Array2::from_shape_vec((h, w), valid).unwrap();
            let tup = (arr.into_pyarray_bound(py).unbind(), v.into_pyarray_bound(py).unbind())
                .to_object(py);
            Ok(tup)
        } else {
            Ok(arr.into_pyarray_bound(py).unbind().to_object(py))
        }
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

#[pyclass(name = "VecSnakeEngine")]
struct PyVecSnakeEngine {
    games: Vec<EngineSnakeEngine>,
}

#[pymethods]
impl PyVecSnakeEngine {
    #[new]
    #[pyo3(signature = (n, board, food_count, seeds = None))]
    fn new(
        n: usize,
        board: &PyBoard,
        food_count: usize,
        seeds: Option<Vec<u64>>,
    ) -> PyResult<Self> {
        let mut games = Vec::with_capacity(n);
        for i in 0..n {
            let seed = seeds.as_ref().and_then(|s| s.get(i)).copied();
            let mut g = EngineSnakeEngine::new(
                board.inner.width(),
                board.inner.height(),
                food_count,
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

    fn step_cardinal(&mut self, actions: Vec<u8>) -> PyResult<Vec<u32>> {
        if actions.len() != self.games.len() {
            return Err(PyValueError::new_err("actions length must match num envs"));
        }
        let mut out: Vec<u32> = Vec::with_capacity(self.games.len());
        for (i, g) in self.games.iter_mut().enumerate() {
            let m = g.step_cardinal(actions[i] as i32).map_err(map_engine_err)?;
            out.push(m);
        }
        Ok(out)
    }

    #[pyo3(signature = (view_radius, rotate_to_head = true, oob_fill_value = 0, return_valid = false))]
    fn head_pixel_views<'py>(
        &self,
        py: Python<'py>,
        view_radius: (i32, i32),
        rotate_to_head: bool,
        oob_fill_value: u8,
        return_valid: bool,
    ) -> PyResult<PyObject> {
        let n = self.games.len();
        if n == 0 {
            let arr = ndarray::Array3::<u8>::zeros((0, 0, 0));
            if return_valid {
                let v = ndarray::Array3::from_shape_vec((0, 0, 0), Vec::<bool>::new()).unwrap();
                let tup = (arr.into_pyarray_bound(py).unbind(), v.into_pyarray_bound(py).unbind())
                    .to_object(py);
                return Ok(tup);
            }
            return Ok(arr.into_pyarray_bound(py).unbind().to_object(py));
        }

        let (view0, valid0, h, w) = self.games[0]
            .head_pixel_view(view_radius.0, view_radius.1, rotate_to_head, oob_fill_value)
            .map_err(map_engine_err)?;
        let mut data = Vec::with_capacity(n * h * w);
        data.extend_from_slice(&view0);

        let mut vdata: Vec<bool> = Vec::new();
        if return_valid {
            vdata = Vec::with_capacity(n * h * w);
            vdata.extend_from_slice(&valid0);
        }

        for g in self.games.iter().skip(1) {
            let (view, valid, hh, ww) =
                g.head_pixel_view(view_radius.0, view_radius.1, rotate_to_head, oob_fill_value)
                    .map_err(map_engine_err)?;
            if hh != h || ww != w {
                return Err(PyValueError::new_err("head_pixel_views: inconsistent shapes"));
            }
            data.extend_from_slice(&view);
            if return_valid {
                vdata.extend_from_slice(&valid);
            }
        }

        let arr = ndarray::Array3::from_shape_vec((n, h, w), data).unwrap();
        if return_valid {
            let v = ndarray::Array3::from_shape_vec((n, h, w), vdata).unwrap();
            let tup = (arr.into_pyarray_bound(py).unbind(), v.into_pyarray_bound(py).unbind())
                .to_object(py);
            Ok(tup)
        } else {
            Ok(arr.into_pyarray_bound(py).unbind().to_object(py))
        }
    }

    #[pyo3(signature = (view_radius, rotate_to_head = true, empty_id = None, return_valid = false))]
    fn head_tile_views<'py>(
        &self,
        py: Python<'py>,
        view_radius: (i32, i32),
        rotate_to_head: bool,
        empty_id: Option<u8>,
        return_valid: bool,
    ) -> PyResult<PyObject> {
        let n = self.games.len();
        if n == 0 {
            let arr = ndarray::Array3::<u8>::zeros((0, 0, 0));
            if return_valid {
                let v = ndarray::Array3::from_shape_vec((0, 0, 0), Vec::<bool>::new()).unwrap();
                let tup = (arr.into_pyarray_bound(py).unbind(), v.into_pyarray_bound(py).unbind())
                    .to_object(py);
                return Ok(tup);
            }
            return Ok(arr.into_pyarray_bound(py).unbind().to_object(py));
        }

        let empty = empty_id.unwrap_or(TILE_OOB);
        let (view0, valid0, h, w) = self.games[0]
            .head_tile_view(view_radius.0, view_radius.1, rotate_to_head, empty)
            .map_err(map_engine_err)?;
        let mut data = Vec::with_capacity(n * h * w);
        data.extend_from_slice(&view0);

        let mut vdata: Vec<bool> = Vec::new();
        if return_valid {
            vdata = Vec::with_capacity(n * h * w);
            vdata.extend_from_slice(&valid0);
        }

        for g in self.games.iter().skip(1) {
            let (view, valid, hh, ww) =
                g.head_tile_view(view_radius.0, view_radius.1, rotate_to_head, empty)
                    .map_err(map_engine_err)?;
            if hh != h || ww != w {
                return Err(PyValueError::new_err("head_tile_views: inconsistent shapes"));
            }
            data.extend_from_slice(&view);
            if return_valid {
                vdata.extend_from_slice(&valid);
            }
        }

        let arr = ndarray::Array3::from_shape_vec((n, h, w), data).unwrap();
        if return_valid {
            let v = ndarray::Array3::from_shape_vec((n, h, w), vdata).unwrap();
            let tup = (arr.into_pyarray_bound(py).unbind(), v.into_pyarray_bound(py).unbind())
                .to_object(py);
            Ok(tup)
        } else {
            Ok(arr.into_pyarray_bound(py).unbind().to_object(py))
        }
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

#[pyfunction(name = "tileset_tile_size")]
fn tileset_tile_size_py() -> usize {
    tileset_tile_size()
}

#[pyfunction(name = "tileset_tiles")]
fn tileset_tiles_py(py: Python<'_>) -> Py<PyArray3<u8>> {
    let tiles = tileset_tiles();
    let tile_count = tiles.len();
    let tile_size = tileset_tile_size();
    let mut data = Vec::with_capacity(tile_count * tile_size * tile_size);
    for t in tiles.iter() {
        data.extend_from_slice(t);
    }
    let arr = ndarray::Array3::from_shape_vec((tile_count, tile_size, tile_size), data).unwrap();
    arr.into_pyarray_bound(py).unbind()
}

#[pyfunction(name = "tileset_tile_count")]
fn tileset_tile_count_py() -> usize {
    tileset_tile_count()
}

#[pyfunction(name = "tileset_tile_names")]
fn tileset_tile_names_py() -> Vec<String> {
    tileset_tile_names()
}

#[pyfunction(name = "tile_vocab_defs")]
fn tile_vocab_defs_py(py: Python<'_>) -> PyResult<Vec<PyObject>> {
    let mut out: Vec<PyObject> = Vec::new();
    for def in vocab_defs() {
        let dict = PyDict::new_bound(py);
        dict.set_item("name", def.name)?;

        let mut classes: Vec<PyObject> = Vec::with_capacity(def.classes.len());
        for class_def in def.classes {
            let members: Vec<String> = class_def.members.iter().map(|s| s.to_string()).collect();
            let tup = (class_def.name.to_string(), members).to_object(py);
            classes.push(tup);
        }

        dict.set_item("classes", classes)?;
        out.push(dict.into());
    }
    Ok(out)
}

#[pymodule]
fn _core(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBoard>()?;
    m.add_class::<PySnakeEngine>()?;
    m.add_class::<PyVecSnakeEngine>()?;

    m.add_function(wrap_pyfunction!(tileset_tile_size_py, m)?)?;
    m.add_function(wrap_pyfunction!(tileset_tiles_py, m)?)?;
    m.add_function(wrap_pyfunction!(tileset_tile_count_py, m)?)?;
    m.add_function(wrap_pyfunction!(tileset_tile_names_py, m)?)?;
    m.add_function(wrap_pyfunction!(tile_vocab_defs_py, m)?)?;

    m.add("MOVE_OK", MOVE_OK)?;
    m.add("MOVE_FOOD", MOVE_FOOD)?;
    m.add("MOVE_HIT_BOUNDARY", MOVE_HIT_BOUNDARY)?;
    m.add("MOVE_HIT_WALL", MOVE_HIT_WALL)?;
    m.add("MOVE_HIT_SELF", MOVE_HIT_SELF)?;
    m.add("MOVE_NOT_RUNNING", MOVE_NOT_RUNNING)?;
    m.add("MOVE_TIMEOUT", MOVE_TIMEOUT)?;
    m.add("MOVE_WIN", MOVE_WIN)?;

    m.add("TILE_OOB", TILE_OOB)?;
    m.add("TILE_EMPTY", TILE_EMPTY)?;
    m.add("TILE_WALL_TL", TILE_WALL_TL)?;
    m.add("TILE_WALL_TR", TILE_WALL_TR)?;
    m.add("TILE_WALL_BL", TILE_WALL_BL)?;
    m.add("TILE_WALL_BR", TILE_WALL_BR)?;
    m.add("TILE_WALL_TOP", TILE_WALL_TOP)?;
    m.add("TILE_WALL_BOTTOM", TILE_WALL_BOTTOM)?;
    m.add("TILE_WALL_LEFT", TILE_WALL_LEFT)?;
    m.add("TILE_WALL_RIGHT", TILE_WALL_RIGHT)?;
    m.add("TILE_HEAD_UP", TILE_HEAD_UP)?;
    m.add("TILE_HEAD_DOWN", TILE_HEAD_DOWN)?;
    m.add("TILE_HEAD_LEFT", TILE_HEAD_LEFT)?;
    m.add("TILE_HEAD_RIGHT", TILE_HEAD_RIGHT)?;
    m.add("TILE_BODY_VERTICAL_UP", TILE_BODY_VERTICAL_UP)?;
    m.add("TILE_BODY_VERTICAL_DOWN", TILE_BODY_VERTICAL_DOWN)?;
    m.add("TILE_BODY_HORIZONTAL_LEFT", TILE_BODY_HORIZONTAL_LEFT)?;
    m.add("TILE_BODY_HORIZONTAL_RIGHT", TILE_BODY_HORIZONTAL_RIGHT)?;
    m.add("TILE_BODY_BR", TILE_BODY_BR)?;
    m.add("TILE_BODY_BL", TILE_BODY_BL)?;
    m.add("TILE_BODY_TR", TILE_BODY_TR)?;
    m.add("TILE_BODY_TL", TILE_BODY_TL)?;
    m.add("TILE_TAIL_UP", TILE_TAIL_UP)?;
    m.add("TILE_TAIL_DOWN", TILE_TAIL_DOWN)?;
    m.add("TILE_TAIL_LEFT", TILE_TAIL_LEFT)?;
    m.add("TILE_TAIL_RIGHT", TILE_TAIL_RIGHT)?;
    m.add("TILE_FOOD", TILE_FOOD)?;

    Ok(())
}
