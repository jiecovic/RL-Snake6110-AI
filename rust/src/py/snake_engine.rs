// rust/src/py/snake_engine.rs

use numpy::{ndarray, IntoPyArray, PyArray2, PyArray3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::ToPyObject;

use crate::engine::constants::TILE_OOB;
use crate::engine::vocab_lut;
use crate::engine::SnakeEngine as EngineSnakeEngine;
use crate::py::board::PyBoard;
use crate::py::map_engine_err;

#[pyclass(name = "SnakeEngine")]
pub(crate) struct PySnakeEngine {
    inner: EngineSnakeEngine,
}

fn apply_lut(data: &mut [u8], lut: &[u8]) {
    for v in data.iter_mut() {
        let idx = *v as usize;
        *v = lut[idx];
    }
}

#[pymethods]
impl PySnakeEngine {
    #[new]
    #[pyo3(signature = (
        board,
        food_count,
        seed = None,
        frame_stack_n = 1,
        enable_pixel_grid = true,
        enable_world_tile_stack = true,
        enable_world_pixel_stack = true
    ))]
    fn new(
        board: &PyBoard,
        food_count: usize,
        seed: Option<u64>,
        frame_stack_n: usize,
        enable_pixel_grid: bool,
        enable_world_tile_stack: bool,
        enable_world_pixel_stack: bool,
    ) -> PyResult<Self> {
        let game = EngineSnakeEngine::new(
            board.inner.width(),
            board.inner.height(),
            food_count,
            seed,
            frame_stack_n,
            enable_pixel_grid,
            enable_world_tile_stack,
            enable_world_pixel_stack,
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

    #[pyo3(signature = (view_radius, rotate_to_head = true, oob_fill_value = 0, return_valid = false))]
    fn head_pixel_view_stacked<'py>(
        &mut self,
        py: Python<'py>,
        view_radius: (i32, i32),
        rotate_to_head: bool,
        oob_fill_value: u8,
        return_valid: bool,
    ) -> PyResult<PyObject> {
        let (view, valid, n, h, w) = self
            .inner
            .head_pixel_view_stacked(view_radius.0, view_radius.1, rotate_to_head, oob_fill_value)
            .map_err(map_engine_err)?;
        let arr = ndarray::Array3::from_shape_vec((n, h, w), view).unwrap();
        if return_valid {
            let v = ndarray::Array3::from_shape_vec((n, h, w), valid).unwrap();
            let tup = (arr.into_pyarray_bound(py).unbind(), v.into_pyarray_bound(py).unbind())
                .to_object(py);
            Ok(tup)
        } else {
            Ok(arr.into_pyarray_bound(py).unbind().to_object(py))
        }
    }

    #[pyo3(signature = (view_radius, rotate_to_head = true, empty_id = None))]
    fn head_tile_view_stacked<'py>(
        &mut self,
        py: Python<'py>,
        view_radius: (i32, i32),
        rotate_to_head: bool,
        empty_id: Option<u8>,
    ) -> PyResult<PyObject> {
        let empty = empty_id.unwrap_or(TILE_OOB);
        let (view, n, h, w) = self
            .inner
            .head_tile_view_stacked(view_radius.0, view_radius.1, rotate_to_head, empty)
            .map_err(map_engine_err)?;
        let arr = ndarray::Array3::from_shape_vec((n, h, w), view).unwrap();
        Ok(arr.into_pyarray_bound(py).unbind().to_object(py))
    }

    #[pyo3(signature = (view_radius, rotate_to_head = true, empty_id = None, vocab = ""))]
    fn head_tile_view_stacked_vocab<'py>(
        &mut self,
        py: Python<'py>,
        view_radius: (i32, i32),
        rotate_to_head: bool,
        empty_id: Option<u8>,
        vocab: &str,
    ) -> PyResult<PyObject> {
        let lut = vocab_lut(vocab).map_err(PyValueError::new_err)?;
        let empty = empty_id.unwrap_or(TILE_OOB);
        let (view, n, h, w) = self
            .inner
            .head_tile_view_stacked(view_radius.0, view_radius.1, rotate_to_head, empty)
            .map_err(map_engine_err)?;
        let mut data = view;
        apply_lut(&mut data, &lut);
        let arr = ndarray::Array3::from_shape_vec((n, h, w), data).unwrap();
        Ok(arr.into_pyarray_bound(py).unbind().to_object(py))
    }

    fn tile_grid_stacked<'py>(&self, py: Python<'py>) -> Py<PyArray3<u8>> {
        let (view, n, h, w) = self.inner.tile_grid_stacked();
        let arr = ndarray::Array3::from_shape_vec((n, h, w), view).unwrap();
        arr.into_pyarray_bound(py).unbind()
    }

    fn tile_grid_stacked_vocab<'py>(
        &self,
        py: Python<'py>,
        vocab: &str,
    ) -> PyResult<Py<PyArray3<u8>>> {
        let lut = vocab_lut(vocab).map_err(PyValueError::new_err)?;
        let (view, n, h, w) = self.inner.tile_grid_stacked();
        let mut data = view;
        apply_lut(&mut data, &lut);
        let arr = ndarray::Array3::from_shape_vec((n, h, w), data).unwrap();
        Ok(arr.into_pyarray_bound(py).unbind())
    }

    fn pixel_grid_stacked<'py>(&self, py: Python<'py>) -> Py<PyArray3<u8>> {
        let (view, n, h, w) = self.inner.pixel_grid_stacked();
        let arr = ndarray::Array3::from_shape_vec((n, h, w), view).unwrap();
        arr.into_pyarray_bound(py).unbind()
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
        let arr = ndarray::Array2::from_shape_vec((h, w), self.inner.pixel_grid().to_vec()).unwrap();
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
    fn frame_stack_n(&self) -> usize {
        self.inner.frame_stack_n()
    }

    #[getter]
    fn score(&self) -> i32 {
        self.inner.score()
    }

    #[getter]
    fn running(&self) -> bool {
        self.inner.running()
    }

    fn episode_steps(&self) -> u64 {
        self.inner.episode_steps()
    }

    fn direction(&self) -> i8 {
        self.inner.direction().unwrap_or(-1)
    }

    fn direction_id(&self) -> u8 {
        self.inner.direction_id()
    }

    fn snake_len(&self) -> usize {
        self.inner.snake_len()
    }

    fn snake_progress(&self) -> f32 {
        self.inner.snake_progress()
    }

    fn time_since_food_norm(&self, max_steps: usize) -> f32 {
        self.inner.time_since_food_norm(max_steps)
    }

    fn head_pos(&self) -> (i32, i32) {
        self.inner.head_pos()
    }

    fn food_positions(&self) -> Vec<(i32, i32)> {
        self.inner.food_positions()
    }

    fn closest_food(&self) -> Option<(i32, i32, i32)> {
        self.inner.closest_food()
    }

    fn closest_food_norm(&self, metric: u8) -> (f32, f32, f32) {
        self.inner.closest_food_norm(metric)
    }

    fn steps_since_food(&self) -> usize {
        self.inner.steps_since_food()
    }

    fn set_max_steps(&mut self, max_steps: Option<usize>) {
        self.inner.set_max_steps(max_steps);
    }

    fn collision_flags(&self) -> (bool, bool, bool) {
        self.inner.collision_flags()
    }

    fn max_playable_tiles(&self) -> usize {
        self.inner.max_playable_tiles()
    }

    fn spawnable_count(&self) -> usize {
        self.inner.spawnable_count()
    }
}
