// rust/src/py/vec_engine.rs

use numpy::{ndarray, IntoPyArray, PyArray3, PyArray4};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::ToPyObject;

use crate::engine::constants::TILE_OOB;
use crate::engine::vocab_lut;
use crate::engine::SnakeEngine as EngineSnakeEngine;
use crate::py::board::PyBoard;
use crate::py::map_engine_err;

#[pyclass(name = "VecSnakeEngine")]
pub(crate) struct PyVecSnakeEngine {
    games: Vec<EngineSnakeEngine>,
}

fn apply_lut(data: &mut [u8], lut: &[u8]) {
    for v in data.iter_mut() {
        let idx = *v as usize;
        *v = lut[idx];
    }
}

#[pymethods]
impl PyVecSnakeEngine {
    #[new]
    #[pyo3(signature = (
        n,
        board,
        food_count,
        seeds = None,
        frame_stack_n = 1,
        enable_pixel_grid = true,
        enable_world_tile_stack = true,
        enable_world_pixel_stack = true
    ))]
    fn new(
        n: usize,
        board: &PyBoard,
        food_count: usize,
        seeds: Option<Vec<u64>>,
        frame_stack_n: usize,
        enable_pixel_grid: bool,
        enable_world_tile_stack: bool,
        enable_world_pixel_stack: bool,
    ) -> PyResult<Self> {
        let mut games = Vec::with_capacity(n);
        for i in 0..n {
            let seed = seeds.as_ref().and_then(|s| s.get(i)).copied();
            let mut g = EngineSnakeEngine::new(
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

    #[pyo3(signature = (view_radius, rotate_to_head = true, oob_fill_value = 0, return_valid = false))]
    fn head_pixel_views_stacked<'py>(
        &mut self,
        py: Python<'py>,
        view_radius: (i32, i32),
        rotate_to_head: bool,
        oob_fill_value: u8,
        return_valid: bool,
    ) -> PyResult<PyObject> {
        let n_envs = self.games.len();
        if n_envs == 0 {
            let arr = ndarray::Array4::<u8>::zeros((0, 0, 0, 0));
            if return_valid {
                let v = ndarray::Array4::from_shape_vec((0, 0, 0, 0), Vec::<u8>::new()).unwrap();
                let tup = (arr.into_pyarray_bound(py).unbind(), v.into_pyarray_bound(py).unbind())
                    .to_object(py);
                return Ok(tup);
            }
            return Ok(arr.into_pyarray_bound(py).unbind().to_object(py));
        }

        let (view0, valid0, n_stack, h, w) = self.games[0]
            .head_pixel_view_stacked(view_radius.0, view_radius.1, rotate_to_head, oob_fill_value)
            .map_err(map_engine_err)?;
        let mut data = Vec::with_capacity(n_envs * n_stack * h * w);
        data.extend_from_slice(&view0);

        let mut vdata: Vec<u8> = Vec::new();
        if return_valid {
            vdata = Vec::with_capacity(n_envs * n_stack * h * w);
            vdata.extend_from_slice(&valid0);
        }

        for g in self.games.iter_mut().skip(1) {
            let (view, valid, ns, hh, ww) = g
                .head_pixel_view_stacked(view_radius.0, view_radius.1, rotate_to_head, oob_fill_value)
                .map_err(map_engine_err)?;
            if ns != n_stack || hh != h || ww != w {
                return Err(PyValueError::new_err("head_pixel_views_stacked: inconsistent shapes"));
            }
            data.extend_from_slice(&view);
            if return_valid {
                vdata.extend_from_slice(&valid);
            }
        }

        let arr = ndarray::Array4::from_shape_vec((n_envs, n_stack, h, w), data).unwrap();
        if return_valid {
            let v = ndarray::Array4::from_shape_vec((n_envs, n_stack, h, w), vdata).unwrap();
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

    #[pyo3(signature = (view_radius, rotate_to_head = true, empty_id = None))]
    fn head_tile_views_stacked<'py>(
        &mut self,
        py: Python<'py>,
        view_radius: (i32, i32),
        rotate_to_head: bool,
        empty_id: Option<u8>,
    ) -> PyResult<PyObject> {
        let n_envs = self.games.len();
        if n_envs == 0 {
            let arr = ndarray::Array4::<u8>::zeros((0, 0, 0, 0));
            return Ok(arr.into_pyarray_bound(py).unbind().to_object(py));
        }

        let empty = empty_id.unwrap_or(TILE_OOB);
        let (view0, n_stack, h, w) = self.games[0]
            .head_tile_view_stacked(view_radius.0, view_radius.1, rotate_to_head, empty)
            .map_err(map_engine_err)?;
        let mut data = Vec::with_capacity(n_envs * n_stack * h * w);
        data.extend_from_slice(&view0);

        for g in self.games.iter_mut().skip(1) {
            let (view, ns, hh, ww) = g
                .head_tile_view_stacked(view_radius.0, view_radius.1, rotate_to_head, empty)
                .map_err(map_engine_err)?;
            if ns != n_stack || hh != h || ww != w {
                return Err(PyValueError::new_err("head_tile_views_stacked: inconsistent shapes"));
            }
            data.extend_from_slice(&view);
        }

        let arr = ndarray::Array4::from_shape_vec((n_envs, n_stack, h, w), data).unwrap();
        Ok(arr.into_pyarray_bound(py).unbind().to_object(py))
    }

    #[pyo3(signature = (view_radius, rotate_to_head = true, empty_id = None, vocab = ""))]
    fn head_tile_views_stacked_vocab<'py>(
        &mut self,
        py: Python<'py>,
        view_radius: (i32, i32),
        rotate_to_head: bool,
        empty_id: Option<u8>,
        vocab: &str,
    ) -> PyResult<PyObject> {
        let n_envs = self.games.len();
        if n_envs == 0 {
            let arr = ndarray::Array4::<u8>::zeros((0, 0, 0, 0));
            return Ok(arr.into_pyarray_bound(py).unbind().to_object(py));
        }

        let lut = vocab_lut(vocab).map_err(PyValueError::new_err)?;
        let empty = empty_id.unwrap_or(TILE_OOB);
        let (view0, n_stack, h, w) = self.games[0]
            .head_tile_view_stacked(view_radius.0, view_radius.1, rotate_to_head, empty)
            .map_err(map_engine_err)?;
        let mut data = Vec::with_capacity(n_envs * n_stack * h * w);
        data.extend_from_slice(&view0);

        for g in self.games.iter_mut().skip(1) {
            let (view, ns, hh, ww) = g
                .head_tile_view_stacked(view_radius.0, view_radius.1, rotate_to_head, empty)
                .map_err(map_engine_err)?;
            if ns != n_stack || hh != h || ww != w {
                return Err(PyValueError::new_err(
                    "head_tile_views_stacked_vocab: inconsistent shapes",
                ));
            }
            data.extend_from_slice(&view);
        }
        apply_lut(&mut data, &lut);

        let arr = ndarray::Array4::from_shape_vec((n_envs, n_stack, h, w), data).unwrap();
        Ok(arr.into_pyarray_bound(py).unbind().to_object(py))
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

    fn tile_grids_stacked<'py>(&self, py: Python<'py>) -> Py<PyArray4<u8>> {
        let n = self.games.len();
        if n == 0 {
            let arr = ndarray::Array4::<u8>::zeros((0, 0, 0, 0));
            return arr.into_pyarray_bound(py).unbind();
        }
        let (view0, n_stack, h, w) = self.games[0].tile_grid_stacked();
        let mut data = Vec::with_capacity(n * n_stack * h * w);
        data.extend_from_slice(&view0);
        for g in self.games.iter().skip(1) {
            let (view, ns, hh, ww) = g.tile_grid_stacked();
            if ns != n_stack || hh != h || ww != w {
                panic!("tile_grids_stacked: inconsistent shapes");
            }
            data.extend_from_slice(&view);
        }
        let arr = ndarray::Array4::from_shape_vec((n, n_stack, h, w), data).unwrap();
        arr.into_pyarray_bound(py).unbind()
    }

    fn tile_grids_stacked_vocab<'py>(&self, py: Python<'py>, vocab: &str) -> PyResult<Py<PyArray4<u8>>> {
        let n = self.games.len();
        if n == 0 {
            let arr = ndarray::Array4::<u8>::zeros((0, 0, 0, 0));
            return Ok(arr.into_pyarray_bound(py).unbind());
        }
        let lut = vocab_lut(vocab).map_err(PyValueError::new_err)?;
        let (view0, n_stack, h, w) = self.games[0].tile_grid_stacked();
        let mut data = Vec::with_capacity(n * n_stack * h * w);
        data.extend_from_slice(&view0);
        for g in self.games.iter().skip(1) {
            let (view, ns, hh, ww) = g.tile_grid_stacked();
            if ns != n_stack || hh != h || ww != w {
                panic!("tile_grids_stacked_vocab: inconsistent shapes");
            }
            data.extend_from_slice(&view);
        }
        apply_lut(&mut data, &lut);
        let arr = ndarray::Array4::from_shape_vec((n, n_stack, h, w), data).unwrap();
        Ok(arr.into_pyarray_bound(py).unbind())
    }

    fn pixel_grids<'py>(&self, py: Python<'py>) -> PyResult<Py<PyArray3<u8>>> {
        let n = self.games.len();
        if n == 0 {
            let arr = ndarray::Array3::<u8>::zeros((0, 0, 0));
            return Ok(arr.into_pyarray_bound(py).unbind());
        }
        let h = self.games[0].height() * self.games[0].tile_size();
        let w = self.games[0].width() * self.games[0].tile_size();
        let mut data = Vec::with_capacity(n * h * w);
        for g in self.games.iter() {
            data.extend_from_slice(g.pixel_grid().map_err(map_engine_err)?);
        }
        let arr = ndarray::Array3::from_shape_vec((n, h, w), data).unwrap();
        Ok(arr.into_pyarray_bound(py).unbind())
    }

    fn pixel_grids_stacked<'py>(&self, py: Python<'py>) -> PyResult<Py<PyArray4<u8>>> {
        let n = self.games.len();
        if n == 0 {
            let arr = ndarray::Array4::<u8>::zeros((0, 0, 0, 0));
            return Ok(arr.into_pyarray_bound(py).unbind());
        }
        let (view0, n_stack, h, w) = self.games[0].pixel_grid_stacked().map_err(map_engine_err)?;
        let mut data = Vec::with_capacity(n * n_stack * h * w);
        data.extend_from_slice(&view0);
        for g in self.games.iter().skip(1) {
            let (view, ns, hh, ww) = g.pixel_grid_stacked().map_err(map_engine_err)?;
            if ns != n_stack || hh != h || ww != w {
                panic!("pixel_grids_stacked: inconsistent shapes");
            }
            data.extend_from_slice(&view);
        }
        let arr = ndarray::Array4::from_shape_vec((n, n_stack, h, w), data).unwrap();
        Ok(arr.into_pyarray_bound(py).unbind())
    }

    fn directions(&self) -> Vec<i8> {
        self.games
            .iter()
            .map(|g| g.direction().unwrap_or(-1))
            .collect()
    }

    fn direction_ids(&self) -> Vec<u8> {
        self.games.iter().map(|g| g.direction_id()).collect()
    }

    fn snake_progresses(&self) -> Vec<f32> {
        self.games.iter().map(|g| g.snake_progress()).collect()
    }

    fn steps_since_foods(&self) -> Vec<usize> {
        self.games.iter().map(|g| g.steps_since_food()).collect()
    }

    fn set_max_steps(&mut self, max_steps: Option<usize>) {
        for g in self.games.iter_mut() {
            g.set_max_steps(max_steps);
        }
    }

    fn time_since_foods_norm(&self, max_steps: usize) -> Vec<f32> {
        self.games
            .iter()
            .map(|g| g.time_since_food_norm(max_steps))
            .collect()
    }

    fn closest_foods(&self) -> Vec<(i32, i32, i32)> {
        self.games
            .iter()
            .map(|g| g.closest_food().unwrap_or((0, 0, 0)))
            .collect()
    }

    fn closest_foods_norm(&self, metric: u8) -> Vec<(f32, f32, f32)> {
        self.games
            .iter()
            .map(|g| g.closest_food_norm(metric))
            .collect()
    }

    fn collision_aheads(&self) -> Vec<bool> {
        self.games.iter().map(|g| g.collision_flags().0).collect()
    }

    fn collision_lefts(&self) -> Vec<bool> {
        self.games.iter().map(|g| g.collision_flags().1).collect()
    }

    fn collision_rights(&self) -> Vec<bool> {
        self.games.iter().map(|g| g.collision_flags().2).collect()
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
