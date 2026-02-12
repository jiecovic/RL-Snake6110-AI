// rust/src/py/tileset.rs

use numpy::{ndarray, IntoPyArray, PyArray3};
use pyo3::prelude::*;

use crate::engine::{tileset_tile_count, tileset_tile_names, tileset_tile_size, tileset_tiles};

#[pyfunction(name = "tileset_tile_size")]
pub(crate) fn tileset_tile_size_py() -> usize {
    tileset_tile_size()
}

#[pyfunction(name = "tileset_tiles")]
pub(crate) fn tileset_tiles_py(py: Python<'_>) -> Py<PyArray3<u8>> {
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
pub(crate) fn tileset_tile_count_py() -> usize {
    tileset_tile_count()
}

#[pyfunction(name = "tileset_tile_names")]
pub(crate) fn tileset_tile_names_py() -> Vec<String> {
    tileset_tile_names()
}

#[pyfunction(name = "pixel_frame_channels")]
pub(crate) fn pixel_frame_channels_py(add_oob_mask: bool) -> usize {
    if add_oob_mask {
        2
    } else {
        1
    }
}

#[pyfunction(name = "categorical_frame_channels")]
pub(crate) fn categorical_frame_channels_py() -> usize {
    1
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(tileset_tile_size_py, m)?)?;
    m.add_function(wrap_pyfunction!(tileset_tiles_py, m)?)?;
    m.add_function(wrap_pyfunction!(tileset_tile_count_py, m)?)?;
    m.add_function(wrap_pyfunction!(tileset_tile_names_py, m)?)?;
    m.add_function(wrap_pyfunction!(pixel_frame_channels_py, m)?)?;
    m.add_function(wrap_pyfunction!(categorical_frame_channels_py, m)?)?;
    Ok(())
}
