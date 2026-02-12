// rust/src/lib.rs
#![allow(unsafe_op_in_unsafe_fn)]

mod engine;
mod py;

use pyo3::prelude::*;

#[pymodule]
fn _core(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    py::register(m)
}
