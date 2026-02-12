// rust/src/py/board.rs

use pyo3::prelude::*;

use crate::engine::Board as EngineBoard;
use crate::py::map_engine_err;

#[pyclass(name = "Board")]
pub(crate) struct PyBoard {
    pub(crate) inner: EngineBoard,
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
