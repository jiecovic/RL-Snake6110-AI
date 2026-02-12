// rust/src/py/vocab.rs

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::engine::vocab_defs;

#[pyfunction(name = "tile_vocab_defs")]
pub(crate) fn tile_vocab_defs_py(py: Python<'_>) -> PyResult<Vec<PyObject>> {
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

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(tile_vocab_defs_py, m)?)?;
    Ok(())
}
