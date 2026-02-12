// rust/src/py/vocab.rs

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::engine::{vocab_class_names, vocab_classes, vocab_defs, vocab_num_classes};

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

#[pyfunction(name = "tile_vocab_names")]
pub(crate) fn tile_vocab_names_py() -> PyResult<Vec<String>> {
    let mut names: Vec<String> = vocab_defs().iter().map(|d| d.name.to_string()).collect();
    names.sort();
    Ok(names)
}

#[pyfunction(name = "tile_vocab_num_classes")]
pub(crate) fn tile_vocab_num_classes_py(name: &str) -> PyResult<usize> {
    vocab_num_classes(name).map_err(|e| PyValueError::new_err(e))
}

#[pyfunction(name = "tile_vocab_class_names")]
pub(crate) fn tile_vocab_class_names_py(name: &str) -> PyResult<Vec<String>> {
    vocab_class_names(name).map_err(|e| PyValueError::new_err(e))
}

#[pyfunction(name = "tile_vocab_classes")]
pub(crate) fn tile_vocab_classes_py(py: Python<'_>, name: &str) -> PyResult<Vec<PyObject>> {
    let classes = vocab_classes(name).map_err(|e| PyValueError::new_err(e))?;
    let mut out: Vec<PyObject> = Vec::with_capacity(classes.len());
    for (cname, members) in classes {
        let tup = (cname, members).to_object(py);
        out.push(tup);
    }
    Ok(out)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(tile_vocab_defs_py, m)?)?;
    m.add_function(wrap_pyfunction!(tile_vocab_names_py, m)?)?;
    m.add_function(wrap_pyfunction!(tile_vocab_num_classes_py, m)?)?;
    m.add_function(wrap_pyfunction!(tile_vocab_class_names_py, m)?)?;
    m.add_function(wrap_pyfunction!(tile_vocab_classes_py, m)?)?;
    Ok(())
}
