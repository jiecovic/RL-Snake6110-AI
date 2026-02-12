// rust/src/py/mod.rs

mod board;
mod snake_engine;
mod tileset;
mod vec_engine;
mod vocab;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate::engine::constants::*;
use crate::engine::EngineError;

pub(crate) fn map_engine_err(err: EngineError) -> PyErr {
    match err {
        EngineError::DirectionNone => PyRuntimeError::new_err(err.to_string()),
        _ => PyValueError::new_err(err.to_string()),
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<board::PyBoard>()?;
    m.add_class::<snake_engine::PySnakeEngine>()?;
    m.add_class::<vec_engine::PyVecSnakeEngine>()?;

    tileset::register(m)?;
    vocab::register(m)?;

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
