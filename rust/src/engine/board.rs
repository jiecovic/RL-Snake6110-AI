// rust/src/engine/board.rs

use super::EngineError;

pub struct Board {
    width: usize,
    height: usize,
}

impl Board {
    pub fn new(width: usize, height: usize) -> Result<Self, EngineError> {
        if width == 0 || height == 0 {
            return Err(EngineError::InvalidGridLen);
        }
        Ok(Self { width, height })
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }
}
