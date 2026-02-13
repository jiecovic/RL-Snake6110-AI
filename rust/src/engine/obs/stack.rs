// rust/src/engine/obs/stack.rs

#[derive(Debug)]
pub struct FrameStacker {
    n_stack: usize,
    frame_len: usize,
    buf: Vec<u8>,
}

impl FrameStacker {
    pub fn new(n_stack: usize) -> Self {
        Self {
            n_stack: n_stack.max(1),
            frame_len: 0,
            buf: Vec::new(),
        }
    }

    pub fn reset_with(&mut self, frame: &[u8]) {
        let len = frame.len();
        self.frame_len = len;
        self.buf.resize(self.n_stack * len, 0);
        for i in 0..self.n_stack {
            let dst = i * len;
            self.buf[dst..dst + len].copy_from_slice(frame);
        }
    }

    pub fn push(&mut self, frame: &[u8]) {
        let len = frame.len();
        if self.frame_len != len || self.buf.len() != self.n_stack * len {
            self.reset_with(frame);
            return;
        }

        if self.n_stack <= 1 {
            self.buf[..len].copy_from_slice(frame);
            return;
        }

        let shift = len;
        self.buf.copy_within(shift.., 0);
        let dst = (self.n_stack - 1) * len;
        self.buf[dst..dst + len].copy_from_slice(frame);
    }

    pub fn stacked(&self) -> &[u8] {
        &self.buf
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HeadViewKey {
    pub ry: i32,
    pub rx: i32,
    pub rotate: bool,
    pub param: u8,
}

#[derive(Debug)]
pub struct HeadStacker {
    key: Option<HeadViewKey>,
    last_step: u64,
    stacker: FrameStacker,
}

impl HeadStacker {
    pub fn new(n_stack: usize) -> Self {
        Self {
            key: None,
            last_step: 0,
            stacker: FrameStacker::new(n_stack),
        }
    }

    pub fn update(&mut self, step_id: u64, key: HeadViewKey, frame: &[u8]) {
        if self.key != Some(key) {
            self.key = Some(key);
            self.last_step = step_id;
            self.stacker.reset_with(frame);
            return;
        }

        if step_id == self.last_step {
            return;
        }

        self.last_step = step_id;
        self.stacker.push(frame);
    }

    pub fn stacked(&self) -> &[u8] {
        self.stacker.stacked()
    }

}
