// rust/src/engine/obs/stack.rs

#[derive(Debug)]
pub struct FrameStacker {
    n_stack: usize,
    frame_len: usize,
    buf: Vec<u8>,
    head: usize,
    stacked: Vec<u8>,
    dirty: bool,
}

impl FrameStacker {
    pub fn new(n_stack: usize) -> Self {
        Self {
            n_stack: n_stack.max(1),
            frame_len: 0,
            buf: Vec::new(),
            head: 0,
            stacked: Vec::new(),
            dirty: true,
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
        self.head = self.n_stack.saturating_sub(1);
        self.stacked.clone_from(&self.buf);
        self.dirty = false;
    }

    pub fn push(&mut self, frame: &[u8]) {
        let len = frame.len();
        if self.frame_len != len || self.buf.len() != self.n_stack * len {
            self.reset_with(frame);
            return;
        }

        if self.n_stack <= 1 {
            self.buf[..len].copy_from_slice(frame);
            self.head = 0;
            self.stacked.clone_from(&self.buf);
            self.dirty = false;
            return;
        }

        self.head = (self.head + 1) % self.n_stack;
        let dst = self.head * len;
        self.buf[dst..dst + len].copy_from_slice(frame);
        self.dirty = true;
    }

    pub fn stacked(&mut self) -> &[u8] {
        if self.frame_len == 0 || self.buf.is_empty() {
            self.stacked.clear();
            self.dirty = false;
            return &self.stacked;
        }
        if self.n_stack <= 1 {
            if self.dirty {
                self.stacked.clone_from(&self.buf);
                self.dirty = false;
            }
            return &self.stacked;
        }
        if self.dirty {
            let len = self.frame_len;
            if self.stacked.len() != self.buf.len() {
                self.stacked.resize(self.buf.len(), 0);
            }
            let start = (self.head + 1) % self.n_stack;
            for i in 0..self.n_stack {
                let src_slot = (start + i) % self.n_stack;
                let src = src_slot * len;
                let dst = i * len;
                self.stacked[dst..dst + len].copy_from_slice(&self.buf[src..src + len]);
            }
            self.dirty = false;
        }
        &self.stacked
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

    pub fn stacked(&mut self) -> &[u8] {
        self.stacker.stacked()
    }

}
