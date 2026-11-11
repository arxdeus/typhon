//! Port of `codon/cir/util/context.h`.

/// Port of `util::CIRContext<Frame>`: a simple frame stack.
#[derive(Clone, Debug)]
pub struct CirContext<Frame> {
    frames: Vec<Frame>,
}

impl<Frame> Default for CirContext<Frame> {
    fn default() -> Self {
        Self { frames: Vec::new() }
    }
}

impl<Frame> CirContext<Frame> {
    pub fn new() -> Self {
        Self::default()
    }
    /// `emplaceFrame(args...)`.
    pub fn emplace_frame(&mut self, f: Frame) {
        self.frames.push(f);
    }
    /// `replaceFrame(newFrame)`.
    pub fn replace_frame(&mut self, f: Frame) {
        self.frames.pop();
        self.frames.push(f);
    }
    pub fn frames(&self) -> &Vec<Frame> {
        &self.frames
    }
    pub fn frames_mut(&mut self) -> &mut Vec<Frame> {
        &mut self.frames
    }
    /// `getFrame()` (top of the stack).
    pub fn frame(&self) -> &Frame {
        self.frames.last().expect("empty context")
    }
    pub fn frame_mut(&mut self) -> &mut Frame {
        self.frames.last_mut().expect("empty context")
    }
    pub fn pop_frame(&mut self) {
        self.frames.pop();
    }
}
