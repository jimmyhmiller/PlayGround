/// Address type distinguishing return addresses from instruction pointers
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameAddress {
    /// Return address - the address we will return to (points after call instruction)
    ReturnAddress(u64),
    /// Instruction pointer - the actual instruction being executed (first frame only)
    InstructionPointer(u64),
}

impl FrameAddress {
    /// Get the raw address value
    #[inline]
    pub fn address(&self) -> u64 {
        match self {
            FrameAddress::ReturnAddress(addr) => *addr,
            FrameAddress::InstructionPointer(addr) => *addr,
        }
    }

    /// Convert to an instruction pointer for symbol lookup
    /// (subtracts 1 from return addresses to point within the call instruction)
    #[inline]
    pub fn to_lookup_address(&self) -> u64 {
        match self {
            FrameAddress::ReturnAddress(addr) => addr.saturating_sub(1),
            FrameAddress::InstructionPointer(addr) => *addr,
        }
    }

    /// Check if this is a return address
    #[inline]
    pub fn is_return_address(&self) -> bool {
        matches!(self, FrameAddress::ReturnAddress(_))
    }
}

/// How was this frame unwound?
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnwindMethod {
    /// First frame - registers taken directly
    InitialFrame,
    /// Frame pointer chain was followed
    FramePointer,
}

/// A single stack frame
#[derive(Debug, Clone)]
pub struct Frame {
    /// Frame index (0 = innermost/current frame)
    pub index: usize,

    /// Address of this frame (return address or IP)
    pub address: FrameAddress,

    /// Stack pointer at this frame
    pub stack_pointer: u64,

    /// Frame pointer at this frame (if available)
    pub frame_pointer: Option<u64>,

    /// How this frame was unwound
    pub unwind_method: UnwindMethod,

    /// Is this frame considered reliable?
    pub is_reliable: bool,
}

impl Frame {
    /// Create a new frame
    pub fn new(
        index: usize,
        address: FrameAddress,
        stack_pointer: u64,
        frame_pointer: Option<u64>,
        unwind_method: UnwindMethod,
    ) -> Self {
        Self {
            index,
            address,
            stack_pointer,
            frame_pointer,
            unwind_method,
            is_reliable: true,
        }
    }

    /// Get the address for symbol lookup (adjusts return addresses)
    #[inline]
    pub fn lookup_address(&self) -> u64 {
        self.address.to_lookup_address()
    }

    /// Get the raw address
    #[inline]
    pub fn raw_address(&self) -> u64 {
        self.address.address()
    }
}

/// Complete stack trace
#[derive(Debug, Clone)]
pub struct StackTrace {
    /// Frames from innermost to outermost
    pub frames: Vec<Frame>,

    /// Was the walk terminated early?
    pub truncated: bool,

    /// Reason for truncation (if any)
    pub truncation_reason: Option<String>,
}

impl StackTrace {
    /// Create a new empty stack trace
    pub fn new() -> Self {
        Self {
            frames: Vec::new(),
            truncated: false,
            truncation_reason: None,
        }
    }

    /// Create with pre-allocated capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            frames: Vec::with_capacity(capacity),
            truncated: false,
            truncation_reason: None,
        }
    }

    /// Add a frame to the trace
    pub fn push(&mut self, frame: Frame) {
        self.frames.push(frame);
    }

    /// Get the number of frames
    #[inline]
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// Check if the trace is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Iterate over frames
    pub fn iter(&self) -> impl Iterator<Item = &Frame> {
        self.frames.iter()
    }

    /// Mark the trace as truncated with a reason
    pub fn truncate_with_reason(&mut self, reason: impl Into<String>) {
        self.truncated = true;
        self.truncation_reason = Some(reason.into());
    }
}

impl Default for StackTrace {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> IntoIterator for &'a StackTrace {
    type Item = &'a Frame;
    type IntoIter = std::slice::Iter<'a, Frame>;

    fn into_iter(self) -> Self::IntoIter {
        self.frames.iter()
    }
}

impl IntoIterator for StackTrace {
    type Item = Frame;
    type IntoIter = std::vec::IntoIter<Frame>;

    fn into_iter(self) -> Self::IntoIter {
        self.frames.into_iter()
    }
}
