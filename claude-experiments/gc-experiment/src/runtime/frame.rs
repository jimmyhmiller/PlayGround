use std::ffi::c_void;

/// Metadata about a function's frame - how many roots it contains.
/// This is generated as a constant in the compiled code and referenced by Frame.
#[repr(C)]
pub struct FrameOrigin {
    /// Number of GC root slots in frames for this function
    pub num_roots: u32,
    /// Function name for debugging (null-terminated C string)
    pub function_name: *const u8,
}

impl FrameOrigin {
    pub fn name(&self) -> &str {
        if self.function_name.is_null() {
            "<unknown>"
        } else {
            unsafe {
                let len = libc_strlen(self.function_name);
                std::str::from_utf8_unchecked(std::slice::from_raw_parts(self.function_name, len))
            }
        }
    }
}

fn libc_strlen(s: *const u8) -> usize {
    let mut len = 0;
    unsafe {
        while *s.add(len) != 0 {
            len += 1;
        }
    }
    len
}

/// A stack frame in the GC frame chain.
///
/// Each function that can hold GC roots allocates one of these on its stack.
/// The frames form a linked list from Thread.top_frame back through callers.
///
/// Layout in memory (what the generated code creates):
/// ```
/// struct ActualFrame {
///     parent: *mut Frame,      // offset 0
///     origin: *const FrameOrigin, // offset 8
///     roots: [*mut Object; N], // offset 16, where N = origin.num_roots
/// }
/// ```
///
/// We can't express variable-length arrays in Rust, so Frame is just the header.
/// The roots array follows immediately after in memory.
#[repr(C)]
pub struct Frame {
    /// Pointer to the caller's frame (or null if this is the bottom)
    pub parent: *mut Frame,
    /// Metadata about this frame (number of roots, function name)
    pub origin: *const FrameOrigin,
    // roots: [*mut c_void; ?] follows in memory
}

impl Frame {
    /// Get the number of root slots in this frame
    pub fn num_roots(&self) -> u32 {
        if self.origin.is_null() {
            0
        } else {
            unsafe { (*self.origin).num_roots }
        }
    }

    /// Get a pointer to the roots array (immediately follows the Frame header)
    pub fn roots_ptr(&self) -> *const *mut c_void {
        unsafe {
            let frame_end = (self as *const Frame).add(1);
            frame_end as *const *mut c_void
        }
    }

    /// Get a mutable pointer to the roots array
    pub fn roots_ptr_mut(&mut self) -> *mut *mut c_void {
        unsafe {
            let frame_end = (self as *mut Frame).add(1);
            frame_end as *mut *mut c_void
        }
    }

    /// Iterate over all roots in this frame (returns values)
    pub fn roots(&self) -> FrameRootsIter {
        FrameRootsIter {
            ptr: self.roots_ptr(),
            remaining: self.num_roots() as usize,
        }
    }

    /// Iterate over all root slots in this frame (returns slot addresses and values)
    pub fn root_slots(&self) -> FrameRootSlotsIter {
        FrameRootSlotsIter {
            ptr: self.roots_ptr() as *mut *mut c_void,
            remaining: self.num_roots() as usize,
        }
    }

    /// Get the function name for debugging
    pub fn function_name(&self) -> &str {
        if self.origin.is_null() {
            "<unknown>"
        } else {
            unsafe { (*self.origin).name() }
        }
    }
}

/// Iterator over roots in a single frame
pub struct FrameRootsIter {
    ptr: *const *mut c_void,
    remaining: usize,
}

impl Iterator for FrameRootsIter {
    type Item = *mut c_void;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            None
        } else {
            let root = unsafe { *self.ptr };
            self.ptr = unsafe { self.ptr.add(1) };
            self.remaining -= 1;
            Some(root)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl ExactSizeIterator for FrameRootsIter {}

/// Iterator over root slots in a single frame (returns slot address and value)
pub struct FrameRootSlotsIter {
    ptr: *mut *mut c_void,
    remaining: usize,
}

impl Iterator for FrameRootSlotsIter {
    /// Returns (slot_address, value)
    type Item = (*mut *mut c_void, *mut c_void);

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            None
        } else {
            let slot_addr = self.ptr;
            let value = unsafe { *self.ptr };
            self.ptr = unsafe { self.ptr.add(1) };
            self.remaining -= 1;
            Some((slot_addr, value))
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl ExactSizeIterator for FrameRootSlotsIter {}

/// Iterator over all frames in a thread's frame chain
pub struct FrameChainIter {
    current: *mut Frame,
}

impl FrameChainIter {
    pub fn new(top_frame: *mut Frame) -> Self {
        Self { current: top_frame }
    }
}

impl Iterator for FrameChainIter {
    type Item = &'static Frame;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current.is_null() {
            None
        } else {
            let frame = unsafe { &*self.current };
            self.current = frame.parent;
            Some(frame)
        }
    }
}
