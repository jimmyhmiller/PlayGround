use std::ffi::c_void;
use crate::runtime::frame::{Frame, FrameChainIter};

/// Thread state flags
pub const THREAD_STATE_CHECK_REQUESTED: u8 = 1;
pub const THREAD_STATE_STOP_REQUESTED: u8 = 2;

/// Per-thread GC state.
///
/// In a real implementation, this would be accessed via thread-local storage.
/// For simplicity, we pass it explicitly as the first parameter to all functions.
#[repr(C)]
pub struct Thread {
    /// Flags for GC synchronization (CHECK_REQUESTED, STOP_REQUESTED, etc.)
    pub state: u8,

    /// Padding to align top_frame to 8 bytes
    pub _padding: [u8; 7],

    /// Head of the frame chain - points to the current function's frame
    pub top_frame: *mut Frame,

    /// Pointer to the heap (for allocation)
    pub heap: *mut c_void,
}

// Verify offsets match what codegen expects
const _: () = {
    assert!(std::mem::offset_of!(Thread, state) == 0);
    assert!(std::mem::offset_of!(Thread, top_frame) == 8);
    assert!(std::mem::offset_of!(Thread, heap) == 16);
};

impl Thread {
    /// Create a new thread with no frames
    pub fn new(heap: *mut c_void) -> Self {
        Self {
            state: 0,
            _padding: [0; 7],
            top_frame: std::ptr::null_mut(),
            heap,
        }
    }

    /// Check if the GC has requested this thread to check in
    pub fn check_requested(&self) -> bool {
        self.state & THREAD_STATE_CHECK_REQUESTED != 0
    }

    /// Check if the GC has requested this thread to stop
    pub fn stop_requested(&self) -> bool {
        self.state & THREAD_STATE_STOP_REQUESTED != 0
    }

    /// Iterate over all frames in this thread's stack
    pub fn frames(&self) -> FrameChainIter {
        FrameChainIter::new(self.top_frame)
    }

    /// Walk the stack and collect all non-null GC roots.
    /// This is the key function that demonstrates stack walking works.
    pub fn collect_roots(&self) -> Vec<*mut c_void> {
        let mut roots = Vec::new();
        for frame in self.frames() {
            for root in frame.roots() {
                if !root.is_null() {
                    roots.push(root);
                }
            }
        }
        roots
    }

    /// Walk the stack and call a callback for each non-null root.
    /// This is what the GC would use for marking.
    pub fn visit_roots<F>(&self, mut visitor: F)
    where
        F: FnMut(*mut c_void),
    {
        for frame in self.frames() {
            for root in frame.roots() {
                if !root.is_null() {
                    visitor(root);
                }
            }
        }
    }

    /// Debug: print the current stack with all roots
    pub fn dump_stack(&self) {
        println!("=== Stack Dump ===");
        for (i, frame) in self.frames().enumerate() {
            println!("Frame {}: {} ({} roots)", i, frame.function_name(), frame.num_roots());
            for (j, root) in frame.roots().enumerate() {
                if root.is_null() {
                    println!("  root[{}]: null", j);
                } else {
                    println!("  root[{}]: {:p}", j, root);
                }
            }
        }
        println!("=== End Stack Dump ===");
    }
}

/// Thread offsets for codegen - these must match the struct layout
pub mod offsets {
    pub const STATE: u64 = 0;
    pub const TOP_FRAME: u64 = 8;
    pub const HEAP: u64 = 16;
}
