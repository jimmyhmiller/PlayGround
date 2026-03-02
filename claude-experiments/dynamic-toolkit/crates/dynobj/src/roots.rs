use std::cell::Cell;
use std::sync::Mutex;

/// Trait for types that can enumerate GC root references.
///
/// The visitor receives `*mut u64` so the GC can update slots in-place
/// (e.g., for moving/forwarding pointers). `&self` enables composability —
/// multiple root sources can be scanned through shared references, behind
/// `Arc`, etc. Interior mutability (`Cell` for per-thread, `Mutex` for shared)
/// makes this safe.
pub trait RootSource {
    fn scan_roots(&self, visitor: &mut dyn FnMut(*mut u64));
}

// Blanket impl: scan a slice of root sources.
impl<T: RootSource> RootSource for [T] {
    fn scan_roots(&self, visitor: &mut dyn FnMut(*mut u64)) {
        for source in self {
            source.scan_roots(visitor);
        }
    }
}

// ─── ShadowStack ─────────────────────────────────────────────────────

/// Per-thread root storage for compiled code.
///
/// Pre-allocated `u64` buffer organized into frames. The compiler emits
/// push/pop calls in function prologues/epilogues.
///
/// Frame layout (grows upward from base):
/// ```text
/// [prev_fp_offset: u64, slot_count: u64, slot[0], ..., slot[N-1]]
/// ```
///
/// `Send` (can be moved to a thread), `!Sync` (owned by one thread).
/// The raw `*mut u64` buffer means `scan_roots(&self, ...)` works naturally.
#[repr(C)]
pub struct ShadowStack {
    /// Offset (in u64 slots) to the current frame header. 0 when empty.
    fp: usize,
    /// Offset (in u64 slots) to the next free position.
    sp: usize,
    /// Pointer to the buffer.
    base: *mut u64,
    /// Buffer capacity in u64 slots.
    capacity: usize,
}

// ShadowStack owns its buffer and contains no references — Send is safe.
// NOT Sync: only the owning thread should access it.
unsafe impl Send for ShadowStack {}

impl ShadowStack {
    /// Create a new shadow stack with the given capacity (in u64 slots).
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "ShadowStack capacity must be > 0");
        let layout = std::alloc::Layout::array::<u64>(capacity).unwrap();
        let base = unsafe { std::alloc::alloc_zeroed(layout) as *mut u64 };
        assert!(!base.is_null(), "ShadowStack allocation failed");
        ShadowStack {
            fp: 0,
            sp: 0,
            base,
            capacity,
        }
    }

    /// Push a new frame with `slot_count` root slots.
    ///
    /// Panics if there isn't enough capacity for the frame header + slots.
    pub fn push_frame(&mut self, slot_count: usize) {
        let frame_size = 2 + slot_count; // prev_fp + count + slots
        assert!(
            self.sp + frame_size <= self.capacity,
            "ShadowStack overflow: need {} slots but only {} remain",
            frame_size,
            self.capacity - self.sp
        );

        let frame_start = self.sp;
        unsafe {
            // Write prev_fp_offset
            *self.base.add(frame_start) = self.fp as u64;
            // Write slot_count
            *self.base.add(frame_start + 1) = slot_count as u64;
            // Zero-initialize the slots
            for i in 0..slot_count {
                *self.base.add(frame_start + 2 + i) = 0;
            }
        }
        self.fp = frame_start;
        self.sp = frame_start + frame_size;
    }

    /// Pop the current frame, restoring the previous frame pointer.
    ///
    /// Panics if there is no frame to pop.
    pub fn pop_frame(&mut self) {
        assert!(self.sp > 0, "ShadowStack underflow: no frame to pop");

        let prev_fp = unsafe { *self.base.add(self.fp) } as usize;
        let slot_count = unsafe { *self.base.add(self.fp + 1) } as usize;

        // sp goes back to the start of this frame
        self.sp = self.fp;
        // fp goes back to the previous frame
        self.fp = prev_fp;

        // If we just popped the last frame, fp should be 0 and sp should be 0
        // (prev_fp of the first frame is always 0)
        let _ = slot_count;
    }

    /// Returns true if no frames are pushed.
    pub fn is_empty(&self) -> bool {
        self.sp == 0
    }

    /// Set a slot in the current frame.
    ///
    /// Panics if `index` is out of bounds for the current frame.
    pub fn set(&mut self, index: usize, bits: u64) {
        assert!(self.sp > 0, "ShadowStack: no frame to set slot in");
        let slot_count = unsafe { *self.base.add(self.fp + 1) } as usize;
        assert!(
            index < slot_count,
            "ShadowStack: slot index {} out of bounds (frame has {} slots)",
            index,
            slot_count
        );
        unsafe {
            *self.base.add(self.fp + 2 + index) = bits;
        }
    }

    /// Get a slot from the current frame.
    ///
    /// Panics if `index` is out of bounds for the current frame.
    pub fn get(&self, index: usize) -> u64 {
        assert!(self.sp > 0, "ShadowStack: no frame to get slot from");
        let slot_count = unsafe { *self.base.add(self.fp + 1) } as usize;
        assert!(
            index < slot_count,
            "ShadowStack: slot index {} out of bounds (frame has {} slots)",
            index,
            slot_count
        );
        unsafe { *self.base.add(self.fp + 2 + index) }
    }

    /// Create an RAII frame guard that pops the frame on drop.
    pub fn frame(&mut self, slot_count: usize) -> ShadowFrame<'_> {
        self.push_frame(slot_count);
        ShadowFrame { stack: self }
    }
}

impl RootSource for ShadowStack {
    fn scan_roots(&self, visitor: &mut dyn FnMut(*mut u64)) {
        if self.sp == 0 {
            return;
        }

        // Walk frames from the current one backward
        let mut fp = self.fp;
        loop {
            let prev_fp = unsafe { *self.base.add(fp) } as usize;
            let slot_count = unsafe { *self.base.add(fp + 1) } as usize;

            for i in 0..slot_count {
                let slot_ptr = unsafe { self.base.add(fp + 2 + i) };
                visitor(slot_ptr);
            }

            if fp == 0 {
                // This was the first frame (prev_fp stored as 0 for the bottom frame,
                // but fp itself is also 0 for the first frame pushed at offset 0).
                break;
            }
            // Check: if prev_fp == fp we'd loop forever. But that can't happen
            // because push_frame always sets prev_fp to the *previous* fp value.
            // The first frame's prev_fp is 0 and it starts at offset 0, so
            // we detect "first frame" by checking if prev_fp points to the same
            // frame or if we've reached offset 0.
            if prev_fp == fp {
                break;
            }
            fp = prev_fp;
        }
    }
}

impl Drop for ShadowStack {
    fn drop(&mut self) {
        let layout = std::alloc::Layout::array::<u64>(self.capacity).unwrap();
        unsafe {
            std::alloc::dealloc(self.base as *mut u8, layout);
        }
    }
}

/// RAII guard that pops a shadow stack frame on drop.
pub struct ShadowFrame<'a> {
    stack: &'a mut ShadowStack,
}

impl<'a> ShadowFrame<'a> {
    pub fn set(&mut self, index: usize, bits: u64) {
        self.stack.set(index, bits);
    }

    pub fn get(&self, index: usize) -> u64 {
        self.stack.get(index)
    }
}

impl<'a> Drop for ShadowFrame<'a> {
    fn drop(&mut self) {
        self.stack.pop_frame();
    }
}

// ─── ShadowStack FFI ─────────────────────────────────────────────────

/// Push a frame with `slot_count` slots onto the shadow stack.
///
/// # Safety
/// `stack` must point to a valid `ShadowStack`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn shadow_stack_push(stack: *mut ShadowStack, slot_count: usize) {
    unsafe { (*stack).push_frame(slot_count) }
}

/// Pop the current frame from the shadow stack.
///
/// # Safety
/// `stack` must point to a valid `ShadowStack` with at least one frame.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn shadow_stack_pop(stack: *mut ShadowStack) {
    unsafe { (*stack).pop_frame() }
}

/// Set slot `index` in the current frame to `bits`.
///
/// # Safety
/// `stack` must point to a valid `ShadowStack` with a current frame,
/// and `index` must be in bounds.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn shadow_stack_set(stack: *mut ShadowStack, index: usize, bits: u64) {
    unsafe { (*stack).set(index, bits) }
}

/// Get slot `index` from the current frame.
///
/// # Safety
/// `stack` must point to a valid `ShadowStack` with a current frame,
/// and `index` must be in bounds.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn shadow_stack_get(stack: *const ShadowStack, index: usize) -> u64 {
    unsafe { (*stack).get(index) }
}

// ─── RootStack ───────────────────────────────────────────────────────

/// Per-thread root storage for interpreters and Rust runtime code.
///
/// Growable `Vec<Cell<u64>>` with watermark-based scoping.
/// `Cell<u64>` enables `scan_roots(&self, ...)` — `Cell::as_ptr()` returns
/// `*mut u64` from a shared reference.
///
/// `Send` (can be moved to a thread), `!Sync` (owned by one thread).
pub struct RootStack {
    slots: Vec<Cell<u64>>,
}

impl RootStack {
    pub fn new() -> Self {
        RootStack { slots: Vec::new() }
    }

    /// Push a value and return its index.
    pub fn push(&mut self, bits: u64) -> usize {
        let index = self.slots.len();
        self.slots.push(Cell::new(bits));
        index
    }

    /// Get the value at `index`.
    ///
    /// Panics if `index` is out of bounds.
    pub fn get(&self, index: usize) -> u64 {
        self.slots[index].get()
    }

    /// Set the value at `index`.
    ///
    /// Panics if `index` is out of bounds.
    pub fn set(&self, index: usize, bits: u64) {
        self.slots[index].set(bits);
    }

    /// Return the current watermark (number of slots).
    /// Save this before pushing temporary roots, then call `truncate` to pop them.
    pub fn watermark(&self) -> usize {
        self.slots.len()
    }

    /// Truncate back to a saved watermark, removing all slots pushed after it.
    ///
    /// Panics if `watermark` is greater than the current length.
    pub fn truncate(&mut self, watermark: usize) {
        assert!(
            watermark <= self.slots.len(),
            "RootStack::truncate: watermark {} > length {}",
            watermark,
            self.slots.len()
        );
        self.slots.truncate(watermark);
    }

    /// Return the number of active slots.
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    /// Return true if empty.
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }
}

impl Default for RootStack {
    fn default() -> Self {
        Self::new()
    }
}

impl RootSource for RootStack {
    fn scan_roots(&self, visitor: &mut dyn FnMut(*mut u64)) {
        for cell in &self.slots {
            visitor(cell.as_ptr());
        }
    }
}

// ─── PinnedRoots ─────────────────────────────────────────────────────

/// Thread-safe root storage for long-lived values (globals, constants).
///
/// Uses `Mutex<Vec<u64>>` — the lock gives `&mut` access during scanning.
///
/// `Send + Sync`. Lock is acquired during both mutation and scanning.
pub struct PinnedRoots {
    slots: Mutex<Vec<u64>>,
}

impl PinnedRoots {
    pub fn new() -> Self {
        PinnedRoots {
            slots: Mutex::new(Vec::new()),
        }
    }

    /// Pin a value and return its index.
    pub fn pin(&self, bits: u64) -> usize {
        let mut slots = self.slots.lock().unwrap();
        let index = slots.len();
        slots.push(bits);
        index
    }

    /// Get the value at `index`.
    ///
    /// Panics if `index` is out of bounds.
    pub fn get(&self, index: usize) -> u64 {
        let slots = self.slots.lock().unwrap();
        slots[index]
    }

    /// Set the value at `index`.
    ///
    /// Panics if `index` is out of bounds.
    pub fn set(&self, index: usize, bits: u64) {
        let mut slots = self.slots.lock().unwrap();
        slots[index] = bits;
    }
}

impl Default for PinnedRoots {
    fn default() -> Self {
        Self::new()
    }
}

impl RootSource for PinnedRoots {
    fn scan_roots(&self, visitor: &mut dyn FnMut(*mut u64)) {
        let mut slots = self.slots.lock().unwrap();
        for slot in slots.iter_mut() {
            visitor(slot as *mut u64);
        }
    }
}
