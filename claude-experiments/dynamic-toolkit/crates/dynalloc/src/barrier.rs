use std::sync::Mutex;

// ─── SATB Write Barrier Buffer ──────────────────────────────────────

/// Thread-local buffer for Snapshot-At-The-Beginning (SATB) write barriers.
///
/// When a mutator overwrites a pointer field during concurrent GC, it
/// logs the *old* value here before the store. This ensures the GC
/// sees all objects that were reachable at the start of the cycle —
/// even if the mutator disconnects them.
///
/// Each mutator thread has its own `SATBBuffer`. The GC drains all
/// buffers during or after the concurrent phase to process any
/// objects the mutator "hid" from the collector.
///
/// Uses a simple Vec<u64> with a flush threshold. When the buffer
/// fills up, it should be flushed to a global queue for the GC to
/// process.
pub struct SATBBuffer {
    /// Local log of old pointer values (pre-write snapshots).
    buf: Vec<u64>,
    /// Max entries before suggesting a flush.
    capacity: usize,
}

impl SATBBuffer {
    /// Create a new SATB buffer with the given capacity.
    pub fn new(capacity: usize) -> Self {
        SATBBuffer {
            buf: Vec::with_capacity(capacity),
            capacity,
        }
    }

    /// Log an old value before it is overwritten.
    ///
    /// This is the write barrier's hot path — called before every
    /// pointer store during concurrent GC. Must be fast.
    #[inline(always)]
    pub fn log(&mut self, old_value: u64) {
        self.buf.push(old_value);
    }

    /// Check if the buffer is at or over capacity and should be flushed.
    #[inline(always)]
    pub fn should_flush(&self) -> bool {
        self.buf.len() >= self.capacity
    }

    /// Drain all logged values, returning them.
    pub fn drain(&mut self) -> Vec<u64> {
        std::mem::take(&mut self.buf)
    }

    /// Number of entries currently buffered.
    pub fn len(&self) -> usize {
        self.buf.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }
}

// ─── Global SATB Queue ──────────────────────────────────────────────

/// Global queue that collects flushed SATB buffers from all threads.
///
/// Thread-safe: mutator threads push their full buffers here,
/// and the GC thread drains them during collection.
pub struct SATBQueue {
    /// Flushed buffers waiting for GC processing.
    queue: Mutex<Vec<Vec<u64>>>,
}

impl SATBQueue {
    pub fn new() -> Self {
        SATBQueue {
            queue: Mutex::new(Vec::new()),
        }
    }

    /// Push a flushed buffer from a mutator thread.
    pub fn push(&self, buf: Vec<u64>) {
        if buf.is_empty() {
            return;
        }
        self.queue.lock().unwrap().push(buf);
    }

    /// Drain all queued buffers for GC processing.
    /// Returns a flat iterator of all logged values.
    pub fn drain_all(&self) -> Vec<u64> {
        let mut queue = self.queue.lock().unwrap();
        let buffers = std::mem::take(&mut *queue);
        buffers.into_iter().flatten().collect()
    }

    /// Check if there are any pending buffers.
    pub fn is_empty(&self) -> bool {
        self.queue.lock().unwrap().is_empty()
    }
}

// ─── Read Barrier ───────────────────────────────────────────────────

/// Read barrier: follow forwarding pointer if the object has been relocated.
///
/// During concurrent GC, an object may have been copied to to-space
/// and a forwarding pointer installed in the old location. The read
/// barrier ensures the mutator always accesses the most up-to-date copy.
///
/// # Safety
/// - `ptr` must point to a valid heap object (or be null).
/// - `type_info_offset` must be the correct offset for the header type.
#[inline(always)]
pub unsafe fn read_barrier(ptr: *mut u8, type_info_offset: usize) -> *mut u8 {
    if ptr.is_null() {
        return ptr;
    }
    let slot = unsafe { ptr.add(type_info_offset) as *const u64 };
    let word = unsafe { *slot };
    if word & crate::semi_space::FORWARDING_BIT != 0 {
        (word & !crate::semi_space::FORWARDING_BIT) as *mut u8
    } else {
        // No forwarding — object is in place.
        ptr
    }
}

/// Atomic read barrier using atomic load for concurrent safety.
///
/// Use this when the GC might be concurrently installing forwarding
/// pointers. Uses `Acquire` ordering to ensure we see the complete
/// copied object after following the forwarding pointer.
///
/// # Safety
/// Same as `read_barrier`.
#[inline(always)]
pub unsafe fn read_barrier_atomic(ptr: *mut u8, type_info_offset: usize) -> *mut u8 {
    if ptr.is_null() {
        return ptr;
    }
    let slot = unsafe { ptr.add(type_info_offset) as *const std::sync::atomic::AtomicU64 };
    let word = unsafe { (*slot).load(std::sync::atomic::Ordering::Acquire) };
    if word & crate::semi_space::FORWARDING_BIT != 0 {
        (word & !crate::semi_space::FORWARDING_BIT) as *mut u8
    } else {
        ptr
    }
}
