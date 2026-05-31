//! Process-global GC root table for Var root values.
//!
//! A `Var` is an indirection cell: code never bakes a Var's value, it bakes
//! a reference to the Var and dereferences at runtime. The value a Var holds,
//! when it is a GC-heap pointer (a string, a closure, a cons, …), must be a
//! genuine GC root so the collector forwards it when the object moves.
//!
//! This table provides that, exactly the way the JIT call table provides
//! stable indirection for functions: every Var owns a stable slot index, its
//! root value lives in that slot as a NanBox, and the table is registered with
//! the GC as a [`RootSource`]. On every collection the GC scans each live slot
//! and rewrites moved pointers in place, so a Var deref always reads the
//! current location of its value.
//!
//! Storage is a chunked array of `AtomicU64`:
//!   * **Chunked** so growth never relocates existing slots — a slot's address
//!     is stable for the program's lifetime, and the GC may hold a `*mut u64`
//!     into it across the whole scan.
//!   * **`AtomicU64`** so concurrent `bind_root` from multiple mutator threads
//!     and the GC's atomic slot access (see `Heap::process_slot`) never race.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{LazyLock, RwLock};

use dynobj::RootSource;

/// Slots per chunk. A chunk is a heap-stable `Box<[AtomicU64]>`; once
/// allocated its address never changes, so slot pointers stay valid across
/// table growth.
const CHUNK: usize = 4096;

/// Growable, GC-rooted table of Var root values (NanBox-encoded).
pub struct VarRoots {
    /// Each chunk holds `CHUNK` slots. New chunks are appended under the
    /// write lock; existing chunks are never moved or freed.
    chunks: RwLock<Vec<Box<[AtomicU64]>>>,
    /// Number of slots handed out by `alloc_slot`. Slots `0..len` are live.
    len: AtomicUsize,
}

// The slots are `AtomicU64`, so concurrent access is sound; the `RwLock` only
// guards chunk-vector growth. The contained pointers are opaque NanBoxes.
unsafe impl Sync for VarRoots {}
unsafe impl Send for VarRoots {}

impl VarRoots {
    fn new() -> Self {
        VarRoots { chunks: RwLock::new(Vec::new()), len: AtomicUsize::new(0) }
    }

    /// Reserve a fresh slot, growing the table by a chunk if needed. The
    /// slot starts at `0` (decodes as the double `0.0`, never a heap
    /// pointer, so an unused slot is inert to the GC). Returns the slot
    /// index, stable for the Var's lifetime.
    pub fn alloc_slot(&self) -> usize {
        let idx = self.len.fetch_add(1, Ordering::AcqRel);
        let chunk = idx / CHUNK;
        // Fast path: the chunk already exists.
        {
            let chunks = self.chunks.read().unwrap();
            if chunk < chunks.len() {
                return idx;
            }
        }
        // Grow. Multiple threads may race here; the loop is idempotent under
        // the write lock (only append chunks that don't exist yet).
        let mut chunks = self.chunks.write().unwrap();
        while chunks.len() <= chunk {
            let c: Box<[AtomicU64]> =
                (0..CHUNK).map(|_| AtomicU64::new(0)).collect();
            chunks.push(c);
        }
        idx
    }

    /// Read the NanBox currently stored in `idx`.
    pub fn get(&self, idx: usize) -> u64 {
        let chunks = self.chunks.read().unwrap();
        chunks[idx / CHUNK][idx % CHUNK].load(Ordering::Acquire)
    }

    /// Store `bits` into `idx`.
    pub fn set(&self, idx: usize, bits: u64) {
        let chunks = self.chunks.read().unwrap();
        chunks[idx / CHUNK][idx % CHUNK].store(bits, Ordering::Release);
    }
}

impl RootSource for VarRoots {
    fn scan_roots(&self, visitor: &mut dyn FnMut(*mut u64)) {
        let chunks = self.chunks.read().unwrap();
        let len = self.len.load(Ordering::Acquire);
        for idx in 0..len {
            let slot = &chunks[idx / CHUNK][idx % CHUNK];
            // `AtomicU64::as_ptr` yields `*mut u64`; the GC accesses it
            // atomically (see `Heap::process_slot`), matching our storage.
            visitor(slot.as_ptr());
        }
    }
}

/// The one process-global table. Vars persist for the program's lifetime
/// (they live in the namespace registry), so slots are never reclaimed —
/// the same monotonic model the namespace registry already uses.
pub static VAR_ROOTS: LazyLock<VarRoots> = LazyLock::new(VarRoots::new);

/// A `'static` `RootSource` handle for registering the table with a GC via
/// `DynGcRuntime::register_extra_root_source`. Valid for the whole program.
pub fn var_roots_root_source() -> *const dyn RootSource {
    &*VAR_ROOTS as *const dyn RootSource
}
