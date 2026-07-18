//! Deterministic allocation accounting.
//!
//! Diffpack's thesis is a fast, *incremental*, *low-memory* module graph. Wall
//! clock and OS RSS are noisy and machine-dependent; allocation counts are
//! deterministic and reproducible, so they make the sharpest regression guard.
//! [`TrackingAllocator`] wraps the system allocator and records live bytes, a
//! high-water mark, and cumulative totals with relaxed atomics (negligible,
//! uniform overhead). The guard suite asserts on these to catch a change that
//! starts retaining every AST, leaking per-edit revisions, or ballooning peak
//! memory.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

static LIVE_BYTES: AtomicUsize = AtomicUsize::new(0);
static PEAK_BYTES: AtomicUsize = AtomicUsize::new(0);
static TOTAL_BYTES: AtomicUsize = AtomicUsize::new(0);
static TOTAL_ALLOCS: AtomicUsize = AtomicUsize::new(0);

/// A `System`-backed global allocator that tallies live, peak, and cumulative
/// allocation so the guard suite can measure memory deterministically.
pub struct TrackingAllocator;

impl TrackingAllocator {
    fn on_alloc(size: usize) {
        TOTAL_BYTES.fetch_add(size, Ordering::Relaxed);
        TOTAL_ALLOCS.fetch_add(1, Ordering::Relaxed);
        let live = LIVE_BYTES.fetch_add(size, Ordering::Relaxed) + size;
        PEAK_BYTES.fetch_max(live, Ordering::Relaxed);
    }

    fn on_free(size: usize) {
        LIVE_BYTES.fetch_sub(size, Ordering::Relaxed);
    }
}

// SAFETY: every branch defers the actual (de)allocation to the system allocator
// with the exact same pointer/layout it was given; the atomic bookkeeping only
// reads and updates counters and never touches the returned memory.
unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            Self::on_alloc(layout.size());
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        Self::on_free(layout.size());
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            Self::on_alloc(layout.size());
        }
        ptr
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_ptr = unsafe { System.realloc(ptr, layout, new_size) };
        if !new_ptr.is_null() {
            let old_size = layout.size();
            if new_size >= old_size {
                Self::on_alloc(new_size - old_size);
            } else {
                Self::on_free(old_size - new_size);
            }
        }
        new_ptr
    }
}

/// A point-in-time view of allocation counters.
#[derive(Debug, Clone, Copy)]
pub struct MemorySnapshot {
    /// Bytes currently allocated and not yet freed.
    pub live_bytes: usize,
    /// The largest `live_bytes` seen since the last [`reset_peak`].
    pub peak_bytes: usize,
    /// Cumulative bytes ever allocated.
    pub total_bytes: usize,
    /// Cumulative number of allocations.
    pub total_allocs: usize,
}

/// Reads the current allocation counters.
pub fn snapshot() -> MemorySnapshot {
    MemorySnapshot {
        live_bytes: LIVE_BYTES.load(Ordering::Relaxed),
        peak_bytes: PEAK_BYTES.load(Ordering::Relaxed),
        total_bytes: TOTAL_BYTES.load(Ordering::Relaxed),
        total_allocs: TOTAL_ALLOCS.load(Ordering::Relaxed),
    }
}

/// Resets the high-water mark to the current live total, so a subsequent
/// [`snapshot`] reports the peak of just the region that follows.
pub fn reset_peak() {
    PEAK_BYTES.store(LIVE_BYTES.load(Ordering::Relaxed), Ordering::Relaxed);
}
