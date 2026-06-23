//! `memscope-core` — a tracking [`GlobalAlloc`] wrapper that maintains a live
//! heap table, interns allocation sites (stack traces), and emits an event
//! stream, with a runtime-switchable Full / Sampled / Off mode.
//!
//! It records only the type-erased facts available at allocation time (size,
//! align, address, stack trace). Turning a site's stack trace into a concrete
//! Rust type is `memscope-symbols`' job, done off the hot path against the
//! binary's DWARF. See the crate docs / `project-rust-memory-allocator`.
//!
//! # Reentrancy
//! Symbolication, the interner, and even the live table allocate. To avoid
//! recursing into ourselves, every allocation made *while recording* is routed
//! straight to the inner allocator and never recorded. This is enforced by a
//! per-thread [`HookScope`] guard, used both on the hot path and around the
//! public query API (which also allocates while holding internal locks).
//!
//! # Usage
//! ```ignore
//! use memscope_core::MemScope;
//! #[global_allocator]
//! static GLOBAL: MemScope = MemScope::system();
//!
//! fn main() {
//!     memscope_core::set_mode(memscope_core::Mode::Full);
//!     // ... run workload ...
//!     let dump = memscope_core::snapshot();
//! }
//! ```

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

mod recorder;
mod unwind;

pub use memscope_proto::{EventKind, RawEvent, SiteId, Snapshot};
pub use recorder::{Mode, Stats, MAX_FRAMES};
use recorder::recorder;
use unwind::{DefaultUnwind, Unwind};

thread_local! {
    /// Set while we are inside our own bookkeeping. Const-initialized so reading
    /// it never allocates.
    static IN_HOOK: Cell<bool> = const { Cell::new(false) };
}

/// RAII reentrancy guard. On enter, records the previous flag and sets it; on
/// drop, restores the previous value (so it nests correctly).
struct HookScope {
    prev: bool,
}

impl HookScope {
    #[inline]
    fn enter() -> Self {
        let prev = IN_HOOK.with(|x| x.replace(true));
        HookScope { prev }
    }
    /// Were we already inside the recorder when this scope was entered?
    #[inline]
    fn was_active(&self) -> bool {
        self.prev
    }
}

impl Drop for HookScope {
    #[inline]
    fn drop(&mut self) {
        let p = self.prev;
        IN_HOOK.with(|x| x.set(p));
    }
}

/// The default stack unwinder. Behind a function so the strategy can be swapped
/// later without touching the hot path call sites.
static UNWIND: DefaultUnwind = DefaultUnwind;

#[inline]
fn capture(buf: &mut [usize], skip: usize) -> usize {
    UNWIND.capture(buf, skip)
}

/// A tracking global allocator wrapping an inner [`GlobalAlloc`] (default
/// [`System`]; wrap jemalloc/mimalloc by constructing with [`MemScope::new`]).
pub struct MemScope<A = System> {
    inner: A,
}

impl MemScope<System> {
    /// A tracking allocator over the system allocator.
    pub const fn system() -> Self {
        MemScope { inner: System }
    }
}

impl<A> MemScope<A> {
    pub const fn new(inner: A) -> Self {
        MemScope { inner }
    }
}

unsafe impl<A: GlobalAlloc> GlobalAlloc for MemScope<A> {
    #[inline]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = self.inner.alloc(layout);
        if ptr.is_null() {
            return ptr;
        }
        let scope = HookScope::enter();
        if !scope.was_active() {
            recorder().on_alloc(
                ptr as u64,
                layout.size() as u64,
                layout.align() as u32,
                capture,
                EventKind::Alloc,
            );
        }
        ptr
    }

    #[inline]
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = self.inner.alloc_zeroed(layout);
        if ptr.is_null() {
            return ptr;
        }
        let scope = HookScope::enter();
        if !scope.was_active() {
            recorder().on_alloc(
                ptr as u64,
                layout.size() as u64,
                layout.align() as u32,
                capture,
                EventKind::Alloc,
            );
        }
        ptr
    }

    #[inline]
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // Record the free *before* handing memory back, so a concurrent
        // snapshot never sees a freed address as live.
        {
            let scope = HookScope::enter();
            if !scope.was_active() {
                recorder().on_dealloc(ptr as u64);
            }
        }
        self.inner.dealloc(ptr, layout);
    }

    #[inline]
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new = self.inner.realloc(ptr, layout, new_size);
        if new.is_null() {
            return new;
        }
        let scope = HookScope::enter();
        if !scope.was_active() {
            let r = recorder();
            r.on_dealloc(ptr as u64);
            r.on_alloc(
                new as u64,
                new_size as u64,
                layout.align() as u32,
                capture,
                EventKind::ReallocGrow,
            );
        }
        new
    }
}

// --- public control + query API (all guarded: they allocate) -----------------

/// Switch recording mode at runtime.
pub fn set_mode(mode: Mode) {
    let _g = HookScope::enter();
    recorder().set_mode(mode);
}

/// In [`Mode::Sampled`], record roughly 1 in `rate` allocations.
pub fn set_sample_rate(rate: u32) {
    let _g = HookScope::enter();
    recorder().set_sample_rate(rate);
}

/// Limit captured stack depth (1..=[`MAX_FRAMES`]).
pub fn set_backtrace_depth(depth: usize) {
    let _g = HookScope::enter();
    recorder().set_backtrace_depth(depth);
}

/// Toggle stack-trace capture (off = track sizes/live set but no sites).
pub fn set_capture_sites(on: bool) {
    let _g = HookScope::enter();
    recorder().set_capture_sites(on);
}

/// Take a self-contained heap dump of the current live set.
pub fn snapshot() -> Snapshot {
    let _g = HookScope::enter();
    recorder().snapshot()
}

/// Drain up to `max` queued events (oldest first) for live streaming.
pub fn drain_events(out: &mut Vec<RawEvent>, max: usize) -> usize {
    let _g = HookScope::enter();
    recorder().drain_events(out, max)
}

/// Current aggregate counters.
pub fn stats() -> Stats {
    let _g = HookScope::enter();
    recorder().stats()
}

/// Raw return-address frames for an interned site.
pub fn site_frames(site: SiteId) -> Option<Vec<u64>> {
    let _g = HookScope::enter();
    recorder().site_frames(site)
}
