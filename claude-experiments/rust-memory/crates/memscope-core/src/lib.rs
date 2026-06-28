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
mod sink;
mod tls_ring;
mod unwind;

pub use memscope_proto::{EventKind, MetaValue, RawEvent, SiteId, Snapshot};
pub use recorder::{Mode, Stats, MAX_FRAMES};
pub use tls_ring::RingMode;
pub use sink::{spawn_consumer, Consumer, EventSink, FanOut, FnSink, LiveRec, LiveSet};
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
        // `try_with`: during thread teardown the TLS is gone — treat that as
        // "already active" so we bypass recording rather than panic.
        let prev = IN_HOOK.try_with(|x| x.replace(true)).unwrap_or(true);
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
        let _ = IN_HOOK.try_with(|x| x.set(p));
    }
}

/// Selects the stack-capture strategy at runtime. Defaults to frame-pointer
/// unwinding on supported architectures (≈10–20× cheaper than libunwind), and
/// can be flipped to the always-correct backtrace path if a build omits frame
/// pointers.
static USE_FRAME_POINTER: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(unwind::FRAME_POINTER_SUPPORTED);

static FP_UNWIND: unwind::FramePointerUnwind = unwind::FramePointerUnwind;
static BT_UNWIND: DefaultUnwind = DefaultUnwind;

#[inline]
fn capture(buf: &mut [usize], skip: usize) -> usize {
    if USE_FRAME_POINTER.load(std::sync::atomic::Ordering::Relaxed) {
        FP_UNWIND.capture(buf, skip)
    } else {
        BT_UNWIND.capture(buf, skip)
    }
}

/// Choose the stack-capture strategy: `true` = frame-pointer unwinding (fast,
/// needs frame pointers), `false` = libunwind via the `backtrace` crate (slow,
/// always correct). No effect on architectures without frame-pointer support.
pub fn set_frame_pointer_unwinding(on: bool) {
    let _g = HookScope::enter();
    USE_FRAME_POINTER.store(
        on && unwind::FRAME_POINTER_SUPPORTED,
        std::sync::atomic::Ordering::Relaxed,
    );
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
                recorder().on_dealloc(ptr as u64, layout.size() as u64, layout.align() as u32);
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
            r.on_dealloc(ptr as u64, layout.size() as u64, layout.align() as u32);
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

// --- external interposer recording API ---------------------------------------
//
// For a malloc shim (LD_PRELOAD / DYLD_INSERT_LIBRARIES) that observes the
// process's allocations from *outside* the `GlobalAlloc` — same recording the
// `MemScope` allocator does, but callable without owning allocation. Each is
// reentrancy-guarded (the recorder's own allocations must not recurse) and
// captures the call stack internally, so a shim just forwards address + size.

/// Record an observed allocation at `addr` of `size` bytes (alignment best-effort).
#[inline]
pub fn note_alloc(addr: u64, size: u64, align: u32) {
    let scope = HookScope::enter();
    if !scope.was_active() {
        recorder().on_alloc(addr, size, align, capture, EventKind::Alloc);
    }
}

/// Record an observed free of the block at `addr` (its `size`, e.g. from
/// `malloc_size`, keeps the live-bytes accounting exact).
#[inline]
pub fn note_free(addr: u64, size: u64, align: u32) {
    let scope = HookScope::enter();
    if !scope.was_active() {
        recorder().on_dealloc(addr, size, align);
    }
}

/// Record an observed reallocation (frees `old`, allocates `new`).
#[inline]
pub fn note_realloc(old: u64, old_size: u64, new: u64, new_size: u64, align: u32) {
    let scope = HookScope::enter();
    if !scope.was_active() {
        let r = recorder();
        r.on_dealloc(old, old_size, align);
        r.on_alloc(new, new_size, align, capture, EventKind::ReallocGrow);
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

/// Enable/disable the live event stream. Off by default: the event ring is only
/// written when a consumer is draining events. (Installing a consumer via
/// [`spawn_consumer`] enables it automatically.)
pub fn set_event_streaming(on: bool) {
    let _g = HookScope::enter();
    recorder().set_event_streaming(on);
}

/// Switch the event ring between [`RingMode::Overwrite`] (never block; lose
/// oldest under pressure) and [`RingMode::Reliable`] (bounded backpressure so the
/// consumer doesn't lose events).
pub fn set_ring_mode(mode: RingMode) {
    let _g = HookScope::enter();
    recorder().set_ring_mode(mode);
}

/// Total events the ring has dropped because the consumer fell behind (the
/// reconstruction is lossy from that point).
pub fn ring_dropped() -> u64 {
    recorder().ring_dropped()
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

// --- metadata (`meta!`) -------------------------------------------------------

/// Intern a metadata key name to a stable id (cache this at the call site).
pub fn key_id(name: &str) -> u32 {
    let _g = HookScope::enter();
    recorder().intern_key(name)
}

/// Key name for an interned key id.
pub fn key_name(id: u32) -> Option<String> {
    let _g = HookScope::enter();
    recorder().key_name(id)
}

/// The key/value pairs of an interned metadata context.
pub fn meta_context(id: u32) -> Option<Vec<(u32, MetaValue)>> {
    let _g = HookScope::enter();
    recorder().meta_context(id)
}

/// Drop a named checkpoint into the event stream. The live set *at this instant*
/// can later be reconstructed (and two checkpoints diffed) from a recording — see
/// `memscope diff`. A no-op while recording is off; cost is one ring push (≈ a
/// `meta!` enter), no allocation tracked.
pub fn mark(label: &str) {
    let _g = HookScope::enter();
    let r = recorder();
    if r.mode() == Mode::Off {
        return;
    }
    let id = r.intern_mark(label);
    r.on_mark(id);
}

/// The label string for an interned mark id (for the file recorder's `TAG_MARK`
/// table).
pub fn mark_label(id: u32) -> Option<String> {
    let _g = HookScope::enter();
    recorder().mark_label(id)
}

/// Enter a metadata scope: every allocation made on this thread while the
/// returned guard is alive is attributed to `kvs` (merged with any enclosing
/// scopes). A no-op while recording is off. Drop the guard to leave the scope.
///
/// Prefer the `meta!` macro in the `memscope` facade over calling this directly.
#[must_use = "the metadata scope ends when the guard is dropped"]
pub fn push_meta(kvs: &[(u32, MetaValue)]) -> MetaGuard {
    let _g = HookScope::enter();
    let r = recorder();
    if r.mode() == Mode::Off {
        return MetaGuard { meta_id: None };
    }
    let id = r.intern_meta(kvs);
    r.on_meta(EventKind::MetaEnter, id);
    MetaGuard { meta_id: Some(id) }
}

/// RAII handle for an active metadata scope; emits the matching `MetaExit` on
/// drop. See [`push_meta`].
pub struct MetaGuard {
    meta_id: Option<u32>,
}

impl Drop for MetaGuard {
    fn drop(&mut self) {
        if let Some(id) = self.meta_id {
            let _g = HookScope::enter();
            recorder().on_meta(EventKind::MetaExit, id);
        }
    }
}

/// Permanently exclude the *calling thread* from allocation tracking. Intended
/// for the agent/transport thread so its own serialization and symbolication
/// allocations never enter the table or the event stream. Irreversible for that
/// thread (by design).
pub fn exclude_current_thread() {
    IN_HOOK.with(|x| x.set(true));
}
