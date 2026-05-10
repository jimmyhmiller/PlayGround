use std::sync::atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use dynobj::{
    AtomicRootSet, ObjHeader, RootSource, TypeInfo, VarLenKind, read_type_id, read_varlen_count,
    scan_object,
};

use crate::alloc::{Alloc, AtomicBumpAllocator, HeapWalker};
use crate::barrier::SATBQueue;
use crate::semi_space::FORWARDING_BIT;
use crate::card_table::CardTable;
use crate::semi_space::PtrPolicy;
use crate::statemap::{StatemapTracer, TraceState};
use crate::thread::ThreadState;

// ─── Heap ───────────────────────────────────────────────────────────

/// Thread-safe heap with stop-the-world semi-space collection.
///
/// Multiple mutator threads allocate concurrently from shared from-space
/// (via `AtomicBumpAllocator`). When space is exhausted, one thread
/// triggers STW collection: all threads park at safepoints, the GC
/// copies live objects to to-space, swaps spaces, and resumes threads.
/// GC phase for concurrent collection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum GcPhase {
    /// No GC in progress. Write barriers are disabled.
    Idle = 0,
    /// Concurrent copying in progress. Write barriers are active.
    Copying = 1,
}

/// State for the generational nursery (young generation).
struct NurseryState {
    /// The nursery bump allocator (young generation).
    nursery: AtomicBumpAllocator,
    /// Card tables for each tenured space — tracks old→young pointer writes.
    /// Index matches the `spaces` array: card_tables[0] covers spaces[0], etc.
    card_tables: [CardTable; 2],
    /// Number of minor collections performed.
    minor_collections: AtomicUsize,
}

pub struct Heap {
    spaces: [AtomicBumpAllocator; 2],
    /// Index into `spaces` for the current from-space (0 or 1).
    from_idx: AtomicUsize,

    /// Registered mutator threads. The GC scans their roots during STW.
    threads: Mutex<Vec<Arc<ThreadState>>>,

    /// Global roots (interned symbols, module-level constants, etc.)
    pub globals: AtomicRootSet,

    /// Flag polled by mutator threads: "should I enter safepoint?"
    gc_requested: AtomicBool,

    /// Prevents concurrent GC triggers.
    gc_lock: Mutex<()>,

    type_id_offset: usize,
    type_table: Vec<TypeInfo>,
    collections: AtomicUsize,

    /// Current GC phase. Mutators check this to know if write barriers
    /// should be active.
    gc_phase: AtomicU8,

    /// Global SATB queue. Mutator threads flush their local SATB buffers
    /// here, and the GC drains it during collection.
    pub satb_queue: SATBQueue,

    /// Optional statemap tracer for GC visualization.
    pub tracer: Option<Arc<StatemapTracer>>,

    /// When true, trigger GC on every allocation (stress testing).
    gc_every_alloc: AtomicBool,

    /// Generational nursery state. `None` for non-generational heaps.
    nursery_state: Option<NurseryState>,

    /// Optional walker for the JIT frame chain pointed to by each
    /// parked thread's `parked_jit_fp`. dynalloc cannot itself
    /// understand JIT-frame layouts (that's dynlower's job), so the
    /// embedder installs this hook at runtime via `set_jit_frame_walker`.
    /// `None` means "no JIT integration" — used by tests and
    /// non-JIT clients.
    ///
    /// SIGNATURE: `(jit_fp, visitor) -> ()`. The walker calls `visitor`
    /// with each `*mut u64` slot it finds, exactly as a `RootSource`
    /// would.
    jit_frame_walker: std::sync::atomic::AtomicPtr<()>,
}

// Safety: All fields are either Sync (atomics, mutexes) or accessed
// only under proper synchronization.
unsafe impl Sync for Heap {}
unsafe impl Send for Heap {}

impl Heap {
    /// Create a new heap with two spaces of `space_size` bytes each.
    pub fn new<H: ObjHeader>(space_size: usize, type_table: Vec<TypeInfo>) -> Self {
        Heap {
            spaces: [
                AtomicBumpAllocator::new::<H>(space_size),
                AtomicBumpAllocator::new::<H>(space_size),
            ],
            from_idx: AtomicUsize::new(0),
            threads: Mutex::new(Vec::new()),
            globals: AtomicRootSet::new(),
            gc_requested: AtomicBool::new(false),
            gc_lock: Mutex::new(()),
            type_id_offset: H::TYPE_ID_OFFSET,
            type_table,
            collections: AtomicUsize::new(0),
            gc_phase: AtomicU8::new(GcPhase::Idle as u8),
            satb_queue: SATBQueue::new(),
            tracer: None,
            gc_every_alloc: AtomicBool::new(false),
            nursery_state: None,
            jit_frame_walker: std::sync::atomic::AtomicPtr::new(std::ptr::null_mut()),
        }
    }

    /// Create a new generational heap with a nursery (young generation)
    /// and two tenured spaces (old generation).
    ///
    /// New allocations go to the nursery. When the nursery fills, a minor
    /// GC promotes survivors to tenured from-space. When tenured space fills,
    /// a major GC (STW or concurrent) collects the old generation.
    pub fn new_generational<H: ObjHeader>(nursery_size: usize, tenured_size: usize, type_table: Vec<TypeInfo>) -> Self {
        let spaces = [
            AtomicBumpAllocator::new::<H>(tenured_size),
            AtomicBumpAllocator::new::<H>(tenured_size),
        ];
        let card_tables = [
            CardTable::new(spaces[0].base(), tenured_size),
            CardTable::new(spaces[1].base(), tenured_size),
        ];
        Heap {
            spaces,
            from_idx: AtomicUsize::new(0),
            threads: Mutex::new(Vec::new()),
            globals: AtomicRootSet::new(),
            gc_requested: AtomicBool::new(false),
            gc_lock: Mutex::new(()),
            type_id_offset: H::TYPE_ID_OFFSET,
            type_table,
            collections: AtomicUsize::new(0),
            gc_phase: AtomicU8::new(GcPhase::Idle as u8),
            satb_queue: SATBQueue::new(),
            tracer: None,
            gc_every_alloc: AtomicBool::new(false),
            nursery_state: Some(NurseryState {
                nursery: AtomicBumpAllocator::new::<H>(nursery_size),
                card_tables,
                minor_collections: AtomicUsize::new(0),
            }),
            jit_frame_walker: std::sync::atomic::AtomicPtr::new(std::ptr::null_mut()),
        }
    }

    /// Install a walker for JIT frames pointed to by parked threads.
    /// Called by frontends with JIT integration (e.g. dynlang) at
    /// runtime-construction time. The walker is consulted at every
    /// GC: each registered thread's `parked_jit_fp` is read; if
    /// non-null, the walker is invoked with that fp and a visitor.
    ///
    /// # Safety
    /// The walker must remain valid for the lifetime of the Heap.
    /// The supplied `*mut u64` slots must be live for the duration of
    /// the visitor call (they are, by the safepoint contract).
    pub fn set_jit_frame_walker(
        &self,
        walker: unsafe fn(*const u8, &mut dyn FnMut(*mut u64)),
    ) {
        let p = walker as *mut ();
        self.jit_frame_walker
            .store(p, std::sync::atomic::Ordering::Release);
    }

    /// Walk a single parked-thread JIT frame using the registered
    /// walker. No-op if no walker is installed.
    fn walk_jit_frame(&self, jit_fp: *const u8, visitor: &mut dyn FnMut(*mut u64)) {
        let walker = self
            .jit_frame_walker
            .load(std::sync::atomic::Ordering::Acquire);
        if walker.is_null() {
            return;
        }
        let walker: unsafe fn(*const u8, &mut dyn FnMut(*mut u64)) =
            unsafe { std::mem::transmute(walker as *const ()) };
        unsafe { walker(jit_fp, visitor) };
    }

    /// Enable or disable GC-on-every-allocation (stress testing mode).
    pub fn set_gc_every_alloc(&self, enabled: bool) {
        self.gc_every_alloc.store(enabled, Ordering::Release);
    }

    /// Check if GC should be triggered on every allocation.
    #[inline(always)]
    pub fn gc_every_alloc(&self) -> bool {
        self.gc_every_alloc.load(Ordering::Relaxed)
    }

    /// Get the current from-space.
    #[inline(always)]
    fn from_space(&self) -> &AtomicBumpAllocator {
        unsafe {
            self.spaces
                .get_unchecked(self.from_idx.load(Ordering::Acquire))
        }
    }

    /// Get the current to-space.
    #[inline(always)]
    fn to_space(&self) -> &AtomicBumpAllocator {
        unsafe {
            self.spaces
                .get_unchecked(1 - self.from_idx.load(Ordering::Acquire))
        }
    }

    /// Swap from-space and to-space by flipping the index.
    fn swap_spaces(&self) {
        let old = self.from_idx.load(Ordering::Acquire);
        self.from_idx.store(1 - old, Ordering::Release);
    }

    /// Create a new heap with statemap tracing enabled.
    pub fn new_with_tracer<H: ObjHeader>(space_size: usize, type_table: Vec<TypeInfo>, tracer: Arc<StatemapTracer>) -> Self {
        let mut heap = Self::new::<H>(space_size, type_table);
        heap.tracer = Some(tracer);
        heap
    }

    /// Record a trace event if a tracer is attached.
    #[inline(always)]
    #[allow(dead_code)]
    fn trace(&self, entity: &str, state: TraceState) {
        if let Some(ref tracer) = self.tracer {
            tracer.record(entity, state);
        }
    }

    /// Record a trace event for the GC entity.
    #[inline(always)]
    fn trace_gc(&self, state: TraceState) {
        if let Some(ref tracer) = self.tracer {
            tracer.record_gc(state);
        }
    }

    /// Record a trace event for a thread by index.
    #[inline(always)]
    fn trace_thread(&self, id: usize, state: TraceState) {
        if let Some(ref tracer) = self.tracer {
            tracer.record_thread(id, state);
        }
    }

    // ─── Thread registration ────────────────────────────────────

    /// Register a new mutator thread. Returns a shared handle to
    /// the thread's state and its index.
    ///
    /// If a GC is currently active, waits for it to finish first.
    /// This prevents a newly registered thread from calling safepoint()
    /// during an active GC that doesn't know about it — which would
    /// cause the thread to block in enter_safepoint with nobody to
    /// resume it.
    pub fn register_thread(&self) -> (Arc<ThreadState>, usize) {
        let ts = Arc::new(ThreadState::new());
        loop {
            // Wait for any active GC to finish.
            while self.gc_requested() || self.barriers_active() {
                std::thread::yield_now();
            }
            // Register under the threads lock. Since GC methods
            // (stw_collect, concurrent_collect, mutator_triggered_gc) set
            // gc_requested under this same lock before snapshotting, if
            // gc_requested is still false while we hold the lock, no GC
            // has us in (or will add us to) its snapshot without including us.
            let mut threads = self.threads.lock().unwrap();
            if self.gc_requested() || self.barriers_active() {
                // GC started between our check and acquiring the lock.
                drop(threads);
                continue;
            }
            let id = threads.len();
            threads.push(ts.clone());
            return (ts, id);
        }
    }

    /// Deregister a mutator thread by its ThreadState pointer.
    ///
    /// The thread must not be running mutator code when this is called.
    /// Typically called after the thread has finished all work.
    pub fn deregister_thread(&self, state: &Arc<ThreadState>) {
        let mut threads = self.threads.lock().unwrap();
        let ptr = Arc::as_ptr(state);
        if let Some(pos) = threads.iter().position(|t| Arc::as_ptr(t) == ptr) {
            threads.swap_remove(pos);
        }
    }

    /// Safely deregister a mutator thread, first participating in any
    /// pending GC. This avoids the race where a GC snapshots this thread
    /// and then waits for it to reach safepoint, but the thread deregisters
    /// and never reaches safepoint.
    ///
    /// The atomicity guarantee: gc_requested is checked under the threads
    /// lock. If no GC is requested, we deregister while holding the lock,
    /// so no new mutator_triggered_gc can snapshot us. If GC IS requested,
    /// we drop the lock, enter safepoint, and retry.
    pub fn safe_deregister_thread(&self, state: &Arc<ThreadState>) {
        loop {
            if self.gc_requested() {
                state.enter_safepoint();
                continue;
            }
            if self.barriers_active() {
                std::thread::yield_now();
                continue;
            }
            // Try to deregister atomically with gc_requested check.
            let mut threads = self.threads.lock().unwrap();
            if self.gc_requested() || self.barriers_active() {
                // GC started between our check and acquiring the lock.
                drop(threads);
                continue;
            }
            let ptr = Arc::as_ptr(state);
            if let Some(pos) = threads.iter().position(|t| Arc::as_ptr(t) == ptr) {
                threads.swap_remove(pos);
            }
            break;
        }
    }

    // ─── Allocation ─────────────────────────────────────────────

    /// Allocate from the shared from-space (atomic bump).
    ///
    /// Returns null if from-space is exhausted. Caller should
    /// trigger GC via `collect()` and retry.
    pub fn alloc(&self, info: &TypeInfo, varlen_len: usize) -> *mut u8 {
        self.from_space().alloc(info, varlen_len)
    }

    /// Allocate and initialize header + varlen count.
    pub fn alloc_obj<H: ObjHeader>(&self, info: &TypeInfo, varlen_len: usize) -> *mut u8 {
        unsafe { crate::alloc::alloc_obj::<H>(self.from_space(), info, varlen_len) }
    }

    // ─── Generational helpers ────────────────────────────────────

    /// Check if this heap has a nursery (generational mode).
    #[inline(always)]
    pub fn has_nursery(&self) -> bool {
        self.nursery_state.is_some()
    }

    /// Check if a pointer is in the nursery.
    #[inline(always)]
    pub fn is_nursery(&self, ptr: *const u8) -> bool {
        match &self.nursery_state {
            Some(ns) => ns.nursery.contains(ptr),
            None => false,
        }
    }

    /// Check if a pointer is in tenured space (either from or to).
    #[inline(always)]
    pub fn is_tenured(&self, ptr: *const u8) -> bool {
        self.from_space().contains(ptr) || self.to_space().contains(ptr)
    }

    /// Allocate from the nursery if generational, otherwise from from-space.
    pub fn alloc_nursery(&self, info: &TypeInfo, varlen_len: usize) -> *mut u8 {
        match &self.nursery_state {
            Some(ns) => ns.nursery.alloc(info, varlen_len),
            None => self.from_space().alloc(info, varlen_len),
        }
    }

    /// Allocate and init header in the nursery if generational, otherwise from-space.
    pub fn alloc_nursery_obj<H: ObjHeader>(
        &self,
        info: &TypeInfo,
        varlen_len: usize,
    ) -> *mut u8 {
        match &self.nursery_state {
            Some(ns) => unsafe { crate::alloc::alloc_obj::<H>(&ns.nursery, info, varlen_len) },
            None => unsafe { crate::alloc::alloc_obj::<H>(self.from_space(), info, varlen_len) },
        }
    }

    /// Allocate directly in tenured from-space (for promotion during minor GC).
    pub fn alloc_tenured(&self, info: &TypeInfo, varlen_len: usize) -> *mut u8 {
        self.from_space().alloc(info, varlen_len)
    }

    /// Mark the card covering `obj` as dirty (old→young pointer write).
    #[inline(always)]
    pub fn mark_card_dirty(&self, obj: *const u8) {
        if let Some(ns) = &self.nursery_state {
            let from_idx = self.from_idx.load(Ordering::Acquire);
            ns.card_tables[from_idx].mark_dirty(obj);
        }
    }

    /// Number of minor collections performed.
    pub fn minor_collections(&self) -> usize {
        match &self.nursery_state {
            Some(ns) => ns.minor_collections.load(Ordering::Relaxed),
            None => 0,
        }
    }

    /// Nursery bytes used.
    pub fn nursery_used(&self) -> usize {
        match &self.nursery_state {
            Some(ns) => ns.nursery.used(),
            None => 0,
        }
    }

    // ─── Safepoint polling ──────────────────────────────────────

    /// Check if GC has been requested. Mutator threads should call
    /// this at safepoints (loop backedges, allocation slow paths).
    #[inline(always)]
    pub fn gc_requested(&self) -> bool {
        self.gc_requested.load(Ordering::Acquire)
    }

    pub fn safepoint(&self, thread: &ThreadState) {
        if self.gc_requested.load(Ordering::Acquire) {
            thread.enter_safepoint();
        }
    }

    // ─── Collection ─────────────────────────────────────────────

    /// Bytes used in from-space.
    pub fn from_used(&self) -> usize {
        self.from_space().used()
    }

    /// Base pointer of from-space (for diagnostics).
    pub fn from_base(&self) -> *const u8 {
        self.from_space().base()
    }

    /// Base pointer of to-space (for diagnostics).
    pub fn to_base(&self) -> *const u8 {
        self.to_space().base()
    }

    /// Check if a pointer is specifically in tenured from-space (not nursery).
    pub fn from_space_contains(&self, ptr: *const u8) -> bool {
        self.from_space().contains(ptr)
    }

    /// Check if a pointer is in from-space (or nursery for generational heaps).
    pub fn contains(&self, ptr: *const u8) -> bool {
        self.from_space().contains(ptr) || self.is_nursery(ptr)
    }

    /// Check if a pointer is in either tenured space or nursery.
    pub fn contains_either(&self, ptr: *const u8) -> bool {
        self.from_space().contains(ptr) || self.to_space().contains(ptr) || self.is_nursery(ptr)
    }

    /// Total size of each space.
    pub fn space_size(&self) -> usize {
        self.from_space().size()
    }

    /// Number of collections performed.
    pub fn collections(&self) -> usize {
        self.collections.load(Ordering::Relaxed)
    }

    /// Run a stop-the-world collection.
    ///
    /// This is designed to be called by one thread (the triggering thread)
    /// while all other threads are at safepoints, OR by a dedicated GC
    /// thread after requesting all mutators to stop.
    ///
    /// # Safety
    /// - All objects in from-space must have valid headers.
    /// - All mutator threads must be at safepoints (not running mutator code).
    /// - `extra_roots` can provide additional root sources beyond the
    ///   registered threads and globals.
    pub unsafe fn collect<P: PtrPolicy>(&self, extra_roots: &[&dyn RootSource]) {
        let _gc_guard = self.gc_lock.lock().unwrap();
        unsafe { self.collect_inner::<P>(extra_roots) };
    }

    /// Internal collection logic — caller must hold `gc_lock`.
    unsafe fn collect_inner<P: PtrPolicy>(&self, extra_roots: &[&dyn RootSource]) {
        // Phase 1: scan all roots → copy/forward targets into to-space

        self.globals.scan_roots(&mut |slot| {
            unsafe { self.process_slot::<P>(slot) };
        });

        // Per-thread roots (threads are at safepoints — safe to scan)
        {
            let threads = self.threads.lock().unwrap();
            for ts in threads.iter() {
                ts.scan_roots(&mut |slot| {
                    unsafe { self.process_slot::<P>(slot) };
                });
                let jit_fp = ts.parked_jit_fp();
                if !jit_fp.is_null() {
                    self.walk_jit_frame(jit_fp, &mut |slot| {
                        unsafe { self.process_slot::<P>(slot) };
                    });
                }
            }
        }

        // Extra roots (caller-provided)
        for source in extra_roots.iter() {
            source.scan_roots(&mut |slot| {
                unsafe { self.process_slot::<P>(slot) };
            });
        }

        // Nursery objects as roots: during major GC, nursery objects may
        // hold pointers to tenured from-space objects that need forwarding.
        if let Some(ns) = &self.nursery_state {
            unsafe {
                ns.nursery.walk(&self.type_table, &mut |obj, info| {
                    scan_object(obj, info, |slot| {
                        self.process_slot::<P>(slot);
                    });
                });
            }
        }

        // Phase 2: Cheney scan — walk to-space linearly
        let mut scan_offset = 0usize;
        while scan_offset < self.to_space().used() {
            let obj = unsafe { self.to_space().base().add(scan_offset) };
            let type_id = unsafe { read_type_id(obj, self.type_id_offset) };
            let info = &self.type_table[type_id as usize];

            unsafe {
                scan_object(obj, info, |slot| {
                    self.process_slot::<P>(slot);
                });
            }

            let varlen_len = match info.varlen {
                VarLenKind::None => 0,
                _ => unsafe { read_varlen_count(obj, info) },
            };
            let obj_size = info.allocation_size(varlen_len);
            let align = 1usize << info.align_log2;
            scan_offset = (scan_offset + obj_size + align - 1) & !(align - 1);
        }

        // Phase 3: swap spaces (flip atomic index — no UB, no ptr::swap)
        self.swap_spaces();
        self.to_space().reset();
        self.collections.fetch_add(1, Ordering::Relaxed);

        // Clear card table for new from-space after swap
        if let Some(ns) = &self.nursery_state {
            let from_idx = self.from_idx.load(Ordering::Acquire);
            ns.card_tables[from_idx].clear_all();
        }
    }

    /// Trigger a STW collection cycle: request all threads to stop,
    /// wait for them, collect, then resume them.
    ///
    /// # Safety
    /// Same as `collect`. The calling thread must NOT be registered
    /// as a mutator (or must have already entered its safepoint).
    pub unsafe fn stw_collect<P: PtrPolicy>(&self) {
        let _gc_guard = self.gc_lock.lock().unwrap();

        self.trace_gc(TraceState::GcWaitingForSafepoints);

        // Snapshot thread refs and set gc_requested atomically under threads lock.
        // This prevents a thread from deregistering between gc_requested being set
        // and the snapshot being taken.
        let thread_snapshot: Vec<Arc<ThreadState>> = {
            let threads = self.threads.lock().unwrap();
            self.gc_requested.store(true, Ordering::Release);
            threads.iter().cloned().collect()
        };
        for (i, ts) in thread_snapshot.iter().enumerate() {
            while !ts.is_safely_at_safepoint() {
                std::thread::yield_now();
            }
            self.trace_thread(i, TraceState::AtSafepoint);
        }

        self.trace_gc(TraceState::GcStw);
        unsafe { self.collect_inner::<P>(&[]) };

        self.trace_gc(TraceState::GcResuming);
        self.gc_requested.store(false, Ordering::Release);

        for (i, ts) in thread_snapshot.iter().enumerate() {
            ts.resume();
            self.trace_thread(i, TraceState::Running);
        }
        self.trace_gc(TraceState::GcIdle);
    }

    /// Bytes remaining in from-space.
    pub fn from_remaining(&self) -> usize {
        self.from_space().remaining()
    }

    /// Trigger STW collection from a mutator thread.
    ///
    /// If another thread is already collecting, this thread enters its
    /// safepoint and waits for that collection to finish instead.
    ///
    /// # Safety
    /// Same as `collect`. Convenience wrapper: no extra root sources.
    pub unsafe fn mutator_triggered_gc<P: PtrPolicy>(&self, triggering_thread: &ThreadState) {
        unsafe { self.mutator_triggered_gc_with_extras::<P>(triggering_thread, &[]) }
    }

    /// Read each registered `ThreadState`'s `parked_jit_fp`. Returns
    /// only non-null FPs (threads currently parked at a JIT
    /// safepoint). Used by the JIT safepoint session to walk every
    /// parked thread's JIT frame as a root source during STW GC,
    /// ensuring the GC's relocations update spill slots in those
    /// frames.
    pub fn parked_thread_jit_fps(&self) -> Vec<*const u8> {
        let threads = self.threads.lock().unwrap();
        threads
            .iter()
            .filter_map(|ts| {
                let fp = ts.parked_jit_fp();
                if fp.is_null() { None } else { Some(fp) }
            })
            .collect()
    }

    /// Like [`mutator_triggered_gc`], but also scans `extra_roots`
    /// after the registered threads. Used by JIT frontends to thread
    /// per-safepoint frame roots and module-level literal pools into
    /// the trace alongside the standard root set.
    pub unsafe fn mutator_triggered_gc_with_extras<P: PtrPolicy>(
        &self,
        triggering_thread: &ThreadState,
        extra_roots: &[&dyn RootSource],
    ) {
        // Try to become the GC thread. If someone else is already collecting,
        // enter our safepoint and wait for them to finish.
        let gc_guard = match self.gc_lock.try_lock() {
            Ok(guard) => guard,
            Err(std::sync::TryLockError::WouldBlock) => {
                // Another thread is already collecting.
                // Enter safepoint so they can proceed, but only while GC
                // is actively requesting safepoints. If gc_requested is false,
                // the GC winner is between its resume pass and dropping gc_lock —
                // entering safepoint now would hang because nobody will resume us.
                while self.gc_requested() {
                    triggering_thread.enter_safepoint();
                }
                return;
            }
            Err(e) => panic!("gc_lock poisoned: {}", e),
        };

        // We won the race — we're the GC thread now.

        // Snapshot thread refs and set gc_requested atomically (under threads lock).
        // This prevents a thread from deregistering between gc_requested being set
        // and the snapshot being taken — which would leave a "ghost" in the snapshot
        // that never reaches safepoint.
        let trigger_ptr = triggering_thread as *const ThreadState as usize;
        let thread_snapshot: Vec<Arc<ThreadState>> = {
            let threads = self.threads.lock().unwrap();
            self.gc_requested.store(true, Ordering::Release);
            threads.iter().cloned().collect()
        };

        // Wait for all threads EXCEPT ourselves to reach safepoints
        for ts in thread_snapshot.iter() {
            if Arc::as_ptr(ts) as usize == trigger_ptr {
                continue;
            }
            while !ts.is_safely_at_safepoint() {
                std::thread::yield_now();
            }
        }

        // All other threads suspended — run collection.
        // Scan ALL threads' roots (including ours — our frame chain is stable
        // since we're running GC code, not mutator code).

        // Phase 1: scan all roots
        self.globals.scan_roots(&mut |slot| {
            unsafe { self.process_slot::<P>(slot) };
        });

        for ts in thread_snapshot.iter() {
            ts.scan_roots(&mut |slot| {
                unsafe { self.process_slot::<P>(slot) };
            });
            // If this thread is parked at a JIT safepoint, walk its
            // JIT frame chain too. Without this, GC misses the
            // spill slots holding live GC pointers in the parked
            // thread's recursion stack — and the thread reads stale
            // (relocated) pointers when it resumes.
            let jit_fp = ts.parked_jit_fp();
            if !jit_fp.is_null() {
                self.walk_jit_frame(jit_fp, &mut |slot| {
                    unsafe { self.process_slot::<P>(slot) };
                });
            }
        }

        // Caller-provided extra roots (e.g. JIT-frame stack-map roots,
        // module-level literal pools, host-side scoped frame chains).
        for source in extra_roots.iter() {
            source.scan_roots(&mut |slot| {
                unsafe { self.process_slot::<P>(slot) };
            });
        }

        // Nursery objects as roots during major GC
        if let Some(ns) = &self.nursery_state {
            unsafe {
                ns.nursery.walk(&self.type_table, &mut |obj, info| {
                    scan_object(obj, info, |slot| {
                        self.process_slot::<P>(slot);
                    });
                });
            }
        }

        // Phase 2: Cheney scan
        let mut scan_offset = 0usize;
        while scan_offset < self.to_space().used() {
            let obj = unsafe { self.to_space().base().add(scan_offset) };
            let type_id = unsafe { read_type_id(obj, self.type_id_offset) };
            let info = &self.type_table[type_id as usize];

            unsafe {
                scan_object(obj, info, |slot| {
                    self.process_slot::<P>(slot);
                });
            }

            let varlen_len = match info.varlen {
                VarLenKind::None => 0,
                _ => unsafe { read_varlen_count(obj, info) },
            };
            let obj_size = info.allocation_size(varlen_len);
            let align = 1usize << info.align_log2;
            scan_offset = (scan_offset + obj_size + align - 1) & !(align - 1);
        }

        // Phase 3: swap spaces
        self.swap_spaces();
        self.to_space().reset();
        self.collections.fetch_add(1, Ordering::Relaxed);

        // Clear card table for new from-space after swap (all tenured pointers updated)
        if let Some(ns) = &self.nursery_state {
            let from_idx = self.from_idx.load(Ordering::Acquire);
            ns.card_tables[from_idx].clear_all();
        }

        // Debug: verify all root slots point to from-space after swap
        if cfg!(debug_assertions) && self.gc_every_alloc() {
            for ts in thread_snapshot.iter() {
                ts.scan_roots(&mut |slot| {
                    let bits = unsafe { *slot };
                    if let Some(ptr) = P::try_decode_ptr(bits) {
                        assert!(
                            self.from_space().contains(ptr),
                            "POST-SWAP: root slot {:p} points to {:p} which is NOT in from-space \
                             [{:p}..+{}). In to-space: {}. collections={}",
                            slot,
                            ptr,
                            self.from_space().base(),
                            self.from_space().size(),
                            self.to_space().contains(ptr),
                            self.collections.load(Ordering::Relaxed),
                        );
                    }
                });
            }
        }

        // Clear request flag and resume all other threads.
        self.gc_requested.store(false, Ordering::Release);

        for ts in thread_snapshot.iter() {
            if Arc::as_ptr(ts) as usize == trigger_ptr {
                continue;
            }
            ts.resume();
        }

        drop(gc_guard);
    }

    // ─── GC Phase ─────────────────────────────────────────────

    /// Current GC phase.
    pub fn gc_phase(&self) -> GcPhase {
        match self.gc_phase.load(Ordering::Acquire) {
            1 => GcPhase::Copying,
            _ => GcPhase::Idle,
        }
    }

    /// Check if write barriers should be active.
    ///
    /// Relaxed ordering is correct here because:
    /// - Before STW #1: mutations are captured in the snapshot copy, no SATB needed.
    /// - After STW #1 resume: the safepoint Mutex provides a full memory barrier,
    ///   so gc_phase=Copying is visible to all mutators before they resume.
    #[inline(always)]
    pub fn barriers_active(&self) -> bool {
        self.gc_phase.load(Ordering::Relaxed) != GcPhase::Idle as u8
    }

    /// The type_id_offset for this heap's header type.
    pub fn type_id_offset(&self) -> usize {
        self.type_id_offset
    }

    /// Replication barrier: if `obj` has been forwarded to to-space,
    /// replicate a field write to the to-space copy.
    ///
    /// During concurrent GC, the collector copies objects from from-space
    /// to to-space. If a mutator modifies a from-space object's field
    /// after the copy, the to-space copy becomes stale. This barrier
    /// ensures the to-space copy stays in sync by writing the new value
    /// to the same offset in the forwarded copy.
    ///
    /// Fast path when no GC is active: single atomic load + branch.
    ///
    /// # Safety
    /// - `obj` must point to a valid heap object.
    /// - `field_offset` must be a valid offset within the object.
    #[inline(always)]
    pub unsafe fn replication_barrier(&self, obj: *mut u8, field_offset: usize, new_bits: u64) {
        if !self.barriers_active() {
            return;
        }
        // Check if this object has been forwarded to to-space
        if let Some(forwarded) = unsafe { self.check_forwarded_atomic(obj) } {
            unsafe {
                let atomic_slot =
                    &*(forwarded.add(field_offset) as *const std::sync::atomic::AtomicU64);
                atomic_slot.store(new_bits, Ordering::Relaxed);
            };
        }
    }

    /// Read barrier: follow forwarding pointer if object has been relocated.
    ///
    /// During concurrent GC, an object's type_info slot may contain a
    /// forwarding pointer (bit 0 set). This returns the relocated address.
    ///
    /// Fast path when no GC is active: returns the pointer unchanged
    /// (single atomic load + branch).
    ///
    /// # Safety
    /// - `ptr` must point to a valid heap object or be null.
    #[inline(always)]
    pub unsafe fn read_barrier(&self, ptr: *mut u8) -> *mut u8 {
        if ptr.is_null() {
            return ptr;
        }
        // Fast path: no GC active
        if !self.barriers_active() {
            return ptr;
        }
        // Slow path: check forwarding pointer
        unsafe { crate::barrier::read_barrier_atomic(ptr, self.type_id_offset) }
    }

    // ─── Concurrent Collection ──────────────────────────────────

    /// Run a concurrent collection cycle.
    ///
    /// Protocol:
    /// 1. Short STW pause #1: snapshot roots, enable write barriers
    /// 2. Concurrent copy phase: copy objects while mutators run
    /// 3. Short STW pause #2: drain SATB buffers, process remaining
    ///    gray objects, swap spaces, disable barriers, resume
    ///
    /// # Safety
    /// Same as `collect`. The calling thread must NOT be a registered mutator.
    pub unsafe fn concurrent_collect<P: PtrPolicy>(&self) {
        let _gc_guard = self.gc_lock.lock().unwrap();

        // ── STW Pause #1: Snapshot roots ─────────────────────────

        self.trace_gc(TraceState::GcWaitingForSafepoints);
        self.gc_phase
            .store(GcPhase::Copying as u8, Ordering::Release);

        // Snapshot thread refs and set gc_requested atomically under threads lock.
        let thread_snapshot: Vec<Arc<ThreadState>> = {
            let threads = self.threads.lock().unwrap();
            self.gc_requested.store(true, Ordering::Release);
            threads.iter().cloned().collect()
        };

        for (i, ts) in thread_snapshot.iter().enumerate() {
            while !ts.is_safely_at_safepoint() {
                std::thread::yield_now();
            }
            self.trace_thread(i, TraceState::AtSafepoint);
        }

        self.trace_gc(TraceState::GcStw);
        // Snapshot roots into to-space (short work under STW)
        self.globals.scan_roots(&mut |slot| {
            unsafe { self.process_slot::<P>(slot) };
        });

        for ts in thread_snapshot.iter() {
            ts.scan_roots(&mut |slot| {
                unsafe { self.process_slot::<P>(slot) };
            });
        }

        // Nursery objects as roots during concurrent major GC
        if let Some(ns) = &self.nursery_state {
            unsafe {
                ns.nursery.walk(&self.type_table, &mut |obj, info| {
                    scan_object(obj, info, |slot| {
                        self.process_slot::<P>(slot);
                    });
                });
            }
        }

        // Resume threads — they now run with write barriers active.
        // Snapshot generations so STW #2 can distinguish stale safepoints.
        self.trace_gc(TraceState::GcResuming);
        self.gc_requested.store(false, Ordering::Release);
        let stw1_gens: Vec<u64> = thread_snapshot
            .iter()
            .map(|ts| ts.safepoint_gen())
            .collect();
        for ts in thread_snapshot.iter() {
            ts.resume();
        }

        // ── Concurrent Copy Phase ────────────────────────────────
        self.trace_gc(TraceState::GcConcurrent);

        // Discover and copy the transitive closure of root objects.
        //
        // Standard concurrent copying GC approach: scan to-space objects
        // to discover children, but read each child's fields from the
        // FROM-SPACE original (not the to-space copy). This is critical
        // because mutators may modify from-space objects during the
        // concurrent phase (e.g., trimming a linked list by nulling a
        // next pointer). Reading from from-space sees the latest values.
        // The SATB write barrier ensures that any old pointer values
        // overwritten by mutators are captured for later processing.
        //
        // We do NOT update to-space pointer fields during the concurrent
        // phase (no write to to-space objects). This avoids data races
        // between the GC updating a to-space field and a mutator reading
        // it. All pointer updates are deferred to STW #2.
        let mut scan_offset = 0usize;
        while scan_offset < self.to_space().used() {
            let to_obj = unsafe { self.to_space().base().add(scan_offset) };
            let type_id = unsafe { read_type_id(to_obj, self.type_id_offset) };
            let info = &self.type_table[type_id as usize];

            // Read fields from the to-space copy to discover from-space
            // children. For each from-space pointer, copy the target to
            // to-space (installs forwarding pointer) but don't update the
            // parent's field — that's deferred to STW #2's re-scan.
            unsafe {
                scan_object(to_obj, info, |slot| {
                    self.process_slot_concurrent::<P>(slot);
                });
            }

            let varlen_len = match info.varlen {
                VarLenKind::None => 0,
                _ => unsafe { read_varlen_count(to_obj, info) },
            };
            let obj_size = info.allocation_size(varlen_len);
            let align = 1usize << info.align_log2;
            scan_offset = (scan_offset + obj_size + align - 1) & !(align - 1);
        }

        // ── STW Pause #2: Final drain + swap ────────────────────

        self.trace_gc(TraceState::GcWaitingForSafepoints);
        self.gc_requested.store(true, Ordering::Release);

        for (i, (ts, stw1_gen)) in thread_snapshot.iter().zip(stw1_gens.iter()).enumerate() {
            // Wait for thread to be at a NEW safepoint (gen > stw1_gen)
            // AND truly blocked (no stale resume pending).
            //
            // Threads in STATE_BLOCKED are accepted unconditionally:
            // they cannot be running mutator code, so by definition no
            // new allocations or root mutations have occurred since
            // STW #1. Re-scanning their (unchanged) roots is sound
            // even though their gen hasn't advanced.
            loop {
                if ts.is_blocked() && ts.is_safely_at_safepoint() {
                    break;
                }
                if ts.safepoint_gen() > *stw1_gen && ts.is_safely_at_safepoint() {
                    break;
                }
                std::thread::yield_now();
            }
            self.trace_thread(i, TraceState::AtSafepoint);
        }

        self.trace_gc(TraceState::GcStw);

        // Re-scan all roots. Between STW #1 and now, mutators may have:
        // - Allocated new objects in from-space and stored them in roots
        // - Changed root slots to point to different objects
        // Re-scanning catches all of these.
        self.globals.scan_roots(&mut |slot| {
            unsafe { self.process_slot::<P>(slot) };
        });

        for ts in thread_snapshot.iter() {
            ts.scan_roots(&mut |slot| {
                unsafe { self.process_slot::<P>(slot) };
            });
        }

        // Re-scan nursery objects as roots
        if let Some(ns) = &self.nursery_state {
            unsafe {
                ns.nursery.walk(&self.type_table, &mut |obj, info| {
                    scan_object(obj, info, |slot| {
                        self.process_slot::<P>(slot);
                    });
                });
            }
        }

        // Drain SATB queue (flushed by mutators) + thread-local SATB buffers
        let satb_values = self.satb_queue.drain_all();
        for bits in satb_values {
            if let Some(ptr) = P::try_decode_ptr(bits) {
                if self.from_space().contains(ptr) {
                    unsafe { self.copy_or_forward::<P>(ptr) };
                }
            }
        }

        // Drain per-thread SATB buffers (threads are at safepoints — safe)
        for ts in thread_snapshot.iter() {
            let buf = unsafe { ts.satb_buffer() };
            let values = buf.drain();
            for bits in values {
                if let Some(ptr) = P::try_decode_ptr(bits) {
                    if self.from_space().contains(ptr) {
                        unsafe { self.copy_or_forward::<P>(ptr) };
                    }
                }
            }
        }

        // Full re-scan of to-space. During the concurrent phase, mutators
        // may have stored from-space pointers into to-space objects' fields
        // (e.g., allocating a new object and storing it in a field of an
        // already-copied object). The SATB barrier only logs old values,
        // so these new from-space pointers are not captured. Re-scanning
        // from offset 0 catches all such pointers. process_slot is
        // idempotent — already-correct to-space pointers are skipped.
        scan_offset = 0;
        while scan_offset < self.to_space().used() {
            let obj = unsafe { self.to_space().base().add(scan_offset) };
            let type_id = unsafe { read_type_id(obj, self.type_id_offset) };
            let info = &self.type_table[type_id as usize];

            unsafe {
                scan_object(obj, info, |slot| {
                    self.process_slot::<P>(slot);
                });
            }

            let varlen_len = match info.varlen {
                VarLenKind::None => 0,
                _ => unsafe { read_varlen_count(obj, info) },
            };
            let obj_size = info.allocation_size(varlen_len);
            let align = 1usize << info.align_log2;
            scan_offset = (scan_offset + obj_size + align - 1) & !(align - 1);
        }

        // Swap spaces
        self.swap_spaces();
        self.to_space().reset();
        self.collections.fetch_add(1, Ordering::Relaxed);

        // Clear card table for new from-space after swap
        if let Some(ns) = &self.nursery_state {
            let from_idx = self.from_idx.load(Ordering::Acquire);
            ns.card_tables[from_idx].clear_all();
        }

        // Debug: verify all root slots point to from-space after swap
        if cfg!(debug_assertions) && self.gc_every_alloc() {
            for ts in thread_snapshot.iter() {
                ts.scan_roots(&mut |slot| {
                    let bits = unsafe { *slot };
                    if let Some(ptr) = P::try_decode_ptr(bits) {
                        assert!(
                            self.from_space().contains(ptr),
                            "CONCURRENT POST-SWAP: root slot {:p} -> {:p} NOT in from [{:p}..+{}). \
                             In to: {}. collections={}",
                            slot,
                            ptr,
                            self.from_space().base(),
                            self.from_space().size(),
                            self.to_space().contains(ptr),
                            self.collections.load(Ordering::Relaxed),
                        );
                    }
                });
            }
        }

        // Disable write barriers and resume threads
        self.trace_gc(TraceState::GcResuming);
        self.gc_phase.store(GcPhase::Idle as u8, Ordering::Release);
        self.gc_requested.store(false, Ordering::Release);

        for (i, ts) in thread_snapshot.iter().enumerate() {
            ts.resume();
            self.trace_thread(i, TraceState::Running);
        }
        self.trace_gc(TraceState::GcIdle);
    }

    /// Copy `size` bytes word-by-word using Relaxed atomic loads/stores.
    /// Guarantees each 64-bit field is non-torn even if a mutator is
    /// concurrently writing to `src`.
    unsafe fn atomic_copy_words(src: *const u8, dst: *mut u8, size: usize) {
        use std::sync::atomic::AtomicU64;
        debug_assert!(size % 8 == 0);
        let words = size / 8;
        for i in 0..words {
            unsafe {
                let s = &*((src as *const AtomicU64).add(i));
                let d = &*((dst as *const AtomicU64).add(i));
                d.store(s.load(Ordering::Relaxed), Ordering::Relaxed);
            }
        }
    }

    // ─── Internal GC machinery ──────────────────────────────────

    unsafe fn process_slot<P: PtrPolicy>(&self, slot: *mut u64) {
        // Use atomic loads/stores even in STW: the slot was last written by
        // a mutator thread via Cell::set (non-atomic write). Even though the
        // mutator is parked at a safepoint with a Release/Acquire edge,
        // Miri requires matching atomic access types across threads.
        let bits = unsafe {
            let atomic = &*(slot as *const std::sync::atomic::AtomicU64);
            atomic.load(Ordering::Relaxed)
        };
        if let Some(ptr) = P::try_decode_ptr(bits) {
            if self.is_nursery(ptr) {
                return;
            }
            if self.from_space().contains(ptr) {
                let new_ptr = unsafe { self.copy_or_forward::<P>(ptr) };
                unsafe {
                    let atomic = &*(slot as *const std::sync::atomic::AtomicU64);
                    atomic.store(P::encode_ptr(new_ptr), Ordering::Relaxed);
                };
            }
        }
    }

    /// Process a slot during concurrent collection: copy the target
    /// object to to-space (if not already copied) but do NOT update
    /// the parent's pointer slot.
    ///
    /// Pointer updates are deferred to the STW #2 re-scan. This avoids
    /// a data race where the GC writes a to-space pointer into a
    /// parent field while a mutator concurrently reads that field.
    /// On ARM64, the mutator could follow the updated pointer before
    /// the memcpy that populated the to-space copy is visible
    /// (store-store reordering), causing it to see stale data.
    unsafe fn process_slot_concurrent<P: PtrPolicy>(&self, slot: *mut u64) {
        let bits = unsafe {
            let atomic = &*(slot as *const std::sync::atomic::AtomicU64);
            atomic.load(Ordering::Relaxed)
        };
        if let Some(ptr) = P::try_decode_ptr(bits) {
            // Skip nursery pointers during major GC
            if self.is_nursery(ptr) {
                return;
            }
            if self.from_space().contains(ptr) {
                // Copy the object (installs forwarding pointer) but
                // don't write the new address back to the parent slot.
                unsafe { self.copy_or_forward_atomic::<P>(ptr) };
            }
        }
    }

    unsafe fn check_forwarded(&self, old: *mut u8) -> Option<*mut u8> {
        let slot = unsafe { old.add(self.type_id_offset) as *const u64 };
        let word = unsafe { *slot };
        if word & FORWARDING_BIT != 0 {
            Some((word & !FORWARDING_BIT) as *mut u8)
        } else {
            None
        }
    }

    /// Atomic check for forwarding pointer (for concurrent use).
    unsafe fn check_forwarded_atomic(&self, old: *mut u8) -> Option<*mut u8> {
        use std::sync::atomic::AtomicU64;
        let slot = unsafe { old.add(self.type_id_offset) as *const AtomicU64 };
        let word = unsafe { (*slot).load(Ordering::Acquire) };
        if word & FORWARDING_BIT != 0 {
            Some((word & !FORWARDING_BIT) as *mut u8)
        } else {
            None
        }
    }

    unsafe fn install_forwarding(&self, old: *mut u8, new: *mut u8) {
        let slot = unsafe { old.add(self.type_id_offset) as *mut u64 };
        unsafe { *slot = (new as u64) | FORWARDING_BIT };
    }

    /// Atomically install a forwarding pointer using CAS.
    ///
    /// Returns the winning new address (may be from another thread
    /// if we lost the race).
    unsafe fn install_forwarding_atomic(
        &self,
        old: *mut u8,
        old_type_info: u64,
        new: *mut u8,
    ) -> *mut u8 {
        use std::sync::atomic::AtomicU64;
        let slot = unsafe { old.add(self.type_id_offset) as *const AtomicU64 };
        let forwarding = (new as u64) | FORWARDING_BIT;
        match unsafe {
            (*slot).compare_exchange(
                old_type_info,
                forwarding,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
        } {
            Ok(_) => new, // We won — our copy is the canonical one
            Err(current) => {
                // Someone else already installed a forwarding pointer
                debug_assert!(current & FORWARDING_BIT != 0, "CAS failed but not forwarded?");
                (current & !FORWARDING_BIT) as *mut u8
            }
        }
    }

    unsafe fn copy_or_forward<P: PtrPolicy>(&self, old: *mut u8) -> *mut u8 {
        if let Some(forwarded) = unsafe { self.check_forwarded(old) } {
            return forwarded;
        }

        let type_id = unsafe { read_type_id(old, self.type_id_offset) };
        let info = &self.type_table[type_id as usize];
        let varlen_len = match info.varlen {
            VarLenKind::None => 0,
            _ => unsafe { read_varlen_count(old, info) },
        };
        let size = info.allocation_size(varlen_len);

        let new = self.to_space().alloc(info, varlen_len);
        assert!(!new.is_null(), "to-space exhausted during collection");

        unsafe {
            core::ptr::copy_nonoverlapping(old, new, size);
        }

        unsafe { self.install_forwarding(old, new) };

        new
    }

    /// Atomic copy-or-forward for concurrent collection.
    ///
    /// Uses CAS to install the forwarding pointer, so multiple threads
    /// can safely race to copy the same object. Losers discard their
    /// copy (wasted to-space, but correct).
    unsafe fn copy_or_forward_atomic<P: PtrPolicy>(&self, old: *mut u8) -> *mut u8 {
        // Check if already forwarded (atomic load)
        if let Some(forwarded) = unsafe { self.check_forwarded_atomic(old) } {
            return forwarded;
        }

        // Read type info BEFORE it gets overwritten by a forwarding pointer.
        // This is safe because:
        // - If no one else has forwarded yet, the type_info slot is valid.
        // - If someone else forwards between our check and this read,
        //   the CAS below will fail and we'll use their copy.
        let type_info_word = unsafe {
            use std::sync::atomic::AtomicU64;
            let slot = old.add(self.type_id_offset) as *const AtomicU64;
            (*slot).load(Ordering::Acquire)
        };

        // Re-check: might have been forwarded between our two loads
        if type_info_word & FORWARDING_BIT != 0 {
            return (type_info_word & !FORWARDING_BIT) as *mut u8;
        }

        let info = &self.type_table[type_info_word as u16 as usize];
        let varlen_len = match info.varlen {
            VarLenKind::None => 0,
            _ => unsafe { read_varlen_count(old, info) },
        };
        let size = info.allocation_size(varlen_len);

        // Allocate in to-space (atomic bump — safe for concurrent use)
        let new = self.to_space().alloc(info, varlen_len);
        assert!(!new.is_null(), "to-space exhausted during collection");

        // Copy the object word-by-word with atomic loads/stores to avoid
        // torn reads if a mutator is concurrently writing to the source.
        unsafe {
            Self::atomic_copy_words(old, new, size);
        }

        // Try to install forwarding pointer atomically
        let winner = unsafe { self.install_forwarding_atomic(old, type_info_word, new) };

        // If we lost the race, our copy is wasted (to-space leak, but
        // correct). The winner's copy is the canonical one.
        // In a production GC, we'd want to reclaim this space.
        winner
    }

    // ─── Minor GC (Generational) ────────────────────────────────

    /// Promote a nursery object to tenured from-space, or follow its
    /// forwarding pointer if already promoted.
    ///
    /// # Safety
    /// - `old` must point to a valid nursery object.
    /// - Must be called during STW (no concurrent access).
    unsafe fn promote_or_forward<P: PtrPolicy>(&self, old: *mut u8) -> *mut u8 {
        // Check if already promoted (forwarding pointer installed)
        if let Some(forwarded) = unsafe { self.check_forwarded(old) } {
            return forwarded;
        }

        let type_id = unsafe { read_type_id(old, self.type_id_offset) };
        let info = &self.type_table[type_id as usize];
        let varlen_len = match info.varlen {
            VarLenKind::None => 0,
            _ => unsafe { read_varlen_count(old, info) },
        };
        let size = info.allocation_size(varlen_len);

        // Allocate in tenured from-space
        let new = self.from_space().alloc(info, varlen_len);
        assert!(
            !new.is_null(),
            "tenured from-space exhausted during minor GC promotion"
        );

        unsafe {
            core::ptr::copy_nonoverlapping(old, new, size);
        }

        // Install forwarding pointer in nursery copy
        unsafe { self.install_forwarding(old, new) };

        new
    }

    /// Process a slot during minor GC: if it points to the nursery,
    /// promote the target and update the slot.
    ///
    /// # Safety
    /// - Must be called during STW.
    unsafe fn promote_slot<P: PtrPolicy>(&self, slot: *mut u64) {
        let bits = unsafe {
            let atomic = &*(slot as *const std::sync::atomic::AtomicU64);
            atomic.load(Ordering::Relaxed)
        };
        if let Some(ptr) = P::try_decode_ptr(bits) {
            if self.is_nursery(ptr) {
                let new_ptr = unsafe { self.promote_or_forward::<P>(ptr) };
                unsafe {
                    let atomic = &*(slot as *const std::sync::atomic::AtomicU64);
                    atomic.store(P::encode_ptr(new_ptr), Ordering::Relaxed);
                };
            }
        }
    }

    /// Run a minor (nursery) collection triggered by a mutator thread.
    ///
    /// All live nursery objects are promoted to tenured from-space.
    /// The nursery is then reset.
    ///
    /// # Safety
    /// - All objects in the nursery and tenured spaces must have valid headers.
    /// - The triggering thread's frame chain must be stable.
    pub unsafe fn mutator_triggered_minor_gc<P: PtrPolicy>(&self, triggering_thread: &ThreadState) {
        let ns = match &self.nursery_state {
            Some(ns) => ns,
            None => return,
        };

        // Try to become the GC thread
        let gc_guard = match self.gc_lock.try_lock() {
            Ok(guard) => guard,
            Err(std::sync::TryLockError::WouldBlock) => {
                // Another thread is already collecting.
                while self.gc_requested() {
                    triggering_thread.enter_safepoint();
                }
                return;
            }
            Err(e) => panic!("gc_lock poisoned: {}", e),
        };

        // Under gc_lock: if nursery is empty, another thread already did the minor GC.
        if ns.nursery.used() == 0 {
            drop(gc_guard);
            return;
        }

        // Under gc_lock: if tenured doesn't have enough room for worst-case
        // promotion, run a major GC first. We already hold gc_lock, so use
        // the shared STW-from-mutator pattern.
        if ns.nursery.used() > self.from_space().remaining() {
            let trigger_ptr = triggering_thread as *const ThreadState as usize;
            let thread_snapshot: Vec<Arc<ThreadState>> = {
                let threads = self.threads.lock().unwrap();
                self.gc_requested.store(true, Ordering::Release);
                threads.iter().cloned().collect()
            };
            for ts in thread_snapshot.iter() {
                if Arc::as_ptr(ts) as usize == trigger_ptr {
                    continue;
                }
                while !ts.is_safely_at_safepoint() {
                    std::thread::yield_now();
                }
            }
            // Reuse collect_inner — all threads are at safepoints, gc_lock held.
            unsafe { self.collect_inner::<P>(&[]) };
            self.gc_requested.store(false, Ordering::Release);
            for ts in thread_snapshot.iter() {
                if Arc::as_ptr(ts) as usize == trigger_ptr {
                    continue;
                }
                ts.resume();
            }
        }

        // Snapshot threads and set gc_requested
        let trigger_ptr = triggering_thread as *const ThreadState as usize;
        let thread_snapshot: Vec<Arc<ThreadState>> = {
            let threads = self.threads.lock().unwrap();
            self.gc_requested.store(true, Ordering::Release);
            threads.iter().cloned().collect()
        };

        // Wait for all threads EXCEPT ourselves to reach safepoints
        for ts in thread_snapshot.iter() {
            if Arc::as_ptr(ts) as usize == trigger_ptr {
                continue;
            }
            while !ts.is_safely_at_safepoint() {
                std::thread::yield_now();
            }
        }

        // Record promotion start offset for Cheney scanning
        let promotion_start = self.from_space().used();

        // Phase 1: Scan roots — promote nursery objects
        self.globals.scan_roots(&mut |slot| {
            unsafe { self.promote_slot::<P>(slot) };
        });

        for ts in thread_snapshot.iter() {
            ts.scan_roots(&mut |slot| {
                unsafe { self.promote_slot::<P>(slot) };
            });
            let jit_fp = ts.parked_jit_fp();
            if !jit_fp.is_null() {
                self.walk_jit_frame(jit_fp, &mut |slot| {
                    unsafe { self.promote_slot::<P>(slot) };
                });
            }
        }

        // Phase 2: Scan dirty cards in tenured from-space
        let from_idx = self.from_idx.load(Ordering::Acquire);
        {
            let card_table = &ns.card_tables[from_idx];
            let tenured = &self.spaces[from_idx];
            let tenured_used = promotion_start; // only scan pre-existing tenured objects

            // Build object-start index: for each card, the offset of the last
            // object that starts at or before the card boundary. This lets us
            // jump directly to the right object instead of walking from offset 0.
            let obj_starts = unsafe {
                Self::build_object_start_index(
                    tenured,
                    tenured_used,
                    card_table,
                    self.type_id_offset,
                    &self.type_table,
                )
            };

            for (card_idx, card_addr) in card_table.iter_dirty() {
                let start_offset = if card_idx < obj_starts.len() {
                    obj_starts[card_idx]
                } else {
                    continue;
                };
                unsafe {
                    self.scan_card_from_offset::<P>(card_addr, tenured, tenured_used, start_offset);
                }
            }
        }

        // Phase 3: Cheney scan of promoted objects
        // Walk tenured from-space from promotion_start to current used(),
        // processing each newly promoted object's fields.
        let mut scan_offset = promotion_start;
        while scan_offset < self.from_space().used() {
            let obj = unsafe { self.from_space().base().add(scan_offset) };
            let type_id = unsafe { read_type_id(obj, self.type_id_offset) };
            let info = &self.type_table[type_id as usize];

            unsafe {
                scan_object(obj, info, |slot| {
                    self.promote_slot::<P>(slot);
                });
            }

            let varlen_len = match info.varlen {
                VarLenKind::None => 0,
                _ => unsafe { read_varlen_count(obj, info) },
            };
            let obj_size = info.allocation_size(varlen_len);
            let align = 1usize << info.align_log2;
            scan_offset = (scan_offset + obj_size + align - 1) & !(align - 1);
        }

        // Phase 4: Reset nursery and clear card table
        ns.nursery.reset();
        ns.card_tables[from_idx].clear_all();
        ns.minor_collections.fetch_add(1, Ordering::Relaxed);

        // Resume threads
        self.gc_requested.store(false, Ordering::Release);
        for ts in thread_snapshot.iter() {
            if Arc::as_ptr(ts) as usize == trigger_ptr {
                continue;
            }
            ts.resume();
        }

        drop(gc_guard);
    }

    /// Build an object-start index for card scanning.
    ///
    /// Returns a Vec where entry[i] is the offset of the last object that
    /// starts at or before card i's boundary. This allows O(1) lookup of
    /// where to start scanning for a given dirty card, instead of walking
    /// from offset 0.
    ///
    /// # Safety
    /// All objects in the tenured space must have valid headers.
    unsafe fn build_object_start_index(
        tenured: &AtomicBumpAllocator,
        tenured_used: usize,
        card_table: &CardTable,
        type_id_offset: usize,
        type_table: &[TypeInfo],
    ) -> Vec<usize> {
        let num_cards = card_table.card_count();
        // Sentinel: usize::MAX means "no object starts in this card".
        let mut obj_starts = vec![usize::MAX; num_cards];
        let tenured_base = tenured.base() as usize;
        let card_size = card_table.card_size();

        let mut offset = 0usize;
        while offset < tenured_used {
            let obj = unsafe { tenured.base().add(offset) };
            let type_id = unsafe { read_type_id(obj, type_id_offset) };
            let info = &type_table[type_id as usize];
            let varlen_len = match info.varlen {
                VarLenKind::None => 0,
                _ => unsafe { read_varlen_count(obj, info) },
            };
            let obj_size = info.allocation_size(varlen_len);

            let obj_addr = obj as usize;
            let card_idx = (obj_addr - tenured_base) / card_size;

            // Keep the FIRST object per card — earlier objects must also be scanned.
            if card_idx < num_cards && obj_starts[card_idx] == usize::MAX {
                obj_starts[card_idx] = offset;
            }

            let align = 1usize << info.align_log2;
            offset = (offset + obj_size + align - 1) & !(align - 1);
        }

        // Forward-fill: cards with no objects inherit the previous card's
        // start offset. This handles objects that span from an earlier card.
        // scan_card_from_offset walks forward from this offset and checks
        // overlap, so scanning a few extra pre-card objects is harmless.
        if num_cards > 0 && obj_starts[0] == usize::MAX {
            obj_starts[0] = 0;
        }
        for i in 1..num_cards {
            if obj_starts[i] == usize::MAX {
                obj_starts[i] = obj_starts[i - 1];
            }
        }

        obj_starts
    }

    /// Scan objects overlapping a dirty card, starting from a known offset.
    ///
    /// # Safety
    /// - Must be called during STW.
    /// - `start_offset` must be the offset of a valid object in tenured space.
    unsafe fn scan_card_from_offset<P: PtrPolicy>(
        &self,
        card_addr: *const u8,
        tenured: &AtomicBumpAllocator,
        tenured_used: usize,
        start_offset: usize,
    ) {
        let card_start = card_addr as usize;
        let card_end = card_start + 512; // CARD_SHIFT = 9 → 512 bytes

        let mut offset = start_offset;
        while offset < tenured_used {
            let obj = unsafe { tenured.base().add(offset) };
            let obj_addr = obj as usize;
            let type_id = unsafe { read_type_id(obj, self.type_id_offset) };
            let info = &self.type_table[type_id as usize];

            let varlen_len = match info.varlen {
                VarLenKind::None => 0,
                _ => unsafe { read_varlen_count(obj, info) },
            };
            let obj_size = info.allocation_size(varlen_len);
            let obj_end = obj_addr + obj_size;

            // Check if object overlaps with the card region
            if obj_end > card_start && obj_addr < card_end {
                unsafe {
                    scan_object(obj, info, |slot| {
                        self.promote_slot::<P>(slot);
                    });
                }
            }

            // Stop if we've passed the card entirely
            if obj_addr >= card_end {
                break;
            }

            let align = 1usize << info.align_log2;
            offset = (offset + obj_size + align - 1) & !(align - 1);
        }
    }
}
