use std::sync::atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use dynobj::{
    read_type_info, read_varlen_count, scan_object,
    ObjHeader, RootSource, TypeInfo, VarLenKind, AtomicRootSet,
};

use crate::alloc::{Alloc, AtomicBumpAllocator};
use crate::barrier::SATBQueue;
use crate::semi_space::PtrPolicy;
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

pub struct Heap {
    from: AtomicBumpAllocator,
    to: AtomicBumpAllocator,

    /// Registered mutator threads. The GC scans their roots during STW.
    threads: Mutex<Vec<Arc<ThreadState>>>,

    /// Global roots (interned symbols, module-level constants, etc.)
    pub globals: AtomicRootSet,

    /// Flag polled by mutator threads: "should I enter safepoint?"
    gc_requested: AtomicBool,

    /// Prevents concurrent GC triggers.
    gc_lock: Mutex<()>,

    type_info_offset: usize,
    collections: AtomicUsize,

    /// Current GC phase. Mutators check this to know if write barriers
    /// should be active.
    gc_phase: AtomicU8,

    /// Global SATB queue. Mutator threads flush their local SATB buffers
    /// here, and the GC drains it during collection.
    pub satb_queue: SATBQueue,
}

// Safety: All fields are either Sync (atomics, mutexes) or accessed
// only under proper synchronization.
unsafe impl Sync for Heap {}
unsafe impl Send for Heap {}

impl Heap {
    /// Create a new heap with two spaces of `space_size` bytes each.
    pub fn new<H: ObjHeader>(space_size: usize) -> Self {
        Heap {
            from: AtomicBumpAllocator::new::<H>(space_size),
            to: AtomicBumpAllocator::new::<H>(space_size),
            threads: Mutex::new(Vec::new()),
            globals: AtomicRootSet::new(),
            gc_requested: AtomicBool::new(false),
            gc_lock: Mutex::new(()),
            type_info_offset: H::TYPE_INFO_OFFSET,
            collections: AtomicUsize::new(0),
            gc_phase: AtomicU8::new(GcPhase::Idle as u8),
            satb_queue: SATBQueue::new(),
        }
    }

    // ─── Thread registration ────────────────────────────────────

    /// Register a new mutator thread. Returns a shared handle to
    /// the thread's state and its index.
    pub fn register_thread(&self) -> (Arc<ThreadState>, usize) {
        let ts = Arc::new(ThreadState::new());
        let mut threads = self.threads.lock().unwrap();
        let id = threads.len();
        threads.push(ts.clone());
        (ts, id)
    }

    /// Deregister a mutator thread by index.
    ///
    /// The thread must not be running mutator code when this is called.
    /// Typically called after the thread has finished all work.
    pub fn deregister_thread(&self, id: usize) {
        let mut threads = self.threads.lock().unwrap();
        // Swap-remove to avoid shifting. IDs of other threads don't change
        // because we don't use the index for anything after registration
        // (the Arc<ThreadState> is the real handle).
        if id < threads.len() {
            threads.swap_remove(id);
        }
    }

    // ─── Allocation ─────────────────────────────────────────────

    /// Allocate from the shared from-space (atomic bump).
    ///
    /// Returns null if from-space is exhausted. Caller should
    /// trigger GC via `collect()` and retry.
    pub fn alloc(&self, info: &'static TypeInfo, varlen_len: usize) -> *mut u8 {
        self.from.alloc(info, varlen_len)
    }

    /// Allocate and initialize header + varlen count.
    pub fn alloc_obj<H: ObjHeader>(
        &self,
        info: &'static TypeInfo,
        varlen_len: usize,
    ) -> *mut u8 {
        unsafe { crate::alloc::alloc_obj::<H>(&self.from, info, varlen_len) }
    }

    // ─── Safepoint polling ──────────────────────────────────────

    /// Check if GC has been requested. Mutator threads should call
    /// this at safepoints (loop backedges, allocation slow paths).
    #[inline(always)]
    pub fn gc_requested(&self) -> bool {
        self.gc_requested.load(Ordering::Relaxed)
    }

    /// Poll for GC and enter safepoint if requested.
    ///
    /// This is the main safepoint check for mutator threads.
    /// Fast path: single atomic load + branch (no contention).
    #[inline(always)]
    pub fn safepoint(&self, thread: &ThreadState) {
        if self.gc_requested.load(Ordering::Relaxed) {
            thread.enter_safepoint();
        }
    }

    // ─── Collection ─────────────────────────────────────────────

    /// Bytes used in from-space.
    pub fn from_used(&self) -> usize {
        self.from.used()
    }

    /// Check if a pointer is in from-space.
    pub fn contains(&self, ptr: *const u8) -> bool {
        self.from.contains(ptr)
    }

    /// Total size of each space.
    pub fn space_size(&self) -> usize {
        self.from.size()
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
    pub unsafe fn collect<P: PtrPolicy>(
        &self,
        extra_roots: &[&dyn RootSource],
    ) {
        let _gc_guard = self.gc_lock.lock().unwrap();

        // Phase 1: scan all roots → copy/forward targets into to-space

        // Global roots
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
            }
        }

        // Extra roots (caller-provided)
        for source in extra_roots {
            source.scan_roots(&mut |slot| {
                unsafe { self.process_slot::<P>(slot) };
            });
        }

        // Phase 2: Cheney scan — walk to-space linearly
        let mut scan_offset = 0usize;
        while scan_offset < self.to.used() {
            let obj = unsafe { self.to.base().add(scan_offset) };
            let info = unsafe { &*read_type_info(obj, self.type_info_offset) };

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
        // We can't std::mem::swap because AtomicBumpAllocator isn't easily
        // swappable (contains atomics). Instead, we use pointer swapping.
        // For AtomicBumpAllocator, we swap the base pointers and reset cursors.
        //
        // Actually, since we can't swap atomics directly, we'll use a
        // different approach: the from and to are behind raw pointers that
        // we can swap at the memory level.
        //
        // For now, with immutable fields, we need a different approach.
        // We'll swap the underlying memory using unsafe pointer casts.
        unsafe {
            let from_ptr = &self.from as *const AtomicBumpAllocator as *mut AtomicBumpAllocator;
            let to_ptr = &self.to as *const AtomicBumpAllocator as *mut AtomicBumpAllocator;
            core::ptr::swap(from_ptr, to_ptr);
        }
        self.to.reset();
        self.collections.fetch_add(1, Ordering::Relaxed);
    }

    /// Trigger a STW collection cycle: request all threads to stop,
    /// wait for them, collect, then resume them.
    ///
    /// # Safety
    /// Same as `collect`. The calling thread must NOT be registered
    /// as a mutator (or must have already entered its safepoint).
    pub unsafe fn stw_collect<P: PtrPolicy>(&self) {
        // Signal all threads to reach safepoints
        self.gc_requested.store(true, Ordering::Release);

        // Wait for all threads to reach safepoints
        {
            let threads = self.threads.lock().unwrap();
            for ts in threads.iter() {
                while !ts.is_at_safepoint() {
                    std::thread::yield_now();
                }
            }
        }

        // All threads suspended — run collection
        unsafe { self.collect::<P>(&[]) };

        // Clear the request flag before resuming
        self.gc_requested.store(false, Ordering::Release);

        // Resume all threads
        {
            let threads = self.threads.lock().unwrap();
            for ts in threads.iter() {
                ts.resume();
            }
        }
    }

    /// Bytes remaining in from-space.
    pub fn from_remaining(&self) -> usize {
        self.from.remaining()
    }

    /// Trigger STW collection from a mutator thread.
    ///
    /// If another thread is already collecting, this thread enters its
    /// safepoint and waits for that collection to finish instead.
    ///
    /// # Safety
    /// Same as `collect`.
    pub unsafe fn mutator_triggered_gc<P: PtrPolicy>(&self, triggering_thread: &ThreadState) {
        // Try to become the GC thread. If someone else is already collecting,
        // enter our safepoint and wait for them to finish.
        let gc_guard = match self.gc_lock.try_lock() {
            Ok(guard) => guard,
            Err(std::sync::TryLockError::WouldBlock) => {
                // Another thread is already collecting.
                // Enter safepoint so they can proceed.
                triggering_thread.enter_safepoint();
                return;
            }
            Err(e) => panic!("gc_lock poisoned: {}", e),
        };

        // We won the race — we're the GC thread now.

        // Signal all threads to reach safepoints
        self.gc_requested.store(true, Ordering::Release);

        // Wait for all threads EXCEPT ourselves to reach safepoints
        {
            let threads = self.threads.lock().unwrap();
            for ts in threads.iter() {
                let ts_ptr = Arc::as_ptr(ts) as usize;
                let trigger_ptr = triggering_thread as *const ThreadState as usize;
                if ts_ptr == trigger_ptr {
                    continue;
                }
                while !ts.is_at_safepoint() {
                    std::thread::yield_now();
                }
            }
        }

        // All other threads suspended — run collection.
        // Scan ALL threads' roots (including ours — our frame chain is stable
        // since we're running GC code, not mutator code).

        // Phase 1: scan all roots
        self.globals.scan_roots(&mut |slot| {
            unsafe { self.process_slot::<P>(slot) };
        });

        {
            let threads = self.threads.lock().unwrap();
            for ts in threads.iter() {
                ts.scan_roots(&mut |slot| {
                    unsafe { self.process_slot::<P>(slot) };
                });
            }
        }

        // Phase 2: Cheney scan
        let mut scan_offset = 0usize;
        while scan_offset < self.to.used() {
            let obj = unsafe { self.to.base().add(scan_offset) };
            let info = unsafe { &*read_type_info(obj, self.type_info_offset) };

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
        unsafe {
            let from_ptr = &self.from as *const AtomicBumpAllocator as *mut AtomicBumpAllocator;
            let to_ptr = &self.to as *const AtomicBumpAllocator as *mut AtomicBumpAllocator;
            core::ptr::swap(from_ptr, to_ptr);
        }
        self.to.reset();
        self.collections.fetch_add(1, Ordering::Relaxed);

        // Clear request flag and resume all other threads
        self.gc_requested.store(false, Ordering::Release);

        {
            let threads = self.threads.lock().unwrap();
            for ts in threads.iter() {
                let ts_ptr = Arc::as_ptr(ts) as usize;
                let trigger_ptr = triggering_thread as *const ThreadState as usize;
                if ts_ptr == trigger_ptr {
                    continue;
                }
                ts.resume();
            }
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
    #[inline(always)]
    pub fn barriers_active(&self) -> bool {
        self.gc_phase.load(Ordering::Relaxed) != GcPhase::Idle as u8
    }

    /// The type_info_offset for this heap's header type.
    pub fn type_info_offset(&self) -> usize {
        self.type_info_offset
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
        unsafe { crate::barrier::read_barrier_atomic(ptr, self.type_info_offset) }
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

        // Enable write barriers before stopping threads, so any
        // stores that happen while we're waiting are captured.
        self.gc_phase.store(GcPhase::Copying as u8, Ordering::Release);

        // Signal threads to reach safepoints
        self.gc_requested.store(true, Ordering::Release);

        // Wait for all threads
        {
            let threads = self.threads.lock().unwrap();
            for ts in threads.iter() {
                while !ts.is_at_safepoint() {
                    std::thread::yield_now();
                }
            }
        }

        // Snapshot roots into to-space (short work under STW)
        self.globals.scan_roots(&mut |slot| {
            unsafe { self.process_slot::<P>(slot) };
        });

        {
            let threads = self.threads.lock().unwrap();
            for ts in threads.iter() {
                ts.scan_roots(&mut |slot| {
                    unsafe { self.process_slot::<P>(slot) };
                });
            }
        }

        // Resume threads — they now run with write barriers active
        self.gc_requested.store(false, Ordering::Release);
        {
            let threads = self.threads.lock().unwrap();
            for ts in threads.iter() {
                ts.resume();
            }
        }

        // ── Concurrent Copy Phase ────────────────────────────────

        // Cheney scan of to-space. Objects were seeded by root scanning above.
        // Mutators are running concurrently with write barriers active.
        // New objects allocated by mutators go into from-space (still the
        // active allocation space until we swap).
        let mut scan_offset = 0usize;
        while scan_offset < self.to.used() {
            let obj = unsafe { self.to.base().add(scan_offset) };
            let info = unsafe { &*read_type_info(obj, self.type_info_offset) };

            unsafe {
                scan_object(obj, info, |slot| {
                    self.process_slot_concurrent::<P>(slot);
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

        // ── STW Pause #2: Final drain + swap ────────────────────

        // Stop all threads again to drain SATB buffers and finalize
        self.gc_requested.store(true, Ordering::Release);
        {
            let threads = self.threads.lock().unwrap();
            for ts in threads.iter() {
                while !ts.is_at_safepoint() {
                    std::thread::yield_now();
                }
            }
        }

        // Re-scan all roots. Between STW #1 and now, mutators may have:
        // - Allocated new objects in from-space and stored them in roots
        // - Changed root slots to point to different objects
        // Re-scanning catches all of these.
        self.globals.scan_roots(&mut |slot| {
            unsafe { self.process_slot::<P>(slot) };
        });
        {
            let threads = self.threads.lock().unwrap();
            for ts in threads.iter() {
                ts.scan_roots(&mut |slot| {
                    unsafe { self.process_slot::<P>(slot) };
                });
            }
        }

        // Drain SATB queue (flushed by mutators) + thread-local SATB buffers
        let satb_values = self.satb_queue.drain_all();
        for bits in satb_values {
            if let Some(ptr) = P::try_decode_ptr(bits) {
                if self.from.contains(ptr) {
                    unsafe { self.copy_or_forward::<P>(ptr) };
                }
            }
        }

        // Drain per-thread SATB buffers (threads are at safepoints — safe)
        {
            let threads = self.threads.lock().unwrap();
            for ts in threads.iter() {
                let buf = unsafe { ts.satb_buffer() };
                let values = buf.drain();
                for bits in values {
                    if let Some(ptr) = P::try_decode_ptr(bits) {
                        if self.from.contains(ptr) {
                            unsafe { self.copy_or_forward::<P>(ptr) };
                        }
                    }
                }
            }
        }

        // Final Cheney scan pass — process objects from re-scan + SATB drain
        while scan_offset < self.to.used() {
            let obj = unsafe { self.to.base().add(scan_offset) };
            let info = unsafe { &*read_type_info(obj, self.type_info_offset) };

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
        unsafe {
            let from_ptr = &self.from as *const AtomicBumpAllocator as *mut AtomicBumpAllocator;
            let to_ptr = &self.to as *const AtomicBumpAllocator as *mut AtomicBumpAllocator;
            core::ptr::swap(from_ptr, to_ptr);
        }
        self.to.reset();
        self.collections.fetch_add(1, Ordering::Relaxed);

        // Disable write barriers and resume threads
        self.gc_phase.store(GcPhase::Idle as u8, Ordering::Release);
        self.gc_requested.store(false, Ordering::Release);

        {
            let threads = self.threads.lock().unwrap();
            for ts in threads.iter() {
                ts.resume();
            }
        }
    }

    // ─── Internal GC machinery ──────────────────────────────────

    unsafe fn process_slot<P: PtrPolicy>(&self, slot: *mut u64) {
        let bits = unsafe { *slot };
        if let Some(ptr) = P::try_decode_ptr(bits) {
            if self.from.contains(ptr) {
                let new_ptr = unsafe { self.copy_or_forward::<P>(ptr) };
                unsafe { *slot = P::encode_ptr(new_ptr) };
            }
        }
    }

    /// Process a slot during concurrent collection. Uses atomic CAS
    /// for forwarding pointer installation.
    unsafe fn process_slot_concurrent<P: PtrPolicy>(&self, slot: *mut u64) {
        let bits = unsafe { *slot };
        if let Some(ptr) = P::try_decode_ptr(bits) {
            if self.from.contains(ptr) {
                let new_ptr = unsafe { self.copy_or_forward_atomic::<P>(ptr) };
                unsafe { *slot = P::encode_ptr(new_ptr) };
            }
        }
    }

    unsafe fn check_forwarded(&self, old: *mut u8) -> Option<*mut u8> {
        let slot = unsafe { old.add(self.type_info_offset) as *const u64 };
        let word = unsafe { *slot };
        if word & 1 == 1 {
            Some((word & !1) as *mut u8)
        } else {
            None
        }
    }

    /// Atomic check for forwarding pointer (for concurrent use).
    unsafe fn check_forwarded_atomic(&self, old: *mut u8) -> Option<*mut u8> {
        use std::sync::atomic::AtomicU64;
        let slot = unsafe { old.add(self.type_info_offset) as *const AtomicU64 };
        let word = unsafe { (*slot).load(Ordering::Acquire) };
        if word & 1 == 1 {
            Some((word & !1) as *mut u8)
        } else {
            None
        }
    }

    unsafe fn install_forwarding(&self, old: *mut u8, new: *mut u8) {
        let slot = unsafe { old.add(self.type_info_offset) as *mut u64 };
        unsafe { *slot = (new as u64) | 1 };
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
        let slot = unsafe { old.add(self.type_info_offset) as *const AtomicU64 };
        let forwarding = (new as u64) | 1;
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
                debug_assert!(current & 1 == 1, "CAS failed but not forwarded?");
                (current & !1) as *mut u8
            }
        }
    }

    unsafe fn copy_or_forward<P: PtrPolicy>(&self, old: *mut u8) -> *mut u8 {
        if let Some(forwarded) = unsafe { self.check_forwarded(old) } {
            return forwarded;
        }

        let info = unsafe { &*read_type_info(old, self.type_info_offset) };
        let varlen_len = match info.varlen {
            VarLenKind::None => 0,
            _ => unsafe { read_varlen_count(old, info) },
        };
        let size = info.allocation_size(varlen_len);

        let new = self.to.alloc(info, varlen_len);
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
            let slot = old.add(self.type_info_offset) as *const AtomicU64;
            (*slot).load(Ordering::Acquire)
        };

        // Re-check: might have been forwarded between our two loads
        if type_info_word & 1 == 1 {
            return (type_info_word & !1) as *mut u8;
        }

        let info = unsafe { &*(type_info_word as *const TypeInfo) };
        let varlen_len = match info.varlen {
            VarLenKind::None => 0,
            _ => unsafe { read_varlen_count(old, info) },
        };
        let size = info.allocation_size(varlen_len);

        // Allocate in to-space (atomic bump — safe for concurrent use)
        let new = self.to.alloc(info, varlen_len);
        assert!(!new.is_null(), "to-space exhausted during collection");

        // Copy the object
        unsafe {
            core::ptr::copy_nonoverlapping(old, new, size);
        }

        // Try to install forwarding pointer atomically
        let winner = unsafe { self.install_forwarding_atomic(old, type_info_word, new) };

        if winner as usize != new as usize {
            // We lost the race — our copy is wasted (to-space leak, but
            // correct). The winner's copy is the canonical one.
            // In a production GC, we'd want to reclaim this space.
        }

        winner
    }
}
